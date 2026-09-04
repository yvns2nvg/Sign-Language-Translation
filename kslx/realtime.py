"""웹캠 데모 (Windows).

기본은 --mode auto: 손 움직임 에너지로 "단어 끝 -> 다음 단어 시작"을 자동
감지해서 매번 SPACE를 누르지 않아도 이어서 인식한다 (kslx.stream.SegmentGate).
--mode manual 로 기존처럼 SPACE 수동 녹화도 가능. 실행 중 M 키로 전환.

노트북에 옮겨 쓸 때 필요한 것: kslx/ 폴더 전체(특히 mp_models/holistic_landmarker.task,
data/word_labels.json) + runs/ 안의 체크포인트(.pt) 하나.

    pip install torch opencv-python mediapipe pillow numpy
    python -m kslx.realtime --ckpt runs/signer_out_aug.pt

★ 얼굴 매핑은 근사치다 (kslx.adapters.mediapipe_adapter 상단 주석 참고).
★ 자동 세그멘테이션은 휴리스틱이다 (kslx.stream 상단 주석 참고) — 카메라가
   손을 순간적으로 놓치면 단어가 중간에 잘못 끊길 수 있다. 그럴 땐 --mode manual
   또는 화면에 뜨는 에너지 수치를 보고 --start-energy/--end-energy 를 조정할 것.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from kslx.adapters.mediapipe_adapter import holistic_result_to_frame89
from kslx.data.aihub import word_label_map
from kslx.models.conv_transformer import ConvTransformer
from kslx.normalize import featurize
from kslx.stream import LiveNormalizer, SegmentGate, hand_energy

WINDOWS_KOREAN_FONT = r"C:\Windows\Fonts\malgun.ttf"
MODEL_BUNDLE_DEFAULT = Path(__file__).parent / "mp_models" / "holistic_landmarker.task"


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ConvTransformer(feature_dim=ckpt["feature_dim"], num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["classes"]


def make_landmarker(bundle_path: Path):
    BaseOptions = mp.tasks.BaseOptions
    HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
    HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(bundle_path)),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.3,
    )
    return HolisticLandmarker.create_from_options(options)


def put_korean_text(frame_bgr: np.ndarray, text: str, org: tuple[int, int],
                     font: ImageFont.FreeTypeFont, color=(0, 255, 0)) -> np.ndarray:
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(org, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def predict(model, classes, label_map, buffer: list[np.ndarray], t_out: int, device: str, topk: int = 5):
    seq = np.stack(buffer, axis=0)  # (T, 89, 2)
    feat = featurize(seq, t_out=t_out, sign_span=None)
    x = torch.from_numpy(feat).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0]
    top_probs, top_idx = probs.topk(min(topk, probs.shape[0]))
    results = []
    for p, idx in zip(top_probs.tolist(), top_idx.tolist()):
        word_id = int(classes[idx])
        gloss = label_map.get(word_id, f"WORD{word_id:04d}")
        results.append((gloss, p))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--model-bundle", type=Path, default=MODEL_BUNDLE_DEFAULT)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-face", action="store_true")
    ap.add_argument("--t-out", type=int, default=64)
    ap.add_argument("--mode", choices=["auto", "manual"], default="auto")
    ap.add_argument("--start-energy", type=float, default=0.06)
    ap.add_argument("--end-energy", type=float, default=0.03)
    ap.add_argument("--end-hold", type=int, default=10, help="이 프레임 수만큼 저에너지가 지속되면 단어 끝으로 판정")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, classes = load_model(args.ckpt, args.device)
    label_map = word_label_map()
    landmarker = make_landmarker(args.model_bundle)
    font_small = ImageFont.truetype(WINDOWS_KOREAN_FONT, 22)
    font_big = ImageFont.truetype(WINDOWS_KOREAN_FONT, 34)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {args.camera} 를 열 수 없습니다.")

    mode = args.mode
    manual_recording = False
    manual_buffer: list[np.ndarray] = []
    gate = SegmentGate(start_energy=args.start_energy, end_energy=args.end_energy, end_hold=args.end_hold)
    live_norm = LiveNormalizer()
    prev_norm_frame = None
    last_result = None
    t0 = time.time()

    print("모드: auto(에너지 자동 분할) / manual(SPACE 수동) — M 으로 전환")
    print("SPACE: manual 모드에서 녹화 시작/중지, auto 모드에서 현재 구간 강제 종료")
    print("R: 취소 | Q/ESC: 종료")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - t0) * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)
        frame89 = holistic_result_to_frame89(result, w, h, use_face=not args.no_face)

        norm_frame = live_norm.normalize(frame89)
        energy = hand_energy(prev_norm_frame, norm_frame) if prev_norm_frame is not None else 0.0
        prev_norm_frame = norm_frame

        status = ""
        if mode == "auto":
            label, finished_buf = gate.step(frame89, energy)
            if label == "finished" and finished_buf is not None:
                if len(finished_buf) >= 4:
                    last_result = predict(model, classes, label_map, finished_buf, args.t_out, args.device)
                    print(last_result)
                else:
                    print("구간이 너무 짧습니다 (4프레임 미만) — 무시함")
            state_kr = {"idle": "대기", "started": "녹화중", "recording": "녹화중", "finished": "대기"}[label]
            status = f"[auto] {state_kr}  energy={energy:.3f}  frames={len(gate.buffer)}"
        else:
            if manual_recording:
                manual_buffer.append(frame89)
            status = f"[manual] {'REC ' + str(len(manual_buffer)) + 'f' if manual_recording else '대기 (SPACE)'}"

        color = (0, 0, 255) if (mode == "auto" and gate.state == "recording") or \
                                (mode == "manual" and manual_recording) else (0, 255, 0)
        frame = put_korean_text(frame, status, (10, 10), font_small, color=color)
        if last_result:
            y = 50
            for gloss, p in last_result:
                frame = put_korean_text(frame, f"{gloss}  {p*100:.1f}%", (10, y), font_big)
                y += 40

        cv2.imshow("KSL realtime (kslx)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("m"):
            mode = "manual" if mode == "auto" else "auto"
            gate = SegmentGate(start_energy=args.start_energy, end_energy=args.end_energy, end_hold=args.end_hold)
            manual_recording = False
            manual_buffer = []
            last_result = None
            print(f"-> 모드 전환: {mode}")
        elif key == ord(" "):
            if mode == "manual":
                if not manual_recording:
                    manual_recording = True
                    manual_buffer = []
                    last_result = None
                else:
                    manual_recording = False
                    if len(manual_buffer) >= 4:
                        last_result = predict(model, classes, label_map, manual_buffer, args.t_out, args.device)
                        print(last_result)
                    else:
                        print("녹화가 너무 짧습니다 (4프레임 미만) — 무시함")
                    manual_buffer = []
            else:
                if gate.state == "recording" and len(gate.buffer) >= 4:
                    last_result = predict(model, classes, label_map, gate.buffer, args.t_out, args.device)
                    print(last_result)
                gate.state = "idle"
                gate.buffer = []
                gate.low_run = 0
        elif key == ord("r"):
            manual_recording = False
            manual_buffer = []
            gate.state = "idle"
            gate.buffer = []
            gate.low_run = 0
            last_result = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
