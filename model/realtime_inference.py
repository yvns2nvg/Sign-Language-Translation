"""
KSL 수어 실시간 인식 - 웹캠 추론 스크립트
================================================
모델 감지 우선순위:
  1. ksl_tcn_improved.pt   (TCN, 99%+ 정확도)  ← 기본 사용
  2. ksl_dtw_knn_model.pkl (DTW+KNN, 최후 폴백)

사용법:
  python realtime_inference.py
  python realtime_inference.py --model path/to/model.pt
  python realtime_inference.py --temperature 1.0   # 원본 logit 그대로
  python realtime_inference.py --no-tta            # TTA 비활성화 (빠름)

조작:
  SPACE  : 녹화 시작 / 중지 후 예측
  R      : 녹화 취소
  Q / ESC: 종료
"""

import argparse
import os
import sys
import numpy as np
import cv2
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFont, ImageDraw, Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from features import (
    extract_improved_features,
    resample_keypoints,
)

# ─────────────────────────────────────────────
# 추론 하이퍼파라미터
# ─────────────────────────────────────────────
TTA_SPEEDS    = [0.85, 1.0, 1.15]   # 속도 TTA 배율
CONF_THRESHOLD = 0.20               # 이 미만이면 "불확실" 표시


# ─────────────────────────────────────────────
# P1-B: 손 미감지 프레임 보간
# ─────────────────────────────────────────────
def _fill_missing_hands(seq):
    """
    seq: (T, 67, 3)
    MediaPipe 미감지 프레임(전체 0)을 이웃 유효 프레임으로 선형 보간.
    연속 6프레임 초과 공백은 마지막 유효값으로 채운다.
    """
    result = seq.copy()
    for hand_sl in (slice(0, 21), slice(21, 42)):
        missing = np.all(result[:, hand_sl, :] == 0, axis=(1, 2))
        if not missing.any():
            continue
        valid_idx = np.where(~missing)[0]
        if len(valid_idx) == 0:
            continue
        for t in np.where(missing)[0]:
            prev_v = valid_idx[valid_idx < t]
            next_v = valid_idx[valid_idx > t]
            if len(prev_v) == 0:
                result[t, hand_sl] = result[next_v[0], hand_sl]
            elif len(next_v) == 0:
                result[t, hand_sl] = result[prev_v[-1], hand_sl]
            else:
                t0, t1 = prev_v[-1], next_v[0]
                if t1 - t0 <= 6:
                    w = (t - t0) / (t1 - t0)
                    result[t, hand_sl] = ((1 - w) * result[t0, hand_sl]
                                          + w * result[t1, hand_sl])
                else:
                    result[t, hand_sl] = result[prev_v[-1], hand_sl]
    return result


# ─────────────────────────────────────────────
# TCN 아키텍처
# ─────────────────────────────────────────────
class AttnPool1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        return (x * torch.softmax(self.attn(x), dim=1)).sum(dim=1)


class AttnAvgMaxPool1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_pool = AttnPool1D(dim)

    def forward(self, x):
        return torch.cat([self.attn_pool(x), x.mean(dim=1), x.max(dim=1).values], dim=-1)


class Chomp1d(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, x):
        return x[:, :, :-self.s].contiguous() if self.s else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, dilation, dropout):
        super().__init__()
        p = (ks - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, ks, padding=p, dilation=dilation)
        self.chomp1 = Chomp1d(p)
        self.bn1    = nn.BatchNorm1d(out_ch)
        self.conv2  = nn.Conv1d(out_ch, out_ch, ks, padding=p, dilation=dilation)
        self.chomp2 = Chomp1d(p)
        self.bn2    = nn.BatchNorm1d(out_ch)
        self.dropout    = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.dropout(F.gelu(self.bn1(self.chomp1(self.conv1(x)))))
        y = self.dropout(F.gelu(self.bn2(self.chomp2(self.conv2(y)))))
        return F.gelu(y + (x if self.downsample is None else self.downsample(x)))


class TCNClassifier(nn.Module):
    def __init__(self, input_dim, channels, num_classes, dropout, kernel_size=5):
        super().__init__()
        layers, in_ch = [], input_dim
        for d in [1, 2, 4, 8, 16]:
            layers.append(TemporalBlock(in_ch, channels, kernel_size, d, dropout))
            in_ch = channels
        self.tcn  = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(channels)
        self.pool = AttnAvgMaxPool1D(channels)
        self.head = nn.Sequential(
            nn.Linear(channels * 3, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.head(self.pool(self.norm(x)))


# ─────────────────────────────────────────────
# MediaPipe 키포인트 추출
# ─────────────────────────────────────────────

# MediaPipe Pose 랜드마크 인덱스
_MP_NOSE       = 0
_MP_L_EYE      = 2
_MP_R_EYE      = 5
_MP_L_EAR      = 7
_MP_R_EAR      = 8
_MP_L_SHOULDER = 11
_MP_R_SHOULDER = 12
_MP_L_ELBOW    = 13
_MP_R_ELBOW    = 14
_MP_L_WRIST    = 15
_MP_R_WRIST    = 16
_MP_L_HIP      = 23
_MP_R_HIP      = 24
_MP_L_KNEE     = 25
_MP_R_KNEE     = 26
_MP_L_ANKLE    = 27
_MP_R_ANKLE    = 28
_MP_L_HEEL     = 29
_MP_R_HEEL     = 30
_MP_L_FOOT_IDX = 31
_MP_R_FOOT_IDX = 32


def extract_keypoints(results):
    """
    MediaPipe Holistic → (67, 3)
    순서: left hand(0:21) + right hand(21:42) + pose first 25(42:67)
    학습 NPZ 포맷과 동일 (Colab 전처리 코드로 확인).
    """
    lh = (np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                   dtype=np.float32)
          if results.left_hand_landmarks else np.zeros((21, 3), np.float32))
    rh = (np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                   dtype=np.float32)
          if results.right_hand_landmarks else np.zeros((21, 3), np.float32))
    # 포즈 z=0 고정: AI Hub는 OpenPose BODY_25(2D) 기반 → z 없음
    pose = (np.array([[lm.x, lm.y, 0.0] for lm in results.pose_landmarks.landmark[:25]],
                     dtype=np.float32)
            if results.pose_landmarks else np.zeros((25, 3), np.float32))
    return np.concatenate([lh, rh, pose], axis=0)


# ─────────────────────────────────────────────
# 모델 로드 (TCN 전용 + DTW+KNN 최후 폴백)
# ─────────────────────────────────────────────
def load_model(model_path, temperature=1.0, use_tta=True):
    """
    temperature : 소프트맥스 온도 (1.0 = 원본 logit, >1 → 분포 평탄화)
    use_tta     : True 시 TTA_SPEEDS 3배속 예측 후 평균
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tcn_pt  = os.path.join(SCRIPT_DIR, 'ksl_tcn_improved.pt')
    dtw_pkl = os.path.join(SCRIPT_DIR, 'ksl_dtw_knn_model.pkl')

    if model_path == 'auto':
        if os.path.exists(tcn_pt):    model_path = tcn_pt
        elif os.path.exists(dtw_pkl): model_path = dtw_pkl
        else:
            print('[오류] 모델 파일을 찾을 수 없습니다.'); sys.exit(1)

    # ── TCN (.pt) ─────────────────────────────
    if model_path.endswith('.pt'):
        ckpt   = torch.load(model_path, map_location='cpu', weights_only=False)
        config = ckpt['config']

        if config.get('model') != 'tcn':
            print(f'[오류] TCN 모델 파일이 아닙니다 (model={config.get("model")}). '
                  f'ksl_tcn_improved.pt 를 사용하세요.')
            sys.exit(1)

        model = TCNClassifier(
            config['input_dim'], config['tcn_channels'],
            config['num_classes'], config['dropout'], config['kernel_size'],
        )
        model.load_state_dict(ckpt['model_state'])
        model.eval().to(device)

        meta_path = model_path.replace('.pt', '_meta.pkl')
        if not os.path.exists(meta_path):
            print(f'[오류] 메타 파일 없음: {meta_path}'); sys.exit(1)
        label_names = joblib.load(meta_path)['label_names']
        tta_speeds  = TTA_SPEEDS if use_tta else [1.0]

        print(f'[TCN] {model_path}')
        print(f'  클래스: {len(label_names)}  디바이스: {device}')
        print(f'  온도: {temperature}  TTA: {tta_speeds}')

        def predict(kp_seq):
            # P1-B: 손 미감지 보간
            kp_seq = _fill_missing_hands(kp_seq)
            T = len(kp_seq)

            # 학습 데이터 포맷 재현:
            # 녹화된 T프레임을 앞에 두고 뒤를 0으로 패딩 → (148, 67, 3)
            # 속도 TTA: 각 speed로 T를 리샘플 후 제로패딩
            all_probs = []
            for speed in tta_speeds:
                eff_len = max(15, int(T * speed))

                if eff_len != T:
                    kp_adj = resample_keypoints(kp_seq, eff_len)
                else:
                    kp_adj = kp_seq

                kp_148 = np.zeros((148, 67, 3), dtype=np.float32)
                copy_len = min(eff_len, 148)
                kp_148[:copy_len] = kp_adj[:copy_len]

                feat = extract_improved_features(kp_148)
                x = torch.from_numpy(feat).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(x) / temperature, dim=1)[0].cpu().numpy()
                all_probs.append(probs)

            avg_probs = np.mean(all_probs, axis=0)
            top_idx   = avg_probs.argsort()[::-1][:5]
            return (label_names[top_idx[0]], float(avg_probs[top_idx[0]]),
                    [(float(avg_probs[i]), label_names[i]) for i in top_idx])

        return predict, label_names, 'TCN'

    # ── DTW+KNN (최후 폴백) ────────────────────
    try:
        from dtaidistance import dtw as _dtw  # noqa: F401
    except ImportError:
        print('[오류] dtaidistance 없음. pip install dtaidistance'); sys.exit(1)

    from features import extract_features_legacy
    data        = joblib.load(model_path)
    clf         = data['classifier']
    label_names = data['label_names']
    print(f'[DTW+KNN] {model_path}  템플릿={len(clf.templates)}')

    def predict(kp_seq):
        kp_seq = _fill_missing_hands(kp_seq)
        feat   = extract_features_legacy(kp_seq)
        return clf.predict_one(feat)

    return predict, label_names, 'DTW+KNN'


# ─────────────────────────────────────────────
# 화면 UI (PIL 기반 한글 렌더링)
# ─────────────────────────────────────────────
_FONT_PATHS = [
    '/System/Library/Fonts/AppleSDGothicNeo.ttc',
    '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
]


def _get_font(size):
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_status(frame, recording, frame_count, result_text, conf_text, top5, uncertain=False):
    h, w = frame.shape[:2]

    if recording:
        cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
        bar_w = int(min(frame_count / 60, 1.0) * (w - 20))
        cv2.rectangle(frame, (10, h - 15), (w - 10, h - 5), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, h - 15), (10 + bar_w, h - 5), (0, 0, 255), -1)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)

    def _put(pos, text, size=18, color=(255, 255, 255), bg=True):
        font = _get_font(size)
        x, y = pos
        if bg:
            try:
                bbox = draw.textbbox((x, y), text, font=font)
            except AttributeError:
                tw, th = draw.textsize(text, font=font)
                bbox = (x, y, x + tw, y + th)
            draw.rectangle([bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4], fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=(color[0], color[1], color[2]))

    if recording:
        _put((50, 18), f'REC  {frame_count} frames', size=18, color=(0, 0, 255))
    else:
        _put((10, 10), 'SPACE: 녹화시작  R: 취소  Q: 종료', size=16)

    if result_text:
        text_color = (0, 165, 255) if uncertain else (0, 255, 100)
        _put((10, h - 70), result_text, size=26, color=text_color)
        _put((10, h - 38), conf_text,   size=16, color=(200, 200, 200))

    if top5:
        _put((w - 260, 10), '--- Top-5 ---', size=15, color=(200, 200, 200))
        for i, (prob, lbl) in enumerate(top5):
            _put((w - 260, 35 + i * 22), f'{i + 1}. {lbl}  ({prob:.3f})',
                 size=14, color=(180, 220, 255))

    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)[:]


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main(model_path, temperature, use_tta):
    try:
        import mediapipe as mp
    except ImportError:
        print('[오류] mediapipe 없음. pip install mediapipe'); sys.exit(1)

    predict_fn, _, model_type = load_model(model_path, temperature=temperature, use_tta=use_tta)
    print('준비 완료! SPACE를 눌러 녹화 시작\n')

    mp_holistic   = mp.solutions.holistic
    mp_drawing    = mp.solutions.drawing_utils
    mp_draw_style = mp.solutions.drawing_styles
    mp_hands      = mp.solutions.hands

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[오류] 웹캠을 열 수 없습니다.'); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recording   = False
    frames_buf  = []
    result_text = conf_text = ''
    top5_result = []
    uncertain   = False
    MIN_FRAMES  = 30
    MAX_FRAMES  = 90

    def do_predict():
        nonlocal result_text, conf_text, top5_result, frames_buf, recording, uncertain
        kp_seq = np.array(frames_buf)

        # P2-A: 경계 트리밍 (앞뒤 10%)
        T      = len(kp_seq)
        trim_n = max(1, int(T * 0.10))
        if T - 2 * trim_n >= MIN_FRAMES:
            kp_seq = kp_seq[trim_n:-trim_n]

        n_tta = len(TTA_SPEEDS) if use_tta else 1
        print(f'예측 중... ({len(kp_seq)} 프레임, TTA {n_tta}회)', end='', flush=True)
        pred, conf, top5 = predict_fn(kp_seq)

        word = pred.split('_', 1)[-1] if '_' in pred else pred

        if conf < CONF_THRESHOLD:
            uncertain   = True
            result_text = f'불확실: {word}?'
            conf_text   = f'신뢰도 낮음: {conf * 100:.1f}%  |  {pred}'
        else:
            uncertain   = False
            result_text = f'예측: {word}'
            conf_text   = f'신뢰도: {conf * 100:.1f}%  |  {pred}'

        top5_result = top5
        frames_buf  = []
        recording   = False
        print(f'\n→ {pred}  ({conf * 100:.1f}%)' + (' [불확실]' if uncertain else ''))
        for i, (p, l) in enumerate(top5):
            print(f'  {i + 1}. {l}  {p:.4f}')

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.flip(frame, 1)

            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS,
                                      mp_draw_style.get_default_hand_landmarks_style(),
                                      mp_draw_style.get_default_hand_connections_style())
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS,
                                      mp_draw_style.get_default_hand_landmarks_style(),
                                      mp_draw_style.get_default_hand_connections_style())
            mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                      mp_holistic.POSE_CONNECTIONS,
                                      mp_draw_style.get_default_pose_landmarks_style())

            if recording:
                frames_buf.append(extract_keypoints(results))
                if len(frames_buf) >= MAX_FRAMES:
                    do_predict()

            draw_status(frame, recording, len(frames_buf),
                        result_text, conf_text, top5_result, uncertain)
            cv2.imshow(f'KSL 수어 인식 [{model_type}]', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                if not recording:
                    recording   = True
                    frames_buf  = []
                    result_text = conf_text = ''
                    top5_result = []
                    uncertain   = False
                    print('녹화 시작...')
                else:
                    if len(frames_buf) < MIN_FRAMES:
                        print(f'프레임 부족 ({len(frames_buf)}/{MIN_FRAMES})')
                        frames_buf = []
                        recording  = False
                    else:
                        do_predict()
            elif key in (ord('r'), ord('R')):
                frames_buf  = []
                recording   = False
                result_text = conf_text = ''
                top5_result = []
                uncertain   = False
                print('취소')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='auto',
                        help='TCN 모델 경로 (.pt) 또는 "auto" (기본값)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='소프트맥스 온도 (기본 1.0, 높을수록 분포 평탄화)')
    parser.add_argument('--no-tta', action='store_true',
                        help='속도 TTA 비활성화 (빠른 예측)')
    args = parser.parse_args()
    main(args.model, args.temperature, not args.no_tta)
