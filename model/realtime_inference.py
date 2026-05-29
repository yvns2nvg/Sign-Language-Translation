"""
KSL 수어 실시간 인식 - 웹캠 추론 스크립트
================================================
LSTM 모델(ksl_lstm_model.pt)과 DTW+KNN 모델(ksl_dtw_knn_model.pkl) 모두 지원.
LSTM 모델이 있으면 자동으로 사용합니다 (추론 속도 0.1초 이하).

사용법:
  python realtime_inference.py
  python realtime_inference.py --model path/to/ksl_lstm_model.pt

조작:
  SPACE  : 녹화 시작 / 중지 후 예측
  R      : 녹화 취소 (버퍼 초기화)
  Q / ESC: 종료
"""

import argparse
import os
import sys
import numpy as np
import cv2
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 관절 각도 특징 추출 (학습 코드와 동일) ──
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]
FEAT_DIM = 40

def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))

def extract_hand_angles(hand_kp):
    angles = []
    for (parent, child) in HAND_CONNECTIONS:
        if parent == 0:
            angles.append(np.linalg.norm(hand_kp[child] - hand_kp[parent]))
        else:
            v1 = hand_kp[parent] - hand_kp[parent - 1]
            v2 = hand_kp[child]  - hand_kp[parent]
            angles.append(angle_between(v1, v2))
    return np.array(angles, dtype=np.float32)

def extract_features(sequence):
    """sequence: (T, 67, 3) → (T, 40)"""
    return np.array([
        np.concatenate([
            extract_hand_angles(sequence[f, 0:21, :]),
            extract_hand_angles(sequence[f, 21:42, :])
        ])
        for f in range(sequence.shape[0])
    ], dtype=np.float32)

def resample_sequence(feat, target_len=148):
    """특징 시퀀스를 target_len 프레임으로 보간"""
    T = feat.shape[0]
    if T == target_len:
        return feat
    idx = np.linspace(0, T - 1, target_len)
    lo  = np.floor(idx).astype(int)
    hi  = np.minimum(lo + 1, T - 1)
    w   = (idx - lo)[:, None]
    return (feat[lo] * (1 - w) + feat[hi] * w).astype(np.float32)


# ── MediaPipe 키포인트 추출 ──
def extract_keypoints(results):
    """MediaPipe Holistic 결과 → (67, 3) numpy 배열"""
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
    else:
        lh = np.zeros((21, 3), dtype=np.float32)

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
    else:
        rh = np.zeros((21, 3), dtype=np.float32)

    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.pose_landmarks.landmark[:25]], dtype=np.float32)
    else:
        pose = np.zeros((25, 3), dtype=np.float32)

    return np.concatenate([lh, rh, pose], axis=0)  # (67, 3)


# ── LSTM 모델 정의 (train_lstm.py와 동일) ──
def _build_lstm_model(config):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    class BiLSTMAttention(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.attn = nn.Linear(hidden_dim * 2, 1)
            self.norm = nn.LayerNorm(hidden_dim * 2)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            out, _  = self.lstm(x)
            attn_w  = torch.softmax(self.attn(out), dim=1)
            context = (out * attn_w).sum(dim=1)
            context = self.norm(context)
            return self.head(context)

    return BiLSTMAttention(
        config['input_dim'], config['hidden_dim'],
        config['num_layers'], config['num_classes'], config['dropout']
    )


# ── 모델 로드 ──
def load_model(model_path):
    """
    .pt  → LSTM 모델
    .pkl → DTW+KNN 모델
    자동 감지: 같은 폴더에서 .pt 우선
    """
    # 자동 감지: 경로가 기본값이면 LSTM .pt 우선 탐색
    lstm_path = os.path.join(SCRIPT_DIR, 'ksl_lstm_model.pt')
    meta_path = os.path.join(SCRIPT_DIR, 'ksl_lstm_meta.pkl')
    dtw_path  = os.path.join(SCRIPT_DIR, 'ksl_dtw_knn_model.pkl')

    # 명시적으로 .pt 경로가 주어졌거나, 자동 감지로 .pt가 있는 경우
    use_lstm = model_path.endswith('.pt') or (
        not model_path.endswith('.pkl') and os.path.exists(lstm_path)
    )

    if use_lstm:
        pt_path = model_path if model_path.endswith('.pt') else lstm_path
        mk_path = meta_path
        if not os.path.exists(pt_path):
            print(f'[오류] LSTM 모델 파일 없음: {pt_path}')
            sys.exit(1)
        if not os.path.exists(mk_path):
            print(f'[오류] 메타 파일 없음: {mk_path}')
            sys.exit(1)

        try:
            import torch
        except ImportError:
            print('[오류] PyTorch 없음. pip install torch')
            sys.exit(1)

        print(f'[LSTM 모델 로드] {pt_path}')
        ckpt   = torch.load(pt_path, map_location='cpu')
        config = ckpt['config']
        model  = _build_lstm_model(config)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = model.to(device)

        meta         = joblib.load(mk_path)
        label_names  = meta['label_names']
        le           = meta['label_encoder']

        print(f'  클래스 수: {len(label_names)}  디바이스: {device}')

        def predict(feat_seq):
            """feat_seq: (T, 40) → (pred_label, confidence, top5)"""
            feat_148 = resample_sequence(feat_seq, target_len=148)
            x = torch.from_numpy(feat_148).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            top_idx = probs.argsort()[::-1][:5]
            pred    = label_names[top_idx[0]]
            conf    = float(probs[top_idx[0]])
            top5    = [(float(probs[i]), label_names[i]) for i in top_idx]
            return pred, conf, top5

        return predict, label_names, 'lstm'

    else:
        # DTW+KNN 폴백
        pkl = model_path if model_path.endswith('.pkl') else dtw_path
        if not os.path.exists(pkl):
            print(f'[오류] 모델 파일 없음: {pkl}')
            sys.exit(1)

        try:
            from dtaidistance import dtw as dtw_lib
        except ImportError:
            print('[오류] dtaidistance 없음. pip install dtaidistance')
            sys.exit(1)

        print(f'[DTW+KNN 모델 로드] {pkl}')
        data        = joblib.load(pkl)
        clf         = data['classifier']
        label_names = data['label_names']
        print(f'  클래스 수: {len(label_names)}  템플릿 수: {len(clf.templates)}')

        def predict(feat_seq):
            pred, conf, top5 = clf.predict_one(feat_seq)
            return pred, conf, top5

        return predict, label_names, 'dtw'


# ── 화면 UI ──
def put_text(img, text, pos, scale=0.7, color=(255,255,255), thickness=2, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    if bg:
        cv2.rectangle(img, (x-4, y-th-4), (x+tw+4, y+bl+4), (0,0,0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_status(frame, recording, frame_count, result_text, conf_text, top5):
    h, w = frame.shape[:2]
    if recording:
        cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
        put_text(frame, f'REC  {frame_count} frames', (50, 40), color=(0,0,255))
        bar_w = int(min(frame_count / 60, 1.0) * (w - 20))
        cv2.rectangle(frame, (10, h-15), (w-10,  h-5), (50,50,50), -1)
        cv2.rectangle(frame, (10, h-15), (10+bar_w, h-5), (0,0,255), -1)
    else:
        put_text(frame, 'SPACE: 녹화시작  R: 취소  Q: 종료', (10, 30), scale=0.55)

    if result_text:
        put_text(frame, result_text, (10, h-60), scale=0.9, color=(0,255,100))
        put_text(frame, conf_text,   (10, h-30), scale=0.65, color=(200,200,200))

    if top5:
        put_text(frame, '--- Top-5 ---', (w-240, 30), scale=0.55, color=(200,200,200))
        for idx, (prob, lbl) in enumerate(top5):
            put_text(frame, f'{idx+1}. {lbl}  ({prob:.3f})', (w-240, 55+idx*22),
                     scale=0.5, color=(180,220,255))


# ── 메인 ──
def main(model_path):
    try:
        import mediapipe as mp
    except ImportError:
        print('[오류] mediapipe 없음. pip install mediapipe')
        sys.exit(1)

    predict_fn, label_names, model_type = load_model(model_path)
    print(f'모델 타입: {model_type.upper()}\n준비 완료! SPACE를 눌러 녹화 시작\n')

    mp_holistic   = mp.solutions.holistic
    mp_drawing    = mp.solutions.drawing_utils
    mp_draw_style = mp.solutions.drawing_styles
    mp_hands      = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[오류] 웹캠을 열 수 없습니다.')
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recording   = False
    frames_buf  = []
    result_text = ''
    conf_text   = ''
    top5_result = []
    MIN_FRAMES  = 10
    MAX_FRAMES  = 90

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw_style.get_default_hand_landmarks_style(),
                mp_draw_style.get_default_hand_connections_style())
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw_style.get_default_hand_landmarks_style(),
                mp_draw_style.get_default_hand_connections_style())
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_draw_style.get_default_pose_landmarks_style())

            kp = extract_keypoints(results)
            if recording:
                frames_buf.append(kp)
                if len(frames_buf) >= MAX_FRAMES:
                    _do_predict(frames_buf, predict_fn)
                    result_text, conf_text, top5_result = _format_result(
                        *predict_fn(extract_features(np.array(frames_buf))))
                    frames_buf = []
                    recording  = False

            draw_status(frame, recording, len(frames_buf), result_text, conf_text, top5_result)
            cv2.imshow('KSL 수어 인식', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                if not recording:
                    recording   = True
                    frames_buf  = []
                    result_text = conf_text = ''
                    top5_result = []
                    print('녹화 시작...')
                else:
                    if len(frames_buf) < MIN_FRAMES:
                        print(f'프레임 부족 ({len(frames_buf)}/{MIN_FRAMES})')
                    else:
                        feat = extract_features(np.array(frames_buf))
                        print(f'예측 중... ({len(frames_buf)} 프레임)', end='', flush=True)
                        pred, conf, top5 = predict_fn(feat)
                        word = pred.split('_', 1)[-1] if '_' in pred else pred
                        result_text = f'예측: {word}'
                        conf_text   = f'신뢰도: {conf*100:.1f}%  |  {pred}'
                        top5_result = top5
                        print(f'\n→ {pred}  ({conf*100:.1f}%)')
                        for i, (p, l) in enumerate(top5):
                            print(f'  {i+1}. {l}  {p:.4f}')
                    frames_buf = []
                    recording  = False
            elif key in (ord('r'), ord('R')):
                frames_buf = []; recording = False
                result_text = conf_text = ''; top5_result = []
                print('취소')

    cap.release()
    cv2.destroyAllWindows()


def _do_predict(frames_buf, predict_fn):
    pass  # MAX_FRAMES 자동 예측 시 사용 (현재는 키 입력 방식 사용)

def _format_result(pred, conf, top5):
    word = pred.split('_', 1)[-1] if '_' in pred else pred
    return f'예측: {word}', f'신뢰도: {conf*100:.1f}%  |  {pred}', top5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KSL 수어 실시간 인식')
    parser.add_argument('--model', default='auto',
                        help='모델 파일 경로 (.pt 또는 .pkl). 기본값: 자동 감지')
    args = parser.parse_args()
    main(args.model)
