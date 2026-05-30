"""
KSL 수어 실시간 인식 - 웹캠 추론 스크립트
================================================
모델 자동 감지 우선순위:
  1. ksl_tcn_improved.pt   (TCN, 99%+ 정확도)
  2. ksl_lstm_model.pt     (BiLSTM)
  3. ksl_dtw_knn_model.pkl (DTW+KNN 폴백)

사용법:
  python realtime_inference.py
  python realtime_inference.py --model path/to/model.pt

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 특징 추출 (ksl_tcn_model.py 와 완전히 동일)
# ─────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]
FINGER_TIPS = [4, 8, 12, 16, 20]

def _safe_norm(x, axis=-1, keepdims=False, eps=1e-6):
    return np.linalg.norm(x, axis=axis, keepdims=keepdims) + eps

def _normalize_hand_batch(hand):
    """hand: (N, T, 21, 3)"""
    center = hand[:, :, 0:1, :]
    hand_c = hand - center
    scale  = np.linalg.norm(hand_c[:, :, 9, :], axis=-1, keepdims=True)
    scale  = scale[:, :, :, None]
    return hand_c / (scale + 1e-6), center.squeeze(2), scale.squeeze(2)

def _extract_angle_features(X_batch):
    """(N,T,67,3) → (N,T,40) 관절 각도 특징"""
    N, T = X_batch.shape[:2]
    features = np.zeros((N, T, 40), dtype=np.float32)
    for hand_i, hs in enumerate([0, 21]):
        hand = X_batch[:, :, hs:hs+21, :]
        hand_norm, _, _ = _normalize_hand_batch(hand)
        base = hand_i * 20
        for ci, (parent, child) in enumerate(HAND_CONNECTIONS):
            if parent == 0:
                diff = hand_norm[:, :, child] - hand_norm[:, :, parent]
                features[:, :, base+ci] = np.linalg.norm(diff, axis=-1)
            else:
                v1  = hand_norm[:, :, parent]   - hand_norm[:, :, parent-1]
                v2  = hand_norm[:, :, child]    - hand_norm[:, :, parent]
                n1  = np.linalg.norm(v1, axis=-1)
                n2  = np.linalg.norm(v2, axis=-1)
                cos = np.clip(np.sum(v1*v2, axis=-1) / (n1*n2+1e-6), -1.0, 1.0)
                features[:, :, base+ci] = np.arccos(cos)
    return features

def extract_improved_features_batch(X_batch):
    """
    X_batch: (N, T, 67, 3) → (N, T, feat_dim)
    ksl_tcn_model.py 의 extract_improved_features_batch 와 동일
    """
    X = X_batch.astype(np.float32)
    N, T = X.shape[:2]

    left  = X[:, :, 0:21, :]
    right = X[:, :, 21:42, :]
    left_norm,  left_wrist,  left_scale  = _normalize_hand_batch(left)
    right_norm, right_wrist, right_scale = _normalize_hand_batch(right)

    angle_feats = _extract_angle_features(X)

    left_tips  = left_norm[:, :, FINGER_TIPS, :].reshape(N, T, -1)
    right_tips = right_norm[:, :, FINGER_TIPS, :].reshape(N, T, -1)
    tip_feats  = np.concatenate([left_tips, right_tips], axis=-1).astype(np.float32)

    global_center = np.mean(X, axis=2, keepdims=True)
    global_scale  = _safe_norm(X - global_center, axis=-1, keepdims=True).mean(axis=2, keepdims=True)
    Xg = (X - global_center) / (global_scale + 1e-6)

    left_wrist_g  = Xg[:, :, 0, :]
    right_wrist_g = Xg[:, :, 21, :]
    wrist_rel  = right_wrist_g - left_wrist_g
    wrist_dist = _safe_norm(wrist_rel, axis=-1, keepdims=True)
    wrist_mid  = 0.5 * (left_wrist_g + right_wrist_g)
    wrist_feats = np.concatenate(
        [left_wrist_g, right_wrist_g, wrist_rel, wrist_dist, wrist_mid], axis=-1
    ).astype(np.float32)

    left_td  = _safe_norm(left_norm[:, :, FINGER_TIPS, :] - left_norm[:, :, 0:1, :], axis=-1)
    right_td = _safe_norm(right_norm[:, :, FINGER_TIPS, :] - right_norm[:, :, 0:1, :], axis=-1)
    hand_spread = np.concatenate([
        left_td.mean(axis=-1, keepdims=True),
        left_td.std(axis=-1, keepdims=True),
        right_td.mean(axis=-1, keepdims=True),
        right_td.std(axis=-1, keepdims=True),
        left_scale.reshape(N, T, -1).mean(axis=-1, keepdims=True),
        right_scale.reshape(N, T, -1).mean(axis=-1, keepdims=True),
    ], axis=-1).astype(np.float32)

    extra = []
    if X.shape[2] > 42:
        ex  = Xg[:, :, 42:, :]
        rc  = ex.mean(axis=2)
        rs  = ex.std(axis=2)
        lwr = left_wrist_g  - rc
        rwr = right_wrist_g - rc
        extra = [rc.astype(np.float32), rs.astype(np.float32),
                 lwr.astype(np.float32), rwr.astype(np.float32),
                 _safe_norm(lwr, axis=-1, keepdims=True).astype(np.float32),
                 _safe_norm(rwr, axis=-1, keepdims=True).astype(np.float32)]

    base  = np.concatenate([angle_feats, tip_feats, wrist_feats, hand_spread] + extra, axis=-1)
    delta = np.zeros_like(base); delta[:, 1:] = base[:, 1:] - base[:, :-1]
    feats = np.concatenate([base, delta], axis=-1).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

def extract_improved_features(sequence):
    """sequence: (T, 67, 3) → (T, feat_dim)"""
    return extract_improved_features_batch(sequence[np.newaxis])[0]

# 구형 LSTM 호환용 40-dim 추출
def _angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return 0.0
    return np.arccos(np.clip(np.dot(v1,v2)/(n1*n2), -1.0, 1.0))

def _extract_hand_angles_legacy(hand_kp):
    c = hand_kp[0]; h = (hand_kp - c) / (np.linalg.norm(hand_kp[9]-c)+1e-6)
    angles = []
    for parent, child in HAND_CONNECTIONS:
        if parent == 0:
            angles.append(np.linalg.norm(h[child]-h[parent]))
        else:
            angles.append(_angle_between(h[parent]-h[parent-1], h[child]-h[parent]))
    return np.array(angles, dtype=np.float32)

def extract_features_legacy(sequence):
    """구형 LSTM용: (T,67,3) → (T,40)"""
    return np.array([np.concatenate([
        _extract_hand_angles_legacy(sequence[f, 0:21]),
        _extract_hand_angles_legacy(sequence[f, 21:42])
    ]) for f in range(sequence.shape[0])], dtype=np.float32)

def resample_sequence(feat, target_len=148):
    T = feat.shape[0]
    if T == target_len: return feat
    idx = np.linspace(0, T-1, target_len)
    lo  = np.floor(idx).astype(int)
    hi  = np.minimum(lo+1, T-1)
    w   = (idx - lo)[:, None]
    return (feat[lo]*(1-w) + feat[hi]*w).astype(np.float32)


# ─────────────────────────────────────────────
# MediaPipe 키포인트 추출
# ─────────────────────────────────────────────
def extract_keypoints(results):
    """MediaPipe Holistic → (67, 3)"""
    lh = np.array([[lm.x,lm.y,lm.z] for lm in results.left_hand_landmarks.landmark],
                  dtype=np.float32) if results.left_hand_landmarks else np.zeros((21,3),np.float32)
    rh = np.array([[lm.x,lm.y,lm.z] for lm in results.right_hand_landmarks.landmark],
                  dtype=np.float32) if results.right_hand_landmarks else np.zeros((21,3),np.float32)
    pose = np.array([[lm.x,lm.y,lm.z] for lm in results.pose_landmarks.landmark[:25]],
                    dtype=np.float32) if results.pose_landmarks else np.zeros((25,3),np.float32)
    return np.concatenate([lh, rh, pose], axis=0)


# ─────────────────────────────────────────────
# 모델 아키텍처 (ksl_tcn_model.py 와 동일)
# ─────────────────────────────────────────────
class AttnPool1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim//2), nn.Tanh(), nn.Linear(dim//2, 1))
    def forward(self, x):
        return (x * torch.softmax(self.attn(x), dim=1)).sum(dim=1)

class AttnAvgMaxPool1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_pool = AttnPool1D(dim)
    def forward(self, x):
        return torch.cat([self.attn_pool(x), x.mean(dim=1), x.max(dim=1).values], dim=-1)

class Chomp1d(nn.Module):
    def __init__(self, s): super().__init__(); self.s = s
    def forward(self, x): return x[:,:,:-self.s].contiguous() if self.s else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, dilation, dropout):
        super().__init__()
        p = (ks-1)*dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch,out_ch,ks,padding=p,dilation=dilation), Chomp1d(p),
            nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch,out_ch,ks,padding=p,dilation=dilation), Chomp1d(p),
            nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    def forward(self, x):
        return F.gelu(self.net(x) + (x if self.downsample is None else self.downsample(x)))

class TCNClassifier(nn.Module):
    def __init__(self, input_dim, channels, num_classes, dropout, kernel_size=5):
        super().__init__()
        layers, in_ch = [], input_dim
        for d in [1,2,4,8,16]:
            layers.append(TemporalBlock(in_ch, channels, kernel_size, d, dropout))
            in_ch = channels
        self.tcn  = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(channels)
        self.pool = AttnAvgMaxPool1D(channels)
        self.head = nn.Sequential(
            nn.Linear(channels*3, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.tcn(x.transpose(1,2)).transpose(1,2)
        return self.head(self.pool(self.norm(x)))

class BiLSTMImproved(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers>1 else 0.0)
        d = hidden_dim*2
        self.norm = nn.LayerNorm(d)
        self.pool = AttnAvgMaxPool1D(d)
        self.head = nn.Sequential(nn.Linear(d*3,512), nn.LayerNorm(512), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(512, num_classes))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.pool(self.norm(out)))

class GRUImproved(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers>1 else 0.0)
        d = hidden_dim*2
        self.norm = nn.LayerNorm(d)
        self.pool = AttnAvgMaxPool1D(d)
        self.head = nn.Sequential(nn.Linear(d*3,512), nn.LayerNorm(512), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(512, num_classes))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(self.pool(self.norm(out)))

# 구형 BiLSTM+Attention (ksl_lstm_model.pt 호환)
class BiLSTMAttentionLegacy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers>1 else 0.0)
        self.attn = nn.Linear(hidden_dim*2, 1)
        self.norm = nn.LayerNorm(hidden_dim*2)
        self.head = nn.Sequential(nn.Linear(hidden_dim*2,512), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(512, num_classes))
    def forward(self, x):
        out, _  = self.lstm(x)
        ctx = (out * torch.softmax(self.attn(out), dim=1)).sum(dim=1)
        return self.head(self.norm(ctx))


# ─────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────
def _build_improved_model(config):
    m = config['model']
    if m == 'tcn':
        return TCNClassifier(config['input_dim'], config['tcn_channels'],
                             config['num_classes'], config['dropout'], config['kernel_size'])
    if m == 'bilstm':
        return BiLSTMImproved(config['input_dim'], config['hidden_dim'],
                              config['num_layers'], config['num_classes'], config['dropout'])
    if m == 'gru':
        return GRUImproved(config['input_dim'], config['hidden_dim'],
                           config['num_layers'], config['num_classes'], config['dropout'])
    raise ValueError(f'알 수 없는 모델: {m}')

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 자동 감지 우선순위: tcn_improved > lstm > dtw
    tcn_pt   = os.path.join(SCRIPT_DIR, 'ksl_tcn_improved.pt')
    tcn_meta = os.path.join(SCRIPT_DIR, 'ksl_tcn_improved_meta.pkl')
    lstm_pt  = os.path.join(SCRIPT_DIR, 'ksl_lstm_model.pt')
    lstm_meta= os.path.join(SCRIPT_DIR, 'ksl_lstm_meta.pkl')
    dtw_pkl  = os.path.join(SCRIPT_DIR, 'ksl_dtw_knn_model.pkl')

    if model_path == 'auto':
        if os.path.exists(tcn_pt):   model_path = tcn_pt
        elif os.path.exists(lstm_pt): model_path = lstm_pt
        elif os.path.exists(dtw_pkl): model_path = dtw_pkl
        else:
            print('[오류] 모델 파일을 찾을 수 없습니다.'); sys.exit(1)

    # ── TCN / BiLSTM-improved / GRU (ksl_tcn_model.py 계열) ──
    if model_path.endswith('.pt'):
        ckpt   = torch.load(model_path, map_location='cpu', weights_only=False)
        config = ckpt['config']
        is_improved = 'model' in config  # 구형 LSTM은 'model' 키 없음

        if is_improved:
            model = _build_improved_model(config)
            meta_path = model_path.replace('.pt', '_meta.pkl')
            feat_fn   = extract_improved_features
            model_type = config['model'].upper()
        else:
            model = BiLSTMAttentionLegacy(
                config['input_dim'], config['hidden_dim'],
                config['num_layers'], config['num_classes'], config['dropout'])
            meta_path  = lstm_meta
            feat_fn    = extract_features_legacy
            model_type = 'LSTM(legacy)'

        model.load_state_dict(ckpt['model_state'])
        model.eval().to(device)

        if not os.path.exists(meta_path):
            print(f'[오류] 메타 파일 없음: {meta_path}'); sys.exit(1)
        meta        = joblib.load(meta_path)
        label_names = meta['label_names']
        print(f'[{model_type}] {model_path}')
        print(f'  클래스: {len(label_names)}  디바이스: {device}')

        def predict(kp_seq):
            feat = feat_fn(kp_seq)                     # (T, D)
            feat = resample_sequence(feat, 148)        # → (148, D)
            x    = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            top_idx = probs.argsort()[::-1][:5]
            return (label_names[top_idx[0]], float(probs[top_idx[0]]),
                    [(float(probs[i]), label_names[i]) for i in top_idx])

        return predict, label_names, model_type

    # ── DTW+KNN ──
    try:
        from dtaidistance import dtw as dtw_lib
    except ImportError:
        print('[오류] dtaidistance 없음. pip install dtaidistance'); sys.exit(1)

    data        = joblib.load(model_path)
    clf         = data['classifier']
    label_names = data['label_names']
    print(f'[DTW+KNN] {model_path}  템플릿={len(clf.templates)}')

    def predict(kp_seq):
        feat = extract_features_legacy(kp_seq)
        return clf.predict_one(feat)

    return predict, label_names, 'DTW+KNN'


# ─────────────────────────────────────────────
# 화면 UI
# ─────────────────────────────────────────────
def put_text(img, text, pos, scale=0.7, color=(255,255,255), thickness=2, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    if bg:
        cv2.rectangle(img, (x-4,y-th-4), (x+tw+4,y+bl+4), (0,0,0), -1)
    cv2.putText(img, text, (x,y), font, scale, color, thickness, cv2.LINE_AA)

def draw_status(frame, recording, frame_count, result_text, conf_text, top5):
    h, w = frame.shape[:2]
    if recording:
        cv2.circle(frame, (30,30), 12, (0,0,255), -1)
        put_text(frame, f'REC  {frame_count} frames', (50,40), color=(0,0,255))
        bar_w = int(min(frame_count/60,1.0)*(w-20))
        cv2.rectangle(frame, (10,h-15),(w-10,h-5),(50,50,50),-1)
        cv2.rectangle(frame, (10,h-15),(10+bar_w,h-5),(0,0,255),-1)
    else:
        put_text(frame, 'SPACE: 녹화시작  R: 취소  Q: 종료', (10,30), scale=0.55)
    if result_text:
        put_text(frame, result_text, (10,h-60), scale=0.9, color=(0,255,100))
        put_text(frame, conf_text,   (10,h-30), scale=0.65, color=(200,200,200))
    if top5:
        put_text(frame, '--- Top-5 ---', (w-240,30), scale=0.55, color=(200,200,200))
        for i,(prob,lbl) in enumerate(top5):
            put_text(frame, f'{i+1}. {lbl}  ({prob:.3f})', (w-240,55+i*22),
                     scale=0.5, color=(180,220,255))


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main(model_path):
    try:
        import mediapipe as mp
    except ImportError:
        print('[오류] mediapipe 없음. pip install mediapipe'); sys.exit(1)

    predict_fn, _, model_type = load_model(model_path)
    print(f'준비 완료! SPACE를 눌러 녹화 시작\n')

    mp_holistic   = mp.solutions.holistic
    mp_drawing    = mp.solutions.drawing_utils
    mp_draw_style = mp.solutions.drawing_styles
    mp_hands      = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[오류] 웹캠을 열 수 없습니다.'); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recording   = False
    frames_buf  = []       # raw keypoints (T, 67, 3)
    result_text = conf_text = ''
    top5_result = []
    MIN_FRAMES  = 10
    MAX_FRAMES  = 90

    def do_predict():
        nonlocal result_text, conf_text, top5_result, frames_buf, recording
        kp_seq = np.array(frames_buf)  # (T, 67, 3)
        print(f'예측 중... ({len(frames_buf)} 프레임)', end='', flush=True)
        pred, conf, top5 = predict_fn(kp_seq)
        word = pred.split('_',1)[-1] if '_' in pred else pred
        result_text = f'예측: {word}'
        conf_text   = f'신뢰도: {conf*100:.1f}%  |  {pred}'
        top5_result = top5
        frames_buf  = []
        recording   = False
        print(f'\n→ {pred}  ({conf*100:.1f}%)')
        for i,(p,l) in enumerate(top5):
            print(f'  {i+1}. {l}  {p:.4f}')

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # MediaPipe는 원본 방향으로 처리 → 학습 데이터와 손 방향 일치
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.flip(frame, 1)  # 화면만 거울 모드

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

            draw_status(frame, recording, len(frames_buf), result_text, conf_text, top5_result)
            cv2.imshow(f'KSL 수어 인식 [{model_type}]', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                if not recording:
                    recording = True; frames_buf = []
                    result_text = conf_text = ''; top5_result = []
                    print('녹화 시작...')
                else:
                    if len(frames_buf) < MIN_FRAMES:
                        print(f'프레임 부족 ({len(frames_buf)}/{MIN_FRAMES})')
                        frames_buf = []; recording = False
                    else:
                        do_predict()
            elif key in (ord('r'), ord('R')):
                frames_buf = []; recording = False
                result_text = conf_text = ''; top5_result = []
                print('취소')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='auto',
                        help='모델 경로 (.pt/.pkl) 또는 "auto" (기본값)')
    args = parser.parse_args()
    main(args.model)
