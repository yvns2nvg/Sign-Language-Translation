"""
KSL 수어 인식 — 공통 특징 추출 모듈
=====================================
학습(train_ksl_tcn.py)과 추론(realtime_inference.py)이 이 파일을 공유한다.
특징 추출 로직을 변경할 경우 반드시 이 파일 하나만 수정할 것.

주요 심볼:
  extract_improved_features_batch(X)  → (N, T, 206)  TCN/BiLSTM용
  extract_improved_features(seq)      → (T, 206)      단일 시퀀스 래퍼
  extract_features_legacy(seq)        → (T, 40)       구형 LSTM용
  resample_sequence(feat, target=148) → (target, D)   시간축 리샘플링
"""

import numpy as np

# ── 상수 ───────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGER_TIPS = [4, 8, 12, 16, 20]


# ── 내부 헬퍼 ──────────────────────────────────────
def _safe_norm(x, axis=-1, keepdims=False, eps=1e-6):
    return np.linalg.norm(x, axis=axis, keepdims=keepdims) + eps


def _normalize_hand_batch(hand):
    """hand: (N, T, 21, 3) → 손목 원점 + 중지MCP 스케일 정규화"""
    center = hand[:, :, 0:1, :]
    hand_c = hand - center
    scale = np.linalg.norm(hand_c[:, :, 9, :], axis=-1, keepdims=True)
    scale = scale[:, :, :, None]
    return hand_c / (scale + 1e-6), center.squeeze(2), scale.squeeze(2)


def _extract_angle_features(X_batch):
    """(N, T, 67, 3) → (N, T, 40) 관절 각도/거리 특징"""
    N, T = X_batch.shape[:2]
    features = np.zeros((N, T, 40), dtype=np.float32)
    for hand_i, hs in enumerate([0, 21]):
        hand = X_batch[:, :, hs:hs + 21, :]
        hand_norm, _, _ = _normalize_hand_batch(hand)
        base = hand_i * 20
        for ci, (parent, child) in enumerate(HAND_CONNECTIONS):
            if parent == 0:
                diff = hand_norm[:, :, child] - hand_norm[:, :, parent]
                features[:, :, base + ci] = np.linalg.norm(diff, axis=-1)
            else:
                v1 = hand_norm[:, :, parent] - hand_norm[:, :, parent - 1]
                v2 = hand_norm[:, :, child] - hand_norm[:, :, parent]
                n1 = np.linalg.norm(v1, axis=-1)
                n2 = np.linalg.norm(v2, axis=-1)
                cos = np.clip(np.sum(v1 * v2, axis=-1) / (n1 * n2 + 1e-6), -1.0, 1.0)
                features[:, :, base + ci] = np.arccos(cos)
    return features


# ── 공개 API ───────────────────────────────────────
def extract_improved_features_batch(X_batch):
    """
    TCN/BiLSTM 학습 및 추론에 사용하는 개선 특징 추출.

    X_batch : (N, T, 67, 3)  왼손21 + 오른손21 + 포즈25
    반환     : (N, T, 206)

    특징 그룹:
      관절각도(40) + 손끝좌표(30) + 손목관계(13) + 손크기(6)
      + 몸통참조(14) + delta(103) = 206
    """
    X = X_batch.astype(np.float32)
    N, T = X.shape[:2]

    left = X[:, :, 0:21, :]
    right = X[:, :, 21:42, :]
    left_norm, left_wrist, left_scale = _normalize_hand_batch(left)
    right_norm, right_wrist, right_scale = _normalize_hand_batch(right)

    angle_feats = _extract_angle_features(X)

    left_tips = left_norm[:, :, FINGER_TIPS, :].reshape(N, T, -1)
    right_tips = right_norm[:, :, FINGER_TIPS, :].reshape(N, T, -1)
    tip_feats = np.concatenate([left_tips, right_tips], axis=-1).astype(np.float32)

    global_center = np.mean(X, axis=2, keepdims=True)
    global_scale = _safe_norm(X - global_center, axis=-1, keepdims=True).mean(axis=2, keepdims=True)
    Xg = (X - global_center) / (global_scale + 1e-6)

    left_wrist_g = Xg[:, :, 0, :]
    right_wrist_g = Xg[:, :, 21, :]
    wrist_rel = right_wrist_g - left_wrist_g
    wrist_dist = _safe_norm(wrist_rel, axis=-1, keepdims=True)
    wrist_mid = 0.5 * (left_wrist_g + right_wrist_g)
    wrist_feats = np.concatenate(
        [left_wrist_g, right_wrist_g, wrist_rel, wrist_dist, wrist_mid], axis=-1
    ).astype(np.float32)  # 3+3+3+1+3 = 13

    left_td = _safe_norm(left_norm[:, :, FINGER_TIPS, :] - left_norm[:, :, 0:1, :], axis=-1)
    right_td = _safe_norm(right_norm[:, :, FINGER_TIPS, :] - right_norm[:, :, 0:1, :], axis=-1)
    hand_spread = np.concatenate([
        left_td.mean(axis=-1, keepdims=True),
        left_td.std(axis=-1, keepdims=True),
        right_td.mean(axis=-1, keepdims=True),
        right_td.std(axis=-1, keepdims=True),
        left_scale.reshape(N, T, -1).mean(axis=-1, keepdims=True),
        right_scale.reshape(N, T, -1).mean(axis=-1, keepdims=True),
    ], axis=-1).astype(np.float32)  # 6

    extra = []
    if X.shape[2] > 42:
        ex = Xg[:, :, 42:, :]
        rc = ex.mean(axis=2)
        rs = ex.std(axis=2)
        lwr = left_wrist_g - rc
        rwr = right_wrist_g - rc
        extra = [
            rc.astype(np.float32), rs.astype(np.float32),
            lwr.astype(np.float32), rwr.astype(np.float32),
            _safe_norm(lwr, axis=-1, keepdims=True).astype(np.float32),
            _safe_norm(rwr, axis=-1, keepdims=True).astype(np.float32),
        ]  # 3+3+3+3+1+1 = 14

    base = np.concatenate([angle_feats, tip_feats, wrist_feats, hand_spread] + extra, axis=-1)
    delta = np.zeros_like(base)
    delta[:, 1:] = base[:, 1:] - base[:, :-1]
    feats = np.concatenate([base, delta], axis=-1).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_improved_features(sequence):
    """단일 시퀀스 래퍼. sequence: (T, 67, 3) → (T, 206)"""
    return extract_improved_features_batch(sequence[np.newaxis])[0]


def resample_sequence(feat, target_len=148):
    """선형 보간으로 시퀀스 길이를 target_len으로 맞춤."""
    T = feat.shape[0]
    if T == target_len:
        return feat
    idx = np.linspace(0, T - 1, target_len)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (idx - lo)[:, None]
    return (feat[lo] * (1 - w) + feat[hi] * w).astype(np.float32)


def resample_keypoints(kp, target_len=148):
    """
    원시 키포인트 리샘플링. delta 계산 전에 호출해야 한다.

    kp: (T, K, 3)  — 67개 키포인트 또는 임의의 K
    반환: (target_len, K, 3)
    """
    T, K, C = kp.shape
    if T == target_len:
        return kp
    flat = kp.reshape(T, K * C)
    resampled = resample_sequence(flat, target_len)
    return resampled.reshape(target_len, K, C)


# ── 구형 LSTM 호환 (40-dim, 비정규화) ─────────────────
def _angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))


def _extract_hand_angles_legacy(hand_kp):
    c = hand_kp[0]
    h = (hand_kp - c) / (np.linalg.norm(hand_kp[9] - c) + 1e-6)
    angles = []
    for parent, child in HAND_CONNECTIONS:
        if parent == 0:
            angles.append(np.linalg.norm(h[child] - h[parent]))
        else:
            angles.append(_angle_between(h[parent] - h[parent - 1], h[child] - h[parent]))
    return np.array(angles, dtype=np.float32)


def extract_features_legacy(sequence):
    """구형 BiLSTM 추론용: (T, 67, 3) → (T, 40)"""
    return np.array([
        np.concatenate([
            _extract_hand_angles_legacy(sequence[f, 0:21]),
            _extract_hand_angles_legacy(sequence[f, 21:42]),
        ])
        for f in range(sequence.shape[0])
    ], dtype=np.float32)
