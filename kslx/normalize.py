"""정규화: 목 원점 이동 + 어깨너비 스케일, 리샘플링, 속도 특징.

레이아웃/스케일 파라미터는 kslx/layout.py 를 따른다 (89점, pose13 안의
neck=index1, r_shoulder=index2, l_shoulder=index5).
"""

from __future__ import annotations

import numpy as np

from kslx.layout import POSE13_NECK_IDX, POSE13_R_SHOULDER_IDX, POSE13_L_SHOULDER_IDX

# AI Hub 키포인트는 OpenPose 컨벤션대로 미검출 지점을 (0,0,conf=0) 으로 채운다.
# 한 프레임에서 양쪽 어깨가 동시에 미검출이면 shoulder_vec 이 (0,0)이 되어
# 스케일이 0에 붕괴한다 — 이때 예전에는 고정 최소값(1e-3)으로 나눠서 그 프레임의
# (미검출과 무관한) 다른 관절 좌표까지 최대 100만배까지 폭발시키는 버그가 있었다
# (실측: word_271.npz 의 |X| 최댓값이 1.2e6 까지 나갔다). 고정 floor 대신
# "그 클립 안에서 정상적으로 검출된 프레임들의 중앙값 스케일" 로 대체한다.
DEGENERATE_SCALE_PX = 5.0   # 이보다 작으면 검출 실패로 간주 (실제 어깨너비는 항상 수십~수백 px)
MIN_SCALE = 1e-3            # 클립 전체가 미검출인 극단적 예외에서만 쓰는 최후 방어선


def center_and_scale(seq: np.ndarray) -> np.ndarray:
    """seq: (T, 89, 2). 프레임마다 목을 원점으로, 어깨너비로 스케일.

    프레임별로 정규화하는 이유: 카메라와의 거리/위치가 take 마다, 심지어 같은
    클립 안에서도(제스처 중 상체가 움직이면) 달라지기 때문이다.
    """
    neck = seq[:, POSE13_NECK_IDX:POSE13_NECK_IDX + 1, :]  # (T,1,2)
    centered = seq - neck
    shoulder_vec = (seq[:, POSE13_R_SHOULDER_IDX, :] - seq[:, POSE13_L_SHOULDER_IDX, :])  # (T,2)
    scale = np.linalg.norm(shoulder_vec, axis=-1)  # (T,)

    valid = scale >= DEGENERATE_SCALE_PX
    if valid.any():
        fallback = np.median(scale[valid])
    else:
        fallback = MIN_SCALE  # 클립 전체가 미검출 — 정상적인 fallback 기준이 없음
    scale = np.where(valid, scale, fallback)
    scale = np.maximum(scale, MIN_SCALE)
    return centered / scale[:, None, None]


def resample_time(seq: np.ndarray, t_out: int) -> np.ndarray:
    """(T_in, ...) -> (t_out, ...) 선형 보간. T_in==1 이면 반복."""
    t_in = seq.shape[0]
    if t_in == t_out:
        return seq
    if t_in == 1:
        return np.repeat(seq, t_out, axis=0)
    x_in = np.linspace(0.0, 1.0, t_in)
    x_out = np.linspace(0.0, 1.0, t_out)
    flat = seq.reshape(t_in, -1)
    out = np.empty((t_out, flat.shape[1]), dtype=seq.dtype)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(x_out, x_in, flat[:, j])
    return out.reshape((t_out,) + seq.shape[1:])


def velocity(seq: np.ndarray) -> np.ndarray:
    """(T, ...) -> (T, ...) 프레임간 차분. 첫 프레임은 0."""
    v = np.zeros_like(seq)
    v[1:] = seq[1:] - seq[:-1]
    return v


def featurize(seq_xy: np.ndarray, t_out: int = 64,
              sign_span: tuple[int, int] | None = None) -> np.ndarray:
    """(T_raw, 89, 2) 원시 좌표 -> (t_out, 356) 학습 입력 피처.

    356 = 89*2(위치) + 89*2(속도). sign_span 이 주어지면 그 구간만 잘라 쓴다
    (형태소 어노테이션 기반 수어 구간 크롭 — 배경/정지 프레임 제거).
    """
    if sign_span is not None:
        s, e = sign_span
        seq_xy = seq_xy[s:e]
        if seq_xy.shape[0] == 0:
            raise ValueError("empty sign span after crop")
    norm = center_and_scale(seq_xy)
    # 개별 관절(어깨 이외) 미검출로 인한 잔여 이상치에 대한 최후 방어선.
    # 정상 신호는 어깨너비 단위로 몇 배를 넘지 않는다.
    norm = np.clip(norm, -20.0, 20.0)
    norm = resample_time(norm, t_out)
    vel = velocity(norm)
    pos_flat = norm.reshape(t_out, -1)
    vel_flat = vel.reshape(t_out, -1)
    return np.concatenate([pos_flat, vel_flat], axis=-1).astype(np.float32)


FEATURE_DIM = 89 * 2 * 2  # 356
N_POINTS = FEATURE_DIM // 4  # 89


def unpack_position(feat):
    """(..., T, 356) -> (..., T, 89, 2) 위치 채널만 (torch/np 둘 다 동작).

    kslx.augment 와 kslx.eval_robust 가 이미 정규화+리샘플된 피처의 위치 성분에
    변형을 걸 때 공용으로 쓴다 (속도는 변형 후 repack_from_position 이 다시 계산).
    """
    *lead, t, _ = feat.shape
    pos_flat = feat[..., : N_POINTS * 2]
    return pos_flat.reshape(*lead, t, N_POINTS, 2)


def repack_from_position(pos):
    """(..., T, 89, 2) -> (..., T, 356). 속도는 여기서 재계산한다."""
    import torch as _torch
    if isinstance(pos, _torch.Tensor):
        vel = _torch.zeros_like(pos)
        vel[..., 1:, :, :] = pos[..., 1:, :, :] - pos[..., :-1, :, :]
        cat = _torch.cat
    else:
        vel = np.zeros_like(pos)
        vel[..., 1:, :, :] = pos[..., 1:, :, :] - pos[..., :-1, :, :]
        cat = np.concatenate
    *lead, t, p, c = pos.shape
    pos_flat = pos.reshape(*lead, t, p * c)
    vel_flat = vel.reshape(*lead, t, p * c)
    return cat([pos_flat, vel_flat], axis=-1)
