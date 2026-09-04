"""학습 시간 증강 — 배치 단위, GPU 이전 전에 CPU 텐서에 적용한다.

eval_robust.py 의 변형과 종류는 같지만 거기서는 "한 번에 하나씩" 걸어서
열화를 측정하는 게 목적이고, 여기서는 매 배치마다 샘플별로 독립적인 확률로
여러 개를 동시에 적용해 실사용 변형 조합에 대한 강건성을 학습시키는 게 목적이다.

★ yaw 는 실제 3D 회전이 아니라 2D 전단 근사다 (kslx.eval_robust 상단 주석과 동일한 이유).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from kslx.layout import LHAND_SLICE, RHAND_SLICE, POSE13_MIRROR_PAIRS
from kslx.normalize import unpack_position, repack_from_position


def _sub_assign(out: torch.Tensor, mask: torch.Tensor, fn) -> None:
    idx = mask.nonzero(as_tuple=True)[0]
    if len(idx) == 0:
        return
    sub = out[idx].clone()
    out[idx] = fn(sub)


def _mirror(sub: torch.Tensor) -> torch.Tensor:
    sub[..., 0] = -sub[..., 0]
    lhand = sub[:, :, LHAND_SLICE, :].clone()
    rhand = sub[:, :, RHAND_SLICE, :].clone()
    sub[:, :, LHAND_SLICE, :] = rhand
    sub[:, :, RHAND_SLICE, :] = lhand
    for a, b in POSE13_MIRROR_PAIRS:
        tmp = sub[:, :, a, :].clone()
        sub[:, :, a, :] = sub[:, :, b, :]
        sub[:, :, b, :] = tmp
    return sub


def _rotate(sub: torch.Tensor, max_deg: float, rng: np.random.Generator) -> torch.Tensor:
    n = sub.shape[0]
    theta = torch.from_numpy(rng.uniform(-max_deg, max_deg, size=n)).float() * math.pi / 180.0
    cos, sin = torch.cos(theta), torch.sin(theta)
    x, y = sub[..., 0].clone(), sub[..., 1].clone()
    sub[..., 0] = cos[:, None, None] * x - sin[:, None, None] * y
    sub[..., 1] = sin[:, None, None] * x + cos[:, None, None] * y
    return sub


def _shear(sub: torch.Tensor, max_deg: float, rng: np.random.Generator) -> torch.Tensor:
    n = sub.shape[0]
    deg = torch.from_numpy(rng.uniform(-max_deg, max_deg, size=n)).float()
    k = torch.tan(deg * math.pi / 180.0)
    sub[..., 0] = sub[..., 0] + k[:, None, None] * sub[..., 1]
    return sub


def _scale(sub: torch.Tensor, lo: float, hi: float, rng: np.random.Generator) -> torch.Tensor:
    n = sub.shape[0]
    factor = torch.from_numpy(rng.uniform(lo, hi, size=n)).float()
    return sub * factor[:, None, None, None]


def _noise(sub: torch.Tensor, sigma: float, rng: np.random.Generator) -> torch.Tensor:
    return sub + torch.from_numpy(rng.normal(0.0, sigma, size=tuple(sub.shape))).float()


def _hand_dropout(sub: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    n = sub.shape[0]
    which = rng.integers(0, 2, size=n)
    for i in range(n):
        sl = LHAND_SLICE if which[i] == 0 else RHAND_SLICE
        sub[i, :, sl, :] = 0.0
    return sub


def augment_features(feat: torch.Tensor, rng: np.random.Generator,
                      p_mirror: float = 0.5, p_rotate: float = 0.5, p_shear: float = 0.3,
                      p_scale: float = 0.5, p_noise: float = 0.5, p_hand_dropout: float = 0.2) -> torch.Tensor:
    """feat: (B, T, 356) CPU 텐서. 같은 shape 반환."""
    pos = unpack_position(feat)  # (B, T, 89, 2)
    n = pos.shape[0]

    _sub_assign(pos, torch.from_numpy(rng.random(n) < p_mirror), _mirror)
    _sub_assign(pos, torch.from_numpy(rng.random(n) < p_rotate), lambda s: _rotate(s, 15.0, rng))
    _sub_assign(pos, torch.from_numpy(rng.random(n) < p_shear), lambda s: _shear(s, 20.0, rng))
    _sub_assign(pos, torch.from_numpy(rng.random(n) < p_scale), lambda s: _scale(s, 0.8, 1.2, rng))
    _sub_assign(pos, torch.from_numpy(rng.random(n) < p_noise), lambda s: _noise(s, 0.02, rng))
    _sub_assign(pos, torch.from_numpy(rng.random(n) < p_hand_dropout), lambda s: _hand_dropout(s, rng))

    pos = torch.clamp(pos, -20.0, 20.0)
    return repack_from_position(pos)
