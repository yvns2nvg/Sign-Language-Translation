"""강건성 평가 — 모델 선택은 이걸로 한다 (angle_out top-1 만 보면 안 된다).

val 셋(기본 angle_out)에 실사용 변형(합성)을 준 뒤 top-1 열화를 잰다.
피처는 이미 정규화+리샘플된 (T,356) = pos(T,89,2) ++ vel(T,89,2) 이므로,
변형은 pos 채널에 적용한 뒤 속도를 다시 계산해 피처를 재구성한다.

★ yaw 는 3D 회전이 아니라 2D 전단(shear) 근사다 ★ (layout.py 의 3D 금지 원칙과
동일한 이유 — 3D 키포인트는 앵글 간 leak 이 있어 학습/평가 어디에도 못 쓴다.
실제 카메라 시점 변화의 대략적인 대리 지표로만 쓸 것.

python -m kslx.eval_robust --data data/kslx/word_271.npz --ckpt runs/angle_out_base.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from kslx.models.conv_transformer import ConvTransformer
from kslx.normalize import FEATURE_DIM
from kslx.train import ClipMeta, load_clips_meta, topk_accuracy
from kslx.splits import PROTOCOLS


N_POINTS = FEATURE_DIM // 4  # 89


def _unpack(feat: np.ndarray) -> np.ndarray:
    """(N, T, 356) -> pos (N, T, 89, 2)."""
    n, t, _ = feat.shape
    pos_flat = feat[:, :, : N_POINTS * 2]
    return pos_flat.reshape(n, t, N_POINTS, 2)


def _repack(pos: np.ndarray) -> np.ndarray:
    """pos (N,T,89,2) -> feat (N,T,356), 속도는 재계산."""
    n, t, p, _ = pos.shape
    vel = np.zeros_like(pos)
    vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
    return np.concatenate([pos.reshape(n, t, -1), vel.reshape(n, t, -1)], axis=-1).astype(np.float32)


def perturb_shear(pos: np.ndarray, deg: float, rng: np.random.Generator) -> np.ndarray:
    k = math.tan(math.radians(deg))
    sign = rng.choice([-1.0, 1.0], size=(pos.shape[0], 1, 1))  # (N,1,1) broadcasts vs (N,T,89)
    out = pos.copy()
    out[..., 0] = pos[..., 0] + sign * k * pos[..., 1]
    return out


def perturb_rotate(pos: np.ndarray, deg: float, rng: np.random.Generator) -> np.ndarray:
    sign = rng.choice([-1.0, 1.0], size=pos.shape[0])
    theta = np.radians(deg) * sign
    cos, sin = np.cos(theta), np.sin(theta)
    out = np.empty_like(pos)
    x, y = pos[..., 0], pos[..., 1]
    out[..., 0] = cos[:, None, None] * x - sin[:, None, None] * y
    out[..., 1] = sin[:, None, None] * x + cos[:, None, None] * y
    return out


def perturb_hand_dropout(pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from kslx.layout import LHAND_SLICE, RHAND_SLICE
    out = pos.copy()
    which = rng.choice([LHAND_SLICE, RHAND_SLICE], size=pos.shape[0])
    for i, sl in enumerate(which):
        out[i, :, sl, :] = 0.0
    return out


def perturb_scale(pos: np.ndarray, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    factor = rng.uniform(lo, hi, size=(pos.shape[0], 1, 1, 1))
    return pos * factor


def perturb_noise(pos: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return pos + rng.normal(0.0, sigma, size=pos.shape)


def perturb_speed(pos: np.ndarray, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    from kslx.normalize import resample_time
    t = pos.shape[1]
    out = np.empty_like(pos)
    for i in range(pos.shape[0]):
        factor = rng.uniform(lo, hi)
        t_mid = max(2, int(round(t * factor)))
        warped = resample_time(pos[i], t_mid)
        out[i] = resample_time(warped, t)
    return out


VARIANTS = {
    "clean": lambda pos, rng: pos,
    "yaw_15": lambda pos, rng: perturb_shear(pos, 15, rng),
    "yaw_30": lambda pos, rng: perturb_shear(pos, 30, rng),
    "rotate_15": lambda pos, rng: perturb_rotate(pos, 15, rng),
    "hand_dropout": lambda pos, rng: perturb_hand_dropout(pos, rng),
    "scale_25": lambda pos, rng: perturb_scale(pos, 0.75, 1.25, rng),
    "noise_03": lambda pos, rng: perturb_noise(pos, 0.03, rng),
    "speed_0.6_1.6": lambda pos, rng: perturb_speed(pos, 0.6, 1.6, rng),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--protocol", type=str, default="angle_out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    npz = np.load(args.data, allow_pickle=False)
    X, y = npz["X"], npz["y"]
    clips = load_clips_meta(npz)
    _, val_idx = PROTOCOLS[args.protocol](clips, val_angles=("L", "U")) \
        if args.protocol == "angle_out" else PROTOCOLS[args.protocol](clips)
    x_val, y_val = X[val_idx], y[val_idx]
    pos_val = _unpack(x_val)
    print(f"[eval_robust] protocol={args.protocol} n_val={len(val_idx)}")

    rng = np.random.default_rng(args.seed)
    header = f"{'variant':<16}" + "".join(f"{Path(c).stem:>18}" for c in args.ckpt)
    print(header)
    baseline = {}
    for name, fn in VARIANTS.items():
        pos_p = fn(pos_val, rng)
        feat_p = _repack(pos_p)
        x_t = torch.from_numpy(feat_p)
        y_t = torch.from_numpy(y_val).long()
        row = f"{name:<16}"
        for ckpt_path in args.ckpt:
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            model = ConvTransformer(feature_dim=ckpt["feature_dim"], num_classes=ckpt["num_classes"])
            model.load_state_dict(ckpt["state_dict"])
            model.to(args.device).eval()
            with torch.no_grad():
                logits = model(x_t.to(args.device))
                top1 = topk_accuracy(logits, y_t.to(args.device), ks=(1,))["top1"]
            if name == "clean":
                baseline[str(ckpt_path)] = top1
            delta = top1 - baseline.get(str(ckpt_path), top1)
            row += f"{top1*100:>10.1f}%({delta*100:+.1f})"
        print(row)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
