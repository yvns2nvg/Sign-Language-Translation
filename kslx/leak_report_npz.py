"""팀원이 미리 만들어둔 3D npz(`DataSet/Dataset_NPZ/*.npz`)로 누수 주장을 독립 재확인.

이 npz 는 kslx 파이프라인과 무관하게 다른 스크립트(`*_view_filter.py` 등, 전부
"기본은 3D keypoint의 x,y,z 사용" 이라고 명시)로 만들어졌다. 즉 서로 다른 코드베이스
두 개가 같은 원본 AI Hub 데이터에서 3D 를 뽑았을 때 똑같은 누수가 나오는지 보는
독립 검증이다.

npz 스키마: X (N,148,67,3) float32, Y (N,) 클래스 인덱스, V (N,) "WORDxxxx_한글" —
같은 V 를 공유하는 최대 5개 샘플이 같은 take 의 5개 카메라 앵글이다.

python -m kslx.leak_report_npz --npz DataSet/Dataset_NPZ/02_dataset.npz
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def check_3d_duplication(npz_path: Path, n_samples: int = 50) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    X, V = d["X"], d["V"]

    by_v: dict[str, list[int]] = defaultdict(list)
    for i, v in enumerate(V):
        by_v[str(v)].append(i)

    takes_with_multi = [idxs for idxs in by_v.values() if len(idxs) >= 2]
    checked = 0
    identical = 0
    max_diff = 0.0
    for idxs in takes_with_multi:
        if checked >= n_samples:
            break
        checked += 1
        base = X[idxs[0]]
        all_same = True
        for j in idxs[1:]:
            diff = np.abs(X[j] - base).max()
            max_diff = max(max_diff, float(diff))
            if diff > 1e-6:
                all_same = False
        if all_same:
            identical += 1

    return {
        "n_samples": len(X), "n_takes_total": len(by_v),
        "n_takes_multi_angle": len(takes_with_multi),
        "takes_checked": checked, "takes_identical_across_angles": identical,
        "max_abs_diff_seen": max_diff,
    }


def take_out_vs_random_gap(npz_path: Path, seed: int = 0) -> dict:
    """아주 단순한 1-NN 분류기로 random split과 take_out split의 top-1 격차를 잰다.
    (모델 학습이 아니라 "같은 take 복사본이 옆에 있으면 최근접이 항상 정답이 된다"는
    사실 자체를 보여주는 데 목적이 있다 — 1-NN 이 이 효과를 가장 극적으로 드러낸다.)
    """
    d = np.load(npz_path, allow_pickle=True)
    X, Y, V = d["X"], d["Y"], d["V"]
    n = len(X)
    # 시간축 평균 풀링: (N,148,67,3) -> (N,201). 완전 동일한 복사본은 평균을 내도
    # 여전히 완전 동일하므로 leak 탐지 목적에는 영향이 없고, 브루트포스 1-NN을
    # 13585개 샘플에서도 돌릴 수 있게 차원을 크게 줄여준다.
    feat = X.mean(axis=1).reshape(n, -1)
    feat = feat - feat.mean(axis=0, keepdims=True)
    feat = feat / (feat.std(axis=0, keepdims=True) + 1e-6)

    rng = np.random.default_rng(seed)

    def nn_accuracy(train_idx, val_idx):
        tr = feat[train_idx]
        va = feat[val_idx]
        tr_sq = (tr ** 2).sum(axis=1)
        va_sq = (va ** 2).sum(axis=1)
        d2 = va_sq[:, None] + tr_sq[None, :] - 2.0 * va @ tr.T
        nn = d2.argmin(axis=1)
        pred = Y[train_idx][nn]
        return (pred == Y[val_idx]).mean()

    # random: 클래스별 20% val, take 무시
    by_class = defaultdict(list)
    for i, y in enumerate(Y):
        by_class[int(y)].append(i)
    train_idx, val_idx = [], []
    for idxs in by_class.values():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = max(1, round(len(idxs) * 0.2)) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    random_acc = nn_accuracy(np.array(train_idx), np.array(val_idx))

    # take_out: V(=take) 단위로 20%를 통째로 뺀다
    by_v = defaultdict(list)
    for i, v in enumerate(V):
        by_v[str(v)].append(i)
    takes = list(by_v.keys())
    rng.shuffle(takes)
    n_val_takes = max(1, round(len(takes) * 0.2))
    val_takes = set(takes[:n_val_takes])
    train_idx2 = [i for v, idxs in by_v.items() if v not in val_takes for i in idxs]
    val_idx2 = [i for v, idxs in by_v.items() if v in val_takes for i in idxs]
    takeout_acc = nn_accuracy(np.array(train_idx2), np.array(val_idx2))

    return {"random_1nn_top1": random_acc, "take_out_1nn_top1": takeout_acc,
            "gap": random_acc - takeout_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    args = ap.parse_args()

    print(f"[leak_report_npz] {args.npz}")
    r1 = check_3d_duplication(args.npz)
    print("\n-- 3D 키포인트 앵글(take) 간 동일성 --")
    print(r1)
    if r1["takes_checked"] > 0:
        pct = 100 * r1["takes_identical_across_angles"] / r1["takes_checked"]
        print(f"  -> {pct:.0f}% 의 take 가 여러 '앵글' 샘플에서 3D 좌표 완전 동일.")

    print("\n-- random vs take_out 1-NN 격차 (3D 원시좌표, 정규화만) --")
    r2 = take_out_vs_random_gap(args.npz)
    print(r2)
    print(f"  -> random 분할이 take_out 대비 top-1 {100*r2['gap']:.1f}%p 더 높게 나옴 "
          f"— 같은 take 복사본이 val 옆(train)에 있을 때 생기는 순수 누수 효과.")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
