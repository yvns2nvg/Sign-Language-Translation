"""데이터만으로 누수를 정량화한다 (모델 학습 불필요).

python -m kslx.leak_report --root 004.수어영상 --words 1-271 --signers 1-16

두 가지를 증명한다:
1. 3D 키포인트가 같은 take 의 5개 앵글 파일에서 비트 단위로 동일하다
   (멀티뷰 삼각측량 결과 복사) — 그래서 layout.py 가 3D 를 배제한다.
2. `random` 분할 프로토콜에서 val 클립의 몇 %가 train 에 같은 take(다른 앵글)를
   갖는지 — 이게 기존에 보고된 고정확도의 정체다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kslx.data.aihub import ANGLES, scan_clips
from kslx.splits import random_split, take_out_split, SplitError


def check_3d_duplication(root: Path, clips, n_samples: int = 30) -> dict:
    """같은 take 의 서로 다른 앵글에서 첫 프레임 3D pose keypoints 를 비교."""
    by_take: dict[str, list] = {}
    for c in clips:
        by_take.setdefault(c.take_id, []).append(c)

    checked = 0
    identical = 0
    max_diff_seen = 0.0
    for take_id, take_clips in by_take.items():
        if len(take_clips) < 2:
            continue
        if checked >= n_samples:
            break
        arrays = {}
        for c in take_clips:
            frame0 = sorted(c.keypoint_dir().glob("*_keypoints.json"))
            if not frame0:
                continue
            with open(frame0[0], "r", encoding="utf-8") as fh:
                data = json.load(fh)
            people = data["people"]
            if isinstance(people, list):
                people = people[0] if people else None
            if people is None or "pose_keypoints_3d" not in people:
                continue
            arrays[c.angle] = np.array(people["pose_keypoints_3d"], dtype=np.float64)
        if len(arrays) < 2:
            continue
        checked += 1
        vals = list(arrays.values())
        base = vals[0]
        all_same = True
        for other in vals[1:]:
            diff = np.abs(other - base).max()
            max_diff_seen = max(max_diff_seen, float(diff))
            if diff > 0:
                all_same = False
        if all_same:
            identical += 1

    return {"takes_checked": checked, "takes_identical_across_angles": identical,
            "max_abs_diff_seen": max_diff_seen}


def check_random_split_leak(clips) -> dict:
    train_idx, val_idx = random_split(clips, val_ratio=0.15, seed=0)
    train_takes = {clips[i].take_id for i in train_idx}
    val_leaked = sum(1 for i in val_idx if clips[i].take_id in train_takes)
    return {
        "n_train": len(train_idx), "n_val": len(val_idx),
        "val_clips_with_take_sibling_in_train": val_leaked,
        "val_leak_ratio": val_leaked / max(1, len(val_idx)),
    }


def check_take_out_feasibility(clips) -> dict:
    try:
        train_idx, val_idx = take_out_split(clips, val_ratio=0.15, seed=0)
        return {"feasible": True, "n_train": len(train_idx), "n_val": len(val_idx)}
    except SplitError as e:
        return {"feasible": False, "reason": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--signers", type=str, default=None)
    ap.add_argument("--words", type=str, default=None)
    args = ap.parse_args()

    from kslx.data.build_dataset import _parse_word_range
    signers = _parse_word_range(args.signers)
    words = _parse_word_range(args.words)
    clips = scan_clips(args.root, signers=signers, words=words)
    print(f"[leak_report] {len(clips)} clips, "
          f"{len({c.signer for c in clips})} signers, {len({c.word for c in clips})} words")

    print("\n-- 1) 3D 키포인트 앵글 간 동일성 --")
    r1 = check_3d_duplication(args.root, clips)
    print(r1)
    if r1["takes_checked"] > 0:
        pct = 100 * r1["takes_identical_across_angles"] / r1["takes_checked"]
        print(f"  -> 표본 {r1['takes_checked']}개 take 중 {pct:.0f}% 가 5앵글 3D 완전 동일. "
              f"3D 는 학습에 쓰지 말 것.")

    print("\n-- 2) random 분할의 take 누수 --")
    r2 = check_random_split_leak(clips)
    print(r2)
    print(f"  -> val 클립의 {100*r2['val_leak_ratio']:.1f}% 가 train 에 같은 take(다른 앵글)를 가짐.")

    print("\n-- 3) take_out 분할 가능 여부 --")
    r3 = check_take_out_feasibility(clips)
    print(r3)

    print("\n-- 4) signer 현황 --")
    signers_present = sorted({c.signer for c in clips})
    print(f"  signers: {signers_present} (n={len(signers_present)})")
    print(f"  signer_out {'가능' if len(signers_present) >= 2 else '불가능 (수어사 1명뿐)'}")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
