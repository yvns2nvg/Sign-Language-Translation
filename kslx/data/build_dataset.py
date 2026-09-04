"""JSON 키포인트 -> npz 캐시 빌더.

사용:
    python -m kslx.data.build_dataset --root 004.수어영상 --out data/kslx/word.npz \
        --words 1-271 --workers 10

npz 내용:
    X        (N, T, FEATURE_DIM) float32   정규화된 위치+속도 피처
    y        (N,) int32                    0..C-1 클래스 인덱스 (word id 아님)
    signer   (N,) int32                    수어사(session) 번호
    angle    (N,) <U1                      카메라 앵글 F/D/L/R/U
    take_id  (N,) <U7                      "{signer:02d}_{word:04d}" — 분할 그룹 키
    clip_id  (N,) <U40                     원본 클립 식별자
    classes  (C,) int32                    class index -> 원래 word id
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from kslx.data.aihub import Clip, load_keypoint_sequence, load_sign_span, scan_clips
from kslx.normalize import FEATURE_DIM, featurize


def _parse_word_range(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _process_one(args) -> dict | None:
    clip, t_out = args
    try:
        seq = load_keypoint_sequence(clip)
        span = load_sign_span(clip, seq.shape[0])
        feat = featurize(seq, t_out=t_out, sign_span=span)
    except Exception as e:  # noqa: BLE001 — 손상된 클립은 건너뛰고 로그만 남긴다
        return {"error": f"{clip.clip_id}: {e}"}
    return {
        "feat": feat,
        "word": clip.word,
        "signer": clip.signer,
        "angle": clip.angle,
        "take_id": clip.take_id,
        "clip_id": clip.clip_id,
    }


def build(root: Path, out_path: Path, signers, words, angles, workers: int, t_out: int) -> None:
    clips = scan_clips(root, signers=signers, words=words, angles=angles)
    if not clips:
        raise SystemExit("no clips matched the given filters")
    print(f"[build_dataset] {len(clips)} clips matched, {workers} workers, t_out={t_out}")

    feats, ys, signers_arr, angles_arr, take_ids, clip_ids = [], [], [], [], [], []
    n_err = 0
    t0 = time.time()
    tasks = [(c, t_out) for c in clips]
    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_one, tasks, chunksize=8), 1):
            if "error" in result:
                n_err += 1
                if n_err <= 20:
                    print(f"  [skip] {result['error']}", file=sys.stderr)
            else:
                feats.append(result["feat"])
                ys.append(result["word"])
                signers_arr.append(result["signer"])
                angles_arr.append(result["angle"])
                take_ids.append(result["take_id"])
                clip_ids.append(result["clip_id"])
            if i % 500 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                print(f"  {i}/{len(tasks)} ({elapsed:.1f}s, {n_err} errors)")

    if not feats:
        raise SystemExit("all clips failed to load — check --root path")

    classes = sorted(set(ys))
    word_to_idx = {w: i for i, w in enumerate(classes)}
    y_idx = np.array([word_to_idx[w] for w in ys], dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=np.stack(feats, axis=0),
        y=y_idx,
        signer=np.array(signers_arr, dtype=np.int32),
        angle=np.array(angles_arr, dtype="<U1"),
        take_id=np.array(take_ids, dtype="<U8"),
        clip_id=np.array(clip_ids, dtype="<U40"),
        classes=np.array(classes, dtype=np.int32),
        feature_dim=FEATURE_DIM,
        t_out=t_out,
    )
    print(f"[build_dataset] wrote {out_path} : X={np.stack(feats).shape}, "
          f"{len(classes)} classes, {len(set(signers_arr))} signers, "
          f"{n_err} clips skipped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="004.수어영상 폴더 경로")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--signers", type=str, default=None, help="예: 1-16 또는 1,2,3")
    ap.add_argument("--words", type=str, default=None, help="예: 1-271")
    ap.add_argument("--angles", type=str, default="F,D,L,R,U")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--t-out", type=int, default=64)
    args = ap.parse_args()

    signers = _parse_word_range(args.signers)
    words = _parse_word_range(args.words)
    angles = tuple(args.angles.split(","))
    build(args.root, args.out, signers, words, angles, args.workers, args.t_out)


if __name__ == "__main__":
    main()
