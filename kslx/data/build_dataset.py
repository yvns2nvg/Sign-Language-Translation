"""JSON 키포인트 -> npz/npy 캐시 빌더.

사용:
    python -m kslx.data.build_dataset --root 004.수어영상 --out data/kslx/word.npz \
        --words 1-271 --workers 10

    # 대규모(예: 3000단어 전체)는 --out 을 디렉토리로 주면 X 를 압축 없는
    # .npy 로 저장한다 — 학습 때 np.memmap 으로 열어서 전체를 RAM에 한 번에
    # 안 올리고 배치 단위로만 읽을 수 있다. (.npz 는 zip 압축이라 진짜
    # 메모리맵이 안 된다 — numpy 가 어차피 전체를 풀어서 RAM에 얹는다.)
    python -m kslx.data.build_dataset --root 004.수어영상 --out data/kslx/word_full \
        --words 1-3000 --workers 14

내용 (두 형식 공통):
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
import json
import os
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


def build(root: Path, out_path: Path, signers, words, angles, workers: int, t_out: int,
          as_dir: bool) -> None:
    clips = scan_clips(root, signers=signers, words=words, angles=angles)
    if not clips:
        raise SystemExit("no clips matched the given filters")
    n_total = len(clips)
    print(f"[build_dataset] {n_total} clips matched, {workers} workers, t_out={t_out}, "
          f"format={'npy-dir(memmap 가능)' if as_dir else 'npz'}")

    # 대용량(npy-dir)은 메모리맵을 계속 열어둔 채 쓰지 않는다 — Windows 에서
    # 20GB짜리 파일을 mmap 으로 오래 열어두고 쓰면 더티 페이지가 플러시되기
    # 전까지 시스템 캐시에 쌓여 "메모리 부족"으로 프로세스가 강제 종료되는 걸
    # 실측했다. 대신 일반 파일 쓰기 + 주기적 flush/fsync 로 디스크에 그때그때
    # 내려보낸다. 학습 때 읽을 땐 read-only np.memmap 을 쓰는데, 읽기 전용
    # 메모리맵은 더티 페이지가 없어(캐시가 차면 OS가 그냥 버리고 필요할 때
    # 디스크에서 다시 읽으면 됨) 이 문제가 없다.
    x_file = None
    if as_dir:
        out_path.mkdir(parents=True, exist_ok=True)
        x_file = open(out_path / "X.raw", "wb")
    else:
        feats: list[np.ndarray] = []

    ys, signers_arr, angles_arr, take_ids, clip_ids = [], [], [], [], []
    n_err = 0
    n_valid = 0
    t0 = time.time()
    tasks = [(c, t_out) for c in clips]
    FLUSH_EVERY = 2000
    with Pool(workers) as pool:
        # imap(순서 보장)을 써서 결과를 받는 즉시 그대로 순차적으로(seek 없이)
        # 파일에 이어붙인다.
        for i, result in enumerate(pool.imap(_process_one, tasks, chunksize=8), 1):
            if "error" in result:
                n_err += 1
                if n_err <= 20:
                    print(f"  [skip] {result['error']}", file=sys.stderr)
            else:
                if as_dir:
                    x_file.write(np.ascontiguousarray(result["feat"], dtype=np.float32).tobytes())
                else:
                    feats.append(result["feat"])
                ys.append(result["word"])
                signers_arr.append(result["signer"])
                angles_arr.append(result["angle"])
                take_ids.append(result["take_id"])
                clip_ids.append(result["clip_id"])
                n_valid += 1
                if as_dir and n_valid % FLUSH_EVERY == 0:
                    x_file.flush()
                    os.fsync(x_file.fileno())
            if i % 2000 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                print(f"  {i}/{len(tasks)} ({elapsed:.1f}s, {n_err} errors)")

    if n_valid == 0:
        raise SystemExit("all clips failed to load — check --root path")

    classes = sorted(set(ys))
    word_to_idx = {w: i for i, w in enumerate(classes)}
    y_idx = np.array([word_to_idx[w] for w in ys], dtype=np.int32)
    signer_arr = np.array(signers_arr, dtype=np.int32)
    angle_arr = np.array(angles_arr, dtype="<U1")
    take_id_arr = np.array(take_ids, dtype="<U8")
    clip_id_arr = np.array(clip_ids, dtype="<U40")
    classes_arr = np.array(classes, dtype=np.int32)

    if as_dir:
        x_file.flush()
        os.fsync(x_file.fileno())
        x_file.close()
        np.save(out_path / "y.npy", y_idx)
        np.save(out_path / "signer.npy", signer_arr)
        np.save(out_path / "angle.npy", angle_arr)
        np.save(out_path / "take_id.npy", take_id_arr)
        np.save(out_path / "clip_id.npy", clip_id_arr)
        np.save(out_path / "classes.npy", classes_arr)
        with open(out_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"feature_dim": FEATURE_DIM, "t_out": t_out, "n_valid": n_valid,
                       "n_total": n_total}, f)
        print(f"[build_dataset] wrote {out_path}/X.raw : X=({n_valid},{t_out},{FEATURE_DIM}) "
              f"(read시 np.memmap 으로 열 것), "
              f"{len(classes)} classes, {len(set(signers_arr))} signers, {n_err} clips skipped")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            X=np.stack(feats, axis=0),
            y=y_idx, signer=signer_arr, angle=angle_arr, take_id=take_id_arr,
            clip_id=clip_id_arr, classes=classes_arr,
            feature_dim=FEATURE_DIM, t_out=t_out,
        )
        print(f"[build_dataset] wrote {out_path} : X=({n_valid},{t_out},{FEATURE_DIM}), "
              f"{len(classes)} classes, {len(set(signers_arr))} signers, {n_err} clips skipped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="004.수어영상 폴더 경로")
    ap.add_argument("--out", type=Path, required=True,
                     help=".npz 로 끝나면 압축 npz, 아니면 메모리맵 가능한 npy 디렉토리로 저장")
    ap.add_argument("--signers", type=str, default=None, help="예: 1-16 또는 1,2,3")
    ap.add_argument("--words", type=str, default=None, help="예: 1-271")
    ap.add_argument("--angles", type=str, default="F,D,L,R,U")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--t-out", type=int, default=64)
    args = ap.parse_args()

    signers = _parse_word_range(args.signers)
    words = _parse_word_range(args.words)
    angles = tuple(args.angles.split(","))
    as_dir = args.out.suffix.lower() != ".npz"
    build(args.root, args.out, signers, words, angles, args.workers, args.t_out, as_dir)


if __name__ == "__main__":
    main()
