"""분할 프로토콜.

핵심 원칙: **무엇을 밖에 낼지(out)** 로 프로토콜 이름을 짓는다. 같은 원본
데이터라도 무엇을 기준으로 잘라내느냐에 따라 측정하는 일반화 능력이 다르다.

- random      : 클래스별 층화 랜덤 분할. 같은 take 의 다른 앵글이 train/val 에
                동시에 들어갈 수 있어 실질적으로 거의 같은 샘플이 양쪽에 존재한다
                (leak). 기존에 보고된 고정확도의 정체.
- take_out    : (signer, word) 조합 — take — 단위로 분할. 한 take 의 5개 앵글이
                통째로 train 또는 val 한쪽에만 들어간다.
- angle_out   : 카메라 앵글 자체로 분할 (예: F/D/R 은 train, L/U 는 val).
                같은 take 가 양쪽에 걸치므로 시점 일반화의 **상한선**이지
                정직한 signer_out 대체재가 아니다.
- signer_out  : 수어사(session) 단위로 분할. val 수어사는 train 에 전혀
                등장하지 않는다 — 실사용(미학습 수어자) 성능에 가장 가까운 지표.

각 함수는 안전장치로 `SplitError` 를 던진다 — 조건이 안 맞는데 조용히 이상한
숫자를 내는 것보다 낫다.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from kslx.data.aihub import Clip


class SplitError(RuntimeError):
    pass


def _stratified_random(clips: list[Clip], val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    by_word: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(clips):
        by_word[c.word].append(i)
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for word, idxs in by_word.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = max(1, round(len(idxs) * val_ratio)) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


def _group_split(clips: list[Clip], group_key, val_groups: set, ) -> tuple[np.ndarray, np.ndarray]:
    train_idx, val_idx = [], []
    for i, c in enumerate(clips):
        if group_key(c) in val_groups:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return np.array(train_idx), np.array(val_idx)


def random_split(clips: list[Clip], val_ratio: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    train_idx, val_idx = _stratified_random(clips, val_ratio, seed)
    if len(val_idx) == 0:
        raise SplitError("random split produced empty val set")
    return train_idx, val_idx


def take_out_split(clips: list[Clip], val_ratio: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(signer, word) take 단위로 val 을 뺀다. 어떤 word 가 train 에서 완전히
    사라지면(take 수가 1개뿐이라 전부 val 로 빠짐) SplitError."""
    by_word_takes: dict[int, set[str]] = defaultdict(set)
    for c in clips:
        by_word_takes[c.word].add(c.take_id)

    rng = random.Random(seed)
    val_takes: set[str] = set()
    for word, takes in by_word_takes.items():
        takes = sorted(takes)
        if len(takes) < 2:
            raise SplitError(
                f"word {word} has only {len(takes)} take(s) — take_out split would "
                f"remove it entirely from train (zero-shot). 수어사를 더 받아야 한다."
            )
        rng.shuffle(takes)
        n_val = max(1, round(len(takes) * val_ratio))
        n_val = min(n_val, len(takes) - 1)  # 최소 1개는 train 에 남긴다
        val_takes.update(takes[:n_val])

    train_idx, val_idx = _group_split(clips, lambda c: c.take_id, val_takes)
    _audit_disjoint_takes(clips, train_idx, val_idx)
    return train_idx, val_idx


def angle_out_split(clips: list[Clip], val_angles: tuple[str, ...] = ("L", "U")) -> tuple[np.ndarray, np.ndarray]:
    angles_present = {c.angle for c in clips}
    if not set(val_angles) <= angles_present:
        raise SplitError(f"val_angles {val_angles} not subset of present angles {angles_present}")
    if set(val_angles) == angles_present:
        raise SplitError("val_angles covers all present angles — no train data would remain")
    train_idx, val_idx = _group_split(clips, lambda c: c.angle, set(val_angles))
    if len(val_idx) == 0:
        raise SplitError("angle_out split produced empty val set")
    return train_idx, val_idx


def signer_out_split(clips: list[Clip], val_signers: tuple[int, ...] | None = None,
                      n_val_signers: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """수어사 단위 분할 — 진짜 일반화 평가. val 수어사는 train 에 단 한 프레임도
    등장하지 않는다."""
    signers = sorted({c.signer for c in clips})
    if len(signers) < 2:
        raise SplitError(
            f"only {len(signers)} signer(s) present — signer_out is impossible. "
            f"AI Hub 에서 다른 REAL 세션을 더 받아야 한다."
        )
    if val_signers is None:
        rng = random.Random(seed)
        pool = signers[:]
        rng.shuffle(pool)
        n_val = max(1, min(n_val_signers, len(signers) - 1))
        val_signers = tuple(sorted(pool[:n_val]))
    missing = set(val_signers) - set(signers)
    if missing:
        raise SplitError(f"val_signers {missing} not present in data")
    if set(val_signers) == set(signers):
        raise SplitError("val_signers covers all signers — no train signer would remain")

    train_idx, val_idx = _group_split(clips, lambda c: c.signer, set(val_signers))
    train_words = {clips[i].word for i in train_idx}
    val_words = {clips[i].word for i in val_idx}
    oov = val_words - train_words
    if oov:
        raise SplitError(
            f"{len(oov)} word(s) in val have zero train examples across remaining signers "
            f"(e.g. {sorted(oov)[:5]}) — classification target undefined for them."
        )
    _audit_disjoint_signers(clips, train_idx, val_idx)
    return train_idx, val_idx


# ---- 사후 감사 (post-hoc audit): leak 없음을 코드로 증명 ----

def _audit_disjoint_takes(clips: list[Clip], train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    train_takes = {clips[i].take_id for i in train_idx}
    val_takes = {clips[i].take_id for i in val_idx}
    overlap = train_takes & val_takes
    if overlap:
        raise SplitError(f"take leak: {len(overlap)} take(s) appear in both train and val")


def _audit_disjoint_signers(clips: list[Clip], train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    train_signers = {clips[i].signer for i in train_idx}
    val_signers = {clips[i].signer for i in val_idx}
    overlap = train_signers & val_signers
    if overlap:
        raise SplitError(f"signer leak: signer(s) {overlap} appear in both train and val")


PROTOCOLS = {
    "random": random_split,
    "take_out": take_out_split,
    "angle_out": angle_out_split,
    "signer_out": signer_out_split,
}
