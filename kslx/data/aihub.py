"""AI Hub 수어영상 키포인트 데이터셋 스캐너 + 로더.

디렉토리 구조 (WORD, REAL 만 다룬다 — SEN 은 범위 밖):

    <root>/1.Training/라벨링데이터/REAL/WORD/<session:02d>/
        NIA_SL_WORD<word:04d>_REAL<session:02d>_<angle>/
            NIA_SL_WORD<word:04d>_REAL<session:02d>_<angle>_<frame:012d>_keypoints.json
    <root>/1.Training/라벨링데이터/REAL/WORD/morpheme/<session:02d>/
        NIA_SL_WORD<word:04d>_REAL<session:02d>_<angle>_morpheme.json

session = 수어사(signer) ID, angle ∈ {F,D,L,R,U} = 카메라 5앵글, word = 클래스.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from kslx.layout import extract_openpose_2d, N_TOTAL

TAKE_DIR_RE = re.compile(r"^NIA_SL_WORD(\d{4})_REAL(\d{2})_([FDLRU])$")
ANGLES = ("F", "D", "L", "R", "U")


@dataclass(frozen=True)
class Clip:
    root: Path
    word: int
    signer: int
    angle: str

    @property
    def take_id(self) -> str:
        """(signer, word) — 앵글이 달라도 같은 take 면 같은 값. 분할 시 그룹 키로 쓴다."""
        return f"{self.signer:02d}_{self.word:04d}"

    @property
    def clip_id(self) -> str:
        return f"NIA_SL_WORD{self.word:04d}_REAL{self.signer:02d}_{self.angle}"

    def keypoint_dir(self) -> Path:
        return self.root / "1.Training" / "라벨링데이터" / "REAL" / "WORD" / f"{self.signer:02d}" / self.clip_id

    def morpheme_path(self) -> Path:
        return (self.root / "1.Training" / "라벨링데이터" / "REAL" / "WORD" / "morpheme"
                / f"{self.signer:02d}" / f"{self.clip_id}_morpheme.json")


def list_signers(root: Path) -> list[int]:
    word_dir = root / "1.Training" / "라벨링데이터" / "REAL" / "WORD"
    out = []
    for entry in os.scandir(word_dir):
        if entry.is_dir() and re.fullmatch(r"\d{2}", entry.name):
            out.append(int(entry.name))
    return sorted(out)


def scan_clips(root: Path, signers: Iterable[int] | None = None,
               words: Iterable[int] | None = None,
               angles: Iterable[str] | None = None) -> list[Clip]:
    """조건에 맞는 (signer, word, angle) 클립을 나열한다. 디스크 IO는 디렉토리 목록만."""
    root = Path(root)
    signers = set(signers) if signers is not None else set(list_signers(root))
    words = set(words) if words is not None else None
    angles = set(angles) if angles is not None else set(ANGLES)

    clips: list[Clip] = []
    word_root = root / "1.Training" / "라벨링데이터" / "REAL" / "WORD"
    for signer in sorted(signers):
        session_dir = word_root / f"{signer:02d}"
        if not session_dir.is_dir():
            continue
        for entry in os.scandir(session_dir):
            m = TAKE_DIR_RE.match(entry.name)
            if not m:
                continue
            word, signer_from_name, angle = int(m.group(1)), int(m.group(2)), m.group(3)
            if words is not None and word not in words:
                continue
            if angle not in angles:
                continue
            clips.append(Clip(root=root, word=word, signer=signer_from_name, angle=angle))
    clips.sort(key=lambda c: (c.signer, c.word, c.angle))
    return clips


def load_keypoint_sequence(clip: Clip) -> np.ndarray:
    """클립의 모든 프레임 JSON을 읽어 (T, 89, 2) 배열로 반환한다. 프레임 순서 = 파일명 정렬."""
    d = clip.keypoint_dir()
    files = sorted(d.glob("*_keypoints.json"))
    if not files:
        raise FileNotFoundError(f"no keypoint json under {d}")
    frames = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        people = data["people"]
        if isinstance(people, list):
            people = people[0] if people else None
        if people is None:
            xs = ys = [0.0] * N_TOTAL
        else:
            xs, ys = extract_openpose_2d(people)
        frames.append(np.stack([xs, ys], axis=-1))  # (89, 2)
    return np.stack(frames, axis=0).astype(np.float32)  # (T, 89, 2)


def load_sign_span(clip: Clip, n_frames: int) -> tuple[int, int]:
    """형태소 어노테이션의 start/end(초)를 프레임 인덱스로 변환.

    fps 는 클립마다 duration/n_frames 로부터 역산한다 (전역 상수로 가정하지 않음 —
    실측상 세션/클립별로 29~30fps 로 미세하게 다르다).
    파일이 없거나 파싱 실패하면 전체 구간을 반환한다 (fallback, 조용히 죽지 않게 로깅은
    호출부에서 처리).
    """
    mp = clip.morpheme_path()
    if not mp.exists():
        return 0, n_frames
    with open(mp, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    duration = meta["metaData"]["duration"]
    entries = meta.get("data", [])
    if not entries or duration <= 0:
        return 0, n_frames
    fps = n_frames / duration
    start_s = min(e["start"] for e in entries)
    end_s = max(e["end"] for e in entries)
    start_f = max(0, int(round(start_s * fps)))
    end_f = min(n_frames, int(round(end_s * fps)))
    if end_f <= start_f:
        return 0, n_frames
    return start_f, end_f


@lru_cache(maxsize=1)
def word_label_map(path: str | None = None) -> dict[int, str]:
    """WORD#### -> 한글 단어. kslx/data/word_labels.json (ksl-realtime-infer 의
    models/label_map.json 을 원본으로 복사해둔 캐시) 를 읽는다."""
    p = Path(path) if path else Path(__file__).parent / "word_labels.json"
    with open(p, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {int(k.replace("WORD", "")): v for k, v in raw.items()}
