"""손 움직임 에너지로 "단어 끝 → 다음 단어 시작"을 자동으로 감지하는 게이트.

학습에 쓴 형태소(수어구간) 어노테이션과 같은 원리다 — AI Hub 라벨도 클립 안에서
"손이 움직이는 구간"을 수어 구간으로 봤다. 여기서는 실시간으로 그 구간의
시작/끝을 손 키포인트의 프레임간 이동량(에너지)으로 근사한다.

★ 이건 README 가 설계했던 정식 스트리밍 파이프라인(학습된 background 클래스 +
conf_threshold 튜닝 + WER 평가, kslx.eval_stream/stream 원안)이 아니라 그보다
훨씬 가벼운 휴리스틱이다. 카메라가 한쪽 손을 순간적으로 놓치면 "에너지가
0으로 떨어졌다"고 오판해서 단어가 끝난 걸로 잘못 끊길 수 있다. 그게 문제가
되면 realtime.py 의 SPACE 방식(수동)으로 돌아가거나, 다음 단계로 학습된
경계 검출기를 붙일 것 (원본 README §4.1 "남은 길" 참고).
"""

from __future__ import annotations

import numpy as np

from kslx.layout import LHAND_SLICE, RHAND_SLICE
from kslx.normalize import DEGENERATE_SCALE_PX, MIN_SCALE
from kslx.layout import POSE13_NECK_IDX, POSE13_R_SHOULDER_IDX, POSE13_L_SHOULDER_IDX


class LiveNormalizer:
    """실시간용 프레임 단위 목-원점/어깨너비 정규화. 클립 전체를 미리 볼 수
    없으므로 kslx.normalize.center_and_scale 의 "클립 중앙값 fallback" 대신
    "마지막으로 정상 검출됐던 스케일"을 이월해서 쓴다."""

    def __init__(self):
        self.last_scale: float | None = None

    def normalize(self, frame89: np.ndarray) -> np.ndarray:
        neck = frame89[POSE13_NECK_IDX]
        centered = frame89 - neck
        shoulder_vec = frame89[POSE13_R_SHOULDER_IDX] - frame89[POSE13_L_SHOULDER_IDX]
        scale = float(np.linalg.norm(shoulder_vec))
        if scale < DEGENERATE_SCALE_PX:
            scale = self.last_scale if self.last_scale is not None else MIN_SCALE
        else:
            self.last_scale = scale
        return centered / max(scale, MIN_SCALE)


def hand_energy(prev_norm: np.ndarray, curr_norm: np.ndarray) -> float:
    """두 정규화된 프레임 사이의 양손 평균 이동량 (어깨너비 단위)."""
    diff = curr_norm[LHAND_SLICE] - prev_norm[LHAND_SLICE]
    diff_r = curr_norm[RHAND_SLICE] - prev_norm[RHAND_SLICE]
    both = np.concatenate([diff, diff_r], axis=0)
    return float(np.linalg.norm(both, axis=-1).mean())


class SegmentGate:
    """에너지 기반 상태 머신: idle -> recording -> finished(예측 트리거) -> idle."""

    def __init__(self, start_energy: float = 0.06, end_energy: float = 0.03,
                 end_hold: int = 10, min_frames: int = 6, max_frames: int = 90):
        self.start_energy = start_energy
        self.end_energy = end_energy
        self.end_hold = end_hold
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.state = "idle"
        self.low_run = 0
        self.buffer: list[np.ndarray] = []

    def step(self, frame89_raw: np.ndarray, energy: float) -> tuple[str, list[np.ndarray] | None]:
        """반환: (상태전이 라벨, 완료됐을 때만 잘라낸 프레임 버퍼).
        상태전이 라벨: 'idle' | 'started' | 'recording' | 'finished'"""
        if self.state == "idle":
            if energy > self.start_energy:
                self.state = "recording"
                self.buffer = [frame89_raw]
                self.low_run = 0
                return "started", None
            return "idle", None

        # recording
        self.buffer.append(frame89_raw)
        self.low_run = self.low_run + 1 if energy < self.end_energy else 0

        finished = (self.low_run >= self.end_hold and len(self.buffer) >= self.min_frames) \
            or len(self.buffer) >= self.max_frames
        if not finished:
            return "recording", None

        buf = self.buffer
        trim = min(self.low_run, max(0, len(buf) - self.min_frames))
        buf_trimmed = buf[: len(buf) - trim] if trim > 0 else buf
        self.state = "idle"
        self.buffer = []
        self.low_run = 0
        return "finished", buf_trimmed
