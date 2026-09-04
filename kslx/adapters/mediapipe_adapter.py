"""MediaPipe Tasks HolisticLandmarker 결과 -> kslx.layout 의 89점 규약 변환.

학습 데이터는 AI Hub 라벨(OpenPose BODY_25 + hand21x2 + face70)이고, 실시간
추론은 MediaPipe(pose33 + hand21x2 + face478)를 쓰므로 인덱스 체계가 다르다.
이 모듈이 그 변환을 담당한다.

★ 미검증 ★ pose/hand 매핑은 관절 대응이 명확해 신뢰도가 높지만, 얼굴 34점은
dlib68 좌표를 정확히 재현한 게 아니라 MediaPipe FaceMesh 공식 영역 상수
(FACEMESH_LEFT_EYE 등, mediapipe/python/solutions/face_mesh_connections.py)
에서 각 영역을 균등 서브샘플링한 근사치다. 눈썹/눈/입의 대략적인 모양 변화는
잡지만 학습 시 얼굴 포인트와 위치가 1:1로 대응하지는 않는다.
실사용 성능이 나쁘면 먼저 `--no-face` 로 얼굴을 꺼서 pose+hand 만으로
비교해볼 것 (README §4.2 와 동일한 주의).
"""

from __future__ import annotations

import numpy as np

# ---- pose: MediaPipe Pose(33) -> 우리 pose13 ----
# MediaPipe 도 OpenPose 와 마찬가지로 "피사체 자신의" 좌/우 기준이다.
MP_POSE_NOSE = 0
MP_POSE_L_SHOULDER, MP_POSE_R_SHOULDER = 11, 12
MP_POSE_L_ELBOW, MP_POSE_R_ELBOW = 13, 14
MP_POSE_L_WRIST, MP_POSE_R_WRIST = 15, 16
MP_POSE_L_HIP, MP_POSE_R_HIP = 23, 24
MP_POSE_L_EYE, MP_POSE_R_EYE = 2, 5
MP_POSE_L_EAR, MP_POSE_R_EAR = 7, 8

# ---- face: MediaPipe FaceMesh 공식 영역 상수에서 서브샘플링한 34점 ----
# (눈썹5+5, 눈6+6, 입12 = 34). kslx.layout.FACE_KEEP 순서(눈썹→눈→입)에 맞춘다.
FACE_RIGHT_EYEBROW5 = [46, 53, 63, 70, 107]
FACE_LEFT_EYEBROW5 = [276, 283, 293, 300, 336]
FACE_RIGHT_EYE6 = [7, 144, 154, 158, 161, 246]
FACE_LEFT_EYE6 = [249, 373, 381, 385, 388, 466]
FACE_LIPS12 = [0, 37, 61, 82, 88, 178, 191, 291, 311, 318, 375, 415]
FACE_INDICES_34 = (FACE_RIGHT_EYEBROW5 + FACE_LEFT_EYEBROW5
                    + FACE_RIGHT_EYE6 + FACE_LEFT_EYE6 + FACE_LIPS12)
assert len(FACE_INDICES_34) == 34


def _xy(landmarks, idx: int, w: int, h: int) -> tuple[float, float]:
    lm = landmarks[idx]
    return lm.x * w, lm.y * h


def _mid(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def pose33_to_pose13(pose_landmarks, w: int, h: int) -> np.ndarray:
    if not pose_landmarks:
        return np.zeros((13, 2), dtype=np.float32)
    lm = pose_landmarks
    r_sho = _xy(lm, MP_POSE_R_SHOULDER, w, h)
    l_sho = _xy(lm, MP_POSE_L_SHOULDER, w, h)
    neck = _mid(r_sho, l_sho)
    r_hip = _xy(lm, MP_POSE_R_HIP, w, h)
    l_hip = _xy(lm, MP_POSE_L_HIP, w, h)
    mid_hip = _mid(r_hip, l_hip)
    pts = [
        _xy(lm, MP_POSE_NOSE, w, h), neck, r_sho,
        _xy(lm, MP_POSE_R_ELBOW, w, h), _xy(lm, MP_POSE_R_WRIST, w, h),
        l_sho, _xy(lm, MP_POSE_L_ELBOW, w, h), _xy(lm, MP_POSE_L_WRIST, w, h),
        mid_hip,
        _xy(lm, MP_POSE_R_EYE, w, h), _xy(lm, MP_POSE_L_EYE, w, h),
        _xy(lm, MP_POSE_R_EAR, w, h), _xy(lm, MP_POSE_L_EAR, w, h),
    ]
    return np.array(pts, dtype=np.float32)


def hand21_to_hand21(hand_landmarks, w: int, h: int) -> np.ndarray:
    """MediaPipe Hands 와 OpenPose hand 는 같은 21점 토폴로지(손목+손가락 4점x5)
    라 순서 변환 없이 그대로 쓴다."""
    if not hand_landmarks:
        return np.zeros((21, 2), dtype=np.float32)
    return np.array([_xy(hand_landmarks, i, w, h) for i in range(21)], dtype=np.float32)


def face_to_face34(face_landmarks, w: int, h: int) -> np.ndarray:
    if not face_landmarks:
        return np.zeros((34, 2), dtype=np.float32)
    return np.array([_xy(face_landmarks, i, w, h) for i in FACE_INDICES_34], dtype=np.float32)


def holistic_result_to_frame89(result, width: int, height: int, use_face: bool = True) -> np.ndarray:
    """HolisticLandmarkerResult (단일 프레임) -> (89, 2) kslx 레이아웃 배열."""
    pose = pose33_to_pose13(result.pose_landmarks[0] if result.pose_landmarks else None, width, height)
    lhand = hand21_to_hand21(result.left_hand_landmarks[0] if result.left_hand_landmarks else None, width, height)
    rhand = hand21_to_hand21(result.right_hand_landmarks[0] if result.right_hand_landmarks else None, width, height)
    if use_face:
        face = face_to_face34(result.face_landmarks[0] if result.face_landmarks else None, width, height)
    else:
        face = np.zeros((34, 2), dtype=np.float32)
    return np.concatenate([pose, lhand, rhand, face], axis=0)
