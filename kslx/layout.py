"""키포인트 레이아웃 정의.

AI Hub 라벨링 JSON은 OpenPose BODY_25 + hand_left21 + hand_right21 + face70
포맷이다 (2D는 x,y,conf 세 값씩, 3D도 별도 필드로 존재).

★ 3D 키포인트는 쓰지 않는다 ★
같은 take(수어사 x 단어)의 5개 카메라 앵글(F/D/L/R/U) 파일에 든
`*_keypoints_3d` 값은 멀티뷰 삼각측량 결과 하나를 5개 파일에 그대로
복사해 넣은 것이라 앵글마다 비트 단위로 동일하다. 이걸 피처로 쓰면
랜덤 분할에서 val 샘플이 train 샘플(다른 앵글, 같은 take)의 문자 그대로의
복사본이 되어 leak이 생긴다. 이 프로젝트는 반드시 `*_keypoints_2d` 만 쓴다.
(증거: kslx.leak_report)

89점 축소 레이아웃 = pose13 + hand_left21 + hand_right21 + face34.
전신(다리/발) 25점 중 수어와 무관한 하반신을 버리고, 얼굴 70점 중
비수지 문법 표지(눈썹/눈/입)만 남겨 34점으로 줄인다.
"""

from __future__ import annotations

# OpenPose BODY_25 인덱스 정의는 https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 참고. 수어 인식에 쓰이는 상반신 13점만 남긴다.
POSE25_KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18]
POSE13_NAMES = [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist", "mid_hip",
    "r_eye", "l_eye", "r_ear", "l_ear",
]
POSE13_NECK_IDX = 1          # POSE13_NAMES 안에서의 위치 (정규화 원점)
POSE13_R_SHOULDER_IDX = 2
POSE13_L_SHOULDER_IDX = 5

N_POSE = 13
N_HAND = 21          # 좌/우 각각, 토폴로지 변경 없음
N_FACE_RAW = 70

# OpenPose face70 = dlib68(잡선0-16, 눈썹17-26, 코27-35, 눈36-47, 입48-67) + 동공68-69.
# 수어의 비수지 문법 표지(눈썹 올림/찡그림, 눈 크기, 입모양)만 남기고
# 턱선/코/입 안쪽/동공은 버려 34점으로 줄인다.
FACE_KEEP = list(range(17, 27)) + list(range(36, 48)) + list(range(48, 60))
assert len(FACE_KEEP) == 34
N_FACE = 34

N_TOTAL = N_POSE + 2 * N_HAND + N_FACE  # 13 + 21*2 + 34 = 89
assert N_TOTAL == 89

# 축소된 89점 배열 안에서 각 파트의 슬라이스
POSE_SLICE = slice(0, N_POSE)
LHAND_SLICE = slice(N_POSE, N_POSE + N_HAND)
RHAND_SLICE = slice(N_POSE + N_HAND, N_POSE + 2 * N_HAND)
FACE_SLICE = slice(N_POSE + 2 * N_HAND, N_TOTAL)

# 미러링(좌우 반전) 시 좌/우가 바뀌는 관계를 augment.py 에서 쓴다.
POSE13_MIRROR_PAIRS = [
    (2, 5), (3, 6), (4, 7),   # 어깨/팔꿈치/손목
    (9, 10), (11, 12),        # 눈/귀
]


def extract_openpose_2d(person: dict) -> "tuple[list[float], list[float]]":
    """OpenPose 스타일 `people` dict 하나에서 89점 (x, y) 를 뽑는다.

    입력 JSON의 `*_keypoints_2d` 는 [x0,y0,c0, x1,y1,c1, ...] 플랫 배열이다.
    반환: (xs, ys) 각각 길이 89.
    """
    def take(flat, keep_idx):
        pts = []
        for i in keep_idx:
            pts.append((flat[3 * i], flat[3 * i + 1]))
        return pts

    pose_pts = take(person["pose_keypoints_2d"], POSE25_KEEP)
    lhand_pts = take(person["hand_left_keypoints_2d"], range(N_HAND))
    rhand_pts = take(person["hand_right_keypoints_2d"], range(N_HAND))
    face_pts = take(person["face_keypoints_2d"], FACE_KEEP)

    all_pts = pose_pts + lhand_pts + rhand_pts + face_pts
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    return xs, ys
