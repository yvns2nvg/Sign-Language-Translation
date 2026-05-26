"""
KSL 수어 실시간 인식 - 웹캠 추론 스크립트
================================================
사용법:
  python realtime_inference.py
  python realtime_inference.py --model path/to/ksl_dtw_knn_model.pkl

조작:
  SPACE  : 녹화 시작 / 중지 후 예측
  R      : 녹화 취소 (버퍼 초기화)
  Q      : 종료
"""

import argparse
import os
import sys
import numpy as np
import cv2
import joblib

# ── 경로 기본값 ──
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ksl_dtw_knn_model.pkl')

# ── 관절 각도 특징 추출 (학습 코드와 동일) ──
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))

def extract_hand_angles(hand_kp):
    angles = []
    for (parent, child) in HAND_CONNECTIONS:
        if parent == 0:
            angles.append(np.linalg.norm(hand_kp[child] - hand_kp[parent]))
        else:
            v1 = hand_kp[parent] - hand_kp[parent - 1]
            v2 = hand_kp[child] - hand_kp[parent]
            angles.append(angle_between(v1, v2))
    return np.array(angles, dtype=np.float32)

def extract_features(sequence):
    """sequence: (frames, 67, 3) → (frames, 40)"""
    return np.array([
        np.concatenate([
            extract_hand_angles(sequence[f, 0:21, :]),
            extract_hand_angles(sequence[f, 21:42, :])
        ])
        for f in range(sequence.shape[0])
    ], dtype=np.float32)

# ── DTW+KNN 분류기 (학습 코드와 동일) ──
try:
    from dtaidistance import dtw as dtw_lib

    class DTWKNNClassifier:
        def __init__(self, k=5):
            self.k = k
            self.templates = []

        def fit(self, X_feat, y):
            self.templates = list(zip(X_feat, y))

        def _dist(self, s1, s2):
            return np.mean([
                dtw_lib.distance(s1[:, d].astype(np.double), s2[:, d].astype(np.double))
                for d in range(s1.shape[1])
            ])

        def predict_one(self, query):
            dists = sorted([(self._dist(query, t), l) for t, l in self.templates])
            votes = {}
            for d, l in dists[:self.k]:
                votes[l] = votes.get(l, 0) + 1.0 / (d + 1e-6)
            best = max(votes, key=votes.get)
            conf = votes[best] / sum(votes.values())
            return best, conf, dists[:5]

except ImportError:
    print('[오류] dtaidistance 패키지가 없습니다.')
    print('  pip install dtaidistance')
    sys.exit(1)

# ── MediaPipe 키포인트 추출 ──
def extract_keypoints(results):
    """MediaPipe Holistic 결과 → (67, 3) numpy 배열"""
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
    else:
        lh = np.zeros((21, 3), dtype=np.float32)

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
    else:
        rh = np.zeros((21, 3), dtype=np.float32)

    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.pose_landmarks.landmark[:25]], dtype=np.float32)
    else:
        pose = np.zeros((25, 3), dtype=np.float32)

    return np.concatenate([lh, rh, pose], axis=0)  # (67, 3)

# ── 화면 UI 헬퍼 ──
def put_text(img, text, pos, scale=0.7, color=(255,255,255), thickness=2, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    if bg:
        cv2.rectangle(img, (x-4, y-th-4), (x+tw+4, y+bl+4), (0,0,0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_status(frame, recording, frame_count, result_text, conf_text, top5):
    h, w = frame.shape[:2]

    # 녹화 상태 표시
    if recording:
        cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
        put_text(frame, f'REC  {frame_count} frames', (50, 40), color=(0,0,255))
        # 프레임 수 프로그레스 바
        bar_w = int((frame_count / 60) * (w - 20))
        cv2.rectangle(frame, (10, h-15), (w-10, h-5), (50,50,50), -1)
        cv2.rectangle(frame, (10, h-15), (10+bar_w, h-5), (0,0,255), -1)
    else:
        put_text(frame, 'SPACE: 녹화시작  R: 취소  Q: 종료', (10, 30), scale=0.55)

    # 예측 결과 표시
    if result_text:
        put_text(frame, result_text, (10, h-60), scale=0.9, color=(0,255,100))
        put_text(frame, conf_text,   (10, h-30), scale=0.65, color=(200,200,200))

    # Top-5
    if top5:
        put_text(frame, '--- Top-5 ---', (w-240, 30), scale=0.55, color=(200,200,200))
        for idx, (dist, lbl) in enumerate(top5):
            put_text(frame, f'{idx+1}. {lbl}  ({dist:.3f})', (w-240, 55+idx*22),
                     scale=0.5, color=(180,220,255))

# ── 메인 ──
def main(model_path):
    # 패키지 확인
    try:
        import mediapipe as mp
    except ImportError:
        print('[오류] mediapipe 패키지가 없습니다.')
        print('  pip install mediapipe')
        sys.exit(1)

    # 모델 로드
    if not os.path.exists(model_path):
        print(f'[오류] 모델 파일을 찾을 수 없습니다: {model_path}')
        print('  --model 옵션으로 경로를 지정하거나 같은 폴더에 ksl_dtw_knn_model.pkl을 복사하세요.')
        sys.exit(1)

    print(f'모델 로드 중: {model_path}')
    data = joblib.load(model_path)
    clf = data['classifier']
    label_names = data['label_names']
    print(f'  클래스 수: {len(label_names)}')
    print(f'  템플릿 수: {len(clf.templates)}')

    # MediaPipe Holistic 초기화
    mp_holistic   = mp.solutions.holistic
    mp_drawing    = mp.solutions.drawing_utils
    mp_draw_style = mp.solutions.drawing_styles
    mp_hands      = mp.solutions.hands

    # 웹캠
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[오류] 웹캠을 열 수 없습니다.')
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recording    = False
    frames_buf   = []
    result_text  = ''
    conf_text    = ''
    top5_result  = []
    MIN_FRAMES   = 10
    MAX_FRAMES   = 60   # 자동 예측 최대 프레임 수

    print('\n준비 완료! 웹캠 창을 클릭하고 SPACE를 눌러 녹화 시작')

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # 랜드마크 그리기
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw_style.get_default_hand_landmarks_style(),
                mp_draw_style.get_default_hand_connections_style()
            )
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw_style.get_default_hand_landmarks_style(),
                mp_draw_style.get_default_hand_connections_style()
            )
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_draw_style.get_default_pose_landmarks_style()
            )

            # 키포인트 추출 및 버퍼 저장
            kp = extract_keypoints(results)
            if recording:
                frames_buf.append(kp)
                # 최대 프레임 도달 시 자동 예측
                if len(frames_buf) >= MAX_FRAMES:
                    sequence = np.array(frames_buf)
                    feat = extract_features(sequence)
                    pred, conf, top5 = clf.predict_one(feat)
                    word = pred.split('_', 1)[-1] if '_' in pred else pred
                    result_text = f'예측: {word}'
                    conf_text   = f'신뢰도: {conf*100:.1f}%  |  {pred}'
                    top5_result = top5
                    print(f'\n[예측] {pred}  (신뢰도: {conf*100:.1f}%)')
                    for i, (d, l) in enumerate(top5):
                        print(f'  {i+1}. {l}  dist={d:.4f}')
                    frames_buf = []
                    recording  = False

            # UI 그리기
            draw_status(frame, recording, len(frames_buf), result_text, conf_text, top5_result)
            cv2.imshow('KSL 수어 인식 (SPACE: 녹화, Q: 종료)', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                if not recording:
                    recording   = True
                    frames_buf  = []
                    result_text = ''
                    conf_text   = ''
                    top5_result = []
                    print('\n[녹화 시작] 수화를 보여주세요... (다시 SPACE를 눌러 예측)')
                else:
                    if len(frames_buf) < MIN_FRAMES:
                        print(f'[경고] 프레임이 너무 적습니다 ({len(frames_buf)}개, 최소 {MIN_FRAMES}개 필요)')
                    else:
                        sequence = np.array(frames_buf)
                        feat = extract_features(sequence)
                        print(f'  예측 중... ({len(frames_buf)} 프레임)', end='', flush=True)
                        pred, conf, top5 = clf.predict_one(feat)
                        word = pred.split('_', 1)[-1] if '_' in pred else pred
                        result_text = f'예측: {word}'
                        conf_text   = f'신뢰도: {conf*100:.1f}%  |  {pred}'
                        top5_result = top5
                        print(f'\n[예측] {pred}  (신뢰도: {conf*100:.1f}%)')
                        for i, (d, l) in enumerate(top5):
                            print(f'  {i+1}. {l}  dist={d:.4f}')
                    frames_buf = []
                    recording  = False
            elif key == ord('r') or key == ord('R'):
                frames_buf  = []
                recording   = False
                result_text = ''
                conf_text   = ''
                top5_result = []
                print('[취소] 버퍼 초기화')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KSL 수어 실시간 인식')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='모델 파일 경로 (기본: ksl_dtw_knn_model.pkl)')
    args = parser.parse_args()
    main(args.model)
