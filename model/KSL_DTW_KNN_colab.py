# %% [markdown]
# # KSL 수어 인식 - MediaPipe DTW+KNN (로컬 실행 버전)
# gabguerin/Sign-Language-Recognition--MediaPipe-DTW 기반
#
# **데이터 형식**: `.npz` 파일, X shape = `(N, 148, 67, 3)`
# - `0:21` → 왼손 21개 키포인트
# - `21:42` → 오른손 21개 키포인트
# - `42:67` → 포즈 25개 키포인트
#
# **데이터 경로**: Dataset_NPZ/Dataset_NPZ/*.npz

# %%
# 셀 1: 패키지 설치 (최초 1회)
# !pip install dtaidistance scikit-learn joblib -q

# %%
# 셀 2: 경로 설정
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, 'Dataset_NPZ', 'Dataset_NPZ')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'ksl_dtw_knn_model.pkl')

print(f'데이터 경로: {DATA_ROOT}')
print(f'모델 저장 경로: {MODEL_SAVE_PATH}')

# %%
# 셀 3: 임포트 및 함수 정의
import numpy as np
import joblib
from sklearn.model_selection import LeaveOneOut
from dtaidistance import dtw

# ── 관절 각도 특징 추출 (위치/크기 불변) ──
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

# ── 데이터 증강 ──
def augment_sequence(sequence, n=15):
    augmented = []
    for _ in range(n):
        aug = sequence.copy()
        aug += np.random.normal(0, 0.005, aug.shape).astype(np.float32)
        scale = np.random.uniform(0.8, 1.2)
        new_len = max(5, int(aug.shape[0] * scale))
        aug = aug[np.linspace(0, aug.shape[0]-1, new_len).astype(int)]
        if np.random.random() < 0.5:
            tmp = aug.copy()
            tmp[:, 0:21, :] = aug[:, 21:42, :]
            tmp[:, 21:42, :] = aug[:, 0:21, :]
            tmp[:, :42, 0] = 1.0 - tmp[:, :42, 0]
            aug = tmp
        augmented.append(aug)
    return augmented

def augment_dataset(X, y, n=15):
    X_aug, y_aug = list(X), list(y)
    for seq, label in zip(X, y):
        X_aug.extend(augment_sequence(seq, n))
        y_aug.extend([label] * n)
    print(f'증강: {len(X)}개 → {len(X_aug)}개')
    return X_aug, y_aug

# ── DTW + KNN 분류기 ──
class DTWKNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.templates = []

    def fit(self, X_feat, y):
        self.templates = list(zip(X_feat, y))

    def _dist(self, s1, s2):
        return np.mean([
            dtw.distance_fast(s1[:, d].astype(np.double), s2[:, d].astype(np.double))
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

# ── 데이터 로드 (NPZ 형식) ──
def load_dataset(root, max_samples_per_class=None):
    """
    NPZ 파일에서 데이터 로드.
    max_samples_per_class: 클래스당 최대 샘플 수 (None이면 전체 로드)
    전체 90,000샘플은 DTW+KNN에 매우 느리므로 소규모 실험 시 제한 권장.
    """
    X, y = [], []
    class_counts = {}

    npz_files = sorted([f for f in os.listdir(root) if f.endswith('.npz')])
    if not npz_files:
        raise FileNotFoundError(f'NPZ 파일을 찾을 수 없습니다: {root}')

    print(f'NPZ 파일 {len(npz_files)}개 로드 중...')
    for fname in npz_files:
        data = np.load(os.path.join(root, fname), allow_pickle=True)
        X_batch = data['X']   # (N, frames, 67, 3)
        V_batch = data['V']   # (N,) string labels e.g. "WORD0001_고민"

        for i in range(X_batch.shape[0]):
            label = str(V_batch[i])
            if max_samples_per_class is not None:
                if class_counts.get(label, 0) >= max_samples_per_class:
                    continue
            X.append(X_batch[i].astype(np.float32))
            y.append(label)
            class_counts[label] = class_counts.get(label, 0) + 1

    labels = sorted(list(set(y)))
    print(f'로드 완료: {len(X)}개 샘플 / {len(labels)}개 클래스')
    return X, y, labels

print('함수 정의 완료!')

# %%
# 셀 4: 데이터 로드 + 특징 추출
# 주의: 전체 로드(max_samples_per_class=None)는 90,000샘플로 DTW+KNN이 매우 느립니다.
# 빠른 테스트는 max_samples_per_class=5 권장.
MAX_TRAIN_SAMPLES = 100  # 테스트용: 총 학습 데이터 수 제한
X_raw, y_raw, label_names = load_dataset(DATA_ROOT, max_samples_per_class=5)
if len(X_raw) > MAX_TRAIN_SAMPLES:
    X_raw = X_raw[:MAX_TRAIN_SAMPLES]
    y_raw = y_raw[:MAX_TRAIN_SAMPLES]
    label_names = sorted(list(set(y_raw)))
    print(f'테스트 모드: {MAX_TRAIN_SAMPLES}개 샘플 / {len(label_names)}개 클래스로 제한')

print('\n특징 추출 중...')
X_feat = [extract_features(seq) for seq in X_raw]
print(f'특징 shape 예시: {X_feat[0].shape}  (프레임수 × 40차원)')

# %%
# 셀 5: LOO 교차검증
print('[LOO 교차검증]')
loo = LeaveOneOut()
y_arr = np.array(y_raw)
correct = 0

for i, (tr_idx, te_idx) in enumerate(loo.split(X_feat)):
    clf = DTWKNNClassifier(k=5)
    clf.fit([X_feat[j] for j in tr_idx], y_arr[tr_idx].tolist())
    pred, conf, _ = clf.predict_one(X_feat[te_idx[0]])
    ok = pred == y_arr[te_idx[0]]
    if ok:
        correct += 1
    mark = '✓' if ok else '✗'
    print(f'  [{i+1:3d}] 정답={y_arr[te_idx[0]]:<30} 예측={pred:<30} {mark}')

print(f'\nLOO 정확도: {correct}/{len(X_feat)} = {correct/len(X_feat)*100:.1f}%')

# %%
# 셀 6: 데이터 증강 (×15) + 최종 모델 학습
print('[데이터 증강 × 15]')
X_aug_raw, y_aug = augment_dataset(X_raw, y_raw, n=15)

print('특징 추출 중 (증강 데이터)...')
X_aug_feat = [extract_features(seq) for seq in X_aug_raw]

final_clf = DTWKNNClassifier(k=5)
final_clf.fit(X_aug_feat, y_aug)
print('최종 모델 학습 완료!')

# %%
# 셀 7: 모델 저장 (로컬)
joblib.dump({'classifier': final_clf, 'label_names': label_names}, MODEL_SAVE_PATH)
print(f'모델 저장 완료: {MODEL_SAVE_PATH}')

# %%
# 셀 8: 테스트 - 첫 번째 샘플로 예측 확인
test_seq = X_raw[0]
test_feat = extract_features(test_seq)
pred, conf, top5 = final_clf.predict_one(test_feat)

print(f'실제 레이블: {y_raw[0]}')
print(f'예측 결과:   {pred}  (신뢰도: {conf*100:.1f}%)')
print('\nTop-5 유사도:')
for dist, lbl in top5:
    print(f'  {lbl}: {dist:.4f}')
