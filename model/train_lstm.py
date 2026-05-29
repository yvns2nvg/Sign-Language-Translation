"""
KSL 수어 인식 - BiLSTM + Attention 학습 스크립트
================================================
실행:
  python model/train_lstm.py

GPU 강력 권장 (CPU 단독은 수십 시간 소요)
  GPU 예상: ~4~12시간 (80 에포크 기준)
  CPU 예상: 60시간+

첫 실행 시 feat_cache/ 폴더에 특징 캐시를 자동 생성합니다 (30~60분).
이후 재실행 시 캐시를 재사용하므로 바로 학습 시작됩니다.

저장 파일:
  model/ksl_lstm_model.pt   ← 학습된 모델 가중치
  model/ksl_lstm_meta.pkl   ← 레이블 인코더
"""

import os, sys, time
import numpy as np
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder

# ── 경로 설정 ──
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_ROOT   = os.path.join(PROJECT_DIR, 'Dataset_NPZ', 'Dataset_NPZ')
CACHE_DIR   = os.path.join(SCRIPT_DIR, 'feat_cache')
MODEL_PATH  = os.path.join(SCRIPT_DIR, 'ksl_lstm_model.pt')
META_PATH   = os.path.join(SCRIPT_DIR, 'ksl_lstm_meta.pkl')

# ── 하이퍼파라미터 ──
FEAT_DIM   = 40    # 관절 각도 특징 차원
HIDDEN_DIM = 256   # BiLSTM 은닉층 (양방향이므로 출력은 512)
NUM_LAYERS = 2     # LSTM 레이어 수
DROPOUT    = 0.4
BATCH_SIZE = 128   # GPU 메모리 부족 시 64로 줄이기
LR         = 1e-3
EPOCHS     = 80
PATIENCE   = 12    # 검증 정확도가 N 에포크 동안 개선 없으면 조기 종료
VAL_RATIO  = 0.2

# ── 관절 각도 특징 추출 (벡터화) ──
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def extract_features_batch(X_batch):
    """
    X_batch : (N, T, 67, 3) numpy float32
    반환값  : (N, T, 40)  — 왼손 20 + 오른손 20 관절 각도/거리
    """
    N, T = X_batch.shape[:2]
    features = np.zeros((N, T, FEAT_DIM), dtype=np.float32)

    for hand_i, hand_start in enumerate([0, 21]):
        hand = X_batch[:, :, hand_start:hand_start+21, :]  # (N, T, 21, 3)
        base = hand_i * 20

        for ci, (parent, child) in enumerate(HAND_CONNECTIONS):
            if parent == 0:
                diff = hand[:, :, child] - hand[:, :, parent]
                features[:, :, base + ci] = np.linalg.norm(diff, axis=-1)
            else:
                v1  = hand[:, :, parent]   - hand[:, :, parent-1]
                v2  = hand[:, :, child]    - hand[:, :, parent]
                n1  = np.linalg.norm(v1, axis=-1)
                n2  = np.linalg.norm(v2, axis=-1)
                dot = np.sum(v1 * v2, axis=-1)
                cos = np.clip(dot / (n1 * n2 + 1e-6), -1.0, 1.0)
                features[:, :, base + ci] = np.arccos(cos)

    return features


# ── 1단계: 전처리 (NPZ → .npy 캐시) ──
def preprocess():
    os.makedirs(CACHE_DIR, exist_ok=True)
    npz_files = sorted([f for f in os.listdir(DATA_ROOT) if f.endswith('.npz')])

    if not npz_files:
        raise FileNotFoundError(f'NPZ 파일 없음: {DATA_ROOT}')

    print(f'[전처리] NPZ {len(npz_files)}개 → {CACHE_DIR}')
    total_t = time.time()

    for i, fname in enumerate(npz_files):
        feat_path = os.path.join(CACHE_DIR, f'feats_{i:02d}.npy')
        lbl_path  = os.path.join(CACHE_DIR, f'labels_{i:02d}.npy')

        if os.path.exists(feat_path) and os.path.exists(lbl_path):
            print(f'  [{i+1}/{len(npz_files)}] {fname} — 캐시 존재, 건너뜀')
            continue

        t0 = time.time()
        data = np.load(os.path.join(DATA_ROOT, fname), allow_pickle=True)
        X    = data['X'].astype(np.float32)          # (N, 148, 67, 3)
        y    = np.array([str(v) for v in data['V']]) # (N,)

        feats = extract_features_batch(X)             # (N, 148, 40)
        np.save(feat_path, feats)
        np.save(lbl_path,  y)

        print(f'  [{i+1}/{len(npz_files)}] {fname}  {feats.shape}  ({time.time()-t0:.0f}s)')

    print(f'전처리 완료  총 {time.time()-total_t:.0f}s\n')


# ── 2단계: 레이블 인코더 구성 ──
def build_label_encoder():
    print('[레이블 수집 중...]')
    all_labels = set()
    for f in sorted(os.listdir(CACHE_DIR)):
        if f.startswith('labels_'):
            all_labels.update(np.load(os.path.join(CACHE_DIR, f)))
    le = LabelEncoder()
    le.fit(sorted(all_labels))
    print(f'  총 클래스 수: {len(le.classes_)}\n')
    return le


# ── Dataset ──
class KSLDataset(Dataset):
    def __init__(self, feat_path, lbl_path, label_encoder, indices):
        # memory-map: 실제 데이터는 필요할 때만 디스크에서 읽음
        self.X    = np.load(feat_path, mmap_mode='r')
        y_str     = np.load(lbl_path)
        self.y    = label_encoder.transform(y_str)
        self.idx  = indices

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        ri = self.idx[i]
        # mmap 슬라이스를 copy()해서 실제 메모리로 가져옴
        return torch.from_numpy(self.X[ri].copy()), int(self.y[ri])


def build_loaders(label_encoder):
    feat_files = sorted([f for f in os.listdir(CACHE_DIR) if f.startswith('feats_')])
    train_sets, val_sets = [], []

    for ff in feat_files:
        num = ff.split('_')[1].split('.')[0]
        fp  = os.path.join(CACHE_DIR, ff)
        lp  = os.path.join(CACHE_DIR, f'labels_{num}.npy')
        n   = np.load(fp, mmap_mode='r').shape[0]

        idx = np.random.permutation(n)
        cut = int(n * (1 - VAL_RATIO))
        train_sets.append(KSLDataset(fp, lp, label_encoder, idx[:cut]))
        val_sets.append(  KSLDataset(fp, lp, label_encoder, idx[cut:]))

    train_ds = ConcatDataset(train_sets)
    val_ds   = ConcatDataset(val_sets)

    # Windows는 num_workers=0 (멀티프로세싱 이슈 방지)
    nw = 0 if sys.platform == 'win32' else 4

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=nw, pin_memory=True)
    return train_loader, val_loader


# ── 모델: BiLSTM + Attention ──
class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out, _   = self.lstm(x)                       # (B, T, H*2)
        attn_w   = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
        context  = (out * attn_w).sum(dim=1)          # (B, H*2)
        context  = self.norm(context)
        return self.head(context)


# ── 에포크 실행 ──
def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train(is_train)
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for step, (X_b, y_b) in enumerate(loader):
            X_b = X_b.to(device, non_blocking=True)
            y_b = torch.as_tensor(y_b, dtype=torch.long, device=device)

            logits = model(X_b)
            loss   = criterion(logits, y_b)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_b)
            correct    += (logits.argmax(1) == y_b).sum().item()
            n          += len(y_b)

            # 배치 진행 상황 출력 (100 배치마다)
            if is_train and (step + 1) % 100 == 0:
                print(f'    step {step+1}/{len(loader)}  '
                      f'loss={total_loss/n:.4f}  acc={correct/n:.4f}', flush=True)

    return total_loss / n, correct / n


# ── 메인 ──
def main():
    print('=' * 60)
    print('KSL BiLSTM+Attention 학습')
    print('=' * 60)

    assert os.path.isdir(DATA_ROOT), f'데이터 폴더 없음: {DATA_ROOT}'

    # 1. 전처리
    preprocess()

    # 2. 레이블 인코더
    le          = build_label_encoder()
    num_classes = len(le.classes_)

    # 3. DataLoader
    train_loader, val_loader = build_loaders(le)
    print(f'훈련 샘플: {len(train_loader.dataset):,}  '
          f'검증 샘플: {len(val_loader.dataset):,}')

    # 4. 디바이스 & 모델
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'디바이스: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('  ⚠ GPU 없음 — 학습이 매우 느립니다 (60h+)')

    model = BiLSTMAttention(
        FEAT_DIM, HIDDEN_DIM, NUM_LAYERS, num_classes, DROPOUT
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'모델 파라미터: {n_params:,}\n')

    # 5. 손실함수 / 옵티마이저 / 스케줄러
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    # 6. 학습 루프
    best_val_acc = 0.0
    no_improve   = 0

    print(f'[학습 시작] epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}')
    print('-' * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, is_train=False)

        elapsed = time.time() - t0
        print(f'Epoch {epoch:3d}/{EPOCHS}  '
              f'train={tr_acc:.4f}({tr_loss:.4f})  '
              f'val={vl_acc:.4f}({vl_loss:.4f})  '
              f'{elapsed:.0f}s', flush=True)

        scheduler.step(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                'model_state': model.state_dict(),
                'config': {
                    'input_dim':   FEAT_DIM,
                    'hidden_dim':  HIDDEN_DIM,
                    'num_layers':  NUM_LAYERS,
                    'num_classes': num_classes,
                    'dropout':     DROPOUT,
                }
            }, MODEL_PATH)
            joblib.dump({'label_encoder': le,
                         'label_names':   le.classes_.tolist()}, META_PATH)
            no_improve = 0
            print(f'  ✓ 모델 저장  best_val_acc={best_val_acc:.4f}')
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f'\nEarly stopping (patience={PATIENCE} 에포크 동안 개선 없음)')
                break

    print('\n' + '=' * 60)
    print(f'학습 완료!  최고 val_acc = {best_val_acc:.4f}')
    print(f'모델: {MODEL_PATH}')
    print(f'메타: {META_PATH}')
    print('=' * 60)


if __name__ == '__main__':
    main()
