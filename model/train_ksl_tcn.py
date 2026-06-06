"""
KSL Sign Recognition - Improved Training Script  (v2)
======================================================

v1 대비 주요 개선:
  1. 학습 데이터 증강 (속도 변환 + 공간 노이즈)
     - 추론의 TTA [0.85, 1.0, 1.15] 와 동일한 속도 분포로 학습
     - 손/포즈 키포인트 변동에 강건해짐
  2. 서명자 독립 검증 (Signer-Independent Evaluation)
     - NPZ에 서명자 ID가 있으면 자동으로 서명자 단위 분리
     - 없으면 기존 랜덤 분리로 폴백 (경고 출력)
  3. 기존 v1 기능 전체 유지

Run:
  python model/train_ksl_tcn.py --model tcn
  python model/train_ksl_tcn.py --model tcn --aug          # 증강 활성화 (권장)
  python model/train_ksl_tcn.py --model tcn --aug --signer-key S  # 서명자 분리
  python model/train_ksl_tcn.py --model tcn --epochs 120 --batch-size 128 --aug
"""

import os
import sys
import csv
import time
import json
import argparse
import random

import numpy as np
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_DIR, "Dataset_NPZ", "Dataset_NPZ")
CACHE_DIR = os.path.join(SCRIPT_DIR, "feat_cache_improved")
MODEL_DIR = SCRIPT_DIR

from features import extract_improved_features_batch


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------------------
# Preprocessing and dataset
# -----------------------------------------------------------------------------
def preprocess(force: bool = False, signer_key: str = ""):
    """
    NPZ → (features, labels, [signers]) 캐시 생성.
    signer_key: NPZ에서 서명자 ID를 읽을 키 이름 (예: "S", "signer_id").
                비어 있으면 공통 키 이름을 자동으로 시도.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    npz_files = sorted([f for f in os.listdir(DATA_ROOT) if f.endswith(".npz")])
    if not npz_files:
        raise FileNotFoundError(f"NPZ files not found: {DATA_ROOT}")

    print(f"[Preprocess] NPZ {len(npz_files)} files -> {CACHE_DIR}")
    total_t = time.time()

    meta = {"files": [], "feature_dim": None}

    _SIGNER_KEY_CANDIDATES = [signer_key] if signer_key else ["S", "signer_id", "signer", "speaker", "person"]

    for i, fname in enumerate(npz_files):
        feat_path   = os.path.join(CACHE_DIR, f"feats_{i:02d}.npy")
        lbl_path    = os.path.join(CACHE_DIR, f"labels_{i:02d}.npy")
        signer_path = os.path.join(CACHE_DIR, f"signers_{i:02d}.npy")

        if not force and os.path.exists(feat_path) and os.path.exists(lbl_path):
            arr = np.load(feat_path, mmap_mode="r")
            meta["files"].append({"name": fname, "shape": tuple(arr.shape)})
            meta["feature_dim"] = int(arr.shape[-1])
            print(f"  [{i+1}/{len(npz_files)}] {fname} - cache exists {arr.shape}")
            continue

        t0 = time.time()
        data = np.load(os.path.join(DATA_ROOT, fname), allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = np.array([str(v) for v in data["V"]])

        # 서명자 ID 저장 시도
        if not os.path.exists(signer_path) or force:
            for key in _SIGNER_KEY_CANDIDATES:
                if key and key in data:
                    np.save(signer_path, np.array(data[key]))
                    print(f"    서명자 ID 저장: key='{key}', {len(data[key])} samples")
                    break

        feats = extract_improved_features_batch(X)
        np.save(feat_path, feats)
        np.save(lbl_path, y)

        meta["files"].append({"name": fname, "shape": tuple(feats.shape)})
        meta["feature_dim"] = int(feats.shape[-1])
        print(f"  [{i+1}/{len(npz_files)}] {fname} {feats.shape} ({time.time() - t0:.0f}s)")

    with open(os.path.join(CACHE_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[Preprocess done] total {time.time() - total_t:.0f}s\n")
    return meta["feature_dim"]


def build_label_encoder():
    all_labels = []
    for f in sorted(os.listdir(CACHE_DIR)):
        if f.startswith("labels_") and f.endswith(".npy"):
            all_labels.extend(np.load(os.path.join(CACHE_DIR, f)).tolist())
    le = LabelEncoder()
    le.fit(sorted(set(all_labels)))
    print(f"[Labels] num_classes={len(le.classes_)}")
    return le


# -----------------------------------------------------------------------------
# Feature-level augmentation
# -----------------------------------------------------------------------------
def _find_actual_length(feat: np.ndarray) -> int:
    """제로패딩된 특징에서 실제 유효 프레임 수 반환."""
    norms = np.linalg.norm(feat, axis=1)
    nz = np.where(norms > 1e-6)[0]
    return int(nz[-1]) + 1 if len(nz) > 0 else feat.shape[0]


def augment_features(feat: np.ndarray, speed: float = 1.0, jitter_std: float = 0.0) -> np.ndarray:
    """
    특징 레벨 증강. 기존 캐시를 수정하지 않고 런타임에 적용한다.

    feat       : (T, 206) 제로패딩된 특징 시퀀스
    speed      : 속도 배율 (0.7~1.3 범위 권장)
    jitter_std : base feature(0:103)에 추가할 Gaussian 노이즈 표준편차

    반환: (T, 206) (shape 불변, 제로패딩 유지)
    """
    DIM_BASE = 103   # base features; delta는 103:206
    T = feat.shape[0]

    T_actual = _find_actual_length(feat)
    if T_actual == 0:
        return feat

    result = feat.copy()

    # ── 속도 변환: base feature 리샘플 → delta 재계산 ──
    if abs(speed - 1.0) > 1e-4:
        T_new = max(10, int(T_actual * speed))
        base = feat[:T_actual, :DIM_BASE]
        idx = np.linspace(0, T_actual - 1, T_new)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, T_actual - 1)
        w = (idx - lo)[:, np.newaxis]
        base_new = (base[lo] * (1 - w) + base[hi] * w).astype(np.float32)
        delta_new = np.zeros_like(base_new)
        delta_new[1:] = base_new[1:] - base_new[:-1]
        result = np.zeros_like(feat)
        copy_len = min(T_new, T)
        result[:copy_len, :DIM_BASE]  = base_new[:copy_len]
        result[:copy_len, DIM_BASE:]  = delta_new[:copy_len]
        T_actual = copy_len

    # ── 공간 노이즈: base feature에 Gaussian 추가 → delta 재계산 ──
    if jitter_std > 0.0 and T_actual > 0:
        noise = np.random.randn(T_actual, DIM_BASE).astype(np.float32) * jitter_std
        result[:T_actual, :DIM_BASE] += noise
        if T_actual > 1:
            result[1:T_actual, DIM_BASE:] = (
                result[1:T_actual, :DIM_BASE] - result[:T_actual - 1, :DIM_BASE]
            )

    return result


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class KSLDataset(Dataset):
    def __init__(
        self,
        feat_path: str,
        lbl_path: str,
        label_encoder: LabelEncoder,
        indices,
        augment: bool = False,
        speed_range: tuple = (0.8, 1.2),
        jitter_std: float = 0.01,
    ):
        self.X = np.load(feat_path, mmap_mode="r")
        y_str = np.load(lbl_path)
        self.y = label_encoder.transform(y_str)
        self.idx = np.asarray(indices, dtype=np.int64)
        self.augment = augment
        self.speed_range = speed_range
        self.jitter_std = jitter_std

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        ri = self.idx[i]
        x = self.X[ri].copy()

        if self.augment:
            speed = float(np.random.uniform(*self.speed_range))
            x = augment_features(x, speed=speed, jitter_std=self.jitter_std)

        return torch.from_numpy(x).float(), int(self.y[ri])


def _signer_split(indices, signer_ids, num_val_signers: int, seed: int):
    """서명자 단위 학습/검증 분리."""
    unique = np.unique(signer_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_val = max(1, min(num_val_signers, len(unique) // 5))
    val_set = set(shuffled[:n_val].tolist())
    train_mask = np.array([s not in val_set for s in signer_ids])
    return indices[train_mask], indices[~train_mask], len(unique), n_val


def build_loaders(
    label_encoder,
    batch_size: int,
    val_ratio: float,
    seed: int,
    num_workers: int,
    use_weighted_sampler: bool,
    augment: bool = False,
    speed_range: tuple = (0.8, 1.2),
    jitter_std: float = 0.01,
    num_val_signers: int = 3,
):
    feat_files = sorted(
        [f for f in os.listdir(CACHE_DIR) if f.startswith("feats_") and f.endswith(".npy")]
    )
    train_sets, val_sets = [], []
    train_labels_all = []
    signer_split_used = False

    for ff in feat_files:
        num = ff.split("_")[1].split(".")[0]
        fp = os.path.join(CACHE_DIR, ff)
        lp = os.path.join(CACHE_DIR, f"labels_{num}.npy")
        sp = os.path.join(CACHE_DIR, f"signers_{num}.npy")

        y_str = np.load(lp)
        y     = label_encoder.transform(y_str)
        n     = len(y)
        indices = np.arange(n)

        # ── 서명자 독립 분리 (signer ID 캐시가 있을 때) ──
        if os.path.exists(sp):
            signer_ids = np.load(sp)
            train_idx, val_idx, n_total, n_val = _signer_split(
                indices, signer_ids, num_val_signers, seed
            )
            if not signer_split_used:
                print(f"  [서명자 분리] 총 {n_total}명, val {n_val}명 (서명자 독립)")
                signer_split_used = True
        else:
            # ── 폴백: 랜덤 stratified split ──
            try:
                train_idx, val_idx = train_test_split(
                    indices,
                    test_size=val_ratio,
                    random_state=seed,
                    shuffle=True,
                    stratify=y,
                )
            except ValueError:
                rng = np.random.default_rng(seed)
                idx = rng.permutation(n)
                cut = int(n * (1.0 - val_ratio))
                train_idx, val_idx = idx[:cut], idx[cut:]

        train_sets.append(
            KSLDataset(
                fp, lp, label_encoder, train_idx,
                augment=augment, speed_range=speed_range, jitter_std=jitter_std,
            )
        )
        val_sets.append(
            KSLDataset(fp, lp, label_encoder, val_idx, augment=False)
        )
        train_labels_all.extend(y[train_idx].tolist())

    if not signer_split_used:
        print("  [경고] 서명자 ID 캐시 없음 → 랜덤 분리 사용 (서명자 의존 평가)")
        print("         재학습 시 --signer-key 옵션으로 서명자 키를 지정하세요.")

    train_ds = ConcatDataset(train_sets)
    val_ds   = ConcatDataset(val_sets)

    pin = torch.cuda.is_available()

    if use_weighted_sampler:
        train_labels_all = np.asarray(train_labels_all)
        counts = np.bincount(train_labels_all, minlength=len(label_encoder.classes_))
        class_weight  = 1.0 / np.maximum(counts, 1)
        sample_weight = class_weight[train_labels_all]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weight, dtype=torch.double),
            num_samples=len(sample_weight),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, np.asarray(train_labels_all)


# -----------------------------------------------------------------------------
# Pooling layers
# -----------------------------------------------------------------------------
class AttnPool1D(nn.Module):
    """Input: (B, T, C), output: (B, C)."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


class AttnAvgMaxPool1D(nn.Module):
    """Attention + average + max pooling."""
    def __init__(self, dim):
        super().__init__()
        self.attn_pool = AttnPool1D(dim)

    def forward(self, x):
        attn = self.attn_pool(x)
        avg  = x.mean(dim=1)
        mx   = x.max(dim=1).values
        return torch.cat([attn, avg, mx], dim=-1)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class BiLSTMImproved(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * 2
        self.norm = nn.LayerNorm(out_dim)
        self.pool = AttnAvgMaxPool1D(out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim * 3, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.pool(self.norm(out)))


class GRUImproved(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * 2
        self.norm = nn.LayerNorm(out_dim)
        self.pool = AttnAvgMaxPool1D(out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim * 3, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(self.pool(self.norm(out)))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1    = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1   = Chomp1d(padding)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2   = Chomp1d(padding)
        self.bn2      = nn.BatchNorm1d(out_ch)
        self.dropout  = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.dropout(F.gelu(self.bn1(self.chomp1(self.conv1(x)))))
        y = self.dropout(F.gelu(self.bn2(self.chomp2(self.conv2(y)))))
        return F.gelu(y + (x if self.downsample is None else self.downsample(x)))


class TCNClassifier(nn.Module):
    def __init__(self, input_dim, channels, num_classes, dropout, kernel_size=5):
        super().__init__()
        layers, in_ch = [], input_dim
        for d in [1, 2, 4, 8, 16]:
            layers.append(TemporalBlock(in_ch, channels, kernel_size, d, dropout))
            in_ch = channels
        self.tcn  = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(channels)
        self.pool = AttnAvgMaxPool1D(channels)
        self.head = nn.Sequential(
            nn.Linear(channels * 3, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.head(self.pool(self.norm(x)))


def build_model(args, input_dim, num_classes):
    if args.model == "bilstm":
        return BiLSTMImproved(input_dim, args.hidden_dim, args.num_layers, num_classes, args.dropout)
    if args.model == "gru":
        return GRUImproved(input_dim, args.hidden_dim, args.num_layers, num_classes, args.dropout)
    if args.model == "tcn":
        return TCNClassifier(input_dim, args.tcn_channels, num_classes, args.dropout, args.kernel_size)
    raise ValueError(f"Unknown model: {args.model}")


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------
def topk_correct(logits, y, ks=(1, 3, 5)):
    max_k = min(max(ks), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1)
    pred    = pred.t()
    correct = pred.eq(y.view(1, -1).expand_as(pred))
    out = {}
    for k in ks:
        kk = min(k, logits.shape[1])
        out[k] = correct[:kk].reshape(-1).float().sum().item()
    return out


def run_epoch(model, loader, criterion, optimizer, device, scaler, is_train, grad_clip):
    model.train(is_train)
    total_loss = total_n = correct_top1 = correct_top3 = correct_top5 = 0
    all_pred, all_true = [], []

    with torch.set_grad_enabled(is_train):
        for step, (X_b, y_b) in enumerate(loader):
            X_b = X_b.to(device, non_blocking=True)
            y_b = torch.as_tensor(y_b, dtype=torch.long, device=device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            use_amp = scaler is not None and device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(X_b)
                loss   = criterion(logits, y_b)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            bs           = len(y_b)
            total_loss  += loss.item() * bs
            total_n     += bs
            tk           = topk_correct(logits.detach(), y_b, ks=(1, 3, 5))
            correct_top1 += tk[1]
            correct_top3 += tk[3]
            correct_top5 += tk[5]

            pred = logits.argmax(dim=1)
            all_pred.extend(pred.detach().cpu().numpy().tolist())
            all_true.extend(y_b.detach().cpu().numpy().tolist())

            if is_train and (step + 1) % 100 == 0:
                print(
                    f"    step {step+1}/{len(loader)} "
                    f"loss={total_loss/total_n:.4f} "
                    f"top1={correct_top1/total_n:.4f} "
                    f"top3={correct_top3/total_n:.4f}",
                    flush=True,
                )

    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return {
        "loss": total_loss / max(total_n, 1),
        "top1": correct_top1 / max(total_n, 1),
        "top3": correct_top3 / max(total_n, 1),
        "top5": correct_top5 / max(total_n, 1),
        "macro_f1": macro_f1,
        "y_true": all_true,
        "y_pred": all_pred,
    }


def make_class_weights(train_labels, num_classes, device):
    counts  = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def save_history_row(path, row, write_header=False):
    fields = [
        "epoch", "lr", "train_loss", "train_top1", "train_top3", "train_top5", "train_macro_f1",
        "val_loss", "val_top1", "val_top3", "val_top5", "val_macro_f1", "elapsed_sec",
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_final_reports(out_prefix, label_encoder, y_true, y_pred):
    labels       = list(range(len(label_encoder.classes_)))
    target_names = [str(x) for x in label_encoder.classes_]

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        zero_division=0, digits=4,
    )
    with open(out_prefix + "_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.save(out_prefix + "_confusion_matrix.npy", cm)

    confused = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                confused.append((cm[i, j], target_names[i], target_names[j]))
    confused.sort(reverse=True, key=lambda x: x[0])
    with open(out_prefix + "_top_confusions.txt", "w", encoding="utf-8") as f:
        for count, true_name, pred_name in confused[:100]:
            f.write(f"{count}\tTRUE={true_name}\tPRED={pred_name}\n")

    print("\n[Classification report]")
    print(report)
    print(f"[Reports saved] {out_prefix}_*.txt/npy")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # 모델 설정
    p.add_argument("--model", choices=["tcn", "bilstm", "gru"], default="tcn")
    p.add_argument("--epochs",       type=int,   default=120)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.35)
    p.add_argument("--hidden-dim",   type=int,   default=320)
    p.add_argument("--num-layers",   type=int,   default=2)
    p.add_argument("--tcn-channels", type=int,   default=256)
    p.add_argument("--kernel-size",  type=int,   default=5)

    # 학습 설정
    p.add_argument("--val-ratio",         type=float, default=0.2)
    p.add_argument("--patience",          type=int,   default=18)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--label-smoothing",   type=float, default=0.08)
    p.add_argument("--class-weight",      action="store_true")
    p.add_argument("--weighted-sampler",  action="store_true")
    p.add_argument("--force-preprocess",  action="store_true")
    p.add_argument("--num-workers",       type=int,   default=None)
    p.add_argument("--no-amp",            action="store_true")
    p.add_argument("--grad-clip",         type=float, default=1.0)

    # ── 데이터 증강 (v2 신규) ──────────────────────────────────
    p.add_argument(
        "--aug", action="store_true",
        help="학습 데이터 증강 활성화 (속도 변환 + 공간 노이즈). 권장."
    )
    p.add_argument(
        "--aug-speed-min", type=float, default=0.8,
        help="속도 변환 최솟값 (기본 0.8). TTA 범위와 일치시킬 것."
    )
    p.add_argument(
        "--aug-speed-max", type=float, default=1.2,
        help="속도 변환 최댓값 (기본 1.2)."
    )
    p.add_argument(
        "--aug-jitter", type=float, default=0.01,
        help="공간 노이즈 표준편차 (기본 0.01). 0이면 비활성화."
    )

    # ── 서명자 독립 분리 (v2 신규) ────────────────────────────
    p.add_argument(
        "--signer-key", type=str, default="",
        help="NPZ에서 서명자 ID를 읽을 키 이름 (예: S, signer_id). "
             "비워두면 S/signer_id/signer/speaker 순으로 자동 시도."
    )
    p.add_argument(
        "--num-val-signers", type=int, default=3,
        help="검증에 사용할 서명자 수 (기본 3). 전체 서명자의 최대 20%%까지 허용."
    )

    return p.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print("KSL Improved Training  v2")
    print("=" * 80)
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    assert os.path.isdir(DATA_ROOT), f"Dataset folder not found: {DATA_ROOT}"

    input_dim     = preprocess(force=args.force_preprocess, signer_key=args.signer_key)
    label_encoder = build_label_encoder()
    num_classes   = len(label_encoder.classes_)

    if args.num_workers is None:
        num_workers = 0 if sys.platform == "win32" else 4
    else:
        num_workers = args.num_workers

    if args.aug:
        print(f"[증강] speed=[{args.aug_speed_min}, {args.aug_speed_max}], jitter={args.aug_jitter}")
    else:
        print("[증강] 비활성화 (--aug 플래그 추가 권장)")

    train_loader, val_loader, train_labels = build_loaders(
        label_encoder=label_encoder,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=num_workers,
        use_weighted_sampler=args.weighted_sampler,
        augment=args.aug,
        speed_range=(args.aug_speed_min, args.aug_speed_max),
        jitter_std=args.aug_jitter,
        num_val_signers=args.num_val_signers,
    )

    print(f"[Data] train={len(train_loader.dataset):,}, val={len(val_loader.dataset):,}")
    print(f"[Data] input_dim={input_dim}, num_classes={num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    model    = build_model(args, input_dim, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {args.model}, params={n_params:,}")

    if args.class_weight:
        weights   = make_class_weights(train_labels, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
        print("[Loss] CrossEntropyLoss with class weights")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print("[Loss] CrossEntropyLoss")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=6, factor=0.5)
    scaler    = None if args.no_amp else torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    run_name     = f"ksl_{args.model}_improved"
    model_path   = os.path.join(MODEL_DIR, run_name + ".pt")
    meta_path    = os.path.join(MODEL_DIR, run_name + "_meta.pkl")
    history_path = os.path.join(MODEL_DIR, run_name + "_history.csv")
    out_prefix   = os.path.join(MODEL_DIR, run_name)

    if os.path.exists(history_path):
        os.remove(history_path)

    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    best_val_true = best_val_pred = None

    print("\n[Training start]")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0           = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler, True,  args.grad_clip)
        val_metrics   = run_epoch(model, val_loader,   criterion, optimizer, device, scaler, False, args.grad_clip)
        elapsed       = time.time() - t0

        score = val_metrics["macro_f1"]
        scheduler.step(score)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"lr={lr:.2e} "
            f"train_top1={train_metrics['top1']:.4f} train_f1={train_metrics['macro_f1']:.4f} "
            f"val_top1={val_metrics['top1']:.4f} val_top3={val_metrics['top3']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"loss={val_metrics['loss']:.4f} "
            f"{elapsed:.0f}s",
            flush=True,
        )

        save_history_row(
            history_path,
            {
                "epoch": epoch, "lr": lr,
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"],
                "train_top3": train_metrics["top3"],
                "train_top5": train_metrics["top5"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
                "val_top3": val_metrics["top3"],
                "val_top5": val_metrics["top5"],
                "val_macro_f1": val_metrics["macro_f1"],
                "elapsed_sec": elapsed,
            },
            write_header=(epoch == 1),
        )

        if score > best_score:
            best_score     = score
            best_epoch     = epoch
            no_improve     = 0
            best_val_true  = val_metrics["y_true"]
            best_val_pred  = val_metrics["y_pred"]

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "model": args.model,
                        "input_dim": input_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "tcn_channels": args.tcn_channels,
                        "kernel_size": args.kernel_size,
                        "num_classes": num_classes,
                        "dropout": args.dropout,
                    },
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_score,
                    "val_top1_at_best": val_metrics["top1"],
                    "val_top3_at_best": val_metrics["top3"],
                },
                model_path,
            )
            joblib.dump(
                {
                    "label_encoder": label_encoder,
                    "label_names": label_encoder.classes_.tolist(),
                    "feature_cache": CACHE_DIR,
                    "feature_dim": input_dim,
                },
                meta_path,
            )
            print(
                f"  saved best  epoch={best_epoch}  "
                f"val_macro_f1={best_score:.4f}  val_top1={val_metrics['top1']:.4f}"
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n[Early stopping] no improvement for {args.patience} epochs")
                break

    print("\n" + "=" * 80)
    print(f"Training done.  best_epoch={best_epoch}  best_val_macro_f1={best_score:.4f}")
    print(f"Model : {model_path}")
    print(f"Meta  : {meta_path}")
    print(f"History: {history_path}")
    print("=" * 80)

    if best_val_true is not None:
        save_final_reports(out_prefix, label_encoder, best_val_true, best_val_pred)


if __name__ == "__main__":
    main()
