"""
KSL Sign Recognition - Improved Training Script
================================================

Key improvements over the original BiLSTM+Attention script:
1. Richer features
   - Original hand angle/distance features: 40 dims
   - Hand wrist relative position features
   - Left/right hand relative features
   - Fingertip coordinates
   - Velocity/delta features
   - Optional body/face reference features from remaining landmarks

2. Model choices
   - BiLSTM + improved pooling
   - Residual TCN + attention pooling
   - GRU + improved pooling

3. Better training / evaluation
   - Random seed
   - Stratified global train/val split when possible
   - Class-weighted loss option
   - Label smoothing
   - Mixed precision training
   - ReduceLROnPlateau
   - Early stopping
   - CSV history save
   - Classification report
   - Confusion matrix
   - Top-k accuracy

Run:
  python model/train_ksl_improved.py --model tcn
  python model/train_ksl_improved.py --model bilstm
  python model/train_ksl_improved.py --model gru

Recommended first experiment:
  python model/train_ksl_improved.py --model tcn --epochs 120 --batch-size 128

If GPU memory is insufficient:
  python model/train_ksl_improved.py --model tcn --batch-size 64
"""

import os
import sys
import csv
import time
import math
import json
import argparse
import random
from dataclasses import dataclass, asdict

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
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_DIR, "Dataset_NPZ", "Dataset_NPZ")
CACHE_DIR = os.path.join(SCRIPT_DIR, "feat_cache_improved")
MODEL_DIR = SCRIPT_DIR


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
# Feature extraction
# -----------------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_TIPS = [4, 8, 12, 16, 20]


def safe_norm(x, axis=-1, keepdims=False, eps=1e-6):
    return np.linalg.norm(x, axis=axis, keepdims=keepdims) + eps


def normalize_hand_batch(hand):
    """
    hand: (N, T, 21, 3)
    Normalize each hand by wrist origin and wrist-to-middle-MCP scale.
    """
    center = hand[:, :, 0:1, :]
    hand_c = hand - center
    scale = np.linalg.norm(hand_c[:, :, 9, :], axis=-1, keepdims=True)
    scale = scale[:, :, :, None]
    return hand_c / (scale + 1e-6), center.squeeze(2), scale.squeeze(2)


def extract_original_angle_features(X_batch):
    """
    Returns original 40-dim features:
    left hand 20 + right hand 20 angle/distance features.
    """
    N, T = X_batch.shape[:2]
    features = np.zeros((N, T, 40), dtype=np.float32)

    for hand_i, hand_start in enumerate([0, 21]):
        hand = X_batch[:, :, hand_start:hand_start + 21, :]
        hand_norm, _, _ = normalize_hand_batch(hand)
        base = hand_i * 20

        for ci, (parent, child) in enumerate(HAND_CONNECTIONS):
            if parent == 0:
                diff = hand_norm[:, :, child] - hand_norm[:, :, parent]
                features[:, :, base + ci] = np.linalg.norm(diff, axis=-1)
            else:
                v1 = hand_norm[:, :, parent] - hand_norm[:, :, parent - 1]
                v2 = hand_norm[:, :, child] - hand_norm[:, :, parent]
                n1 = np.linalg.norm(v1, axis=-1)
                n2 = np.linalg.norm(v2, axis=-1)
                dot = np.sum(v1 * v2, axis=-1)
                cos = np.clip(dot / (n1 * n2 + 1e-6), -1.0, 1.0)
                features[:, :, base + ci] = np.arccos(cos)

    return features


def temporal_delta(feat):
    """feat: (N, T, C) -> delta: (N, T, C)."""
    d = np.zeros_like(feat, dtype=np.float32)
    d[:, 1:] = feat[:, 1:] - feat[:, :-1]
    return d


def extract_improved_features_batch(X_batch):
    """
    X_batch: (N, T, 67, 3)

    Feature groups:
    1. Original hand angle/distance features: 40
    2. Normalized fingertip coordinates: 30
    3. Wrist positions and left-right relation: 15
    4. Hand open/size descriptors: 6
    5. Optional body/face reference relation from remaining landmarks: up to 18
    6. Delta features for selected dynamic groups

    Output dim may be around 180~220 depending on whether extra landmarks exist.
    """
    X = X_batch.astype(np.float32)
    N, T = X.shape[:2]

    # Split hands. Existing code assumes left hand = 0:21, right hand = 21:42.
    left = X[:, :, 0:21, :]
    right = X[:, :, 21:42, :]

    left_norm, left_wrist, left_scale = normalize_hand_batch(left)
    right_norm, right_wrist, right_scale = normalize_hand_batch(right)

    # 1) Original angle/distance features.
    angle_feats = extract_original_angle_features(X)

    # 2) Fingertip normalized coordinates.
    left_tips = left_norm[:, :, FINGER_TIPS, :].reshape(N, T, -1)
    right_tips = right_norm[:, :, FINGER_TIPS, :].reshape(N, T, -1)
    tip_feats = np.concatenate([left_tips, right_tips], axis=-1).astype(np.float32)

    # 3) Wrist and left-right relation.
    # Use global center/scale so position information is not completely removed.
    # If body landmarks exist, use all-landmark center/scale as a rough camera normalization.
    valid = X
    global_center = np.mean(valid, axis=2, keepdims=True)
    global_scale = safe_norm(valid - global_center, axis=-1, keepdims=True).mean(axis=2, keepdims=True)
    Xg = (X - global_center) / (global_scale + 1e-6)

    left_wrist_g = Xg[:, :, 0, :]
    right_wrist_g = Xg[:, :, 21, :]
    wrist_rel = right_wrist_g - left_wrist_g
    wrist_dist = safe_norm(wrist_rel, axis=-1, keepdims=True)
    wrist_mid = 0.5 * (left_wrist_g + right_wrist_g)
    wrist_feats = np.concatenate(
        [left_wrist_g, right_wrist_g, wrist_rel, wrist_dist, wrist_mid], axis=-1
    ).astype(np.float32)  # 3+3+3+1+3 = 13

    # 4) Hand spread/open descriptors.
    left_tip_dists = safe_norm(left_norm[:, :, FINGER_TIPS, :] - left_norm[:, :, 0:1, :], axis=-1)
    right_tip_dists = safe_norm(right_norm[:, :, FINGER_TIPS, :] - right_norm[:, :, 0:1, :], axis=-1)
    hand_spread = np.concatenate(
        [
            left_tip_dists.mean(axis=-1, keepdims=True),
            left_tip_dists.std(axis=-1, keepdims=True),
            right_tip_dists.mean(axis=-1, keepdims=True),
            right_tip_dists.std(axis=-1, keepdims=True),
            left_scale.reshape(N, T, -1).mean(axis=-1, keepdims=True),
            right_scale.reshape(N, T, -1).mean(axis=-1, keepdims=True),
        ],
        axis=-1,
    ).astype(np.float32)

    # 5) Optional reference features using remaining landmarks 42:67.
    # The exact semantic meaning depends on the dataset, but for sign recognition,
    # hand-to-body/face relative position is often very useful.
    extra_feats = []
    if X.shape[2] > 42:
        extra = Xg[:, :, 42:, :]  # (N, T, M, 3)
        ref_center = extra.mean(axis=2)  # rough face/body center
        ref_std = extra.std(axis=2)
        lw_to_ref = left_wrist_g - ref_center
        rw_to_ref = right_wrist_g - ref_center
        extra_feats.append(ref_center.astype(np.float32))
        extra_feats.append(ref_std.astype(np.float32))
        extra_feats.append(lw_to_ref.astype(np.float32))
        extra_feats.append(rw_to_ref.astype(np.float32))
        extra_feats.append(safe_norm(lw_to_ref, axis=-1, keepdims=True).astype(np.float32))
        extra_feats.append(safe_norm(rw_to_ref, axis=-1, keepdims=True).astype(np.float32))

    base_groups = [angle_feats, tip_feats, wrist_feats, hand_spread] + extra_feats
    base = np.concatenate(base_groups, axis=-1).astype(np.float32)

    # 6) Dynamic features: delta of selected groups.
    # Include delta for base; this gives movement direction/speed information.
    delta = temporal_delta(base)

    feats = np.concatenate([base, delta], axis=-1).astype(np.float32)

    # Clean NaN/Inf just in case landmark detector produced bad values.
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feats


# -----------------------------------------------------------------------------
# Preprocessing and dataset
# -----------------------------------------------------------------------------
def preprocess(force: bool = False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    npz_files = sorted([f for f in os.listdir(DATA_ROOT) if f.endswith(".npz")])
    if not npz_files:
        raise FileNotFoundError(f"NPZ files not found: {DATA_ROOT}")

    print(f"[Preprocess] NPZ {len(npz_files)} files -> {CACHE_DIR}")
    total_t = time.time()

    meta = {"files": [], "feature_dim": None}

    for i, fname in enumerate(npz_files):
        feat_path = os.path.join(CACHE_DIR, f"feats_{i:02d}.npy")
        lbl_path = os.path.join(CACHE_DIR, f"labels_{i:02d}.npy")

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


class KSLDataset(Dataset):
    def __init__(self, feat_path, lbl_path, label_encoder, indices):
        self.X = np.load(feat_path, mmap_mode="r")
        y_str = np.load(lbl_path)
        self.y = label_encoder.transform(y_str)
        self.idx = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        ri = self.idx[i]
        x = torch.from_numpy(self.X[ri].copy()).float()
        y = int(self.y[ri])
        return x, y


def build_loaders(label_encoder, batch_size, val_ratio, seed, num_workers, use_weighted_sampler):
    feat_files = sorted([f for f in os.listdir(CACHE_DIR) if f.startswith("feats_") and f.endswith(".npy")])
    train_sets, val_sets = [], []
    train_labels_all = []

    for ff in feat_files:
        num = ff.split("_")[1].split(".")[0]
        fp = os.path.join(CACHE_DIR, ff)
        lp = os.path.join(CACHE_DIR, f"labels_{num}.npy")
        y_str = np.load(lp)
        y = label_encoder.transform(y_str)
        n = len(y)
        indices = np.arange(n)

        # Stratified split per file if possible.
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

        train_sets.append(KSLDataset(fp, lp, label_encoder, train_idx))
        val_sets.append(KSLDataset(fp, lp, label_encoder, val_idx))
        train_labels_all.extend(y[train_idx].tolist())

    train_ds = ConcatDataset(train_sets)
    val_ds = ConcatDataset(val_sets)

    pin = torch.cuda.is_available()

    if use_weighted_sampler:
        train_labels_all = np.asarray(train_labels_all)
        counts = np.bincount(train_labels_all, minlength=len(label_encoder.classes_))
        class_weight = 1.0 / np.maximum(counts, 1)
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
        avg = x.mean(dim=1)
        mx = x.max(dim=1).values
        return torch.cat([attn, avg, mx], dim=-1)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class BiLSTMImproved(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * 2
        self.norm = nn.LayerNorm(out_dim)
        self.pool = AttnAvgMaxPool1D(out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out)
        pooled = self.pool(out)
        return self.head(pooled)


class GRUImproved(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * 2
        self.norm = nn.LayerNorm(out_dim)
        self.pool = AttnAvgMaxPool1D(out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.norm(out)
        pooled = self.pool(out)
        return self.head(pooled)


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
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.conv1(x)
        y = self.chomp1(y)
        y = self.bn1(y)
        y = F.gelu(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = self.bn2(y)
        y = F.gelu(y)
        y = self.dropout(y)

        res = x if self.downsample is None else self.downsample(x)
        return F.gelu(y + res)


class TCNClassifier(nn.Module):
    def __init__(self, input_dim, channels, num_classes, dropout, kernel_size=5):
        super().__init__()
        layers = []
        dilations = [1, 2, 4, 8, 16]
        in_ch = input_dim
        for d in dilations:
            layers.append(TemporalBlock(in_ch, channels, kernel_size, d, dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(channels)
        self.pool = AttnAvgMaxPool1D(channels)
        self.head = nn.Sequential(
            nn.Linear(channels * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C) -> Conv1d expects (B, C, T)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        pooled = self.pool(x)
        return self.head(pooled)


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
    pred = pred.t()
    correct = pred.eq(y.view(1, -1).expand_as(pred))
    out = {}
    for k in ks:
        kk = min(k, logits.shape[1])
        out[k] = correct[:kk].reshape(-1).float().sum().item()
    return out


def run_epoch(model, loader, criterion, optimizer, device, scaler, is_train, grad_clip):
    model.train(is_train)
    total_loss = 0.0
    total_n = 0
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0

    all_pred = []
    all_true = []

    with torch.set_grad_enabled(is_train):
        for step, (X_b, y_b) in enumerate(loader):
            X_b = X_b.to(device, non_blocking=True)
            y_b = torch.as_tensor(y_b, dtype=torch.long, device=device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            use_amp = scaler is not None and device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(X_b)
                loss = criterion(logits, y_b)

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

            bs = len(y_b)
            total_loss += loss.item() * bs
            total_n += bs

            tk = topk_correct(logits.detach(), y_b, ks=(1, 3, 5))
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
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
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
    labels = list(range(len(label_encoder.classes_)))
    target_names = [str(x) for x in label_encoder.classes_]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        digits=4,
    )
    with open(out_prefix + "_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.save(out_prefix + "_confusion_matrix.npy", cm)

    # Top confused pairs, excluding diagonal.
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
    print(f"[Reports saved] {out_prefix}_classification_report.txt")
    print(f"[Reports saved] {out_prefix}_confusion_matrix.npy")
    print(f"[Reports saved] {out_prefix}_top_confusions.txt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["tcn", "bilstm", "gru"], default="tcn")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--hidden-dim", type=int, default=320)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--tcn-channels", type=int, default=256)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=18)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-smoothing", type=float, default=0.08)
    p.add_argument("--class-weight", action="store_true", help="Use class weights in CrossEntropyLoss.")
    p.add_argument("--weighted-sampler", action="store_true", help="Use WeightedRandomSampler for imbalanced classes.")
    p.add_argument("--force-preprocess", action="store_true")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print("KSL Improved Training")
    print("=" * 80)
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    assert os.path.isdir(DATA_ROOT), f"Dataset folder not found: {DATA_ROOT}"

    input_dim = preprocess(force=args.force_preprocess)
    label_encoder = build_label_encoder()
    num_classes = len(label_encoder.classes_)

    if args.num_workers is None:
        num_workers = 0 if sys.platform == "win32" else 4
    else:
        num_workers = args.num_workers

    train_loader, val_loader, train_labels = build_loaders(
        label_encoder=label_encoder,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=num_workers,
        use_weighted_sampler=args.weighted_sampler,
    )

    print(f"[Data] train={len(train_loader.dataset):,}, val={len(val_loader.dataset):,}")
    print(f"[Data] input_dim={input_dim}, num_classes={num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    model = build_model(args, input_dim, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {args.model}, params={n_params:,}")

    if args.class_weight:
        weights = make_class_weights(train_labels, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
        print("[Loss] CrossEntropyLoss with class weights")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print("[Loss] CrossEntropyLoss")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=6, factor=0.5)
    scaler = None if args.no_amp else torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    run_name = f"ksl_{args.model}_improved"
    model_path = os.path.join(MODEL_DIR, run_name + ".pt")
    meta_path = os.path.join(MODEL_DIR, run_name + "_meta.pkl")
    history_path = os.path.join(MODEL_DIR, run_name + "_history.csv")
    out_prefix = os.path.join(MODEL_DIR, run_name)

    if os.path.exists(history_path):
        os.remove(history_path)

    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    best_val_true, best_val_pred = None, None

    print("\n[Training start]")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler, True, args.grad_clip)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, scaler, False, args.grad_clip)
        elapsed = time.time() - t0

        # Use macro F1 as the main score if class imbalance exists. It is often more meaningful than accuracy.
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
                "epoch": epoch,
                "lr": lr,
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
            best_score = score
            best_epoch = epoch
            no_improve = 0
            best_val_true = val_metrics["y_true"]
            best_val_pred = val_metrics["y_pred"]

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
                f"  saved best model: epoch={best_epoch}, "
                f"val_macro_f1={best_score:.4f}, val_top1={val_metrics['top1']:.4f}"
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n[Early stopping] no improvement for {args.patience} epochs")
                break

    print("\n" + "=" * 80)
    print(f"Training done. best_epoch={best_epoch}, best_val_macro_f1={best_score:.4f}")
    print(f"Model: {model_path}")
    print(f"Meta: {meta_path}")
    print(f"History: {history_path}")
    print("=" * 80)

    if best_val_true is not None:
        save_final_reports(out_prefix, label_encoder, best_val_true, best_val_pred)


if __name__ == "__main__":
    main()
