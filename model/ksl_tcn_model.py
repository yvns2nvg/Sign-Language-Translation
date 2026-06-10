"""
KSL Sign Recognition - Improved Training Script  v3
=====================================================

v3 주요 변경 (v2 대비):
  1. 서명자 독립 검증 (Signer-Independent Evaluation)
     - 클래스 내 위치 기반 서명자 ID 자동 추론 (--num-signers, --angles-per-signer)
     - NPZ에 서명자 키가 있으면 우선 사용 (--signer-key)
     - val 서명자를 train 서명자와 완전 분리 → 실제 일반화 성능 측정 가능

  2. 데이터 증강 (--aug 플래그로 활성화)
     - 속도 변환 [--aug-speed-min, --aug-speed-max]: 추론 TTA 범위와 동일한 분포 학습
     - 공간 노이즈 (--aug-jitter): 웹캠 키포인트 흔들림·조명 오차 시뮬레이션
     - 랜덤 크롭 (--aug-crop-prob, --aug-crop-min): 30~90 프레임 짧은 녹화 시뮬레이션

실행 (권장):
  python model/ksl_tcn_model.py --model tcn --epochs 120 --batch-size 128 ^
      --aug --num-signers 16 --angles-per-signer 5 --val-signers 2

기존 방식 (하위 호환):
  python model/ksl_tcn_model.py --model tcn --epochs 120 --batch-size 128
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
# Signer ID inference
# -----------------------------------------------------------------------------
def build_signer_ids(y_str, num_signers=16, angles_per_signer=5):
    """
    클래스 내 위치 기반으로 서명자 ID를 추론한다.

    AI Hub KSL 데이터셋 구조 가정:
      각 클래스(단어) 내 샘플이
      [signer0_ang0..4, signer1_ang0..4, ..., signer15_ang0..4] 순으로 저장됨.

    y_str            : (N,) str  레이블 배열
    num_signers      : 전체 서명자 수 (기본 16)
    angles_per_signer: 서명자당 각도 수 (기본 5)
    반환             : (N,) int32  서명자 ID (0 ~ num_signers-1)
    """
    signer_ids = np.zeros(len(y_str), dtype=np.int32)
    for cls in np.unique(y_str):
        indices = np.where(y_str == cls)[0]
        for local_pos, global_idx in enumerate(indices):
            signer_ids[global_idx] = (local_pos // angles_per_signer) % num_signers
    return signer_ids


# -----------------------------------------------------------------------------
# Augmentation helpers
# -----------------------------------------------------------------------------
def _find_actual_length(feat):
    """제로패딩된 특징 (T, D)에서 실제 유효 프레임 수 반환."""
    norms = np.linalg.norm(feat, axis=1)
    nz = np.where(norms > 1e-6)[0]
    return int(nz[-1]) + 1 if len(nz) > 0 else feat.shape[0]


def augment_features(feat, speed=1.0, jitter_std=0.0, crop_ratio=None):
    """
    특징 레벨 증강. 입출력 shape 불변 (T, 206).

    speed      : 속도 배율 (0.8~1.2). 1.0이면 생략.
    jitter_std : base feature(0:103)에 추가할 Gaussian 노이즈 std.
                 웹캠 키포인트 흔들림·조명 차이를 시뮬레이션.
    crop_ratio : None이면 생략. 0~1 범위면 실제 길이를 이 비율로 줄이고 이후 제로패딩.
                 추론 시 30~90 프레임 짧은 녹화를 학습 중에도 경험하게 한다.
    """
    DIM_BASE = 103
    T = feat.shape[0]
    T_actual = _find_actual_length(feat)
    if T_actual == 0:
        return feat

    result = feat.copy()

    # 1. 크롭: 짧은 녹화 시뮬레이션
    if crop_ratio is not None and crop_ratio < 1.0:
        T_crop = max(15, int(T_actual * crop_ratio))
        result[T_crop:] = 0.0
        T_actual = T_crop

    # 2. 속도 변환: base 리샘플 → delta 재계산
    if abs(speed - 1.0) > 1e-4:
        T_new = max(10, int(T_actual * speed))
        base = result[:T_actual, :DIM_BASE]
        idx = np.linspace(0, T_actual - 1, T_new)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, T_actual - 1)
        w = (idx - lo)[:, np.newaxis]
        base_new = (base[lo] * (1 - w) + base[hi] * w).astype(np.float32)
        delta_new = np.zeros_like(base_new)
        delta_new[1:] = base_new[1:] - base_new[:-1]
        result = np.zeros_like(feat)
        copy_len = min(T_new, T)
        result[:copy_len, :DIM_BASE] = base_new[:copy_len]
        result[:copy_len, DIM_BASE:] = delta_new[:copy_len]
        T_actual = copy_len

    # 3. 공간 노이즈: base에 Gaussian 추가 → delta 재계산
    if jitter_std > 0.0 and T_actual > 0:
        noise = np.random.randn(T_actual, DIM_BASE).astype(np.float32) * jitter_std
        result[:T_actual, :DIM_BASE] += noise
        if T_actual > 1:
            result[1:T_actual, DIM_BASE:] = (
                result[1:T_actual, :DIM_BASE] - result[:T_actual - 1, :DIM_BASE]
            )

    return result


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
    def __init__(
        self,
        feat_path,
        lbl_path,
        label_encoder,
        indices,
        augment=False,
        speed_range=(0.8, 1.2),
        jitter_std=0.0,
        crop_prob=0.0,
        crop_min=0.5,
    ):
        self.X = np.load(feat_path, mmap_mode="r")
        y_str = np.load(lbl_path)
        self.y = label_encoder.transform(y_str)
        self.idx = np.asarray(indices, dtype=np.int64)
        self.augment = augment
        self.speed_range = speed_range
        self.jitter_std = jitter_std
        self.crop_prob = crop_prob
        self.crop_min = crop_min

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        ri = self.idx[i]
        x = self.X[ri].copy()

        if self.augment:
            speed = float(np.random.uniform(*self.speed_range))
            crop_ratio = (
                float(np.random.uniform(self.crop_min, 1.0))
                if self.crop_prob > 0 and np.random.random() < self.crop_prob
                else None
            )
            x = augment_features(x, speed=speed, jitter_std=self.jitter_std,
                                  crop_ratio=crop_ratio)

        return torch.from_numpy(x).float(), int(self.y[ri])


def build_loaders(
    label_encoder,
    batch_size,
    val_ratio,
    seed,
    num_workers,
    use_weighted_sampler,
    augment=False,
    speed_range=(0.8, 1.2),
    jitter_std=0.0,
    crop_prob=0.0,
    crop_min=0.5,
    num_signers=0,
    angles_per_signer=5,
    val_signers=0,
    signer_key="",
):
    """
    서명자 독립 분리 전략:
      AI Hub KSL 구조 = NPZ 파일 1개 = 서명자 1명의 데이터
        (1명 × 5각도 × 3000단어 ≒ 15,000 샘플/파일)
      따라서 파일 인덱스 = 서명자 ID.
      마지막 val_signers 개 파일을 검증 세트로 분리.

    val_signers=0 또는 num_signers=0 이면 기존 랜덤 stratified 분리.
    """
    feat_files = sorted([f for f in os.listdir(CACHE_DIR) if f.startswith("feats_") and f.endswith(".npy")])
    total_files = len(feat_files)
    train_sets, val_sets = [], []
    train_labels_all = []

    # ── 서명자 독립 분리: 파일 인덱스 기반 ──────────────────────
    use_signer_split = (num_signers > 0 and val_signers > 0
                        and val_signers < total_files)
    if use_signer_split:
        val_file_indices = set(range(total_files - val_signers, total_files))
        print(f"[서명자 독립 분리] 전체 {total_files}개 파일(서명자) 중 "
              f"마지막 {val_signers}개를 검증 세트로 분리")
        print(f"  val 파일: {sorted(val_file_indices)}  "
              f"train 파일: 0~{total_files - val_signers - 1}")
    else:
        val_file_indices = None
        print(f"[분리 방식] 랜덤 stratified 분리 (val_ratio={val_ratio})")

    for file_idx, ff in enumerate(feat_files):
        num = ff.split("_")[1].split(".")[0]
        fp = os.path.join(CACHE_DIR, ff)
        lp = os.path.join(CACHE_DIR, f"labels_{num}.npy")

        y_str = np.load(lp)
        y = label_encoder.transform(y_str)
        n = len(y)
        indices = np.arange(n)

        if use_signer_split:
            if file_idx in val_file_indices:
                # 이 파일 전체 → 검증 세트
                val_sets.append(KSLDataset(fp, lp, label_encoder, indices))
            else:
                # 이 파일 전체 → 학습 세트
                train_sets.append(KSLDataset(
                    fp, lp, label_encoder, indices,
                    augment=augment, speed_range=speed_range,
                    jitter_std=jitter_std, crop_prob=crop_prob, crop_min=crop_min,
                ))
                train_labels_all.extend(y.tolist())
        else:
            # ── 폴백: 랜덤 stratified 분리 ──
            try:
                train_idx, val_idx = train_test_split(
                    indices, test_size=val_ratio, random_state=seed,
                    shuffle=True, stratify=y,
                )
            except ValueError:
                rng = np.random.default_rng(seed)
                idx = rng.permutation(n)
                cut = int(n * (1.0 - val_ratio))
                train_idx, val_idx = idx[:cut], idx[cut:]

            train_sets.append(KSLDataset(
                fp, lp, label_encoder, train_idx,
                augment=augment, speed_range=speed_range,
                jitter_std=jitter_std, crop_prob=crop_prob, crop_min=crop_min,
            ))
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
            with torch.amp.autocast("cuda", enabled=use_amp):
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
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="KSL TCN Training v3\n\n"
                    "권장:\n"
                    "  python model/ksl_tcn_model.py --model tcn --epochs 120 "
                    "--batch-size 128 --aug --num-signers 16 --angles-per-signer 5 --val-signers 2",
    )
    # ── 모델 ──
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

    # ── 학습 ──
    p.add_argument("--val-ratio",        type=float, default=0.2,
                   help="서명자 분리 미사용 시 val 비율 (기본 0.2)")
    p.add_argument("--patience",         type=int,   default=18)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--label-smoothing",  type=float, default=0.08)
    p.add_argument("--class-weight",     action="store_true")
    p.add_argument("--weighted-sampler", action="store_true")
    p.add_argument("--force-preprocess", action="store_true")
    p.add_argument("--num-workers",      type=int,   default=None)
    p.add_argument("--no-amp",           action="store_true")
    p.add_argument("--grad-clip",        type=float, default=1.0)

    # ── 서명자 독립 검증 ──
    p.add_argument("--num-signers",       type=int, default=0,
                   help="데이터셋 내 전체 서명자 수. 0이면 서명자 분리 비활성화 (기본 0)")
    p.add_argument("--angles-per-signer", type=int, default=5,
                   help="서명자 1명당 각도 수 (기본 5). 클래스 내 위치 기반 추론에 사용")
    p.add_argument("--val-signers",       type=int, default=2,
                   help="검증에 사용할 서명자 수 (기본 2). num-signers > 0 일 때 활성화")
    p.add_argument("--signer-key",        type=str, default="",
                   help="NPZ에서 서명자 ID를 읽을 키 이름 (예: S). 비워두면 자동 탐색")

    # ── 데이터 증강 ──
    p.add_argument("--aug",            action="store_true",
                   help="학습 데이터 증강 활성화 (속도·노이즈·크롭). 강력 권장")
    p.add_argument("--aug-speed-min",  type=float, default=0.8,
                   help="속도 변환 최솟값 (기본 0.8). 추론 TTA 범위와 맞출 것")
    p.add_argument("--aug-speed-max",  type=float, default=1.2,
                   help="속도 변환 최댓값 (기본 1.2)")
    p.add_argument("--aug-jitter",     type=float, default=0.02,
                   help="공간 노이즈 std (기본 0.02). 웹캠 키포인트 흔들림 시뮬레이션")
    p.add_argument("--aug-crop-prob",  type=float, default=0.4,
                   help="크롭 적용 확률 (기본 0.4). 짧은 녹화 시뮬레이션")
    p.add_argument("--aug-crop-min",   type=float, default=0.5,
                   help="크롭 후 최소 유지 비율 (기본 0.5 → 최소 50%%만 유지)")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print("KSL Improved Training  v3")
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

    if args.aug:
        print(f"[증강] speed=[{args.aug_speed_min}, {args.aug_speed_max}], "
              f"jitter={args.aug_jitter}, "
              f"crop_prob={args.aug_crop_prob}, crop_min={args.aug_crop_min}")
    else:
        print("[증강] 비활성화  (--aug 추가 강력 권장)")

    if args.num_signers > 0:
        print(f"[서명자 분리] num_signers={args.num_signers}, "
              f"angles_per_signer={args.angles_per_signer}, "
              f"val_signers={args.val_signers}")
    else:
        print("[서명자 분리] 비활성화  (--num-signers 16 --val-signers 2 추가 권장)")

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
        crop_prob=args.aug_crop_prob,
        crop_min=args.aug_crop_min,
        num_signers=args.num_signers,
        angles_per_signer=args.angles_per_signer,
        val_signers=args.val_signers,
        signer_key=args.signer_key,
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
    scaler = None if args.no_amp else torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

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
