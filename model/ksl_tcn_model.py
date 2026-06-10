"""
KSL 수어 인식 - TCN 학습 스크립트
===================================

데이터셋 구조:
  Dataset_NPZ/01_dataset.npz ~ 16_dataset.npz
  · 파일 1개 = 서명자 1명의 데이터
  · 파일 1개 내용: 3000개 단어 × 5각도 = 15,000 샘플
  · 파일 인덱스 = 서명자 ID (0~15)

서명자 독립 분리 (기본):
  · 학습: 파일 00~13 (서명자 14명)
  · 검증: 파일 14~15 (서명자 2명) ← 학습에 전혀 등장하지 않은 사람

실행 (권장):
  python model/ksl_tcn_model.py --model tcn --epochs 120 --batch-size 128 --aug

옵션:
  --val-signers 2    검증에 사용할 파일 수 (마지막 N개, 기본 2)
  --aug              데이터 증강 활성화 (속도·노이즈·크롭)
  --no-signer-split  서명자 분리 비활성화 → 랜덤 stratified 분리로 폴백
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


# ──────────────────────────────────────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_ROOT   = os.path.join(PROJECT_DIR, "Dataset_NPZ", "Dataset_NPZ")
CACHE_DIR   = os.path.join(SCRIPT_DIR, "feat_cache_improved")
MODEL_DIR   = SCRIPT_DIR


# ──────────────────────────────────────────────────────────────────────────────
# 재현성
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 증강
# ──────────────────────────────────────────────────────────────────────────────
def _find_actual_length(feat):
    """제로패딩된 특징 (T, D)에서 실제 유효 프레임 수 반환."""
    norms = np.linalg.norm(feat, axis=1)
    nz    = np.where(norms > 1e-6)[0]
    return int(nz[-1]) + 1 if len(nz) > 0 else feat.shape[0]


def augment_features(feat, speed=1.0, jitter_std=0.0, crop_ratio=None):
    """
    특징 레벨 증강. 입출력 shape 불변 (T, D).

    speed      : 속도 배율 (0.8~1.2). 추론 TTA 범위와 동일하게 학습.
    jitter_std : base feature에 더할 Gaussian 노이즈 std.
                 웹캠 키포인트 흔들림·조명 차이 시뮬레이션.
    crop_ratio : 실제 길이의 이 비율만 유지하고 이후 제로패딩.
                 추론 시 30~90 프레임 짧은 녹화를 학습 중에도 경험하게 함.
    """
    DIM_BASE = feat.shape[1] // 2      # base / delta 절반씩 구성
    T        = feat.shape[0]
    T_actual = _find_actual_length(feat)
    if T_actual == 0:
        return feat

    result = feat.copy()

    # 1. 크롭
    if crop_ratio is not None and crop_ratio < 1.0:
        T_crop           = max(15, int(T_actual * crop_ratio))
        result[T_crop:]  = 0.0
        T_actual         = T_crop

    # 2. 속도 변환: base 리샘플 → delta 재계산
    if abs(speed - 1.0) > 1e-4:
        T_new    = max(10, int(T_actual * speed))
        base     = result[:T_actual, :DIM_BASE]
        idx      = np.linspace(0, T_actual - 1, T_new)
        lo       = np.floor(idx).astype(int)
        hi       = np.minimum(lo + 1, T_actual - 1)
        w        = (idx - lo)[:, np.newaxis]
        base_new = (base[lo] * (1 - w) + base[hi] * w).astype(np.float32)
        d_new    = np.zeros_like(base_new)
        d_new[1:] = base_new[1:] - base_new[:-1]
        result   = np.zeros_like(feat)
        copy_len = min(T_new, T)
        result[:copy_len, :DIM_BASE] = base_new[:copy_len]
        result[:copy_len, DIM_BASE:] = d_new[:copy_len]
        T_actual = copy_len

    # 3. 공간 노이즈: base에 Gaussian → delta 재계산
    if jitter_std > 0.0 and T_actual > 0:
        noise = np.random.randn(T_actual, DIM_BASE).astype(np.float32) * jitter_std
        result[:T_actual, :DIM_BASE] += noise
        if T_actual > 1:
            result[1:T_actual, DIM_BASE:] = (
                result[1:T_actual, :DIM_BASE] - result[:T_actual - 1, :DIM_BASE]
            )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 특징 추출
# ──────────────────────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGER_TIPS = [4, 8, 12, 16, 20]


def _safe_norm(x, axis=-1, keepdims=False, eps=1e-6):
    return np.linalg.norm(x, axis=axis, keepdims=keepdims) + eps


def _normalize_hand_batch(hand):
    """hand: (N, T, 21, 3) → 손목 원점 + 중지MCP 스케일 정규화."""
    center  = hand[:, :, 0:1, :]
    hand_c  = hand - center
    scale   = np.linalg.norm(hand_c[:, :, 9, :], axis=-1, keepdims=True)
    scale   = scale[:, :, :, None]
    return hand_c / (scale + 1e-6), center.squeeze(2), scale.squeeze(2)


def _extract_angle_features(X_batch):
    """(N, T, 67, 3) → (N, T, 40) 관절 각도·거리 특징."""
    N, T     = X_batch.shape[:2]
    features = np.zeros((N, T, 40), dtype=np.float32)
    for hand_i, hs in enumerate([0, 21]):
        hand      = X_batch[:, :, hs:hs + 21, :]
        hand_norm, _, _ = _normalize_hand_batch(hand)
        base      = hand_i * 20
        for ci, (parent, child) in enumerate(HAND_CONNECTIONS):
            if parent == 0:
                diff = hand_norm[:, :, child] - hand_norm[:, :, parent]
                features[:, :, base + ci] = np.linalg.norm(diff, axis=-1)
            else:
                v1  = hand_norm[:, :, parent]     - hand_norm[:, :, parent - 1]
                v2  = hand_norm[:, :, child]      - hand_norm[:, :, parent]
                n1  = np.linalg.norm(v1, axis=-1)
                n2  = np.linalg.norm(v2, axis=-1)
                cos = np.clip(np.sum(v1 * v2, axis=-1) / (n1 * n2 + 1e-6), -1.0, 1.0)
                features[:, :, base + ci] = np.arccos(cos)
    return features


def extract_improved_features_batch(X_batch):
    """
    X_batch : (N, T, 67, 3)  왼손21 + 오른손21 + 포즈25
    반환    : (N, T, 206)

    특징 그룹:
      관절각도(40) + 손끝좌표(30) + 손목관계(13) + 손크기(6)
      + 몸통참조(14) + delta(103) = 206
    """
    X    = X_batch.astype(np.float32)
    N, T = X.shape[:2]

    left  = X[:, :, 0:21, :]
    right = X[:, :, 21:42, :]
    left_norm,  _, left_scale  = _normalize_hand_batch(left)
    right_norm, _, right_scale = _normalize_hand_batch(right)

    angle_feats = _extract_angle_features(X)   # (N,T,40)

    left_tips  = left_norm[:, :, FINGER_TIPS, :].reshape(N, T, -1)
    right_tips = right_norm[:, :, FINGER_TIPS, :].reshape(N, T, -1)
    tip_feats  = np.concatenate([left_tips, right_tips], axis=-1).astype(np.float32)  # 30

    global_center = np.mean(X, axis=2, keepdims=True)
    global_scale  = _safe_norm(X - global_center, axis=-1, keepdims=True).mean(axis=2, keepdims=True)
    Xg            = (X - global_center) / (global_scale + 1e-6)

    lw_g = Xg[:, :, 0, :]
    rw_g = Xg[:, :, 21, :]
    wrist_rel  = rw_g - lw_g
    wrist_dist = _safe_norm(wrist_rel, axis=-1, keepdims=True)
    wrist_mid  = 0.5 * (lw_g + rw_g)
    wrist_feats = np.concatenate(
        [lw_g, rw_g, wrist_rel, wrist_dist, wrist_mid], axis=-1
    ).astype(np.float32)  # 13

    lt_d = _safe_norm(left_norm[:, :, FINGER_TIPS, :] - left_norm[:, :, 0:1, :], axis=-1)
    rt_d = _safe_norm(right_norm[:, :, FINGER_TIPS, :] - right_norm[:, :, 0:1, :], axis=-1)
    hand_spread = np.concatenate([
        lt_d.mean(axis=-1, keepdims=True), lt_d.std(axis=-1, keepdims=True),
        rt_d.mean(axis=-1, keepdims=True), rt_d.std(axis=-1, keepdims=True),
        left_scale.reshape(N, T, -1).mean(axis=-1, keepdims=True),
        right_scale.reshape(N, T, -1).mean(axis=-1, keepdims=True),
    ], axis=-1).astype(np.float32)  # 6

    extra = []
    if X.shape[2] > 42:
        ex  = Xg[:, :, 42:, :]
        rc  = ex.mean(axis=2)
        rs  = ex.std(axis=2)
        lwr = lw_g - rc
        rwr = rw_g - rc
        extra = [
            rc.astype(np.float32), rs.astype(np.float32),
            lwr.astype(np.float32), rwr.astype(np.float32),
            _safe_norm(lwr, axis=-1, keepdims=True).astype(np.float32),
            _safe_norm(rwr, axis=-1, keepdims=True).astype(np.float32),
        ]  # 14

    base  = np.concatenate([angle_feats, tip_feats, wrist_feats, hand_spread] + extra, axis=-1)
    delta = np.zeros_like(base)
    delta[:, 1:] = base[:, 1:] - base[:, :-1]
    feats = np.concatenate([base, delta], axis=-1).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 전처리 (NPZ → 특징 캐시)
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(force: bool = False):
    """
    NPZ 파일들을 읽어 특징 캐시(feat_cache_improved/)를 생성한다.
    캐시가 이미 있으면 건너뛴다 (force=True 이면 재생성).

    파일 인덱스가 곧 서명자 ID이므로, 캐시 파일명 feats_NN.npy 의 NN이
    서명자 ID(0-indexed)와 1:1 대응된다.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    npz_files = sorted([f for f in os.listdir(DATA_ROOT) if f.endswith(".npz")])
    if not npz_files:
        raise FileNotFoundError(f"NPZ 파일 없음: {DATA_ROOT}")

    print(f"[전처리] NPZ {len(npz_files)}개 → {CACHE_DIR}")
    total_t = time.time()
    meta    = {"files": [], "feature_dim": None, "num_signers": len(npz_files)}

    for signer_id, fname in enumerate(npz_files):
        feat_path = os.path.join(CACHE_DIR, f"feats_{signer_id:02d}.npy")
        lbl_path  = os.path.join(CACHE_DIR, f"labels_{signer_id:02d}.npy")

        if not force and os.path.exists(feat_path) and os.path.exists(lbl_path):
            arr = np.load(feat_path, mmap_mode="r")
            meta["files"].append({"signer_id": signer_id, "name": fname, "shape": tuple(arr.shape)})
            meta["feature_dim"] = int(arr.shape[-1])
            print(f"  [서명자{signer_id:02d}] {fname} — 캐시 존재 {arr.shape}")
            continue

        t0   = time.time()
        data = np.load(os.path.join(DATA_ROOT, fname), allow_pickle=True)
        X    = data["X"].astype(np.float32)          # (N, 148, 67, 3)
        y    = np.array([str(v) for v in data["V"]]) # (N,)

        feats = extract_improved_features_batch(X)   # (N, 148, 206)
        np.save(feat_path, feats)
        np.save(lbl_path,  y)

        meta["files"].append({"signer_id": signer_id, "name": fname, "shape": tuple(feats.shape)})
        meta["feature_dim"] = int(feats.shape[-1])
        print(f"  [서명자{signer_id:02d}] {fname}  {feats.shape}  ({time.time()-t0:.0f}s)")

    with open(os.path.join(CACHE_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[전처리 완료] 총 {time.time()-total_t:.0f}s\n")
    return meta["feature_dim"]


# ──────────────────────────────────────────────────────────────────────────────
# 레이블 인코더
# ──────────────────────────────────────────────────────────────────────────────
def build_label_encoder():
    all_labels = []
    for f in sorted(os.listdir(CACHE_DIR)):
        if f.startswith("labels_") and f.endswith(".npy"):
            all_labels.extend(np.load(os.path.join(CACHE_DIR, f)).tolist())
    le = LabelEncoder()
    le.fit(sorted(set(all_labels)))
    print(f"[레이블] num_classes={len(le.classes_)}")
    return le


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class KSLDataset(Dataset):
    def __init__(
        self,
        feat_path,
        lbl_path,
        label_encoder,
        augment     = False,
        speed_range = (0.8, 1.2),
        jitter_std  = 0.0,
        crop_prob   = 0.0,
        crop_min    = 0.5,
    ):
        self.X          = np.load(feat_path, mmap_mode="r")
        y_str           = np.load(lbl_path)
        self.y          = label_encoder.transform(y_str)
        self.augment    = augment
        self.speed_range = speed_range
        self.jitter_std = jitter_std
        self.crop_prob  = crop_prob
        self.crop_min   = crop_min

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.augment:
            speed      = float(np.random.uniform(*self.speed_range))
            crop_ratio = (
                float(np.random.uniform(self.crop_min, 1.0))
                if self.crop_prob > 0 and np.random.random() < self.crop_prob
                else None
            )
            x = augment_features(x, speed=speed, jitter_std=self.jitter_std,
                                  crop_ratio=crop_ratio)
        return torch.from_numpy(x).float(), int(self.y[i])


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader 구성
# ──────────────────────────────────────────────────────────────────────────────
def build_loaders(
    label_encoder,
    batch_size,
    val_ratio,
    seed,
    num_workers,
    use_weighted_sampler,
    augment      = False,
    speed_range  = (0.8, 1.2),
    jitter_std   = 0.0,
    crop_prob    = 0.0,
    crop_min     = 0.5,
    val_signers  = 2,
    signer_split = True,
):
    """
    signer_split=True (기본):
      파일 인덱스 = 서명자 ID.
      마지막 val_signers 개 파일 → 검증, 나머지 → 학습.
      예) 16개 파일, val_signers=2 → train: 파일00~13, val: 파일14~15

    signer_split=False:
      파일별 랜덤 stratified 분리 (val_ratio 사용).
      서명자 독립 보장 없음.
    """
    feat_files  = sorted([f for f in os.listdir(CACHE_DIR)
                           if f.startswith("feats_") and f.endswith(".npy")])
    total_files = len(feat_files)

    if signer_split:
        assert val_signers < total_files, \
            f"val_signers({val_signers}) >= 전체 파일 수({total_files})"
        val_file_set = set(range(total_files - val_signers, total_files))
        print(f"[서명자 독립 분리]")
        print(f"  총 파일(서명자): {total_files}명")
        print(f"  학습 서명자: 파일 00 ~ {total_files - val_signers - 1:02d}  ({total_files - val_signers}명)")
        print(f"  검증 서명자: 파일 {total_files - val_signers:02d} ~ {total_files - 1:02d}  ({val_signers}명)")
    else:
        val_file_set = None
        print(f"[랜덤 stratified 분리]  val_ratio={val_ratio}")

    train_sets, val_sets = [], []
    train_labels_all     = []

    for file_idx, ff in enumerate(feat_files):
        num      = ff.split("_")[1].split(".")[0]
        fp       = os.path.join(CACHE_DIR, ff)
        lp       = os.path.join(CACHE_DIR, f"labels_{num}.npy")
        y_str    = np.load(lp)
        y        = label_encoder.transform(y_str)

        if signer_split:
            if file_idx in val_file_set:
                val_sets.append(KSLDataset(fp, lp, label_encoder))
            else:
                train_sets.append(KSLDataset(
                    fp, lp, label_encoder,
                    augment=augment, speed_range=speed_range,
                    jitter_std=jitter_std, crop_prob=crop_prob, crop_min=crop_min,
                ))
                train_labels_all.extend(y.tolist())
        else:
            indices = np.arange(len(y))
            try:
                train_idx, val_idx = train_test_split(
                    indices, test_size=val_ratio, random_state=seed,
                    shuffle=True, stratify=y,
                )
            except ValueError:
                rng      = np.random.default_rng(seed)
                idx      = rng.permutation(len(y))
                cut      = int(len(y) * (1.0 - val_ratio))
                train_idx, val_idx = idx[:cut], idx[cut:]

            train_sets.append(KSLDataset(
                fp, lp, label_encoder,
                augment=augment, speed_range=speed_range,
                jitter_std=jitter_std, crop_prob=crop_prob, crop_min=crop_min,
            ))
            val_sets.append(KSLDataset(fp, lp, label_encoder))
            # 랜덤 분리 시에는 인덱스를 직접 슬라이싱할 수 없으므로
            # Dataset 자체에 인덱스 정보를 전달해야 한다.
            # → signer_split=False 시에는 아래 _KSLDatasetSubset 사용
            train_labels_all.extend(y[train_idx].tolist())

    # signer_split=False 일 때는 전체 파일을 인덱스 기반으로 분리
    if not signer_split:
        train_sets, val_sets = [], []
        train_labels_all     = []
        for file_idx, ff in enumerate(feat_files):
            num   = ff.split("_")[1].split(".")[0]
            fp    = os.path.join(CACHE_DIR, ff)
            lp    = os.path.join(CACHE_DIR, f"labels_{num}.npy")
            y_str = np.load(lp)
            y     = label_encoder.transform(y_str)
            n     = len(y)
            indices = np.arange(n)
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

            train_sets.append(_KSLSubset(fp, lp, label_encoder, train_idx,
                                          augment=augment, speed_range=speed_range,
                                          jitter_std=jitter_std, crop_prob=crop_prob,
                                          crop_min=crop_min))
            val_sets.append(_KSLSubset(fp, lp, label_encoder, val_idx))
            train_labels_all.extend(y[train_idx].tolist())

    train_ds = ConcatDataset(train_sets)
    val_ds   = ConcatDataset(val_sets)
    pin      = torch.cuda.is_available()

    if use_weighted_sampler:
        tlabels      = np.asarray(train_labels_all)
        counts       = np.bincount(tlabels, minlength=len(label_encoder.classes_))
        w_class      = 1.0 / np.maximum(counts, 1)
        w_sample     = w_class[tlabels]
        sampler      = WeightedRandomSampler(
            torch.as_tensor(w_sample, dtype=torch.double),
            num_samples=len(w_sample), replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, np.asarray(train_labels_all)


class _KSLSubset(Dataset):
    """랜덤 분리 폴백 시 사용하는 인덱스 기반 서브셋."""
    def __init__(self, feat_path, lbl_path, label_encoder, indices,
                 augment=False, speed_range=(0.8,1.2),
                 jitter_std=0.0, crop_prob=0.0, crop_min=0.5):
        self.X          = np.load(feat_path, mmap_mode="r")
        y_str           = np.load(lbl_path)
        self.y          = label_encoder.transform(y_str)
        self.idx        = np.asarray(indices, dtype=np.int64)
        self.augment    = augment
        self.speed_range = speed_range
        self.jitter_std = jitter_std
        self.crop_prob  = crop_prob
        self.crop_min   = crop_min

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        ri = self.idx[i]
        x  = self.X[ri].copy()
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


# ──────────────────────────────────────────────────────────────────────────────
# Pooling
# ──────────────────────────────────────────────────────────────────────────────
class AttnPool1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1)
        )
    def forward(self, x):
        return (x * torch.softmax(self.attn(x), dim=1)).sum(dim=1)


class AttnAvgMaxPool1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_pool = AttnPool1D(dim)
    def forward(self, x):
        return torch.cat([self.attn_pool(x), x.mean(dim=1), x.max(dim=1).values], dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# 모델 정의
# ──────────────────────────────────────────────────────────────────────────────
class Chomp1d(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s
    def forward(self, x):
        return x[:, :, :-self.s].contiguous() if self.s else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad         = (kernel_size - 1) * dilation
        self.conv1  = nn.Conv1d(in_ch,  out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.bn1    = nn.BatchNorm1d(out_ch)
        self.conv2  = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.bn2    = nn.BatchNorm1d(out_ch)
        self.drop   = nn.Dropout(dropout)
        self.down   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.drop(F.gelu(self.bn1(self.chomp1(self.conv1(x)))))
        y = self.drop(F.gelu(self.bn2(self.chomp2(self.conv2(y)))))
        return F.gelu(y + (x if self.down is None else self.down(x)))


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


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        out_dim   = hidden_dim * 2
        self.norm = nn.LayerNorm(out_dim)
        self.pool = AttnAvgMaxPool1D(out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim * 3, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, num_classes),
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.pool(self.norm(out)))


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.gru  = nn.GRU(input_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        out_dim   = hidden_dim * 2
        self.norm = nn.LayerNorm(out_dim)
        self.pool = AttnAvgMaxPool1D(out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim * 3, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, num_classes),
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(self.pool(self.norm(out)))


def build_model(args, input_dim, num_classes):
    if args.model == "tcn":
        return TCNClassifier(input_dim, args.tcn_channels, num_classes,
                             args.dropout, args.kernel_size)
    if args.model == "bilstm":
        return BiLSTMClassifier(input_dim, args.hidden_dim, args.num_layers,
                                num_classes, args.dropout)
    if args.model == "gru":
        return GRUClassifier(input_dim, args.hidden_dim, args.num_layers,
                             num_classes, args.dropout)
    raise ValueError(f"Unknown model: {args.model}")


# ──────────────────────────────────────────────────────────────────────────────
# 학습 / 평가
# ──────────────────────────────────────────────────────────────────────────────
def topk_correct(logits, y, ks=(1, 3, 5)):
    max_k  = min(max(ks), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1)
    pred   = pred.t()
    correct = pred.eq(y.view(1, -1).expand_as(pred))
    return {k: correct[:min(k, max_k)].reshape(-1).float().sum().item() for k in ks}


def run_epoch(model, loader, criterion, optimizer, device, scaler, is_train, grad_clip):
    model.train(is_train)
    total_loss = total_n = c1 = c3 = c5 = 0
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

            bs = len(y_b)
            total_loss += loss.item() * bs
            total_n    += bs
            tk          = topk_correct(logits.detach(), y_b)
            c1 += tk[1]; c3 += tk[3]; c5 += tk[5]

            all_pred.extend(logits.argmax(1).detach().cpu().numpy().tolist())
            all_true.extend(y_b.detach().cpu().numpy().tolist())

            if is_train and (step + 1) % 100 == 0:
                print(f"    step {step+1}/{len(loader)}  "
                      f"loss={total_loss/total_n:.4f}  "
                      f"top1={c1/total_n:.4f}  top3={c3/total_n:.4f}", flush=True)

    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return {
        "loss": total_loss / max(total_n, 1),
        "top1": c1 / max(total_n, 1),
        "top3": c3 / max(total_n, 1),
        "top5": c5 / max(total_n, 1),
        "macro_f1": macro_f1,
        "y_true": all_true,
        "y_pred": all_pred,
    }


def make_class_weights(train_labels, num_classes, device):
    counts  = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    return torch.as_tensor(weights / weights.mean(), dtype=torch.float32, device=device)


def save_history_row(path, row, write_header=False):
    fields = ["epoch", "lr",
              "train_loss", "train_top1", "train_top3", "train_top5", "train_macro_f1",
              "val_loss",   "val_top1",   "val_top3",   "val_top5",   "val_macro_f1",
              "elapsed_sec"]
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


def save_final_reports(out_prefix, label_encoder, y_true, y_pred):
    labels       = list(range(len(label_encoder.classes_)))
    target_names = [str(x) for x in label_encoder.classes_]

    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=target_names, zero_division=0, digits=4)
    with open(out_prefix + "_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.save(out_prefix + "_confusion_matrix.npy", cm)

    confused = sorted(
        [(cm[i, j], target_names[i], target_names[j])
         for i in range(cm.shape[0]) for j in range(cm.shape[1]) if i != j and cm[i, j] > 0],
        reverse=True,
    )
    with open(out_prefix + "_top_confusions.txt", "w", encoding="utf-8") as f:
        for cnt, tn, pn in confused[:100]:
            f.write(f"{cnt}\tTRUE={tn}\tPRED={pn}\n")

    print("\n[Classification Report]")
    print(report)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "KSL TCN 학습\n\n"
            "권장:\n"
            "  python model/ksl_tcn_model.py --model tcn --epochs 120 "
            "--batch-size 128 --aug\n\n"
            "서명자 독립 분리 기본 활성화 (파일 14·15번 → 검증)\n"
            "--no-signer-split 으로 비활성화 가능"
        ),
    )
    # ── 모델 ──
    p.add_argument("--model",        choices=["tcn","bilstm","gru"], default="tcn")
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
    p.add_argument("--patience",         type=int,   default=18)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--label-smoothing",  type=float, default=0.08)
    p.add_argument("--class-weight",     action="store_true")
    p.add_argument("--weighted-sampler", action="store_true")
    p.add_argument("--force-preprocess", action="store_true")
    p.add_argument("--num-workers",      type=int,   default=None)
    p.add_argument("--no-amp",           action="store_true")
    p.add_argument("--grad-clip",        type=float, default=1.0)

    # ── 서명자 독립 분리 ──
    p.add_argument("--val-signers",     type=int, default=2,
                   help="검증 서명자 수 (마지막 N개 파일, 기본 2)")
    p.add_argument("--no-signer-split", action="store_true",
                   help="서명자 분리 비활성화 → 랜덤 stratified 분리")
    p.add_argument("--val-ratio",       type=float, default=0.2,
                   help="--no-signer-split 시 val 비율 (기본 0.2)")

    # ── 데이터 증강 ──
    p.add_argument("--aug",           action="store_true",
                   help="학습 데이터 증강 활성화 (강력 권장)")
    p.add_argument("--aug-speed-min", type=float, default=0.8)
    p.add_argument("--aug-speed-max", type=float, default=1.2)
    p.add_argument("--aug-jitter",    type=float, default=0.02,
                   help="공간 노이즈 std (기본 0.02)")
    p.add_argument("--aug-crop-prob", type=float, default=0.4,
                   help="크롭 적용 확률 (기본 0.4)")
    p.add_argument("--aug-crop-min",  type=float, default=0.5,
                   help="크롭 후 최소 유지 비율 (기본 0.5)")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 70)
    print("KSL 수어 인식 학습")
    print("=" * 70)
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print()

    assert os.path.isdir(DATA_ROOT), f"데이터 폴더 없음: {DATA_ROOT}"

    input_dim     = preprocess(force=args.force_preprocess)
    label_encoder = build_label_encoder()
    num_classes   = len(label_encoder.classes_)

    num_workers = (0 if sys.platform == "win32" else 4) \
                  if args.num_workers is None else args.num_workers

    signer_split = not args.no_signer_split
    if args.aug:
        print(f"[증강] speed=[{args.aug_speed_min},{args.aug_speed_max}]  "
              f"jitter={args.aug_jitter}  "
              f"crop_prob={args.aug_crop_prob}  crop_min={args.aug_crop_min}")
    else:
        print("[증강] 비활성화  (--aug 추가 강력 권장)")

    train_loader, val_loader, train_labels = build_loaders(
        label_encoder    = label_encoder,
        batch_size       = args.batch_size,
        val_ratio        = args.val_ratio,
        seed             = args.seed,
        num_workers      = num_workers,
        use_weighted_sampler = args.weighted_sampler,
        augment          = args.aug,
        speed_range      = (args.aug_speed_min, args.aug_speed_max),
        jitter_std       = args.aug_jitter,
        crop_prob        = args.aug_crop_prob,
        crop_min         = args.aug_crop_min,
        val_signers      = args.val_signers,
        signer_split     = signer_split,
    )

    print(f"\n[데이터] train={len(train_loader.dataset):,}  val={len(val_loader.dataset):,}")
    print(f"[특징]   input_dim={input_dim}  num_classes={num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[디바이스] {device}", end="")
    if device.type == "cuda":
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print()

    model    = build_model(args, input_dim, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[모델] {args.model}  파라미터: {n_params:,}\n")

    criterion = (
        nn.CrossEntropyLoss(
            weight=make_class_weights(train_labels, num_classes, device),
            label_smoothing=args.label_smoothing,
        ) if args.class_weight
        else nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    )

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=6, factor=0.5)
    scaler    = (None if args.no_amp
                 else torch.amp.GradScaler("cuda", enabled=(device.type == "cuda")))

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

    print("[학습 시작]")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, optimizer,
                       device, scaler, True,  args.grad_clip)
        vl = run_epoch(model, val_loader,   criterion, optimizer,
                       device, scaler, False, args.grad_clip)
        elapsed = time.time() - t0

        score = vl["macro_f1"]
        scheduler.step(score)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs}  lr={lr:.2e}  "
            f"train_top1={tr['top1']:.4f}  train_f1={tr['macro_f1']:.4f}  "
            f"val_top1={vl['top1']:.4f}  val_top3={vl['top3']:.4f}  "
            f"val_f1={vl['macro_f1']:.4f}  loss={vl['loss']:.4f}  {elapsed:.0f}s",
            flush=True,
        )

        save_history_row(history_path, {
            "epoch": epoch, "lr": lr,
            "train_loss": tr["loss"], "train_top1": tr["top1"],
            "train_top3": tr["top3"], "train_top5": tr["top5"],
            "train_macro_f1": tr["macro_f1"],
            "val_loss": vl["loss"], "val_top1": vl["top1"],
            "val_top3": vl["top3"], "val_top5": vl["top5"],
            "val_macro_f1": vl["macro_f1"], "elapsed_sec": elapsed,
        }, write_header=(epoch == 1))

        if score > best_score:
            best_score     = score
            best_epoch     = epoch
            no_improve     = 0
            best_val_true  = vl["y_true"]
            best_val_pred  = vl["y_pred"]

            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "model":       args.model,
                    "input_dim":   input_dim,
                    "hidden_dim":  args.hidden_dim,
                    "num_layers":  args.num_layers,
                    "tcn_channels": args.tcn_channels,
                    "kernel_size": args.kernel_size,
                    "num_classes": num_classes,
                    "dropout":     args.dropout,
                },
                "args":               vars(args),
                "best_epoch":         best_epoch,
                "best_val_macro_f1":  best_score,
                "val_top1_at_best":   vl["top1"],
                "val_top3_at_best":   vl["top3"],
            }, model_path)

            joblib.dump({
                "label_encoder": label_encoder,
                "label_names":   label_encoder.classes_.tolist(),
                "feature_dim":   input_dim,
            }, meta_path)

            print(f"  ✓ 모델 저장  epoch={best_epoch}  "
                  f"val_f1={best_score:.4f}  val_top1={vl['top1']:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n[Early stopping] {args.patience} 에포크 동안 개선 없음")
                break

    print("\n" + "=" * 70)
    print(f"학습 완료  best_epoch={best_epoch}  best_val_macro_f1={best_score:.4f}")
    print(f"모델: {model_path}")
    print(f"메타: {meta_path}")
    print("=" * 70)

    if best_val_true is not None:
        save_final_reports(out_prefix, label_encoder, best_val_true, best_val_pred)


if __name__ == "__main__":
    main()
