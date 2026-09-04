"""1D Depthwise Conv + Transformer 인코더 기반 고립단어 분류기.

입력: (B, T, FEATURE_DIM) 정규화된 위치+속도 피처 (kslx.normalize.featurize 출력).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class DepthwiseConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x + residual


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ConvTransformer(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, d_model: int = 160,
                 n_conv_blocks: int = 3, n_transformer_layers: int = 3,
                 n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.conv_blocks = nn.ModuleList(
            [DepthwiseConvBlock(d_model) for _ in range(n_conv_blocks)]
        )
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_transformer_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, feature_dim)
        h = self.input_proj(x)                # (B, T, d_model)
        h = h.transpose(1, 2)                 # (B, d_model, T)
        for block in self.conv_blocks:
            h = block(h)
        h = h.transpose(1, 2)                 # (B, T, d_model)
        h = self.pos_enc(h)
        h = self.transformer(h)               # (B, T, d_model)
        pooled = h.mean(dim=1)                # (B, d_model)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
