"""
src/preprocessing/augmentation.py
==================================
Data augmentation transformations for EOG time-series windows.

Applied ONLY during training to enhance generalization across subjects,
session variability, and sensor placement noise without causing subject leakage.

Transformations:
  1. Random Amplitude Scaling: simulates gain/skin impedance variations.
  2. Gaussian Noise Injection: simulates sensor/thermal noise.
  3. Micro Time-Shifting: simulates slight trigger offset jitter.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple, Optional


def augment_eog_windows_numpy(
    X: np.ndarray,
    scale_range: Tuple[float, float] = (0.90, 1.10),
    noise_std: float = 0.02,
    shift_max: int = 2,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Augment a numpy array batch of EOG windows X of shape (N, 2, window_len).

    Parameters
    ----------
    X           : (N, 2, window_len) float array
    scale_range : (min_scale, max_scale) multiplier per channel
    noise_std   : standard deviation of additive Gaussian noise relative to signal std
    shift_max   : maximum sample shift (roll) per window

    Returns
    -------
    Augmented array of same shape (N, 2, window_len).
    """
    if len(X) == 0:
        return X

    rng = np.random.RandomState(seed)
    X_aug = X.copy()
    N, C, T = X_aug.shape

    scales = rng.uniform(scale_range[0], scale_range[1], size=(N, C, 1)).astype(np.float32)
    X_aug *= scales

    if noise_std > 0:
        noise = rng.normal(0, noise_std, size=(N, C, T)).astype(np.float32)
        X_aug += noise

    if shift_max > 0:
        shifts = rng.randint(-shift_max, shift_max + 1, size=N)
        for i in range(N):
            if shifts[i] != 0:
                X_aug[i] = np.roll(X_aug[i], shift=shifts[i], axis=-1)

    return X_aug


def augment_eog_batch_tensor(
    X_tensor: torch.Tensor,
    scale_range: Tuple[float, float] = (0.90, 1.10),
    noise_std: float = 0.02,
    channel_dropout_prob: float = 0.1,
) -> torch.Tensor:
    """
    Augment a PyTorch batch tensor X of shape (batch_size, 2, window_len) on GPU/CPU.
    Fast execution during PyTorch training loops.
    
    Includes:
      - Random channel amplitude scaling
      - Additive Gaussian noise
      - Channel dropout (randomly zero one channel)
    """
    if not X_tensor.requires_grad:
        N, C, T = X_tensor.shape
        device = X_tensor.device

        scales = (scale_range[1] - scale_range[0]) * torch.rand((N, C, 1), device=device) + scale_range[0]
        X_aug = X_tensor * scales

        if noise_std > 0:
            noise = torch.randn((N, C, T), device=device) * noise_std
            X_aug = X_aug + noise

        if channel_dropout_prob > 0 and C > 0:
            drop_mask = torch.rand(N, device=device) < channel_dropout_prob
            if drop_mask.any():
                drop_channel = torch.randint(0, C, (N,), device=device)
                ch_mask = torch.ones(N, C, 1, device=device)
                ch_mask[drop_mask, drop_channel[drop_mask]] = 0.0
                X_aug = X_aug * ch_mask

        return X_aug
    return X_tensor


def mixup_batch(
    X: torch.Tensor,
    y_cls: torch.Tensor,
    y_ang: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply Mixup augmentation: blend random pairs of windows and their labels.
    
    This forces the network to learn smoother decision boundaries, proven to
    reduce RMSE by 5-10% on time-series regression tasks.
    
    Parameters
    ----------
    X      : (batch, 2, window_len) input tensor
    y_cls  : (batch,) classification labels
    y_ang  : (batch, 2) regression targets
    alpha  : Beta distribution parameter (higher = more mixing)
    
    Returns
    -------
    X_mixed, y_cls_a, y_cls_b, y_ang_mixed, lam
    """
    if alpha <= 0:
        return X, y_cls, y_cls, y_ang, 1.0
    
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam = max(lam, 1 - lam)
    
    batch_size = X.size(0)
    index = torch.randperm(batch_size, device=X.device)
    
    X_mixed = lam * X + (1 - lam) * X[index]
    y_ang_mixed = lam * y_ang + (1 - lam) * y_ang[index]
    
    return X_mixed, y_cls, y_cls[index], y_ang_mixed, lam

