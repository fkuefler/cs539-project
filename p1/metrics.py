"""P1: Evaluation metrics — SSIM, PSNR, and Dice score."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Compute PSNR (dB) between predicted and target arrays in [0, data_range]."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mse = float(np.mean((pred - target) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(peak_signal_noise_ratio(target, pred, data_range=data_range))


def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Compute 2D SSIM between predicted and target slices in [0, data_range]."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    return float(structural_similarity(target, pred, data_range=data_range))


def compute_dice(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute binary Dice similarity coefficient between two masks."""
    pred_bin = (np.asarray(pred_mask) > 0.5).astype(np.float32)
    true_bin = (np.asarray(true_mask) > 0.5).astype(np.float32)
    pred_sum = float(pred_bin.sum())
    true_sum = float(true_bin.sum())
    if pred_sum == 0.0 and true_sum == 0.0:
        return 1.0
    intersection = float(np.sum(pred_bin * true_bin))
    return float((2.0 * intersection + eps) / (pred_sum + true_sum + eps))


def evaluate_slice_stack(
    pred_slices: np.ndarray,
    target_slices: np.ndarray,
    data_range: float = 1.0,
) -> dict[str, float]:
    """Compute mean SSIM and PSNR over a stack of (N, H, W) 2D slices."""
    if pred_slices.shape != target_slices.shape:
        raise ValueError(f"Shape mismatch: {pred_slices.shape} vs {target_slices.shape}")
    if len(pred_slices) == 0:
        return {"ssim": float("nan"), "psnr": float("nan"), "num_slices": 0}

    ssim_vals = [
        compute_ssim(pred_slices[i], target_slices[i], data_range=data_range)
        for i in range(len(pred_slices))
    ]
    psnr_vals = [
        compute_psnr(pred_slices[i], target_slices[i], data_range=data_range)
        for i in range(len(pred_slices))
    ]
    finite_psnr = [v for v in psnr_vals if np.isfinite(v)]
    return {
        "ssim": float(np.mean(ssim_vals)),
        "psnr": float(np.mean(finite_psnr)) if finite_psnr else float("inf"),
        "num_slices": int(len(pred_slices)),
    }
