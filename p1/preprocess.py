"""P1: Prepare Scans — NIfTI loading, alignment checks, normalization, and 2D slicing."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

# In BraTS 2023 GLI, label 3 is Enhancing Tumor (ET). (Legacy BraTS 2021 used 4.)
ENHANCING_TUMOR_LABEL = 3


@dataclass
class CaseData:
    case_id: str
    t1n: np.ndarray
    t1c: np.ndarray | None
    seg: np.ndarray | None
    affine: np.ndarray
    zooms: tuple[float, ...]


def find_modality_file(case_dir: Path, modality: str) -> Path:
    """Locate the NIfTI file for a modality (t1n, t1c, seg) inside case_dir."""
    matches = sorted(case_dir.glob(f"*-{modality}.nii.gz"))
    if not matches:
        matches = sorted(case_dir.glob(f"*-{modality}.nii"))
    if not matches:
        raise FileNotFoundError(f"No '{modality}' NIfTI file found in {case_dir}")
    return matches[0]


def load_case(case_dir: Path, require_targets: bool = True) -> CaseData:
    """Load T1n (and optionally T1c and seg) NIfTI volumes from a case directory."""
    t1n_img = nib.load(str(find_modality_file(case_dir, "t1n")))
    t1n = np.asarray(t1n_img.get_fdata(), dtype=np.float32)

    t1c = None
    seg = None
    if require_targets:
        t1c_img = nib.load(str(find_modality_file(case_dir, "t1c")))
        seg_img = nib.load(str(find_modality_file(case_dir, "seg")))
        t1c = np.asarray(t1c_img.get_fdata(), dtype=np.float32)
        seg = np.rint(seg_img.get_fdata()).astype(np.int16)

    return CaseData(
        case_id=case_dir.name,
        t1n=t1n,
        t1c=t1c,
        seg=seg,
        affine=t1n_img.affine,
        zooms=tuple(float(z) for z in t1n_img.header.get_zooms()[:3]),
    )


def verify_case_alignment(case_dir: Path, atol: float = 1e-3) -> dict:
    """Verify shape/affine alignment across t1n, t1c, and seg, and inspect seg labels."""
    t1n_img = nib.load(str(find_modality_file(case_dir, "t1n")))
    t1c_img = nib.load(str(find_modality_file(case_dir, "t1c")))
    seg_img = nib.load(str(find_modality_file(case_dir, "seg")))

    shapes = {
        "t1n": tuple(t1n_img.shape),
        "t1c": tuple(t1c_img.shape),
        "seg": tuple(seg_img.shape),
    }
    shapes_match = len(set(shapes.values())) == 1
    affines_match = bool(
        np.allclose(t1n_img.affine, t1c_img.affine, atol=atol)
        and np.allclose(t1n_img.affine, seg_img.affine, atol=atol)
    )

    seg_data = np.rint(seg_img.get_fdata()).astype(np.int16)
    unique_labels = [int(x) for x in np.unique(seg_data)]
    et_voxels = int(np.sum(seg_data == ENHANCING_TUMOR_LABEL))
    legacy_label4_voxels = int(np.sum(seg_data == 4))

    return {
        "case_id": case_dir.name,
        "shapes": shapes,
        "shapes_match": shapes_match,
        "affines_match": affines_match,
        "zooms_mm": [round(float(z), 4) for z in t1n_img.header.get_zooms()[:3]],
        "unique_seg_labels": unique_labels,
        "enhancing_tumor_voxels": et_voxels,
        "legacy_label4_voxels": legacy_label4_voxels,
    }


def normalize_volume(
    vol: np.ndarray,
    brain_mask: np.ndarray | None = None,
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
    method: str = "percentile",
) -> np.ndarray:
    """Normalize foreground intensities to [0, 1], keeping background at 0.0.

    Methods:
      - 'percentile': clip foreground to [lower_pct, upper_pct] and scale to [0, 1].
      - 'zscore': z-score foreground, clip to [-3, 3], and scale linearly to [0, 1].
    """
    vol = np.asarray(vol, dtype=np.float32)
    mask = (vol > 0) if brain_mask is None else np.asarray(brain_mask, dtype=bool)
    out = np.zeros_like(vol, dtype=np.float32)
    if not np.any(mask):
        return out

    fg = vol[mask]
    if method == "zscore":
        mean = float(np.mean(fg))
        std = float(np.std(fg))
        if std <= 1e-8:
            return out
        z = np.clip((fg - mean) / std, -3.0, 3.0)
        out[mask] = (z + 3.0) / 6.0
        return out

    if method != "percentile":
        raise ValueError(f"Unknown normalization method: {method}")

    lo, hi = np.percentile(fg, [lower_pct, upper_pct])
    if hi <= lo:
        return out

    clipped = np.clip(fg, lo, hi)
    out[mask] = (clipped - lo) / (hi - lo)
    return out


def extract_enhancing_tumor_mask(
    seg: np.ndarray,
    et_label: int = ENHANCING_TUMOR_LABEL,
) -> np.ndarray:
    """Convert multi-class BraTS segmentation volume to binary enhancing-tumor mask."""
    return (np.asarray(seg) == et_label).astype(np.float32)


def prepare_scans(
    t1n: np.ndarray,
    t1c: np.ndarray | None = None,
    seg: np.ndarray | None = None,
    min_brain_fraction: float = 0.05,
    et_label: int = ENHANCING_TUMOR_LABEL,
    norm_method: str = "zscore",
) -> dict[str, np.ndarray]:
    """Prepare normalized 2D axial slices from 3D volumes (Prepare Scans — P1).

    Returns a dict with:
      - 't1n': (N, H, W) float32 in [0, 1]
      - 'slice_indices': (N,) int32 axial indices kept
      - 't1c': (N, H, W) float32 in [0, 1] (if t1c is provided)
      - 'et_mask': (N, H, W) float32 in {0, 1} (if seg is provided)
    """
    brain_mask = t1n > 0
    t1n_norm = normalize_volume(t1n, brain_mask=brain_mask, method=norm_method)

    t1c_norm = (
        normalize_volume(t1c, brain_mask=(t1c > 0), method=norm_method)
        if t1c is not None
        else None
    )
    et_vol = extract_enhancing_tumor_mask(seg, et_label=et_label) if seg is not None else None

    num_z = t1n_norm.shape[2]
    kept_indices = [
        z for z in range(num_z) if float(np.mean(brain_mask[:, :, z])) >= min_brain_fraction
    ]

    result: dict[str, np.ndarray] = {
        "t1n": np.stack([t1n_norm[:, :, z] for z in kept_indices], axis=0).astype(np.float32)
        if kept_indices
        else np.empty((0, t1n.shape[0], t1n.shape[1]), dtype=np.float32),
        "slice_indices": np.asarray(kept_indices, dtype=np.int32),
    }
    if t1c_norm is not None:
        result["t1c"] = (
            np.stack([t1c_norm[:, :, z] for z in kept_indices], axis=0).astype(np.float32)
            if kept_indices
            else np.empty((0, t1n.shape[0], t1n.shape[1]), dtype=np.float32)
        )
    if et_vol is not None:
        result["et_mask"] = (
            np.stack([et_vol[:, :, z] for z in kept_indices], axis=0).astype(np.float32)
            if kept_indices
            else np.empty((0, t1n.shape[0], t1n.shape[1]), dtype=np.float32)
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify alignment and preprocess BraTS cases.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional dir to save .npz slices.")
    parser.add_argument("--min-brain-fraction", type=float, default=0.05)
    parser.add_argument("--norm-method", choices=("percentile", "zscore"), default="zscore")
    args = parser.parse_args()

    case_dirs = sorted(p for p in args.raw_dir.iterdir() if p.is_dir() and p.name.startswith("BraTS-"))
    if not case_dirs:
        raise SystemExit(f"No BraTS-* directories found in {args.raw_dir}")

    reports = []
    for case_dir in case_dirs:
        rep = verify_case_alignment(case_dir)
        reports.append(rep)
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            case = load_case(case_dir, require_targets=True)
            prepared = prepare_scans(
                case.t1n,
                case.t1c,
                case.seg,
                min_brain_fraction=args.min_brain_fraction,
                norm_method=args.norm_method,
            )
            np.savez_compressed(args.out_dir / f"{case.case_id}.npz", **prepared)
            rep["num_kept_slices"] = int(len(prepared["slice_indices"]))
            rep["num_tumor_slices"] = int(np.sum(prepared["et_mask"].sum(axis=(1, 2)) > 0))

    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
