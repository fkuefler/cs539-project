"""P1: Baseline evaluation — compare raw T1n against real T1c (SSIM & PSNR) and plot examples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p1.metrics import evaluate_slice_stack
from p1.preprocess import load_case, prepare_scans
from p1.splits import load_splits


def run_baseline(
    raw_dir: Path,
    splits_path: Path | None = None,
    split_name: str | None = None,
    out_json: Path = Path("p1/baseline_results.json"),
    out_fig: Path = Path("p1/baseline_examples.png"),
    max_plot_cases: int = 3,
    norm_method: str = "zscore",
) -> dict:
    """Evaluate raw T1n vs real T1c across cases and save metrics + example figure."""
    if splits_path is not None and splits_path.exists() and split_name is not None:
        splits = load_splits(splits_path)
        case_ids = splits[split_name]
    else:
        case_ids = sorted(
            p.name for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("BraTS-")
        )

    per_case: list[dict] = []
    plot_samples: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]] = []

    for cid in case_ids:
        case = load_case(raw_dir / cid, require_targets=True)
        prep = prepare_scans(case.t1n, case.t1c, case.seg, norm_method=norm_method)
        metrics = evaluate_slice_stack(prep["t1n"], prep["t1c"])
        tumor_per_slice = prep["et_mask"].sum(axis=(1, 2))
        metrics["case_id"] = cid
        metrics["num_tumor_slices"] = int(np.sum(tumor_per_slice > 0))
        per_case.append(metrics)

        if len(plot_samples) < max_plot_cases and len(prep["slice_indices"]) > 0:
            # Pick slice with largest enhancing tumor area (or middle slice if none)
            best_idx = (
                int(np.argmax(tumor_per_slice))
                if float(tumor_per_slice.max()) > 0
                else len(prep["slice_indices"]) // 2
            )
            plot_samples.append(
                (
                    cid,
                    int(prep["slice_indices"][best_idx]),
                    prep["t1n"][best_idx],
                    prep["t1c"][best_idx],
                    prep["et_mask"][best_idx],
                )
            )

    summary = {
        "split": split_name or "all",
        "num_cases": len(per_case),
        "mean_ssim": float(np.mean([c["ssim"] for c in per_case])) if per_case else float("nan"),
        "std_ssim": float(np.std([c["ssim"] for c in per_case])) if per_case else float("nan"),
        "mean_psnr": float(np.mean([c["psnr"] for c in per_case])) if per_case else float("nan"),
        "std_psnr": float(np.std([c["psnr"] for c in per_case])) if per_case else float("nan"),
        "per_case": per_case,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    if plot_samples:
        _save_baseline_figure(plot_samples, out_fig)

    return summary


def _save_baseline_figure(
    samples: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]],
    out_fig: Path,
) -> None:
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.3 * n), squeeze=False)
    col_titles = ["T1n (Input)", "Real T1c (Target)", "|T1c - T1n|", "T1c + Enhancing Tumor"]

    for row, (cid, z, t1n, t1c, et_mask) in enumerate(samples):
        diff = np.abs(t1c - t1n)
        axes[row, 0].imshow(np.rot90(t1n), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"{cid}\nz={z}", fontsize=9)
        axes[row, 1].imshow(np.rot90(t1c), cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(np.rot90(diff), cmap="magma", vmin=0, vmax=0.6)
        axes[row, 3].imshow(np.rot90(t1c), cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(
            np.ma.masked_where(np.rot90(et_mask) < 0.5, np.rot90(et_mask)),
            cmap="autumn",
            alpha=0.65,
        )
        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=10)

    plt.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P1 baseline (T1n vs real T1c).")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits", type=Path, default=Path("p1/splits.json"))
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--out-json", type=Path, default=Path("p1/baseline_results.json"))
    parser.add_argument("--out-fig", type=Path, default=Path("p1/baseline_examples.png"))
    parser.add_argument("--norm-method", choices=("percentile", "zscore"), default="zscore")
    args = parser.parse_args()

    summary = run_baseline(
        raw_dir=args.raw_dir,
        splits_path=args.splits,
        split_name=args.split_name,
        out_json=args.out_json,
        out_fig=args.out_fig,
        norm_method=args.norm_method,
    )
    print(
        f"Baseline ({summary['split']}, n={summary['num_cases']}): "
        f"SSIM={summary['mean_ssim']:.4f} +/- {summary['std_ssim']:.4f}, "
        f"PSNR={summary['mean_psnr']:.2f} +/- {summary['std_psnr']:.2f} dB"
    )


if __name__ == "__main__":
    main()
