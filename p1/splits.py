"""P1: Patient-level train / dev / eval splits with zero patient leakage."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from p1.download_brats import extract_patient_id


def create_patient_splits(
    case_ids: list[str],
    train_frac: float = 0.70,
    dev_frac: float = 0.15,
    eval_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split case IDs into train/dev/eval grouped strictly by patient ID."""
    if abs((train_frac + dev_frac + eval_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + dev_frac + eval_frac must sum to 1.0")

    by_patient: dict[str, list[str]] = defaultdict(list)
    for cid in sorted(case_ids):
        by_patient[extract_patient_id(cid)].append(cid)

    patients = sorted(by_patient.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    n = len(patients)
    if n < 3:
        raise ValueError(f"Need at least 3 unique patients to form train/dev/eval splits, got {n}.")

    n_eval = max(1, int(round(n * eval_frac)))
    n_dev = max(1, int(round(n * dev_frac)))
    n_train = n - n_dev - n_eval
    if n_train < 1:
        n_train = 1
        remaining = n - 1
        n_dev = remaining // 2
        n_eval = remaining - n_dev

    train_patients = patients[:n_train]
    dev_patients = patients[n_train : n_train + n_dev]
    eval_patients = patients[n_train + n_dev :]

    # Strict leak check
    assert set(train_patients).isdisjoint(dev_patients)
    assert set(train_patients).isdisjoint(eval_patients)
    assert set(dev_patients).isdisjoint(eval_patients)

    return {
        "train": sorted(cid for p in train_patients for cid in by_patient[p]),
        "dev": sorted(cid for p in dev_patients for cid in by_patient[p]),
        "eval": sorted(cid for p in eval_patients for cid in by_patient[p]),
    }


def save_splits(splits: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n")


def load_splits(path: Path) -> dict[str, list[str]]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-level train/dev/eval splits.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("p1/splits.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    case_ids = sorted(
        p.name for p in args.raw_dir.iterdir() if p.is_dir() and p.name.startswith("BraTS-")
    )
    splits = create_patient_splits(case_ids, seed=args.seed)
    save_splits(splits, args.out)
    print(
        f"Saved splits to {args.out}: "
        f"train={len(splits['train'])}, dev={len(splits['dev'])}, eval={len(splits['eval'])}"
    )


if __name__ == "__main__":
    main()
