# P1 — Data Access, Preprocessing, Splits & Baseline

Owner: Francis Kuefler (`fkuefler`)

- `download_brats.py`: Downloads complete BraTS-GLI 2023 cases (`t1n`, `t1c`, `seg`) from `MedOtter/brats2023-gli-dataset`, enforcing one case per unique patient.
- `preprocess.py`: Verifies shape/affine alignment and segmentation labels (`label == 3` for enhancing tumor in BraTS 2023), normalizes foreground intensities to `[0, 1]`, and extracts 2D axial slices.
- `splits.py`: Generates deterministic patient-level `train` / `dev` / `eval` splits (`splits.json`) with zero patient leakage.
- `metrics.py`: Implements SSIM, PSNR, and binary Dice metrics.
- `baseline.py`: Evaluates raw `T1n` vs real `T1c` (SSIM/PSNR) and generates visual comparison figures.
