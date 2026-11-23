"""Download a starter subset of BraTS-GLI 2023 (t1n, t1c, seg) from Hugging Face.

Enforces one complete case per patient (deduplicating timepoints like -000, -001)
and downloads only the three modalities needed for our project (t1n, t1c, seg).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

REPO_ID = "MedOtter/brats2023-gli-dataset"
REQUIRED_SUFFIXES = ("-t1n.nii.gz", "-t1c.nii.gz", "-seg.nii.gz")
CASE_PATTERN = re.compile(r"(BraTS-GLI-(\d+)-(\d+))")


def extract_patient_id(case_id: str) -> str:
    """Return the patient identifier (e.g. 'BraTS-GLI-00000') from a case ID."""
    match = CASE_PATTERN.search(case_id)
    if match:
        return f"BraTS-GLI-{match.group(2)}"
    return case_id


def discover_complete_cases(repo_id: str = REPO_ID) -> dict[str, dict[str, str]]:
    """List complete cases (t1n, t1c, seg) in the Hugging Face dataset repo."""
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    for filepath in files:
        for suffix in REQUIRED_SUFFIXES:
            if filepath.endswith(suffix):
                match = CASE_PATTERN.search(filepath)
                if not match:
                    continue
                case_id = match.group(1)
                modality = suffix.removeprefix("-").removesuffix(".nii.gz")
                grouped[case_id][modality] = filepath
                break

    complete = {
        case_id: mods
        for case_id, mods in sorted(grouped.items())
        if set(mods.keys()) == {"t1n", "t1c", "seg"}
    }
    return complete


def select_one_case_per_patient(
    complete_cases: dict[str, dict[str, str]],
    num_cases: int | None = None,
    explicit_cases: list[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Select one complete case per unique patient."""
    if explicit_cases:
        selected: dict[str, dict[str, str]] = {}
        for raw_id in explicit_cases:
            case_id = raw_id if raw_id.startswith("BraTS-GLI-") else f"BraTS-GLI-{raw_id}"
            if case_id not in complete_cases:
                raise ValueError(f"Requested case '{case_id}' not found among complete cases.")
            selected[case_id] = complete_cases[case_id]
        return selected

    seen_patients: set[str] = set()
    selected = {}
    for case_id, mods in sorted(complete_cases.items()):
        patient_id = extract_patient_id(case_id)
        if patient_id in seen_patients:
            continue
        seen_patients.add(patient_id)
        selected[case_id] = mods
        if num_cases is not None and len(selected) >= num_cases:
            break
    return selected


def download_cases(
    output_dir: Path,
    num_cases: int = 10,
    explicit_cases: list[str] | None = None,
    repo_id: str = REPO_ID,
) -> list[dict]:
    """Download selected BraTS-GLI 2023 cases into output_dir/<case_id>/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    complete = discover_complete_cases(repo_id=repo_id)
    selected = select_one_case_per_patient(
        complete,
        num_cases=num_cases,
        explicit_cases=explicit_cases,
    )

    manifest: list[dict] = []
    for case_id, mods in tqdm(selected.items(), desc="Downloading BraTS cases"):
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        local_paths: dict[str, str] = {}
        for modality in ("t1n", "t1c", "seg"):
            remote_path = mods[modality]
            downloaded = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=remote_path,
                local_dir=output_dir / "_hf_cache",
            )
            target = case_dir / Path(remote_path).name
            if not target.exists():
                target.symlink_to(Path(downloaded).resolve())
            local_paths[modality] = str(target)

        manifest.append(
            {
                "case_id": case_id,
                "patient_id": extract_patient_id(case_id),
                "paths": local_paths,
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloaded {len(manifest)} cases to {output_dir} (manifest: {manifest_path})")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BraTS-GLI 2023 starter subset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional explicit case IDs (e.g. 00000-000 00002-000 00003-000).",
    )
    args = parser.parse_args()
    download_cases(
        output_dir=args.output_dir,
        num_cases=args.num_cases,
        explicit_cases=args.cases,
    )


if __name__ == "__main__":
    main()
