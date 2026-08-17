#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import nilearn
import pandas as pd
from nilearn import datasets


N_SUBJECTS = int(os.environ.get("RSFMRI_N_SUBJECTS", "10"))

REPO = Path(__file__).resolve().parents[1]
WORK = Path(
    os.environ.get(
        "RSFMRI_WORK_DIR",
        Path.home()
        / "neuroimaging-data"
        / "fmri-rest_fsl-graphs_braph_adhd200-preproc",
    )
).expanduser().resolve()

DATA_DIR = WORK / "data" / "nilearn"
TABLES = REPO / "results" / "tables"

DATA_DIR.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


def subject_id_from_path(path: str | Path) -> str:
    """Extract the seven-digit ADHD-200 subject identifier."""
    match = re.search(
        r"(?<!\d)(\d{7})(?!\d)",
        str(path),
    )
    if match is None:
        raise RuntimeError(
            f"Could not extract a seven-digit subject ID from {path}"
        )
    return match.group(1)


def normalize_subject_id(value: object) -> str:
    """Normalize a phenotypic Subject value to seven digits."""
    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    match = re.search(r"(\d+)", text)

    if match is None:
        raise RuntimeError(
            f"Could not normalize phenotypic Subject value: {value!r}"
        )

    return match.group(1).zfill(7)


print(f"Downloading {N_SUBJECTS} ADHD resting-state subjects...")

adhd = datasets.fetch_adhd(
    n_subjects=N_SUBJECTS,
    data_dir=str(DATA_DIR),
    resume=True,
    verbose=1,
)

print(
    "Downloading Schaefer 2018 atlas: "
    "100 parcels, 7 networks..."
)

atlas = datasets.fetch_atlas_schaefer_2018(
    n_rois=100,
    yeo_networks=7,
    resolution_mm=2,
    data_dir=str(DATA_DIR),
    verbose=1,
)

functional_files = [
    Path(path).resolve()
    for path in adhd.func
]

confound_files = [
    Path(path).resolve()
    for path in adhd.confounds
]

if len(functional_files) != len(confound_files):
    raise RuntimeError(
        "Functional and confound file counts do not match."
    )

subject_ids = []

for func_file, confound_file in zip(
    functional_files,
    confound_files,
):
    func_id = subject_id_from_path(func_file)
    confound_id = subject_id_from_path(confound_file)

    if func_id != confound_id:
        raise RuntimeError(
            "Functional/confound subject mismatch: "
            f"{func_file} vs {confound_file}"
        )

    subject_ids.append(func_id)

if len(set(subject_ids)) != len(subject_ids):
    raise RuntimeError(
        "Duplicate subject IDs detected in downloaded files."
    )

phenotypic = pd.DataFrame(
    adhd.phenotypic
).copy()

if "Subject" not in phenotypic.columns:
    raise RuntimeError(
        "ADHD phenotypic table does not contain a Subject column."
    )

phenotypic["_subject_id"] = (
    phenotypic["Subject"]
    .map(normalize_subject_id)
)

if phenotypic["_subject_id"].duplicated().any():
    duplicates = sorted(
        phenotypic.loc[
            phenotypic["_subject_id"].duplicated(
                keep=False
            ),
            "_subject_id",
        ].unique()
    )

    raise RuntimeError(
        "Duplicate phenotypic Subject IDs: "
        + ", ".join(duplicates)
    )

phenotypic_by_subject = (
    phenotypic
    .set_index("_subject_id", drop=False)
)

missing = [
    subject_id
    for subject_id in subject_ids
    if subject_id not in phenotypic_by_subject.index
]

if missing:
    raise RuntimeError(
        "Missing phenotypic rows for downloaded subjects: "
        + ", ".join(missing)
    )

aligned = (
    phenotypic_by_subject
    .loc[subject_ids]
    .copy()
    .reset_index(drop=True)
)

aligned = aligned.drop(
    columns=["_subject_id"]
)

aligned.insert(
    0,
    "subject_id",
    subject_ids,
)

aligned.insert(
    1,
    "func_file",
    [str(path) for path in functional_files],
)

aligned.insert(
    2,
    "confounds_file",
    [str(path) for path in confound_files],
)

aligned.to_csv(
    TABLES / "table0_adhd200_subject_manifest.tsv",
    sep="\t",
    index=False,
)

manifest = {
    "nilearn_version": nilearn.__version__,
    "n_subjects": len(functional_files),
    "subject_ids": subject_ids,
    "repetition_time_seconds": float(adhd.t_r),
    "functional_files": [
        str(path)
        for path in functional_files
    ],
    "confounds_files": [
        str(path)
        for path in confound_files
    ],
    "atlas_maps": str(
        Path(atlas.maps).resolve()
    ),
    "atlas_labels": [
        str(label)
        for label in atlas.labels
    ],
    "atlas_n_rois": 100,
    "atlas_yeo_networks": 7,
    "atlas_resolution_mm": 2,
}

with (
    TABLES / "dataset_manifest.json"
).open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        manifest,
        handle,
        indent=2,
    )

print("\nDOWNLOAD COMPLETE")
print(f"Subjects: {len(functional_files)}")
print(f"TR: {adhd.t_r} s")
print(f"Atlas: {atlas.maps}")
print(
    "Manifest: "
    f"{TABLES / 'table0_adhd200_subject_manifest.tsv'}"
)
