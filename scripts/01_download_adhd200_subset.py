#!/usr/bin/env python3
from __future__ import annotations

import json
import os
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

print(f"Downloading {N_SUBJECTS} ADHD resting-state subjects...")
adhd = datasets.fetch_adhd(
    n_subjects=N_SUBJECTS,
    data_dir=str(DATA_DIR),
    resume=True,
    verbose=1,
)

print("Downloading Schaefer 2018 atlas: 100 parcels, 7 networks...")
atlas = datasets.fetch_atlas_schaefer_2018(
    n_rois=100,
    yeo_networks=7,
    resolution_mm=2,
    data_dir=str(DATA_DIR),
    verbose=1,
)

phenotypic = pd.DataFrame(adhd.phenotypic).copy()
phenotypic.insert(
    0,
    "func_file",
    [str(Path(path).resolve()) for path in adhd.func],
)
phenotypic.insert(
    1,
    "confounds_file",
    [str(Path(path).resolve()) for path in adhd.confounds],
)
phenotypic.to_csv(
    TABLES / "table0_adhd200_subject_manifest.tsv",
    sep="\t",
    index=False,
)

manifest = {
    "nilearn_version": nilearn.__version__,
    "n_subjects": len(adhd.func),
    "repetition_time_seconds": float(adhd.t_r),
    "functional_files": [str(Path(path).resolve()) for path in adhd.func],
    "confounds_files": [str(Path(path).resolve()) for path in adhd.confounds],
    "atlas_maps": str(Path(atlas.maps).resolve()),
    "atlas_labels": [str(label) for label in atlas.labels],
    "atlas_n_rois": 100,
    "atlas_yeo_networks": 7,
    "atlas_resolution_mm": 2,
}

with (TABLES / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)

print("\nDOWNLOAD COMPLETE")
print(f"Subjects: {len(adhd.func)}")
print(f"TR: {adhd.t_r} s")
print(f"Atlas: {atlas.maps}")
print(
    "Manifest: "
    f"{TABLES / 'table0_adhd200_subject_manifest.tsv'}"
)
