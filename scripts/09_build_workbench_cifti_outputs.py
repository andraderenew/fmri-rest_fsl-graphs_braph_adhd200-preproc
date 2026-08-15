#!/usr/bin/env python3
"""Build native CIFTI outputs for Connectome Workbench from validated Schaefer-100 results.

Creates dense cortical scalar maps for hub score and the top-hub connectivity
fingerprint, parcel scalar equivalents, and the full group-mean 100x100 pconn.
The official CBIG Schaefer2018 fsLR32k dlabel is the authoritative dense mapping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
from nibabel import cifti2
import numpy as np
import pandas as pd

SEED_LABEL = "LH_Default_Par_2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--repo-dir", type=Path, default=Path(__file__).resolve().parents[1])
    return p.parse_args()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def save_cifti(data: np.ndarray, axes, path: Path) -> None:
    header = cifti2.Cifti2Header.from_axes(axes)
    img = cifti2.Cifti2Image(dataobj=np.asarray(data, dtype=np.float32), header=header)
    img.update_headers()
    nib.save(img, str(path))
    if not path.exists() or path.stat().st_size < 1024:
        raise RuntimeError(f"Failed to write CIFTI: {path}")


def main() -> None:
    args = parse_args()
    repo = args.repo_dir.resolve()
    work = args.work_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    ref = require(work / "reference_surface_connectome")
    dlabel = require(ref / "Schaefer2018_100Parcels_7Networks_order.dlabel.nii")
    matrix_tsv = require(repo / "results/tables/table1_group_mean_connectome.tsv")
    nodes_tsv = require(repo / "results/tables/table5_group_node_metrics.tsv")

    matrix_df = pd.read_csv(matrix_tsv, sep="\t", index_col=0)
    labels = list(matrix_df.index)
    if len(labels) != 100 or labels != list(matrix_df.columns):
        raise RuntimeError("Expected square 100x100 matrix with matching row/column labels")
    matrix = matrix_df.to_numpy(dtype=float)

    nodes = pd.read_csv(nodes_tsv, sep="\t").sort_values("node")
    if len(nodes) != 100 or list(nodes["label"]) != labels:
        raise RuntimeError("Node metrics do not match matrix parcel order")
    hub_values = nodes["hub_score"].to_numpy(dtype=float)

    seed_idx = labels.index(SEED_LABEL)
    seed_key = seed_idx + 1
    fingerprint = matrix[seed_idx].copy()
    fingerprint[seed_idx] = np.nan

    atlas = nib.load(str(dlabel))
    label_axis = atlas.header.get_axis(0)
    brain_axis = atlas.header.get_axis(1)
    dense_keys = np.asarray(atlas.dataobj)[0].astype(np.int32)

    present = sorted(set(np.unique(dense_keys).astype(int)) - {0})
    if present != list(range(1, 101)):
        raise RuntimeError(f"Official dlabel keys are not exactly 1..100: {present}")

    table = label_axis.label[0]
    for key in range(1, 101):
        if key not in table:
            raise RuntimeError(f"Missing key {key} in official dlabel label table")
        official_name = table[key][0]
        normalized = official_name.replace("7Networks_", "", 1)
        if normalized != labels[key - 1]:
            raise RuntimeError(
                f"Parcel-name mismatch at key {key}: official={official_name}, matrix={labels[key - 1]}"
            )

    hub_dense = np.full(dense_keys.shape, np.nan, dtype=float)
    fingerprint_dense = np.full(dense_keys.shape, np.nan, dtype=float)
    for key in range(1, 101):
        mask = dense_keys == key
        hub_dense[mask] = hub_values[key - 1]
        fingerprint_dense[mask] = fingerprint[key - 1]

    parcel_models = []
    for key, name in enumerate(labels, start=1):
        mask = dense_keys == key
        if not np.any(mask):
            raise RuntimeError(f"Parcel {key} has no dense brainordinates")
        parcel_models.append((name, brain_axis[mask]))
    parcels_axis = cifti2.ParcelsAxis.from_brain_models(parcel_models)

    hub_dscalar = out / "hub_score.dscalar.nii"
    fingerprint_dscalar = out / f"{SEED_LABEL}_fingerprint.dscalar.nii"
    hub_pscalar = out / "hub_score.pscalar.nii"
    fingerprint_pscalar = out / f"{SEED_LABEL}_fingerprint.pscalar.nii"
    pconn = out / "group_mean_connectome.pconn.nii"

    save_cifti(
        hub_dense[None, :],
        (cifti2.ScalarAxis(["Hub score (z)"]), brain_axis),
        hub_dscalar,
    )
    save_cifti(
        fingerprint_dense[None, :],
        (cifti2.ScalarAxis([f"Pearson r with {SEED_LABEL}"]), brain_axis),
        fingerprint_dscalar,
    )
    save_cifti(
        hub_values[None, :],
        (cifti2.ScalarAxis(["Hub score (z)"]), parcels_axis),
        hub_pscalar,
    )
    save_cifti(
        fingerprint[None, :],
        (cifti2.ScalarAxis([f"Pearson r with {SEED_LABEL}"]), parcels_axis),
        fingerprint_pscalar,
    )
    save_cifti(matrix, (parcels_axis, parcels_axis), pconn)

    # Re-open all files and validate axes/data exactly.
    hd = nib.load(str(hub_dscalar))
    fd = nib.load(str(fingerprint_dscalar))
    hp = nib.load(str(hub_pscalar))
    fp = nib.load(str(fingerprint_pscalar))
    pc = nib.load(str(pconn))

    if np.asanyarray(hd.dataobj).shape != (1, brain_axis.size):
        raise RuntimeError("Unexpected hub dscalar shape")
    if np.asanyarray(fd.dataobj).shape != (1, brain_axis.size):
        raise RuntimeError("Unexpected fingerprint dscalar shape")
    if np.asanyarray(hp.dataobj).shape != (1, 100):
        raise RuntimeError("Unexpected hub pscalar shape")
    if np.asanyarray(fp.dataobj).shape != (1, 100):
        raise RuntimeError("Unexpected fingerprint pscalar shape")
    if np.asanyarray(pc.dataobj).shape != (100, 100):
        raise RuntimeError("Unexpected pconn shape")

    pconn_data = np.asanyarray(pc.dataobj).astype(float)
    pconn_max_diff = float(np.nanmax(np.abs(pconn_data - matrix)))
    if pconn_max_diff > 1e-6:
        raise RuntimeError(f"pconn differs from public matrix: max diff {pconn_max_diff}")

    hub_reloaded = np.asanyarray(hd.dataobj)[0].astype(float)
    fp_reloaded = np.asanyarray(fd.dataobj)[0].astype(float)
    max_hub_dense_error = 0.0
    max_fp_dense_error = 0.0
    for key in range(1, 101):
        mask = dense_keys == key
        expected_hub = hub_values[key - 1]
        max_hub_dense_error = max(
            max_hub_dense_error,
            float(np.nanmax(np.abs(hub_reloaded[mask] - expected_hub))),
        )
        if key == seed_key:
            if not np.all(np.isnan(fp_reloaded[mask])):
                raise RuntimeError("Seed self-correlation is not NaN in fingerprint dscalar")
        else:
            expected_fp = fingerprint[key - 1]
            max_fp_dense_error = max(
                max_fp_dense_error,
                float(np.nanmax(np.abs(fp_reloaded[mask] - expected_fp))),
            )

    parcel_names_row = list(pc.header.get_axis(0).name)
    parcel_names_col = list(pc.header.get_axis(1).name)
    if parcel_names_row != labels or parcel_names_col != labels:
        raise RuntimeError("pconn parcel-axis names do not match matrix labels")

    validation = {
        "source_dlabel": "results/workbench/Schaefer2018_100Parcels_7Networks_order.dlabel.nii",
        "source_matrix": "results/tables/table1_group_mean_connectome.tsv",
        "source_node_metrics": "results/tables/table5_group_node_metrics.tsv",
        "brainordinates": int(brain_axis.size),
        "parcels": 100,
        "seed_label": SEED_LABEL,
        "seed_key": seed_key,
        "seed_hub_rank": int(nodes.iloc[seed_idx]["hub_rank"]),
        "seed_hub_score": float(nodes.iloc[seed_idx]["hub_score"]),
        "fingerprint_self_correlation_excluded": True,
        "pconn_shape": [100, 100],
        "pconn_vs_public_matrix_max_abs_diff": pconn_max_diff,
        "hub_dense_max_abs_error": max_hub_dense_error,
        "fingerprint_dense_max_abs_error": max_fp_dense_error,
        "outputs": [
            hub_dscalar.name,
            fingerprint_dscalar.name,
            hub_pscalar.name,
            fingerprint_pscalar.name,
            pconn.name,
        ],
    }
    (out / "workbench_cifti_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    print("WORKBENCH_CIFTI_EXPORT_PASSED")
    for path in [hub_dscalar, fingerprint_dscalar, hub_pscalar, fingerprint_pscalar, pconn]:
        print(f"WROTE\t{path.name}\t{path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
