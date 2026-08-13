#!/usr/bin/env python3
"""Publication-style cortical surface maps for the validated Schaefer-100 connectome.

This script does not recompute fMRI preprocessing or connectivity. It maps the
validated parcel-level results onto the official Schaefer-100 fsLR32k cortical
indexing and renders Conte69/fsLR32k cortical surfaces with BrainSpace/Surfplot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd

from brainspace.datasets import load_conte69, load_marker, load_parcellation
from surfplot import Plot

NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NETWORK_DISPLAY = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal attention",
    "SalVentAttn": "Salience / ventral attention",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default",
}
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


def network_from_label(label: str) -> str:
    parts = label.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected Schaefer label: {label}")
    return parts[1]


def load_official_label_colors(path: Path) -> dict[int, tuple[float, float, float]]:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    colors: dict[int, tuple[float, float, float]] = {}
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        parts = lines[i + 1].split()
        if len(parts) < 5:
            continue
        key = int(parts[0])
        colors[key] = tuple(float(x) / 255.0 for x in parts[1:4])
    return colors


def cifti_surface_labels(dlabel: Path, n_lh: int, n_rh: int) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(dlabel))
    data = np.asanyarray(img.dataobj)
    if data.shape[0] != 1:
        raise RuntimeError(f"Expected one CIFTI label map, got {data.shape}")
    axis = img.header.get_axis(1)
    lh = np.zeros(n_lh, dtype=np.int32)
    rh = np.zeros(n_rh, dtype=np.int32)
    found_lh = False
    found_rh = False
    for structure, slc, model in axis.iter_structures():
        name = str(structure).upper()
        vertices = np.asarray(model.vertex, dtype=int)
        values = np.asarray(data[0, slc], dtype=int)
        if values.size != vertices.size:
            raise RuntimeError(f"CIFTI {structure}: value/vertex mismatch")
        if "LEFT" in name:
            lh[vertices] = values
            found_lh = True
        elif "RIGHT" in name:
            rh[vertices] = values
            found_rh = True
    if not found_lh or not found_rh:
        raise RuntimeError("CIFTI did not contain both cortical hemispheres")
    return lh, rh


def map_parcel_values(values: np.ndarray, label_lh: np.ndarray, label_rh: np.ndarray):
    if values.shape != (100,):
        raise RuntimeError(f"Expected 100 parcel values, got {values.shape}")
    out_lh = np.full(label_lh.shape, np.nan, dtype=float)
    out_rh = np.full(label_rh.shape, np.nan, dtype=float)
    for key in range(1, 101):
        out_lh[label_lh == key] = float(values[key - 1])
        out_rh[label_rh == key] = float(values[key - 1])
    return out_lh, out_rh


def save_surface_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig)
    if not path.exists() or path.stat().st_size < 50_000:
        raise RuntimeError(f"Surface figure failed validation: {path}")


def new_plot(surf_lh, surf_rh) -> Plot:
    return Plot(
        surf_lh=surf_lh,
        surf_rh=surf_rh,
        layout="grid",
        views=["lateral", "medial"],
        mirror_views=True,
        size=(1100, 620),
        zoom=1.45,
        background=(1, 1, 1),
        brightness=0.60,
    )


def add_curvature(p: Plot, curv_lh: np.ndarray, curv_rh: np.ndarray) -> None:
    q = float(np.nanpercentile(np.abs(np.concatenate([curv_lh, curv_rh])), 97.5))
    q = max(q, 1e-6)
    p.add_layer(
        {"left": curv_lh, "right": curv_rh},
        cmap="Greys",
        color_range=(-q, q),
        alpha=0.58,
        cbar=False,
        zero_transparent=False,
    )


def add_parcel_outlines(p: Plot, label_lh: np.ndarray, label_rh: np.ndarray, alpha: float = 0.28) -> None:
    p.add_layer(
        {"left": label_lh, "right": label_rh},
        cmap="gray",
        as_outline=True,
        cbar=False,
        alpha=alpha,
        zero_transparent=True,
    )


def figure_networks(
    surf_lh, surf_rh, curv_lh, curv_rh, label_lh, label_rh,
    parcel_labels: list[str], parcel_colors: dict[int, tuple[float, float, float]], out: Path,
) -> list[str]:
    network_index = {n: i + 1 for i, n in enumerate(NETWORK_ORDER)}
    parcel_network_values = np.array(
        [network_index[network_from_label(x)] for x in parcel_labels], dtype=float
    )
    net_lh, net_rh = map_parcel_values(parcel_network_values, label_lh, label_rh)
    network_colors = []
    for network in NETWORK_ORDER:
        keys = [i + 1 for i, label in enumerate(parcel_labels) if network_from_label(label) == network]
        rgb = np.mean([parcel_colors[k] for k in keys], axis=0)
        network_colors.append(tuple(float(x) for x in rgb))
    cmap = ListedColormap(network_colors, name="SchaeferYeo7OfficialFamilyColors")
    p = new_plot(surf_lh, surf_rh)
    add_curvature(p, curv_lh, curv_rh)
    p.add_layer(
        {"left": net_lh, "right": net_rh}, cmap=cmap, color_range=(1, 7),
        cbar=False, alpha=0.96, zero_transparent=True,
    )
    add_parcel_outlines(p, label_lh, label_rh, alpha=0.32)
    fig = p.build()
    fig.suptitle(
        "Schaefer-100 parcellation grouped by Yeo 7 functional networks",
        fontsize=14, fontweight="bold", y=0.98,
    )
    handles = [
        Patch(facecolor=network_colors[i], edgecolor="none", label=NETWORK_DISPLAY[n])
        for i, n in enumerate(NETWORK_ORDER)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=9)
    fig.subplots_adjust(top=0.90, bottom=0.11)
    save_surface_figure(fig, out)
    return [to_hex(c) for c in network_colors]


def figure_hub_score(
    surf_lh, surf_rh, curv_lh, curv_rh, label_lh, label_rh,
    hub_values: np.ndarray, seed_key: int, out: Path,
) -> None:
    hub_lh, hub_rh = map_parcel_values(hub_values, label_lh, label_rh)
    vmax = float(np.nanmax(np.abs(hub_values)))
    seed_lh = (label_lh == seed_key).astype(float)
    seed_rh = (label_rh == seed_key).astype(float)
    p = new_plot(surf_lh, surf_rh)
    add_curvature(p, curv_lh, curv_rh)
    p.add_layer(
        {"left": hub_lh, "right": hub_rh}, cmap="RdBu_r", color_range=(-vmax, vmax),
        cbar=True, cbar_label="Composite hub score (z)", alpha=0.96, zero_transparent=False,
    )
    add_parcel_outlines(p, label_lh, label_rh, alpha=0.20)
    p.add_layer(
        {"left": seed_lh, "right": seed_rh}, cmap=ListedColormap(["black"]),
        as_outline=True, cbar=False, alpha=1.0, zero_transparent=True,
    )
    fig = p.build(cbar_kws={"location": "right", "draw_border": False, "aspect": 12,
                            "shrink": 0.28, "decimals": 1, "pad": 0.01})
    fig.suptitle("Parcel-wise functional-connectome hub score", fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.012, f"Black outline marks the highest-ranked hub: {SEED_LABEL}.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=0.90, bottom=0.08)
    save_surface_figure(fig, out)


def figure_seed_fingerprint(
    surf_lh, surf_rh, curv_lh, curv_rh, label_lh, label_rh,
    fingerprint: np.ndarray, seed_key: int, out: Path,
) -> float:
    vals = fingerprint.copy().astype(float)
    vals[seed_key - 1] = np.nan
    map_lh, map_rh = map_parcel_values(vals, label_lh, label_rh)
    finite = vals[np.isfinite(vals)]
    vmax = float(np.max(np.abs(finite)))
    seed_lh = (label_lh == seed_key).astype(float)
    seed_rh = (label_rh == seed_key).astype(float)
    p = new_plot(surf_lh, surf_rh)
    add_curvature(p, curv_lh, curv_rh)
    p.add_layer(
        {"left": map_lh, "right": map_rh}, cmap="RdBu_r", color_range=(-vmax, vmax),
        cbar=True, cbar_label=f"Pearson r with {SEED_LABEL}", alpha=0.96, zero_transparent=False,
    )
    add_parcel_outlines(p, label_lh, label_rh, alpha=0.20)
    p.add_layer(
        {"left": seed_lh, "right": seed_rh}, cmap=ListedColormap(["black"]),
        as_outline=True, cbar=False, alpha=1.0, zero_transparent=True,
    )
    fig = p.build(cbar_kws={"location": "right", "draw_border": False, "aspect": 12,
                            "shrink": 0.28, "decimals": 2, "pad": 0.01})
    fig.suptitle(f"Connectivity fingerprint of {SEED_LABEL}", fontsize=14, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.012,
        "Group-mean Pearson r (10 subjects; averaged in Fisher-z and back-transformed). "
        "Seed self-correlation is excluded; black outline marks the seed parcel.",
        ha="center", fontsize=8.8,
    )
    fig.subplots_adjust(top=0.90, bottom=0.08)
    save_surface_figure(fig, out)
    return vmax


def main() -> None:
    args = parse_args()
    repo = args.repo_dir.resolve()
    work = args.work_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ref = require(work / "reference_surface_connectome")
    dlabel = require(ref / "Schaefer2018_100Parcels_7Networks_order.dlabel.nii")
    label_table = require(ref / "Schaefer2018_100Parcels_7Networks_workbench_labels.txt")

    matrix_df = pd.read_csv(require(repo / "results/tables/table1_group_mean_connectome.tsv"), sep="\t", index_col=0)
    labels = list(matrix_df.index)
    if labels != list(matrix_df.columns) or len(labels) != 100:
        raise RuntimeError("Public connectome matrix labels are invalid")
    nodes = pd.read_csv(require(repo / "results/tables/table5_group_node_metrics.tsv"), sep="\t").sort_values("node")
    if list(nodes["label"]) != labels:
        raise RuntimeError("Node metrics do not match matrix label order")

    matrix = matrix_df.to_numpy(dtype=float)
    seed_idx = labels.index(SEED_LABEL)
    seed_key = seed_idx + 1
    hub_values = nodes["hub_score"].to_numpy(dtype=float)
    fingerprint = matrix[seed_idx].copy()

    surf_lh, surf_rh = load_conte69()
    n_lh, n_rh = int(surf_lh.n_points), int(surf_rh.n_points)
    if (n_lh, n_rh) != (32492, 32492):
        raise RuntimeError(f"Unexpected Conte69 vertex counts: {n_lh}, {n_rh}")
    curv_lh, curv_rh = load_marker("curvature")
    bs_lh, bs_rh = load_parcellation("schaefer", scale=100, join=False)
    bs_lh, bs_rh = np.asarray(bs_lh, dtype=np.int32), np.asarray(bs_rh, dtype=np.int32)
    cifti_lh, cifti_rh = cifti_surface_labels(dlabel, n_lh, n_rh)

    exact_left = bool(np.array_equal(cifti_lh, bs_lh))
    exact_right = bool(np.array_equal(cifti_rh, bs_rh))
    mismatches_left = int(np.sum(cifti_lh != bs_lh))
    mismatches_right = int(np.sum(cifti_rh != bs_rh))
    if not exact_left or not exact_right:
        raise RuntimeError(
            "BrainSpace Schaefer-100 vertex labels do not exactly match the official fsLR32k CIFTI: "
            f"LH mismatches={mismatches_left}, RH mismatches={mismatches_right}"
        )

    parcel_colors = load_official_label_colors(label_table)
    if sorted(parcel_colors) != list(range(1, 101)):
        raise RuntimeError("Official Workbench label table does not contain keys 1..100")

    network_hex = figure_networks(
        surf_lh, surf_rh, curv_lh, curv_rh, cifti_lh, cifti_rh,
        labels, parcel_colors, output / "surface_schaefer100_yeo7.png",
    )
    figure_hub_score(
        surf_lh, surf_rh, curv_lh, curv_rh, cifti_lh, cifti_rh,
        hub_values, seed_key, output / "surface_hub_score.png",
    )
    fingerprint_vmax = figure_seed_fingerprint(
        surf_lh, surf_rh, curv_lh, curv_rh, cifti_lh, cifti_rh,
        fingerprint, seed_key, output / "surface_seed_fingerprint_LH_Default_Par_2.png",
    )

    validation = {
        "surface": "Conte69 / fsLR32k",
        "vertices_lh": n_lh,
        "vertices_rh": n_rh,
        "official_cifti_vs_brainspace_exact_lh": exact_left,
        "official_cifti_vs_brainspace_exact_rh": exact_right,
        "cifti_vs_brainspace_mismatches_lh": mismatches_left,
        "cifti_vs_brainspace_mismatches_rh": mismatches_right,
        "parcel_keys": [int(x) for x in sorted(parcel_colors)],
        "seed_label": SEED_LABEL,
        "seed_key": seed_key,
        "seed_hub_rank": int(nodes.iloc[seed_idx]["hub_rank"]),
        "seed_hub_score": float(nodes.iloc[seed_idx]["hub_score"]),
        "fingerprint_self_correlation_excluded": True,
        "fingerprint_color_abs_max": fingerprint_vmax,
        "network_family_colors_hex": dict(zip(NETWORK_ORDER, network_hex)),
        "outputs": [
            "surface_schaefer100_yeo7.png",
            "surface_hub_score.png",
            "surface_seed_fingerprint_LH_Default_Par_2.png",
        ],
    }
    (output / "surface_mapping_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    print("SURFACE_PUBLICATION_MAPS_PASSED")
    print(f"CIFTI_VS_BRAINSPACE_EXACT_LH={exact_left}")
    print(f"CIFTI_VS_BRAINSPACE_EXACT_RH={exact_right}")
    print(f"SEED_KEY={seed_key}")
    for path in sorted(output.glob("*.png")):
        print(f"WROTE {path.name}\t{path.stat().st_size} bytes")
    print("WROTE surface_mapping_validation.json\t" + str((output / "surface_mapping_validation.json").stat().st_size) + " bytes")


if __name__ == "__main__":
    main()
