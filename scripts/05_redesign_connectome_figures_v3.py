#!/usr/bin/env python3
"""Final V3 presentation refinements for connectome figures.

This script reuses the numerically validated V2 renderer and changes only figure
wording/layout. It does not recompute preprocessing, subject connectivity, graph
construction, node metrics, or network summaries.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
V2_PATH = HERE / "04_redesign_connectome_figures_v2.py"
_spec = importlib.util.spec_from_file_location("connectome_v2", V2_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load {V2_PATH}")
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

NETWORK_ORDER = v2.NETWORK_ORDER
NETWORK_DISPLAY = v2.NETWORK_DISPLAY


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return p.parse_args()


def figure_anatomical_connectome(graph, coords, atlas_world, out: Path) -> None:
    palette = v2.network_palette()
    display_edges = v2.v1.strongest_edges(graph, fraction=0.20)
    weights = np.array([d["weight"] for _, _, d in display_edges], dtype=float)
    lo, hi = float(weights.min()), float(weights.max())
    span = max(hi - lo, 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.9))
    for ax, view in zip(axes, ["Sagittal", "Coronal", "Axial"]):
        v2.add_atlas_outline(ax, atlas_world, view)
        xx, yy, xlabel, ylabel = v2.v1.project_view(coords, view)

        for i, j, d in display_edges:
            scaled = (d["weight"] - lo) / span
            ax.plot(
                [xx[i], xx[j]], [yy[i], yy[j]],
                linewidth=0.35 + 1.55 * scaled,
                alpha=0.11 + 0.36 * scaled,
                color="0.33", zorder=1,
            )

        for network in NETWORK_ORDER:
            idx = coords.index[coords["network"] == network].to_numpy(dtype=int)
            ax.scatter(
                xx[idx], yy[idx], s=24,
                color=[palette[network]], edgecolors="white",
                linewidths=0.35, zorder=2,
            )

        ax.axhline(0, linewidth=0.45, color="0.55", alpha=0.35)
        ax.axvline(0, linewidth=0.45, color="0.55", alpha=0.35)
        ax.set_title(view, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", markersize=6.5,
            markerfacecolor=palette[n], markeredgecolor="white",
            label=NETWORK_DISPLAY[n],
        )
        for n in NETWORK_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.055),
        fontsize=8.5,
        columnspacing=1.3,
        handletextpad=0.4,
    )
    fig.suptitle(
        "15%-density functional graph in MNI projections",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        (
            f"Strongest {len(display_edges)} of {graph.number_of_edges()} graph edges shown "
            "(all positive); gray contour = convex hull of projected Schaefer cortical atlas support."
        ),
        ha="center",
        fontsize=8.8,
    )
    fig.subplots_adjust(bottom=0.24, top=0.86, wspace=0.22)
    v2.v1.savefig(fig, out)


def figure_hubs(nodes, coords, atlas_world, out: Path) -> None:
    palette = v2.network_palette()
    merged = nodes.merge(
        coords[["node", "x_mni", "y_mni", "z_mni"]],
        on="node", how="left", validate="one_to_one",
    )
    top15 = merged.sort_values("hub_rank").head(15).copy()
    top10 = top15.head(10).copy()
    top_bar = top15.sort_values("hub_score", ascending=True)

    fig = plt.figure(figsize=(15.0, 9.0))
    ax1 = fig.add_axes([0.07, 0.14, 0.45, 0.74])
    ax1.barh(
        top_bar["label"], top_bar["hub_score"],
        color=[palette[n] for n in top_bar["network"]],
    )
    ax1.set_xlabel("Composite hub score (z)")
    ax1.set_title("Top 15 hubs", fontweight="bold")
    ax1.tick_params(axis="y", labelsize=8)

    ax2 = fig.add_axes([0.59, 0.18, 0.37, 0.66])
    v2.add_atlas_outline(ax2, atlas_world, "Axial")
    ax2.scatter(
        merged["x_mni"], merged["y_mni"],
        s=22, color="0.72", alpha=0.65,
        edgecolors="white", linewidths=0.25, zorder=1,
    )

    hmin = float(top10["hub_score"].min())
    hmax = float(top10["hub_score"].max())
    denom = max(hmax - hmin, 1e-12)
    for network in NETWORK_ORDER:
        sub = top10.loc[top10["network"] == network]
        if sub.empty:
            continue
        sizes = 95 + 135 * (sub["hub_score"] - hmin) / denom
        ax2.scatter(
            sub["x_mni"], sub["y_mni"],
            s=sizes, color=[palette[network]],
            alpha=0.92, edgecolors="white", linewidths=0.6, zorder=2,
        )

    for row in top10.itertuples(index=False):
        ax2.annotate(
            str(int(row.hub_rank)),
            (row.x_mni, row.y_mni),
            xytext=(4, 4), textcoords="offset points",
            fontsize=8, fontweight="bold", zorder=3,
        )

    ax2.axhline(0, linewidth=0.45, color="0.55", alpha=0.35)
    ax2.axvline(0, linewidth=0.45, color="0.55", alpha=0.35)
    ax2.set_xlabel("MNI x (mm)")
    ax2.set_ylabel("MNI y (mm)")
    ax2.set_title("Top-10 hubs — axial MNI projection", fontweight="bold")
    ax2.set_aspect("equal", adjustable="box")

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", markersize=7,
            markerfacecolor=palette[n], markeredgecolor="white",
            label=NETWORK_DISPLAY[n],
        )
        for n in NETWORK_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.76, 0.055),
    )
    fig.suptitle(
        "Functional-connectome hub ranking and MNI spatial distribution",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.025,
        (
            "Ranks 1–10 are highlighted; gray points are the remaining Schaefer-100 parcel centroids. "
            "Gray contour = projected atlas-support convex hull."
        ),
        ha="center",
        fontsize=8.8,
    )
    v2.v1.savefig(fig, out)


def figure_network_graph(coords, summary, out: Path) -> None:
    v2.figure_network_graph(coords, summary, out)


def main() -> None:
    args = parse_args()

    v2.figure_anatomical_connectome = figure_anatomical_connectome
    v2.figure_hubs = figure_hubs
    v2.figure_network_graph = figure_network_graph
    v2.main()

    validation_path = args.output_dir.resolve() / "redesign_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["revision"] = "v3"
    validation["anatomical_reference"] = (
        "convex hull of projected Schaefer cortical atlas support; not a structural MRI background"
    )
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    print("CONNECTOME_VISUAL_REDESIGN_V3_PASSED")


if __name__ == "__main__":
    main()
