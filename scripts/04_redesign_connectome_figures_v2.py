#!/usr/bin/env python3
"""Refined V2 connectome figures from previously validated derivatives.

This renderer does not recompute preprocessing, subject connectivity, or graph
metrics. It reuses the validated inputs and graph-building functions from the V1
renderer while improving scientific labeling and figure layout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import networkx as nx
import nibabel as nib
from nibabel.affines import apply_affine
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "03_redesign_connectome_figures.py"
_spec = importlib.util.spec_from_file_location("connectome_v1", V1_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load {V1_PATH}")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

NETWORK_ORDER = v1.NETWORK_ORDER
NETWORK_DISPLAY = v1.NETWORK_DISPLAY


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


def network_palette():
    return v1.network_palette()


def atlas_world_support(atlas_path: Path, max_points: int = 20000) -> np.ndarray:
    img = nib.load(str(atlas_path))
    data = np.asanyarray(img.dataobj)
    vox = np.argwhere(data > 0)
    if vox.shape[0] < 3:
        raise RuntimeError("Atlas cortical support is unexpectedly empty.")
    step = max(1, int(np.ceil(vox.shape[0] / max_points)))
    return apply_affine(img.affine, vox[::step])


def projected_xy(world: np.ndarray, view: str) -> np.ndarray:
    if view == "Axial":
        return world[:, [0, 1]]
    if view == "Coronal":
        return world[:, [0, 2]]
    return world[:, [1, 2]]


def add_atlas_outline(ax, world: np.ndarray, view: str) -> None:
    pts = projected_xy(world, view)
    hull = ConvexHull(pts)
    poly = pts[hull.vertices]
    ax.fill(
        poly[:, 0],
        poly[:, 1],
        facecolor="0.965",
        edgecolor="0.72",
        linewidth=1.0,
        zorder=0,
    )
    dx = float(poly[:, 0].max() - poly[:, 0].min())
    dy = float(poly[:, 1].max() - poly[:, 1].min())
    ax.set_xlim(poly[:, 0].min() - 0.04 * dx, poly[:, 0].max() + 0.04 * dx)
    ax.set_ylim(poly[:, 1].min() - 0.04 * dy, poly[:, 1].max() + 0.04 * dy)


def figure_ordered_matrix(matrix: np.ndarray, coords: pd.DataFrame, out: Path) -> None:
    ordered = v1.ordered_metadata(coords)
    idx = ordered["node"].to_numpy(dtype=int)
    M = matrix[np.ix_(idx, idx)]
    blocks = v1.network_blocks(ordered)

    offdiag = M[~np.eye(M.shape[0], dtype=bool)]
    vmax = max(float(np.percentile(np.abs(offdiag), 99.5)), 0.1)
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(11.6, 10.5))
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, interpolation="nearest", origin="upper")

    centers = [b[3] for b in blocks]
    names = [NETWORK_DISPLAY[b[0]] for b in blocks]
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(names, rotation=42, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    for _, start, _, _ in blocks:
        ax.axhline(start - 0.5, linewidth=0.8, color="0.25", alpha=0.65)
        ax.axvline(start - 0.5, linewidth=0.8, color="0.25", alpha=0.65)
    ax.axhline(len(coords) - 0.5, linewidth=0.8, color="0.25", alpha=0.65)
    ax.axvline(len(coords) - 0.5, linewidth=0.8, color="0.25", alpha=0.65)

    ax.set_title(
        "Group-mean functional connectivity — Schaefer-100 ordered by Yeo network",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Schaefer-100 parcels (LH then RH within each network)", labelpad=13)
    ax.set_ylabel("Schaefer-100 parcels grouped by functional network")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Pearson r (10 subjects averaged in Fisher-z, then back-transformed)")

    fig.text(
        0.5,
        0.018,
        "Full signed 100×100 matrix; diagonal retained at r = 1. Network boundaries are shown explicitly.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.18, right=0.90, bottom=0.25, top=0.92)
    v1.savefig(fig, out)


def figure_network_matrix(network_df: pd.DataFrame, out: Path) -> None:
    M = network_df.to_numpy(dtype=float)
    vmax = float(np.max(np.abs(M)))
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9.8, 8.8))
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, interpolation="nearest")

    names = [NETWORK_DISPLAY[x] for x in NETWORK_ORDER]
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=42, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9)

    ax.set_title(
        "Mean functional connectivity between canonical networks",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Pearson r across parcel pairs")

    fig.text(
        0.5,
        0.020,
        "Diagonal cells average within-network off-diagonal parcel pairs; self-correlations are excluded.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.22, right=0.88, bottom=0.28, top=0.91)
    v1.savefig(fig, out)


def figure_anatomical_connectome(
    graph: nx.Graph,
    coords: pd.DataFrame,
    atlas_world: np.ndarray,
    out: Path,
) -> None:
    palette = network_palette()
    display_edges = v1.strongest_edges(graph, fraction=0.20)
    weights = np.array([d["weight"] for _, _, d in display_edges], dtype=float)
    lo, hi = float(weights.min()), float(weights.max())
    span = max(hi - lo, 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.7))
    for ax, view in zip(axes, ["Sagittal", "Coronal", "Axial"]):
        add_atlas_outline(ax, atlas_world, view)
        xx, yy, xlabel, ylabel = v1.project_view(coords, view)

        for i, j, d in display_edges:
            scaled = (d["weight"] - lo) / span
            ax.plot(
                [xx[i], xx[j]],
                [yy[i], yy[j]],
                linewidth=0.35 + 1.55 * scaled,
                alpha=0.11 + 0.36 * scaled,
                color="0.33",
                zorder=1,
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
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=palette[n], markeredgecolor="white",
               label=NETWORK_DISPLAY[n])
        for n in NETWORK_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.suptitle(
        "Validated 15%-density functional graph in MNI anatomical projections",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.095,
        f"Strongest {len(display_edges)} of {graph.number_of_edges()} graph edges shown; all selected graph edges are positive. Gray outline = Schaefer cortical support.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.24, top=0.86, wspace=0.22)
    v1.savefig(fig, out)


def figure_network_graph(
    coords: pd.DataFrame,
    summary: pd.DataFrame,
    out: Path,
) -> None:
    palette = network_palette()
    counts = coords["network"].value_counts().to_dict()
    within = {
        row.network_a: int(row.n_edges)
        for row in summary.itertuples(index=False)
        if row.network_a == row.network_b
    }

    between = summary.loc[
        (summary["network_a"] != summary["network_b"]) & (summary["n_edges"] > 0)
    ].copy()
    edge_max = max(float(between["n_edges"].max()), 1.0)

    angles_deg = {
        "Vis": 8,
        "SomMot": 57,
        "DorsAttn": 108,
        "SalVentAttn": 160,
        "Limbic": 214,
        "Cont": 270,
        "Default": 326,
    }
    pos = {
        n: np.array([
            np.cos(np.deg2rad(angles_deg[n])),
            np.sin(np.deg2rad(angles_deg[n])),
        ])
        for n in NETWORK_ORDER
    }

    fig, ax = plt.subplots(figsize=(10.8, 9.5))

    for row in between.itertuples(index=False):
        u, v = row.network_a, row.network_b
        width = 0.7 + 5.0 * float(row.n_edges) / edge_max
        ax.plot(
            [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            linewidth=width, alpha=0.27, color="0.38", zorder=1,
        )
        if int(row.n_edges) >= 12:
            mid = (pos[u] + pos[v]) / 2.0
            ax.text(
                mid[0], mid[1], str(int(row.n_edges)),
                fontsize=8, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.8),
                zorder=3,
            )

    short = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal\nattention",
        "SalVentAttn": "Salience /\nventral attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }

    for network in NETWORK_ORDER:
        x, y = pos[network]
        n_parcels = int(counts.get(network, 0))
        size = 1700 + 65 * n_parcels
        ax.scatter(
            [x], [y], s=size,
            color=[palette[network]], edgecolors="white",
            linewidths=1.6, zorder=2,
        )
        ax.text(
            x, y,
            f"{short[network]}\n{n_parcels} parcels\nwithin: {within.get(network, 0)} edges",
            ha="center", va="center", fontsize=8.6,
            linespacing=1.08, zorder=4,
        )

    ax.set_title(
        "Network-level organization of the validated 15%-density graph",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(-1.38, 1.38)
    ax.set_ylim(-1.30, 1.30)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.text(
        0.5,
        0.030,
        "Edge width = number of between-network parcel edges; labels are shown for pairs with ≥12 edges. Node annotations include within-network edge counts.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.12, top=0.91)
    v1.savefig(fig, out)


def figure_hubs(
    nodes: pd.DataFrame,
    coords: pd.DataFrame,
    atlas_world: np.ndarray,
    out: Path,
) -> None:
    palette = network_palette()
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
    add_atlas_outline(ax2, atlas_world, "Axial")
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
    ax2.set_title("Top-10 hubs in axial projection", fontweight="bold")
    ax2.set_aspect("equal", adjustable="box")

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=palette[n], markeredgecolor="white",
               label=NETWORK_DISPLAY[n])
        for n in NETWORK_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.76, 0.055),
    )
    fig.suptitle(
        "Functional-connectome hub ranking and anatomical distribution",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "Ranks 1–10 are highlighted on the Schaefer cortical support; other parcels are shown in gray.",
        ha="center",
        fontsize=9,
    )
    v1.savefig(fig, out)


def main() -> None:
    args = parse_args()
    repo = args.repo_dir.resolve()
    work = args.work_dir.resolve()
    output = args.output_dir.resolve()
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    matrix, labels, nodes, atlas_path, matrix_diff = v1.load_inputs(repo, work)
    coords = v1.atlas_centroids(atlas_path, labels)
    atlas_world = atlas_world_support(atlas_path)
    network_df = v1.network_mean_matrix(matrix, labels)
    graph = v1.connected_graph(matrix, density=0.15)
    summary = v1.network_edge_summary(graph, coords)

    edge_signs = np.array(
        [d["correlation"] for _, _, d in graph.edges(data=True)], dtype=float
    )
    communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
    modularity = float(nx.community.modularity(graph, communities, weight="weight"))

    if graph.number_of_edges() != 742:
        raise RuntimeError(f"Expected 742 graph edges, got {graph.number_of_edges()}")
    if int(np.sum(edge_signs < 0)) != 0:
        raise RuntimeError("Validated graph unexpectedly contains negative edges.")
    if abs(nx.density(graph) - 0.1498989898989899) > 1e-12:
        raise RuntimeError("Unexpected graph density.")
    if abs(modularity - 0.5114978691679645) > 1e-9:
        raise RuntimeError(f"Unexpected modularity: {modularity}")

    coords.to_csv(
        tables / "schaefer100_node_coordinates.tsv",
        sep="\t", index=False, float_format="%.6f",
    )
    network_df.to_csv(
        tables / "network_mean_connectivity.tsv",
        sep="\t", float_format="%.8f",
    )
    summary.to_csv(
        tables / "network_graph_edge_summary.tsv",
        sep="\t", index=False, float_format="%.8f",
    )

    figure_ordered_matrix(matrix, coords, figures / "connectivity_network_ordered.png")
    figure_network_matrix(network_df, figures / "network_mean_connectivity.png")
    figure_anatomical_connectome(
        graph, coords, atlas_world, figures / "connectome_anatomical_views.png"
    )
    figure_network_graph(coords, summary, figures / "network_level_graph.png")
    figure_hubs(nodes, coords, atlas_world, figures / "hubs_anatomical_summary.png")
    v1.interactive_3d(graph, coords, nodes, output / "connectome_interactive.html")

    validation = {
        "revision": "v2",
        "subjects": 10,
        "nodes": int(matrix.shape[0]),
        "public_tsv_vs_npy_max_abs_diff": matrix_diff,
        "graph_edges": int(graph.number_of_edges()),
        "graph_density": float(nx.density(graph)),
        "graph_negative_edges": int(np.sum(edge_signs < 0)),
        "graph_positive_edges": int(np.sum(edge_signs > 0)),
        "n_communities": int(len(communities)),
        "modularity_weighted": modularity,
        "top_hub": str(nodes.sort_values("hub_rank").iloc[0]["label"]),
        "atlas": str(atlas_path),
        "anatomical_reference": "convex-hull projection of Schaefer cortical atlas support",
        "figures": sorted(p.name for p in figures.glob("*.png")),
        "interactive_html": "connectome_interactive.html",
    }
    (output / "redesign_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    print("CONNECTOME_VISUAL_REDESIGN_V2_PASSED")
    for path in sorted(figures.glob("*.png")):
        print(f"WROTE {path.name}\t{path.stat().st_size} bytes")
    print(
        f"WROTE connectome_interactive.html\t"
        f"{(output / 'connectome_interactive.html').stat().st_size} bytes"
    )
    print(
        f"WROTE redesign_validation.json\t"
        f"{(output / 'redesign_validation.json').stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
