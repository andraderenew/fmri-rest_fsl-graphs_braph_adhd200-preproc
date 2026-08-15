#!/usr/bin/env python3
"""Redesign resting-state functional connectome visualizations from validated derivatives.

This script does not recompute subject-level fMRI preprocessing or connectivity.
It reads the validated group-mean correlation matrix, public node metrics, and the
Schaefer-100 atlas to generate network-ordered and anatomically grounded figures.
"""

from __future__ import annotations

import argparse
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
import plotly.graph_objects as go


NETWORK_ORDER = [
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
]

NETWORK_DISPLAY = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal attention",
    "SalVentAttn": "Salience / ventral attention",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--work-dir",
        required=True,
        type=Path,
        help="Local work directory containing derivatives/connectomes and atlas data.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Preview output directory. Repository results are not modified.",
    )
    p.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return p.parse_args()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def network_from_label(label: str) -> str:
    parts = label.split("_")
    return parts[1] if len(parts) >= 3 else "Unknown"


def hemi_from_label(label: str) -> str:
    return label.split("_")[0]


def network_palette() -> dict[str, tuple[float, float, float, float]]:
    cmap = matplotlib.colormaps["tab10"]
    return {network: cmap(i) for i, network in enumerate(NETWORK_ORDER)}


def load_inputs(repo: Path, work: Path):
    public_matrix = pd.read_csv(
        require(repo / "results/tables/table1_group_mean_connectome.tsv"),
        sep="\t",
        index_col=0,
    )
    labels = list(public_matrix.index)
    if labels != list(public_matrix.columns):
        raise RuntimeError("Public matrix row/column labels do not match.")

    group_path = require(work / "derivatives/connectomes/group_mean_correlation.npy")
    matrix = np.load(group_path).astype(float)
    if matrix.shape != (len(labels), len(labels)):
        raise RuntimeError(
            f"Group matrix shape {matrix.shape} does not match {len(labels)} labels."
        )

    public = public_matrix.to_numpy(dtype=float)
    max_diff = float(np.max(np.abs(matrix - public)))
    if max_diff > 1e-6:
        raise RuntimeError(f"Public TSV differs from validated NPY: max diff={max_diff}")

    nodes = pd.read_csv(
        require(repo / "results/tables/table5_group_node_metrics.tsv"),
        sep="\t",
    )
    if list(nodes.sort_values("node")["label"]) != labels:
        raise RuntimeError("Node-metric labels do not match connectivity-matrix order.")

    manifest = json.loads(
        require(repo / "results/tables/dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    atlas_path = require(Path(manifest["atlas_maps"]))
    return matrix, labels, nodes, atlas_path, max_diff


def atlas_centroids(atlas_path: Path, labels: list[str]) -> pd.DataFrame:
    img = nib.load(str(atlas_path))
    data = np.asanyarray(img.dataobj)

    rows = []
    for idx, label in enumerate(labels, start=1):
        vox = np.argwhere(data == idx)
        if vox.size == 0:
            raise RuntimeError(f"Atlas label {idx} has no voxels: {label}")
        center_vox = vox.mean(axis=0)
        x, y, z = apply_affine(img.affine, center_vox)
        rows.append(
            {
                "node": idx - 1,
                "atlas_value": idx,
                "label": label,
                "hemisphere": hemi_from_label(label),
                "network": network_from_label(label),
                "x_mni": float(x),
                "y_mni": float(y),
                "z_mni": float(z),
                "n_voxels": int(vox.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def ordered_metadata(coords: pd.DataFrame) -> pd.DataFrame:
    order_map = {name: i for i, name in enumerate(NETWORK_ORDER)}
    hemi_map = {"LH": 0, "RH": 1}
    out = coords.copy()
    out["_network_order"] = out["network"].map(order_map)
    out["_hemi_order"] = out["hemisphere"].map(hemi_map).fillna(9)
    out = out.sort_values(
        ["_network_order", "_hemi_order", "node"],
        kind="stable",
    ).reset_index(drop=True)
    return out


def network_blocks(ordered: pd.DataFrame):
    blocks = []
    start = 0
    for network in NETWORK_ORDER:
        count = int((ordered["network"] == network).sum())
        stop = start + count
        blocks.append((network, start, stop, (start + stop - 1) / 2.0))
        start = stop
    return blocks


def network_mean_matrix(
    matrix: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    meta = pd.DataFrame(
        {
            "node": np.arange(len(labels)),
            "network": [network_from_label(x) for x in labels],
        }
    )
    out = np.zeros((len(NETWORK_ORDER), len(NETWORK_ORDER)), dtype=float)

    for a, net_a in enumerate(NETWORK_ORDER):
        idx_a = meta.loc[meta["network"] == net_a, "node"].to_numpy(dtype=int)
        for b, net_b in enumerate(NETWORK_ORDER):
            idx_b = meta.loc[meta["network"] == net_b, "node"].to_numpy(dtype=int)
            block = matrix[np.ix_(idx_a, idx_b)]
            if a == b:
                mask = ~np.eye(len(idx_a), dtype=bool)
                vals = block[mask]
            else:
                vals = block.ravel()
            out[a, b] = float(np.mean(vals))

    return pd.DataFrame(out, index=NETWORK_ORDER, columns=NETWORK_ORDER)


def connected_graph(matrix: np.ndarray, density: float = 0.15) -> nx.Graph:
    n_nodes = matrix.shape[0]
    target_edges = max(
        n_nodes - 1,
        int(round(density * n_nodes * (n_nodes - 1) / 2)),
    )

    complete = nx.Graph()
    complete.add_nodes_from(range(n_nodes))
    candidates = []

    ii, jj = np.triu_indices(n_nodes, k=1)
    for i, j in zip(ii, jj):
        signed = float(matrix[i, j])
        magnitude = abs(signed)
        complete.add_edge(
            int(i),
            int(j),
            weight=magnitude,
            correlation=signed,
            distance=1.0 / max(magnitude, 1e-12),
        )
        candidates.append((int(i), int(j), magnitude, signed))

    graph = nx.maximum_spanning_tree(complete, weight="weight")
    existing = {tuple(sorted(edge)) for edge in graph.edges()}

    for i, j, magnitude, signed in sorted(
        candidates,
        key=lambda item: item[2],
        reverse=True,
    ):
        if graph.number_of_edges() >= target_edges:
            break
        key = tuple(sorted((i, j)))
        if key in existing:
            continue
        graph.add_edge(
            i,
            j,
            weight=magnitude,
            correlation=signed,
            distance=1.0 / max(magnitude, 1e-12),
        )
        existing.add(key)

    if not nx.is_connected(graph):
        raise RuntimeError("Rebuilt 15%-density graph is not connected.")
    return graph


def savefig(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.18,
        facecolor="white",
    )
    plt.close(fig)
    if not path.exists() or path.stat().st_size < 20_000:
        raise RuntimeError(f"Figure validation failed: {path}")


def figure_ordered_matrix(
    matrix: np.ndarray,
    coords: pd.DataFrame,
    out: Path,
) -> None:
    ordered = ordered_metadata(coords)
    idx = ordered["node"].to_numpy(dtype=int)
    M = matrix[np.ix_(idx, idx)]
    blocks = network_blocks(ordered)

    offdiag = M[~np.eye(M.shape[0], dtype=bool)]
    vmax = float(np.percentile(np.abs(offdiag), 99.5))
    vmax = max(vmax, 0.1)
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(11.4, 10.2))
    im = ax.imshow(
        M,
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
        origin="upper",
    )

    centers = [b[3] for b in blocks]
    names = [NETWORK_DISPLAY[b[0]] for b in blocks]
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    for _, start, _, _ in blocks:
        ax.axhline(start - 0.5, linewidth=0.8)
        ax.axvline(start - 0.5, linewidth=0.8)
    ax.axhline(len(coords) - 0.5, linewidth=0.8)
    ax.axvline(len(coords) - 0.5, linewidth=0.8)

    ax.set_title(
        "Group-mean functional connectivity — Schaefer-100 ordered by Yeo network",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Parcels grouped by functional network (LH then RH within network)")
    ax.set_ylabel("Parcels grouped by functional network")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Pearson correlation (group mean in Fisher-z space)")

    fig.text(
        0.5,
        0.015,
        "Full signed 100×100 matrix; diagonal retained at r = 1. Network boundaries shown explicitly.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.17)
    savefig(fig, out)


def figure_network_matrix(network_df: pd.DataFrame, out: Path) -> None:
    M = network_df.to_numpy()
    vmax = float(np.max(np.abs(M)))
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9.6, 8.2))
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, interpolation="nearest")

    names = [NETWORK_DISPLAY[x] for x in NETWORK_ORDER]
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(
                j,
                i,
                f"{M[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    ax.set_title(
        "Mean functional connectivity between canonical networks",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Pearson r")

    fig.text(
        0.5,
        0.015,
        "Diagonal cells summarize within-network off-diagonal correlations; self-correlations are excluded.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.18)
    savefig(fig, out)


def strongest_edges(graph: nx.Graph, fraction: float = 0.20):
    edges = sorted(
        graph.edges(data=True),
        key=lambda e: e[2]["weight"],
        reverse=True,
    )
    n_keep = max(1, int(round(fraction * len(edges))))
    return edges[:n_keep]


def project_view(coords: pd.DataFrame, view: str):
    if view == "Axial":
        return coords["x_mni"].to_numpy(), coords["y_mni"].to_numpy(), "MNI x (mm)", "MNI y (mm)"
    if view == "Coronal":
        return coords["x_mni"].to_numpy(), coords["z_mni"].to_numpy(), "MNI x (mm)", "MNI z (mm)"
    return coords["y_mni"].to_numpy(), coords["z_mni"].to_numpy(), "MNI y (mm)", "MNI z (mm)"


def figure_anatomical_connectome(
    graph: nx.Graph,
    coords: pd.DataFrame,
    out: Path,
) -> None:
    palette = network_palette()
    display_edges = strongest_edges(graph, fraction=0.20)
    weights = np.array([d["weight"] for _, _, d in display_edges], dtype=float)
    lo, hi = float(weights.min()), float(weights.max())
    span = max(hi - lo, 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.5))
    for ax, view in zip(axes, ["Sagittal", "Coronal", "Axial"]):
        xx, yy, xlabel, ylabel = project_view(coords, view)
        for i, j, d in display_edges:
            scaled = (d["weight"] - lo) / span
            ax.plot(
                [xx[i], xx[j]],
                [yy[i], yy[j]],
                linewidth=0.35 + 1.65 * scaled,
                alpha=0.10 + 0.38 * scaled,
                color="0.35",
                zorder=1,
            )

        for network in NETWORK_ORDER:
            idx = coords.index[coords["network"] == network].to_numpy(dtype=int)
            ax.scatter(
                xx[idx],
                yy[idx],
                s=20,
                color=[palette[network]],
                edgecolors="white",
                linewidths=0.25,
                label=NETWORK_DISPLAY[network],
                zorder=2,
            )

        ax.axhline(0, linewidth=0.5, alpha=0.35)
        ax.axvline(0, linewidth=0.5, alpha=0.35)
        ax.set_title(view, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="datalim")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=palette[n],
            markeredgecolor="white",
            label=NETWORK_DISPLAY[n],
        )
        for n in NETWORK_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Anatomical projection of the validated 15%-density functional graph",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.045,
        f"Showing the strongest {len(display_edges)} of {graph.number_of_edges()} graph edges for readability; all displayed graph edges are positive.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.20, top=0.86, wspace=0.25)
    savefig(fig, out)


def network_edge_summary(
    graph: nx.Graph,
    coords: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for i, net_a in enumerate(NETWORK_ORDER):
        nodes_a = set(coords.loc[coords["network"] == net_a, "node"].astype(int))
        for j, net_b in enumerate(NETWORK_ORDER[i:], start=i):
            nodes_b = set(coords.loc[coords["network"] == net_b, "node"].astype(int))
            values = []
            for u, v, d in graph.edges(data=True):
                if net_a == net_b:
                    if u in nodes_a and v in nodes_a:
                        values.append(float(d["correlation"]))
                else:
                    if (u in nodes_a and v in nodes_b) or (u in nodes_b and v in nodes_a):
                        values.append(float(d["correlation"]))
            records.append(
                {
                    "network_a": net_a,
                    "network_b": net_b,
                    "n_edges": len(values),
                    "mean_r": float(np.mean(values)) if values else np.nan,
                    "mean_abs_r": float(np.mean(np.abs(values))) if values else np.nan,
                }
            )
    return pd.DataFrame(records)


def figure_network_graph(
    graph: nx.Graph,
    coords: pd.DataFrame,
    summary: pd.DataFrame,
    out: Path,
) -> None:
    palette = network_palette()
    G = nx.Graph()
    counts = coords["network"].value_counts().to_dict()

    for network in NETWORK_ORDER:
        G.add_node(network, n_parcels=int(counts.get(network, 0)))

    for row in summary.itertuples(index=False):
        if row.n_edges > 0 and row.network_a != row.network_b:
            G.add_edge(
                row.network_a,
                row.network_b,
                n_edges=int(row.n_edges),
                mean_r=float(row.mean_r),
            )

    pos = nx.circular_layout(G)
    edge_counts = np.array([d["n_edges"] for _, _, d in G.edges(data=True)], dtype=float)
    edge_max = max(float(edge_counts.max()) if edge_counts.size else 1.0, 1.0)

    fig, ax = plt.subplots(figsize=(9.8, 9.0))
    for u, v, d in G.edges(data=True):
        width = 0.7 + 5.0 * d["n_edges"] / edge_max
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            linewidth=width,
            alpha=0.28,
            color="0.40",
            zorder=1,
        )
        mid = (pos[u] + pos[v]) / 2
        ax.text(
            mid[0],
            mid[1],
            f"{d['n_edges']}",
            fontsize=8,
            ha="center",
            va="center",
            zorder=3,
        )

    for network in NETWORK_ORDER:
        x, y = pos[network]
        size = 650 + 95 * G.nodes[network]["n_parcels"]
        ax.scatter(
            [x],
            [y],
            s=size,
            color=[palette[network]],
            edgecolors="white",
            linewidths=1.4,
            zorder=2,
        )
        ax.text(
            x,
            y,
            f"{NETWORK_DISPLAY[network]}\n{G.nodes[network]['n_parcels']} parcels",
            ha="center",
            va="center",
            fontsize=9,
            zorder=4,
        )

    ax.set_title(
        "Network-level organization of the validated functional graph",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.text(
        0.5,
        0.03,
        "Edge width and label represent the number of parcel-level graph edges connecting each network pair.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    ax.set_axis_off()
    savefig(fig, out)


def figure_hubs(
    nodes: pd.DataFrame,
    coords: pd.DataFrame,
    out: Path,
) -> None:
    palette = network_palette()
    merged = nodes.merge(
        coords[["node", "x_mni", "y_mni", "z_mni"]],
        on="node",
        how="left",
        validate="one_to_one",
    )

    top = merged.sort_values("hub_rank").head(15).copy()
    top_bar = top.sort_values("hub_score", ascending=True)

    fig = plt.figure(figsize=(14.8, 8.6))
    ax1 = fig.add_axes([0.07, 0.12, 0.45, 0.76])
    bar_colors = [palette[n] for n in top_bar["network"]]
    ax1.barh(top_bar["label"], top_bar["hub_score"], color=bar_colors)
    ax1.set_xlabel("Composite hub score (z)")
    ax1.set_title("Top 15 hubs", fontweight="bold")
    ax1.tick_params(axis="y", labelsize=8)

    ax2 = fig.add_axes([0.59, 0.16, 0.37, 0.68])
    sizes = 45 + 125 * (
        (merged["hub_score"] - merged["hub_score"].min())
        / max(merged["hub_score"].max() - merged["hub_score"].min(), 1e-12)
    )
    for network in NETWORK_ORDER:
        m = merged["network"] == network
        ax2.scatter(
            merged.loc[m, "x_mni"],
            merged.loc[m, "y_mni"],
            s=sizes[m],
            color=[palette[network]],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.35,
        )

    for row in top.head(10).itertuples(index=False):
        ax2.annotate(
            str(row.hub_rank),
            (row.x_mni, row.y_mni),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    ax2.set_xlabel("MNI x (mm)")
    ax2.set_ylabel("MNI y (mm)")
    ax2.set_title("Hub-score distribution in axial projection", fontweight="bold")
    ax2.axhline(0, linewidth=0.5, alpha=0.35)
    ax2.axvline(0, linewidth=0.5, alpha=0.35)
    ax2.set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        "Functional-connectome hub ranking and anatomical distribution",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.035,
        "Numbers 1–10 identify the ten highest composite hub scores; node size scales with hub score.",
        ha="center",
        fontsize=9,
    )
    savefig(fig, out)


def interactive_3d(
    graph: nx.Graph,
    coords: pd.DataFrame,
    nodes: pd.DataFrame,
    out: Path,
) -> None:
    palette = network_palette()
    display_edges = strongest_edges(graph, fraction=0.20)

    edge_x, edge_y, edge_z = [], [], []
    for i, j, _ in display_edges:
        a = coords.loc[i]
        b = coords.loc[j]
        edge_x += [a.x_mni, b.x_mni, None]
        edge_y += [a.y_mni, b.y_mni, None]
        edge_z += [a.z_mni, b.z_mni, None]

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(width=1.2, color="rgba(80,80,80,0.28)"),
        hoverinfo="none",
        name="Strongest graph edges",
    )

    merged = coords.merge(
        nodes[["node", "hub_rank", "hub_score", "degree", "strength"]],
        on="node",
        how="left",
        validate="one_to_one",
    )

    traces = [edge_trace]
    for network in NETWORK_ORDER:
        m = merged["network"] == network
        sub = merged.loc[m]
        rgba = palette[network]
        rgb = f"rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})"
        traces.append(
            go.Scatter3d(
                x=sub["x_mni"],
                y=sub["y_mni"],
                z=sub["z_mni"],
                mode="markers",
                marker=dict(
                    size=5 + 5 * np.clip(sub["hub_score"], 0, None),
                    color=rgb,
                    opacity=0.90,
                ),
                text=[
                    (
                        f"{row.label}<br>{NETWORK_DISPLAY[row.network]}"
                        f"<br>hub rank: {int(row.hub_rank)}"
                        f"<br>degree: {int(row.degree)}"
                        f"<br>strength: {row.strength:.2f}"
                    )
                    for row in sub.itertuples(index=False)
                ],
                hoverinfo="text",
                name=NETWORK_DISPLAY[network],
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=(
            "Interactive Schaefer-100 functional graph "
            "— strongest 20% of validated graph edges displayed"
        ),
        scene=dict(
            xaxis_title="MNI x",
            yaxis_title="MNI y",
            zaxis_title="MNI z",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)


def main() -> None:
    args = parse_args()
    repo = args.repo_dir.resolve()
    work = args.work_dir.resolve()
    output = args.output_dir.resolve()
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    matrix, labels, nodes, atlas_path, matrix_diff = load_inputs(repo, work)
    coords = atlas_centroids(atlas_path, labels)
    network_df = network_mean_matrix(matrix, labels)
    graph = connected_graph(matrix, density=0.15)

    edge_signs = np.array(
        [d["correlation"] for _, _, d in graph.edges(data=True)],
        dtype=float,
    )
    if int(np.sum(edge_signs < 0)) != 0:
        raise RuntimeError("Validated graph unexpectedly contains negative edges.")

    communities = list(
        nx.community.greedy_modularity_communities(graph, weight="weight")
    )
    modularity = float(
        nx.community.modularity(graph, communities, weight="weight")
    )

    if graph.number_of_edges() != 742:
        raise RuntimeError(f"Expected 742 graph edges, got {graph.number_of_edges()}")
    if abs(nx.density(graph) - 0.1498989898989899) > 1e-12:
        raise RuntimeError("Unexpected graph density.")
    if abs(modularity - 0.5114978691679645) > 1e-9:
        raise RuntimeError(f"Unexpected modularity: {modularity}")

    coords.to_csv(
        tables / "schaefer100_node_coordinates.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )
    network_df.to_csv(
        tables / "network_mean_connectivity.tsv",
        sep="\t",
        float_format="%.8f",
    )
    summary = network_edge_summary(graph, coords)
    summary.to_csv(
        tables / "network_graph_edge_summary.tsv",
        sep="\t",
        index=False,
        float_format="%.8f",
    )

    figure_ordered_matrix(
        matrix,
        coords,
        figures / "connectivity_network_ordered.png",
    )
    figure_network_matrix(
        network_df,
        figures / "network_mean_connectivity.png",
    )
    figure_anatomical_connectome(
        graph,
        coords,
        figures / "connectome_anatomical_views.png",
    )
    figure_network_graph(
        graph,
        coords,
        summary,
        figures / "network_level_graph.png",
    )
    figure_hubs(
        nodes,
        coords,
        figures / "hubs_anatomical_summary.png",
    )
    interactive_3d(
        graph,
        coords,
        nodes,
        output / "connectome_interactive.html",
    )

    validation = {
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
        "figures": sorted(p.name for p in figures.glob("*.png")),
        "interactive_html": "connectome_interactive.html",
    }
    (output / "redesign_validation.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )

    print("CONNECTOME_VISUAL_REDESIGN_PASSED")
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
