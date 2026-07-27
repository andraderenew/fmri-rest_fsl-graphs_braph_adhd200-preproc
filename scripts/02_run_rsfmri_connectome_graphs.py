#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
from scipy.stats import zscore

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results" / "tables"
FIGURES = REPO / "results" / "figures"
MANIFEST_JSON = TABLES / "dataset_manifest.json"
MANIFEST_TSV = TABLES / "table0_adhd200_subject_manifest.tsv"

WORK = Path(os.environ.get(
    "RSFMRI_WORK_DIR",
    Path.home() / "neuroimaging-data" /
    "fmri-rest_fsl-graphs_braph_adhd200-preproc",
)).expanduser().resolve()
DERIVATIVES = WORK / "derivatives"
TIMESERIES_DIR = DERIVATIVES / "timeseries"
MATRICES_DIR = DERIVATIVES / "connectomes"

TR = 2.0
HIGH_PASS = 0.01
LOW_PASS = 0.10
GRAPH_DENSITY = 0.15
SMALLWORLD_NITER = 5
SMALLWORLD_NRAND = 10
RANDOM_SEED = 42

CONFOUND_COLUMNS = [
    "csf",
    "wm",
    "motion-pitch",
    "motion-roll",
    "motion-yaw",
    "motion-x",
    "motion-y",
    "motion-z",
    "compcor1",
    "compcor2",
    "compcor3",
    "compcor4",
    "compcor5",
]


def clean_label(value: object) -> str:
    text = str(value)
    if (text.startswith("b'") and text.endswith("'")) or (
        text.startswith('b"') and text.endswith('"')
    ):
        try:
            decoded = ast.literal_eval(text)
            if isinstance(decoded, bytes):
                text = decoded.decode("utf-8")
        except (SyntaxError, ValueError):
            pass
    return text.replace("7Networks_", "")


def network_from_label(label: str) -> str:
    parts = label.split("_")
    if len(parts) >= 3:
        return parts[1]
    return "Unknown"


def subject_id_from_path(path: str | Path) -> str:
    path = Path(path)
    candidates = [path.parent.name, path.name]
    for candidate in candidates:
        match = re.search(r"(\d{7})", candidate)
        if match:
            return match.group(1)
    raise ValueError(f"No se pudo extraer el identificador del sujeto: {path}")


def read_confounds(path: str | Path, n_scans: int) -> tuple[pd.DataFrame, list[str]]:
    path = Path(path)
    try:
        frame = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        frame = pd.read_table(path)

    selected = [column for column in CONFOUND_COLUMNS if column in frame.columns]

    if not selected:
        numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
        excluded = {"constant", "linearTrend", "global", "gm"}
        selected = [column for column in numeric if column not in excluded]

    if not selected:
        raise RuntimeError(f"No se encontraron confounds numéricos en {path}")

    confounds = frame[selected].apply(pd.to_numeric, errors="coerce")
    confounds = confounds.interpolate(limit_direction="both").fillna(0.0)

    if len(confounds) < n_scans:
        raise RuntimeError(
            f"{path}: {len(confounds)} filas de confounds para {n_scans} volúmenes."
        )
    if len(confounds) > n_scans:
        confounds = confounds.iloc[:n_scans].copy()

    return confounds, selected


def fisher_z(correlation: np.ndarray) -> np.ndarray:
    clipped = np.clip(correlation, -0.999999, 0.999999)
    transformed = np.arctanh(clipped)
    np.fill_diagonal(transformed, 0.0)
    return transformed


def connected_proportional_graph(
    correlation: np.ndarray,
    density: float,
) -> nx.Graph:
    """Build a connected graph using absolute correlation magnitude.

    A maximum spanning tree guarantees connectivity. Strongest remaining edges
    are added until the requested proportional density is reached.
    """
    n_nodes = correlation.shape[0]
    target_edges = max(
        n_nodes - 1,
        int(round(density * n_nodes * (n_nodes - 1) / 2)),
    )

    complete = nx.Graph()
    complete.add_nodes_from(range(n_nodes))

    upper_i, upper_j = np.triu_indices(n_nodes, k=1)
    edges = []
    for i, j in zip(upper_i, upper_j):
        signed = float(correlation[i, j])
        magnitude = abs(signed)
        edges.append((int(i), int(j), magnitude, signed))
        complete.add_edge(
            int(i),
            int(j),
            weight=magnitude,
            correlation=signed,
            distance=1.0 / max(magnitude, 1e-12),
        )

    graph = nx.maximum_spanning_tree(complete, weight="weight")
    existing = {tuple(sorted(edge)) for edge in graph.edges()}

    for i, j, magnitude, signed in sorted(
        edges, key=lambda item: item[2], reverse=True
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
        raise RuntimeError("El grafo proporcional no quedó conectado.")

    return graph


def graph_global_metrics(graph: nx.Graph, include_sigma: bool) -> dict[str, float]:
    metrics: dict[str, float] = {
        "n_nodes": float(graph.number_of_nodes()),
        "n_edges": float(graph.number_of_edges()),
        "density": float(nx.density(graph)),
        "mean_degree": float(
            np.mean([degree for _, degree in graph.degree()])
        ),
        "mean_strength": float(
            np.mean([degree for _, degree in graph.degree(weight="weight")])
        ),
        "average_clustering_unweighted": float(nx.average_clustering(graph)),
        "average_clustering_weighted": float(
            nx.average_clustering(graph, weight="weight")
        ),
        "transitivity": float(nx.transitivity(graph)),
        "global_efficiency_unweighted": float(nx.global_efficiency(graph)),
        "characteristic_path_length_unweighted": float(
            nx.average_shortest_path_length(graph)
        ),
        "characteristic_path_length_weighted": float(
            nx.average_shortest_path_length(graph, weight="distance")
        ),
        "assortativity_degree": float(
            nx.degree_assortativity_coefficient(graph)
        ),
    }

    communities = list(
        nx.community.greedy_modularity_communities(graph, weight="weight")
    )
    metrics["n_communities"] = float(len(communities))
    metrics["modularity_weighted"] = float(
        nx.community.modularity(graph, communities, weight="weight")
    )

    if include_sigma:
        metrics["small_world_sigma"] = float(
            nx.sigma(
                nx.Graph(graph),
                niter=SMALLWORLD_NITER,
                nrand=SMALLWORLD_NRAND,
                seed=RANDOM_SEED,
            )
        )

    return metrics


def node_metrics(graph: nx.Graph, labels: list[str]) -> pd.DataFrame:
    degree = dict(graph.degree())
    strength = dict(graph.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(
        graph,
        normalized=True,
        weight="distance",
    )
    clustering = nx.clustering(graph, weight="weight")
    eigenvector = nx.eigenvector_centrality_numpy(graph, weight="weight")

    communities = list(
        nx.community.greedy_modularity_communities(graph, weight="weight")
    )
    community_id = {}
    for index, community in enumerate(communities, start=1):
        for node in community:
            community_id[node] = index

    frame = pd.DataFrame(
        {
            "node": np.arange(len(labels), dtype=int),
            "label": labels,
            "network": [network_from_label(label) for label in labels],
            "degree": [degree[i] for i in range(len(labels))],
            "strength": [strength[i] for i in range(len(labels))],
            "betweenness": [betweenness[i] for i in range(len(labels))],
            "weighted_clustering": [clustering[i] for i in range(len(labels))],
            "eigenvector_centrality": [
                eigenvector[i] for i in range(len(labels))
            ],
            "community": [community_id[i] for i in range(len(labels))],
        }
    )

    centrality_columns = [
        "degree",
        "strength",
        "betweenness",
        "eigenvector_centrality",
    ]
    standardized = frame[centrality_columns].apply(
        lambda column: zscore(column.to_numpy(), ddof=0)
    )
    frame["hub_score"] = standardized.mean(axis=1)
    frame = frame.sort_values("hub_score", ascending=False).reset_index(drop=True)
    frame["hub_rank"] = np.arange(1, len(frame) + 1)
    return frame


def save_matrix_tsv(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_path,
        sep="\t",
        float_format="%.8f",
    )


def make_figures(
    group_correlation: np.ndarray,
    labels: list[str],
    atlas_maps: str,
    nodes: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(group_correlation, vmin=-1.0, vmax=1.0)
    axis.set_title("Group-mean functional connectivity — Schaefer 100")
    tick_positions = np.arange(0, len(labels), 10)
    axis.set_xticks(tick_positions)
    axis.set_yticks(tick_positions)
    axis.set_xticklabels([labels[i] for i in tick_positions], rotation=90, fontsize=6)
    axis.set_yticklabels([labels[i] for i in tick_positions], fontsize=6)
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    figure.savefig(FIGURES / "fig1_group_mean_connectome.png", dpi=220)
    plt.close(figure)

    coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas_maps)
    display = plotting.plot_connectome(
        group_correlation,
        coordinates,
        edge_threshold="85%",
        node_size=18,
        title="Group-mean functional connectome — strongest 15% displayed",
    )
    display.savefig(FIGURES / "fig2_group_connectome_brain.png", dpi=220)
    display.close()

    top = nodes.head(15).sort_values("hub_score", ascending=True)
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(top["label"], top["hub_score"])
    axis.set_xlabel("Composite hub score (z)")
    axis.set_title("Top 15 functional-connectome hubs")
    figure.tight_layout()
    figure.savefig(FIGURES / "fig3_top_hubs.png", dpi=220)
    plt.close(figure)


def main() -> None:
    for directory in [
        TABLES,
        FIGURES,
        TIMESERIES_DIR,
        MATRICES_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_JSON.exists() or not MANIFEST_TSV.exists():
        raise FileNotFoundError(
            "Faltan los manifiestos de descarga en results/tables."
        )

    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    raw_table = pd.read_csv(MANIFEST_TSV, sep="\t")

    functional_files = [Path(path) for path in manifest["functional_files"]]
    confound_files = [Path(path) for path in manifest["confounds_files"]]
    atlas_maps = str(manifest["atlas_maps"])
    labels = [clean_label(label) for label in manifest["atlas_labels"]]
    expected_rois = int(manifest.get("atlas_n_rois", len(labels)))
    if len(labels) == expected_rois + 1:
        labels = labels[1:]
    expected_rois = int(manifest.get("atlas_n_rois", len(labels)))
    if len(labels) == expected_rois + 1:
        labels = labels[1:]

    if len(functional_files) != len(confound_files):
        raise RuntimeError("El número de imágenes y confounds no coincide.")

    pheno = raw_table.drop(
        columns=["func_file", "confounds_file"],
        errors="ignore",
    ).copy()
    if "Subject" in pheno.columns:
        pheno["_match_id"] = (
            pheno["Subject"].astype(str).str.extract(r"(\d{7})", expand=False)
        )
    else:
        pheno["_match_id"] = np.nan

    corrected_rows = []
    correlations = []
    fisher_matrices = []
    subject_qc_rows = []
    subject_graph_rows = []
    selected_confound_union: set[str] = set()

    masker = NiftiLabelsMasker(
        labels_img=atlas_maps,
        standardize="zscore_sample",
        standardize_confounds=True,
        detrend=True,
        high_pass=HIGH_PASS,
        low_pass=LOW_PASS,
        t_r=TR,
        resampling_target="data",
        memory=str(WORK / "work" / "nilearn_cache"),
        memory_level=1,
        verbose=1,
    )
    connectivity = ConnectivityMeasure(
        kind="correlation",
        standardize="zscore_sample",
    )

    print(f"Subjects: {len(functional_files)}")
    print(f"Atlas regions: {len(labels)}")
    print(f"Band-pass: {HIGH_PASS}–{LOW_PASS} Hz")
    print(f"Graph density: {GRAPH_DENSITY:.0%}")

    for index, (func_file, confound_file) in enumerate(
        zip(functional_files, confound_files),
        start=1,
    ):
        subject_id = subject_id_from_path(func_file)
        image = nib.load(str(func_file))
        n_scans = int(image.shape[-1])

        confounds, selected_columns = read_confounds(
            confound_file,
            n_scans=n_scans,
        )
        selected_confound_union.update(selected_columns)

        print(
            f"\n[{index}/{len(functional_files)}] "
            f"sub-{subject_id}: {n_scans} scans, "
            f"{len(selected_columns)} confounds"
        )

        time_series = masker.fit_transform(
            str(func_file),
            confounds=confounds,
        )

        if time_series.shape[1] != len(labels):
            raise RuntimeError(
                f"sub-{subject_id}: {time_series.shape[1]} señales ROI, "
                f"pero {len(labels)} etiquetas."
            )

        correlation = connectivity.fit_transform([time_series])[0]
        correlation = np.asarray(correlation, dtype=float)
        np.fill_diagonal(correlation, 1.0)
        z_matrix = fisher_z(correlation)

        correlations.append(correlation)
        fisher_matrices.append(z_matrix)

        np.save(
            MATRICES_DIR / f"sub-{subject_id}_correlation.npy",
            correlation,
        )
        save_matrix_tsv(
            correlation,
            labels,
            MATRICES_DIR / f"sub-{subject_id}_correlation.tsv",
        )
        pd.DataFrame(time_series, columns=labels).to_csv(
            TIMESERIES_DIR / f"sub-{subject_id}_timeseries.tsv",
            sep="\t",
            index=False,
            float_format="%.8f",
        )

        graph = connected_proportional_graph(
            correlation,
            density=GRAPH_DENSITY,
        )
        subject_metrics = graph_global_metrics(graph, include_sigma=False)
        subject_metrics["subject_id"] = subject_id
        subject_graph_rows.append(subject_metrics)

        upper = correlation[np.triu_indices_from(correlation, k=1)]
        subject_qc_rows.append(
            {
                "subject_id": subject_id,
                "n_scans": n_scans,
                "n_rois": int(time_series.shape[1]),
                "n_confounds": int(confounds.shape[1]),
                "mean_correlation": float(np.mean(upper)),
                "mean_absolute_correlation": float(np.mean(np.abs(upper))),
                "timeseries_mean": float(np.mean(time_series)),
                "timeseries_sd": float(np.std(time_series, ddof=1)),
            }
        )

        matching = pheno.loc[pheno["_match_id"] == subject_id].copy()
        base_row = {
            "subject_id": subject_id,
            "func_file": str(func_file),
            "confounds_file": str(confound_file),
        }
        if len(matching) == 1:
            for column, value in matching.drop(columns=["_match_id"]).iloc[0].items():
                base_row[column] = value
        corrected_rows.append(base_row)

    fisher_stack = np.stack(fisher_matrices, axis=0)
    group_z = np.mean(fisher_stack, axis=0)
    group_correlation = np.tanh(group_z)
    np.fill_diagonal(group_correlation, 1.0)

    np.save(MATRICES_DIR / "group_mean_correlation.npy", group_correlation)
    save_matrix_tsv(
        group_correlation,
        labels,
        TABLES / "table1_group_mean_connectome.tsv",
    )

    group_graph = connected_proportional_graph(
        group_correlation,
        density=GRAPH_DENSITY,
    )
    global_metrics = graph_global_metrics(group_graph, include_sigma=True)
    global_frame = pd.DataFrame(
        {
            "metric": list(global_metrics.keys()),
            "value": list(global_metrics.values()),
        }
    )

    nodes = node_metrics(group_graph, labels)
    hubs = nodes.head(10).copy()

    pd.DataFrame(corrected_rows).to_csv(
        TABLES / "table0b_adhd200_subject_manifest_corrected.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(subject_qc_rows).to_csv(
        TABLES / "table2_subject_qc.tsv",
        sep="\t",
        index=False,
        float_format="%.8f",
    )
    subject_graph_frame = pd.DataFrame(subject_graph_rows)
    ordered = ["subject_id"] + [
        column for column in subject_graph_frame.columns
        if column != "subject_id"
    ]
    subject_graph_frame[ordered].to_csv(
        TABLES / "table3_subject_graph_metrics.tsv",
        sep="\t",
        index=False,
        float_format="%.8f",
    )
    global_frame.to_csv(
        TABLES / "table4_group_global_graph_metrics.tsv",
        sep="\t",
        index=False,
        float_format="%.8f",
    )
    nodes.to_csv(
        TABLES / "table5_group_node_metrics.tsv",
        sep="\t",
        index=False,
        float_format="%.8f",
    )
    hubs.to_csv(
        TABLES / "table6_top_hubs.tsv",
        sep="\t",
        index=False,
        float_format="%.8f",
    )

    parameters = {
        "subjects": len(functional_files),
        "atlas": "Schaefer 2018, 100 parcels, 7 networks, 2 mm",
        "tr_seconds": TR,
        "high_pass_hz": HIGH_PASS,
        "low_pass_hz": LOW_PASS,
        "signal_standardization": "zscore_sample",
        "detrend": True,
        "global_signal_regression": False,
        "selected_confounds": sorted(selected_confound_union),
        "connectivity": "Pearson correlation",
        "group_average": "mean in Fisher-z space, transformed back to r",
        "graph_weight": "absolute correlation magnitude",
        "graph_density": GRAPH_DENSITY,
        "graph_connectivity_rule": (
            "maximum spanning tree plus strongest remaining edges"
        ),
        "small_world_sigma_niter": SMALLWORLD_NITER,
        "small_world_sigma_nrand": SMALLWORLD_NRAND,
        "random_seed": RANDOM_SEED,
    }
    (TABLES / "analysis_parameters.json").write_text(
        json.dumps(parameters, indent=2),
        encoding="utf-8",
    )

    make_figures(
        group_correlation=group_correlation,
        labels=labels,
        atlas_maps=atlas_maps,
        nodes=nodes,
    )

    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Subjects processed: {len(functional_files)}")
    print(f"Graph edges: {group_graph.number_of_edges()}")
    print(f"Graph density: {nx.density(group_graph):.6f}")
    print(f"Small-world sigma: {global_metrics['small_world_sigma']:.6f}")
    print("\nTop 10 hubs:")
    print(
        hubs[
            [
                "hub_rank",
                "label",
                "network",
                "degree",
                "strength",
                "betweenness",
                "hub_score",
            ]
        ].to_string(index=False)
    )
    print(f"\nTables: {TABLES}")
    print(f"Figures: {FIGURES}")
    print(f"Large derivatives: {DERIVATIVES}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
