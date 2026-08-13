#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import nibabel as nib
from nibabel.affines import apply_affine
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "03_redesign_connectome_figures.py"
_spec = importlib.util.spec_from_file_location("connectome_v1", V1_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load {V1_PATH}")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

NETWORK_ORDER = v1.NETWORK_ORDER
NETWORK_DISPLAY = v1.NETWORK_DISPLAY
NETWORK_COLORS = {
    "Vis": "#6A3D9A",
    "SomMot": "#1F78B4",
    "DorsAttn": "#33A02C",
    "SalVentAttn": "#E31A1C",
    "Limbic": "#B15928",
    "Cont": "#FF7F00",
    "Default": "#E377C2",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--repo-dir", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--mesh-step", type=int, default=2)
    return p.parse_args()


def hemisphere_mesh(atlas_path: Path, lo: int, hi: int, step: int):
    img = nib.load(str(atlas_path))
    data = np.asanyarray(img.dataobj)
    mask = ((data >= lo) & (data <= hi)).astype(np.float32)
    smooth = gaussian_filter(mask, sigma=0.75)
    verts, faces, _, _ = marching_cubes(
        smooth,
        level=0.45,
        step_size=max(1, int(step)),
        allow_degenerate=False,
    )
    return apply_affine(img.affine, verts), faces.astype(np.int32)


def mesh_trace(world, faces, color: str):
    return go.Mesh3d(
        x=world[:, 0], y=world[:, 1], z=world[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=0.13, flatshading=False,
        hoverinfo="skip", showlegend=False,
        lighting=dict(ambient=0.72, diffuse=0.58, specular=0.12, roughness=0.92),
        lightposition=dict(x=90, y=-120, z=180),
    )


def edge_trace(edges, coords, width: float, rgba: str, visible: bool):
    xs, ys, zs = [], [], []
    for i, j, _ in edges:
        a = coords.loc[int(i)]
        b = coords.loc[int(j)]
        xs.extend([a.x_mni, b.x_mni, None])
        ys.extend([a.y_mni, b.y_mni, None])
        zs.extend([a.z_mni, b.z_mni, None])
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=rgba, width=width),
        hoverinfo="skip", visible=visible, showlegend=False,
    )


def node_traces(coords, nodes):
    merged = coords.merge(
        nodes[["node", "degree", "strength", "betweenness", "hub_score", "hub_rank", "community"]],
        on="node", how="left", validate="one_to_one",
    )
    hub = merged["hub_score"].to_numpy(dtype=float)
    scaled = (hub - hub.min()) / max(hub.max() - hub.min(), 1e-12)
    merged["marker_size"] = 4.8 + 5.8 * scaled
    traces = []
    for network in NETWORK_ORDER:
        sub = merged.loc[merged["network"] == network].copy()
        hover = [
            f"<b>{r.label}</b><br>{NETWORK_DISPLAY[r.network]}"
            f"<br>MNI: ({r.x_mni:.1f}, {r.y_mni:.1f}, {r.z_mni:.1f})"
            f"<br>Hub rank: {int(r.hub_rank)}<br>Hub score: {r.hub_score:.3f}"
            f"<br>Degree: {int(r.degree)}<br>Strength: {r.strength:.3f}"
            f"<br>Betweenness: {r.betweenness:.4f}<br>Community: {int(r.community)}"
            for r in sub.itertuples(index=False)
        ]
        traces.append(go.Scatter3d(
            x=sub["x_mni"], y=sub["y_mni"], z=sub["z_mni"], mode="markers",
            marker=dict(size=sub["marker_size"], color=NETWORK_COLORS[network], opacity=0.96,
                        line=dict(color="white", width=0.9)),
            text=hover, hovertemplate="%{text}<extra></extra>",
            name=NETWORK_DISPLAY[network], legendgroup=network,
        ))
    top10 = merged.sort_values("hub_rank").head(10)
    labels = go.Scatter3d(
        x=top10["x_mni"], y=top10["y_mni"], z=top10["z_mni"], mode="text",
        text=[str(int(x)) for x in top10["hub_rank"]],
        textfont=dict(size=11, color="#111827"), textposition="top center",
        hoverinfo="skip", showlegend=False,
    )
    return traces, labels


def split_edges(graph):
    edges = sorted(graph.edges(data=True), key=lambda e: float(e[2]["weight"]), reverse=True)
    n = len(edges)
    k5, k10, k20 = int(round(.05*n)), int(round(.10*n)), int(round(.20*n))
    return edges[:k5], edges[k5:k10], edges[k10:k20], edges[k20:]


def cam(x, y, z):
    return dict(eye=dict(x=x, y=y, z=z), center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1), projection=dict(type="orthographic"))


def main() -> None:
    args = parse_args()
    repo = args.repo_dir.resolve()
    work = args.work_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    matrix, labels, nodes, atlas_path, matrix_diff = v1.load_inputs(repo, work)
    coords = v1.atlas_centroids(atlas_path, labels).set_index("node", drop=False)
    graph = v1.connected_graph(matrix, density=0.15)
    signs = np.array([float(d["correlation"]) for _, _, d in graph.edges(data=True)])
    if graph.number_of_edges() != 742 or int(np.sum(signs < 0)) != 0:
        raise RuntimeError("Validated graph invariants failed.")

    lh_world, lh_faces = hemisphere_mesh(atlas_path, 1, 50, args.mesh_step)
    rh_world, rh_faces = hemisphere_mesh(atlas_path, 51, 100, args.mesh_step)
    bins = split_edges(graph)

    traces = [
        mesh_trace(lh_world, lh_faces, "#CBD5E1"),
        mesh_trace(rh_world, rh_faces, "#D7DEE8"),
        edge_trace(bins[0], coords, 4.2, "rgba(15,23,42,0.78)", True),
        edge_trace(bins[1], coords, 2.9, "rgba(30,41,59,0.58)", True),
        edge_trace(bins[2], coords, 1.65, "rgba(51,65,85,0.36)", True),
        edge_trace(bins[3], coords, 0.75, "rgba(71,85,105,0.10)", False),
    ]
    ntr, hub_labels = node_traces(coords.reset_index(drop=True), nodes)
    traces.extend(ntr)
    traces.append(hub_labels)
    fig = go.Figure(data=traces)

    total = len(traces)
    always = list(range(0, 2)) + list(range(6, total))
    def visibility(edge_indices):
        v = [False] * total
        for i in always + edge_indices:
            v[i] = True
        return v

    edge_buttons = [
        dict(label="Top 5%", method="update", args=[{"visible": visibility([2])}]),
        dict(label="Top 10%", method="update", args=[{"visible": visibility([2,3])}]),
        dict(label="Top 20%", method="update", args=[{"visible": visibility([2,3,4])}]),
        dict(label="All 742", method="update", args=[{"visible": visibility([2,3,4,5])}]),
        dict(label="Nodes only", method="update", args=[{"visible": visibility([])}]),
    ]
    camera_buttons = [
        dict(label="3D", method="relayout", args=[{"scene.camera": cam(1.55,-1.85,1.15)}]),
        dict(label="Left lateral", method="relayout", args=[{"scene.camera": cam(-2.7,0,0.05)}]),
        dict(label="Right lateral", method="relayout", args=[{"scene.camera": cam(2.7,0,0.05)}]),
        dict(label="Dorsal", method="relayout", args=[{"scene.camera": cam(0,0,2.8)}]),
        dict(label="Anterior", method="relayout", args=[{"scene.camera": cam(0,2.8,0.05)}]),
        dict(label="Posterior", method="relayout", args=[{"scene.camera": cam(0,-2.8,0.05)}]),
    ]

    fig.update_layout(
        title=dict(text="Schaefer-100 resting-state functional connectome<br><sup>10-subject Fisher-z group mean · connected 15% graph · 742 positive edges</sup>", x=.5),
        paper_bgcolor="white", font=dict(family="Arial, sans-serif", color="#111827"),
        width=1500, height=950, margin=dict(l=0,r=0,t=105,b=20),
        legend=dict(x=.01,y=.98,bgcolor="rgba(255,255,255,.84)",bordercolor="#CBD5E1",borderwidth=1,title="Yeo network"),
        updatemenus=[
            dict(type="dropdown",x=.99,y=.99,xanchor="right",yanchor="top",buttons=edge_buttons,active=2,bgcolor="white"),
            dict(type="dropdown",x=.99,y=.91,xanchor="right",yanchor="top",buttons=camera_buttons,active=0,bgcolor="white"),
        ],
        annotations=[dict(
            text="Node size = composite hub score · click legend items to hide/show networks · surface = Schaefer atlas-support mesh",
            x=.5,y=.005,xref="paper",yref="paper",showarrow=False,font=dict(size=11,color="#475569"))],
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", bgcolor="white", camera=cam(1.55,-1.85,1.15), dragmode="orbit"),
    )

    html = out / "connectome_3d_viewer.html"
    fig.write_html(html, include_plotlyjs=True, full_html=True,
                   config={"displaylogo":False,"responsive":True,"scrollZoom":True})
    meta = {
        "viewer":"interactive Plotly 3D cortical connectome",
        "subjects":10,"nodes":100,"graph_edges":742,
        "graph_positive_edges":int(np.sum(signs>0)),"graph_negative_edges":int(np.sum(signs<0)),
        "public_tsv_vs_npy_max_abs_diff":float(matrix_diff),
        "top_hub":str(nodes.sort_values("hub_rank").iloc[0]["label"]),
        "surface_reference":"meshes extracted from Schaefer-100 volumetric atlas support; not subject-specific pial or structural MRI surfaces",
        "default_edge_display":"strongest 20% of validated graph edges",
    }
    (out/"connectome_3d_viewer_validation.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("CONNECTOME_3D_VIEWER_PASSED")
    print(f"HTML={html}")
    print(f"HTML_BYTES={html.stat().st_size}")
    print(f"LH_MESH_VERTICES={len(lh_world)} LH_MESH_FACES={len(lh_faces)}")
    print(f"RH_MESH_VERTICES={len(rh_world)} RH_MESH_FACES={len(rh_faces)}")
    print("DEFAULT_EDGES_DISPLAYED=148")

if __name__ == "__main__":
    main()
