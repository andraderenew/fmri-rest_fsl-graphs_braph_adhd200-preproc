#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def find_first(candidates):
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("None of the candidate files exists:\n" + "\n".join(str(x) for x in candidates))


def crop_fraction(img, left=0.0, right=1.0, top=0.0, bottom=1.0):
    h, w = img.shape[:2]
    x0 = int(round(left * w))
    x1 = int(round(right * w))
    y0 = int(round(top * h))
    y1 = int(round(bottom * h))
    return img[y0:y1, x0:x1]


def draw_panel(ax, img, letter, title, letter_x=-0.025, letter_y=1.02):
    ax.imshow(img)
    ax.axis("off")
    ax.text(letter_x, letter_y, letter, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=18, fontweight="bold")
    ax.set_title(title, fontsize=12.5, fontweight="semibold", pad=5)


def main():
    args = parse_args()
    work = args.work_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    matrix_path = find_first([
        work / "connectome_visual_redesign_v3/figures/connectivity_network_ordered.png",
        work / "connectome_visual_redesign_v2/figures/connectivity_network_ordered.png",
        work / "connectome_visual_redesign_v1/figures/connectivity_network_ordered.png",
    ])
    hub_path = find_first([
        work / "surface_publication_maps_v3/surface_hub_score.png",
        work / "surface_publication_maps_v2/surface_hub_score.png",
        work / "surface_publication_maps_v1/surface_hub_score.png",
    ])
    fp_path = find_first([
        work / "surface_publication_maps_v3/surface_seed_fingerprint_LH_Default_Par_2.png",
        work / "surface_publication_maps_v2/surface_seed_fingerprint_LH_Default_Par_2.png",
        work / "surface_publication_maps_v1/surface_seed_fingerprint_LH_Default_Par_2.png",
    ])
    atlas_path = find_first([
        work / "surface_publication_maps_v3/surface_schaefer100_yeo7.png",
        work / "surface_publication_maps_v2/surface_schaefer100_yeo7.png",
        work / "surface_publication_maps_v1/surface_schaefer100_yeo7.png",
    ])

    matrix = mpimg.imread(matrix_path)
    hub = mpimg.imread(hub_path)
    fp = mpimg.imread(fp_path)
    atlas = mpimg.imread(atlas_path)

    # Crop only presentation-only whitespace/titles/captions from the already validated panels.
    # Scientific content, colorbars, labels, view annotations, and atlas legend are retained.
    matrix = crop_fraction(matrix, left=0.085, right=0.965, top=0.055, bottom=0.915)
    hub = crop_fraction(hub, left=0.055, right=0.975, top=0.070, bottom=0.985)
    fp = crop_fraction(fp, left=0.055, right=0.975, top=0.070, bottom=0.985)
    atlas = crop_fraction(atlas, left=0.055, right=0.965, top=0.070, bottom=0.990)

    fig = plt.figure(figsize=(15.2, 10.8), facecolor="white")
    outer = fig.add_gridspec(
        1, 2,
        width_ratios=[1.05, 1.0],
        left=0.025, right=0.985, top=0.965, bottom=0.035,
        wspace=0.025,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    draw_panel(ax_a, matrix, "A", "Group-mean functional connectivity")

    right = outer[0, 1].subgridspec(
        3, 1,
        height_ratios=[1.0, 1.0, 1.08],
        hspace=0.11,
    )

    ax_b = fig.add_subplot(right[0, 0])
    draw_panel(ax_b, hub, "B", "Functional-connectome hub score")

    ax_c = fig.add_subplot(right[1, 0])
    draw_panel(ax_c, fp, "C", "Connectivity fingerprint — LH_Default_Par_2")

    ax_d = fig.add_subplot(right[2, 0])
    draw_panel(ax_d, atlas, "D", "Schaefer-100 / Yeo 7 reference")

    png = out / "figure_main_connectome_surface_matrix_v2.png"
    pdf = out / "figure_main_connectome_surface_matrix_v2.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)

    if not png.exists() or png.stat().st_size < 250_000:
        raise RuntimeError(f"PNG output failed size validation: {png}")
    if not pdf.exists() or pdf.stat().st_size < 100_000:
        raise RuntimeError(f"PDF output failed size validation: {pdf}")

    print("MAIN_PUBLICATION_FIGURE_V2_PASSED")
    print("MATRIX_SOURCE=", matrix_path)
    print("HUB_SOURCE=", hub_path)
    print("FINGERPRINT_SOURCE=", fp_path)
    print("ATLAS_SOURCE=", atlas_path)
    print("WROTE=", png, png.stat().st_size)
    print("WROTE=", pdf, pdf.stat().st_size)


if __name__ == "__main__":
    main()
