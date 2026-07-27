# Resting-State fMRI Functional Connectomics and Graph Analysis

[![License](https://img.shields.io/github/license/andraderenew/fmri-rest_fsl-graphs_braph_adhd200-preproc)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17715118-blue)](https://doi.org/10.5281/zenodo.17715118)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5627--579X-A6CE39)](https://orcid.org/0000-0001-5627-579X)

Reproducible resting-state fMRI workflow using a public preprocessed ADHD-200 subset, Schaefer-100 parcellation, Nilearn functional connectivity, and NetworkX graph analysis.

> **Implementation note:** the numerical results in this repository were generated with Python, Nilearn, and NetworkX. The exported matrices and node tables are suitable for independent BRAPH import and validation, but this release does not claim a completed independent BRAPH replication.

## Project snapshot

- **Subjects:** 10
- **Dataset:** public ADHD-200 preprocessed sample distributed through Nilearn/NITRC
- **Atlas:** Schaefer 2018, 100 parcels, 7 networks, 2 mm
- **TR:** 2.0 s
- **Temporal filtering:** 0.01–0.10 Hz
- **Connectivity:** Pearson correlation, averaged in Fisher-z space
- **Graph construction:** connected proportional graph at 15% density
- **Final graph:** 100 nodes and 742 edges
- **Small-world sigma:** 2.527
- **Weighted modularity:** 0.511
- **Top composite hub:** `LH_Default_Par_2`

## Workflow

1. Download 10 public preprocessed resting-state fMRI datasets and confounds.
2. Extract 100 regional time series with the Schaefer-100 atlas.
3. Apply detrending, nuisance regression, standardization, and 0.01–0.10 Hz filtering.
4. Compute subject-level Pearson correlation matrices.
5. Average connectivity in Fisher-z space.
6. Construct a connected graph at 15% density using a maximum spanning tree plus the strongest remaining absolute correlations.
7. Calculate global and regional graph metrics.
8. Export QC tables, graph tables, figures, and BRAPH-ready matrices.

## Main results

| Metric | Value |
|---|---:|
| Nodes | 100 |
| Edges | 742 |
| Density | 0.1499 |
| Mean degree | 14.84 |
| Mean strength | 7.590 |
| Average clustering | 0.545 |
| Global efficiency | 0.495 |
| Characteristic path length | 2.362 |
| Weighted modularity | 0.511 |
| Communities | 4 |
| Small-world sigma | 2.527 |

A sigma above 1 is consistent with small-world organization relative to the implemented random-reference graphs. This is a descriptive result for a small technical sample, not a population-level inference.

## Figures

### Group-mean connectivity matrix

![Group-mean functional connectivity matrix](results/figures/fig1_group_mean_connectome.png)

### Strongest group-level connections

![Group connectome displayed on the brain](results/figures/fig2_group_connectome_brain.png)

### Highest-ranked hubs

![Top functional-connectome hubs](results/figures/fig3_top_hubs.png)

## Top 10 hubs

| Rank | Region | Network | Degree | Strength | Betweenness | Hub score |
|---:|---|---|---:|---:|---:|---:|
| 1 | `LH_Default_Par_2` | Default | 29 | 15.054 | 0.0427 | 2.370 |
| 2 | `LH_Default_Temp_1` | Default | 26 | 13.287 | 0.0431 | 1.987 |
| 3 | `LH_Default_pCunPCC_2` | Default | 25 | 13.430 | 0.0305 | 1.804 |
| 4 | `RH_Cont_pCun_1` | Control | 22 | 11.161 | 0.0470 | 1.688 |
| 5 | `RH_DorsAttn_Post_1` | Dorsal attention | 24 | 10.905 | 0.0443 | 1.598 |
| 6 | `RH_SomMot_1` | Somatomotor | 24 | 12.426 | 0.0482 | 1.571 |
| 7 | `LH_SomMot_6` | Somatomotor | 22 | 10.610 | 0.0544 | 1.455 |
| 8 | `LH_Default_pCunPCC_1` | Default | 23 | 11.487 | 0.0311 | 1.443 |
| 9 | `RH_Vis_6` | Visual | 19 | 10.692 | 0.0245 | 1.129 |
| 10 | `LH_SomMot_3` | Somatomotor | 22 | 12.276 | 0.0334 | 1.128 |

The hub score is the mean of standardized degree, strength, betweenness centrality, and eigenvector centrality. It is an analysis-specific ranking rather than a universal biological definition of hub status.

## Reproduction

```bash
conda env create -f env/environment.yml
conda activate rsfmri-graphs
# Optional: keep imaging data on an external drive
export RSFMRI_WORK_DIR="/path/to/rsfmri-work-directory"
python scripts/01_download_adhd200_subset.py
python scripts/02_run_rsfmri_connectome_graphs.py
```

Raw/preprocessed imaging files, cached files, local path manifests, regional time series, and subject-level matrices are excluded from Git.

## Documentation

- [Data sources](DATA_SOURCES.md)
- [Methods](docs/METHODS.md)
- [Results](docs/RESULTS.md)
- [Mini-report](reports/report.md)
- [Software environment](env/TOOL_VERSIONS.md)
- [Citation metadata](CITATION.cff)

## Limitations

- The sample contains only 10 participants and is intended as a technical portfolio demonstration.
- Inputs were already preprocessed; upstream preprocessing was not reproduced locally.
- No group-level inference or motion censoring was performed.
- Results depend on parcellation, nuisance model, edge definition, density, and reference-graph settings.
- Absolute correlations were used for graph construction, so positive and negative edges are not distinguished in topology.
- Independent numerical BRAPH validation remains a separate task.

## Citation

See [`CITATION.cff`](CITATION.cff) or cite:

**Andrade Rey, R.** Resting-State fMRI — Graph Metrics with BRAPH (ADHD-200 preprocessed). Zenodo. https://doi.org/10.5281/zenodo.17715118
