# Resting-State fMRI — Graph Metrics with BRAPH (ADHD-200 preprocessed)
[![License](https://img.shields.io/github/license/andraderenew/fmri-rest_fsl-graphs_braph_adhd200-preproc)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](#cite-this-work)
[![Pages](https://img.shields.io/website?url=https%3A%2F%2Fandraderenew.github.io%2Ffmri-rest_fsl-graphs_braph_adhd200-preproc%2F)](https://andraderenew.github.io/fmri-rest_fsl-graphs_braph_adhd200-preproc/)
![Release](https://img.shields.io/github/v/release/andraderenew/fmri-rest_fsl-graphs_braph_adhd200-preproc?include_prereleases)
![Last commit](https://img.shields.io/github/last-commit/andraderenew/fmri-rest_fsl-graphs_braph_adhd200-preproc)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5627--579X-A6CE39)](https://orcid.org/0000-0001-5627-579X)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Profile-4285F4)](https://scholar.google.es/citations?hl=es&user=Nl3ApFEAAAAJ)

**One-line:** Build functional connectomes from preprocessed time-series and report hubs/small-worldness with BRAPH.

## Overview
Use **ADHD-200 preprocessed** time-series/connectomes to compute **graph metrics** in **BRAPH** (degree, betweenness, small-worldness) and visualize hubs.

## Data & subset
See `DATA_SOURCES.md`. Suggested: <10–30 subjects> (timeseries only) to stay in MB → few GB.

## Pipeline
(Opt) nuisance/bandpass → correlation matrices (Fisher z) → BRAPH metrics → hub summary & figures.

## Results (to be filled)
- Mean connectome heatmap; node degree map  
- Top hubs (degree/betweenness) table

## Reproducibility
- Versions: see `env/TOOL_VERSIONS.md`  
- Steps: “Download preprocessed timeseries → compute matrices → run BRAPH → figures.”  
- Limits: pipeline choice affects metrics; small N

## Cite this work
See `CITATION.cff` (add DOI after first Release).
