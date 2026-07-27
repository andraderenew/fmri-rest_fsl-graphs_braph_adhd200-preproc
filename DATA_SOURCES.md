# Data Sources and Storage

This project uses a public preprocessed subset of the ADHD-200 resting-state fMRI dataset distributed through Nilearn and hosted by NITRC.

- Retrieval: Nilearn `fetch_adhd`
- Local subset: 10 participants
- Functional data: preprocessed 4D resting-state fMRI
- Additional files: confound regressors and phenotypic metadata
- TR: 2.0 seconds

Regional signals used the Schaefer 2018 cortical atlas with 100 parcels, 7 networks, and 2 mm resolution, retrieved with `fetch_atlas_schaefer_2018`.

Imaging data and large derivatives are stored outside the repository. The location is controlled by `RSFMRI_WORK_DIR`.

```bash
export RSFMRI_WORK_DIR="/path/to/rsfmri-work-directory"
```

They are excluded from GitHub. The repository contains scripts, environment files, public summaries, the group matrix, graph metrics, and figures.
