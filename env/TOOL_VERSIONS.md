# Tool Versions and Reproducibility Provenance

## Declared analysis environment

The repository environment specification records:

- Python: 3.11
- Nilearn: 0.14.0
- NetworkX: dependency recorded, exact historical version not pinned
- pandas: dependency recorded, exact historical version not pinned
- SciPy: dependency recorded, exact historical version not pinned
- scikit-learn: dependency recorded, exact historical version not pinned
- Matplotlib: dependency recorded, exact historical version not pinned

A local ignored dataset manifest from the completed analysis also records Nilearn 0.14.0.

The exact historical versions of the unpinned Python dependencies were not preserved and are therefore not reconstructed retrospectively.

## Connectome Workbench provenance

Native CIFTI outputs and a Connectome Workbench scene are retained in the repository. The exact Workbench software version used to prepare those viewer outputs was not preserved.

## Cross-audit validation — 2026-08-17

The retained public group-connectivity matrix was independently reconstructed into a graph using:

- Python: 3.12.4
- NumPy: 2.2.2
- pandas: 2.2.3
- SciPy: 1.17.1
- NetworkX: 3.6.1
- Matplotlib: 3.10.9
- nibabel: 5.4.2

Nilearn and scikit-learn were not installed in the base cross-audit shell, so the upstream image-to-timeseries extraction was not rerun in that shell.

Using the retained public matrix, the audit reproduced:

- nodes: 100
- edges: 742
- density: 0.14989899
- weighted modularity: 0.51149787
- highest composite hub: `LH_Default_Par_2`
- highest hub score: 2.37041592
- small-world sigma: 2.52714410

The reproduced small-world sigma differed from the retained value by approximately `3.1e-09`.

These checks validate the retained matrix-to-graph computation under the documented parameters; they do not establish the missing exact historical versions of every dependency.
