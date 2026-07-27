# Resting-State fMRI Functional Connectomics: Mini-Report

## Aim

To demonstrate a transparent workflow for transforming preprocessed resting-state fMRI into regional time series, functional-connectivity matrices, and graph-theoretical summaries.

## Data and methods

Ten public preprocessed ADHD-200 runs were analyzed with Schaefer-100. Regional signals underwent nuisance regression, detrending, sample-z standardization, and 0.01–0.10 Hz filtering. Subject correlation matrices were averaged in Fisher-z space.

A connected graph was constructed from absolute group correlations at approximately 15% density.

## Results

The graph contained 100 nodes and 742 edges. Mean degree was 14.84, global efficiency 0.495, path length 2.362, weighted modularity 0.511, and small-world sigma 2.527. The highest composite hub score occurred in `LH_Default_Par_2`.

## Interpretation and limitations

Sigma above one is compatible with small-world organization under the implemented reference procedure. These findings are descriptive and should not be generalized beyond this technical sample. No group comparison, hypothesis test, motion censoring, or independent BRAPH replication was conducted.
