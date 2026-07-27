# Methods

## Design

This is a technical demonstration of resting-state functional-connectivity and graph analysis using 10 public preprocessed ADHD-200 datasets. It does not test diagnostic or group-level hypotheses.

## Signal extraction

Regional signals were extracted with Nilearn `NiftiLabelsMasker` and the Schaefer 2018 atlas with 100 parcels, 7 networks, and 2 mm resolution.

Processing included detrending, 0.01–0.10 Hz filtering, sample-z standardization, and regression of CSF, white-matter signal, six motion parameters, and five CompCor components when available. Global-signal regression was not used.

## Functional connectivity

A 100 × 100 Pearson-correlation matrix was calculated per participant. Group connectivity was computed by transforming correlations to Fisher z, averaging across subjects, and converting back to Pearson r.

## Graph construction

Topology used absolute group-correlation magnitude. A maximum spanning tree guaranteed connectivity, after which the strongest remaining edges were added until approximately 15% density. The final graph contained 100 nodes and 742 edges.

For weighted shortest paths, distance was defined as the reciprocal of absolute correlation magnitude.

## Graph metrics

Global metrics included degree, strength, clustering, transitivity, efficiency, characteristic path length, assortativity, modularity, communities, and small-world sigma.

Small-world sigma used 5 rewiring iterations, 10 random reference graphs, and random seed 42.

Regional metrics included degree, strength, weighted clustering, betweenness, eigenvector centrality, and community assignment. The composite hub score was the mean of standardized degree, strength, betweenness, and eigenvector centrality.

## Implementation note

The reported numerical results were generated with Python, Nilearn, and NetworkX. Exported matrices are suitable for BRAPH import, but independent BRAPH replication was not completed in this release.
