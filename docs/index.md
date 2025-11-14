# Resting-State fMRI: Graph Metrics with BRAPH (ADHD-200 preprocessed)

**Goal:** Build functional connectivity matrices from preprocessed resting-state time-series and report basic graph metrics (hubs, small-worldness).

---

## Snapshot
- **Dataset:** ADHD-200 preprocessed (choose one pipeline release)
- **Local subset:** <N subjects> · **Disk:** stay in MB–low GB (timeseries/connectomes only)
- **Tools:** FSL (optional denoise), BRAPH (graph metrics)
- **Status:** <planned / in progress / complete>
- **Last updated:** <YYYY-MM-DD>

---

## Data
- **Source:** ADHD-200 preprocessed (public, NITRC account).  
- **What I downloaded:** atlas timeseries or connectomes; list subjects.  
- **Layout:** subject folders with timeseries per ROI.

---

## Pipeline (high-level)
1) (Optional) Additional nuisance regression/bandpass  
2) Correlation matrix per subject (z-scored)  
3) Import to BRAPH → compute degree, betweenness, small-worldness  
4) Group summary; visualize hubs

---

## Results (to be filled)
- Figure: mean connectome heatmap; node degree map  
- Table: top hubs by degree/betweenness

---

## Reproducibility
- Versions in `env/TOOL_VERSIONS.md`.  
- Steps: “Download preprocessed timeseries → compute matrices → run BRAPH metrics → export figures.”  
- Limitations: pipeline choice affects metrics; small N.

---

**Author:** Rene Andrade Rey · 🧪 ORCID: https://orcid.org/0000-0001-5627-579X · 🌐 Scholar: https://scholar.google.es/citations?hl=es&user=Nl3ApFEAAAAJ
