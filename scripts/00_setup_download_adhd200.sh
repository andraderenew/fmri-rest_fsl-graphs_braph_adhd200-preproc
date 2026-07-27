#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="rsfmri-graphs"
WORK="${RSFMRI_WORK_DIR:-$HOME/neuroimaging-data/fmri-rest_fsl-graphs_braph_adhd200-preproc}"

export RSFMRI_WORK_DIR="$WORK"

mkdir -p \
    "$WORK/data" \
    "$WORK/derivatives" \
    "$WORK/work" \
    "$REPO/results/tables" \
    "$REPO/results/figures"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.11
fi

conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install \
    'nilearn[plotting]==0.14.0' \
    networkx \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    nibabel

conda run --no-capture-output \
    -n "$ENV_NAME" \
    python "$REPO/scripts/01_download_adhd200_subset.py"

echo
echo "Setup and ADHD-200 subset download completed."
echo "Working directory: $WORK"
