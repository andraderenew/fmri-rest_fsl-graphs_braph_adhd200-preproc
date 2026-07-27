#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/github/fmri-rest_fsl-graphs_braph_adhd200-preproc"
BACKUP_DIR="$HOME/Documents/github-backups/fmri-rest_fsl-graphs_braph_adhd200-preproc"
SOURCE="$BACKUP_DIR/08_repair_and_publish_rsfmri_project.sh"
TEMP="/tmp/08_repair_and_publish_rsfmri_project.sh"
LOG="$HOME/Downloads/rsfmri_repair_publish.log"

cd "$REPO"
mkdir -p "$BACKUP_DIR"

# Keep the local resume helper out of the public repository.
if [ -f "$REPO/scripts/09_fix_requirements_and_resume_publish.sh" ]; then
    mv -f \
      "$REPO/scripts/09_fix_requirements_and_resume_publish.sh" \
      "$BACKUP_DIR/"
fi

if [ ! -f "$SOURCE" ]; then
    echo "ERROR: no encuentro $SOURCE"
    exit 1
fi

cp -f "$SOURCE" "$TEMP"
chmod +x "$TEMP"

echo "=== RESUMING PUBLICATION FROM TEMPORARY COPY ==="
bash "$TEMP" 2>&1 | tee "$LOG"
