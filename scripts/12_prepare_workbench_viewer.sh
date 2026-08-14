#!/usr/bin/env bash

REPO="$HOME/github/fmri-rest_fsl-graphs_braph_adhd200-preproc"
VENV="$HOME/.venvs/connectome-surface"
WORK="/media/andraderenew/Elements/neuroimaging/fmri-rest_fsl-graphs_braph_adhd200-preproc"
CIFTI="$WORK/workbench_cifti_outputs_v1"
REF="$WORK/reference_surface_connectome"
VIEW="$WORK/workbench_viewer_v1"

mkdir -p "$VIEW"

SURFROOT="$($VENV/bin/python - <<'PY'
from pathlib import Path
import brainspace
print(Path(brainspace.__file__).resolve().parent / 'datasets' / 'surfaces')
PY
)"

L_SRC="$SURFROOT/conte69_32k_lh.gii"
R_SRC="$SURFROOT/conte69_32k_rh.gii"
LS_SRC="$SURFROOT/conte69_32k_lh_sphere.gii"
RS_SRC="$SURFROOT/conte69_32k_rh_sphere.gii"

for f in "$L_SRC" "$R_SRC" "$LS_SRC" "$RS_SRC"; do
    if [ ! -f "$f" ]; then
        echo "ERROR_MISSING_SURFACE=$f"
        exit 1
    fi
done

L="$VIEW/Conte69.L.midthickness.32k_fs_LR.surf.gii"
R="$VIEW/Conte69.R.midthickness.32k_fs_LR.surf.gii"
LS="$VIEW/Conte69.L.sphere.32k_fs_LR.surf.gii"
RS="$VIEW/Conte69.R.sphere.32k_fs_LR.surf.gii"

cp "$L_SRC" "$L"
cp "$R_SRC" "$R"
cp "$LS_SRC" "$LS"
cp "$RS_SRC" "$RS"

wb_command -set-structure "$L" CORTEX_LEFT -surface-type ANATOMICAL -surface-secondary-type MIDTHICKNESS
wb_command -set-structure "$R" CORTEX_RIGHT -surface-type ANATOMICAL -surface-secondary-type MIDTHICKNESS
wb_command -set-structure "$LS" CORTEX_LEFT -surface-type SPHERICAL
wb_command -set-structure "$RS" CORTEX_RIGHT -surface-type SPHERICAL

cp "$REF/Schaefer2018_100Parcels_7Networks_order.dlabel.nii" "$VIEW/"
cp "$CIFTI/hub_score.dscalar.nii" "$VIEW/"
cp "$CIFTI/LH_Default_Par_2_fingerprint.dscalar.nii" "$VIEW/"
cp "$CIFTI/hub_score.pscalar.nii" "$VIEW/"
cp "$CIFTI/LH_Default_Par_2_fingerprint.pscalar.nii" "$VIEW/"
cp "$CIFTI/group_mean_connectome.pconn.nii" "$VIEW/"
cp "$CIFTI/workbench_cifti_validation.json" "$VIEW/"

SPEC="$VIEW/connectome.spec"
rm -f "$SPEC"

wb_command -spec-file-modify "$SPEC" \
  -add CORTEX_LEFT "$L" \
  -add CORTEX_RIGHT "$R" \
  -add CORTEX_LEFT "$LS" \
  -add CORTEX_RIGHT "$RS" \
  -add INVALID "$VIEW/Schaefer2018_100Parcels_7Networks_order.dlabel.nii" \
  -add INVALID "$VIEW/hub_score.dscalar.nii" \
  -add INVALID "$VIEW/LH_Default_Par_2_fingerprint.dscalar.nii" \
  -add INVALID "$VIEW/hub_score.pscalar.nii" \
  -add INVALID "$VIEW/LH_Default_Par_2_fingerprint.pscalar.nii" \
  -add INVALID "$VIEW/group_mean_connectome.pconn.nii"

echo "WORKBENCH_VIEWER_PACKAGE_READY"
echo "VIEW=$VIEW"
echo "SPEC=$SPEC"
