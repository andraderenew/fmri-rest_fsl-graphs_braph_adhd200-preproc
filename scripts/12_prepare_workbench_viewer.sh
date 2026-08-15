#!/usr/bin/env bash

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$REPO/workbench_viewer_build}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SCENE_NAME="connectome_surface_pconn_LH_Default_Par_2.scene"
ATLAS_NAME="Schaefer2018_100Parcels_7Networks_order.dlabel.nii"
PCONN_NAME="group_mean_connectome_RedWhiteBlue.pconn.nii"

SCENE_SRC="$REPO/results/workbench/$SCENE_NAME"
ATLAS_SRC="$REPO/results/workbench/$ATLAS_NAME"
PCONN_SRC="$REPO/results/workbench/$PCONN_NAME"

echo "REPO=$REPO"
echo "OUT=$OUT"
echo "PYTHON_BIN=$PYTHON_BIN"

if ! command -v wb_command >/dev/null 2>&1; then
    echo "ERROR: wb_command is not available on PATH"
    exit 1
fi

if ! command -v wb_view >/dev/null 2>&1; then
    echo "ERROR: wb_view is not available on PATH"
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import brainspace' >/dev/null 2>&1; then
    echo "ERROR: BrainSpace is not importable with $PYTHON_BIN"
    echo "Activate the connectome-surface environment or set PYTHON_BIN."
    exit 1
fi

for f in "$SCENE_SRC" "$ATLAS_SRC" "$PCONN_SRC"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required repository file missing: $f"
        exit 1
    fi
done

SURFROOT="$(
"$PYTHON_BIN" -c '
from pathlib import Path
import brainspace
print(Path(brainspace.__file__).resolve().parent / "datasets" / "surfaces")
'
)"

L_SRC="$SURFROOT/conte69_32k_lh.gii"
R_SRC="$SURFROOT/conte69_32k_rh.gii"

for f in "$L_SRC" "$R_SRC"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: BrainSpace Conte69 surface missing: $f"
        exit 1
    fi
done

mkdir -p "$OUT"

L="$OUT/Conte69.L.32k_fs_LR.surf.gii"
R="$OUT/Conte69.R.32k_fs_LR.surf.gii"

cp -f "$L_SRC" "$L"
cp -f "$R_SRC" "$R"

wb_command \
  -set-structure \
  "$L" \
  CORTEX_LEFT \
  -surface-type ANATOMICAL \
  -surface-secondary-type MIDTHICKNESS \
  || exit 1

wb_command \
  -set-structure \
  "$R" \
  CORTEX_RIGHT \
  -surface-type ANATOMICAL \
  -surface-secondary-type MIDTHICKNESS \
  || exit 1

cp -f "$SCENE_SRC" "$OUT/$SCENE_NAME"
cp -f "$ATLAS_SRC" "$OUT/$ATLAS_NAME"
cp -f "$PCONN_SRC" "$OUT/$PCONN_NAME"

if [ -f "$REPO/results/cifti/workbench_cifti_validation.json" ]; then
    cp -f \
      "$REPO/results/cifti/workbench_cifti_validation.json" \
      "$OUT/workbench_cifti_validation.json"
fi

cat > "$OUT/01_open_scene.sh" <<LAUNCH
#!/usr/bin/env bash
HERE="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
wb_view "\$HERE/$SCENE_NAME"
LAUNCH

chmod +x "$OUT/01_open_scene.sh"

echo
echo "========================================"
echo "PORTABLE WORKBENCH PACKAGE"
echo "========================================"

for f in \
  "$L" \
  "$R" \
  "$OUT/$ATLAS_NAME" \
  "$OUT/$PCONN_NAME" \
  "$OUT/$SCENE_NAME"
do
    if [ ! -s "$f" ]; then
        echo "ERROR: missing or empty output: $f"
        exit 1
    fi
    stat -c '%n	%s bytes' "$f"
done

echo
echo "Scene:"
echo "  $OUT/$SCENE_NAME"
echo
echo "Launch:"
echo "  $OUT/01_open_scene.sh"
echo
echo "WORKBENCH_PORTABLE_PACKAGE_READY"
