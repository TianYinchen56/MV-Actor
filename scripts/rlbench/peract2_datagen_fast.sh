#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${DATA_PATH:-peract2_raw}
ZARR_PATH=${ZARR_PATH:-zarr_datasets/peract2}

mkdir -p "$DATA_PATH" "$ZARR_PATH"

# Download the official 3DFA PerAct2 zarr data.
(
  cd "$ZARR_PATH"
  wget -c https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2.zip
  unzip -o peract2.zip
  rm -f peract2.zip
)

# Download the official 3DFA PerAct2 test seeds used for online RLBench evaluation.
(
  cd "$DATA_PATH"
  wget -c https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2_test.zip
  unzip -o peract2_test.zip
  rm -f peract2_test.zip
)

echo "PerAct2 zarr data: $ZARR_PATH"
echo "PerAct2 test seeds: $DATA_PATH/peract2_test"
