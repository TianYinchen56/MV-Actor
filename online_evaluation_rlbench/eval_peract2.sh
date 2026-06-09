#!/usr/bin/env bash
set -euo pipefail

# Short public entry point, matching the style of the official 3DFA repository.
# Set CKPT, DATA_DIR, CLIP_HF_LOCAL_PATH, COPPELIASIM_ROOT, RLBENCH_ROOT, and PYREP_ROOT first.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/run_eval_mv_actor_alltasks.sh"
