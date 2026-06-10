#!/usr/bin/env bash
set -euo pipefail

WEIGHT_ROOT=${WEIGHT_ROOT:-third_party_weights}
CLIP_DIR="$WEIGHT_ROOT/openai_clip"
CLIP_TEXT_DIR="$WEIGHT_ROOT/openai_clip_vit_base_patch32"
export CLIP_DIR CLIP_TEXT_DIR

mkdir -p "$CLIP_DIR" "$CLIP_TEXT_DIR"

python - <<'PY'
from pathlib import Path
import os
import shutil
from clip import clip as openai_clip

clip_dir = Path(os.environ.get("CLIP_DIR", "third_party_weights/openai_clip")).resolve()
clip_dir.mkdir(parents=True, exist_ok=True)
model_path = Path(openai_clip._download(openai_clip._MODELS["RN50"], str(clip_dir))).resolve()
target = clip_dir / "RN50.pt"
if model_path != target:
    shutil.copy2(model_path, target)
print(f"CLIP RN50 checkpoint: {target}")
PY

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is required for the CLIP text snapshot. Install it with: pip install huggingface_hub" >&2
  exit 1
fi

huggingface-cli download openai/clip-vit-base-patch32 \
  --local-dir "$CLIP_TEXT_DIR" \
  --local-dir-use-symlinks False

printf 'export CLIP_RN50_LOCAL_PATH=%s\n' "$(cd "$CLIP_DIR" && pwd)/RN50.pt"
printf 'export CLIP_HF_LOCAL_PATH=%s\n' "$(cd "$CLIP_TEXT_DIR" && pwd)"
