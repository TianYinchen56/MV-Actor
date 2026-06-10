#!/usr/bin/env bash
set -euo pipefail

PI3X_DIR=${PI3X_DIR:-third_party_weights/pi3x}
PI3X_REPO=${PI3X_REPO:-yyfz233/Pi3X}
mkdir -p "$PI3X_DIR"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is required. Install it with: pip install huggingface_hub" >&2
  exit 1
fi

huggingface-cli download "$PI3X_REPO" \
  --include "*.safetensors" \
  --local-dir "$PI3X_DIR" \
  --local-dir-use-symlinks False

mapfile -t ckpts < <(find "$PI3X_DIR" -type f -name '*.safetensors' | sort)
if [ "${#ckpts[@]}" -ne 1 ]; then
  printf 'Expected exactly one Pi3X .safetensors checkpoint under %s, found %s:\n' "$PI3X_DIR" "${#ckpts[@]}" >&2
  printf '  %s\n' "${ckpts[@]}" >&2
  exit 1
fi

printf 'export PI3X_CKPT=%s\n' "$(cd "$(dirname "${ckpts[0]}")" && pwd)/$(basename "${ckpts[0]}")"
