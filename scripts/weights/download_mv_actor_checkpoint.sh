#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR=${CKPT_DIR:-checkpoints}
CKPT_NAME=${CKPT_NAME:-best_iter200000_snapshot.pth}
CKPT_URL=${CKPT_URL:-https://github.com/TianYinchen56/MV-Actor/releases/download/v0.1-eval/best_iter200000_snapshot.pth}
CKPT_SHA256=${CKPT_SHA256:-2b5655e217b20fada49ba6757e4158e276b04a504ad8d9bbaf8b28e5888f6793}
GITHUB_REPO=${GITHUB_REPO:-TianYinchen56/MV-Actor}
GITHUB_RELEASE_TAG=${GITHUB_RELEASE_TAG:-v0.1-eval}

mkdir -p "$CKPT_DIR"
TARGET="$CKPT_DIR/$CKPT_NAME"

if [ -n "${GITHUB_TOKEN:-}" ]; then
  if ! command -v python >/dev/null 2>&1; then
    echo "python is required to resolve private GitHub release assets." >&2
    exit 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to download private GitHub release assets." >&2
    exit 1
  fi
  ASSET_ID=$(curl -fsS \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${GITHUB_REPO}/releases/tags/${GITHUB_RELEASE_TAG}" \
    | CKPT_NAME="$CKPT_NAME" python -c '
import json
import os
import sys

release = json.load(sys.stdin)
ckpt_name = os.environ["CKPT_NAME"]
for asset in release.get("assets", []):
    if asset.get("name") == ckpt_name:
        print(asset["id"])
        break
else:
    names = [asset.get("name") for asset in release.get("assets", [])]
    raise SystemExit(f"Asset {ckpt_name!r} not found in release. Available assets: {names}")
')
  curl -fL \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/octet-stream" \
    "https://api.github.com/repos/${GITHUB_REPO}/releases/assets/${ASSET_ID}" \
    -o "$TARGET"
else
  if command -v wget >/dev/null 2>&1; then
    wget -c "$CKPT_URL" -O "$TARGET"
  else
    curl -fL "$CKPT_URL" -o "$TARGET"
  fi
fi

echo "$CKPT_SHA256  $TARGET" | sha256sum -c -

echo "export CKPT=$(cd "$CKPT_DIR" && pwd)/$CKPT_NAME"
