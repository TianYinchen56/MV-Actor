#!/usr/bin/env bash
set -euo pipefail

DISPLAY_NUM=${DISPLAY_NUM:?DISPLAY_NUM is required}
if [ -n "${XVFB_BIN:-}" ]; then
  XVFB_BIN=$XVFB_BIN
elif command -v Xvfb >/dev/null 2>&1; then
  XVFB_BIN=$(command -v Xvfb)
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/bin/Xvfb" ]; then
  XVFB_BIN=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/bin/Xvfb
else
  echo "Xvfb binary not found. Set XVFB_BIN or activate an environment that provides Xvfb."
  exit 1
fi

XVFB_SCREEN=${XVFB_SCREEN:-1280x1024x24}
XVFB_LOG_FILE=${XVFB_LOG_FILE:-/tmp/xvfb_${DISPLAY_NUM}.log}
REUSE_EXISTING_DISPLAY=${REUSE_EXISTING_DISPLAY:-0}

if [ -n "${OPENSSL10_LIB:-}" ]; then
  export LD_LIBRARY_PATH=$OPENSSL10_LIB:${LD_LIBRARY_PATH:-}
fi

if [ -n "${COPPELIASIM_ROOT:-}" ]; then
  export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$COPPELIASIM_ROOT/lib:${LD_LIBRARY_PATH:-}
  export QT_QPA_PLATFORM_PLUGIN_PATH=${QT_QPA_PLATFORM_PLUGIN_PATH:-$COPPELIASIM_ROOT}
fi
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}

XVFB_PID=
CMD_PID=
export DISPLAY=:$DISPLAY_NUM

if [ "$REUSE_EXISTING_DISPLAY" = "1" ] && DISPLAY=:$DISPLAY_NUM xdpyinfo >/dev/null 2>&1; then
  USE_EXISTING_DISPLAY=1
else
  USE_EXISTING_DISPLAY=0
fi

cleanup() {
  if [ -n "${CMD_PID:-}" ] && kill -0 "$CMD_PID" >/dev/null 2>&1; then
    kill -TERM -- "-$CMD_PID" >/dev/null 2>&1 || true
    wait "$CMD_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${XVFB_PID:-}" ] && kill -0 "$XVFB_PID" >/dev/null 2>&1; then
    kill "$XVFB_PID" >/dev/null 2>&1 || true
    wait "$XVFB_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if [ "$USE_EXISTING_DISPLAY" = "0" ]; then
  if DISPLAY=:$DISPLAY_NUM xdpyinfo >/dev/null 2>&1; then
    echo "Display :$DISPLAY_NUM is already in use. Set REUSE_EXISTING_DISPLAY=1 to reuse it."
    exit 1
  fi
  "$XVFB_BIN" ":$DISPLAY_NUM" -screen 0 "$XVFB_SCREEN" -nolisten tcp >"$XVFB_LOG_FILE" 2>&1 &
  XVFB_PID=$!
  READY=0
  for _ in $(seq 1 50); do
    if DISPLAY=:$DISPLAY_NUM xdpyinfo >/dev/null 2>&1; then
      READY=1
      break
    fi
    sleep 0.2
  done
  if [ "$READY" -ne 1 ]; then
    echo "Failed to start Xvfb on :$DISPLAY_NUM"
    cat "$XVFB_LOG_FILE"
    exit 1
  fi
fi

setsid "$@" &
CMD_PID=$!
set +e
wait "$CMD_PID"
EXIT_CODE=$?
set -e
CMD_PID=
exit "$EXIT_CODE"
