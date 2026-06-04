#!/usr/bin/env bash
set -euo pipefail

export M2SVID_SERVICE_ROOT="${M2SVID_SERVICE_ROOT:-/workspace/m2svid_service}"
export M2SVID_OUTPUT_ROOT="${M2SVID_OUTPUT_ROOT:-/workspace/outputs/m2svid_runpod_v0.1}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export PORT="${PORT:-${GRADIO_SERVER_PORT:-7864}}"
export GRADIO_CONCURRENCY="${GRADIO_CONCURRENCY:-1}"

# Auth guard. Reject default placeholder; require explicit override on public Pods.
if [[ "${GRADIO_AUTH:-}" == "" || "${GRADIO_AUTH:-}" == "user:change-me" ]]; then
  if [[ "${ALLOW_NO_AUTH:-0}" != "1" ]]; then
    echo "[entrypoint] GRADIO_AUTH not set (or still placeholder)." >&2
    echo "  Set GRADIO_AUTH=user:strong-password to enable HTTP basic auth," >&2
    echo "  or set ALLOW_NO_AUTH=1 to launch without auth (RunPod TCP-only / private)." >&2
    exit 2
  fi
  echo "[entrypoint] WARNING: launching without auth (ALLOW_NO_AUTH=1)."
fi

IMAGE_APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${APP_DIR:-}" ]]; then
  if [[ -f /workspace/m2svid_runpod_v0.1/app.py ]]; then
    APP_DIR=/workspace/m2svid_runpod_v0.1
  else
    APP_DIR="${IMAGE_APP_DIR}"
  fi
fi
APP_PYTHON="${APP_PYTHON:-${M2SVID_SERVICE_ROOT}/.venv/bin/python}"

mkdir -p "${M2SVID_OUTPUT_ROOT}"

if [[ ! -x "${APP_PYTHON}" ]]; then
  echo "[entrypoint] ${APP_PYTHON} not found; falling back to python3"
  APP_PYTHON="$(command -v python3)"
fi

cd "${APP_DIR}"
exec "${APP_PYTHON}" app.py
