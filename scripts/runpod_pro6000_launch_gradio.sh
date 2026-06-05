#!/usr/bin/env bash
# Launch Gradio only after Pro6000 preflight passes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROFILE="${PROFILE:-${APP_DIR}/runpod_profiles/pro6000-safe.env}"

if [[ -f "${PROFILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${PROFILE}"
  set +a
fi

export M2SVID_SERVICE_ROOT="${M2SVID_SERVICE_ROOT:-/workspace/m2svid_service}"
export M2SVID_OUTPUT_ROOT="${M2SVID_OUTPUT_ROOT:-/workspace/outputs/m2svid_runpod_v0.1}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export PORT="${PORT:-7864}"
export GRADIO_CONCURRENCY="${GRADIO_CONCURRENCY:-1}"
export BLACKWELL_SHIM_FORCE="${BLACKWELL_SHIM_FORCE:-1}"

if [[ -z "${GRADIO_AUTH:-}" && "${ALLOW_NO_AUTH:-0}" != "1" ]]; then
  echo "[pro6000-launch] GRADIO_AUTH is required. Example: export GRADIO_AUTH=user:strong-password" >&2
  exit 20
fi

"${M2SVID_SERVICE_ROOT}/.venv/bin/python" "${APP_DIR}/scripts/check_runpod_paths.py"
exec "${APP_DIR}/runpod_entrypoint.sh"
