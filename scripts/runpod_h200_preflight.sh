#!/usr/bin/env bash
# H200 paid-run gate. Run on the Pod after prepare_env finishes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROFILE="${PROFILE:-${APP_DIR}/runpod_profiles/h200-safe.env}"

if [[ -f "${PROFILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${PROFILE}"
  set +a
fi

export M2SVID_SERVICE_ROOT="${M2SVID_SERVICE_ROOT:-/workspace/m2svid_service}"
export M2SVID_OUTPUT_ROOT="${M2SVID_OUTPUT_ROOT:-/workspace/outputs/m2svid_runpod_v0.1}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export INSTALL_FA2="${INSTALL_FA2:-0}"
export INSTALL_BLACKWELL_SHIM="${INSTALL_BLACKWELL_SHIM:-0}"

APP_PY="${M2SVID_SERVICE_ROOT}/.venv/bin/python"
VDA_PY="${M2SVID_SERVICE_ROOT}/.venv-vda/bin/python"

log() { printf '\033[1;36m[h200-preflight]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[h200-preflight]\033[0m %s\n' "$*" >&2; }

log "profile=${PROFILE}"
log "service=${M2SVID_SERVICE_ROOT}"
log "output=${M2SVID_OUTPUT_ROOT}"

log "1. nvidia-smi + H200 capability"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  err "nvidia-smi not found; this is not a GPU Pod"
  exit 10
fi
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

log "2. required paths"
"${APP_PY:-python3}" "${APP_DIR}/scripts/check_runpod_paths.py"

log "3. torch/cuda H200 smoke"
"${APP_PY}" - <<'PY'
import sys
import torch

print(f"torch={torch.__version__} cuda={torch.version.cuda}")
if not torch.cuda.is_available():
    print("cuda not available", file=sys.stderr)
    sys.exit(11)
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"gpu={name} cap={cap} mem_gb={mem_gb:.1f}")
if cap[0] != 9:
    print(f"expected Hopper/H200 cap major 9, got {cap}", file=sys.stderr)
    sys.exit(12)
x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
print(f"matmul_ok mean={float(y.float().mean()):.6f}")
PY

log "4. app imports"
"${APP_PY}" - <<'PY'
mods = [
    "gradio",
    "diffusers",
    "transformers",
    "open_clip",
    "cv2",
    "decord",
    "transnetv2_pytorch",
]
for name in mods:
    __import__(name)
    print(f"import_ok {name}")
PY

log "5. xformers/SDPA attention smoke"
"${APP_PY}" - <<'PY'
import sys
import torch

try:
    import xformers
    from xformers.ops import memory_efficient_attention as mea
    print(f"xformers={getattr(xformers, '__version__', 'unknown')} module={mea.__module__}")
    q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.float16)
    out = mea(q, q, q)
    torch.cuda.synchronize()
    print(f"xformers_attention_ok shape={tuple(out.shape)}")
except Exception as e:
    print(f"xformers_attention_failed {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(13)
PY

log "6. VDA venv import smoke"
"${VDA_PY}" - <<'PY'
import torch
import cv2
import transformers
print(f"vda_torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
print("vda_imports_ok")
PY

log "7. AutoShot weight sanity"
"${APP_PY}" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["M2SVID_SERVICE_ROOT"])
p = root / "ckpts" / "autoshot.pth"
size = p.stat().st_size if p.exists() else 0
print(f"autoshot={p} size={size}")
if size < 1024 * 1024:
    raise SystemExit("autoshot.pth missing or too small")
PY

log "PREFLIGHT_OK"
