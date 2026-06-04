#!/usr/bin/env bash
# RunPod one-time env prepare for m2svid_runpod_v0.1.
#
# Idempotent steps:
#   1. Create venvs (.venv, .venv-vda; optional .venv-flashdepth, .venv-depthcrafter)
#   2. Install torch 2.9.0+cu128 + project requirements
#   3. Clone third_party repos at pinned commits
#   4. Download ckpts (skip if present)
#   5. Deploy Blackwell xformers SDPA shim (auto-detect sm_120)
#   6. Verify (xformers.info, paths, GPU capability)
#
# Env overrides:
#   M2SVID_SERVICE_ROOT     /workspace/m2svid_service
#   PYTHON_BIN              python3 (RunPod base image default; py3.11 stable, py3.12 dev)
#   TORCH_INDEX_URL         https://download.pytorch.org/whl/cu128
#   TORCH_VERSION           2.9.0
#   INSTALL_FLASHDEPTH      0 (set 1 to add .venv-flashdepth; needs cp310/cp312 kurogane wheels)
#   INSTALL_DEPTHCRAFTER    0
#   INSTALL_BLACKWELL_SHIM  auto | 0 | 1 (auto = nvidia-smi capability check)
#   SKIP_CKPTS              0 (set 1 to skip downloads, e.g. when volume already has ckpts)
#   SKIP_THIRD_PARTY        0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SERVICE_ROOT="${M2SVID_SERVICE_ROOT:-/workspace/m2svid_service}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.9.0}"
INSTALL_FLASHDEPTH="${INSTALL_FLASHDEPTH:-0}"
INSTALL_DEPTHCRAFTER="${INSTALL_DEPTHCRAFTER:-0}"
INSTALL_BLACKWELL_SHIM="${INSTALL_BLACKWELL_SHIM:-auto}"
SKIP_CKPTS="${SKIP_CKPTS:-0}"
SKIP_THIRD_PARTY="${SKIP_THIRD_PARTY:-0}"

SHIM_SRC="${SCRIPT_DIR}/blackwell_xformers_shim.py"

log() { printf '\033[1;36m[prepare_env]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; }

mkdir -p "${SERVICE_ROOT}" /workspace/outputs

# ---- step 1+2: venvs + requirements ----------------------------------------
create_env() {
  local env_dir="$1"
  if [[ ! -x "${env_dir}/bin/python" ]]; then
    log "create venv: ${env_dir}"
    "${PYTHON_BIN}" -m venv "${env_dir}"
  else
    log "venv exists: ${env_dir}"
  fi
  "${env_dir}/bin/python" -m pip install --upgrade pip wheel setuptools
  "${env_dir}/bin/python" -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"
}

log "=== step 1+2: venvs + requirements ==="
create_env "${SERVICE_ROOT}/.venv"
"${SERVICE_ROOT}/.venv/bin/python" -m pip install -r "${APP_DIR}/requirements-runpod-app.txt"
"${SERVICE_ROOT}/.venv/bin/python" -m pip install -r "${APP_DIR}/requirements-m2svid-runpod.txt"

create_env "${SERVICE_ROOT}/.venv-vda"
"${SERVICE_ROOT}/.venv-vda/bin/python" -m pip install -r "${APP_DIR}/requirements-vda-runpod.txt"

if [[ "${INSTALL_FLASHDEPTH}" == "1" ]]; then
  warn "FlashDepth venv requires cp310/cp312 + kurogane Linux wheels — verify upstream."
  warn "Skipping automated install for now; manual setup expected."
fi
if [[ "${INSTALL_DEPTHCRAFTER}" == "1" ]]; then
  create_env "${SERVICE_ROOT}/.venv-depthcrafter"
  warn "DepthCrafter requirements not bundled in this repo — install from upstream m2svid_service."
fi

# ---- step 3: m2svid_service vendored source (Hi3D + bidavideo + RAFT + ...) -
# Pinned upstream repos (vendored in m2svid_service):
#   Hi3D-Official, bidavideo, RAFT, pytorch-msssim, mamba — NOT in github clone list.
# External repos (still cloned from upstream):
#   DepthCrafter   github.com/Tencent/DepthCrafter         v1.0.1  fc83d365
#   Video-Depth-Anything  github.com/DepthAnything/...     4f5ae231
#   AutoShot       github.com/wentaozhu/AutoShot           77c82ff8
#   FlashDepth     github.com/Eyeline-Labs/FlashDepth      3e08f313  (INSTALL_FLASHDEPTH=1)
M2SVID_SERVICE_REPO="${M2SVID_SERVICE_REPO:-https://github.com/nca2stu59/m2svid_service.git}"

clone_repo() {
  local url="$1"
  local dest="$2"
  local sha="${3:-}"
  if [[ -d "${dest}/.git" ]]; then
    log "third_party present: ${dest}"
    return
  fi
  mkdir -p "$(dirname "${dest}")"
  log "clone: ${url} -> ${dest}"
  git clone "${url}" "${dest}"
  if [[ -n "${sha}" ]]; then
    (cd "${dest}" && git checkout "${sha}")
  fi
}

clone_m2svid_service() {
  if [[ -f "${SERVICE_ROOT}/configs/m2svid.yaml" ]]; then
    log "m2svid_service vendored source already present"
    return
  fi
  if [[ -d "${SERVICE_ROOT}/.git" ]]; then
    log "m2svid_service .git exists, pull"
    (cd "${SERVICE_ROOT}" && git pull --ff-only)
    return
  fi
  # Source missing but dir may already hold venvs/ckpts from previous partial run.
  # Clone to temp, overlay code into service root (preserve .venv*/ckpts/outputs).
  local tmp
  tmp="$(mktemp -d)"
  log "clone m2svid_service to ${tmp}/src"
  git clone "${M2SVID_SERVICE_REPO}" "${tmp}/src"
  log "overlay code -> ${SERVICE_ROOT} (preserve .venv*, ckpts/, outputs/)"
  if ! command -v rsync >/dev/null 2>&1; then
    apt-get install -y rsync
  fi
  rsync -a \
    --exclude='.venv' --exclude='.venv-*' --exclude='.venv.bak' --exclude='.venv-*.bak' \
    --exclude='ckpts/' --exclude='outputs/' --exclude='runs/' --exclude='temp/' --exclude='work/' \
    "${tmp}/src/" "${SERVICE_ROOT}/"
  rm -rf "${tmp}"
}

if [[ "${SKIP_THIRD_PARTY}" != "1" ]]; then
  log "=== step 3: m2svid_service source + external third_party ==="
  clone_m2svid_service

  # External repos (NOT vendored)
  clone_repo "https://github.com/DepthAnything/Video-Depth-Anything.git" \
    "${SERVICE_ROOT}/third_party/Video-Depth-Anything" \
    "4f5ae23172ba60fd7bc11ef671cca678842c7072"
  clone_repo "https://github.com/wentaozhu/AutoShot.git" \
    "${SERVICE_ROOT}/third_party/AutoShot" \
    "77c82ff826a9301bb173d9be786297a49d73d081"

  if [[ "${INSTALL_DEPTHCRAFTER}" == "1" ]]; then
    clone_repo "https://github.com/Tencent/DepthCrafter.git" \
      "${SERVICE_ROOT}/third_party/DepthCrafter_new" \
      "fc83d365f2b781ab05aeb94b13f7e97417df7d97"
  fi
  if [[ "${INSTALL_FLASHDEPTH}" == "1" ]]; then
    clone_repo "https://github.com/Eyeline-Labs/FlashDepth.git" \
      "${SERVICE_ROOT}/third_party/FlashDepth" \
      "3e08f313b9f1b08efde5e6ebacc671a173cb9f36"
  fi
else
  log "SKIP_THIRD_PARTY=1 — skipping clones"
fi

# ---- step 4: ckpts ---------------------------------------------------------
download_ckpt() {
  local label="$1"
  local url="$2"
  local dest="$3"
  if [[ -f "${dest}" ]]; then
    log "ckpt present: ${dest}"
    return
  fi
  mkdir -p "$(dirname "${dest}")"
  log "download ${label}: ${url}"
  curl -L --fail --retry 3 -o "${dest}.partial" "${url}"
  mv "${dest}.partial" "${dest}"
}

if [[ "${SKIP_CKPTS}" != "1" ]]; then
  log "=== step 4: ckpts ==="
  download_ckpt "m2svid_weights" \
    "https://storage.googleapis.com/gresearch/m2svid/m2svid_weights.pt" \
    "${SERVICE_ROOT}/ckpts/m2svid_weights.pt"
  download_ckpt "open_clip" \
    "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin" \
    "${SERVICE_ROOT}/ckpts/open_clip_pytorch_model.bin"
  download_ckpt "vgg_lpips" \
    "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1" \
    "${SERVICE_ROOT}/ckpts/vgg.pth"
  download_ckpt "VDA-S" \
    "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth" \
    "${SERVICE_ROOT}/third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vits.pth"
  download_ckpt "VDA-L" \
    "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth" \
    "${SERVICE_ROOT}/third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vitl.pth"
  if [[ ! -f "${SERVICE_ROOT}/ckpts/autoshot.pth" ]]; then
    warn "autoshot.pth missing — Baidu Pan manual download required."
    warn "  https://pan.baidu.com/s/1CdCVNzFdF3U6I4ajfejYNQ (passcode: sfkq)"
    warn "  Place as ${SERVICE_ROOT}/ckpts/autoshot.pth (shot detection disabled if absent)."
  fi
else
  log "SKIP_CKPTS=1 — skipping downloads"
fi

# ---- step 5: Blackwell shim deploy ------------------------------------------
detect_blackwell() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  local caps
  caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  case "${caps}" in
    12.*) return 0 ;;
    *)    return 1 ;;
  esac
}

deploy_shim_to_venv() {
  local venv_dir="$1"
  local site
  site="$(${venv_dir}/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || echo "")"
  if [[ -z "${site}" || ! -d "${site}" ]]; then
    warn "site-packages not found for ${venv_dir}, skipping shim"
    return
  fi
  cp "${SHIM_SRC}" "${site}/blackwell_xformers_shim.py"
  printf 'import blackwell_xformers_shim\n' > "${site}/_blackwell_xformers_shim.pth"
  log "shim deployed -> ${site}"
}

case "${INSTALL_BLACKWELL_SHIM}" in
  1)    SHIM_ACTIVE=1 ;;
  0)    SHIM_ACTIVE=0 ;;
  auto) if detect_blackwell; then SHIM_ACTIVE=1; else SHIM_ACTIVE=0; fi ;;
  *)    err "INSTALL_BLACKWELL_SHIM must be auto|0|1"; exit 1 ;;
esac

if [[ "${SHIM_ACTIVE}" == "1" ]]; then
  log "=== step 5: deploy Blackwell shim (sm_120 detected or forced) ==="
  deploy_shim_to_venv "${SERVICE_ROOT}/.venv"
  deploy_shim_to_venv "${SERVICE_ROOT}/.venv-vda"
  [[ -d "${SERVICE_ROOT}/.venv-flashdepth" ]] && deploy_shim_to_venv "${SERVICE_ROOT}/.venv-flashdepth"
  [[ -d "${SERVICE_ROOT}/.venv-depthcrafter" ]] && deploy_shim_to_venv "${SERVICE_ROOT}/.venv-depthcrafter"
else
  log "Blackwell shim skipped (non-sm_120 GPU or explicitly disabled)"
fi

# ---- step 6: verify ---------------------------------------------------------
log "=== step 6: verify ==="
"${SERVICE_ROOT}/.venv/bin/python" - <<'PY'
import sys
try:
    import torch
    print(f"  torch: {torch.__version__} cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"  gpu:   {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    else:
        print("  gpu:   not available (CPU-only env)")
except Exception as e:
    print(f"  torch import failed: {e}", file=sys.stderr); sys.exit(2)
try:
    import xformers
    print(f"  xformers: {xformers.__version__}")
    from xformers.ops import memory_efficient_attention as mea
    print(f"  mea module: {mea.__module__}")
except Exception as e:
    print(f"  xformers import failed: {e}", file=sys.stderr); sys.exit(3)
PY

if [[ -x "${SERVICE_ROOT}/.venv/bin/python" ]]; then
  M2SVID_SERVICE_ROOT="${SERVICE_ROOT}" \
    "${SERVICE_ROOT}/.venv/bin/python" "${APP_DIR}/scripts/check_runpod_paths.py" || true
fi

log "done"
log "  app python:    ${SERVICE_ROOT}/.venv/bin/python"
log "  vda python:    ${SERVICE_ROOT}/.venv-vda/bin/python"
log "  service root:  ${SERVICE_ROOT}"
log "  Blackwell shim active: ${SHIM_ACTIVE}"
log "next: ./runpod_entrypoint.sh"
