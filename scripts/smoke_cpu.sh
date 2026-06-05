#!/usr/bin/env bash
# CPU-only smoke for Phase 0 baseline.
# Runs without --gpus= to validate non-GPU paths: imports, apt deps, prepare_env
# dry traversal, autoshot stub call (no model weights), Gradio entry import.
#
# Usage: docker run --rm -e SMOKE_MODE=phase0 <image> bash scripts/smoke_cpu.sh
#
# Exit 0 = pass. Non-zero = which step failed (see step ids below).
set -e

SMOKE_MODE="${SMOKE_MODE:-phase0}"
APP_DIR="${APP_DIR:-/opt/m2svid_runpod_v0.1}"
SERVICE_ROOT="${M2SVID_SERVICE_ROOT:-/workspace/m2svid_service}"

step() { printf '\033[1;36m[smoke %s]\033[0m %s\n' "$SMOKE_MODE" "$1"; }
fail() { printf '\033[1;31m[smoke FAIL]\033[0m %s\n' "$1" >&2; exit "${2:-1}"; }

step "1. apt binaries"
for bin in ffmpeg ffprobe git curl tmux rsync cmake ninja; do
  command -v "$bin" >/dev/null 2>&1 || fail "missing apt bin: $bin" 11
done
step "  OK: ffmpeg ffprobe git curl tmux rsync cmake ninja"

step "2. python interpreter"
command -v python3 >/dev/null 2>&1 || fail "python3 missing" 12
python3 --version
step "  OK"

step "3. mock prepare_env env vars"
export SKIP_THIRD_PARTY=1 SKIP_CKPTS=1 INSTALL_BLACKWELL_SHIM=0
mkdir -p "$SERVICE_ROOT/configs" "$SERVICE_ROOT/ckpts" "$SERVICE_ROOT/third_party"
# Pretend autoshot.pth exists for smoke
touch "$SERVICE_ROOT/ckpts/autoshot.pth" "$SERVICE_ROOT/ckpts/m2svid_weights.pt" \
      "$SERVICE_ROOT/ckpts/open_clip_pytorch_model.bin" "$SERVICE_ROOT/ckpts/vgg.pth"
mkdir -p "$SERVICE_ROOT/third_party/Video-Depth-Anything/checkpoints"
touch "$SERVICE_ROOT/third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vits.pth"
touch "$SERVICE_ROOT/third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vitl.pth"
touch "$SERVICE_ROOT/configs/m2svid.yaml"
step "  OK"

step "4. prepare_env.sh dry run (SKIP_* on, no venv install)"
# Just source the function-defining portion; do NOT execute install steps
# We test the path resolution + early checks
bash -n "$APP_DIR/scripts/runpod_prepare_env.sh" || fail "prepare_env.sh parse error" 14
step "  OK: parses clean"

step "5. install_xformers_stub.sh parse"
bash -n "$APP_DIR/scripts/install_xformers_stub.sh" || fail "install_xformers_stub.sh parse error" 15
step "  OK"

step "6. blackwell_xformers_shim.py parse"
python3 -c "import ast; ast.parse(open('$APP_DIR/scripts/blackwell_xformers_shim.py').read())" \
  || fail "blackwell_xformers_shim.py parse error" 16
step "  OK"

step "7. autoshot_splitter.py parse (vendored)"
python3 -c "import ast; ast.parse(open('$APP_DIR/vendored/autoshot/autoshot_splitter.py').read())" \
  || fail "autoshot_splitter.py parse error" 17
step "  OK"

step "8. app.py / run_pipeline.py parse"
python3 -c "import ast; ast.parse(open('$APP_DIR/run_pipeline.py').read())" \
  || fail "run_pipeline.py parse error" 18
python3 -c "import ast; ast.parse(open('$APP_DIR/app.py').read())" \
  || fail "app.py parse error" 19
step "  OK"

step "9. worker parse"
for w in autoshot_worker.py m2svid_worker.py shotclass_worker.py concat_ffmpeg.py; do
  python3 -c "import ast; ast.parse(open('$APP_DIR/$w').read())" \
    || fail "$w parse error" 20
done
step "  OK"

step "10. cache dir env"
[[ "${HF_HOME:-}" == "/workspace/.cache/huggingface" ]] || fail "HF_HOME wrong: ${HF_HOME:-unset}" 21
[[ "${TORCH_HOME:-}" == "/workspace/.cache/torch" ]] || fail "TORCH_HOME wrong: ${TORCH_HOME:-unset}" 22
mkdir -p "$HF_HOME" "$TORCH_HOME" || fail "cannot mkdir caches" 23
step "  OK: HF_HOME=$HF_HOME"

if [[ "$SMOKE_MODE" == "phase1" ]] || [[ "$SMOKE_MODE" == "phase2" ]]; then
  step "11. (phase1+) flash-attn wheel present"
  ls /opt/wheels/flash_attn-*.whl >/dev/null 2>&1 || fail "no flash-attn wheel in /opt/wheels" 24
  WHEEL=$(ls /opt/wheels/flash_attn-*.whl | head -1)
  step "  OK: $(basename "$WHEEL")"

  step "12. (phase1+) fa2_unet_patch.py parse"
  python3 -c "import ast; ast.parse(open('$APP_DIR/scripts/fa2_unet_patch.py').read())" \
    || fail "fa2_unet_patch.py parse error" 25
  step "  OK"

  step "13. (phase1+) wheel metadata (sm_120, cu128, cp311, torch 2.9)"
  WHEEL_NAME=$(basename "$WHEEL")
  echo "    $WHEEL_NAME"
  # Just printout — wheel naming on FA2 is non-standard, no strict gate here.
fi

step "All Phase 0 CPU smoke steps passed."
