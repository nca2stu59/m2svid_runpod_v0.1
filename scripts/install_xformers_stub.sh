#!/usr/bin/env bash
# Replace broken xformers wheel with a stub package that routes
# memory_efficient_attention through PyTorch SDPA (cuDNN flash on Hopper / Blackwell).
#
# Use when the PyPI xformers wheel cannot load its C++ extension on this env
# (ABI mismatch: cu130 vs cu128, cp310 vs cp311, torch 2.10 vs 2.9, etc.).
#
# The stub presents the same import surface that m2svid (SGM / Hi3D-Official)
# and DINOv2 / VDA call:
#   xformers.ops.memory_efficient_attention(q, k, v, ...)
#   xformers.ops.fmha.memory_efficient_attention(q, k, v, ...)
#
# Both 3D (B*H, M, D) and 4D (B, M, H, D) layouts handled.
set -euo pipefail

SERVICE_ROOT="${M2SVID_SERVICE_ROOT:-/workspace/m2svid_service}"

apply_stub() {
  local venv_dir="$1"
  local py="${venv_dir}/bin/python"
  if [[ ! -x "${py}" ]]; then
    echo "[stub] skip ${venv_dir} (no python)"
    return
  fi
  local site
  site="$(${py} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  echo "[stub] target site-packages: ${site}"

  # 1. Remove broken xformers (best effort)
  "${py}" -m pip uninstall -y xformers >/dev/null 2>&1 || true

  # 2. Wipe any leftover xformers/ dirs (sometimes pip leaves debris)
  rm -rf "${site}/xformers" "${site}"/xformers-*.dist-info 2>/dev/null || true

  # 3. Write stub package
  mkdir -p "${site}/xformers/ops/fmha"
  cat > "${site}/xformers/__init__.py" <<'PY'
"""Stub xformers package — routes attention through PyTorch SDPA.

The real xformers wheel for this Python/torch/CUDA combo was unavailable or
broken (cu130 wheel on cu128 env, cp310 wheel on cp311, etc.). PyTorch SDPA
with cuDNN flash backend gives parity-or-better perf on Hopper (sm_90) and
Blackwell (sm_120) for SVD-class video diffusion workloads.
"""
__version__ = "0.0.33-stub"
from . import ops  # noqa: F401
PY

  cat > "${site}/xformers/ops/__init__.py" <<'PY'
from .fmha import memory_efficient_attention
__all__ = ["memory_efficient_attention", "fmha"]
from . import fmha  # noqa: F401
PY

  cat > "${site}/xformers/ops/fmha/__init__.py" <<'PY'
"""xformers.ops.fmha stub — SDPA implementation.

Handles both tensor layouts xformers accepts:
  3D (B*H, M, D)    — SGM / Hi3D-Official (heads pre-merged into batch)
  4D (B, M, H, D)   — DINOv2 / VDA / standard multi-head
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0,
                               scale=None, op=None):
    attn_mask = attn_bias if isinstance(attn_bias, torch.Tensor) else None
    if query.dim() == 3:
        q = query.unsqueeze(1)
        k = key.unsqueeze(1)
        v = value.unsqueeze(1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=p, scale=scale,
        )
        return out.squeeze(1)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    if not q.is_contiguous(): q = q.contiguous()
    if not k.is_contiguous(): k = k.contiguous()
    if not v.is_contiguous(): v = v.contiguous()
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=p, scale=scale,
    )
    return out.transpose(1, 2).contiguous()


def memory_efficient_attention_forward(*args, **kwargs):
    return memory_efficient_attention(*args, **kwargs)


__all__ = ["memory_efficient_attention", "memory_efficient_attention_forward"]
PY

  # 4. Verify
  "${py}" - <<'PY'
import xformers
from xformers.ops import memory_efficient_attention as mea
import xformers.ops.fmha as fmha
print(f"[stub-verify] xformers={xformers.__version__} mea={mea.__module__}")
import torch
if torch.cuda.is_available():
    q = torch.randn(2, 1024, 8, 64, device='cuda', dtype=torch.float16)
    out = mea(q, q, q)
    print(f"[stub-verify] 4D attn OK shape={tuple(out.shape)}")
    q3 = torch.randn(16, 256, 64, device='cuda', dtype=torch.float16)
    out3 = mea(q3, q3, q3)
    print(f"[stub-verify] 3D attn OK shape={tuple(out3.shape)}")
PY

  # 5. Remove shim .pth + module (stub takes over the same role,
  # leaving shim active would cause double-patch).
  rm -f "${site}/_blackwell_xformers_shim.pth" \
        "${site}/blackwell_xformers_shim.py" 2>/dev/null || true
  echo "[stub] done: ${venv_dir}"
}

apply_stub "${SERVICE_ROOT}/.venv"
apply_stub "${SERVICE_ROOT}/.venv-vda"
[[ -d "${SERVICE_ROOT}/.venv-flashdepth" ]]  && apply_stub "${SERVICE_ROOT}/.venv-flashdepth"
[[ -d "${SERVICE_ROOT}/.venv-depthcrafter" ]] && apply_stub "${SERVICE_ROOT}/.venv-depthcrafter"

echo "[stub] all venvs patched. Real xformers replaced with SDPA stub."
