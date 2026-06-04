"""Monkey-patch xformers.memory_efficient_attention to use torch SDPA.

Activation criteria (any of):
  1. SM 12.x (Blackwell — sm_120 RTX 5090 / Pro 6000 / B100/B200) — xformers
     prebuilt wheels lack Blackwell cutlass kernels.
  2. xformers C++ extension fails to load (ABI mismatch — cu130 vs cu128,
     cp310 vs cp311, torch 2.10 vs 2.9, etc.). Common on RunPod Linux pods
     because PyPI xformers 0.0.33 ships cu130/cp310 bytes but advertises
     wider compat.
  3. Live attention call raises (any reason).

If xformers loads + runs an attention call cleanly, shim is a no-op.

PyTorch SDPA + cuDNN flash-attn supports sm_80+ (A100/H100/H200) and
sm_120 (Blackwell) and gives parity-or-better perf for SVD-class video
diffusion. xformers fallback path is uniformly equivalent or slower.

Handles BOTH tensor layouts xformers accepts:
  3D (B, M, D)        — SGM / Hi3D-Official pre-merge heads into batch
  4D (B, M, H, D)     — DINOv2 / VDA / standard multi-head

Activation: paired `_blackwell_xformers_shim.pth` in same site-packages
auto-imports this module on Python startup.

Origin: m2svid_service Windows migration (port/buildlog/2026-05-10_m2svid_cu128_blackwell_shim.md).
"""
from __future__ import annotations

import os
import sys


def _xformers_works() -> bool:
    """Smoke-test xformers C++ extension + a tiny attention call."""
    try:
        import torch
        from xformers.ops import memory_efficient_attention as _mea
        # Trigger the C++ extension. If it's broken the warning fires here.
        if not torch.cuda.is_available():
            return False
        q = torch.zeros(1, 8, 4, 32, device="cuda", dtype=torch.float16)
        _mea(q, q, q)
        return True
    except Exception:
        return False


def _install() -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return

    force = os.environ.get("BLACKWELL_SHIM_FORCE", "").lower() in ("1", "yes", "true", "on")
    disable = os.environ.get("BLACKWELL_SHIM_DISABLE", "").lower() in ("1", "yes", "true", "on")
    if disable:
        return

    is_blackwell = False
    try:
        cap = torch.cuda.get_device_capability(0)
        is_blackwell = cap[0] >= 12
    except Exception:
        pass

    # Decide whether to patch.
    if not force and not is_blackwell:
        # Non-Blackwell: patch only if xformers is actually broken.
        if _xformers_works():
            return

    try:
        import xformers.ops as _xops
        import xformers.ops.fmha as _xfmha
    except Exception as e:
        print(f"[blackwell_xformers_shim] xformers import failed: {e}", file=sys.stderr)
        return

    import torch.nn.functional as F

    def _sdpa_mea(query, key, value, attn_bias=None, p=0.0, scale=None, op=None):
        attn_mask = attn_bias if isinstance(attn_bias, torch.Tensor) else None
        if query.dim() == 3:
            # (B*H, M, D) — heads pre-merged into batch (SGM pattern)
            q = query.unsqueeze(1)
            k = key.unsqueeze(1)
            v = value.unsqueeze(1)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=p, scale=scale,
            )
            return out.squeeze(1)
        # 4D (B, M, H, D)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=p, scale=scale,
        )
        return out.transpose(1, 2).contiguous()

    _xops.memory_efficient_attention = _sdpa_mea
    _xfmha.memory_efficient_attention = _sdpa_mea


_install()
