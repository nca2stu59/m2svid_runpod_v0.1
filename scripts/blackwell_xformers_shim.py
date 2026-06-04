"""Monkey-patch xformers.memory_efficient_attention to use torch SDPA on Blackwell.

Defensive shim for sm_120 (RTX 5090 / Pro 6000 Blackwell / B100/B200).
On Linux + xformers 0.0.33, native cutlass-blackwell kernel is usually
available — this shim is then a benign override (perf parity via cuDNN SDPA).

On non-Blackwell GPUs (sm_90 / sm_80 / etc.) shim is a no-op (capability check).

Handles BOTH tensor layouts xformers accepts:
  3D (B, M, D)        — SGM / Hi3D-Official pre-merge heads into batch
  4D (B, M, H, D)     — DINOv2 / VDA / standard multi-head

Activation: paired `_blackwell_xformers_shim.pth` in same site-packages
auto-imports this module on Python startup.

Origin: m2svid_service Windows migration (port/buildlog/2026-05-10_m2svid_cu128_blackwell_shim.md).
"""
from __future__ import annotations

import sys


def _install() -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        cap = torch.cuda.get_device_capability(0)
    except Exception:
        return
    if cap[0] < 12:
        return  # not Blackwell, leave xformers alone

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
