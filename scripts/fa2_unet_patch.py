"""Phase 1 — Flash-attn 2 UNet attention monkey-patch (module-name gated).

Routes m2svid SGM (Hi3D-Official) UNet attention through flash-attn 2.x while
leaving VAE / temporal_ae attention on PyTorch SDPA. The VAE has head_dim=512
and a different memory pattern that Sage/FA2 don't accelerate cleanly (vault
DO_NOT.md §A.1 — Blackwell SVD+VAE NaN root cause is the VAE mid-block).

Gate strategy:
  - Patch only `MemoryEfficientCrossAttention.forward` and `CrossAttention.forward`
    inside `sgm.modules.attention`.
  - Inside the patched function, inspect the call site via `self`'s module class
    name; if the module path contains `first_stage` / `autoencoder` / `temporal_ae`
    / `vae`, fall back to original SDPA path.

Activation: `_fa2_unet_patch.pth` in the same venv site-packages auto-imports
this module at Python startup.

Disabled when:
  - flash_attn import fails (FA2 wheel missing or wrong ABI)
  - env var FA2_UNET_PATCH=0
  - GPU capability < (8, 0) — FA2 needs Ampere or newer
"""
from __future__ import annotations

import os
import sys


def _install() -> None:
    if os.environ.get("FA2_UNET_PATCH", "1").lower() in ("0", "no", "false", "off"):
        return

    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        cap = torch.cuda.get_device_capability(0)
        if cap[0] < 8:
            return  # Need Ampere or newer for FA2
    except Exception:
        return

    try:
        from flash_attn import flash_attn_func  # noqa: F401
    except Exception as e:
        print(f"[fa2_unet_patch] flash_attn import failed: {e}", file=sys.stderr)
        return

    # Defer SGM attention import — only patch when m2svid loads.
    import importlib

    def _is_vae_path(module) -> bool:
        try:
            qualname = type(module).__module__ + "." + type(module).__qualname__
            qualname_low = qualname.lower()
            for marker in ("first_stage", "autoencod", "temporal_ae", "vae", "decoder", "encoder"):
                if marker in qualname_low:
                    return True
        except Exception:
            pass
        return False

    def _patch_mea(orig_forward):
        from flash_attn import flash_attn_func

        def patched(self, x, context=None, mask=None, additional_tokens=None,
                    n_times_crossframe_attn_in_self=0):
            if _is_vae_path(self) or mask is not None or additional_tokens is not None:
                return orig_forward(self, x, context=context, mask=mask,
                                    additional_tokens=additional_tokens,
                                    n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self)
            try:
                h = self.heads
                q = self.to_q(x)
                ctx_in = context if context is not None else x
                k = self.to_k(ctx_in)
                v = self.to_v(ctx_in)
                b, m, d = q.shape
                head_dim = d // h
                q = q.view(b, m, h, head_dim)
                k = k.view(b, k.shape[1], h, head_dim)
                v = v.view(b, v.shape[1], h, head_dim)
                # flash_attn_func: (B, M, H, D) fp16/bf16, returns (B, M, H, D)
                out = flash_attn_func(q, k, v, causal=False, dropout_p=0.0)
                out = out.reshape(b, m, d)
                return self.to_out(out)
            except Exception:
                return orig_forward(self, x, context=context, mask=mask,
                                    additional_tokens=additional_tokens,
                                    n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self)

        return patched

    # SGM attention path is in third_party/Hi3D-Official.
    # Import lazily; if SGM not on sys.path yet, register a meta-path hook.
    try:
        mod = importlib.import_module("sgm.modules.attention")
    except Exception:
        # Register a finder that patches on first SGM import.
        class _Hook:
            def find_module(self, fullname, path=None):
                if fullname == "sgm.modules.attention":
                    return self
                return None

            def load_module(self, fullname):
                sys.meta_path.remove(self)
                module = importlib.import_module(fullname)
                _do_patch(module)
                return module

        sys.meta_path.insert(0, _Hook())
        return

    _do_patch(mod)


def _do_patch(mod) -> None:
    try:
        if hasattr(mod, "MemoryEfficientCrossAttention"):
            cls = mod.MemoryEfficientCrossAttention
            if not getattr(cls, "_fa2_patched", False):
                from importlib import import_module as _im  # noqa: F401
                orig = cls.forward
                cls.forward = _make_patched(cls, orig)
                cls._fa2_patched = True
        if hasattr(mod, "CrossAttention"):
            cls = mod.CrossAttention
            if not getattr(cls, "_fa2_patched", False):
                orig = cls.forward
                cls.forward = _make_patched(cls, orig)
                cls._fa2_patched = True
    except Exception as e:
        print(f"[fa2_unet_patch] patch apply failed: {e}", file=sys.stderr)


def _make_patched(cls_unused, orig_forward):
    import sys as _sys

    from flash_attn import flash_attn_func

    def _is_vae(self):
        try:
            qualname = type(self).__module__ + "." + type(self).__qualname__
            qualname_low = qualname.lower()
            return any(m in qualname_low for m in
                       ("first_stage", "autoencod", "temporal_ae", "vae"))
        except Exception:
            return False

    def patched(self, x, context=None, mask=None, additional_tokens=None,
                n_times_crossframe_attn_in_self=0):
        # Conservative: fall back when any complex bias/mask path active.
        if _is_vae(self) or mask is not None or additional_tokens is not None \
                or n_times_crossframe_attn_in_self:
            return orig_forward(self, x, context=context, mask=mask,
                                additional_tokens=additional_tokens,
                                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self)
        try:
            h = self.heads
            q = self.to_q(x)
            ctx_in = context if context is not None else x
            k = self.to_k(ctx_in)
            v = self.to_v(ctx_in)
            b, m, d = q.shape
            head_dim = d // h
            q = q.view(b, m, h, head_dim)
            k = k.view(b, k.shape[1], h, head_dim)
            v = v.view(b, v.shape[1], h, head_dim)
            out = flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(),
                                  causal=False, dropout_p=0.0)
            out = out.reshape(b, m, d)
            return self.to_out(out)
        except Exception as e:
            print(f"[fa2_unet_patch] runtime fallback: {e}", file=_sys.stderr)
            return orig_forward(self, x, context=context, mask=mask,
                                additional_tokens=additional_tokens,
                                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self)

    return patched


_install()
