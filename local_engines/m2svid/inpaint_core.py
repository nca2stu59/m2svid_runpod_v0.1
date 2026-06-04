"""Core M2SVid inpaint functions, usable from gradio_app (in-process)
or from inpaint_worker.py (subprocess-per-shot for VRAM=0 guarantee).

v0.16m vendoring notes
----------------------
This file lives at port/stereo_pipeline_v0.16m/local_engines/m2svid/inpaint_core.py.
The actual runtime (sgm package, third_party/Hi3D-Official + pytorch-msssim,
configs/, ckpts/) is shared with m2svid_service to avoid duplicating ~9 GB of
weights and a 4-venv dependency stack.

Resolution order:
  1. M2SVID_SERVICE_ROOT env var (explicit override)
  2. C:\\Users\\PC\\Desktop\\m2svid_service (pinned default — present on this PC)
  3. ROOT (fall back to vendored layout if user actually copies everything)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent

# Resolve where m2svid_service ACTUALLY lives so we can find its third_party/,
# configs/, and ckpts/ without duplicating them into the app tree.
_SERVICE_ROOT_DEFAULT = (
    Path(r"C:\Users\PC\Desktop\m2svid_service")
    if os.name == "nt" else Path("/workspace/m2svid_service")
)
SERVICE_ROOT = Path(
    os.environ.get("M2SVID_SERVICE_ROOT")
    or os.environ.get("RUNPOD_M2SVID_SERVICE")
    or _SERVICE_ROOT_DEFAULT
).resolve()

# Path resolver — try service first, then vendored ROOT.
def _resolve(*relative: str) -> Path:
    for base in (SERVICE_ROOT, ROOT):
        cand = base.joinpath(*relative)
        if cand.exists():
            return cand
    # Final: return service-rooted path (caller will hit FileNotFoundError if missing)
    return SERVICE_ROOT.joinpath(*relative)

for p in [
    str(ROOT),
    str(SERVICE_ROOT / "third_party" / "Hi3D-Official"),
    str(SERVICE_ROOT / "third_party" / "pytorch-msssim"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import ffmpeg
from torchvision import transforms
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from sgm.util import instantiate_from_config
from m2svid.utils.video_utils import get_video_fps
from m2svid.data.utils import get_video_frames, apply_closing, apply_dilation

CHUNK_SIZE = 25  # m2svid trained on 25-frame windows

DEFAULT_CONFIG = _resolve("configs", "m2svid.yaml")
DEFAULT_CKPT = _resolve("ckpts", "m2svid_weights.pt")

_MODEL = None
_CLIP_OFFLOAD_DISABLED = os.environ.get("M2SVID_DISABLE_CLIP_OFFLOAD", "0") == "1"


def _attach_clip_offload(model, log_fn=print) -> bool:
    """v0.17 Option A — CPU offload for FrozenOpenCLIP vision encoder.

    The vision encoder (~683M params, ~1.4 GB fp16) is invoked exactly once per
    chunk inside model.conditioner.get_unconditional_conditioning() to extract
    cond_frames embeddings. After that single forward pass the cached
    embeddings drive all 25 denoising steps; the encoder weights just sit on
    GPU consuming VRAM.

    We wrap that conditioner call so the CLIP image encoder is moved to GPU
    only for the duration of the call, then evicted to CPU + cuda cache cleared
    immediately after. Net effect: ~1.4 GB freed during the heavy denoising
    loop, allowing higher processing dim within the 32 GB VRAM budget.

    Returns True if the hook was successfully installed, False otherwise
    (so we don't crash if the embedder layout changes).
    """
    if _CLIP_OFFLOAD_DISABLED:
        log_fn("[v0.17 offload] CLIP offload disabled via M2SVID_DISABLE_CLIP_OFFLOAD=1")
        return False

    image_embedder = None
    try:
        embedders = model.conditioner.embedders
    except AttributeError:
        log_fn("[v0.17 offload] WARN: model.conditioner.embedders not found, skip offload")
        return False

    for emb in embedders:
        # FrozenOpenCLIPImageEmbedder / FrozenOpenCLIPImagePredictionEmbedder
        if hasattr(emb, "open_clip") or emb.__class__.__name__.lower().startswith("frozenopen"):
            image_embedder = emb
            break

    if image_embedder is None:
        log_fn("[v0.17 offload] WARN: no FrozenOpenCLIP embedder found, skip offload")
        return False

    # Locate the actual nn.Module holding vision weights (varies by class).
    target_module = None
    for attr_name in ("open_clip", "model", "transformer"):
        sub = getattr(image_embedder, attr_name, None)
        if sub is not None and hasattr(sub, "to") and hasattr(sub, "parameters"):
            target_module = sub
            break

    if target_module is None:
        log_fn("[v0.17 offload] WARN: open_clip submodule not located, skip offload")
        return False

    # Initial eviction so VRAM is freed before first chunk runs
    try:
        target_module.to("cpu")
        torch.cuda.empty_cache()
    except Exception as e:
        log_fn(f"[v0.17 offload] WARN: initial CPU move failed ({e}), skip offload")
        return False

    orig_fn = model.conditioner.get_unconditional_conditioning

    def wrapped(*args, **kwargs):
        target_module.to("cuda")
        try:
            return orig_fn(*args, **kwargs)
        finally:
            target_module.to("cpu")
            torch.cuda.empty_cache()

    model.conditioner.get_unconditional_conditioning = wrapped
    log_fn(
        f"[v0.17 offload] CLIP vision encoder offloaded to CPU "
        f"(class={image_embedder.__class__.__name__}, target={target_module.__class__.__name__})"
    )
    return True


def get_model(config_path: Optional[Path] = None, ckpt_path: Optional[Path] = None,
              log_fn=print):
    global _MODEL
    if _MODEL is None:
        cfg_p = Path(config_path) if config_path else DEFAULT_CONFIG
        ck_p = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
        cfg = OmegaConf.load(cfg_p)
        m = instantiate_from_config(cfg.model).cpu()
        m.init_from_ckpt(str(ck_p))
        _MODEL = m.cuda().half().eval()
        _attach_clip_offload(_MODEL, log_fn=log_fn)
    return _MODEL


def _pad_last_chunk(t: torch.Tensor, pad_t: int) -> torch.Tensor:
    if pad_t <= 0:
        return t
    last = t[:, :, -1:, :, :].expand(-1, -1, pad_t, -1, -1)
    return torch.cat([t, last], dim=2)


def generate_chunked(model, batch: dict, chunk_size: int = CHUNK_SIZE,
                     log_fn=print) -> torch.Tensor:
    """Run model.generate over the time axis in chunks of `chunk_size` frames."""
    T = batch["video"].shape[2]
    if T <= chunk_size:
        return model.generate(batch)["generated-video"][0]
    outs = []
    n_chunks = (T + chunk_size - 1) // chunk_size
    log_fn(f"  chunked inference: T={T} into {n_chunks} chunks of {chunk_size}")
    for i in range(n_chunks):
        s, e = i * chunk_size, min((i + 1) * chunk_size, T)
        pad_t = chunk_size - (e - s)
        sub = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.ndim == 5:
                sub[k] = _pad_last_chunk(v[:, :, s:e], pad_t)
            else:
                sub[k] = v
        log_fn(f"  chunk {i+1}/{n_chunks}: frames [{s}:{e}] (pad={pad_t})")
        t0 = time.perf_counter()
        chunk_out = model.generate(sub)["generated-video"][0]
        if pad_t > 0:
            chunk_out = chunk_out[:, : (e - s)]
        outs.append(chunk_out)
        log_fn(f"  chunk {i+1}/{n_chunks} done in {time.perf_counter()-t0:.1f}s")
    return torch.cat(outs, dim=1)


def build_inpaint_tensors(video_path: Path, repro: Path, mask: Path,
                          mask_antialias: int = 0,
                          reprojected_closing_holes_kernel: int = 11):
    """Returns (iv, rp, rm, fps) — CPU tensors in [c, t, h, w] format."""
    input_video = get_video_frames(str(video_path))
    reprojected = get_video_frames(str(repro))
    reprojected_mask = get_video_frames(str(mask), video_is_grayscale=True)
    fps = get_video_fps(str(video_path), ffmpeg.probe(str(video_path)))

    reprojected_mask = apply_closing(reprojected_mask, reprojected_closing_holes_kernel)
    reprojected[reprojected_mask.repeat(1, 3, 1, 1) > 0.5] = 0
    reprojected_mask = apply_dilation(reprojected_mask, 3)
    reprojected_mask = reprojected_mask.repeat(1, 3, 1, 1)

    iv = input_video.permute(1, 0, 2, 3).float() * 2 - 1
    rp = reprojected.permute(1, 0, 2, 3).float() * 2 - 1
    rm = reprojected_mask.permute(1, 0, 2, 3).float() * 2 - 1

    c, t, h, w = rm.shape
    rm = rm.permute(1, 0, 2, 3).float()
    rm = transforms.Resize([h // 8, w // 8], antialias=bool(mask_antialias))(rm)
    rm = rm[:, [0]].permute(1, 0, 2, 3).float()
    return iv, rp, rm, fps


def generate_shot_tensor(video_path: Path, repro: Path, mask: Path,
                         shot_start: int, shot_end: int,
                         seed: int = 42,
                         config_path: Optional[Path] = None,
                         ckpt_path: Optional[Path] = None,
                         mask_antialias: int = 0,
                         reprojected_closing_holes_kernel: int = 11,
                         chunk_size: int = CHUNK_SIZE,
                         log_fn=print) -> torch.Tensor:
    """Run the M2SVid generator on a single shot range. Returns CPU tensor [c,t,h,w].

    chunk_size: temporal window per generate() call. Default 25 (M2SVid training
    window). Smaller values free VRAM (allowing higher processing dim) at the
    cost of weaker temporal coherence + more chunk seams. Used by Resolution
    Overdrive presets.
    """
    seed_everything(int(seed))
    iv, rp, rm, fps = build_inpaint_tensors(
        video_path, repro, mask,
        mask_antialias=mask_antialias,
        reprojected_closing_holes_kernel=reprojected_closing_holes_kernel,
    )
    T = iv.shape[1]
    s = max(0, int(shot_start))
    e_excl = min(int(shot_end) + 1, T)
    iv_s = iv[:, s:e_excl]
    rp_s = rp[:, s:e_excl]
    rm_s = rm[:, s:e_excl]
    model = get_model(config_path, ckpt_path, log_fn=log_fn)
    batch = {
        "video": iv_s[None].cuda(),
        "video_2nd_view": iv_s[None].cuda(),
        "reprojected_video": rp_s[None].cuda(),
        "reprojected_mask": rm_s[None].cuda(),
        "fps_id": torch.tensor([fps]).cuda(),
        "caption": [""],
        "motion_bucket_id": torch.tensor([127]).cuda(),
    }
    cs = max(1, int(chunk_size))
    if cs != CHUNK_SIZE:
        log_fn(f"  using chunk_size={cs} (default {CHUNK_SIZE})")
    with torch.inference_mode():
        out = generate_chunked(model, batch, chunk_size=cs, log_fn=log_fn).cpu()
    return out
