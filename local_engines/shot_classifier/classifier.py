"""
classifier.py — v0.16b shot-scale 분류 (wide / normal / closeup)

Phase A change vs v0.16:
  - Default backend swapped from CLIP-ViT-B/32 (2021) to **SigLIP-2 base**
    (google/siglip2-base-patch16-256, Feb 2025)
  - +10-15pt accuracy on ImageNet zero-shot (~78% vs 63%)
  - 한국어 / 다언어 prompt 지원 (109 langs vs EN-only)
  - 모델 더 작음 (~370 MB vs 600 MB)
  - 동일 cosine-sim API, max_disp 매핑 동일

Backwards-compat: 기존 ClipClassifier ('clip') / DepthClassifier ('depth')
백엔드도 그대로 유지. 추가로 'siglip2' 가 새 default 백엔드.

공통 인터페이스 ShotClassifier:
  load(cache_dir, device) → 모델 로드
  predict(frame_rgb_uint8_HxWx3) → {class, confidence, scores, extra}
  unload() → VRAM 정리
"""
from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("shot_classifier.backend")

CLASSES = ("wide", "normal", "closeup")


class ShotClassifier(ABC):
    name: str = "base"

    def __init__(self):
        self.device: str = "cpu"
        self.loaded: bool = False

    @abstractmethod
    def load(self, cache_dir: Path, device: str = "cuda") -> None: ...

    @abstractmethod
    def predict(self, frame_rgb: np.ndarray) -> dict: ...

    def unload(self) -> None:
        for attr in ("model", "processor", "text_embeds", "class_embeds"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass
        self.loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        gc.collect()
        log.info(f"[{self.name}] unloaded, VRAM cleared")


# ──────────────────────────────────────────────────────────────────────
# Common prompts (used by CLIP and SigLIP-2)
# ──────────────────────────────────────────────────────────────────────

DEFAULT_PROMPTS: dict[str, list[str]] = {
    "wide": [
        "a wide establishing shot of a landscape",
        "a long shot with a small subject in a large environment",
        "a wide angle shot showing the full scene",
        "an extreme long shot of a vast space",
    ],
    "normal": [
        "a medium shot at moderate distance",
        "a standard shot with balanced framing",
        "a regular shot of a person from waist up",
        "a medium full shot showing most of the body",
    ],
    "closeup": [
        "a close-up shot filling the frame with the subject",
        "a tight telephoto shot with compressed background",
        "an extreme close-up with shallow depth of field",
        "a portrait close-up of a face",
    ],
}


# ──────────────────────────────────────────────────────────────────────
# SigLIP-2 (NEW DEFAULT in v0.16b)
# ──────────────────────────────────────────────────────────────────────

class Siglip2Classifier(ShotClassifier):
    """SigLIP-2 base zero-shot classifier.

    Uses sigmoid-based pairwise scoring (SigLIP) instead of softmax (CLIP).
    For our 3-class argmax it produces equivalent or better selections.
    """
    name = "siglip2"
    model_id = "google/siglip2-base-patch16-256"

    def __init__(self, prompts: Optional[dict[str, list[str]]] = None):
        super().__init__()
        self.prompts = prompts or DEFAULT_PROMPTS
        for c in CLASSES:
            if c not in self.prompts or not self.prompts[c]:
                raise ValueError(f"prompts missing class '{c}' or empty list")

    def load(self, cache_dir: Path, device: str = "cuda") -> None:
        from transformers import AutoModel, AutoProcessor

        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"[{self.name}] loading {self.model_id} to {self.device} (cache={cache_dir})")
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=str(cache_dir))
        self.model = AutoModel.from_pretrained(
            self.model_id, cache_dir=str(cache_dir),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

        # Per-class average text embedding.
        # transformers >= 4.49 (SigLIP-2 support): SiglipProcessor doesn't define
        # a default max_length, so `padding="max_length"` silently does nothing
        # and varying-length prompts fail to stack. Use `padding=True` to pad
        # to the longest in the batch (works regardless of model max_length).
        class_embeds = []
        for cls in CLASSES:
            ps = self.prompts[cls]
            inputs = self.processor(
                text=ps, return_tensors="pt",
                padding=True, truncation=True, max_length=64,
            ).to(self.device)
            with torch.no_grad():
                result = self.model.get_text_features(**inputs)
                # transformers 5.x compat: returns BaseModelOutputWithPooling
                # (4.x returned a Tensor). Use pooler_output for pooled embedding.
                embeds = result if torch.is_tensor(result) else result.pooler_output
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                avg = embeds.mean(dim=0)
                avg = avg / avg.norm()
            class_embeds.append(avg)
        self.class_embeds = torch.stack(class_embeds, dim=0)  # [3, D]
        self.loaded = True
        n_prompts = sum(len(self.prompts[c]) for c in CLASSES)
        log.info(f"[{self.name}] loaded ({n_prompts} prompts across {len(CLASSES)} classes)")

    def predict(self, frame_rgb: np.ndarray) -> dict:
        from PIL import Image

        if not self.loaded:
            raise RuntimeError(f"{self.name} not loaded")

        img = Image.fromarray(frame_rgb)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            result = self.model.get_image_features(**inputs)
            # transformers 5.x compat (BaseModelOutputWithPooling vs Tensor)
            img_embed = result if torch.is_tensor(result) else result.pooler_output
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
            sims = (img_embed @ self.class_embeds.T).squeeze(0).float()  # [3]
            # SigLIP uses sigmoid; for argmax that's equivalent to ranking by sims
            # We still return softmax-like scores for consistency with v0.16 output
            probs = torch.softmax(sims * 10.0, dim=0).cpu().numpy()  # temp=10 for stability

        pred_idx = int(np.argmax(probs))
        return {
            "class": CLASSES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "scores": {c: float(probs[i]) for i, c in enumerate(CLASSES)},
            "extra": {
                "cosine_similarities": {c: float(sims[i].item()) for i, c in enumerate(CLASSES)},
                "model_id": self.model_id,
            },
        }


# ──────────────────────────────────────────────────────────────────────
# CLIP zero-shot (legacy backend, kept for backward compat)
# ──────────────────────────────────────────────────────────────────────

class ClipClassifier(ShotClassifier):
    name = "clip"
    model_id = "openai/clip-vit-base-patch32"

    def __init__(self, prompts: Optional[dict[str, list[str]]] = None):
        super().__init__()
        self.prompts = prompts or DEFAULT_PROMPTS
        for c in CLASSES:
            if c not in self.prompts or not self.prompts[c]:
                raise ValueError(f"prompts missing class '{c}' or empty list")

    def load(self, cache_dir: Path, device: str = "cuda") -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"[{self.name}] loading {self.model_id} to {self.device} (cache={cache_dir})")
        self.processor = CLIPProcessor.from_pretrained(self.model_id, cache_dir=str(cache_dir))
        self.model = CLIPModel.from_pretrained(
            self.model_id, cache_dir=str(cache_dir)
        ).to(self.device).eval()

        class_embeds = []
        for cls in CLASSES:
            ps = self.prompts[cls]
            inputs = self.processor(text=ps, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                result = self.model.get_text_features(**inputs)
                # transformers 5.x compat: returns BaseModelOutputWithPooling
                # (4.x returned a Tensor). Use pooler_output for pooled embedding.
                embeds = result if torch.is_tensor(result) else result.pooler_output
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                avg = embeds.mean(dim=0)
                avg = avg / avg.norm()
            class_embeds.append(avg)
        self.class_embeds = torch.stack(class_embeds, dim=0)
        self.loaded = True
        log.info(f"[{self.name}] loaded ({sum(len(self.prompts[c]) for c in CLASSES)} prompts)")

    def predict(self, frame_rgb: np.ndarray) -> dict:
        from PIL import Image

        if not self.loaded:
            raise RuntimeError("ClipClassifier not loaded")

        img = Image.fromarray(frame_rgb)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_embed = self.model.get_image_features(**inputs)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
            sims = (img_embed @ self.class_embeds.T).squeeze(0)
            logit_scale = self.model.logit_scale.exp().item()
            probs = torch.softmax(sims * logit_scale, dim=0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        return {
            "class": CLASSES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "scores": {c: float(probs[i]) for i, c in enumerate(CLASSES)},
            "extra": {
                "cosine_similarities": {c: float(sims[i].item()) for i, c in enumerate(CLASSES)},
                "model_id": self.model_id,
            },
        }


# ──────────────────────────────────────────────────────────────────────
# Depth-Anything-V2 (legacy backend)
# ──────────────────────────────────────────────────────────────────────

class DepthClassifier(ShotClassifier):
    name = "depth"
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, std_wide: float = 0.25, std_closeup: float = 0.12):
        super().__init__()
        if not (0 < std_closeup < std_wide < 1):
            raise ValueError("must: 0 < std_closeup < std_wide < 1")
        self.std_wide = float(std_wide)
        self.std_closeup = float(std_closeup)

    def load(self, cache_dir: Path, device: str = "cuda") -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"[{self.name}] loading {self.model_id} to {self.device} (cache={cache_dir})")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id, cache_dir=str(cache_dir))
        self.model = AutoModelForDepthEstimation.from_pretrained(
            self.model_id, cache_dir=str(cache_dir)
        ).to(self.device).eval()
        self.loaded = True

    def predict(self, frame_rgb: np.ndarray) -> dict:
        from PIL import Image

        if not self.loaded:
            raise RuntimeError("DepthClassifier not loaded")

        img = Image.fromarray(frame_rgb)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            depth = out.predicted_depth.squeeze().detach().cpu().numpy()

        dmin, dmax = float(depth.min()), float(depth.max())
        rng = dmax - dmin
        if rng < 1e-6:
            norm = np.zeros_like(depth)
        else:
            norm = (depth - dmin) / rng
        std = float(norm.std())

        if std <= self.std_closeup:
            s_wide, s_normal, s_closeup = 0.0, 0.0, 1.0
        elif std >= self.std_wide:
            s_wide, s_normal, s_closeup = 1.0, 0.0, 0.0
        else:
            span = self.std_wide - self.std_closeup
            t = (std - self.std_closeup) / span
            if t <= 0.5:
                s_closeup = 1.0 - 2.0 * t
                s_normal = 2.0 * t
                s_wide = 0.0
            else:
                s_closeup = 0.0
                s_normal = 2.0 * (1.0 - t)
                s_wide = 2.0 * t - 1.0

        scores = {"wide": s_wide, "normal": s_normal, "closeup": s_closeup}
        tot = sum(scores.values())
        if tot > 0:
            scores = {k: v / tot for k, v in scores.items()}
        pred_idx = int(np.argmax([scores[c] for c in CLASSES]))

        return {
            "class": CLASSES[pred_idx],
            "confidence": float(scores[CLASSES[pred_idx]]),
            "scores": {c: float(scores[c]) for c in CLASSES},
            "extra": {"depth_std": std, "depth_range": rng},
        }


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────

def make_classifier(
    backend: str,
    *,
    clip_prompts: Optional[dict] = None,
    depth_std_wide: float = 0.25,
    depth_std_closeup: float = 0.12,
) -> ShotClassifier:
    backend = backend.lower()
    if backend in ("siglip2", "siglip-2", "siglip"):
        return Siglip2Classifier(prompts=clip_prompts)
    if backend == "clip":
        return ClipClassifier(prompts=clip_prompts)
    if backend == "depth":
        return DepthClassifier(std_wide=depth_std_wide, std_closeup=depth_std_closeup)
    raise ValueError(f"unknown backend: {backend} (expected 'siglip2', 'clip', or 'depth')")
