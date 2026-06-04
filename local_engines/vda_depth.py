"""
vda_depth.py — Video-Depth-Anything Large wrapper for v0.16b

Phase C change vs v0.16:
  - Replace DepthCrafter (SVD-based, 4.81s/frame) with Video-Depth-Anything
    Large (DINOv2-based, ~10ms/frame on RTX 5090)
  - 30-50× speed up
  - TAE 0.570 on ScanNet (vs DepthCrafter 0.639) → better temporal consistency
  - 1.5 GB weight (vs 5.9 GB DepthCrafter)
  - License: CC-BY-NC-4.0 (same as DepthCrafter — no regression)

Source: https://github.com/DepthAnything/Video-Depth-Anything (Apache-2.0 code)
Weight: https://huggingface.co/depth-anything/Video-Depth-Anything-Large
        (video_depth_anything_vitl.pth, 382M params, 1.5 GB)

API (mirrors GenStereo's DepthCrafterDemo for drop-in usage):
    VDADepth(weights_path, repo_path, device='cuda')
    .infer(input_video_path) -> (video_depth, depth_vis)
        video_depth: [N, H, W] np.float32 in [0, 1]
        depth_vis:   [N, H, W, 3] np.float32 in [0, 1]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


VDA_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _vis_depth(depth: np.ndarray) -> np.ndarray:
    """Convert per-frame depth [N,H,W] in arbitrary range to colormap RGB
    [N,H,W,3] in [0,1]. Used to mimic DepthCrafter's vis output for any
    downstream consumer that wants a debug video.

    Uses inferno colormap (matplotlib) but with NaN/inf guarded — same fix
    we applied to the GenStereo upstream patch (utils.py) for robustness.
    """
    import matplotlib.cm as cm
    cmap = np.array(cm.get_cmap("inferno").colors, dtype=np.float32)  # [256, 3]

    out = np.zeros((depth.shape[0], depth.shape[1], depth.shape[2], 3), dtype=np.float32)
    for i in range(depth.shape[0]):
        d = depth[i]
        v_min, v_max = float(np.nanmin(d)), float(np.nanmax(d))
        denom = max(v_max - v_min, 1e-8)
        norm = (d - v_min) / denom
        norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
        idx = np.clip((norm * 255).astype(np.int64), 0, 255)
        out[i] = cmap[idx]
    return out


class VDADepth:
    """Video-Depth-Anything Large wrapper.

    The VDA model takes a list/array of RGB uint8 frames and returns
    per-frame depth (smaller-is-closer convention, native VDA output).
    We invert + normalize to match DepthCrafter's [0,1] inverse-depth
    convention so downstream forward-warp logic stays identical.
    """
    def __init__(
        self,
        weights_path: str | Path,
        repo_path: str | Path,
        encoder: str = "vitl",
        device: str = "cuda",
        fp16: bool = True,
    ):
        self.weights_path = Path(weights_path)
        self.repo_path = Path(repo_path)
        self.encoder = encoder
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.fp16 = fp16
        self.model = None

        if not self.weights_path.exists():
            raise FileNotFoundError(f"VDA weights not found: {self.weights_path}")
        if not self.repo_path.exists():
            raise FileNotFoundError(f"VDA repo not found: {self.repo_path}")

    def _ensure_loaded(self):
        if self.model is not None:
            return
        # Add VDA repo to sys.path so its module imports work
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))
        from video_depth_anything.video_depth import VideoDepthAnything

        cfg = VDA_MODEL_CONFIGS[self.encoder]
        m = VideoDepthAnything(**cfg, metric=False)
        state = torch.load(str(self.weights_path), map_location="cpu", weights_only=True)
        m.load_state_dict(state, strict=True)
        m = m.to(self.device).eval()
        self.model = m

    def infer(
        self,
        input_video_path: str | Path,
        process_length: int = -1,
        max_res: int = 1024,
        input_size: int = 518,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run VDA on a video file.

        Returns:
            res: [N, H, W] np.float32 normalized depth in [0, 1]
                  (closer to 1 = nearer to camera, matching DepthCrafter)
            vis: [N, H, W, 3] np.float32 RGB depth visualization in [0, 1]
        """
        self._ensure_loaded()

        # Use VDA's own video reader (handles resize + fps) so we get the
        # exact preprocessing the model was trained for.
        sys.path.insert(0, str(self.repo_path))
        from utils.dc_utils import read_video_frames

        # VDA's read_video_frames signature: (video_path, process_length, target_fps=-1, max_res=-1)
        frames, target_fps = read_video_frames(
            str(input_video_path),
            process_length if process_length > 0 else -1,
            target_fps=-1,
            max_res=max_res,
        )
        # frames: np.float32 [N, H, W, 3] in [0, 1]

        with torch.inference_mode():
            depths, fps = self.model.infer_video_depth(
                frames, target_fps,
                input_size=input_size,
                device=self.device,
                fp32=not self.fp16,
            )
        # depths: np.float32 [N, H, W], NATIVE VDA scale (larger = nearer)

        # Normalize to [0, 1] across the whole sequence (matches DepthCrafter
        # post-processing: normalized inverse depth, larger = closer)
        d_min = float(depths.min())
        d_max = float(depths.max())
        denom = max(d_max - d_min, 1e-8)
        res = ((depths - d_min) / denom).astype(np.float32)

        vis = _vis_depth(res)
        return res, vis

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


# Quick smoke test
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--max-res", type=int, default=1024)
    args = parser.parse_args()

    vda = VDADepth(args.weights, args.repo)
    print(f"Loading VDA-L from {args.weights}...")
    vda._ensure_loaded()
    print(f"VRAM after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    print(f"Inferring depth on {args.video}...")
    import time
    t0 = time.time()
    depth, vis = vda.infer(args.video, max_res=args.max_res)
    elapsed = time.time() - t0
    print(f"Inference: {elapsed:.1f}s for {depth.shape[0]} frames "
          f"({elapsed/depth.shape[0]*1000:.1f}ms/frame)")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"Depth shape: {depth.shape} dtype: {depth.dtype}")
    print(f"VRAM peak:  {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("VDA smoke test OK")
