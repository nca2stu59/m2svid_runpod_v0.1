"""
depth_splat_local.py — v0.16b local Step1 (depth + splatting)

Replaces GenStereo's `depth_splatting_inference.py` with a stack that uses:
  - Video-Depth-Anything Large (Phase C, ~30-50× faster than DepthCrafter)
  - Pure-PyTorch softmax splatting (Phase B, no CUDA toolkit dependency)

Output format is identical to GenStereo's depth_splatting_inference.py:
  - Output mp4 is a 2H×2W grid:
      top-left  : original frame
      top-right : depth visualization (inferno colormap)
      bot-left  : occlusion mask (white = hole)
      bot-right : warped right-eye view
  - Used by GenStereo's `inpainting_inference.py` (Step2) downstream

Usage (CLI, drop-in for GenStereo's script):
    python depth_splat_local.py INPUT.mp4 OUTPUT.mp4 \
        --vda-weights /path/to/video_depth_anything_vitl.pth \
        --vda-repo /path/to/Video-Depth-Anything/repo \
        --max_disp 20 --process_length -1 --batch_size 10 --max_res 1024
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Local module imports
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from softmax_splatting import ForwardWarpStereoSoftmax  # noqa: E402
from vda_depth import VDADepth  # noqa: E402


def _emit(event: str, **kwargs):
    """JSONL event for orchestrator monitoring."""
    import json
    print(json.dumps({"event": event, **kwargs}, ensure_ascii=False), flush=True)


def depth_splatting_local(
    input_video_path: str,
    output_video_path: str,
    vda_weights: str,
    vda_repo: str,
    max_disp: float = 20.0,
    process_length: int = -1,
    batch_size: int = 10,
    max_res: int = 1024,
):
    t_total = time.time()

    # Sanitize Windows paths for fragile downstream readers
    input_video_path = str(Path(input_video_path).resolve())
    output_video_path = str(Path(output_video_path).resolve())

    # ── Step 1a: depth via VDA-L
    _emit("vda_loading", weights=vda_weights)
    t0 = time.time()
    vda = VDADepth(vda_weights, vda_repo, encoder="vitl", fp16=True)
    vda._ensure_loaded()
    t_load = time.time() - t0
    _emit("vda_loaded", sec=round(t_load, 2),
          vram_gb=round(torch.cuda.memory_allocated() / 1024**3, 2))

    t0 = time.time()
    _emit("vda_inferring", input=input_video_path, max_res=max_res)
    res, depth_vis = vda.infer(input_video_path,
                                process_length=process_length,
                                max_res=max_res)
    t_depth = time.time() - t0
    _emit("vda_done", sec=round(t_depth, 2),
          frames=int(res.shape[0]),
          shape=list(res.shape),
          per_frame_ms=round(t_depth / max(res.shape[0], 1) * 1000, 1))

    # Free VDA model — we don't need it for splatting
    vda.unload()
    torch.cuda.empty_cache()

    # ── Step 1b: read original (uncompressed-resolution) frames for splat output
    # We splat onto FULL resolution to match GenStereo's behavior
    from decord import VideoReader, cpu
    vr = VideoReader(input_video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    n_frames = len(vr)
    if process_length > 0:
        n_frames = min(n_frames, process_length)
    input_frames = vr[:n_frames].asnumpy() / 255.0  # [N, H, W, 3] in [0,1]
    H_orig, W_orig = input_frames.shape[1:3]

    # Resize VDA depth (which is at max_res<=1024 internally) to original H×W
    if res.shape[1] != H_orig or res.shape[2] != W_orig:
        depth_t = torch.from_numpy(res).unsqueeze(1).float().cuda()
        depth_t = F.interpolate(depth_t, size=(H_orig, W_orig),
                                mode="bilinear", align_corners=False)
        res = depth_t.cpu().numpy()[:, 0]
        del depth_t
        torch.cuda.empty_cache()

    # Match VDA depth length to input frames (in case of rounding)
    if res.shape[0] < input_frames.shape[0]:
        input_frames = input_frames[:res.shape[0]]
        depth_vis = depth_vis[:res.shape[0]]
    elif res.shape[0] > input_frames.shape[0]:
        res = res[:input_frames.shape[0]]
        depth_vis = depth_vis[:input_frames.shape[0]]

    n_frames = input_frames.shape[0]

    # Resize depth_vis to original size too (for the visualization quadrant)
    if depth_vis.shape[1] != H_orig or depth_vis.shape[2] != W_orig:
        dv = np.zeros((depth_vis.shape[0], H_orig, W_orig, 3), dtype=np.float32)
        for i in range(depth_vis.shape[0]):
            dv[i] = cv2.resize(depth_vis[i], (W_orig, H_orig),
                               interpolation=cv2.INTER_LINEAR)
        depth_vis = dv

    # ── Step 1c: splatting (pure-PyTorch softmax warp)
    _emit("splat_start", n_frames=n_frames, height=H_orig, width=W_orig,
          max_disp=max_disp, batch_size=batch_size)

    splatter = ForwardWarpStereoSoftmax(occlu_map=True).cuda()

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W_orig * 2, H_orig * 2),
    )

    t0 = time.time()
    for i in range(0, n_frames, batch_size):
        batch_frames = input_frames[i:i + batch_size]
        batch_depth = res[i:i + batch_size]
        batch_dvis = depth_vis[i:i + batch_size]

        left_video = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).float().cuda()
        disp_map = torch.from_numpy(batch_depth).unsqueeze(1).float().cuda()

        # Match GenStereo convention: depth in [0,1] → disp in [-max_disp, +max_disp]
        disp_map = disp_map * 2.0 - 1.0
        disp_map = disp_map * max_disp

        with torch.no_grad():
            right_video, occlusion_mask = splatter(left_video, disp_map)

        right_video = right_video.cpu().permute(0, 2, 3, 1).numpy()
        occlusion_mask = occlusion_mask.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)

        for j in range(len(batch_frames)):
            video_grid_top = np.concatenate([batch_frames[j], batch_dvis[j]], axis=1)
            video_grid_bot = np.concatenate([occlusion_mask[j], right_video[j]], axis=1)
            video_grid = np.concatenate([video_grid_top, video_grid_bot], axis=0)

            video_grid_uint8 = np.clip(video_grid * 255.0, 0, 255).astype(np.uint8)
            video_grid_bgr = cv2.cvtColor(video_grid_uint8, cv2.COLOR_RGB2BGR)
            out.write(video_grid_bgr)

        # Per-batch cleanup
        del left_video, disp_map, right_video, occlusion_mask
        torch.cuda.empty_cache()
        gc.collect()

    out.release()
    t_splat = time.time() - t0
    _emit("splat_done", sec=round(t_splat, 2),
          per_frame_ms=round(t_splat / n_frames * 1000, 1),
          output=output_video_path)

    _emit("done", total_sec=round(time.time() - t_total, 2),
          depth_sec=round(t_depth, 2),
          splat_sec=round(t_splat, 2),
          n_frames=n_frames)


def main():
    p = argparse.ArgumentParser(
        description="v0.16b local Step1 (VDA-L depth + softmax splatting)"
    )
    # Positional args mirror GenStereo's depth_splatting_inference.py for
    # easy drop-in (GenStereo uses 4 positionals + flags). Here the first
    # two are required, others have sensible defaults.
    p.add_argument("input_video_path")
    p.add_argument("output_video_path")
    p.add_argument("--vda-weights", required=True,
                   help="Path to video_depth_anything_vitl.pth")
    p.add_argument("--vda-repo", required=True,
                   help="Path to cloned Video-Depth-Anything repo")
    p.add_argument("--max_disp", type=float, default=20.0)
    p.add_argument("--process_length", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--max_res", type=int, default=1024)
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    depth_splatting_local(
        args.input_video_path, args.output_video_path,
        args.vda_weights, args.vda_repo,
        max_disp=args.max_disp,
        process_length=args.process_length,
        batch_size=args.batch_size,
        max_res=args.max_res,
    )


if __name__ == "__main__":
    main()
