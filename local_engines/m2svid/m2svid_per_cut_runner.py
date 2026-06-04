"""
m2svid_per_cut_runner.py — per-cut M2SVid pipeline executor.

Runs in m2svid_service/.venv (sgm + pytorch_lightning + torch 2.9 cu128 + ffmpeg-python).
Spawned as subprocess by `m2svid_worker.py` (which itself runs in port/S3D_Pipeline/m2svid/venv).

Per-cut flow:
  1. preprocess: resize cut .mp4 to processing_dim (64-divisible, max-axis bound)
  2. depth: spawn .venv-vda subprocess (or .venv-flashdepth for FlashDepth) -> depth.npz
  3. warp: in-process (uses m2svid.warping.scatter_image) -> repro.mp4 + mask.mp4
  4. inpaint: in-process (inpaint_core.generate_shot_tensor) -> right view tensor
  5. compose SBS: side-by-side(left=src_resized, right=inpainted)
  6. upscale (optional): lanczos via ffmpeg, or RTX VSR via nvidia-vfx (if installed)
  7. write final SBS .mp4

All log lines go to stdout as plain text so the parent can capture as
stage_log events (compatible with v0.16b's worker contract).

CLI:
    --cut PATH                  per-cut input .mp4 (required)
    --out PATH                  final SBS .mp4 path (required)
    --tmp-dir PATH              workdir for intermediates (required)
    --processing-dim N          max processing dim, 64-div (default 512)
    --output-dim N              output max dim, 0=processing (default 0)
    --depth-backend NAME        VDA-S | VDA-L | FlashDepth-L | FlashDepth-S | DepthCrafter
                                (default VDA-S)
    --upscaler NAME             lanczos | rtx_vsr (default lanczos)
    --rtx-vsr-quality N         0-19, only used if upscaler=rtx_vsr (default 4 = ULTRA)
                                Modes: 0=BICUBIC, 1=LOW, 2=MEDIUM, 3=HIGH, 4=ULTRA,
                                       8-11 DENOISE_LOW..ULTRA, 12-15 DEBLUR_LOW..ULTRA,
                                       16-19 HIGHBITRATE_LOW..ULTRA
    --disparity-perc F          warp disparity (default 0.02)
    --seed N                    inpaint seed (default 42)
    --mask-antialias N          mask resize antialias 0/1 (default 0)
    --chunk-size N              M2SVid temporal window per generate() call
                                (default 25 = training window). Smaller values
                                free VRAM but reduce temporal coherence — used
                                by Resolution Overdrive (12) / Extreme (8).
    --m2svid-service PATH       m2svid_service root (default
                                C:\\Users\\PC\\Desktop\\m2svid_service)

Exit codes:
    0  success
    1  arg / preprocessing failure
    2  depth subprocess failure
    3  warp failure
    4  inpaint failure
    5  compose / upscale failure
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Make stdout line-buffered + UTF-8 so the parent can read events cleanly
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent  # v0.16m/local_engines/m2svid


def _default_service_root() -> Path:
    env = os.environ.get("M2SVID_SERVICE_ROOT") or os.environ.get("RUNPOD_M2SVID_SERVICE")
    if env:
        return Path(env)
    if os.name == "nt":
        return Path(r"C:\Users\PC\Desktop\m2svid_service")
    return Path("/workspace/m2svid_service")


def _venv_python(root: Path, env_name: str = ".venv") -> Path:
    if os.name == "nt":
        return root / env_name / "Scripts" / "python.exe"
    return root / env_name / "bin" / "python"


def _log(msg: str) -> None:
    print(f"[per_cut] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Step 1 — preprocess (ffmpeg resize)
# --------------------------------------------------------------------------- #

def _resize_cut(cut_path: Path, out_path: Path, max_dim: int) -> tuple[int, int]:
    """Resize cut to max_dim with both axes 64-divisible. Returns (W, H)."""
    import ffmpeg  # available in m2svid_service .venv

    probe = ffmpeg.probe(str(cut_path))
    vstream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    src_w = int(vstream["width"])
    src_h = int(vstream["height"])

    # Compute target maintaining aspect, max dim <= max_dim, both 64-div
    if src_w >= src_h:
        new_w = max_dim
        new_h = int(round(src_h * max_dim / src_w))
    else:
        new_h = max_dim
        new_w = int(round(src_w * max_dim / src_h))
    new_w = (new_w // 64) * 64
    new_h = (new_h // 64) * 64

    _log(f"[1/5] preprocess: {src_w}x{src_h} -> {new_w}x{new_h} (max_dim={max_dim}, 64-div)")
    (
        ffmpeg.input(str(cut_path))
        .filter("scale", new_w, new_h, flags="lanczos")
        .output(str(out_path), **{"c:v": "libx264", "crf": 18, "preset": "veryfast", "pix_fmt": "yuv420p"})
        .overwrite_output()
        .run(quiet=True)
    )
    return new_w, new_h


# --------------------------------------------------------------------------- #
# Step 2 — depth (subprocess to backend's venv)
# --------------------------------------------------------------------------- #

def _depth(src_resized: Path, work_dir: Path, backend: str,
           m2svid_service: Path) -> Path:
    """Run depth on src_resized, return path to depth.npz with key='depth'.

    Cache: if out_npz already exists with valid 'depth' array (>=4KB), skip.
    """
    out_dir = work_dir / "depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"{src_resized.stem}.npz"

    # Fast cache check: file exists, non-trivial size, contains 'depth' key.
    if out_npz.exists() and out_npz.stat().st_size > 4096:
        try:
            import numpy as np
            with np.load(str(out_npz)) as z:
                if "depth" in z.files and z["depth"].size > 0:
                    _log(f"[2/5] depth ({backend}) cached: {out_npz.name} "
                         f"({out_npz.stat().st_size//1024} KB)")
                    return out_npz
        except Exception:
            pass  # invalid cache, fall through to re-run

    if backend in ("VDA-S", "VDA-L"):
        encoder = "vits" if backend == "VDA-S" else "vitl"
        vda_python = _venv_python(m2svid_service, ".venv-vda")
        vda_dir = m2svid_service / "third_party" / "Video-Depth-Anything"
        vda_run = vda_dir / "run.py"
        cmd = [
            str(vda_python), "-u", str(vda_run),
            "--input_video", str(src_resized),
            "--output_dir", str(out_dir),
            "--encoder", encoder,
            "--max_res", str(max(_get_video_dim(src_resized))),
            "--save_npz",
        ]
        cwd = str(vda_dir)
    elif backend in ("FlashDepth-L", "FlashDepth-S", "FlashDepth"):
        # FlashDepth's train.py uses torchrun + PyTorch distributed.
        # m2svid_service applies a v0.17 Windows patch in
        # third_party/FlashDepth/utils/init_setup.py: stub dist.* + skip DDP.
        # Verify checkpoint presence (only FlashDepth-L weights ship by default).
        variant = {"FlashDepth-L": "flashdepth-l", "FlashDepth-S": "flashdepth-s",
                   "FlashDepth": "flashdepth"}[backend]
        ckpt_dir = m2svid_service / "third_party" / "FlashDepth" / "configs" / variant
        ckpts = list(ckpt_dir.glob("*.pth")) if ckpt_dir.exists() else []
        if not ckpts:
            raise RuntimeError(
                f"FlashDepth checkpoint missing for variant '{variant}'. "
                f"Expected a .pth file in {ckpt_dir}. "
                f"Only FlashDepth-L ships pretrained weights by default. "
                f"Use --depth-backend FlashDepth-L (or VDA-S for fastest)."
            )
        fd_python = _venv_python(m2svid_service, ".venv-flashdepth")
        fd_run = m2svid_service / "flashdepth_run.py"
        cmd = [
            str(fd_python), "-u", str(fd_run),
            "--input_video", str(src_resized),
            "--output_npz", str(out_npz),    # FlashDepth writes directly to canonical npz path
            "--variant", variant,
        ]
        cwd = str(m2svid_service)
    elif backend == "DepthCrafter":
        dc_python = _venv_python(m2svid_service, ".venv-depthcrafter")
        dc_dir = m2svid_service / "third_party" / "DepthCrafter_new"
        dc_run = dc_dir / "run.py"
        cmd = [
            str(dc_python), "-u", str(dc_run),
            "--video-path", str(src_resized),
            "--save_folder", str(out_dir),
            "--save_npz", "True",
            "--num_inference_steps", "25",
            "--max_res", str(max(_get_video_dim(src_resized))),
        ]
        # DepthCrafter writes {stem}.npz inside save_folder by its own convention
        cwd = str(dc_dir)
    else:
        raise ValueError(f"Unknown depth backend: {backend}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    _log(f"[2/5] depth ({backend}) ...")
    t0 = time.perf_counter()
    # Stream output line-by-line so chunk progress (tqdm) is visible upstream.
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        bufsize=1,
    )
    tail: list[str] = []  # keep last ~50 lines for failure diagnostics
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\r\n")
        if not line:
            continue
        # Filter to important markers to avoid massive tqdm spam in worker logs.
        # Show: stage messages, chunk progress at 25%/50%/75%/100%, errors.
        important = (
            "Loading" in line or "Loaded" in line
            or "Extracting" in line or "Extracted" in line
            or "Inferring" in line or "Inferred" in line
            or "Processing" in line or "Processed" in line
            or "0%|" in line or "25%|" in line or "50%|" in line
            or "75%|" in line or "100%|" in line
            or "FPS" in line.upper() or "frame" in line.lower()
            or "ERROR" in line.upper() or "FAIL" in line.upper()
            or "Traceback" in line
        )
        if important:
            _log(f"  depth: {line}")
        tail.append(line)
        if len(tail) > 50:
            tail.pop(0)
    rc = proc.wait()
    if rc != 0:
        _log(f"  depth STDOUT (last lines):\n" + "\n".join(tail[-30:]))
        raise RuntimeError(f"depth subprocess failed (exit {rc})")
    _log(f"  depth done in {time.perf_counter()-t0:.1f}s")

    # VDA outputs *_depths.npz with key='depths'; convert to canonical 'depth' [0,1]
    if backend in ("VDA-S", "VDA-L"):
        raw = out_dir / f"{src_resized.stem}_depths.npz"
        if not raw.exists():
            raise RuntimeError(f"VDA NPZ not found: {raw}")
        import numpy as np
        arr = np.load(raw)["depths"].astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        np.savez_compressed(out_npz, depth=arr)
    # FlashDepth + DepthCrafter expected to write depth-keyed NPZ already.
    # If not, callers will need adapter (out of scope for first prototype).

    if not out_npz.exists():
        raise RuntimeError(f"depth output NPZ not found: {out_npz}")
    return out_npz


def _get_video_dim(path: Path) -> tuple[int, int]:
    import ffmpeg
    probe = ffmpeg.probe(str(path))
    vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
    return int(vs["width"]), int(vs["height"])


# --------------------------------------------------------------------------- #
# Step 3 — warp (in-process, m2svid_service warping.py)
# --------------------------------------------------------------------------- #

def _warp(src_resized: Path, depth_npz: Path, work_dir: Path,
          disparity_perc: float, m2svid_service: Path) -> tuple[Path, Path]:
    """Warp in-process using m2svid_service's warping module."""
    repro = work_dir / "reprojected.mp4"
    mask = work_dir / "reprojected_mask.mp4"
    if repro.exists() and mask.exists() and repro.stat().st_size > 0 and mask.stat().st_size > 0:
        _log(f"[3/5] warp: cached -> {repro.name} + {mask.name}")
        return repro, mask

    # Bring m2svid_service onto sys.path so `m2svid.warping.warping.scatter_image`
    # imports correctly. Use vendored copy (HERE is local_engines/m2svid/),
    # which has m2svid/ package and warping.py, plus we need third_party/ from
    # m2svid_service for any pytorch-msssim deps that warping.py pulls.
    if str(m2svid_service) not in sys.path:
        sys.path.insert(0, str(m2svid_service))
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    _log(f"[3/5] warp: disparity_perc={disparity_perc}")
    t0 = time.perf_counter()
    # warping.py is vendored at HERE/warping.py — import its function
    sys.path.insert(0, str(HERE))
    import warping as _warp_mod  # vendored copy
    # warping.py exposes process_video_with_depth (we saw it earlier)
    _warp_mod.process_video_with_depth(
        video_path=str(src_resized),
        depth_path=str(depth_npz),
        output_path_reprojected=str(repro),
        output_path_mask=str(mask),
        disparity_perc=disparity_perc,
        batch_size=10,
    )
    _log(f"  warp done in {time.perf_counter()-t0:.1f}s")
    return repro, mask


# --------------------------------------------------------------------------- #
# Step 4 — inpaint (in-process via vendored inpaint_core)
# --------------------------------------------------------------------------- #

def _inpaint(src_resized: Path, repro: Path, mask: Path,
             shot_start: int, shot_end: int, seed: int,
             mask_antialias: int, chunk_size: int,
             m2svid_service: Path) -> "torch.Tensor":
    """Returns CPU tensor [c=3, t, h, w] of the inpainted right view.

    NOTE: M2SVid's OmegaConf model config references weights by relative paths
    (e.g. 'ckpts/open_clip_pytorch_model.bin'). These resolve via CWD, so we
    must chdir into m2svid_service before instantiating the model. We restore
    CWD afterwards so the rest of the runner (compose, upscale) writes outputs
    to the caller-specified absolute paths.
    """
    sys.path.insert(0, str(HERE))  # so inpaint_core imports vendored m2svid pkg
    import inpaint_core  # vendored, points to m2svid_service ckpts/configs/third_party

    _log(f"[4/5] inpaint: shot_range=[{shot_start}:{shot_end}], seed={seed}")
    t0 = time.perf_counter()
    prev_cwd = os.getcwd()
    os.chdir(str(m2svid_service))
    try:
        out = inpaint_core.generate_shot_tensor(
            video_path=src_resized,
            repro=repro,
            mask=mask,
            shot_start=shot_start,
            shot_end=shot_end,
            seed=seed,
            mask_antialias=mask_antialias,
            chunk_size=chunk_size,
            log_fn=lambda m: _log(f"  inpaint: {m}"),
        )
    finally:
        os.chdir(prev_cwd)
    _log(f"  inpaint done in {time.perf_counter()-t0:.1f}s, out.shape={tuple(out.shape)}")
    return out


# --------------------------------------------------------------------------- #
# Step 5 — compose SBS  (left=src_resized | right=inpainted) + upscale
# --------------------------------------------------------------------------- #

def _compose_and_save_sbs(src_resized: Path, right_tensor, fps: float,
                          out_path: Path, output_dim: int,
                          upscaler: str, rtx_vsr_quality: int,
                          tmp_dir: Path,
                          src_aspect: float | None = None) -> None:
    """Build SBS = [LEFT(src_resized) | RIGHT(right_tensor)] and write to out_path.

    Optionally upscales to output_dim with lanczos or RTX VSR (per-eye split-process-repack).
    `src_aspect` (W/H of ORIGINAL cut, before 64-div crop) is used to restore the natural
    aspect at upscale time. If None, falls back to processing aspect (may be distorted
    due to 64-div).
    """
    import torch
    import numpy as np
    sys.path.insert(0, str(HERE))
    from m2svid.data.utils import get_video_frames  # for left

    _log(f"[5/5] compose SBS + upscale ({upscaler})")
    t0 = time.perf_counter()

    # Load LEFT from src_resized (matches inpaint domain)
    left = get_video_frames(str(src_resized))  # [t, c, h, w] in [0,1]
    # right_tensor: [c, t, h, w] in [-1, 1] (M2SVid output)
    right = right_tensor.permute(1, 0, 2, 3).clamp(-1, 1).add(1).div(2)  # [t, c, h, w] in [0,1]

    T = min(left.shape[0], right.shape[0])
    left = left[:T]
    right = right[:T]
    _, C, H, W = left.shape
    assert C == 3 and right.shape[1:] == (3, H, W), f"shape mismatch L={left.shape} R={right.shape}"

    # SBS: width-concat
    sbs = torch.cat([left, right], dim=-1)  # [t, 3, h, 2*w]
    sbs_np = (sbs.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()  # [t, h, 2w, 3]

    # Write SBS at processing dim first
    sbs_proc_path = tmp_dir / "sbs_proc.mp4"
    _write_video(sbs_np, fps, sbs_proc_path)

    # Decide final output: rescale or copy?
    # output_dim = per-eye HEIGHT (== final SBS height). For 16:9 mono source
    # processed at 512 (-> per-eye 512x288), output_dim=1080 gives per-eye
    # 1920x1080 -> SBS 3840x1080 (matches v0.16b output shape).
    final_w = sbs_np.shape[2]
    final_h = sbs_np.shape[1]
    target = int(output_dim) if output_dim > 0 else 0

    if target == 0 or final_h >= target:
        # No upscale needed (output_dim == 0 or already >= target height)
        shutil.copy2(sbs_proc_path, out_path)
        _log(f"  compose+save done in {time.perf_counter()-t0:.1f}s -> {out_path.name} ({final_w}x{final_h})")
        return

    # Upscale path. Compute target SBS dims using ORIGINAL aspect (before 64-div crop)
    # if known, else processing aspect.
    per_eye_h_target = target
    if src_aspect is not None:
        per_eye_w_target = max(2, int(round(per_eye_h_target * src_aspect)) // 2 * 2)
    else:
        per_eye_w_proc = final_w // 2
        per_eye_w_target = max(2, int(round(per_eye_w_proc * (per_eye_h_target / final_h))) // 2 * 2)
    target_h = per_eye_h_target
    target_w = per_eye_w_target * 2
    _log(f"  upscale target SBS: {target_w}x{target_h} (per-eye {per_eye_w_target}x{target_h})")

    if upscaler == "rtx_vsr":
        _upscale_rtx_vsr_to(sbs_proc_path, out_path, target_w, target_h, rtx_vsr_quality, fps)
    else:
        _upscale_lanczos_to(sbs_proc_path, out_path, target_w, target_h)
    _log(f"  compose+save+upscale done in {time.perf_counter()-t0:.1f}s -> {out_path.name}")


def _write_video(arr_thwc_uint8, fps: float, out_path: Path) -> None:
    """Write [t,h,w,3] uint8 video to out_path via imageio-ffmpeg (libx264)."""
    import imageio
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,  # honor exact dims (no extra cropping)
    )
    for frame in arr_thwc_uint8:
        writer.append_data(frame)
    writer.close()


def _upscale_lanczos_to(src: Path, dst: Path, target_w: int, target_h: int) -> None:
    """ffmpeg lanczos rescale to exact dims (target_w, target_h)."""
    import ffmpeg
    (
        ffmpeg.input(str(src))
        .filter("scale", target_w, target_h, flags="lanczos")
        .output(str(dst), **{"c:v": "libx264", "crf": 18, "preset": "medium", "pix_fmt": "yuv420p"})
        .overwrite_output()
        .run(quiet=True)
    )


def _upscale_rtx_vsr_to(src: Path, dst: Path, target_w: int, target_h: int,
                        quality: int, fps: float) -> None:
    """RTX VSR to exact SBS dims target_w x target_h. Per-eye = (target_w/2, target_h)."""
    try:
        from nvvfx import VideoSuperRes
        from nvvfx.effects import QualityLevel
    except ImportError:
        _log("  WARN: nvidia-vfx not installed; falling back to lanczos")
        _upscale_lanczos_to(src, dst, target_w, target_h)
        return
    import torch, numpy as np, imageio
    reader = imageio.get_reader(str(src))
    frames = [np.asarray(f) for f in reader]
    reader.close()
    arr = np.stack(frames, axis=0)  # [t, h, 2w, 3]
    T, H, W2, _ = arr.shape
    W = W2 // 2
    new_W = target_w // 2
    new_H = target_h
    _log(f"  RTX VSR: {W}x{H} per-eye -> {new_W}x{new_H} (quality={quality})")
    vsr = VideoSuperRes(quality=QualityLevel(quality))
    vsr.output_width = new_W
    vsr.output_height = new_H
    vsr.load()
    out_frames = np.empty((T, new_H, new_W * 2, 3), dtype=np.uint8)
    for ti in range(T):
        for half_idx in range(2):
            half = arr[ti, :, half_idx*W:(half_idx+1)*W, :]
            t = torch.from_numpy(half).permute(2, 0, 1).contiguous().float().div(255).cuda()
            # vsr.run returns VideoSuperResOutput; the DLPack capsule is on .image.
            # The capsule references C++ memory and must be cloned BEFORE the next
            # run() call invalidates it.
            result = vsr.run(t)
            out_t = torch.from_dlpack(result.image).clone()
            out_np = (out_t.clamp(0, 1) * 255).round().to(torch.uint8).cpu().permute(1, 2, 0).numpy()
            out_frames[ti, :, half_idx*new_W:(half_idx+1)*new_W, :] = out_np
    _write_video(out_frames, fps, dst)


def _upscale_lanczos(src: Path, dst: Path, target_height: int) -> None:
    """ffmpeg lanczos rescale of an SBS frame.

    Semantics: target_height = final SBS HEIGHT (== per-eye height).
    For SBS 1024x288 with target=1080 -> SBS 3840x1080 (per-eye 1920x1080).

    Output: even-rounded dims (libx264 yuv420p requires even).
    """
    import ffmpeg
    probe = ffmpeg.probe(str(src))
    vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
    src_w = int(vs["width"])
    src_h = int(vs["height"])
    scale = target_height / src_h
    new_h = max(2, int(round(src_h * scale)) // 2 * 2)
    new_w = max(2, int(round(src_w * scale)) // 2 * 2)
    (
        ffmpeg.input(str(src))
        .filter("scale", new_w, new_h, flags="lanczos")
        .output(str(dst), **{"c:v": "libx264", "crf": 18, "preset": "medium", "pix_fmt": "yuv420p"})
        .overwrite_output()
        .run(quiet=True)
    )


def _upscale_rtx_vsr(src: Path, dst: Path, target_height: int,
                     quality: int, fps: float) -> None:
    """RTX Video Super Resolution upscaling for SBS.

    Strategy: split L/R halves, run VSR on each separately, repack.
    target_height = final SBS height (mirrors lanczos semantics).
    """
    try:
        from nvvfx import VideoSuperRes
        from nvvfx.effects import QualityLevel  # nvidia-vfx wheel
    except ImportError:
        _log("  WARN: nvidia-vfx not installed in this venv; falling back to lanczos")
        _upscale_lanczos(src, dst, target_height)
        return

    import torch
    import numpy as np
    import imageio

    reader = imageio.get_reader(str(src))
    frames = [np.asarray(f) for f in reader]
    reader.close()
    arr = np.stack(frames, axis=0)  # [t, h, 2w, 3]
    T, H, W2, _ = arr.shape
    W = W2 // 2

    # per-eye height -> per-eye new dims (preserve aspect, even-rounded)
    s = target_height / H
    new_W = max(2, int(round(W * s)) // 2 * 2)
    new_H = max(2, int(round(H * s)) // 2 * 2)
    new_W2 = new_W * 2

    _log(f"  RTX VSR: {W}x{H} per-eye -> {new_W}x{new_H} (quality={quality})")

    # Build VSR for per-eye dim
    vsr = VideoSuperRes(quality=QualityLevel(quality))
    vsr.output_width = new_W
    vsr.output_height = new_H
    vsr.load()

    out_frames = np.empty((T, new_H, new_W2, 3), dtype=np.uint8)
    for ti in range(T):
        # split L/R
        left = arr[ti, :, :W, :]   # [h, w, 3] uint8
        right = arr[ti, :, W:, :]
        for half_idx, half in enumerate((left, right)):
            t = torch.from_numpy(half).permute(2, 0, 1).contiguous().float().div(255).cuda()
            out_dl = vsr.run(t)
            out_t = torch.from_dlpack(out_dl).clone()
            out_np = (out_t.clamp(0, 1) * 255).round().to(torch.uint8).cpu().permute(1, 2, 0).numpy()
            out_frames[ti, :, half_idx*new_W:(half_idx+1)*new_W, :] = out_np

    _write_video(out_frames, fps, dst)


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cut", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tmp-dir", required=True)
    p.add_argument("--processing-dim", type=int, default=512)
    p.add_argument("--output-dim", type=int, default=0)
    p.add_argument("--depth-backend", default="VDA-S")
    p.add_argument("--upscaler", default="lanczos", choices=["lanczos", "rtx_vsr"])
    p.add_argument("--rtx-vsr-quality", type=int, default=4)
    p.add_argument("--disparity-perc", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask-antialias", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=25,
                   help="M2SVid temporal window per generate() call "
                        "(default 25 = training; Resolution Overdrive uses 12)")
    p.add_argument("--m2svid-service",
                   default=str(_default_service_root()))
    args = p.parse_args()

    cut = Path(args.cut).resolve()
    out = Path(args.out).resolve()
    tmp = Path(args.tmp_dir).resolve()
    service = Path(args.m2svid_service).resolve()
    tmp.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not cut.exists():
        _log(f"[ERR] cut not found: {cut}")
        return 1
    if not service.exists():
        _log(f"[ERR] m2svid_service not found: {service}")
        return 1

    # Tell vendored inpaint_core where to look for ckpts/configs/third_party
    os.environ["M2SVID_SERVICE_ROOT"] = str(service)

    t_total = time.perf_counter()
    try:
        # Step 1
        # capture original aspect BEFORE 64-div crop, so upscale at end can
        # restore the natural source aspect (e.g. 16:9 -> 3840x1080 SBS)
        import ffmpeg as _ffmpeg
        _orig = _ffmpeg.probe(str(cut))
        _ovs = next(s for s in _orig["streams"] if s["codec_type"] == "video")
        src_aspect = float(_ovs["width"]) / float(_ovs["height"])

        src_resized = tmp / "src_resized.mp4"
        new_w, new_h = _resize_cut(cut, src_resized, args.processing_dim)

        # Step 2
        depth_npz = _depth(src_resized, tmp, args.depth_backend, service)

        # Step 3
        repro, mask = _warp(src_resized, depth_npz, tmp, args.disparity_perc, service)

        # Step 4
        # M2SVid wants frame range; for whole cut: 0 .. T-1
        import ffmpeg
        probe = ffmpeg.probe(str(src_resized))
        vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
        # nb_frames may be missing for some encoders; fall back to duration*fps
        if "nb_frames" in vs and vs["nb_frames"] != "N/A":
            T = int(vs["nb_frames"])
        else:
            num_str, den_str = vs["r_frame_rate"].split("/")
            fps_v = float(num_str) / max(float(den_str), 1.0)
            T = int(round(float(vs["duration"]) * fps_v))
        right_tensor = _inpaint(src_resized, repro, mask,
                                shot_start=0, shot_end=T - 1,
                                seed=args.seed, mask_antialias=args.mask_antialias,
                                chunk_size=args.chunk_size,
                                m2svid_service=service)

        # Step 5
        num_str, den_str = vs["r_frame_rate"].split("/")
        fps = float(num_str) / max(float(den_str), 1.0)
        _compose_and_save_sbs(src_resized, right_tensor, fps, out,
                              args.output_dim, args.upscaler,
                              args.rtx_vsr_quality, tmp,
                              src_aspect=src_aspect)
    except RuntimeError as e:
        _log(f"[ERR] {e}")
        return 2 if "depth" in str(e).lower() else 4
    except Exception as e:
        import traceback
        _log(f"[FATAL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return 5

    elapsed = time.perf_counter() - t_total
    _log(f"DONE: {out.name} elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
