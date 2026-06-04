"""Subprocess worker — generates the M2SVid output for one shot, then exits.

The OS-level exit guarantees VRAM is fully reclaimed before the next shot.
Use this for long batches on smaller GPUs or when you need restartability.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _log(msg: str) -> None:
    print(f"[inpaint_worker] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--repro", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--shot_start", type=int, required=True)
    p.add_argument("--shot_end", type=int, required=True)
    p.add_argument("--output", required=True, help=".pt file path for the generated tensor")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--mask_antialias", type=int, default=0)
    args = p.parse_args()

    from inpaint_core import generate_shot_tensor  # imports torch — slow

    t0 = time.perf_counter()
    out = generate_shot_tensor(
        video_path=Path(args.video),
        repro=Path(args.repro),
        mask=Path(args.mask),
        shot_start=args.shot_start,
        shot_end=args.shot_end,
        seed=args.seed,
        config_path=Path(args.config) if args.config else None,
        ckpt_path=Path(args.ckpt) if args.ckpt else None,
        mask_antialias=args.mask_antialias,
        log_fn=_log,
    )
    import torch
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    _log(f"shot [{args.shot_start}:{args.shot_end}] -> {args.output} "
         f"shape={tuple(out.shape)} elapsed={time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
