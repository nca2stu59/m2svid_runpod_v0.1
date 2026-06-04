"""
concat_ffmpeg.py
──────────────────────────────────────────────────────────────────────
SBS mp4들을 ffmpeg으로 이어붙이기.

전략:
  1) stream copy: ffmpeg -f concat -safe 0 -i list.txt -c copy out.mp4
  2) 실패 시 libx264 CRF18 re-encode

단독 실행:
    python concat_ffmpeg.py --sbs-dir DIR --out OUT.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def _build_list_file(files: list[Path], list_path: Path):
    with open(list_path, "w", encoding="utf-8") as f:
        for p in files:
            p_str = str(p.resolve()).replace("'", r"'\''")
            f.write(f"file '{p_str}'\n")


def concat_sbs(files: list[Path], out_path: Path) -> tuple[bool, str]:
    if not files:
        return False, "no input files"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpd:
        list_path = Path(tmpd) / "list.txt"
        _build_list_file(files, list_path)

        # 1단계: stream copy
        cmd_copy = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(out_path),
        ]
        r = subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return True, f"stream copy OK: {out_path.name}"

        # 2단계: re-encode fallback
        cmd_enc = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        r2 = subprocess.run(cmd_enc, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r2.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return True, f"re-encoded (copy failed): {out_path.name}"

        err = r2.stderr.decode("utf-8", errors="replace")[-500:]
        return False, f"ffmpeg concat failed: {err}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Concat SBS mp4s")
    parser.add_argument("--sbs-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pattern", default="shot*_sbs.mp4")
    args = parser.parse_args()

    sbs_dir = Path(args.sbs_dir)
    out_path = Path(args.out)
    files = sorted(sbs_dir.glob(args.pattern))
    if not files:
        print(f"[concat] no files matched: {sbs_dir}/{args.pattern}", file=sys.stderr)
        return 2

    print(f"[concat] {len(files)} files → {out_path}")
    ok, msg = concat_sbs(files, out_path)
    print(f"[concat] {msg}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
