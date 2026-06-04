"""
autoshot_worker.py
──────────────────────────────────────────────────────────────────────
Subprocess A — AutoShot으로 영상을 컷 단위로 분할.

단독 프로세스로 실행되어 종료 시 OS가 VRAM을 전부 회수.

stdout에 JSON 라인으로 진행상황 출력.

입력:
    --video         원본 영상 경로
    --out           분할 mp4들이 저장될 디렉토리
    --threshold     AutoShot 임계값 (기본 0.296)
    --min-duration  최소 컷 길이 (초, 기본 0.0)
    --weights       ckpt_0_200_0.pth 경로

출력:
    {out}/{stem}_shot###.mp4
    {out}/cuts_metadata.json

이벤트:
    {"event":"start", ...}
    {"event":"done", "n_segments":N, ...}
    {"event":"error", "message":...}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Windows cp949 stdout 회피
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


HERE = Path(__file__).resolve().parent
PORT_ROOT = HERE.parent
# 2026-05-10: Tier 1 consolidation — autoshot moved to GenStereoBackend/dependency/
AUTOSHOT_DIR = PORT_ROOT / "GenStereoBackend" / "dependency" / "autoshot"
DEFAULT_WEIGHTS = AUTOSHOT_DIR / "ckpt_0_200_0.pth"


def _emit(event: str, **kwargs):
    print(json.dumps({"event": event, **kwargs}, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoShot subprocess worker")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.296)
    parser.add_argument("--min-duration", type=float, default=0.0)
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    args = parser.parse_args()

    video = Path(args.video).resolve()
    out_dir = Path(args.out).resolve()
    weights = Path(args.weights).resolve()

    if not video.exists():
        _emit("error", message=f"video not found: {video}")
        return 2
    if not weights.exists():
        _emit("error", message=f"AutoShot weights not found: {weights}")
        return 2

    sys.path.insert(0, str(AUTOSHOT_DIR))
    try:
        import autoshot_splitter  # noqa: E402
    except Exception as e:
        _emit("error", message=f"import autoshot_splitter failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return 3

    out_dir.mkdir(parents=True, exist_ok=True)

    _emit("start", video=str(video), out=str(out_dir), threshold=args.threshold)
    t0 = time.time()

    try:
        meta = autoshot_splitter.run_pipeline(
            video_path=str(video),
            output_dir=str(out_dir),
            threshold=args.threshold,
            min_duration=args.min_duration,
            weights_path=str(weights),
        )
    except Exception as e:
        _emit("error", message=f"autoshot pipeline failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return 4

    n_segments = meta.get("n_segments", 0)
    _emit(
        "done",
        n_segments=n_segments,
        n_scenes=meta.get("n_scenes", 0),
        sec=round(time.time() - t0, 2),
        meta_path=str(out_dir / "cuts_metadata.json"),
    )
    return 0 if n_segments > 0 else 5


if __name__ == "__main__":
    sys.exit(main())
