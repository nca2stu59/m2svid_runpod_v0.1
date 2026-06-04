"""
m2svid_worker.py — v0.16m M2SVid per-cut orchestrator.

Replaces v0.16b's `genstereo_worker.py` (StereoCrafter SVD inpaint).

흐름:
  cuts_metadata.json (AutoShot 산출)
    -> 컷마다 m2svid_per_cut_runner.py subprocess (m2svid_service .venv) 1회
       (runner 내부 단계: preprocess -> depth(.venv-vda) -> warp -> inpaint
                        -> SBS compose -> upscale lanczos/RTX VSR)
    -> 각 process exit 시 OS-level VRAM 100% 회수 ✓
  -> out_sbs_dir/shot###_sbs.mp4

CLI:
    --cuts-meta PATH            cuts_metadata.json (필수)
    --out PATH                  SBS mp4 출력 디렉토리 (필수)
    --shot-classes PATH         (선택) shotclass_worker.py 산출 shot_classes.json
                                지정 시 컷별 class (closeup/normal/wide) 에 따라
                                disparity_perc 자동 스케일 (closeup×0.5, wide×1.5)
    --processing-dim N          처리 해상도 max (default 512, 64-div)
    --output-dim N              출력 해상도 max (default 0 = same as processing)
    --depth-backend NAME        VDA-S | VDA-L | FlashDepth-L | FlashDepth-S |
                                FlashDepth | DepthCrafter (default VDA-S)
    --upscaler NAME             lanczos | rtx_vsr (default lanczos)
    --rtx-vsr-quality N         0-19 (default 4 = ULTRA)
    --disparity-perc F          0.02 default
    --seed N                    42 default
    --mask-antialias N          0 default
    --chunk-size N              M2SVid temporal window (default 25;
                                Resolution Overdrive uses 12)
    --fail-fast                 첫 컷 실패 시 즉시 중단

m2svid_service 경로 (default, override 가능):
    --m2svid-service PATH       C:\\Users\\PC\\Desktop\\m2svid_service
    --m2svid-python PATH        m2svid_service/.venv/Scripts/python.exe

이벤트 (stdout JSONL):
    {"event":"start", n_cuts:N, ...}
    {"event":"cut_start", "shot_id":i, ...}
    {"event":"stage_log", "shot_id":i, "label":"runner", "line":"..."}
    {"event":"cut_done", "shot_id":i, "sbs":path, "total_sec":T}
    {"event":"cut_error", "shot_id":i, "rc":rc, "message":...}
    {"event":"done", "sec":T, "n_ok":N, "n_fail":M, "results":[...]}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent


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


DEFAULT_M2SVID_SERVICE = _default_service_root()
RUNNER_SCRIPT = HERE / "local_engines" / "m2svid" / "m2svid_per_cut_runner.py"


# Mirror human-readable progress to stderr ONLY when invoked directly (TTY).
# When run via orchestrator (run_pipeline._stream_subprocess), stdout/stderr
# are pipes — orchestrator's _cli_progress handles formatting; mirror would
# cause duplicate lines.
_MIRROR_STDERR = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False


def _emit(event: str, **kwargs):
    """Emit JSONL on stdout (machine) + (optional) human-readable on stderr (TTY)."""
    payload = {"event": event, **kwargs}
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    if not _MIRROR_STDERR:
        return
    # human-readable mirror for terminal viewing
    ts = time.strftime("%H:%M:%S")
    if event == "start":
        sys.stderr.write(f"[{ts}] === m2svid_worker start (cuts={kwargs.get('n_cuts','?')},"
                         f" depth={kwargs.get('depth_backend','?')},"
                         f" upscaler={kwargs.get('upscaler','?')},"
                         f" per_shot_disp={kwargs.get('per_shot_disparity', False)}) ===\n")
    elif event == "cut_start":
        sid = kwargs.get("shot_id")
        cls = kwargs.get("shot_class")
        disp = kwargs.get("disparity_perc")
        sys.stderr.write(f"[{ts}] [shot{sid}] start"
                         + (f" class={cls}" if cls else "")
                         + (f" disp={disp}" if disp else "")
                         + "\n")
    elif event == "cut_done":
        sid = kwargs.get("shot_id")
        sec = kwargs.get("total_sec", 0)
        sz = kwargs.get("size_mb", 0)
        sys.stderr.write(f"[{ts}] [shot{sid}] done {sec:.1f}s ({sz:.2f} MB)\n")
    elif event == "cut_skipped":
        sid = kwargs.get("shot_id")
        sz = kwargs.get("size_mb", 0)
        sys.stderr.write(f"[{ts}] [shot{sid}] cached ({sz:.2f} MB)\n")
    elif event in ("cut_error", "subprocess_error", "subprocess_timeout", "subprocess_spawn_error"):
        sid = kwargs.get("shot_id", "?")
        msg = kwargs.get("message") or kwargs.get("rc") or ""
        sys.stderr.write(f"[{ts}] [ERR] [shot{sid}] {event}: {msg}\n")
    elif event == "done":
        ok = kwargs.get("n_ok", 0)
        fail = kwargs.get("n_fail", 0)
        n = kwargs.get("n_total", 0)
        sec = kwargs.get("sec", 0)
        if fail:
            sys.stderr.write(f"[{ts}] === DONE: {ok}/{n} ok, {fail} FAIL ({sec:.1f}s) ===\n")
        else:
            sys.stderr.write(f"[{ts}] === DONE: {ok}/{n} ok ({sec:.1f}s) ===\n")
    elif event == "warn":
        sys.stderr.write(f"[{ts}] [WARN] {kwargs.get('message','')}\n")
    elif event == "stage_log":
        # forwarded runner stdout — show only important markers (filter noise)
        line = kwargs.get("line", "")
        important_markers = ("[1/5]", "[2/5]", "[3/5]", "[4/5]", "[5/5]",
                             "DONE:", "[ERR]", "[FATAL]", "Traceback",
                             "depth done", "warp done", "inpaint done",
                             "RTX VSR:", "upscale target",
                             "[STEREO_")
        if any(m in line for m in important_markers):
            sid = kwargs.get("shot_id", "?")
            sys.stderr.write(f"[{ts}] [shot{sid}] {line}\n")
    sys.stderr.flush()


# Discrete class → disparity multiplier (matches m2svid_service _disparity_multiplier
# at endpoints; normal=1.0). Continuous multiplier from shot_scale would be more
# precise but v0.16b shotclass output is discrete.
_DISPARITY_MULTIPLIER = {
    "closeup": 0.5,
    "normal": 1.0,
    "wide": 1.5,
}


def _load_shot_classes(path: Path | None) -> dict:
    """Returns dict {shot_id (int) -> 'closeup'|'normal'|'wide'} or empty."""
    if path is None or not path.exists():
        return {}
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        _emit("warn", message=f"shot_classes load failed: {e}")
        return {}
    shots = m.get("shots", {})
    out: dict[int, str] = {}
    if isinstance(shots, dict):
        for k, v in shots.items():
            try:
                sid = int(k)
                cls = v.get("class") or v.get("shot_class")
                if cls in _DISPARITY_MULTIPLIER:
                    out[sid] = cls
            except Exception:
                continue
    elif isinstance(shots, list):
        for v in shots:
            try:
                sid = int(v.get("shot_id"))
                cls = v.get("class") or v.get("shot_class")
                if cls in _DISPARITY_MULTIPLIER:
                    out[sid] = cls
            except Exception:
                continue
    return out


def _run_subprocess(cmd: list[str], shot_id: int, label: str,
                    timeout: int | None = None) -> int:
    """Run subprocess, forward stdout lines as stage_log events."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1, env=env,
        )
    except Exception as e:
        _emit("subprocess_spawn_error", shot_id=shot_id, label=label,
              message=str(e))
        return 99

    rc = 0
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                _emit("stage_log", shot_id=shot_id, label=label, line=line)
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = 124
        _emit("subprocess_timeout", shot_id=shot_id, label=label)
    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        rc = 125
        _emit("subprocess_error", shot_id=shot_id, label=label, message=str(e))
    return rc


def _process_cut(seg: dict, sbs_out_dir: Path, args, m2svid_python: Path,
                 m2svid_service: Path,
                 shot_class_lookup: dict) -> tuple[bool, str]:
    """단일 컷: m2svid_per_cut_runner.py subprocess 1회 -> SBS .mp4."""
    shot_id = int(seg["shot_id"])
    cut_file = Path(seg["file"]).resolve()
    if not cut_file.exists():
        _emit("cut_error", shot_id=shot_id, step="preflight",
              message=f"cut file not found: {cut_file}")
        return False, "cut file not found"

    sbs_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = sbs_out_dir / f"shot{shot_id:03d}_sbs.mp4"
    tmp_dir = sbs_out_dir / "_m2svid_tmp" / f"shot{shot_id:03d}"

    # Per-cut cache: skip if SBS already produced (and --force-rerun NOT set).
    # File must be at least 4 KB to count as valid (tiny files = corrupt).
    if not args.force_rerun and out_path.exists() and out_path.stat().st_size > 4096:
        size_mb = round(out_path.stat().st_size / 1024**2, 2)
        _emit("cut_skipped", shot_id=shot_id, sbs=str(out_path),
              size_mb=size_mb, reason="cache hit")
        return True, ""

    # Per-shot disparity scaling (if shot_classes loaded)
    cls = shot_class_lookup.get(shot_id)
    mult = _DISPARITY_MULTIPLIER.get(cls, 1.0) if cls else 1.0
    effective_disp = round(args.disparity_perc * mult, 5)

    _emit("cut_start", shot_id=shot_id, file=str(cut_file),
          processing_dim=args.processing_dim, output_dim=args.output_dim,
          depth_backend=args.depth_backend, upscaler=args.upscaler,
          shot_class=cls, disparity_perc=effective_disp,
          disparity_multiplier=mult)

    cmd = [
        str(m2svid_python), "-u", str(RUNNER_SCRIPT),
        "--cut", str(cut_file),
        "--out", str(out_path),
        "--tmp-dir", str(tmp_dir),
        "--processing-dim", str(args.processing_dim),
        "--output-dim", str(args.output_dim),
        "--depth-backend", str(args.depth_backend),
        "--upscaler", str(args.upscaler),
        "--rtx-vsr-quality", str(args.rtx_vsr_quality),
        "--disparity-perc", str(effective_disp),
        "--seed", str(args.seed),
        "--mask-antialias", str(args.mask_antialias),
        "--chunk-size", str(args.chunk_size),
        "--m2svid-service", str(m2svid_service),
    ]

    t0 = time.time()
    rc = _run_subprocess(cmd, shot_id, "runner", timeout=args.timeout)
    elapsed = round(time.time() - t0, 2)

    if rc != 0:
        _emit("cut_error", shot_id=shot_id, step="runner", rc=rc,
              message=f"runner exit {rc}")
        return False, f"runner rc={rc}"

    if not out_path.exists():
        _emit("cut_error", shot_id=shot_id, step="output",
              message=f"runner ok but SBS not produced: {out_path}")
        return False, "no output"

    size_mb = round(out_path.stat().st_size / 1024**2, 2)
    _emit("cut_done", shot_id=shot_id, sbs=str(out_path),
          total_sec=elapsed, size_mb=size_mb)
    return True, ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cuts-meta", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shot-classes", default=None,
                   help="(optional) shot_classes.json — per-cut disparity scaling")
    p.add_argument("--processing-dim", type=int, default=512)
    p.add_argument("--output-dim", type=int, default=0)
    p.add_argument("--depth-backend", default="VDA-S",
                   choices=["VDA-S", "VDA-L", "FlashDepth-L", "FlashDepth-S",
                            "FlashDepth", "DepthCrafter"])
    p.add_argument("--upscaler", default="lanczos",
                   choices=["lanczos", "rtx_vsr"])
    p.add_argument("--rtx-vsr-quality", type=int, default=4)
    p.add_argument("--disparity-perc", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask-antialias", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=25,
                   help="M2SVid temporal window per generate() call "
                        "(default 25 = training window; Resolution Overdrive uses 12)")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--force-rerun", action="store_true",
                   help="이전 결과 무시하고 모든 컷 재처리 (default: skip-if-exists)")
    p.add_argument("--timeout", type=int, default=None,
                   help="per-cut subprocess timeout (seconds)")
    p.add_argument("--m2svid-service",
                   default=str(DEFAULT_M2SVID_SERVICE))
    p.add_argument("--m2svid-python",
                   default=str(_venv_python(DEFAULT_M2SVID_SERVICE)))
    args = p.parse_args()

    cuts_meta_p = Path(args.cuts_meta).resolve()
    sbs_out = Path(args.out).resolve()
    m2svid_service = Path(args.m2svid_service).resolve()
    m2svid_python = Path(args.m2svid_python).resolve()

    if not cuts_meta_p.exists():
        _emit("done", sec=0, n_ok=0, n_fail=0, n_total=0,
              error=f"cuts-meta not found: {cuts_meta_p}")
        return 1
    if not RUNNER_SCRIPT.exists():
        _emit("done", sec=0, n_ok=0, n_fail=0, n_total=0,
              error=f"runner script missing: {RUNNER_SCRIPT}")
        return 1
    if not m2svid_python.exists():
        _emit("done", sec=0, n_ok=0, n_fail=0, n_total=0,
              error=f"m2svid python missing: {m2svid_python}")
        return 1

    meta = json.loads(cuts_meta_p.read_text(encoding="utf-8"))
    segments = meta.get("segments", meta if isinstance(meta, list) else [])

    shot_class_lookup = _load_shot_classes(
        Path(args.shot_classes).resolve() if args.shot_classes else None
    )

    _emit("start",
          n_cuts=len(segments),
          processing_dim=args.processing_dim,
          output_dim=args.output_dim,
          depth_backend=args.depth_backend,
          upscaler=args.upscaler,
          m2svid_service=str(m2svid_service),
          shot_classes=len(shot_class_lookup),
          per_shot_disparity=bool(shot_class_lookup))

    t_all = time.time()
    results: list[dict] = []
    n_ok = 0
    n_fail = 0
    failed_at: int | None = None

    for seg in segments:
        ok, err = _process_cut(seg, sbs_out, args, m2svid_python, m2svid_service,
                               shot_class_lookup)
        results.append({"shot_id": int(seg["shot_id"]), "ok": ok,
                        **({"error": err} if not ok else {})})
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            failed_at = int(seg["shot_id"])
            if args.fail_fast:
                break

    total = round(time.time() - t_all, 2)
    _emit("done", sec=total, n_ok=n_ok, n_fail=n_fail,
          failed_at=failed_at, n_total=len(segments), results=results)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
