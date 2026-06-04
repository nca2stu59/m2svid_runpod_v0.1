"""
run_pipeline.py — v0.16 (원점 재구현, simple GenStereo wrapper)
──────────────────────────────────────────────────────────────────────
오케스트레이터: AutoShot(A) → Shot Classifier(C) → GenStereo wrapper(G) → concat.

설계 원칙 (v0.16):
  - **manifest 인프라 없음**: cuts_metadata.json + shot_classes.json 두 단일 파일만 사용
  - **GenStereo native CLI를 그대로 활용**: depth+splat / inpaint 두 step을 컷마다 별도
    subprocess로 실행. 각 step exit 시 OS-level VRAM 100% 회수.
  - "하나의 파일 처리가 끝나면 반드시 VRAM을 반환"의 가장 단순한 구현.
  - StereoCrafter venv / DepthCrafter / per-stage worker / quanto / cpu_offload 등
    v0.13s ~ v0.15의 복잡한 옵션은 모두 제거.

사용:
    python run_pipeline.py --video INPUT.mp4 --out ./output [...]
    python run_pipeline.py --video INPUT.mp4 --out ./output --no-shotclass

출력:
    {out}/{stem}_{ts}/
        cuts/                       # AutoShot
            {stem}_shot###.mp4
            cuts_metadata.json
        shot_classes/               # Shot Classifier (skip 시 부재)
            shot_classes.json
            thumbnails/
        sbs/                        # GenStereo 컷별 SBS
            shot###_sbs.mp4
        final_sbs.mp4               # concat 활성 시
        logs/
            autoshot_stdout.jsonl
            shotclass_stdout.jsonl  (skip 시 부재)
            genstereo_stdout.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Optional

# Windows cp949 stdout 회피 — Korean 경로/em-dash 등 출력 시 크래시 방지
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


HERE = Path(__file__).resolve().parent
PORT_ROOT = HERE.parent


def _default_service_root() -> Path:
    env = os.environ.get("M2SVID_SERVICE_ROOT") or os.environ.get("RUNPOD_M2SVID_SERVICE")
    if env:
        return Path(env)
    if os.name == "nt":
        return Path(r"C:\Users\PC\Desktop\m2svid_service")
    return Path("/workspace/m2svid_service")


def venv_python(root: Path, env_name: str = ".venv") -> Path:
    """Return venv Python path for Windows or Linux containers."""
    if os.name == "nt":
        return root / env_name / "Scripts" / "python.exe"
    return root / env_name / "bin" / "python"


# 2026-05-10: Tier 1 consolidation - shot_classifier moved to GenStereoBackend/dependency/
SHOT_CLASSIFIER_DIR = Path(
    os.environ.get("SHOT_CLASSIFIER_DIR")
    or os.environ.get("RUNPOD_SHOTCLASS_DIR")
    or PORT_ROOT / "GenStereoBackend" / "dependency" / "shot_classifier"
)
# 2026-05-10: StereoCrafter folder removed; backend now at GenStereoBackend
STEREOCRAFTER_DIR = Path(os.environ.get("GENSTEREO_BACKEND_DIR") or PORT_ROOT / "GenStereoBackend")

# Shot Classifier 전용 venv → 없으면 SC venv (동일 deps) → 둘 다 없으면 에러
DEFAULT_SHOTCLASS_VENV_PY = venv_python(SHOT_CLASSIFIER_DIR, "venv")
# Fallback (Tier 1): shot_classifier no longer has standalone venv.
# Use GenStereoBackend/python_embed (has torch/diffusers/transformers, sentencepiece manual install if needed).
DEFAULT_SHOTCLASS_FALLBACK_PY = (
    STEREOCRAFTER_DIR / "python_embed" / "python.exe"
    if os.name == "nt" else Path(sys.executable)
)
DEFAULT_SHOTCLASS_MODELS = Path(os.environ.get("SHOTCLASS_MODELS_DIR") or SHOT_CLASSIFIER_DIR / "models")

# GenStereo (외부 프로젝트, 별도 portable Python)
DEFAULT_M2SVID_SERVICE = _default_service_root()
DEFAULT_M2SVID_PYTHON = venv_python(DEFAULT_M2SVID_SERVICE)


def _detect_shotclass_python(override: Optional[str]) -> Path:
    """shot_classifier 전용 venv → 없으면 StereoCrafter venv → 둘 다 없으면 에러."""
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"shotclass python not found: {p}")
        return p
    if DEFAULT_SHOTCLASS_VENV_PY.exists():
        return DEFAULT_SHOTCLASS_VENV_PY
    if DEFAULT_SHOTCLASS_FALLBACK_PY.exists():
        return DEFAULT_SHOTCLASS_FALLBACK_PY
    raise FileNotFoundError(
        f"shotclass python not found. Tried:\n"
        f"  {DEFAULT_SHOTCLASS_VENV_PY}\n"
        f"  {DEFAULT_SHOTCLASS_FALLBACK_PY}\n"
        f"Pass --shotclass-python /path/to/python.exe or run shot_classifier/setup.bat."
    )


def _detect_autoshot_python(override: Optional[str]) -> Path:
    """transnetv2_pytorch가 설치된 Python을 찾는다.

    탐색 순서: override → py launcher (-3.11/-3.10/-3) → PATH의 'python' →
    sys.executable. 각 후보에서 `import transnetv2_pytorch` 시도.
    """
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"autoshot python not found: {p}")
        return p

    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path):
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp not in seen:
            seen.add(rp)
            candidates.append(p)

    for arg in ["-3.11", "-3.10", "-3"]:
        try:
            r = subprocess.run(
                ["py", arg, "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                p = Path(r.stdout.strip())
                if p.exists():
                    _add(p)
        except Exception:
            pass

    try:
        r = subprocess.run(
            ["where", "python"], capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if "WindowsApps" in str(p):
                    continue
                if p.exists():
                    _add(p)
    except Exception:
        pass

    _add(Path(sys.executable))

    errs: list[str] = []
    for c in candidates:
        try:
            r = subprocess.run(
                [str(c), "-c", "import transnetv2_pytorch"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                return c
            errs.append(f"  {c}: {r.stderr.strip()[:100]}")
        except Exception as e:
            errs.append(f"  {c}: {e}")

    raise FileNotFoundError(
        "No Python with transnetv2_pytorch installed.\n"
        "Tried:\n" + "\n".join(errs) + "\n\n"
        "Install in any Python: pip install transnetv2-pytorch ffmpeg-python\n"
        "Then pass --autoshot-python /path/to/python.exe"
    )


@dataclass
class PipelineConfig:
    video: str
    out: str
    # AutoShot
    threshold: float = 0.296
    min_duration: float = 0.0
    autoshot_weights: Optional[str] = None
    autoshot_python: Optional[str] = None
    # Shot Classifier (v0.16b: default 'siglip2', was 'clip' in v0.16)
    use_shotclass: bool = True
    shotclass_required: bool = False
    shotclass_backend: str = "siglip2"
    shotclass_python: Optional[str] = None
    shotclass_models_dir: Optional[str] = None
    depth_std_wide: float = 0.25
    depth_std_closeup: float = 0.12
    max_disp_wide: float = 30.0
    max_disp_normal: float = 20.0
    max_disp_closeup: float = 12.0
    # M2SVid (v0.16m)
    processing_dim: int = 512        # max processing dim, 64-divisible (slider 384..1024)
    output_dim: int = 0              # max output dim, 0=same as processing (slider 0..2160)
    depth_backend: str = "VDA-S"     # VDA-S | VDA-L | FlashDepth-L | FlashDepth-S | FlashDepth | DepthCrafter
    upscaler: str = "lanczos"        # lanczos | rtx_vsr
    rtx_vsr_quality: int = 4         # 0-19 (4=ULTRA default; only used if upscaler=rtx_vsr)
    disparity_perc: float = 0.02     # warp disparity percent of width
    seed: int = 42
    mask_antialias: int = 0
    per_cut_timeout: int = 0         # per-cut subprocess timeout (0 = no limit)
    m2svid_service: Optional[str] = None      # default: C:\Users\PC\Desktop\m2svid_service
    m2svid_python: Optional[str] = None       # default: {m2svid_service}/.venv/Scripts/python.exe
    # Resolution Overdrive (v0.17.3+)
    m2svid_chunk_size: int = 25      # M2SVid temporal window per generate() call.
                                     # Default 25 = training window. Smaller (12, 8) frees
                                     # VRAM at the cost of weaker temporal coherence /
                                     # more chunk seams. Used by Overdrive presets.
    m2svid_output_suffix: str = ""   # appended to sbs/ + final_sbs.mp4 names
                                     # (e.g. "_overdrive_12f720" → sbs_overdrive_12f720/,
                                     # final_sbs_overdrive_12f720.mp4). "" = default paths.
    # FPS normalization (29.97/23.976 등 비정수 fps drift 방지):
    #   "ceil"  — 올림 정수 fps로 사전 transcode (기본값)
    #   "round" — 반올림 정수 fps로 사전 transcode
    #   "off"   — 변환 없음
    normalize_fps: str = "ceil"
    # Orchestration
    concat: bool = True
    fail_fast: bool = True
    # Stage caching (v0.17.1+)
    out_dir: Optional[str] = None    # 정확한 base_out 경로 override (None = 자동 hash 기반)
    force_rerun: str = ""            # "all" | comma-separated subset of {autoshot,classifier,m2svid,concat}
    # Per-stage external imports (v0.17.1+)
    # Path 지정 시 해당 자료를 base_out 으로 복사 → 이후 cache hit 으로 단계 skip.
    import_cuts: Optional[str] = None             # 외부 cuts dir (cuts_metadata.json + shot*.mp4)
    import_shot_classes: Optional[str] = None     # 외부 shot_classes.json 파일
    import_sbs_dir: Optional[str] = None          # 외부 SBS dir (shot*_sbs.mp4)
    import_final_sbs: Optional[str] = None        # 외부 final_sbs.mp4 파일
    # Stage-only execution (v0.17.2+)
    # True 면 M2SVid 단계 자체를 skip (AutoShot/Classifier 만 실행할 때 사용)
    skip_m2svid: bool = False


@dataclass
class PipelineResult:
    out_dir: str
    cuts_dir: str
    shotclass_dir: Optional[str]
    shot_classes_json: Optional[str]
    sbs_dir: str
    cuts_metadata: str
    cut_sbs_files: list[str]
    final_sbs: Optional[str]
    logs_dir: str
    elapsed_sec: float
    n_cuts: int
    n_ok: int
    n_fail: int
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


ProgressCB = Callable[[str, dict], None]


def _noop_progress(event: str, payload: dict):
    pass


# ── Stage caching helpers (v0.17.1+) ────────────────────────────────────── #

def _video_content_hash(path: Path, n_bytes: int = 1024 * 1024) -> str:
    """Quick content-addressed hash of a video file.

    Hashes file size + first N bytes + last N bytes (default 1 MB each).
    Stable for same content, fast for large files (no full read).
    """
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    h.update(path.name.encode("utf-8", errors="replace"))
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
        if size > 2 * n_bytes:
            f.seek(-n_bytes, 2)
            h.update(f.read(n_bytes))
    return h.hexdigest()[:12]


def _parse_force_rerun(spec: str) -> set[str]:
    """Parse --force-rerun argument into set of stages.

    Accepted: 'all' | 'autoshot' | 'classifier' | 'm2svid' | 'concat'
    Multiple via comma: 'autoshot,classifier' → {'autoshot', 'classifier'}
    Empty/None → empty set (no force).
    """
    if not spec:
        return set()
    parts = {p.strip().lower() for p in str(spec).split(",") if p.strip()}
    if "all" in parts:
        return {"autoshot", "classifier", "m2svid", "concat"}
    valid = {"autoshot", "classifier", "m2svid", "concat"}
    return parts & valid


def _autoshot_cache_valid(cuts_dir: Path) -> tuple[bool, int, str]:
    """Returns (is_valid, n_cuts, reason). True if AutoShot can be skipped."""
    meta = cuts_dir / "cuts_metadata.json"
    if not meta.exists():
        return False, 0, "cuts_metadata.json not found"
    try:
        with open(meta, "r", encoding="utf-8") as f:
            m = json.load(f)
    except Exception as e:
        return False, 0, f"cuts_metadata.json invalid: {e}"
    n = int(m.get("n_segments", 0)) or len(m.get("segments", []))
    if n == 0:
        return False, 0, "n_segments=0"
    segments = m.get("segments", [])
    missing = [s.get("file") for s in segments if not s.get("file") or not Path(s["file"]).exists()]
    if missing:
        return False, n, f"{len(missing)}/{n} cut files missing on disk"
    return True, n, "OK"


def _classifier_cache_valid(shotclass_dir: Path) -> tuple[bool, str]:
    """Returns (is_valid, reason)."""
    j = shotclass_dir / "shot_classes.json"
    if not j.exists():
        return False, "shot_classes.json not found"
    try:
        with open(j, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        return False, f"shot_classes.json invalid: {e}"
    shots = d.get("shots")
    if not shots or (isinstance(shots, dict) and len(shots) == 0):
        return False, "shots empty"
    return True, "OK"


def _import_cuts(src_dir: Path, dst_cuts_dir: Path, progress: ProgressCB) -> bool:
    """Copy external cuts dir (cuts_metadata.json + shot*.mp4) into dst_cuts_dir.

    Re-writes 'file' paths in cuts_metadata.json so they point to dst_cuts_dir.
    """
    src_meta = src_dir / "cuts_metadata.json"
    if not src_meta.exists():
        progress("orchestrator:import_failed",
                 {"stage": "cuts", "reason": f"cuts_metadata.json not in {src_dir}"})
        return False
    dst_cuts_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(src_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # copy per-cut .mp4 files into dst, rewrite paths
        new_segments = []
        for seg in meta.get("segments", []):
            old_path = Path(seg["file"])
            base_name = old_path.name
            new_path = dst_cuts_dir / base_name
            if not new_path.exists() and old_path.exists():
                shutil.copy2(old_path, new_path)
            elif not old_path.exists() and not new_path.exists():
                # try locating by name in src_dir
                cand = src_dir / base_name
                if cand.exists():
                    shutil.copy2(cand, new_path)
            seg = dict(seg, file=str(new_path))
            new_segments.append(seg)
        meta = dict(meta, segments=new_segments)
        with open(dst_cuts_dir / "cuts_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        progress("orchestrator:import_done",
                 {"stage": "cuts", "n_segments": len(new_segments),
                  "src": str(src_dir), "dst": str(dst_cuts_dir)})
        return True
    except Exception as e:
        progress("orchestrator:import_failed",
                 {"stage": "cuts", "reason": str(e)})
        return False


def _import_shot_classes(src_json: Path, dst_dir: Path, progress: ProgressCB) -> bool:
    if not src_json.exists():
        progress("orchestrator:import_failed",
                 {"stage": "shot_classes", "reason": f"file not found: {src_json}"})
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src_json, dst_dir / "shot_classes.json")
        progress("orchestrator:import_done",
                 {"stage": "shot_classes",
                  "src": str(src_json), "dst": str(dst_dir / "shot_classes.json")})
        return True
    except Exception as e:
        progress("orchestrator:import_failed",
                 {"stage": "shot_classes", "reason": str(e)})
        return False


def _import_sbs_dir(src_dir: Path, dst_dir: Path, progress: ProgressCB) -> bool:
    if not src_dir.exists() or not src_dir.is_dir():
        progress("orchestrator:import_failed",
                 {"stage": "sbs", "reason": f"not a directory: {src_dir}"})
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        for src_file in sorted(src_dir.glob("shot*_sbs.mp4")):
            dst_file = dst_dir / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                n += 1
        progress("orchestrator:import_done",
                 {"stage": "sbs", "n_copied": n,
                  "src": str(src_dir), "dst": str(dst_dir)})
        return True
    except Exception as e:
        progress("orchestrator:import_failed",
                 {"stage": "sbs", "reason": str(e)})
        return False


def _import_final_sbs(src_file: Path, dst_file: Path, progress: ProgressCB) -> bool:
    if not src_file.exists():
        progress("orchestrator:import_failed",
                 {"stage": "final_sbs", "reason": f"file not found: {src_file}"})
        return False
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src_file, dst_file)
        progress("orchestrator:import_done",
                 {"stage": "final_sbs", "src": str(src_file), "dst": str(dst_file)})
        return True
    except Exception as e:
        progress("orchestrator:import_failed",
                 {"stage": "final_sbs", "reason": str(e)})
        return False


def _stream_subprocess(
    cmd: list[str],
    jsonl_log_path: Path,
    progress: ProgressCB,
    source_tag: str,
) -> int:
    """자식 프로세스 stdout(JSONL) 라인을 progress 콜백 + 로그 파일로 forward."""
    jsonl_log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Windows pipe deadlock 회피 (~64KB 버퍼)
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    with open(jsonl_log_path, "w", encoding="utf-8") as log_f:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip("\r\n")
            if not line:
                continue
            log_f.write(line + "\n")
            log_f.flush()
            try:
                payload = json.loads(line)
                event = payload.get("event", "message")
                progress(f"{source_tag}:{event}", payload)
            except json.JSONDecodeError:
                progress(f"{source_tag}:raw", {"line": line})
    proc.wait()
    return proc.returncode


def _normalize_fps(video_path: Path, mode: str, output_dir: Path,
                   progress: ProgressCB) -> tuple[Path, dict]:
    """fps 정수 정규화 (NTSC drop-frame: 29.97→30, 23.976→24).

    Args:
        video_path: 입력 비디오
        mode: "off" | "ceil" | "round"
        output_dir: per-run base_out (transcode 결과 위치)
        progress: 진행 콜백

    Returns:
        (사용할 비디오 path, 진단 dict). 실패 시 원본 path 반환.
    """
    diag: dict = {"mode": mode, "normalized": False}
    if mode == "off":
        diag["reason"] = "off"
        return video_path, diag

    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate,r_frame_rate",
             "-of", "default=nokey=0:noprint_wrappers=1",
             str(video_path)],
            text=True, encoding="utf-8", errors="replace",
        )
    except Exception as e:
        diag.update({"reason": f"ffprobe failed: {e}"})
        progress("orchestrator:fps_normalize_skip", diag)
        return video_path, diag

    avg_str = r_str = None
    for line in out.strip().splitlines():
        if line.startswith("avg_frame_rate="):
            avg_str = line.split("=", 1)[1].strip()
        elif line.startswith("r_frame_rate="):
            r_str = line.split("=", 1)[1].strip()
    fps_str = avg_str or r_str
    if not fps_str or fps_str == "0/0":
        diag.update({"reason": f"fps detect failed (avg={avg_str} r={r_str})"})
        progress("orchestrator:fps_normalize_skip", diag)
        return video_path, diag

    try:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            num, den = int(num), int(den)
            if den == 0:
                raise ValueError("0 denominator")
            fps = num / den
        else:
            fps = float(fps_str)
    except Exception as e:
        diag.update({"reason": f"parse fps '{fps_str}' failed: {e}"})
        progress("orchestrator:fps_normalize_skip", diag)
        return video_path, diag

    diag["original_fps"] = round(fps, 6)

    if abs(fps - round(fps)) < 1e-6:
        diag.update({"reason": "already integer", "target_fps": int(round(fps))})
        progress("orchestrator:fps_normalize_skip", diag)
        return video_path, diag

    import math
    if mode == "ceil":
        target_fps = math.ceil(fps)
    elif mode == "round":
        target_fps = int(round(fps))
    else:
        diag.update({"reason": f"unknown mode: {mode}"})
        progress("orchestrator:fps_normalize_skip", diag)
        return video_path, diag

    diag["target_fps"] = target_fps

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_path.stem}__norm{target_fps}fps.mp4"

    progress("orchestrator:fps_normalize_start", diag)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={target_fps}",
        "-fps_mode", "cfr",
        "-c:v", "libx264", "-crf", "17", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    try:
        rc = subprocess.call(cmd)
    except Exception as e:
        diag.update({"reason": f"ffmpeg spawn failed: {e}"})
        progress("orchestrator:fps_normalize_failed", diag)
        return video_path, diag

    if rc != 0 or not out_path.exists():
        diag.update({"reason": f"ffmpeg rc={rc}", "fallback": "use original"})
        progress("orchestrator:fps_normalize_failed", diag)
        return video_path, diag

    diag.update({
        "normalized": True,
        "transcoded_path": str(out_path),
        "size_mb": round(out_path.stat().st_size / (1024 * 1024), 2),
    })
    progress("orchestrator:fps_normalize_done", diag)
    return out_path, diag


def _early_return(base_out, cuts_dir, sbs_dir, logs_dir, t_start,
                  err, warnings, **kwargs) -> PipelineResult:
    return PipelineResult(
        out_dir=str(base_out), cuts_dir=str(cuts_dir),
        shotclass_dir=kwargs.get("shotclass_dir"),
        shot_classes_json=kwargs.get("shot_classes_json"),
        sbs_dir=str(sbs_dir),
        cuts_metadata=kwargs.get("cuts_metadata", ""),
        cut_sbs_files=kwargs.get("cut_sbs_files", []),
        final_sbs=None,
        logs_dir=str(logs_dir),
        elapsed_sec=round(time.time() - t_start, 2),
        n_cuts=kwargs.get("n_cuts", 0),
        n_ok=kwargs.get("n_ok", 0),
        n_fail=kwargs.get("n_fail", 0),
        error=err, warnings=warnings,
    )


def run(cfg: PipelineConfig, progress: ProgressCB = _noop_progress) -> PipelineResult:
    t_start = time.time()
    warnings: list[str] = []

    # video is required only when out_dir is NOT explicitly provided
    # (out_dir given → stage-only run from existing data, video optional).
    video_path: Optional[Path] = None
    if cfg.video:
        video_path = Path(cfg.video).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"video not found: {video_path}")
    elif not cfg.out_dir:
        raise ValueError("either cfg.video or cfg.out_dir must be provided "
                         "(out_dir for stage-only runs without video)")

    if cfg.out_dir:
        base_out = Path(cfg.out_dir).resolve()
        stem = base_out.name  # for logging
    else:
        # video_path is guaranteed non-None here
        stem = video_path.stem
        vhash = _video_content_hash(video_path)
        base_out = Path(cfg.out).resolve() / f"{stem}_{vhash}"
    cuts_dir = base_out / "cuts"
    # Resolution Overdrive: m2svid_output_suffix changes sbs/ + final_sbs.mp4 paths
    # so multiple presets coexist (sbs/, sbs_overdrive_12f720/, sbs_extreme_8f832/, ...).
    sbs_suffix = (cfg.m2svid_output_suffix or "").strip()
    sbs_dir = base_out / f"sbs{sbs_suffix}"
    logs_dir = base_out / "logs"
    shotclass_dir = base_out / "shot_classes"
    for d in (cuts_dir, sbs_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    force = _parse_force_rerun(cfg.force_rerun)

    progress("orchestrator:start", {
        "video": str(video_path) if video_path else None, "out_dir": str(base_out),
        "use_shotclass": cfg.use_shotclass, "backend": cfg.shotclass_backend,
        "version": "0.17.1",
        "cache_dir": str(base_out),
        "force_rerun": sorted(force) if force else [],
    })

    # ── FPS normalize (default ceil — NTSC drop-frame fps 정수화) ────
    norm_dir = base_out / "_normalized_input"
    video_path, fps_diag = _normalize_fps(
        video_path, cfg.normalize_fps, norm_dir, progress,
    )
    if fps_diag.get("normalized"):
        warnings.append(
            f"input fps normalized: {fps_diag.get('original_fps')} → "
            f"{fps_diag.get('target_fps')} ({fps_diag.get('mode')}) "
            f"@ {fps_diag.get('transcoded_path')}"
        )

    # ── External imports (apply BEFORE cache checks so they trigger skip) ──
    if cfg.import_cuts:
        _import_cuts(Path(cfg.import_cuts).resolve(), cuts_dir, progress)
    if cfg.import_shot_classes:
        _import_shot_classes(Path(cfg.import_shot_classes).resolve(),
                             shotclass_dir, progress)
    if cfg.import_sbs_dir:
        _import_sbs_dir(Path(cfg.import_sbs_dir).resolve(), sbs_dir, progress)
    if cfg.import_final_sbs:
        _import_final_sbs(Path(cfg.import_final_sbs).resolve(),
                          base_out / f"final_sbs{sbs_suffix}.mp4", progress)

    # ── Subprocess A: AutoShot ────────────────────────────────────────
    cuts_metadata = cuts_dir / "cuts_metadata.json"

    # Cache check: skip AutoShot if cuts already produced and valid.
    autoshot_cached = False
    if "autoshot" not in force:
        ok, n_cached, why = _autoshot_cache_valid(cuts_dir)
        if ok:
            progress("orchestrator:autoshot_skipped",
                     {"n_cuts": n_cached, "reason": "cache hit", "cuts_dir": str(cuts_dir)})
            autoshot_cached = True
        else:
            progress("orchestrator:autoshot_cache_miss", {"reason": why})

    if not autoshot_cached:
        if video_path is None:
            err = ("AutoShot needs a video but cfg.video is empty. "
                   "Either provide --video, or import existing cuts first "
                   "(e.g., --import-cuts PATH or via Tab UI).")
            progress("orchestrator:error", {"message": err})
            return _early_return(base_out, cuts_dir, sbs_dir, logs_dir, t_start,
                                 err, warnings)
        try:
            autoshot_python = str(_detect_autoshot_python(cfg.autoshot_python))
        except FileNotFoundError as e:
            err = f"autoshot python not found: {e}"
            progress("orchestrator:error", {"message": err})
            return _early_return(base_out, cuts_dir, sbs_dir, logs_dir, t_start,
                                 err, warnings)
        progress("orchestrator:autoshot_python", {"path": autoshot_python})
        autoshot_cmd = [
            autoshot_python,
            str(HERE / "autoshot_worker.py"),
            "--video", str(video_path),
            "--out", str(cuts_dir),
            "--threshold", str(cfg.threshold),
            "--min-duration", str(cfg.min_duration),
        ]
        if cfg.autoshot_weights:
            autoshot_cmd += ["--weights", cfg.autoshot_weights]

        progress("orchestrator:autoshot_launch", {"cmd": autoshot_cmd})
        rc_a = _stream_subprocess(
            autoshot_cmd,
            logs_dir / "autoshot_stdout.jsonl",
            progress,
            source_tag="autoshot",
        )
        if rc_a != 0:
            err = f"autoshot_worker failed (rc={rc_a})"
            progress("orchestrator:error", {"message": err})
            return _early_return(base_out, cuts_dir, sbs_dir, logs_dir, t_start,
                                 err, warnings)

        if not cuts_metadata.exists():
            err = f"cuts_metadata.json missing: {cuts_metadata}"
            return _early_return(base_out, cuts_dir, sbs_dir, logs_dir, t_start,
                                 err, warnings)

    with open(cuts_metadata, "r", encoding="utf-8") as f:
        meta = json.load(f)
    n_cuts = meta.get("n_segments", 0) or len(meta.get("segments", []))
    progress("orchestrator:autoshot_done", {"n_cuts": n_cuts, "cached": autoshot_cached})

    if n_cuts == 0:
        err = "AutoShot produced 0 segments"
        return _early_return(base_out, cuts_dir, sbs_dir, logs_dir, t_start,
                             err, warnings, cuts_metadata=str(cuts_metadata))

    # ── Subprocess C: Shot Classifier (선택) ──────────────────────────
    shot_classes_json: Optional[Path] = None
    if cfg.use_shotclass:
        shotclass_dir.mkdir(parents=True, exist_ok=True)

        # Cache check
        classifier_cached = False
        if "classifier" not in force:
            ok, why = _classifier_cache_valid(shotclass_dir)
            if ok:
                shot_classes_json = shotclass_dir / "shot_classes.json"
                progress("orchestrator:shotclass_skipped",
                         {"reason": "cache hit", "json": str(shot_classes_json)})
                classifier_cached = True
            else:
                progress("orchestrator:shotclass_cache_miss", {"reason": why})

        if not classifier_cached:
            try:
                shotclass_python = _detect_shotclass_python(cfg.shotclass_python)
                models_dir = (Path(cfg.shotclass_models_dir or DEFAULT_SHOTCLASS_MODELS)
                              / cfg.shotclass_backend)

                sc_cmd = [
                    str(shotclass_python),
                    str(HERE / "shotclass_worker.py"),
                    "--cuts-meta", str(cuts_metadata),
                    "--out-dir", str(shotclass_dir),
                    "--backend", cfg.shotclass_backend,
                    "--models-dir", str(models_dir),
                    "--depth-std-wide", str(cfg.depth_std_wide),
                    "--depth-std-closeup", str(cfg.depth_std_closeup),
                    "--max-disp-wide", str(cfg.max_disp_wide),
                    "--max-disp-normal", str(cfg.max_disp_normal),
                    "--max-disp-closeup", str(cfg.max_disp_closeup),
                ]
                progress("orchestrator:shotclass_launch", {"cmd": sc_cmd})
                rc_c = _stream_subprocess(
                    sc_cmd,
                    logs_dir / "shotclass_stdout.jsonl",
                    progress,
                    source_tag="shotclass",
                )
                candidate = shotclass_dir / "shot_classes.json"
                if rc_c == 0 and candidate.exists():
                    shot_classes_json = candidate
                    progress("orchestrator:shotclass_done", {
                        "rc": rc_c, "json": str(candidate),
                    })
                else:
                    msg = f"shotclass_worker failed (rc={rc_c}) or output missing"
                    progress("orchestrator:shotclass_failed", {"rc": rc_c})
                    if cfg.shotclass_required:
                        return _early_return(
                            base_out, cuts_dir, sbs_dir, logs_dir, t_start,
                            msg, warnings,
                            shotclass_dir=str(shotclass_dir),
                            cuts_metadata=str(cuts_metadata),
                            n_cuts=n_cuts,
                        )
                    warnings.append(msg + " — falling back to fixed max_disp")
            except FileNotFoundError as e:
                msg = f"shotclass python missing: {e}"
                progress("orchestrator:shotclass_skipped", {"message": msg})
                if cfg.shotclass_required:
                    return _early_return(
                        base_out, cuts_dir, sbs_dir, logs_dir, t_start,
                        msg, warnings,
                        shotclass_dir=str(shotclass_dir),
                        cuts_metadata=str(cuts_metadata),
                        n_cuts=n_cuts,
                    )
                warnings.append(msg + " — falling back to fixed max_disp")
    else:
        progress("orchestrator:shotclass_skipped", {"reason": "use_shotclass=False"})

    # ── Subprocess M: M2SVid wrapper (v0.16m) ──────────────────────────
    # 현재 Python (오케스트레이터 venv)으로 m2svid_worker.py를 실행.
    # worker 내부에서 m2svid_per_cut_runner.py를 m2svid_service .venv 로 spawn.
    if cfg.skip_m2svid:
        progress("orchestrator:m2svid_skipped", {"reason": "skip_m2svid=True"})
        cut_sbs_files = sorted(str(p) for p in sbs_dir.glob("shot*_sbs.mp4"))
        elapsed = round(time.time() - t_start, 2)
        # For stage-only runs (AutoShot/Classifier tabs), report success based on
        # cuts/classes work, not SBS count (M2SVid intentionally not run).
        progress("orchestrator:done", {
            "elapsed_sec": elapsed,
            "n_ok": n_cuts,      # AutoShot/Classifier completed; n_cuts is the success metric
            "n_fail": 0,
            "n_total": n_cuts,
            "n_sbs_existing": len(cut_sbs_files),
            "final_sbs": None,
            "warnings": warnings,
            "stage_skipped": "m2svid",
        })
        return PipelineResult(
            out_dir=str(base_out), cuts_dir=str(cuts_dir),
            shotclass_dir=str(shotclass_dir) if cfg.use_shotclass else None,
            shot_classes_json=str(shot_classes_json) if shot_classes_json else None,
            sbs_dir=str(sbs_dir), cuts_metadata=str(cuts_metadata),
            cut_sbs_files=cut_sbs_files, final_sbs=None,
            logs_dir=str(logs_dir), elapsed_sec=elapsed,
            n_cuts=n_cuts, n_ok=n_cuts, n_fail=0,
            error=None, warnings=warnings,
        )

    m2s_service = Path(cfg.m2svid_service).resolve() if cfg.m2svid_service \
        else DEFAULT_M2SVID_SERVICE
    m2s_python = (Path(cfg.m2svid_python).resolve() if cfg.m2svid_python
                  else venv_python(m2s_service))

    if not m2s_service.exists():
        err = f"m2svid_service dir not found: {m2s_service}"
        progress("orchestrator:error", {"message": err})
        return _early_return(base_out, cuts_dir, sbs_dir, logs_dir, t_start,
                             err, warnings,
                             shotclass_dir=str(shotclass_dir) if cfg.use_shotclass else None,
                             shot_classes_json=str(shot_classes_json) if shot_classes_json else None,
                             cuts_metadata=str(cuts_metadata),
                             n_cuts=n_cuts)
    if not m2s_python.exists():
        err = f"m2svid python not found: {m2s_python}"
        progress("orchestrator:error", {"message": err})
        return _early_return(base_out, cuts_dir, sbs_dir, logs_dir, t_start,
                             err, warnings,
                             shotclass_dir=str(shotclass_dir) if cfg.use_shotclass else None,
                             shot_classes_json=str(shot_classes_json) if shot_classes_json else None,
                             cuts_metadata=str(cuts_metadata),
                             n_cuts=n_cuts)

    m_cmd = [
        sys.executable,
        str(HERE / "m2svid_worker.py"),
        "--cuts-meta", str(cuts_metadata),
        "--out", str(sbs_dir),
        "--processing-dim", str(cfg.processing_dim),
        "--output-dim", str(cfg.output_dim),
        "--depth-backend", str(cfg.depth_backend),
        "--upscaler", str(cfg.upscaler),
        "--rtx-vsr-quality", str(cfg.rtx_vsr_quality),
        "--disparity-perc", str(cfg.disparity_perc),
        "--seed", str(cfg.seed),
        "--mask-antialias", str(cfg.mask_antialias),
        "--chunk-size", str(cfg.m2svid_chunk_size),
    ]
    if cfg.per_cut_timeout > 0:
        m_cmd += ["--timeout", str(cfg.per_cut_timeout)]
    if cfg.m2svid_service:
        m_cmd += ["--m2svid-service", cfg.m2svid_service]
    if cfg.m2svid_python:
        m_cmd += ["--m2svid-python", cfg.m2svid_python]
    if cfg.fail_fast:
        m_cmd += ["--fail-fast"]
    # Force-rerun: pass through if user requested m2svid stage rerun
    if "m2svid" in force:
        m_cmd += ["--force-rerun"]
    # Per-shot disparity: forward shot_classes.json if Classifier produced it
    if shot_classes_json:
        m_cmd += ["--shot-classes", str(shot_classes_json)]

    progress("orchestrator:m2svid_launch", {"cmd": m_cmd})
    rc_g = _stream_subprocess(
        m_cmd,
        logs_dir / "m2svid_stdout.jsonl",
        progress,
        source_tag="m2svid",
    )

    cut_sbs_files = sorted(str(p) for p in sbs_dir.glob("shot*_sbs.mp4"))
    n_ok = len(cut_sbs_files)
    n_fail = n_cuts - n_ok

    gs_error = None
    if rc_g != 0:
        gs_error = f"m2svid_worker rc={rc_g}"
        progress("orchestrator:m2svid_failed", {"rc": rc_g, "n_ok": n_ok})
        if cfg.fail_fast:
            return PipelineResult(
                out_dir=str(base_out), cuts_dir=str(cuts_dir),
                shotclass_dir=str(shotclass_dir) if cfg.use_shotclass else None,
                shot_classes_json=str(shot_classes_json) if shot_classes_json else None,
                sbs_dir=str(sbs_dir),
                cuts_metadata=str(cuts_metadata),
                cut_sbs_files=cut_sbs_files, final_sbs=None,
                logs_dir=str(logs_dir),
                elapsed_sec=round(time.time() - t_start, 2),
                n_cuts=n_cuts, n_ok=n_ok, n_fail=n_fail,
                error=gs_error, warnings=warnings,
            )

    # ── Concat (선택, cache-skip 지원) ────────────────────────────────
    # Final concat name also gets the suffix so Overdrive output is stored as
    # final_sbs_overdrive_12f720.mp4 alongside the standard final_sbs.mp4.
    final_sbs = None
    final_path = base_out / f"final_sbs{sbs_suffix}.mp4"
    if cfg.concat and cut_sbs_files:
        if "concat" not in force and final_path.exists() and final_path.stat().st_size > 4096:
            # Cache hit: final_sbs.mp4 already exists.
            progress("orchestrator:concat_skipped",
                     {"reason": "cache hit", "final_sbs": str(final_path),
                      "size_mb": round(final_path.stat().st_size / 1024**2, 2)})
            final_sbs = str(final_path)
        else:
            from concat_ffmpeg import concat_sbs as _concat_sbs  # local import
            progress("orchestrator:concat_start",
                     {"n": len(cut_sbs_files), "out": str(final_path)})
            ok, msg = _concat_sbs([Path(p) for p in cut_sbs_files], final_path)
            progress("orchestrator:concat_done", {"ok": ok, "msg": msg})
            if ok:
                final_sbs = str(final_path)
            else:
                warnings.append(f"concat failed: {msg}")

    elapsed = round(time.time() - t_start, 2)
    progress("orchestrator:done", {
        "elapsed_sec": elapsed, "n_ok": n_ok, "n_fail": n_fail,
        "n_total": n_cuts,
        "final_sbs": final_sbs, "warnings": warnings,
    })

    return PipelineResult(
        out_dir=str(base_out), cuts_dir=str(cuts_dir),
        shotclass_dir=str(shotclass_dir) if cfg.use_shotclass else None,
        shot_classes_json=str(shot_classes_json) if shot_classes_json else None,
        sbs_dir=str(sbs_dir),
        cuts_metadata=str(cuts_metadata),
        cut_sbs_files=cut_sbs_files, final_sbs=final_sbs,
        logs_dir=str(logs_dir), elapsed_sec=elapsed,
        n_cuts=n_cuts, n_ok=n_ok, n_fail=n_fail,
        error=gs_error, warnings=warnings,
    )


_VERBOSE_CLI = False  # toggled by --verbose flag in main()


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _term(msg: str, file=sys.stderr):
    """Always-visible terminal output (stderr) — survives stdout redirect."""
    print(msg, file=file, flush=True)


def _cli_progress(event: str, payload: dict):
    """Format pipeline events into human-readable terminal lines (stderr).

    Behavior:
      - errors always shown with [ERR] tag
      - key events (start, cut_done, step done, total, done) shown as one-liners
      - stage_log filtered to important markers (preprocess/depth/warp/inpaint/compose/DONE/ERR)
      - --verbose flag (env STEREO_VERBOSE=1) shows everything
    """
    line = payload.get("line") if event.endswith(":raw") or event.endswith(":stage_log") else None

    # Errors first — always loud
    err_keys = ("error", "fail", "err", "timeout", "spawn_error")
    if any(k in event.lower() for k in err_keys) or payload.get("ok") is False:
        msg = payload.get("message") or payload.get("error") or json.dumps(payload, ensure_ascii=False)
        _term(f"[{_ts()}] [ERR] {event}: {msg}")
        return

    # stage_log: filter noisy lines unless verbose
    if event.endswith(":stage_log") and line:
        important = any(t in line for t in [
            "[1/5]", "[2/5]", "[3/5]", "[4/5]", "[5/5]",
            "DONE:", "[ERR]", "[FATAL]", "Traceback", "Error:",
            "depth done", "warp done", "inpaint done", "chunk ",
            "RTX VSR:", "upscale target",
            "cuts:", "cuts found", "n_cuts",
            "STEREO_TAESDV", "[STEREO_",
        ])
        if important or _VERBOSE_CLI:
            sid = payload.get("shot_id")
            tag = f"shot{sid}" if sid is not None else (payload.get("label") or "")
            _term(f"[{_ts()}] [{tag}] {line}")
        return

    # Top-level events
    if event.endswith(":start"):
        n = payload.get("n_cuts", "?")
        _term(f"[{_ts()}] === pipeline start (cuts={n}) ===")
        return
    if event.endswith(":cut_start"):
        sid = payload.get("shot_id")
        info = []
        for k in ("shot_class", "disparity_perc", "depth_backend"):
            if k in payload:
                info.append(f"{k}={payload[k]}")
        _term(f"[{_ts()}] [shot{sid}] start ({', '.join(info)})")
        return
    if event.endswith(":cut_done"):
        sid = payload.get("shot_id")
        sec = payload.get("total_sec", 0)
        sz = payload.get("size_mb", 0)
        _term(f"[{_ts()}] [shot{sid}] done {sec:.1f}s ({sz:.2f} MB)")
        return
    if event.endswith(":done"):
        sec = payload.get("sec", 0)
        ok = payload.get("n_ok", 0)
        fail = payload.get("n_fail", 0)
        n_total = payload.get("n_total", 0)
        if fail:
            _term(f"[{_ts()}] === DONE: {ok}/{n_total} ok, {fail} FAIL ({sec:.1f}s) ===")
        else:
            _term(f"[{_ts()}] === DONE: {ok}/{n_total} ok ({sec:.1f}s) ===")
        return
    if event.startswith("orchestrator:"):
        sub = event.split(":", 1)[1]
        if sub in ("autoshot_launch", "shotclass_launch", "m2svid_launch", "concat_start"):
            _term(f"[{_ts()}] >> {sub.replace('_', ' ')}")
        elif sub == "concat_done":
            _term(f"[{_ts()}] >> concat done -> {payload.get('final_sbs', '?')}")
        elif sub.endswith("_failed") or sub == "error":
            _term(f"[{_ts()}] [ERR] orchestrator: {payload.get('message') or payload.get('rc') or payload}")
        elif _VERBOSE_CLI:
            _term(f"[{_ts()}] [orch] {sub}")
        return

    # Fallback: verbose dump
    if _VERBOSE_CLI:
        _term(f"[{_ts()}] [{event}] {json.dumps(payload, ensure_ascii=False)[:200]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="v0.16 Stereo pipeline: AutoShot → ShotClass → GenStereo (CLI wrapper)"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)

    # AutoShot
    g_a = parser.add_argument_group("AutoShot")
    g_a.add_argument("--threshold", type=float, default=0.296)
    g_a.add_argument("--min-duration", type=float, default=0.0)
    g_a.add_argument("--autoshot-weights", type=str, default=None)
    g_a.add_argument("--autoshot-python", type=str, default=None)

    # Shot Classifier
    g_sc = parser.add_argument_group("Shot Classifier")
    g_sc.add_argument("--use-shotclass", dest="use_shotclass",
                      action="store_true", default=True)
    g_sc.add_argument("--no-shotclass", dest="use_shotclass",
                      action="store_false")
    g_sc.add_argument("--shotclass-required", action="store_true", default=False,
                      help="shotclass 실패 시 파이프라인 중단 (기본: 경고만)")
    g_sc.add_argument("--shotclass-backend", choices=["clip", "depth"], default="clip")
    g_sc.add_argument("--shotclass-python", type=str, default=None)
    g_sc.add_argument("--shotclass-models-dir", type=str, default=None)
    g_sc.add_argument("--depth-std-wide", type=float, default=0.25)
    g_sc.add_argument("--depth-std-closeup", type=float, default=0.12)
    g_sc.add_argument("--max-disp-wide", type=float, default=30.0)
    g_sc.add_argument("--max-disp-normal", type=float, default=20.0)
    g_sc.add_argument("--max-disp-closeup", type=float, default=12.0)

    # M2SVid (v0.16m)
    g_m = parser.add_argument_group("M2SVid (v0.16m)")
    g_m.add_argument("--processing-dim", type=int, default=512,
                     help="처리 해상도 max (64-divisible, default 512)")
    g_m.add_argument("--output-dim", type=int, default=0,
                     help="출력 해상도 max (0=processing 동일, default 0)")
    g_m.add_argument("--depth-backend", default="VDA-S",
                     choices=["VDA-S", "VDA-L", "FlashDepth-L", "FlashDepth-S",
                              "FlashDepth", "DepthCrafter"],
                     help="depth backend (default VDA-S)")
    g_m.add_argument("--upscaler", default="lanczos",
                     choices=["lanczos", "rtx_vsr"],
                     help="upscale 방식 (default lanczos)")
    g_m.add_argument("--rtx-vsr-quality", type=int, default=4,
                     help="RTX VSR quality 0-19 (default 4=ULTRA; 0..4 standard, 8..11 denoise, 12..15 deblur, 16..19 highbitrate)")
    g_m.add_argument("--disparity-perc", type=float, default=0.02,
                     help="warp disparity 비율 (default 0.02)")
    g_m.add_argument("--seed", type=int, default=42)
    g_m.add_argument("--mask-antialias", type=int, default=0)
    g_m.add_argument("--per-cut-timeout", type=int, default=0,
                     help="컷당 timeout (초, 0=무제한)")
    g_m.add_argument("--m2svid-service", type=str, default=None,
                     help=f"m2svid_service root (default: {DEFAULT_M2SVID_SERVICE})")
    g_m.add_argument("--m2svid-python", type=str, default=None,
                     help="m2svid_service .venv python (default: {service}/.venv/Scripts/python.exe)")

    # Resolution Overdrive (v0.17.3+)
    g_od = parser.add_argument_group("Resolution Overdrive (M2SVid 처리 해상도 ↑ ↔ chunk T ↓)")
    g_od.add_argument("--m2svid-chunk-size", type=int, default=25,
                      help="M2SVid 시간 윈도우 (default 25 = 학습 윈도우; "
                           "Overdrive 12, Extreme 8). 작을수록 VRAM 여유 + 더 큰 dim 가능, "
                           "단 temporal coherence 저하 / chunk seam 빈도 증가")
    g_od.add_argument("--m2svid-output-suffix", type=str, default="",
                      help="sbs/ + final_sbs.mp4 이름에 붙는 접미사 "
                           "(예: '_overdrive_12f720' → sbs_overdrive_12f720/, "
                           "final_sbs_overdrive_12f720.mp4)")
    g_od.add_argument("--overdrive", action="store_true", default=False,
                      help="Resolution Overdrive preset shortcut: "
                           "--processing-dim 720 --m2svid-chunk-size 12 "
                           "--m2svid-output-suffix _overdrive_12f720")
    g_od.add_argument("--overdrive-mild", action="store_true", default=False,
                      help="Mild preset: 16f @ 576")
    g_od.add_argument("--overdrive-extreme", action="store_true", default=False,
                      help="Extreme preset: 8f @ 832 (위험)")

    # Orchestration
    g_o = parser.add_argument_group("Orchestration")
    g_o.add_argument("--concat", dest="concat", action="store_true", default=True)
    g_o.add_argument("--no-concat", dest="concat", action="store_false")
    g_o.add_argument("--fail-fast", dest="fail_fast", action="store_true", default=True)
    g_o.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    g_o.add_argument("--normalize-fps", choices=["off", "ceil", "round"],
                     default="ceil",
                     help="비정수 fps (29.97/23.976) 입력을 사전 transcode "
                          "(기본 ceil — 정수 fps 강제, off로 원본 유지)")
    g_o.add_argument("--verbose", "-v", action="store_true", default=False,
                     help="terminal 에 모든 stage_log 이벤트 출력 (기본: 핵심 이벤트만)")
    g_o.add_argument("--out-dir", type=str, default=None,
                     help="정확한 base_out 경로 override (default: outputs/{stem}_{video_hash[:12]}/)")
    g_o.add_argument("--force-rerun", type=str, default="",
                     help="강제 재실행 단계 (콤마 구분): autoshot,classifier,m2svid,concat / all")
    # Per-stage external imports (v0.17.1+)
    g_i = parser.add_argument_group("Per-stage import (외부 결과물 가져오기)")
    g_i.add_argument("--import-cuts", type=str, default=None,
                     help="외부 cuts 폴더 (cuts_metadata.json + shot*.mp4) 경로 → AutoShot skip")
    g_i.add_argument("--import-shot-classes", type=str, default=None,
                     help="외부 shot_classes.json 파일 경로 → Classifier skip")
    g_i.add_argument("--import-sbs-dir", type=str, default=None,
                     help="외부 SBS 폴더 (shot*_sbs.mp4) 경로 → M2SVid 컷 skip")
    g_i.add_argument("--import-final-sbs", type=str, default=None,
                     help="외부 final_sbs.mp4 파일 경로 → Concat skip")

    args = parser.parse_args()

    # toggle global verbosity for _cli_progress
    global _VERBOSE_CLI
    _VERBOSE_CLI = bool(args.verbose) or os.environ.get("STEREO_VERBOSE") == "1"

    # Resolution Overdrive preset shortcuts. Apply BEFORE cfg construction.
    # Preset overrides explicit values; warn if user passed both.
    _preset = None
    if args.overdrive_extreme:
        _preset = ("extreme", 8, 832)
    elif args.overdrive:
        _preset = ("overdrive", 12, 720)
    elif args.overdrive_mild:
        _preset = ("mild", 16, 576)
    if _preset is not None:
        name, chunk, dim = _preset
        if (args.processing_dim != 512 or args.m2svid_chunk_size != 25
                or args.m2svid_output_suffix):
            _term(f"[{_ts()}] [WARN] --{name}/preset overrides --processing-dim/"
                  f"--m2svid-chunk-size/--m2svid-output-suffix")
        args.processing_dim = dim
        args.m2svid_chunk_size = chunk
        args.m2svid_output_suffix = f"_{name}_{chunk}f{dim}"
        _term(f"[{_ts()}] === Resolution Overdrive preset '{name}' "
              f"({chunk}f @ {dim}) → suffix={args.m2svid_output_suffix} ===")

    # remove --verbose + preset shortcut flags from cfg construction
    # (not part of PipelineConfig dataclass fields).
    args_dict = vars(args)
    for k in ("verbose", "overdrive", "overdrive_mild", "overdrive_extreme"):
        args_dict.pop(k, None)
    cfg = PipelineConfig(**args_dict)

    _term(f"[{_ts()}] === stereo pipeline v0.17 (verbose={_VERBOSE_CLI}) ===")
    _term(f"[{_ts()}]   video       : {cfg.video}")
    _term(f"[{_ts()}]   out         : {cfg.out}")
    _term(f"[{_ts()}]   depth       : {cfg.depth_backend}")
    _term(f"[{_ts()}]   processing  : {cfg.processing_dim}, output: {cfg.output_dim}")
    _term(f"[{_ts()}]   upscaler    : {cfg.upscaler}"
          + (f" (quality={cfg.rtx_vsr_quality})" if cfg.upscaler == "rtx_vsr" else ""))

    try:
        res = run(cfg, progress=_cli_progress)
    except Exception as e:
        import traceback
        _term("")
        _term("=" * 60)
        _term(f"[{_ts()}] [FATAL] {type(e).__name__}: {e}")
        _term("=" * 60)
        if _VERBOSE_CLI:
            traceback.print_exc(file=sys.stderr)
        else:
            _term("(stack trace suppressed; rerun with --verbose to see)")
        return 2

    _term("")
    _term("=" * 60)
    _term(f"  elapsed   : {res.elapsed_sec:.1f}s")
    _term(f"  cuts      : {res.n_ok}/{res.n_cuts} ok, {res.n_fail} fail")
    _term(f"  final SBS : {res.final_sbs or '(none)'}")
    if res.error:
        _term(f"  ERROR     : {res.error}")
    if res.warnings:
        _term(f"  warnings  : {len(res.warnings)}")
        for w in res.warnings:
            _term(f"    - {w}")
    _term("=" * 60)

    # full result JSON to stdout (for machine consumers / pipe-friendly)
    print(json.dumps(asdict(res), ensure_ascii=False, indent=2))
    return 0 if res.error is None else 1


if __name__ == "__main__":
    sys.exit(main())
