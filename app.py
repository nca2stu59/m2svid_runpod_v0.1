"""
app.py — Stereo Pipeline v0.16m Gradio UI (포트 7864)

v0.16b 의 인터페이스 패턴을 유지하되, Step2 엔진을 GenStereo SVD inpaint 에서
M2SVid full-attention 으로 교체한 분기.

흐름:
  AutoShot → SigLIP-2 classifier → m2svid_worker.py
    └── per-cut runner (.venv): preprocess → depth → warp → inpaint → SBS → upscale

Visual theme: GenStereo (gr.themes.Soft + 이모지 + Tab 레이아웃)
포트:
  7860 = m2svid_service standalone
  7861 = StereoCrafter standalone
  7862 = stereo_pipeline_v0.16b
  7864 = stereo_pipeline_v0.16m (이 파일)

실행:
    python app.py
    → http://127.0.0.1:7864
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import gradio as gr

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_pipeline import (  # noqa: E402
    PipelineConfig,
    run as run_pipeline,
    DEFAULT_M2SVID_SERVICE,
    DEFAULT_M2SVID_PYTHON,
    venv_python,
)
from local_engines import ui_kit as uk  # noqa: E402

# UI metadata (used by header bar)
VERSION = "m2svid_runpod_v0.1"
PORT = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7864")))
DEFAULT_OUT_ROOT = str(
    Path(os.environ.get("M2SVID_OUTPUT_ROOT") or (
        Path("/workspace/outputs/m2svid_runpod_v0.1") if os.name != "nt" else HERE / "outputs"
    ))
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _term_log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def _gradio_auth():
    raw = os.environ.get("GRADIO_AUTH", "").strip()
    if not raw:
        return None
    if ":" not in raw:
        print("[warn] GRADIO_AUTH ignored; expected user:password", file=sys.stderr, flush=True)
        return None
    user, password = raw.split(":", 1)
    return (user, password)


def list_outputs(out_root: str | Path) -> list[tuple[str, str]]:
    root = Path(out_root)
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        # show all final_sbs*.mp4 (default + Overdrive/Mild/Extreme/Custom variants)
        for final in sorted(d.glob("final_sbs*.mp4")):
            if final.is_file():
                size_mb = final.stat().st_size / (1024 * 1024)
                label = f"{d.name}/{final.name} ({size_mb:.1f} MB)"
                out.append((label, str(final)))
    return out


# ─── Per-stage output scanners (Tab dropdowns) ────────────────────────── #

def _scan_run_dirs(out_root: str | Path) -> list[Path]:
    root = Path(out_root)
    if not root.exists():
        return []
    return [d for d in sorted(root.iterdir(), reverse=True) if d.is_dir()]


def list_cuts_dirs(out_root: str | Path) -> list[tuple[str, str]]:
    """List existing cuts dirs (with valid cuts_metadata.json)."""
    out = []
    for d in _scan_run_dirs(out_root):
        cuts = d / "cuts"
        meta = cuts / "cuts_metadata.json"
        if meta.exists():
            try:
                with open(meta, "r", encoding="utf-8") as f:
                    m = json.load(f)
                n = m.get("n_segments") or len(m.get("segments", []))
            except Exception:
                n = "?"
            out.append((f"{d.name}/cuts/  ({n} cuts)", str(cuts)))
    return out


def list_shot_classes(out_root: str | Path) -> list[tuple[str, str]]:
    """List existing shot_classes.json files."""
    out = []
    for d in _scan_run_dirs(out_root):
        j = d / "shot_classes" / "shot_classes.json"
        if j.exists():
            try:
                with open(j, "r", encoding="utf-8") as f:
                    sc = json.load(f)
                shots = sc.get("shots", {})
                n = len(shots) if isinstance(shots, dict) else len(shots) if isinstance(shots, list) else "?"
            except Exception:
                n = "?"
            out.append((f"{d.name}/shot_classes.json  ({n} shots)", str(j)))
    return out


def list_sbs_dirs(out_root: str | Path) -> list[tuple[str, str]]:
    """List existing SBS dirs (with shot*_sbs.mp4).

    v0.17.3+: also enumerates suffixed dirs (sbs_overdrive_12f720/, ...)
    so Resolution Overdrive results can be picked up by the Concat tab.
    """
    out = []
    for d in _scan_run_dirs(out_root):
        # find every "sbs*" subdirectory (default 'sbs' plus suffixed variants)
        for sbs in sorted(d.glob("sbs*")):
            if not sbs.is_dir():
                continue
            files = list(sbs.glob("shot*_sbs.mp4"))
            if files:
                total_mb = sum(f.stat().st_size for f in files) / 1024**2
                out.append((f"{d.name}/{sbs.name}/  ({len(files)} files, {total_mb:.1f} MB)", str(sbs)))
    return out


def list_final_sbs(out_root: str | Path) -> list[tuple[str, str]]:
    """List existing final_sbs*.mp4 files (default + suffixed variants)."""
    out = []
    for d in _scan_run_dirs(out_root):
        for final in sorted(d.glob("final_sbs*.mp4")):
            if final.is_file():
                size_mb = final.stat().st_size / 1024**2
                out.append((f"{d.name}/{final.name}  ({size_mb:.1f} MB)", str(final)))
    return out


def _list_available_depth_backends() -> list[str]:
    """Return depth backends that have verified weights/install on disk.

    Inspects:
      - VDA-S / VDA-L: always listed (lightweight, weights ship with the script
        directory). Their venv check happens at runtime.
      - FlashDepth-{L,S,(no-suffix)}: only listed when the corresponding config
        directory contains at least one .pth checkpoint (only FlashDepth-L
        ships pretrained weights upstream — others must be added manually).
      - DepthCrafter: only listed when its .venv exists.

    Called once at build_ui() time. Adding new weights → restart app to pick up.
    """
    backends: list[str] = ["VDA-S", "VDA-L"]

    fd_root = DEFAULT_M2SVID_SERVICE / "third_party" / "FlashDepth" / "configs"
    for variant_dir, label in [
        ("flashdepth-l", "FlashDepth-L"),
        ("flashdepth-s", "FlashDepth-S"),
        ("flashdepth", "FlashDepth"),
    ]:
        cfg_dir = fd_root / variant_dir
        if cfg_dir.exists() and any(cfg_dir.glob("*.pth")):
            backends.append(label)

    dc_venv = venv_python(DEFAULT_M2SVID_SERVICE, ".venv-depthcrafter")
    if dc_venv.exists():
        backends.append("DepthCrafter")

    return backends


def check_environment() -> str:
    lines = ["### v0.16m 환경 점검\n"]

    def _check(label: str, path: str, kind: str = "file"):
        p = Path(path)
        ok = p.exists() and (p.is_file() if kind == "file" else p.is_dir())
        status = "✅" if ok else "❌"
        lines.append(f"{status} {label}: `{path}`")

    _check("m2svid_service root", str(DEFAULT_M2SVID_SERVICE), "dir")
    _check("m2svid .venv python", str(DEFAULT_M2SVID_PYTHON), "file")
    _check("m2svid .venv-vda python",
           str(venv_python(DEFAULT_M2SVID_SERVICE, ".venv-vda")), "file")
    _check("m2svid weights (5 GB)",
           str(DEFAULT_M2SVID_SERVICE / "ckpts" / "m2svid_weights.pt"), "file")
    _check("open_clip weights (4 GB)",
           str(DEFAULT_M2SVID_SERVICE / "ckpts" / "open_clip_pytorch_model.bin"), "file")
    _check("VDA repo",
           str(DEFAULT_M2SVID_SERVICE / "third_party" / "Video-Depth-Anything"), "dir")
    _check("vendored m2svid_per_cut_runner.py",
           str(HERE / "local_engines" / "m2svid" / "m2svid_per_cut_runner.py"), "file")

    # nvidia-vfx (RTX VSR) — optional
    try:
        import importlib
        spec = importlib.util.find_spec("nvvfx")
        if spec is not None:
            lines.append("✅ nvidia-vfx (RTX VSR) installed in current process")
        else:
            lines.append("⚠️ nvidia-vfx not installed (RTX VSR option will fall back to lanczos)")
    except Exception:
        lines.append("⚠️ nvidia-vfx import probe failed")

    # Show which depth backends are usable right now
    backends = _list_available_depth_backends()
    lines.append(f"\n**Depth backends 사용 가능**: {', '.join(backends)}")
    fd_root = DEFAULT_M2SVID_SERVICE / "third_party" / "FlashDepth" / "configs"
    fd_missing = []
    for variant_dir, label in [("flashdepth-l", "FlashDepth-L"),
                                ("flashdepth-s", "FlashDepth-S"),
                                ("flashdepth", "FlashDepth")]:
        cfg_dir = fd_root / variant_dir
        if not (cfg_dir.exists() and any(cfg_dir.glob("*.pth"))):
            fd_missing.append(f"{label} ({cfg_dir.name}/)")
    if fd_missing:
        lines.append(f"⚠️ FlashDepth 가중치 부재 (dropdown 에서 자동 제외): {', '.join(fd_missing)}")

    return "\n".join(lines)


# ─── Per-stage run handlers (Tab Run buttons) ────────────────────────── #

def run_autoshot_stage(video, out_root, threshold, min_duration, force_rerun_self,
                       progress=gr.Progress(track_tqdm=False)):
    """AutoShot only. Skips classifier/m2svid/concat.

    Propagation: on completion, fills cuts_path textbox in
    Classifier / M2SVid / Overdrive tabs with res.cuts_dir.
    """
    if not video:
        raise gr.Error("입력 영상을 업로드 해주세요.")
    cfg = PipelineConfig(
        video=video, out=out_root,
        threshold=float(threshold), min_duration=float(min_duration),
        use_shotclass=False, shotclass_required=False,
        skip_m2svid=True, concat=False,
        force_rerun="autoshot" if force_rerun_self else "",
    )
    # Wrap _run_cfg_streaming to add 3 cuts_path propagation outputs.
    # During streaming (res=None) keep targets unchanged via gr.update();
    # on final yield, fill them with res.cuts_dir.
    for log, sbs, gallery, res in _run_cfg_streaming(cfg, progress):
        if res is None:
            yield log, sbs, gallery, gr.update(), gr.update(), gr.update()
        else:
            cuts_dir = res.cuts_dir or ""
            log = log + f"\n→ Classifier / M2SVid / Overdrive 탭의 cuts 경로에 자동 입력됨: {cuts_dir}"
            yield log, sbs, gallery, cuts_dir, cuts_dir, cuts_dir


def _derive_base_out(stage_dir_path: str | None,
                     parent_levels: int = 1) -> str | None:
    """Given a path like '.../outputs/STEM_HASH/cuts/' return '.../outputs/STEM_HASH/'.

    parent_levels: 1 for cuts/sbs dir, 2 for shot_classes/shot_classes.json
    (which is one level deeper). For a file path, count from file's parent.
    """
    if not stage_dir_path:
        return None
    p = Path(stage_dir_path).resolve()
    if p.is_file():
        p = p.parent
    for _ in range(parent_levels):
        p = p.parent
    return str(p) if p.exists() else None


def run_classifier_stage(video, out_root, cuts_import, force_rerun_self,
                         progress=gr.Progress(track_tqdm=False)):
    """Classifier only. video는 cuts_import이 있으면 선택사항 (out_dir 자동 도출).

    Propagation: on completion, fills shot_classes_path textbox in
    M2SVid / Overdrive tabs with res.shot_classes_json.
    """
    cuts_path = (cuts_import.strip() or None) if isinstance(cuts_import, str) else None
    base_out = _derive_base_out(cuts_path, parent_levels=1) if cuts_path else None
    if not video and not base_out:
        raise gr.Error("입력 영상 또는 cuts 폴더 경로 중 하나는 필요합니다.")
    cfg = PipelineConfig(
        video=video or "", out=out_root,
        out_dir=base_out,  # 자동 도출 시 hash 계산 skip
        use_shotclass=True, shotclass_required=False,
        shotclass_backend="siglip2",
        skip_m2svid=True, concat=False,
        import_cuts=cuts_path,
        force_rerun="classifier" if force_rerun_self else "",
    )
    for log, sbs, gallery, res in _run_cfg_streaming(cfg, progress):
        if res is None:
            yield log, sbs, gallery, gr.update(), gr.update()
        else:
            sc_path = res.shot_classes_json or ""
            if sc_path:
                log = log + f"\n→ M2SVid / Overdrive 탭의 shot_classes 경로에 자동 입력됨: {sc_path}"
            yield log, sbs, gallery, sc_path, sc_path


def run_m2svid_stage(video, out_root,
                     cuts_import, shotclass_import,
                     processing_dim, output_dim, depth_backend,
                     upscaler, rtx_vsr_quality, disparity_perc,
                     seed, mask_antialias, force_rerun_self,
                     progress=gr.Progress(track_tqdm=False)):
    """M2SVid only. cuts_import 가 있으면 video 선택사항."""
    cuts_path = (cuts_import.strip() or None) if isinstance(cuts_import, str) else None
    sc_path = (shotclass_import.strip() or None) if isinstance(shotclass_import, str) else None
    base_out = _derive_base_out(cuts_path, parent_levels=1) if cuts_path else None
    if not base_out and sc_path:
        base_out = _derive_base_out(sc_path, parent_levels=2)
    if not video and not base_out:
        raise gr.Error("입력 영상 또는 cuts 폴더 경로 중 하나는 필요합니다.")
    cfg = PipelineConfig(
        video=video or "", out=out_root,
        out_dir=base_out,
        use_shotclass=bool(sc_path),
        processing_dim=int(processing_dim), output_dim=int(output_dim),
        depth_backend=str(depth_backend), upscaler=str(upscaler),
        rtx_vsr_quality=int(rtx_vsr_quality),
        disparity_perc=float(disparity_perc),
        seed=int(seed), mask_antialias=int(mask_antialias),
        skip_m2svid=False, concat=False,
        import_cuts=cuts_path,
        import_shot_classes=sc_path,
        force_rerun="m2svid" if force_rerun_self else "",
    )
    # Propagate sbs_dir → Concat tab's c_sbs_path on completion.
    for log, sbs, gallery, res in _run_cfg_streaming(cfg, progress):
        if res is None:
            yield log, sbs, gallery, gr.update()
        else:
            sbs_dir = res.sbs_dir or ""
            if sbs_dir:
                log = log + f"\n→ Concat 탭의 SBS 폴더 경로에 자동 입력됨: {sbs_dir}"
            yield log, sbs, gallery, sbs_dir


# ─── Resolution Overdrive presets ────────────────────────────────────── #
# preset_name → (chunk_size, processing_dim)
OVERDRIVE_PRESETS = {
    "Mild (16f @ 576)":      ("mild",      16, 576),
    "Overdrive (12f @ 720)": ("overdrive", 12, 720),
    "Extreme (8f @ 832)":    ("extreme",    8, 832),
    "Custom":                ("custom",    12, 720),  # default values for sliders
}


def _overdrive_suffix(preset_label: str, chunk: int, dim: int) -> str:
    """Compute suffix string from preset selection / custom values.

    Mapping mirrors run_pipeline._preset logic:
      Mild      → _mild_16f576
      Overdrive → _overdrive_12f720
      Extreme   → _extreme_8f832
      Custom    → _custom_{chunk}f{dim}
    """
    name, _, _ = OVERDRIVE_PRESETS.get(preset_label, ("custom", chunk, dim))
    return f"_{name}_{int(chunk)}f{int(dim)}"


def run_overdrive_stage(video, out_root,
                        cuts_import, shotclass_import,
                        preset_label, chunk_size, processing_dim,
                        depth_backend, upscaler, rtx_vsr_quality,
                        disparity_perc, seed, mask_antialias,
                        output_dim, force_rerun_self,
                        progress=gr.Progress(track_tqdm=False)):
    """Resolution Overdrive: M2SVid 를 더 큰 해상도 + 짧은 chunk T 로 재실행.

    출력은 `sbs{suffix}/` + `final_sbs{suffix}.mp4` 별도 폴더로 저장되어
    Standard 결과와 공존. Concat 은 Concat 탭에서 별도 실행.
    """
    cuts_path = (cuts_import.strip() or None) if isinstance(cuts_import, str) else None
    sc_path = (shotclass_import.strip() or None) if isinstance(shotclass_import, str) else None
    base_out = _derive_base_out(cuts_path, parent_levels=1) if cuts_path else None
    if not base_out and sc_path:
        base_out = _derive_base_out(sc_path, parent_levels=2)
    if not video and not base_out:
        raise gr.Error("입력 영상 또는 cuts 폴더 경로 중 하나는 필요합니다.")

    # Resolve preset → effective (chunk, dim, suffix)
    name, def_chunk, def_dim = OVERDRIVE_PRESETS.get(preset_label, ("custom", 12, 720))
    if name == "custom":
        eff_chunk = max(1, int(chunk_size))
        # round dim down to 64-divisible
        eff_dim = max(64, (int(processing_dim) // 64) * 64)
    else:
        eff_chunk, eff_dim = def_chunk, def_dim
    suffix = f"_{name}_{eff_chunk}f{eff_dim}"

    cfg = PipelineConfig(
        video=video or "", out=out_root,
        out_dir=base_out,
        use_shotclass=bool(sc_path),
        processing_dim=eff_dim, output_dim=int(output_dim),
        depth_backend=str(depth_backend), upscaler=str(upscaler),
        rtx_vsr_quality=int(rtx_vsr_quality),
        disparity_perc=float(disparity_perc),
        seed=int(seed), mask_antialias=int(mask_antialias),
        m2svid_chunk_size=eff_chunk,
        m2svid_output_suffix=suffix,
        skip_m2svid=False, concat=False,
        import_cuts=cuts_path,
        import_shot_classes=sc_path,
        force_rerun="m2svid" if force_rerun_self else "",
    )
    # Propagate Overdrive sbs_dir (e.g. sbs_overdrive_12f720/) → Concat tab.
    for log, sbs, gallery, res in _run_cfg_streaming(cfg, progress):
        if res is None:
            yield log, sbs, gallery, gr.update()
        else:
            sbs_dir = res.sbs_dir or ""
            if sbs_dir:
                log = log + f"\n→ Concat 탭의 SBS 폴더 경로에 자동 입력됨: {sbs_dir}"
            yield log, sbs, gallery, sbs_dir


def run_concat_stage(video, out_root, sbs_dir_import,
                     progress=gr.Progress(track_tqdm=False)):
    """Concat only. sbs_dir_import 가 있으면 video 선택사항.

    v0.17.3+: sbs_path 디렉토리명이 'sbs_overdrive_12f720' 처럼 suffix 가 붙은 경우,
    suffix 를 추출하여 cfg.m2svid_output_suffix 로 설정 → final_sbs 도 동일 suffix 적용
    (예: final_sbs_overdrive_12f720.mp4). Standard sbs/ 면 빈 문자열.
    """
    sbs_path = (sbs_dir_import.strip() or None) if isinstance(sbs_dir_import, str) else None
    base_out = _derive_base_out(sbs_path, parent_levels=1) if sbs_path else None
    if not video and not base_out:
        raise gr.Error("입력 영상 또는 SBS 폴더 경로 중 하나는 필요합니다.")

    # Extract suffix from SBS dir name: 'sbs' → '', 'sbs_overdrive_12f720' → '_overdrive_12f720'
    sbs_suffix = ""
    if sbs_path:
        dir_name = Path(sbs_path).name
        if dir_name.startswith("sbs") and dir_name != "sbs":
            sbs_suffix = dir_name[3:]  # strip "sbs" prefix → "_overdrive_12f720"

    cfg = PipelineConfig(
        video=video or "", out=out_root,
        out_dir=base_out,
        use_shotclass=False,
        skip_m2svid=False, concat=True,
        m2svid_output_suffix=sbs_suffix,
        import_sbs_dir=sbs_path,
        force_rerun="concat",  # always re-do concat
    )
    # Concat is the terminal stage — no propagation. Drop the 4th tuple element
    # so .click() outputs only need the 3 standard components (log, video, gallery).
    for log, sbs, gallery, _res in _run_cfg_streaming(cfg, progress):
        yield log, sbs, gallery


def _run_cfg_streaming(cfg, progress):
    """Generator that runs run_pipeline(cfg) in a worker thread and yields
    intermediate (log_text, sbs_path, cuts, res_or_None) 4-tuples for live UI
    updates. Reusable across all per-stage tab Run buttons.

    v0.17.4+: 4th tuple element is the PipelineResult on the FINAL yield only,
    None during streaming. Handlers use this to extract paths for stage→stage
    propagation (cuts_dir → cl_cuts_path / m_cuts_path / od_cuts_path, etc.).
    """
    import queue as _queue
    import threading as _threading

    log_lines: list[str] = []
    event_q: _queue.Queue = _queue.Queue()
    holder: dict = {"res": None, "exc": None}

    def _cb(event: str, payload: dict):
        try:
            _format_progress(event, payload, log_lines)
            try:
                last = log_lines[-1] if log_lines else ""
                progress(0.5, desc=last[:120])
            except Exception:
                pass
        finally:
            event_q.put_nowait("update")

    def _runner():
        try:
            holder["res"] = run_pipeline(cfg, progress=_cb)
        except BaseException as e:
            holder["exc"] = e
        finally:
            event_q.put_nowait("done")

    t = _threading.Thread(target=_runner, daemon=True, name="stage-pipeline")
    t.start()

    yield ("\n".join(log_lines) or "(시작 중...)", None, [], None)
    while True:
        try:
            msg = event_q.get(timeout=60)
        except _queue.Empty:
            yield ("\n".join(log_lines) + "\n... (60s 무응답)", None, [], None)
            continue
        if msg == "done":
            break
        # de-dup
        while True:
            try:
                event_q.get_nowait()
            except _queue.Empty:
                break
        yield ("\n".join(log_lines), None, [], None)

    if holder["exc"] is not None:
        log_lines.append("")
        log_lines.append(f"❌ 예외: {type(holder['exc']).__name__}: {holder['exc']}")
        yield ("\n".join(log_lines), None, [], None)
        raise holder["exc"]

    res = holder["res"]
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append(json.dumps({
        "elapsed_sec": res.elapsed_sec,
        "n_cuts": res.n_cuts, "n_ok": res.n_ok, "n_fail": res.n_fail,
        "out_dir": res.out_dir,
        "final_sbs": res.final_sbs,
        "error": res.error,
    }, ensure_ascii=False, indent=2))
    yield (
        "\n".join(log_lines),
        res.final_sbs or None,
        [(Path(p).name, p) for p in res.cut_sbs_files],
        res,  # final yield: full PipelineResult for path propagation
    )


def _format_progress(event: str, payload: dict, log: list[str]) -> str:
    """오케스트레이터 progress event → 사용자용 메시지.

    이벤트는 `{source_tag}:{event_name}` 형태로 옴 (예: `m2svid:cut_done`,
    `autoshot:done`, `orchestrator:start`). source_tag 별로 매칭.
    """
    desc = ""

    # 오케스트레이터 자체 이벤트
    if event == "orchestrator:start":
        desc = f"🚀 시작 — out: {payload.get('out_dir', '?')}"
    elif event == "orchestrator:autoshot_launch":
        desc = "🎞 AutoShot 시작"
    elif event == "orchestrator:shotclass_launch":
        desc = "🏷 컷 분류기 (SigLIP-2)"
    elif event == "orchestrator:m2svid_launch":
        desc = "🎬 M2SVid 시작 (per-cut subprocess: depth → warp → inpaint → SBS → upscale)"
    elif event == "orchestrator:concat_start":
        desc = "🎞 ffmpeg concat"
    elif event in ("orchestrator:m2svid_failed", "orchestrator:autoshot_failed",
                   "orchestrator:shotclass_failed", "orchestrator:error"):
        desc = f"❌ {event}: {payload.get('message') or payload}"
    elif event == "orchestrator:done":
        elapsed = payload.get("elapsed_sec", 0)
        n_ok = payload.get("n_ok", 0)
        n_total = payload.get("n_total", 0)
        desc = f"✅ 완료 ({elapsed:.1f}s, {n_ok}/{n_total} 컷)"

    # AutoShot 단계 이벤트
    elif event.startswith("autoshot:"):
        sub = event.split(":", 1)[1]
        if sub == "done":
            n = payload.get("n_segments") or payload.get("n_scenes") or payload.get("n_cuts") or "?"
            desc = f"🎞 AutoShot 완료 — {n}개 컷"
        elif sub == "raw":
            line = payload.get("line", "")
            if any(t in line for t in ("Detected", "shot", "Frame", "FPS", "Processed", "ERROR", "Error")):
                desc = f"  [AutoShot] {line}"

    # Shot Classifier 단계
    elif event.startswith("shotclass:"):
        sub = event.split(":", 1)[1]
        if sub == "done":
            desc = "🏷 컷 분류 완료"
        elif sub == "raw":
            line = payload.get("line", "")
            if any(t in line for t in ("classified", "shot", "closeup", "normal", "wide", "ERROR", "Error")):
                desc = f"  [Classifier] {line}"

    # M2SVid 단계 — 핵심: 백엔드 엔진 진행상황 노출
    elif event.startswith("m2svid:"):
        sub = event.split(":", 1)[1]
        sid = payload.get("shot_id", "?")
        if sub == "start":
            n = payload.get("n_cuts", "?")
            psd = payload.get("per_shot_disparity", False)
            desc = f"🎬 M2SVid worker 시작 — {n}개 컷, per-shot disparity={'ON' if psd else 'OFF'}"
        elif sub == "cut_start":
            cls = payload.get("shot_class", "?")
            disp = payload.get("disparity_perc", "?")
            desc = f"  ▶ shot{sid} 시작 (class={cls}, disparity={disp})"
        elif sub == "cut_done":
            secs = payload.get("total_sec", 0)
            size_mb = payload.get("size_mb", 0)
            desc = f"  ✅ shot{sid} 완료 ({secs:.1f}s, {size_mb:.2f} MB)"
        elif sub == "cut_error":
            desc = f"  ❌ shot{sid} 실패: {payload.get('message', '')}"
        elif sub in ("subprocess_error", "subprocess_timeout", "subprocess_spawn_error"):
            desc = f"  ❌ shot{sid} {sub}: {payload.get('message', '')}"
        elif sub == "stage_log":
            # 핵심: per-cut runner 의 5단계 진행 + chunk 진행 노출
            line = payload.get("line", "")
            # show: stage markers, depth/warp/inpaint timing, RTX VSR, chunk progress, errors
            important = (
                "[1/5]" in line or "[2/5]" in line or "[3/5]" in line
                or "[4/5]" in line or "[5/5]" in line
                or "DONE:" in line or "[ERR]" in line or "[FATAL]" in line
                or "Traceback" in line
                or "depth done" in line or "warp done" in line or "inpaint done" in line
                or "chunk " in line.lower() or "RTX VSR:" in line
                or "upscale target" in line
                or "[STEREO_" in line  # env-gated hooks (TAESDV etc.)
                or "OOM" in line.upper() or "out of memory" in line.lower()
                # sub-stage internal messages (depth subprocess streamed output etc.)
                or "  depth: " in line or "  warp:" in line or "  inpaint:" in line
                or "%|" in line  # tqdm progress bar lines (already filtered upstream)
                or "VideoTransformer" in line  # m2svid model load progress
                or "Loading pipeline" in line or "Restored from" in line
            )
            if important:
                # strip the "[per_cut] HH:MM:SS " prefix to keep lines compact
                clean = line
                if clean.startswith("[per_cut] ") and len(clean) > 19:
                    clean = clean[19:].lstrip()
                desc = f"  [shot{sid}] {clean}"
        elif sub == "raw":
            # raw lines (worker stderr, non-JSON) — show errors/warnings only
            line = payload.get("line", "")
            if any(t in line.upper() for t in ("ERROR", "FAIL", "TRACEBACK", "OOM", "WARNING")):
                desc = f"  [m2svid:raw] {line}"
        elif sub == "done":
            sec = payload.get("sec", 0)
            ok = payload.get("n_ok", 0)
            fail = payload.get("n_fail", 0)
            n = payload.get("n_total", 0)
            if fail:
                desc = f"🎬 M2SVid 종료 — {ok}/{n} ok, {fail} FAIL ({sec:.1f}s)"
            else:
                desc = f"🎬 M2SVid 종료 — {ok}/{n} ok ({sec:.1f}s)"

    # Concat 단계
    elif event.startswith("concat:"):
        sub = event.split(":", 1)[1]
        if sub == "done":
            desc = f"🎞 concat 완료 → {Path(payload.get('final_sbs', '?')).name}"
        elif sub in ("error", "raw"):
            line = payload.get("line", "") or payload.get("message", "")
            if line and any(t in line.upper() for t in ("ERROR", "FAIL")):
                desc = f"  [concat] {line}"

    if desc:
        log.append(desc)
        _term_log(desc)
    return "\n".join(log)


def pipeline(
    video_path: str,
    out_root: str,
    # AutoShot
    threshold: float, min_duration: float,
    # Classifier
    use_shotclass: bool, shotclass_required: bool,
    # M2SVid
    processing_dim: int, output_dim: int,
    depth_backend: str, upscaler: str, rtx_vsr_quality: int,
    disparity_perc: float, seed: int, mask_antialias: int,
    # Orchestration
    concat: bool, fail_fast: bool,
    # Cache / rerun
    force_rerun: list, out_dir_override: str,
    # Per-stage imports
    import_cuts: str, import_shot_classes: str,
    import_sbs_dir: str, import_final_sbs: str,
    progress=gr.Progress(track_tqdm=False),
):
    if not video_path:
        raise gr.Error("입력 영상을 업로드 해주세요.")

    log_lines: list[str] = []
    progress(0.0, desc="setup")

    cfg = PipelineConfig(
        video=video_path,
        out=out_root,
        threshold=float(threshold),
        min_duration=float(min_duration),
        use_shotclass=bool(use_shotclass),
        shotclass_required=bool(shotclass_required),
        shotclass_backend="siglip2",
        processing_dim=int(processing_dim),
        output_dim=int(output_dim),
        depth_backend=str(depth_backend),
        upscaler=str(upscaler),
        rtx_vsr_quality=int(rtx_vsr_quality),
        disparity_perc=float(disparity_perc),
        seed=int(seed),
        mask_antialias=int(mask_antialias),
        concat=bool(concat),
        fail_fast=bool(fail_fast),
        out_dir=(out_dir_override.strip() or None) if isinstance(out_dir_override, str) else None,
        force_rerun=",".join(force_rerun) if isinstance(force_rerun, list) else "",
        import_cuts=(import_cuts.strip() or None) if isinstance(import_cuts, str) else None,
        import_shot_classes=(import_shot_classes.strip() or None) if isinstance(import_shot_classes, str) else None,
        import_sbs_dir=(import_sbs_dir.strip() or None) if isinstance(import_sbs_dir, str) else None,
        import_final_sbs=(import_final_sbs.strip() or None) if isinstance(import_final_sbs, str) else None,
    )

    # 진행률 추적용 mutable state (per-cut subprocess 진행 보간)
    _state = {"n_cuts": 1, "last_done": 0, "last_prog": 0.05}

    def _cb(event: str, payload: dict):
        msg = _format_progress(event, payload, log_lines)
        # event prefix-based progress mapping
        prog = _state["last_prog"]
        if event.endswith(":start") and "n_cuts" in payload:
            _state["n_cuts"] = max(1, int(payload.get("n_cuts", 1)))
        if event == "autoshot:done":
            n_cuts = payload.get("n_segments") or payload.get("n_scenes") or payload.get("n_cuts")
            if isinstance(n_cuts, int) and n_cuts > 0:
                _state["n_cuts"] = n_cuts
            prog = 0.20
        elif event == "shotclass:done":
            prog = 0.30
        elif event == "orchestrator:m2svid_launch":
            prog = 0.35
        elif event == "m2svid:start":
            n = payload.get("n_cuts")
            if isinstance(n, int) and n > 0:
                _state["n_cuts"] = n
            prog = 0.36
        elif event == "m2svid:cut_start":
            sid = int(payload.get("shot_id", 0))
            # interpolate inside cut window: each cut spans 0.55 / n_cuts of bar
            base = 0.40 + 0.55 * max(0, sid - 1) / max(1, _state["n_cuts"])
            prog = base
        elif event == "m2svid:cut_done":
            sid = int(payload.get("shot_id", 0))
            _state["last_done"] = sid
            prog = 0.40 + 0.55 * sid / max(1, _state["n_cuts"])
        elif event == "m2svid:stage_log":
            # interpolate within current cut based on stage marker [N/5]
            sid = int(payload.get("shot_id", 0)) or _state["last_done"] + 1
            line = payload.get("line", "")
            stage_idx = 0
            for i in range(1, 6):
                if f"[{i}/5]" in line:
                    stage_idx = i
                    break
            if stage_idx:
                cut_start = 0.40 + 0.55 * max(0, sid - 1) / max(1, _state["n_cuts"])
                cut_end = 0.40 + 0.55 * sid / max(1, _state["n_cuts"])
                prog = cut_start + (cut_end - cut_start) * (stage_idx - 1) / 5
        elif event == "m2svid:done":
            prog = 0.95
        elif event == "concat:done":
            prog = 0.98
        elif event == "orchestrator:done":
            prog = 1.0
        prog = max(_state["last_prog"], min(0.999, prog))
        _state["last_prog"] = prog
        # use last log line as live description
        last_line = log_lines[-1] if log_lines else ""
        try:
            progress(prog, desc=last_line)
        except Exception:
            pass

    # ── Live streaming via background thread + queue ──────────────────
    # Gradio yields intermediate (log_text, sbs, cuts) tuples while the
    # pipeline runs in a worker thread. _cb signals the queue on each event,
    # the main thread (this generator) yields the latest log to update the UI.
    import queue as _queue
    import threading as _threading

    event_q: _queue.Queue = _queue.Queue()
    holder: dict = {"res": None, "exc": None}

    # Wrap _cb to also signal the queue
    _cb_inner = _cb
    def _cb_streaming(event: str, payload: dict):
        try:
            _cb_inner(event, payload)
        finally:
            event_q.put_nowait("update")

    def _runner():
        try:
            holder["res"] = run_pipeline(cfg, progress=_cb_streaming)
        except BaseException as e:
            holder["exc"] = e
        finally:
            event_q.put_nowait("done")

    t = _threading.Thread(target=_runner, daemon=True, name="m2svid-pipeline")
    t.start()

    # Yield initial empty state, then drain queue
    yield ("\n".join(log_lines) or "(시작 중...)", None, [])
    while True:
        try:
            msg = event_q.get(timeout=60)
        except _queue.Empty:
            yield ("\n".join(log_lines) + "\n... (60s 무응답, 진행 중)", None, [])
            continue
        if msg == "done":
            break
        # de-duplicate: drain any queued updates so we don't yield N times in a row
        while True:
            try:
                event_q.get_nowait()
            except _queue.Empty:
                break
        yield ("\n".join(log_lines), None, [])

    # Re-raise worker exceptions in the main thread (Gradio shows them properly)
    if holder["exc"] is not None:
        log_lines.append("")
        log_lines.append("=" * 60)
        log_lines.append(f"❌ 예외: {type(holder['exc']).__name__}: {holder['exc']}")
        yield ("\n".join(log_lines), None, [])
        raise holder["exc"]  # type: ignore[misc]

    res = holder["res"]

    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append(json.dumps({
        "elapsed_sec": res.elapsed_sec,
        "n_cuts": res.n_cuts,
        "n_ok": res.n_ok,
        "n_fail": res.n_fail,
        "final_sbs": res.final_sbs,
        "error": res.error,
    }, ensure_ascii=False, indent=2))

    yield (
        "\n".join(log_lines),
        res.final_sbs or None,
        [(Path(p).name, p) for p in res.cut_sbs_files],
    )


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title=f"Stereo Pipeline {VERSION} — M2SVid") as demo:
        # ── Header bar (sticky, system info) ─────────────────────────── #
        gr.HTML(uk.header_bar_html(
            VERSION, DEFAULT_M2SVID_PYTHON, DEFAULT_OUT_ROOT, PORT,
            extra_lines=[
                f"<b>m2svid_service:</b> {Path(DEFAULT_M2SVID_SERVICE).name}",
            ],
        ))
        # ── Status pill + Reset state row ────────────────────────────── #
        with gr.Row():
            status_pill = gr.HTML(uk.status_pill_html("idle"))
            with gr.Column(scale=0, min_width=200):
                with gr.Row():
                    reset_gpu_cb = gr.Checkbox(label="GPU cleanup", value=True,
                                               container=False, scale=0,
                                               min_width=120)
                    reset_btn = gr.Button("🔄 Reset state", size="sm",
                                          variant="secondary", scale=0,
                                          min_width=130)
        status_msg = gr.Markdown("", elem_id="sp-status-msg")

        gr.Markdown(
            "**AutoShot → SigLIP-2 → VDA depth → M2SVid inpaint → SBS**\n\n"
            f"v0.17 = M2SVid full-attention (Step2 -90.8% vs v0.16b SVD UNet). "
            f"⚠ 1080p 처리 시 96GB VRAM 권장 (RTX Pro 6000+) — 32GB 환경은 v0.16b 권장."
        )

        # Shared state (stage tabs share these)
        out_root_state = str(HERE / "outputs")
        # Depth backends with verified weights/install — checked once at app
        # startup. To pick up newly-added weights, restart the app.
        available_depth_backends = _list_available_depth_backends()

        with gr.Tabs():
            # ─── Stage tab: AutoShot ─────────────────────────────────────────
            with gr.TabItem("🎞 AutoShot"):
                gr.Markdown("### 컷 분할 (AutoShot)\n비디오 → cuts/cuts_metadata.json + shot###.mp4")
                with gr.Row():
                    with gr.Column():
                        as_video = gr.Video(label="입력 영상", sources=["upload"])
                        as_out_root = gr.Textbox(label="출력 root",
                                                 value=out_root_state, scale=3)
                        as_threshold = gr.Slider(0.10, 0.50, value=0.296, step=0.005,
                                                 label="threshold")
                        as_min_dur = gr.Slider(0.0, 1.0, value=0.5, step=0.1,
                                               label="min duration (초)")
                        as_force = gr.Checkbox(value=False, label="기존 결과 무시 (강제 재실행)",
                                               info="미체크 시 cache hit → instant skip")
                        as_run = gr.Button("🎞 AutoShot 실행", variant="primary")
                    with gr.Column():
                        as_log = gr.Textbox(label="진행 로그", lines=20, interactive=False)
                        # Hidden outputs absorb the 3-tuple yields (log, sbs, cuts).
                        # AutoShot stage doesn't produce video/gallery; hidden.
                        as_hidden_video = gr.Video(visible=False)
                        as_hidden_gallery = gr.Gallery(visible=False)
                # NOTE: as_run.click(...) wired at end of build_ui so it can
                # reference cl_cuts_path / m_cuts_path / od_cuts_path defined
                # in subsequent tabs.

            # ─── Stage tab: Classifier ──────────────────────────────────────
            with gr.TabItem("🏷 Classifier"):
                gr.Markdown("### 컷 분류 (SigLIP-2)\ncuts → shot_classes.json (closeup/normal/wide)")
                with gr.Row():
                    with gr.Column():
                        cl_video = gr.Video(label="입력 영상 (hash 계산용)", sources=["upload"])
                        cl_out_root = gr.Textbox(label="출력 root",
                                                 value=out_root_state, scale=3)
                        with gr.Row():
                            cl_cuts_dd = gr.Dropdown(
                                choices=list_cuts_dirs(out_root_state),
                                label="기존 cuts 폴더 (선택하면 경로 자동 입력)",
                                interactive=True, scale=4,
                            )
                            cl_cuts_refresh = gr.Button("🔄 Refresh", scale=1)
                        cl_cuts_path = gr.Textbox(
                            label="cuts 폴더 경로 (수동 입력 또는 dropdown 선택)",
                            placeholder="예: outputs/.../cuts/",
                            info="비우면 같은 영상 hash 의 자동 폴더 사용",
                        )
                        cl_force = gr.Checkbox(value=False, label="기존 분류 결과 무시 (강제 재실행)")
                        cl_run = gr.Button("🏷 Classifier 실행", variant="primary")
                    with gr.Column():
                        cl_log = gr.Textbox(label="진행 로그", lines=20, interactive=False)
                        cl_hidden_video = gr.Video(visible=False)
                        cl_hidden_gallery = gr.Gallery(visible=False)
                cl_cuts_dd.change(lambda v: v or "", inputs=[cl_cuts_dd], outputs=[cl_cuts_path])
                cl_cuts_refresh.click(
                    lambda r: gr.Dropdown(choices=list_cuts_dirs(r)),
                    inputs=[cl_out_root], outputs=[cl_cuts_dd],
                )
                # NOTE: cl_run.click(...) wired at end of build_ui (forward
                # ref to m_sc_path / od_sc_path).

            # ─── Stage tab: M2SVid ──────────────────────────────────────────
            with gr.TabItem("🎬 M2SVid"):
                gr.Markdown("### 스테레오 inpaint (M2SVid)\ncuts + (선택) shot_classes → SBS .mp4 per cut")
                with gr.Row():
                    with gr.Column():
                        m_video = gr.Video(label="입력 영상 (hash 계산용)", sources=["upload"])
                        m_out_root = gr.Textbox(label="출력 root",
                                                value=out_root_state, scale=3)
                        with gr.Row():
                            m_cuts_dd = gr.Dropdown(
                                choices=list_cuts_dirs(out_root_state),
                                label="cuts 폴더 (필수)", interactive=True, scale=4,
                            )
                            m_cuts_refresh = gr.Button("🔄", scale=1)
                        m_cuts_path = gr.Textbox(label="cuts 경로",
                                                 placeholder="dropdown 또는 수동")
                        with gr.Row():
                            m_sc_dd = gr.Dropdown(
                                choices=list_shot_classes(out_root_state),
                                label="shot_classes.json (선택, per-shot disparity)",
                                interactive=True, scale=4,
                            )
                            m_sc_refresh = gr.Button("🔄", scale=1)
                        m_sc_path = gr.Textbox(label="shot_classes.json 경로 (선택)",
                                               placeholder="비우면 균일 disparity 사용")
                        m_depth = gr.Dropdown(
                            choices=available_depth_backends,
                            value="VDA-S", label="Depth backend",
                            info="가중치/venv 가 검증된 항목만 표시됨 (재기동 시 갱신)",
                        )
                        m_proc = gr.Slider(384, 1024, value=512, step=64, label="처리 dim")
                        m_out_dim = gr.Slider(0, 2160, value=0, step=64, label="출력 dim (0=처리 그대로)")
                        m_upscaler = gr.Dropdown(choices=["lanczos", "rtx_vsr"],
                                                 value="lanczos", label="upscaler")
                        m_vsr_q = gr.Slider(0, 19, value=4, step=1, label="RTX VSR quality")
                        m_disp = gr.Slider(0.005, 0.05, value=0.02, step=0.005, label="disparity perc")
                        m_seed = gr.Number(value=42, label="seed", precision=0)
                        m_mask = gr.Slider(0, 1, value=0, step=1, label="mask antialias")
                        m_force = gr.Checkbox(value=False, label="기존 SBS 무시 (강제 재실행)")
                        m_run = gr.Button("🎬 M2SVid 실행", variant="primary")
                    with gr.Column():
                        m_log = gr.Textbox(label="진행 로그", lines=20, interactive=False)
                        m_gallery = gr.Gallery(label="컷별 SBS", columns=2)
                        m_hidden_video = gr.Video(visible=False)
                m_cuts_dd.change(lambda v: v or "", inputs=[m_cuts_dd], outputs=[m_cuts_path])
                m_cuts_refresh.click(
                    lambda r: gr.Dropdown(choices=list_cuts_dirs(r)),
                    inputs=[m_out_root], outputs=[m_cuts_dd],
                )
                m_sc_dd.change(lambda v: v or "", inputs=[m_sc_dd], outputs=[m_sc_path])
                m_sc_refresh.click(
                    lambda r: gr.Dropdown(choices=list_shot_classes(r)),
                    inputs=[m_out_root], outputs=[m_sc_dd],
                )
                # NOTE: m_run.click(...) wired at end of build_ui
                # (forward ref to c_sbs_path).

            # ─── Stage tab: Resolution Overdrive ────────────────────────────
            with gr.TabItem("🚀 Resolution Overdrive"):
                gr.Markdown(
                    "### 🚀 Resolution Overdrive — 처리 해상도 ↑ ↔ chunk T ↓\n"
                    "Standard 25f@512 와 별도 폴더 (`sbs_{preset}_{chunk}f{dim}/`) 로 저장.\n"
                    "Concat 은 Concat 탭에서 별도 실행."
                )
                gr.Markdown(
                    "> ⚠️ **주의** — 처리시간 ~2-2.5x, VRAM 28GB+ 필요, "
                    "M2SVid 학습 윈도우(25f) 외 영역 → temporal flicker / chunk seam 가능. "
                    "OOM 시 자동 fallback 없음 (실패 시 chunk T 키우거나 dim 낮춰서 재시도)."
                )
                with gr.Row():
                    with gr.Column():
                        od_video = gr.Video(label="입력 영상 (hash 계산용, cuts 폴더 있으면 선택)",
                                            sources=["upload"])
                        od_out_root = gr.Textbox(label="출력 root",
                                                 value=out_root_state, scale=3)
                        with gr.Row():
                            od_cuts_dd = gr.Dropdown(
                                choices=list_cuts_dirs(out_root_state),
                                label="cuts 폴더 (필수)", interactive=True, scale=4,
                            )
                            od_cuts_refresh = gr.Button("🔄", scale=1)
                        od_cuts_path = gr.Textbox(label="cuts 경로",
                                                  placeholder="dropdown 또는 수동")
                        with gr.Row():
                            od_sc_dd = gr.Dropdown(
                                choices=list_shot_classes(out_root_state),
                                label="shot_classes.json (선택, per-shot disparity)",
                                interactive=True, scale=4,
                            )
                            od_sc_refresh = gr.Button("🔄", scale=1)
                        od_sc_path = gr.Textbox(label="shot_classes.json 경로 (선택)",
                                                placeholder="비우면 균일 disparity 사용")

                        gr.Markdown("**Preset**")
                        od_preset = gr.Radio(
                            choices=list(OVERDRIVE_PRESETS.keys()),
                            value="Overdrive (12f @ 720)",
                            label="처리 모드",
                            info="Mild(안전) → Overdrive(권장) → Extreme(공격적) / Custom 은 슬라이더 활성화",
                        )
                        with gr.Group():
                            gr.Markdown("**Custom params** (Custom 선택 시 사용)")
                            od_chunk = gr.Slider(
                                4, 25, value=12, step=1,
                                label="chunk size (시간 윈도우)",
                                info="default 25 = M2SVid 학습 윈도우. 작을수록 VRAM 여유 + dim ↑ 가능",
                                interactive=False,
                            )
                            od_dim = gr.Slider(
                                256, 960, value=720, step=64,
                                label="processing dim (64-div)",
                                info="64 단위 자동 round",
                                interactive=False,
                            )

                        od_suffix_preview = gr.Textbox(
                            label="output suffix (preview)",
                            value="_overdrive_12f720",
                            interactive=False,
                        )

                        with gr.Accordion("M2SVid 공통 옵션", open=False):
                            od_depth = gr.Dropdown(
                                choices=available_depth_backends,
                                value="VDA-S", label="Depth backend",
                                info="가중치/venv 가 검증된 항목만 표시됨",
                            )
                            od_upscaler = gr.Dropdown(choices=["lanczos", "rtx_vsr"],
                                                      value="lanczos", label="upscaler")
                            od_vsr_q = gr.Slider(0, 19, value=4, step=1, label="RTX VSR quality")
                            od_out_dim = gr.Slider(
                                0, 2160, value=1080, step=64,
                                label="출력 dim (per-eye height; 0=처리 그대로)",
                                info=("0 = upscale skip (단순 copy). >0 = lanczos / RTX VSR 로 "
                                      "per-eye 높이를 이 값으로 키움. 1080 → SBS 3840×1080. "
                                      "RTX VSR 동작에 필수."),
                            )
                            od_disp = gr.Slider(0.005, 0.05, value=0.02, step=0.005,
                                                label="disparity perc")
                            od_seed = gr.Number(value=42, label="seed", precision=0)
                            od_mask = gr.Slider(0, 1, value=0, step=1, label="mask antialias")

                        od_force = gr.Checkbox(value=False,
                                               label="기존 Overdrive SBS 무시 (강제 재실행)")
                        od_run = gr.Button("🚀 Run Resolution Overdrive", variant="primary")
                    with gr.Column():
                        od_log = gr.Textbox(label="진행 로그", lines=22, interactive=False)
                        od_gallery = gr.Gallery(label="컷별 SBS (Overdrive)", columns=2)
                        od_hidden_video = gr.Video(visible=False)

                # Wiring: dropdowns → text paths
                od_cuts_dd.change(lambda v: v or "", inputs=[od_cuts_dd], outputs=[od_cuts_path])
                od_cuts_refresh.click(
                    lambda r: gr.Dropdown(choices=list_cuts_dirs(r)),
                    inputs=[od_out_root], outputs=[od_cuts_dd],
                )
                od_sc_dd.change(lambda v: v or "", inputs=[od_sc_dd], outputs=[od_sc_path])
                od_sc_refresh.click(
                    lambda r: gr.Dropdown(choices=list_shot_classes(r)),
                    inputs=[od_out_root], outputs=[od_sc_dd],
                )

                # Preset → toggle Custom slider interactivity + update suffix preview
                def _on_preset_change(label, cur_chunk, cur_dim):
                    is_custom = (label == "Custom")
                    name, def_chunk, def_dim = OVERDRIVE_PRESETS.get(label, ("custom", 12, 720))
                    if is_custom:
                        eff_chunk = int(cur_chunk)
                        eff_dim = max(64, (int(cur_dim) // 64) * 64)
                    else:
                        eff_chunk, eff_dim = def_chunk, def_dim
                    suffix = f"_{name}_{eff_chunk}f{eff_dim}"
                    return (
                        gr.update(interactive=is_custom, value=eff_chunk),
                        gr.update(interactive=is_custom, value=eff_dim),
                        suffix,
                    )

                od_preset.change(
                    _on_preset_change,
                    inputs=[od_preset, od_chunk, od_dim],
                    outputs=[od_chunk, od_dim, od_suffix_preview],
                )

                # Custom slider edits → update suffix preview
                def _on_custom_change(label, cur_chunk, cur_dim):
                    name, _, _ = OVERDRIVE_PRESETS.get(label, ("custom", 12, 720))
                    eff_chunk = int(cur_chunk)
                    eff_dim = max(64, (int(cur_dim) // 64) * 64)
                    return f"_{name}_{eff_chunk}f{eff_dim}"

                od_chunk.change(
                    _on_custom_change,
                    inputs=[od_preset, od_chunk, od_dim],
                    outputs=[od_suffix_preview],
                )
                od_dim.change(
                    _on_custom_change,
                    inputs=[od_preset, od_chunk, od_dim],
                    outputs=[od_suffix_preview],
                )

                # NOTE: od_run.click(...) wired at end of build_ui
                # (forward ref to c_sbs_path).

            # ─── Stage tab: Concat ──────────────────────────────────────────
            with gr.TabItem("🎞 Concat"):
                gr.Markdown("### ffmpeg concat\nSBS dir → final_sbs.mp4")
                with gr.Row():
                    with gr.Column():
                        c_video = gr.Video(label="입력 영상 (hash 계산용)", sources=["upload"])
                        c_out_root = gr.Textbox(label="출력 root",
                                                value=out_root_state, scale=3)
                        with gr.Row():
                            c_sbs_dd = gr.Dropdown(
                                choices=list_sbs_dirs(out_root_state),
                                label="SBS 폴더", interactive=True, scale=4,
                            )
                            c_sbs_refresh = gr.Button("🔄", scale=1)
                        c_sbs_path = gr.Textbox(label="SBS 폴더 경로",
                                                placeholder="dropdown 또는 수동")
                        c_run = gr.Button("🎞 Concat 실행", variant="primary")
                    with gr.Column():
                        c_log = gr.Textbox(label="진행 로그", lines=15, interactive=False)
                        c_video_out = gr.Video(label="final_sbs.mp4")
                        c_hidden_gallery = gr.Gallery(visible=False)
                c_sbs_dd.change(lambda v: v or "", inputs=[c_sbs_dd], outputs=[c_sbs_path])
                c_sbs_refresh.click(
                    lambda r: gr.Dropdown(choices=list_sbs_dirs(r)),
                    inputs=[c_out_root], outputs=[c_sbs_dd],
                )
                c_run.click(
                    fn=run_concat_stage,
                    inputs=[c_video, c_out_root, c_sbs_path],
                    outputs=[c_log, c_video_out, c_hidden_gallery],
                )

            # ─── Tab: 🚀 All-in-one Pipeline (existing) ──────────────────────
            with gr.TabItem("🚀 Pipeline"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 입력")
                        video_in = gr.Video(label="입력 영상", sources=["upload"])
                        out_root = gr.Textbox(
                            label="출력 디렉토리 root",
                            value=str(HERE / "outputs"),
                            info="결과는 {root}/{video_stem}_{timestamp}/final_sbs.mp4 에 저장",
                        )

                        gr.Markdown("### 🎞 AutoShot")
                        threshold = gr.Slider(0.10, 0.50, value=0.296, step=0.005,
                                              label="threshold",
                                              info="0.296 = 논문 best-F1")
                        min_duration = gr.Slider(0.0, 1.0, value=0.5, step=0.1,
                                                 label="min duration (초)",
                                                 info="컷 최소 길이 (micro-cut 차단)")

                        gr.Markdown("### 🏷 Shot Classifier (SigLIP-2)")
                        use_shotclass = gr.Checkbox(value=True, label="shot classifier 사용",
                                                    info="컷별 closeup/normal/wide 분류 (현재는 정보 수집만)")
                        shotclass_required = gr.Checkbox(value=False,
                                                         label="shotclass 실패 시 중단",
                                                         info="기본: 경고만 + 진행")

                    with gr.Column():
                        gr.Markdown("### 🎬 M2SVid")
                        depth_backend = gr.Dropdown(
                            choices=available_depth_backends,
                            value="VDA-S", label="Depth backend",
                            info="가중치/venv 가 검증된 항목만 표시됨. VDA-S: 빠름·균형 (default) / VDA-L: 정확, NC 라이선스 / FlashDepth-L: 가장 빠름 (~30 FPS @ 518²) / DepthCrafter: 가장 느림. (Windows 에서 FlashDepth-L 은 dist stub patch 적용됨)",
                        )
                        processing_dim = gr.Slider(384, 1024, value=512, step=64,
                                                   label="처리 해상도 max",
                                                   info="M2SVid 학습 해상도 512. 1024 까지 OK, 64 단위")
                        output_dim = gr.Slider(0, 2160, value=0, step=64,
                                               label="출력 해상도 max",
                                               info="0 = 처리 해상도 그대로, >0 = 업스케일 (1080 → ~3840×1080 SBS)")
                        upscaler = gr.Dropdown(
                            choices=["lanczos", "rtx_vsr"],
                            value="lanczos", label="업스케일 방식",
                            info="rtx_vsr 은 nvidia-vfx wheel 필요 (RTX 5090 Blackwell native)",
                        )
                        rtx_vsr_quality = gr.Slider(0, 19, value=4, step=1,
                                                    label="RTX VSR quality",
                                                    info="0=BICUBIC, 1=LOW, 2=MEDIUM, 3=HIGH, 4=ULTRA(default) / 8-11 DENOISE_LOW..ULTRA / 12-15 DEBLUR_LOW..ULTRA / 16-19 HIGHBITRATE_LOW..ULTRA")
                        disparity_perc = gr.Slider(0.005, 0.05, value=0.02, step=0.005,
                                                   label="disparity 비율",
                                                   info="warp 강도 (영상 폭의 %)")
                        with gr.Accordion("고급 옵션", open=False):
                            seed = gr.Number(value=42, label="seed", precision=0)
                            mask_antialias = gr.Slider(0, 1, value=0, step=1,
                                                       label="mask antialias")
                            concat = gr.Checkbox(value=True, label="ffmpeg concat",
                                                 info="컷별 SBS 합쳐 최종 final_sbs.mp4 생성")
                            fail_fast = gr.Checkbox(value=True,
                                                    label="첫 컷 실패 시 중단")
                            gr.Markdown("**캐시 / 재실행** — 동일 영상 재실행 시 이전 결과 자동 재사용 (content-hash 기반 폴더)")
                            force_rerun = gr.CheckboxGroup(
                                choices=["autoshot", "classifier", "m2svid", "concat"],
                                value=[], label="강제 재실행 단계",
                                info="체크된 단계는 캐시 무시하고 재실행. 미체크 = 출력 존재 시 skip",
                            )
                            out_dir_override = gr.Textbox(
                                label="출력 폴더 직접 지정 (외부 import용)",
                                value="", placeholder="예: C:/Users/PC/Desktop/.../이전실행폴더 (비우면 자동 hash 경로)",
                                info="기존 다른 실행의 결과 폴더를 가리키면 그 단계들 skip 가능",
                            )
                            gr.Markdown("**단계별 외부 결과 가져오기 (선택)** — 경로 지정 시 base_out 으로 복사 → 해당 단계 자동 skip")
                            import_cuts = gr.Textbox(
                                label="cuts 폴더 import (AutoShot skip)",
                                value="", placeholder="예: outputs/.../cuts/",
                                info="cuts_metadata.json + shot*.mp4 가 있는 폴더",
                            )
                            import_shot_classes = gr.Textbox(
                                label="shot_classes.json import (Classifier skip)",
                                value="", placeholder="예: outputs/.../shot_classes/shot_classes.json",
                            )
                            import_sbs_dir = gr.Textbox(
                                label="SBS 폴더 import (M2SVid skip)",
                                value="", placeholder="예: outputs/.../sbs/",
                                info="shot*_sbs.mp4 들이 있는 폴더",
                            )
                            import_final_sbs = gr.Textbox(
                                label="final_sbs.mp4 import (Concat skip)",
                                value="", placeholder="예: outputs/.../final_sbs.mp4",
                            )

                with gr.Row():
                    run_btn = gr.Button("🎬 Run", variant="primary", size="lg")

                gr.Markdown("### 결과")
                with gr.Row():
                    with gr.Column():
                        log_out = gr.Textbox(label="진행 로그", lines=18, interactive=False)
                    with gr.Column():
                        sbs_out = gr.Video(label="final_sbs.mp4")
                        cuts_out = gr.Gallery(label="컷별 SBS", columns=2)

                run_btn.click(
                    fn=pipeline,
                    inputs=[
                        video_in, out_root,
                        threshold, min_duration,
                        use_shotclass, shotclass_required,
                        processing_dim, output_dim,
                        depth_backend, upscaler, rtx_vsr_quality,
                        disparity_perc, seed, mask_antialias,
                        concat, fail_fast,
                        force_rerun, out_dir_override,
                        import_cuts, import_shot_classes,
                        import_sbs_dir, import_final_sbs,
                    ],
                    outputs=[log_out, sbs_out, cuts_out],
                )

            # Tab 2: Outputs
            with gr.TabItem("📂 Outputs"):
                gr.Markdown("최근 실행 결과 — 각 항목 클릭 시 final_sbs.mp4 로드")
                out_root_view = gr.Textbox(label="root", value=str(HERE / "outputs"))
                out_list = gr.Dropdown(label="실행 기록", choices=[])
                refresh_btn = gr.Button("🔄 새로고침")
                preview = gr.Video(label="preview")

                def _refresh(root):
                    items = list_outputs(root)
                    return gr.Dropdown(choices=items, value=(items[0][1] if items else None))

                def _preview(path):
                    return path

                refresh_btn.click(_refresh, inputs=[out_root_view], outputs=[out_list])
                out_list.change(_preview, inputs=[out_list], outputs=[preview])

            # Tab 3: Settings/Environment
            with gr.TabItem("⚙️ Settings"):
                gr.Markdown(check_environment())
                gr.Markdown("""
                ### v0.16m 변경 사항

                - **Step2 엔진 교체**: GenStereo SVD UNet → M2SVid full-attention
                - **출력 규격**: 처리 해상도 + lanczos / RTX VSR 업스케일
                - **vendored 코드**: `local_engines/m2svid/` (inpaint_core.py, warping.py 외)
                - **runtime 의존**: `m2svid_service/.venv` (sgm + pytorch_lightning + torch 2.9 cu128)

                ### RTX VSR 활성화 (선택)

                ```
                pip install nvidia-vfx
                ```
                → RTX 5090 Blackwell 네이티브 가속, DLPack zero-copy interop.
                fallback: 자동으로 lanczos.
                """)

                # ── Force Reboot (v0.17.5+) ───────────────────────────
                gr.Markdown("### 🔄 Force Reboot")
                gr.Markdown(
                    "⚠️ **진행 중 작업 모두 중단됨.** 코드 변경 (.venv-flashdepth, "
                    "app.py, run_pipeline.py 등) 적용 시 사용.  \n"
                    "재기동 절차:  \n"
                    "1. 아래 [재기동 확인] 체크 → [🔴 강제 재기동] 활성화  \n"
                    "2. 클릭하면 즉시 종료 + .bat watchdog 가 자동 재시작 (~2초)  \n"
                    "3. **브라우저는 5.5초 후 자동 새로고침** (수동 F5 불필요)"
                )
                reboot_confirm = gr.Checkbox(
                    value=False, label="재기동 확인 (실수 방지)",
                )
                reboot_btn = gr.Button(
                    "🔴 강제 재기동", variant="stop", interactive=False,
                )
                reboot_status = gr.Markdown("")

                def _toggle_reboot_btn(checked):
                    return gr.update(interactive=bool(checked))

                def _do_reboot():
                    """Schedule forced exit (code 42) so .bat watchdog restarts."""
                    import threading
                    def _kill():
                        time.sleep(0.5)  # let HTTP response flush
                        # SIGKILL-equivalent — bypass atexit / cleanup so any
                        # hung subprocess parent is forcibly torn down.
                        os._exit(42)
                    threading.Thread(target=_kill, daemon=True,
                                     name="force-reboot").start()
                    return ("🔄 재기동 중... 5.5초 후 자동 새로고침\n"
                            "(.bat watchdog 가 exit code 42 감지 후 자동 재시작)")

                reboot_confirm.change(
                    _toggle_reboot_btn,
                    inputs=[reboot_confirm], outputs=[reboot_btn],
                )
                # Q-B: client-side JS schedules location.reload() in 5.5s so
                # the browser picks up the freshly-restarted Gradio without
                # the user pressing F5.  The empty-array return passes through
                # to the Python handler unchanged.
                reboot_btn.click(
                    _do_reboot,
                    outputs=[reboot_status],
                    js="() => { setTimeout(() => location.reload(), 5500); return []; }",
                )

        # ── Cross-tab .click() wiring (v0.17.4+ stage→stage propagation) ──
        # Defined here (after ALL tab components exist) so AutoShot can write
        # to Classifier/M2SVid/Overdrive textboxes, etc. Each stage handler
        # yields its standard 3 outputs (log, video, gallery) PLUS extra
        # textbox values for downstream tabs (filled on completion only).
        as_run.click(
            fn=run_autoshot_stage,
            inputs=[as_video, as_out_root, as_threshold, as_min_dur, as_force],
            outputs=[as_log, as_hidden_video, as_hidden_gallery,
                     # propagation: cuts_dir → Classifier / M2SVid / Overdrive
                     cl_cuts_path, m_cuts_path, od_cuts_path],
        )
        cl_run.click(
            fn=run_classifier_stage,
            inputs=[cl_video, cl_out_root, cl_cuts_path, cl_force],
            outputs=[cl_log, cl_hidden_video, cl_hidden_gallery,
                     # propagation: shot_classes_json → M2SVid / Overdrive
                     m_sc_path, od_sc_path],
        )
        m_run.click(
            fn=run_m2svid_stage,
            inputs=[m_video, m_out_root, m_cuts_path, m_sc_path,
                    m_proc, m_out_dim, m_depth, m_upscaler, m_vsr_q,
                    m_disp, m_seed, m_mask, m_force],
            outputs=[m_log, m_hidden_video, m_gallery,
                     # propagation: sbs_dir → Concat
                     c_sbs_path],
        )
        od_run.click(
            fn=run_overdrive_stage,
            inputs=[od_video, od_out_root, od_cuts_path, od_sc_path,
                    od_preset, od_chunk, od_dim,
                    od_depth, od_upscaler, od_vsr_q,
                    od_disp, od_seed, od_mask,
                    od_out_dim, od_force],
            outputs=[od_log, od_hidden_video, od_gallery,
                     # propagation: sbs_dir (suffixed) → Concat
                     c_sbs_path],
        )
        # c_run.click(...) is wired inline within the Concat tab (terminal,
        # no propagation needed).

        # ── Reset state wiring (header bar) ──────────────────────────── #
        def _reset_v17(gpu_cleanup):
            msg = "🔄 reset"
            if gpu_cleanup:
                msg = uk.gpu_cleanup_subprocess(DEFAULT_M2SVID_PYTHON)
            return (uk.status_pill_html("idle"),
                    f"⏱ {time.strftime('%H:%M:%S')} · {msg}")
        reset_btn.click(fn=_reset_v17, inputs=[reset_gpu_cb],
                        outputs=[status_pill, status_msg])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=int(os.environ.get("GRADIO_CONCURRENCY", "1"))).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=PORT,
        show_error=True,
        inbrowser=os.environ.get("GRADIO_INBROWSER", "0") == "1",
        auth=_gradio_auth(),
        share=False,
        theme=uk.THEME, css=uk.HEADER_CSS,
    )
