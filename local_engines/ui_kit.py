"""
ui_kit.py — shared Gradio widgets for stereo_pipeline branches.

Copy-pasted (not imported across folders) to keep version isolation. To update
all four folders in sync:
  port/stereo_pipeline_v0.16b/local_engines/ui_kit.py
  port/stereo_pipeline_v0.16xf/local_engines/ui_kit.py
  port/stereo_pipeline_v0.17/local_engines/ui_kit.py
  port/stereo_pipeline_v0.16ta/local_engines/ui_kit.py

Public API:
  HEADER_CSS                — extra CSS for the header bar + status pill
  THEME                     — gr.themes.Default(primary_hue="purple")
  PRESETS_DIR               — default JSON preset directory (~/.stereo_pipeline_presets/)
  HISTORY_PATH              — default history JSON (~/.stereo_pipeline_history.json)

  diag_gpu()                  -> str   (e.g. "NVIDIA RTX 5090 (32 GB)")
  diag_lan_url(port)          -> str   ("http://192.168.x.x:7862")
  diag_free_space(path)       -> str   ("Free: 421 GB")
  diag_xformers(sc_python)    -> str   ("xformers 0.0.35 patched · cutlassF-pt") | "fallback (sdpa)"
  diag_sc_python(path)        -> str   ("xformers_build" | "python_embed" | "missing")

  status_pill_html(state, msg) -> str  rich HTML for status indicator
  header_bar_html(...)         -> str  full header HTML

  scan_runs(out_root)         -> list[dict]  (each: name, video_path, n_cuts, status, mtime)
  history_load() / history_save_entry(...)  — past video paths

  build_dataset_selector(out_root) -> tuple   (dropdown, path_textbox, refresh_btn, on_change_fn)
  build_outputs_gallery()          -> tuple   (gallery, refresh_btn)
  build_logs_viewer()              -> tuple   (textbox, grep_input, scroll_btn)
  build_presets_manager(form_state) -> tuple  (load_dd, save_name_box, save_btn, load_btn)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import gradio as gr


# ─── Theme + CSS ─────────────────────────────────────────────────────────── #

THEME = gr.themes.Default(
    primary_hue="purple",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    spacing_size=gr.themes.sizes.spacing_md,
)

HEADER_CSS = """
.sp-header {
    background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
    border: 1px solid #e9d5ff;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.sp-header-title {
    font-size: 22px;
    font-weight: 700;
    color: #4c1d95;
    margin: 0 0 6px 0;
    letter-spacing: -0.2px;
}
.sp-header-info {
    font-family: 'JetBrains Mono', 'Consolas', 'Menlo', monospace;
    font-size: 12px;
    color: #6b21a8;
    line-height: 1.5;
    word-break: break-all;
}
.sp-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}
.sp-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    border: 1px solid;
    white-space: nowrap;
}
.sp-pill-idle    { background: #f3f4f6; color: #6b7280; border-color: #d1d5db; }
.sp-pill-running { background: #ede9fe; color: #6d28d9; border-color: #c4b5fd; }
.sp-pill-done    { background: #d1fae5; color: #065f46; border-color: #6ee7b7; }
.sp-pill-error   { background: #fee2e2; color: #991b1b; border-color: #fca5a5; }
.sp-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
}
.sp-pill-running .sp-dot { animation: sp-pulse 1.4s ease-in-out infinite; }
@keyframes sp-pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.35; }
}
.sp-tab-emoji { font-size: 16px; margin-right: 4px; }
.sp-section-title { color: #6d28d9; font-weight: 600; font-size: 13px; margin: 0 0 4px 2px; }
"""


# ─── Defaults ────────────────────────────────────────────────────────────── #

HOME = Path(os.path.expanduser("~"))
PRESETS_DIR = HOME / ".stereo_pipeline_presets"
HISTORY_PATH = HOME / ".stereo_pipeline_history.json"
HISTORY_LIMIT = 30


# ─── Diag helpers (subprocess-cached on first call) ──────────────────────── #

_diag_cache: dict[str, tuple[float, str]] = {}
_DIAG_TTL = 30.0  # seconds


def _cache_get(key: str, ttl: float = _DIAG_TTL) -> Optional[str]:
    e = _diag_cache.get(key)
    if e and time.time() - e[0] < ttl:
        return e[1]
    return None


def _cache_set(key: str, value: str) -> str:
    _diag_cache[key] = (time.time(), value)
    return value


def diag_gpu() -> str:
    cached = _cache_get("gpu")
    if cached:
        return cached
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=4,
        )
        line = r.stdout.strip().splitlines()[0]
        name, mem = [s.strip() for s in line.split(",")]
        mem_gb = int(int(mem.replace(" MiB", "").strip()) / 1024 + 0.5)
        return _cache_set("gpu", f"{name} ({mem_gb} GB)")
    except Exception as e:
        return _cache_set("gpu", f"(GPU probe failed: {type(e).__name__})")


def diag_lan_url(port: int = 7862) -> str:
    cached = _cache_get(f"lan:{port}")
    if cached:
        return cached
    try:
        # Pick the LAN-facing interface (not loopback)
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                return _cache_set(f"lan:{port}", f"http://{ip}:{port}")
        return _cache_set(f"lan:{port}", f"http://127.0.0.1:{port}")
    except Exception:
        return _cache_set(f"lan:{port}", f"http://127.0.0.1:{port}")


def diag_free_space(path: str | Path) -> str:
    try:
        anchor = Path(path)
        # walk up until we hit a real existing directory
        while anchor and not anchor.exists():
            anchor = anchor.parent
        usage = shutil.disk_usage(str(anchor))
        free_gb = usage.free / 1024**3
        return f"Free: {free_gb:.0f} GB"
    except Exception:
        return "Free: ?"


def diag_sc_python(sc_python: str | Path) -> str:
    p = Path(sc_python)
    if not p.exists():
        return f"❌ missing: {p.name}"
    parent = p.parent.parent.name
    grand = p.parent.parent.parent.name
    return f"{grand}/{parent}"


def diag_xformers(sc_python: str | Path, force: bool = False) -> str:
    """Probe xformers status by running a tiny script in sc_python.

    Cached for 5 minutes (changes only on env reinstall).
    """
    p = Path(sc_python)
    cache_key = f"xf:{p}"
    if not force:
        cached = _cache_get(cache_key, ttl=300.0)
        if cached:
            return cached
    if not p.exists():
        return _cache_set(cache_key, "n/a (sc_python missing)")
    probe = (
        "import sys\n"
        "try:\n"
        "    import xformers\n"
        "    v = xformers.__version__\n"
        "    from xformers.ops.fmha import cutlass\n"
        "    cap = cutlass.FwOp.CUDA_MAXIMUM_COMPUTE_CAPABILITY\n"
        "    cap_ok = cap[0] >= 12\n"
        "    print(f'xf={v} cap={cap[0]}.{cap[1]} cap_ok={int(cap_ok)}')\n"
        "except ImportError as e:\n"
        "    print('xf=missing')\n"
        "except Exception as e:\n"
        "    print(f'xf=err {type(e).__name__}')\n"
    )
    try:
        r = subprocess.run(
            [str(p), "-c", probe],
            capture_output=True, text=True, timeout=12,
        )
        out = r.stdout.strip()
        if out.startswith("xf=missing"):
            return _cache_set(cache_key, "xformers: not installed (sdpa fallback)")
        if out.startswith("xf=err"):
            return _cache_set(cache_key, f"xformers: {out[3:]}")
        m = re.match(r"xf=([\w.+-]+) cap=([\d.]+) cap_ok=([01])", out)
        if m:
            ver, cap, cap_ok = m.group(1), m.group(2), m.group(3) == "1"
            label = "sm_120 unlocked" if cap_ok else f"cap≤{cap} (sdpa fallback)"
            return _cache_set(cache_key, f"xformers {ver} · {label}")
        return _cache_set(cache_key, f"xformers: {out[:60]}")
    except subprocess.TimeoutExpired:
        return _cache_set(cache_key, "xformers: probe timeout")
    except Exception as e:
        return _cache_set(cache_key, f"xformers: probe failed ({type(e).__name__})")


# ─── Status pill HTML ────────────────────────────────────────────────────── #

def status_pill_html(state: str = "idle", message: str = "") -> str:
    state = state if state in ("idle", "running", "done", "error") else "idle"
    label = {"idle": "idle", "running": "running", "done": "done", "error": "error"}[state]
    if message:
        label = f"{label}: {message}"
    return (
        f'<span class="sp-pill sp-pill-{state}">'
        f'<span class="sp-dot"></span>'
        f'<span>{label}</span>'
        f'</span>'
    )


# ─── Header bar ──────────────────────────────────────────────────────────── #

def header_bar_html(
    version: str,
    sc_python: str | Path,
    out_dir: str | Path,
    port: int = 7862,
    extra_lines: Optional[list[str]] = None,
) -> str:
    parts = [
        f"<b>GPU:</b> {diag_gpu()}",
        f"<b>sc_python:</b> {diag_sc_python(sc_python)}",
        f"<b>{diag_xformers(sc_python)}</b>",
        f"<b>LAN:</b> <a href='{diag_lan_url(port)}' target='_blank' style='color:inherit'>{diag_lan_url(port)}</a>",
        f"<b>{diag_free_space(out_dir)}</b>",
    ]
    if extra_lines:
        parts.extend(extra_lines)
    info = " · ".join(parts)
    return (
        f'<div class="sp-header">'
        f'<div class="sp-header-title">Stereo Pipeline {version}</div>'
        f'<div class="sp-header-info">{info}</div>'
        f'</div>'
    )


def refresh_header_html(version: str, sc_python: str | Path, out_dir: str | Path,
                       port: int = 7862, force_xformers: bool = False) -> str:
    """Re-render header (called on Refresh button)."""
    if force_xformers:
        diag_xformers(sc_python, force=True)
    # Clear gpu+lan+free cache so fresh probe runs
    for k in list(_diag_cache.keys()):
        if k in ("gpu",) or k.startswith("lan:"):
            _diag_cache.pop(k, None)
    return header_bar_html(version, sc_python, out_dir, port)


# ─── Run scanner (Dataset dropdown source) ───────────────────────────────── #

def scan_runs(out_root: str | Path) -> list[dict]:
    """Scan <out_root> for run subfolders. Returns list of dicts:
       {label, video_name, video_path (best-guess), cuts_dir, n_cuts, status, mtime}
    """
    root = Path(out_root)
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        # name pattern: <video_stem>_<timestamp>
        name = d.name
        m = re.match(r"^(.+?)_(\d{10})$", name)
        if m:
            video_stem = m.group(1)
            ts = m.group(2)
        else:
            video_stem = name
            ts = ""
        cuts = d / "cuts"
        meta = cuts / "cuts_metadata.json"
        n_cuts = "?"
        original_video = ""
        if meta.exists():
            try:
                with open(meta, "r", encoding="utf-8") as f:
                    md = json.load(f)
                segs = md.get("segments", [])
                n_cuts = md.get("n_segments") or len(segs)
                # Try to recover original video path from first segment's `file`
                if segs and isinstance(segs, list):
                    first = segs[0].get("file", "")
                    if first:
                        # cuts/<video_stem>_shot001.mp4 → infer original
                        # Or cuts metadata may have 'source_video' / 'video' key
                        original_video = md.get("source_video") or md.get("video") or ""
            except Exception:
                pass
        # Status: check for final_sbs.mp4
        finals = list(d.glob("final_sbs*.mp4"))
        if finals:
            status = f"✓ {len(finals)} final"
        elif (d / "sbs").exists() and any((d / "sbs").glob("shot*_sbs.mp4")):
            n_sbs = len(list((d / "sbs").glob("shot*_sbs.mp4")))
            status = f"⚠ partial {n_sbs} SBS"
        elif meta.exists():
            status = "cuts only"
        else:
            status = "empty"
        try:
            mtime = d.stat().st_mtime
        except Exception:
            mtime = 0
        # Heuristic for original video: check parent of cuts/ for the input mp4
        # (orchestrator usually doesn't copy it, but history.json may have it)
        label = f"{video_stem} · {n_cuts} cuts · {status}"
        if ts:
            label += f"   ({_fmt_ts(ts)})"
        out.append(dict(
            label=label,
            video_name=video_stem,
            video_path=original_video,
            cuts_dir=str(cuts) if meta.exists() else "",
            n_cuts=n_cuts,
            status=status,
            mtime=mtime,
            run_dir=str(d),
        ))
    return out


def _fmt_ts(ts: str) -> str:
    try:
        # ts is unix seconds string
        t = int(ts)
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
    except Exception:
        return ts


# ─── History (~/.stereo_pipeline_history.json) ───────────────────────────── #

def history_load() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[:HISTORY_LIMIT]
    except Exception:
        pass
    return []


def history_save_entry(video_path: str, run_dir: str = "", n_cuts: int | str = "",
                       status: str = "") -> None:
    if not video_path:
        return
    items = history_load()
    # de-dup by video_path
    items = [it for it in items if it.get("video_path") != video_path]
    items.insert(0, dict(
        video_path=video_path,
        run_dir=run_dir,
        n_cuts=n_cuts,
        status=status,
        ts=time.time(),
    ))
    items = items[:HISTORY_LIMIT]
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ─── Dataset selector (dropdown of past runs + Refresh) ──────────────────── #

def build_dataset_choices(out_root: str | Path) -> tuple[list[tuple[str, str]], dict[str, dict]]:
    """Build dropdown choices. Returns (choices, metadata_lookup).

    choices: [(display_label, value_id), ...] where value_id is unique key
    metadata_lookup: {value_id: {video_path, cuts_dir, run_dir, ...}}
    """
    runs = scan_runs(out_root)
    history = history_load()

    choices: list[tuple[str, str]] = []
    lookup: dict[str, dict] = {}

    # 1. Past runs from outputs/
    for r in runs:
        key = f"run:{r['run_dir']}"
        choices.append((f"📁 {r['label']}", key))
        lookup[key] = r

    # 2. History entries (if not already covered by a run)
    seen_paths = {r.get("video_path") for r in runs if r.get("video_path")}
    for h in history:
        vp = h.get("video_path", "")
        if not vp or vp in seen_paths:
            continue
        key = f"hist:{vp}"
        label = f"🕘 {Path(vp).name} · {h.get('n_cuts','?')} cuts · {h.get('status','')}"
        choices.append((label, key))
        lookup[key] = dict(
            label=label,
            video_path=vp,
            cuts_dir="",
            run_dir=h.get("run_dir", ""),
            n_cuts=h.get("n_cuts", "?"),
            status=h.get("status", ""),
        )

    if not choices:
        choices = [("(no past runs found — use textbox below)", "")]
        lookup[""] = {}

    return choices, lookup


# Module-level lookup cache (populated on dropdown rebuild)
_dataset_lookup: dict[str, dict] = {}


def dataset_on_change(selected_key: str) -> tuple[str, str]:
    """Dropdown change handler. Returns (video_path, cuts_dir)."""
    info = _dataset_lookup.get(selected_key, {})
    return info.get("video_path", ""), info.get("cuts_dir", "")


def dataset_refresh(out_root: str) -> tuple[gr.Dropdown, str]:
    """Refresh button handler. Returns (new_dropdown, status_msg)."""
    global _dataset_lookup
    choices, lookup = build_dataset_choices(out_root)
    _dataset_lookup = lookup
    msg = f"refreshed: {len(choices)} entries from {out_root}"
    return gr.Dropdown(choices=choices, value=None), msg


# ─── Presets manager (JSON load/save) ────────────────────────────────────── #

def presets_list() -> list[str]:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted([p.stem for p in PRESETS_DIR.glob("*.json")])


def preset_save(name: str, data: dict) -> str:
    if not name or not name.strip():
        return "❌ preset name empty"
    name = re.sub(r"[^\w\-]+", "_", name.strip())
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    path = PRESETS_DIR / f"{name}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return f"✓ saved {name}.json"
    except Exception as e:
        return f"❌ save failed: {e}"


def preset_load(name: str) -> dict:
    if not name:
        return {}
    path = PRESETS_DIR / f"{name}.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ─── Reset state helper ──────────────────────────────────────────────────── #

def gpu_cleanup_subprocess(sc_python: str | Path) -> str:
    """Spawn a tiny subprocess to call torch.cuda.empty_cache(). Returns status."""
    p = Path(sc_python)
    if not p.exists():
        return "(skipped: sc_python missing)"
    cmd = (
        "import torch, gc\n"
        "if torch.cuda.is_available():\n"
        "    gc.collect()\n"
        "    torch.cuda.empty_cache()\n"
        "    torch.cuda.synchronize()\n"
        "    f = torch.cuda.mem_get_info()\n"
        "    print(f'free={f[0]/1024**3:.1f}GB total={f[1]/1024**3:.1f}GB')\n"
        "else:\n"
        "    print('cuda unavailable')\n"
    )
    try:
        r = subprocess.run([str(p), "-c", cmd], capture_output=True, text=True, timeout=15)
        return f"GPU cleanup: {r.stdout.strip()}"
    except subprocess.TimeoutExpired:
        return "GPU cleanup: timeout"
    except Exception as e:
        return f"GPU cleanup: {type(e).__name__}"


# ─── Logs viewer (post-run, with grep) ───────────────────────────────────── #

def logs_filter(log_text: str, grep: str) -> str:
    if not grep or not grep.strip():
        return log_text
    pat = grep.strip()
    try:
        rx = re.compile(pat, re.IGNORECASE)
        out = "\n".join(line for line in log_text.splitlines() if rx.search(line))
        return out or "(no matches)"
    except re.error:
        # treat as plain substring
        out = "\n".join(line for line in log_text.splitlines() if pat.lower() in line.lower())
        return out or "(no matches)"


# ─── Tab title helpers ───────────────────────────────────────────────────── #

TAB_TITLES = {
    "input":   "🎬 Input",
    "cuts":    "🎞 Cuts",
    "stereo":  "🎨 Stereo",
    "outputs": "📂 Outputs",
    "settings": "⚙️ Settings",
}


def section_title(text: str) -> str:
    """Small purple section heading inside a tab."""
    return f'<div class="sp-section-title">{text}</div>'
