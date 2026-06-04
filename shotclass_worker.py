"""
shotclass_worker.py
──────────────────────────────────────────────────────────────────────
Subprocess C — Shot Classifier로 컷별 wide/normal/closeup 분류.

단독 프로세스로 실행되어 종료 시 OS가 VRAM을 전부 회수.
shot_classifier/classifier.py를 import해 모델 로드는 한 번만, 모든 컷 순차 분류.

실행 요구:
    port/shot_classifier/venv/Scripts/python.exe 로 실행

입력:
    --cuts-meta             cuts_metadata.json 경로
    --out-dir               shot_classes/ 출력 디렉토리
    --backend               clip | depth (기본 clip)
    --models-dir            shot_classifier/models/{backend}/ 캐시 경로
    --depth-std-wide        depth 백엔드 임계값 (기본 0.25)
    --depth-std-closeup     (기본 0.12)
    --max-disp-wide         wide 클래스 → max_disp (기본 30)
    --max-disp-normal       (기본 20)
    --max-disp-closeup      (기본 12)

출력:
    {out_dir}/shot_classes.json — shot_id 키로 {class, confidence, scores, max_disp, ...}
    {out_dir}/thumbnails/shot###.jpg — middle frame 썸네일

이벤트:
    {"event":"start", "n_cuts":N, "backend":...}
    {"event":"models_loading"}
    {"event":"models_loaded", "sec":T, "vram":...}
    {"event":"shot_classified", "shot_id":i, "class":..., "max_disp":...}
    {"event":"shot_error", "shot_id":i, "message":...}
    {"event":"done", "sec":T, "n_ok":N, "n_fail":M}
    {"event":"error", "message":...}
"""
from __future__ import annotations

import argparse
import json
import os
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
# v0.16b: prefer LOCAL classifier (with SigLIP-2 backend) over the shared
# port/shot_classifier (which still ships CLIP-only). Falls back to shared
# location if local module is unavailable.
LOCAL_CLASSIFIER_DIR = HERE / "local_engines" / "shot_classifier"
SHOT_CLASSIFIER_DIR = LOCAL_CLASSIFIER_DIR if (LOCAL_CLASSIFIER_DIR / "classifier.py").exists() \
    else (PORT_ROOT / "shot_classifier")


def _emit(event: str, **kwargs):
    print(json.dumps({"event": event, **kwargs}, ensure_ascii=False), flush=True)


def _vram_status() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"VRAM {used:.1f}/{reserved:.1f}/{total:.0f}GB (used/res/tot)"
    except Exception:
        pass
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Shot classifier subprocess worker")
    parser.add_argument("--cuts-meta", required=True)
    parser.add_argument("--out-dir", required=True)
    # v0.16b: 'siglip2' is new default backend (was 'clip' in v0.16)
    parser.add_argument("--backend", choices=["siglip2", "clip", "depth"], default="siglip2")
    parser.add_argument("--models-dir", required=True,
                        help="shot_classifier/models/{backend}/ 경로")
    parser.add_argument("--depth-std-wide", type=float, default=0.25)
    parser.add_argument("--depth-std-closeup", type=float, default=0.12)
    parser.add_argument("--max-disp-wide", type=float, default=30.0)
    parser.add_argument("--max-disp-normal", type=float, default=20.0)
    parser.add_argument("--max-disp-closeup", type=float, default=12.0)
    args = parser.parse_args()

    meta_path = Path(args.cuts_meta).resolve()
    out_dir = Path(args.out_dir).resolve()
    models_dir = Path(args.models_dir).resolve()

    if not meta_path.exists():
        _emit("error", message=f"cuts_metadata.json not found: {meta_path}")
        return 2
    if not SHOT_CLASSIFIER_DIR.exists():
        _emit("error", message=f"shot_classifier dir not found: {SHOT_CLASSIFIER_DIR}")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = out_dir / "thumbnails"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    segments = meta.get("segments", [])
    total = len(segments)
    if total == 0:
        _emit("error", message="no segments in cuts_metadata.json")
        return 5

    # shot_classifier/classifier.py import
    sys.path.insert(0, str(SHOT_CLASSIFIER_DIR))
    try:
        from classifier import make_classifier, CLASSES  # noqa: E402
    except Exception as e:
        _emit("error", message=f"import shot_classifier.classifier failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return 3

    _emit("start", n_cuts=total, backend=args.backend, out_dir=str(out_dir))

    try:
        from decord import VideoReader, cpu
        from PIL import Image
        import numpy as np
        import torch
    except Exception as e:
        _emit("error", message=f"missing deps (decord/PIL/numpy/torch): {e}")
        return 3

    # 모델 로드
    _emit("models_loading", backend=args.backend)
    t_load = time.time()
    try:
        clf = make_classifier(
            args.backend,
            depth_std_wide=args.depth_std_wide,
            depth_std_closeup=args.depth_std_closeup,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clf.load(cache_dir=models_dir, device=device)
    except Exception as e:
        _emit("error", message=f"classifier load failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return 3
    _emit("models_loaded", sec=round(time.time() - t_load, 2),
          vram=_vram_status(), device=device)

    max_disp_map = {
        "wide": float(args.max_disp_wide),
        "normal": float(args.max_disp_normal),
        "closeup": float(args.max_disp_closeup),
    }

    results: dict[str, dict] = {}
    n_ok = n_fail = 0
    overall_t0 = time.time()

    for seg in segments:
        shot_id = seg["shot_id"]
        cut_file = Path(seg["file"]).resolve()
        if not cut_file.exists():
            _emit("shot_error", shot_id=shot_id,
                  message=f"cut file missing: {cut_file}")
            n_fail += 1
            continue

        try:
            vr = VideoReader(str(cut_file), ctx=cpu(0))
            n = len(vr)
            if n == 0:
                raise RuntimeError("empty video")
            mid = n // 2
            frame = vr[mid].asnumpy()  # RGB H×W×3 uint8

            # 썸네일 저장 (max 512px)
            thumb_path = thumbs_dir / f"shot{shot_id:03d}.jpg"
            img = Image.fromarray(frame)
            w, h = img.size
            scale = 512 / max(w, h)
            if scale < 1:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            img.save(str(thumb_path), "JPEG", quality=85)

            pred = clf.predict(frame)
            cls = pred["class"]
            max_disp = max_disp_map[cls]

            results[str(shot_id)] = {
                "shot_id": shot_id,
                "file": str(cut_file),
                "class": cls,
                "confidence": pred["confidence"],
                "scores": pred["scores"],
                "max_disp": max_disp,
                "thumbnail": str(thumb_path),
                "extra": pred.get("extra", {}),
            }
            _emit(
                "shot_classified",
                shot_id=shot_id,
                **{
                    "class": cls,
                    "confidence": round(pred["confidence"], 3),
                    "max_disp": max_disp,
                },
            )
            n_ok += 1
        except Exception as e:
            _emit("shot_error", shot_id=shot_id, message=str(e))
            traceback.print_exc(file=sys.stderr)
            n_fail += 1

    # 저장
    out_payload = {
        "backend": args.backend,
        "max_disp_map": max_disp_map,
        "depth_std_wide": args.depth_std_wide if args.backend == "depth" else None,
        "depth_std_closeup": args.depth_std_closeup if args.backend == "depth" else None,
        "n_total": total,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "shots": results,
    }
    out_json = out_dir / "shot_classes.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    # 명시적 unload (이미 프로세스 exit으로 회수되지만 로그 차원)
    try:
        clf.unload()
    except Exception:
        pass

    _emit(
        "done",
        sec=round(time.time() - overall_t0, 2),
        n_ok=n_ok,
        n_fail=n_fail,
        out_json=str(out_json),
    )
    return 0 if n_fail == 0 else (0 if n_ok > 0 else 6)


if __name__ == "__main__":
    sys.exit(main())
