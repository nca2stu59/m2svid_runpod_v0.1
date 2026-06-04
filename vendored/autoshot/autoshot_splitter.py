"""
autoshot_splitter.py
────────────────────────────────────────────────────────────────────────────
Shot Boundary Detection + 영상 분할 (멀티 백엔드)

지원 백엔드:
  - autoshot (기본): TransNetV2 + AutoShot 가중치. GPU 사용.
      pip install transnetv2-pytorch ffmpeg-python
  - ecr: Edge Change Ratio. CPU 전용, 가벼움.
      pip install opencv-python
  - psd-adaptive: PySceneDetect AdaptiveDetector. CPU 전용.
      pip install scenedetect[opencv]

AutoShot 가중치 파일 위치 (우선순위):
  1. autoshot/ckpt_0_200_0.pth   (현재 폴더 기준)
  2. 환경변수 AUTOSHOT_WEIGHTS 가 지정한 경로

인터페이스:
  run_pipeline(video_path, output_dir, threshold, min_duration, backend=...) -> dict
"""

import subprocess
import json
import os
import time
import argparse
from pathlib import Path


# ══════════════════════════════════════════════════════════════
# 1. ffprobe 기준 실제 비디오 메타
# ══════════════════════════════════════════════════════════════

def _get_video_info(video_path: str) -> tuple[float, float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,duration",
        "-of", "json", video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info   = json.loads(result.stdout)
    stream = info["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    if "duration" in stream:
        duration = float(stream["duration"])
    else:
        cmd2 = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", video_path
        ]
        r2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        duration = float(json.loads(r2.stdout)["format"]["duration"])
    return fps, duration


# ══════════════════════════════════════════════════════════════
# 2. AutoShot 모델 로드
# ══════════════════════════════════════════════════════════════

def _load_autoshot_model(weights_path: str = None):
    """
    TransNetV2 아키텍처에 AutoShot 가중치를 로드한다.

    weights_path 미지정 시 탐색 순서:
      1. ./autoshot/ckpt_0_200_0.pth
      2. 환경변수 AUTOSHOT_WEIGHTS
    """
    try:
        import torch
        from transnetv2_pytorch import TransNetV2
    except ImportError:
        raise ImportError(
            "transnetv2-pytorch 또는 torch가 설치되지 않았습니다.\n"
            "설치: pip install transnetv2-pytorch"
        )

    # 가중치 경로 결정
    if weights_path is None:
        candidates = [
            Path(__file__).parent / "autoshot" / "ckpt_0_200_0.pth",
            Path(os.environ.get("AUTOSHOT_WEIGHTS", "")),
        ]
        for c in candidates:
            if c.exists():
                weights_path = str(c)
                break

    if weights_path is None or not Path(weights_path).exists():
        raise FileNotFoundError(
            "AutoShot 가중치 파일을 찾을 수 없습니다.\n"
            "확인: ./autoshot/ckpt_0_200_0.pth 위치에 파일이 있는지 확인하세요.\n"
            "또는 환경변수 AUTOSHOT_WEIGHTS=/path/to/ckpt_0_200_0.pth 를 설정하세요."
        )

    print(f"[AutoShot] Loading weights from: {weights_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = TransNetV2(device=device)

    # AutoShot 체크포인트 로드
    # ckpt_0_200_0.pth 는 Lightning checkpoint 형식일 수 있음 — 두 방식 시도
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict):
        # Lightning checkpoint: 'state_dict' 키 안에 있음
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Lightning prefix 제거 (예: "model." prefix)
            state_dict = {
                k.replace("model.", "").replace("net.", ""): v
                for k, v in state_dict.items()
            }
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # strict=False: 키 불일치 허용 (아키텍처 세부 차이 흡수)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[AutoShot] Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"[AutoShot] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

    model.eval()
    print(f"[AutoShot] Model loaded on {device}")
    return model


# ══════════════════════════════════════════════════════════════
# 3. AutoShot 씬 감지
# ══════════════════════════════════════════════════════════════

def detect_scenes(video_path: str,
                  threshold: float = 0.5,
                  weights_path: str = None,
                  ) -> tuple[list[dict], float, float]:
    """
    AutoShot 모델로 씬 경계를 감지한다.

    Returns
    -------
    scenes   : list[dict]  start_frame, end_frame, probability, start_sec, end_sec
    fps      : float  (ffprobe 기준)
    duration : float
    """
    fps, duration = _get_video_info(video_path)
    model = _load_autoshot_model(weights_path)

    # predict_video: transnetv2-pytorch의 ffmpeg 기반 프레임 추출 사용
    result = model.analyze_video(video_path, threshold=threshold, quiet=False)

    scenes_raw = result.get("scenes", [])
    scenes = []
    for s in scenes_raw:
        sf        = s["start_frame"]
        ef        = s["end_frame"]
        start_sec = sf / fps
        end_sec   = ef / fps
        scenes.append({
            "start_frame": int(sf),
            "end_frame":   int(ef),
            "probability": round(float(s.get("probability", threshold)), 4),
            "start_sec":   round(start_sec, 4),
            "end_sec":     round(end_sec,   4),
        })

    return scenes, fps, duration


# ══════════════════════════════════════════════════════════════
# 3b. ECR (Edge Change Ratio) 씬 감지
# ══════════════════════════════════════════════════════════════

def detect_scenes_ecr(video_path: str,
                      threshold: float = 0.5,
                      ) -> tuple[list[dict], float, float]:
    """
    ECR (Edge Change Ratio) 기반 씬 경계 감지.

    클래식 CV 방법: Canny edge → dilate → 프레임간 entering/exiting 픽셀 비율.
    GPU 불필요, CPU 전용, 빠르고 가벼움.

    Returns: (scenes, fps, duration) — detect_scenes() 와 동일 포맷.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise ImportError("ECR backend requires opencv-python: pip install opencv-python")

    fps, duration = _get_video_info(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[ECR] Analyzing {total_frames} frames (threshold={threshold})", flush=True)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    boundaries: list[tuple[int, float]] = []  # (frame_idx, ecr_value)

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return [], fps, duration

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_edges = cv2.Canny(prev_gray, 50, 150)
    prev_dilated = cv2.dilate(prev_edges, kernel)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        dilated = cv2.dilate(edges, kernel)

        # Entering: edges in current frame not covered by dilated previous
        entering = cv2.bitwise_and(edges, cv2.bitwise_not(prev_dilated))
        # Exiting: edges in previous frame not covered by dilated current
        exiting = cv2.bitwise_and(prev_edges, cv2.bitwise_not(dilated))

        n_edges = max(np.count_nonzero(edges), 1)
        n_prev_edges = max(np.count_nonzero(prev_edges), 1)

        ecr = max(
            np.count_nonzero(entering) / n_edges,
            np.count_nonzero(exiting) / n_prev_edges,
        )

        if ecr > threshold:
            boundaries.append((frame_idx, ecr))

        prev_edges = edges
        prev_dilated = dilated

    cap.release()

    # Convert boundaries to scenes
    scenes = []
    prev_boundary = 0
    for bnd_frame, ecr_val in boundaries:
        if bnd_frame > prev_boundary:
            scenes.append({
                "start_frame": prev_boundary,
                "end_frame":   bnd_frame - 1,
                "probability": round(ecr_val, 4),
                "start_sec":   round(prev_boundary / fps, 4),
                "end_sec":     round((bnd_frame - 1) / fps, 4),
            })
        prev_boundary = bnd_frame

    # Last scene to end of video
    if prev_boundary < frame_idx:
        scenes.append({
            "start_frame": prev_boundary,
            "end_frame":   frame_idx,
            "probability": 0.0,
            "start_sec":   round(prev_boundary / fps, 4),
            "end_sec":     round(frame_idx / fps, 4),
        })

    print(f"[ECR] Detected {len(scenes)} scenes ({len(boundaries)} boundaries)", flush=True)
    return scenes, fps, duration


# ══════════════════════════════════════════════════════════════
# 3c. PSD-Adaptive (PySceneDetect AdaptiveDetector) 씬 감지
# ══════════════════════════════════════════════════════════════

def detect_scenes_psd_adaptive(video_path: str,
                               threshold: float = 3.0,
                               ) -> tuple[list[dict], float, float]:
    """
    PySceneDetect AdaptiveDetector 기반 씬 경계 감지.

    adaptive_threshold: 프레임간 content 변화의 rolling-average 대비 배수.
    기본값 3.0 (PySceneDetect 기본). 낮을수록 민감.

    Returns: (scenes, fps, duration) — detect_scenes() 와 동일 포맷.
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import AdaptiveDetector
    except ImportError:
        raise ImportError(
            "PSD-Adaptive backend requires scenedetect: "
            "pip install scenedetect[opencv]"
        )

    fps, duration = _get_video_info(video_path)
    print(f"[PSD-Adaptive] Analyzing video (adaptive_threshold={threshold})", flush=True)

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=float(threshold)))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    scenes = []
    for start_tc, end_tc in scene_list:
        sf = start_tc.get_frames()
        ef = end_tc.get_frames()
        scenes.append({
            "start_frame": sf,
            "end_frame":   ef,
            "probability": 1.0,
            "start_sec":   round(start_tc.get_seconds(), 4),
            "end_sec":     round(end_tc.get_seconds(), 4),
        })

    print(f"[PSD-Adaptive] Detected {len(scenes)} scenes", flush=True)
    return scenes, fps, duration


# ══════════════════════════════════════════════════════════════
# 4. 영상 분할 (libx264 CRF18 재인코딩)
# ══════════════════════════════════════════════════════════════

def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_coverage_windows(scenes: list[dict],
                            actual_duration: float,
                            min_duration: float) -> tuple[list[dict], dict]:
    min_len = max(_as_float(min_duration), 0.1)
    ordered = sorted(
        scenes,
        key=lambda s: (_as_float(s.get("start_sec")), _as_float(s.get("end_sec"))),
    )
    intervals = []
    prev_end = 0.0

    for idx, scene in enumerate(ordered):
        raw_start = max(0.0, min(_as_float(scene.get("start_sec")), actual_duration))
        start = 0.0 if idx == 0 else max(prev_end, raw_start)
        raw_end = max(start, min(_as_float(scene.get("end_sec"), start), actual_duration))

        if idx + 1 < len(ordered):
            next_start = max(0.0, min(_as_float(ordered[idx + 1].get("start_sec"), raw_end), actual_duration))
            end = max(raw_end, next_start)
        else:
            end = max(raw_end, actual_duration)
        end = min(max(end, start), actual_duration)

        if end <= start:
            continue

        intervals.append({
            "start": start,
            "end": end,
            "probability": _as_float(scene.get("probability")),
            "source_scene_count": 1,
            "merged_short_scenes": 0,
        })
        prev_end = end

    windows = []
    pending = None
    merged_short = 0

    for interval in intervals:
        dur = interval["end"] - interval["start"]
        if dur >= min_len:
            if pending:
                interval["start"] = pending["start"]
                interval["probability"] = max(interval["probability"], pending["probability"])
                interval["source_scene_count"] += pending["source_scene_count"]
                interval["merged_short_scenes"] += pending["source_scene_count"]
                pending = None
            windows.append(interval)
            continue

        merged_short += 1
        if windows:
            prev = windows[-1]
            prev["end"] = max(prev["end"], interval["end"])
            prev["probability"] = max(prev["probability"], interval["probability"])
            prev["source_scene_count"] += interval["source_scene_count"]
            prev["merged_short_scenes"] += interval["source_scene_count"]
        elif pending:
            pending["end"] = max(pending["end"], interval["end"])
            pending["probability"] = max(pending["probability"], interval["probability"])
            pending["source_scene_count"] += interval["source_scene_count"]
        else:
            pending = dict(interval)

    if pending:
        windows.append(pending)

    for window in windows:
        window["duration"] = max(0.0, window["end"] - window["start"])

    stats = {
        "policy": "preserve_coverage_merge_short_scenes",
        "min_effective_duration": round(min_len, 4),
        "input_scenes": len(scenes),
        "input_intervals": len(intervals),
        "output_windows": len(windows),
        "merged_short_scenes": merged_short,
    }
    return windows, stats


def split_scenes(video_path: str,
                 scenes: list[dict],
                 actual_duration: float,
                 output_dir: str,
                 min_duration: float = 0.0) -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)
    stem     = Path(video_path).stem
    segments = []
    shot_id  = 0

    windows, split_stats = _build_coverage_windows(scenes, actual_duration, min_duration)
    split_scenes.last_stats = split_stats
    if split_stats["merged_short_scenes"]:
        print(
            "[ShotSplit] Coverage-preserving merge: "
            f"{split_stats['input_scenes']} scenes -> {split_stats['output_windows']} segments, "
            f"merged_short_scenes={split_stats['merged_short_scenes']}",
            flush=True,
        )

    for scene in windows:
        start_sec = scene["start"]
        end_sec   = scene["end"]
        if start_sec >= actual_duration:
            continue
        end_sec = min(end_sec, actual_duration)
        dur = end_sec - start_sec

        shot_id += 1
        print(f"[ShotSplit] Splitting shot {shot_id}/{len(windows)}  {start_sec:.2f}s~{end_sec:.2f}s", flush=True)
        out_file = os.path.join(output_dir, f"{stem}_shot{shot_id:03d}.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.6f}",
            "-i",  video_path,
            "-t",  f"{dur:.6f}",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-avoid_negative_ts", "1",
            out_file
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if r.returncode != 0 or not os.path.exists(out_file):
            print(f"  [WARN] shot{shot_id:03d} encode failed, skipping.")
            shot_id -= 1
            continue

        segments.append({
            "shot_id":     shot_id,
            "file":        out_file,
            "start":       round(start_sec, 4),
            "end":         round(end_sec,   4),
            "duration":    round(dur,       4),
            "probability": scene.get("probability", 0.0),
            "source_scene_count": scene.get("source_scene_count", 1),
            "merged_short_scenes": scene.get("merged_short_scenes", 0),
        })

    return segments


def _coverage_summary(segments: list[dict], actual_duration: float) -> dict:
    if actual_duration <= 0:
        return {
            "duration": 0.0,
            "ratio": 0.0,
            "gap_count": 0,
            "gap_sec": 0.0,
            "largest_gap_sec": 0.0,
            "overlap_sec": 0.0,
        }

    gap_sec = 0.0
    largest_gap = 0.0
    overlap_sec = 0.0
    gap_count = 0
    prev_end = 0.0

    for seg in sorted(segments, key=lambda s: _as_float(s.get("start"))):
        start = max(0.0, min(_as_float(seg.get("start")), actual_duration))
        end = max(start, min(_as_float(seg.get("end")), actual_duration))
        gap = start - prev_end
        if gap > 0.05:
            gap_count += 1
            gap_sec += gap
            largest_gap = max(largest_gap, gap)
        elif gap < -0.05:
            overlap_sec += abs(gap)
        prev_end = max(prev_end, end)

    tail_gap = actual_duration - prev_end
    if tail_gap > 0.05:
        gap_count += 1
        gap_sec += tail_gap
        largest_gap = max(largest_gap, tail_gap)

    covered = max(0.0, actual_duration - gap_sec)
    return {
        "duration": round(covered, 4),
        "ratio": round(covered / actual_duration, 6),
        "gap_count": gap_count,
        "gap_sec": round(gap_sec, 4),
        "largest_gap_sec": round(largest_gap, 4),
        "overlap_sec": round(overlap_sec, 4),
    }


# ══════════════════════════════════════════════════════════════
# 5. 메인 파이프라인
# ══════════════════════════════════════════════════════════════

def run_pipeline(video_path: str,
                 output_dir:    str   = None,
                 threshold:     float = 0.5,
                 min_duration:  float = 0.0,
                 weights_path:  str   = None,
                 backend:       str   = "autoshot",
                 ) -> dict:
    """
    Shot boundary detection + splitting pipeline.

    Parameters
    ----------
    weights_path : ckpt_0_200_0.pth 경로 (None → ./autoshot/ckpt_0_200_0.pth 자동 탐색)
    backend : "autoshot" | "ecr" | "psd-adaptive"
    """
    video_path = str(video_path)
    backend = (backend or "autoshot").strip().lower()
    engine_label = {"autoshot": "AutoShot", "ecr": "ECR",
                    "psd-adaptive": "PSD-Adaptive"}.get(backend, backend)

    if output_dir is None:
        stem = Path(video_path).stem
        output_dir = str(Path(video_path).parent / f"{stem}_autoshot_cuts")

    t0 = time.time()
    print(f"[{engine_label}] Detecting scenes: {Path(video_path).name}")
    fps_pre, dur_pre = _get_video_info(video_path)
    total_f = int(dur_pre * fps_pre)
    print(f"[{engine_label}] Video: {dur_pre:.1f}s  {fps_pre:.2f}fps  {total_f} frames", flush=True)
    print(f"           threshold={threshold}")

    if backend == "ecr":
        scenes, fps, duration = detect_scenes_ecr(video_path, threshold)
    elif backend == "psd-adaptive":
        scenes, fps, duration = detect_scenes_psd_adaptive(video_path, threshold)
    else:
        scenes, fps, duration = detect_scenes(video_path, threshold, weights_path)
    t_detect = time.time() - t0

    print(f"[{engine_label}] FPS: {fps:.4f}  Duration: {duration:.4f}s")
    print(f"[{engine_label}] Detected {len(scenes)} scenes  ({t_detect:.1f}s)")

    if len(scenes) == 0:
        print(f"[{engine_label}] No scenes detected. Try lowering threshold.")
        meta = {
            "engine":      engine_label,
            "video":       video_path,
            "fps":         round(fps, 4),
            "duration":    round(duration, 4),
            "threshold":   threshold,
            "detect_time": round(t_detect, 2),
            "split_time":  0,
            "total_time":  round(t_detect, 2),
            "n_scenes":    0,
            "n_segments":  0,
            "coverage":    _coverage_summary([], duration),
            "segments":    [],
        }
        os.makedirs(output_dir, exist_ok=True)
        _save_metadata(output_dir, meta, engine_label)
        return meta

    print(f"[{engine_label}] Splitting video...")
    t1 = time.time()
    segments = split_scenes(video_path, scenes, duration, output_dir, min_duration)
    split_stats = getattr(split_scenes, "last_stats", {})
    t_split  = time.time() - t1
    t_total  = time.time() - t0

    print(f"[{engine_label}] {len(segments)} segments  "
          f"split={t_split:.1f}s  total={t_total:.1f}s")

    ok = True
    for i in range(len(segments) - 1):
        gap = segments[i+1]["start"] - segments[i]["end"]
        if abs(gap) > 0.1:
            print(f"  [WARN] gap shot{i+1}~{i+2}: {gap:.3f}s")
            ok = False
    if ok:
        print(f"[{engine_label}] Gap check: OK")

    coverage = _coverage_summary(segments, duration)
    print(
        f"[{engine_label}] Coverage: {coverage['ratio'] * 100:.2f}% "
        f"(gap={coverage['gap_sec']:.3f}s, gaps={coverage['gap_count']})"
    )

    meta = {
        "engine":      engine_label,
        "video":       video_path,
        "fps":         round(fps, 4),
        "duration":    round(duration, 4),
        "threshold":   threshold,
        "min_duration": min_duration,
        "detect_time": round(t_detect, 2),
        "split_time":  round(t_split,  2),
        "total_time":  round(t_total,  2),
        "n_scenes":    len(scenes),
        "n_segments":  len(segments),
        "split_policy": split_stats,
        "coverage":    coverage,
        "segments":    segments,
    }

    _save_metadata(output_dir, meta, engine_label)
    return meta


def _save_metadata(output_dir: str, meta: dict, label: str = "AutoShot"):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "cuts_metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[{label}] Metadata -> {json_path}")


# ══════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shot Boundary Detection + Splitting")
    parser.add_argument("video",                type=str)
    parser.add_argument("-o", "--output",       type=str,   default=None)
    parser.add_argument("--threshold",          type=float, default=0.5)
    parser.add_argument("--min-duration",       type=float, default=0.0)
    parser.add_argument("--weights",            type=str,   default=None,
                        help="ckpt_0_200_0.pth path (default: ./autoshot/ckpt_0_200_0.pth)")
    parser.add_argument("--backend",            type=str,   default="autoshot",
                        choices=["autoshot", "ecr", "psd-adaptive"],
                        help="Shot detection backend (default: autoshot)")
    args = parser.parse_args()

    meta = run_pipeline(
        video_path   = args.video,
        output_dir   = args.output,
        threshold    = args.threshold,
        min_duration = args.min_duration,
        weights_path = args.weights,
        backend      = args.backend,
    )

    print(f"\n{'='*50}")
    print(f"Engine    : {meta['engine']}")
    print(f"FPS       : {meta['fps']}")
    print(f"Duration  : {meta['duration']}s")
    print(f"Segments  : {meta['n_segments']}")
    print(f"Total     : {meta['total_time']}s")
    print(f"{'='*50}")
