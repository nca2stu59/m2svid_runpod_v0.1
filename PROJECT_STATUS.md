# Stereo Pipeline v0.16 — PROJECT_STATUS

> **Status: GREEN ✓** · 9/9 cuts end-to-end verified · 43m 42s for 15s 1080p input
> Last updated: 2026-05-02
>
> 이 문서는 Claude가 v0.16 작업을 이어받을 때 즉시 컨텍스트를 잡을 수 있도록 작성됨.
> 새 세션에서 이 파일을 가장 먼저 읽으면 충분.

---

## TL;DR (30초 안에 파악)

- **무엇**: 2D 영상 → AutoShot 컷 분할 → Shot Classifier (max_disp 자동) → **GenStereo CLI wrapping** (per-cut 2-step subprocess) → SBS concat
- **왜 이 구조**: "한 파일 처리 후 VRAM 100% 회수"를 OS-level subprocess exit으로 보장
- **검증**: 아임비타 광고 영상 (15s, 9컷, 1080p, 30fps) → 9/9 성공
- **외부 의존**: `C:\Users\PC\Desktop\port\S3D_Pipeline\GenStereoBackend` (GenStereo project + portable Python)
- **다음에 할 수 있는 일**: throughput 개선, 다른 영상으로 robustness 검증, UI 개선

---

## 1. 검증된 아키텍처 (재현 가능 ✓)

### 데이터 흐름

```
input.mp4
   ↓
[AutoShot subprocess]  ← system Python (transnetv2_pytorch 필요)
   ↓ cuts/cuts_metadata.json + cuts/{stem}_shot###.mp4
   ↓ (process exit → VRAM 0)
[Shot Classifier subprocess]  ← StereoCrafter venv (clip backend)
   ↓ shot_classes/shot_classes.json
   ↓ (process exit → VRAM 0)
[GenStereo wrapper subprocess]  ← StereoCrafter venv (orchestrator만)
   ├─ for each cut:
   │   ├─ Step1: depth_splatting_inference.py  ← GenStereo portable Python
   │   │         ↓ _genstereo_tmp/shot###_splatting_results.mp4
   │   │         ↓ (process exit → VRAM 0)
   │   └─ Step2: inpainting_inference.py        ← GenStereo portable Python
   │             ↓ sbs/shot###_sbs.mp4
   │             ↓ (process exit → VRAM 0)
   ↓
[ffmpeg concat] → final_sbs.mp4 (stream copy)
```

### 핵심 원칙 (이거 깨지 마세요)

1. **각 stage = 별도 subprocess.** in-process로 통합하면 VRAM 누수 / PCIe spill 위험.
2. **컷마다 GenStereo Step1, Step2도 별도 subprocess로 spawn.** 모델 로드 비용은 지불해도 OS-level VRAM 회수가 가치 있음.
3. **manifest 인프라 사용 금지.** `cuts_metadata.json` + `shot_classes.json` 두 단일 파일로 충분 (v0.15에서 manifest 시도 → 복잡도만 늘고 이득 없음).
4. **stderr → stdout merge** (`stderr=subprocess.STDOUT`). Windows pipe 버퍼(~64KB) deadlock 회피.
5. **자식에 UTF-8 강제**: `PYTHONIOENCODING=utf-8`, `PYTHONUTF8=1`, `PYTHONUNBUFFERED=1` + `sys.stdout.reconfigure(encoding="utf-8")`. Korean 경로 안전.

### 파일 인벤토리 (v0.16 디렉토리)

| 파일 | 역할 | 변경 빈도 |
|---|---|---|
| `run_pipeline.py` | 오케스트레이터 (CLI + Gradio 공용 `run()`) | 중간 |
| `genstereo_worker.py` | per-cut 2-step GenStereo CLI wrapper | 낮음 (잘 작동) |
| `autoshot_worker.py` | AutoShot subprocess wrapper | 안정 (v0.13s 그대로) |
| `shotclass_worker.py` | Shot Classifier subprocess wrapper | 안정 (v0.13s 그대로) |
| `concat_ffmpeg.py` | ffmpeg concat 헬퍼 | 안정 |
| `app.py` | Gradio single-tab UI (port 7862) | 중간 |
| `run.bat` | Windows 런처 (UI/CLI 분기, pause 포함) | 안정 |
| `VERSION` | "0.16.0" | 릴리스 시만 |
| `README.md` | 사용자용 문서 | 변경 사항 시 |
| `CHANGELOG.md` | v0.15 → v0.16 변환 기록 | 변경 사항 시 |

---

## 2. 외부 의존 — GenStereo Native CLI

### 경로 (모두 hardcoded default, override 가능)

```
C:\Users\PC\Desktop\port\S3D_Pipeline\GenStereoBackend\
├── depth_splatting_inference.py        ← Step1 entry
├── inpainting_inference.py             ← Step2 entry  [PATCHED, see §3]
├── python_embed/python.exe             ← 자체 Portable Python (diffusers 0.37, transformers 5.x)
├── weights/
│   ├── DepthCrafter/
│   ├── StereoCrafter/
│   └── stable-video-diffusion-img2vid-xt-1-1/
└── dependency/
    └── DepthCrafter/depthcrafter/utils.py  ← [PATCHED, see §3]
```

### CLI 호출 형식

**Step1 — depth+splat**:
```
python_embed\python.exe depth_splatting_inference.py \
  INPUT.mp4 OUTPUT_SPLAT.mp4 \
  weights/DepthCrafter weights/stable-video-diffusion-img2vid-xt-1-1 \
  --max_disp=N --process_length=-1 --batch_size=10
```

**Step2 — SVD inpaint + SBS**:
```
python_embed\python.exe inpainting_inference.py \
  weights/stable-video-diffusion-img2vid-xt-1-1 \
  weights/StereoCrafter \
  INPUT_SPLAT.mp4 SAVE_DIR \
  --frames_chunk=23 --overlap=3 --tile_num=2
```

출력 파일명: `{splat_stem.replace("_splatting_results", "")}_inpainting_results_sbs.mp4`

---

## 3. 적용된 패치 (반드시 유지)

세 GenStereo 상류 파일에 패치 적용. 모두 백업 (`*.bak.20260502_v016`) 옆에 보관.

### 3.1. `dependency/DepthCrafter/depthcrafter/utils.py` — `ColorMapper.apply`

**증상**: 균일 depth 컷 (로고/블랭크 프레임)에서 `IndexError: index -9223372036854775808`

**원인**: `v_max == v_min` → 0/0 = NaN → `(NaN * 255).long()` = INT64_MIN → 256-크기 colormap 인덱싱 실패

**수정**:
```python
denom = v_max - v_min
denom = denom.clamp(min=1e-8) if isinstance(denom, torch.Tensor) else max(float(denom), 1e-8)
image = (image - v_min) / denom
image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
image = (image * 255).long().clamp(0, self.colormap.shape[0] - 1)
image = self.colormap[image]
```

### 3.2. `inpainting_inference.py` L188 — Windows path

**증상**: SBS 파일이 우리 지정 `SAVE_DIR`이 아니라 입력 splat과 같은 디렉토리에 저장됨

**원인**: `input_video_path.split("/")[-1]` — Windows 백슬래시 경로에서 `/`가 없어 전체 경로 반환 → `os.path.join(save_dir, full_path)` = 두 번째 인자가 절대 경로면 첫 번째 무시

**수정**:
```python
video_name = os.path.basename(input_video_path).replace(".mp4", "").replace("_splatting_results", "") + "_inpainting_results"
```

### 3.3. `inpainting_inference.py` L290-299 — anaglyph 비활성

**이유**: 사용자 명시적 요청. 컷당 ~10-30s 인코딩 + ~3-10MB 디스크 낭비. 파이프라인은 SBS만 필요.

**수정**: anaglyph 생성 블록 전체 주석 처리 (필요 시 주석 해제).

---

## 4. 검증된 성능 베이스라인

### 입력
- 영상: `아임비타 에너지가 필요한 모든 순간, 아임비타 이뮨샷.mp4` (Korean filename)
- 길이: 15.0s @ 29.97fps (450 frames)
- 해상도: 1920×1080 → GenStereo 내부에서 1024×576로 다운스케일

### 결과 (Total: 43m 42s)

| Stage | 시간 | 비고 |
|---|---|---|
| AutoShot | 13s | 9 컷 분할 |
| Shot Classifier | 12s | clip backend, 9컷 분류 |
| GenStereo (9컷) | 2588s (43m 8s) | 핵심 부하 |
| Concat | <1s | stream copy 성공 |

### Per-Cut

| Shot | Class | maxD | Frames | Step1 | Step2 | Total | SBS Size |
|---|---|---|---|---|---|---|---|
| 1 | normal | 20 | 34 | 66s | 147s | **3:33** | 3.3MB |
| 2 | normal | 20 | 50 | 83s | 221s | **5:04** | 6.9MB |
| 3 | normal | 20 | 72 | 130s | 292s | **7:02** | 8.4MB |
| 4 | wide | 30 | 86 | 139s | 363s | **8:22** | 8.4MB |
| 5 | wide | 30 | 38 | 71s | 150s | **3:41** | 3.6MB |
| 6 | normal | 20 | 10 | 39s | 50s | **1:29** | 0.5MB |
| 7 | normal | 20 | 29 | 60s | 149s | **3:29** | 1.5MB |
| 8 | normal | 20 | 74 | 125s | 288s | **6:53** | 5.4MB |
| 9 | normal | 20 | 34 | 66s | 149s | **3:35** | 5.4MB |

### Throughput
- 평균: ~6.06s/frame
- GenStereo native baseline (단일 80-frame 컷): 4.81s/frame
- **Wrapper overhead: ~26%** (per-cut 모델 로드 2회 × 9컷 = ~3분)
- 입력 15s → 출력 43m 42s = 174× sub-realtime

---

## 5. 사용법 (재현)

```bat
cd C:\Users\PC\Desktop\port\S3D_Pipeline\stereo_pipeline_v0.16

run.bat                                              # UI (default, port 7862)
run.bat ui                                           # UI (명시)
run.bat --video INPUT.mp4 --out outputs              # CLI
run.bat cli --video INPUT.mp4 --out outputs          # CLI (legacy 키워드)
run.bat help                                         # 옵션 보기
```

전체 옵션:
```bat
run.bat --video INPUT.mp4 --out outputs ^
  --threshold 0.296 --min-duration 0.0 ^
  --use-shotclass --shotclass-backend clip ^
  --max-disp-wide 30 --max-disp-normal 20 --max-disp-closeup 12 ^
  --max-disp-fallback 20 ^
  --tile-num 2 --frames-chunk 23 --overlap 3 ^
  --splat-batch-size 10 ^
  --concat --fail-fast
```

---

## 6. ⛔ DO NOT (학습된 함정)

다음은 시도했고 실패했거나, 검증된 안티패턴. 다시 시도하지 마세요.

### 6.1. 아키텍처 함정

| ❌ 시도 금지 | 이유 (한 줄) |
|---|---|
| StereoCrafter `app.py` `run_pipeline()`을 in-process로 통합 호출 | 1080p에서 PCIe spill (54+ GB reserved on 32GB VRAM, 30-50× slowdown) |
| Per-stage worker 분리 (depth/splat/inpaint 각각 subprocess) — v0.13s 패턴 | 작동은 하지만 GenStereo native 6m25s/컷 baseline을 못 따라감 |
| Manifest 인프라 (workspace.json + 컷×stage JSON) — v0.15 패턴 | 추가 복잡도만 발생, 단일 `cuts_metadata.json` + `shot_classes.json`로 충분 |
| P-1 / P-2 통합 stereo 모드 (v0.15) | 1080p에서 둘 다 PCIe spill, 14-28분 stuck |
| GenStereo 모델 캐시해서 컷마다 재사용 (in-process) | "VRAM 100% 회수" 규칙 깨짐. subprocess exit이 가장 단순 +확실 |

### 6.2. 코드 패턴 함정

| ❌ 시도 금지 | 대신 사용 |
|---|---|
| `subprocess.Popen(stderr=PIPE)` 자식 stdout과 분리 | `stderr=subprocess.STDOUT` (Windows 64KB 버퍼 deadlock 회피) |
| 자식에 `PYTHONIOENCODING` 설정 안 함 | 항상 `env["PYTHONIOENCODING"]="utf-8"` + `PYTHONUTF8=1` (Korean 경로) |
| `bufsize=-1` (block buffered) | `bufsize=1` + `PYTHONUNBUFFERED=1` (라인 단위 forward) |
| `shift` + `%*` (Windows .bat) | `shift` 후 `%*`는 변하지 않음. 수동 ARGS 누적 루프 사용 |
| .bat에 `pause` 없이 `goto :eof` | 더블클릭 시 창 깜빡 사라짐. 모든 종료 경로에 `pause >nul` |
| GenStereo `os.path.join(save_dir, abs_path_string)` 신뢰 | Windows에서 두 번째가 절대 경로면 첫 인자 무시. 패치 §3.2 적용 |
| GenStereo `vis_sequence_depth(constant_array)` | NaN 크래시. 패치 §3.1 적용 |

### 6.3. 옵션 함정

| ❌ 시도 금지 | 이유 |
|---|---|
| `tile_num=4` at 1080p | latent (44→22→11→5.5) 비정수 분할 → "Expected 12 got 11" shape error |
| `tile_num=1` at 1080p | OOM 위험. 검증된 default = `2` |
| `cpu_offload="model"` (DepthCrafter) | 32GB VRAM에서 1080p+ 입력 시 PCIe spill 유발 |
| `frames_chunk > 23` at 1080p | SVD temporal attention O(T²), VRAM 폭증 |
| StereoCrafter venv 무시하고 system Python 사용 | torch CUDA 12.8 + diffusers 0.29.2 의존성. system은 보통 다름 |
| Shot Classifier `--no-shotclass` 기본화 | wide 컷에 max_disp=20 적용되면 disparity 부족, 평면 느낌 |

### 6.4. 디버깅 함정

| ❌ 시도 금지 | 대신 사용 |
|---|---|
| `find /v ""` Windows에서 stdin 처리 | sandbox-blocked. PowerShell `cmd /c "..." 2>&1 \| Select -First 25` |
| GenStereo upstream 파일 백업 없이 수정 | 항상 `*.bak.20260502_v016` 백업 (백업 없이 destructive 수정 금지) |
| SC venv `app.py`의 SageAttention 코드 건드림 | 영구 제외 (항상 충돌). 사용자 룰. |

---

## 7. 알려진 개선 가능 영역

다음은 작동은 하지만 더 개선할 수 있는 항목.

### 7.1. Throughput
- **현재**: ~6.06s/frame (wrapper overhead 26%)
- **목표**: native baseline ~4.81s/frame에 근접
- **가능한 접근**:
  - GenStereo의 model loading을 daemon 형태로 keep-alive (1 worker process가 여러 컷을 받음)
  - 단, "VRAM 100% 회수" 규칙과 트레이드오프 — 사용자 룰 우선

### 7.2. Robustness
- 검증된 영상: 1개 (아임비타 광고, 9컷, 15s)
- 미검증: 다른 해상도 (4K?), 더 긴 영상 (>30s), 더 많은 컷 (>20), 다양한 콘텐츠 타입
- **권장**: 5-10개 영상으로 확장 벤치마크

### 7.3. UI/UX
- 현재 UI는 단일 탭, 진행 로그 표시
- 컷별 progress bar 없음 (전체 frac만 추정)
- 결과 갤러리는 컷별 SBS 파일 리스트만
- **개선 가능**: 컷별 썸네일, max_disp 시각화, GPU memory 그래프

### 7.4. CLI/Gradio 동시 실행
- 현재: Gradio 7862만 점유
- StereoCrafter UI 7861, GenAI UI 7860, Shot Classifier UI 7863과 충돌 회피됨
- 동시 GenStereo CLI 사용 금지 (외부 GenStereo가 단일 인스턴스 가정 가능)

### 7.5. Concat fallback
- 현재: stream copy 성공 → fallback re-encode 미사용
- 다른 영상에서 stream copy 실패 시 libx264 CRF18 fallback이 작동하는지 미검증

---

## 8. 버전 진화 (참고)

| 버전 | 핵심 변화 | 결과 |
|---|---|---|
| v0.13s | StereoCrafter 3-stage subprocess (depth/splat/inpaint) | 작동하나 느림 |
| v0.21 | manifest 인프라 도입 (workspace.json + per-cut JSON) | 복잡도 ↑ |
| v0.15 | v0.21 fork + P-1/P-2 통합 모드 + GenStereo wrapper 탭 (G) | P-1/P-2 PCIe spill, G 탭만 작동 |
| **v0.16** | **원점 재구현. manifest 제거, GenStereo wrapper만, single-tab UI** | **9/9 검증 ✓** |

v0.2 (안정 라인)는 별도. 기능 개선 완료 후 main 채택 예정.

---

## 9. Claude 새 세션 진입 가이드

새 Claude 세션에서 v0.16 작업을 이어받을 때:

1. **이 파일 (`PROJECT_STATUS.md`) 먼저 읽기** — 가장 빠른 컨텍스트
2. 최근 변경 사항은 `CHANGELOG.md`
3. 사용자 룰:
   - Destructive ops (file delete, third-party file edit) 전 백업 필수
   - 설계 논의 → Q&A 1라운드만으로 코딩 들어가지 말 것 (명시적 go 신호 대기)
   - SageAttention 영구 제외 (StereoCrafter)
4. 출력 디렉토리: `outputs/{stem}_{timestamp}/` — gitignore 권장
5. 핵심 외부 의존: `C:\Users\PC\Desktop\port\S3D_Pipeline\GenStereoBackend` — 이 경로는 패치된 상태 (§3)
6. 새 영상 테스트는 `run.bat` 사용 (UI 또는 CLI)
7. 디버깅 시 우선 확인:
   - `outputs/.../logs/{autoshot,shotclass,genstereo}_stdout.jsonl` (모든 이벤트 기록)
   - `nvidia-smi -l 1` (별도 터미널, VRAM 회수 확인)
   - `outputs/.../sbs/_genstereo_tmp/` 가 비어있는지 (남아있으면 cleanup 실패 = 버그)
