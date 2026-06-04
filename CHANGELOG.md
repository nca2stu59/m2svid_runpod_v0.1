# Changelog

## RunPod bootstrap hardening — 2026-06-04

P0/P1 충돌 평가 후 적용. 리서치: `runpod/pytorch:*` stable = py3.11+torch2.8+cu128, xformers 0.0.33 Linux wheel 이 Blackwell sm_120 커널 포함 (release notes), RunPod Cloudflare proxy 100s cap 공식 확인.

- `requirements-m2svid-runpod.txt`, `requirements-vda-runpod.txt`: **xformers==0.0.33 + triton** 추가 (이전 누락 → `ModuleNotFoundError: xformers` 즉시 fail).
- `scripts/blackwell_xformers_shim.py`: m2svid_service Windows 마이그레이션의 SDPA shim 이식. 3D + 4D layout 분기. 비-Blackwell GPU 에서 no-op.
- `scripts/runpod_prepare_env.sh`: 전면 재작성.
  - venv 생성 + torch 2.9.0+cu128 + project requirements
  - third_party 자동 clone (Hi3D-Official, Video-Depth-Anything, pytorch-msssim, AutoShot)
  - ckpts 자동 download (m2svid_weights 5GB, open_clip 4GB, vgg, VDA-S, VDA-L). AutoShot Baidu Pan 만 수동.
  - Blackwell shim auto-detect (`INSTALL_BLACKWELL_SHIM=auto`, `nvidia-smi compute_cap` 으로 12.x 판정 → 자동 배포)
  - 검증 단계 (torch + xformers import + GPU capability print + check_runpod_paths.py)
  - env flag: `INSTALL_FLASHDEPTH`, `INSTALL_DEPTHCRAFTER`, `SKIP_CKPTS`, `SKIP_THIRD_PARTY`
- `runpod_entrypoint.sh`: `GRADIO_AUTH` 가드. 미설정/placeholder 시 fail-fast (override: `ALLOW_NO_AUTH=1`).
- `Dockerfile`: 베이스 이미지 명시 — `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` (stable; venv 내 torch 2.9.0 으로 업그레이드).
- `RUNPOD.md`: 전면 재작성.
  - 100s proxy 우회: TCP exposure 권장 (Community Cloud Pod IP 변동 주의)
  - GPU 권장: H200 vs Pro 6000 vs A100 비교 ($500 크레딧 effective hr 계산)
  - Network Volume region-lock 명시
  - 설치 순서 단계 (volume → Pod → clone → prepare_env → 검증 → AutoShot 수동 → Template 저장)
  - prepare_env 모든 env flag 표
  - troubleshoot 표 (cutlassF crash, ModuleNotFoundError, 524 timeout, autoshot 부재, region mismatch)

호환성:
- 기존 Windows m2svid_service: 영향 없음 (Linux-only 변경)
- Pro 6000 / 5090 (sm_120): Blackwell shim 자동 활성. xformers 0.0.33 native Blackwell 도 있으면 shim 이 override 되지만 perf parity (cuDNN SDPA).
- H200 / A100 (sm_90/sm_80): shim 비활성, native xformers 사용.

## fps-normalize patch — 2026-05-09 (cross-cutting, 7 폴더 일괄 적용)

**입력 비정수 fps (29.97 / 23.976 / 59.94 등 NTSC drop-frame) 의 sync drift 방지.**

### 동기

`imageio.get_writer(fps=int(fps))` 등 파이프라인 곳곳에서 fps를 int cast하면서 1초당 0.97 frame 손실 → 3분 영상에서 ~174 frame drift. AutoShot 컷 분할 시간 round-off, ffmpeg concat의 timestamp 누적 drift 등이 동시 발생.

### ⚠ 동작 명확화 (자주 오해되는 부분)

**30 fps hardcode 강제 아님.** 입력 영상의 fps를 ffprobe로 검출 후 `math.ceil(input_fps)` 결과로 transcode. 입력별로 결과 다름:

- 23.976 fps 입력 → **24 fps 출력** (영화 표준)
- 29.97 fps 입력 → **30 fps 출력** (NTSC SD/HD)
- 47.952 fps 입력 → **48 fps 출력**
- 59.94 fps 입력 → **60 fps 출력** (NTSC HFR)
- 50 fps 입력 → **50 fps 출력** (PAL HFR, 이미 정수 → skip)
- 30 / 25 / 60 등 정수 fps 입력 → **skip** (no transcode, ffprobe 비용만)

### 변경 (모든 폴더 동일)

- `run_pipeline.py`에 `_normalize_fps(video, mode, output_dir, progress)` 헬퍼 추가
- `PipelineConfig.normalize_fps: str = "ceil"` 필드 (default = ceil = 정수 fps 강제)
- CLI: `--normalize-fps {off|ceil|round}` 추가 (Orchestration group)
- `run()` 진입점에서 `base_out` 생성 직후, AutoShot 호출 전에 호출
- 비정수 fps 검출 시 ffmpeg로 `output/<run>/_normalized_input/<stem>__norm{N}fps.mp4` 사전 transcode 후 그것을 후속 단계 입력으로 사용
- 진행 이벤트: `orchestrator:fps_normalize_{start,done,skip,failed}`

### 동작

| 입력 fps | mode=ceil (default) | mode=round | mode=off |
|---|---|---|---|
| 23.976 (24000/1001) | 24 (transcode) | 24 (transcode) | 23.976 그대로 |
| 29.97 (30000/1001) | 30 (transcode) | 30 (transcode) | 29.97 그대로 |
| 59.94 (60000/1001) | 60 (transcode) | 60 (transcode) | 59.94 그대로 |
| 30 / 25 / 60 (이미 정수) | skip (no-op) | skip (no-op) | 그대로 |

### 검증 (7 폴더 모두 통과)

- 23.976 fps 72 frames 3.003 s 입력 → 24/1 fps 72 frames 3.000 s 출력
- 30 fps 입력 → skip (no transcode)
- mode=off → 원본 path 그대로 반환

### 비용

- 1회 추가 ffmpeg transcode (libx264 CRF 17 preset fast + AAC 192k, +faststart)
- 1080p 3분 영상 기준 ~20-40 s 추가
- 정수 fps 입력은 비용 거의 0 (ffprobe만)
- 사용자 명시 fallback: `--normalize-fps off`

### 적용 폴더 (7개)

`v0.16`, `v0.16b`, `v0.16fp`, `v0.16m`, `v0.16ta`, `v0.16xf`, `v0.17`

---

## v0.17.0 — 2026-05-05 (운영 권장)

**v0.16m 검증 통과 → 정식 릴리스.** Step2 inpaint = M2SVid full-attention. Per-shot disparity 통합. 단일컷 7.06× / 9컷 E2E 2.51× 가속, 9/9 통과.

### 운영 권장 변경

- **이전**: v0.16b (StereoCrafter SVD UNet, ~37min for 9-cut benchmark)
- **현재**: v0.17 (M2SVid full-attention + per-shot disparity, ~15min for 9-cut benchmark)

### 측정된 성능

#### 단일 컷 (shot001, 1.17s, 35 frames @ 1080p)

| 항목 | v0.16b | v0.17 | 변화 |
|---|---|---|---|
| Step1 (depth+warp) | 200.4s | ~80s | -60% |
| Step2 (inpaint) | 948.5s | 87s | **-90.8% (10.9× 가속)** |
| **Total** | **1148.85s** | **162.78s** | **-85.8% (7.06× 가속)** |
| 출력 SBS PSNR (vs v0.16b) | (reference) | **39.43 dB** | ✅ 합리적 |
| L/R 비대칭 | (reference) | **6.62 dB** | ✅ stereo cue 안전 |
| Per-frame 안정성 | — | mean 40.02 dB, 평탄 | ✅ flicker 없음 |

#### 9컷 풀 E2E (아임비타 14.51s, 1080p)

| 항목 | v0.16 | v0.16b | v0.17 | 변화 (vs v0.16b) |
|---|---|---|---|---|
| Step1 total (9컷) | 779.1s | 437.4s | (depth+warp 통합) | — |
| Step2 total (9컷) | 1809.0s | 1823.0s | (inpaint+compose) | — |
| **Grand total** | 2588.2s (43m 8s) | **2260.6s (37m 40s)** | **901.17s (15m 1s)** | **-60.1% (2.51× 가속)** |
| Per-cut 평균 | — | 251.2s | **100.13s** | -60% |
| Per-cut 범위 | — | — | 79.2-116.0s | 안정 |
| 통과율 | 9/9 | 9/9 | **9/9** | ✅ |

테스트 환경: RTX 5090 (Blackwell sm_120), Windows 11. v0.17 처리 dim 512, 출력 dim 1080 (lanczos).

### v0.16b → v0.17 핵심 변화

| 단계 | v0.16b | v0.17 | 비고 |
|---|---|---|---|
| Depth | VDA-L (port venv, sdpa-patched) | **VDA-S (default), selectable** (VDA-S/L, FlashDepth-L/S, DepthCrafter) | m2svid_service venv 사용 |
| Warp | Pure-PyTorch softmax splat | **m2svid native forward warp** | M2SVid 학습 분포 일치 |
| Inpaint | StereoCrafter SVD UNet 8-step | **M2SVid full-attention** 25-frame chunk | upstream 주장 6× 실측 7×|
| Per-shot disparity | (간접 max_disp 매핑) | **closeup×0.5 / normal / wide×1.5** | shot_classes.json 자동 forward |
| 출력 규격 | 3840×1024 (고정) | **slider 처리 384..1024 + lanczos / RTX VSR upscale 0..2160** | source aspect 자동 보정 |
| Micro-cut (0.1-0.4s) | ❌ 실패 | **✅ 성공** | M2SVid chunk_size=25 padding |
| Per-cut VRAM=0 | ✅ | ✅ | OS-level subprocess exit |

### 신규 / 수정 파일

#### NEW
- `m2svid_worker.py` — top 오케스트레이터, JSONL 이벤트, per-shot disparity 자동 forward
- `local_engines/m2svid/m2svid_per_cut_runner.py` — per-cut 5단계 runner (preprocess → depth → warp → inpaint → SBS compose → upscale)
- `local_engines/m2svid/inpaint_core.py` (vendored, path-resolver 패치)
- `local_engines/m2svid/inpaint_worker.py` (vendored)
- `local_engines/m2svid/warping.py` (vendored)
- `local_engines/m2svid/m2svid/` (m2svid 패키지 통째 vendoring)
- `_compare_sbs.py` — SBS A/B 비교 도구

#### MODIFIED
- `run_pipeline.py` — PipelineConfig + dispatch GenStereo → M2SVid, shot_classes_json forward
- `app.py` — Gradio UI 재작성 (port 7864), depth dropdown, processing/output slider, RTX VSR toggle
- `run.bat` — port 7864
- `VERSION` → `0.17.0`

#### REMOVED (v0.16b 잔재)
- `genstereo_worker.py` (StereoCrafter SVD wrapper)
- `local_engines/depth_splat_local.py`, `vda_depth.py`, `softmax_splatting.py` (m2svid_service VDA wrapper 사용)

### Q1-Q4 결정 (B-3 deep integration)

- **Q1 depth**: selectable, default = VDA-S
- **Q2 warp**: m2svid native forward warp
- **Q3 venv**: m2svid_service 의 4개 venv (`.venv`, `.venv-vda`, `.venv-flashdepth`, `.venv-depthcrafter`) 재사용
- **Q4 해상도**: slider 384..1024 step 64 (default 512)
- **Upscaler**: lanczos (default) / rtx_vsr (nvidia-vfx wheel, RTX 5090 native, DLPack zero-copy)

### 사용 (CLI)

```bash
# 기본 (VDA-S, lanczos, 처리 512, 출력 처리 그대로)
run.bat --video INPUT.mp4 --out outputs

# 1080p output + RTX VSR + VDA-L
run.bat --video INPUT.mp4 --out outputs ^
    --depth-backend VDA-L --output-dim 1080 --upscaler rtx_vsr

# 모든 stage_log 이벤트 출력 (디버그)
run.bat --video INPUT.mp4 --out outputs --verbose
```

### 터미널 출력 정책

- **기본 동작**: 핵심 진행상황 + 모든 오류를 **stderr** 로 즉시 출력
  - 단계 시작/종료, per-cut 시간/크기, [ERR]/[FATAL] 표시
  - stdout 은 최종 결과 JSON (machine-friendly)
  - stdout 을 파일로 redirect 해도 terminal 에서 진행상황 + 오류 모두 볼 수 있음
- **`--verbose` / `-v`**: 모든 stage_log 이벤트 (depth/warp/inpaint 내부 라인 포함) 까지 출력
- **환경 변수**: `STEREO_VERBOSE=1` 도 동일 효과
- **오류**: 항상 `[ERR]` 또는 `[FATAL]` 태그로 강조. unhandled exception 은 main() 에서 캐치되어 깔끔 표시 (`--verbose` 시 traceback 포함)

### 사용 (Gradio UI)

```bash
run.bat        # http://127.0.0.1:7864
```

### 디버깅 메모 (재시도 시 함정)

1. **m2svid_service venv 모두 torch 2.9.0+cu128 + sm_120 으로 업그레이드 확인 필수** (May 4 검증, `.bak` 백업 존재)
2. **vendored inpaint_core.py 의 `M2SVID_SERVICE_ROOT` env / hardcoded default** — m2svid_service ckpts (5GB+4GB) 중복 회피
3. **runner 의 inpaint stage 에서 `os.chdir(m2svid_service)`** — m2svid 의 OmegaConf config 가 `ckpts/open_clip_pytorch_model.bin` 등 상대 경로 사용
4. **upscale 시 source aspect 보정**: 64-div 강제로 처리 단계에서 aspect 왜곡 (16:9 → 2:1) 발생 → 업스케일 시 source aspect 로 stretch 복원

### 명시적 DO-NOT (v0.16b 에서 상속)

SageAttention 활성화 절대 금지:
- [`ROADMAP_NOTES.md §F.1`](ROADMAP_NOTES.md)

---

## v0.16m.0-wip — 2026-05-04 (실험 분기, v0.17 의 staging)

**v0.16b 분기. Step2 inpaint 엔진을 StereoCrafter SVD UNet → M2SVid (Google) full-attention 으로 교체.**

### 동기

v0.16fp / v0.16xf / v0.16ta 3개 분기 모두 ❌ 운영 비채택 → "Step2 가속" long-term 후보였던 **M2SVid** 가 m2svid_service 형태로 로컬 셋업 완료 → 본격 통합 시도.

### 설계 (B-3 deep integration)

- `local_engines/m2svid/` 에 m2svid_service 코드 vendoring (inpaint_core.py, warping.py, m2svid/ 패키지)
- vendored `inpaint_core.py` path-resolver 패치: `M2SVID_SERVICE_ROOT` env / hardcoded default → ckpts(5+4 GB), configs/, third_party/ 는 m2svid_service 재사용 (중복 회피)
- `m2svid_worker.py` (port venv) — top 오케스트레이터 (cuts_metadata 분배)
- `m2svid_per_cut_runner.py` (m2svid `.venv`) — per-cut 5단계: preprocess → depth → warp → inpaint → SBS compose → upscale
- 매 cut 종료 시 OS-level subprocess exit → VRAM 100% 회수 (v0.16b 의 룰 유지)

**Q1 — depth backend selectable**: VDA-S (default) / VDA-L / FlashDepth-L / FlashDepth-S / FlashDepth / DepthCrafter

**Q2 — warp**: m2svid native `warping.py` (forward warp, repro+mask 별도 mp4 출력)

**Q3 — venv**: m2svid_service 의 4개 기존 venv 재사용 (`.venv`, `.venv-vda`, `.venv-flashdepth`, `.venv-depthcrafter`)

**Q4 — 해상도 슬라이더**:
- processing-dim: 384..1024 step 64 (default 512, M2SVid 학습 해상도)
- output-dim: 0..2160 step 64 (default 0 = 처리 해상도 그대로)
- 64-divisible 강제

**Upscaler**:
- `lanczos` (default, ffmpeg)
- `rtx_vsr` (선택, `pip install nvidia-vfx`) — RTX 5090 Blackwell 지원, DLPack zero-copy interop, SBS L/R 분리 → eye-별 VSR → repack
- nvidia-vfx 미설치 시 자동 lanczos fallback

### 신규 파일

```
stereo_pipeline_v0.16m/
├── m2svid_worker.py                      # NEW (top orchestrator)
├── _smoke_m2svid_single_cut.py           # NEW (gradio_app.pipeline 호출 검증)
└── local_engines/
    └── m2svid/                           # NEW (vendored)
        ├── inpaint_core.py               (path 패치됨)
        ├── inpaint_worker.py
        ├── warping.py
        ├── m2svid_per_cut_runner.py      # NEW (per-cut 5-stage runner)
        └── m2svid/                       (m2svid_service 패키지 통째 vendoring)
```

### 수정된 파일

- `run_pipeline.py` — PipelineConfig GenStereo 필드 → M2SVid 필드. dispatch `genstereo_worker.py` → `m2svid_worker.py`. argparse 갱신.
- `app.py` — Gradio UI 재작성, port 7862 → 7864. M2SVid slider/dropdown.
- `run.bat` — port 7864.
- `VERSION` → `0.16m.0-wip`.

### 사용

```bash
# CLI 기본 (VDA-S, lanczos, 처리 512)
python run_pipeline.py --video INPUT.mp4 --out outputs

# CLI VDA-L + 출력 1080p + RTX VSR
python run_pipeline.py --video INPUT.mp4 --out outputs \
    --depth-backend VDA-L --output-dim 1080 --upscaler rtx_vsr

# Gradio UI
run.bat        # http://127.0.0.1:7864
```

### 검증 상태 (TODO)

- [x] 폴더 클론 + vendoring
- [x] inpaint_core.py path 패치 + import smoke test
- [x] m2svid_per_cut_runner.py / m2svid_worker.py 작성 + --help 통과
- [x] run_pipeline.py / app.py / run.bat M2SVid 어댑터
- [x] RTX VSR 조사 (nvidia-vfx wheel, RTX 5090 지원, DLPack OK)
- [ ] **단일 컷 prototype 실행 (GPU 필요 — 가용 시 대기)**
- [ ] shot001 A/B vs v0.16b (PSNR/SSIM/L-R/per-frame)
- [ ] 9-cut mini E2E (15s 1080p 아임비타)
- [ ] 결과 채택/비채택 결정 → 본 CHANGELOG 갱신

### 환경 메모 (May 4 20:52-21:04 시점 확인)

m2svid_service 의 4개 venv 모두 **torch 2.9.0+cu128 + sm_120** 으로 업그레이드 완료, `.bak` 백업 존재. 추가:
- `.venv-vda`: xformers 0.0.33 + triton 3.5.1 (GPU smoke OK, NaN 없음)
- `.venv-flashdepth`: xformers 0.0.33 + triton 3.5.1 + flash_attn 2.8.3 + causal_conv1d 1.5.0 (mamba_ssm MISSING — FlashDepth 실제 사용 시 설치 필요)

### 명시적 DO-NOT (v0.16b 에서 상속)

SageAttention 활성화 절대 금지:
- [`ROADMAP_NOTES.md §F.1`](ROADMAP_NOTES.md)

---

## v0.16ta.0 — 2026-05-04 (실험 분기) — ❌ ROI 음수, stereo cue 파괴

(자세한 내용: [`../stereo_pipeline_v0.16ta/CHANGELOG.md`](../stereo_pipeline_v0.16ta/CHANGELOG.md))

## v0.16fp.0 — 2026-05-04 (실험 분기) — ❌ ROI 음수, compile 14.6min/proc

(자세한 내용: [`../stereo_pipeline_v0.16fp/CHANGELOG.md`](../stereo_pipeline_v0.16fp/CHANGELOG.md))

## v0.16xf.0 — 2026-05-03 (실험 분기) — ❌ wheel CUDA extensions 미빌드

(자세한 내용: [`../stereo_pipeline_v0.16xf/CHANGELOG.md`](../stereo_pipeline_v0.16xf/CHANGELOG.md))

## v0.16b.0 — 2026-05-02

**v0.16 위에 v0.17 풀 패키지(A+B+C+D) + E 모니터링 적용.**

`port/stereo_pipeline_v0.16/IMPROVEMENT_ROADMAP.md` 의 권장 사항을 모두 구현.

### Phase A — Shot Classifier: CLIP-ViT-B/32 → SigLIP-2 base

- 새 default backend: `siglip2` (`google/siglip2-base-patch16-256`)
- 기존 backend (`clip`, `depth`) 도 유지 (backward compat)
- 효과: 정확도 +10-15pt (ImageNet zero-shot 63% → 78%), 모델 크기 600MB → 370MB,
  한국어 / 다언어 prompt 지원 (109 langs)
- 변경 파일: `local_engines/shot_classifier/classifier.py` (forked + modified)
- worker (`shotclass_worker.py`) 가 local 모듈 우선 import

### Phase B — Forward-Warp: Custom CUDA → Pure-PyTorch softmax splatting

- 새 모듈: `local_engines/softmax_splatting.py` (Niklaus & Liu 알고리즘)
- 효과: CUDA toolkit 의존 제거, .pyd 재빌드 불필요, 새 PC 셋업 단순화
- API: `ForwardWarpStereoSoftmax(eps, occlu_map)` — GenStereo `ForwardWarpStereo` drop-in
- 검증: 3/3 unit tests PASS (zero disp identity, shift correctness, autograd)

### Phase C — Depth: DepthCrafter → Video-Depth-Anything Large

- 새 wrapper: `local_engines/vda_depth.py`
- 가중치: `C:\Users\PC\Desktop\port\S3D_Pipeline\GenStereoBackend\dependency\Video-Depth-Anything\video_depth_anything_vitl.pth` (1.5 GB)
- repo: `C:\Users\PC\Desktop\port\S3D_Pipeline\GenStereoBackend\dependency\Video-Depth-Anything\repo` (cloned from GitHub)
- 효과: depth **30× 빠름** (DepthCrafter ~50s → VDA-L 5.6s for 34 frames)
- 시간 일관성 (TAE on ScanNet): 0.639 → 0.570 (더 좋음)
- VRAM peak: 26 GB → **9.92 GB**
- 라이선스: 동일 CC-BY-NC-4.0

### Phase D — Step1 통합: depth_splat_local.py

- GenStereo 의 `depth_splatting_inference.py` 를 우리 local 스크립트로 교체
- VDA depth → softmax splat → splat.mp4 (GenStereo Step2 가 그대로 사용)
- worker (`genstereo_worker.py`) Step1 호출 라인이 local 스크립트 경유

### Phase E — ROADMAP_NOTES.md (외부 의존 모니터링)

- M2SVid 출시 알림 (Q3 2026 예상, 6× speedup, drop-in)
- DA3-Streaming (24GB GPU 지원), Elastic3D / StereoPilot / StereoWorld
- xformers Blackwell 지원 (현재 sdpa fallback)
- SigLIP-2 → DINOv3 (라벨 데이터 확보 시)

### 측정된 성능 (v0.16 vs v0.16b, ★ 9컷 풀 E2E 실측 ★)

| 단계 | v0.16 (실측) | v0.16b (실측) | 변화 |
|---|---|---|---|
| Step1 total (9컷) | 779.1s | **437.4s** | **-43.9%** ✓ |
| Step2 total (9컷) | 1809.0s | 1823.0s | +0.8% (변동 없음) |
| **Grand total** | **2588.2s (43m 8s)** | **2260.6s (37m 40s)** | **-12.7%** ✓ |
| Per-cut depth | ~50s (DepthCrafter) | **5.6s** (VDA-L sdpa) | -89% |
| Per-cut splat | ~16s (CUDA Forward-Warp) | 15.2s (pure-PyTorch) | -5% |
| 9/9 통과 | ✓ | **✓** | (n_fail=0) |

### 적용된 GenStereo 패치 (v0.16 + 추가)

기존 v0.16 패치 3종 (utils.py NaN guard, inpainting Windows path, anaglyph 비활성)
+ v0.16b 신규 2종 (Blackwell xformers fallback):
- `Video-Depth-Anything/repo/video_depth_anything/dinov2_layers/attention.py`
- `Video-Depth-Anything/repo/video_depth_anything/motion_module/attention.py`

모두 `*.bak.20260502_v016b` 백업 보관.

### 디렉토리 구조

```
stereo_pipeline_v0.16b/
├── (v0.16 의 모든 .py 그대로 + 일부 수정)
│   ├── run_pipeline.py        ← VDA-L args 추가
│   ├── genstereo_worker.py    ← Step1 → local depth_splat_local.py
│   └── shotclass_worker.py    ← local classifier 우선 import
├── local_engines/             ← v0.16b 신규
│   ├── softmax_splatting.py   (Phase B)
│   ├── vda_depth.py           (Phase C)
│   ├── depth_splat_local.py   (Phase D)
│   └── shot_classifier/
│       └── classifier.py      (Phase A — SigLIP-2 추가)
├── ROADMAP_NOTES.md           (Phase E)
├── CHANGELOG.md               (이 파일)
└── VERSION                    "0.16b.0"
```

### 추가 의존성 (설치 필요)

SC venv (`port/S3D_Pipeline/m2svid/venv/`):
```
pip install einops imageio imageio-ffmpeg easydict xformers \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

GenStereo `Video-Depth-Anything/repo/.../attention.py` 두 곳에 v0.16b sdpa patch 적용 필요.

### 알려진 한계

- splat 단계가 pure-PyTorch 라 CUDA 보다 약간 느림 (446 ms/frame at 1080p) —
  현재 step1 의 ~30%, 전체 시간의 ~5% → 병목 아님
- M2SVid 출시 전엔 Step2 (inpaint) 가 여전히 전체 시간의 ~80%
- xformers 0.0.36+ 가 Blackwell 지원하면 sdpa patch 되돌릴 수 있음

---

## v0.16.0 — 2026-05-02 (이전 릴리스)

**원점 재구현 (fresh simple)**: v0.15의 manifest 인프라 + per-stage worker를 모두 제거.
"AutoShot/Classifier + GenStereo" 두 개 블록으로만 구성.

(v0.16 변경사항은 ../stereo_pipeline_v0.16/CHANGELOG.md 참조)
