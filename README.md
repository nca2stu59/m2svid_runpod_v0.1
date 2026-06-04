# m2svid_runpod_v0.1 — RunPod M2SVid Gradio fork

RunPod setup lives in [RUNPOD.md](RUNPOD.md). This fork keeps the v0.17 M2SVid
pipeline but changes runtime defaults toward `/workspace/m2svid_service`,
Linux venv paths, and Gradio port exposure.

# Stereo Pipeline v0.17 — M2SVid + per-shot disparity (source baseline)

> **Version: v0.17.0** · 2026-05-05 · **운영 권장**.
> 부모: v0.16m.0-wip (실험 분기, 2026-05-04) — v0.16b 위에 M2SVid Step2 교체 + per-shot disparity 검증.
> 자매 분기 (모두 ❌ 운영 비채택, 실험 기록): [v0.16fp](../stereo_pipeline_v0.16fp/) (FP8), [v0.16xf](../stereo_pipeline_v0.16xf/) (xformers), [v0.16ta](../stereo_pipeline_v0.16ta/) (TAESDV)
> 변경 내역 + 실측 결과: [CHANGELOG.md](CHANGELOG.md)

입력 영상 → **AutoShot** → **Shot Classifier (SigLIP-2)** → **VDA depth + warp + M2SVid inpaint + per-shot disparity** → 최종 SBS (lanczos 또는 RTX VSR upscale).

## v0.16b → v0.17 핵심 변화

| 단계 | v0.16b | v0.17 | 비고 |
|---|---|---|---|
| Depth | VDA-L (port venv) | **VDA-S (default), selectable**: VDA-S/L, FlashDepth-L/S, DepthCrafter | m2svid_service venv 사용 |
| Warp | Pure-PyTorch softmax splat | **m2svid native forward warp** | M2SVid 학습 분포 일치 |
| Inpaint | StereoCrafter SVD UNet 8-step | **M2SVid full-attention** 25-frame chunk | **6.85× 가속 실측** |
| Per-shot disparity | (간접 max_disp 매핑) | **closeup×0.5 / normal / wide×1.5** scaling | shot_classes.json 자동 forward |
| 출력 규격 | 3840×1024 (고정) | **slider 처리 384..1024 + lanczos / RTX VSR upscale 0..2160** | source aspect 자동 보정 |
| Per-cut VRAM=0 | ✅ | ✅ | OS-level subprocess exit |
| Micro-cut (0.1-0.4s) 처리 | ❌ 실패 (n_fail) | **✅ 성공** | M2SVid chunk_size=25 패딩 |

## 측정된 성능 (★ 9컷 풀 E2E + shot001 단일 컷 실측 ★)

### 단일 컷 (shot001, 1.17s, 35 frames @ 1080p)

| 항목 | v0.16b | v0.17 | 변화 |
|---|---|---|---|
| Step1 (depth+warp) | 200.4s | ~80s | -60% |
| Step2 (inpaint) | 948.5s | 87s | **-90.8% (10.9× 가속)** |
| **Total** | **1148.85s** | **162.78s** | **-85.8% (7.06× 가속)** |
| 출력 SBS PSNR (vs v0.16b) | (reference) | 39.43 dB | ✅ 합리적 일치 |
| L/R 비대칭 | (reference) | 6.62 dB | ✅ stereo cue 안전 |

### 9컷 풀 E2E (아임비타 14.51s, 1080p)

| 항목 | v0.16 | v0.16b | v0.17 | 변화 (vs v0.16b) |
|---|---|---|---|---|
| Step1 total (9컷) | 779.1s | 437.4s | (depth+warp 통합) | — |
| Step2 total (9컷) | 1809.0s | 1823.0s | (inpaint+compose) | — |
| **Grand total** | 2588.2s (43m 8s) | **2260.6s (37m 40s)** | **901.17s (15m 1s)** | **-60.1% (2.51× 가속)** |
| Per-cut 평균 | — | 251.2s | **100.13s** | -60% |
| Per-cut 범위 | — | — | 79.2-116.0s | 안정 |
| 통과율 | 9/9 | 9/9 | **9/9** | ✅ |

테스트 환경: RTX 5090 (Blackwell sm_120), Windows 11, 처리 dim 512, 출력 dim 1080 (lanczos).

## 설계 원칙

> **"하나의 컷 처리가 끝나면 반드시 VRAM을 반환"** — OS-level subprocess exit 으로 100% 보장.

각 컷마다:
1. `m2svid_worker.py` (top 오케스트레이터) 가 `m2svid_per_cut_runner.py` 를 m2svid_service `.venv` 로 spawn
2. runner 내부 5단계: **preprocess → depth subprocess (.venv-vda) → warp → inpaint → SBS compose → upscale (lanczos / RTX VSR)**
3. runner 종료 시 OS-level VRAM 회수 (PCIe spill 방지)

## 흐름

```
input.mp4
    │
    ▼
[AutoShot subprocess] ─────────────────► cuts_metadata.json + shot###.mp4
    │
    ▼
[Shot Classifier subprocess (SigLIP-2)] ─► shot_classes.json (closeup/normal/wide)
    │
    ▼
[m2svid_worker.py] (per-shot disparity 자동 forward)
    │   for each cut:
    │     [m2svid_per_cut_runner.py] (m2svid_service .venv, single subprocess per cut)
    │       1. preprocess: ffmpeg lanczos resize → src_resized.mp4 (64-div)
    │       2. depth subprocess (.venv-vda or .venv-flashdepth)
    │       3. warp (in-process, m2svid native forward warp)
    │       4. inpaint (in-process, M2SVid generate_shot_tensor, 25-frame chunks)
    │       5. compose SBS (left=src_resized | right=inpainted)
    │       6. upscale (lanczos via ffmpeg / rtx_vsr via nvidia-vfx, source aspect 보정)
    │     → VRAM 0 (subprocess exit)
    ▼
[ffmpeg concat] ──────────────────────────► final_sbs.mp4
```

## 구성

```
stereo_pipeline_v0.17/
├── run_pipeline.py              # 오케스트레이터 (CLI + Gradio 공용)
├── autoshot_worker.py           # subprocess A: AutoShot
├── shotclass_worker.py          # subprocess C: SigLIP-2
├── m2svid_worker.py             # subprocess M: per-cut M2SVid 분배 + per-shot disparity
├── concat_ffmpeg.py             # ffmpeg concat
├── app.py                       # Gradio UI (포트 7864)
├── run.bat                      # Windows 런처
├── VERSION                      # 0.17.0
├── CHANGELOG.md
├── README.md                    # 이 파일
├── _compare_sbs.py              # SBS A/B 비교 도구
└── local_engines/
    ├── m2svid/                  # M2SVid vendored
    │   ├── inpaint_core.py            (path 패치 — m2svid_service ckpts/configs/third_party 참조)
    │   ├── inpaint_worker.py          (m2svid_service 원본)
    │   ├── warping.py                 (m2svid_service 원본)
    │   ├── m2svid_per_cut_runner.py   (per-cut 5단계 runner)
    │   └── m2svid/                    (m2svid_service m2svid 패키지 통째)
    └── shot_classifier/         # SigLIP-2 분류기 (v0.16b 그대로)
```

## 사전 요구사항

| 컴포넌트 | 위치 |
|---|---|
| **m2svid_service root** | `C:\Users\PC\Desktop\m2svid_service\` |
| m2svid weights (5 GB) | `m2svid_service\ckpts\m2svid_weights.pt` |
| open_clip weights (4 GB) | `m2svid_service\ckpts\open_clip_pytorch_model.bin` |
| m2svid `.venv` (sgm + pytorch_lightning + torch 2.9 cu128) | `m2svid_service\.venv\Scripts\python.exe` |
| m2svid `.venv-vda` (xformers + triton, Blackwell verified) | `m2svid_service\.venv-vda\Scripts\python.exe` |
| (선택) RTX VSR | `pip install nvidia-vfx` (m2svid_service `.venv` 또는 port venv) |

→ m2svid_service 의 4개 venv 가 모두 torch 2.9.0+cu128 + sm_120 으로 업그레이드되어 있어야 함 (May 4 검증, `.bak` 백업 존재).

## 실행

### Gradio UI
```bat
run.bat
```
→ http://127.0.0.1:7864

### CLI
```bat
rem 기본 (VDA-S, lanczos, 처리 512, 출력 처리해상도)
run.bat --video INPUT.mp4 --out outputs

rem 1080p output + RTX VSR + VDA-L
run.bat --video INPUT.mp4 --out outputs ^
    --depth-backend VDA-L --output-dim 1080 --upscaler rtx_vsr

rem 전체 옵션
run.bat --help
```

## 옵션 요약

| 옵션 | 기본 | 범위 | 설명 |
|---|---|---|---|
| `--depth-backend` | VDA-S | VDA-S/L, FlashDepth-L/S/(default), DepthCrafter | depth 엔진 선택 |
| `--processing-dim` | 512 | 384..1024 step 64 | M2SVid 학습 해상도 = 512 |
| `--output-dim` | 0 | 0..2160 step 64 | 0 = 처리 그대로, >0 = 업스케일 (per-eye height) |
| `--upscaler` | lanczos | lanczos / rtx_vsr | rtx_vsr 은 nvidia-vfx wheel 필요 |
| `--rtx-vsr-quality` | 12 (HIGH) | 0..19 | BICUBIC..ULTRA / DENOISE / DEBLUR / HIGHBITRATE |
| `--disparity-perc` | 0.02 | 0.005..0.05 | warp 강도. shot_classes 있으면 컷별 자동 스케일 |
| `--seed` | 42 | — | M2SVid inpaint 재현성 |
| `--mask-antialias` | 0 | 0/1 | mask resize antialias |

## 향후 (ROADMAP_NOTES.md 참조)

- nvidia-vfx (RTX VSR) 정식 채용 검토 (현재 옵션, 검증 후 default 변경 가능)
- DA3-Streaming 으로 24GB 미만 GPU 지원 (Q2 2026)
- Backend 패키지 분리 + NiceGUI / Flet / CLI adapter (BACKEND_EXTRACTION.md)
- M2SVid 업스트림 신규 release 흡수 (vendored 코드 갱신)
