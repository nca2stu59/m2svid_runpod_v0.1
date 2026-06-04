# v0.17 — Documentation Index

> **운영 권장 버전.** v0.16b 의 Step2 (StereoCrafter SVD UNet) → M2SVid full-attention 교체 + per-shot disparity.
> 9컷 풀 E2E 실측: **901.17s (15m 1s)** vs v0.16b 의 2260.6s (37m 40s) — **2.51× 가속**, 9/9 통과.
> 단일컷 (shot001): **162.78s** vs 1148.85s — **7.06× 가속**, PSNR 39.43 dB (vs v0.16b 출력 비교).

---

## 처음 오는 사람용 (순서대로)

| # | 문서 | 무엇 | 언제 읽나 |
|---|---|---|---|
| 1 | [README.md](README.md) | v0.16m 소개 + 사용법 + (TBD) 측정 결과 | 항상 (1분) |
| 2 | [CHANGELOG.md](CHANGELOG.md) | v0.16b → v0.16m 변경 + 4개 분기(xf/fp/ta/m) 흐름 | 변경/실험 흐름 (3분) |
| 3 | [PHASE_RESULTS.md](PHASE_RESULTS.md) | (역사적) v0.16b A+B+C+D+E 결과 | v0.16b 가 무엇이었는지 (5분) |

---

## 작업 / 결정용

| 문서 | 용도 |
|---|---|
| [BACKEND_EXTRACTION.md](BACKEND_EXTRACTION.md) | 다른 UI 프레임워크 추가 시 참조 (NiceGUI / Flet / CLI / REST) |
| [ROADMAP_NOTES.md](ROADMAP_NOTES.md) | 외부 의존 모니터링 (M2SVid 출시, DA3-Streaming 등) |
| [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md) | v0.16 → v0.16b 의 의사결정 근거 (역사적, 참고용) |

---

## 운영 / 이식용

| 문서 | 용도 |
|---|---|
| [MIGRATION.md](MIGRATION.md) | 새 PC로 이식할 때 필요 사양 (v0.16 기준, v0.16b는 + VDA-L 1.5GB + xformers + einops 등) |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | v0.16 검증 베이스라인 (역사적, v0.16b는 PHASE_RESULTS.md 참조) |

---

## 핵심 사실 (이것만 알면 됨)

- **현재 빌드**: v0.17.0 (2026-05-05, **운영 권장**)
- **검증**: 단일컷 7.06× 가속, 9컷 E2E 2.51× 가속, 9/9 통과, PSNR 39.4 dB, L/R delta 6.6 dB
- **변경 (v0.16b → v0.17)**: Step2 = StereoCrafter SVD UNet → **M2SVid full-attention** + warp m2svid native + per-shot disparity (closeup×0.5/wide×1.5) + depth backend selectable + processing/output dim slider + RTX VSR 옵션 + micro-cut 처리 안정
- **출력 규격**: 처리 해상도 (default 512, 64-div) → lanczos / RTX VSR 업스케일 (source aspect 자동 보정)
- **외부 의존**: m2svid_service (`C:\Users\PC\Desktop\m2svid_service\`) — 4개 venv 모두 torch 2.9.0+cu128 sm_120 검증됨 + ckpts (m2svid_weights.pt 5GB + open_clip 4GB)
- **운영 권장 변경**: **v0.16b → v0.17** (이전 v0.16b 는 fallback 으로 보존)

## ⛔ 절대 시도 금지

- **SageAttention 활성화 (모든 변종)** — Blackwell sm_120 + SVD/DepthCrafter 에서 NaN 검증됨.
  자세한 증상 / 재시도 조건 / "first-run 함정": [`ROADMAP_NOTES.md §F.1`](ROADMAP_NOTES.md)
  추가 detail: [`C:\Users\PC\Desktop\port\S3D_Pipeline\GenStereoBackend\stereocraft-optimization.md`](../../GenAI/StereoCrafter/stereocraft-optimization.md)
- **flash-attn 무계획 설치 on Win+sm_120** — pre-built 없음, source build fragile, cuDNN SDPA 대비 perf gain 없음
- **`scaled_dot_product_attention` global monkey-patch** — state leak 위험

---

## 빠른 시작

```bat
cd C:\Users\PC\Desktop\port\S3D_Pipeline\stereo_pipeline_v0.17
run.bat                                     # Gradio UI (port 7864)
run.bat --video INPUT.mp4 --out outputs     # CLI (VDA-S, lanczos, 처리 512)

rem 다른 옵션 예시
run.bat --video INPUT.mp4 --out outputs ^
    --depth-backend VDA-L --output-dim 1080 --upscaler rtx_vsr
```

---

## 코드 진입점

| 무엇 | 파일 |
|---|---|
| Gradio UI (port 7864) | `app.py` |
| 오케스트레이터 (UI-agnostic) | `run_pipeline.py` |
| AutoShot worker | `autoshot_worker.py` |
| Shot Classifier worker (SigLIP-2) | `shotclass_worker.py` |
| **M2SVid orchestrator (per-shot disparity 자동)** | `m2svid_worker.py` |
| **per-cut runner (5 stage)** | `local_engines/m2svid/m2svid_per_cut_runner.py` |
| M2SVid inpaint core (path 패치) | `local_engines/m2svid/inpaint_core.py` |
| m2svid native warp | `local_engines/m2svid/warping.py` |
| m2svid 패키지 (vendored) | `local_engines/m2svid/m2svid/` |
| SigLIP-2 classifier | `local_engines/shot_classifier/classifier.py` |
| SBS A/B 비교 도구 | `_compare_sbs.py` |

---

## v0.17 옵션 요약

| 옵션 | 기본 | 범위 |
|---|---|---|
| `--depth-backend` | VDA-S | VDA-S/L, FlashDepth-L/S/(default), DepthCrafter |
| `--processing-dim` | 512 | 384..1024 step 64 |
| `--output-dim` | 0 | 0..2160 step 64 (0=처리 그대로) |
| `--upscaler` | lanczos | lanczos / rtx_vsr |
| `--rtx-vsr-quality` | 12 (HIGH) | 0..19 |
| `--disparity-perc` | 0.02 | 0.005..0.05 step 0.005 |

---

## 향후 작업 후보 (참고)

- ✅ ~~단일컷 prototype + 9컷 mini E2E~~ → 통과, **v0.17 승격 완료** (2026-05-05)
- ✅ ~~per-shot disparity~~ → 통합 완료, 자동 forward 동작 검증
- nvidia-vfx (RTX VSR) 정식 채용 검토 — 현재 옵션, 시각 비교 후 default 화 가능
- DA3-Streaming 으로 24GB 미만 GPU 지원 (Q2 2026 검토)
- 출력 영상 시각 검증 (직접 시청, anaglyph 합성 검토)
- M2SVid 업스트림 신규 release 흡수 (vendored 코드 갱신)
- 정식 installer (PyInstaller onedir + Inno Setup) — 배포용
