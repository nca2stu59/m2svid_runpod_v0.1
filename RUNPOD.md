# m2svid_runpod_v0.1 — RunPod setup

Target: RunPod Pod (not Serverless — m2svid jobs are 1-2 hr, persistent worker fits).

## Layout

```text
/workspace/m2svid_runpod_v0.1      app code
/workspace/m2svid_service          M2SVid runtime, venvs, ckpts, third_party
/workspace/outputs/m2svid_runpod_v0.1
```

Files materialized under `/workspace/m2svid_service` by `prepare_env.sh`:

```text
ckpts/m2svid_weights.pt              (~5 GB, GCS)
ckpts/open_clip_pytorch_model.bin    (~4 GB, HF)
ckpts/vgg.pth                        (LPIPS)
ckpts/autoshot.pth                   (Baidu Pan, MANUAL)
configs/m2svid.yaml                  (copy from upstream m2svid_service repo)
third_party/Hi3D-Official            (auto-clone)
third_party/Video-Depth-Anything     (auto-clone + VDA-S/L ckpts)
third_party/pytorch-msssim           (auto-clone)
third_party/AutoShot                 (auto-clone)
.venv                                (sgm + xformers 0.0.33 + triton)
.venv-vda                            (DINOv2 + xformers 0.0.33)
```

## Pod template choice

| Item | Value |
|---|---|
| Base image | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` (stable). `prepare_env.sh` upgrades venv torch → 2.9.0+cu128. |
| Container disk | 50 GB (code + venvs ≈ 15 GB) |
| Volume | 50 GB persistent (current baseline; ckpts/cache/outputs). Use 100 GB+ for repeated max-res tests. |
| GPU | H200 141GB recommended ($4.39/hr Community). RTX Pro 6000 Blackwell 96GB ($2.09/hr) also supported — Blackwell shim auto-deploys. |
| HTTP ports | 7864/http |
| TCP ports | 7864/tcp (RECOMMENDED — avoids 100s Cloudflare cap) |

## One-time setup

1. Create RunPod Network Volume in the region where you want to launch GPU Pods (volumes are region-locked).
2. Launch Pod with the template above, mount volume at `/workspace`.
3. SSH or web terminal into Pod, then:

```bash
cd /workspace
git clone <this-repo-url> m2svid_runpod_v0.1
cd m2svid_runpod_v0.1
bash scripts/runpod_prepare_env.sh        # ~15-25 min: venvs + 14 GB ckpts + git clones
```

4. Verify:

```bash
M2SVID_SERVICE_ROOT=/workspace/m2svid_service \
  /workspace/m2svid_service/.venv/bin/python scripts/check_runpod_paths.py
# expect: all OK
```

5. **AutoShot ckpt** — manual download from Baidu Pan (Chinese phone/QR auth):
   `https://pan.baidu.com/s/1CdCVNzFdF3U6I4ajfejYNQ` (passcode `sfkq`).
   Save file as `/workspace/m2svid_service/ckpts/autoshot.pth`.
   Without it, shot detection is disabled but pipeline still runs (single-shot mode).

6. Save Pod as a custom template via RunPod dashboard. Future Pod launches skip steps 3-5 (volume + image persist).

## Launch Gradio

```bash
cd /workspace/m2svid_runpod_v0.1
export GRADIO_AUTH="user:strong-password"      # mandatory unless ALLOW_NO_AUTH=1
export M2SVID_SERVICE_ROOT=/workspace/m2svid_service
export M2SVID_OUTPUT_ROOT=/workspace/outputs/m2svid_runpod_v0.1
bash scripts/runpod_h200_preflight.sh
bash scripts/runpod_h200_launch_gradio.sh
```

Do not run Gradio before `runpod_h200_preflight.sh` prints `PREFLIGHT_OK`.

## Reaching Gradio — TCP vs HTTP proxy

RunPod's HTTP proxy goes through Cloudflare which caps responses at **100 seconds**. M2SVid jobs run 1-2 hours. The Gradio queue keeps the WebSocket alive in most cases, but unstable bursts hit 524 errors.

### Option A — TCP exposure (recommended)

In RunPod Pod edit dialog → "Expose TCP Ports" → add 7864. RunPod assigns a public IP + external port:

```text
tcp://<public-ip>:<external-port>      # e.g. tcp://213.173.109.39:23107
```

Connect from browser as `http://<public-ip>:<external-port>`. No Cloudflare in path → no timeout. Caveat: Community Cloud Pod IP can change on restart/migration.

### Option B — HTTP proxy + Gradio queue

```text
https://<POD_ID>-7864.proxy.runpod.net
```

Works for short jobs. For long jobs ensure Gradio `.queue(default_concurrency_limit=1)` is set and progress bar pushes updates ≥ once / 60s. If unstable, switch to A.

## Cost reference

| Component | $/month idle | $/job 1080p 30s video |
|---|---|---|
| 50 GB Network Volume | current baseline | included |
| Pod H200 boot (~90s) | — | ~$0.11 |
| Pod H200 1 hr inference | — | ~$4.39 |
| Pod Pro 6000 1.5 hr inference (≈ H200 1 hr work) | — | ~$3.15 |

$500 credit ≈ 113 hr H200 or 239 hr Pro 6000 of compute. H200 wins on $/effective-hour (~2.5× faster) for SVD inpaint.

## Build Docker image (optional)

```bash
docker build -f Dockerfile.h200-safe -t <dockerhub-user>/m2svid-runpod:h200-safe .
docker build -f Dockerfile.pro6000-safe -t <dockerhub-user>/m2svid-runpod:pro6000-safe .
docker build -t <dockerhub-user>/m2svid-runpod:v0.1 .
docker push <dockerhub-user>/m2svid-runpod:v0.1
```

Use `Dockerfile.h200-safe` for the first H200 validation pass. It skips the
flash-attn source build and pins the runtime profile to sm90/H200-safe defaults.
Use `Dockerfile.pro6000-safe` for the first RTX PRO 6000 validation pass. It
skips the flash-attn source build, pins sm120 defaults, and forces the
Blackwell xformers shim path.

### Pro6000 launch variant

```bash
cd /workspace/m2svid_runpod_v0.1
PROFILE=runpod_profiles/pro6000-safe.env bash scripts/runpod_prepare_env.sh
bash scripts/runpod_pro6000_preflight.sh
export GRADIO_AUTH="user:strong-password"
bash scripts/runpod_pro6000_launch_gradio.sh
```

In RunPod template editor:

```text
Image: <dockerhub-user>/m2svid-runpod:v0.1
Container disk: 50 GB
Volume mount: /workspace (your network volume)
HTTP ports: 7864/http
TCP ports: 7864/tcp
Container command: /opt/m2svid_runpod_v0.1/runpod_entrypoint.sh
Env vars:
  GRADIO_AUTH=user:your-password
  M2SVID_SERVICE_ROOT=/workspace/m2svid_service
```

Models stay on the network volume — never bake into image.

## prepare_env.sh env flags

| Flag | Default | Effect |
|---|---|---|
| `M2SVID_SERVICE_ROOT` | `/workspace/m2svid_service` | service root |
| `PYTHON_BIN` | `python3` | base interpreter (3.11 on stable image) |
| `TORCH_VERSION` | `2.9.0` | torch wheel pin |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu128` | wheel index |
| `INSTALL_BLACKWELL_SHIM` | `auto` | `auto` = nvidia-smi cap=12.x check. `1` force, `0` skip. |
| `INSTALL_FLASHDEPTH` | `0` | FlashDepth venv (mamba2 dep, Linux build expected) |
| `INSTALL_DEPTHCRAFTER` | `0` | DepthCrafter venv |
| `SKIP_CKPTS` | `0` | skip downloads if volume already populated |
| `SKIP_THIRD_PARTY` | `0` | skip git clones |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `RuntimeError: cutlassF: no kernel found to launch!` on Pro 6000 / 5090 | `INSTALL_BLACKWELL_SHIM=1 bash scripts/runpod_prepare_env.sh` (re-deploys shim). |
| `ModuleNotFoundError: xformers` | Re-run prepare_env.sh; check that `requirements-m2svid-runpod.txt` has `xformers==0.0.33`. |
| 524 timeout via proxy | Switch to TCP exposure (Option A above). |
| `autoshot.pth missing` warning | Manual Baidu Pan download. Pipeline still runs in single-shot fallback. |
| Volume region mismatch with desired GPU | Volume is region-locked. Either reprovision volume in target region, or pick GPU in current volume region. |
