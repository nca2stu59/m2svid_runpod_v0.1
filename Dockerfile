# RunPod stable base: torch 2.8 + py3.11 + cu128 + cudnn-devel + ubuntu 22.04.
# prepare_env.sh upgrades torch to 2.9.0+cu128 inside the venv (host torch left alone).
# Override BASE_IMAGE for py3.12 dev image or custom.
ARG BASE_IMAGE=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# ─── Stage 1: flash-attn 2 source build (cu128 + torch 2.9 + cp311) ───
# Built once at image-build time, wheel copied to final stage.
# Default build target is H200/H100 (sm_90). Override args for Pro 6000:
#   --build-arg FA2_TORCH_CUDA_ARCH_LIST=12.0 --build-arg FA2_FLASH_ATTN_CUDA_ARCHS=120
# MAX_JOBS=2 to avoid OOM during nvcc concurrent compile (~80GB RAM usage at 4 jobs).
FROM ${BASE_IMAGE} AS fa2-builder

ARG FA2_TORCH_CUDA_ARCH_LIST=9.0
ARG FA2_FLASH_ATTN_CUDA_ARCHS=90

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    TORCH_CUDA_ARCH_LIST="${FA2_TORCH_CUDA_ARCH_LIST}" \
    FLASH_ATTN_CUDA_ARCHS="${FA2_FLASH_ATTN_CUDA_ARCHS}" \
    MAX_JOBS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Build venv with torch 2.9.0+cu128 (matches runtime exactly).
RUN python3 -m venv /opt/fa2-venv \
    && /opt/fa2-venv/bin/python -m pip install --upgrade pip wheel setuptools \
    && /opt/fa2-venv/bin/python -m pip install \
       torch==2.9.0 torchvision==0.24.0 \
       --index-url https://download.pytorch.org/whl/cu128 \
    && /opt/fa2-venv/bin/python -m pip install ninja packaging psutil

# Clone + build FA2.
ARG FLASH_ATTN_TAG=v2.7.4.post1
RUN git clone --depth=1 --branch ${FLASH_ATTN_TAG} \
        https://github.com/Dao-AILab/flash-attention /opt/flash-attention

# Source build with NO --no-build-isolation conflict. Output to /opt/wheels/.
RUN mkdir -p /opt/wheels \
    && cd /opt/flash-attention \
    && /opt/fa2-venv/bin/python setup.py bdist_wheel \
    && cp dist/flash_attn-*.whl /opt/wheels/ \
    && ls -lh /opt/wheels/

# ─── Stage 2: final image ─────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    M2SVID_SERVICE_ROOT=/workspace/m2svid_service \
    M2SVID_OUTPUT_ROOT=/workspace/outputs/m2svid_runpod_v0.1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7864 \
    PORT=7864 \
    GRADIO_CONCURRENCY=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    XDG_CACHE_HOME=/workspace/.cache \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# apt deps baked into image (Pod restart container disk wipe defense).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg curl ca-certificates \
    tmux rsync nano less \
    build-essential cmake ninja-build \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Bring FA2 wheel from builder stage. prepare_env.sh installs it into m2svid venv.
COPY --from=fa2-builder /opt/wheels /opt/wheels

WORKDIR /opt/m2svid_runpod_v0.1
COPY . /opt/m2svid_runpod_v0.1

RUN chmod +x /opt/m2svid_runpod_v0.1/runpod_entrypoint.sh \
    /opt/m2svid_runpod_v0.1/scripts/*.sh

EXPOSE 7864
CMD ["/opt/m2svid_runpod_v0.1/runpod_entrypoint.sh"]
