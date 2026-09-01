# syntax=docker/dockerfile:1
# OVRTX uses the host NVIDIA driver. Run this image only on an RTX-capable
# RunPod GPU whose driver satisfies the OVRTX release requirements.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    XDG_CACHE_HOME=/workspace/.cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        libvulkan1 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /usr/local/bin/uv

WORKDIR /opt/palatial/newton
COPY . .
RUN uv sync --frozen --no-dev --extra sim --extra ovrtx

ENV PATH=/opt/palatial/newton/.venv/bin:$PATH
ENTRYPOINT ["/opt/palatial/newton/deploy/runpod/entrypoint.sh"]
CMD ["sleep", "infinity"]
