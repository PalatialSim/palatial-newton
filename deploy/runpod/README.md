# Newton + OVRTX on RunPod

This image is an interactive GPU pod workbench. It combines Newton's CUDA
simulation with OVRTX's RTX USD rendering; it is not a serverless endpoint.
The pod model lets shader caches and output recordings survive in a network
volume while an engineer iterates over SSH.

## Build and publish

Run from the repository root, substituting your approved immutable registry
tag. Do not use `latest`.

```bash
docker build --platform=linux/amd64 \
  -f deploy/runpod/Dockerfile \
  -t ghcr.io/palatialsim/palatial-newton-ovrtx:<tag> .
docker push ghcr.io/palatialsim/palatial-newton-ovrtx:<tag>
```

The image pins the Python OVRTX and ovstage wheels in `pyproject.toml` and
installs from the committed `uv.lock`. It places transient application files in
`/opt/palatial/newton`; mount a RunPod network volume at `/workspace` for output
and `XDG_CACHE_HOME`, which includes OVRTX's shader cache. The image requests
the NVIDIA `graphics` driver capability required for headless Vulkan and ships
FFmpeg for direct video encoding.

## Create a guarded pod

Before creation, authenticate `runpodctl`, check current GPU availability, and
choose a data center that can host both the volume and the GPU. OVRTX requires
an RTX-capable GPU and a compatible NVIDIA driver; an RTX 4090 is an appropriate
initial development target, but validate the actual pod driver before reporting
success.

```bash
runpodctl user
runpodctl gpu list --include-unavailable
runpodctl datacenter list
runpodctl ssh list-keys

runpodctl pod create \
  --name palatial-newton-ovrtx-dev \
  --image ghcr.io/palatialsim/palatial-newton-ovrtx:<immutable-tag> \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --ports "22/tcp" \
  --env '{"NVIDIA_DRIVER_CAPABILITIES":"compute,utility,graphics,video"}' \
  --network-volume-id <volume-id> \
  --volume-mount-path /workspace \
  --terminate-after <ISO-8601-deadline>
```

`--terminate-after` is deliberate: a stopped pod still leaves disk and network
volume storage billable. The volume must be selected at creation and is pinned
to its data center.

## Validate the real renderer

After the pod reports running, use `runpodctl ssh info <pod-id>` and execute the
reported SSH command non-interactively. The first OVRTX frame can spend one or
two minutes compiling shaders, so retain `/workspace/.cache` between tests.

First validate NVIDIA's own minimal example. An import proves only that Python
can find the wheels; this command must produce the upstream PNG on the exact
GPU and driver selected for the pod.

```bash
git clone --depth 1 https://github.com/nvidia-omniverse/ovrtx.git /workspace/ovrtx-upstream
cd /workspace/ovrtx-upstream/examples/python/minimal
uv run main.py --png
test -s _output/render.png
```

Then validate the Newton bridge. The render target is inferred from its path;
there is no separate output mode. An image path captures the last authored
frame, a video path renders the complete timeline, and a suffix-free path
receives numbered PNG frames.

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
mkdir -p /workspace/newton-ovrtx
cd /workspace/newton-ovrtx
python -m newton.examples basic_shapes --device cuda:0 --viewer ovrtx \
  --num-frames 120 --output-path simulation.usd --ovrtx-output-path render.png
python -m newton.examples basic_shapes --device cuda:0 --viewer ovrtx \
  --num-frames 120 --output-path simulation-video.usd --ovrtx-output-path render.mp4
python - <<'PY'
from pathlib import Path
from PIL import Image

image_path = Path("render.png")
with Image.open(image_path) as image:
    print({"path": str(image_path.resolve()), "size": image.size, "mode": image.mode})
PY
```

## Stage scripting

Pass `--ovrtx-script render_scene.py` to customize the same live stage session.
The script can define `compose_stage(source_path, default_stage)`,
`on_stage_open(context)`, `before_frame(context)`,
`after_frame(context, pixels)`, `on_render_complete(context, output_path)`, and
`on_stage_close(context)`. All hooks are optional. Per-frame stage mutations
must reserve and publish an OVStage ordinal through `context.mutation()`:

```python
def before_frame(context):
    with context.mutation() as ordinal:
        # Author through context.ovstage using context.stage and this ordinal.
        pass


def after_frame(context, pixels):
    # Return None to keep the frame, or a uint8 HxWx4 array to replace it.
    return pixels
```

Programmatic callers can use `newton.ovrtx.OVRTXStage` as a context manager and
call `render_frame()` repeatedly. The renderer attaches once and is reused for
the entire timeline; video frames stream to FFmpeg over a raw RGBA pipe instead
of being round-tripped through temporary PNG files.

Compare the reported driver with NVIDIA's [OVRTX driver requirements](https://github.com/nvidia-omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/docs/driver_requirements.rst)
before treating the bridge result as valid.

This is the acceptance check: a RunPod `Running` state or a successful image
pull does not prove that OVRTX initialized, selected the GPU, loaded the USD
recording, and emitted a readable PNG. Record the pod ID, image digest, GPU,
driver, command, USD path, and PNG check independently.
