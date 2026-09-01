<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# OVRTX + Palatial Newton on Runpod: integration findings

**Status:** research findings. Implementation status is tracked in the
repository change set; sources were checked 2026-09-01 against OVRTX `0.4.1`
([source revision](https://github.com/NVIDIA-Omniverse/ovrtx/tree/9240b8200f044c11e8998442f927b50942dc2c06)).

## Decision

The branch implementation intentionally takes a narrower first slice than the
upstream `ViewerRTX`: `--viewer ovrtx` records Newton's existing time-sampled
USD, composes an OVRTX camera and `RenderProduct` over that immutable
recording, and renders the final authored USD timecode. It is a generic USD
playback path, not a claim of a tested per-frame source-prim adapter for
Palatial articulated assets or deforming cables.

An OVRTX rendering option is feasible. Current upstream Newton has already
chosen the public contract `--viewer rtx` and implements `ViewerRTX`; this
Palatial branch does **not** contain that viewer. Upstream's implementation
builds its own USD scene from Newton viewer calls, updates rigid transforms and
deforming-mesh vertices, supports headless screenshot capture, and uses
OVRTX's standalone `open_usd`/attribute APIs. [upstream CLI](https://github.com/newton-physics/newton/blob/000de3f1ff965122e3307b776c5d38a1b7c9661c/newton/examples/__init__.py#L643-L673), [upstream viewer](https://github.com/newton-physics/newton/blob/000de3f1ff965122e3307b776c5d38a1b7c9661c/newton/_src/viewer/viewer_rtx.py), [headless capture](https://github.com/newton-physics/newton/blob/000de3f1ff965122e3307b776c5d38a1b7c9661c/newton/_src/viewer/viewer_rtx.py#L2040-L2065)

This is a useful reference, not a safe one-file backport: the local branch
lacks `viewer_rtx.py`, `viewer_gui.py`, `utils.py`, and the upstream layer
infrastructure, while its `ViewerBase`, `ViewerUSD`, and example CLI have
substantial upstream divergence. Also, NVIDIA OVRTX 0.4 marks that
renderer-owned scene path as deprecated. New Palatial-specific code should use
the attached `ovstage` flow: Newton publishes a renderable USD scene and
per-frame state at a new ordinal; OVRTX renders the named `RenderProduct` and
returns pixels. [OVRTX 0.4 direction](https://github.com/NVIDIA-Omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/README.md), [attached-stage contract](https://github.com/NVIDIA-Omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/docs/core/ovstage_integration.rst)

For the first usable branch slice, expose the upstream-compatible spelling
`--viewer rtx` (with `--renderer ovrtx` only as an optional Palatial alias),
and scope it to **rigid,
source-mapped Palatial USD** and make unsupported cloth/cable geometry updates
fail clearly. The existing textured articulated replay already maps
`model.body_label` to source-USD prim paths and writes sampled rigid transforms;
that is the most concrete local evidence for the required mapping. A direct
OVRTX adapter can use the same map and write `omni:xform` to an `ovstage` stage
each frame. Deformable cloth/cables require a separate, tested points/topology
update path; a rigid transform adapter would produce misleading renders.

## Runtime and container gate

| Area | Evidence-backed requirement | Consequence for Runpod |
| --- | --- | --- |
| GPU | OVRTX runtime validation requires an **NVIDIA RTX-capable GPU**. Its Linux baseline for Ada/Ampere/Turing data-center GPUs is driver `570.158.01`; supported newer branches are listed by NVIDIA. | Do not select a pod merely because it exposes CUDA. Preflight the exact pod GPU and host driver against OVRTX's table. |
| Newton | This branch requires Python `>=3.10`; its docs require an NVIDIA GPU and driver `545+` for CUDA 12. | Use a single Python version in the overlap (the existing Palatial Runpod runtime has used Python 3.11); lock both dependency sets together rather than allowing ordinary-pip drift. |
| OVRTX Python | OVRTX/ovstage publish wheels for Python `>=3.10,<3.14`; NVIDIA's minimal project pins `ovrtx==0.4.1.364340` and `ovstage==0.1.1.355824`. | Pin an exact compatible OVRTX + ovstage pair in the image lock. Do not add them as a core Newton dependency. |
| Linux graphics | On Linux, OVRTX uses Vulkan. Its dynamic runtime expects its package `bin/` layout, including `cache`, `library`, `libs`, `mdl`, `plugins`, `rendering-data`, and `usd_plugins`, to remain together. | The image needs graphics/Vulkan-capable NVIDIA runtime exposure and the wheel/runtime layout intact; a CUDA-only image assertion is insufficient. |
| Headless workers | NVIDIA documents an EGL crash risk when a Linux no-display process repeatedly destroys/recreates renderers. It recommends `RendererConfig(keep_system_alive=True)` for that lifecycle; `VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING=1` is a fallback. | Keep one attached renderer alive per worker/job. Make the configuration explicit and test worker reuse, rather than creating a renderer per frame/request. |
| Warmup | The first application step can take 1–2 minutes for shader compilation. In real-time path tracing, textures and accumulation need warmup; NVIDIA gives 40 frames as a conservative default. | Separate startup/warmup time from simulation throughput and cache the shader directory in the image/volume only after a measured runtime trial. |

Sources: [OVRTX driver requirements](https://github.com/NVIDIA-Omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/docs/driver_requirements.rst), [Newton installation requirements](../guide/installation.rst), [OVRTX minimal project lock](https://github.com/NVIDIA-Omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/examples/python/minimal/pyproject.toml), [OVRTX renderer configuration](https://github.com/NVIDIA-Omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/docs/core/renderer_configuration.rst), [OVRTX warmup guidance](https://github.com/NVIDIA-Omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/skills/warmup/SKILL.md).

## Proposed CLI and integration seam

Keep the current GL/null behavior compatible and add the upstream-compatible
selection to the shared example parser where feasible, or a focused Palatial
entry point where the existing custom parser requires it:

```text
--viewer {null,gl,rtx}
--ovrtx-output <png path>
--ovrtx-render-product /Render/Camera
--ovrtx-warmup-frames 40
```

`--viewer rtx` should validate at startup that `ovrtx`, `ovstage`, a
compatible GPU/driver, a source USD with a stable body-path map, and the chosen
RenderProduct are present. If the input scene lacks an OVRTX camera/render
configuration, compose an in-memory USDA root with the input as a sublayer and
author a Camera, RenderProduct, and `LdrColor` RenderVar. This leaves the
converter input immutable. OVRTX documents that `step()` receives
RenderProduct paths, not Camera paths. [inline sublayer + render product](https://github.com/NVIDIA-Omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/tests/docs/usd/data/inline_sublayers_camera_renderproduct.usda), [render lifecycle](https://github.com/NVIDIA-Omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/examples/python/minimal/main.py)

OVRTX itself is a Python/C application library, not a generic `--renderer`
executable. Select its visual mode in the USD RenderProduct via
`omni:rtx:rendermode`: `RealTimePathTracing` (default, warm up before capture),
`PathTracing` (reference-quality single-step convergence), or `Minimal`
(throughput-oriented rasterization). Keep this distinct from Newton's CLI
backend selection. [OVRTX render modes](https://github.com/NVIDIA-Omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/docs/sensors/cameras/render_modes.rst), [PathTracing behavior](https://github.com/NVIDIA-Omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/docs/sensors/cameras/render_modes/path_tracing.rst)

The per-frame contract should be:

1. Newton advances `state_0`.
2. The adapter converts the mapped rigid `body_q` values to an `(N, 4, 4)`
   `float64` USD-row-vector transform batch and writes `omni:xform` at a new,
   monotonically increasing ovstage ordinal.
3. It waits for `advance_write_floor(ordinal, Scope.ALL)`, then invokes
   `renderer.step({render_product}, delta_time, ordinal=ordinal)`.
4. It maps `LdrColor` to CPU only when a PNG is requested, otherwise use the
   CUDA/DLPack output path. Mappings must be released before the next frame.

The matrix convention is material: OVRTX stores translation in row `3`, columns
`0..2`, and needs `float64`, not a Newton/Warp float32 transform array. OVRTX's
Python `step()` performs the attached-stage update, wait, and result fetch after
the caller has published its ovstage ordinal. [transform write contract](https://github.com/NVIDIA-Omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/tests/docs/python/test_attribute_shapes.py), [attached update loop](https://github.com/NVIDIA-Omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/tests/docs/python/test_stage_mutation.py), [render-output mapping](https://github.com/NVIDIA-Omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/examples/python/minimal/main.py)

The existing local touchpoints are:

| Path | Why it matters |
| --- | --- |
| `newton/examples/palatial/example_palatial_load.py` | Owns the current `--gui` / headless viewer selection and frame loop. |
| `newton/examples/palatial/example_palatial_articulated.py` | Already records USD and implements the rigid `model.body_label` to source-prim transform mapping for textured playback. |
| `newton/_src/palatial/load.py` | `NewtonBundle` exposes the original USD path, model, state, solver, and `fps` needed by the adapter. |
| `newton/viewer.py` | Public viewer export surface; do not import `newton._src` from an external caller. A general `ViewerOVRTX` is a later public-API decision, not required for the first Palatial CLI slice. |
| `pyproject.toml` | Core package intentionally has no OVRTX dependency; preserve that boundary and install OVRTX only in the Runpod runtime profile. |

Upstream calls the backend `ViewerRTX`, not `ViewerOVRTX`, and packages it in a
separate `rtx` optional dependency group. Retain that naming if the goal is
eventual upstream alignment, but pin the version pair in the Runpod profile:
upstream only declares `ovrtx>=0.3`, whereas OVRTX 0.4's standalone API is a
deprecated compatibility surface. [upstream dependency group](https://github.com/newton-physics/newton/blob/000de3f1ff965122e3307b776c5d38a1b7c9661c/pyproject.toml#L91-L96), [OVRTX standalone deprecation](https://github.com/NVIDIA-Omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/python/ovrtx/_src/renderer.py#L502-L552)

## Validation order

1. **Static:** `--help` exposes `rtx`; absent optional packages and unsupported
   asset modes fail with actionable errors; existing GL/null tests remain green.
2. **Image:** record Python/OVRTX/ovstage/Warp versions and the immutable image
   digest; verify the OVRTX package's runtime directories are present.
3. **OVRTX runtime:** on the exact Runpod pod, run NVIDIA's minimal Python
   example with `--png` before Newton integration. Success is its
   `_output/render.png`, not an import.
4. **Bridge runtime:** run one named rigid Palatial USD with `--viewer rtx`;
   retain the source USD, output PNG, renderer log, GPU/driver information, and
   image digest. Confirm dimensions, non-empty RGBA output, and body motion
   across two distinct simulation frames.
5. **Acceptance:** compare a known camera view against a reviewed baseline and
   separately report simulation success, OVRTX rendering success, and artifact
   publication. A Newton simulation completing or an OVRTX PNG existing alone
   does not prove the bridge is correct.

No OVRTX runtime check was attempted here: the local research host is not an
approved unsandboxed, RTX/driver-verified Runpod runtime. NVIDIA requires those
conditions for runtime validation. [OVRTX Python prerequisites](https://github.com/NVIDIA-Omniverse/ovrtx/blob/9240b8200f044c11e8998442f927b50942dc2c06/examples/python/minimal/README.md)
