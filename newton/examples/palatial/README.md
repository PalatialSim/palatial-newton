# Palatial examples

These examples drive Newton from a **converted USDA** produced by the palatial
converter. Each one reads the bundle via `newton.palatial.load(usd_path)` —
solver, fps, substeps, body type and solver params all come baked into the
USDA, so the same script runs for any asset of the matching body type.

All commands assume the `test_venv` interpreter set up in
`~/Documents/Research_work/test_venv` (replace with your own Python).

---

## 0. `generate_palatial_cable_usd` — author a cable asset

Generates a **first-kind** palatial cable asset: a high-level
`*.newton.usda` with `NewtonRodAPI`, `NewtonRodMaterialAPI`, a
`BasisCurves` centerline, an authored `UsdGeom.Mesh` surface, and baked solver
metadata. The default output is a flat rectangular cable so it exercises the
anisotropic rod path.

```bash
/home/achuthan_palatial/Documents/Research_work/test_venv/bin/python \
  -m newton.examples.palatial.generate_palatial_cable_usd \
  ~/power_cable.newton.usda \
  --cross-section-type flatRect \
  --length 1.5 --segment-count 16 \
  --drop-height 0.3 \
  --solver vbd --solver-substeps 2
```

Then run it with the cable example:

```bash
... -m newton.examples.palatial.example_palatial_cable \
  ~/power_cable.newton.usda \
  --device cuda:0 --gui --steps 1000
```

Or let the example generate its own temporary flatRect ribbon asset:

```bash
... -m newton.examples.palatial.example_palatial_cable \
  --device cuda:0 --gui --steps 1000
```

Named anisotropic presets are also available for quick VBDPalatial checks:

```bash
... -m newton.examples.palatial.generate_palatial_cable_usd \
  ~/flat_balanced_demo.newton.usda \
  --preset flat_balanced_demo

... -m newton.examples.palatial.generate_palatial_cable_usd \
  ~/flat_bend_z_dominant.newton.usda \
  --preset flat_bend_z_dominant

... -m newton.examples.palatial.generate_palatial_cable_usd \
  ~/flat_bend_y_dominant.newton.usda \
  --preset flat_bend_y_dominant

... -m newton.examples.palatial.generate_palatial_cable_usd \
  ~/round_low_torsion_demo.newton.usda \
  --preset round_low_torsion_demo
```

Good starting ranges for generated anisotropic cables:

- `segment_count`: `12` to `24`
- `stretch_stiffness`: `8e4` to `2e5` for demos, up to `1e6` for stiffer cables
- `bend_y_stiffness`, `bend_z_stiffness`: `4e2` to `3e3`
- `torsion_stiffness`: `8e1` to `4e2` for torsion-soft ribbons, up to `1e3`
- `width`: `8e-3` to `2e-2` m and `thickness`: `2.5e-3` to `6e-3` m for flat cables
- `radius`: `5e-3` to `6e-3` m for round demo cables
- `solver_substeps`: `4` to `6` when using `vbd_palatial`

---

## 1. `example_palatial_load` — generic settle / drop

Works for **cloth**, **rigid**, and **rod** bundles. Cloth-only knobs are
ignored on rigid/rod assets; joint-drive flags only apply to rigid
articulations.

```bash
/home/achuthan_palatial/Documents/Research_work/test_venv/bin/python \
  -m newton.examples.palatial.example_palatial_load \
  ~/new_jean.newton.usda \
  --device cuda:0 --substeps 2 \
  --cloth-particle-radius 0.005 \
  --soft-contact-ke 1e5 --soft-contact-kd 1 \
  --bending-ke 1e-2 \
  --drop-height 0.5 \
  --gui --steps 1000
```

Add `--record-mp4 /tmp/jean.mp4` to dump a front-facing camera view to disk.
Recording works headless (no `--gui`) or alongside `--gui`.

```bash
... example_palatial_load ~/new_jean.newton.usda \
  --device cuda:0 --substeps 2 \
  --record-mp4 /tmp/jean.mp4 --mp4-fps 60 --steps 1000
```

Select the first-class OVRTX backend on an RTX Linux host with the same video
target. It keeps one live renderer for the run, writes `/tmp/jean.mp4`, and
preserves the complete animated USD stage plus copied base-color textures at
`/tmp/jean.usd` and `/tmp/jean_textures/`:

```bash
... example_palatial_load ~/new_jean.newton.usda \
  --device cuda:0 --viewer ovrtx \
  --record-mp4 /tmp/jean.mp4 --mp4-fps 60 --steps 1000
```

There is no Palatial-specific output-mode flag. `--viewer ovrtx` changes the
viewer backend, while the existing `--record-mp4` and `--validation-report`
contracts remain the artifact destinations. This is a single-scene path: one
Newton model, one OVRTX renderer, and one retained timeline.

For a complete render plus semantic report:

```bash
uv run python -m newton.examples.palatial.example_palatial_load \
  /workspace/asset.newton.usda \
  --device cuda:0 --viewer ovrtx \
  --steps 600 --mp4-fps 60 \
  --record-mp4 /workspace/results/asset.mp4 \
  --validation-report /workspace/results/simulation_report.json
```

The output set is:

- `asset.mp4`: encoded OVRTX frames.
- `asset.usd`: the full animated USD stage, including camera, lights,
  materials, render product, and simulation time samples.
- `asset_textures/`: portable copies of referenced base-color textures.
- `simulation_report.json`: `palatial.newton-validation.v1` physics telemetry,
  evaluated separately from visual output.

`--ovrtx-script render_scene.py` exposes the same stage composition and
per-frame scripting hooks as the shared Newton OVRTX viewer.

OVRTX also preserves mesh `opacity` and index of refraction (`ior`) alongside
base color, texture, roughness, and metallic values. These are authored as
portable `UsdPreviewSurface` inputs in the retained recording.

### Red/cyan transparent glasses

The supplied glasses asset has texture image files beside it, but its USD does
not connect those files to either material. Both lenses also arrive as one
combined mesh. Create a corrected, portable asset by splitting the two
connected lens components and binding red/cyan transparent materials:

```bash
uv run python -m newton.examples.palatial.author_anaglyph_glasses \
  /workspace/glasses/glasses.usd \
  /workspace/glasses/glasses_anaglyph.usd \
  --opacity 0.38 --roughness 0.08 --ior 1.49
```

The author preserves collision and rigid-body schemas, makes the frame an
opaque `UsdPreviewSurface`, removes the unresolved `OmniSurface.mdl`
dependency, hides the old combined lens visual, and binds red to the negative-X
lens and cyan to the positive-X lens. Render the result head-on with the same
Palatial playback command:

```bash
uv run python -m newton.examples.palatial.example_palatial_load \
  /workspace/glasses/glasses_anaglyph.usd \
  --device cuda:0 --viewer ovrtx \
  --camera-position 0 0.42 0.08 \
  --camera-target 0 0.02 0.018 \
  --steps 600 --mp4-fps 60 \
  --record-mp4 /workspace/results/glasses_anaglyph.mp4 \
  --validation-report /workspace/results/glasses_anaglyph_validation.json
```

`--camera-position` and `--camera-target` are a paired generic look-at override
for both ViewerGL and ViewerOVRTX; omitting them keeps the automatic camera.

`--drop-height` is support-plane clearance, not a raw Z translation. For
example, `--drop-height 0.2` places the asset's live collision lower bound
0.2 m above the ground (or table top) while preserving relative transforms in
an articulation. Rod playback uses a fixed 0.5 m clearance for consistent
framing.

Use `--validation-report` when a machine-readable result is required:

```bash
... example_palatial_load ~/new_jean.newton.usda \
  --drop-height 0.2 --steps 1000 \
  --record-mp4 /tmp/jean.mp4 \
  --validation-report /tmp/simulation_report.json
```

The `palatial.newton-validation.v1` report checks initial and maximum
support-plane penetration, finite transforms, centroid displacement, linear
speed, and the fraction of trajectory samples whose centroid stays inside a
size-derived displacement envelope. That envelope ratio is simulator-state
telemetry; it is not rendered-pixel visibility or a texture-quality score.
MP4 recording is independent, so callers should validate the semantic report
and the video artifact separately.

Rod example:

```bash
/home/achuthan_palatial/Documents/Research_work/test_venv/bin/python \
  -m newton.examples.palatial.example_palatial_load \
  ~/hdmi_rod.newton.usda \
  --device cuda:0 --substeps 2 \
  --zero-gravity \
  --gui --steps 1000
```

## 2. `example_palatial_articulated` — drive a rigid joint chain

Default solver is MuJoCo (needs `mujoco` + `mujoco_warp`). Drives one
revolute/prismatic joint with a sine target; everything else holds zero.

```bash
/home/achuthan_palatial/Documents/Research_work/test_venv/bin/python \
  -m newton.examples.palatial.example_palatial_articulated \
  "$HOME/nvidia-asset-test/A00191 Articulated Microwave/articulated_mwave.newton.usda" \
  --gui --device cuda:0 --substeps 16 \
  --joint-target-ke 1000 --joint-target-kd 50 \
  --rotate-z 90 --drive-joint 1
```

Add `--record-mp4 /tmp/mwave.mp4` for a front-facing recording:

```bash
... example_palatial_articulated "<asset>.newton.usda" \
  --device cuda:0 --substeps 16 \
  --joint-target-ke 1000 --joint-target-kd 50 \
  --drive-joint 1 \
  --record-mp4 /tmp/mwave.mp4 --mp4-fps 60 --steps 600
```

## 3. `example_palatial_twist` — cloth twist test

Pins two opposing edges of a cloth and counter-rotates them, like
`example_cloth_twist`. Uses a **top-down** camera by default when recording.

```bash
/home/achuthan_palatial/Documents/Research_work/test_venv/bin/python \
  -m newton.examples.palatial.example_palatial_twist \
  ~/new_jean.newton.usda \
  --twist-axis x --angular-velocity 1.5 \
  --device cuda:0 --substeps 2 \
  --cloth-particle-radius 0.005 \
  --soft-contact-ke 1e5 --soft-contact-kd 1 \
  --bending-ke 1e-2 \
  --drop-height 0.5 --drop-frames 240 \
  --gui --steps 1000 \
  --record-mp4 /tmp/jean_twist.mp4 --top-view
```

## 4. `example_palatial_cable` — cable load / anchored twist

Requires a **cable** bundle (authored with `NewtonRodAPI`), but if you omit the
USD path it will generate a temporary flatRect ribbon asset for you. By
default it anchors the first segment and spins it around the cable axis so you
can see twist propagation through the loaded rod. Set `--spin-rate 0` for a
pure hanging / settling run, or `--no-anchor-first` to let the whole cable
move.

```bash
/home/achuthan_palatial/Documents/Research_work/test_venv/bin/python \
  -m newton.examples.palatial.example_palatial_cable \
  ~/power_cable.newton.usda \
  --device cuda:0 --substeps 2 \
  --anchor-first --spin-rate 0.5 \
  --gui --steps 1000
```

Record a front-facing mp4:

```bash
... example_palatial_cable ~/power_cable.newton.usda \
  --device cuda:0 --substeps 2 \
  --anchor-first --spin-rate 0.5 \
  --record-mp4 /tmp/power_cable_twist.mp4 --mp4-fps 60 --steps 1000
```

The same example also accepts the v1 Power cable assembly source USDA:

```bash
... -m newton.examples.palatial.example_palatial_cable \
  D:\palatial-sim-newton-solvers-usd\02_WorkingFiles\Cables.usda \
  --device cuda:0 --gui --substeps 10
```

### Scenario recipes (3 boundary conditions x 3 input types)

Generate the simple inputs first:

```bash
... -m newton.examples.palatial.generate_palatial_cable_usd \
  D:\Temp\flat_default.newton.usda \
  --cross-section-type flatRect \
  --length 1.5 --segment-count 16 \
  --drop-height 0.3 \
  --width 0.012 --thickness 0.004 \
  --stretch-stiffness 1.0e5 --stretch-damping 0.05 \
  --compress-stiffness 1.0e5 --compress-damping 0.05 \
  --bend-y-stiffness 8.0e2 --bend-y-damping 0.1 \
  --bend-z-stiffness 1.6e3 --bend-z-damping 0.1 \
  --torsion-stiffness 4.0e2 --torsion-damping 0.05 \
  --solver vbd_palatial --solver-substeps 4

... -m newton.examples.palatial.generate_palatial_cable_usd \
  D:\Temp\round_default.newton.usda \
  --cross-section-type roundSolid \
  --length 1.6 --segment-count 16 \
  --drop-height 0.3 \
  --radius 0.006 \
  --stretch-stiffness 1.0e5 --stretch-damping 0.05 \
  --compress-stiffness 1.0e5 --compress-damping 0.05 \
  --bend-y-stiffness 1.2e3 --bend-y-damping 0.1 \
  --bend-z-stiffness 1.2e3 --bend-z-damping 0.1 \
  --torsion-stiffness 2.0e2 --torsion-damping 0.05 \
  --solver vbd_palatial --solver-substeps 4
```

1. One end fixed, natural bend

```bash
... -m newton.examples.palatial.example_palatial_cable \
  D:\Temp\flat_default.newton.usda \
  --device cuda:0 --gui --substeps 4 --spin-rate 0.0

... -m newton.examples.palatial.example_palatial_cable \
  D:\Temp\round_default.newton.usda \
  --device cuda:0 --gui --substeps 4 --spin-rate 0.0

... -m newton.examples.palatial.example_palatial_cable \
  D:\palatial-sim-newton-solvers-usd\02_WorkingFiles\Cables.usda \
  --device cuda:0 --gui --substeps 10 --spin-rate 0.0
```

2. Both ends fixed and twisted

```bash
... -m newton.examples.palatial.example_palatial_cable \
  D:\Temp\flat_default.newton.usda \
  --device cuda:0 --gui --substeps 4 \
  --anchor-last --spin-rate 0.4 --spin-last-rate -0.4

... -m newton.examples.palatial.example_palatial_cable \
  D:\Temp\round_default.newton.usda \
  --device cuda:0 --gui --substeps 5 \
  --anchor-last --spin-rate 0.4 --spin-last-rate -0.4

... -m newton.examples.palatial.example_palatial_cable \
  D:\palatial-sim-newton-solvers-usd\02_WorkingFiles\Cables.usda \
  --device cuda:0 --gui --substeps 10 \
  --anchor-last --spin-rate 0.3 --spin-last-rate -0.3
```

3. Drop onto a box obstacle

```bash
... -m newton.examples.palatial.example_palatial_cable \
  D:\Temp\flat_default.newton.usda \
  --device cuda:0 --gui --substeps 6 \
  --no-anchor-first --spin-rate 0.0 \
  --obstacle-box --extra-drop-height 0.2

... -m newton.examples.palatial.example_palatial_cable \
  D:\Temp\round_default.newton.usda \
  --device cuda:0 --gui --substeps 6 \
  --no-anchor-first --spin-rate 0.0 \
  --obstacle-box --extra-drop-height 0.2

... -m newton.examples.palatial.example_palatial_cable \
  D:\palatial-sim-newton-solvers-usd\02_WorkingFiles\Cables.usda \
  --device cuda:0 --gui --substeps 10 \
  --no-anchor-first --spin-rate 0.0 --extra-drop-height 0.2
```

---

## Camera conventions

| Example                          | Recording camera | Flag           |
|----------------------------------|------------------|----------------|
| `example_palatial_load`          | front (-Y → +Y)  | always front   |
| `example_palatial_articulated`   | front (-Y → +Y)  | always front   |
| `example_palatial_twist`         | top-down         | `--top-view`   |
| `example_palatial_cable`         | front (-Y → +Y)  | `--top-view`   |

Newton's `ViewerGL.set_camera(pos, pitch, yaw)` is degree-based, Z-up:
* yaw=0/pitch=0 → look toward +X
* yaw=90/pitch=0 → look toward +Y (front view used here)
* pitch=-90 → look straight down (twist top view)

## Requirements

For an RTX Linux or RunPod environment using OVRTX:

```bash
uv sync --frozen --extra examples --extra ovrtx
```

The container needs FFmpeg plus EGL, OpenGL, and Vulkan runtime libraries and
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video`. The OVRTX extra is
platform-gated to Linux x86-64/aarch64.

In `test_venv`:
```bash
VIRTUAL_ENV=~/Documents/Research_work/test_venv uv pip install \
  -e ~/Documents/Research_work/palatial-sim-newton \
  'usd-core>=25.5' 'newton-usd-schemas>=0.2.0' \
  'pyglet>=2.0' mujoco mujoco_warp
```
`ffmpeg` must be on PATH for `--record-mp4`.
