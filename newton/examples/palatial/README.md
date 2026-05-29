# Palatial examples

These examples drive Newton from a **converted USDA** produced by the palatial
converter. Each one reads the bundle via `newton.palatial.load(usd_path)` —
solver, fps, substeps, body type and solver params all come baked into the
USDA, so the same script runs for any asset of the matching body type.

All commands assume the `test_venv` interpreter set up in
`~/Documents/Research_work/test_venv` (replace with your own Python).

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

---

## Camera conventions

| Example                          | Recording camera | Flag           |
|----------------------------------|------------------|----------------|
| `example_palatial_load`          | front (-Y → +Y)  | always front   |
| `example_palatial_articulated`   | front (-Y → +Y)  | always front   |
| `example_palatial_twist`         | top-down         | `--top-view`   |

Newton's `ViewerGL.set_camera(pos, pitch, yaw)` is degree-based, Z-up:
* yaw=0/pitch=0 → look toward +X
* yaw=90/pitch=0 → look toward +Y (front view used here)
* pitch=-90 → look straight down (twist top view)

## Requirements

In `test_venv`:
```bash
VIRTUAL_ENV=~/Documents/Research_work/test_venv uv pip install \
  -e ~/Documents/Research_work/palatial-sim-newton \
  'usd-core>=25.5' 'newton-usd-schemas>=0.2.0' \
  'pyglet>=2.0' mujoco mujoco_warp
```
`ffmpeg` must be on PATH for `--record-mp4`.
