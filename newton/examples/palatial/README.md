# Palatial Examples

Examples that load converter-produced `*.newton.usda` assets via
`newton.palatial.load()` and drive them with the appropriate solver
(MuJoCo for rigid/articulated, VBD for cloth, FEM-rod for cable).

All commands assume the repo root as the working directory.

## Articulated

Drive one joint of an articulated mechanism (microwave door, drawer,
hinge, etc.) with a sine wave. Other joints settle to zero.

```bash
python -m newton.examples palatial_articulated \
    newton/examples/assets/palatial_assets/door/articulated_machine.netwon.usda \
    --drive-amplitude 1.5 --drive-frequency 0.5 --fix-base --drive-joint 0
```

Key flags:

- `--drive-joint N` — joint index to drive (default: first revolute/prismatic).
- `--drive-amplitude` — sine amplitude in rad (revolute) or m (prismatic); `0` disables.
- `--drive-frequency` — sine frequency in Hz.
- `--fix-base` / `--no-fix-base` — anchor the root body (default: fixed).
- `--joint-target-ke` / `--joint-target-kd` — override position-drive gains.

## Cloth

Load a garment / sheet / panel, optionally rotate, then drop it on the
ground. Uses the VBD solver with self-contact knobs baked in at load time.

```bash
python -m newton.examples palatial_cloth \
    newton/examples/assets/palatial_assets/shirt/Newton_asset_6a229ab66eb9ed9182c2565b.usda \
    --rotate-x -90 --substeps 1 --soft-contact-kd 1 --soft-contact-ke 1e4
```

Key flags:

- `--rotate-x/-y/-z` — euler degrees about world axes before drop.
- `--drop-height` — meters lifted in +Z before falling (default `0.2`).
- `--substeps` — solver substeps per frame.
- `--soft-contact-ke` / `--soft-contact-kd` — cloth-vs-ground normal stiffness / damping.

## Cable

Load a rod/cable asset (FEM-rod solver). Endpoint 0 is pinned; gravity
pulls the rest of the chain.

```bash
python -m newton.examples palatial_cable \
    newton/examples/assets/palatial_assets/cable/Newton_asset.usda
```

## Rigid

Load a rigid body (no joints), let it settle on the ground.

```bash
python -m newton.examples palatial_rigid \
    newton/examples/assets/palatial_assets/bed/new_bed.newton.usda
```

## Common flags

Every example accepts the standard `newton.examples` flags:

- `--device cuda:0` / `--device cpu`
- `--viewer gl` (default, interactive) / `--viewer usd --output-path out.usd` / `--viewer null`
- `--num-frames N` — run headless for N frames then exit.

See `python -m newton.examples palatial_<name> --help` for the full list.
