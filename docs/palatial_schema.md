# Writing a custom example

This guide explains how to use `newton.palatial` — the USDA-to-Newton loader
shipped in `newton/_src/palatial/` — to write your own example script that
simulates a converted Palatial asset.

The public surface lives at `newton.palatial` (re-exported from
`newton/_src/palatial/__init__.py`). **Never import from `newton._src`.**

```python
from newton.palatial import (
    NewtonBundle,            # dataclass returned by load()
    load,                    # USDA -> ready-to-step bundle
    find_shell_prim_path,    # prim path of the cloth/shell mesh (or None)
    find_cloth_prim_path,    # legacy: bodyType="cloth" mesh path
    find_rod_prim_path,      # prim path of the rod guide (or None)
    read_shell_params,       # resolved cloth/shell param dict
    read_rod_params,         # resolved rod centerline + material param dict
)
```

---

## Installation

From the repository root, create the project virtual environment and pull
in everything needed to run the Palatial loader and the example viewer:

```bash
cd /path/to/palatial-newton
uv sync --extra examples
```

This installs:

* `newton` (this repo, in editable mode via `uv_build`)
* the upstream `newton-usd-schemas` package from PyPI (pinned in
  `uv.lock` at `0.3.1`, required as `>=0.3.1` in `pyproject.toml`) —
  provides `NewtonSceneAPI`, `NewtonMaterialAPI`, `NewtonXpbdSceneAPI`,
  etc.
* USD core (`usd-core`), Warp, MuJoCo-Warp, and the viewer / importer
  stack used by `newton.examples`.

The Palatial-specific deformable APIs (`NewtonDeformableAPI`,
`NewtonShellAPI`, `NewtonShellMaterialAPI`, `NewtonClothAPI`,
`NewtonRodAPI`, `NewtonRodMaterialAPI`) are **not** shipped via PyPI.
They live in-tree at `newton/_src/usd/schemas_ext/generatedSchema.usda`
and are auto-registered as a sibling USD plugin on first import of
`newton._src.usd`, so no separate install step is required.

Verify the install:

```bash
uv run python -c "import newton_usd_schemas, newton.palatial; \
    print('upstream schemas:', newton_usd_schemas.__version__); \
    print('palatial loader OK')"
```

Other useful extras:

| Command | Adds |
| --- | --- |
| `uv sync --extra dev` | examples + `coverage` (for `python -m newton.tests`) |
| `uv sync --extra torch-cu12` | examples + PyTorch (CUDA 12) for RL policies |
| `uv sync --extra rtx` | OVRTX real-time ray-tracing viewer |

If you need to develop against a fork of `newton-usd-schemas` instead of
the PyPI release, add a `[tool.uv.sources]` override in `pyproject.toml`,
e.g.:

```toml
[tool.uv.sources]
newton-usd-schemas = { path = "../newton-usd-schemas", editable = true }
# or:
# newton-usd-schemas = { git = "https://github.com/<you>/newton-usd-schemas", branch = "<branch>" }
```

then re-run `uv sync --extra examples`.

---

## 1. What the converted USDA already contains

`load()` reads a `*.newton.usda` produced by the Palatial pipeline. That
file already encodes everything needed to step a simulation:

| USDA prim / attribute | Drives |
| --- | --- |
| `physicsScene.newton:solver` | Solver class (`mujoco`, `xpbd`, `featherstone`, `vbd`, `semi_implicit`, `style3d`) |
| `physicsScene.newton:timeStepsPerSecond` | `fps` → `dt = 1 / fps` |
| `physicsScene.newton:solver:<key>` | Solver kwargs (`iterations`, `substeps`, ...) — forwarded only when the solver's `__init__` accepts them |
| `physicsScene` gravity attrs | World gravity |
| Mesh with `NewtonShellAPI` / `NewtonClothAPI`, or `newton:deformable:simulationIntent` in {`cloth`,`shell`}, or `newton:bodyType="cloth"` | Triggers cloth path; cloth params resolved from `newton:shell:*` (+ bound material) |
| `BasisCurves` with `NewtonRodAPI` / `newton:deformable:simulationIntent="rod"` | Triggers isotropic rod path; centerline/radius resolved from guide geometry plus optional helper rigid bodies such as `CablePath` / `CableJacket` |
| Standard `UsdPhysics` rigid body / collision / mass / material APIs | Rigid path via `builder.add_usd` |

You should not hand-pick the solver, stiffness, or fps in your example —
the converter writes them and `load()` reads them. Your example tunes the
margins, it does not re-specify the physics.

---

## 2. The single entry point: `load()`

`load()` has one positional argument and the rest are keyword-only (note the
bare `*`). The full signature is wider than it first looks:

```python
from newton.palatial import load

bundle = load(
    "/path/to/asset.newton.usda",
    *,
    solver_override=None,          # force "mujoco" / "vbd" / ... if you must
    device=None,                   # "cuda:0", "cpu"; default = wp.get_preferred_device()
    fix_base=False,                # rigid only: anchor floating roots with FIXED joints
    table=None,                    # cloth only: drop the garment onto a box table
    rod_textured_tube=False,       # rod only: sweep a textured tube along the cable
    rod_tube_radial_segments=12,   # rod only: tube cross-section resolution
    solver_param_overrides=None,   # extra solver __init__ kwargs (camelCase or snake_case)
    on_model=None,                 # callback(model) run right before the solver is built
)
```

Because everything after `usd_path` is keyword-only, `load(path, "vbd")`
raises — it has to be `load(path, solver_override="vbd")`.

`bundle` is a `NewtonBundle` dataclass:

```python
@dataclass
class NewtonBundle:
    usd_path: str
    body_type: str        # "rigid", "cloth", or "rod"
    solver_name: str
    fps: int
    model: Any            # newton.Model
    solver: Any           # SolverMuJoCo / SolverVBD / ...
    state_in: Any         # model.state()
    state_out: Any        # model.state()
    control: Any          # model.control()
    solver_params: dict   # resolved kwargs the solver was built with (see below)

    @property
    def dt(self) -> float: ...   # 1 / fps
```

That is the entire contract. Every example below builds on top of it.

A few things worth saying out loud:

* `state_in` and `state_out` are two separate state objects on purpose. The
  step loop ping-pongs them: read one, write the other, swap. Do not alias
  them.
* `control` is the input channel for actuated assets. Joint targets are
  written here, not on the model.
* `dt` is derived, not stored. It is always `1 / fps`. Substeps are your
  example's concern: you divide `dt` yourself, `load()` does not.

### `solver_params` is resolved, not raw

`bundle.solver_params` is **not** a verbatim copy of the `newton:solver:*`
attributes. By the time the bundle exists, that dict has been through:

1. the scene-attribute read,
2. the shell-knob merge (cloth only),
3. the anti-pinch self-contact defaults (cloth + VBD only, see §2.1),
4. your `solver_param_overrides` (if any), and
5. a dedupe pass that keeps the canonical camelCase form.

It is the resolved set of kwargs that were actually forwarded to the solver
constructor. If you print it and see keys the USDA never authored (e.g.
`particleRestShapeContactExclusionRadius`), that is expected — they were
injected. Treat it as "what the solver was built with."

### Notes on `fix_base`

Set `fix_base=True` for rigid articulated assets that should be anchored to
the world (so MuJoCo treats them as fixed-base articulations, like a robot
arm bolted to a workbench). `load()` patches `ModelBuilder.add_joint_free`
during `add_usd(...)` to emit `add_joint_fixed(parent=-1, child=body)`
instead, then sweeps any remaining unconnected massive bodies into fixed
joints + articulations.

### Deformable solver auto-pick

If the USDA does not pin a solver (no authored `newton:solver`) and you do
not pass `solver_override`, `load()` picks:

* for cloth: `style3d` when any `newton:shell:style3d:*` attr is authored,
  else `vbd` if `SolverVBD` is available, else `xpbd`
* for rod: `vbd` if `SolverVBD` is available, else `xpbd`

`solver_override="..."` forces a named solver regardless.

### 2.1 What `load()` does, in order

Where your override lands in this pipeline decides whether it wins or gets
overwritten, so the sequence matters:

1. Open the stage; read `PhysicsScene` solver name, fps, and every
   `newton:solver:*` attribute into `solver_params`.
2. Detect body type by walking prims (schema check, not filename).
3. Auto-pick a solver, unless the scene pinned one or you passed
   `solver_override`.
4. **Cloth only:** fold shell-level VBD knobs
   (`vbdSelfContactRadius/Margin`, `vbdConservativeBoundRelaxation`) into the
   params, then `setdefault` the anti-pinch self-contact knobs
   (`particleEnableSelfContact`, `particleRestShapeContactExclusionRadius`
   = 0.005, `particleTopologicalContactFilterThreshold` = 1,
   `particleVertexContactBufferSize` = 64) for VBD. These keep dense
   garments from erupting into self-contact spikes; an authored USDA value
   still wins because `setdefault` only fills gaps.
5. Build the model (`_build_cloth` / `_build_rod` / `_build_rigid`).
6. Apply `solver_param_overrides`. Each key pops both its camelCase and
   snake_case forms first so it cannot collide with a USDA duplicate, then
   updates and dedupes.
7. Run `on_model(model)` if provided.
8. Build the solver from the final params.
9. Allocate `state_in`, `state_out`, `control`; seed rod initial pose if
   needed; pack the `NewtonBundle`.

The takeaway: scene attrs go in first, shell + anti-pinch defaults layer on
top, your explicit overrides win last, and **only then** is the solver
constructed. Mutating `bundle.solver_params` after `load()` returns changes
nothing — the solver already exists.

---

## 3. Minimal custom example

The smallest useful example is ~30 lines: open a viewer, load the bundle,
step a fixed number of frames.

```python
# my_example.py
from __future__ import annotations
import argparse
import warp as wp  # noqa: F401 — must come before pxr.Usd
import newton
from newton.palatial import load
from newton import viewer as v


class Example:
    def __init__(self, viewer, usd_path: str):
        self.viewer = viewer

        bundle = load(usd_path)
        self.bundle = bundle
        self.model = bundle.model
        self.solver = bundle.solver
        self.state_0 = bundle.state_in
        self.state_1 = bundle.state_out
        self.control = bundle.control

        self.fps = bundle.fps
        self.frame_dt = 1.0 / float(self.fps)
        self.sim_substeps = max(1, int(bundle.solver_params.get("substeps", 1)))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # model.collide(state) allocates a Contacts buffer and returns it.
        # Pass an existing buffer (model.collide(state, contacts)) to reuse it.
        self.contacts = self.model.collide(self.state_0)
        self.viewer.set_model(self.model)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control,
                             self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("usd")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--gui", action="store_true")
    args = p.parse_args()

    viewer = v.ViewerGL(headless=False) if args.gui else v.ViewerNull()
    ex = Example(viewer, args.usd)
    for _ in range(args.steps):
        ex.step()
        ex.render()
    viewer.close()


if __name__ == "__main__":
    main()
```

Run with:

```bash
uv run python my_example.py /path/to/asset.newton.usda --gui --steps 1000
```

That single class works for **cloth**, **rigid**, and **rod** bundles — the body
type is detected inside `load()` and the right solver is constructed for
you.

> Note on `model.collide`: its signature is
> `collide(state, contacts=None, *, collision_pipeline=None) -> Contacts`.
> When `contacts` is `None` it allocates a fresh buffer (equivalent to
> calling `model.contacts()` then collide). In the per-frame loop, allocate
> once and reuse the buffer to avoid per-step allocation.

---

## 4. The step / render loop pattern

The Palatial examples follow the same shape that all `newton/examples/*`
do, so they integrate with the test harness:

* `__init__(self, viewer, ...)` — build the model + solver, set
  `self.frame_dt`, `self.sim_dt`, `self.sim_substeps`, `self.sim_time`,
  call `viewer.set_model(self.model)`.
* `simulate(self)` — run `sim_substeps` solver steps inside one frame,
  ping-pong `state_0` / `state_1`.
* `step(self)` — call `simulate()` and advance `sim_time` by
  `frame_dt`.
* `render(self)` — `begin_frame / log_state / log_contacts / end_frame`.

If you register the example in `newton/examples/` per the project
conventions in `AGENTS.md`, add `test_final()` and optionally
`test_post_step()` methods so `python -m newton.tests` can drive it.

### A note on real-time pacing

`newton.examples.run` calls `step()` once per rendered frame, and each
`step()` advances `1/fps` of sim time. If your per-frame compute cannot keep
up with `fps`, the live GUI looks like slow motion even though the sim is
time-correct. Substeps multiply per-frame compute (`substeps=10` does 10×
the solver work per frame), so a high substep count is the usual cause of a
"floaty / slow" drop in the viewer. mp4 recording paths decouple frame
output from wall-clock, so a recording can look real-time even when the GUI
drags. Pick substeps for stability, not for playback speed, and record to
mp4 when you need a faithful-speed clip.

---

## 5. Reading cloth/shell params yourself

Sometimes you want to inspect the converted USDA without building a full
model — e.g. to print stiffness, to derive a particle radius from
`newton:shell:particleRadius`, or to look up `dropHeight`.

```python
from newton.palatial import find_shell_prim_path, read_shell_params

prim_path = find_shell_prim_path("/path/to/asset.newton.usda")
if prim_path is None:
    raise SystemExit("not a cloth asset")

p = read_shell_params("/path/to/asset.newton.usda")
print(p["density"], p["triStiffness"], p["bendStiffness"], p["dropHeight"])
```

`read_shell_params` always returns every key — the defaults from
`newton/_src/palatial/shell.py::DEFAULTS` are used for any attribute that
the USDA does not author. The resolved dict contains:

```
addBendingEdges, bendDamping, bendStiffness, density, dropHeight, intent,
particleRadius, thickness, triAreaStiffness, triDamping, triDrag, triLift,
triStiffness, style3dTriAnisoKe, style3dEdgeAnisoKe,
vbdConservativeBoundRelaxation, vbdSelfContactMargin, vbdSelfContactRadius
```

The `style3d*` and `vbd*` extras come back as `None` when unauthored. New
schema (`newton:shell:*` on mesh + bound `NewtonShellMaterialAPI` material)
takes precedence over legacy (`newton:cloth:*` on mesh).

For legacy assets that only carry `newton:bodyType="cloth"`,
`find_cloth_prim_path` is the fallback locator.

---

## 6. Reading rod params yourself

For rod assets, `read_rod_params()` resolves a normalized isotropic dict from
the authored guide plus any helper bodies used to reconstruct a bent
centerline or estimate radius.

```python
from newton.palatial import find_rod_prim_path, read_rod_params

prim_path = find_rod_prim_path("/path/to/rod_asset.newton.usda")
if prim_path is None:
    raise SystemExit("not a rod asset")

p = read_rod_params("/path/to/rod_asset.newton.usda")
print(p["radius"], p["segmentCount"], p["bendStiffness"])
print(p["centerlineSourcePath"], p["radiusSourcePath"])
```

The resolved dict includes (among others) `axialStiffness`, `axialDamping`,
`bendStiffness`, `bendDamping`, `stretchStiffness`, `stretchDamping`,
`compressStiffness`, `compressDamping`, `radius`, `length`, `segmentCount`,
`closed`, `twistTotal`, `frameDefinition`, `effectiveDensity`, `points`,
`guidePrimPath`, `centerlineSourcePath`, `radiusSourcePath`,
`diffuseTexturePath`, and `displayColor`.

For a straight authored rod, the guide itself becomes the centerline and
radius source. For converted cable assets whose visible bend is represented
by helper rigid bodies, `read_rod_params()` prefers path-like helpers (for
example `CablePath`) as the centerline source and jacket-like helpers (for
example `CableJacket`) as the radius source. Rods require
`frameDefinition="parallelTransport"` and at least 3 centerline points.

---

## 7. Parameter tuning

The asset is the source of truth. When you must deviate (to replicate a
reference behavior, or because a knob the schema does not author needs a
value), there are exactly three places to do it, and which one is correct
depends on **when the solver reads the value**:

* read fresh every step → set it on `bundle.model` after `load()`;
* baked into the solver at `__init__` → pass `solver_param_overrides=...`
  to `load()` (before the solver is built);
* baked from model state at solver `__init__` (MuJoCo PD gains) → use the
  `on_model=...` hook.

Mutating `bundle.solver_params` after `load()` does nothing; the solver is
already constructed.

### a. Override solver substeps

`bundle.solver_params` carries `substeps` (when the USDA authored it) — but
your example usually drives it from a CLI flag and divides `dt` itself:

```python
sim_substeps = args.substeps or int(bundle.solver_params.get("substeps", 1))
sim_dt = (1.0 / bundle.fps) / max(1, sim_substeps)
```

### b. Contact tuning (step-read knobs on the model)

These live on `bundle.model` and are read every step, so they are safe to
set after `load()`. The defaults Newton ships are weak (`soft_contact_ke`
defaults to `1e3`), which lets cloth slowly compress through the ground over
time. A stiff `ke` paired with real damping holds it on the surface:

```python
import numpy as np
n = int(bundle.model.particle_count)
bundle.model.particle_radius.assign(np.full(n, 0.005, dtype=np.float32))
bundle.model.soft_contact_ke = 1.0e5   # 1e3/1e4 lets cloth seep through the floor
bundle.model.soft_contact_kd = 1.0     # only helps paired with a stiff ke
bundle.model.soft_contact_mu = 0.5
```

> Raising `soft_contact_kd` alone, without a stiff `ke`, makes ground
> penetration **worse** (over-damped penalty drift). Tune them together.

Per-edge bending lives in `model.edge_bending_properties`, an
`[edge_count, 2]` array of `[ke, kd]`. There is **no** scalar
`model.bending_ke` / `bending_kd` on the current build — write the array
columns instead:

```python
bp = bundle.model.edge_bending_properties.numpy().copy()
bp[:, 0] = 1.0e-2   # bend ke
bp[:, 1] = 1.0e-3   # bend kd
bundle.model.edge_bending_properties.assign(bp)
```

To flatten the rest shape so the cloth drapes instead of curling toward its
authored rest angle, zero `model.edge_rest_angle`.

### c. VBD self-contact knobs (baked at solver `__init__`)

The anti-pinch knobs and self-contact radii are VBD constructor arguments,
so they must be set **before** the solver is built — pass them through
`solver_param_overrides` (keys accept camelCase or snake_case):

```python
bundle = load(
    path,
    solver_param_overrides={
        "particle_rest_shape_contact_exclusion_radius": 0.005,
        "particle_topological_contact_filter_threshold": 1,
        "particle_vertex_contact_buffer_size": 64,
        "particle_self_contact_radius": 0.002,
        "particle_self_contact_margin": 0.002,
    },
)
```

For cloth + VBD, `load()` already injects sane defaults for these (§2.1), so
you only override when you need different values. A few VBD attributes are
also writable on the live solver after construction — for example
`bundle.solver.particle_collision_detection_interval` and
`bundle.solver.iterations`. Verify an attribute exists on your build before
relying on it; not every knob from upstream Newton is present here (e.g.
`rigid_contact_k_start` and `particle_edge_contact_buffer_size` are not
exposed as settable attributes on the current `SolverVBD`).

### d. PD gains for rigid articulated assets (MuJoCo, baked from model)

Converted articulated assets use PD control. MuJoCo bakes actuator type and
PD gains from model state at solver `__init__`, so editing
`model.joint_target_ke` / `joint_target_kd` after `load()` has no effect on
the already-built solver. Use the `on_model` hook so the edit happens before
the solver is constructed:

```python
def set_gains(model):
    ke = model.joint_target_ke.numpy().copy()
    ke[:] = 1000.0
    model.joint_target_ke.assign(ke)

bundle = load(path, on_model=set_gains)
```

If you must change gains after the fact, rebuild the solver from the edited
model instead:

```python
ke = bundle.model.joint_target_ke.numpy().copy()
ke[:] = 1000.0
bundle.model.joint_target_ke.assign(ke)
bundle.solver = type(bundle.solver)(bundle.model)
```

Joint position targets are per-step inputs and go on `bundle.control`:

```python
arr_name = "joint_target_pos" if hasattr(bundle.control, "joint_target_pos") else "joint_target"
arr = getattr(bundle.control, arr_name)
jt = arr.numpy().copy()
jt[bundle.model.joint_qd_start.numpy()[joint_index]] = target_value
arr.assign(jt)
```

---

## 8. Steps to run

* Drop the script under `newton/examples/palatial/example_palatial_<name>.py`.
* Follow the `Example` class layout. Implement `test_final()` (runs once
  after the run) so `uv run --extra dev -m newton.tests` can validate it.
* Match the project conventions in `AGENTS.md`: PEP 604 unions,
  prefix-first naming, Google-style docstrings, SI units.

For a reference implementation that exercises every section above, read
`newton/examples/palatial/example_palatial_load.py`.
