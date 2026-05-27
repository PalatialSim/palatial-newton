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
    read_shell_params,       # resolved cloth/shell param dict
)
```

---

## 1. What the converted USDA already contains

`load()` reads a `*.newton.usda` produced by the Palatial pipelikne. That file already encodes everything needed to
step a simulation:

| USDA prim / attribute | Drives |
| --- | --- |
| `physicsScene.newton:solver` | Solver class (`mujoco`, `xpbd`, `featherstone`, `vbd`, `semi_implicit`, `style3d`) |
| `physicsScene.newton:timeStepsPerSecond` | `fps` → `dt = 1 / fps` |
| `physicsScene.newton:solver:<key>` | Solver kwargs (`iterations`, `substeps`, ...) — forwarded only when the solver accepts them |
| `physicsScene` gravity attrs | World gravity |
| Mesh with `NewtonShellAPI` / `NewtonClothAPI` / `newton:bodyType="cloth"` | Triggers cloth path; cloth params resolved from `newton:shell:*` (+ bound material) |
| Standard `UsdPhysics` rigid body / collision / mass / material APIs | Rigid path via `builder.add_usd` |

You should not hand-pick the solver, stiffness, or fps in your example —
the converter writes them and `load()` reads them.

---

## 2. The single entry point: `load()`

```python
from newton.palatial import load

bundle = load(
    "/path/to/asset.newton.usda",
    solver_override=None,   # force "mujoco" / "vbd" / ... if you must
    device=None,            # "cuda:0", "cpu"; default = wp.get_preferred_device()
    fix_base=False,         # rigid only: anchor floating roots with FIXED joints
)
```

`bundle` is a `NewtonBundle` dataclass:

```python
@dataclass
class NewtonBundle:
    usd_path: str
    body_type: str        # "rigid" or "cloth"
    solver_name: str
    fps: int
    model: Any            # newton Model
    solver: Any           # SolverMuJoCo / SolverVBD / ...
    state_in: Any         # model.state()
    state_out: Any        # model.state()
    control: Any          # model.control()
    solver_params: dict   # raw newton:solver:* attrs
    @property
    def dt(self) -> float: ...   # 1 / fps
```

That is the entire contract. Every example below builds on top of it.

### Notes on `fix_base`

Set `fix_base=True` for rigid articulated assets that should be anchored to
the world (so MuJoCo treats them as fixed-base articulations, like a robot
arm bolted to a workbench). `load()` patches `ModelBuilder.add_joint_free`
during `add_usd(...)` to emit `add_joint_fixed(parent=-1, child=body)`
instead.

### Cloth solver auto-pick

If the USDA does not pin a solver and the asset is cloth, `load()` picks:

* `style3d` when any `newton:shell:style3d:*` attr is authored,
* `vbd` if `SolverVBD` is available,
* else `xpbd`.

`solver_override="..."` can override the custom solver.

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

        self.contacts = self.model.collide(self.state_0)
        self.viewer.set_model(self.model)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
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

That single class works for **both** cloth and rigid bundles — the body
type is detected inside `load()` and the right solver is constructed for
you.

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
the USDA does not author. New-schema (`newton:shell:*` on mesh + bound
`NewtonShellMaterialAPI` material) takes precedence over legacy
(`newton:cloth:*` on mesh). Optional Style3D and VBD extras come back as
`None` when unauthored.

For legacy assets that only carry `newton:bodyType="cloth"`,
`find_cloth_prim_path` is the fallback locator.

---

## 6. Parameter Optimization

In case of parameter tweaking for replicating behavior, these can be changed using the NewtonBundle class.

### a. Override solver substeps

`bundle.solver_params` carries `substeps` straight from the USDA — but
your example may want to override it via a CLI flag:

```python
sim_substeps = args.substeps or int(bundle.solver_params.get("substeps", 1))
sim_dt = (1.0 / bundle.fps) / max(1, sim_substeps)
```

### b. Contact tuning

After `load()`, you can still write per-particle radii and global
soft-contact knobs onto the cloth model:

```python
import numpy as np
n = int(bundle.model.particle_count)
bundle.model.particle_radius.assign(np.full(n, 0.005, dtype=np.float32))
bundle.model.soft_contact_ke = 1e5
bundle.model.soft_contact_kd = 1.0
bundle.model.soft_contact_mu = 1.0
bundle.model.cloth_body_contact_margin = 0.01
bundle.model.bending_ke = 1e-2
bundle.model.bending_kd = 1e-3
```

VBD-only knobs go on the solver, not the model:

```python
if type(bundle.solver).__name__ == "SolverVBD":
    bundle.solver.particle_edge_contact_buffer_size = 64
    bundle.solver.particle_collision_detection_interval = -1
    bundle.solver.rigid_contact_k_start = 100.0
```

### c. PD gains for rigid articulated assets

Since assets are converted using PhysX , articulated assets mostly support PD controls compared to PID. MuJoCo bakes PD gains at solver-init time, so if you change
`model.joint_target_ke` / `joint_target_kd`, rebuild the solver:

```python
ke = bundle.model.joint_target_ke.numpy().copy()
ke[:] = 1000.0
bundle.model.joint_target_ke.assign(ke)
bundle.solver = type(bundle.solver)(bundle.model)
```

Joint position targets go on `bundle.control`:

```python
arr_name = "joint_target_pos" if hasattr(bundle.control, "joint_target_pos") else "joint_target"
arr = getattr(bundle.control, arr_name)
jt = arr.numpy().copy()
jt[bundle.model.joint_qd_start.numpy()[joint_index]] = target_value
arr.assign(jt)
```

---

## 7. Steps to Run

* Drop the script under `newton/examples/palatial/example_palatial_<name>.py`.
* Follow the `Example` class layout. Implement `test_final()` (runs once
  after the run) so `uv run --extra dev -m newton.tests` can validate it.
* Match the project conventions in `AGENTS.md`: PEP 604 unions,
  prefix-first naming, Google-style docstrings, SI units.

For a reference implementation that exercises every section above, read
`newton/examples/palatial/example_palatial_load.py`.
