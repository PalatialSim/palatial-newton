# `newton.palatial` Package Reference

Internal package: `newton/_src/palatial/`. Reads `*.newton.usda` files
produced by the Palatial converter and returns a ready-to-step Newton
`model + solver + state` bundle. The public surface is re-exported from
`newton.palatial` — examples and docs must import from there, never from
`newton._src.palatial`.

This document covers the inline behavior previously documented in source
comments. For end-user tutorials and asset authoring, see
`docs/palatial_schema.md`.

---

## Module layout

| File | Role |
| --- | --- |
| `__init__.py` | Re-exports the public API. Keep `__all__` in sync. |
| `load.py` | Single entry point. Detects body type (`rigid` / `cloth` / `rod`), picks a solver, builds the model, returns `NewtonBundle`. |
| `shell.py` | Read-side helper for `NewtonShellAPI` / `NewtonShellMaterialAPI`. Owns `DEFAULTS` (mirror of `generatedSchema.usda`). |
| `cloth.py` | Legacy `newton:bodyType="cloth"` reader + mesh extraction fallback. |
| `rod.py` | `NewtonRodAPI` reader: resolves guide centerline, radius, material params. |
| `rod_connectors.py` | Rigid connector planning/attachment around rod endpoints. |
| `usd_utils.py` | Shared USD geometry helpers (units, transforms, point reads). |
| `_resolvers.py` | Locates Newton's schema resolver classes regardless of API location. Underscore-prefixed — do not export. |

## Plugin registration

Every module imports `newton` before any `pxr.Usd` call. That import
registers the bundled USD plugins (`newton_usd_schemas`, `newton_shell`)
via `newton/_src/usd/__init__.py`, which authorize `NewtonSceneAPI`,
`NewtonXpbdSceneAPI`, `NewtonShellAPI`, `NewtonClothAPI`,
`NewtonShellMaterialAPI`, `NewtonRodAPI`, `NewtonRodMaterialAPI`, etc.
The plugin registration must precede any `Usd.Stage.Open` in the same
process.

---

## Public API

```python
from newton.palatial import (
    NewtonBundle,            # dataclass returned by load()
    load,                    # USDA -> ready-to-step bundle
    find_shell_prim_path,    # cloth/shell mesh path
    find_cloth_prim_path,    # legacy bodyType="cloth" mesh path
    find_rod_prim_path,      # rod guide curve path
    read_shell_params,       # resolved cloth/shell param dict
    read_rod_params,         # resolved rod centerline + material dict
)
```

### `NewtonBundle`

Dataclass returned by `load()`. Carries every object needed to step the
asset:

| Field | Meaning |
| --- | --- |
| `usd_path` | Source USDA path. |
| `body_type` | `"rigid"`, `"cloth"`, or `"rod"`. |
| `solver_name` | Name passed to `_build_solver`. |
| `fps` | Time steps per second authored on the scene (default 240). |
| `model` | Built `newton.Model`. |
| `solver` | Built solver instance. |
| `state_in`, `state_out` | Double-buffer states. |
| `control` | `model.control()`. |
| `solver_params` | The full dict of `newton:solver:*` attrs the scene authored. |

`bundle.dt` returns `1.0 / fps`.

### `load(usd_path, *, solver_override=None, device=None, fix_base=False, table=None, rod_textured_tube=False, rod_tube_radial_segments=12, solver_param_overrides=None, on_model=None)`

Opens the USD, classifies the body type, builds the model, instantiates
the solver, and returns the bundle.

- `device`: warp device string (e.g. `"cuda:0"`, `"cpu"`). Defaults to
  `wp.get_preferred_device()`.
- `fix_base` (rigid only): anchor every floating root body to world via
  a fixed joint instead of the FREE joints `parse_usd` adds by default.
- `table` (cloth only): optional `{"pos": vec3, "size": vec3, ...}`.
  When provided, a static box is added under the cloth and the cloth
  is rotated/lifted so it lands on the table top. See "Cloth table
  support" below.
- `rod_textured_tube` (rod only): when True and the USD binds a diffuse
  texture to the cable mesh, hide the underlying capsule shapes and
  render the rod as a swept textured cylinder. No effect on closed rods
  or when no on-disk texture can be resolved.
- `rod_tube_radial_segments`: number of radial segments for the swept
  tube when `rod_textured_tube` is enabled.
- `solver_param_overrides`: optional `dict` of extra solver kwargs.
  Keys accept either USDA camelCase (e.g. `particleSelfContactRadius`)
  or solver snake_case (e.g. `particle_self_contact_radius`).
  Caller-supplied values win over scene-level and shell-level USDA
  attrs. Use this to feed values parsed from a companion physics JSON
  or to set VBD knobs the schema does not author.
- `on_model`: optional `Callable[[Model], None]`. Invoked once with the
  finalized model immediately before the solver is constructed. Use
  this when a solver bakes model state at `__init__` time — e.g.
  MuJoCo bakes actuator type and `joint_target_ke / kd` at solver
  construction, so writing those fields *after* `load()` returns has
  no effect.

### Solver selection

`_read_scene_params(stage)` reads from the first `PhysicsScene`:

1. `newton:solver` (preferred), then legacy `palatial:solver`.
2. `newton:timeStepsPerSecond` -> `fps`.
3. Every `newton:solver:<key>` (or `palatial:solver:<key>`) custom
   attribute becomes an entry of `solver_params`.

If the scene does not pin a solver, defaults are picked by body type:
- **cloth**: `style3d` if any `newton:shell:style3d:*` attr is authored,
  else `vbd` if available, else `xpbd`.
- **rod**: `vbd` if available, else `xpbd`.
- **rigid**: `mujoco`.

`_build_solver` filters `solver_params` against the solver's
`__init__` signature, so unknown keys are silently dropped. The alias
table `_SOLVER_PARAM_ALIAS` maps USD-canonical camelCase keys (e.g.
`particleEnableSelfContact`) to the solver constructor's snake_case
kwarg (`particle_enable_self_contact`). `_dedupe_solver_params` warns
and removes snake_case duplicates when both forms are present with
different values.

### Body-type detection

`_detect_body_type(stage)` returns `"cloth"`/`"rod"`/`"rigid"`:

- `cloth`: any prim has `NewtonShellAPI`/`NewtonClothAPI` applied, has
  `newton:deformable:simulationIntent ∈ {"cloth","shell"}`, or has the
  legacy `newton:bodyType="cloth"` marker.
- `rod`: any prim has `NewtonRodAPI` applied, or has
  `newton:deformable:simulationIntent == "rod"`.

Applied schemas are read both from `GetAppliedSchemas()` and the raw
`apiSchemas` listOp so detection works whether or not the schema plugin
is loaded in the current process.

---

## `_build_rigid`

Uses Newton's `parse_usd` via `builder.add_usd(...)`. SimReady assets
ship pre-decomposed convex pieces, so `skip_mesh_approximation=True` is
passed.

`fix_base=True` patches `builder.add_joint_free` during `add_usd(...)`
to emit `add_joint_fixed(parent=-1, child=...)` for every floating-base
body — same pattern as `basic_joints`. The patch covers the four inline
call sites in `import_usd.py` and the
`add_free_joints_to_floating_bodies` post-pass. Any rigid body added
without an articulation that escaped that path is anchored as FIXED
afterwards.

After import, `_untint_textured_shapes` forces textured shapes to a
white vertex color so the diffuse texture renders untinted.

If `solver_name == "vbd"`, `builder.color()` is called before
`finalize()`; SolverVBD requires a per-body graph coloring when any
rigid bodies are present.

## `_build_cloth`

Reads cloth/shell params via `read_shell_params` (which already merges
the new `newton:shell:*` namespace with the legacy `newton:cloth:*`
attrs and walks bound Materials). Triangulates the first
`UsdGeom.Mesh` via `cloth._extract_first_mesh`, which:

- scales by `metersPerUnit`,
- rotates Y-up source to Z-up via `(x, y, z) -> (x, -z, y)`,
- recenters XY to the origin and lifts so `min_z = 0`,
- fan-triangulates non-triangle faces.

Only kwargs the installed `builder.add_cloth_mesh` actually accepts are
forwarded (signature is introspected). Style3D-only anisotropic
stiffness (`tri_aniso_ke`, `edge_aniso_ke`) is forwarded when authored.

If `solver_name == "style3d"`, `SolverStyle3D.register_custom_attributes`
is called on the builder before `finalize()` to register the
per-particle/per-edge rest-state buffers Style3D expects.

`builder.color()` is always called before finalize for cloth — required
by SolverVBD, harmless for XPBD/Style3D.

### Cloth table support (the `table` kwarg)

| Key | Default | Meaning |
| --- | --- | --- |
| `pos` | `(0.0, 0.0, 0.1)` | World-space box position [m]. |
| `size` | `(1.0, 1.0, 0.1)` | Half-extents [m]. |
| `margin` | `0.01` | Clearance between the cloth's lowest rotated+scaled vertex and the table top [m]. |
| `rot` | `quat(-pi/2, x)` | Cloth rest orientation. Default lays a gown flat. |
| `cloth_scale` | `1.0` | Uniform mesh scale applied via `add_cloth_mesh`. Useful for garments authored at full real-world size (e.g. a ~1.9 m gown on an 0.8 m table needs `cloth_scale ~ 0.4`). |

`add_cloth_mesh` applies scale -> rot -> translation. To center the
asset's geometric centroid on the table, the loader replicates scale +
rot in NumPy to compute the world-space AABB, then picks a translation
that aligns the AABB center in X/Y with `table_pos` (or scene origin
when no table). Without this, the translation would place the mesh's
local origin (often chest/waist of a gown, not its centroid) at the
target, producing asymmetric drape.

When a table is configured, the spawn Z is raised so the cloth's lowest
vertex is exactly `margin` above the table top — the asset settles
immediately instead of free-falling.

The table itself is rendered pure white (RGB).

### `read_shell_params`

Returns a dict containing every key in `shell.DEFAULTS` plus optional
solver-specific extras (`style3dTriAnisoKe`, `style3dEdgeAnisoKe`,
`vbdSelfContactRadius`, `vbdSelfContactMargin`,
`vbdConservativeBoundRelaxation`). Optional extras are `None` when
unauthored.

Reads the new `newton:shell:*` attrs from the cloth mesh + bound
material first, falls back to legacy `newton:cloth:*` attrs for older
USDAs. Defaults come from `DEFAULTS` (mirror of `generatedSchema.usda`).
`intent` is read from `newton:deformable:simulationIntent` on the mesh,
defaulting to `"cloth"`.

Walks bound Materials under both the all-purpose binding and the
`physics` purpose binding (the purpose the converter uses when it
creates `<defaultPrim>/cloth_material`).

### Shell-level VBD knobs forwarded to solver

The loader forwards three shell-level VBD tunables authored on the
cloth material into `solver_params`. Shell-level is the canonical
source — any scene-level `newton:solver:*` equivalent is silently
superseded:

| Shell attr | Solver key |
| --- | --- |
| `vbdSelfContactRadius` | `particleSelfContactRadius` |
| `vbdSelfContactMargin` | `particleSelfContactMargin` |
| `vbdConservativeBoundRelaxation` | `particleConservativeBoundRelaxation` |

## `_build_rod`

Builds an isotropic rod via `builder.add_rod(...)` from a
`NewtonRodAPI`-authored USDA. Requires
`newton:rod:frameDefinition == "parallelTransport"` and >= 3 centerline
points (resolved by `read_rod_params`).

After construction:
- self-collisions among rod shapes are filtered via
  `filter_body_self_collisions`,
- every rod segment is painted with the color resolved via
  `_resolve_display_color` (falls back to neutral grey),
- when `rod_textured_tube` is True and a diffuse texture is bound, the
  capsule shapes are hidden and a swept textured cylinder is added per
  segment via `_build_rod_textured_tube`,
- rigid connector components are attached to the rod endpoints via
  `attach_rod_connector_component`,
- remaining rigid content is imported via
  `import_remaining_rigid_content`,
- the rod's rest pose is straightened (zero curvature) by
  `_set_rod_zero_curvature_rest_poses`, which preserves the curved
  initial body_q in `_RodBuildResult.initial_body_q` so the solver
  starts from the authored shape but relaxes toward straight.

`load()` copies `rod_initial_body_q` into `state_in.body_q`,
`state_out.body_q`, and `solver.body_q_prev` (when present), then zeros
the body velocities.

### `read_rod_params`

Returns a dict containing every key in `rod.DEFAULTS` plus resolved
fields: `points` (centerline in Newton world coords), `radius` (with
`crossSectionType=flatRect` -> `0.5 * thickness`),
`effectiveDensity` (rescales `density` so a `flatRect` rod has the same
mass per unit length as the equivalent isotropic rod), `displayColor`,
`diffuseTexturePath`, plus source-path provenance fields
(`guidePrimPath`, `centerlineSourcePath`, `radiusSourcePath`).

Geometry attrs are read from the rod guide prim and its ancestors;
material attrs are read from bound Materials carrying
`NewtonRodMaterialAPI`, falling back to the geometry sources.

`axialStiffness`/`axialDamping` use the canonical `stretch*` value when
authored; fall back to the `compress*` compatibility attrs otherwise.

### Centerline resolution (`_read_centerline_spec`)

In order of preference:

1. Authored `BasisCurves.points` on the rod prim (>= 3 nodes).
2. Ordered collider centers from the best matching rigid-body
   candidate. Candidates are scored by:
   - keyword preference for centerline (`path` > `centerline/guide` >
     others > `jacket`),
   - smaller cross-section,
   - endpoint distance to the guide curve endpoints.
   Centers are MST-ordered (`_order_polyline_centers`), oriented to
   guide endpoints, and converted to polyline nodes
   (`_segment_centers_to_nodes`).
3. Two authored guide points (degenerate but valid).
4. A straight fallback along +X at `dropHeight`
   (`_synthesize_straight_centerline`).

If `segmentCount` is authored, the resolved centerline is resampled to
exactly `segmentCount + 1` points.

### Radius resolution

In order of preference:

1. Explicit `newton:rod:radius` attribute.
2. `flatRect` cross-section: `0.5 * thickness`.
3. Visual mesh bbox of the radius candidate (`min(dims) / 2`).
4. Collider point clouds: per-segment radial distance projected onto
   the centerline tangent (90th percentile per group, median across
   groups).
5. Curve `widths` median, halved.
6. `DEFAULTS["radius"]`.

### Color / texture resolution

`_resolve_display_color` walks, in order:

1. Material bound to the rod guide.
2. Materials bound to mesh descendants in the centerline source's
   ancestor chain (the cable-side rigid body).
3. `primvars:displayColor` on those meshes.
4. Alpha-weighted mean RGB of the diffuse texture bound to those
   meshes (via `_sample_texture_mean_rgb`).

`_resolve_diffuse_texture_path` performs the same walk but returns a
filesystem path. Both fall back to `<usda_dir>/<name>` and
`<usda_dir>/textures/<name>` because some exports record absolute
paths that don't exist on the consumer machine.

## `rod_connectors.py`

Plans which rigid bodies in a rod USDA are connectors that should
attach to a rod endpoint vs. unrelated rigid content.

### `plan_rod_rigid_imports(usd_path, params, points, quaternions)`

Returns `(components, remaining_body_paths, remaining_joint_paths,
all_body_paths, all_joint_paths)`.

1. Collect every `PhysicsRigidBodyAPI` prim and every joint linking two
   of them.
2. Mark the rod's helper prims (`guidePrimPath`,
   `centerlineSourcePath`, `radiusSourcePath`) as blocked.
3. From every joint touching the `radiusSourcePath`, walk the
   non-helper neighbor as a component root.
4. For each root, flood-fill the joint adjacency (`_collect_component_paths`)
   to find every connected rigid body and joint.
5. For each component, build a `_RodAttachmentComponent` with:
   - `endpoint_name` (`"start"` or `"end"`, by closer centroid),
   - `relative_xform` / `parent_xform` / `child_xform` (computed by
     `_component_relative_xform` — aligns the connector mouth + outward
     axis to the rod endpoint frame),
   - `proxy_bounds` (a bbox in root-body local space used to add a
     non-visible collision proxy that contacts the rod without
     re-introducing connector-vs-connector contacts).
6. Whatever rigid bodies/joints remain are returned for import as plain
   rigid content.

### `attach_rod_connector_component(builder, *, component, ...)`

Imports a single component via `builder.add_usd` with `ignore_paths`
filtered to exactly the bodies/joints belonging to that component,
then:

- recenters imported body frames to local geometry centers
  (`recenter_body_frames`) so joint transforms remain stable across
  rod attachment,
- hides connector visuals that are absurdly larger than their
  colliders, falling back to the colliders rendered white
  (`hide_oversized_connector_visuals`),
- reparents the root body's existing fixed joint to the rod endpoint
  body (or inserts a new `add_joint_fixed` when the import didn't
  produce one),
- disables connector-vs-connector shape collisions
  (`_disable_shape_collisions`),
- when `proxy_bounds` is set, adds a hidden, non-particle-colliding
  box on the root body that is filtered against every rod shape — gives
  the connector real contact volume for static objects without
  re-introducing rod-side contacts.

### `import_remaining_rigid_content(builder, *, ...)`

Imports non-connector rigid bodies/joints via `builder.add_usd` with
`ignore_paths` filtered to the leftover set.

### Helper math

Pure-NumPy quaternion helpers (`_quat_mul_np`, `_quat_rotate_np`,
`_quat_conjugate_np`, `_quat_between_vectors_np`) and a robust
`_normalize_vector` live at the top of the file because they're used
both during planning and during attachment. `_rigid_transform_to_newton`
converts a USD local-to-world matrix into a `wp.transform` in Newton
world coordinates by transforming the basis points
`(0, e_x, e_y, e_z)`, re-orthonormalizing, and building a quaternion
from the resulting rotation matrix.

---

## `usd_utils.py`

Shared geometry helpers used by `rod.py` and `rod_connectors.py`:

| Function | Behavior |
| --- | --- |
| `has_api_schema(prim, name)` | True iff `name` is in `GetAppliedSchemas()` or in any of the raw `apiSchemas` listOp items. |
| `stage_units(stage)` | Returns `(meters_per_unit, up_axis)`. |
| `to_newton_world(points, mpu, up_axis)` | Scales by `mpu`; rotates Y-up to Z-up via `(x, y, z) -> (x, -z, y)`. |
| `matrix_transform_points(matrix, points)` | Applies a `Gf.Matrix4d` to a point cloud. |
| `read_mesh_world_points(stage, prim, *, ...)` | Reads a mesh's points, transforms to world, then to Newton world coords. |

## `_resolvers.py`

`get_default_resolvers()` returns
`[SchemaResolverNewton, SchemaResolverPhysx, SchemaResolverMjc]`.
Tries the public path first, falls back to the internal `_src.usd.schemas`
path used by newton 0.2.0. Returns `None` when neither is importable
so callers can pass that through unchanged and let `add_usd` use its
own defaults.

---

## Adding a new schema-backed body type

See `newton/_src/palatial/AGENTS.md` for the step-by-step recipe (new
schema -> new `<body>.py` -> `load.py` dispatch -> public re-exports
-> docs + example).

## Running the CLI

`python -m newton._src.palatial.load <usda>` prints what the loader
detected (body type, solver, fps, solver params, particle/shape/body
counts). Use it as a quick sanity check on a converted asset.
