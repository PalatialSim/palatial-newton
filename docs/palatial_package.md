# Newton Palatial Package

Loads USDA files produced by the Palatial converter and returns a bundle ready to step: a Newton model, a solver, and the state buffers, wired together.

Public surface lives under `newton.palatial`. Import from there. `newton._src.palatial` is internal and the layout will change.

This doc covers the loader internals. For usage, see `docs/palatial_schema.md`.

---

## Package layout

| File | Purpose |
| --- | --- |
| `__init__.py` | Re-exports the public API. |
| `load.py` | Main entry point. |
| `shell.py` | Shell parameter reader. |
| `cloth.py` | Cloth model builder. |
| `rod.py` | Rod model builder. |
| `rod_connectors.py` | Identifies which rigid bodies should attach to a rod's endpoints. |
| `usd_utils.py` | USD geometry helpers. |
| `_resolvers.py` | Picks the Newton schema resolver classes. |

### A note on plugin registration

Every module does `import newton` before touching `pxr.Usd`. This is required: the import is what registers the USD plugins. Skip it and you get confusing failures downstream.

---

## Public API

```python
from newton.palatial import (
    NewtonBundle,
    load,
    find_shell_prim_path,
    find_cloth_prim_path,
    find_rod_prim_path,
    read_shell_params,
    read_rod_params,
)
```

### NewtonBundle

Dataclass returned by `load()`. Contains everything you need to start stepping a scene.

| Field | Type / meaning |
| --- | --- |
| `usd_path` | Source USDA path. |
| `body_type` | `"rigid"`, `"cloth"`, or `"rod"`. |
| `solver_name` | Selected solver. |
| `fps` | Steps per second. |
| `model` | Built Newton model. |
| `solver` | Constructed solver instance. |
| `state_in`, `state_out` | Double-buffered states; swap each step. |
| `control` | Model's control object. |
| `solver_params` | Dict of solver parameters used at construction. |

`bundle.dt` is shorthand for `1.0 / fps`.

### load

Single-call pipeline: opens the USD, determines body type, builds the model, constructs the solver, returns a bundle.

| Parameter | Purpose |
| --- | --- |
| `usd_path` | USDA file to open. |
| `solver_override` | Force a specific solver. Overrides whatever the scene declares. |
| `device` | e.g. `"cuda:0"` or `"cpu"`. |
| `fix_base` | Pin every floating root body to the world. Useful for debugging. |
| `table` | Dict describing a table for cloth to land on (see below). |
| `rod_textured_tube` | Replace the rod's capsules with a swept textured cylinder. |
| `rod_tube_radial_segments` | Radial subdivision count for that tube. |
| `solver_param_overrides` | Extra solver params layered over the scene's. |
| `on_model` | Callback fired on the model before solver construction. Last chance to tweak. |

### Solver selection

If the scene authors a `newton:solver` attribute, that wins. Otherwise the loader picks a default based on body type.

### Body type detection

If any prim has `NewtonShellAPI` or `NewtonClothAPI` applied, it's cloth. Otherwise the loader checks `newton:deformable:simulationIntent`.

---

## Inside `load.py`

The file is around 800 lines but the structure is straightforward: a few readers that interrogate the stage, three `_build_*` functions (one per body type), a solver factory, and the `load()` orchestrator. Walking them top to bottom:

### Readers (stage → facts)

| Function | Behavior |
| --- | --- |
| `_read_scene_params(stage)` | Walks to the first `PhysicsScene` and reads three things: the solver name (`newton:solver`, falling back to `palatial:solver`), the fps (`newton:timeStepsPerSecond`), and all `newton:solver:*` / `palatial:solver:*` attributes into a `solver_params` dict. Defaults to `mujoco` @ 240 fps when nothing's authored. The dict is passed through `_dedupe_solver_params` before return. |
| `_detect_body_type(stage)` | Returns `"cloth"`, `"rod"`, or `"rigid"`. Cloth wins on `NewtonShellAPI` / `NewtonClothAPI`, `simulationIntent` in `{cloth, shell}`, or the legacy `newton:bodyType="cloth"`. Rod requires `NewtonRodAPI` or `simulationIntent="rod"`. Cloth short-circuits; rod is only checked after cloth comes up empty. |
| `_scene_pins_solver(stage)` | Boolean: did the scene explicitly author a solver? Used by `load()` to decide whether auto-selection is allowed. |

### Builders (facts → `newton.Model`)

| Function | Behavior |
| --- | --- |
| `_build_rigid(...)` | Wraps Newton's `add_usd`. Adds a ground plane. With `fix_base=True`, monkeypatches `add_joint_free` so floating roots come in as fixed joints, then sweeps any massive unconnected bodies into fixed-joint articulations. Calls `builder.color()` only when the solver is VBD. |
| `_build_cloth(...)` | Reads shell params, triangulates the first mesh via `cloth._extract_first_mesh`, then computes a spawn transform (scale, rotate to Z-up, translate so the mesh AABB centers on the target). Optionally drops it onto a `table` box. Forwards only the `add_cloth_mesh` kwargs the installed Newton accepts (signature-introspected). Always calls `color()`. Registers Style3D custom attrs when that solver is selected. |
| `_build_rod(...)` | Validates `frameDefinition="parallelTransport"` and ≥3 points, builds parallel-transport quaternions, calls `builder.add_rod(...)` with the resolved stiffness and damping, kills rod self-collisions, colors segments, optionally swaps capsules for a textured tube, attaches connector components, imports any leftover rigid bodies, then straightens the rest pose. Returns `_RodBuildResult(model, initial_body_q)` so `load()` can seed the curved start pose. |

Two helpers sit alongside these. `_untint_textured_shapes` forces imported textured shapes to white vertex color so the texture isn't tinted. `_copy_transform` is a value-copy of `wp.transform`, used by the rod rest-pose math.

### Solver factory

`_build_solver(name, model, params)` is the only place a solver is constructed. It holds a registry of six solver classes (`mujoco`, `xpbd`, `featherstone`, `vbd`, `semi_implicit`, `style3d`), looks up the requested one, and raises if the installed Newton build doesn't ship it.

The factory introspects the solver's `__init__` signature and only forwards kwargs that constructor accepts, after mapping each key through `_SOLVER_PARAM_ALIAS` (USDA camelCase to solver snake_case). Attributes the USDA authors but the solver doesn't understand are silently dropped instead of raising. This is what lets one `solver_params` dict feed six different solvers.

### Param-name plumbing

`_SOLVER_PARAM_ALIAS` is the camelCase-to-snake_case table (e.g. `particleSelfContactRadius` → `particle_self_contact_radius`). `_dedupe_solver_params` uses it to collapse duplicates: when both forms of the same knob are present, the camelCase form wins and the snake form is dropped. A warning is printed if the two values actually disagreed. This matters because params arrive from three places (scene attrs, shell knobs, and your overrides) and can spell the same thing two ways.

### `load()`: orchestration order

The sequence here is deliberate. Where a value enters decides whether it survives.

1. Open the stage. Run `_read_scene_params` for solver/fps/params, then `_detect_body_type`.
2. Auto-pick the deformable solver only if the scene didn't pin one and no `solver_override` was passed. Cloth: style3d if Style3D attrs are authored, else vbd, else xpbd. Rod: vbd, else xpbd. `solver_override` trumps all.
3. Cloth only: fold the shell-level VBD knobs (`vbdSelfContact*`, `vbdConservativeBoundRelaxation`) into `solver_params`, then `setdefault` the anti-pinch self-contact defaults for VBD (`particleEnableSelfContact=True`, `particleRestShapeContactExclusionRadius=0.005`, `particleTopologicalContactFilterThreshold=1`, `particleVertexContactBufferSize=64`). `setdefault` means an authored USDA value still wins.
4. Build the model via the matching `_build_*`.
5. Apply `solver_param_overrides` if passed. Each key pops both spelling variants first to cleanly displace the USDA value, then updates and dedupes. Overrides win over everything.
6. Run `on_model(model)` if given. Last chance to mutate the model before the solver bakes anything from it.
7. `_build_solver(...)`. The solver now exists; `solver_params` is frozen into it.
8. Allocate `state_in`, `state_out`, `control`. For rods, copy the curved `initial_body_q` into both states (and into `solver.body_q_prev`) and zero velocities so the rod starts at rest in its bent shape.
9. Pack and return the `NewtonBundle`.

Net effect: scene attrs are the floor, shell and anti-pinch defaults layer on top, `solver_param_overrides` is the ceiling. Mutating `bundle.solver_params` after `load()` returns does nothing, because the solver was already built in step 7.

> `load.py` is also runnable as a module: `python -m ...load <usd>` prints the detected body type, chosen solver, fps/dt, resolved solver params, and particle/shape/body counts. Useful for a quick "what would this load as?" check without writing an example.

---

## Inside `shell.py`

The read side of the cloth schema. Turns a `*.newton.usda` into a plain Python dict of cloth parameters without building anything. Three public-ish functions and a `DEFAULTS` table.

`DEFAULTS` mirrors `generatedSchema.usda` so the reader always returns a full dict even for a sparsely-authored asset: `thickness=1e-3`, `particleRadius=0.01`, `addBendingEdges=True`, `dropHeight=1.0`, `density=300.0`, `triStiffness=triAreaStiffness=1e2`, `triDamping=triDrag=triLift=0.0`, `bendStiffness=1e-3`, `bendDamping=0.0`. Keep in sync with the schema when knobs are added.

| Function | Behavior |
| --- | --- |
| `_has_shell_api(prim)` | True iff the prim carries `NewtonShellAPI` or `NewtonClothAPI`. Checks both the resolved applied-schemas set and the raw `apiSchemas` list items (prepended, appended, explicit), because composition doesn't always surface a schema in `GetAppliedSchemas()`. |
| `find_shell_prim_path(usd_path)` | Returns the prim path of the first cloth/shell mesh. Prefers a `_has_shell_api` mesh (new schema); falls back to the first mesh tagged legacy `newton:bodyType="cloth"` if no schema'd mesh exists. |
| `_bound_shell_material_prims(mesh)` | Finds Material prims bound to the mesh that carry `NewtonShellMaterialAPI`. Checks both the all-purpose binding and the `physics`-purpose binding, dedupes, and filters to materials that actually have the shell-material schema. |
| `read_shell_params(usd_path)` | The payload. Resolves every param into a normalized dict. |

How `read_shell_params` resolves a value: it builds a source list of `[mesh, *bound_materials]` and walks it in order, taking the first authored value it finds. The internal `_walk(*names, default)` accepts multiple attribute names per param so it can try the new name first and a legacy alias second. For example, `triStiffness` reads `newton:shell:triStiffness`, then `newton:cloth:triKe`, then falls back to `DEFAULTS`. The mesh is consulted before the material, so a mesh-level override beats the bound material.

Beyond the always-present `DEFAULTS` keys, the dict also gets:

* `style3dTriAnisoKe` / `style3dEdgeAnisoKe`: optional anisotropic vec3s for the Style3D solver. `None` when unauthored.
* `vbdSelfContactRadius` / `vbdSelfContactMargin` / `vbdConservativeBoundRelaxation`: optional VBD knobs. `None` when unauthored. `load()` is what later copies these into `solver_params` under their `particle*` names.
* `intent`: from `newton:deformable:simulationIntent`. Defaults to `"cloth"`.

That's the whole contract. `find_shell_prim_path` tells you whether it's cloth and where. `read_shell_params` tells you how it should behave. Neither touches Warp or builds anything, so they're cheap to call for inspection.

---

## Inside `rod.py`

This is the heaviest reader (around 930 lines) because rod geometry is rarely authored cleanly. The converter often represents a cable's visible bend with helper rigid bodies rather than a clean curve, so `rod.py` has to reconstruct the centerline and estimate the radius from whatever's available. Public surface is just `find_rod_prim_path` and `read_rod_params`; everything else is the resolution machinery behind them.

`DEFAULTS` covers geometry plus isotropic material: `radius=0.01`, `frameDefinition="parallelTransport"`, `closed=False`, `crossSectionType="roundSolid"`, `width=0.01`, `thickness=0.002`, `length=1.0`, `dropHeight=0.3`, `twistTotal=0.0`, `density=1000.0`, `stretch/compressStiffness=1e5`, all dampings `0.0`, `bendStiffness=0.0`, `segmentCount=None` (meaning "use whatever the centerline resolves to").

### Schema and binding helpers

| Function | Behavior |
| --- | --- |
| `_has_rod_api` / `_has_rod_material_api` | Applied-schema checks for `NewtonRodAPI` / `NewtonRodMaterialAPI`. Same raw-`apiSchemas` fallback as shell. |
| `_find_rod_prim(stage)` | First prim carrying `NewtonRodAPI`. |
| `find_rod_prim_path(usd_path)` | Its path, or `None`. |
| `_geometry_source_prims(rod_prim)` | The rod prim plus its ancestor chain. Where geometry attributes (`newton:rod:*`) are searched. |
| `_bound_rod_material_prims(rod_prim)` | Bound Materials carrying `NewtonRodMaterialAPI`, for material attributes. |
| `_authored_attribute_value(sources, *names)` | First authored value across a source list, with the prim path it came from (provenance). |

### Centerline reconstruction

This is the core. `_read_centerline_spec(...)` resolves the polyline the rod is built on, trying sources in priority order:

1. Authored `BasisCurves.points` on the rod prim (`_read_basis_curve_points`), if there are more than 2 points.
2. Otherwise, ordered collider centers from the best matching rigid-body candidate (`_collect_rigid_body_candidates` → `_select_centerline_candidate` → `_order_polyline_centers` → `_segment_centers_to_nodes`), with the guide's two endpoints pinned if available.
3. Otherwise, two authored guide points joined by a straight line.
4. Last resort: a synthetic straight centerline along +X at `dropHeight` (`_synthesize_straight_centerline`).

If `segmentCount` is authored, the result is resampled to `segmentCount + 1` nodes (`_resample_polyline`). A pile of small NumPy helpers supports this: `_complete_distance_matrix`, `_order_polyline_centers` (nearest-neighbor polyline ordering), `_endpoint_cost`, `_orient_centers_to_guide`, `_polyline_lengths`, `_nearest_segment_index`.

### Radius estimation

Radius is resolved separately, also by priority (inside `_read_centerline_spec`, finalized in `read_rod_params`):

1. Explicit `newton:rod:radius`.
2. A `flatRect` cross-section → `0.5 * thickness`.
3. Visual-mesh bbox of the radius candidate (`_estimate_radius_from_visual_bbox`).
4. Collider point-group spread (`_estimate_radius_from_point_groups`).
5. Median of the curve's authored `widths`, halved.
6. `DEFAULTS["radius"]`.

`_candidate_cross_section`, `_candidate_keyword_rank`, and `_select_radius_candidate` score which helper body is most likely the radius source (jacket-like names rank higher).

### Color and texture resolution

`_resolve_display_color` and `_resolve_diffuse_texture_path` walk, in order: the Material bound to the rod guide, then materials bound to mesh descendants along the centerline source's ancestor chain, then `primvars:displayColor`, then (for color) the alpha-weighted mean RGB of the bound diffuse texture (`_sample_texture_mean_rgb`). Texture path falls back to `<usda_dir>/<name>` and `<usda_dir>/textures/<name>`. `_coerce_color_triplet` and `_resolve_existing_texture_path` are the small normalizers.

### `read_rod_params(usd_path)`: putting it together

Starts from `DEFAULTS`, then resolves geometry attrs from `_geometry_source_prims` and material attrs from `_bound_rod_material_prims`. The geometry vs material split matters: stiffness comes from the material, dimensions from the geometry. Then:

* Calls `_read_centerline_spec` for `points`, `guidePrimPath`, `centerlineSourcePath`, and the fallback radius.
* Finalizes `radius` and `radiusSourcePath` by the priority above.
* Computes `axialStiffness` / `axialDamping` via `_canonical_or_compat_value`: prefer the canonical `stretch*`, fall back to an authored `compress*`.
* Computes `effectiveDensity`. For a `flatRect` cross-section, the authored density is rescaled by `rect_area / circular_area` so the round simulated rod carries the same mass per length as the real flat ribbon.
* Resolves `intent`, `displayColor`, `diffuseTexturePath`.

The returned dict is what `_build_rod` consumes to call `builder.add_rod(...)`. Everything is provenance-tagged (`*SourcePath`) so you can see where each geometric decision came from when a cable reconstructs oddly.

## Building rigid bodies

`_build_rigid` is mostly a thin wrapper. It leans on Newton's own `parse_usd` and adds a joint to every floating-base body so they're anchored properly.

## Building cloth

`_build_cloth` reads the cloth params, triangulates the mesh, scales it by `metersPerUnit`, rotates it into Z-up, and recenters it. It also wires up the custom attributes that the Style3D solver wants.

One thing worth knowing: `builder.color()` always gets called before `finalize` for cloth. SolverVBD needs it. XPBD and Style3D don't care, so it's safe either way.

### Cloth tables

The `table` keyword arg lets you drop a flat surface under the cloth so it lands on something instead of falling forever.

| Key | Purpose | Default |
| --- | --- | --- |
| `pos` | Box position in world space (m). | `(0.0, 0.0, 0.1)` |
| `size` | Box size (m). | `(1.0, 1.0, 0.1)` |
| `margin` | Clearance between the cloth's lowest point and the table top (m). | `0.01` |
| `rot` | Cloth's rest orientation. | Lays flat. |
| `cloth_scale` | Mesh scale passed through to `add_cloth_mesh`. Useful when your garment was authored at real-world size. | — |

`add_cloth_mesh` applies scale, then rotation, then translation, in that order. To center the asset on the table, the loader figures out its world-space AABB and picks a translation so the AABB's center lines up with `table_pos` in X and Y.

If you've configured a table, the spawn Z gets bumped up so the cloth's lowest vertex sits exactly `margin` above the table top. That way it settles right away instead of dropping in from the sky. The table itself is rendered as plain white.

### `read_shell_params`

Returns a dict with every key in `shell.DEFAULTS`, plus a handful of optional solver-specific extras that are `None` when nothing was authored.

It checks `newton:shell:*` on the mesh and its bound material first. Falls back to the older `newton:cloth:*` names for legacy USDAs if those aren't there. Defaults all come from `DEFAULTS`, hand-kept in sync with `generatedSchema.usda`. `intent` is read from `newton:deformable:simulationIntent` on the mesh, defaulting to `"cloth"` if missing.

When walking bindings, it checks both the all-purpose binding and the `physics`-purpose binding.

### VBD knobs that get forwarded

A few shell-level VBD knobs authored on the cloth's material get copied straight into `solver_params`:

| Material attribute | `solver_params` key |
| --- | --- |
| `vbdSelfContactRadius` | `particleSelfContactRadius` |
| `vbdSelfContactMargin` | `particleSelfContactMargin` |
| `vbdConservativeBoundRelaxation` | `particleConservativeBoundRelaxation` |

## Building rods

`_build_rod` builds the rod through `builder.add_rod(...)`. It expects `newton:rod:frameDefinition == "parallelTransport"` and at least 3 centerline points; anything less and you don't really have a curve.

Once the rod is built, the function:

- Turns off self-collisions between the rod's own shapes.
- Colors every segment with whatever `_resolve_display_color` returned.
- If `rod_textured_tube` is on and there's a diffuse texture bound, hides the capsules and replaces them with a swept textured cylinder per segment.
- Hooks up any connector components to the rod's endpoints.
- Brings in any leftover content that wasn't a connector.
- Straightens the rod's rest pose by keeping the initial `body_q` it computed.

After all that, `load()` copies `rod_initial_body_q` into `state_in.body_q`, `state_out.body_q`, and `solver.body_q_prev`, then zeroes out the body velocities so the rod starts at rest.

### `read_rod_params`

Same idea as `read_shell_params`: returns every key from `rod.DEFAULTS` plus some resolved fields (`points`, `radius`, `effectiveDensity`, `displayColor`, `diffuseTexturePath`, and a few provenance fields recording where each value came from).

Geometry attributes get read from the rod's guide prim and its ancestors. Material attributes come from bound Materials carrying `NewtonRodMaterialAPI`, falling back to the geometry sources when those aren't there.

### Where the centerline comes from

The loader tries these in order and takes whichever works first:

1. Authored `BasisCurves.points` on the rod prim itself.
2. Ordered collider centers from the matching rigid-body candidate.
3. Two authored guide points (joined with a straight line).
4. A straight fallback along +X at `dropHeight`. Last resort.

If `segmentCount` is set, the result is resampled to `segmentCount + 1` points.

### Where the radius comes from

Same idea, tried in order:

1. An explicit `newton:rod:radius`.
2. A `FlatRect` cross-section, in which case it's `0.5 * thickness`.
3. The bbox of the visual mesh on the radius candidate.
4. Collider point clouds.
5. The median of the curve's `widths`, halved.
6. `DEFAULTS["radius"]`.

### Color and texture

Display color is resolved by walking, in order:

1. The Material bound to the rod guide.
2. Materials bound to mesh descendants in the ancestor chain of whatever provided the centerline.
3. `primvars:displayColor` on those same meshes.
4. The alpha-weighted mean RGB of the diffuse texture bound to those meshes.

The diffuse texture path goes through the same walk but returns a filesystem path. Both fall back to `<usda_dir>/<name>` and `<usda_dir>/textures/<name>` if nothing's bound.

## `rod_connectors.py`

A USDA full of a rod usually has other rigid bodies in it too: plugs, end caps, hardware. This module decides which of those are connectors that belong attached to the rod, and which are unrelated stuff that should just be imported as-is.

### `plan_rod_rigid_imports`

Returns a 5-tuple: `(connector_components, remaining_body_paths, remaining_joint_paths, all_body_paths, all_joint_paths)`.

What it does:

1. Walks the stage and grabs every prim with `PhysicsRigidBodyAPI`, plus every joint that links two of them.
2. Marks the rod's helper prims as off-limits.
3. For every joint that touches the `radiusSourcePath`, picks the non-helper side as a component root.
4. Flood-fills the rigid-body adjacency from each root so the whole connected mechanism comes along.
5. Builds a `_RodAttachmentComponent` for each one: endpoint name, the relative transform, the parent and child transforms, and any proxy bounds.

### `attach_rod_connector_component`

Brings a component in via `builder.add_usd` with `ignore_paths` set so it only imports the bodies and joints that belong to that one component. Then it cleans up:

- Recenters the imported body frames on their geometry.
- Hides connector visuals that are ridiculously bigger than their colliders (usually a modeling artifact).
- Reparents the root body's existing fixed joint so it attaches to the rod endpoint body.
- Turns off connector-vs-connector shape collisions.
- If `proxy_bounds` is set, adds a box to the main body that doesn't collide with particles. This box gets checked against every rod shape, which gives the connector a volume for resting against static stuff in the scene without bringing back contacts on the rod side (which is what we wanted to get rid of in the first place).

### `import_remaining_rigid_content`

Brings in everything that wasn't a connector, again through `builder.add_usd`, with the connector paths filtered out.

### Helper math

A small pile of NumPy helpers lives at the top of the file: quaternion multiply, quaternion rotate, quaternion conjugate, the quaternion between two vectors, and a safe normalize. Both planning and attachment need them, which is why they sit at the top. There's also a helper that turns a USD to-world matrix into a `wp.transform` in Newton's world coordinates: it transforms a few basis points, reorthonormalizes them, and builds a quaternion from the resulting rotation matrix.

---

## `usd_utils.py`

Shared geometry helpers that `rod.py` and `rod_connectors.py` both lean on:

| Function | Behavior |
| --- | --- |
| `has_api_schema(prim, name)` | `True` if `name` is in the applied schemas or in any `apiSchemas` list items. |
| `stage_units(stage)` | Returns `(meters_per_unit, up_axis)`. |
| `to_newton_world(points, mpu, up_axis)` | Scales by meters-per-unit, then rotates Y-up to Z-up by mapping `(x, y, z)` to `(x, -z, y)`. |
| `matrix_transform_points(matrix, points)` | Applies a `Gf.Matrix4d` to a batch of points. |
| `read_mesh_world_points(stage, prim, ...)` | Reads a mesh's points, transforms them into world coordinates, and then into Newton world coordinates. |

## `_resolvers.py`

`get_default_resolvers()` returns a list of resolvers: `SchemaResolverNewton`, `SchemaResolverPhysx`, `SchemaResolverMjc`. It tries the public path first, then falls back to the internal `_src.usd.schemas` path that `newton 0.2.0` still uses. If neither imports cleanly it returns `None`, and the caller can pass that straight through to `add_usd` so it'll use whatever defaults it has built in.