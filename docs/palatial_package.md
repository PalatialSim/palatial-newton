# Newton Palatial Package

This package loads the USDA files that come out of the Palatial converter and hands you back a bundle you can simulate with right away — a Newton model, a solver, and the state buffers, all wired up.

Everything public lives under `newton.palatial`. Always import from there. Don't reach into `newton._src.palatial` — that's internal and it'll move.

This doc is about how the loader works under the hood. If you just want to use it, head over to `docs/palatial_schema.md` instead.

---

## What's in the package

| File | What it does |
| --- | --- |
| `__init__.py` | Re-exports the public API. |
| `load.py` | The main entry point. Call this. |
| `shell.py` | Reads shell parameters. |
| `cloth.py` | Builds cloth models. |
| `rod.py` | Builds rod models. |
| `rod_connectors.py` | Figures out which rigid bodies are connectors that should attach to a rod's ends. |
| `usd_utils.py` | Small USD geometry helpers. |
| `_resolvers.py` | Picks the right Newton schema resolver classes. |

### A note on plugin registration

Every module does `import newton` before it touches `pxr.Usd`. That's not cosmetic — it's what registers the USD plugins. Skip it and things break in confusing ways.

---

## Public API

Everything you should ever need to import:

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

A dataclass. Think of it as the result of "open this USDA and get me something I can step."

| Field | What it is |
| --- | --- |
| `usd_path` | Where the USDA came from. |
| `body_type` | `"rigid"`, `"cloth"`, or `"rod"`. |
| `solver_name` | Which solver got picked. |
| `fps` | Steps per second. |
| `model` | The built Newton model. |
| `solver` | The solver, already constructed. |
| `state_in`, `state_out` | The double-buffered states — you swap between them every step. |
| `control` | The model's control object. |
| `solver_params` | The dict of solver parameters that were used. |

`bundle.dt` is just shorthand for `1.0 / fps`.

### load

`load` is the whole pipeline in one call: open the USD, work out what kind of body it is, build the model, spin up the solver, hand you a bundle.

| Parameter | What it does |
| --- | --- |
| `usd_path` | The USDA file to open. |
| `solver_override` | Force a specific solver instead of letting the scene pick. |
| `device` | Something like `"cuda:0"` or `"cpu"`. |
| `fix_base` | Pin every floating root body to the world. Handy for debugging. |
| `table` | A dict describing a table for cloth to land on. See below. |
| `rod_textured_tube` | Swap the rod's capsules for a swept textured cylinder. |
| `rod_tube_radial_segments` | How many sides that tube should have. |
| `solver_param_overrides` | Extra solver params to layer on top of whatever the scene specifies. |
| `on_model` | A callback that runs on the model just before the solver is built — your last chance to tweak things. |

### How the solver gets picked

The scene's `newton:solver` attribute wins if it's there. If it isn't, the loader picks a sensible default based on what kind of body it's dealing with.

### How the body type gets figured out

If any prim has `NewtonShellAPI` or `NewtonClothAPI` applied, that decides it. Otherwise the loader falls back to the `newton:deformable:simulationIntent` attribute.

---

---

## Inside `load.py`

This is the file you actually call. It's ~800 lines, but the shape is simple: a
handful of small readers that interrogate the stage, three `_build_*` functions
(one per body type), a solver factory, and the `load()` orchestrator that ties
them together. Here's every function, top to bottom, and what it's for.

### The readers (stage → facts)

| Function | What it does |
| --- | --- |
| `_read_scene_params(stage)` | Walks to the first `PhysicsScene` and pulls three things: the solver name (`newton:solver`, falling back to `palatial:solver`), the fps (`newton:timeStepsPerSecond`), and every `newton:solver:*` / `palatial:solver:*` attribute into a `solver_params` dict. Defaults to `mujoco` @ 240 fps if nothing's authored. Runs the result through `_dedupe_solver_params` before handing it back. |
| `_detect_body_type(stage)` | Decides `"cloth"`, `"rod"`, or `"rigid"` by walking prims. Cloth wins on `NewtonShellAPI` / `NewtonClothAPI`, or `simulationIntent` in `{cloth, shell}`, or the legacy `newton:bodyType="cloth"`. Rod is flagged by `NewtonRodAPI` or `simulationIntent="rod"`. Cloth short-circuits immediately; rod is only returned if nothing cloth-y showed up first. |
| `_scene_pins_solver(stage)` | Just a boolean: did the scene explicitly author a solver? `load()` uses this to decide whether it's allowed to auto-pick. |

### The builders (facts → `newton.Model`)

| Function | What it does |
| --- | --- |
| `_build_rigid(...)` | Thin wrapper over Newton's own `add_usd`. Adds a ground plane, and if `fix_base=True`, monkeypatches `add_joint_free` so floating roots come in as *fixed* joints instead — then sweeps any leftover massive unconnected bodies into fixed joints + articulations. Calls `builder.color()` only when the solver is VBD. |
| `_build_cloth(...)` | Reads shell params, triangulates the first mesh (`cloth._extract_first_mesh`), then computes a spawn transform: scale → rotate to Z-up → translate so the mesh AABB centers on the target. Optionally drops it onto a `table` box. Forwards only the `add_cloth_mesh` kwargs the installed Newton accepts (signature-introspected), always calls `color()`, and registers Style3D custom attrs when that solver is selected. |
| `_build_rod(...)` | The big one. Validates `frameDefinition="parallelTransport"` and ≥3 points, builds parallel-transport quaternions, calls `builder.add_rod(...)` with the resolved stiffness/damping, kills rod self-collisions, colors segments, optionally swaps capsules for a textured tube, attaches connector components, imports leftover rigids, then straightens the rest pose. Returns a `_RodBuildResult(model, initial_body_q)` so `load()` can seed the curved start pose. |

Two small helpers live alongside these: `_untint_textured_shapes` (forces
textured imported shapes to white vertex color so the texture shows through
instead of being tinted), and `_copy_transform` (a value-copy of a
`wp.transform`, used by the rod rest-pose math).

### The solver factory

`_build_solver(name, model, params)` is the only place a solver is constructed.
It keeps a registry of the six solver classes (`mujoco`, `xpbd`,
`featherstone`, `vbd`, `semi_implicit`, `style3d`), looks up the requested one,
and raises if this Newton build doesn't ship it. The important detail: it
**introspects the solver's `__init__` signature** and forwards only the kwargs
that constructor actually accepts, after mapping each key through
`_SOLVER_PARAM_ALIAS` (USDA camelCase → solver snake_case). So an attribute the
USDA authors but the solver doesn't understand is silently dropped rather than
blowing up — which is what lets one `solver_params` dict feed six different
solvers.

### The param-name plumbing

`_SOLVER_PARAM_ALIAS` is the camelCase→snake_case table
(`particleSelfContactRadius` → `particle_self_contact_radius`, etc.).
`_dedupe_solver_params` uses it to collapse duplicates: if both the camelCase
and snake_case form of the same knob are present, the canonical camelCase wins
and the snake form is dropped (with a printed warning if their values actually
disagreed). This matters because params arrive from three places — scene attrs,
shell knobs, and your overrides — and they can spell the same thing two ways.

### `load()` — the orchestration order

Everything above is wired together here. The sequence is deliberate, because
where a value enters decides whether it survives:

1. Open the stage; `_read_scene_params` for solver/fps/params; `_detect_body_type`.
2. Auto-pick the deformable solver **only if** the scene didn't pin one and you
   didn't pass `solver_override` (cloth → style3d if Style3D attrs are authored,
   else vbd, else xpbd; rod → vbd, else xpbd). `solver_override` trumps all.
3. **Cloth only:** fold the shell-level VBD knobs (`vbdSelfContact*`,
   `vbdConservativeBoundRelaxation`) into `solver_params`, then `setdefault` the
   anti-pinch self-contact defaults for VBD (`particleEnableSelfContact=True`,
   `particleRestShapeContactExclusionRadius=0.005`,
   `particleTopologicalContactFilterThreshold=1`,
   `particleVertexContactBufferSize=64`). `setdefault` means an authored USDA
   value still wins.
4. Build the model via the matching `_build_*`.
5. Apply `solver_param_overrides` if you passed any — each key pops both spelling
   variants first so it cleanly displaces the USDA value, then updates + dedupes.
   **These win over everything.**
6. Run the `on_model(model)` hook, if given. Last chance to mutate the model
   before the solver bakes anything from it.
7. `_build_solver(...)`. The solver now exists; `solver_params` is frozen into it.
8. Allocate `state_in`, `state_out`, `control`. For rods, copy the curved
   `initial_body_q` into both states (and `solver.body_q_prev`) and zero
   velocities so it starts at rest in its bent shape.
9. Pack and return the `NewtonBundle`.

The practical upshot: scene attrs are the floor, shell + anti-pinch defaults
layer on top, `solver_param_overrides` is the ceiling, and mutating
`bundle.solver_params` *after* `load()` returns does nothing — the solver was
already built in step 7.

> `load.py` is also runnable as a module: `python -m ...load <usd>` prints the
> detected body type, chosen solver, fps/dt, resolved solver params, and
> particle/shape/body counts. Handy for a quick "what would this load as?" check
> without writing an example.

---

## Inside `shell.py`

This is the read side of the cloth schema — it turns a `*.newton.usda` into a
plain Python dict of cloth parameters, with no model building. Three public-ish
functions and a `DEFAULTS` table.

`DEFAULTS` mirrors `generatedSchema.usda` so the reader always returns a full
dict even for a sparsely-authored asset: `thickness=1e-3`,
`particleRadius=0.01`, `addBendingEdges=True`, `dropHeight=1.0`,
`density=300.0`, `triStiffness=triAreaStiffness=1e2`,
`triDamping=triDrag=triLift=0.0`, `bendStiffness=1e-3`, `bendDamping=0.0`. Keep
this in sync with the schema when knobs are added.

| Function | What it does |
| --- | --- |
| `_has_shell_api(prim)` | True iff the prim carries `NewtonShellAPI` or `NewtonClothAPI`. Checks both the resolved applied-schemas set *and* the raw `apiSchemas` list items (prepended/appended/explicit), because composition doesn't always surface a schema in `GetAppliedSchemas()`. |
| `find_shell_prim_path(usd_path)` | Returns the prim path of the first cloth/shell mesh. Prefers a `_has_shell_api` mesh (new schema); falls back to the first mesh tagged legacy `newton:bodyType="cloth"` if no schema'd mesh exists. |
| `_bound_shell_material_prims(mesh)` | Finds Material prims bound to the mesh that carry `NewtonShellMaterialAPI`. Checks both the all-purpose binding and the `physics`-purpose binding, dedupes, and filters to materials that actually have the shell-material schema. |
| `read_shell_params(usd_path)` | The payload. Resolves every param into a normalized dict. |

How `read_shell_params` resolves a value: it builds a source list of
`[mesh, *bound_materials]` and walks it in order, taking the first authored
value it finds. The internal `_walk(*names, default)` accepts multiple attribute
names per param so it can try the new name first and a legacy alias second — e.g.
`triStiffness` reads `newton:shell:triStiffness`, then `newton:cloth:triKe`,
then falls back to `DEFAULTS`. The mesh is consulted before the material, so a
mesh-level override beats the bound material.

On top of the always-present `DEFAULTS` keys, it adds:

* `style3dTriAnisoKe` / `style3dEdgeAnisoKe` — optional anisotropic vec3s for
  the Style3D solver; `None` when unauthored.
* `vbdSelfContactRadius` / `vbdSelfContactMargin` /
  `vbdConservativeBoundRelaxation` — optional VBD knobs; `None` when unauthored.
  (`load()` is what later copies these into `solver_params` under their
  `particle*` names.)
* `intent` — from `newton:deformable:simulationIntent`, default `"cloth"`.

That's the whole contract: `find_shell_prim_path` to know *whether* it's cloth
and *where*, `read_shell_params` to know *how* it should behave. Neither touches
Warp or builds anything, so they're cheap to call for inspection.

---

## Inside `rod.py`

This is the heaviest reader (~930 lines) because rod geometry is rarely authored
cleanly — the converter often represents a cable's visible bend with helper
rigid bodies rather than a clean curve, so `rod.py` has to *reconstruct* the
centerline and estimate the radius from whatever's there. Public surface is just
`find_rod_prim_path` and `read_rod_params`; everything else is the resolution
machinery behind them.

`DEFAULTS` covers geometry + isotropic material: `radius=0.01`,
`frameDefinition="parallelTransport"`, `closed=False`,
`crossSectionType="roundSolid"`, `width=0.01`, `thickness=0.002`, `length=1.0`,
`dropHeight=0.3`, `twistTotal=0.0`, `density=1000.0`,
`stretch/compressStiffness=1e5`, all dampings `0.0`, `bendStiffness=0.0`,
`segmentCount=None` (meaning "use whatever the centerline resolves to").

### Schema + binding helpers

| Function | What it does |
| --- | --- |
| `_has_rod_api` / `_has_rod_material_api` | Applied-schema checks for `NewtonRodAPI` / `NewtonRodMaterialAPI` (same raw-`apiSchemas` fallback as shell). |
| `_find_rod_prim(stage)` | First prim carrying `NewtonRodAPI`. |
| `find_rod_prim_path(usd_path)` | Its path, or `None`. |
| `_geometry_source_prims(rod_prim)` | The rod prim plus its ancestor chain — where geometry attributes (`newton:rod:*`) are searched. |
| `_bound_rod_material_prims(rod_prim)` | Bound Materials carrying `NewtonRodMaterialAPI`, for material attributes. |
| `_authored_attribute_value(sources, *names)` | First authored value across a source list, with the prim path it came from (provenance). |

### Centerline reconstruction

This is the core. `_read_centerline_spec(...)` resolves the polyline the rod is
built on, trying sources in priority order:

1. Authored `BasisCurves.points` on the rod prim (`_read_basis_curve_points`),
   if there are >2 points.
2. Otherwise, ordered collider centers from the best matching rigid-body
   candidate (`_collect_rigid_body_candidates` → `_select_centerline_candidate`
   → `_order_polyline_centers` → `_segment_centers_to_nodes`), with the guide's
   two endpoints pinned if available.
3. Otherwise, two authored guide points joined by a straight line.
4. Last resort, a synthetic straight centerline along +X at `dropHeight`
   (`_synthesize_straight_centerline`).

If `segmentCount` is authored, the result is resampled to `segmentCount + 1`
nodes (`_resample_polyline`). A pile of small NumPy helpers supports this:
`_complete_distance_matrix`, `_order_polyline_centers` (nearest-neighbor
polyline ordering), `_endpoint_cost`, `_orient_centers_to_guide`,
`_polyline_lengths`, `_nearest_segment_index`.

### Radius estimation

Radius is resolved separately, also by priority (inside `_read_centerline_spec`,
finalized in `read_rod_params`):

1. Explicit `newton:rod:radius`.
2. A `flatRect` cross-section → `0.5 * thickness`.
3. Visual-mesh bbox of the radius candidate (`_estimate_radius_from_visual_bbox`).
4. Collider point-group spread (`_estimate_radius_from_point_groups`).
5. Median of the curve's authored `widths`, halved.
6. `DEFAULTS["radius"]`.

`_candidate_cross_section`, `_candidate_keyword_rank`, `_select_radius_candidate`
score which helper body is most likely the radius source (jacket-like names
rank higher).

### Color / texture resolution

`_resolve_display_color` and `_resolve_diffuse_texture_path` walk, in order: the
Material bound to the rod guide, then materials bound to mesh descendants along
the centerline source's ancestor chain, then `primvars:displayColor`, then (for
color) the alpha-weighted mean RGB of the bound diffuse texture
(`_sample_texture_mean_rgb`). Texture path falls back to `<usda_dir>/<name>` and
`<usda_dir>/textures/<name>`. `_coerce_color_triplet` and
`_resolve_existing_texture_path` are the small normalizers.

### `read_rod_params(usd_path)` — putting it together

Starts from `DEFAULTS`, then resolves geometry attrs from `_geometry_source_prims`
and material attrs from `_bound_rod_material_prims` (geometry vs material split
matters — stiffness comes from the material, dimensions from the geometry). Then:

* Calls `_read_centerline_spec` for `points`, `guidePrimPath`,
  `centerlineSourcePath`, and the fallback radius.
* Finalizes `radius` + `radiusSourcePath` by the priority above.
* Computes `axialStiffness` / `axialDamping` via `_canonical_or_compat_value`:
  prefer the canonical `stretch*`, fall back to an authored `compress*`.
* Computes `effectiveDensity`: for a `flatRect` cross-section, the authored
  density is rescaled by `rect_area / circular_area` so the round simulated rod
  carries the same mass per length as the real flat ribbon.
* Resolves `intent`, `displayColor`, `diffuseTexturePath`.

The returned dict is what `_build_rod` consumes to call `builder.add_rod(...)`.
Everything provenance-tagged (`*SourcePath`) so you can see *where* each
geometric decision came from when a cable reconstructs oddly.

## Building rigid bodies

`_build_rigid` is mostly a thin wrapper — it leans on Newton's own `parse_usd` and then adds a joint to every floating-base body so they're anchored properly.

## Building cloth

`_build_cloth` reads the cloth params, triangulates the mesh, scales it by `metersPerUnit`, rotates it into Z-up, and recenters it. It also wires up the custom attributes that the Style3D solver wants.

One thing worth knowing: `builder.color()` always gets called before `finalize` for cloth. SolverVBD needs it. XPBD and Style3D don't care, so it's safe either way.

### Cloth tables

The `table` keyword arg lets you drop a flat surface under the cloth so it lands on something instead of falling forever.

| Key | What it does | Default |
| --- | --- | --- |
| `pos` | Where the box sits in world space (m). | `(0.0, 0.0, 0.1)` |
| `size` | How big the box is (m). | `(1.0, 1.0, 0.1)` |
| `margin` | How much clearance to leave between the cloth's lowest point and the table top (m). | `0.01` |
| `rot` | The cloth's rest orientation. | Lays flat. |
| `cloth_scale` | Mesh scale passed through to `add_cloth_mesh`. Useful when your garment was authored at real-world size. | — |

`add_cloth_mesh` applies scale, then rotation, then translation — in that order. To get the asset centered on the table, the loader figures out its world-space AABB and picks a translation so the AABB's center lines up with `table_pos` in X and Y.

If you've configured a table, the spawn Z gets bumped up so the cloth's lowest vertex sits exactly `margin` above the table top. That way it settles right away instead of dropping in from the sky. The table itself is rendered as plain white.

### `read_shell_params`

Returns a dict with every key in `shell.DEFAULTS`, plus a handful of optional solver-specific extras that are `None` when nothing was authored.

It looks at `newton:shell:*` on the mesh and its bound material first. If those aren't there, it falls back to the older `newton:cloth:*` names for legacy USDAs. The defaults all come from `DEFAULTS`, which is hand-kept in sync with `generatedSchema.usda`. `intent` is read from `newton:deformable:simulationIntent` on the mesh, and defaults to `"cloth"` if it's missing.

When it walks bindings, it looks at both the all-purpose binding and the `physics`-purpose binding.

### VBD knobs that get forwarded

A few shell-level VBD knobs authored on the cloth's material get copied straight into `solver_params`:

| Material attribute | `solver_params` key |
| --- | --- |
| `vbdSelfContactRadius` | `particleSelfContactRadius` |
| `vbdSelfContactMargin` | `particleSelfContactMargin` |
| `vbdConservativeBoundRelaxation` | `particleConservativeBoundRelaxation` |

## Building rods

`_build_rod` builds the rod through `builder.add_rod(...)`. It expects `newton:rod:frameDefinition == "parallelTransport"` and at least 3 centerline points — anything less and you don't really have a curve.

Once the rod is built, the function:

- Turns off self-collisions between the rod's own shapes.
- Colors every segment with whatever `_resolve_display_color` came back with.
- If `rod_textured_tube` is on and there's a diffuse texture bound, hides the capsules and replaces them with a swept textured cylinder per segment.
- Hooks up any connector components to the rod's endpoints.
- Brings in any leftover content that wasn't a connector.
- Straightens the rod's rest pose by keeping the initial `body_q` it computed.

After all that, `load()` copies `rod_initial_body_q` into `state_in.body_q`, `state_out.body_q`, and `solver.body_q_prev`, and zeroes out the body velocities so the rod starts at rest.

### `read_rod_params`

Same idea as `read_shell_params`: returns every key from `rod.DEFAULTS` plus some resolved fields — `points`, `radius`, `effectiveDensity`, `displayColor`, `diffuseTexturePath`, and a few "where did this come from" provenance fields.

Geometry attributes get read from the rod's guide prim and its ancestors. Material attributes come from bound Materials carrying `NewtonRodMaterialAPI`, falling back to the geometry sources when those aren't there.

### Where the centerline comes from

The loader tries these in order and takes whichever works first:

1. Authored `BasisCurves.points` on the rod prim itself.
2. Ordered collider centers from the matching rigid-body candidate.
3. Two authored guide points (joined with a straight line).
4. A straight fallback along +X at `dropHeight`. Last resort.

If `segmentCount` is set, whatever you got out of the above is resampled to `segmentCount + 1` points.

### Where the radius comes from

Same idea — tried in order:

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

The diffuse texture path goes through the same walk but returns a filesystem path instead. Both fall back to `<usda_dir>/<name>` and `<usda_dir>/textures/<name>` if nothing's bound.

## `rod_connectors.py`

A USDA full of a rod usually has some other rigid bodies in it too — plugs, end caps, hardware. This module is what decides which of those are *connectors* that belong attached to the rod, and which are unrelated stuff that should just be imported as-is.

### `plan_rod_rigid_imports`

Returns a 5-tuple: `(connector_components, remaining_body_paths, remaining_joint_paths, all_body_paths, all_joint_paths)`.

What it does:

1. Walks the stage and grabs every prim with `PhysicsRigidBodyAPI`, plus every joint that links two of them.
2. Marks the rod's helper prims as off-limits.
3. For every joint that touches the `radiusSourcePath`, picks the non-helper side as a component root.
4. Flood-fills the rigid-body adjacency from each root so the whole connected mechanism comes along.
5. Builds a `_RodAttachmentComponent` for each one — endpoint name, the relative transform, the parent/child transforms, and any proxy bounds.

### `attach_rod_connector_component`

Brings a component in via `builder.add_usd` with `ignore_paths` set so it only imports the bodies and joints that belong to that one component. Then it cleans up:

- Recenters the imported body frames on their geometry.
- Hides connector visuals that are ridiculously bigger than their colliders (usually a modeling artifact).
- Reparents the root body's existing fixed joint so it attaches to the rod endpoint body.
- Turns off connector-vs-connector shape collisions.
- If `proxy_bounds` is set, adds a box to the main body that doesn't collide with particles. This box gets checked against every rod shape, which gives the connector a volume for resting against static stuff in the scene — without bringing back contacts on the rod side, which is what we wanted to get rid of in the first place.

### `import_remaining_rigid_content`

Brings in everything that *wasn't* a connector, again through `builder.add_usd`, with the connector paths filtered out.

### Helper math

There's a small pile of NumPy helpers up at the top of the file — quaternion multiply, quaternion rotate, quaternion conjugate, the quaternion between two vectors, and a safe normalize. They live at the top because both planning and attachment need them. There's also a helper that turns a USD to-world matrix into a `wp.transform` in Newton's world coordinates: it transforms a few basis points, reorthonormalizes them, and builds a quaternion from the resulting rotation matrix.

---

## `usd_utils.py`

The shared geometry helpers that `rod.py` and `rod_connectors.py` both lean on:

| Function | What it does |
| --- | --- |
| `has_api_schema(prim, name)` | `True` if `name` is in the applied schemas or in any `apiSchemas` list items. |
| `stage_units(stage)` | Returns `(meters_per_unit, up_axis)`. |
| `to_newton_world(points, mpu, up_axis)` | Scales by meters-per-unit, then rotates Y-up to Z-up by mapping `(x, y, z)` to `(x, -z, y)`. |
| `matrix_transform_points(matrix, points)` | Applies a `Gf.Matrix4d` to a batch of points. |
| `read_mesh_world_points(stage, prim, ...)` | Reads a mesh's points, transforms them into world coordinates, and then into Newton world coordinates. |

## `_resolvers.py`

`get_default_resolvers()` hands back a list of resolvers — `SchemaResolverNewton`, `SchemaResolverPhysx`, `SchemaResolverMjc`. It tries the public path first, then falls back to the internal `_src.usd.schemas` path that `newton 0.2.0` still uses. If neither one imports cleanly it just returns `None`, and the caller can pass that straight through to `add_usd` so it'll use whatever defaults it has built in.
