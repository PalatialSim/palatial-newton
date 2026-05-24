# Newton Rod/Cable USD API - Palatial Integration Guide

This document replaces the earlier `NewtonCableAPI` / `NewtonCableMaterialAPI`
draft.

The updated recommendation is to reuse the USD schema shape already explored in:

- `D:/palatial-sim-newton-solvers-usd`
- `D:/palatial-sim-newton-solvers-usd-handoff`

That schema family is:

- `NewtonDeformableAPI`
- `NewtonRodAPI`
- `NewtonRodMaterialAPI`

In other words:

- keep the Python/runtime feature name "cable" on the palatial side if that is
  more readable for this repo,
- but align authored USD API names and attribute namespaces with the existing
  `rod` schema work instead of inventing a parallel `newton:cable:*` family.

This gives us maximum reuse of existing authoring patterns, `BasisCurves`
centerline semantics, material layout, and future anisotropic compatibility.

---

## 1. Reuse Strategy

### What to reuse from the other repos

Reuse these ideas directly:

1. A cable is authored as a 1-D deformable with:
   - a root `Xform` / `Scope`,
   - a `BasisCurves` centerline,
   - a bound `Material`.
2. The root applies:
   - `NewtonDeformableAPI`
   - `NewtonRodAPI`
3. The material applies:
   - `NewtonRodMaterialAPI`
4. The authored simulation intent is:
   - `newton:deformable:simulationIntent = "rod"`
5. The centerline orientation convention is:
   - `newton:rod:frameDefinition = "parallelTransport"`

### What to stage now vs. later

The external repos contain two different layers:

1. USD schema + authoring + runtime mapping
2. vendored Newton subtree changes for true anisotropic runtime support

For this repo, the immediate target is:

- add the rod/cable USD schema surface,
- add a palatial read-side loader for it,
- and extend the **stock** `builder.add_rod()` path so it can accept
  anisotropic cable semantics as an intermediate validation step.

That means the first implementation phase in this repo can be:

- schema-aligned,
- loader-aligned,
- call-surface-aligned with anisotropic inputs,
- while still collapsing those anisotropic inputs into the current isotropic
  `JointType.CABLE` behavior underneath.

What stays for a later phase is the full vendored Newton runtime port, for
example:

- `JointType.ANISOTROPIC_CABLE`
- `add_joint_anisotropic_cable()`
- `add_rod_anisotropic()`
- `add_rod_graph_anisotropic()`
- `SolverVBDPalatial`

So the split is:

- **phase A**: stock `add_rod()` learns the richer anisotropic parameter surface
  and uses a deterministic collapse for validation,
- **phase B**: port the subtree-backed Newton runtime changes so those same
  parameters are consumed natively without collapse.

---

## 2. Schema Names to Use

Use the same schema names as the related repos:

| Schema name | Applied on | Role |
|---|---|---|
| `NewtonDeformableAPI` | `Xform`, `Gprim`, `Scope` | shared deformable flags |
| `NewtonRodAPI` | `BasisCurves`, `Xform`, `Scope` | 1-D cable/rod geometry |
| `NewtonRodMaterialAPI` | `Material` | 1-D material response |

Important naming decision:

- internal Python helper module can still be `newton/_src/palatial/cable.py`
- public helper names can still be `find_cable_*` / `read_cable_*`
- but the authored USD API and attribute namespace should be `rod`, not `cable`

This keeps the runtime feature easy to read in this repo while aligning the USD
surface with the existing work.

---

## 3. Attribute Inventory to Reuse

### 3.1 Shared deformable attrs

These already fit the existing `NewtonDeformableAPI`:

| USD attribute | Meaning | Default |
|---|---|---|
| `newton:deformable:enabled` | runtime enable switch | `true` |
| `newton:deformable:selfCollisionEnabled` | self-collision toggle | `false` |
| `newton:deformable:simulationIntent` | must be `"rod"` for cables | `"rod"` |
| `newton:deformable:velocityDamping` | global velocity damping | `0.0` |

### 3.2 Rod geometry attrs

These should live on the cable root `Xform` / `Scope` and optionally also be
available on the `BasisCurves` prim because `NewtonRodAPI` can apply there too.

| USD attribute | Meaning | Default | Notes |
|---|---|---|---|
| `newton:rod:frameDefinition` | frame convention | `"parallelTransport"` | required by loader |
| `newton:rod:closed` | closed loop | `false` | open chain by default |
| `newton:rod:crossSectionType` | cross-section kind | `"roundSolid"` | `"flatRect"` allowed for ribbon-like assets |
| `newton:rod:radius` | round cable radius [m] | `0.005` | primary stock runtime field |
| `newton:rod:width` | ribbon width [m] | `0.01` | optional, for `flatRect` |
| `newton:rod:thickness` | ribbon thickness [m] | `0.002` | optional, for `flatRect` |
| `newton:rod:segmentCount` | segment count | `16` | points count is `segmentCount + 1` |
| `newton:rod:length` | centerline length [m] | `1.0` | author on the centerline `BasisCurves` prim |

### 3.3 Palatial convenience attrs

These are worth keeping in this repo even though they are not the core
cross-repo rod material model. They match the sort of loader-side convenience
that already proved useful in the shell path.

| USD attribute | Meaning | Default | Notes |
|---|---|---|---|
| `newton:rod:dropHeight` | procedural spawn height [m] | `0.3` | only used when the loader has to synthesize a straight centerline |
| `newton:rod:twistTotal` | total distributed twist [rad] | `0.0` | forwarded to `create_parallel_transport_cable_quaternions()` |

Important:

- these are local palatial convenience attrs,
- they should not replace authored `BasisCurves` points,
- they should not replace the explicit anisotropic stiffness or damping fields,
- do **not** reintroduce `newton:rod:youngsModulus`.

### 3.4 Temporary compatibility attrs

The other repos still keep a small migration layer. For this repo, the loader
may read these as fallbacks, but they should not drive the formal schema design:

| USD attribute | Meaning |
|---|---|
| `newton:rod:isClosed` | legacy fallback for `newton:rod:closed` |
| `newton:rod:verticesPerSegment` | temporary compatibility attr; expected value is `2` |
| `newton:rodMaterial:*` | root-level legacy fallback material attrs |

### 3.5 Rod material attrs

Reuse the full material layout from the related repos even if this repo first
lowers it into stock isotropic `add_rod()`.

| USD attribute | Meaning | Default |
|---|---|---|
| `newton:rod:density` | density [kg/m^3] | `1000.0` |
| `newton:rod:stretchStiffness` | axial tension stiffness | `1.0e5` |
| `newton:rod:stretchDamping` | axial tension damping | `0.0` |
| `newton:rod:compressStiffness` | axial compression stiffness | `1.0e5` |
| `newton:rod:compressDamping` | axial compression damping | `0.0` |
| `newton:rod:bendYStiffness` | bend stiffness around local X | `1.0e3` |
| `newton:rod:bendYDamping` | bend damping around local X | `0.0` |
| `newton:rod:bendZStiffness` | bend stiffness around local Y | `1.0e3` |
| `newton:rod:bendZDamping` | bend damping around local Y | `0.0` |
| `newton:rod:torsionStiffness` | torsion stiffness around local Z | `1.0e3` |
| `newton:rod:torsionDamping` | torsion damping around local Z | `0.0` |

These fields are future-proof:

- phase A stock runtime can accept and collapse them to isotropic rod
  parameters,
- a later subtree-backed anisotropic runtime port can consume them one-to-one.

---

## 4. Current-Repo Integration Plan

### Step 1 - Update `generatedSchema.usda`

Path:

- `newton/_src/usd/schemas_ext/generatedSchema.usda`

Keep using the current plugin package in this repo, but add rod schemas with the
same names and field layout as the related repos.

Also keep `NewtonDeformableAPI`; it already exists and already allows
`simulationIntent = "rod"`.

Recommended blocks to add:

```usda
class "NewtonRodAPI" (
    prepend apiSchemas = ["NewtonDeformableAPI"]
    doc = """1D deformable rod/cable geometry.
    Apply on the cable root prim (Xform or Scope) and/or the authored
    BasisCurves centerline prim. Material-response attrs live on a bound
    Material via NewtonRodMaterialAPI."""
)
{
    uniform token newton:rod:frameDefinition = "parallelTransport" (
        allowedTokens = ["parallelTransport"]
        doc = """Frame convention used to interpret per-segment orientation.
        Current palatial loader expects parallel transport frames."""
    )

    bool newton:rod:closed = false (
        doc = "If true, close the rod/cable into a loop."
    )

    uniform token newton:rod:crossSectionType = "roundSolid" (
        allowedTokens = ["roundSolid", "flatRect"]
        doc = """Cross-section kind. roundSolid maps directly to stock
        add_rod(); flatRect can be approximated in the stock phase."""
    )

    float newton:rod:radius = 0.005 (
        doc = "Round cable radius [m]."
    )

    float newton:rod:width = 0.01 (
        doc = "Flat-rect ribbon width [m]. Optional."
    )

    float newton:rod:thickness = 0.002 (
        doc = "Flat-rect ribbon thickness [m]. Optional."
    )

    int newton:rod:segmentCount = 16 (
        doc = "Number of rod segments. Number of points is segmentCount + 1."
    )

    float newton:rod:length = 1.0 (
        doc = """Centerline length [m]. Prefer authoring this on the
        BasisCurves centerline prim."""
    )

    float newton:rod:dropHeight = 0.3 (
        doc = """Palatial loader convenience height [m]. Only used when the
        loader needs to synthesize a straight centerline because no
        BasisCurves points were authored."""
    )

    float newton:rod:twistTotal = 0.0 (
        doc = """Palatial loader convenience total twist [rad]. Applied when
        generating segment quaternions from the centerline."""
    )
}

class "NewtonRodMaterialAPI" (
    prepend apiSchemas = ["PhysicsMaterialAPI"]
    doc = """1D deformable rod/cable material response.
    Apply on a Material prim and bind it to the cable root via
    MaterialBindingAPI."""
)
{
    float newton:rod:density = 1000.0 (
        doc = "Bulk density [kg/m^3]."
    )

    float newton:rod:stretchStiffness = 100000.0 (
        doc = "Axial tension stiffness."
    )

    float newton:rod:stretchDamping = 0.0 (
        doc = "Axial tension damping."
    )

    float newton:rod:compressStiffness = 100000.0 (
        doc = "Axial compression stiffness."
    )

    float newton:rod:compressDamping = 0.0 (
        doc = "Axial compression damping."
    )

    float newton:rod:bendYStiffness = 1000.0 (
        doc = "Bend stiffness around local X."
    )

    float newton:rod:bendYDamping = 0.0 (
        doc = "Bend damping around local X."
    )

    float newton:rod:bendZStiffness = 1000.0 (
        doc = "Bend stiffness around local Y."
    )

    float newton:rod:bendZDamping = 0.0 (
        doc = "Bend damping around local Y."
    )

    float newton:rod:torsionStiffness = 1000.0 (
        doc = "Torsion stiffness around local Z."
    )

    float newton:rod:torsionDamping = 0.0 (
        doc = "Torsion damping around local Z."
    )
}
```

Important:

- do **not** add a parallel `NewtonCableAPI` / `NewtonCableMaterialAPI`
- use the existing `rod` naming as the schema surface
- do **not** reintroduce `newton:rod:youngsModulus`

### Step 2 - Update `plugInfo.json`

Path:

- `newton/_src/usd/schemas_ext/plugInfo.json`

Add `NewtonPhysicsRodAPI` and `NewtonPhysicsRodMaterialAPI`, and tighten
`NewtonPhysicsDeformableAPI` to match the other repos.

Recommended entries:

```json
"NewtonPhysicsDeformableAPI": {
  "schemaIdentifier": "NewtonDeformableAPI",
  "alias": { "UsdSchemaBase": "NewtonDeformableAPI" },
  "autoGenerated": false,
  "bases": [ "UsdAPISchemaBase" ],
  "schemaKind": "singleApplyAPI",
  "apiSchemaCanOnlyApplyTo": [ "Xform", "Gprim", "Scope" ]
},
"NewtonPhysicsRodAPI": {
  "schemaIdentifier": "NewtonRodAPI",
  "alias": { "UsdSchemaBase": "NewtonRodAPI" },
  "autoGenerated": false,
  "bases": [ "UsdAPISchemaBase" ],
  "schemaKind": "singleApplyAPI",
  "apiSchemaCanOnlyApplyTo": [ "BasisCurves", "Xform", "Scope" ]
},
"NewtonPhysicsRodMaterialAPI": {
  "schemaIdentifier": "NewtonRodMaterialAPI",
  "alias": { "UsdSchemaBase": "NewtonRodMaterialAPI" },
  "autoGenerated": false,
  "bases": [ "UsdAPISchemaBase" ],
  "schemaKind": "singleApplyAPI",
  "apiSchemaCanOnlyApplyTo": [ "Material" ]
}
```

Keep the current plugin-loading path in:

- `newton/_src/usd/__init__.py`

No plugin-package split is required for this repo right now.

### Step 3 - Create `newton/_src/palatial/cable.py`

This remains the read-side helper module for the palatial loader, but it should
read `newton:rod:*` fields.

Recommended responsibilities:

```python
DEFAULTS = {
    "frameDefinition": "parallelTransport",
    "closed": False,
    "crossSectionType": "roundSolid",
    "radius": 0.005,
    "width": None,
    "thickness": None,
    "segmentCount": 16,
    "length": 1.0,
    "dropHeight": 0.3,
    "twistTotal": 0.0,
    "density": 1000.0,
    "stretchStiffness": 1.0e5,
    "stretchDamping": 0.0,
    "compressStiffness": 1.0e5,
    "compressDamping": 0.0,
    "bendYStiffness": 1.0e3,
    "bendYDamping": 0.0,
    "bendZStiffness": 1.0e3,
    "bendZDamping": 0.0,
    "torsionStiffness": 1.0e3,
    "torsionDamping": 0.0,
}


def find_cable_prim_path(usd_path: str) -> str | None:
    """Return the root prim path of the first rod/cable asset."""


def find_cable_centerline_prim_path(usd_path: str) -> str | None:
    """Return the BasisCurves centerline prim path for the first cable."""


def _bound_cable_material_prims(prim: Usd.Prim) -> list[Usd.Prim]:
    """Return bound Material prims carrying NewtonRodMaterialAPI."""


def read_cable_params(usd_path: str) -> dict:
    """Resolve rod/cable params from root + centerline + bound Material.

    Loader fallback behavior:
    - prefer newton:rod:* on bound Material / root / centerline,
    - keep local support for newton:rod:dropHeight and newton:rod:twistTotal,
    - accept newton:rodMaterial:* and newton:rod:isClosed as compatibility
      fallbacks when present.
    """


def extract_cable_points(usd_path: str) -> list[wp.vec3] | None:
    """Read centerline points from the cable's BasisCurves prim."""
```

### Step 4 - Update `load.py`

Path:

- `newton/_src/palatial/load.py`

#### 4a. Detect cable body type from rod schema

Teach `_detect_body_type()` to treat either of these as cable/rod:

- `NewtonRodAPI` applied
- `newton:deformable:simulationIntent == "rod"`

The internal body type string can stay `"cable"` in this repo.

#### 4b. Add `_build_cable(...)`

Build path for the current stock Newton runtime:

1. Read:
   - root attrs via `read_cable_params()`
   - `BasisCurves` points via `extract_cable_points()`
2. If the USD has no authored points:
   - fall back to a straight cable with `segmentCount`, `length`, and
     `dropHeight`
3. Generate segment quaternions with:
   - `newton.utils.create_parallel_transport_cable_quaternions()`
   - pass `twist_total=newton:rod:twistTotal`
4. Lower into stock `builder.add_rod(...)`

The intended split is:

- authored `BasisCurves` points remain the primary geometric source,
- `dropHeight` is only a procedural fallback convenience,
- `twistTotal` is a loader-side framing convenience,
- neither field should replace the explicit anisotropic rod material attrs.

#### 4c. Intermediate anisotropic validation in stock `add_rod()`

This repo's current target is not just "loader collapse" in isolation.
As an intermediate validation step, the stock builder path itself should accept
anisotropic cable arguments, even before the full subtree runtime port lands.

Recommended phase-A builder surface:

```python
builder.add_rod(
    positions=points,
    quaternions=edge_q,
    radius=radius,
    stretch_stiffness=...,
    stretch_damping=...,
    bend_stiffness=...,          # existing isotropic arg remains supported
    bend_damping=...,            # existing isotropic arg remains supported
    bend_y_stiffness=...,        # new intermediate arg
    bend_y_damping=...,          # new intermediate arg
    bend_z_stiffness=...,        # new intermediate arg
    bend_z_damping=...,          # new intermediate arg
    torsion_stiffness=...,       # new intermediate arg
    torsion_damping=...,         # new intermediate arg
    closed=...,
    label=...,
)
```

Phase-A behavior inside stock `add_rod()`:

- if only legacy isotropic args are provided, preserve current behavior,
- if anisotropic args are provided, accept them and collapse them
  deterministically to the current isotropic cable joint path,
- do not silently ignore the new args.

Suggested stock collapse rule for phase A:

```python
effective_bend_stiffness = (
    bend_stiffness
    if bend_stiffness is not None
    else 0.5 * (bend_y_stiffness + bend_z_stiffness)
)

effective_bend_damping = max(
    bend_damping or 0.0,
    bend_y_damping or 0.0,
    bend_z_damping or 0.0,
    torsion_damping or 0.0,
)
```

This is intentionally only a validation approximation. It gives us:

- stable schema-to-builder mapping now,
- minimal churn at callsites later,
- a clean stepping stone toward the true subtree-backed anisotropic runtime.

For the stock phase, collapse anisotropic material to isotropic rod inputs the
same way the related repo's `asset_example/runner_core.py` does:

```python
    stretch_stiffness = float(p["stretchStiffness"])
    stretch_damping = float(p["stretchDamping"])
    bend_y_stiffness = float(p["bendYStiffness"])
    bend_y_damping = float(p["bendYDamping"])
    bend_z_stiffness = float(p["bendZStiffness"])
    bend_z_damping = float(p["bendZDamping"])
    torsion_stiffness = float(p["torsionStiffness"])
    torsion_damping = float(p["torsionDamping"])

    bend_stiffness = 0.5 * (bend_y_stiffness + bend_z_stiffness)
    bend_damping = max(
        bend_y_damping,
        bend_z_damping,
        torsion_damping,
        stretch_damping,
    )
```

Radius handling for the stock phase:

- use `newton:rod:radius` when present,
- if `crossSectionType == "flatRect"` and only `thickness` is authored, use
  `radius = 0.5 * thickness` as an approximation.

Suggested skeleton:

```python
def _build_cable(usd_path: str, *, device: str | None = None,
                 solver_name: str | None = None) -> Any:
    from .cable import extract_cable_points, read_cable_params

    p = read_cable_params(usd_path)
    points = extract_cable_points(usd_path)

    if not points:
        points = newton.utils.create_straight_cable_points(
            start=wp.vec3(0.0, 0.0, 0.0),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=float(p["length"]),
            num_segments=int(p["segmentCount"]),
        )

    edge_q = newton.utils.create_parallel_transport_cable_quaternions(points)

    radius = (
        float(p["radius"])
        if p["radius"] is not None
        else 0.5 * float(p["thickness"])
    )

    stretch_stiffness = float(p["stretchStiffness"])
    stretch_damping = float(p["stretchDamping"])
    bend_y_stiffness = float(p["bendYStiffness"])
    bend_y_damping = float(p["bendYDamping"])
    bend_z_stiffness = float(p["bendZStiffness"])
    bend_z_damping = float(p["bendZDamping"])
    torsion_stiffness = float(p["torsionStiffness"])
    torsion_damping = float(p["torsionDamping"])

    bend_stiffness = 0.5 * (bend_y_stiffness + bend_z_stiffness)
    bend_damping = max(
        bend_y_damping,
        bend_z_damping,
        torsion_damping,
        stretch_damping,
    )

    with wp.ScopedDevice(device) if device else wp.ScopedDevice(wp.get_preferred_device()):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_rod(
            positions=points,
            quaternions=edge_q,
            radius=radius,
            stretch_stiffness=stretch_stiffness,
            stretch_damping=stretch_damping,
            bend_stiffness=bend_stiffness,
            bend_damping=bend_damping,
            bend_y_stiffness=bend_y_stiffness,
            bend_y_damping=bend_y_damping,
            bend_z_stiffness=bend_z_stiffness,
            bend_z_damping=bend_z_damping,
            torsion_stiffness=torsion_stiffness,
            torsion_damping=torsion_damping,
            closed=bool(p["closed"]),
            label="cable_0",
        )
        try:
            builder.color()
        except Exception:
            pass
        return builder.finalize()
```

#### 4d. Loader dispatch and solver default

Update `load()` so:

```python
if body_type == "cloth":
    model = _build_cloth(...)
elif body_type == "cable":
    model = _build_cable(...)
else:
    model = _build_rigid(...)
```

For default solver selection, cable should follow the cloth rule of preferring
VBD when the scene did not pin a solver:

```python
if body_type in ("cloth", "cable") and not scene_pinned and not solver_override:
    if body_type == "cloth" and style3d_used and getattr(newton.solvers, "SolverStyle3D", None):
        solver_name = "style3d"
    elif getattr(newton.solvers, "SolverVBD", None):
        solver_name = "vbd"
    else:
        solver_name = "xpbd"
```

Also update `NewtonBundle.body_type` comments to include `"cable"`.

### Step 5 - Export the new helpers publicly

Update both:

- `newton/_src/palatial/__init__.py`
- `newton/palatial.py`

The public facade matters. In this repo, users import from `newton.palatial`,
not from `_src.palatial`.

Recommended exports:

```python
from ._src.palatial.cable import (
    find_cable_prim_path,
    find_cable_centerline_prim_path,
    read_cable_params,
)
```

---

## 5. Tests to Reuse

### Schema registration tests

Mirror the rod schema tests from the related repos:

- create an in-memory stage,
- define an `Xform` root and a `BasisCurves` centerline,
- apply `NewtonDeformableAPI` and `NewtonRodAPI`,
- define a `Material`, apply `NewtonRodMaterialAPI`,
- assert authored attrs round-trip.

This is the best template:

- `third_party/newton/tests/test_schema_resolver.py`
  `TestRegisteredRodSchemas`

### Palatial loader tests

Add a current-repo test that:

1. writes a tiny USDA with:
   - root `Xform`
   - child `BasisCurves`
   - bound `Material`
2. calls:
   - `find_cable_prim_path()`
   - `extract_cable_points()`
   - `read_cable_params()`
3. builds the model through `newton.palatial.load()`
4. verifies:
   - `bundle.body_type == "cable"`
   - rod bodies were created
   - first/last segment count matches `segmentCount`

### Important behavior to test explicitly

1. `simulationIntent == "rod"` triggers cable detection
2. `BasisCurves` point extraction works
3. bound material lookup works through `MaterialBindingAPI`
4. `flatRect` + `thickness` falls back to stock radius approximation
5. procedural fallback centerline generation honors `dropHeight`
6. quaternion generation honors `twistTotal`
7. stock runtime collapse from:
   - `bendYStiffness`
   - `bendZStiffness`
   - `torsionDamping`
   into stock `add_rod()` args is deterministic

---

## 6. Deliberate Non-Goals for This Change

This guide is intentionally **not** asking for:

- direct `add_usd()` rod import support,
- a vendored `newton_usd_schemas` package split,
- a separate `NewtonCableAPI` namespace,
- a second parallel cable schema family,
- a revived `newton:rod:youngsModulus` field.

This guide also does **not** require the full subtree runtime port in the first
patch. The intermediate target is narrower:

- stock `add_rod()` accepts anisotropic inputs,
- palatial schema/loader maps USD into those inputs,
- the underlying runtime may still collapse to isotropic cable behavior.

Those can all happen later if needed.

For now, the most compatible path is:

1. reuse `NewtonDeformableAPI` + `NewtonRodAPI` + `NewtonRodMaterialAPI`,
2. read that schema in `newton/_src/palatial/cable.py`,
3. extend stock `builder.add_rod()` so it can accept anisotropic cable fields,
4. lower those fields through a deterministic phase-A collapse,
5. preserve the richer anisotropic fields in USD so a later subtree runtime
   port does not require another schema redesign.

---

## 7. Best Source Files to Follow

For this guide, the most useful concrete references are:

### Current repo

- `newton/_src/usd/schemas_ext/generatedSchema.usda`
- `newton/_src/usd/schemas_ext/plugInfo.json`
- `newton/_src/palatial/shell.py`
- `newton/_src/palatial/load.py`
- `newton/palatial.py`

### Related repos

- `asset_example/usd_asset.py`
  - concrete authoring of `NewtonDeformableAPI`, `NewtonRodAPI`,
    `NewtonRodMaterialAPI`
- `asset_example/runner_core.py`
  - concrete read-side consumption of `BasisCurves` and `newton:rod:*`
- `third_party/newton/tests/test_schema_resolver.py`
  - schema registration test pattern
- `docs/usd/current_usd_schema.md`
  - current status and migration-layer notes

If there is any conflict between older cable-only notes and the related repos'
rod schema, prefer the rod schema.

One more practical note:

- the `third_party/newton` changes in the related repos came in through
  `git subtree`

So when later porting the true anisotropic runtime pieces, compare and transplant
changes with the subtree layout in mind rather than treating them as unrelated
handwritten fork files.
