"""Read-side helpers for NewtonRodAPI rod or cable assets."""
from __future__ import annotations

# `import newton` registers the bundled USD plugins via
# newton/_src/usd/__init__.py. Must precede any pxr.Usd usage in the same
# process.
import newton  # noqa: F401

from . import _resolvers  # noqa: F401  (kept for parity with shell.py init)

import warp as wp
from pxr import Usd, UsdGeom, UsdShade


DEFAULTS = {
    "frameDefinition": "parallelTransport",
    "closed": False,
    "crossSectionType": "roundSolid",
    "radius": 0.005,
    "width": 0.01,
    "thickness": 0.002,
    "segmentCount": 16,
    "verticesPerSegment": 2,
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


def _applied_schema_tokens(prim: Usd.Prim) -> set[str]:
    """Return applied schema tokens, including raw listOp metadata."""
    applied = set(prim.GetAppliedSchemas())
    raw = prim.GetMetadata("apiSchemas")
    if raw is not None:
        for items_attr in ("prependedItems", "appendedItems", "explicitItems"):
            for tok in getattr(raw, items_attr, []) or []:
                applied.add(str(tok))
    return applied


def _has_rod_api(prim: Usd.Prim) -> bool:
    """True iff the prim carries NewtonRodAPI."""
    return "NewtonRodAPI" in _applied_schema_tokens(prim)


def _has_rod_intent(prim: Usd.Prim) -> bool:
    """True iff the prim declares deformable intent 'rod'."""
    a = prim.GetAttribute("newton:deformable:simulationIntent")
    return bool(a and a.HasAuthoredValue() and str(a.Get()) == "rod")


def find_cable_prim_path(usd_path: str) -> str | None:
    """Return the prim path of the first rod or cable root on the stage."""
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return None

    root_candidate = None
    curves_candidate = None
    for prim in stage.Traverse():
        is_rod = _has_rod_api(prim) or _has_rod_intent(prim)
        if not is_rod:
            continue
        if prim.IsA(UsdGeom.BasisCurves):
            curves_candidate = curves_candidate or prim
            continue
        root_candidate = prim
        break

    if root_candidate is not None:
        return root_candidate.GetPath().pathString
    if curves_candidate is None:
        return None

    parent = curves_candidate.GetParent()
    if parent and parent.IsValid() and parent.GetTypeName() in ("Xform", "Scope"):
        return parent.GetPath().pathString
    return curves_candidate.GetPath().pathString


def find_cable_centerline_prim_path(usd_path: str) -> str | None:
    """Return the prim path of the first rod or cable BasisCurves centerline."""
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return None

    cable_path = find_cable_prim_path(usd_path)
    cable_root = stage.GetPrimAtPath(cable_path) if cable_path else None
    if cable_root and cable_root.IsValid():
        root_path = cable_root.GetPath()
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.BasisCurves) and prim.GetPath().HasPrefix(root_path):
                return prim.GetPath().pathString

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.BasisCurves) and (_has_rod_api(prim) or _has_rod_intent(prim)):
            return prim.GetPath().pathString
    return None


def _bound_cable_material_prims(prim: Usd.Prim) -> list[Usd.Prim]:
    """Return bound Material prims that carry NewtonRodMaterialAPI."""
    if not prim or not prim.IsValid():
        return []

    binding = UsdShade.MaterialBindingAPI(prim)
    candidates: list[Usd.Prim] = []
    for purpose in ("", "physics"):
        try:
            result = binding.ComputeBoundMaterial(purpose) if purpose else binding.ComputeBoundMaterial()
        except Exception:
            result = None
        if result is None:
            continue
        material = result[0] if isinstance(result, tuple) else result
        if material and material.GetPrim().IsValid():
            candidates.append(material.GetPrim())

    out: list[Usd.Prim] = []
    seen = set()
    for material_prim in candidates:
        path = material_prim.GetPath()
        if path in seen:
            continue
        seen.add(path)
        if "NewtonRodMaterialAPI" in _applied_schema_tokens(material_prim):
            out.append(material_prim)
    return out


def read_cable_params(usd_path: str) -> dict[str, object]:
    """Resolve rod or cable params to a normalized dict."""
    out: dict[str, object] = dict(DEFAULTS)
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return out

    cable_path = find_cable_prim_path(usd_path)
    centerline_path = find_cable_centerline_prim_path(usd_path)
    if not cable_path:
        return out

    root = stage.GetPrimAtPath(cable_path)
    centerline = stage.GetPrimAtPath(centerline_path) if centerline_path else None
    geometry_sources = [prim for prim in (root, centerline) if prim and prim.IsValid()]

    material_sources: list[Usd.Prim] = []
    seen = set()
    for prim in geometry_sources:
        for material_prim in _bound_cable_material_prims(prim):
            path = material_prim.GetPath()
            if path in seen:
                continue
            seen.add(path)
            material_sources.append(material_prim)
    for prim in geometry_sources:
        path = prim.GetPath()
        if path in seen:
            continue
        seen.add(path)
        material_sources.append(prim)

    def _walk_optional(sources: list[Usd.Prim], *names: str) -> object | None:
        for prim in sources:
            for name in names:
                attr = prim.GetAttribute(name)
                if attr and attr.HasAuthoredValue():
                    return attr.Get()
        return None

    def _walk_float(sources: list[Usd.Prim], *names: str, default: float) -> float:
        value = _walk_optional(sources, *names)
        return default if value is None else float(value)

    def _walk_int(sources: list[Usd.Prim], *names: str, default: int) -> int:
        value = _walk_optional(sources, *names)
        return default if value is None else int(value)

    def _walk_bool(sources: list[Usd.Prim], *names: str, default: bool) -> bool:
        value = _walk_optional(sources, *names)
        return default if value is None else bool(value)

    def _walk_token(sources: list[Usd.Prim], *names: str, default: str) -> str:
        value = _walk_optional(sources, *names)
        return default if value is None else str(value)

    out["frameDefinition"] = _walk_token(
        geometry_sources,
        "newton:rod:frameDefinition",
        default=DEFAULTS["frameDefinition"],
    )
    out["closed"] = _walk_bool(
        geometry_sources,
        "newton:rod:closed",
        "newton:rod:isClosed",
        default=DEFAULTS["closed"],
    )
    out["crossSectionType"] = _walk_token(
        geometry_sources,
        "newton:rod:crossSectionType",
        default=DEFAULTS["crossSectionType"],
    )
    out["width"] = _walk_float(
        geometry_sources,
        "newton:rod:width",
        default=DEFAULTS["width"],
    )
    out["thickness"] = _walk_float(
        geometry_sources,
        "newton:rod:thickness",
        default=DEFAULTS["thickness"],
    )
    out["segmentCount"] = _walk_int(
        geometry_sources,
        "newton:rod:segmentCount",
        default=DEFAULTS["segmentCount"],
    )
    out["verticesPerSegment"] = _walk_int(
        geometry_sources,
        "newton:rod:verticesPerSegment",
        default=DEFAULTS["verticesPerSegment"],
    )
    out["length"] = _walk_float(
        geometry_sources,
        "newton:rod:length",
        default=DEFAULTS["length"],
    )
    out["dropHeight"] = _walk_float(
        geometry_sources,
        "newton:rod:dropHeight",
        default=DEFAULTS["dropHeight"],
    )
    out["twistTotal"] = _walk_float(
        geometry_sources,
        "newton:rod:twistTotal",
        default=DEFAULTS["twistTotal"],
    )

    radius = _walk_optional(geometry_sources, "newton:rod:radius")
    if radius is None and out["crossSectionType"] == "flatRect":
        out["radius"] = 0.5 * float(out["thickness"])
    else:
        out["radius"] = DEFAULTS["radius"] if radius is None else float(radius)

    legacy_bend_stiffness = _walk_optional(material_sources, "newton:rodMaterial:bendStiffness")
    legacy_damping = _walk_optional(material_sources, "newton:rodMaterial:damping")

    out["density"] = _walk_float(
        material_sources,
        "newton:rod:density",
        "newton:rodMaterial:density",
        default=DEFAULTS["density"],
    )
    out["stretchStiffness"] = _walk_float(
        material_sources,
        "newton:rod:stretchStiffness",
        "newton:rodMaterial:stretchStiffness",
        default=DEFAULTS["stretchStiffness"],
    )
    out["stretchDamping"] = _walk_float(
        material_sources,
        "newton:rod:stretchDamping",
        default=float(legacy_damping) if legacy_damping is not None else DEFAULTS["stretchDamping"],
    )
    out["compressStiffness"] = _walk_float(
        material_sources,
        "newton:rod:compressStiffness",
        default=DEFAULTS["compressStiffness"],
    )
    out["compressDamping"] = _walk_float(
        material_sources,
        "newton:rod:compressDamping",
        default=DEFAULTS["compressDamping"],
    )
    out["bendYStiffness"] = _walk_float(
        material_sources,
        "newton:rod:bendYStiffness",
        default=float(legacy_bend_stiffness) if legacy_bend_stiffness is not None else DEFAULTS["bendYStiffness"],
    )
    out["bendYDamping"] = _walk_float(
        material_sources,
        "newton:rod:bendYDamping",
        default=float(legacy_damping) if legacy_damping is not None else DEFAULTS["bendYDamping"],
    )
    out["bendZStiffness"] = _walk_float(
        material_sources,
        "newton:rod:bendZStiffness",
        default=float(legacy_bend_stiffness) if legacy_bend_stiffness is not None else DEFAULTS["bendZStiffness"],
    )
    out["bendZDamping"] = _walk_float(
        material_sources,
        "newton:rod:bendZDamping",
        default=float(legacy_damping) if legacy_damping is not None else DEFAULTS["bendZDamping"],
    )
    out["torsionStiffness"] = _walk_float(
        material_sources,
        "newton:rod:torsionStiffness",
        default=float(legacy_bend_stiffness) if legacy_bend_stiffness is not None else DEFAULTS["torsionStiffness"],
    )
    out["torsionDamping"] = _walk_float(
        material_sources,
        "newton:rod:torsionDamping",
        "newton:rodMaterial:damping",
        default=float(legacy_damping) if legacy_damping is not None else DEFAULTS["torsionDamping"],
    )
    out["intent"] = _walk_token(
        geometry_sources,
        "newton:deformable:simulationIntent",
        default="rod",
    )

    return out


def extract_cable_points(usd_path: str) -> list[wp.vec3]:
    """Extract authored centerline points from a rod or cable BasisCurves prim."""
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return []

    centerline_path = find_cable_centerline_prim_path(usd_path)
    if not centerline_path:
        return []

    centerline_prim = stage.GetPrimAtPath(centerline_path)
    if not centerline_prim or not centerline_prim.IsValid():
        return []

    basis_curves = UsdGeom.BasisCurves(centerline_prim)
    points = basis_curves.GetPointsAttr().Get()
    if points is None:
        return []

    return [wp.vec3(float(point[0]), float(point[1]), float(point[2])) for point in points]
