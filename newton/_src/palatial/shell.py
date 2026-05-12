"""Read-side helpers for the NewtonShellAPI / NewtonClothAPI schema.

Authoring lives in the converter package (palatial-sim-gen-isaac). This
module only resolves cloth/shell parameters off a converted USD so the
loader can build a Newton model.
"""
from __future__ import annotations

# `import newton` registers the bundled USD plugins (newton + newton_shell)
# via newton/_src/usd/__init__.py. Must precede any pxr.Usd usage in the
# same process.
import newton  # noqa: F401

from . import _resolvers  # noqa: F401  (kept for parity with cloth.py init)

from pxr import Usd, UsdGeom, UsdShade


# Default values (mirrored from the schema plugin's generatedSchema.usda
# so the read path always returns a complete dict even when the source
# USD omits explicit attribute values).
DEFAULTS = {
    # geometry (NewtonShellAPI on Mesh)
    "thickness":         1e-3,
    "particleRadius":    0.01,
    "addBendingEdges":   True,
    "dropHeight":        1.0,
    # material (NewtonShellMaterialAPI on Material)
    "density":           300.0,
    "triStiffness":      1.0e2,
    "triAreaStiffness":  1.0e2,
    "triDamping":        0.0,
    "triDrag":           0.0,
    "triLift":           0.0,
    "bendStiffness":     1.0e-3,
    "bendDamping":       0.0,
}


def _has_shell_api(prim: Usd.Prim) -> bool:
    """True iff the prim carries NewtonShellAPI or NewtonClothAPI.

    Checks both `GetAppliedSchemas()` and the raw `apiSchemas` listOp so it
    works whether or not the plugin is registered in the current process.
    """
    applied = set(prim.GetAppliedSchemas())
    raw = prim.GetMetadata("apiSchemas")
    if raw is not None:
        for items_attr in ("prependedItems", "appendedItems", "explicitItems"):
            for tok in getattr(raw, items_attr, []) or []:
                applied.add(str(tok))
    return "NewtonShellAPI" in applied or "NewtonClothAPI" in applied


def find_shell_prim_path(usd_path: str) -> str | None:
    """Return the prim path of the first cloth/shell mesh on the stage.

    Preference order:
      1. Mesh with NewtonShellAPI / NewtonClothAPI applied (new schema).
      2. Mesh tagged with the legacy `newton:bodyType="cloth"` marker.
    """
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return None
    found_new = None
    found_legacy = None
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if found_new is None and _has_shell_api(prim):
            found_new = prim.GetPath().pathString
            break
        if found_legacy is None:
            a = prim.GetAttribute("newton:bodyType")
            if a and a.HasAuthoredValue() and str(a.Get()) == "cloth":
                found_legacy = prim.GetPath().pathString
    return found_new or found_legacy


def _bound_shell_material_prims(mesh_prim: Usd.Prim) -> list[Usd.Prim]:
    """Return Material prims bound to the mesh that carry NewtonShellMaterialAPI.

    Walks the all-purpose binding plus the `physics` purpose binding (the
    purpose used by the converter when it creates `<defaultPrim>/cloth_material`).
    """
    out: list[Usd.Prim] = []
    binding = UsdShade.MaterialBindingAPI(mesh_prim)
    candidates = []
    for purpose in ("", "physics"):
        try:
            r = binding.ComputeBoundMaterial(purpose) if purpose else binding.ComputeBoundMaterial()
        except Exception:
            r = None
        if r is None:
            continue
        # ComputeBoundMaterial returns either a Material or a (Material, relationship) tuple.
        mat = r[0] if isinstance(r, tuple) else r
        if mat and mat.GetPrim().IsValid():
            candidates.append(mat.GetPrim())
    seen = set()
    for p in candidates:
        if p.GetPath() in seen:
            continue
        seen.add(p.GetPath())
        applied = set(p.GetAppliedSchemas())
        raw = p.GetMetadata("apiSchemas")
        if raw is not None:
            for items_attr in ("prependedItems", "appendedItems", "explicitItems"):
                for tok in getattr(raw, items_attr, []) or []:
                    applied.add(str(tok))
        if "NewtonShellMaterialAPI" in applied:
            out.append(p)
    return out


def read_shell_params(usd_path: str) -> dict:
    """Resolve all cloth/shell params to a normalized dict.

    Reads new `newton:shell:*` attrs from the cloth mesh + bound material
    first; falls back to legacy `newton:cloth:*` attrs for older USDAs.
    Missing values fall back to DEFAULTS, so the result always contains
    every key.
    """
    out = dict(DEFAULTS)
    stage = Usd.Stage.Open(usd_path)
    cloth_path = find_shell_prim_path(usd_path)
    if not cloth_path:
        return out
    mesh = stage.GetPrimAtPath(cloth_path)
    materials = _bound_shell_material_prims(mesh)
    sources = [mesh, *materials]

    def _walk(*names, default):
        for prim in sources:
            for n in names:
                a = prim.GetAttribute(n)
                if a and a.HasAuthoredValue():
                    return a.Get()
        return default

    # Geometry (mesh-only).
    out["thickness"]       = float(_walk("newton:shell:thickness",       default=DEFAULTS["thickness"]))
    out["particleRadius"]  = float(_walk("newton:shell:particleRadius",  default=DEFAULTS["particleRadius"]))
    out["addBendingEdges"] = bool (_walk("newton:shell:addBendingEdges", default=DEFAULTS["addBendingEdges"]))
    out["dropHeight"]      = float(_walk("newton:shell:dropHeight",
                                          "newton:cloth:dropHeight",
                                          default=DEFAULTS["dropHeight"]))
    # Material (Material + legacy-on-mesh fallback).
    out["density"]          = float(_walk("newton:shell:density",
                                           "newton:cloth:density",         default=DEFAULTS["density"]))
    out["triStiffness"]     = float(_walk("newton:shell:triStiffness",
                                           "newton:cloth:triKe",           default=DEFAULTS["triStiffness"]))
    out["triAreaStiffness"] = float(_walk("newton:shell:triAreaStiffness",
                                           "newton:cloth:triKa",           default=DEFAULTS["triAreaStiffness"]))
    out["triDamping"]       = float(_walk("newton:shell:triDamping",
                                           "newton:cloth:triKd",           default=DEFAULTS["triDamping"]))
    out["triDrag"]          = float(_walk("newton:shell:triDrag",          default=DEFAULTS["triDrag"]))
    out["triLift"]          = float(_walk("newton:shell:triLift",          default=DEFAULTS["triLift"]))
    out["bendStiffness"]    = float(_walk("newton:shell:bendStiffness",
                                           "newton:cloth:edgeKe",          default=DEFAULTS["bendStiffness"]))
    out["bendDamping"]      = float(_walk("newton:shell:bendDamping",
                                           "newton:cloth:edgeKd",          default=DEFAULTS["bendDamping"]))
    # Optional extensions (None if unauthored).
    def _opt_vec3(*names):
        for prim in sources:
            for n in names:
                a = prim.GetAttribute(n)
                if a and a.HasAuthoredValue():
                    v = a.Get()
                    return (float(v[0]), float(v[1]), float(v[2]))
        return None

    out["style3dTriAnisoKe"]  = _opt_vec3("newton:shell:style3d:triAnisoKe")
    out["style3dEdgeAnisoKe"] = _opt_vec3("newton:shell:style3d:edgeAnisoKe")
    out["vbdSelfContactRadius"]            = _walk("newton:shell:vbd:selfContactRadius",            default=None)
    out["vbdSelfContactMargin"]            = _walk("newton:shell:vbd:selfContactMargin",            default=None)
    out["vbdConservativeBoundRelaxation"]  = _walk("newton:shell:vbd:conservativeBoundRelaxation",  default=None)

    # Intent (advisory).
    a = mesh.GetAttribute("newton:deformable:simulationIntent")
    out["intent"] = str(a.Get()) if a and a.HasAuthoredValue() else "cloth"

    del mesh, stage
    return out
