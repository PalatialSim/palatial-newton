# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read-side helpers for legacy ``newton:bodyType="cloth"`` USDAs."""
from __future__ import annotations

import newton  # noqa: F401

from pxr import Usd, UsdGeom

import numpy as np


CLOTH_BODY_TOKEN = "cloth"


def find_cloth_prim_path(usd_path: str) -> str | None:
    """Return the first mesh prim path tagged ``newton:bodyType="cloth"``."""
    stage = Usd.Stage.Open(usd_path)
    for prim in stage.Traverse():
        attr = prim.GetAttribute("newton:bodyType")
        if attr and attr.HasAuthoredValue() and str(attr.Get()) == CLOTH_BODY_TOKEN:
            return prim.GetPath().pathString
    return None


def read_cloth_params(usd_path: str) -> dict:
    """Read legacy ``newton:cloth:density`` / ``dropHeight`` if present."""
    out = {}
    stage = Usd.Stage.Open(usd_path)
    for prim in stage.Traverse():
        if prim.GetAttribute("newton:bodyType") and prim.GetAttribute("newton:bodyType").Get() == CLOTH_BODY_TOKEN:
            d = prim.GetAttribute("newton:cloth:density")
            h = prim.GetAttribute("newton:cloth:dropHeight")
            if d and d.HasAuthoredValue(): out["density"] = float(d.Get())
            if h and h.HasAuthoredValue(): out["drop_height"] = float(h.Get())
            break
    return out


def _extract_first_mesh(usd_path: str):
    """Return ``(vertices, tri_indices)`` from the first mesh, in Newton world coords."""
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Cannot open USD: {usd_path}")
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    up_axis = str(UsdGeom.GetStageUpAxis(stage) or "Y")
    mesh = None
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            break
    if mesh is None:
        raise RuntimeError("No UsdGeom.Mesh found on stage.")

    pts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    idx = mesh.GetFaceVertexIndicesAttr().Get()
    if pts is None or counts is None or idx is None:
        raise RuntimeError("Mesh missing points / face data.")

    arr = np.asarray([(float(p[0]), float(p[1]), float(p[2])) for p in pts], dtype=np.float32)
    arr *= mpu
    if up_axis == "Y":
        arr = np.stack([arr[:, 0], -arr[:, 2], arr[:, 1]], axis=1)
    arr[:, 0] -= float(arr[:, 0].mean())
    arr[:, 1] -= float(arr[:, 1].mean())
    arr[:, 2] -= float(arr[:, 2].min())
    verts = [(float(v[0]), float(v[1]), float(v[2])) for v in arr]

    tri = []
    cursor = 0
    for c in counts:
        if c < 3:
            cursor += c
            continue
        a = idx[cursor]
        for k in range(1, c - 1):
            tri.append(int(a))
            tri.append(int(idx[cursor + k]))
            tri.append(int(idx[cursor + k + 1]))
        cursor += c
    return verts, tri
