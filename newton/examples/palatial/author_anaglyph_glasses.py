# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Author subtle red/green transparent lenses on a two-lens USD glasses asset.

The input asset may keep both lenses in one mesh. This utility finds the two
connected mesh components, writes each as an independent visual mesh, binds
portable ``UsdPreviewSurface`` materials, and hides the original combined
visual. Collision geometry and rigid-body schemas are left untouched.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
except ImportError as error:  # pragma: no cover - exercised by the CLI error path
    raise ImportError("Anaglyph USD authoring requires the optional USD dependencies") from error


DEFAULT_LENS_MESH = "/World/part_3/part_3_NewtonVisual"
DEFAULT_RED_COLOR = (0.75, 0.04, 0.03)
DEFAULT_GREEN_COLOR = (0.02, 0.58, 0.08)
DEFAULT_FRAME_COLOR = (0.004, 0.005, 0.006)


def _face_components(face_indices: np.ndarray, point_count: int) -> list[np.ndarray]:
    """Return face-index arrays grouped by shared-vertex connectivity."""
    parent = np.arange(point_count, dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for face in face_indices:
        anchor = int(face[0])
        for vertex in face[1:]:
            union(anchor, int(vertex))

    grouped: dict[int, list[int]] = {}
    for face_index, face in enumerate(face_indices):
        root = find(int(face[0]))
        grouped.setdefault(root, []).append(face_index)
    return [np.asarray(indices, dtype=np.int32) for indices in grouped.values()]


def _copy_mesh_component(
    stage: Usd.Stage,
    source: UsdGeom.Mesh,
    target_path: Sdf.Path,
    face_indices: np.ndarray,
) -> UsdGeom.Mesh:
    """Copy one triangle component into a compact sibling mesh."""
    points = np.asarray(source.GetPointsAttr().Get(), dtype=np.float32)
    source_indices = np.asarray(source.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    selected = source_indices[face_indices]
    used_points, inverse = np.unique(selected.reshape(-1), return_inverse=True)

    target = UsdGeom.Mesh.Define(stage, target_path)
    target.CreatePointsAttr(points[used_points])
    target.CreateFaceVertexCountsAttr(np.full(len(selected), 3, dtype=np.int32))
    target.CreateFaceVertexIndicesAttr(inverse.astype(np.int32))
    target.CreateSubdivisionSchemeAttr(source.GetSubdivisionSchemeAttr().Get() or UsdGeom.Tokens.none)
    target.CreateOrientationAttr(source.GetOrientationAttr().Get() or UsdGeom.Tokens.rightHanded)
    target.CreateDoubleSidedAttr(bool(source.GetDoubleSidedAttr().Get()))
    bounds = points[used_points]
    target.CreateExtentAttr(
        [
            Gf.Vec3f(*bounds.min(axis=0).tolist()),
            Gf.Vec3f(*bounds.max(axis=0).tolist()),
        ]
    )

    normals = source.GetNormalsAttr().Get()
    if normals is not None:
        normals_np = np.asarray(normals, dtype=np.float32)
        interpolation = source.GetNormalsInterpolation()
        if interpolation == UsdGeom.Tokens.vertex and len(normals_np) == len(points):
            normals_np = normals_np[used_points]
        elif interpolation == UsdGeom.Tokens.faceVarying and len(normals_np) == len(source_indices) * 3:
            normals_np = normals_np.reshape(-1, 3, 3)[face_indices].reshape(-1, 3)
        elif interpolation == UsdGeom.Tokens.uniform and len(normals_np) == len(source_indices):
            normals_np = normals_np[face_indices]
        target.CreateNormalsAttr(normals_np)
        target.SetNormalsInterpolation(interpolation)

    return target


def _author_preview_material(
    stage: Usd.Stage,
    path: Sdf.Path,
    color: tuple[float, float, float],
    *,
    opacity: float,
    roughness: float,
    ior: float,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(ior))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _repair_source_materials(
    stage: Usd.Stage,
    *,
    frame_color: tuple[float, float, float],
    frame_roughness: float,
) -> None:
    """Replace non-portable MDL material state inherited from the source asset."""
    plastic_path = Sdf.Path("/World/Looks/material_plastic")
    stage.RemovePrim(plastic_path)
    _author_preview_material(
        stage,
        plastic_path,
        frame_color,
        opacity=1.0,
        roughness=frame_roughness,
        ior=1.5,
    )

    # The combined source lens is hidden below and its collision mesh must not
    # be promoted to a visual by the importer. Keep the original binding target
    # valid, but remove its unresolved OmniSurface.mdl shader implementation.
    omni_surface_path = Sdf.Path("/World/Looks/OmniSurface")
    stage.RemovePrim(omni_surface_path)
    UsdShade.Material.Define(stage, omni_surface_path)

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if any(rel.GetName().startswith("material:binding") for rel in prim.GetRelationships()):
            UsdShade.MaterialBindingAPI.Apply(prim)


def author_anaglyph_glasses(
    source_path: str | Path,
    output_path: str | Path,
    *,
    lens_mesh_path: str = DEFAULT_LENS_MESH,
    opacity: float = 0.14,
    roughness: float = 0.04,
    ior: float = 1.49,
    red_color: tuple[float, float, float] = DEFAULT_RED_COLOR,
    green_color: tuple[float, float, float] = DEFAULT_GREEN_COLOR,
    frame_color: tuple[float, float, float] = DEFAULT_FRAME_COLOR,
    frame_roughness: float = 0.32,
) -> Path:
    """Create a Newton-ready copy with a matte frame and subtle red/green lenses."""
    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Glasses USD does not exist: {source_path}")
    if source_path == output_path:
        raise ValueError("Anaglyph output must differ from the source USD")
    if not 0.0 < opacity < 1.0:
        raise ValueError("Lens opacity must be between zero and one")
    if not 0.0 <= roughness <= 1.0:
        raise ValueError("Lens roughness must be between zero and one")
    if not 0.0 <= frame_roughness <= 1.0:
        raise ValueError("Frame roughness must be between zero and one")
    if ior <= 1.0:
        raise ValueError("Lens index of refraction must be greater than one")
    for name, color in (("Red lens", red_color), ("Green lens", green_color), ("Frame", frame_color)):
        if len(color) != 3 or any(channel < 0.0 or channel > 1.0 for channel in color):
            raise ValueError(f"{name} color must contain three values between zero and one")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    stage = Usd.Stage.Open(str(output_path))
    if stage is None:
        raise RuntimeError(f"Could not open copied USD stage: {output_path}")
    source = UsdGeom.Mesh.Get(stage, lens_mesh_path)
    if not source:
        raise ValueError(f"Lens mesh not found: {lens_mesh_path}")

    counts = np.asarray(source.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    if len(counts) == 0 or not np.all(counts == 3):
        raise ValueError("Anaglyph lens authoring currently requires a triangle mesh")
    indices = np.asarray(source.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    points = np.asarray(source.GetPointsAttr().Get(), dtype=np.float32)
    components = _face_components(indices, len(points))
    if len(components) != 2:
        raise ValueError(f"Expected exactly two disconnected lens components, found {len(components)}")
    components.sort(key=lambda faces: float(points[indices[faces].reshape(-1), 0].mean()))

    _repair_source_materials(stage, frame_color=frame_color, frame_roughness=frame_roughness)

    looks_path = Sdf.Path("/World/Looks")
    UsdGeom.Scope.Define(stage, looks_path)
    red = _author_preview_material(
        stage,
        looks_path.AppendChild("AnaglyphRed"),
        red_color,
        opacity=opacity,
        roughness=roughness,
        ior=ior,
    )
    green = _author_preview_material(
        stage,
        looks_path.AppendChild("AnaglyphGreen"),
        green_color,
        opacity=opacity,
        roughness=roughness,
        ior=ior,
    )

    parent_path = source.GetPath().GetParentPath()
    red_mesh = _copy_mesh_component(stage, source, parent_path.AppendChild("AnaglyphLensRed"), components[0])
    green_mesh = _copy_mesh_component(stage, source, parent_path.AppendChild("AnaglyphLensGreen"), components[1])
    UsdShade.MaterialBindingAPI.Apply(red_mesh.GetPrim()).Bind(red)
    UsdShade.MaterialBindingAPI.Apply(green_mesh.GetPrim()).Bind(green)
    red_mesh.GetPrim().CreateAttribute("palatial:renderRole", Sdf.ValueTypeNames.Token, custom=True).Set(
        "anaglyph-red-lens"
    )
    green_mesh.GetPrim().CreateAttribute("palatial:renderRole", Sdf.ValueTypeNames.Token, custom=True).Set(
        "anaglyph-green-lens"
    )
    UsdGeom.Imageable(source.GetPrim()).MakeInvisible()
    stage.GetRootLayer().Save()
    return output_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Source glasses USD")
    parser.add_argument("output", help="Corrected output USD")
    parser.add_argument("--lens-mesh", default=DEFAULT_LENS_MESH)
    parser.add_argument("--opacity", type=float, default=0.14)
    parser.add_argument("--roughness", type=float, default=0.04)
    parser.add_argument("--ior", type=float, default=1.49)
    parser.add_argument("--red-color", type=float, nargs=3, default=DEFAULT_RED_COLOR, metavar=("R", "G", "B"))
    parser.add_argument("--green-color", type=float, nargs=3, default=DEFAULT_GREEN_COLOR, metavar=("R", "G", "B"))
    parser.add_argument("--frame-color", type=float, nargs=3, default=DEFAULT_FRAME_COLOR, metavar=("R", "G", "B"))
    parser.add_argument("--frame-roughness", type=float, default=0.32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    output = author_anaglyph_glasses(
        args.source,
        args.output,
        lens_mesh_path=args.lens_mesh,
        opacity=args.opacity,
        roughness=args.roughness,
        ior=args.ior,
        red_color=tuple(args.red_color),
        green_color=tuple(args.green_color),
        frame_color=tuple(args.frame_color),
        frame_roughness=args.frame_roughness,
    )
    print(f"Anaglyph glasses USD saved in: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
