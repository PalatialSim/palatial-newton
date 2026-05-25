# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate a NewtonRodAPI cable asset for the palatial cable examples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import newton  # noqa: F401

from newton.palatial import read_cable_params

DEFAULT_OUTPUT_PATH = Path("palatial_cable_example.newton.usda")
DEFAULT_ROOT_PATH = "/Cable"
DEFAULT_CENTERLINE_PATH = f"{DEFAULT_ROOT_PATH}/Centerline"
DEFAULT_MATERIAL_PATH = "/Materials/CableMaterial"


def _usd_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import newton_usd_schemas  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Generating palatial cable USD assets requires 'usd-core' and the "
            "Newton USD schemas plugin."
        ) from exc

    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def _create_custom_attribute(prim: Any, name: str, value_type: Any, value: object) -> None:
    prim.CreateAttribute(name, value_type, custom=True).Set(value)


def _apply_api_schema(prim: Any, schema_name: str, *, Sdf: Any) -> None:
    try:
        if prim.HasAPI(schema_name):
            return
    except Exception:
        pass

    try:
        prim.ApplyAPI(schema_name)
        if prim.HasAPI(schema_name):
            return
    except Exception:
        pass

    token_list = prim.GetMetadata("apiSchemas")
    prepended_items: list[str] = []
    appended_items: list[str] = []
    explicit_items: list[str] = []
    if token_list is not None:
        prepended_items.extend(str(token) for token in (getattr(token_list, "prependedItems", []) or []))
        appended_items.extend(str(token) for token in (getattr(token_list, "appendedItems", []) or []))
        explicit_items.extend(str(token) for token in (getattr(token_list, "explicitItems", []) or []))
        if schema_name in prepended_items or schema_name in appended_items or schema_name in explicit_items:
            return

    prepended_items.append(schema_name)
    updated = Sdf.TokenListOp()
    updated.prependedItems = prepended_items
    updated.appendedItems = appended_items
    updated.explicitItems = explicit_items
    prim.SetMetadata("apiSchemas", updated)


def _effective_radius(
    *,
    cross_section_type: str,
    radius: float,
    width: float,
    thickness: float,
) -> float:
    if cross_section_type == "roundSolid":
        if radius <= 0.0:
            raise ValueError("radius must be > 0 for roundSolid cables")
        return float(radius)
    if cross_section_type == "flatRect":
        if width <= 0.0:
            raise ValueError("width must be > 0 for flatRect cables")
        if thickness <= 0.0:
            raise ValueError("thickness must be > 0 for flatRect cables")
        return 0.5 * float(thickness)
    raise ValueError(f"Unsupported cross_section_type: {cross_section_type!r}")


def _build_straight_points(
    *,
    length: float,
    segment_count: int,
    z_height: float,
    Gf: Any,
) -> list[Any]:
    if length <= 0.0:
        raise ValueError("length must be > 0")
    if segment_count < 2:
        raise ValueError("segment_count must be >= 2")

    step = float(length) / float(segment_count)
    return [Gf.Vec3f(float(index) * step, 0.0, float(z_height)) for index in range(segment_count + 1)]


def author_cable_usd(
    output_path: str | Path,
    *,
    cross_section_type: str = "flatRect",
    length: float = 1.5,
    segment_count: int = 16,
    drop_height: float = 0.3,
    twist_total: float = 0.0,
    radius: float = 0.005,
    width: float = 0.012,
    thickness: float = 0.004,
    density: float = 1000.0,
    stretch_stiffness: float = 1.0e5,
    stretch_damping: float = 0.05,
    compress_stiffness: float = 1.0e5,
    compress_damping: float = 0.05,
    bend_y_stiffness: float = 8.0e2,
    bend_y_damping: float = 0.1,
    bend_z_stiffness: float = 1.6e3,
    bend_z_damping: float = 0.1,
    torsion_stiffness: float = 4.0e2,
    torsion_damping: float = 0.05,
    fps: int = 120,
    solver: str = "vbd",
    solver_iterations: int = 2,
    solver_substeps: int = 2,
) -> Path:
    """Author a NewtonRodAPI cable asset on disk.

    Args:
        output_path: Destination ``*.newton.usda`` path.
        cross_section_type: Cable cross-section token. Supported values are
            ``"roundSolid"`` and ``"flatRect"``.
        length: Cable centerline length [m].
        segment_count: Number of cable segments. The authored centerline uses
            ``segment_count + 1`` points.
        drop_height: World-space lift [m] encoded into the authored centerline
            points and mirrored in ``newton:rod:dropHeight``.
        twist_total: Total twist [rad] for parallel transport quaternion
            generation during load.
        radius: Round cable radius [m].
        width: Flat cable width [m].
        thickness: Flat cable thickness [m].
        density: Cable density [kg/m^3].
        stretch_stiffness: Axial stretch stiffness [N/m].
        stretch_damping: Axial stretch damping.
        compress_stiffness: Axial compression stiffness [N/m].
        compress_damping: Axial compression damping.
        bend_y_stiffness: Bend stiffness around one transverse cable axis [N*m].
        bend_y_damping: Bend damping around one transverse cable axis.
        bend_z_stiffness: Bend stiffness around the other transverse cable axis [N*m].
        bend_z_damping: Bend damping around the other transverse cable axis.
        torsion_stiffness: Torsion stiffness around the cable tangent axis [N*m].
        torsion_damping: Torsion damping around the cable tangent axis.
        fps: Authored simulation rate [Hz].
        solver: Authored solver token for the example loader.
        solver_iterations: Authored ``newton:solver:iterations`` value.
        solver_substeps: Authored ``newton:solver:substeps`` value.

    Returns:
        Absolute path to the authored USD file.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if solver_iterations <= 0:
        raise ValueError("solver_iterations must be > 0")
    if solver_substeps <= 0:
        raise ValueError("solver_substeps must be > 0")
    if density <= 0.0:
        raise ValueError("density must be > 0")

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = _usd_modules()
    effective_radius = _effective_radius(
        cross_section_type=cross_section_type,
        radius=radius,
        width=width,
        thickness=thickness,
    )
    centerline_points = _build_straight_points(
        length=length,
        segment_count=segment_count,
        z_height=drop_height + effective_radius,
        Gf=Gf,
    )

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, DEFAULT_ROOT_PATH).GetPrim()
    stage.SetDefaultPrim(root)
    _apply_api_schema(root, "NewtonDeformableAPI", Sdf=Sdf)
    _apply_api_schema(root, "NewtonRodAPI", Sdf=Sdf)
    UsdShade.MaterialBindingAPI.Apply(root)
    _create_custom_attribute(root, "newton:deformable:enabled", Sdf.ValueTypeNames.Bool, True)
    _create_custom_attribute(root, "newton:deformable:simulationIntent", Sdf.ValueTypeNames.Token, "rod")
    _create_custom_attribute(root, "newton:rod:frameDefinition", Sdf.ValueTypeNames.Token, "parallelTransport")
    _create_custom_attribute(root, "newton:rod:closed", Sdf.ValueTypeNames.Bool, False)
    _create_custom_attribute(root, "newton:rod:isClosed", Sdf.ValueTypeNames.Bool, False)
    _create_custom_attribute(root, "newton:rod:crossSectionType", Sdf.ValueTypeNames.Token, cross_section_type)
    _create_custom_attribute(root, "newton:rod:segmentCount", Sdf.ValueTypeNames.Int, int(segment_count))
    _create_custom_attribute(root, "newton:rod:verticesPerSegment", Sdf.ValueTypeNames.Int, 2)
    _create_custom_attribute(root, "newton:rod:length", Sdf.ValueTypeNames.Float, float(length))
    _create_custom_attribute(root, "newton:rod:dropHeight", Sdf.ValueTypeNames.Float, float(drop_height))
    _create_custom_attribute(root, "newton:rod:twistTotal", Sdf.ValueTypeNames.Float, float(twist_total))
    if cross_section_type == "roundSolid":
        _create_custom_attribute(root, "newton:rod:radius", Sdf.ValueTypeNames.Float, float(radius))
    else:
        _create_custom_attribute(root, "newton:rod:width", Sdf.ValueTypeNames.Float, float(width))
        _create_custom_attribute(root, "newton:rod:thickness", Sdf.ValueTypeNames.Float, float(thickness))

    centerline = UsdGeom.BasisCurves.Define(stage, DEFAULT_CENTERLINE_PATH)
    centerline_prim = centerline.GetPrim()
    _apply_api_schema(centerline_prim, "NewtonRodAPI", Sdf=Sdf)
    centerline.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    centerline.CreateCurveVertexCountsAttr().Set([segment_count + 1])
    centerline.CreatePointsAttr().Set(centerline_points)
    _create_custom_attribute(centerline_prim, "newton:rod:length", Sdf.ValueTypeNames.Float, float(length))

    material = UsdShade.Material.Define(stage, DEFAULT_MATERIAL_PATH)
    material_prim = material.GetPrim()
    _apply_api_schema(material_prim, "NewtonRodMaterialAPI", Sdf=Sdf)
    _create_custom_attribute(material_prim, "newton:rod:density", Sdf.ValueTypeNames.Float, float(density))
    _create_custom_attribute(
        material_prim, "newton:rod:stretchStiffness", Sdf.ValueTypeNames.Float, float(stretch_stiffness)
    )
    _create_custom_attribute(
        material_prim, "newton:rod:stretchDamping", Sdf.ValueTypeNames.Float, float(stretch_damping)
    )
    _create_custom_attribute(
        material_prim, "newton:rod:compressStiffness", Sdf.ValueTypeNames.Float, float(compress_stiffness)
    )
    _create_custom_attribute(
        material_prim, "newton:rod:compressDamping", Sdf.ValueTypeNames.Float, float(compress_damping)
    )
    _create_custom_attribute(
        material_prim, "newton:rod:bendYStiffness", Sdf.ValueTypeNames.Float, float(bend_y_stiffness)
    )
    _create_custom_attribute(
        material_prim, "newton:rod:bendYDamping", Sdf.ValueTypeNames.Float, float(bend_y_damping)
    )
    _create_custom_attribute(
        material_prim, "newton:rod:bendZStiffness", Sdf.ValueTypeNames.Float, float(bend_z_stiffness)
    )
    _create_custom_attribute(
        material_prim, "newton:rod:bendZDamping", Sdf.ValueTypeNames.Float, float(bend_z_damping)
    )
    _create_custom_attribute(
        material_prim, "newton:rod:torsionStiffness", Sdf.ValueTypeNames.Float, float(torsion_stiffness)
    )
    _create_custom_attribute(
        material_prim, "newton:rod:torsionDamping", Sdf.ValueTypeNames.Float, float(torsion_damping)
    )
    _create_custom_attribute(material_prim, "physics:density", Sdf.ValueTypeNames.Float, float(density))
    UsdShade.MaterialBindingAPI(root).Bind(material)

    scene = UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim()
    _create_custom_attribute(scene, "newton:solver", Sdf.ValueTypeNames.Token, solver)
    _create_custom_attribute(scene, "newton:timeStepsPerSecond", Sdf.ValueTypeNames.Int, int(fps))
    _create_custom_attribute(scene, "newton:solver:iterations", Sdf.ValueTypeNames.Int, int(solver_iterations))
    _create_custom_attribute(scene, "newton:solver:substeps", Sdf.ValueTypeNames.Int, int(solver_substeps))

    stage.Save()
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_palatial_cable_usd",
        description="Generate a NewtonRodAPI cable *.newton.usda asset for the palatial cable examples.",
    )
    parser.add_argument(
        "output_usd",
        nargs="?",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Destination *.newton.usda path.",
    )
    parser.add_argument(
        "--cross-section-type",
        choices=("flatRect", "roundSolid"),
        default="flatRect",
        help="Cable cross-section authored into NewtonRodAPI.",
    )
    parser.add_argument("--length", type=float, default=1.5, help="Cable length [m].")
    parser.add_argument("--segment-count", type=int, default=16, help="Cable segment count.")
    parser.add_argument(
        "--drop-height",
        type=float,
        default=0.3,
        help="Lift baked into the authored centerline and mirrored in newton:rod:dropHeight [m].",
    )
    parser.add_argument("--twist-total", type=float, default=0.0, help="Authored total twist [rad].")
    parser.add_argument("--radius", type=float, default=0.005, help="Round cable radius [m].")
    parser.add_argument("--width", type=float, default=0.012, help="Flat cable width [m].")
    parser.add_argument("--thickness", type=float, default=0.004, help="Flat cable thickness [m].")
    parser.add_argument("--density", type=float, default=1000.0, help="Cable density [kg/m^3].")
    parser.add_argument("--stretch-stiffness", type=float, default=1.0e5, help="Stretch stiffness [N/m].")
    parser.add_argument("--stretch-damping", type=float, default=0.05, help="Stretch damping.")
    parser.add_argument("--compress-stiffness", type=float, default=1.0e5, help="Compression stiffness [N/m].")
    parser.add_argument("--compress-damping", type=float, default=0.05, help="Compression damping.")
    parser.add_argument("--bend-y-stiffness", type=float, default=8.0e2, help="Bend-Y stiffness [N*m].")
    parser.add_argument("--bend-y-damping", type=float, default=0.1, help="Bend-Y damping.")
    parser.add_argument("--bend-z-stiffness", type=float, default=1.6e3, help="Bend-Z stiffness [N*m].")
    parser.add_argument("--bend-z-damping", type=float, default=0.1, help="Bend-Z damping.")
    parser.add_argument("--torsion-stiffness", type=float, default=4.0e2, help="Torsion stiffness [N*m].")
    parser.add_argument("--torsion-damping", type=float, default=0.05, help="Torsion damping.")
    parser.add_argument("--fps", type=int, default=120, help="Authored simulation rate [Hz].")
    parser.add_argument(
        "--solver",
        choices=("vbd", "vbd_palatial"),
        default="vbd",
        help="Authored solver token written onto /physicsScene.",
    )
    parser.add_argument(
        "--solver-iterations",
        type=int,
        default=2,
        help="Authored newton:solver:iterations value.",
    )
    parser.add_argument(
        "--solver-substeps",
        type=int,
        default=2,
        help="Authored newton:solver:substeps value.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    output_path = author_cable_usd(
        args.output_usd,
        cross_section_type=args.cross_section_type,
        length=args.length,
        segment_count=args.segment_count,
        drop_height=args.drop_height,
        twist_total=args.twist_total,
        radius=args.radius,
        width=args.width,
        thickness=args.thickness,
        density=args.density,
        stretch_stiffness=args.stretch_stiffness,
        stretch_damping=args.stretch_damping,
        compress_stiffness=args.compress_stiffness,
        compress_damping=args.compress_damping,
        bend_y_stiffness=args.bend_y_stiffness,
        bend_y_damping=args.bend_y_damping,
        bend_z_stiffness=args.bend_z_stiffness,
        bend_z_damping=args.bend_z_damping,
        torsion_stiffness=args.torsion_stiffness,
        torsion_damping=args.torsion_damping,
        fps=args.fps,
        solver=args.solver,
        solver_iterations=args.solver_iterations,
        solver_substeps=args.solver_substeps,
    )
    params = read_cable_params(str(output_path))

    print(f"[write] {output_path}")
    print(
        f"[cable] cross_section={params['crossSectionType']} length={float(params['length']):.3f}m "
        f"segments={int(params['segmentCount'])} radius={float(params['radius']):.4f}m "
        f"fps={args.fps} solver={args.solver}"
    )
    print("Run:")
    print(
        "  python -m newton.examples.palatial.example_palatial_cable "
        f"\"{output_path}\" --gui --device cuda:0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
