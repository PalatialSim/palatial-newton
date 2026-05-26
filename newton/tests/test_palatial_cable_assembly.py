# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Palatial Power cable assembly loader path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import newton  # noqa: F401
import numpy as np

from newton.examples.palatial.generate_palatial_cable_usd import author_cable_usd
from newton.palatial import load
from newton.tests.unittest_utils import USD_AVAILABLE

if USD_AVAILABLE:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom
    except (ImportError, ModuleNotFoundError):
        Usd = None  # type: ignore[assignment]
else:
    Usd = None  # type: ignore[assignment]

from newton._src.palatial.load import _detect_scene_kind

from newton._src.palatial.cable_assembly import (
    build_power_cable_assembly_model,
    is_power_cable_assembly_stage,
)


def _create_mesh_prim(stage: Any, prim_path: str, *, size: tuple[float, float, float]) -> None:
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    sx, sy, sz = size
    mesh.CreatePointsAttr().Set(
        [
            Gf.Vec3f(-sx, -sy, -sz),
            Gf.Vec3f(sx, -sy, -sz),
            Gf.Vec3f(sx, sy, -sz),
            Gf.Vec3f(-sx, sy, -sz),
            Gf.Vec3f(-sx, -sy, sz),
            Gf.Vec3f(sx, -sy, sz),
            Gf.Vec3f(sx, sy, sz),
            Gf.Vec3f(-sx, sy, sz),
        ]
    )
    mesh.CreateFaceVertexCountsAttr().Set([4, 4, 4, 4, 4, 4])
    mesh.CreateFaceVertexIndicesAttr().Set(
        [
            0, 1, 2, 3,
            4, 5, 6, 7,
            0, 1, 5, 4,
            1, 2, 6, 5,
            2, 3, 7, 6,
            3, 0, 4, 7,
        ]
    )
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)


def _apply_translate(prim: Any, translate: tuple[float, float, float]) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))


def _author_power_assembly_stage(output_path: Path, *, add_rod_api: bool = False) -> None:
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    geometry = UsdGeom.Scope.Define(stage, "/World/Geometry").GetPrim()
    _ = geometry

    cable = UsdGeom.Xform.Define(stage, "/World/Geometry/Power_Cable_Body").GetPrim()
    _apply_translate(cable, (0.0, 0.0, 0.0))
    if add_rod_api:
        cable.ApplyAPI("NewtonRodAPI")
        cable.CreateAttribute("newton:deformable:simulationIntent", Sdf.ValueTypeNames.Token, custom=True).Set("rod")
    _create_mesh_prim(stage, "/World/Geometry/Power_Cable_Body/Mesh", size=(0.9, 0.05, 0.05))

    iec_offsets = {
        "Pow_IEC_Strain": (1.25, 0.10, 0.0),
        "Pow_IEC_Body": (1.35, 0.0, 0.0),
        "Pow_IEC_Recess": (1.34, 0.0, 0.08),
        "Pow_IEC_Slot0": (1.40, 0.05, 0.0),
        "Pow_IEC_Slot1": (1.40, -0.05, 0.0),
        "Pow_IEC_ESlot": (1.45, 0.0, 0.0),
    }
    nema_offsets = {
        "Pow_NEMA_Strain": (-1.25, -0.10, 0.0),
        "Pow_NEMA_Body": (-1.35, 0.0, 0.0),
        "Pow_NEMA_Face": (-1.40, 0.0, 0.0),
        "Pow_NEMA_Hot": (-1.42, 0.08, 0.0),
        "Pow_NEMA_Neut": (-1.42, 0.0, 0.08),
        "Pow_NEMA_Gnd": (-1.42, -0.08, 0.0),
        "Pow_NEMA_FS0": (-1.48, 0.05, 0.0),
        "Pow_NEMA_FS1": (-1.48, -0.05, 0.0),
        "Pow_NEMA_GS": (-1.48, 0.0, 0.05),
    }
    for name, translate in {**iec_offsets, **nema_offsets}.items():
        prim = UsdGeom.Xform.Define(stage, f"/World/Geometry/{name}").GetPrim()
        _apply_translate(prim, translate)
        _create_mesh_prim(stage, f"/World/Geometry/{name}/Mesh", size=(0.08, 0.04, 0.04))

    stage.Save()


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestPalatialCableAssembly(unittest.TestCase):
    """Tests for the Power cable assembly loader path."""

    def test_scene_kind_detection_prefers_assembly_over_rod_tokens(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "power_assembly.usda"
            _author_power_assembly_stage(usd_path, add_rod_api=True)

            stage = Usd.Stage.Open(str(usd_path))
            self.assertTrue(is_power_cable_assembly_stage(stage))
            self.assertEqual(_detect_scene_kind(stage), "cable_assembly")

    def test_load_builds_power_cable_assembly_bundle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "power_assembly.usda"
            _author_power_assembly_stage(usd_path)

            bundle = load(str(usd_path), device="cpu")

            self.assertEqual(bundle.body_type, "cable")
            self.assertEqual(bundle.scene_kind, "cable_assembly")
            self.assertEqual(bundle.fps, 60)
            self.assertEqual(bundle.solver_params.get("iterations"), 2)
            self.assertEqual(bundle.solver_params.get("substeps"), 10)
            self.assertIs(bundle.solver.__class__, newton.solvers.SolverVBDPalatial)
            self.assertGreater(bundle.model.body_count, 2)
            self.assertGreater(bundle.model.joint_count, 2)

            shape_types = bundle.model.shape_type.numpy().tolist()
            self.assertEqual(shape_types.count(int(newton.GeoType.MESH)), 2)
            self.assertGreaterEqual(shape_types.count(int(newton.GeoType.BOX)), 1)
            self.assertGreaterEqual(shape_types.count(int(newton.GeoType.CAPSULE)), 1)

            joint_types = bundle.model.joint_type.numpy().tolist()
            self.assertGreaterEqual(joint_types.count(int(newton.JointType.FIXED)), 2)

            body_q = bundle.state_in.body_q.numpy()
            self.assertTrue(np.isfinite(body_q).all())

            contacts = bundle.model.contacts()
            bundle.state_in.clear_forces()
            bundle.model.collide(bundle.state_in, contacts)
            dt = bundle.dt / float(bundle.solver_params.get("substeps", 1))
            bundle.solver.step(bundle.state_in, bundle.state_out, bundle.control, contacts, dt)

    def test_build_power_cable_assembly_model_is_finite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "power_assembly.usda"
            _author_power_assembly_stage(usd_path)

            model = build_power_cable_assembly_model(str(usd_path), device="cpu")
            self.assertGreater(model.body_count, 0)
            self.assertTrue(np.isfinite(model.body_q.numpy()).all())

    def test_simple_cable_still_detects_as_cable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "simple_cable.newton.usda"
            author_cable_usd(usd_path, solver="vbd_palatial", solver_substeps=2)

            stage = Usd.Stage.Open(str(usd_path))
            self.assertEqual(_detect_scene_kind(stage), "cable")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
