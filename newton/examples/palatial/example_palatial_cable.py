# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Load a Palatial cable USDA and simulate it with optional anchored twist.

This is the cable-specific sibling of ``example_palatial_load.py``. It consumes
the public ``newton.palatial.load()`` entry point, requires a cable or rod USDA,
and optionally anchors the first segment and spins it about the cable axis to
demonstrate twist propagation through the loaded cable bundle.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
import warp as wp

import newton

from newton.examples.palatial.generate_palatial_cable_usd import author_cable_usd
from newton.palatial import extract_cable_points, load, read_cable_params

DEFAULT_SIMPLE_CABLE_CONTACT_KE = 1.0e4
DEFAULT_SIMPLE_CABLE_CONTACT_KD = 0.0
DEFAULT_SIMPLE_CABLE_CONTACT_KF = 1.0e3
DEFAULT_SIMPLE_CABLE_CONTACT_MU = 1.0
DEFAULT_SIMPLE_CABLE_CONTACT_MARGIN = 2.5e-4
DEFAULT_SIMPLE_CABLE_CONTACT_GAP = 1.0e-3
DEFAULT_OBSTACLE_BOX_HX = 0.18
DEFAULT_OBSTACLE_BOX_HY = 0.12
DEFAULT_OBSTACLE_BOX_HZ = 0.10
_IDENTITY_QUAT = wp.quat(0.0, 0.0, 0.0, 1.0)


@wp.kernel
def spin_first_capsules_kernel(
    body_indices: wp.array[wp.int32],
    twist_rates: wp.array[float],
    dt: float,
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    """Apply continuous twist to the anchored cable segment."""
    tid = wp.tid()
    body_id = body_indices[tid]

    transform = body_q0[body_id]
    pos = wp.transform_get_translation(transform)
    rot = wp.transform_get_rotation(transform)

    # Cable rod capsules are authored with their long axis along local +Z.
    axis_world = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
    angle = twist_rates[tid] * dt
    dq = wp.quat_from_axis_angle(axis_world, angle)
    rot_new = wp.mul(dq, rot)

    updated = wp.transform(pos, rot_new)
    body_q0[body_id] = updated
    body_q1[body_id] = updated


def _filter_solver_kwargs(solver_cls: type, solver_params: dict[str, object]) -> dict[str, object]:
    """Keep only solver kwargs accepted by the current constructor."""
    alias = {
        "iterations": "iterations",
        "substeps": "substeps",
        "particleEnableSelfContact": "particle_enable_self_contact",
        "particleSelfContactRadius": "particle_self_contact_radius",
        "particleSelfContactMargin": "particle_self_contact_margin",
    }
    try:
        accepted = set(inspect.signature(solver_cls.__init__).parameters)
    except (TypeError, ValueError):
        accepted = set()

    kwargs: dict[str, object] = {}
    for name, value in solver_params.items():
        py_name = alias.get(name, name)
        if py_name in accepted:
            kwargs[py_name] = value
    return kwargs


def _viewer_running(viewer) -> bool:
    """Normalize ``viewer.is_running`` across ViewerGL, ViewerNull, and wrappers."""

    is_running_attr = getattr(viewer, "is_running", True)
    if callable(is_running_attr):
        try:
            return bool(is_running_attr())
        except Exception:
            return True
    return bool(is_running_attr)


def _set_front_camera(viewer, points: np.ndarray) -> None:
    """Frame a front-facing camera from the current cable bounds."""
    cx = float(points[:, 0].mean())
    cy = float(points[:, 1].mean())
    cz_mid = float((points[:, 2].min() + points[:, 2].max()) * 0.5)
    ext_x = float(points[:, 0].max() - points[:, 0].min())
    ext_y = float(points[:, 1].max() - points[:, 1].min())
    ext_z = float(points[:, 2].max() - points[:, 2].min())
    dist = max(ext_x, ext_y, ext_z, 1.0) * 2.2
    viewer.set_camera(wp.vec3(cx, cy - dist, cz_mid), 0.0, 90.0)
    print(f"  front-view camera: pos=({cx:.3f},{cy - dist:.3f},{cz_mid:.3f})")


def _set_top_camera(viewer, points: np.ndarray) -> None:
    """Frame a top-down camera from the current cable bounds."""
    cx = float(points[:, 0].mean())
    cy = float(points[:, 1].mean())
    zmax = float(points[:, 2].max())
    ext_x = float(points[:, 0].max() - points[:, 0].min())
    ext_y = float(points[:, 1].max() - points[:, 1].min())
    height = zmax + max(ext_x, ext_y, 1.0) * 1.5
    viewer.set_camera(wp.vec3(cx, cy, height), -90.0, 0.0)
    print(f"  top-view camera: pos=({cx:.3f},{cy:.3f},{height:.3f})")


def _resolve_input_usd(
    usd_path: str | None,
    *,
    substeps: int | None,
    solver_override: str | None,
) -> str:
    """Return the user USD path or author a temporary flatRect ribbon asset."""
    if usd_path:
        return usd_path

    generated_path = Path(tempfile.gettempdir()) / "palatial_ribbon_example.newton.usda"
    author_cable_usd(
        generated_path,
        cross_section_type="flatRect",
        solver=solver_override or "vbd",
        solver_substeps=max(1, int(substeps)) if substeps is not None else 2,
    )
    print(f"[write] generated default cable asset: {generated_path}")
    return str(generated_path)


def _reconstruct_cable_points_from_bundle(bundle, params: dict[str, object]) -> list[wp.vec3]:
    """Reconstruct centerline points from the loaded cable body transforms."""

    body_q = bundle.model.body_q.numpy()
    if body_q.shape[0] <= 0:
        raise RuntimeError("Cable bundle has zero rigid bodies")

    points = [
        wp.vec3(float(body_q[index, 0]), float(body_q[index, 1]), float(body_q[index, 2]))
        for index in range(body_q.shape[0])
    ]
    segment_count = max(1, int(params["segmentCount"]))
    segment_length = float(params["length"]) / float(segment_count)
    last_rotation = wp.quat(
        float(body_q[-1, 3]),
        float(body_q[-1, 4]),
        float(body_q[-1, 5]),
        float(body_q[-1, 6]),
    )
    tip_point = points[-1] + wp.quat_rotate(last_rotation, wp.vec3(0.0, 0.0, segment_length))
    points.append(tip_point)
    return points


def _build_simple_cable_model(
    *,
    bundle,
    usd_path: str,
    params: dict[str, object],
    device: str | None,
    obstacle_box: bool,
    obstacle_box_hx: float,
    obstacle_box_hy: float,
    obstacle_box_hz: float,
) -> newton.Model:
    """Build a simple cable model for the example, optionally with a static obstacle box."""

    points = extract_cable_points(usd_path, world_space=True)
    if not points:
        points = _reconstruct_cable_points_from_bundle(bundle, params)

    quaternions = newton.utils.create_parallel_transport_cable_quaternions(
        points,
        twist_total=float(params["twistTotal"]),
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=float(params["density"]))

    with wp.ScopedDevice(device) if device else wp.ScopedDevice(wp.get_preferred_device()):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        shape_type = "box" if str(params["crossSectionType"]) == "flatRect" else "capsule"
        builder.add_rod(
            positions=points,
            quaternions=quaternions,
            radius=float(params["radius"]),
            cfg=cfg,
            stretch_stiffness=float(params["stretchStiffness"]),
            stretch_damping=float(params["stretchDamping"]),
            bend_y_stiffness=float(params["bendYStiffness"]),
            bend_y_damping=float(params["bendYDamping"]),
            bend_z_stiffness=float(params["bendZStiffness"]),
            bend_z_damping=float(params["bendZDamping"]),
            torsion_stiffness=float(params["torsionStiffness"]),
            torsion_damping=float(params["torsionDamping"]),
            closed=bool(params["closed"]),
            label="cable",
            shape_type=shape_type,
            width=float(params["width"]),
            thickness=float(params["thickness"]),
        )
        if obstacle_box:
            point_array = np.asarray([(float(point[0]), float(point[1]), float(point[2])) for point in points], dtype=float)
            box_center = wp.vec3(
                float((point_array[:, 0].min() + point_array[:, 0].max()) * 0.5),
                float(point_array[:, 1].mean()),
                float(obstacle_box_hz),
            )
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(box_center, _IDENTITY_QUAT),
                hx=float(obstacle_box_hx),
                hy=float(obstacle_box_hy),
                hz=float(obstacle_box_hz),
                label="cable_drop_box",
            )
        try:
            builder.color()
        except Exception:
            pass
        return builder.finalize()


class Example:
    def __init__(
        self,
        viewer,
        usd_path: str,
        *,
        substeps: int | None = None,
        device: str | None = None,
        solver_override: str | None = None,
        anchor_first: bool = True,
        anchor_last: bool = False,
        spin_rate: float = 0.5,
        spin_last_rate: float = 0.0,
        extra_drop_height: float = 0.0,
        obstacle_box: bool = False,
        obstacle_box_hx: float = DEFAULT_OBSTACLE_BOX_HX,
        obstacle_box_hy: float = DEFAULT_OBSTACLE_BOX_HY,
        obstacle_box_hz: float = DEFAULT_OBSTACLE_BOX_HZ,
        contact_ke: float | None = None,
        contact_kd: float | None = None,
        contact_kf: float | None = None,
        contact_mu: float | None = None,
        contact_margin: float | None = None,
        contact_gap: float | None = None,
    ):
        self.viewer = viewer
        self.usd_path = usd_path

        bundle = load(usd_path, solver_override=solver_override, device=device)
        if bundle.body_type != "cable":
            raise RuntimeError(
                f"example_palatial_cable needs a cable or rod USDA, got body_type={bundle.body_type!r}"
            )

        self.bundle = bundle
        self.scene_kind = bundle.scene_kind or bundle.body_type

        if self.scene_kind == "cable_assembly":
            self.cable_params = {}
            self.centerline_points = []
        else:
            self.cable_params = read_cable_params(usd_path)
            self.centerline_points = extract_cable_points(usd_path, world_space=True)

        self.obstacle_box = bool(obstacle_box)
        self.obstacle_box_hx = float(obstacle_box_hx)
        self.obstacle_box_hy = float(obstacle_box_hy)
        self.obstacle_box_hz = float(obstacle_box_hz)
        self._solver_kwargs = _filter_solver_kwargs(type(bundle.solver), bundle.solver_params)

        if self.scene_kind != "cable_assembly" and self.obstacle_box:
            self.model = _build_simple_cable_model(
                bundle=bundle,
                usd_path=usd_path,
                params=self.cable_params,
                device=device,
                obstacle_box=self.obstacle_box,
                obstacle_box_hx=self.obstacle_box_hx,
                obstacle_box_hy=self.obstacle_box_hy,
                obstacle_box_hz=self.obstacle_box_hz,
            )
            self.solver = type(bundle.solver)(self.model, **self._solver_kwargs)
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()
            self.contacts = self.model.contacts()
            self.bundle.model = self.model
            self.bundle.solver = self.solver
            self.bundle.state_in = self.state_0
            self.bundle.state_out = self.state_1
            self.bundle.control = self.control
        else:
            self.model = bundle.model
            self.solver = bundle.solver
            self.state_0 = bundle.state_in
            self.state_1 = bundle.state_out
            self.control = bundle.control
            self.contacts = self.model.contacts()

        self.anchor_first = bool(anchor_first)
        self.anchor_last = bool(anchor_last)
        self.spin_rate = float(spin_rate)
        self.spin_last_rate = float(spin_last_rate)
        if abs(self.spin_rate) > 0.0 and not self.anchor_first:
            raise RuntimeError("--spin-rate requires --anchor-first so the driven segment stays kinematic")
        if abs(self.spin_last_rate) > 0.0 and not self.anchor_last:
            raise RuntimeError("--spin-last-rate requires --anchor-last so the driven segment stays kinematic")

        body_count = int(self.model.body_count)
        if body_count <= 0:
            raise RuntimeError("Cable bundle has zero rigid bodies")
        self.anchor_body_index = 0
        self.tip_body_index = body_count - 1

        if self.scene_kind != "cable_assembly":
            if contact_ke is None:
                contact_ke = DEFAULT_SIMPLE_CABLE_CONTACT_KE
            if contact_kd is None:
                contact_kd = DEFAULT_SIMPLE_CABLE_CONTACT_KD
            if contact_kf is None:
                contact_kf = DEFAULT_SIMPLE_CABLE_CONTACT_KF
            if contact_mu is None:
                contact_mu = DEFAULT_SIMPLE_CABLE_CONTACT_MU
            if contact_margin is None:
                contact_margin = DEFAULT_SIMPLE_CABLE_CONTACT_MARGIN
            if contact_gap is None:
                contact_gap = DEFAULT_SIMPLE_CABLE_CONTACT_GAP

        self.contact_ke = contact_ke
        self.contact_kd = contact_kd
        self.contact_kf = contact_kf
        self.contact_mu = contact_mu
        self.contact_margin = contact_margin
        self.contact_gap = contact_gap
        self._apply_contact_tuning()

        if extra_drop_height:
            self._shift_cable_z(float(extra_drop_height))

        anchored_body_indices: list[int] = []
        if self.anchor_first:
            anchored_body_indices.append(self.anchor_body_index)
        if self.anchor_last and self.tip_body_index not in anchored_body_indices:
            anchored_body_indices.append(self.tip_body_index)
        for body_index in anchored_body_indices:
            self._make_body_kinematic(body_index)
        if anchored_body_indices:
            self._rebuild_solver()

        self.fps = bundle.fps
        self.frame_dt = 1.0 / float(self.fps)
        if substeps is not None:
            self.sim_substeps = max(1, int(substeps))
        else:
            self.sim_substeps = max(1, int(bundle.solver_params.get("substeps", 1)))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        body_device = self.state_0.body_q.device if self.state_0.body_q is not None else self.model.device
        spin_rate_by_body: dict[int, float] = {}
        if abs(self.spin_rate) > 0.0:
            spin_rate_by_body[self.anchor_body_index] = spin_rate_by_body.get(self.anchor_body_index, 0.0) + self.spin_rate
        if abs(self.spin_last_rate) > 0.0:
            spin_rate_by_body[self.tip_body_index] = spin_rate_by_body.get(self.tip_body_index, 0.0) + self.spin_last_rate
        if spin_rate_by_body:
            self.spin_body_indices = wp.array(list(spin_rate_by_body.keys()), dtype=wp.int32, device=body_device)
            self.spin_rates = wp.array(list(spin_rate_by_body.values()), dtype=wp.float32, device=body_device)
        else:
            self.spin_body_indices = wp.zeros(0, dtype=wp.int32, device=body_device)
            self.spin_rates = wp.zeros(0, dtype=wp.float32, device=body_device)

        self.model.collide(self.state_0, self.contacts)
        self.viewer.set_model(self.model)

        self.capture()
        self._print_bundle_summary(extra_drop_height)

    def capture(self) -> None:
        """Capture the cable simulation into a CUDA graph when available."""
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def _rebuild_solver(self) -> None:
        self.solver = type(self.solver)(self.model, **self._solver_kwargs)

    def _apply_contact_tuning(self) -> None:
        """Apply per-shape contact tuning overrides directly to the finalized model."""

        tuning = {
            "contact_ke": self.contact_ke,
            "contact_kd": self.contact_kd,
            "contact_kf": self.contact_kf,
            "contact_mu": self.contact_mu,
            "contact_margin": self.contact_margin,
            "contact_gap": self.contact_gap,
        }
        for name, value in tuning.items():
            if value is not None and float(value) < 0.0:
                raise RuntimeError(f"{name} must be >= 0, got {value}")

        shape_count = int(self.model.shape_count)
        if shape_count <= 0:
            return

        def _assign_full(attr_name: str, value: float | None) -> None:
            if value is None:
                return
            array = getattr(self.model, attr_name, None)
            if array is None:
                return
            array.assign(np.full(shape_count, float(value), dtype=np.float32))

        _assign_full("shape_material_ke", self.contact_ke)
        _assign_full("shape_material_kd", self.contact_kd)
        _assign_full("shape_material_kf", self.contact_kf)
        _assign_full("shape_material_mu", self.contact_mu)
        _assign_full("shape_margin", self.contact_margin)
        _assign_full("shape_gap", self.contact_gap)

    def _shift_cable_z(self, dz: float) -> None:
        """Lift every cable segment by ``dz`` in world +Z."""
        for state in (self.state_0, self.state_1):
            body_q = state.body_q.numpy().copy()
            body_q[:, 2] += dz
            state.body_q.assign(body_q)

    def _make_body_kinematic(self, body_index: int) -> None:
        """Freeze one cable segment by zeroing its mass and inertia."""
        body_mass = self.model.body_mass.numpy().copy()
        body_inv_mass = self.model.body_inv_mass.numpy().copy()
        body_inertia = self.model.body_inertia.numpy().copy()
        body_inv_inertia = self.model.body_inv_inertia.numpy().copy()

        body_mass[body_index] = 0.0
        body_inv_mass[body_index] = 0.0
        body_inertia[body_index] = np.zeros((3, 3), dtype=np.float32)
        body_inv_inertia[body_index] = np.zeros((3, 3), dtype=np.float32)

        self.model.body_mass.assign(body_mass)
        self.model.body_inv_mass.assign(body_inv_mass)
        self.model.body_inertia.assign(body_inertia)
        self.model.body_inv_inertia.assign(body_inv_inertia)

    def _print_bundle_summary(self, extra_drop_height: float) -> None:
        if self.scene_kind == "cable_assembly":
            shape_types = self.model.shape_type.numpy().tolist()
            mesh_count = shape_types.count(int(newton.GeoType.MESH))
            box_count = shape_types.count(int(newton.GeoType.BOX))
            print(
                f"[cable_assembly] usd={self.usd_path}  solver={self.bundle.solver_name}  "
                f"fps={self.fps}  substeps={self.sim_substeps}  bodies={int(self.model.body_count)}  "
                f"joints={int(self.model.joint_count)}"
            )
            print(
                f"        connector_meshes={mesh_count}  static_boxes={box_count}  "
                f"anchor_first={self.anchor_first}  anchor_last={self.anchor_last}  "
                f"spin_rate={self.spin_rate:.3f}rad/s  spin_last_rate={self.spin_last_rate:.3f}rad/s  "
                f"extra_drop={extra_drop_height:.3f}m"
            )
            if any(value is not None for value in (self.contact_ke, self.contact_kd, self.contact_kf, self.contact_mu)):
                print(
                    "        contact: "
                    f"ke={self.contact_ke!s}  kd={self.contact_kd!s}  "
                    f"kf={self.contact_kf!s}  mu={self.contact_mu!s}  "
                    f"margin={self.contact_margin!s}  gap={self.contact_gap!s}"
                )
            return

        params = self.cable_params
        cross_section = str(params["crossSectionType"])
        if cross_section == "flatRect":
            cross_section_summary = (
                f"cross_section={cross_section}  width={float(params['width']):.4f}m  "
                f"thickness={float(params['thickness']):.4f}m  radius={float(params['radius']):.4f}m"
            )
        else:
            cross_section_summary = (
                f"cross_section={cross_section}  radius={float(params['radius']):.4f}m"
            )
        print(
            f"[cable] usd={self.usd_path}  solver={self.bundle.solver_name}  "
            f"fps={self.fps}  substeps={self.sim_substeps}  bodies={int(self.model.body_count)}  "
            f"joints={int(self.model.joint_count)}"
        )
        print(
            f"        {cross_section_summary}  "
            f"length={float(params['length']):.4f}m  segments={int(params['segmentCount'])}  "
            f"anchor_first={self.anchor_first}  anchor_last={self.anchor_last}  "
            f"spin_rate={self.spin_rate:.3f}rad/s  spin_last_rate={self.spin_last_rate:.3f}rad/s  "
            f"extra_drop={extra_drop_height:.3f}m"
        )
        print(
            "        stiffness: "
            f"stretch={float(params['stretchStiffness']):.3g}  "
            f"bend_y={float(params['bendYStiffness']):.3g}  "
            f"bend_z={float(params['bendZStiffness']):.3g}  "
            f"torsion={float(params['torsionStiffness']):.3g}"
        )
        print(
            "        damping: "
            f"stretch={float(params['stretchDamping']):.3g}  "
            f"bend_y={float(params['bendYDamping']):.3g}  "
            f"bend_z={float(params['bendZDamping']):.3g}  "
            f"torsion={float(params['torsionDamping']):.3g}"
        )
        if self.centerline_points:
            print(
                f"        authored centerline points={len(self.centerline_points)}  "
                f"dropHeight={float(params['dropHeight']):.3f}m  twistTotal={float(params['twistTotal']):.3f}rad"
            )
        print(
            "        contact: "
            f"ke={self.contact_ke!s}  kd={self.contact_kd!s}  "
            f"kf={self.contact_kf!s}  mu={self.contact_mu!s}  "
            f"margin={self.contact_margin!s}  gap={self.contact_gap!s}"
        )
        if self.obstacle_box:
            print(
                "        obstacle_box: "
                f"hx={self.obstacle_box_hx:.3f}m  hy={self.obstacle_box_hy:.3f}m  hz={self.obstacle_box_hz:.3f}m"
            )

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            if self.spin_body_indices.shape[0] > 0:
                wp.launch(
                    kernel=spin_first_capsules_kernel,
                    dim=self.spin_body_indices.shape[0],
                    inputs=[self.spin_body_indices, self.spin_rates, self.sim_dt],
                    outputs=[self.state_0.body_q, self.state_1.body_q],
                    device=self.spin_body_indices.device,
                )

            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        if not _viewer_running(self.viewer):
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="example_palatial_cable")
    parser.add_argument(
        "usd",
        nargs="?",
        default=None,
        help="Path to a converted cable *.newton.usda. If omitted, a temporary flatRect ribbon asset is generated.",
    )
    parser.add_argument("--steps", type=int, default=600, help="Frames to simulate when not in GUI mode")
    parser.add_argument("--substeps", type=int, default=None, help="Override newton:solver:substeps from the USDA")
    parser.add_argument("--gui", action="store_true", help="Open ViewerGL and run until the window is closed")
    parser.add_argument("--device", default=None, help="Warp device, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument(
        "--solver-override",
        choices=("vbd", "vbd_palatial"),
        default=None,
        help="Override the solver baked into the USDA",
    )
    parser.add_argument(
        "--anchor-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the first cable segment so the bundle hangs from one end",
    )
    parser.add_argument(
        "--anchor-last",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the last cable segment as well, useful for two-end twist cases.",
    )
    parser.add_argument(
        "--spin-rate",
        type=float,
        default=0.5,
        help="Angular speed [rad/s] applied to the anchored first segment around the cable axis",
    )
    parser.add_argument(
        "--spin-last-rate",
        type=float,
        default=0.0,
        help="Angular speed [rad/s] applied to the anchored last segment around the cable axis.",
    )
    parser.add_argument(
        "--extra-drop-height",
        type=float,
        default=0.0,
        help="Additional world-space lift [m] applied to the cable before simulation",
    )
    parser.add_argument(
        "--obstacle-box",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For simple ribbon or rod assets, add a static box above the ground so the cable can drop onto it.",
    )
    parser.add_argument("--obstacle-box-hx", type=float, default=DEFAULT_OBSTACLE_BOX_HX, help="Obstacle box half-width in X [m].")
    parser.add_argument("--obstacle-box-hy", type=float, default=DEFAULT_OBSTACLE_BOX_HY, help="Obstacle box half-width in Y [m].")
    parser.add_argument("--obstacle-box-hz", type=float, default=DEFAULT_OBSTACLE_BOX_HZ, help="Obstacle box half-height in Z [m].")
    parser.add_argument(
        "--record-mp4",
        default=None,
        help="Output mp4 path. Uses ViewerGL (headless unless --gui) and pipes frames to ffmpeg.",
    )
    parser.add_argument("--mp4-fps", type=int, default=60, help="Output mp4 framerate (default 60)")
    parser.add_argument("--top-view", action="store_true", help="Place camera straight above the cable looking down")
    parser.add_argument("--contact-ke", type=float, default=None, help="Override rigid contact stiffness for all shapes.")
    parser.add_argument("--contact-kd", type=float, default=None, help="Override rigid contact damping for all shapes.")
    parser.add_argument("--contact-kf", type=float, default=None, help="Override friction damping for all shapes.")
    parser.add_argument("--contact-mu", type=float, default=None, help="Override Coulomb friction for all shapes.")
    parser.add_argument("--contact-margin", type=float, default=None, help="Override per-shape contact margin [m].")
    parser.add_argument("--contact-gap", type=float, default=None, help="Override per-shape contact gap [m].")
    parser.add_argument(
        "--no-auto-camera",
        action="store_true",
        help="Skip the recording auto-camera so the viewer keeps its default camera",
    )
    args = parser.parse_args(argv)
    resolved_usd = _resolve_input_usd(
        args.usd,
        substeps=args.substeps,
        solver_override=args.solver_override,
    )

    from newton import viewer as v

    if args.record_mp4:
        viewer = v.ViewerGL(headless=not args.gui)
    elif args.gui:
        viewer = v.ViewerGL(headless=False)
    else:
        viewer = v.ViewerNull()

    example = Example(
        viewer,
        resolved_usd,
        substeps=args.substeps,
        device=args.device,
        solver_override=args.solver_override,
        anchor_first=args.anchor_first,
        anchor_last=args.anchor_last,
        spin_rate=args.spin_rate,
        spin_last_rate=args.spin_last_rate,
        extra_drop_height=args.extra_drop_height,
        obstacle_box=args.obstacle_box,
        obstacle_box_hx=args.obstacle_box_hx,
        obstacle_box_hy=args.obstacle_box_hy,
        obstacle_box_hz=args.obstacle_box_hz,
        contact_ke=args.contact_ke,
        contact_kd=args.contact_kd,
        contact_kf=args.contact_kf,
        contact_mu=args.contact_mu,
        contact_margin=args.contact_margin,
        contact_gap=args.contact_gap,
    )

    body_points = example.state_0.body_q.numpy()[:, 0:3]
    if args.top_view and hasattr(example.viewer, "set_camera"):
        _set_top_camera(example.viewer, body_points)
    elif args.record_mp4 and not args.no_auto_camera and hasattr(example.viewer, "set_camera"):
        _set_front_camera(example.viewer, body_points)

    ffmpeg_proc = None
    if args.record_mp4:
        import shutil
        import subprocess

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not on PATH; cannot record mp4")
        example.render()
        frame = example.viewer.get_frame()
        height, width, _ = frame.shape
        enc_width = width - (width % 2)
        enc_height = height - (height % 2)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{enc_width}x{enc_height}",
            "-r",
            str(args.mp4_fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-preset",
            "fast",
            args.record_mp4,
        ]
        ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        ffmpeg_proc.stdin.write(frame.numpy()[:enc_height, :enc_width, :].tobytes())
        print(f"  recording mp4: {args.record_mp4}  size={enc_width}x{enc_height}  fps={args.mp4_fps}")

    frame_index = 1 if ffmpeg_proc is not None else 0
    use_step_count = bool(args.record_mp4) or not args.gui
    try:
        while True:
            if not _viewer_running(viewer):
                break
            if use_step_count and frame_index >= args.steps:
                break
            example.step()
            example.render()
            if ffmpeg_proc is not None:
                ffmpeg_proc.stdin.write(example.viewer.get_frame().numpy()[:enc_height, :enc_width, :].tobytes())
            frame_index += 1
    finally:
        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait(timeout=30)
            except Exception:
                ffmpeg_proc.kill()
        try:
            viewer.close()
        except Exception:
            pass

    print(f"[done] frames={frame_index}  sim_time={example.sim_time:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
