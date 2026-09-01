# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""OVRTX rendering helpers for Newton USD recordings.

The optional ``newton[ovrtx]`` extra adds NVIDIA's OVRTX and ovstage wheels.
This module intentionally imports those packages only when rendering is
requested, so simulations and non-RTX viewers do not acquire a new runtime
dependency.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

RENDER_MODES = frozenset({"RealTimePathTracing", "PathTracing", "Minimal"})


def _format_vec3(values: tuple[float, float, float]) -> str:
    """Format a three-element vector for inline USDA."""
    return ", ".join(f"{value:.9g}" for value in values)


def _camera_orientation(
    position: tuple[float, float, float], target: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    """Return the USD orientation quaternion for a camera looking at *target*.

    USD cameras look along their local negative-Z axis. The construction
    matches Newton's existing USD rendering utility: rotate around Z to face
    the target in the ground plane, then pitch around X.
    """
    delta_x = target[0] - position[0]
    delta_y = target[1] - position[1]
    delta_z = target[2] - position[2]
    planar_distance = math.hypot(delta_x, delta_y)
    if math.isclose(planar_distance, 0.0) and math.isclose(delta_z, 0.0):
        raise ValueError("OVRTX camera position and target must be different")

    rotation_z = math.atan2(-delta_x, delta_y)
    rotation_x = math.atan2(delta_z, planar_distance) + math.pi / 2.0
    half_z = rotation_z / 2.0
    half_x = rotation_x / 2.0
    cos_z, sin_z = math.cos(half_z), math.sin(half_z)
    cos_x, sin_x = math.cos(half_x), math.sin(half_x)

    # Quaternion product q_z * q_x, returned in USD order: real, i, j, k.
    return (cos_z * cos_x, cos_z * sin_x, sin_z * sin_x, sin_z * cos_x)


def _compose_render_stage(
    source_path: Path,
    render_product_path: str,
    width: int,
    height: int,
    camera_position: tuple[float, float, float],
    camera_target: tuple[float, float, float],
    render_mode: str,
) -> str:
    """Compose an OVRTX render configuration over a Newton USD recording."""
    if not render_product_path.startswith("/"):
        raise ValueError("OVRTX render product path must be absolute")
    if width <= 0 or height <= 0:
        raise ValueError("OVRTX output width and height must be positive")
    if render_mode not in RENDER_MODES:
        raise ValueError(f"Unsupported OVRTX render mode: {render_mode}")

    product_parts = [part for part in render_product_path.split("/") if part]
    if len(product_parts) != 2 or product_parts[0] != "Render":
        raise ValueError("OVRTX render product path must be of the form '/Render/<name>'")

    quat_w, quat_x, quat_y, quat_z = _camera_orientation(camera_position, camera_target)
    source_asset_path = source_path.as_posix().replace("@", "%40")
    return f'''#usda 1.0
(
    subLayers = [
        @{source_asset_path}@
    ]
)

def Camera "NewtonCamera"
{{
    float2 clippingRange = (0.01, 100000)
    float focalLength = 24
    float horizontalAperture = 20.955
    quatf xformOp:orient = ({quat_w:.9g}, {quat_x:.9g}, {quat_y:.9g}, {quat_z:.9g})
    double3 xformOp:translate = ({_format_vec3(camera_position)})
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
}}

def "Render"
{{
    def RenderProduct "{product_parts[1]}"
    {{
        custom token omni:rtx:rendermode = "{render_mode}"
        uniform int2 resolution = ({width}, {height})
        rel camera = </NewtonCamera>
        rel orderedVars = [<LdrColor>]

        def RenderVar "LdrColor"
        {{
            uniform string sourceName = "LdrColor"
        }}
    }}
}}
'''


def render_usd(
    source_path: str | Path,
    output_path: str | Path,
    *,
    render_product_path: str = "/Render/Newton",
    width: int = 1280,
    height: int = 720,
    camera_position: tuple[float, float, float] = (3.0, -3.0, 2.5),
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.5),
    render_mode: str = "RealTimePathTracing",
    warmup_frames: int = 40,
    usd_time_code: float | None = None,
    fps: float = 60.0,
    app_id: str = "newton.ovrtx",
) -> Path:
    """Render one LDR PNG from a Newton time-sampled USD recording.

    The function uses OVRTX's current ``ovstage`` integration rather than its
    deprecated renderer-owned USD APIs. If ``usd_time_code`` is supplied, it
    selects that authored USD timecode.

    Args:
        source_path: Existing Newton USD recording to render.
        output_path: PNG file to write.
        render_product_path: Absolute OVRTX render product path.
        width: Render width in pixels.
        height: Render height in pixels.
        camera_position: Camera position in stage metres.
        camera_target: Point at which the camera is aimed in stage metres.
        render_mode: OVRTX camera render mode.
        warmup_frames: Number of renderer steps before downloading the final PNG.
        usd_time_code: Authored USD timecode, or ``None`` for the default time.
        fps: Sensor step rate used for the render request.
        app_id: OVRTX application identifier used for cache isolation.

    Returns:
        Absolute path of the rendered PNG.

    Raises:
        FileNotFoundError: If ``source_path`` does not exist.
        ImportError: If the optional OVRTX runtime is not installed.
        RuntimeError: If OVRTX returns no LDR frame for the render product.
    """
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Newton USD recording does not exist: {source}")
    if fps <= 0.0:
        raise ValueError("OVRTX render fps must be positive")
    if warmup_frames <= 0:
        raise ValueError("OVRTX warmup frames must be positive")

    try:
        import ovrtx  # noqa: PLC0415 - optional GPU runtime
        import ovstage  # noqa: PLC0415 - optional GPU runtime
        from PIL import Image
    except ImportError as error:
        raise ImportError(
            "OVRTX rendering requires the optional runtime. Install it with: uv sync --extra ovrtx"
        ) from error

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = ovrtx.Renderer()
    stage = ovstage.Stage(app_id)
    renderer.attach_ovstage(stage)
    try:
        ordinal = 1
        ovstage.population.open_usd_from_string(
            stage,
            _compose_render_stage(
                source,
                render_product_path,
                width,
                height,
                camera_position,
                camera_target,
                render_mode,
            ),
            ordinal=ordinal,
        )
        stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()

        if usd_time_code is not None:
            ordinal += 1
            ovstage.population.update_from_usd_time_async(stage, ordinal=ordinal, time_code=usd_time_code).wait()
            stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()

        for _ in range(warmup_frames):
            products = renderer.step(
                render_products={render_product_path},
                delta_time=1.0 / fps,
                ordinal=ordinal,
            )
        product = products.get(render_product_path)
        if product is None:
            raise RuntimeError(f"OVRTX returned no render product for {render_product_path}")

        for frame in product.frames:
            ldr_color = frame.render_vars.get("LdrColor")
            if ldr_color is None:
                continue
            mapped = ldr_color.map(device=ovrtx.Device.CPU)
            try:
                pixels = np.from_dlpack(mapped).copy()
            finally:
                mapped.unmap()
            Image.fromarray(pixels).save(output)
            return output

        raise RuntimeError(f"OVRTX returned no LdrColor frame for {render_product_path}")
    finally:
        renderer.detach_ovstage()
        stage.destroy()
        renderer.destroy()
