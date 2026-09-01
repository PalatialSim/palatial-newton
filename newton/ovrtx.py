# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stage-first OVRTX rendering for Newton USD recordings.

The optional ``newton[ovrtx]`` extra adds NVIDIA's OVRTX and OVStage wheels.
Imports stay lazy so simulations and non-RTX viewers do not acquire a GPU
runtime dependency.

``OVRTXStage`` is the public rendering seam. It opens one complete USD stage,
attaches one OVRTX renderer for the lifetime of the session, and routes the
same timeline to an image, a video, or a frame directory based on the output
target. A render script may customize stage composition and lifecycle hooks.
"""

from __future__ import annotations

import importlib.util
import math
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

RENDER_MODES = frozenset({"RealTimePathTracing", "PathTracing", "Minimal"})
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
VIDEO_SUFFIXES = frozenset({".mkv", ".mov", ".mp4"})


@dataclass(frozen=True)
class RenderedFrame:
    """One rendered USD timeline sample."""

    index: int
    time_seconds: float
    pixels: np.ndarray


class OVRTXScriptContext:
    """Stable context exposed to optional OVRTX render scripts.

    Scripts may inspect ``stage`` and ``renderer`` directly. Mutations must use
    :meth:`mutation` so OVStage receives a monotonically increasing ordinal and
    publishes the write before OVRTX consumes it.
    """

    def __init__(self, backend: OVRTXStage):
        self._backend = backend
        self.frame_index: int | None = None
        self.time_seconds: float | None = None

    @property
    def source_path(self) -> Path:
        """Absolute path of the source USD stage."""
        return self._backend.source_path

    @property
    def stage(self) -> Any:
        """The attached ``ovstage.Stage`` instance."""
        return self._backend.stage

    @property
    def renderer(self) -> Any:
        """The attached ``ovrtx.Renderer`` instance."""
        return self._backend.renderer

    @property
    def ovstage(self) -> ModuleType:
        """The loaded ``ovstage`` module."""
        return self._backend._ovstage

    @property
    def ovrtx(self) -> ModuleType:
        """The loaded ``ovrtx`` module."""
        return self._backend._ovrtx

    @property
    def ordinal(self) -> int:
        """Latest committed OVStage ordinal."""
        return self._backend.ordinal

    @contextmanager
    def mutation(self) -> Iterator[int]:
        """Reserve and publish one OVStage mutation ordinal."""
        ordinal = self._backend._next_ordinal()
        yield ordinal
        self._backend._publish(ordinal)


class OVRTXStage:
    """Own one OVStage + OVRTX session for a complete USD timeline.

    The source stage is composed beneath a renderer-owned camera and render
    product without modifying the source file. The renderer is attached once,
    reused for every frame, and destroyed when the context manager exits.

    A script file may define any of these optional hooks:

    ``compose_stage(source_path, default_stage) -> str``
        Replace or edit the inline USDA root before OVStage opens it.
    ``on_stage_open(context)``
        Configure the populated stage before timeline rendering.
    ``before_frame(context)``
        Mutate the current frame after USD time evaluation and before rendering.
    ``after_frame(context, pixels) -> ndarray | None``
        Inspect or replace the copied CPU pixel array.
    ``on_stage_close(context)``
        Perform final script cleanup while the native stage is still alive.
    """

    def __init__(
        self,
        source_path: str | Path,
        *,
        render_product_path: str = "/Render/Newton",
        width: int = 1280,
        height: int = 720,
        camera_position: tuple[float, float, float] = (3.0, -3.0, 2.5),
        camera_target: tuple[float, float, float] = (0.0, 0.0, 0.5),
        render_mode: str = "RealTimePathTracing",
        warmup_frames: int = 40,
        samples_per_frame: int = 1,
        fps: float = 60.0,
        script_path: str | Path | None = None,
        app_id: str = "newton.ovrtx",
    ):
        self.source_path = Path(source_path).expanduser().resolve()
        if not self.source_path.is_file():
            raise FileNotFoundError(f"Newton USD recording does not exist: {self.source_path}")
        if fps <= 0.0:
            raise ValueError("OVRTX render fps must be positive")
        if warmup_frames <= 0:
            raise ValueError("OVRTX warmup frames must be positive")
        if samples_per_frame <= 0:
            raise ValueError("OVRTX samples per frame must be positive")

        self._stage_text = _compose_render_stage(
            self.source_path,
            render_product_path,
            width,
            height,
            camera_position,
            camera_target,
            render_mode,
        )
        self.render_product_path = render_product_path
        self.width = width
        self.height = height
        self.warmup_frames = warmup_frames
        self.samples_per_frame = samples_per_frame
        self.fps = fps
        self.script_path = Path(script_path).expanduser().resolve() if script_path is not None else None
        self.app_id = app_id

        self._ovrtx: ModuleType | None = None
        self._ovstage: ModuleType | None = None
        self._script: ModuleType | None = None
        self._renderer: Any = None
        self._stage: Any = None
        self._ordinal = 0
        self._context = OVRTXScriptContext(self)
        self._closed = False

    @property
    def stage(self) -> Any:
        """The live OVStage instance; available after entering the session."""
        if self._stage is None:
            raise RuntimeError("OVRTX stage is not open")
        return self._stage

    @property
    def renderer(self) -> Any:
        """The live OVRTX renderer; available after entering the session."""
        if self._renderer is None:
            raise RuntimeError("OVRTX renderer is not open")
        return self._renderer

    @property
    def ordinal(self) -> int:
        """Latest committed OVStage ordinal."""
        return self._ordinal

    def __enter__(self) -> OVRTXStage:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def open(self) -> None:
        """Create and attach the native renderer and populate the full stage."""
        if self._renderer is not None:
            return
        if self._closed:
            raise RuntimeError("OVRTX stage sessions cannot be reopened after close")

        try:
            import ovrtx  # noqa: PLC0415 - optional GPU runtime
            import ovstage  # noqa: PLC0415 - optional GPU runtime
        except ImportError as error:
            raise ImportError(
                "OVRTX rendering requires the optional runtime. Install it with: uv sync --extra ovrtx"
            ) from error

        self._ovrtx = ovrtx
        self._ovstage = ovstage
        self._script = _load_render_script(self.script_path)
        stage_text = self._call_script("compose_stage", self.source_path, self._stage_text)
        if stage_text is None:
            stage_text = self._stage_text
        if not isinstance(stage_text, str):
            raise TypeError("OVRTX render script compose_stage() must return a USDA string or None")

        self._renderer = ovrtx.Renderer()
        self._stage = ovstage.Stage(self.app_id)
        try:
            self._renderer.attach_ovstage(self._stage)
            ordinal = self._next_ordinal()
            ovstage.population.open_usd_from_string(self._stage, stage_text, ordinal=ordinal)
            self._publish(ordinal)
            self._call_script("on_stage_open", self._context)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Detach and destroy native resources in ownership order."""
        if self._closed:
            return
        self._closed = True
        if self._renderer is None:
            return
        try:
            self._call_script("on_stage_close", self._context)
        finally:
            try:
                self._renderer.detach_ovstage()
            finally:
                if self._stage is not None:
                    self._stage.destroy()
                self._renderer.destroy()
                self._stage = None
                self._renderer = None

    def render(
        self,
        output_path: str | Path,
        *,
        frame_start: int = 0,
        frame_end: int | None = None,
        frame_step: int = 1,
        video_codec: str = "libx264",
        video_crf: int = 18,
        video_preset: str = "medium",
    ) -> Path:
        """Render the USD timeline to the target implied by ``output_path``.

        Image suffixes capture ``frame_end`` (or ``frame_start`` when no end is
        supplied). Video suffixes stream the inclusive frame range directly to
        FFmpeg as raw pixels. A suffix-free path receives a numbered PNG frame
        sequence. There is deliberately no separate output-mode flag.
        """
        self.open()
        if frame_start < 0:
            raise ValueError("OVRTX frame start must be non-negative")
        if frame_step <= 0:
            raise ValueError("OVRTX frame step must be positive")
        if frame_end is None:
            frame_end = frame_start
        if frame_end < frame_start:
            raise ValueError("OVRTX frame end must be greater than or equal to frame start")

        output = Path(output_path).expanduser().resolve()
        suffix = output.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            frame = self.render_frame(frame_end, steps=self.warmup_frames)
            _save_image(output, frame.pixels)
        elif suffix in VIDEO_SUFFIXES:
            self._render_video(
                output,
                range(frame_start, frame_end + 1, frame_step),
                codec=video_codec,
                crf=video_crf,
                preset=video_preset,
            )
        elif not suffix:
            self._render_sequence(output, range(frame_start, frame_end + 1, frame_step))
        else:
            supported = sorted(IMAGE_SUFFIXES | VIDEO_SUFFIXES)
            raise ValueError(
                f"Unsupported OVRTX render target suffix {suffix!r}; use one of {supported} or a directory"
            )

        self._call_script("on_render_complete", self._context, output)
        return output

    def render_frame(self, frame_index: int, *, steps: int | None = None) -> RenderedFrame:
        """Evaluate and render one authored USD frame using the live session."""
        self.open()
        if frame_index < 0:
            raise ValueError("OVRTX frame index must be non-negative")
        if steps is None:
            steps = self.samples_per_frame
        if steps <= 0:
            raise ValueError("OVRTX renderer steps must be positive")

        time_seconds = frame_index / self.fps
        ordinal = self._next_ordinal()
        # OVStage's usd-time parameter is seconds despite the historic
        # ``time_code`` keyword name in the 0.1 Python binding.
        self._ovstage.population.update_from_usd_time_async(
            self.stage,
            ordinal=ordinal,
            time_code=time_seconds,
        ).wait()
        self._publish(ordinal)

        self._context.frame_index = frame_index
        self._context.time_seconds = time_seconds
        self._call_script("before_frame", self._context)
        pixels = self._step_and_copy(steps)
        scripted_pixels = self._call_script("after_frame", self._context, pixels)
        if scripted_pixels is not None:
            pixels = np.asarray(scripted_pixels)
        pixels = _validate_pixels(pixels, self.width, self.height)
        return RenderedFrame(index=frame_index, time_seconds=time_seconds, pixels=pixels)

    def _render_sequence(self, output_dir: Path, frame_indices: range) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for sequence_index, frame_index in enumerate(frame_indices):
            steps = self.warmup_frames if sequence_index == 0 else self.samples_per_frame
            frame = self.render_frame(frame_index, steps=steps)
            _save_image(output_dir / f"frame_{sequence_index:06d}.png", frame.pixels)

    def _render_video(
        self,
        output: Path,
        frame_indices: range,
        *,
        codec: str,
        crf: int,
        preset: str,
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("OVRTX video output requires FFmpeg on PATH")
        if not 0 <= crf <= 51:
            raise ValueError("OVRTX video CRF must be between 0 and 51")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgba",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps / frame_indices.step),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            if process.stdin is None or process.stderr is None:
                raise RuntimeError("FFmpeg did not expose its input/output pipes")
            for sequence_index, frame_index in enumerate(frame_indices):
                steps = self.warmup_frames if sequence_index == 0 else self.samples_per_frame
                frame = self.render_frame(frame_index, steps=steps)
                process.stdin.write(np.ascontiguousarray(frame.pixels).tobytes())
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"FFmpeg failed with exit code {return_code}: {stderr}")
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg did not produce a video: {output}")

    def _step_and_copy(self, steps: int) -> np.ndarray:
        products: dict[str, Any] = {}
        for _ in range(steps):
            products = self.renderer.step(
                render_products={self.render_product_path},
                delta_time=1.0 / self.fps,
                ordinal=self.ordinal,
            )
        product = products.get(self.render_product_path)
        if product is None:
            raise RuntimeError(f"OVRTX returned no render product for {self.render_product_path}")

        for frame in reversed(product.frames):
            ldr_color = frame.render_vars.get("LdrColor")
            if ldr_color is None:
                continue
            mapped = ldr_color.map(device=self._ovrtx.Device.CPU)
            try:
                return np.from_dlpack(mapped).copy()
            finally:
                mapped.unmap()
        raise RuntimeError(f"OVRTX returned no LdrColor frame for {self.render_product_path}")

    def _next_ordinal(self) -> int:
        self._ordinal += 1
        return self._ordinal

    def _publish(self, ordinal: int) -> None:
        self.stage.advance_write_floor(ordinal, self._ovstage.Scope.ALL).wait()

    def _call_script(self, name: str, *args: Any) -> Any:
        if self._script is None:
            return None
        hook: Callable[..., Any] | None = getattr(self._script, name, None)
        if hook is None:
            return None
        if not callable(hook):
            raise TypeError(f"OVRTX render script attribute {name!r} must be callable")
        return hook(*args)


def _load_render_script(script_path: Path | None) -> ModuleType | None:
    if script_path is None:
        return None
    if not script_path.is_file():
        raise FileNotFoundError(f"OVRTX render script does not exist: {script_path}")
    module_name = f"newton_ovrtx_script_{abs(hash(script_path))}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load OVRTX render script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_pixels(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    pixels = np.asarray(pixels)
    expected_shape = (height, width, 4)
    if pixels.shape != expected_shape:
        raise ValueError(f"OVRTX pixels must have shape {expected_shape}, got {pixels.shape}")
    if pixels.dtype != np.uint8:
        raise ValueError(f"OVRTX pixels must use uint8, got {pixels.dtype}")
    return pixels


def _save_image(output: Path, pixels: np.ndarray) -> None:
    from PIL import Image

    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(pixels)
    if output.suffix.lower() in {".jpeg", ".jpg"}:
        image = image.convert("RGB")
    image.save(output)


def _format_vec3(values: tuple[float, float, float]) -> str:
    """Format a three-element vector for inline USDA."""
    return ", ".join(f"{value:.9g}" for value in values)


def _camera_orientation(
    position: tuple[float, float, float], target: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    """Return the USD orientation quaternion for a camera looking at *target*."""
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
    """Compose renderer-owned presentation over the complete source USD stage."""
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
    samples_per_frame: int = 1,
    frame_start: int = 0,
    frame_end: int | None = None,
    frame_step: int = 1,
    usd_time_code: float | None = None,
    fps: float = 60.0,
    script_path: str | Path | None = None,
    video_codec: str = "libx264",
    video_crf: int = 18,
    video_preset: str = "medium",
    app_id: str = "newton.ovrtx",
) -> Path:
    """Render a full USD stage to an image, video, or frame directory.

    ``usd_time_code`` remains as a compatibility alias for selecting one image
    frame. New callers should use the inclusive ``frame_start`` / ``frame_end``
    timeline interface. USD frames are converted to seconds before being sent
    to OVStage.
    """
    if usd_time_code is not None:
        if frame_end is not None or frame_start != 0:
            raise ValueError("usd_time_code cannot be combined with frame_start or frame_end")
        frame_start = int(usd_time_code)
        frame_end = int(usd_time_code)

    with OVRTXStage(
        source_path,
        render_product_path=render_product_path,
        width=width,
        height=height,
        camera_position=camera_position,
        camera_target=camera_target,
        render_mode=render_mode,
        warmup_frames=warmup_frames,
        samples_per_frame=samples_per_frame,
        fps=fps,
        script_path=script_path,
        app_id=app_id,
    ) as stage:
        return stage.render(
            output_path,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_step=frame_step,
            video_codec=video_codec,
            video_crf=video_crf,
            video_preset=video_preset,
        )
