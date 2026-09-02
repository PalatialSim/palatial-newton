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
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from ._src.geometry.types import OpenPBRMaterial

RENDER_MODES = frozenset({"RealTimePathTracing", "PathTracing", "Minimal"})
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
VIDEO_SUFFIXES = frozenset({".mkv", ".mov", ".mp4"})

# Friendly public names map to the exact OVRTX 0.4.1 render-var source names.
# Keep this table explicit: Replicator names such as ``Normals`` and
# ``InstanceSegmentation`` are not OVRTX camera AOVs.
RENDER_VARS = {
    "rgb": "LdrColor",
    "hdr": "HdrColor",
    "normals": "NormalSD",
    "depth": "DepthSD",
    "distance_to_camera": "DistanceToCameraSD",
    "distance_to_image_plane": "DistanceToImagePlaneSD",
    "albedo": "DiffuseAlbedoSD",
    "camera_position": "Camera3dPositionSD",
    "semantic_segmentation": "SemanticSegmentation",
    "semantic_id_map": "SemanticIdMap",
}


def _normalize_datastore_cache(datastore_cache: str | Path | None) -> str | None:
    """Return the URI syntax required by OVRTX's datastore cache config."""
    if datastore_cache is None:
        return None
    value = str(datastore_cache)
    supported_prefixes = ("local://", "grpcdns://", "grpcdns_notls://")
    if value.startswith(supported_prefixes):
        return value
    if "://" in value:
        raise ValueError("OVRTX datastore_cache must use local://, grpcdns://, or grpcdns_notls://")
    return f"local://{Path(value).expanduser().resolve()}"


@dataclass(frozen=True)
class OVRTXMaterial(OpenPBRMaterial):
    """One dual-context visual material for OVRTX and portable USD viewers.

    OVRTX consumes the MaterialX/OpenPBR context, including transmission,
    coating, and physical thin-film interference. Other USD viewers receive a
    ``UsdPreviewSurface`` fallback authored from the same description.
    ``preview_opacity`` may approximate transmission for viewers whose preview
    surface implementation has no transmissive BSDF.
    """


def author_material(
    stage: Any,
    path: Any,
    material_spec: OpenPBRMaterial,
    *,
    base_color_texture: Any | None = None,
) -> Any:
    """Author one MaterialX/OpenPBR material with a Preview Surface fallback.

    Args:
        stage: Writable ``pxr.Usd.Stage``.
        path: Absolute material prim path.
        material_spec: Renderer-independent material values.
        base_color_texture: Optional staged texture asset path. The mesh must
            provide its primary UV set as ``primvars:st``/MaterialX ``UV0``.

    Returns:
        The authored ``pxr.UsdShade.Material``.
    """
    try:
        from pxr import Gf, Sdf, UsdShade
    except ImportError as error:  # pragma: no cover - import error is environment-specific
        raise ImportError("OVRTX material authoring requires the optional USD dependencies") from error

    if not isinstance(material_spec, OpenPBRMaterial):
        raise TypeError("material_spec must be an OpenPBRMaterial")
    material_path = path if isinstance(path, Sdf.Path) else Sdf.Path(str(path))
    if not material_path.IsAbsolutePath() or not material_path.IsPrimPath():
        raise ValueError(f"OVRTX material path must be an absolute prim path: {material_path}")

    for child_name in ("PreviewSurface", "TexCoordReader", "BaseColorTexture", "OpenPBR", "BaseColorImage"):
        stage.RemovePrim(material_path.AppendChild(child_name))

    material = UsdShade.Material.Define(stage, material_path)
    preview = UsdShade.Shader.Define(stage, material_path.AppendChild("PreviewSurface"))
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(material_spec.roughness))
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(material_spec.metallic))
    preview.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(material_spec.fallback_opacity))
    preview.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(material_spec.ior))
    preview.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(float(material_spec.coat))
    preview.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(float(material_spec.coat_roughness))

    texture_asset = None
    if base_color_texture is not None:
        texture_asset = (
            base_color_texture
            if isinstance(base_color_texture, Sdf.AssetPath)
            else Sdf.AssetPath(str(base_color_texture))
        )
        st_reader = UsdShade.Shader.Define(stage, material_path.AppendChild("TexCoordReader"))
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        texture = UsdShade.Shader.Define(stage, material_path.AppendChild("BaseColorTexture"))
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_asset)
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    else:
        preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*material_spec.color))
    preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")

    openpbr = UsdShade.Shader.Define(stage, material_path.AppendChild("OpenPBR"))
    openpbr.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
    if texture_asset is not None:
        image = UsdShade.Shader.Define(stage, material_path.AppendChild("BaseColorImage"))
        image.CreateIdAttr("ND_image_color3")
        image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_asset)
        image.GetInput("file").GetAttr().SetColorSpace("sRGB")
        image.CreateOutput("out", Sdf.ValueTypeNames.Color3f)
        openpbr.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(image.ConnectableAPI(), "out")
    else:
        openpbr.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*material_spec.color))
    openpbr.CreateInput("base_metalness", Sdf.ValueTypeNames.Float).Set(float(material_spec.metallic))
    openpbr.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(float(material_spec.roughness))
    openpbr.CreateInput("specular_ior", Sdf.ValueTypeNames.Float).Set(float(material_spec.ior))
    openpbr.CreateInput("transmission_weight", Sdf.ValueTypeNames.Float).Set(float(material_spec.transmission))
    openpbr.CreateInput("transmission_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*material_spec.transmitted_color)
    )
    openpbr.CreateInput("transmission_depth", Sdf.ValueTypeNames.Float).Set(float(material_spec.transmission_depth))
    openpbr.CreateInput("coat_weight", Sdf.ValueTypeNames.Float).Set(float(material_spec.coat))
    openpbr.CreateInput("coat_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*material_spec.coat_color))
    openpbr.CreateInput("coat_roughness", Sdf.ValueTypeNames.Float).Set(float(material_spec.coat_roughness))
    openpbr.CreateInput("coat_ior", Sdf.ValueTypeNames.Float).Set(float(material_spec.coat_ior))
    openpbr.CreateInput("thin_film_weight", Sdf.ValueTypeNames.Float).Set(float(material_spec.thin_film))
    openpbr.CreateInput("thin_film_thickness", Sdf.ValueTypeNames.Float).Set(float(material_spec.thin_film_thickness))
    openpbr.CreateInput("thin_film_ior", Sdf.ValueTypeNames.Float).Set(float(material_spec.thin_film_ior))
    openpbr.CreateInput("geometry_opacity", Sdf.ValueTypeNames.Float).Set(float(material_spec.opacity))
    openpbr.CreateInput("geometry_thin_walled", Sdf.ValueTypeNames.Bool).Set(material_spec.thin_walled)
    openpbr.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mtlx").ConnectToSource(openpbr.ConnectableAPI(), "out")
    return material


@dataclass(frozen=True)
class OVRTXConfig:
    """Live OVRTX viewer configuration.

    Delivery is intentionally absent from this object. A target path is an
    image, video, or suffix-free frame directory by its shape; callers do not
    select a second output mode.
    """

    width: int = 1280
    height: int = 720
    render_mode: str = "Minimal"
    render_every: int = 1
    warmup_frames: int = 4
    samples_per_frame: int = 1
    render_vars: tuple[str, ...] = ("rgb",)
    camera_focal_length: float = 24.0
    camera_horizontal_aperture: float = 20.955
    dome_light_intensity: float = 1000.0
    key_light_intensity: float = 3000.0
    default_material_roughness: float = 0.6
    script_path: str | None = None
    video_codec: str = "libx264"
    video_crf: int = 18
    video_preset: str = "medium"
    render_product_path: str = "/Render/Newton"
    app_id: str = "newton.ovrtx"
    keep_system_alive: bool = True
    datastore_cache: str | None = None
    on_frame: Callable[[int, dict[str, np.ndarray]], None] | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("OVRTX width and height must be positive")
        if self.render_mode not in RENDER_MODES:
            raise ValueError(f"Unsupported OVRTX render mode: {self.render_mode}")
        if self.render_every <= 0:
            raise ValueError("OVRTX render_every must be positive")
        if self.warmup_frames <= 0 or self.samples_per_frame <= 0:
            raise ValueError("OVRTX warmup and samples-per-frame values must be positive")
        if not self.render_vars:
            raise ValueError("OVRTX render_vars must contain at least one render variable")
        unknown = set(self.render_vars) - RENDER_VARS.keys()
        if unknown:
            raise ValueError(f"Unknown OVRTX render vars {sorted(unknown)}; choose from {sorted(RENDER_VARS)}")
        if "rgb" not in self.render_vars:
            raise ValueError("OVRTX render_vars must include 'rgb' for image/video delivery")
        if not self.render_product_path.startswith("/Render/"):
            raise ValueError("OVRTX render product path must be of the form '/Render/<name>'")


def ovrtx_available(*, verbose: bool = False) -> bool:
    """Return whether the optional native OVRTX runtime imports successfully."""
    try:
        import ovrtx  # noqa: F401, PLC0415
        import ovstage  # noqa: F401, PLC0415
    except ImportError as error:
        if verbose:
            print(f"OVRTX unavailable: {error} (install with: uv sync --extra ovrtx)")
        return False
    return True


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

    def bind_attribute(
        self,
        prim_paths: Sequence[str],
        attribute_name: str,
        *,
        is_array: bool = False,
        semantic: Any = None,
    ) -> OVRTXAttributeBinding:
        """Bind a populated attribute for efficient per-frame script writes.

        The script owns the returned binding and should close it from its
        ``on_stage_close`` hook.
        """
        return self._backend.bind_attribute(
            prim_paths,
            attribute_name,
            is_array=is_array,
            semantic=semantic,
        )


class OVRTXAttributeBinding:
    """Persistent OVStage query used for low-overhead live attribute writes."""

    def __init__(
        self,
        backend: OVRTXStage,
        prim_paths: Sequence[str],
        attribute_name: str,
        *,
        is_array: bool,
        semantic: Any,
    ):
        if not prim_paths:
            raise ValueError("OVRTX attribute bindings require at least one prim path")
        self._backend = backend
        self._size = len(prim_paths)
        self._paths = backend._ovstage.PathDictionary(backend.stage)
        self._paths.__enter__()
        self._path_list = self._paths.create_path_list_from_strings(list(prim_paths))
        self._query = backend.stage.query_from_path_list(self._path_list)
        self._attribute = self._paths.intern_token(attribute_name)
        self._is_array = is_array
        self._semantic = semantic
        self._closed = False

    def write(self, values: np.ndarray, *, ordinal: int | None = None, publish: bool = True) -> int:
        """Write one value per bound prim and optionally publish the ordinal."""
        if self._closed:
            raise RuntimeError("OVRTX attribute binding is closed")
        values = np.ascontiguousarray(values)
        if len(values) != self._size:
            raise ValueError(f"OVRTX binding expected {self._size} values, got {len(values)}")
        if ordinal is None:
            ordinal = self._backend._next_ordinal()
        lanes = int(np.prod(values.shape[1:])) if values.ndim > 1 else 1
        dtype = self._backend._ovstage.numpy_to_dldatatype(values.dtype, lanes=lanes)
        tensor = self._backend._ovstage.make_dltensor(values, dtype=dtype, shape=[len(values)], ndim=1)
        kwargs = {
            "ordinal": ordinal,
            "tensors": tensor,
            "is_array": self._is_array,
        }
        if self._semantic is not None:
            kwargs["semantic"] = self._semantic
        self._backend.stage.write_attribute(self._query, self._attribute, **kwargs).wait()
        if publish:
            self._backend._publish(ordinal)
        return ordinal

    def close(self) -> None:
        """Release the native query, path list, and dictionary."""
        if self._closed:
            return
        self._closed = True
        try:
            self._query.release().wait()
        finally:
            try:
                self._paths.destroy_path_list(self._path_list)
            finally:
                self._paths.__exit__(None, None, None)


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
        render_vars: Sequence[str] = ("rgb",),
        keep_system_alive: bool = True,
        datastore_cache: str | Path | None = None,
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
            render_vars,
        )
        self.render_product_path = render_product_path
        self.width = width
        self.height = height
        self.warmup_frames = warmup_frames
        self.samples_per_frame = samples_per_frame
        self.fps = fps
        self.script_path = Path(script_path).expanduser().resolve() if script_path is not None else None
        self.app_id = app_id
        self.render_vars = tuple(render_vars)
        self.keep_system_alive = keep_system_alive
        self.datastore_cache = _normalize_datastore_cache(datastore_cache)
        unknown = set(self.render_vars) - RENDER_VARS.keys()
        if unknown:
            raise ValueError(f"Unknown OVRTX render vars {sorted(unknown)}; choose from {sorted(RENDER_VARS)}")

        self._ovrtx: ModuleType | None = None
        self._ovstage: ModuleType | None = None
        self._script: ModuleType | None = None
        self._renderer: Any = None
        self._stage: Any = None
        self._ordinal = 0
        self._context = OVRTXScriptContext(self)
        self._closed = False

    def bind_attribute(
        self,
        prim_paths: Sequence[str],
        attribute_name: str,
        *,
        is_array: bool = False,
        semantic: Any = None,
    ) -> OVRTXAttributeBinding:
        """Create a persistent direct-write binding for populated prims."""
        self.open()
        return OVRTXAttributeBinding(self, prim_paths, attribute_name, is_array=is_array, semantic=semantic)

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

        renderer_config = ovrtx.RendererConfig(
            keep_system_alive=self.keep_system_alive,
            datastore_cache=self.datastore_cache,
        )
        self._renderer = ovrtx.Renderer(renderer_config)
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
        render_vars = self.step_and_read(steps)
        pixels = render_vars["rgb"]
        scripted_pixels = self._call_script("after_frame", self._context, pixels)
        if scripted_pixels is not None:
            pixels = np.asarray(scripted_pixels)
        pixels = _validate_pixels(pixels, self.width, self.height)
        return RenderedFrame(index=frame_index, time_seconds=time_seconds, pixels=pixels)

    def step_and_read(self, steps: int | None = None) -> dict[str, np.ndarray]:
        """Step the attached renderer and copy configured render vars to NumPy."""
        self.open()
        if steps is None:
            steps = self.samples_per_frame
        if steps <= 0:
            raise ValueError("OVRTX renderer steps must be positive")
        return self._step_and_copy(steps)

    def render_live_frame(self, frame_index: int, *, steps: int | None = None) -> dict[str, np.ndarray]:
        """Render the current directly-mutated OVStage state.

        Unlike :meth:`render_frame`, this method does not evaluate USD time.
        It is the live-viewer path used after simulation attributes have been
        written through persistent OVStage bindings.
        """
        if frame_index < 0:
            raise ValueError("OVRTX frame index must be non-negative")
        self._context.frame_index = frame_index
        self._context.time_seconds = frame_index / self.fps
        self._call_script("before_frame", self._context)
        render_vars = self.step_and_read(steps)
        scripted_pixels = self._call_script("after_frame", self._context, render_vars["rgb"])
        if scripted_pixels is not None:
            render_vars["rgb"] = _validate_pixels(np.asarray(scripted_pixels), self.width, self.height)
        return render_vars

    def notify_render_complete(self, output: Path | None) -> None:
        """Invoke the optional completion hook for a live viewer target."""
        self._call_script("on_render_complete", self._context, output)

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

    def _step_and_copy(self, steps: int) -> dict[str, np.ndarray]:
        products: Any = None
        for _ in range(steps):
            products = self.renderer.step(
                render_products={self.render_product_path},
                delta_time=1.0 / self.fps,
                ordinal=self.ordinal,
            )
        try:
            product = products[self.render_product_path]
        except (KeyError, TypeError) as error:
            raise RuntimeError(f"OVRTX returned no render product for {self.render_product_path}") from error

        frames = list(product.frames)
        if not frames:
            raise RuntimeError(f"OVRTX returned no frames for {self.render_product_path}")
        frame = frames[-1]
        outputs: dict[str, np.ndarray] = {}
        for friendly_name in self.render_vars:
            source_name = RENDER_VARS[friendly_name]
            try:
                render_var = frame.render_vars[source_name]
            except (KeyError, TypeError) as error:
                raise RuntimeError(
                    f"OVRTX returned no {source_name} render var for {self.render_product_path}"
                ) from error
            mapped = render_var.map(device=self._ovrtx.Device.CPU)
            try:
                outputs[friendly_name] = np.from_dlpack(mapped).copy()
            finally:
                mapped.unmap()
        return outputs

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


def camera_matrix(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
) -> np.ndarray:
    """Return an OVStage row-major camera transform looking at ``target``."""
    qw, qx, qy, qz = _camera_orientation(position, target)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    matrix[0, 1] = 2.0 * (qx * qy + qz * qw)
    matrix[0, 2] = 2.0 * (qx * qz - qy * qw)
    matrix[1, 0] = 2.0 * (qx * qy - qz * qw)
    matrix[1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    matrix[1, 2] = 2.0 * (qy * qz + qx * qw)
    matrix[2, 0] = 2.0 * (qx * qz + qy * qw)
    matrix[2, 1] = 2.0 * (qy * qz - qx * qw)
    matrix[2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
    matrix[3, :3] = position
    matrix[3, 3] = 1.0
    return matrix


def _compose_render_stage(
    source_path: Path,
    render_product_path: str,
    width: int,
    height: int,
    camera_position: tuple[float, float, float],
    camera_target: tuple[float, float, float],
    render_mode: str,
    render_vars: Sequence[str] = ("rgb",),
) -> str:
    """Compose renderer-owned presentation over the complete source USD stage."""
    if not render_product_path.startswith("/"):
        raise ValueError("OVRTX render product path must be absolute")
    if width <= 0 or height <= 0:
        raise ValueError("OVRTX output width and height must be positive")
    if render_mode not in RENDER_MODES:
        raise ValueError(f"Unsupported OVRTX render mode: {render_mode}")
    unknown = set(render_vars) - RENDER_VARS.keys()
    if unknown:
        raise ValueError(f"Unknown OVRTX render vars {sorted(unknown)}; choose from {sorted(RENDER_VARS)}")

    product_parts = [part for part in render_product_path.split("/") if part]
    if len(product_parts) != 2 or product_parts[0] != "Render":
        raise ValueError("OVRTX render product path must be of the form '/Render/<name>'")

    quat_w, quat_x, quat_y, quat_z = _camera_orientation(camera_position, camera_target)
    source_asset_path = source_path.as_posix().replace("@", "%40")
    source_names = [RENDER_VARS[name] for name in render_vars]
    ordered_vars = ", ".join(f"<{name}>" for name in source_names)
    render_var_defs = "\n".join(
        f'''        def RenderVar "{name}"
        {{
            uniform string sourceName = "{name}"
        }}'''
        for name in source_names
    )
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

def "Lighting"
{{
    def DomeLight "Dome"
    {{
        float inputs:intensity = 1000
    }}

    def DistantLight "Key"
    {{
        float inputs:angle = 1
        float inputs:intensity = 3000
        float3 xformOp:rotateXYZ = (-55, 30, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}
}}

def "Render"
{{
    def RenderProduct "{product_parts[1]}"
    {{
        custom token omni:rtx:rendermode = "{render_mode}"
        uniform int2 resolution = ({width}, {height})
        rel camera = </NewtonCamera>
        rel orderedVars = [{ordered_vars}]

{render_var_defs}
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
    render_vars: Sequence[str] = ("rgb",),
    keep_system_alive: bool = True,
    datastore_cache: str | Path | None = None,
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
        render_vars=render_vars,
        keep_system_alive=keep_system_alive,
        datastore_cache=datastore_cache,
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
