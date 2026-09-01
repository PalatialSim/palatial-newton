# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent live OVRTX viewer for Newton simulations."""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from ...ovrtx import (
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    OVRTXConfig,
    OVRTXStage,
    _camera_orientation,
    _save_image,
    author_material,
)
from ..core.types import override
from ..geometry.types import Mesh, OpenPBRMaterial
from .viewer_usd import ViewerUSD

try:
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdRender, UsdShade
except ImportError:  # pragma: no cover - ViewerUSD reports the dependency
    Gf = Sdf = UsdGeom = UsdLux = UsdRender = UsdShade = None


def _transforms_to_matrices(xforms: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Convert Warp transforms to OVStage's row-major ``matrix4d`` layout."""
    xforms = np.asarray(xforms, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if xforms.ndim != 2 or xforms.shape[1] != 7:
        raise ValueError(f"Expected transforms with shape (N, 7), got {xforms.shape}")
    if scales.shape != (len(xforms), 3):
        raise ValueError(f"Expected scales with shape ({len(xforms)}, 3), got {scales.shape}")

    qx, qy, qz, qw = (xforms[:, index] for index in range(3, 7))
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    norm = np.where(norm > 0.0, norm, 1.0)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    matrices = np.zeros((len(xforms), 4, 4), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    matrices[:, 0, 1] = 2.0 * (qx * qy + qz * qw)
    matrices[:, 0, 2] = 2.0 * (qx * qz - qy * qw)
    matrices[:, 1, 0] = 2.0 * (qx * qy - qz * qw)
    matrices[:, 1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    matrices[:, 1, 2] = 2.0 * (qy * qz + qx * qw)
    matrices[:, 2, 0] = 2.0 * (qx * qz + qy * qw)
    matrices[:, 2, 1] = 2.0 * (qy * qz - qx * qw)
    matrices[:, 2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
    matrices[:, :3, :3] *= scales[:, :, None]
    matrices[:, 3, :3] = xforms[:, :3]
    matrices[:, 3, 3] = 1.0
    return matrices


@dataclass(frozen=True)
class _MeshMaterial:
    color: tuple[float, float, float]
    has_explicit_color: bool
    roughness: float
    metallic: float
    opacity: float
    ior: float
    texture: np.ndarray | str | None
    has_uvs: bool
    material: OpenPBRMaterial | None


class ViewerOVRTX(ViewerUSD):
    """USD-stage viewer whose live frames are rendered by one OVRTX session.

    Static scene data is populated once from the editable USD stage. Newton
    instance transforms then travel directly to OVStage, avoiding per-frame
    USD serialization and re-population. The target path itself determines
    delivery: image, video, or suffix-free frame directory.
    """

    @override
    def _mesh_log_options(self, geo_src: Mesh) -> dict[str, Any]:
        material = geo_src.visual_material
        return {"material": material} if material is not None else {}

    def __init__(
        self,
        output_path: str,
        render_output_path: str | None = None,
        *,
        config: OVRTXConfig | None = None,
        width: int | None = None,
        height: int | None = None,
        camera_position: tuple[float, float, float] = (3.0, -3.0, 2.5),
        camera_target: tuple[float, float, float] = (0.0, 0.0, 0.5),
        render_mode: str | None = None,
        warmup_frames: int | None = None,
        samples_per_frame: int | None = None,
        frame_step: int | None = None,
        script_path: str | None = None,
        video_codec: str | None = None,
        video_crf: int | None = None,
        video_preset: str | None = None,
        fps: int = 60,
        up_axis: str = "Z",
        num_frames: int | None = 100,
    ):
        if config is None:
            defaults = OVRTXConfig()
            config = OVRTXConfig(
                width=defaults.width if width is None else width,
                height=defaults.height if height is None else height,
                render_mode=defaults.render_mode if render_mode is None else render_mode,
                render_every=defaults.render_every if frame_step is None else frame_step,
                warmup_frames=defaults.warmup_frames if warmup_frames is None else warmup_frames,
                samples_per_frame=defaults.samples_per_frame if samples_per_frame is None else samples_per_frame,
                script_path=script_path,
                video_codec=defaults.video_codec if video_codec is None else video_codec,
                video_crf=defaults.video_crf if video_crf is None else video_crf,
                video_preset=defaults.video_preset if video_preset is None else video_preset,
            )
        elif any(
            value is not None
            for value in (
                width,
                height,
                render_mode,
                warmup_frames,
                samples_per_frame,
                frame_step,
                script_path,
                video_codec,
                video_crf,
                video_preset,
            )
        ):
            raise ValueError("Pass either OVRTXConfig or individual OVRTX settings, not both")

        self.config = config
        self.render_output_path = (
            Path(render_output_path).expanduser().resolve() if render_output_path is not None else None
        )
        self._camera_position = tuple(float(value) for value in camera_position)
        self._camera_target = tuple(float(value) for value in camera_target)
        self._session: OVRTXStage | None = None
        self._xform_binding: Any = None
        self._camera_binding: Any = None
        self._instance_paths: dict[str, list[str]] = {}
        self._pending_matrices: dict[str, np.ndarray] = {}
        self._mesh_materials: dict[str, _MeshMaterial] = {}
        self._latest_render_vars: dict[str, np.ndarray] = {}
        self._render_count = 0
        self._video_process: subprocess.Popen | None = None
        self._closed = False
        super().__init__(output_path=output_path, fps=fps, up_axis=up_axis, num_frames=num_frames)

    @override
    def set_model(self, model, max_worlds=None):
        self._close_runtime()
        self._instance_paths.clear()
        self._pending_matrices.clear()
        self._mesh_materials.clear()
        super().set_model(model, max_worlds)
        if model is None or self.stage is None:
            return

        # ViewerBase builds instance batches in set_model(), while ViewerUSD
        # normally creates their USD prims lazily on the first log_state().
        # OVRTX population happens here, so author those stable paths before
        # opening OVStage; the first live frame replaces these seed transforms.
        for shapes in self._shape_instances.values():
            visible = self._should_show_shape(shapes.flags, shapes.static)
            self.log_instances(
                shapes.name,
                shapes.mesh,
                shapes.xforms,
                shapes.scales,
                shapes.colors,
                shapes.materials,
                hidden=not visible,
            )

        self._author_render_stage()
        self.stage.GetRootLayer().Save()
        self._session = OVRTXStage(
            self.output_path,
            render_product_path=self.config.render_product_path,
            width=self.config.width,
            height=self.config.height,
            camera_position=self._camera_position,
            camera_target=self._camera_target,
            render_mode=self.config.render_mode,
            warmup_frames=self.config.warmup_frames,
            samples_per_frame=self.config.samples_per_frame,
            fps=self.fps,
            script_path=self.config.script_path,
            app_id=self.config.app_id,
            render_vars=self.config.render_vars,
        )
        try:
            self._session.open()
            self._rebind_instance_transforms()
            self._camera_binding = self._session.bind_attribute(
                ["/NewtonCamera"],
                "omni:xform",
                semantic=self._session._ovstage.AttributeSemantic.MATRIX,
            )
            self._write_camera()
        except BaseException:
            self._close_runtime()
            raise

    @override
    def log_mesh(
        self,
        name: str,
        points: wp.array[wp.vec3],
        indices: wp.array[wp.int32] | wp.array[wp.uint32],
        normals: wp.array[wp.vec3] | None = None,
        uvs: wp.array[wp.vec2] | None = None,
        texture: np.ndarray | str | None = None,
        hidden: bool = False,
        backface_culling: bool = True,
        color: tuple[float, float, float] | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
        opacity: float | None = None,
        ior: float | None = None,
        material: OpenPBRMaterial | None = None,
    ):
        if material is not None:
            if not isinstance(material, OpenPBRMaterial):
                raise TypeError("ViewerOVRTX material must be an OpenPBRMaterial")
            color = material.color
            roughness = material.roughness
            metallic = material.metallic
            opacity = material.fallback_opacity
            ior = material.ior
        result = super().log_mesh(
            name,
            points,
            indices,
            normals,
            uvs,
            texture,
            hidden,
            backface_culling,
            color,
            roughness,
            metallic,
            opacity,
            ior,
        )
        mesh = self._meshes.get(name)
        has_uvs = False
        if mesh is not None and uvs is not None:
            uv_values = uvs.numpy().astype(np.float32)
            point_count = len(points)
            index_count = len(indices)
            if len(uv_values) == point_count:
                interpolation = UsdGeom.Tokens.vertex
            elif len(uv_values) == index_count:
                interpolation = UsdGeom.Tokens.faceVarying
            else:
                raise ValueError(
                    f"Mesh {name!r} has {len(uv_values)} UVs; expected {point_count} vertex "
                    f"or {index_count} face-varying values"
                )
            primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st",
                Sdf.ValueTypeNames.TexCoord2fArray,
                interpolation,
            )
            primvar.Set(uv_values)
            has_uvs = True
        self._mesh_materials[name] = _MeshMaterial(
            color=(0.8, 0.8, 0.8) if color is None else tuple(float(value) for value in color),
            has_explicit_color=color is not None,
            roughness=self.config.default_material_roughness if roughness is None else float(roughness),
            metallic=0.0 if metallic is None else float(metallic),
            opacity=1.0 if opacity is None else float(opacity),
            ior=1.5 if ior is None else float(ior),
            texture=texture,
            has_uvs=has_uvs,
            material=material,
        )
        return result

    def _stage_texture_asset(self, texture: np.ndarray | str, index: int) -> Sdf.AssetPath:
        """Copy a mesh texture beside the USD recording and return its relative asset path."""
        output = Path(self.output_path)
        texture_dir = output.parent / f"{output.stem}_textures"
        texture_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(texture, np.ndarray):
            target = texture_dir / f"mesh_{index:03d}.png"
            pixels = texture
            if np.issubdtype(pixels.dtype, np.floating):
                pixels = (np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
            _save_image(target, pixels)
        else:
            source = Path(texture).expanduser()
            if not source.is_file():
                return Sdf.AssetPath(str(texture))
            target = texture_dir / f"mesh_{index:03d}_{source.name}"
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)

        return Sdf.AssetPath(target.relative_to(output.parent).as_posix())

    def _author_render_stage(self) -> None:
        """Author camera, lights, materials, AOVs, and render product into USD."""
        camera = UsdGeom.Camera.Define(self.stage, "/NewtonCamera")
        camera.CreateFocalLengthAttr(self.config.camera_focal_length)
        camera.CreateHorizontalApertureAttr(self.config.camera_horizontal_aperture)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100000.0))
        camera_xform = UsdGeom.Xformable(camera)
        camera_xform.ClearXformOpOrder()
        matrix = self._camera_matrix()[0]
        camera_xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Matrix4d(*matrix.ravel()))

        lights = UsdGeom.Scope.Define(self.stage, "/Lighting")
        dome = UsdLux.DomeLight.Define(self.stage, lights.GetPath().AppendChild("Dome"))
        dome.CreateIntensityAttr(self.config.dome_light_intensity)
        key = UsdLux.DistantLight.Define(self.stage, lights.GetPath().AppendChild("Key"))
        key.CreateIntensityAttr(self.config.key_light_intensity)
        key.CreateAngleAttr(1.0)
        key_xform = UsdGeom.Xformable(key)
        key_xform.ClearXformOpOrder()
        key_xform.AddRotateXYZOp().Set(Gf.Vec3f(-55.0, 30.0, 0.0))

        material_scope = UsdGeom.Scope.Define(self.stage, "/Materials")
        for index, (name, values) in enumerate(self._mesh_materials.items()):
            mesh = self._meshes.get(name)
            if mesh is None:
                continue
            material_path = material_scope.GetPath().AppendChild(f"Mesh_{index}")
            texture_asset = None
            if values.texture is not None and values.has_uvs:
                texture_asset = self._stage_texture_asset(values.texture, index)
            if values.material is not None:
                material = author_material(
                    self.stage,
                    material_path,
                    values.material,
                    base_color_texture=texture_asset,
                )
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
                continue

            material = UsdShade.Material.Define(self.stage, material_path)
            shader = UsdShade.Shader.Define(self.stage, material_path.AppendChild("PreviewSurface"))
            shader.CreateIdAttr("UsdPreviewSurface")
            if texture_asset is not None:
                st_reader = UsdShade.Shader.Define(self.stage, material_path.AppendChild("TexCoordReader"))
                st_reader.CreateIdAttr("UsdPrimvarReader_float2")
                st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

                texture_shader = UsdShade.Shader.Define(self.stage, material_path.AppendChild("BaseColorTexture"))
                texture_shader.CreateIdAttr("UsdUVTexture")
                texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_asset)
                texture_shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
                texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                    st_reader.ConnectableAPI(), "result"
                )
                texture_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                    texture_shader.ConnectableAPI(), "rgb"
                )
            elif values.has_explicit_color:
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*values.color))
            else:
                reader = UsdShade.Shader.Define(self.stage, material_path.AppendChild("DisplayColor"))
                reader.CreateIdAttr("UsdPrimvarReader_float3")
                reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
                reader.CreateInput("fallback", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*values.color))
                reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                    reader.ConnectableAPI(), "result"
                )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(values.roughness)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(values.metallic)
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(values.opacity)
            shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(values.ior)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

        product_path = Sdf.Path(self.config.render_product_path)
        UsdGeom.Scope.Define(self.stage, product_path.GetParentPath())
        product = UsdRender.Product.Define(self.stage, product_path)
        product.CreateResolutionAttr(Gf.Vec2i(self.config.width, self.config.height))
        product.CreateCameraRel().SetTargets([camera.GetPath()])
        product.GetPrim().CreateAttribute("omni:rtx:rendermode", Sdf.ValueTypeNames.Token, custom=True).Set(
            self.config.render_mode
        )
        render_var_paths = []
        from ...ovrtx import RENDER_VARS  # noqa: PLC0415

        for friendly_name in self.config.render_vars:
            source_name = RENDER_VARS[friendly_name]
            render_var = UsdRender.Var.Define(self.stage, product_path.AppendChild(source_name))
            render_var.CreateSourceNameAttr(source_name)
            render_var_paths.append(render_var.GetPath())
        product.CreateOrderedVarsRel().SetTargets(render_var_paths)

    def _camera_matrix(self) -> np.ndarray:
        qw, qx, qy, qz = _camera_orientation(self._camera_position, self._camera_target)
        xform = np.asarray([[*self._camera_position, qx, qy, qz, qw]], dtype=np.float64)
        return _transforms_to_matrices(xform, np.ones((1, 3), dtype=np.float64))

    @override
    def log_instances(
        self,
        name: str,
        mesh: str,
        xforms: wp.array[wp.transform] | None,
        scales: wp.array[wp.vec3] | None,
        colors: wp.array[wp.vec3] | None,
        materials: wp.array[wp.vec4] | None,
        hidden: bool = False,
    ):
        super().log_instances(name, mesh, xforms, scales, colors, materials, hidden)
        if xforms is None:
            return
        xforms_np = xforms.numpy()
        scales_np = np.ones((len(xforms_np), 3), dtype=np.float32) if scales is None else scales.numpy()
        paths = [self._get_path(name) + f"/instance_{index}" for index in range(len(xforms_np))]
        paths_changed = self._instance_paths.get(name) != paths
        self._instance_paths[name] = paths
        self._pending_matrices[name] = _transforms_to_matrices(xforms_np, scales_np)
        if paths_changed and self._session is not None:
            self._rebind_instance_transforms()

    def _rebind_instance_transforms(self) -> None:
        if self._xform_binding is not None:
            self._xform_binding.close()
            self._xform_binding = None
        paths = [path for batch_paths in self._instance_paths.values() for path in batch_paths]
        if paths and self._session is not None:
            self._xform_binding = self._session.bind_attribute(
                paths,
                "omni:xform",
                semantic=self._session._ovstage.AttributeSemantic.MATRIX,
            )

    def _write_instance_transforms(self) -> None:
        if self._xform_binding is None:
            return
        matrices = np.concatenate(
            [self._pending_matrices[name] for name in self._instance_paths],
            axis=0,
        )
        self._xform_binding.write(matrices)

    @override
    def set_camera(self, pos: wp.vec3, pitch: float, yaw: float):
        position = np.asarray([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)
        pitch = max(min(float(pitch), 89.0), -89.0)
        yaw = (float(yaw) + 180.0) % 360.0 - 180.0
        pitch_radians = math.radians(pitch)
        yaw_radians = math.radians(yaw)
        if self.up_axis.upper() == "Z":
            front = np.asarray(
                [
                    math.cos(yaw_radians) * math.cos(pitch_radians),
                    math.sin(yaw_radians) * math.cos(pitch_radians),
                    math.sin(pitch_radians),
                ]
            )
        else:
            front = np.asarray(
                [
                    math.cos(yaw_radians) * math.cos(pitch_radians),
                    math.sin(pitch_radians),
                    math.sin(yaw_radians) * math.cos(pitch_radians),
                ]
            )
        self._camera_position = tuple(position)
        self._camera_target = tuple(position + front * 5.0)
        if self.stage is not None:
            camera = UsdGeom.Camera.Get(self.stage, "/NewtonCamera")
            if camera:
                xformable = UsdGeom.Xformable(camera)
                ops = xformable.GetOrderedXformOps()
                if ops:
                    ops[0].Set(Gf.Matrix4d(*self._camera_matrix()[0].ravel()))
        self._write_camera()

    def _write_camera(self) -> None:
        if self._camera_binding is None:
            return
        self._camera_binding.write(self._camera_matrix())

    @override
    def end_frame(self):
        super().end_frame()
        if self._session is None or self.model is None:
            return
        if self._frame_count % self.config.render_every != 0:
            return

        self._write_instance_transforms()
        steps = self.config.warmup_frames if self._render_count == 0 else self.config.samples_per_frame
        self._latest_render_vars = self._session.render_live_frame(self._frame_index, steps=steps)
        self._render_count += 1
        self._emit_frame()
        if self.config.on_frame is not None:
            self.config.on_frame(self._frame_index, self.read_render_vars())

    def read_render_vars(self) -> dict[str, np.ndarray]:
        """Return the latest copied OVRTX RGB/sensor buffers."""
        return dict(self._latest_render_vars)

    def snapshot(
        self,
        path: str | Path,
        *,
        render_mode: str = "PathTracing",
        warmup_frames: int = 64,
    ) -> Path:
        """Render a high-quality current-frame still in an isolated session."""
        if self.stage is None:
            raise RuntimeError("Cannot snapshot a closed OVRTX viewer")
        self.stage.GetRootLayer().Save()
        output = Path(path).expanduser().resolve()
        with OVRTXStage(
            self.output_path,
            width=self.config.width,
            height=self.config.height,
            camera_position=self._camera_position,
            camera_target=self._camera_target,
            render_mode=render_mode,
            warmup_frames=warmup_frames,
            samples_per_frame=self.config.samples_per_frame,
            fps=self.fps,
            render_vars=("rgb",),
        ) as session:
            frame = session.render_frame(self._frame_index, steps=warmup_frames)
        _save_image(output, frame.pixels)
        return output

    def _emit_frame(self) -> None:
        target = self.render_output_path
        if target is None:
            return
        pixels = self._latest_render_vars["rgb"]
        suffix = target.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            _save_image(target, pixels)
        elif suffix in VIDEO_SUFFIXES:
            self._write_video_frame(target, pixels)
        elif not suffix:
            target.mkdir(parents=True, exist_ok=True)
            sequence_index = self._render_count - 1
            _save_image(target / f"frame_{sequence_index:06d}.png", pixels)
            for name, values in self._latest_render_vars.items():
                if name != "rgb":
                    np.save(target / f"{name}_{sequence_index:06d}.npy", values)
        else:
            supported = sorted(IMAGE_SUFFIXES | VIDEO_SUFFIXES)
            raise ValueError(f"Unsupported OVRTX target suffix {suffix!r}; use one of {supported} or a directory")

    def _write_video_frame(self, target: Path, pixels: np.ndarray) -> None:
        if self._video_process is None:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise RuntimeError("OVRTX video delivery requires FFmpeg on PATH")
            target.parent.mkdir(parents=True, exist_ok=True)
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
                f"{self.config.width}x{self.config.height}",
                "-framerate",
                str(self.fps / self.config.render_every),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                self.config.video_codec,
                "-preset",
                self.config.video_preset,
                "-crf",
                str(self.config.video_crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(target),
            ]
            self._video_process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if self._video_process.stdin is None:
            raise RuntimeError("FFmpeg did not expose its input pipe")
        self._video_process.stdin.write(np.ascontiguousarray(pixels).tobytes())

    def _close_video(self) -> None:
        if self._video_process is None:
            return
        process, self._video_process = self._video_process, None
        if process.stdin is not None:
            process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {return_code}: {stderr}")
        if self.render_output_path is not None and (
            not self.render_output_path.is_file() or self.render_output_path.stat().st_size == 0
        ):
            raise RuntimeError(f"FFmpeg did not produce a video: {self.render_output_path}")

    def _close_runtime(self) -> None:
        if self._camera_binding is not None:
            self._camera_binding.close()
            self._camera_binding = None
        if self._xform_binding is not None:
            self._xform_binding.close()
            self._xform_binding = None
        if self._session is not None:
            self._session.close()
            self._session = None

    @override
    def close(self):
        if self._closed:
            return
        self._closed = True
        error: BaseException | None = None
        try:
            self._close_video()
            if self._session is not None:
                self._session.notify_render_complete(self.render_output_path)
        except BaseException as caught:
            error = caught
        finally:
            self._close_runtime()
            if self.stage is not None:
                super().close()
        if self._render_count:
            target = self.render_output_path if self.render_output_path is not None else "memory"
            print(f"OVRTX rendered {self._render_count} live frame(s) to: {target}")
        if error is not None:
            raise error
