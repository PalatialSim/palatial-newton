# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared USD appearance extraction, independent of physics graph parsing."""

from __future__ import annotations

import copy
import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from pxr import Usd


from ..core import quat_between_axes
from ..core.types import Axis
from ..geometry import Mesh
from ..sim.builder import ModelBuilder
from ..usd import utils as usd
from .color import color_linear_to_srgb
from .import_usd_deformable_utils import _LOADABLE_VISUAL_TYPE_NAMES_LOWER

logger = logging.getLogger("newton")
_UNMATERIALED_VISUAL_COLOR = color_linear_to_srgb((0.18, 0.18, 0.18))


def _is_uniform_scale(scale, rel_tol: float = 1.0e-6) -> bool:
    """Whether the three components of a scale vector agree to within ``rel_tol``.

    Scales reach the importer through single-precision transform decomposition, so an
    exactly uniform scale routinely comes back with components a few ULP apart. An exact
    ``==`` comparison reports those as non-uniform.
    """
    lo, hi = min(scale), max(scale)
    return hi - lo <= rel_tol * max(abs(lo), abs(hi))


class _UsdVisualImporter:
    """Own visual extraction and its caches for both simulation and replay imports."""

    def __init__(self, builder, stage, *, ignore_paths=(), load_sites=True, verbose=False, visuals_only=False):
        from pxr import Usd, UsdGeom

        self.builder = builder
        self.stage = stage
        self.ignore_paths = ignore_paths
        self.load_sites = load_sites
        self.verbose = verbose
        self.visuals_only = visuals_only
        self.xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        self.traverse_instance_proxies = Usd.TraverseInstanceProxies()
        self.path_shape_map = {}
        self.path_shape_scale = {}
        self.bodies_with_visual_shapes = set()
        self.material_props_cache = {}
        self.mesh_cache = {}
        self.incoming_world_xform = wp.transform_identity()
        self.visual_shape_cfg = ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
        )

    def _is_enabled_collider(self, prim: Usd.Prim) -> bool:
        from pxr import UsdPhysics

        if collider := UsdPhysics.CollisionAPI(prim):
            return collider.GetCollisionEnabledAttr().Get()
        return False

    def _xform_to_mat44(self, xform: wp.transform) -> wp.mat44:
        return wp.transform_compose(xform.p, xform.q, wp.vec3(1.0))

    def _get_material_props_cached(self, prim: Usd.Prim) -> dict[str, Any]:
        """Get material properties with caching to avoid repeated traversal."""
        prim_path = str(prim.GetPath())
        if prim_path not in self.material_props_cache:
            self.material_props_cache[prim_path] = usd.resolve_material_properties_for_prim(prim)
        return self.material_props_cache[prim_path]

    def _get_mesh_cached(self, prim: Usd.Prim, *, load_uvs: bool = False, load_normals: bool = False) -> Mesh:
        """Load and cache mesh data to avoid repeated expensive USD mesh extraction."""
        prim_path = str(prim.GetPath())
        key = (prim_path, load_uvs, load_normals)
        if key in self.mesh_cache:
            return self.mesh_cache[key]

        # A mesh loaded with more data is a superset of simpler representations.
        for cached_key in [
            (prim_path, True, True),
            (prim_path, load_uvs, True),
            (prim_path, True, load_normals),
        ]:
            if cached_key != key and cached_key in self.mesh_cache:
                return self.mesh_cache[cached_key]

        mesh = usd.get_mesh(
            prim,
            load_uvs=load_uvs,
            load_normals=load_normals,
            load_visual_materials=False,
            compute_inertia=not self.visuals_only,
        )
        self.mesh_cache[key] = mesh
        return mesh

    def _apply_visual_material(self, mesh: Mesh, material_props: dict[str, Any]) -> None:
        """Apply one resolved USD visual material to its owning mesh."""
        texture = material_props.get("texture")
        if texture is not None:
            mesh.texture = texture
        if mesh.texture is not None:
            # Textures provide albedo; do not tint them with the shape palette.
            mesh.color = (1.0, 1.0, 1.0)
        elif material_props.get("color") is not None:
            mesh.color = material_props["color"]

        for key in ("opacity", "roughness", "metallic", "texture_transform"):
            value = material_props.get(key)
            if value is not None:
                setattr(mesh, key, value)

    def _get_mesh_with_visual_material(self, prim: Usd.Prim, *, path_name: str) -> Mesh:
        """Load a renderable mesh without changing physics mass properties."""
        material_props = self._get_material_props_cached(prim)
        texture = material_props.get("texture")
        physics_mesh = self._get_mesh_cached(prim)
        if texture is not None:
            render_mesh = self._get_mesh_cached(prim, load_uvs=True)
            # Texture UV expansion is render-only. Preserve the collision mesh's
            # mass/inertia so visibility changes do not perturb simulation.
            mesh = Mesh(
                render_mesh.vertices,
                render_mesh.indices,
                normals=render_mesh.normals,
                uvs=render_mesh.uvs,
                compute_inertia=False,
                is_solid=physics_mesh.is_solid,
                maxhullvert=physics_mesh.maxhullvert,
                sdf=physics_mesh.sdf,
            )
            mesh.mass = physics_mesh.mass
            mesh.com = physics_mesh.com
            mesh.inertia = physics_mesh.inertia
            mesh.has_inertia = physics_mesh.has_inertia
        else:
            mesh = physics_mesh.copy(recompute_inertia=False)
        self._apply_visual_material(mesh, material_props)
        if mesh.texture is not None and mesh.uvs is None:
            logger.info("Mesh %s has a texture but no UV coordinates; texture sampling is disabled.", path_name)
        return mesh

    def _get_face_material_subsets(self, prim: Usd.Prim) -> list[Usd.Prim]:
        """Return face-based material subsets authored directly under a mesh prim."""
        from pxr import UsdGeom

        subsets = []
        for child in prim.GetChildren():
            try:
                is_subset = child.IsA(UsdGeom.Subset)
            except Exception:
                is_subset = False
            if not is_subset:
                continue

            subset = UsdGeom.Subset(child)
            element_type = subset.GetElementTypeAttr().Get()
            if element_type != UsdGeom.Tokens.face:
                continue
            family_name = subset.GetFamilyNameAttr().Get()
            if family_name and family_name != "materialBind":
                continue
            indices = subset.GetIndicesAttr().Get()
            if not indices:
                continue
            subsets.append(child)
        return subsets

    def _get_subset_uvs(self, prim: Usd.Prim, used_vertices: np.ndarray, expected_count: int) -> np.ndarray | None:
        """Return UVs for a material subset when a matching primvar is authored."""
        from pxr import UsdGeom

        max_used_vertex = int(np.max(used_vertices, initial=-1))
        full_mesh_uvs = None
        for primvar in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            name = primvar.GetBaseName()
            if not name.startswith("st"):
                continue
            values = primvar.Get()
            if values is None:
                continue
            uvs = np.asarray(values, dtype=np.float32)
            if primvar.IsIndexed():
                indices = primvar.GetIndices()
                if indices is None:
                    continue
                indices = np.asarray(indices, dtype=np.int32)
                if len(indices) == expected_count:
                    uvs = uvs[indices]
                    if len(uvs) == expected_count:
                        return uvs
                    continue
                if len(indices) > max_used_vertex:
                    uvs = uvs[indices]
                else:
                    continue
            if len(uvs) == expected_count:
                return uvs
            if full_mesh_uvs is None and len(uvs) > max_used_vertex:
                full_mesh_uvs = uvs[used_vertices]
        return full_mesh_uvs

    def _make_visual_submesh(
        self,
        mesh: Mesh,
        triangle_indices: np.ndarray,
        material_props: dict[str, Any],
        *,
        prim: Usd.Prim,
        path_name: str,
    ) -> Mesh | None:
        """Create a render-only mesh slice for the selected triangle rows."""
        if len(triangle_indices) == 0:
            return None

        triangles = mesh.indices.reshape(-1, 3)[triangle_indices]
        used_vertices = np.unique(triangles)
        vertex_remap = np.full(len(mesh.vertices), -1, dtype=np.int32)
        vertex_remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int32)

        normals = None
        if mesh.normals is not None and len(mesh.normals) == len(mesh.vertices):
            normals = mesh.normals[used_vertices]

        uvs = None
        if mesh.uvs is not None and len(mesh.uvs) == len(mesh.vertices):
            uvs = mesh.uvs[used_vertices]
        elif material_props.get("texture") is not None:
            uvs = self._get_subset_uvs(prim, used_vertices, len(used_vertices))

        submesh = Mesh(
            mesh.vertices[used_vertices],
            vertex_remap[triangles].reshape(-1),
            normals=normals,
            uvs=uvs,
            compute_inertia=False,
            is_solid=mesh.is_solid,
            maxhullvert=mesh.maxhullvert,
        )

        self._apply_visual_material(submesh, material_props)
        if submesh.texture is not None and submesh.uvs is None:
            logger.info(
                "Mesh material subset %s has a texture but no UV coordinates; texture sampling is disabled.",
                path_name,
            )
        return submesh

    def _get_visual_material_subset_meshes(self, prim: Usd.Prim) -> list[tuple[str, Mesh]]:
        """Load one render mesh per USD material subset when subsets are authored."""
        from pxr import UsdGeom

        subsets = self._get_face_material_subsets(prim)
        if not subsets:
            return []

        mesh_schema = UsdGeom.Mesh(prim)
        face_counts = mesh_schema.GetFaceVertexCountsAttr().Get()
        if face_counts is None:
            return []
        face_counts = np.asarray(face_counts, dtype=np.int32)
        if len(face_counts) == 0 or np.any(face_counts < 3):
            return []

        subset_props = [(str(subset.GetPath()), usd.resolve_material_properties_for_prim(subset)) for subset in subsets]
        # Load UVs (and matching authored normals) so each submesh slices real
        # per-corner texture coordinates instead of recovering per-vertex UVs,
        # which scrambles faceVarying UV sets. UV loading unwelds vertices while
        # preserving triangle order, so the per-face subset selection still aligns.
        mesh = self._get_mesh_cached(prim, load_uvs=True, load_normals=True)
        triangle_face_indices = np.repeat(np.arange(len(face_counts), dtype=np.int32), face_counts - 2)
        covered_faces = np.zeros(len(face_counts), dtype=bool)

        submeshes = []
        for subset_path, material_props in subset_props:
            # Split on authored binding structure, not on whether the bound material's properties
            # resolve: a subset that binds a material Newton does not recognize still becomes its
            # own (unshaded) submesh, so import topology never depends on material vocabulary.
            # The gate is "a binding authored on the subset itself" — direct or collection-based,
            # with or without MaterialBindingAPI applied. ComputeBoundMaterial is deliberately not
            # used here: every subset inherits the parent mesh's binding through it, so full
            # resolution would split unbound subsets, and an ancestor rebind with
            # strongerThanDescendants would make topology depend on rebinding again. Subsets with
            # no authored binding fall through to the uncovered-faces fallback below, which
            # applies the parent mesh material.
            subset = UsdGeom.Subset(self.stage.GetPrimAtPath(subset_path))
            has_authored_binding = any(
                rel.GetName().startswith("material:binding") and rel.GetTargets()
                for rel in subset.GetPrim().GetRelationships()
            )
            if not has_authored_binding:
                continue
            subset_indices = np.asarray(subset.GetIndicesAttr().Get(), dtype=np.int32)
            valid = (subset_indices >= 0) & (subset_indices < len(face_counts))
            if not np.all(valid):
                logger.info(
                    "Mesh material subset %s: face indices outside the mesh face range; "
                    "out-of-range indices will be ignored.",
                    subset_path,
                )
                subset_indices = subset_indices[valid]
            if len(subset_indices) == 0:
                continue

            face_mask = np.zeros(len(face_counts), dtype=bool)
            face_mask[subset_indices] = True
            triangle_indices = np.nonzero(face_mask[triangle_face_indices])[0]
            submesh = self._make_visual_submesh(
                mesh, triangle_indices, material_props, prim=prim, path_name=subset_path
            )
            if submesh is None:
                continue
            covered_faces[subset_indices] = True
            submeshes.append((subset_path, submesh))

        if not submeshes:
            return []

        uncovered_faces = np.nonzero(~covered_faces)[0]
        if len(uncovered_faces) > 0:
            face_mask = np.zeros(len(face_counts), dtype=bool)
            face_mask[uncovered_faces] = True
            triangle_indices = np.nonzero(face_mask[triangle_face_indices])[0]
            fallback_mesh = self._make_visual_submesh(
                mesh,
                triangle_indices,
                self._get_material_props_cached(prim),
                prim=prim,
                path_name=str(prim.GetPath()),
            )
            if fallback_mesh is not None:
                submeshes.insert(0, (str(prim.GetPath()), fallback_mesh))

        return submeshes

    def _get_axial_visual_dimensions(
        self, prim: Usd.Prim, scale: wp.vec3, axis: Axis, default_radius: float, default_height: float
    ) -> tuple[float, float]:
        """Return scaled (radius, half_height); radius uses the largest perpendicular scale to match UsdPhysics."""
        radius = usd.get_float(prim, "radius", default_radius)
        half_height = usd.get_float(prim, "height", default_height) / 2
        axis_index = int(axis)
        radius_scale = max(scale[index] for index in range(3) if index != axis_index)
        return radius * radius_scale, half_height * scale[axis_index]

    def _get_planar_visual_dimensions(self, prim: Usd.Prim, scale: wp.vec3, axis: Axis) -> tuple[float, float]:
        """Return scaled (width, length); UsdGeomPlane aligns width to Z for X-axis planes and length to Z for Y-axis planes."""
        width_scale = scale[2] if axis == Axis.X else scale[0]
        length_scale = scale[2] if axis == Axis.Y else scale[1]
        width = usd.get_float(prim, "width", 0.0) * width_scale
        length = usd.get_float(prim, "length", 0.0) * length_scale
        return width, length

    def _is_effectively_visible(self, prim: Usd.Prim) -> bool:
        """Return whether ``prim`` is effectively visible in USD.

        A prim is effectively visible only when it is a :class:`UsdGeom.Imageable`
        whose inherited visibility is not ``invisible``. Non-imageable prims are
        not renderable in USD, so they are treated as not effectively visible.
        """
        from pxr import UsdGeom

        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            return False
        return imageable.ComputeVisibility() != UsdGeom.Tokens.invisible

    def _is_viewport_drawn(self, prim: Usd.Prim) -> bool:
        """Return whether a prim is drawn under viewport semantics.

        USD viewports draw the ``default`` and ``proxy`` purposes and hide ``guide`` and
        ``render``; the allowlist also keeps any future purpose hidden until explicitly
        handled. This is what decides whether a collider is drawn: ``guide`` is the
        conventional purpose for authored collision geometry (e.g. the MuJoCo USD
        exporter), and such a prim is not viewport geometry. ``force_show_colliders``
        is the explicit override for inspecting it anyway.
        """
        from pxr import UsdGeom

        if not self._is_effectively_visible(prim):
            return False
        return UsdGeom.Imageable(prim).ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy)

    def _get_prim_world_mat(self, prim, articulation_root_xform, incoming_world_xform):
        prim_world_mat = usd.get_transform_matrix(prim, local=False, xform_cache=self.xform_cache)
        if articulation_root_xform is not None:
            rebase_mat = self._xform_to_mat44(wp.transform_inverse(articulation_root_xform))
            prim_world_mat = rebase_mat @ prim_world_mat
        if incoming_world_xform is not None:
            # Apply the incoming world transform in model space (static shapes or when using body_xform).
            incoming_mat = self._xform_to_mat44(incoming_world_xform)
            prim_world_mat = incoming_mat @ prim_world_mat
        return prim_world_mat

    def _load_visual_shape_children(
        self,
        parent_body_id: int,
        prim: Usd.Prim,
        body_xform: wp.transform | None,
        articulation_root_xform: wp.transform | None,
        allow_visual_shapes: bool,
    ):
        for child in prim.GetFilteredChildren(self.traverse_instance_proxies):
            self._load_visual_shapes_impl(
                parent_body_id, child, body_xform, articulation_root_xform, allow_visual_shapes
            )

    def _load_visual_shapes_impl(
        self,
        parent_body_id: int,
        prim: Usd.Prim,
        body_xform: wp.transform | None = None,
        articulation_root_xform: wp.transform | None = None,
        allow_visual_shapes: bool = True,
        recurse: bool = True,
    ):
        """Load visual shapes and sites for a prim subtree.

        Args:
            parent_body_id: ModelBuilder body id to attach shapes to. Use -1 for
                static shapes that are not bound to any rigid body.
            prim: USD prim to inspect for visual geometry and recurse into.
            body_xform: Rigid body transform actually used by the self.builder.
                This matches any physics-authored pose, scene-level transforms,
                and incoming transforms that were applied when the body was created.
            articulation_root_xform: The articulation root's world-space transform,
                passed when override_root_xform=True. Strips the root's original
                pose from visual prim transforms to match the rebased body transforms.
            allow_visual_shapes: Whether non-site geometry may be loaded from this subtree.
            recurse: Whether to inspect child prims after processing ``prim``.
        """
        from pxr import UsdPhysics

        if not self.visuals_only and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return
        path_name = str(prim.GetPath())
        if any(re.match(path, path_name) for path in self.ignore_paths):
            return
        if not self.visuals_only and self._is_enabled_collider(prim):
            if recurse:
                self._load_visual_shape_children(parent_body_id, prim, body_xform, articulation_root_xform, False)
            return

        type_name = str(prim.GetTypeName()).lower()
        if type_name.endswith("joint"):
            return

        is_site = usd.has_applied_api_schema(prim, "NewtonSiteAPI") or usd.has_applied_api_schema(prim, "MjcSiteAPI")
        if is_site and not self.load_sites:
            return
        if not is_site and not allow_visual_shapes:
            if recurse:
                self._load_visual_shape_children(
                    parent_body_id, prim, body_xform, articulation_root_xform, allow_visual_shapes
                )
            return
        if type_name not in _LOADABLE_VISUAL_TYPE_NAMES_LOWER:
            # Skip the transform/material work below for prims that cannot produce a shape.
            if (
                len(type_name) > 0
                and type_name not in {"geomsubset", "material", "scope", "shader", "xform", "tetmesh"}
                and path_name not in self.path_shape_map
                and self.verbose
            ):
                print(f"Warning: Unsupported geometry type {type_name} at {path_name} while loading visual shapes.")
            if recurse:
                self._load_visual_shape_children(
                    parent_body_id, prim, body_xform, articulation_root_xform, allow_visual_shapes
                )
            return

        prim_world_mat = self._get_prim_world_mat(
            prim,
            articulation_root_xform,
            self.incoming_world_xform if (parent_body_id == -1 or body_xform is not None) else None,
        )
        if body_xform is not None:
            # Use the body transform used by the self.builder to avoid USD/physics pose mismatches.
            body_world_mat = self._xform_to_mat44(body_xform)
            rel_mat = wp.inverse(body_world_mat) @ prim_world_mat
        else:
            rel_mat = prim_world_mat

        xform_pos, xform_rot, scale = wp.transform_decompose(rel_mat)
        xform = wp.transform(xform_pos, xform_rot)
        if self.visuals_only:
            reconstructed = wp.transform_compose(xform_pos, xform_rot, scale)
            if not np.allclose(np.asarray(reconstructed), np.asarray(rel_mat), rtol=1e-5, atol=1e-6):
                raise ValueError(f"Visual transform contains unsupported shear or reflection: {path_name}")
            if type_name in {"sphere", "capsule"} and not _is_uniform_scale(scale):
                raise ValueError(f"Nonuniform curved primitive scale cannot be replayed faithfully: {path_name}")
            if type_name in {"cylinder", "cone"}:
                axis_index = int(usd.get_gprim_axis(prim))
                radial_scale = [scale[index] for index in range(3) if index != axis_index]
                if not np.isclose(*radial_scale, rtol=1e-5, atol=1e-6):
                    raise ValueError(f"Elliptical primitive cross-section cannot be replayed faithfully: {path_name}")

        shape_id = -1

        visual_shape_cfg_for_prim = copy.copy(self.visual_shape_cfg)
        visual_shape_cfg_for_prim.is_visible = is_site or self._is_viewport_drawn(prim)
        material_props = self._get_material_props_cached(prim)
        shape_color = material_props.get("color")
        shape_visual_kwargs = {}
        if material_props.get("opacity") is not None:
            shape_visual_kwargs["opacity"] = material_props["opacity"]
        # A textured mesh resolves no scalar color on purpose, so the texture is not tinted;
        # the mesh path gives it white. Geometry that never receives the texture still wants
        # the neutral, otherwise it falls through to a palette color.
        carries_texture = material_props.get("texture") is not None and type_name == "mesh"
        if shape_color is None and not carries_texture and visual_shape_cfg_for_prim.is_visible:
            shape_color = _UNMATERIALED_VISUAL_COLOR

        if path_name not in self.path_shape_map:
            if type_name == "cube":
                size = usd.get_float(prim, "size", 2.0)
                side_lengths = scale * size
                shape_id = self.builder.add_shape_box(
                    parent_body_id,
                    xform=xform,
                    hx=side_lengths[0] / 2,
                    hy=side_lengths[1] / 2,
                    hz=side_lengths[2] / 2,
                    cfg=visual_shape_cfg_for_prim,
                    color=shape_color,
                    as_site=is_site,
                    label=path_name,
                    **shape_visual_kwargs,
                )
            elif type_name == "sphere":
                if not _is_uniform_scale(scale):
                    print(f"Warning: Non-uniform scaling of spheres is not supported, at {path_name}.")
                radius = usd.get_float(prim, "radius", 1.0) * max(scale)
                shape_id = self.builder.add_shape_sphere(
                    parent_body_id,
                    xform=xform,
                    radius=radius,
                    cfg=visual_shape_cfg_for_prim,
                    color=shape_color,
                    as_site=is_site,
                    label=path_name,
                    **shape_visual_kwargs,
                )
            elif type_name == "plane":
                axis = usd.get_gprim_axis(prim)
                width, length = self._get_planar_visual_dimensions(prim, scale, axis)
                # Apply axis rotation to transform
                xform = wp.transform(xform.p, xform.q * quat_between_axes(Axis.Z, axis))
                shape_id = self.builder.add_shape_plane(
                    body=parent_body_id,
                    xform=xform,
                    width=width,
                    length=length,
                    cfg=visual_shape_cfg_for_prim,
                    color=shape_color,
                    label=path_name,
                    **shape_visual_kwargs,
                )
            elif type_name == "capsule":
                axis = usd.get_gprim_axis(prim)
                radius, half_height = self._get_axial_visual_dimensions(
                    prim, scale, axis, default_radius=0.5, default_height=1.0
                )
                # Apply axis rotation to transform
                xform = wp.transform(xform.p, xform.q * quat_between_axes(Axis.Z, axis))
                shape_id = self.builder.add_shape_capsule(
                    parent_body_id,
                    xform=xform,
                    radius=radius,
                    half_height=half_height,
                    cfg=visual_shape_cfg_for_prim,
                    color=shape_color,
                    as_site=is_site,
                    label=path_name,
                    **shape_visual_kwargs,
                )
            elif type_name == "cylinder":
                axis = usd.get_gprim_axis(prim)
                radius, half_height = self._get_axial_visual_dimensions(
                    prim, scale, axis, default_radius=1.0, default_height=2.0
                )
                # Apply axis rotation to transform
                xform = wp.transform(xform.p, xform.q * quat_between_axes(Axis.Z, axis))
                shape_id = self.builder.add_shape_cylinder(
                    parent_body_id,
                    xform=xform,
                    radius=radius,
                    half_height=half_height,
                    cfg=visual_shape_cfg_for_prim,
                    color=shape_color,
                    as_site=is_site,
                    label=path_name,
                    **shape_visual_kwargs,
                )
            elif type_name == "cone":
                axis = usd.get_gprim_axis(prim)
                radius, half_height = self._get_axial_visual_dimensions(
                    prim, scale, axis, default_radius=1.0, default_height=2.0
                )
                # Apply axis rotation to transform
                xform = wp.transform(xform.p, xform.q * quat_between_axes(Axis.Z, axis))
                shape_id = self.builder.add_shape_cone(
                    parent_body_id,
                    xform=xform,
                    radius=radius,
                    half_height=half_height,
                    cfg=visual_shape_cfg_for_prim,
                    color=shape_color,
                    as_site=is_site,
                    label=path_name,
                    **shape_visual_kwargs,
                )
            elif type_name == "mesh":
                subset_meshes = self._get_visual_material_subset_meshes(prim)
                if subset_meshes:
                    for subset_path, subset_mesh in subset_meshes:
                        subset_shape_id = self.builder.add_shape_mesh(
                            parent_body_id,
                            xform=xform,
                            scale=scale,
                            mesh=subset_mesh,
                            cfg=visual_shape_cfg_for_prim,
                            color=None,
                            label=subset_path,
                        )
                        self.path_shape_map[subset_path] = subset_shape_id
                        self.path_shape_scale[subset_path] = scale
                        if shape_id < 0:
                            shape_id = subset_shape_id
                        if self.verbose:
                            print(
                                f"Added visual shape {subset_path} ({type_name} material subset) "
                                f"with id {subset_shape_id}."
                            )
                else:
                    mesh = self._get_mesh_with_visual_material(prim, path_name=path_name)
                    shape_id = self.builder.add_shape_mesh(
                        parent_body_id,
                        xform=xform,
                        scale=scale,
                        mesh=mesh,
                        cfg=visual_shape_cfg_for_prim,
                        color=shape_color,
                        label=path_name,
                        **shape_visual_kwargs,
                    )
            elif type_name == "particlefield3dgaussiansplat":
                gaussian = usd.get_gaussian(prim)
                shape_id = self.builder.add_shape_gaussian(
                    parent_body_id,
                    gaussian=gaussian,
                    xform=xform,
                    scale=scale,
                    cfg=visual_shape_cfg_for_prim,
                    color=shape_color,
                    label=path_name,
                    **shape_visual_kwargs,
                )
            if shape_id >= 0:
                self.path_shape_map[path_name] = shape_id
                self.path_shape_scale[path_name] = scale
                if not is_site and visual_shape_cfg_for_prim.is_visible:
                    self.bodies_with_visual_shapes.add(parent_body_id)
                if self.verbose:
                    print(f"Added visual shape {path_name} ({type_name}) with id {shape_id}.")

        if recurse:
            self._load_visual_shape_children(
                parent_body_id, prim, body_xform, articulation_root_xform, allow_visual_shapes
            )


def parse_usd_visuals(builder: ModelBuilder, source: str | Usd.Stage, *, body_paths: list[str]) -> dict[str, Any]:
    """Import appearance onto independent rigid carriers without parsing physics."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(source) if isinstance(source, str) else source
    if not stage or stage.GetCompositionErrors():
        raise ValueError(f"Cannot compose USD visual source: {source}")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("Visual pose replay requires metersPerUnit=1")
    if Axis.from_string(UsdGeom.GetStageUpAxis(stage)) != builder.up_axis:
        raise ValueError("Visual pose replay requires matching stage and builder up axes")
    if len(set(body_paths)) != len(body_paths):
        raise ValueError("Visual body paths must be unique")
    visuals = _UsdVisualImporter(builder, stage, load_sites=False, visuals_only=True)
    path_body_map = {}
    for path in body_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Visual body is not an authored rigid body: {path}")
        if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is False:
            raise ValueError(f"Visual body is disabled: {path}")
        matrix = visuals.xform_cache.GetLocalToWorldTransform(prim).RemoveScaleShear()
        position = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        xform = wp.transform(wp.vec3(*position), wp.quat(*rotation.GetImaginary(), rotation.GetReal()))
        path_body_map[path] = builder.add_body(xform=xform, mass=0.0, is_kinematic=True, label=path)
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if any("deformable" in schema.lower() or schema == "SkelBindingAPI" for schema in prim.GetAppliedSchemas()):
            raise ValueError(f"Deformable geometry cannot use rigid pose replay: {prim.GetPath()}")
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if not visuals._is_viewport_drawn(prim):
            continue
        if prim.GetTypeName().lower() not in _LOADABLE_VISUAL_TYPE_NAMES_LOWER:
            raise ValueError(f"Unsupported visible geometry in rigid replay: {prim.GetPath()} ({prim.GetTypeName()})")
        if any(attribute.GetNumTimeSamples() > 0 for attribute in prim.GetAttributes()):
            raise ValueError(f"Time-varying visual geometry cannot use rigid pose replay: {prim.GetPath()}")
        owner = prim
        while owner and str(owner.GetPath()) not in path_body_map:
            if UsdGeom.Xformable(owner) and any(
                op.GetAttr().GetNumTimeSamples() > 0 for op in UsdGeom.Xformable(owner).GetOrderedXformOps()
            ):
                raise ValueError(f"Time-varying local visual transform cannot use rigid pose replay: {owner.GetPath()}")
            owner = owner.GetParent()
        body = path_body_map[str(owner.GetPath())] if owner else -1
        body_xform = builder.body_q[body] if body >= 0 else None
        visuals._load_visual_shapes_impl(body, prim, body_xform=body_xform, recurse=False)
    return {
        "path_body_map": path_body_map,
        "path_shape_map": visuals.path_shape_map,
        "path_shape_scale": visuals.path_shape_scale,
    }
