# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Gaussian appearance with independent deformation for repeated assets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .bvh import compute_gaussian_bounds
from .types import Gaussian, GeoType

if TYPE_CHECKING:
    from ..sim.model import Model
    from ..sim.state import State


@wp.kernel(enable_backward=False)
def _skin_gaussians(
    rest: wp.array[wp.transform],
    indices: wp.array2d[wp.int32],
    weights: wp.array2d[wp.float32],
    reference_nodes: wp.array[wp.int32],
    nodes: wp.array2d[wp.transform],
    poses: wp.array2d[wp.transform],
):
    instance, point = wp.tid()
    rest_pose = rest[point]
    position = wp.vec3(0.0)
    rotation = wp.quat(0.0, 0.0, 0.0, 0.0)
    reference = wp.transform_get_rotation(nodes[instance, reference_nodes[point]])
    for influence in range(indices.shape[1]):
        node = nodes[instance, indices[point, influence]]
        weight = weights[point, influence]
        position += weight * wp.transform_point(node, wp.transform_get_translation(rest_pose))
        q = wp.transform_get_rotation(node)
        if wp.dot(q, reference) < 0.0:
            q = -q
        rotation += weight * q
    if wp.length_sq(rotation) > 1.0e-12:
        rotation = wp.normalize(rotation)
    else:
        rotation = reference
    poses[instance, point] = wp.transform(position, rotation * wp.transform_get_rotation(rest_pose))


@wp.kernel(enable_backward=False)
def _update_gaussian_bounds(
    data: wp.array[Gaussian.Data],
    lowers: wp.array2d[wp.vec3],
    uppers: wp.array2d[wp.vec3],
):
    instance, point = wp.tid()
    lower, upper = compute_gaussian_bounds(data[instance], point)
    lowers[instance, point] = lower
    uppers[instance, point] = upper


@wp.kernel(enable_backward=False)
def _update_shape_bounds(
    shape_ids: wp.array[wp.int32],
    lowers: wp.array2d[wp.vec3],
    uppers: wp.array2d[wp.vec3],
    bounds: wp.array2d[wp.vec3],
):
    instance, lane = wp.tid()
    lower = wp.vec3(1.0e30)
    upper = wp.vec3(-1.0e30)
    for point in range(lane, lowers.shape[1], wp.block_dim()):
        lower = wp.min(lower, lowers[instance, point])
        upper = wp.max(upper, uppers[instance, point])
    lower = wp.tile_reduce(wp.min, wp.tile(lower, preserve_type=True))[0]
    upper = wp.tile_reduce(wp.max, wp.tile(upper, preserve_type=True))[0]
    if lane == 0:
        bounds[shape_ids[instance], 0] = lower
        bounds[shape_ids[instance], 1] = upper


class GaussianSkinning:
    """Deform repeated Gaussian shapes while sharing their appearance arrays.

    .. experimental::

    Register after model finalization and before creating a tiled camera. Each
    selected shape must reference the same finalized Gaussian asset. Binding
    indices and weights are shared across instances. Nodes supplied to
    :meth:`update` are rest-to-current transforms in each shape's local frame.

    Centers use linear blend skinning. Orientations use normalized quaternion
    blending with hemisphere alignment. Scales, opacity and SH coefficients
    remain fixed; this is a rigid skinning approximation, not affine covariance
    deformation. This forward-only adapter does not change physics or collision
    proxies. The caller computes node transforms from its chosen physical model.

    The model retains storage and BVHs for its lifetime. Construct a fresh model
    to change the binding or instance count. Mutating shared appearance arrays
    affects every instance and is unsupported while this adapter is active.
    """

    def __init__(
        self,
        model: Model,
        *,
        shape_ids: np.ndarray,
        node_indices: np.ndarray,
        node_weights: np.ndarray,
        node_count: int,
    ):
        """Register Gaussian instances and upload their shared skinning binding.

        Args:
            model: Finalized model owning the shapes and their render resources.
            shape_ids: Distinct shape indices, shape ``(B,)``, integer.
            node_indices: Node indices per Gaussian, shape ``(G, K)``, integer.
            node_weights: Finite nonnegative weights, shape ``(G, K)``; rows sum to one.
            node_count: Number of deformation nodes in each instance.
        """
        if model._gaussian_render_context_initialized:
            raise RuntimeError("Register GaussianSkinning before creating a render context")
        ids = np.asarray(shape_ids)
        indices = np.asarray(node_indices)
        weights = np.asarray(node_weights, dtype=np.float32)
        if ids.ndim != 1 or not ids.size or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("shape_ids must be a nonempty integer vector")
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= model.shape_count):
            raise ValueError("shape_ids must contain distinct valid shape indices")
        if model._gaussian_deformation_shapes.intersection(ids.tolist()):
            raise ValueError("A selected shape already has a deformation binding")
        if np.any(model.shape_type.numpy()[ids] != int(GeoType.GAUSSIAN)):
            raise ValueError("Every selected shape must be a Gaussian")
        sources = model.shape_source_ptr.numpy()[ids]
        if np.any(sources != sources[0]):
            raise ValueError("Selected shapes must share one finalized Gaussian asset")
        template, _ = model._gaussian_keep_alive[int(sources[0])]
        point_count = int(template.num_points)
        if point_count == 0:
            raise ValueError("The Gaussian asset must contain at least one point")
        if not isinstance(node_count, int) or node_count <= 0:
            raise ValueError("node_count must be a positive integer")
        if (
            indices.ndim != 2
            or indices.shape[0] != point_count
            or indices.shape[1] == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= node_count)
        ):
            raise ValueError("node_indices must have shape (G, K) and contain valid node indices")
        if (
            weights.shape != indices.shape
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0)
        ):
            raise ValueError("node_weights must match node_indices with nonnegative rows summing to one")

        self.model = model
        self.node_count = node_count
        self.instance_count = len(ids)
        self.point_count = point_count
        self._rest = template.transforms
        self._bvhs: list[wp.Bvh] = []
        self._views: list[Gaussian.Data] = []
        with wp.ScopedDevice(model.device):
            self._shape_ids = wp.array(ids, dtype=wp.int32)
            self._indices = wp.array(indices, dtype=wp.int32)
            self._weights = wp.array(weights, dtype=wp.float32)
            self._reference_nodes = wp.array(
                indices[np.arange(point_count), np.argmax(weights, axis=1)], dtype=wp.int32
            )
            self.transforms = wp.empty((self.instance_count, point_count), dtype=wp.transform)
            self._lowers = wp.empty((self.instance_count, point_count), dtype=wp.vec3)
            self._uppers = wp.empty((self.instance_count, point_count), dtype=wp.vec3)
            for instance in range(self.instance_count):
                view = Gaussian.Data()
                view.num_points = point_count
                view.transforms = self.transforms[instance]
                wp.copy(view.transforms, template.transforms)
                view.scales = template.scales
                view.opacities = template.opacities
                view.sh_coeffs = template.sh_coeffs
                view.min_response = template.min_response
                view.sorting_mode = template.sorting_mode
                self._views.append(view)
            self._data = wp.array(self._views, dtype=Gaussian.Data)
            self._refit_bounds()
            for instance, view in enumerate(self._views):
                bvh = wp.Bvh(self._lowers[instance], self._uppers[instance])
                view.bvh_id = bvh.id
                self._bvhs.append(bvh)
            self._data.assign(self._views)
            old_count = model.gaussians_count
            table = wp.empty(old_count + self.instance_count, dtype=Gaussian.Data)
            wp.copy(table, model.gaussians_data, count=old_count)
            wp.copy(table, self._data, dest_offset=old_count)
            pointers = model.shape_source_ptr.numpy()
            pointers[ids] = np.arange(old_count, old_count + self.instance_count, dtype=np.uint64)
            model.shape_source_ptr.assign(pointers)
            model.gaussians_data = table
            model.gaussians_count = table.shape[0]
            model._gaussian_keep_alive.extend(zip(self._views, self._bvhs, strict=True))
            model._gaussian_deformation_shapes.update(ids.tolist())
            model._gaussian_deformations.append(self)

    def _refit_bounds(self) -> None:
        wp.launch(
            _update_gaussian_bounds,
            dim=(self.instance_count, self.point_count),
            inputs=[self._data, self._lowers, self._uppers],
            device=self.model.device,
        )
        wp.launch(
            _update_shape_bounds,
            dim=(self.instance_count, 128),
            inputs=[self._shape_ids, self._lowers, self._uppers, self.model.bvh_shape_bounds],
            device=self.model.device,
            block_dim=128,
        )
        for bvh in self._bvhs:
            bvh.refit()

    def update(self, node_transforms: wp.array2d[wp.transform], state: State) -> None:
        """Update poses and render bounds without changing the physical state.

        Args:
            node_transforms: Unit-quaternion rest-to-current transforms in shape
                local coordinates, shape ``(B, node_count)`` on the model device.
            state: Current state used to refit rigid shape world transforms.

        This method allocates no arrays and performs no host readback. Identity
        transforms reset an instance to its rest pose. CUDA graph capture is
        supported by the underlying Warp kernels and BVH refits.
        """
        if (
            node_transforms.dtype != wp.transform
            or node_transforms.shape != (self.instance_count, self.node_count)
            or node_transforms.device != self.model.device
        ):
            raise ValueError("node_transforms must be a (B, node_count) transform array on the model device")
        wp.launch(
            _skin_gaussians,
            dim=(self.instance_count, self.point_count),
            inputs=[self._rest, self._indices, self._weights, self._reference_nodes, node_transforms, self.transforms],
            device=self.model.device,
        )
        self._refit_bounds()
        self.model.bvh_refit_shapes(state)
