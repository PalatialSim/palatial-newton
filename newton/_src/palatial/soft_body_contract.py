# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Read Palatial soft-body ownership authored into USD."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .solver_plan import SolverPlanPartEntities

_SOFT_BODY_SPEC_ATTR = "palatial:softBodySpec"
_PART_ID_ATTR = "palatial:partId"


def read_soft_body_spec(source: str | Any) -> dict[str, Any] | None:
    """Return the canonical soft-body spec embedded on a PhysicsScene."""
    from pxr import Usd

    stage = Usd.Stage.Open(source) if isinstance(source, str) else source
    if not stage:
        raise RuntimeError(f"Cannot open USD: {source}")
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsScene":
            continue
        attr = prim.GetAttribute(_SOFT_BODY_SPEC_ATTR)
        if not (attr and attr.HasAuthoredValue()):
            continue
        raw = attr.Get()
        if not isinstance(raw, str):
            raise TypeError(f"{_SOFT_BODY_SPEC_ATTR} must be a JSON string")
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {_SOFT_BODY_SPEC_ATTR} JSON: {exc}") from exc
        if not isinstance(spec, dict):
            raise TypeError(f"{_SOFT_BODY_SPEC_ATTR} must decode to an object")
        return spec
    return None


def _part_root_paths(stage: Any, spec: Mapping[str, Any]) -> dict[str, str]:
    ordered_parts = [str(part["part_id"]) for part in spec["parts"]]
    expected = set(ordered_parts)
    tagged: dict[str, str] = {}
    for prim in stage.Traverse():
        attr = prim.GetAttribute(_PART_ID_ATTR)
        if not (attr and attr.HasAuthoredValue()):
            continue
        part_id = str(attr.Get())
        if part_id not in expected:
            raise ValueError(f"USD tags unknown soft-body part {part_id!r}")
        if part_id in tagged:
            raise ValueError(f"USD tags soft-body part {part_id!r} more than once")
        tagged[part_id] = prim.GetPath().pathString
    missing = expected.difference(tagged)
    if missing:
        raise ValueError(f"USD is missing soft-body part tags: {', '.join(sorted(missing))}")
    return {part_id: tagged[part_id] for part_id in ordered_parts}


def _owned_path(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def _extend_range(target: list[int], value: Any, key: str) -> None:
    if not isinstance(value, Mapping) or key not in value:
        return
    start, end = value[key]
    target.extend(range(int(start), int(end)))


def part_entities_from_import(
    stage: Any,
    spec: Mapping[str, Any],
    import_result: Mapping[str, Any],
    builder: Any,
) -> dict[str, SolverPlanPartEntities]:
    """Map tagged Palatial parts to realized Newton importer indices."""
    roots = _part_root_paths(stage, spec)
    mutable = {part_id: {"bodies": [], "particles": [], "joints": [], "shapes": []} for part_id in roots}

    scalar_maps = (
        ("path_body_map", "bodies"),
        ("path_joint_map", "joints"),
        ("path_shape_map", "shapes"),
    )
    for map_name, field in scalar_maps:
        for path, index in import_result.get(map_name, {}).items():
            for part_id, root in roots.items():
                if _owned_path(str(path), root):
                    mutable[part_id][field].append(int(index))

    for path, value in import_result.get("path_cable_map", {}).items():
        for part_id, root in roots.items():
            if not _owned_path(str(path), root):
                continue
            bodies, joints = value
            mutable[part_id]["bodies"].extend(int(index) for index in bodies)
            mutable[part_id]["joints"].extend(int(index) for index in joints)

    for map_name in ("path_cloth_map", "path_soft_map"):
        for path, value in import_result.get(map_name, {}).items():
            for part_id, root in roots.items():
                if _owned_path(str(path), root):
                    _extend_range(mutable[part_id]["particles"], value, "particle")

    # USD joints frequently live beside part roots rather than below them.
    # Assign any such joint to the part that owns its realized child body.
    body_owner = {body: part_id for part_id, fields in mutable.items() for body in fields["bodies"]}
    for joint, child in enumerate(builder.joint_child):
        part_id = body_owner.get(int(child))
        if part_id is not None:
            mutable[part_id]["joints"].append(joint)

    part_specs = {str(part["part_id"]): part for part in spec["parts"]}
    result = {}
    for part_id, fields in mutable.items():
        deduped = {field: tuple(dict.fromkeys(indices)) for field, indices in fields.items()}
        representation = str(part_specs[part_id].get("representation") or "")
        if representation in {"surface", "volume"} and not deduped["particles"]:
            raise ValueError(f"soft-body part {part_id!r} imported no particles")
        if representation == "rod" and not deduped["bodies"]:
            raise ValueError(f"rod part {part_id!r} imported no bodies")
        if representation == "rigid" and not (deduped["bodies"] or deduped["shapes"]):
            raise ValueError(f"rigid part {part_id!r} imported no bodies or shapes")
        result[part_id] = SolverPlanPartEntities(**deduped)
    return result
