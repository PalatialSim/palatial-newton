# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse


def add_palatial_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the one argument every palatial example needs: the USDA path."""
    parser.add_argument(
        "usd_path",
        type=str,
        help="Path to a converter-produced *.newton.usda asset.",
    )
    return parser


def build_contacts(model, state):
    """Allocate a contacts buffer and run one collision pass.

    Works for every body type. Cloth/rod use particle-vs-shape contacts;
    rigid/articulated use shape-vs-shape. `model.contacts()` picks the
    right pipeline for us, so the examples never branch on body type.
    """
    contacts = model.contacts()
    model.collide(state, contacts)
    return contacts


def pin_body_kinematic(model, body_index: int) -> None:
    """Clamp one rigid body in place by zeroing its inverse mass.

    Used by the cable example to hold one cable endpoint while gravity
    pulls the rest of the chain down. Mirrors what the production loader
    does for rod body 0.
    """
    inv_mass = model.body_inv_mass.numpy().copy()
    if not (0 <= body_index < inv_mass.shape[0]):
        return
    inv_mass[body_index] = 0.0
    model.body_inv_mass.assign(inv_mass)
