.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.palatial
===============

Palatial USD-to-Newton bundle loader and helpers.

Example
-------
    import newton
    from newton.palatial import load

    bundle = load("/path/to/asset.newton.usda")
    model, solver = bundle.model, bundle.solver
    dt = bundle.dt
    while running:
        contacts = model.collide(bundle.state_in)
        solver.step(bundle.state_in, bundle.state_out,
                    bundle.control, contacts, dt)
        bundle.state_in, bundle.state_out = bundle.state_out, bundle.state_in

.. py:module:: newton.palatial
.. currentmodule:: newton.palatial

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   NewtonBundle

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   find_cloth_prim_path
   find_rod_prim_path
   find_shell_prim_path
   load
   read_rod_params
   read_shell_params
