# Palatial Package Guidelines

Internal package: `newton/_src/palatial/`. Reads `*.newton.usda` files produced
by the Palatial converter and returns a ready-to-step Newton model + solver.
The public surface is re-exported from `newton.palatial` — examples and docs
must import from there, never from `newton._src.palatial`.

## Module layout

| File | Role |
| --- | --- |
| `__init__.py` | Re-exports the public API (`NewtonBundle`, `load`, `read_shell_params`, `find_shell_prim_path`, `find_cloth_prim_path`). Keep `__all__` in sync. |
| `load.py` | Single entry point. Detects body type (`rigid` / `cloth`), picks a solver, builds the model, returns `NewtonBundle`. |
| `shell.py` | Read-side helper for `NewtonShellAPI` / `NewtonShellMaterialAPI`. Owns `DEFAULTS` (mirror of `generatedSchema.usda`). |
| `cloth.py` | Legacy `newton:bodyType="cloth"` reader + mesh extraction fallback. |
| `_resolvers.py` | Internal attribute/binding resolution helpers. Underscore-prefixed — do not export. |

## Conventions

- `import newton` **before** any `pxr.Usd` use — that registers the
  `newton_usd_schemas` and `newton_shell` plugins.
- New body types follow the same shape as cloth: a `find_<body>_prim_path()`
  locator + a `read_<body>_params()` resolver in a dedicated module
  (`shell.py` / `cloth.py` / future `cable.py`), then a `_build_<body>()`
  branch in `load.py` and a new arm in `_detect_body_type()`.
- All `read_*_params()` resolvers must return every key from their `DEFAULTS`
  dict so callers can treat the result as total. Optional solver-specific
  extras (Style3D, VBD) come back as `None` when unauthored.
- Solver kwargs from `physicsScene.newton:solver:<key>` are forwarded through
  `_build_solver()`, which filters them against the solver's `__init__`
  signature — never hard-code per-solver kwarg lists elsewhere.
- Do not pick solver / fps / stiffness defaults in `load.py` when the USDA
  authors them. Auto-pick is allowed only when the scene does not pin a
  solver (cloth → `style3d` if any `newton:shell:style3d:*` attr is present,
  else `vbd` if available, else `xpbd`).
- `fix_base=True` patches `ModelBuilder.add_joint_free` during `add_usd(...)`
  to emit a fixed root joint. Keep that patch scoped to the `add_usd` call
  and restored in a `finally`.
- Follow project rules in the root `AGENTS.md`: PEP 604 unions, prefix-first
  naming, Google-style docstrings, SI units in public docstrings,
  `wp.array[T]` annotation form, no new required dependencies.

## Adding a new schema-backed body type

1. Author the schema + material classes in
   `newton/_src/usd/schemas_ext/generatedSchema.usda` and register them in
   `plugInfo.json` (no change needed in `newton/_src/usd/__init__.py` — the
   `newton_shell` plugin is already loaded).
2. Add `newton/_src/palatial/<body>.py` with:
   - `DEFAULTS` dict mirroring the schema defaults,
   - `find_<body>_prim_path(usd_path) -> str | None`,
   - `read_<body>_params(usd_path) -> dict`,
   - geometry extractor when applicable (e.g. `extract_<body>_points`).
3. In `load.py`: extend `_detect_body_type()`, add `_build_<body>()`,
   dispatch it in `load()`, and extend the default-solver block.
4. Re-export the new `find_*` / `read_*` symbols from `__init__.py` and
   update `__all__`.
5. Update the user-facing docs under `docs/palatial_*.md` and register any
   new example under `newton/examples/palatial/` per the example rules in
   the root `AGENTS.md`.

## Tests

Add a regression that runs `load()` end-to-end on a small `*.newton.usda`
fixture and asserts the returned `NewtonBundle` fields (body type, solver
class, fps, model counts). Use `unittest`, run via
`uv run --extra dev -m newton.tests -k palatial`.
