# Agent Context

**This repo:** `ffreis-python-onnx-model-converter` — converts PyTorch, TensorFlow,
and scikit-learn models to ONNX. Plugin-based for custom model families.

## Non-obvious facts

- **Pickle loading is unsafe by default.** Require explicit `--allow-unsafe` flag for
  sklearn/joblib models. Never remove this gate.

- **TensorFlow conversion uses a separate `tf_legacy` path** intentionally — `tf2onnx`
  pins old protobuf that conflicts with Python ≥3.13. Keep these dependency pins
  isolated; do not merge them into the main `pyproject.toml`.

- **AutoSklearn has separate dependency pins** (ASKL1 vs ASKL2 conflict resolver
  incompatibilities). Keep in `examples/docker/` only; do not add to main dependencies.

- **HuggingFace export uses `optimum.exporters.onnx` behind the `[hf]` extra (D2).**
  `converters/hf_converter.py` (`convert_hf_to_onnx` + `verify_hf_onnx_parity`)
  lazy-imports optimum/transformers — multi-GB, kept off the core deps (same isolation
  policy as `tf_legacy`). `optimum.main_export` auto-detects the task from
  `model.config.model_type`. The module is **coverage-omitted** (extra absent in the
  baseline run) and its integration test uses `importorskip`; a typed `DependencyError`
  is raised at call time when optimum is missing.

- **Lint gate is `make lint` (ruff + flake8 + mypy), NOT `make fmt-check`.** `check`
  = `grpc-check lint test-unit` — it does **not** run `fmt-check`. `isort` (in
  `fmt-check`) is configured `force_single_line` which *conflicts* with ruff's import
  combining; ruff is authoritative. Make imports ruff-clean (combined); do not split
  them to satisfy isort, or `make lint` will fail.

- **Plugin discovery via entry points.** New model families can be added as plugins
  without modifying the core. Do not add framework-specific adapters to `core/`.

- **Architecture boundary enforcement** via `make architecture-check`. The CLI layer
  must not import from `infrastructure/`. Violations fail CI.

- **Coverage minimum is 90%** — stricter than other repos. Do not lower it.

- **Parity checks** validate that source model and ONNX Runtime outputs match within
  tolerance. These are optional locally but enforced in CI for supported frameworks.

## Structure

```
src/onnx_converter/
  application/      ← orchestration use-cases (typed options)
  adapters/         ← framework-specific loaders/converters
  plugins/          ← plugin protocol for custom models
  infrastructure/   ← ONNX post-processing, optimization
  cli/              ← Typer CLI
  converter/        ← HTTP and gRPC servers
proto/              ← gRPC contract
```

## Build/test

```bash
make env && make install-dev && make check
make test-integration       # scheduled; requires framework installs
make architecture-check
```

## Keeping this file current

- **If you discover a fact not reflected here:** add it before finishing your task.
- **If something here is wrong or outdated:** correct it in the same commit as the code change.
- **If you rename a file, command, or concept referenced here:** update the reference.
