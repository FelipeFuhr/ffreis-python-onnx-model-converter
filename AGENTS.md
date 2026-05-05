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
