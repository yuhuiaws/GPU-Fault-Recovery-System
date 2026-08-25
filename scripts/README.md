# Scripts and tools

`scripts/` contains repository validation, deployment support, E2E drivers and
performance harnesses. `tools/` is limited to catalog/source generators and the
generic fault-case runner.

## Python naming

- Importable modules use `snake_case.py` and live under a package directory
  containing `__init__.py`.
- Path-only CLI checks may retain `kebab-case.py`; callers must invoke them with
  `python scripts/<name>.py`, never import them.
- A Python file has a shebang only when its executable bit is set. Non-executable
  modules are invoked through the configured Python interpreter.

## Directory roles

- `scripts/e2e/`: live-cluster drivers and probes.
- `scripts/e2e/manifests/`: test-only Kubernetes inputs named by procedures.
- `scripts/perf/`: current capacity suites and reusable performance helpers.
- `scripts/perf/legacy/`: superseded one-off Job/benchmark pairs, retained only
  for historical comparison.
- `tools/`: NVIDIA catalog generation and the generic test-case runner.

All Python under `scripts/` and `tools/` is covered by Ruff, compileall and the
architecture ratchet. Shell under `deploy/`, `scripts/` and `tools/` is covered
by `bash -n` and shellcheck.
