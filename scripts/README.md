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

- `build-deploy-host-bundle.py`, `deploy_host_bundle.py`,
  `setup_deploy_host.py` and `setup-deploy-host.sh`: build, verify and install
  the signed offline deployment-host Python environment.
- `release_deploy.py`: developer-facing build, site preparation, deploy,
  single-pass verify and lightweight release-summary pipeline used by
  `make release-deploy`. It persists both reports under the release state
  directory without repeating the live health checks. A proven same-release
  NOOP skips the mutating deploy path and runs the independent verify checks
  with bounded parallelism; uncertain classifications fall back to deploy.
  `ADMIN_EMAIL` optionally overrides the site administrator address; otherwise
  the existing site value or AWS account email discovery is used.
- `scripts/e2e/regional/`: regional acceptance drivers, probes, boot guards,
  recovery helpers and test-only Kubernetes inputs.
- `scripts/e2e/hyperpod/`: focused HyperPod scenario runners.
- `scripts/e2e/`: shared rendering and isolated-API utilities only.
- `scripts/perf/`: current capacity suites and reusable performance helpers.
- `scripts/perf/legacy/`: superseded one-off Job/benchmark pairs, retained only
  for historical comparison.
- `tools/`: NVIDIA catalog generation and the generic test-case runner.

All Python under `scripts/` and `tools/` is covered by Ruff, compileall and the
architecture ratchet. Shell under `deploy/`, `scripts/` and `tools/` is covered
by `bash -n` and shellcheck.
