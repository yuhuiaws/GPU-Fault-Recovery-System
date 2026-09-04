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
  `deploy_host_component.py`, `deploy_host_identity.py`,
  `deploy_source_identity.py`, `setup_deploy_host.py` and
  `setup-deploy-host.sh`: build, verify and install the signed offline
  deployment-host Python environment. The dedicated `gpu-fault-deploy-host`
  wheel owns `gpu-fault-admin` and is excluded from Runtime components.
  Dependency locks are installed once per platform-bound dependency identity;
  release-specific project wheels use lightweight overlay venvs. Bundle payloads
  are content-addressed independently from Git commits.
- `component_artifacts.py` and `component_artifact_cache.py`: validate canonical
  component wheels/Node bundle and persist source-only artifact sets across
  staging snapshots without reusing delivery identity.
- `ci_coverage_gate.py`, `ci_gate_artifacts.py`, `ci_unit_gate.py`,
  `ci_gate.py`, `resolve_ci_run.py` and `restore_ci_candidate.py`: partition runtime, deployment,
  fault-runner and PostgreSQL coverage; content-address, restore and sign each
  shard; combine branch coverage under one floor; bind duration/fault evidence
  into the current unit and commit gates; then resolve and verify that candidate
  for Release or eligible deployment-host promotion.
- `staging_deploy.py`: private source preparation invoked by
  `gpu-fault-admin deploy`; it creates or reuses signing material, isolated
  dirty-worktree commits, signed successful-source authorization, content-addressed
  deploy-host bundles and the internal deployment venv. It is not a public command.
- `release_deploy.py`: developer-facing build, site preparation, deploy,
  single-pass verify and lightweight release-summary pipeline used by
  `make release-deploy`. It persists both reports under the release state
  directory without repeating the live health checks. A lightweight,
  state-bound component diff can classify NOOP even when the delivery release
  ID changed; NOOP skips the mutating deploy path and stability window, then
  runs the independent verify checks with bounded parallelism. Uncertain
  classifications fall back to deploy.
  `ADMIN_EMAIL` optionally overrides the site administrator address; otherwise
  the existing site value or AWS account email discovery is used.
- `run_release_gates.py`: local production fallback DAG. Static/contracts,
  ordinary pytest and PostgreSQL stress run with one to three CPU-bounded workers
  and isolated Python/pytest caches; artifact construction starts only after all
  three pass.
- `run_static_gates.py`: bounded static DAG for Ruff, mypy, compileall,
  architecture, contracts, safety, deployment/config, docs, YAML and Shell.
  The same runner backs local `make check` and the production fallback.
- `scripts/e2e/regional/`: regional acceptance drivers, probes, boot guards,
  recovery helpers and test-only Kubernetes inputs.
- `scripts/e2e/hyperpod/`: focused HyperPod scenario runners.
- `scripts/e2e/`: shared rendering and isolated-API utilities only.
- `scripts/perf/`: current capacity suites and reusable performance helpers.
- `scripts/perf/legacy/`: superseded one-off Job/benchmark pairs, retained only
  for historical comparison.
- `tools/`: NVIDIA catalog generation and the generic test-case runner.
  Compatible unit/component catalog pytest cases are batched into one pytest
  process while preserving per-case evidence. CI can consume the per-node
  results already emitted by the coverage run instead of executing pytest
  again; live, command and manual cases keep the existing scheduler and
  approval rules.

All Python under `scripts/` and `tools/` is covered by Ruff, compileall and the
architecture ratchet. Shell under `deploy/`, `scripts/` and `tools/` is covered
by `bash -n` and shellcheck.
