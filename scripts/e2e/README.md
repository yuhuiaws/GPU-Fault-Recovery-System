# E2E drivers and probes

This directory contains live-cluster drivers and focused audit probes.
Kubernetes inputs are isolated under `manifests/`; follow that directory's
rendering and target-node instructions before applying anything.

## Layout

- `regional/`: regional acceptance drivers, audit probes, helpers and
  regional-only manifests.
- `hyperpod/`: focused HyperPod data-plane scenario runners.
- `isolated_api.py` and `render_manifest.py`: shared utilities.

## Audit probes

Files prefixed `audit_` verify one production contract, usually PostgreSQL,
collector, processor, evidence or policy behavior. They are not generic test
runners and must be invoked only by the procedure that names them.

Regional procedures and safety boundaries are documented in
`regional/README.md`.

## Q118 helpers

- `q118_fallback_marker.py`
- `q118_requeue_containment.py`
- `audit_q118_gpu_result.py`

Their Kubernetes fixtures live in `regional/manifests/q118-*.yaml`.

## Recovery helper

The regional recovery helper lives under `regional/`.
