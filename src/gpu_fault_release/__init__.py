"""Regional release orchestrator: the deploy host's rollout, rollback and checks.

The modules here used to live under ``deploy/control-plane/regional/`` as
scripts importing each other by bare name, which made them reachable only with
that directory on ``sys.path``. They are a package now, imported as
``gpu_fault_release.<module>``; the shell launcher
``deploy/control-plane/regional/rollout-regional-release.sh`` runs
``python3 -m gpu_fault_release.rollout`` with ``src`` on ``PYTHONPATH`` so the
administrator CLI's path-based contract is unchanged.

Non-Python inputs stay under ``deploy/control-plane/regional/``: the rendered
manifests in ``generated/``, the patch and prerequisite YAML, the cleanup
inventory, and the probe programs in ``probes/`` that the engine ships to Pods
as source text. Every module here locates them from the repository root, which
is why the package runs from a checkout rather than from an installed wheel.
"""
