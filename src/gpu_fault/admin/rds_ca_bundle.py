"""Ship the pinned AWS RDS CA bundle ConfigMap ahead of its consumers."""

from __future__ import annotations

import os
from pathlib import Path

from gpu_fault.admin.bootstrap_common import CommandRunner


def ensure_rds_ca_bundle(
    runner: CommandRunner,
    *,
    repository_root: Path,
    cpu_kubeconfig: Path,
    namespace: str,
) -> None:
    """Ship ``gpu-fault-rds-ca-bundle`` before anything that mounts it.

    The credential-refresh CronJob mounts the RDS CA bundle ConfigMap
    non-optionally. This bootstrap task runs before the release engine's own
    ``apply_rds_ca_bundle`` step, so on a cluster that never had the bundle the
    verify Job sat in ``ContainerCreating`` on a missing volume until the deploy
    timed out (live 2026-09-08). The script is idempotent and digest-pinned.
    """

    runner.run(
        [
            "bash",
            str(repository_root / "deploy/control-plane/tools/apply-rds-ca-bundle.sh"),
        ],
        env={
            **os.environ,
            "KUBECONFIG": str(cpu_kubeconfig),
            "GPU_FAULT_NAMESPACE": namespace,
        },
        mutate=True,
        capture=False,
    )
