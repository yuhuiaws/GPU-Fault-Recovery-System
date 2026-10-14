"""Ship the pinned AWS RDS CA bundle ConfigMap ahead of its consumers."""

from __future__ import annotations

import os
from pathlib import Path

from gpu_fault.admin.bootstrap_common import CommandRunner

RDS_CA_BUNDLE_PATH = "/etc/gpu-fault/rds/ca-bundle.pem"
# The installer, the copy it installs (the release's vendored bundle, admitted
# only when it matches the script's RDS_CA_BUNDLE_SHA256 pin), the ConfigMap it
# writes and the URL AWS publishes the bundle at -- the last one read only to
# report drift, never installed (``rds_ca_bundle_drift``).
APPLY_SCRIPT = "deploy/control-plane/tools/apply-rds-ca-bundle.sh"
VENDORED_BUNDLE = "deploy/control-plane/tools/rds-global-bundle.pem"
CONFIGMAP_NAME = "gpu-fault-rds-ca-bundle"
BUNDLE_KEY = "ca-bundle.pem"
PUBLISHED_BUNDLE_URL = (
    "https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem"
)


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
    timed out (live 2026-09-08). The script is idempotent and digest-pinned,
    and installs the release's vendored copy, never a download.
    """

    runner.run(
        ["bash", str(repository_root / APPLY_SCRIPT)],
        env={
            **os.environ,
            "KUBECONFIG": str(cpu_kubeconfig),
            "GPU_FAULT_NAMESPACE": namespace,
        },
        mutate=True,
        capture=False,
    )
