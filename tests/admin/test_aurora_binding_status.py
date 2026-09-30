"""``aurora_binding`` accepts an online Aurora cluster in its backup window.

The freshly created cluster of the 2026-09-30 first deploy sat in
``backing-up`` (first automated backup) when the release preflight built its
database proof; the check demanded ``available`` and reported the failure as a
binding mismatch. Backup and maintenance windows keep the cluster online and
its identity intact; every other status still fails closed, with a message
that names the status instead of blaming the binding.
"""

from __future__ import annotations

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.installation_lifecycle import (
    AURORA_ONLINE_STATUSES,
    aurora_binding,
)

CPU_EKS_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/cpu-eks"
CLUSTER_ID = "gpu-fault-us-west-2-site-aaaaaaaa"
SECRET_ARN = "arn:aws:secretsmanager:us-west-2:123456789012:secret:rds!cluster-x"


def _cluster(status: str) -> dict[str, object]:
    return {
        "DBClusterIdentifier": CLUSTER_ID,
        "DBClusterArn": f"arn:aws:rds:us-west-2:123456789012:cluster:{CLUSTER_ID}",
        "DbClusterResourceId": "cluster-RESOURCE",
        "Engine": "aurora-postgresql",
        "Status": status,
        "Endpoint": "writer.cluster-x.us-west-2.rds.amazonaws.com",
        "Port": 5432,
        "DatabaseName": "gpu_fault",
        "MasterUsername": "administrator",
        "MasterUserSecret": {"SecretArn": SECRET_ARN},
    }


def _bind(cluster: dict[str, object]) -> dict[str, object]:
    return aurora_binding(
        cluster,
        cpu_eks_arn=CPU_EKS_ARN,
        aws_region="us-west-2",
        cluster_id=CLUSTER_ID,
        master_secret_arn=SECRET_ARN,
    )


@pytest.mark.parametrize("status", sorted(AURORA_ONLINE_STATUSES))
def test_online_statuses_bind(status: str) -> None:
    binding = _bind(_cluster(status))

    assert binding["cluster_resource_id"] == "cluster-RESOURCE"
    assert binding["master_secret_arn"] == SECRET_ARN


def test_backing_up_is_an_online_status() -> None:
    assert "backing-up" in AURORA_ONLINE_STATUSES, (
        "the first automated backup after creation must not fail the preflight"
    )


@pytest.mark.parametrize(
    "status", ["creating", "modifying", "failing-over", "deleting", "stopped", None]
)
def test_offline_statuses_fail_closed_naming_the_status(status: str | None) -> None:
    cluster = _cluster("available")
    cluster["Status"] = status

    with pytest.raises(BootstrapError, match="not online: status="):
        _bind(cluster)


def test_identity_mismatch_is_still_reported_as_a_binding_difference() -> None:
    cluster = _cluster("backing-up")
    cluster["DbClusterResourceId"] = ""

    with pytest.raises(BootstrapError, match="binding differs or is incomplete"):
        _bind(cluster)
