"""Aurora bootstrap: the discovery answers it refuses and the creates it issues.

Every ``describe`` the bootstrap reads is checked for the shape it needs before
anything is created, tagged or failed over: a subnet group that is not there
is created with the site tag; CPU nodes without an InternalIP or without an
ENI security group stop the ingress rule; a cluster describe that does not
name exactly the cluster, or names no single writer, stops the writer-zone
reconcile; an instance or parameter group whose describe lacks its ARN cannot
be claimed by tag; and the managed master secret is read again while the
cluster has not published it yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from gpu_fault.admin import bootstrap_aurora as aurora
from gpu_fault.admin.bootstrap_common import BootstrapError
from tests.admin.test_admin_bootstrap_aurora import InstanceCreationRunner

REGION = "us-east-1"
SITE = "site-1"
CLUSTER = "aurora-a"


class ReadRunner:
    """``aws_json``/``aws_text`` answered per operation; every call recorded."""

    def __init__(self, answers: dict[str, list[Any]]) -> None:
        self.answers = {key: list(value) for key, value in answers.items()}
        self.calls: list[tuple[str, ...]] = []

    def _answer(self, arguments: Sequence[str]) -> Any:
        self.calls.append(("aws", *arguments))
        queue = self.answers[arguments[1]]
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def aws_json(self, _region: str, *arguments: str, **_keywords: Any) -> Any:
        return self._answer(arguments)

    def aws_text(self, _region: str, *arguments: str, **_keywords: Any) -> str:
        return str(self._answer(arguments))

    def run(self, arguments: Sequence[str], **_keywords: Any) -> str:
        self.calls.append(tuple(arguments))
        return ""


def _not_found(code: str) -> BootstrapError:
    return BootstrapError(
        f"command failed (254): aws: An error occurred ({code}) when calling the operation"
    )


def test_a_missing_subnet_group_is_created_over_the_subnets_with_the_site_tag() -> None:
    runner = ReadRunner(
        {"describe-db-subnet-groups": [_not_found("DBSubnetGroupNotFoundFault")]}
    )

    aurora.ensure_subnet_group(
        runner,
        aws_region=REGION,
        name="aurora-a-subnets",
        subnet_ids=["subnet-1", "subnet-2"],
        site_id=SITE,
    )

    creates = [call for call in runner.calls if call[2] == "create-db-subnet-group"]
    assert len(creates) == 1
    assert creates[0][creates[0].index("--subnet-ids") + 1 :][:2] == (
        "subnet-1",
        "subnet-2",
    )
    assert creates[0][-1] == f"Key=gpu-fault:site-id,Value={SITE}"


def test_cpu_nodes_without_an_internal_ip_cannot_source_the_ingress_rule() -> None:
    runner = ReadRunner({})
    document = {"items": [{"status": {"addresses": [{"type": "Hostname"}]}}]}

    with pytest.raises(BootstrapError, match="no node InternalIP"):
        aurora.cpu_node_security_groups(runner, aws_region=REGION, document=document)
    assert runner.calls == [], "no ENI read happens without an address"


def test_cpu_node_enis_without_a_security_group_are_refused() -> None:
    runner = ReadRunner(
        {"describe-network-interfaces": [{"NetworkInterfaces": [{"Groups": []}]}]}
    )
    document = {
        "items": [
            {"status": {"addresses": [{"type": "InternalIP", "address": "10.0.0.5"}]}}
        ]
    }

    with pytest.raises(
        BootstrapError, match="cannot discover CPU node Security Groups"
    ):
        aurora.cpu_node_security_groups(runner, aws_region=REGION, document=document)
    assert runner.calls[0][-1].endswith("Values=10.0.0.5"), (
        "the ENI read is filtered by the node addresses"
    )


def test_writer_zone_reconcile_reads_the_zones_from_the_cpu_nodes_when_unrecorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    nodes = {
        "items": [
            {"metadata": {"labels": {"topology.kubernetes.io/zone": "us-east-1a"}}},
            {"metadata": {"labels": {}}},
        ]
    }

    class NodeRunner(ReadRunner):
        def run(self, arguments: Sequence[str], **_keywords: Any) -> str:
            self.calls.append(tuple(arguments))
            return json.dumps(nodes)

    affinity_calls: list[dict[str, Any]] = []

    def affinity(_runner: object, **kwargs: Any) -> dict[str, Any]:
        affinity_calls.append(kwargs)
        return {"action": "none"}

    monkeypatch.setattr(aurora, "ensure_writer_zone_affinity", affinity)
    runner = NodeRunner({})

    aurora.reconcile_writer_zone(
        runner,
        aws_region=REGION,
        cluster_id=CLUSTER,
        instance_ids=["aurora-a-writer"],
        control_plane_zones=[],
        kubeconfig=tmp_path / "cpu.kubeconfig",
        hyperpod_name="cpu-hp",
    )

    assert affinity_calls[0]["control_plane_zones"] == ["us-east-1a"]
    assert runner.calls[0][:3] == (
        "kubectl",
        "--kubeconfig",
        str(tmp_path / "cpu.kubeconfig"),
    )

    runner.calls.clear()
    aurora.reconcile_writer_zone(
        runner,
        aws_region=REGION,
        cluster_id=CLUSTER,
        instance_ids=["aurora-a-writer"],
        control_plane_zones=["us-east-1b"],
        kubeconfig=tmp_path / "cpu.kubeconfig",
        hyperpod_name="cpu-hp",
    )
    assert runner.calls == [], "recorded zones are used without a node read"
    assert affinity_calls[1]["control_plane_zones"] == ["us-east-1b"]


def _placements() -> dict[str, Any]:
    return {
        "DBInstances": [
            {
                "DBInstanceIdentifier": "aurora-a-writer",
                "DBInstanceStatus": "available",
                "AvailabilityZone": "us-east-1c",
            }
        ]
    }


@pytest.mark.parametrize(
    ("clusters", "problem"),
    [
        ([], "cannot describe Aurora cluster"),
        (
            [{"DBClusterIdentifier": CLUSTER}, {"DBClusterIdentifier": CLUSTER}],
            "cannot describe Aurora cluster",
        ),
        ([{"DBClusterIdentifier": "other"}], "cannot describe Aurora cluster"),
        (
            [{"DBClusterIdentifier": CLUSTER, "DBClusterMembers": []}],
            "has no single writer",
        ),
        (
            [
                {
                    "DBClusterIdentifier": CLUSTER,
                    "DBClusterMembers": [
                        {"DBInstanceIdentifier": "a", "IsClusterWriter": True},
                        {"DBInstanceIdentifier": "b", "IsClusterWriter": True},
                    ],
                }
            ],
            "has no single writer",
        ),
    ],
    ids=["none", "two", "other-cluster", "no-writer", "two-writers"],
)
def test_writer_zone_affinity_refuses_an_unusable_cluster_describe(
    clusters: list[dict[str, Any]], problem: str
) -> None:
    runner = ReadRunner(
        {
            "describe-db-instances": [_placements()],
            "describe-db-clusters": [{"DBClusters": clusters}],
        }
    )

    with pytest.raises(BootstrapError, match=problem):
        aurora.ensure_writer_zone_affinity(
            runner,
            aws_region=REGION,
            cluster_id=CLUSTER,
            instance_ids=["aurora-a-writer"],
            control_plane_zones=["us-east-1a"],
        )
    assert not any(call[2] == "failover-db-cluster" for call in runner.calls), (
        "no failover is issued from a describe that cannot be trusted"
    )


def test_an_instance_describe_without_its_arn_cannot_be_claimed() -> None:
    runner = InstanceCreationRunner(
        {"aurora-a-writer": "available", "aurora-a-reader": "available"},
        primary="aurora-a-writer",
        cluster_status="available",
    )

    with pytest.raises(BootstrapError, match="aurora-a-writer describe lacks its ARN"):
        aurora.ensure_serverless_instances(
            runner,
            aws_region=REGION,
            cluster_id=CLUSTER,
            availability_zones=["us-east-1a", "us-east-1b"],
            safe_name=lambda value, maximum: value[:maximum],
            wait=False,
            site_id=SITE,
        )
    assert "add-tags-to-resource" not in {call[2] for call in runner.calls}


def test_a_parameter_group_describe_without_its_arn_cannot_be_claimed() -> None:
    runner = ReadRunner(
        {
            "describe-db-cluster-parameter-groups": [
                {"DBClusterParameterGroups": [{"DBClusterParameterGroupName": "pg"}]}
            ]
        }
    )

    with pytest.raises(BootstrapError, match="describe lacks its ARN"):
        aurora.ensure_cluster_parameter_group(
            runner,
            aws_region=REGION,
            cluster_id=CLUSTER,
            engine_version="16.4",
            safe_name=lambda value, maximum: value[:maximum],
            site_id=SITE,
        )
    assert [call[2] for call in runner.calls] == [
        "describe-db-cluster-parameter-groups"
    ], "nothing is modified before ownership can be proven"


def test_the_master_secret_is_read_again_until_the_cluster_publishes_it() -> None:
    secret_arn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:aurora-a-x"
    runner = ReadRunner(
        {
            "describe-db-clusters": [
                {"DBClusters": [{"Endpoint": "w.example", "MasterUserSecret": {}}]},
                {
                    "DBClusters": [
                        {
                            "Endpoint": "w.example",
                            "MasterUserSecret": {
                                "SecretArn": secret_arn,
                                "KmsKeyId": "kms-1",
                            },
                        }
                    ]
                },
            ],
            "get-secret-value": [
                json.dumps({"username": "gpu_fault", "password": "not-a-real-value"})
            ],
        }
    )

    readiness = aurora.read_master_secret(
        runner, aws_region=REGION, cluster_id=CLUSTER, attempts=2, poll_seconds=0
    )

    assert readiness.secret_arn == secret_arn
    assert readiness.username == "gpu_fault"
    assert readiness.kms_key_arn == "kms-1"
    assert [call[2] for call in runner.calls] == [
        "describe-db-clusters",
        "describe-db-clusters",
        "get-secret-value",
    ]


def test_a_cluster_that_never_publishes_its_secret_fails_closed() -> None:
    runner = ReadRunner(
        {"describe-db-clusters": [{"DBClusters": [{"MasterUserSecret": None}]}]}
    )

    with pytest.raises(BootstrapError, match="no managed master secret yet"):
        aurora.read_master_secret(
            runner, aws_region=REGION, cluster_id=CLUSTER, attempts=2, poll_seconds=0
        )
    assert len(runner.calls) == 2, "bounded to the configured attempts"
