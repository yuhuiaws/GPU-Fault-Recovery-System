from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gpu_fault.admin import bootstrap_aurora as aurora
from gpu_fault.admin.bootstrap_common import BootstrapError, safe_name
from tests.admin.test_admin_bootstrap_aurora import InstanceCreationRunner


@pytest.mark.parametrize(
    "response",
    [
        "invalid-json",
        {"Error": {"Code": "example"}},
        {
            "DBInstances": [
                {"DBInstanceIdentifier": "example", "DBInstanceStatus": "failed"}
            ]
        },
    ],
)
def test_aurora_waiter_rejects_invalid_or_terminal_responses_without_new_mutation(
    response,
):
    calls = []

    def run(arguments, **options):
        calls.append((arguments, options))
        return response if isinstance(response, str) else json.dumps(response)

    with pytest.raises(
        BootstrapError, match="invalid JSON|error response|terminal failure"
    ):
        aurora.await_serverless_instances(
            SimpleNamespace(run=run), aws_region="us-east-1", instance_ids=["example"]
        )
    assert len(calls) == 1
    assert calls[0][0][2] == "describe-db-instances"


@pytest.mark.parametrize(
    "cluster", [[], [{}], [{"DBClusterIdentifier": "foreign", "DBClusterMembers": []}]]
)
def test_replica_create_requires_a_valid_source_cluster(cluster):
    runner = InstanceCreationRunner()
    original = runner.aws_json

    def read(region, *arguments, **options):
        if arguments[1] == "describe-db-clusters":
            return {"DBClusters": cluster}
        return original(region, *arguments, **options)

    runner.aws_json = read
    with pytest.raises(BootstrapError, match="replica source cluster"):
        aurora.ensure_serverless_instances(
            runner,
            aws_region="us-east-1",
            cluster_id="aurora-a",
            availability_zones=["us-east-1a", "us-east-1b"],
            safe_name=safe_name,
            wait=False,
        )
    assert runner.transitions == []


def test_replica_create_requires_exactly_one_source_primary():
    runner = InstanceCreationRunner(
        {"aurora-a-writer": "available"}, primary=None, cluster_status="available"
    )
    with pytest.raises(BootstrapError, match="replica source primary"):
        aurora.ensure_serverless_instances(
            runner,
            aws_region="us-east-1",
            cluster_id="aurora-a",
            availability_zones=["us-east-1a", "us-east-1b"],
            safe_name=safe_name,
            wait=False,
        )
    assert runner.transitions == []


@pytest.mark.parametrize("value", ["invalid-json", "{}", '{"username":"example"}'])
def test_aurora_master_secret_read_never_accepts_partial_or_unparsed_data(
    monkeypatch, value
):
    calls = []
    sleeps = []

    def cluster(*_args, **_options):
        return {
            "DBClusters": [
                {
                    "Endpoint": "example.invalid",
                    "MasterUserSecret": {
                        "SecretArn": "arn:aws:secretsmanager:us-east-1:123456789012:secret:example"
                    },
                }
            ]
        }

    def secret(*_args, **_options):
        calls.append("read")
        return value

    monkeypatch.setattr(aurora, "time", SimpleNamespace(sleep=sleeps.append))
    with pytest.raises(BootstrapError, match="not readable after 2 attempts"):
        aurora.read_master_secret(
            SimpleNamespace(aws_json=cluster, aws_text=secret),
            aws_region="us-east-1",
            cluster_id="example",
            attempts=2,
            poll_seconds=1,
        )
    assert calls == ["read", "read"]
    assert sleeps == [1]


def test_zero_master_secret_attempts_never_reads_credentials():
    def forbidden(*_args, **_options):
        pytest.fail("zero-attempt request read a Secret")

    with pytest.raises(BootstrapError, match="not readable after 0 attempts"):
        aurora.read_master_secret(
            SimpleNamespace(aws_json=forbidden, aws_text=forbidden),
            aws_region="us-east-1",
            cluster_id="example",
            attempts=0,
        )


def test_parameter_group_creation_requires_a_discovered_engine_family():
    mutations = []

    def read(_region, _service, operation, *_arguments, **_options):
        if operation == "describe-db-cluster-parameter-groups":
            raise BootstrapError("DBParameterGroupNotFound")
        return {"DBEngineVersions": []}

    with pytest.raises(
        BootstrapError, match="cannot resolve the parameter group family"
    ):
        aurora.ensure_cluster_parameter_group(
            SimpleNamespace(
                aws_json=read, run=lambda *_args, **_options: mutations.append("write")
            ),
            aws_region="us-east-1",
            cluster_id="example",
            engine_version="15.8",
            safe_name=safe_name,
        )
    assert mutations == []
