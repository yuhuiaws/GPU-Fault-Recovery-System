"""Guard evidence requires complete identities, pagination and transport binding."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import aurora_binding, regional_commands
from scripts.e2e.regional import guardrail_audit_evidence as evidence
from tests.regional import _cov95_residual_support as support
from tests.regional.test_guardrail_audit_evidence import population

residual_isolation = support.residual_isolation


def provider_record():
    nodes = [
        {
            "NodeLogicalId": f"node-{index}",
            "InstanceId": f"instance-{index}",
            "InstanceGroupName": "group",
        }
        for index in range(3)
    ]
    payload = {
        "cluster_name": "private-cluster",
        "describe_cluster": {"ClusterName": "private-cluster"},
        "list_cluster_nodes": [
            {"ClusterNodeSummaries": [nodes[0]], "NextToken": "page-1"},
            {"ClusterNodeSummaries": [nodes[1]], "NextToken": "page-2"},
            {"ClusterNodeSummaries": [nodes[2]]},
        ],
        "describe_cluster_node": {node["NodeLogicalId"]: node for node in nodes},
    }
    return {"payloads": payload, "recorded_at": support.NOW.isoformat()}


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "missing-token",
        "repeat-token",
        "repeat-node",
        "empty-node",
        "extra-detail",
        "wrong-detail",
    ],
)
def test_provider_page_chain_and_node_details_must_match_even_with_valid_digest(defect):
    record = provider_record()
    payload = record["payloads"]
    pages = payload["list_cluster_nodes"]
    if defect == "missing-token":
        pages[0].pop("NextToken")
    elif defect == "repeat-token":
        pages[1]["NextToken"] = pages[0]["NextToken"]
    elif defect == "repeat-node":
        pages[2]["ClusterNodeSummaries"] = pages[0]["ClusterNodeSummaries"]
    elif defect == "empty-node":
        pages[2]["ClusterNodeSummaries"] = [{"NodeLogicalId": ""}]
    elif defect == "extra-detail":
        payload["describe_cluster_node"]["extra"] = {}
    elif defect == "wrong-detail":
        payload["describe_cluster_node"]["node-1"] = {
            **payload["describe_cluster_node"]["node-1"],
            "InstanceId": "other-instance",
        }
    record["payload_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if defect:
        with pytest.raises(RuntimeError, match="provider recording"):
            evidence.validate_provider_record(
                record, "private-cluster", now=support.NOW
            )
    else:
        evidence.validate_provider_record(record, "private-cluster", now=support.NOW)


@pytest.mark.parametrize("registrations", [[None], [{"cluster_id": "a"}], "invalid"])
def test_incomplete_registry_identities_cannot_be_used_for_guard_audit(registrations):
    with pytest.raises(RuntimeError, match="durable registry"):
        evidence.registration_identity(
            {"registrations": registrations}, "managed", "negative", "region-a"
        )


@pytest.mark.parametrize(
    "field,value", [("generation", True), ("generation", 0), ("uid", ""), ("uid", None)]
)
def test_complete_pod_count_is_not_sufficient_without_deployment_identity(field, value):
    deployment, pods = population(1)
    deployment = deepcopy(deployment)
    deployment["metadata"][field] = value
    with pytest.raises(RuntimeError, match="complete stable Ready"):
        evidence.complete_pod_population(deployment, pods)


def test_regional_binding_adapter_preserves_plane_stdin_and_aws_region(monkeypatch):
    kubernetes, aws = [], []
    regional = SimpleNamespace(
        settings=SimpleNamespace(region="region-a", namespace="namespace-a"),
        kubectl=lambda plane, *args, **kwargs: kubernetes.append((plane, args, kwargs))
        or '{"ok":true}',
    )
    monkeypatch.setattr(
        regional_commands,
        "run_fixture_command",
        lambda command, **kwargs: aws.append((command, kwargs))
        or SimpleNamespace(stdout='{"recorded":true}'),
    )
    binding = aurora_binding.regional_binding(regional, "database-a")
    assert binding.control("get", "pod") == '{"ok":true}'
    assert binding.control("exec", "pod", stdin=b"private-input") == '{"ok":true}'
    assert kubernetes[0] == ("cpu", ("get", "pod"), {"input_text": None})
    assert kubernetes[1][2]["input_text"] == "private-input"
    assert binding.aws("rds", "describe-db-clusters") == {"recorded": True}
    assert aws == [
        (
            [
                "aws",
                "rds",
                "describe-db-clusters",
                "--region",
                "region-a",
                "--output",
                "json",
            ],
            {"timeout": 180},
        )
    ]
