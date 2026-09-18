from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from scripts.e2e.regional import audit_warm_spare_guardrails as audit
from scripts.e2e.regional.guardrail_audit_evidence import (
    complete_pod_population,
    registration_identity,
    validate_provider_record,
)


def population(replicas=3):
    deployment = {
        "metadata": {"uid": "deployment-uid", "generation": 2},
        "spec": {"replicas": replicas},
        "status": {
            "observedGeneration": 2,
            "replicas": replicas,
            "readyReplicas": replicas,
            "updatedReplicas": replicas,
            "availableReplicas": replicas,
        },
    }
    pods = {
        "items": [
            {
                "metadata": {"name": f"pod-{index}", "uid": f"uid-{index}"},
                "spec": {"containers": [{"name": "executor"}]},
                "status": {
                    "phase": "Running",
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "containerStatuses": [{"name": "executor", "ready": True}],
                },
            }
            for index in range(replicas)
        ]
    }
    return deployment, pods


def test_guard_inventory_accepts_actual_three_replica_target() -> None:
    deployment, pods = population()
    assert len(complete_pod_population(deployment, pods)) == 3, (
        "guard audits must respect desired replicas, not a hard-coded two"
    )


@pytest.mark.parametrize(
    "defect", ["unready", "missing", "old-generation", "extra", "string", "terminating"]
)
def test_guard_inventory_rejects_incomplete_or_rolling_replicas(defect) -> None:
    deployment, pods = population()
    if defect == "unready":
        pods["items"][0]["status"]["conditions"][0]["status"] = "False"
    elif defect == "missing":
        pods["items"].pop()
    elif defect == "old-generation":
        deployment["status"]["observedGeneration"] = 1
    elif defect == "extra":
        pods["items"].append(deepcopy(pods["items"][0]))
    elif defect == "string":
        deployment["status"]["readyReplicas"] = "3"
    else:
        deployment["metadata"]["deletionTimestamp"] = "2026-09-12T00:00:00Z"
    with pytest.raises(RuntimeError):
        complete_pod_population(deployment, pods)


def recording():
    node = {
        "NodeLogicalId": "node-1",
        "InstanceId": "instance-1",
        "InstanceGroupName": "gpu",
    }
    payload = {
        "cluster_name": "negative",
        "describe_cluster": {"ClusterName": "negative", "NodeRecovery": "Automatic"},
        "list_cluster_nodes": [{"ClusterNodeSummaries": [node], "NextToken": None}],
        "describe_cluster_node": {"node-1": node},
    }
    return {
        "payloads": payload,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "payload_digest": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def test_provider_record_is_fresh_complete_and_bound() -> None:
    snapshot = recording()
    validate_provider_record(snapshot, "negative")
    assert snapshot["payloads"]["describe_cluster"]["NodeRecovery"] == "Automatic"


@pytest.mark.parametrize(
    "defect", ["stale", "future", "naive", "digest", "cluster", "node", "page", "empty"]
)
def test_provider_record_rejects_invalid_facts(defect) -> None:
    snapshot = recording()
    if defect == "stale":
        snapshot["recorded_at"] = (
            datetime.now(timezone.utc) - timedelta(minutes=6)
        ).isoformat()
    elif defect == "future":
        snapshot["recorded_at"] = (
            datetime.now(timezone.utc) + timedelta(minutes=1)
        ).isoformat()
    elif defect == "naive":
        snapshot["recorded_at"] = "2026-09-12T00:00:00"
    elif defect == "digest":
        snapshot["payload_digest"] = "0" * 64
    else:
        payload = snapshot["payloads"]
        if defect == "cluster":
            payload["describe_cluster"]["ClusterName"] = "another"
        elif defect == "node":
            payload["describe_cluster_node"] = {}
        elif defect == "page":
            payload["list_cluster_nodes"][0]["NextToken"] = "unread-page"
        else:
            payload["list_cluster_nodes"][0]["ClusterNodeSummaries"] = []
        snapshot["payload_digest"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    with pytest.raises(RuntimeError, match="provider recording"):
        validate_provider_record(snapshot, "negative")


def test_negative_cluster_must_be_outside_durable_registry() -> None:
    row = {
        "cluster_id": "cluster-a",
        "region": "region-a",
        "hyperpod_cluster_name": "managed",
        "eks_cluster_arn": "arn:aws:eks:region-a:123456789012:cluster/managed",
    }
    value = {"registrations": [row]}
    assert registration_identity(value, "managed", "negative", "region-a") == row
    with pytest.raises(RuntimeError, match="negative cluster is registered"):
        registration_identity(value, "managed", "managed", "region-a")
    with pytest.raises(RuntimeError, match="does not match"):
        registration_identity(value, "managed", "negative", "region-b")
    with pytest.raises(RuntimeError, match="missing"):
        registration_identity({"registrations": []}, "managed", "negative", "region-a")


def test_provider_pagination_repeated_token_is_not_replayed_forever(
    monkeypatch,
) -> None:
    calls = []

    def command(arguments, **_kwargs):
        calls.append(arguments)
        if "describe-cluster" in arguments:
            return '{"ClusterName":"negative","NodeRecovery":"Automatic"}'
        return '{"ClusterNodeSummaries":[],"NextToken":"repeated"}'

    monkeypatch.setattr(audit, "command", command)
    with pytest.raises(RuntimeError, match="pagination repeated"):
        audit.record_provider_snapshot("negative")
    assert len(calls) == 3, "one repeated provider token must bound the read loop"


@pytest.mark.parametrize(
    "event",
    ["BatchRebootClusterNodes", "BatchDeleteClusterNodes", "BatchReplaceClusterNodes"],
)
def test_readonly_guard_checks_every_provider_mutation(monkeypatch, event) -> None:
    monkeypatch.setattr(
        audit,
        "command",
        lambda *_args, **_kwargs: json.dumps(
            {"Events": [{"EventName": event, "EventTime": "2026-09-12T00:00:00Z"}]}
        ),
    )
    now = datetime.now(timezone.utc)
    assert audit.replace_events(now, now)[0]["event_name"] == event


@pytest.mark.parametrize(
    "arguments", [[], ["--case", audit.CASE_IDS[0], "--case", audit.CASE_IDS[1]]]
)
def test_live_audit_requires_one_explicit_case_before_any_reads(
    monkeypatch, arguments
) -> None:
    import sys

    monkeypatch.setattr(sys, "argv", ["guard-audit", *arguments])
    monkeypatch.setattr(
        audit,
        "configure",
        lambda *_args: pytest.fail("invalid selection reached live setup"),
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        audit.main()
