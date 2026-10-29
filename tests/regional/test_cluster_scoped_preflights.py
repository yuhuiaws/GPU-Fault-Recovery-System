"""Gates two GPU clusters' destructive cases can share.

Round 13 ran every destructive case site-serially because its gates read the
whole Region: an open remote command of ANY cluster refused a preflight, and a
HyperPod node reboot on one cluster failed a concurrent case's negative
CloudTrail claim on the other ("an HP reboot fails a concurrent check on EKS").
These pin the per-cluster readings -- ``own_open_remote_commands`` and
``provider_cluster_scope`` -- and that every fail-closed rule survived them.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import audit_warm_spare_guardrails as guardrails
from scripts.e2e.regional import destr015_verdicts, destr017_verdicts
from scripts.e2e.regional import provider_cluster_scope as scope_module
from scripts.e2e.regional.acceptance_runner_common import (
    own_open_remote_commands,
    remote_command_queue_errors,
)
from scripts.e2e.regional.guardrail_audit_evidence import provider_call_inventory
from scripts.e2e.regional.provider_cluster_scope import (
    CONFLICTING,
    EKS_MANAGED,
    FOREIGN,
    HYPERPOD,
    OWN,
    UNBOUND,
    ProviderClusterScope,
    attribution,
    event_identities,
    request_identities,
    scope_from_nodes,
)
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_common_live import LiveModel

OWN_HP = "hp-cluster-a"
OWN_ARN = "arn:aws:sagemaker:test-region:123456789012:cluster/ownclusterid1"
OTHER_HP = "hp-cluster-b"
OTHER_ARN = "arn:aws:sagemaker:test-region:123456789012:cluster/otherclusterid"
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Remote commands: this cluster's open commands, fail closed
# --------------------------------------------------------------------------- #
def test_only_this_clusters_and_unattributed_open_commands_count() -> None:
    reading = {
        "open_by_cluster": {"eks-a": 2, "hp-b": 5, "": 1, "__unscoped__": "1"},
        "by_status": {"PENDING": 9},
    }
    assert own_open_remote_commands(reading, "eks-a") == 4, (
        "own rows plus the rows no cluster can be attributed"
    )
    assert own_open_remote_commands(reading, "hp-b") == 7, reading
    assert own_open_remote_commands({"open_by_cluster": {"hp-b": 3}}, "eks-a") == 0, (
        "another cluster's in-flight case is no backlog here"
    )


@pytest.mark.parametrize(
    "reading",
    [
        None,
        {},
        {"open_by_cluster": None},
        {"open_by_cluster": ["eks-a"]},
        {"open_by_cluster": {"eks-a": -1}},
        {"open_by_cluster": {"eks-a": "many"}},
        {"open_by_cluster": {"eks-a": True}},
    ],
)
def test_a_missing_or_malformed_reading_refuses(reading: Any) -> None:
    with pytest.raises(ValueError, match="open_by_cluster"):
        own_open_remote_commands(reading, "eks-a")
    errors = remote_command_queue_errors(reading, "eks-a")
    assert errors and errors[0].startswith("remote command queue state is unknown"), (
        errors
    )


def test_the_gate_needs_the_cases_own_cluster_id() -> None:
    with pytest.raises(ValueError, match="own cluster ID"):
        own_open_remote_commands({"open_by_cluster": {}}, "")
    assert remote_command_queue_errors({"open_by_cluster": {"eks-a": 1}}, "eks-a") == [
        "remote command queue is not empty (1 open on eks-a)"
    ]
    assert remote_command_queue_errors({"open_by_cluster": {"hp-b": 1}}, "eks-a") == []


# --------------------------------------------------------------------------- #
# Provider events: attribution by the request's cluster binding
# --------------------------------------------------------------------------- #
HP_SCOPE = ProviderClusterScope(HYPERPOD, OWN_HP, OWN_ARN)
EKS_SCOPE = ProviderClusterScope(EKS_MANAGED)


@pytest.mark.parametrize(
    ("identities", "scope", "expected"),
    [
        ([OWN_HP], HP_SCOPE, OWN),
        ([OWN_ARN], HP_SCOPE, OWN),
        ([OWN_HP.upper()], HP_SCOPE, OWN),
        ([OWN_HP, OWN_ARN], HP_SCOPE, OWN),
        ([OTHER_HP], HP_SCOPE, FOREIGN),
        ([OTHER_ARN], HP_SCOPE, FOREIGN),
        ([OWN_HP, OTHER_ARN], HP_SCOPE, CONFLICTING),
        (None, HP_SCOPE, UNBOUND),
        ([], HP_SCOPE, UNBOUND),
        ([OWN_HP], EKS_SCOPE, FOREIGN),
        ([OTHER_ARN], EKS_SCOPE, FOREIGN),
        (None, EKS_SCOPE, UNBOUND),
    ],
)
def test_attribution_by_cluster_binding(
    identities: list[str] | None, scope: ProviderClusterScope, expected: str
) -> None:
    assert attribution(identities, scope) == expected, (identities, scope)


def test_a_binding_that_cannot_be_read_is_unbound() -> None:
    assert request_identities({"clusterName": OWN_HP, "nodeIds": ["x"]}) == [OWN_HP]
    assert request_identities({"ClusterArn": OWN_ARN}) == [OWN_ARN]
    for parameters in (None, "text", {}, {"clusterName": ""}, {"clusterName": 7}):
        assert request_identities(parameters) is None, parameters
    assert event_identities({"request_cluster_name": OWN_HP}) == [OWN_HP]
    assert (
        event_identities(
            {"request_cluster_name": OWN_HP, "normalization_errors": ["bad"]}
        )
        is None
    ), "a record that failed normalisation is never attributed away"


def _nodes(*labels: dict[str, str]) -> dict[str, Any]:
    return {
        "items": [
            {"metadata": {"name": f"node-{index}", "labels": item}}
            for index, item in enumerate(labels)
        ]
    }


def _describe(name: str, arn: str = OWN_ARN) -> Any:
    return lambda requested: {"ClusterName": requested, "ClusterArn": arn}


def test_the_scope_comes_from_the_nodes_and_the_hyperpod_arn() -> None:
    hyperpod = scope_from_nodes(
        _nodes({scope_module.HYPERPOD_CLUSTER_LABEL: OWN_HP}, {}),
        _describe(OWN_HP),
        region="test-region",
    )
    assert hyperpod == HP_SCOPE, hyperpod
    eks = scope_from_nodes(
        _nodes({"eks.amazonaws.com/nodegroup": "gpu-group"}, {}),
        _describe(OWN_HP),
        region="test-region",
    )
    assert eks == EKS_SCOPE and eks.identities == frozenset(), eks


@pytest.mark.parametrize(
    ("document", "describe", "fragment"),
    [
        ({"items": []}, _describe(OWN_HP), "empty or unreadable"),
        ("not-json-object", _describe(OWN_HP), "empty or unreadable"),
        (_nodes({}), _describe(OWN_HP), "neither a HyperPod cluster label"),
        (
            _nodes(
                {scope_module.HYPERPOD_CLUSTER_LABEL: OWN_HP},
                {scope_module.HYPERPOD_CLUSTER_LABEL: OTHER_HP},
            ),
            _describe(OWN_HP),
            "more than one HyperPod cluster",
        ),
        (
            _nodes({scope_module.HYPERPOD_CLUSTER_LABEL: OWN_HP}),
            lambda _name: {"ClusterName": OTHER_HP, "ClusterArn": OWN_ARN},
            "name/ARN",
        ),
        (
            _nodes({scope_module.HYPERPOD_CLUSTER_LABEL: OWN_HP}),
            _describe(OWN_HP, "arn:aws:sagemaker:other-region:1:cluster/x"),
            "name/ARN",
        ),
    ],
)
def test_an_unattributable_cluster_refuses(
    document: Any, describe: Any, fragment: str
) -> None:
    with pytest.raises(RegionalFixtureError, match=fragment):
        scope_from_nodes(document, describe, region="test-region")


# --------------------------------------------------------------------------- #
# The real lookup, judged by the cases' own CloudTrail verdicts
# --------------------------------------------------------------------------- #
def _trail_row(event_id: str, name: str, **parameters: Any) -> dict[str, Any]:
    stamp = NOW.isoformat()
    detail = {
        "eventID": event_id,
        "requestID": f"request-{event_id}",
        "eventName": name,
        "eventTime": stamp,
        "eventSource": "sagemaker.amazonaws.com",
        "awsRegion": "test-region",
        "recipientAccountId": "123456789012",
        "userIdentity": {"type": "AssumedRole", "accountId": "123456789012"},
        "requestParameters": {"nodeLogicalIds": ["logical-1"], **parameters},
        "responseElements": {"successfulNodeLogicalIds": ["logical-1"]},
    }
    return {
        "EventName": name,
        "EventTime": stamp,
        "EventId": event_id,
        "Username": "session",
        "CloudTrailEvent": json.dumps(detail),
    }


class ScopedLive(LiveModel):
    """A fixture whose GPU Nodes and SageMaker answer for one cluster."""

    def __init__(self, root: Path, *, hyperpod: str = "") -> None:
        super().__init__(root)
        labels = (
            {scope_module.HYPERPOD_CLUSTER_LABEL: hyperpod}
            if hyperpod
            else {"eks.amazonaws.com/nodegroup": "gpu-group"}
        )
        self.documents[("gpu", "node", "")] = _nodes(labels)
        self.described: list[list[str]] = []

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["aws", "sagemaker", "describe-cluster"]:
            self.described.append(list(command))
            name = command[command.index("--cluster-name") + 1]
            answer = {"ClusterName": name, "ClusterArn": OWN_ARN}
            return subprocess.CompletedProcess(command, 0, json.dumps(answer), "")
        return super().run(command, **kwargs)


def _lookup(fixture: ScopedLive, *rows: dict[str, Any]) -> list[dict[str, Any]]:
    fixture.events = [{"Events": list(rows)}]
    return fixture.provider_events(NOW, NOW + timedelta(minutes=5))


def test_another_hyperpod_clusters_reboot_does_not_fail_an_eks_case(
    tmp_path: Path,
) -> None:
    eks = ScopedLive(tmp_path)
    events = _lookup(
        eks, _trail_row("e-1", "BatchRebootClusterNodes", clusterName=OTHER_HP)
    )
    assert events == [], events
    assert destr015_verdicts.cloudtrail_errors(events) == []
    assert destr017_verdicts.cloudtrail_errors(events) == []
    assert eks.provider_events_set_aside == [
        {
            "event_name": "BatchRebootClusterNodes",
            "event_time": NOW.isoformat(),
            "event_id": "e-1",
            "request_cluster_name": OTHER_HP,
            "request_cluster_arn": None,
        }
    ], "what was set aside stays visible"
    assert eks.provider_events_scope == EKS_SCOPE.evidence()
    assert eks.described == [], "a plain EKS cluster has no SageMaker cluster"


def test_an_unbound_sagemaker_call_still_fails_the_eks_case(tmp_path: Path) -> None:
    eks = ScopedLive(tmp_path)
    row = _trail_row("e-2", "BatchRebootClusterNodes", clusterName=OTHER_HP)
    row["CloudTrailEvent"] = "{"
    events = _lookup(eks, row)
    assert [item["event_name"] for item in events] == ["BatchRebootClusterNodes"]
    assert destr015_verdicts.cloudtrail_errors(events), "unreadable binding fails"


def test_a_hyperpod_case_keeps_its_own_reboot_and_sets_another_clusters_aside(
    tmp_path: Path,
) -> None:
    hp = ScopedLive(tmp_path, hyperpod=OWN_HP)
    events = _lookup(
        hp,
        _trail_row("h-1", "BatchRebootClusterNodes", clusterName=OWN_HP),
        _trail_row("h-2", "BatchRebootClusterNodes", clusterName=OWN_ARN),
        _trail_row("h-3", "BatchRebootClusterNodes", clusterName=OTHER_HP),
        _trail_row("h-4", "BatchDeleteClusterNodes", clusterName=OTHER_ARN),
        _trail_row(
            "h-5", "BatchRebootClusterNodes", clusterName=OWN_HP, clusterArn=OTHER_ARN
        ),
    )
    assert [item["event_id"] for item in events] == ["h-1", "h-2", "h-5"], (
        "own (by name or ARN) and conflicting calls stay this case's"
    )
    assert destr017_verdicts.cloudtrail_errors(events), "own reboot still fails"
    assert [item["event_id"] for item in hp.provider_events_set_aside] == ["h-3", "h-4"]
    assert len(hp.described) == 1, "the ARN is read once"
    _lookup(hp, _trail_row("h-6", "BatchRebootClusterNodes", clusterName=OTHER_HP))
    assert len(hp.described) == 1, "the scope is kept on the fixture"


def test_a_window_without_bound_events_reads_no_scope(tmp_path: Path) -> None:
    fixture = ScopedLive(tmp_path, hyperpod=OWN_HP)
    assert _lookup(fixture) == []
    assert fixture.described == [] and not any(
        command[0] == "kubectl" for command, _ in fixture.calls
    ), "no node or SageMaker read for an empty window"


def test_ec2_events_are_returned_unscoped(tmp_path: Path) -> None:
    row = _trail_row("x-1", "RebootInstances")
    detail = json.loads(row["CloudTrailEvent"])
    detail.update(
        eventSource="ec2.amazonaws.com",
        requestParameters={"instancesSet": {"items": [{"instanceId": "i-1"}]}},
        responseElements={"_return": True},
    )
    row["CloudTrailEvent"] = json.dumps(detail)
    fixture = ScopedLive(tmp_path)
    fixture.events = [{"Events": [row]}]
    events = fixture.provider_events(NOW, NOW, event_source="ec2.amazonaws.com")
    assert [item["event_id"] for item in events] == ["x-1"], (
        "EC2 callers keep their own instance and role filters"
    )


# --------------------------------------------------------------------------- #
# DESTR-005/006/007: the guardrail inventory, same scoping
# --------------------------------------------------------------------------- #
def _sagemaker_only(rows: list[dict[str, Any]]) -> Any:
    def command(argv: list[str], **_kwargs: Any) -> str:
        source = next(a for a in argv if a.startswith("AttributeKey=EventSource"))
        if not source.endswith("sagemaker.amazonaws.com"):
            return json.dumps({"Events": []})
        return json.dumps({"Events": rows})

    return command


@pytest.mark.parametrize(
    ("provider", "scope", "kept", "aside"),
    [
        ("eks-managed", EKS_SCOPE, ["g-4"], ["g-1", "g-2", "g-3"]),
        ("hyperpod", HP_SCOPE, ["g-1", "g-2", "g-4"], ["g-3"]),
        ("hyperpod", None, ["g-1", "g-2", "g-3", "g-4"], None),
    ],
)
def test_the_guardrail_inventory_sets_only_other_clusters_calls_aside(
    provider: str,
    scope: ProviderClusterScope | None,
    kept: list[str],
    aside: list[str] | None,
) -> None:
    rows = [
        _trail_row("g-1", "BatchRebootClusterNodes", clusterName=OWN_HP),
        _trail_row("g-2", "BatchRebootClusterNodes", clusterArn=OWN_ARN),
        _trail_row("g-3", "BatchRebootClusterNodes", clusterName=OTHER_HP),
        {**_trail_row("g-4", "BatchRebootClusterNodes"), "CloudTrailEvent": "{"},
    ]
    bindings = {"g-1": [OWN_HP], "g-2": [OWN_ARN], "g-3": [OTHER_HP]}
    inventory = provider_call_inventory(
        NOW,
        NOW,
        provider=provider,
        region="test-region",
        command=_sagemaker_only(rows),
        hyperpod_mutations=frozenset({"BatchRebootClusterNodes"}),
        cluster_scope=scope,
    )
    assert len(inventory["mutations"]) == len(kept), inventory
    if aside is None:
        assert "other_cluster_calls" not in inventory, "no scope, nothing aside"
    else:
        assert [
            item["request_cluster"] for item in inventory["other_cluster_calls"]
        ] == [bindings[key] for key in aside], inventory


def test_the_guardrail_scope_needs_the_hyperpod_arn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(guardrails, "MANAGED_GPU_CLUSTER", OWN_HP)
    monkeypatch.setattr(guardrails, "MANAGED_PROVIDER", "hyperpod")
    assert guardrails.managed_cluster_scope(
        {"cluster_name": OWN_HP, "cluster_arn": OWN_ARN}
    ) == ProviderClusterScope(HYPERPOD, OWN_HP, OWN_ARN)
    assert guardrails.managed_cluster_scope({"cluster_name": OWN_HP}) is None, (
        "without the ARN nothing is set aside"
    )
    assert guardrails.managed_cluster_scope(None) is None
    monkeypatch.setattr(guardrails, "MANAGED_PROVIDER", "eks-managed")
    assert guardrails.managed_cluster_scope(None) == EKS_SCOPE
