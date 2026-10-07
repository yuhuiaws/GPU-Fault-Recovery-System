"""GF-REGIONAL-DESTR-014 provider reboot audit, both providers' live shapes.

The HyperPod run leaves one accepted ``BatchRebootClusterNodes`` per node in
``sagemaker.amazonaws.com`` naming the node's logical id; the eks-managed run
leaves one accepted EC2 ``RebootInstances`` per node in ``ec2.amazonaws.com``
naming the instance, beside Karpenter's refused ``RunInstances`` dry-runs
(attempt 14 read the SageMaker source there and saw nothing).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.e2e.regional import run_destr014_branch_exhaustion as destr014
from tests.regional.test_destr014_branch_exhaustion import FAULT, SIBLING

FAULT_INSTANCE = "i-00000000000000001"
SIBLING_INSTANCE = "i-00000000000000002"
HYPERPOD_IDENTITIES = {
    FAULT: {"instance_id": FAULT_INSTANCE, "node_logical_id": "logical-b"},
    SIBLING: {"instance_id": SIBLING_INSTANCE, "node_logical_id": "logical-c"},
}
EKS_IDENTITIES = {
    FAULT: {"instance_id": FAULT_INSTANCE, "node_logical_id": FAULT},
    SIBLING: {"instance_id": SIBLING_INSTANCE, "node_logical_id": SIBLING},
}


def _event(
    name: str,
    *,
    source: str = "sagemaker.amazonaws.com",
    logical_ids: list[str] | None = None,
    instance_ids: list[str] | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    """A CloudTrail row as ``normalize_provider_event`` returns it: the live
    HyperPod reboot names its node by logical id in the SageMaker source; the
    live eks-managed one names the EC2 instance in ``ec2.amazonaws.com``."""

    return {
        "event_name": name,
        "event_source": source,
        "event_time": "t",
        "username": "executor",
        "session_issuer_role_name": "executor-role",
        "api_error": error_code is not None,
        "api_error_code": error_code,
        "request_node_ids": instance_ids,
        "request_node_logical_ids": logical_ids,
        "response_failed_node_count": 0,
    }


def _hyperpod_reboots() -> list[dict[str, Any]]:
    return [
        _event("BatchRebootClusterNodes", logical_ids=["logical-b"]),
        _event("BatchRebootClusterNodes", logical_ids=["logical-c"]),
    ]


def _eks_reboots() -> list[dict[str, Any]]:
    ec2 = "ec2.amazonaws.com"
    return [
        _event("RebootInstances", source=ec2, instance_ids=[FAULT_INSTANCE]),
        _event("RebootInstances", source=ec2, instance_ids=[SIBLING_INSTANCE]),
    ]


def _karpenter_dry_run() -> dict[str, Any]:
    return _event(
        "RunInstances", source="ec2.amazonaws.com", error_code="Client.DryRunOperation"
    )


def test_cloudtrail_requires_one_accepted_hyperpod_reboot_per_node_and_no_replace() -> (
    None
):
    two = _hyperpod_reboots()
    check = lambda events: destr014.cloudtrail_errors(  # noqa: E731
        events, FAULT, SIBLING, provider="hyperpod", identities=HYPERPOD_IDENTITIES
    )
    assert check(two) == []
    one = check(two[:1])
    assert any(SIBLING in item and "got 0" in item for item in one), one
    assert not any(FAULT in item for item in one), one
    same_node_twice = check([two[0], two[0]])
    assert any(FAULT in item and "got 2" in item for item in same_node_twice), (
        same_node_twice
    )
    batched = check(
        [_event("BatchRebootClusterNodes", logical_ids=["logical-b", "logical-c"])]
    )
    assert any(
        "not attributable to exactly one case node" in item for item in batched
    ), batched
    by_instance = check(
        [
            _event("BatchRebootClusterNodes", instance_ids=[FAULT_INSTANCE]),
            _event("BatchRebootClusterNodes", instance_ids=[SIBLING_INSTANCE]),
        ]
    )
    assert by_instance == [], "a request by EC2 instance id names the node too"
    third = check([*two, _event("RebootClusterNodes", logical_ids=["logical-z"])])
    assert third, "a reboot of a node outside the case must be refused"
    replaced = check(
        [*two, _event("BatchReplaceClusterNodes", logical_ids=["logical-c"])]
    )
    assert any("BatchReplaceClusterNodes" in item for item in replaced), replaced
    deleted = check(
        [*two, _event("BatchDeleteClusterNodes", logical_ids=["logical-c"])]
    )
    assert any("BatchDeleteClusterNodes" in item for item in deleted), deleted
    terminated = check(
        [
            *two,
            _event(
                "TerminateInstances",
                source="ec2.amazonaws.com",
                instance_ids=[FAULT_INSTANCE],
            ),
        ]
    )
    assert terminated == [], "EC2 verbs are not the HyperPod forbidden vocabulary"


def test_cloudtrail_reads_the_eks_reboots_in_the_ec2_source_and_sets_karpenter_aside() -> (
    None
):
    """Live eks-managed attempt 14: the two reboots are EC2 ``RebootInstances``
    by the control plane's role in ``ec2.amazonaws.com`` (zero SageMaker
    events), beside Karpenter's refused ``RunInstances`` dry-runs."""

    check = lambda events: destr014.cloudtrail_errors(  # noqa: E731
        events, FAULT, SIBLING, provider="eks-managed", identities=EKS_IDENTITIES
    )
    live = [_karpenter_dry_run(), *_eks_reboots(), _karpenter_dry_run()]
    assert check(live) == []
    nothing = check([_karpenter_dry_run()])
    assert nothing == [
        "CloudTrail does not contain exactly one accepted eks-managed reboot "
        f"['RebootInstances'] for {FAULT}: got 0 (1 refused provider calls set aside)",
        "CloudTrail does not contain exactly one accepted eks-managed reboot "
        f"['RebootInstances'] for {SIBLING}: got 0 (1 refused provider calls set aside)",
    ]
    sagemaker_named = check(_hyperpod_reboots())
    assert len(sagemaker_named) == 2 and all(
        "got 0" in item for item in sagemaker_named
    ), sagemaker_named
    wrong_source = check(
        [_event("RebootInstances", instance_ids=[FAULT_INSTANCE]), _eks_reboots()[1]]
    )
    assert any("not attributable" in item for item in wrong_source), wrong_source
    refused_reboot = check(
        [
            _event(
                "RebootInstances",
                source="ec2.amazonaws.com",
                instance_ids=[FAULT_INSTANCE],
                error_code="UnauthorizedOperation",
            ),
            _eks_reboots()[1],
        ]
    )
    assert any(FAULT in item and "1 refused" in item for item in refused_reboot), (
        refused_reboot
    )
    terminated = check(
        [
            *_eks_reboots(),
            _event(
                "TerminateInstances",
                source="ec2.amazonaws.com",
                instance_ids=[FAULT_INSTANCE],
            ),
        ]
    )
    assert terminated == [
        "CloudTrail contains a forbidden mutation: TerminateInstances"
    ]
    refused_terminate = check(
        [
            *_eks_reboots(),
            _event(
                "TerminateInstances",
                source="ec2.amazonaws.com",
                instance_ids=[FAULT_INSTANCE],
                error_code="Client.DryRunOperation",
            ),
        ]
    )
    assert refused_terminate == [], "a refused call mutated nothing"
    unknown = destr014.cloudtrail_errors(
        _eks_reboots(),
        FAULT,
        SIBLING,
        provider="eks-managed",
        identities={FAULT: EKS_IDENTITIES[FAULT], SIBLING: {}},
    )
    assert unknown == [
        f"{SIBLING} has no provider identity (providerID or inventory row); "
        "its reboot cannot be attributed",
        "CloudTrail contains 1 accepted reboot call(s) in ec2.amazonaws.com "
        "not attributable to exactly one case node",
    ]


def test_node_reboot_identities_bind_provider_id_to_the_inventory_row() -> None:
    inventory = {
        "count": 2,
        "nodes": [
            {"instance_id": FAULT_INSTANCE, "node_logical_id": "logical-b"},
            {"instance_id": SIBLING_INSTANCE, "node_logical_id": "logical-c"},
        ],
    }
    nodes = {
        FAULT: {"name": FAULT, "provider_id": f"aws:///us-west-2a/{FAULT_INSTANCE}"},
        SIBLING: {
            "name": SIBLING,
            "provider_id": f"aws:///us-west-2a/{SIBLING_INSTANCE}",
        },
    }
    assert destr014.node_reboot_identities(inventory, nodes) == HYPERPOD_IDENTITIES
    eks_inventory = {
        "count": 2,
        "nodes": [
            {"instance_id": FAULT_INSTANCE, "node_logical_id": FAULT},
            {"instance_id": SIBLING_INSTANCE, "node_logical_id": SIBLING},
        ],
    }
    assert destr014.node_reboot_identities(eks_inventory, nodes) == EKS_IDENTITIES
    without_provider_id = destr014.node_reboot_identities(
        inventory, {FAULT: {"name": FAULT}, SIBLING: None}
    )
    assert without_provider_id == {
        FAULT: {"instance_id": None, "node_logical_id": None},
        SIBLING: {"instance_id": None, "node_logical_id": None},
    }
    unlisted = destr014.node_reboot_identities({"count": 0, "nodes": []}, nodes)
    assert unlisted[FAULT] == {
        "instance_id": FAULT_INSTANCE,
        "node_logical_id": None,
    }, "an instance the inventory does not list keeps its EC2 id only"


def test_provider_vocabularies_follow_the_cluster_provider() -> None:
    assert (
        destr014.provider_reboot_events(None)
        == destr014.provider_reboot_events("hyperpod")
        == frozenset(destr014.REBOOT_EVENTS)
    )
    assert destr014.provider_reboot_events("eks-managed") == {"RebootInstances"}
    assert destr014.provider_forbidden_events("hyperpod") == frozenset(
        destr014.FORBIDDEN_EVENTS
    )
    assert destr014.provider_forbidden_events("eks-managed") == {
        "RunInstances",
        "StartInstances",
        "StopInstances",
        "TerminateInstances",
    }


class _AuditRegional:
    """Records which CloudTrail source the runner asks for, in order."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.calls: list[tuple[str, str | None, int | None]] = []

    def wait_provider_events(
        self, started_at: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.calls.append(
            ("wait", kwargs.get("event_source"), kwargs.get("expected_count"))
        )
        return [
            item for item in self.events if item["event_name"] in kwargs["event_names"]
        ]

    def provider_events(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("lookup", kwargs.get("event_source"), None))
        return list(self.events)


def _audit_run(
    tmp_path: Path, *, provider: str, events: list[dict[str, Any]]
) -> SimpleNamespace:
    logical = {
        "hyperpod": {FAULT: "logical-b", SIBLING: "logical-c"},
        "eks-managed": {FAULT: FAULT, SIBLING: SIBLING},
    }[provider]
    instances = {FAULT: FAULT_INSTANCE, SIBLING: SIBLING_INSTANCE}
    return SimpleNamespace(
        settings=SimpleNamespace(fault_node=FAULT, sibling_node=SIBLING),
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        case_dir=tmp_path,
        preflight={
            "provider_inventory": {
                "count": 2,
                "nodes": [
                    {"instance_id": instances[node], "node_logical_id": logical[node]}
                    for node in (FAULT, SIBLING)
                ],
            }
        },
        regional=_AuditRegional(events),
        warm=SimpleNamespace(
            provider=provider,
            node_snapshot=lambda node: {
                "name": node,
                "provider_id": f"aws:///us-west-2a/{instances[node]}",
            },
        ),
    )


def test_provider_reboot_audit_reads_the_eks_source_after_the_delivery_wait(
    tmp_path: Path,
) -> None:
    """Attempt 14 on the eks cluster: the audit must ask CloudTrail for
    ``ec2.amazonaws.com`` (not the SageMaker source), wait for the two
    reboots first, attribute each ``RebootInstances`` to its node by instance
    id and keep Karpenter's refused dry-runs visible but uncounted."""

    events = [_karpenter_dry_run(), *_eks_reboots(), _karpenter_dry_run()]
    run = _audit_run(tmp_path, provider="eks-managed", events=events)

    errors, returned = destr014.provider_reboot_audit(run)

    assert errors == []
    assert returned == events
    assert run.regional.calls == [
        ("wait", "ec2.amazonaws.com", 2),
        ("lookup", "ec2.amazonaws.com", None),
    ]
    record = json.loads((tmp_path / "provider-events.json").read_text())
    assert record["provider"] == "eks-managed"
    assert record["event_source"] == "ec2.amazonaws.com"
    assert record["node_identities"] == EKS_IDENTITIES
    assert [item["event_name"] for item in record["refused_provider_calls"]] == [
        "RunInstances",
        "RunInstances",
    ]
    assert record["events"] == events


def test_provider_reboot_audit_keeps_the_hyperpod_source_and_fails_on_a_missing_reboot(
    tmp_path: Path,
) -> None:
    run = _audit_run(tmp_path, provider="hyperpod", events=_hyperpod_reboots()[:1])

    errors, _returned = destr014.provider_reboot_audit(run)

    assert run.regional.calls == [
        ("wait", "sagemaker.amazonaws.com", 2),
        ("lookup", "sagemaker.amazonaws.com", None),
    ]
    assert errors == [
        "CloudTrail does not contain exactly one accepted hyperpod reboot "
        "['BatchRebootClusterNodes', 'RebootClusterNodes'] for node-c: got 0 "
        "(0 refused provider calls set aside)"
    ]
    record = json.loads((tmp_path / "provider-events.json").read_text())
    assert record["node_identities"] == HYPERPOD_IDENTITIES
