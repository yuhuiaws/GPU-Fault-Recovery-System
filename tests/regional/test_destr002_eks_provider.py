"""GF-REGIONAL-DESTR-002 on an eks-managed cluster.

The case keeps its id on both providers: the preflight probe finds the
executor's node lifecycle adapter by the ``nodeReboot`` owner each provider's
profile names, judges the same four read-only preflights in that adapter's
words, and the live run reads CloudTrail from the provider's own event
source -- EC2 ``RebootInstances`` beside Karpenter's refused ``RunInstances``
dry-runs on plain EKS, SageMaker's ``BatchRebootClusterNodes`` on HyperPod.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_destr002_hyperpod_reboot as destr002
from scripts.e2e.regional.collector_reboot_evidence import normalize_provider_event
from tests.regional.test_collector_reboot_scope_eks import (
    ROLE,
    ec2_cloudtrail_item,
    karpenter_run_instances_item,
)

NODE = "ip-10-0-41-164.us-west-2.compute.internal"
CLUSTER = "gpu-eks-cluster"


def _eks_probe(**overrides: Any) -> dict[str, Any]:
    missing = {
        "safe_to_submit": False,
        "gate_failures": [
            f"trusted scheduler isolation evidence is missing for: {NODE}"
        ],
    }
    probe: dict[str, Any] = {
        "provider": "eks-managed",
        "adapter_owner": "gpu-fault-eks-adapter",
        "configured_cluster_name": CLUSTER,
        "positive": {
            "safe_to_submit": True,
            "node_recovery": None,
            "cluster_status": "ACTIVE",
            "gate_failures": [],
            "targets": [
                {"node_logical_id": NODE, "instance_id": "i-00000000000000001"}
            ],
        },
        "missing_isolation": missing,
        "wrong_isolation": missing,
        "disabled": {
            "safe_to_submit": False,
            "gate_failures": ["EKS REBOOT mutation is disabled by configuration"],
        },
    }
    probe.update(overrides)
    return probe


def test_the_eks_probe_passes_the_same_four_preflights() -> None:
    assert destr002.preflight_probe_errors(_eks_probe(), CLUSTER) == []
    assert destr002.probe_provider(_eks_probe()) == "eks-managed"
    assert destr002.probe_provider({}) == "hyperpod"


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        (
            {"positive": {**_eks_probe()["positive"], "node_recovery": "None"}},
            "an eks-managed preflight reports a HyperPod NodeRecovery: 'None'",
        ),
        (
            {"positive": {**_eks_probe()["positive"], "cluster_status": "UPDATING"}},
            "the EKS cluster is not ACTIVE: 'UPDATING'",
        ),
        (
            {"disabled": {"safe_to_submit": False, "gate_failures": ["other"]}},
            "disabled reboot preflight did not fail closed",
        ),
        ({"configured_cluster_name": "other"}, "cluster name differs from the plan"),
        (
            {"wrong_isolation": {"safe_to_submit": True, "gate_failures": []}},
            "wrong isolation preflight did not fail closed",
        ),
    ],
)
def test_each_eks_preflight_gate_is_judged(
    overrides: dict[str, Any], fragment: str
) -> None:
    errors = destr002.preflight_probe_errors(_eks_probe(**overrides), CLUSTER)
    assert errors == [fragment] or any(fragment in item for item in errors), errors


def test_the_hyperpod_probe_still_needs_node_recovery_none() -> None:
    probe = _eks_probe(provider="hyperpod")
    probe["positive"] = {**probe["positive"], "node_recovery": None}
    assert destr002.preflight_probe_errors(probe, CLUSTER) == [
        "HyperPod NodeRecovery is not None"
    ]


def test_the_probes_select_the_adapter_by_the_provider_owner_map() -> None:
    for probe in (destr002.PREFLIGHT_PROBE, destr002.STORE_REPLAY_PROBE):
        assert "PROVIDER_NODE_REBOOT_OWNERS" in probe
        assert "gpu-fault-hyperpod-adapter" not in probe, (
            "no provider owner is spelled out; the map is the single source"
        )
    assert "EksManagedNodeProvider" in destr002.PREFLIGHT_PROBE, (
        "the disabled variant is built from the live eks config"
    )
    assert "_ledger.replay" in destr002.STORE_REPLAY_PROBE, (
        "the eks provider's ledger answers the durable replay"
    )


def _events(*items: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(normalize_provider_event(item)) for item in items]


class _Regional:
    def __init__(
        self, events: list[dict[str, Any]], *, provisional: bool = True
    ) -> None:
        self.events = events
        self.provisional = provisional
        self.calls: list[dict[str, Any]] = []

    def wait_provider_events(self, started_at: datetime, **kwargs: Any) -> list:
        self.calls.append({"wait": kwargs})
        return [
            item for item in self.events if item["event_name"] in kwargs["event_names"]
        ]

    def provider_events(
        self, started_at: datetime, ended_at: datetime, *, event_source: str
    ) -> list[dict[str, Any]]:
        self.calls.append({"read": event_source})
        return self.events

    def provider_events_provisional(self, _ended_at: datetime) -> bool:
        return self.provisional


def _settings() -> Any:
    return SimpleNamespace(executor_role_arn=ROLE)


def test_the_live_eks_window_counts_one_reboot_beside_the_dry_runs() -> None:
    regional = _Regional(
        _events(
            karpenter_run_instances_item(1),
            ec2_cloudtrail_item(),
            karpenter_run_instances_item(2),
            karpenter_run_instances_item(3),
        )
    )
    errors, evidence = destr002.provider_reboot_evidence(
        regional,  # type: ignore[arg-type]
        _settings(),
        injection_started=datetime.now(timezone.utc),
        provider="eks-managed",
    )
    assert errors == [], errors
    assert regional.calls[0]["wait"] == {
        "event_names": {"RebootInstances"},
        "expected_count": 1,
        "event_source": "ec2.amazonaws.com",
    }
    assert regional.calls[1] == {"read": "ec2.amazonaws.com"}
    assert evidence["provider"] == "eks-managed"
    assert [item["event_name"] for item in evidence["reboot_events"]] == [
        "RebootInstances"
    ]
    assert len(evidence["refused_provider_calls"]) == 3
    assert evidence["forbidden_events"] == []
    assert evidence["provider_events_provisional"] is True


def test_an_accepted_launch_is_a_replacement_and_never_provisional() -> None:
    regional = _Regional(
        _events(ec2_cloudtrail_item(), karpenter_run_instances_item(1, dry_run=False))
    )
    errors, evidence = destr002.provider_reboot_evidence(
        regional,  # type: ignore[arg-type]
        _settings(),
        injection_started=datetime.now(timezone.utc),
        provider="eks-managed",
    )
    assert errors == ["CloudTrail contains replace/delete mutation"]
    assert [item["event_name"] for item in evidence["forbidden_events"]] == [
        "RunInstances"
    ]
    assert evidence["provider_events_provisional"] is False


def test_an_empty_or_refused_window_names_what_it_saw() -> None:
    regional = _Regional([])
    errors, _ = destr002.provider_reboot_evidence(
        regional,  # type: ignore[arg-type]
        _settings(),
        injection_started=datetime.now(timezone.utc),
        provider="eks-managed",
    )
    assert errors == [
        "CloudTrail does not contain exactly one reboot event (0 accepted "
        "['RebootInstances']; 0 refused provider calls set aside)"
    ]
    other_actor = _Regional(_events(ec2_cloudtrail_item()))
    errors, _ = destr002.provider_reboot_evidence(
        other_actor,  # type: ignore[arg-type]
        SimpleNamespace(executor_role_arn="arn:aws:iam::123456789012:role/other"),
        injection_started=datetime.now(timezone.utc),
        provider="eks-managed",
    )
    assert errors == ["CloudTrail reboot actor is not the executor role"]


def test_the_hyperpod_window_keeps_its_sagemaker_vocabulary() -> None:
    regional = _Regional(
        [
            {
                "event_name": "BatchRebootClusterNodes",
                "session_issuer_role_name": ROLE.rsplit("/", 1)[-1],
            },
            {"event_name": "RunInstances"},
        ]
    )
    errors, evidence = destr002.provider_reboot_evidence(
        regional,  # type: ignore[arg-type]
        _settings(),
        injection_started=datetime.now(timezone.utc),
        provider="hyperpod",
    )
    assert errors == [], "an EC2 launch is not a SageMaker replacement"
    assert regional.calls[0]["wait"]["event_source"] == "sagemaker.amazonaws.com"
    assert regional.calls[0]["wait"]["event_names"] == destr002.REBOOT_EVENTS
    assert evidence["forbidden_events"] == []
