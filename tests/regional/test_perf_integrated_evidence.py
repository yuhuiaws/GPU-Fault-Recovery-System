"""The integrated verdict judges the drain and names the configuration.

Perf plan §8.4 lists "processor queue and spool back to the depth from before
the round started" among the verdict's checks and §9 says the report records
the global and the three role configuration digests. The runner measured the
drain (``summary.json.queue_drain``) but never judged it, and recorded only
the release id and the wheel and module digests.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from scripts.perf import regional_integrated_evidence as evidence
from scripts.perf import regional_integrated_workflow_capacity_suite as suite

ADMIN = "a" * 64
ROLE = {"ingress": "1" * 64, "worker": "2" * 64, "spool": "3" * 64}


def drain(queue: float | None, spool: float | None = 0.0) -> dict[str, Any]:
    return {
        "target_queue_depth": 1.0,
        "target_spool_depth": 1.0,
        "samples": [
            {
                "elapsed_seconds": 0.0,
                "queue_depth": 900.0,
                "telemetry_spool_depth": 40.0,
            },
            {
                "elapsed_seconds": 901.0,
                "queue_depth": queue,
                "telemetry_spool_depth": spool,
            },
        ],
    }


def test_a_queue_back_at_its_pre_run_depth_passes() -> None:
    assert evidence.drain_errors(drain(1.0, 1.0)) == [], "drained to the target"
    assert evidence.drain_errors(drain(0.0, None)) == [], (
        "a release without a spool depth family drains on the queue alone"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (drain(12.0), "processor queue did not drain to its pre-run depth: 12 > 1"),
        (drain(0.0, 7.0), "telemetry spool did not drain to its pre-run depth: 7 > 1"),
        (drain(None), "processor queue depth was unreadable after the load"),
        ({"samples": []}, "processor queue drain was not observed after the load"),
        (None, "processor queue drain was not observed after the load"),
    ],
)
def test_a_queue_or_spool_left_above_its_depth_is_named(
    value: dict[str, Any] | None, expected: str
) -> None:
    errors = evidence.drain_errors(value)

    assert len(errors) == 1 and errors[0].startswith(expected), errors


def site(
    *,
    admin: dict[str, str | None] | None = None,
    pods: dict[str, tuple[str, str | None]] | None = None,
) -> Any:
    """A fake control() over the three role Deployments and their Pods."""

    admin = admin or {}
    deployments = {
        "items": [
            {
                "metadata": {
                    "name": name,
                    "annotations": {
                        key: value
                        for key, value in {
                            evidence.ADMIN_CONFIG_ANNOTATION: admin.get(role, ADMIN)
                        }.items()
                        if value is not None
                    },
                },
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {evidence.ROLE_CONFIG_ANNOTATION: ROLE[role]}
                        }
                    }
                },
            }
            for role, name in evidence.ROLE_DEPLOYMENTS.items()
        ]
    }
    pods = (
        pods
        if pods is not None
        else {
            "gpu-fault-api-ha-a": ("gpu-fault-api-ha", ROLE["ingress"]),
            "gpu-fault-control-worker-a": ("gpu-fault-control-worker", ROLE["worker"]),
        }
    )
    pod_list = {
        "items": [
            {
                "metadata": {
                    "name": name,
                    "labels": {"app": app},
                    "annotations": {evidence.ROLE_CONFIG_ANNOTATION: digest},
                }
            }
            for name, (app, digest) in pods.items()
        ]
    }

    def control(*args: str, **_kwargs: Any) -> str:
        if args[:2] == ("get", "deployment"):
            assert list(args[2:5]) == list(evidence.ROLE_DEPLOYMENTS.values()), args
            return json.dumps(deployments)
        assert args[:2] == ("get", "pod"), f"read-only gets only: {args}"
        assert "--field-selector=status.phase=Running" in args, args
        return json.dumps(pod_list)

    return control


def test_the_round_records_one_admin_and_three_role_digests() -> None:
    identity = evidence.config_identity(site())

    assert identity["admin_config_sha256"] == ADMIN
    assert identity["role_config_sha256"] == ROLE
    assert identity["roles"]["ingress"]["running_pods"] == 1
    assert identity["roles"]["spool"]["running_pods"] == 0, "spool disabled: 0 Pods"
    assert evidence.config_errors(identity) == [], "one configuration, recorded"


def test_a_pod_from_an_older_replicaset_is_a_roll_in_flight() -> None:
    control = site(
        pods={
            "gpu-fault-api-ha-a": ("gpu-fault-api-ha", ROLE["ingress"]),
            "gpu-fault-api-ha-old": ("gpu-fault-api-ha", "9" * 64),
        }
    )

    identity = evidence.config_identity(control)

    assert identity["roles"]["ingress"]["pods_on_another_role_config"] == {
        "gpu-fault-api-ha-old": "9" * 64
    }
    assert evidence.config_errors(identity) == [
        "control-plane Pods run another role configuration than their Deployment "
        "(a roll in flight): gpu-fault-api-ha-old"
    ]


def test_missing_or_mixed_admin_digests_fail() -> None:
    missing = evidence.config_identity(site(admin={"worker": None}))
    mixed = evidence.config_identity(site(admin={"spool": "b" * 64}))

    assert evidence.config_errors(missing) == [
        "control-plane configuration digests are missing: worker admin_config_sha256"
    ]
    assert mixed["admin_config_sha256"] is None
    assert evidence.config_errors(mixed) == [
        "the role Deployments carry different AdminConfig digests"
    ]
    assert evidence.config_errors(None) == [
        "control-plane configuration digests were not recorded"
    ]


def answered(**tiers: dict[str, int]) -> dict[str, Any]:
    """The probe's final responses: the 32-cluster matrix answered 2xx."""

    counts = {"10": {"200": 1000}, "50": {"200": 1128}, "100": {"202": 128}}
    counts.update({tier.removeprefix("t"): value for tier, value in tiers.items()})
    return {"layout_mode": "detached", "response_status_counts": counts}


def test_every_request_answered_2xx_passes() -> None:
    assert evidence.final_response_errors(answered(), fault_requests=1000) == []


@pytest.mark.parametrize(
    ("latency", "expected"),
    [
        (
            answered(t50={"200": 1100, "422": 28}),
            ["processor final response 422 on 28 tier-50 requests"],
        ),
        (
            answered(t10={"200": 999, "500": 1}),
            ["processor final response 500 on 1 tier-10 requests"],
        ),
        (
            answered(t100={"none": 3}),
            ["processor final response none on 3 tier-100 requests"],
        ),
        (
            answered(t10={"200": 400}),
            [
                "only 400 of the 1000 fault-tier requests have a processor final "
                "response"
            ],
        ),
        (
            {"error": "command failed (1): exec"},
            ["processor final responses were not read: command failed (1): exec"],
        ),
        ({"layout_mode": "inline"}, ["processor final responses were not recorded"]),
        (None, ["processor final responses were not read: None"]),
    ],
)
def test_a_non_2xx_or_unread_final_response_fails(
    latency: dict[str, Any] | None, expected: list[str]
) -> None:
    """An ingress 202 only means enqueued; a dry run once found background
    422s behind 202 receipts (perf plan §13, item 8)."""

    assert evidence.final_response_errors(latency, fault_requests=1000) == expected


def test_the_integrated_verdict_fails_a_background_422() -> None:
    summary = passing_summary()
    summary["processor_priority_latency"] = answered(t50={"200": 1127, "422": 1})

    status, errors = suite.verdict(summary, expected_workflows=128)

    assert status == "FAIL", errors
    assert errors == ["processor final response 422 on 1 tier-50 requests"]


def passing_summary() -> dict[str, Any]:
    expected = 128
    return {
        "clusters": 32,
        "fixed_matrix": {**suite.FORMAL_MODELS[32], "p0": 1000, "p50": 1000},
        "ingress": {
            "requests": 26576,
            "job_status": "Complete",
            "dns_modes": ["process-cache"],
            "dns_resolution_attempts_max": 1,
            "dns_address_count_min": 2,
            "dns_cache_hits": 26576,
            "paths": {
                "NVIDIA_KERNEL": {"count": 500, "status_counts": {"200": 500}},
                "FABRIC_MANAGER_LOG": {"count": 500, "status_counts": {"200": 500}},
            },
        },
        "control_plane_pod_lifecycle": {"restart_count_delta": 0},
        "workflow": {
            "incident_count": expected,
            "context_incident_count": expected,
            "workflow_count": expected,
            "succeeded_workflow_count": expected,
            "terminal_command_count": 960,
            "command_count": 960,
            "simulated_command_count": 960,
            "step_count_min": 8,
            "step_count_max": 12,
            "operation_counts": {
                name: 1
                for name in (
                    "MARK_UNSCHEDULABLE",
                    "STOP_WORKLOADS",
                    "COLLECT_DIAGNOSTIC_BUNDLE",
                    "RESET_GPU",
                    "RESET_ALL_GPUS_NVSWITCHES",
                    "RESTART_NODE",
                    "RESTORE_SCHEDULING",
                )
            },
        },
        "executor": {"expected_commands": 960, "completed_commands": 960},
        "queue_drain": drain(0.0, 0.0),
        "config_identity": evidence.config_identity(site()),
        "processor_priority_latency": answered(),
    }


def test_the_integrated_verdict_fails_an_undrained_round() -> None:
    summary = passing_summary()
    assert suite.verdict(summary, expected_workflows=128) == ("PASS", [])

    undrained = copy.deepcopy(summary)
    undrained["queue_drain"] = drain(0.0, 300.0)
    status, errors = suite.verdict(undrained, expected_workflows=128)

    assert status == "FAIL", errors
    assert errors == [
        "telemetry spool did not drain to its pre-run depth: 300 > 1 after 901 s"
    ]


def test_the_integrated_verdict_fails_a_round_with_two_configurations() -> None:
    summary = passing_summary()
    summary["config_identity"] = evidence.config_identity(
        site(admin={"ingress": "c" * 64})
    )

    status, errors = suite.verdict(summary, expected_workflows=128)

    assert status == "FAIL", errors
    assert errors == ["the role Deployments carry different AdminConfig digests"]
