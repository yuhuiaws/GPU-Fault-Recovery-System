from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from scripts.e2e.regional import workload_acceptance_checks as checks
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional._identity_lifecycle_support import recovery_state


def metadata() -> dict[str, Any]:
    labels = {
        "gpu-fault.io/managed": "true",
        "gpu-fault.io/job-id": "job",
        "gpu-fault.io/attempt-id": "attempt",
    }
    replicas = {
        role: {
            "template": {
                "metadata": {
                    "labels": {
                        **labels,
                        "gpu-fault.io/critical": "true",
                        "gpu-fault.io/role": role.lower(),
                    },
                    "annotations": {
                        "gpu-fault.io/expected-critical-ranks": "3",
                        "gpu-fault.io/rank-offset": str(index),
                        "gpu-fault.io/runtime-profile-version": "profile",
                        "gpu-fault.io/restart-budget": "1",
                        "gpu-fault.io/training-container": "pytorch",
                    },
                }
            }
        }
        for index, role in enumerate(("Master", "Worker"))
    }
    return {"metadata": {"labels": labels}, "spec": {"pytorchReplicaSpecs": replicas}}


@pytest.mark.parametrize(
    "field",
    [
        "none",
        "workload-label",
        "role",
        "critical",
        "expected-critical-ranks",
        "rank-offset",
        "runtime-profile-version",
        "restart-budget",
        "training-container",
    ],
)
def test_training_metadata_requires_all_watcher_identity_fields(field: str) -> None:
    document = metadata()
    template = document["spec"]["pytorchReplicaSpecs"]["Worker"]["template"]["metadata"]
    if field == "workload-label":
        document["metadata"]["labels"]["gpu-fault.io/job-id"] = "foreign"
    elif field in {"role", "critical"}:
        template["labels"]["gpu-fault.io/" + field] = "wrong"
    elif field != "none":
        template["annotations"]["gpu-fault.io/" + field] = "wrong"
    errors = checks.metadata_errors(
        document, job_id="job", attempt_id="attempt", profile_version="profile"
    )
    assert bool(errors) is (field != "none")
    if field != "none":
        assert len(errors) == 1, errors


@pytest.mark.parametrize(
    ("logs", "message"),
    [
        ({"p": "rank=0 step=0 loss=not-a-number"}, "unparseable loss"),
        ({"p": ""}, "no loss lines"),
        ({"p": "rank=0 step=0 loss=1\nrank=0 step=1 loss=2"}, "loss increased"),
        (
            {
                "p": "rank=0 step=0 loss=2\nrank=0 step=1 loss=1",
                "q": "rank=0 step=0 loss=2\nrank=0 step=1 loss=1",
            },
            "more than one Pod",
        ),
    ],
)
def test_loss_proof_cannot_hide_missing_invalid_or_duplicate_rank_series(
    logs: dict[str, str], message: str
) -> None:
    assert any(
        message in error for error in checks.loss_errors(logs, world_size=1, steps=2)
    ), f"invalid training series did not report {message}"


@pytest.mark.parametrize("log", ["", "SUCCESS rank=0/1 all_reduce=invalid"])
def test_collective_proof_requires_parseable_success(log: str) -> None:
    assert checks.collective_errors({"p": log}, world_size=1), (
        "missing or malformed collective evidence must fail"
    )


def virtual_states() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = [
        {"pods": [{"uid": name + "-uid", "name": name + "-pod"}]} for name in ("a", "b")
    ]
    states = [
        {
            "cluster_id": name,
            "decision": None,
            "restart_budget": None,
            "observations": [
                {
                    "cluster_id": name,
                    "containers": [
                        {
                            "pod_uid": name + "-uid",
                            "pod_name": name + "-pod",
                            "node_id": "shared",
                            "gpu_uuids": ["GPU-" + name],
                        }
                    ],
                }
            ],
        }
        for name in ("a", "b")
    ]
    return states, sources


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "count",
        "empty",
        "store",
        "cluster",
        "node",
        "name",
        "decision",
        "budget",
        "same-gpu",
    ],
)
def test_virtual_collision_proof_keeps_both_physical_ownership_sets(
    defect: str,
) -> None:
    states, sources = virtual_states()
    b = states[1]
    if defect == "count":
        states.pop()
    elif defect == "empty":
        b["observations"] = []
    elif defect == "store":
        b["cluster_id"] = "a"
    elif defect == "cluster":
        b["observations"][0]["cluster_id"] = "a"
    elif defect in {"node", "name", "same-gpu"}:
        container = b["observations"][0]["containers"][0]
        key, value = {
            "node": ("node_id", "foreign"),
            "name": ("pod_name", "a-pod"),
            "same-gpu": ("gpu_uuids", ["GPU-a"]),
        }[defect]
        container[key] = value
    elif defect in {"decision", "budget"}:
        b["decision" if defect == "decision" else "restart_budget"] = {}
    errors = checks.virtual_isolation_errors(
        states, sources=sources, cluster_ids=["a", "b"], shared_node="shared"
    )
    assert bool(errors) is (defect != "none"), errors


@pytest.mark.parametrize(
    "defect", ["none", "missing", "invalid", "naive", "stale", "future", "uid"]
)
def test_source_observation_freshness_and_uid_binding_are_both_required(
    defect: str,
) -> None:
    now = datetime.now(timezone.utc)
    source = {"pods": [{"uid": "uid", "name": "pod", "node": "node"}]}
    observation = {
        "cluster_id": "a",
        "job_id": "job",
        "attempt_id": "attempt",
        "workload_phase": "RUNNING",
        "workload_ids": ["training/job/job"],
        "observed_at": now.isoformat(),
        "containers": [{"pod_uid": "uid", "pod_name": "pod", "node_id": "node"}],
    }
    if defect == "missing":
        observation.pop("observed_at")
    elif defect == "invalid":
        observation["observed_at"] = "invalid"
    elif defect == "naive":
        observation["observed_at"] = now.replace(tzinfo=None).isoformat()
    elif defect in {"stale", "future"}:
        observation["observed_at"] = (
            now + timedelta(seconds=-121 if defect == "stale" else 30)
        ).isoformat()
    elif defect == "uid":
        observation["containers"][0]["pod_uid"] = "foreign"
    assert bool(
        checks.source_observation_errors(
            observation, source, cluster_id="a", job_id="job", attempt_id="attempt"
        )
    ) is (defect != "none")


@pytest.mark.parametrize(
    "defect",
    [
        "missing-time",
        "old-time",
        "future-time",
        "step-index",
        "operation",
        "duplicate-step",
    ],
)
def test_causal_recovery_rejects_unknown_time_or_ambiguous_command_steps(
    defect: str,
) -> None:
    now = datetime.now(timezone.utc)
    state = recovery_state("a", "node-a", "unit")
    if defect == "missing-time":
        state["event"].pop("observed_at")
        message = "time is unknown"
    elif defect in {"old-time", "future-time"}:
        state["event"]["observed_at"] = (
            now + timedelta(seconds=-100 if defect == "old-time" else 100)
        ).isoformat()
        message = "outside this injection window"
    elif defect in {"step-index", "operation"}:
        command = state["commands"][0]
        if defect == "step-index":
            command["step_index"] = None
        else:
            command["step"]["operation"] = None
        message = "step identity is missing"
    else:
        original = state["commands"][0]
        duplicate = copy.deepcopy(original)
        duplicate["command_id"] += "-duplicate"
        state["commands"].append(duplicate)
        message = "cover the same workflow step"
    errors = checks.recovery_identity_errors(
        state,
        cluster_id="a",
        job_id="job-test",
        attempt_id="attempt-test",
        node="node-a",
        marker="unit",
        observed_after=now - timedelta(seconds=1),
    )
    assert any(message in error for error in errors), errors


@pytest.mark.parametrize(
    "defect", ["missing-id", "duplicate-id", "count", "status", "receipt"]
)
def test_notification_completion_needs_unique_sent_incident_bound_records(
    defect: str,
) -> None:
    state = recovery_state("a", "node-a", "unit")
    first = state["notifications"][0]
    if defect == "missing-id":
        first["notification"].pop("notification_id")
    elif defect == "duplicate-id":
        first["notification"]["notification_id"] = state["notifications"][1][
            "notification"
        ]["notification_id"]
    elif defect == "count":
        state["notifications"].pop()
    elif defect == "status":
        first["result"]["status"] = "PENDING"
    else:
        first["result"].pop("provider_message_id")
    assert checks.notification_errors(state), (
        "invalid delivery evidence must not prove recovery notification completion"
    )


def pending_completion_state(result: dict[str, Any] | None) -> dict[str, Any]:
    """A terminal-edge snapshot: ACTION_COMPLETED exists, its result does not."""

    state = recovery_state("a", "node-a", "unit")
    state["notifications"][1]["result"] = result
    return state


@pytest.mark.parametrize(
    "result",
    [
        None,
        {"status": "PENDING", "provider_message_id": None},
        {"status": "LEASED", "provider_message_id": None},
        {"status": "RETRY", "provider_message_id": None},
    ],
)
def test_notification_wait_re_reads_the_store_while_delivery_is_in_flight(
    result: dict[str, Any] | None,
) -> None:
    state = pending_completion_state(result)
    settled = recovery_state("a", "node-a", "unit")
    sleeps: list[float] = []
    reads: list[str] = []

    def snapshot() -> dict[str, Any]:
        reads.append("store")
        return settled

    errors = checks.wait_for_notification_results(
        state, snapshot=snapshot, sleep=sleeps.append, monotonic=lambda: 0.0
    )
    assert errors == []
    assert sleeps == [checks.NOTIFICATION_DELIVERY_POLL_SECONDS] and reads == ["store"]
    assert state["notifications"] == settled["notifications"], (
        "the judged reading must be the refreshed one"
    )


@pytest.mark.parametrize("defect", ["DEAD", "FAILED", "sent-without-id", "missing"])
def test_notification_wait_stops_at_once_on_a_defect_no_later_read_cures(
    defect: str,
) -> None:
    state = pending_completion_state(None)
    if defect == "missing":
        state["notifications"].pop()
    elif defect == "sent-without-id":
        state["notifications"][1]["result"] = {"status": "SENT"}
    else:
        state["notifications"][1]["result"] = {
            "status": defect,
            "provider_message_id": None,
        }
    before = copy.deepcopy(state["notifications"])

    def refuse(*args: Any) -> Any:
        raise AssertionError(f"no store read or sleep is allowed: {args}")

    errors = checks.wait_for_notification_results(
        state, snapshot=refuse, sleep=refuse, monotonic=lambda: 0.0
    )
    assert any("ACTION_COMPLETED" in error for error in errors), errors
    assert state["notifications"] == before


def test_notification_wait_returns_the_pending_errors_at_the_deadline() -> None:
    state = pending_completion_state(None)
    clock = {"now": 0.0}
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    errors = checks.wait_for_notification_results(
        state,
        snapshot=lambda: pending_completion_state(None),
        timeout_seconds=12,
        poll_seconds=5,
        sleep=sleep,
        monotonic=lambda: clock["now"],
    )
    assert errors == [
        "ACTION_COMPLETED notification is not SENT",
        "ACTION_COMPLETED notification has no provider message ID",
    ]
    assert sleeps == [5, 5, 5], "the loop stops at the first poll past the deadline"
    assert checks.NOTIFICATION_DELIVERY_WAIT_SECONDS == 120
