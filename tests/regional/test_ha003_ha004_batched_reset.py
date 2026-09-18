"""HA-003 and HA-004 must recognise RESET_GPU inside a round-trip batch.

Since protocol 3 the node command that resets a GPU heads with
QUIESCE_GPU_SERVICES and carries RESET_GPU in ``batched_steps``; HA-003
attempt 1 filtered on ``step.operation`` alone, saw no reset command at all
and failed a workflow that had SUCCEEDED.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004

_BATCHED = {
    "command_id": "remote-batch",
    "status": "WAITING",
    "step": {"operation": "QUIESCE_GPU_SERVICES"},
    "batched_steps": [
        {"step": {"operation": "VERIFY_NO_GPU_CLIENTS"}},
        {"step": {"operation": "RESET_GPU"}},
        {"step": {"operation": "RESTORE_GPU_SERVICES"}},
    ],
}
_MARK = {
    "command_id": "remote-mark",
    "status": "SUCCEEDED",
    "step": {"operation": "MARK_UNSCHEDULABLE"},
}


def test_the_reset_status_is_read_off_the_command_that_batches_it() -> None:
    state = {"commands": [_MARK, _BATCHED]}

    assert ha003.reset_command_status(state) == "WAITING", (
        "RESET_GPU inside batched_steps was not recognised"
    )
    assert ha003.reset_command_status({"commands": [_MARK]}) is None, (
        "a command without RESET_GPU counted as the reset"
    )


@pytest.fixture
def failover(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    waiting = {
        "step_index": 2,
        "operation": "RESET_GPU",
        "status": "WAITING",
        "adapter_operation_id": None,
        "details": {"mutation_submitted_by_control_plane": False},
        "error": None,
    }
    fixture = SimpleNamespace(
        settings=SimpleNamespace(node="node-a", rds_cluster_id="unit-aurora"),
        proof={"identity": "unit-aurora-binding"},
        state={
            "commands": [copy.deepcopy(_BATCHED)],
            "workflow": {"step_executions": [waiting]},
        },
        aurora_moved=False,
        deadline=datetime.now(timezone.utc) + timedelta(minutes=10),
        observed_after=datetime.now(timezone.utc),
        evidence=ha003.WaitingEvidence(),
        events=[],
        requests=[],
        observations=[],
    )

    def aurora(settings: Any) -> dict[str, Any]:
        fixture.events.append("aurora")
        assert settings is fixture.settings, (
            "the pre-failover Aurora check must describe the bound database"
        )
        writer = "unit-writer-moved" if fixture.aurora_moved else "unit-writer"
        return {"status": "available", "writer": writer}

    def snapshot(**kwargs: Any) -> dict[str, Any]:
        fixture.events.append("snapshot")
        assert kwargs == {
            "node": "node-a",
            "marker": "unit-marker",
            "observed_after": fixture.observed_after,
            "queue_attempts": 1,
        }, "failover must re-read the bound reset immediately before dispatch"
        return fixture.state

    fixture.regional = SimpleNamespace(store_snapshot=snapshot)

    def observe(state: dict[str, Any]) -> None:
        fixture.events.append("observe")
        fixture.observations.append(state)

    def provider(settings: Any, *args: str) -> dict[str, str]:
        assert settings is fixture.settings, "failover must retain the target settings"
        fixture.events.append("provider")
        fixture.requests.append(args)
        return {"status": "mock-requested"}

    monkeypatch.setattr(ha003, "rds_snapshot", aurora)
    monkeypatch.setattr(ha003, "aws_rds", provider)

    def request() -> tuple[datetime, dict[str, Any]]:
        return ha003.request_bound_failover(
            fixture.settings,
            fixture.regional,
            {"aurora_binding": fixture.proof, "rds": {"writer": "unit-writer"}},
            {"command_id": _BATCHED["command_id"]},
            fixture.evidence,
            marker="unit-marker",
            observed_after=fixture.observed_after,
            maintenance_window_end=fixture.deadline,
            observe_state=observe,
        )

    fixture.request = request
    return fixture


@pytest.mark.parametrize("batched", [False, True], ids=["single", "batched"])
@pytest.mark.parametrize("status", ["LEASED", "WAITING"])
def test_failover_rechecks_the_same_in_flight_reset(
    failover: SimpleNamespace, batched: bool, status: str
) -> None:
    command = failover.state["commands"][0]
    command["status"] = status
    if not batched:
        command["step"] = {"operation": "RESET_GPU"}
        command.pop("batched_steps")

    requested_at, result = failover.request()

    assert result == {"status": "mock-requested"}, (
        "a uniquely in-flight reset must reach the mocked provider boundary"
    )
    assert requested_at < failover.deadline, (
        "failover must be requested within the maintenance window"
    )
    assert failover.events == ["aurora", "snapshot", "observe", "provider"], (
        "the Aurora writer check and fresh reset evidence must precede the single "
        "provider request"
    )
    assert failover.requests == [
        ("failover-db-cluster", "--db-cluster-identifier", "unit-aurora")
    ], "failover must target only the preflight-bound database"
    assert failover.observations == [failover.state], (
        "the fresh snapshot must remain available to cleanup identity tracking"
    )
    assert (
        failover.evidence.merged_into({})["observed_waiting_step_executions"]
        == failover.state["workflow"]["step_executions"]
    ), "the final dispatch check must preserve its observed WAITING evidence"


@pytest.mark.parametrize(
    "commands",
    [
        [],
        [_MARK],
        [{**_BATCHED, "status": "PENDING"}],
        [{**_BATCHED, "status": "SUCCEEDED"}],
        [{**_BATCHED, "status": "UNKNOWN"}],
        [{**_BATCHED, "command_id": "remote-other"}],
        [_BATCHED, {**_BATCHED, "command_id": "remote-other"}],
    ],
    ids=[
        "missing",
        "unrelated",
        "pending",
        "terminal",
        "unknown",
        "replaced",
        "duplicate",
    ],
)
def test_failover_refuses_unproved_reset_identity_or_state(
    failover: SimpleNamespace, commands: list[dict[str, Any]]
) -> None:
    failover.state["commands"] = copy.deepcopy(commands)
    with pytest.raises(ha003.RegionalFixtureError, match="uniquely in flight"):
        failover.request()
    assert failover.requests == [], "unproved reset state must not trigger failover"
    assert failover.events == ["aurora", "snapshot", "observe"], (
        "a refused dispatch must retain its Aurora check and cleanup observation"
    )


@pytest.mark.parametrize("failure", ["aurora", "deadline"])
def test_batched_reset_does_not_bypass_aurora_or_window(
    failover: SimpleNamespace, failure: str
) -> None:
    failover.aurora_moved = failure == "aurora"
    if failure == "deadline":
        failover.deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
    with pytest.raises(
        ha003.RegionalFixtureError, match="Aurora moved|maintenance window ended"
    ):
        failover.request()
    assert failover.requests == [], (
        "Aurora/window refusal must precede provider mutation"
    )
    assert failover.events == (
        ["aurora"] if failure == "aurora" else ["aurora", "snapshot", "observe"]
    ), (
        "a moved writer must stop before store sampling; expiry must still retain "
        "observations"
    )


def test_failover_request_is_one_describe_call_and_stamps_its_timing(
    failover: SimpleNamespace,
) -> None:
    """The final barrier costs one RDS describe plus the store re-read; the
    full binding re-read (~40 s live) is what made a1 on 2026-09-18
    INCONCLUSIVE. The stamps let a live run show where the window went."""
    timing: dict[str, Any] = {}
    requested_at, _ = ha003.request_bound_failover(
        failover.settings,
        failover.regional,
        {"aurora_binding": failover.proof, "rds": {"writer": "unit-writer"}},
        {"command_id": _BATCHED["command_id"]},
        failover.evidence,
        marker="unit-marker",
        observed_after=failover.observed_after,
        maintenance_window_end=failover.deadline,
        observe_state=lambda state: None,
        timing=timing,
    )
    assert list(timing) == [
        "recheck_started_at",
        "aurora_checked_at",
        "store_snapshot_at",
        "requested_at",
    ]
    assert timing["requested_at"] == requested_at.isoformat()
    assert failover.events.count("aurora") == 1
    source = Path(ha003.__file__).read_text(encoding="utf-8")
    assert "def request_bound_failover" in source
    body = source.split("def request_bound_failover", 1)[1].split("\ndef ", 1)[0]
    assert "regional_binding(" not in body, (
        "the pre-failover barrier must not re-read the full Aurora binding"
    )


def test_reclaim_timeline_tracks_the_batched_reset_and_only_its_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [
        {
            "commands": [
                copy.deepcopy(_MARK),
                {
                    **copy.deepcopy(_BATCHED),
                    "status": status,
                    "last_lease_owner": owner,
                },
            ]
        }
        for status, owner in (
            ("PENDING", None),
            ("WAITING", "cluster/pod-a"),
            ("WAITING", "cluster/pod-b"),
            ("SUCCEEDED", "cluster/pod-b"),
        )
    ]
    snapshots = iter(states)
    killed: list[str] = []
    observed: list[dict[str, Any]] = []
    clock = SimpleNamespace(elapsed=0.0)
    monkeypatch.setattr(
        ha004,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.elapsed,
            sleep=lambda seconds: setattr(clock, "elapsed", clock.elapsed + seconds),
        ),
    )
    state, timeline = ha004.command_timeline(
        SimpleNamespace(store_snapshot=lambda **_kwargs: next(snapshots)),
        SimpleNamespace(node="node-a"),
        marker="unit-marker",
        observed_after=datetime.now(timezone.utc),
        timeout_seconds=2,
        kill_owner=killed.append,
        observe_state=observed.append,
    )
    assert killed == ["cluster/pod-a"], (
        "only the first in-flight reset owner may reach the mocked deletion boundary"
    )
    assert observed == states, (
        "the whole reclaim timeline must retain cleanup observations"
    )
    assert [item["command_id"] for item in timeline] == ["remote-batch"] * 4, (
        "head-step filtering must not lose the batched reset command"
    )
    assert [item["status"] for item in timeline] == [
        "PENDING",
        "WAITING",
        "WAITING",
        "SUCCEEDED",
    ], "the reclaim timeline must preserve the observed command transitions"
    assert state["ha004_first_owner"] == "cluster/pod-a", (
        "the final state must retain the owner actually removed"
    )
    assert ha004.lease_reissue_observed(timeline, first_owner="cluster/pod-a"), (
        "the same batched command must show a post-deletion owner change"
    )


def test_the_env_window_only_assigns_what_the_live_executor_does_not_already_carry() -> (
    None
):
    """The shipped poll interval is 2 s, the case's own value; assigning it made
    executor_env_window read the live env as an unrecorded open and refuse."""

    live = {
        ha004.LEASE_ENV: {"present": True, "value": "120"},
        ha004.POLL_ENV: {"present": True, "value": "2"},
    }
    assert ha004.window_assignments(live) == {ha004.LEASE_ENV: "10"}, (
        "poll=2 was re-assigned"
    )
    assert ha004.window_assignments({}) == {
        ha004.LEASE_ENV: "10",
        ha004.POLL_ENV: "2",
    }, "absent variables must be assigned"
    assert ha004.window_assignments(
        {ha004.LEASE_ENV: {"present": True, "value": "10"}}
    ) == {ha004.POLL_ENV: "2"}, (
        "a lease already at the test value must be left to the window's own refusal"
    )
