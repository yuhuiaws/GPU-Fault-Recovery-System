from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004
from tests.regional._cov95_ha_reset_harness import ResetHarness


def test_reclaim_helpers_require_timestamps_owner_and_stable_command_identity() -> None:
    assert (
        ha004.lease_token_digest({"lease_token_sha256": "unit-digest"}) == "unit-digest"
    )
    assert (
        ha004.lease_token_digest({"lease_token": "unit-test-lease"})
        == hashlib.sha256(b"unit-test-lease").hexdigest()
    )
    assert (
        ha004.effective_sampling_period_seconds(
            [{}, {"observed_at": "2026-01-01T00:00:00Z"}]
        )
        is None
    )
    killed = {
        "command_id": "unit",
        "status": "WAITING",
        "last_lease_owner": "old",
        "killed_owner": "old",
        "kill_completed_at": "2026-01-01T00:00:00Z",
    }
    assert ha004.lease_reissue_observed([], first_owner="old") is False
    assert (
        ha004.lease_reissue_observed(
            [killed, {**killed, "command_id": "foreign", "killed_owner": None}],
            first_owner="old",
        )
        is False
    )
    assert (
        ha004.lease_reissue_observed(
            [killed, {**killed, "killed_owner": None}], first_owner="old"
        )
        is False
    )


@pytest.mark.parametrize("failure", ["owner", "timeout"])
def test_reclaim_timeline_refuses_unknown_owner_or_never_terminal_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    harness = ResetHarness(monkeypatch, tmp_path, ha004)
    state = copy.deepcopy(harness.api.final)
    state["commands"][4].update(
        status="WAITING" if failure == "owner" else "PENDING",
        lease_owner=None,
        last_lease_owner=None,
    )
    harness.api.states = [state]
    kills = []
    with pytest.raises(RuntimeError, match="no owner|did not reach a terminal"):
        ha004.command_timeline(
            harness.api,
            harness.settings,
            marker="unit",
            observed_after=datetime.now(timezone.utc),
            timeout_seconds=0.1,
            kill_owner=kills.append,
        )
    assert kills == []


def test_final_reclaim_requires_a_second_owner_after_the_kill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = ResetHarness(monkeypatch, tmp_path, ha004)
    for state in harness.api.states:
        state["commands"][4]["last_lease_owner"] = "unit/pod-0"
    code, report = harness.execute()
    assert code == 1
    assert "a second executor did not reclaim the command" in report["errors"]
    assert (
        "no second lease was observed after the first owner was removed"
        in report["errors"]
    )
    assert report["cleanup"]["errors"] == []


def test_executor_environment_references_are_not_treated_as_restorable_literals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = ResetHarness(monkeypatch, tmp_path, ha004)
    harness.api.document["spec"]["template"]["spec"]["containers"][0]["env"][0] = {
        "name": ha004.LEASE_ENV,
        "valueFrom": {"secretKeyRef": {"name": "unit", "key": "lease"}},
    }
    with pytest.raises(RuntimeError, match="valueFrom"):
        ha004.deployment_snapshot(harness.api)


def test_failed_reclaim_predecessor_blocks_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = ResetHarness(monkeypatch, tmp_path, ha004)
    harness.settings.predecessor_path.write_text("{}")
    with pytest.raises(RuntimeError, match="predecessor evidence"):
        harness.execute()
    assert "watchdog-start" not in harness.events
    assert "host-create" not in harness.events


@pytest.mark.parametrize(
    "module,phase",
    [(ha003, "injection"), (ha003, "failover"), (ha004, "injection"), (ha004, "owner")],
)
def test_window_expiry_during_observation_stops_the_next_real_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: Any, phase: str
) -> None:
    harness = ResetHarness(monkeypatch, tmp_path, module)
    expired = False
    deadline = harness.deadline

    def now(tz: Any = None) -> datetime:
        return (
            deadline + timedelta(seconds=1) if expired else datetime.now(timezone.utc)
        )

    monkeypatch.setattr(module, "datetime", SimpleNamespace(now=now))
    if phase == "injection":
        execute = harness.host.execute

        def host_execute(operation: str, *args: str, **kwargs: Any) -> dict:
            nonlocal expired
            value = execute(operation, *args, **kwargs)
            if operation == "start-reset-sampler":
                expired = True
            return value

        monkeypatch.setattr(harness.host, "execute", host_execute)
    elif phase == "failover":
        binding = harness.binding

        def read(expected: dict | None = None) -> dict:
            nonlocal expired
            value = binding(expected)
            if harness.binding_reads == 3:
                expired = True
            return value

        monkeypatch.setattr(
            module, "regional_binding", lambda *a: SimpleNamespace(read=read)
        )
    else:
        snapshot = harness.api.store_snapshot

        def store_snapshot(**kwargs: Any) -> dict:
            nonlocal expired
            value = snapshot(**kwargs)
            if harness.api.state_reads == 2:
                expired = True
            return value

        monkeypatch.setattr(harness.api, "store_snapshot", store_snapshot)
    code, report = harness.execute()
    assert code == 1
    assert "maintenance window ended" in report["error"]
    assert "failover-request" not in harness.events
    assert "delete-owner" not in harness.events
    assert report["cleanup"]["errors"] == []
