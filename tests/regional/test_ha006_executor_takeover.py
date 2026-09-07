"""HA-006: two-round probe adapter, summed physical counter, store-timed takeover."""

from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import run_ha006_executor_takeover as ha006
from scripts.e2e.regional.probes import ha006_executor


def _context(key: str = "workflow/0/RUN_DCGM_DIAGNOSTIC"):
    return SimpleNamespace(
        idempotency_key=key,
        step=SimpleNamespace(
            operation=SimpleNamespace(value="RUN_DCGM_DIAGNOSTIC"),
            execution_owner="gpu-fault-ha006-test",
        ),
        incident=SimpleNamespace(cluster_id="cluster-a", incident_id="incident-a"),
    )


class _Registry:
    def __init__(self) -> None:
        self.notification_id: str | None = None

    def save_notification_if_absent(self, candidate):
        if self.notification_id is None:
            self.notification_id = candidate.notification_id
            return candidate
        return candidate.model_copy(update={"notification_id": self.notification_id})


def test_probe_adapter_waits_on_the_first_round_and_counts_one_physical_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ha006_executor, "WINNER", tmp_path / "winner.json")
    monkeypatch.setattr(ha006_executor, "PHYSICAL", tmp_path / "physical.json")
    monkeypatch.setattr(socket, "gethostname", lambda: "pod-a")
    registry = _Registry()
    adapter = ha006_executor.SharedLedgerAdapter(
        registry, run_id="run-a", sleep_seconds=0, wait_first_round=True
    )

    first = adapter.execute(_context())
    assert first.status.value == "WAITING"
    assert first.details["round"] == 1
    assert adapter.physical_actions == 0
    assert not (tmp_path / "physical.json").exists(), (
        "the first WAITING round must not touch the physical action file"
    )

    second = adapter.execute(_context())
    assert second.status.value == "SUCCEEDED"
    assert second.details["cached"] is False
    assert second.details["physical_count"] == 1
    assert json.loads((tmp_path / "physical.json").read_text())["physical_actions"] == 1

    monkeypatch.setattr(socket, "gethostname", lambda: "pod-b")
    loser = ha006_executor.SharedLedgerAdapter(
        registry, run_id="run-a", sleep_seconds=0
    )
    replay = loser.execute(_context())
    assert replay.details["cached"] is True
    assert replay.details["physical_count"] == 0, (
        "a loser has no physical action of its own"
    )
    assert (
        ha006.physical_action_total(
            {"physical_actions": adapter.physical_actions},
            {"physical_actions": loser.physical_actions},
        )
        == 1
    )


def test_remaining_lease_is_none_when_the_kill_missed_the_lease() -> None:
    killed_at = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
    assert ha006.remaining_lease_seconds({"lease_expires_at": None}, killed_at) is None
    assert ha006.remaining_lease_seconds({}, killed_at) is None
    expiry = (killed_at + timedelta(seconds=12)).isoformat()
    assert (
        ha006.remaining_lease_seconds({"lease_expires_at": expiry}, killed_at) == 12.0
    )


def test_takeover_timing_prefers_store_updated_at_and_widens_tolerance_when_sampled() -> (
    None
):
    killed_at = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
    observed = killed_at + timedelta(seconds=31.4)
    from_store = ha006.takeover_timing(
        {"updated_at": (killed_at + timedelta(seconds=31)).isoformat()},
        killed_at=killed_at,
        observed_completed_at=observed,
    )
    assert from_store["completion_source"] == "store.updated_at"
    assert from_store["takeover_seconds"] == 31.0
    assert (
        from_store["tolerance_seconds"]
        == ha006.POLL_SECONDS + ha006.CLAIM_BACKOFF_MAX_SECONDS
    )

    sampled = ha006.takeover_timing(
        {}, killed_at=killed_at, observed_completed_at=observed
    )
    assert sampled["completion_source"] == "sampled"
    assert sampled["takeover_seconds"] == 31.4
    assert sampled["tolerance_seconds"] == (
        ha006.POLL_SECONDS
        + ha006.CLAIM_BACKOFF_MAX_SECONDS
        + ha006.SAMPLE_INTERVAL_SECONDS
    )


def test_waiting_branch_is_read_from_the_pre_kill_timeline() -> None:
    timeline = [
        {
            "status": "PENDING",
            "lease_owner": None,
            "last_lease_owner": None,
            "round": None,
        },
        {
            "status": "WAITING",
            "lease_owner": None,
            "last_lease_owner": "gpu-fault-ha006-a",
            "round": 1,
        },
        {
            "status": "LEASED",
            "lease_owner": "gpu-fault-ha006-b",
            "last_lease_owner": "gpu-fault-ha006-b",
            "round": None,
        },
    ]
    branch = ha006.waiting_branch(timeline)
    assert branch["observed"] is True
    assert branch["waiting_owner"] == "gpu-fault-ha006-a"
    assert branch["second_round_owner"] == "gpu-fault-ha006-b"
    assert branch["reclaimed_by_other_replica"] is True

    unseen = ha006.waiting_branch([timeline[0], timeline[2]])
    assert unseen["observed"] is False
    assert unseen["reclaimed_by_other_replica"] is False


def test_takeover_errors_judge_the_summed_counter_not_a_constant() -> None:
    final = {
        "last_lease_owner": "gpu-fault-ha006-b",
        "updated_at": "2026-09-07T12:00:31+00:00",
        "result_details": {
            "cached": True,
            "physical_count": 0,
            "shared_notification_id": "n-1",
        },
    }
    timing = {"takeover_seconds": 31.0, "tolerance_seconds": 5.0}
    notification = {
        "dedup_link_count": 1,
        "objects": {
            "notification": {"count": 1},
            "notification_delivery": {"count": 1},
            "notification_result": {"count": 1, "status": "SKIPPED"},
        },
    }
    base = dict(
        final=final,
        survivor="gpu-fault-ha006-b",
        notification_id="n-1",
        timing=timing,
        remaining_lease=28.0,
        survivor_state={"claimed_total": 1, "unexpected_failures": 0},
        notification_final=notification,
    )
    assert ha006.takeover_errors(physical_total=1, **base) == []
    errors = ha006.takeover_errors(physical_total=2, **base)
    assert any("sum to 2, not 1" in item for item in errors), errors
    slow = ha006.takeover_errors(
        physical_total=1,
        **{**base, "timing": {"takeover_seconds": 34.0, "tolerance_seconds": 5.0}},
    )
    assert any("poll/backoff tolerance" in item for item in slow), slow


def test_pod_manifest_carries_poll_backoff_and_waiting_round_settings() -> None:
    manifest = ha006.pod_manifest(
        "gpu-fault-ha006-a",
        "image@sha256:x",
        {"executor_artifact_sha256": "a", "executor_compatibility_digest": "b"},
        "run-a",
    )
    env = {
        item["name"]: item.get("value")
        for item in manifest["spec"]["containers"][0]["env"]
    }
    assert env["POLL_SECONDS"] == str(ha006.POLL_SECONDS)
    assert env["CLAIM_BACKOFF_MAX_SECONDS"] == str(ha006.CLAIM_BACKOFF_MAX_SECONDS)
    assert env["WAIT_FIRST_ROUND"] == "true"
