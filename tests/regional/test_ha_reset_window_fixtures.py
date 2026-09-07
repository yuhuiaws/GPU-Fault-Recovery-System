"""HA-003/HA-004 fixtures: catching the in-flight reset window and keeping its evidence.

Split out of test_destructive_acceptance_fixtures.py, which had reached the
architecture size limit; these tests share only the regional fixture helper
with it.
"""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.e2e.regional import executor_env_window as env_window
from scripts.e2e.regional import regional_live_fixture as live_fixture_module
from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
)


def _regional(tmp_path: Path) -> RegionalLiveFixture:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    return RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=cpu,
            gpu_kubeconfig=gpu,
            gpu_context="gpu-context",
            namespace="gpu-fault-system",
            cluster_id="cluster-a",
            region="us-west-2",
        )
    )


def _ha003_settings(tmp_path: Path) -> ha003.Settings:
    return ha003.Settings(
        regional=_regional(tmp_path).settings,
        node="node-a",
        host_probe_image="registry.example/probe@sha256:" + "a" * 64,
        rds_cluster_id="aurora-a",
        predecessor_path=tmp_path / "predecessor.json",
    )


def test_ha003_waits_for_reset_claim_before_failover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    snapshots = iter(
        [
            {"commands": [{"step": {"operation": "RESET_GPU"}, "status": "PENDING"}]},
            {
                "commands": [
                    {
                        "command_id": "remote-a",
                        "step": {"operation": "RESET_GPU"},
                        "status": "LEASED",
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(regional, "store_snapshot", lambda **_kwargs: next(snapshots))
    state, command = ha003.wait_reset_claim(
        regional,
        _ha003_settings(tmp_path),
        marker="marker-a",
        observed_after=datetime.now(timezone.utc),
        timeout_seconds=5,
    )

    assert command["command_id"] == "remote-a", command
    assert state["commands"][0]["status"] == "LEASED", state


def test_ha003_treats_a_waiting_reset_as_the_in_flight_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The executor reports WAITING while the Node Agent resets, so that is the window.

    LEASED lasts only for the claim round trip; a sampler that needs seconds
    per store read would otherwise watch a 45s reset go straight from PENDING
    to SUCCEEDED and wrongly conclude the window was missed.
    """

    regional = _regional(tmp_path)
    snapshots = iter(
        [
            {"commands": [{"step": {"operation": "RESET_GPU"}, "status": "PENDING"}]},
            {
                "commands": [
                    {
                        "command_id": "remote-a",
                        "step": {"operation": "RESET_GPU"},
                        "status": "WAITING",
                        "result_details": {"node_action_state": "PENDING"},
                    }
                ]
            },
            {
                "commands": [
                    {
                        "command_id": "remote-a",
                        "step": {"operation": "RESET_GPU"},
                        "status": "SUCCEEDED",
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(regional, "store_snapshot", lambda **_kwargs: next(snapshots))
    state, command = ha003.wait_reset_claim(
        regional,
        _ha003_settings(tmp_path),
        marker="marker-a",
        observed_after=datetime.now(timezone.utc),
        timeout_seconds=5,
    )

    assert command["command_id"] == "remote-a", command
    assert state["commands"][0]["status"] == "WAITING", state


def test_ha003_fails_when_the_reset_is_terminal_before_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    monkeypatch.setattr(
        regional,
        "store_snapshot",
        lambda **_kwargs: {
            "commands": [
                {
                    "command_id": "remote-a",
                    "step": {"operation": "RESET_GPU"},
                    "status": "SUCCEEDED",
                }
            ]
        },
    )
    with pytest.raises(
        live_fixture_module.RegionalFixtureError, match="completed before"
    ):
        ha003.wait_reset_claim(
            regional,
            _ha003_settings(tmp_path),
            marker="marker-a",
            observed_after=datetime.now(timezone.utc),
            timeout_seconds=5,
        )


def test_ha003_keeps_the_waiting_records_it_saw_before_the_workflow_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Steps that finish before `wait_for_workflow` starts still count as observed.

    The failover run polls the store itself first, and by the time it hands
    over to `wait_for_workflow` the early steps only have their SUCCEEDED
    record left. The WAITING records seen on the way have to survive into the
    evaluated state, or the evidence check reports steps that did happen as
    missing.
    """

    regional = _regional(tmp_path)
    waiting = {
        "step_index": 2,
        "operation": "QUIESCE_GPU_SERVICES",
        "status": "WAITING",
        "details": {"mutation_submitted_by_control_plane": False},
    }
    reset_waiting = {
        "step_index": 4,
        "operation": "RESET_GPU",
        "status": "WAITING",
        "details": {"mutation_submitted_by_control_plane": False},
    }
    snapshots = iter(
        [
            {
                "workflow": {"step_executions": [waiting]},
                "commands": [{"step": {"operation": "RESET_GPU"}, "status": "PENDING"}],
            },
            {
                "workflow": {"step_executions": [reset_waiting]},
                "commands": [
                    {
                        "command_id": "remote-a",
                        "step": {"operation": "RESET_GPU"},
                        "status": "WAITING",
                    }
                ],
            },
        ]
    )
    monkeypatch.setattr(regional, "store_snapshot", lambda **_kwargs: next(snapshots))
    evidence = live_fixture_module.WaitingEvidence()
    ha003.wait_reset_claim(
        regional,
        _ha003_settings(tmp_path),
        marker="marker-a",
        observed_after=datetime.now(timezone.utc),
        timeout_seconds=5,
        evidence=evidence,
    )

    later_state = {
        "workflow": {"status": "SUCCEEDED"},
        "observed_waiting_step_executions": [
            {
                "step_index": 5,
                "operation": "RESTORE_GPU_SERVICES",
                "status": "WAITING",
                "details": {"mutation_submitted_by_control_plane": False},
            }
        ],
    }
    merged = evidence.merged_into(later_state)

    assert [
        item["operation"] for item in merged["observed_waiting_step_executions"]
    ] == ["QUIESCE_GPU_SERVICES", "RESET_GPU", "RESTORE_GPU_SERVICES"], merged
    assert later_state["observed_waiting_step_executions"][0]["step_index"] == 5, (
        "merging must not mutate the state it was given"
    )
    assert len(later_state["observed_waiting_step_executions"]) == 1, later_state


def _ha004_settings(tmp_path: Path) -> ha004.Settings:
    return ha004.Settings(
        regional=_regional(tmp_path).settings,
        node="node-a",
        host_probe_image="registry.example/probe@sha256:" + "a" * 64,
        predecessor_path=tmp_path / "predecessor.json",
    )


def test_ha004_reclaim_timeline_keeps_one_command_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    snapshots = iter(
        [
            {
                "commands": [
                    {
                        "command_id": "remote-a",
                        "status": "LEASED",
                        "lease_owner": "cluster/pod-a",
                        "lease_token": "token-a",
                        "step": {"operation": "RESET_GPU"},
                    }
                ]
            },
            {
                "commands": [
                    {
                        "command_id": "remote-a",
                        "status": "LEASED",
                        "lease_owner": "cluster/pod-b",
                        "lease_token": "token-b",
                        "step": {"operation": "RESET_GPU"},
                    }
                ]
            },
            {
                "commands": [
                    {
                        "command_id": "remote-a",
                        "status": "SUCCEEDED",
                        "last_lease_owner": "cluster/pod-b",
                        "step": {"operation": "RESET_GPU"},
                    }
                ]
            },
        ]
    )
    killed: list[str] = []
    monkeypatch.setattr(regional, "store_snapshot", lambda **_kwargs: next(snapshots))
    state, timeline = ha004.command_timeline(
        regional,
        _ha004_settings(tmp_path),
        marker="marker-a",
        observed_after=datetime.now(timezone.utc),
        timeout_seconds=5,
        kill_owner=killed.append,
    )

    assert killed == ["cluster/pod-a"], killed
    assert {item["command_id"] for item in timeline} == {"remote-a"}, timeline
    assert state["commands"][0]["status"] == "SUCCEEDED", state


def test_ha004_removes_the_dispatching_replica_seen_through_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WAITING is what a sampler sees while the Node Agent resets.

    The executor releases the lease as soon as the agent answers "pending", so
    the replica to remove is the one recorded as `last_lease_owner`; the other
    replica then re-claims the same command ID and the timeline still shows two
    owners and two tokens.
    """

    regional = _regional(tmp_path)
    snapshots = iter(
        [
            {
                "commands": [
                    {
                        "command_id": "remote-a",
                        "status": "WAITING",
                        "lease_owner": None,
                        "last_lease_owner": "cluster/pod-a",
                        "lease_token": "token-a",
                        "step": {"operation": "RESET_GPU"},
                    }
                ]
            },
            {
                "commands": [
                    {
                        "command_id": "remote-a",
                        "status": "WAITING",
                        "lease_owner": None,
                        "last_lease_owner": "cluster/pod-b",
                        "lease_token": "token-b",
                        "step": {"operation": "RESET_GPU"},
                    }
                ]
            },
            {
                "commands": [
                    {
                        "command_id": "remote-a",
                        "status": "SUCCEEDED",
                        "last_lease_owner": "cluster/pod-b",
                        "step": {"operation": "RESET_GPU"},
                    }
                ]
            },
        ]
    )
    killed: list[str] = []
    monkeypatch.setattr(regional, "store_snapshot", lambda **_kwargs: next(snapshots))
    state, timeline = ha004.command_timeline(
        regional,
        _ha004_settings(tmp_path),
        marker="marker-a",
        observed_after=datetime.now(timezone.utc),
        timeout_seconds=5,
        kill_owner=killed.append,
    )

    assert killed == ["cluster/pod-a"], killed
    assert state["ha004_owners"] == ["cluster/pod-a", "cluster/pod-b"], state
    digests = {
        item["lease_token_sha256"] for item in timeline if item["lease_token_sha256"]
    }
    assert digests == {
        hashlib.sha256(b"token-a").hexdigest(),
        hashlib.sha256(b"token-b").hexdigest(),
    }, timeline
    assert all("lease_token" not in item for item in timeline), (
        "the timeline must never carry the raw lease credential"
    )


def test_ha004_reads_a_second_lease_from_the_owner_change_when_tokens_are_hidden() -> (
    None
):
    """Tokens vanish outside LEASED; a new executor identity on the command is the proof."""

    waiting = {
        "command_id": "remote-a",
        "status": "WAITING",
        "lease_owner": None,
        "last_lease_owner": "cluster/pod-a",
        "lease_token": None,
    }
    reclaimed = {**waiting, "status": "SUCCEEDED", "last_lease_owner": "cluster/pod-b"}

    assert ha004.lease_reissue_observed(
        [waiting, reclaimed], first_owner="cluster/pod-a"
    ), "a different owner on the same command is a new claim, hence a new token"
    assert not ha004.lease_reissue_observed(
        [waiting, {**waiting, "status": "SUCCEEDED"}], first_owner="cluster/pod-a"
    ), "the same owner finishing the command proves nothing about a second lease"
    assert not ha004.lease_reissue_observed([waiting, reclaimed], first_owner=""), (
        "without a recorded first owner the owner side cannot stand in for tokens"
    )
    assert ha004.lease_reissue_observed(
        [
            {**waiting, "status": "LEASED", "lease_token_sha256": "digest-a"},
            {**waiting, "status": "LEASED", "lease_token_sha256": "digest-b"},
        ],
        first_owner="cluster/pod-a",
    ), "two distinct token digests remain the direct proof"


def test_ha004_timeline_reads_the_probe_digest_and_never_the_raw_token() -> None:
    """STORE_PROBE emits lease_token_sha256; a raw token from an older probe is digested."""

    assert ha004.lease_token_digest({"lease_token_sha256": "abc"}) == "abc"
    assert ha004.lease_token_digest({"lease_token": "token-a"}) == (
        hashlib.sha256(b"token-a").hexdigest()
    )
    assert ha004.lease_token_digest({"lease_token": None}) is None
    assert ha004.lease_token_digest({}) is None


def test_ha004_watchdog_outlasts_every_phase_and_matches_the_host_probe_deadline() -> (
    None
):
    """The 900s watchdog used to fire inside a 600+900+1200+180s measurement."""

    assert ha004.WATCHDOG_SECONDS > sum(ha004.PHASE_BUDGETS.values())
    assert ha004.WATCHDOG_SECONDS == ha004.HOST_PROBE_ACTIVE_DEADLINE_SECONDS
    assert ha004.PHASE_BUDGETS["executor_rollout"] == 600
    assert ha004.PHASE_BUDGETS["command_timeline"] == 900
    assert ha004.PHASE_BUDGETS["workflow_wait"] == 1200


def test_ha004_effective_sampling_period_is_measured_from_the_timeline() -> None:
    base = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
    timeline = [
        {"observed_at": (base + timedelta(seconds=6 * index)).isoformat()}
        for index in range(5)
    ]
    assert ha004.effective_sampling_period_seconds(timeline) == 6.0
    assert ha004.effective_sampling_period_seconds(timeline[:1]) is None
    assert "6s" in ha004.SAMPLING_LIMITATION


def test_ha004_stop_process_group_kills_the_detached_watchdog(tmp_path: Path) -> None:
    process = subprocess.Popen(
        ["/bin/bash", "-c", "sleep 600"], start_new_session=True, text=True
    )
    try:
        outcome = ha004.stop_process_group(process)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
    assert outcome["disarmed"] is True
    assert process.poll() is not None
    assert ha004.stop_process_group(None) == {"armed": False}


def test_ha004_lease_and_poll_window_goes_through_executor_env_window() -> None:
    assert ha004.LEASE_ENV in env_window.ALLOWED_VARIABLES
    assert ha004.POLL_ENV in env_window.ALLOWED_VARIABLES
    assert ha004.DEPLOYMENT == env_window.DEPLOYMENT


def test_ha003_failover_must_overlap_the_open_command_or_be_inconclusive() -> None:
    """Requesting the failover while LEASED proves nothing about when the writer moved."""

    after_terminal = [
        {"rds_status": "available", "writer": "w1", "reset_command_status": "LEASED"},
        {
            "rds_status": "available",
            "writer": "w1",
            "reset_command_status": "SUCCEEDED",
        },
        {
            "rds_status": "failing-over",
            "writer": "w1",
            "reset_command_status": "SUCCEEDED",
        },
        {
            "rds_status": "available",
            "writer": "w2",
            "reset_command_status": "SUCCEEDED",
        },
    ]
    assert (
        ha003.failover_overlap_observed(after_terminal, previous_writer="w1") is False
    )
    assert ha003.verdict_for([], overlap_observed=False) == "INCONCLUSIVE"
    assert ha003.verdict_for(["x"], overlap_observed=False) == "FAIL"

    overlapping = [
        {"rds_status": "available", "writer": "w1", "reset_command_status": "LEASED"},
        {
            "rds_status": "failing-over",
            "writer": "w1",
            "reset_command_status": "WAITING",
        },
        {
            "rds_status": "available",
            "writer": "w2",
            "reset_command_status": "SUCCEEDED",
        },
    ]
    assert ha003.failover_overlap_observed(overlapping, previous_writer="w1") is True
    assert ha003.verdict_for([], overlap_observed=True) == "PASS"

    writer_switched_while_open = [
        {"rds_status": "available", "writer": "w2", "reset_command_status": "WAITING"}
    ]
    assert ha003.failover_overlap_observed(
        writer_switched_while_open, previous_writer="w1"
    ), "a writer change sampled while the reset is WAITING is the overlap"


def test_ha003_wait_rds_failover_samples_the_command_alongside_each_rds_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rds = iter(
        [
            {"status": "failing-over", "writer": "w1"},
            {"status": "available", "writer": "w2"},
        ]
    )
    stores = iter(
        [
            {"commands": [{"step": {"operation": "RESET_GPU"}, "status": "WAITING"}]},
            {"commands": [{"step": {"operation": "RESET_GPU"}, "status": "SUCCEEDED"}]},
        ]
    )
    monkeypatch.setattr(ha003, "rds_snapshot", lambda _settings: next(rds))

    after, samples = ha003.wait_rds_failover(
        _ha003_settings(tmp_path),
        previous_writer="w1",
        timeout_seconds=30,
        observe=lambda: next(stores),
        sleep=lambda _seconds: None,
    )

    assert after["writer"] == "w2"
    assert [item["reset_command_status"] for item in samples] == [
        "WAITING",
        "SUCCEEDED",
    ]
    assert [item["rds_status"] for item in samples] == ["failing-over", "available"]
    assert ha003.failover_overlap_observed(samples, previous_writer="w1") is True


def test_ha003_executor_result_submission_codes_come_from_rejected_reports() -> None:
    logs = "\n".join(
        [
            "INFO regional cluster executor claimed command=remote-a",
            "ERROR regional cluster executor could not report result: command=remote-a "
            "cluster=c operation=RESET_GPU status=SUCCEEDED",
            "Traceback (most recent call last):",
            '  File "x.py", line 1, in <module>',
            "gpu_fault.cluster_executor.ClusterExecutorError: regional control plane "
            "rejected request (409): lease token is stale",
            "INFO regional control plane rejected request (500): unrelated GET",
            "ERROR regional cluster executor could not report result: command=remote-a "
            "cluster=c operation=RESET_GPU status=SUCCEEDED",
            "ClusterExecutorError: regional control plane rejected request (500): boom",
        ]
    )
    assert ha003.result_submission_codes(logs) == [409, 500]
    assert ha003.result_submission_codes("") == []
    assert ha003.ALLOWED_RESULT_SUBMISSION_CODES == {200, 409}


def test_store_snapshot_drains_the_queue_for_gates_but_samples_once_in_wait_loops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ~10s queue-drain reading belongs to preflight, not to a polling loop."""

    regional = _regional(tmp_path)
    argv: list[tuple[str, ...]] = []

    def cpu_python(_script: str, *arguments: str, **_kwargs: object) -> dict:
        argv.append(arguments)
        return {"release_id": "release-a", "workflow": {"status": "SUCCEEDED"}}

    monkeypatch.setattr(regional, "cpu_python", cpu_python)

    regional.store_snapshot(node="node-a")
    regional.store_snapshot(node="node-a", queue_attempts=1)
    regional.wait_for_workflow(
        node="node-a",
        marker="marker-a",
        observed_after=datetime.now(timezone.utc),
        case_dir=tmp_path,
        timeout_seconds=5,
    )

    assert [call[-1] for call in argv] == ["20", "1", "1"], argv
