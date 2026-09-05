"""HA-003/HA-004 fixtures: catching the in-flight reset window and keeping its evidence.

Split out of test_destructive_acceptance_fixtures.py, which had reached the
architecture size limit; these tests share only the regional fixture helper
with it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

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
    assert {item["lease_token"] for item in timeline if item["lease_token"]} == {
        "token-a",
        "token-b",
    }, timeline


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
            {**waiting, "status": "LEASED", "lease_token": "token-a"},
            {**waiting, "status": "LEASED", "lease_token": "token-b"},
        ],
        first_owner="cluster/pod-a",
    ), "two distinct tokens remain the direct proof"


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
