from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import Environment
from gpu_fault.watcher import AttemptObservation, ContainerObservation, WorkloadPhase
from scripts.e2e.regional import run_destr001_gpu_reset as reset
from scripts.e2e.regional.probes import destructive_node_probe as probe

NOW = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)
NEW_UID = "22222222-2222-4222-8222-222222222222"
OLD_UID = "11111111-1111-4111-8111-111111111111"


def workload_contract() -> dict[str, Any]:
    observation = AttemptObservation(
        cluster_id="cluster-a",
        environment=Environment.KUBERNETES,
        job_id="training-a",
        attempt_id="training-a002",
        workload_phase=WorkloadPhase.RUNNING,
        observed_at=NOW,
        expected_critical_ranks=1,
        runtime_profile_version="profile-a",
        containers=[
            ContainerObservation(
                pod_uid=NEW_UID,
                pod_name="training-worker",
                container_name="trainer",
                role="worker",
                rank=0,
                node_id="node-a",
                gpu_count=2,
                gpu_uuids=["GPU-a", "GPU-b"],
            )
        ],
    )
    return {
        "cluster_id": "cluster-a",
        "job_id": "training-a",
        "node": "node-a",
        "source_attempt_id": "training-a001",
        "source_pod_uids": [OLD_UID],
        "pods": [
            {
                "uid": NEW_UID,
                "name": "training-worker",
                "node": "node-a",
                "phase": "Running",
                "ready": True,
                "attempt_id": "training-a002",
            }
        ],
        "observation": observation.model_dump(mode="json"),
    }


def host_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    before = {
        "gpu_inventory": [
            {"uuid": "GPU-a", "pci_bdf": "0000:01:00"},
            {"uuid": "GPU-b", "pci_bdf": "0000:02:00"},
        ],
        "ledger": [],
        "services": {},
        "gpu_fault_timers": [],
    }
    after = {
        **before,
        "captured_at": (NOW + timedelta(seconds=1)).isoformat(),
        "compute_clients": [
            {"gpu_uuid": uuid, "pid": str(index + 100), "pod_uid": NEW_UID}
            for index, uuid in enumerate(("GPU-a", "GPU-b"))
        ],
        "ledger": [
            {
                "command_id": "reset",
                "operation": "RESET_GPU",
                "state": "SUCCEEDED",
                "attempt": 1,
                "gpu_uuids": ["GPU-a"],
            }
        ],
        "sampler": {
            "sample_count": 3,
            "min_gpu_count": 1,
            "last": {"gpu_count": 2, "gpu_uuids": ["GPU-a", "GPU-b"]},
            "observed_gpu_uuid_sets": [["GPU-a", "GPU-b"], ["GPU-b"]],
        },
    }
    return before, after


def test_only_proven_restarted_training_clients_can_pass_the_reset_contract() -> None:
    before, after = host_pair()
    assert reset.host_errors(
        before, after, expected_gpu_count=2, target_bdf="0000:01:00"
    ), "the idle contract must continue rejecting compute clients"
    assert (
        reset.host_errors(
            before,
            after,
            expected_gpu_count=2,
            target_bdf="0000:01:00",
            post_restart_workload=workload_contract(),
        )
        == []
    ), "the complete owned replacement attempt is allowed"


@pytest.mark.parametrize(
    "change",
    [
        {"pod_uid": "foreign"},
        {"pod_uid": OLD_UID},
        {"pod_uid": None},
        {"gpu_uuid": "GPU-foreign"},
        {"pid": "unknown"},
    ],
)
def test_restarted_workload_does_not_exempt_foreign_or_unidentified_clients(
    change: dict[str, Any],
) -> None:
    before, after = host_pair()
    after["compute_clients"][0].update(change)
    errors = reset.host_errors(
        before,
        after,
        expected_gpu_count=2,
        target_bdf="0000:01:00",
        post_restart_workload=workload_contract(),
    )
    assert errors, change


@pytest.mark.parametrize(
    "change",
    [
        "stale",
        "future",
        "attempt",
        "pod",
        "ready",
        "cluster",
        "allocation",
        "node",
        "lineage",
    ],
)
def test_post_restart_contract_requires_fresh_complete_owned_observations(
    change: str,
) -> None:
    contract = workload_contract()
    _, after = host_pair()
    observation = contract["observation"]
    if change in {"stale", "future"}:
        observation["observed_at"] = (
            NOW + timedelta(seconds=-121 if change == "stale" else 5)
        ).isoformat()
    elif change == "attempt":
        observation["attempt_id"] = contract["source_attempt_id"]
    elif change == "pod":
        observation["containers"][0]["pod_uid"] = OLD_UID
    elif change == "ready":
        contract["pods"][0]["ready"] = False
    elif change == "cluster":
        contract["cluster_id"] = "other-cluster"
    elif change == "allocation":
        observation["containers"][0]["gpu_uuids"] = ["GPU-a"]
    elif change == "node":
        observation["containers"][0]["node_id"] = "other-node"
    else:
        contract["source_pod_uids"] = [NEW_UID]
    assert reset.post_restart_client_errors(after, contract), change


@pytest.mark.parametrize("systemd", [False, True])
def test_client_pod_uid_is_read_from_the_actual_process_cgroup(
    tmp_path: Path, systemd: bool
) -> None:
    directory = tmp_path / "123"
    directory.mkdir()
    uid = NEW_UID.replace("-", "_") if systemd else NEW_UID
    path = (
        f"/kubepods-burstable-pod{uid}.slice/container.scope"
        if systemd
        else f"/kubepods/burstable/pod{uid}/container"
    )
    (directory / "cgroup").write_text(f"0::{path}\n", encoding="utf-8")

    assert probe.client_pod_uid("123", proc_root=tmp_path) == NEW_UID
    assert probe.client_pod_uid("999", proc_root=tmp_path) is None
    assert probe.client_pod_uid("../123", proc_root=tmp_path) is None


def test_sampler_records_per_sample_uuid_sets_without_counting_query_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = [
        subprocess.CompletedProcess([], 0, "GPU-a\nGPU-b\n", ""),
        subprocess.CompletedProcess([], 1, "", "query failed"),
        subprocess.CompletedProcess([], 0, "GPU-b\n", ""),
        subprocess.CompletedProcess([], 0, "GPU-a\nGPU-b\n", ""),
    ]
    monkeypatch.setattr(probe, "run", lambda *_args, **_kwargs: values.pop(0))
    samples = [probe.gpu_sample() for _ in range(4)]
    assert samples[1]["gpu_count"] is None
    assert samples[1]["gpu_uuids"] is None
    path = tmp_path / "sampler.ndjson"
    path.write_text("\n".join(json.dumps(value) for value in samples), encoding="utf-8")
    monkeypatch.setattr(probe, "sampler_paths", lambda _run: ("sampler", path))
    monkeypatch.setattr(
        probe,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "inactive", ""),
    )

    summary = probe.sampler_summary("owned-run")

    assert summary["min_gpu_count"] == 1
    assert [item["gpu_uuids"] for item in summary["identity_samples"]] == [
        ["GPU-a", "GPU-b"],
        None,
        ["GPU-b"],
        ["GPU-a", "GPU-b"],
    ]


def test_workload_contract_does_not_relax_target_gpu_identity() -> None:
    before, after = host_pair()
    changed = deepcopy(after)
    changed["sampler"]["observed_gpu_uuid_sets"] = [["GPU-a"]]
    errors = reset.host_errors(
        before,
        changed,
        expected_gpu_count=2,
        target_bdf="0000:01:00",
        post_restart_workload=workload_contract(),
    )
    assert any("only the target" in error for error in errors), errors
