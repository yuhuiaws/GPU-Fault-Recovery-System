from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault.regional import RemoteCommandStatus
from scripts.e2e.regional import run_ha001_control_plane_failover as ha
from tests.regional._cov95_ha001_harness import CHAIN, HA001Harness, pod


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> HA001Harness:
    return HA001Harness(monkeypatch, tmp_path)


def test_complete_failover_lifecycle_uses_owned_resources_and_real_commands(
    harness: HA001Harness,
) -> None:
    code, result = harness.execute()
    assert code == 0, result.get("error")
    assert result["verdict"] == "PASS"
    assert result["cleanup_errors"] == []
    assert result["release_id"] == CHAIN["identity"]["release_id"]
    assert result["cluster_id"] == CHAIN["identity"]["cluster_id"]
    assert result["postflight"]["database"] == {"objects": 0, "links": 0}
    assert len(harness.deletions) == 3
    assert [name for name, _ in harness.deletions] == [
        target["name"] for target in harness.plan["targets"]
    ]
    assert result["synthetic_closure"]["physical_executions"] == 3
    for command_id in harness.seed["command_ids"]:
        command = harness.store.get_remote_command(command_id)
        assert command is not None, "closure command must exist"
        assert command.status == RemoteCommandStatus.SUCCEEDED
        assert command.last_lease_owner == harness.executor_id
    assert harness.events.index("seed") > harness.events.index(
        f"delete:{harness.plan['targets'][1]['name']}"
    )
    assert harness.events.index("complete-commands") < harness.events.index(
        f"delete:{harness.plan['targets'][2]['name']}"
    )
    assert harness.events.index("cleanup:Pod") < harness.events.index("cleanup-closure")
    assert harness.resources == {}
    receipt = json.loads(
        (harness.path / "cases" / ha.CASE_ID / "probe-resources-a1.json").read_text()
    )
    assert all(record["deleted"] is True for record in receipt["resources"].values()), (
        "both UID-owned resources must have confirmed deletion receipts"
    )


@pytest.mark.parametrize("kind", ["ConfigMap", "Pod"])
def test_create_ack_loss_still_cleans_only_the_receipted_resources(
    harness: HA001Harness, kind: str
) -> None:
    harness.create_failure = (kind, RuntimeError("unit lost create ACK"))
    code, report = harness.execute()
    assert code == 1
    assert "lost create ACK" in report["error"]
    assert report["cleanup_errors"] == []
    assert harness.resources == {}
    assert harness.deletions == []
    assert "seed" not in harness.events


def test_preflight_uid_drift_stops_before_any_resource_creation(
    harness: HA001Harness,
) -> None:
    harness.drift_target(0)
    with pytest.raises(ha.CaseError, match="planned target drifted"):
        harness.execute()
    assert harness.events == []


def test_late_uid_drift_stops_the_next_deletion_and_cleans_the_probe(
    harness: HA001Harness,
) -> None:
    harness.after_delete = lambda: harness.drift_target(1)
    code, report = harness.execute()
    assert code == 1
    assert "target changed before deletion" in report["error"]
    assert len(harness.deletions) == 1
    assert harness.resources == {}
    assert "seed" not in harness.events


@pytest.mark.parametrize("when", ["before-seed", "after-first-delete"])
def test_abort_stops_progress_and_confirms_cleanup(
    harness: HA001Harness, when: str
) -> None:
    if when == "before-seed":
        harness.before_seed = harness.abort
    else:
        harness.after_delete = harness.abort
    code, report = harness.execute()
    assert code == 1
    assert "unit acceptance abort" in report["error"]
    assert len(harness.deletions) == (2 if when == "before-seed" else 1)
    assert harness.resources == {}
    assert report["cleanup_errors"] == []


@pytest.mark.parametrize("when", ["create", "cleanup"])
def test_supervision_loss_never_sends_a_subsequent_remote_command(
    harness: HA001Harness, when: str
) -> None:
    loss = ProcessSupervisionLost("unit supervision lost")
    if when == "create":
        harness.create_failure = ("Pod", loss)
    else:
        harness.delete_failure = loss
    code, report = harness.execute()
    assert code == 1
    assert report["supervision_lost"] is True
    assert report["cleanup_preserved"]
    assert harness.events[-1] == ("create:Pod" if when == "create" else "stop-probe")
    assert len(harness.resources) == 2
    assert harness.cleaned is False


def test_unproven_probe_shutdown_preserves_command_evidence(
    harness: HA001Harness,
) -> None:
    harness.delete_failure = RuntimeError("unit delete unavailable")
    code, report = harness.execute()
    assert code == 1
    assert (
        "probe shutdown unverified; preserving synthetic closure"
        in (report["cleanup_errors"])
    )
    assert harness.cleaned is False
    assert "cleanup-closure" not in harness.events
    assert report["postflight"]["database"]["objects"] == 5


def test_insufficient_load_cannot_pass_a_successful_recovery(
    harness: HA001Harness,
) -> None:
    harness.final_cycles = 0
    code, report = harness.execute()
    assert code == 1
    assert report["errors"] == [
        "probe did not sustain enough claim cycles",
        "probe did not sustain enough health cycles",
    ]
    assert harness.cleaned is True


def test_seed_ack_loss_retains_owned_ids_and_cleans_after_confirmed_probe_stop(
    harness: HA001Harness,
) -> None:
    harness.seed_ack_lost = True
    code, report = harness.execute()
    assert code == 1
    assert "lost seed ACK" in report["error"]
    assert report["cleanup_errors"] == []
    assert harness.cleaned is True
    assert report["postflight"]["database"] == {"objects": 0, "links": 0}
    assert harness.events.count("seed") == 1
    assert harness.events.index("cleanup:Pod") < harness.events.index("cleanup-closure")
    assert len(harness.deletions) == 2


@pytest.mark.parametrize("status", ["http-401", "http-403", "http-500"])
def test_forbidden_probe_response_aborts_before_control_pod_deletion(
    harness: HA001Harness, status: str
) -> None:
    harness.probe_errors = {status: 1}
    code, report = harness.execute()
    assert code == 1
    assert "forbidden probe responses" in report["error"]
    assert harness.deletions == []
    assert harness.resources == {}


@pytest.mark.parametrize("defect", ["terminating", "missing-condition", "unready"])
def test_control_sample_does_not_count_unknown_or_terminating_pods(
    harness: HA001Harness, defect: str
) -> None:
    name = "unhealthy-ingress"
    document = pod(name, ha.INGRESS_APP)
    if defect == "terminating":
        document["metadata"]["deletionTimestamp"] = "2099-01-01T00:00:00Z"
    elif defect == "missing-condition":
        document["status"]["conditions"] = []
    else:
        document["status"]["conditions"][0]["status"] = "False"
    harness.pods[name] = document
    sample = ha.control_sample(include_queue=True)
    assert sample["ingress_ready"] == 3, (
        "availability must use complete Pod readiness, not just container status"
    )
    assert name in sample["ingress_pods"]


@pytest.mark.parametrize("residuals", [{"objects": 1, "links": 0}, {"links": 0}])
def test_dirty_or_incomplete_database_preflight_cannot_delete_foreign_state(
    harness: HA001Harness, monkeypatch: pytest.MonkeyPatch, residuals: dict[str, Any]
) -> None:
    monkeypatch.setattr(ha, "database_residuals", lambda: residuals)
    code, report = harness.execute()
    assert code == 1
    assert "database preflight found residuals" in report["error"]
    assert harness.resources == {}
    assert harness.deletions == []
    assert "cleanup-closure" not in harness.events
