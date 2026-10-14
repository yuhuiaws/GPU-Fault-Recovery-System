from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import run_ha010_aurora_blackout_liveness as ha
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._cov95_ha001_harness import pod
from tests.regional._cov95_ha010_harness import T0, HA010Harness


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> HA010Harness:
    return HA010Harness(monkeypatch, tmp_path)


def test_full_aurora_blackout_lifecycle_measures_every_cpu_pod(
    harness: HA010Harness,
) -> None:
    code, report = harness.execute()
    assert code == 0, report.get("error", report.get("errors"))
    assert report["errors"] == report["cleanup"]["errors"] == []
    assert report["rds_before"]["writer"] == "old"
    assert report["rds_after"]["writer"] == "new"
    assert report["replacement_pod"] == "replacement"
    assert len(harness.processes) == 6
    assert len(report["readiness_outage_seconds_by_pod"]) == 5
    assert all(process.poll() is not None for process in harness.processes), (
        "samplers must finish"
    )
    assert harness.events.index("failover") < harness.events.index("delete-pod")
    assert harness.binding_reads == 2
    assert harness.events[-1] == "verify-runtime"
    # Cleanup fails the writer back before it waits for the Deployments.
    assert harness.restore_targets == ["old"] and harness.writer == "old"
    restore = harness.events.index("restore")
    assert harness.events[restore + 1 :] == [
        "ready:deployment/gpu-fault-api-ha",
        "ready:deployment/gpu-fault-control-worker",
        "ready:deployment/gpu-fault-telemetry-spool-worker",
        "verify-runtime",
    ], "cleanup fails the writer back, then waits for the Deployments"
    recorded = report["cleanup"]["writer_restore"]
    assert recorded["action"] == "failover" and recorded["target"] == "old"
    assert recorded["writer_before_restore"] == "new"
    assert recorded["rds_after"]["writer"] == "old" and recorded["errors"] == []
    evidence = json.loads((harness.directory / "writer-restore.json").read_text())
    assert evidence == recorded


@pytest.mark.parametrize("defect", ["busy", "binding", "uid", "spawn", "expired"])
def test_abort_stops_progress_and_reaps_every_started_sampler(
    harness: HA010Harness, defect: str
) -> None:
    if defect == "busy":
        harness.busy_at_boundary = True
    elif defect == "binding":
        harness.binding_failure_at = 2
    elif defect == "uid":
        harness.replace_on_delete = True
    elif defect == "spawn":
        harness.spawn_failure_at = 3
    else:
        harness.deadline = T0
    code, report = harness.execute()
    assert code == 1
    assert report["error"]
    assert all("wait" in process.calls for process in harness.processes), (
        "all started children must be reaped"
    )
    assert "delete-pod" not in harness.events
    assert ("failover" in harness.events) is (defect == "uid")


@pytest.mark.parametrize(
    "defect", ["liveness", "sampler", "replacement", "runtime", "cleanup"]
)
def test_failed_liveness_or_cleanup_cannot_become_pass(
    harness: HA010Harness, defect: str
) -> None:
    if defect == "liveness":
        harness.liveness_status = 503
    elif defect == "sampler":
        harness.sampler_returncode = 7
    elif defect == "replacement":
        harness.never_ready = True
    elif defect == "runtime":
        harness.verify_failure = True
    else:
        harness.cleanup_failure = True
    code, report = harness.execute()
    assert code == 1
    assert report["verdict"] == "FAIL"
    assert all("wait" in process.calls for process in harness.processes), (
        "failed cases must reap their samplers"
    )
    assert report.get("errors") or report.get("error") or report["cleanup"]["errors"], (
        "failure must be explained"
    )


@pytest.mark.parametrize(
    "defect", ["terminating", "missing-ready", "false-ready", "not-running"]
)
def test_pod_readiness_requires_a_current_running_ready_pod(defect: str) -> None:
    document = pod("unit", "unit-app")
    document["status"]["conditions"][0]["lastTransitionTime"] = T0.isoformat()
    if defect == "terminating":
        document["metadata"]["deletionTimestamp"] = T0.isoformat()
    elif defect == "missing-ready":
        document["status"]["conditions"] = []
    elif defect == "false-ready":
        document["status"]["conditions"][0]["status"] = "False"
    else:
        document["status"]["phase"] = "Pending"
    assert ha.pod_record(document)["ready"] is False


@pytest.mark.parametrize("value", ["", "-1", "unknown"])
def test_unknown_replica_read_is_not_a_scaled_to_zero_deployment(value: str) -> None:
    regional = SimpleNamespace(kubectl=lambda *a: value)
    with pytest.raises(RegionalFixtureError, match="replica"):
        ha.deployments_scaled_to_zero(regional)


def test_sampler_payload_cannot_override_observed_process_identity() -> None:
    process = SimpleNamespace(
        communicate=lambda **kw: (
            json.dumps(
                {
                    "pod": "foreign",
                    "returncode": 0,
                    "started_at": "forged",
                    "stderr": "forged",
                    "samples": [],
                }
            ),
            "actual stderr",
        ),
        returncode=7,
    )
    result = ha.Sampler("unit-pod", process, T0).collect(1)
    assert result["pod"] == "unit-pod"
    assert result["returncode"] == 7
    assert result["started_at"] == T0.isoformat()
    assert result["stderr"] == "actual stderr"


def test_failed_sampler_script_feed_reaps_the_spawned_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    class BrokenPipe(io.StringIO):
        def write(self, value: str) -> int:
            raise BrokenPipeError("unit feed failed")

    process = SimpleNamespace(
        stdin=BrokenPipe(),
        poll=lambda: None,
        kill=lambda: calls.append("kill"),
        wait=lambda **kw: calls.append("wait"),
    )
    monkeypatch.setattr(
        ha,
        "subprocess",
        SimpleNamespace(**{**vars(subprocess), "Popen": lambda *a, **kw: process}),
    )
    regional = SimpleNamespace(
        settings=SimpleNamespace(cpu_kubeconfig=tmp_path / "cpu", namespace="unit")
    )
    with pytest.raises(BrokenPipeError, match="unit feed failed"):
        ha.start_sampler(
            regional, {"name": "unit-pod", "port": 8080}, duration_seconds=120
        )
    assert calls == ["kill", "wait"]


@pytest.mark.parametrize("env", ["0", "NaN", "-1"])
def test_bad_live_budget_fails_preflight_before_sampler_start(
    harness: HA010Harness, env: str
) -> None:
    harness.env[ha.verdicts.STALE_SECONDS_VARIABLE] = env
    with pytest.raises(RegionalFixtureError, match="preflight failed"):
        harness.execute()
    assert harness.processes == []
    assert harness.events == []


def test_a_writer_that_cannot_be_restored_fails_the_case(harness: HA010Harness) -> None:
    harness.restore_ignored = True

    code, report = harness.execute()

    assert code == 1 and report["verdict"] == "FAIL"
    assert report["errors"] == [], "the liveness verdict itself passed"
    assert any(
        item.startswith("writer_restore: RegionalFixtureError: ")
        for item in report["cleanup"]["errors"]
    ), report["cleanup"]["errors"]
    evidence = json.loads((harness.directory / "writer-restore.json").read_text())
    assert evidence["action"] == "failover" and evidence["target"] == "old"
    assert "did not converge" in evidence["errors"][0], evidence["errors"]
    assert harness.events[-1] == "verify-runtime", "the rest of cleanup still runs"


def test_the_restore_waits_for_a_quiet_control_plane_and_then_refuses(
    harness: HA010Harness,
) -> None:
    # Reads: preflight, before the failover, the post-check, then the restore's.
    harness.busy_from_read = 4

    code, report = harness.execute()

    assert code == 1 and report["verdict"] == "FAIL"
    assert harness.restore_targets == [], "no failover lands on busy work"
    assert harness.writer == "new"
    (error,) = [
        item
        for item in report["cleanup"]["errors"]
        if item.startswith("writer_restore")
    ]
    assert "writer restore refused" in error and "aurora-failback" in error, error
    assert (harness.directory / "store-before-restore.json").is_file(), (
        "the refused restore keeps its store evidence"
    )
    assert harness.store_reads > 4, "the quiet rule is waited out, not read once"


def test_no_failover_means_no_restore(harness: HA010Harness) -> None:
    harness.busy_at_boundary = True

    code, report = harness.execute()

    assert code == 1
    assert "failover" not in harness.events and "restore" not in harness.events
    assert "writer_restore" not in report["cleanup"]
    assert not (harness.directory / "writer-restore.json").exists(), (
        "nothing moved, so nothing is restored"
    )


def test_a_failover_that_was_requested_is_restored_even_when_the_case_aborts(
    harness: HA010Harness,
) -> None:
    harness.replace_on_delete = True

    code, report = harness.execute()

    assert code == 1 and "stale Pod UID" in report["error"]
    assert harness.restore_targets == ["old"] and harness.writer == "old"
    assert report["cleanup"]["writer_restore"]["rds_after"]["writer"] == "old"
