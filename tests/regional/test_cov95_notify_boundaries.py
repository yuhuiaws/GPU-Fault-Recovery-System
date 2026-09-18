from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import notification_evidence as evidence
from scripts.e2e.regional import run_notification_acceptance as runner
from tests.regional._cov95_ha001_harness import Clock
from tests.regional._cov95_notify_harness import NODES, NotificationSite


@pytest.mark.parametrize("contents", ["not-json", "[]", "{}"])
def test_external_evidence_cannot_invent_validity_from_malformed_input(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(contents)
    assert evidence.validate_external_evidence(path, "receipt")["valid"] is False
    assert (
        evidence.validate_external_evidence(tmp_path / "missing", "receipt")["valid"]
        is False
    )
    assert evidence.parse_time("invalid-time") is None


def test_external_evidence_rejects_reversed_window_and_unknown_kind(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(
        json.dumps(
            {
                "method": "unit",
                "reference": "unit",
                "received": True,
                "window_start": "2026-01-02T00:00:00Z",
                "window_end": "2026-01-01T00:00:00Z",
            }
        )
    )
    result = evidence.validate_external_evidence(path, "unknown")
    assert result["errors"] == [
        "window_start is after window_end",
        "unknown evidence kind unknown",
    ]


def test_candidate_discovery_ignores_unreadable_and_unrelated_records(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "cases/GF-REGIONAL-DESTR-001"
    directory.mkdir(parents=True)
    (directory / "GF-REGIONAL-DESTR-001.json").write_text("invalid-json")
    assert evidence.action_completed_records_from_evidence(
        tmp_path, cluster_id="unit"
    ) == {"gpu-reset": [], "workload-restart": []}
    (directory / "GF-REGIONAL-DESTR-001.json").write_text(
        json.dumps(
            {
                "notifications": [
                    {
                        "notification": {
                            "category": "ACTION_COMPLETED",
                            "cluster_name": "unit",
                            "deduplication_key": "unrecognized",
                        },
                        "result": {},
                    },
                    {
                        "notification": {
                            "category": "ACTION_COMPLETED",
                            "cluster_name": "unit",
                            "deduplication_key": "/gpu-reset/",
                        },
                        "result": {},
                    },
                ]
            }
        )
    )
    assert evidence.action_completed_records_from_evidence(
        tmp_path, cluster_id="unit"
    ) == {"gpu-reset": [], "workload-restart": []}


@pytest.mark.parametrize("success", [False, True])
def test_workload_readiness_wait_requires_distinct_running_nodes(
    monkeypatch: pytest.MonkeyPatch, success: bool
) -> None:
    clock = Clock()
    monkeypatch.setattr(runner, "time", clock)
    fixture = SimpleNamespace(
        pods=lambda: [
            {"phase": "Running", "ready": True, "node": "a"},
            {
                "phase": "Running",
                "ready": True,
                "node": "b" if success and clock.now >= 5 else "a",
            },
        ]
    )
    if success:
        assert len(runner.wait_running_pods(fixture, 2, timeout_seconds=6)["pods"]) == 2
    else:
        with pytest.raises(RuntimeError, match="did not run"):
            runner.wait_running_pods(fixture, 2, timeout_seconds=6)
    assert clock.now <= 10


@pytest.mark.parametrize("residual", ["pods", "resource", "none"])
def test_workload_deletion_confirms_both_pod_and_owner_absence(
    monkeypatch: pytest.MonkeyPatch, residual: str
) -> None:
    clock = Clock()
    monkeypatch.setattr(runner, "time", clock)
    deleted = []
    fixture = SimpleNamespace(
        name="unit",
        resource="job",
        delete=lambda: deleted.append(True),
        pods=lambda: [{"name": "unit"}] if residual == "pods" or clock.now < 5 else [],
    )
    regional = SimpleNamespace(
        kubectl=lambda *a, **kw: "job/unit" if residual == "resource" else ""
    )
    if residual == "none":
        assert runner.delete_notification_workload(regional, fixture) is None
    else:
        with pytest.raises(
            runner.NotificationAcceptanceError, match="resources remain"
        ):
            runner.delete_notification_workload(regional, fixture)
    assert deleted == [True]


def test_foreign_reservations_include_init_peak_but_not_finished_or_remote_pods() -> (
    None
):
    def pod(name: str, node: str, phase: str, gpu: int) -> dict:
        return {
            "metadata": {"name": name, "namespace": "unit"},
            "spec": {
                "nodeName": node,
                "containers": [{"resources": {"requests": {"nvidia.com/gpu": gpu}}}],
            },
            "status": {"phase": phase},
        }

    active = pod("active", "a", "Pending", 1)
    active["spec"]["initContainers"] = [
        {"resources": {"limits": {"nvidia.com/gpu": 2}}}
    ]
    items = [
        active,
        pod("finished", "a", "Succeeded", 8),
        pod("remote", "other", "Running", 8),
        pod("cpu", "a", "Running", 0),
    ]
    regional = SimpleNamespace(kubectl=lambda *a, **kw: json.dumps({"items": items}))
    assert runner.foreign_gpu_reservations(regional, ("a",)) == [
        {"name": "active", "namespace": "unit", "node": "a", "reserved_gpus": 2}
    ]


def test_latch_timeout_is_distinct_from_telemetry_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr(runner, "time", clock)
    regional = SimpleNamespace(
        cpu_python=lambda *a: {
            "records": [{"node_id": "unit", "notified": True, "active": False}]
        }
    )
    with pytest.raises(
        runner.NotificationAcceptanceError, match="still hold a notified"
    ):
        runner.wait_low_utilization_latch_disarmed(
            regional,
            cluster_id="unit",
            nodes=("unit",),
            deadline_seconds=1,
            poll_seconds=1,
        )
    assert clock.now == 1


@pytest.mark.parametrize(
    "case_id,fault",
    [
        ("GF-REGIONAL-NOTIFY-003", "none"),
        ("GF-REGIONAL-NOTIFY-003", "tests"),
        ("GF-REGIONAL-NOTIFY-004", "executor"),
        ("GF-REGIONAL-NOTIFY-004", "worker"),
        ("GF-REGIONAL-NOTIFY-005", "none"),
        ("GF-REGIONAL-NOTIFY-005", "node"),
        ("GF-REGIONAL-NOTIFY-001", "predecessor"),
    ],
)
def test_plan_records_actual_preflight_result_without_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case_id: str, fault: str
) -> None:
    site = NotificationSite(monkeypatch, tmp_path)
    if fault == "tests":
        site.focused_code = 1
    elif fault == "executor":
        site.gpu_pods = []
    elif fault == "worker":
        site.ready_counts["gpu-fault-control-worker"] = 0
    elif fault == "node":
        site.nodes[NODES[0]]["ready"] = "False"
    plans = []
    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(runner, "install_abort_signals", lambda: None)
    monkeypatch.setattr(runner, "IdentitySite", lambda _: site)
    monkeypatch.setattr(runner, "os", SimpleNamespace(umask=lambda _: None))
    monkeypatch.setattr(
        runner, "predecessor_path", lambda *a: ("unit", tmp_path / "previous")
    )
    monkeypatch.setattr(
        runner,
        "predecessor_evidence",
        lambda *a, **kw: {"valid": fault != "predecessor"},
    )
    monkeypatch.setattr(
        runner,
        "record_focused_tests",
        lambda details, result: details.update(focused_tests=result),
    )
    monkeypatch.setattr(
        runner,
        "build_plan",
        lambda **kw: plans.append(kw) or {"preflight_passed": kw["preflight_passed"]},
    )
    argv = [
        "unit",
        "--run-dir",
        str(tmp_path),
        "--site",
        str(site.site_file),
        "--case",
        case_id,
    ]
    for node in NODES:
        argv.extend(["--node", node])
    monkeypatch.setattr(sys, "argv", argv)
    assert runner.main() == int(fault != "none")
    assert plans[0]["preflight_passed"] is (fault == "none")
    assert site.notifier.sent == []
    assert site.fixtures == []
