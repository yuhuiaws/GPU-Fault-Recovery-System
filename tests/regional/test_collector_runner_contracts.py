"""Contracts of the COLLECT acceptance/destructive runners fixed in the
2026-09-07 regional e2e review: cleanup on every path, assertions that read the
right record, one mutation per case, and no wall-clock sleeps where a poll will do.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import collector_acceptance_fixture as fixture_module
from scripts.e2e.regional import run_collector_acceptance as acceptance
from scripts.e2e.regional import run_collector_destructive as destructive
from scripts.e2e.regional.collector_negative_evidence import (
    SCOPE_ACTIONS,
    SCOPE_REASONS,
)
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._collector_power_receipt import load_receipt
from tests.regional._collector_reset_support import ResetAudit


class FakeClock:
    """``time`` for a runner module: sleeps advance the monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def time(self) -> float:
        return 1_700_000_000 + self.now


def _stamp(offset_seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()


# --------------------------------------------------------------------------- #
# shared fixture helpers
# --------------------------------------------------------------------------- #


def test_select_workflow_reads_the_record_the_case_is_about() -> None:
    restore = {
        "request_id": "wf-restore",
        "status": "SUCCEEDED",
        "official_action": "RESTORE",
        "official_steps": [{"operation": "RESTORE_SCHEDULING"}],
    }
    reboot = {
        "request_id": "wf-reboot",
        "status": "SUCCEEDED",
        "official_action": "RESTART_BM",
        "official_steps": [{"operation": "RESTART_NODE"}],
    }
    workflows = [restore, reboot]  # newest first, as the probes return them
    assert fixture_module.select_workflow(workflows, operation="RESTART_NODE") is reboot
    assert (
        fixture_module.select_workflow(workflows, official_actions={"RESTART_BM"})
        is reboot
    )
    assert fixture_module.select_workflow(workflows, operation="RESET_GPU") is None
    assert fixture_module.select_workflow([], operation="RESTART_NODE") is None


def test_restore_incidents_offers_every_incident_and_refuses_a_still_held_node() -> (
    None
):
    """``[]`` used to be the answer both for "nothing to do" and "another
    incident still owns the node"; the second one is a failure."""

    ownership = {"gpu-fault.io/incident": "inc-3:1"}
    node_state = {"owned": True}
    restored: list[str] = []

    class Regional:
        settings = SimpleNamespace(
            gpu_kubeconfig="k", gpu_context="c", namespace="n", cluster_id="cl"
        )

        def node_snapshot(self, _node: str) -> dict[str, Any]:
            return {
                "ownership_annotations": ownership if node_state["owned"] else {},
                "taints": [],
                "unschedulable": False,
            }

    class Restore:
        def __init__(self, *_: Any) -> None:
            pass

        def create_restore_workflow(self, *, incident_id: str, **_: Any) -> dict:
            restored.append(incident_id)
            node_state["owned"] = False
            return {"workflow_request_id": f"restore-{incident_id}"}

        def wait_workflow_id(self, workflow_id: str) -> dict:
            return {"request_id": workflow_id, "status": "SUCCEEDED"}

        def wait_incident_idle(self, incident_id: str) -> None:
            assert incident_id in {"inc-1", "inc-3"}

    fixture = fixture_module.CollectorAcceptanceFixture.__new__(
        fixture_module.CollectorAcceptanceFixture
    )
    fixture.regional = Regional()  # type: ignore[assignment]
    fixture.node = "node-a"
    original = fixture_module.WarmSpareLiveFixture
    fixture_module.WarmSpareLiveFixture = Restore  # type: ignore[misc,assignment]
    try:
        state = {
            "incidents": [{"incident_id": "inc-1"}, {"incident_id": "inc-3"}],
            "workflows": [],
            "commands": [],
        }
        result = fixture.restore_incidents(state, profile_version="v1", reason="test")
        # Newest first: the owning incident is offered first, the rest skipped
        # once the ownership is gone.
        assert restored == ["inc-3"], restored
        assert [item["status"] for item in result] == ["SUCCEEDED"]

        node_state["owned"] = True
        with pytest.raises(RegionalFixtureError, match="still isolated"):
            fixture.restore_incidents(
                {"incidents": [], "workflows": [], "commands": []},
                profile_version="v1",
                reason="test",
            )
    finally:
        fixture_module.WarmSpareLiveFixture = original  # type: ignore[misc]


def test_wait_marker_polls_light_and_scans_evidence_once_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(fixture_module, "time", clock)
    calls: list[bool] = []
    statuses = iter(["RUNNING", "RUNNING", "SUCCEEDED"])

    fixture = fixture_module.CollectorAcceptanceFixture.__new__(
        fixture_module.CollectorAcceptanceFixture
    )

    def store_snapshot(marker: str, *, observed_after=None, scan_evidence=True):
        calls.append(scan_evidence)
        status = next(statuses, "SUCCEEDED")
        return {
            "evidence": [{"record_id": "r"}] if scan_evidence else [],
            "evidence_scanned": scan_evidence,
            "workflows": [{"status": status}],
        }

    fixture.store_snapshot = store_snapshot  # type: ignore[method-assign]
    result = fixture.wait_marker(
        "m", case_dir=tmp_path, timeout_seconds=60, terminal_workflow=True
    )
    assert result["evidence_scanned"] is True
    assert calls == [False, False, False, True], (
        "two light polls while RUNNING, then the light read that saw SUCCEEDED "
        "and the single evidence scan"
    )
    assert all(s == fixture_module.STORE_POLL_SECONDS for s in clock.sleeps), (
        "every poll waits the store poll interval"
    )


# --------------------------------------------------------------------------- #
# run_collector_acceptance
# --------------------------------------------------------------------------- #


def test_service_state_errors_names_units_that_died() -> None:
    baseline = {
        "services": {
            "gpu-fault-node-agent.service": {"ActiveState": "active"},
            "nvidia-persistenced.service": {"ActiveState": "inactive"},
        }
    }
    after = {"services": {"gpu-fault-node-agent.service": {"ActiveState": "failed"}}}
    errors = acceptance.service_state_errors(baseline, after)
    assert errors == [
        "gpu-fault-node-agent.service is failed after the case, active before it"
    ]
    assert acceptance.service_state_errors(baseline, baseline) == []


def test_seconds_until_next_summary_is_one_computed_pause() -> None:
    now = datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc)
    last = (now - timedelta(seconds=100)).isoformat()
    assert acceptance.seconds_until_next_summary(last, summary=300, now=now) == 200
    assert acceptance.seconds_until_next_summary(None, summary=300, now=now) == 0
    stale = (now - timedelta(seconds=900)).isoformat()
    assert acceptance.seconds_until_next_summary(stale, summary=300, now=now) == 0
    future = (now + timedelta(seconds=900)).isoformat()
    assert acceptance.seconds_until_next_summary(future, summary=300, now=now) == 300


def test_collect001_judges_each_chain_by_its_own_interval_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(acceptance, "time", clock)
    env = {
        "GPU_FAULT_METRICS_INTERVAL_SECONDS": "5",
        "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "60",
        "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
        "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
    }
    statuses = [
        {"collector": "GPU_METRICS", "batch_id": "dcgm-1", "observed_at": _stamp(60)},
        {"collector": "GPU_METRICS", "batch_id": "dcgm-2", "observed_at": _stamp(120)},
        {"collector": "GPU_METRICS", "batch_id": "dcgm-3", "observed_at": _stamp(180)},
        {
            "collector": "HOST_TELEMETRY",
            "batch_id": "host-1",
            "observed_at": _stamp(300),
        },
        {
            "collector": "HOST_TELEMETRY",
            "batch_id": "host-2",
            "observed_at": _stamp(600),
        },
    ]
    monkeypatch.setattr(acceptance, "collector_statuses", lambda *_, **__: statuses)
    monkeypatch.setattr(acceptance, "recent_evidence", lambda *_, **__: [])
    fixture = SimpleNamespace(
        snapshot=lambda: {"collector_env": env}, regional=object(), node="n"
    )

    result = acceptance.run_collect001(fixture, tmp_path)  # type: ignore[arg-type]

    assert result["verdict"] == "PASS", result["errors"]
    assert result["chains"] == {
        "GPU_METRICS": {"interval": 5, "summary": 60},
        "HOST_TELEMETRY": {"interval": 15, "summary": 300},
    }
    assert result["observation_seconds"] == 300 * 2 + 15 * 4
    gpu = result["deliveries"]["GPU_METRICS"]
    host = result["deliveries"]["HOST_TELEMETRY"]
    assert gpu["summary_deliveries_in_window"] == 11
    assert host["summary_deliveries_in_window"] == 2, (
        "the host chain is judged against its own 300s summary, not the max()"
    )
    assert set(clock.sleeps) == {5}, "polling runs at the fastest chain's interval"


def _collect002_fixture(calls: list[str], *, restore_raises: bool) -> Any:
    def execute(*arguments: str, timeout: int = 180) -> dict[str, Any]:
        calls.append(arguments[0])
        if arguments[0] == "restore-gpu-power-limit" and restore_raises:
            raise RuntimeError("restore boom")
        return {
            **load_receipt(
                arguments[arguments.index("--run-id") + 1],
                acceptance.datetime.now(timezone.utc),
            ),
            "mutation_started": True,
            "timer_armed": True,
            "load_unit_active": True,
            "restored": True,
            "load_stopped": True,
            "timer_disarmed": True,
            "cleanup_verified": True,
        }

    power = [
        {
            "index": 0,
            "power_limit_w": 700.0,
            "power_default_limit_w": 700.0,
            "power_min_limit_w": 200.0,
        }
    ]
    env = {
        "GPU_FAULT_METRICS_INTERVAL_SECONDS": "5",
        "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "60",
        "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES": "2",
    }
    return SimpleNamespace(
        snapshot=lambda: {"collector_env": env, "gpu_power": power},
        execute=execute,
        regional=SimpleNamespace(settings=SimpleNamespace(cluster_id="cluster-a")),
        node="n",
    )


def test_collect002_restores_the_power_limit_and_keeps_the_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(acceptance, "time", clock)
    stamps = iter([_stamp(-10), _stamp(0)])

    def gpu_metrics_stamp(_fixture: Any) -> str:
        try:
            return next(stamps)
        except StopIteration:
            raise ValueError("store unreachable") from None

    monkeypatch.setattr(acceptance, "gpu_metrics_stamp", gpu_metrics_stamp)
    calls: list[str] = []
    fixture = _collect002_fixture(calls, restore_raises=True)

    with pytest.raises(ValueError, match="store unreachable"):
        acceptance.run_collect002(fixture, tmp_path, 1)
    assert calls == ["throttle-gpu", "restore-gpu-power-limit"], (
        "the restore is attempted and its own failure does not mask the case error"
    )
    assert (tmp_path / "cleanup-error.json").is_file(), (
        "a cleanup failure is recorded next to the evidence"
    )
    # Phase alignment: one computed pause to the next summary, not a poll storm.
    assert clock.sleeps and 45 <= clock.sleeps[0] <= 60, clock.sleeps


def test_collect002_reports_a_failed_restore_as_a_case_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(acceptance, "time", clock)
    stamps = iter([_stamp(-10), _stamp(0), _stamp(12)])
    monkeypatch.setattr(
        acceptance, "gpu_metrics_stamp", lambda _f: next(stamps, _stamp(12))
    )
    monkeypatch.setattr(
        acceptance,
        "recent_evidence",
        lambda *_, **__: [
            {
                "kind": "GPU_METRICS",
                "payload": {"edge_filter_reasons": ["candidate:power_violation"]},
            }
        ],
    )
    calls: list[str] = []
    fixture = _collect002_fixture(calls, restore_raises=True)

    result = acceptance.run_collect002(fixture, tmp_path, 1)

    assert result["verdict"] == "FAIL"
    assert any("restore failed" in item for item in result["errors"]), result
    assert result["cleanup_error"] == "RuntimeError: restore boom"


def _fm_reading(*, offset: int, size: int, inode: int = 7, active: bool = True):
    return {
        "service": {"ActiveState": "active" if active else "activating"},
        "fabric_manager_log": {"path": "/var/log/fm.log", "size": size, "inode": inode},
        "fabric_manager_cursor": {
            "files": {
                "/var/log/fm.log": {"offset": offset, "inode": inode, "device": 1}
            }
        },
    }


def test_cursor_claims_compare_the_persisted_cursor_to_the_log() -> None:
    assert acceptance.fm_cursor_caught_up(_fm_reading(offset=160, size=160)), (
        "a cursor at EOF of the active log is caught up"
    )
    assert not acceptance.fm_cursor_caught_up(_fm_reading(offset=100, size=160)), (
        "a cursor behind EOF is not caught up"
    )
    assert not acceptance.fm_cursor_caught_up(
        _fm_reading(offset=160, size=160, active=False)
    ), "an inactive log is never caught up"
    before = _fm_reading(offset=100, size=100)
    after = _fm_reading(offset=160, size=160)
    assert acceptance.cursor_errors(before, after) == []
    stuck = acceptance.cursor_errors(before, _fm_reading(offset=100, size=160))
    assert any("not the log's EOF" in item for item in stuck), stuck
    assert any("did not advance" in item for item in stuck), stuck
    rotated = acceptance.cursor_errors(
        before, _fm_reading(offset=160, size=160, inode=9)
    )
    assert rotated == [] or all("inode" not in item for item in rotated)
    rotated_after = _fm_reading(offset=160, size=160)
    rotated_after["fabric_manager_log"]["inode"] = 9
    assert any(
        "inode" in item for item in acceptance.cursor_errors(before, rotated_after)
    ), "a rotated log with the same offsets is reported by inode"
    no_cursor = {"fabric_manager_log": {"path": "/var/log/fm.log", "size": 1}}
    assert acceptance.cursor_errors(before, no_cursor) == [
        "Fabric Manager collector has no persisted cursor for the log"
    ]


def test_collect005_polls_the_cursor_and_samples_the_replay_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.regional._alignment_fm_receipts import FmProducerFixture

    clock = FakeClock()
    monkeypatch.setattr(acceptance, "time", clock)
    fixture = FmProducerFixture()
    original = fixture.execute
    cursor_readings = iter(
        [
            _fm_reading(offset=160, size=160),
            _fm_reading(offset=100, size=160, active=False),
            _fm_reading(offset=160, size=160),
        ]
    )

    def execute(*arguments: str, timeout: int = 180) -> dict[str, Any]:
        if arguments[0] == "fm-cursor":
            fixture.calls.append("fm-cursor")
            return next(cursor_readings)
        return original(*arguments, timeout=timeout)

    monkeypatch.setattr(fixture, "execute", execute)

    result = acceptance.run_collect005(fixture, tmp_path, 1)  # type: ignore[arg-type]

    assert result["verdict"] == "PASS", result["errors"]
    assert fixture.calls.count("fm-cursor") == 3, (
        "persisted before restart, polled after"
    )
    assert (
        len(result["producer_observation_samples"])
        == acceptance.REPLAY_OBSERVATION_SAMPLES
    )
    assert 15 not in clock.sleeps, "no fixed 15s sleep after the restart"
    assert clock.sleeps.count(acceptance.REPLAY_OBSERVATION_SECONDS) == (
        acceptance.REPLAY_OBSERVATION_SAMPLES - 1
    )


def test_collect012_judges_the_idle_node_monitor_only_without_a_workflow(
    tmp_path: Path,
) -> None:
    """55272a0: RESTART_APP on an idle node is MONITOR_ONLY, not a quarantine.

    The two XID 13 writes share one marker and earn distinct kmsg sequences;
    neither opens a workflow, so there is no BLOCKED chain and no restore step
    between the XIDs -- the node was never isolated."""

    calls: list[str] = []
    sequence = iter([100, 107, 300])
    records: dict[str, list[dict[str, Any]]] = {}

    class Regional:
        settings = SimpleNamespace(cluster_id="cluster-a")

        def node_snapshot(self, node: str) -> dict:
            return {"ready": "True", "unschedulable": False, "taints": []}

    class Fixture:
        node = "node-a"
        regional = Regional()

        def snapshot(self) -> dict:
            return {
                "boot_id": "boot-1",
                "gpu_inventory": [{"pci_bdf": "0000:59:00.0", "uuid": "GPU-a"}],
                "kernel_collector": {"kmsg_fds": ["3"]},
                "services": {},
            }

        def execute(self, *arguments: str, timeout: int = 180) -> dict:
            if arguments[0] == "gpu-identity":
                return {"product": "H100", "driver_branch": 575, "cuda_version": "12.9"}
            calls.append("inject:" + arguments[arguments.index("--xid") + 1])
            marker = arguments[arguments.index("--marker") + 1]
            seq = next(sequence)
            records.setdefault(marker, []).append(
                {
                    "record_id": f"nvidia-kernel/kmsg-boot-1-{seq}",
                    "cluster_id": "cluster-a",
                    "node_id": "node-a",
                    "payload": {
                        "record_id": f"kmsg-boot-1-{seq}",
                        "cluster_id": "cluster-a",
                        "node_id": "node-a",
                        "source_boot_id": "boot-1",
                        "evidence_ref": f"kmsg://node-a/boot-1/{seq}",
                    },
                }
            )
            return {}

        def store_snapshot(
            self, marker: str, *, observed_after=None, scan_evidence: bool = True
        ) -> dict:
            return {
                "evidence": list(records.get(marker, [])) if scan_evidence else [],
                "evidence_scanned": scan_evidence,
                "events": [
                    {
                        **item["payload"],
                        "event_id": item["record_id"],
                        "gpu_uuid": "GPU-a",
                        "xid": 13 if marker.startswith("c012-13-") else 31,
                        "product": "H100",
                        "driver_branch": 575,
                        "cuda_version": "12.9",
                    }
                    for item in records.get(marker, [])
                ],
                "decisions": [
                    {
                        "event_id": item["record_id"],
                        "official_action": "RESTART_APP",
                        "disposition": "MONITOR_ONLY",
                        "action": "NO_ACTION",
                        "reasons": ["no managed application to restart"],
                        "workflow_request_id": None,
                        "incident_id": f"inc-{marker}",
                    }
                    for item in records.get(marker, [])
                ],
                "workflows": [],
                "incidents": [
                    {
                        "incident_id": f"inc-{marker}",
                        "state": "RECOVERED",
                        "workflow_request_id": None,
                    }
                ],
            }

    result = acceptance.run_collect012(Fixture(), tmp_path, 1, "hyperpod-v1")

    assert result["verdict"] == "PASS", result
    assert result["markers"][0] == result["markers"][1], "samples 1 and 2 are one text"
    assert calls == ["inject:13", "inject:13", "inject:31"], (
        "no restore step: the node was never quarantined"
    )
    assert len(result["record_ids"]) == 3
    assert "restore_workflows" not in result, "there is nothing to restore"


def test_restart_app_monitor_only_errors_names_every_regression() -> None:
    good_decision = {
        "official_action": "RESTART_APP",
        "disposition": "MONITOR_ONLY",
        "action": "NO_ACTION",
        "reasons": ["no managed application to restart on an IDLE node"],
        "workflow_request_id": None,
    }
    good_incident = {"state": "RECOVERED", "workflow_request_id": None}
    assert (
        acceptance.restart_app_monitor_only_errors([good_decision], [], [good_incident])
        == []
    )
    # The old EXECUTABLE / BLOCKED shape must now fail every reading.
    blocked = {
        **good_decision,
        "disposition": "EXECUTABLE",
        "action": "RESTART_WORKLOAD",
    }
    errors = acceptance.restart_app_monitor_only_errors(
        [blocked],
        [{"status": "BLOCKED"}],
        [{"state": "QUARANTINED", "workflow_request_id": "wf-1"}],
    )
    assert any("MONITOR_ONLY" in item for item in errors), errors
    assert any("NO_ACTION" in item for item in errors), errors
    assert any("opened a workflow" in item for item in errors), errors
    assert any("RECOVERED" in item for item in errors), errors
    missing = acceptance.restart_app_monitor_only_errors([], [], [])
    assert any("no RESTART_APP decision" in item for item in missing), missing


def test_case_cleanup_finish_releases_whatever_the_case_still_holds() -> None:
    kubectl_calls: list[tuple[str, ...]] = []
    restores: list[str] = []

    class Fixture:
        def __init__(self, node: str, *, fail: bool = False) -> None:
            self.node = node
            self.fail = fail
            self.regional = SimpleNamespace(
                kubectl=lambda *args, **kwargs: kubectl_calls.append(args)
            )

        def restore_incidents(self, state: dict, **_: Any) -> list:
            if self.fail:
                raise RegionalFixtureError("still isolated")
            restores.append(str(state["id"]))
            return [{"status": "SUCCEEDED"}]

    good, bad = Fixture("node-a"), Fixture("node-b", fail=True)
    cleanup = acceptance.CaseCleanup()
    cleanup.register_state(good, {"id": "first", "workflows": [], "commands": []})
    cleanup.register_state(bad, {"id": "second", "workflows": [], "commands": []})
    cleanup.register_state(good, {"id": "third", "workflows": [], "commands": []})
    cleanup.register_annotation(good, "gpu-fault.io/mechanical-inspection-complete")

    released = cleanup.finish(profile_version="v1", reason="end")

    assert restores == ["third"], (
        "one successful node restore settles its older registrations"
    )
    assert len(released["errors"]) == 1 and "node-b" in released["errors"][0]
    assert kubectl_calls == [
        (
            "gpu",
            "annotate",
            "node",
            "node-a",
            "gpu-fault.io/mechanical-inspection-complete-",
        )
    ]
    assert cleanup.annotations == []
    assert [state["id"] for _, state in cleanup.incident_states] == ["second"], (
        "a state whose restore failed stays registered for the report"
    )


def test_collect010_registers_the_quarantine_before_it_judges_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Baseline, the inject's own inventory read, then the post-injection read
    # that finds the probe Pod gone.
    snapshots = iter(
        [{"boot_id": "b", "gpu_inventory": [{"pci_bdf": "x"}], "services": {}}] * 2
    )

    class Fixture:
        node = "node-a"
        regional = object()

        def snapshot(self) -> dict:
            value = next(snapshots, None)
            if value is None:
                raise RuntimeError("probe pod evicted")
            return value

        def execute(self, *arguments: str, timeout: int = 180) -> dict:
            return {}

        def wait_marker(self, marker: str, **_: Any) -> dict:
            return {
                "workflows": [{"status": "BLOCKED", "official_steps": []}],
                "incidents": [{"incident_id": "inc-78"}],
            }

    cleanup = acceptance.CaseCleanup()
    monkeypatch.setattr(acceptance, "firmware_premise", lambda *a, **k: {})
    with pytest.raises(RuntimeError, match="probe pod evicted"):
        acceptance.run_collect010(
            Fixture(),
            tmp_path,
            1,
            "v1",
            cleanup=cleanup,  # type: ignore[arg-type]
        )
    assert [state["incidents"] for _, state in cleanup.incident_states] == [
        [{"incident_id": "inc-78"}]
    ], "the finally block in execute_case can still restore the node"


def _collect009_fixture(
    *, taints: list[dict[str, str]], provider_events: list[dict[str, str]]
) -> tuple[Any, list[tuple[str, ...]]]:
    kubectl_calls: list[tuple[str, ...]] = []
    reads = {"count": 0}

    class Regional:
        def node_snapshot(self, _node: str) -> dict:
            return {
                "ready": "True",
                "unschedulable": False,
                "taints": taints,
                "ownership_annotations": {},
            }

        def kubectl(self, *args: str, **_: Any) -> str:
            kubectl_calls.append(args)
            return ""

        def provider_events(self, *_: Any) -> list:
            return provider_events

        @staticmethod
        def provider_events_provisional(_ended_at: datetime) -> bool:
            return True

    waiting = {
        "status": "RUNNING",
        "step_executions": [
            {
                "operation": "CHECK_MECHANICALS",
                "status": "WAITING",
                "details": {
                    "required_annotation": "gpu-fault.io/mechanical-inspection-complete",
                    "required_annotation_value": "inc-54:1",
                },
            }
        ],
    }

    class Fixture:
        node = "node-a"
        regional = Regional()

        def snapshot(self) -> dict:
            return {
                "boot_id": "boot-1",
                "gpu_inventory": [{"pci_bdf": "x"}],
                "services": {},
            }

        def execute(self, *arguments: str, timeout: int = 180) -> dict:
            return {}

        def store_snapshot(self, marker: str, **_: Any) -> dict:
            reads["count"] += 1
            return {"workflows": [waiting], "commands": [], "incidents": []}

        def wait_marker(self, marker: str, **_: Any) -> dict:
            return {"workflows": [{"status": "SUCCEEDED"}], "incidents": []}

    return Fixture(), kubectl_calls


def test_collect009_observes_the_refusal_and_reads_the_node_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(acceptance, "time", clock)
    fixture, kubectl_calls = _collect009_fixture(taints=[], provider_events=[])
    cleanup = acceptance.CaseCleanup()

    result = acceptance.run_collect009(fixture, tmp_path, 1, cleanup=cleanup)

    assert result["verdict"] == "PASS", result["errors"]
    assert result["wrong_acknowledgement_samples"] >= 3, (
        "three dispatcher cycles with the wrong value, each one read"
    )
    assert result["provider_events_provisional"] is True
    assert 10 not in clock.sleeps, "no fixed 10s sleep"
    removals = [call for call in kubectl_calls if call[-1].endswith("-")]
    assert len(removals) == 1 and cleanup.annotations == []


def test_collect009_fails_a_node_that_was_tainted_or_mutated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acceptance, "time", FakeClock())
    fixture, _ = _collect009_fixture(
        taints=[{"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}],
        provider_events=[{"event_name": "BatchRebootClusterNodes"}],
    )

    result = acceptance.run_collect009(fixture, tmp_path, 1)

    assert result["verdict"] == "FAIL"
    assert any(
        "gpu-fault taints" in item and "during WAITING" in item
        for item in result["errors"]
    ), "gpu-fault taints during WAITING fail the case"
    assert any("provider mutations" in item for item in result["errors"]), (
        "provider mutations fail the case"
    )


def test_collect011_judges_the_scope_dependent_workflow_not_the_latest(
    tmp_path: Path,
) -> None:
    class Fixture:
        def __init__(self, node: str) -> None:
            self.node = node
            self.sxid = 11001
            self.scope = "ACCESS" if node == "a" else "UNKNOWN"
            self.regional = SimpleNamespace(
                settings=SimpleNamespace(cluster_id="cluster-a"),
                business_workloads=lambda _: [],
                cpu_python=lambda *a: {
                    "cluster_id": "cluster-a",
                    "node_id": self.node,
                    "workload_state": "IDLE",
                    "link_scope": self.scope,
                    "link_scope_source": "NVIDIA_PRODUCT_INVARIANT",
                    "participating_gpu_uuids": [],
                },
            )

        def snapshot(self) -> dict:
            return {
                "boot_id": "boot",
                "gpu_inventory": [{"pci_bdf": "0000:59:00.0"}],
                "services": {},
            }

        def execute(self, *arguments: str, timeout: int = 180) -> dict:
            if arguments[0] == "append-sxid":
                self.sxid = int(arguments[arguments.index("--sxid") + 1])
            return {}

        def wait_marker(self, marker: str, **_: Any) -> dict:
            return {
                "workflows": [
                    {
                        "request_id": "wf-restore",
                        "status": "SUCCEEDED",
                        "official_action": "RESTORE",
                        "official_steps": [{"operation": "RESTORE_SCHEDULING"}],
                    },
                    {
                        "request_id": "wf-blocked",
                        "incident_id": f"inc-{self.node}",
                        "status": "BLOCKED",
                        "blocked_kind": "SAFETY_SETTLED",
                        "official_action": SCOPE_ACTIONS[self.scope],
                        "official_steps": [{"operation": "RESET_ALL_GPUS_NVSWITCHES"}],
                    },
                ],
                "incidents": [{"incident_id": f"inc-{self.node}", "gpu_uuids": []}],
                "commands": [],
                "fabric_events": [
                    {
                        "event_id": "sxid-a",
                        "sxid": self.sxid,
                        "classification": "FATAL",
                        "port": "12",
                        "switch_id": "nvidia-nvswitch0"
                        if self.scope == "ACCESS"
                        else None,
                    }
                ],
                "decisions": [
                    {
                        "event_id": "sxid-a",
                        "event_type": "SXID",
                        "incident_id": f"inc-{self.node}",
                        "workflow_request_id": "wf-blocked",
                        "official_action": SCOPE_ACTIONS[self.scope],
                        "disposition": "BLOCKED_MISSING_EVIDENCE",
                        "action": None,
                        "reasons": [SCOPE_REASONS[self.scope]],
                    }
                ],
            }

        def restore_incidents(self, state: dict, **_: Any) -> list:
            return [{"status": "SUCCEEDED"}]

    result = acceptance.run_collect011(
        [Fixture("a"), Fixture("b")],
        tmp_path,
        1,
        "v1",  # type: ignore[list-item]
    )

    assert result["verdict"] == "PASS", result["errors"]
    assert result["selected_workflow_ids"] == ["wf-blocked"] * 6


def test_focused_tests_are_reused_from_the_plan_on_the_same_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        acceptance,
        "reusable_focused_tests",
        lambda path: {"passed": True, "returncode": 0, "command": ["pytest"]},
    )

    def never_run(*_: Any, **__: Any) -> Any:
        raise AssertionError("pytest must not run twice on one tree")

    monkeypatch.setattr(acceptance.RegionalLiveFixture, "run", staticmethod(never_run))
    result = acceptance.focused_tests(
        "GF-REGIONAL-COLLECT-003", tmp_path, reuse_plan=tmp_path / "plan.json"
    )
    assert result["passed"] is True
    assert result["reused_from_plan"] == str(tmp_path / "plan.json")

    monkeypatch.setattr(acceptance, "reusable_focused_tests", lambda path: None)
    monkeypatch.setattr(
        acceptance.RegionalLiveFixture,
        "run",
        staticmethod(
            lambda *_, **__: SimpleNamespace(returncode=0, stdout="ok", stderr="")
        ),
    )
    fresh = acceptance.focused_tests(
        "GF-REGIONAL-COLLECT-003", tmp_path, reuse_plan=tmp_path / "plan.json"
    )
    assert fresh["passed"] is True and "reused_from_plan" not in fresh


def test_plan_details_records_the_focused_tests_with_their_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(
        acceptance,
        "record_focused_tests",
        lambda details, result: recorded.append(result),
    )
    settings = SimpleNamespace(case_id="GF-REGIONAL-COLLECT-003", nodes=("n",))
    preflight = {
        "predecessor": {},
        "release_id": "r",
        "nodes": [{"uid": "u"}],
        "focused_tests": {"passed": True},
    }
    acceptance.plan_details(settings, preflight)  # type: ignore[arg-type]
    assert recorded == [{"passed": True}]

    monkeypatch.setattr(
        destructive,
        "record_focused_tests",
        lambda details, result: recorded.append(result),
    )
    destructive.plan_details(
        SimpleNamespace(case_id="GF-REGIONAL-COLLECT-013", node="n", xid=109),  # type: ignore[arg-type]
        {
            "predecessor": {},
            "release_id": "r",
            "node": {"uid": "u"},
            "store": {},
            "focused_tests": {"passed": False},
        },
    )
    assert recorded[-1] == {"passed": False}


def test_read_only_preflight_binds_the_predecessor_to_release_and_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    class Regional:
        def __init__(self, _settings: Any) -> None:
            pass

        def node_snapshot(self, node: str) -> dict:
            return {
                "name": node,
                "uid": "u",
                "ready": "True",
                "ownership_annotations": {},
            }

        def store_snapshot(self, *, node: str) -> dict:
            return {"agent": {"lifecycle_state": "ACTIVE"}}

        def evidence_identity(self) -> dict[str, str]:
            return {"release_id": "rel-1", "cluster_id": "cl-1"}

        def cpu_blast_snapshot(self) -> dict:
            return {}

    def predecessor(path: Path, case_id: str, **kwargs: Any) -> dict:
        seen.update(kwargs)
        return {"valid": True}

    monkeypatch.setattr(acceptance, "RegionalLiveFixture", Regional)
    monkeypatch.setattr(acceptance, "predecessor_evidence", predecessor)
    monkeypatch.setattr(acceptance, "focused_tests", lambda *_, **__: {"passed": True})
    settings = acceptance.Settings(
        regional=object(),  # type: ignore[arg-type]
        case_id="GF-REGIONAL-COLLECT-003",
        nodes=("node-a",),
        host_probe_image="img",
        predecessor_path=tmp_path / "p.json",
    )

    result = acceptance.read_only_preflight(settings, tmp_path)

    assert seen == {"release_id": "rel-1", "cluster_id": "cl-1"}
    assert result["release_id"] == "rel-1" and result["cluster_id"] == "cl-1"
    assert result["errors"] == []


# --------------------------------------------------------------------------- #
# run_collector_destructive
# --------------------------------------------------------------------------- #


def _destructive_settings(case_id: str, **overrides: Any) -> destructive.Settings:
    values: dict[str, Any] = {
        "regional": SimpleNamespace(
            cluster_id="hp-cluster", gpu_kubeconfig="k", gpu_context="c", namespace="n"
        ),
        "case_id": case_id,
        "node": "hyperpod-node",
        "second_node": None,
        "host_probe_image": "img",
        "hyperpod_cluster": "hp",
        "executor_role_arn": "arn:aws:iam::1:role/executor",
        "site_file": None,
        "predecessor_path": Path("/tmp/predecessor.json"),
    }
    values.update(overrides)
    return destructive.Settings(**values)


def test_collect013_runs_one_reset_cycle_for_the_selected_xid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []

    def run_single_reset(*_: Any, xid: int, marker: str, run_id: str, **__: Any):
        seen.append(xid)
        return {"workflow": {"request_id": f"wf-{xid}"}}, []

    monkeypatch.setattr(destructive, "run_single_reset", run_single_reset)
    settings = _destructive_settings("GF-REGIONAL-COLLECT-013", xid=62)

    result = destructive.run_collect013(
        settings,
        object(),
        object(),
        tmp_path,
        1,
        collector=ResetAudit(),
        cleanup=acceptance.CaseCleanup(),
    )  # type: ignore[arg-type]

    assert seen == [62], "one quiesce/reset cycle, for the XID the operator chose"
    assert result["verdict"] == "PASS" and len(result["runs"]) == 1
    assert _destructive_settings("GF-REGIONAL-COLLECT-013").xid == 109
    xid_option = next(
        a for a in destructive.parser()._actions if "--xid" in a.option_strings
    )
    assert tuple(xid_option.choices) == (109, 62) and xid_option.default == 109
    monkeypatch.setattr(destructive, "record_focused_tests", lambda *_: None)
    plan = destructive.plan_details(
        settings,
        {
            "predecessor": {},
            "release_id": "r",
            "node": {"uid": "u"},
            "store": {},
            "focused_tests": {},
        },
    )
    assert "XID62" in plan["mutation"] and "one cycle" in plan["mutation"]


class _Host:
    host_script = "/probe.py"

    def __init__(
        self, *, stop_raises: bool = False, boot_after: str = "boot-1"
    ) -> None:
        self.calls: list[str] = []
        self.stop_raises = stop_raises
        self.boot_after = boot_after

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        self.calls.append(arguments[0])
        if arguments[0] == "stop-reset-sampler" and self.stop_raises:
            raise RuntimeError("systemctl stop hung")
        boot = self.boot_after if "--since-epoch" in arguments else "boot-1"
        return {
            "gpu_inventory": [{"pci_bdf": "0000:59:00.0"}],
            "ledger": [],
            "boot_id": boot,
        }


def test_run_single_reset_stops_the_sampler_on_every_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _destructive_settings("GF-REGIONAL-COLLECT-013")

    def wait_raises(*_: Any, **__: Any) -> dict:
        raise RuntimeError("store unreachable")

    monkeypatch.setattr(destructive, "wait_xid_workflow", wait_raises)
    host = _Host()
    with pytest.raises(RuntimeError, match="store unreachable"):
        destructive.run_single_reset(
            settings,
            object(),
            host,
            tmp_path,
            collector=ResetAudit(),
            cleanup=acceptance.CaseCleanup(),
            xid=109,
            marker="m",
            run_id="r",  # type: ignore[arg-type]
        )
    assert host.calls[-1] == "stop-reset-sampler", host.calls

    monkeypatch.setattr(
        destructive, "wait_xid_workflow", lambda *_, **__: {"workflow": {}}
    )
    monkeypatch.setattr(destructive.reset_case, "workflow_errors", lambda *_, **__: [])
    monkeypatch.setattr(destructive.reset_case, "host_errors", lambda *_, **__: [])
    host = _Host(stop_raises=True, boot_after="boot-2")
    _state, errors = destructive.run_single_reset(
        settings,
        object(),
        host,
        tmp_path,
        collector=ResetAudit(),
        cleanup=acceptance.CaseCleanup(),
        xid=109,
        marker="m",
        run_id="r",  # type: ignore[arg-type]
    )
    assert any("sampler stop failed" in item for item in errors), errors
    assert any("boot ID changed" in item for item in errors), errors


def test_collect008_reads_the_solo_xid63_by_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destructive, "time", FakeClock())
    audit = ResetAudit()
    monkeypatch.setattr(destructive, "wait_xid_workflow", lambda *_, **__: audit.state)
    monkeypatch.setattr(destructive.reset_case, "workflow_errors", lambda *_, **__: [])
    monkeypatch.setattr(destructive.reset_case, "host_errors", lambda *_, **__: [])
    seen: dict[str, Any] = {}

    class Regional:
        def store_snapshot(self, **kwargs: Any) -> dict:
            seen.update(kwargs)
            return {
                "event": {"xid": 63, "evidence_ref": "kmsg://node/boot/1"},
                "decision": {"disposition": "MONITOR_ONLY"},
                "workflow": None,
            }

    host = _Host()
    result = destructive.run_collect008(
        _destructive_settings("GF-REGIONAL-COLLECT-008"),
        Regional(),
        host,
        tmp_path,
        1,  # type: ignore[arg-type]
        collector=audit,
        cleanup=acceptance.CaseCleanup(),
    )
    assert result["verdict"] == "PASS", result["errors"]
    assert seen["marker"].startswith("c008-63-") and seen["queue_attempts"] == 1
    assert seen["observed_after"] is not None
    assert host.calls[-1] == "stop-reset-sampler"

    bad = destructive.companion_xid_errors(
        {
            "event": {"xid": 63, "evidence_ref": "api://x"},
            "decision": {"disposition": "EXECUTABLE"},
            "workflow": {"request_id": "wf"},
        }
    )
    assert len(bad) == 3, bad


def _finding(
    consecutive: int, required: int, offset: float, *, value: int | None = None
) -> dict[str, Any]:
    return {
        "record_id": f"ev-{consecutive}",
        "observed_at": _stamp(offset),
        "edge_filter_reasons": ["baseline"] if consecutive == 1 else ["threshold"],
        "samples": [
            {
                "name": "gpu_inventory_mismatch",
                "value": int(consecutive >= required) if value is None else value,
                "labels": {
                    "consecutive_mismatch_samples": str(consecutive),
                    "required_consecutive_samples": str(required),
                    "expected_count": "9",
                    "observed_count": "8",
                },
            }
        ],
    }


def test_debounce_errors_demand_the_second_sample_at_about_two_intervals() -> None:
    started = datetime.now(timezone.utc)
    ok = destructive.debounce_errors(
        [_finding(1, 2, 0), _finding(2, 2, 15)],
        interval=15,
        required_samples=2,
        started_at=started,
        tolerance=0.5,
    )
    assert ok == [], ok
    first = destructive.debounce_errors(
        [_finding(1, 2, 15, value=1)],
        interval=15,
        required_samples=2,
        started_at=started,
        tolerance=0.5,
    )
    assert any("first mismatching sample" in item for item in first), first
    late = destructive.debounce_errors(
        [_finding(1, 2, 0), _finding(2, 2, 120)],
        interval=15,
        required_samples=2,
        started_at=started,
        tolerance=0.5,
    )
    assert any("latency" in item for item in late), late
    none = destructive.debounce_errors(
        [], interval=15, required_samples=2, started_at=started, tolerance=0.5
    )
    assert none == ["no bound gpu_inventory_mismatch samples reached the control plane"]
    off = destructive.debounce_errors(
        [_finding(1, 1, 15)],
        interval=15,
        required_samples=1,
        started_at=started,
        tolerance=0.5,
    )
    assert any("switched off" in item for item in off), off


def test_fail_closed_post_is_refused_when_inventory_could_vouch_for_it() -> None:
    now = datetime.now(timezone.utc)
    event_time = now - destructive.FAIL_CLOSED_EVENT_AGE
    fresh = {
        "present": True,
        "observed_at": (event_time + timedelta(seconds=60)).isoformat(),
        "legacy_observed_at": [],
    }
    assert destructive.fail_closed_timing_errors(fresh, event_time=event_time), (
        "an event observed only after the fail-closed age is not fresh"
    )
    safe = {
        "present": True,
        "observed_at": (event_time + timedelta(seconds=90)).isoformat(),
        "legacy_observed_at": [],
    }
    assert destructive.fail_closed_timing_errors(safe, event_time=event_time) == []
    assert (
        destructive.fail_closed_timing_errors(
            {"present": False, "legacy_observed_at": []}, event_time=event_time
        )
        == []
    )


def _collect014_stubs(*, inventory_offset_seconds: float):
    from tests.regional._alignment_collector_support import (
        SequenceResetAudit,
        restored_node,
    )

    calls: list[str] = []
    seen: dict[str, Any] = {}
    audit = SequenceResetAudit()

    class Regional:
        def node_snapshot(self, node: str) -> dict:
            return restored_node()

        def business_workloads(self, node: str) -> list:
            return []

        def cpu_python(self, script: str, *args: str, **_: Any) -> dict:
            assert script is destructive.GPU_INVENTORY_SNAPSHOT
            calls.append("inventory")
            observed = datetime.now(timezone.utc) - destructive.FAIL_CLOSED_EVENT_AGE
            observed += timedelta(seconds=inventory_offset_seconds)
            return {
                "present": True,
                "observed_at": observed.isoformat(),
                "legacy_observed_at": [],
            }

        def executor_python(self, script: str, payload: str, **kwargs: Any) -> dict:
            calls.append("fabric-post")
            seen["post_kwargs"] = kwargs
            return {}

    class Collector:
        def wait_marker(self, marker: str, **kwargs: Any) -> dict:
            seen[marker.split("-")[1]] = kwargs
            if marker.startswith("c014-fail-"):
                calls.append("wait-fail")
                return {
                    "workflows": [
                        {
                            "status": "BLOCKED",
                            "blocked_kind": "SAFETY_SETTLED",
                            "official_action": "RESET_ALL_GPUS_AND_NVSWITCHES",
                            "blocked_reasons": [
                                f"SXID 10003 {destructive.FAIL_CLOSED_REASON}"
                            ],
                        }
                    ],
                    "incidents": [{"incident_id": "inc-fail"}],
                    "commands": [],
                }
            calls.append("wait-positive")
            return deepcopy(audit.state)

        def execute(self, *arguments: str, timeout: int = 180) -> dict:
            if arguments[0] == "reset-audit":
                return audit.execute(*arguments, timeout=timeout)
            calls.append("append-sxid")
            audit.start_variant(int(arguments[arguments.index("--sxid") + 1]))
            return {}

        def restore_incidents(self, state: dict, **_: Any) -> list:
            calls.append("restore")
            return [{"status": "SUCCEEDED"}]

    class ResetHost(_Host):
        def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
            result = super().execute(*arguments, timeout=timeout)
            result.update(deepcopy(audit.current))
            return result

    return Regional(), ResetHost(), Collector(), calls, seen


def test_collect014_refuses_to_post_when_the_stored_inventory_is_too_fresh(
    tmp_path: Path,
) -> None:
    regional, host, collector, calls, _ = _collect014_stubs(inventory_offset_seconds=45)
    result = destructive.run_collect014(
        _destructive_settings("GF-REGIONAL-COLLECT-014"),
        regional,
        host,
        collector,
        tmp_path,
        1,
        "v1",  # type: ignore[arg-type]
    )
    assert result["verdict"] == "FAIL"
    assert "fabric-post" not in calls and "append-sxid" not in calls, calls
    assert any("not posted" in item for item in result["errors"]), result["errors"]


def test_collect014_posts_once_and_scopes_both_waits_by_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destructive.time, "sleep", lambda seconds: None)
    regional, host, collector, calls, seen = _collect014_stubs(
        inventory_offset_seconds=300
    )
    result = destructive.run_collect014(
        _destructive_settings("GF-REGIONAL-COLLECT-014"),
        regional,
        host,
        collector,
        tmp_path,
        1,
        "v1",  # type: ignore[arg-type]
    )
    assert result["verdict"] == "PASS", result["errors"]
    assert len(result["positive_variants"]) == 2
    assert seen["post_kwargs"] == {"attempts": 1}, "a mutation is never retried"
    assert seen["fail"]["observed_after"] is not None, "fail-closed wait is time-scoped"
    assert seen["full"]["observed_after"] is not None
    assert calls.index("inventory") < calls.index("fabric-post")
    assert calls[-1] == "restore", calls
    assert host.calls[-2:] == ["stop-reset-sampler", "snapshot"], host.calls
    assert "restore-quiesce" not in host.calls


def test_collect004_tracks_reboot_without_modifying_the_production_collector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.regional._collector_inventory_reboot_support import InventoryHarness

    harness = InventoryHarness(monkeypatch, tmp_path)
    original_state = harness.state

    def state() -> dict[str, Any]:
        value = original_state()
        value["workflows"][0].update(
            request_id="wf-reboot",
            official_steps=[
                {"operation": "FREEZE_EVIDENCE", "node_ids": ["node-a"]},
                {"operation": "RESTART_NODE", "node_ids": ["node-a"]},
            ],
        )
        value["node_workflow_ids"] = ["wf-reboot"]
        return value

    monkeypatch.setattr(harness, "state", state)
    result = harness.run()

    calls = harness.calls
    assert result["verdict"] == "PASS", result["errors"]
    assert "restore-collector-env" not in calls
    assert "override-expected-gpu-count" not in calls
    assert calls.index("sample-gpu-inventory") < calls.index("publish-gpu-inventory")
    assert calls.index("recovery") < calls.index("node-ready")
    assert "provider-events" in calls
    assert result["restart_workflow_id"] == "wf-reboot"
    assert not any(s == 45 for s in harness.clock.sleeps), "no sleep(interval * 3)"
    assert result["production_configuration_modified"] is False
    assert result["isolated_sampling"]["live_delivery_proven"] is True
    assert calls.count("publish-gpu-inventory") == 1
    assert result["cleanup_errors"] == []
    assert len(harness.cleanup.state_readers) == 1
    assert harness.cleanup.finish(profile_version="v1", reason="test")["errors"] == []
    assert harness.calls[-1] == "incident-cleanup"


@pytest.mark.parametrize("problem", ["digest", "extra-workflow", "failed-workflow"])
def test_collect004_rejects_config_drift_and_unsuccessful_product_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    import json

    from tests.regional._collector_inventory_reboot_support import InventoryHarness

    harness = InventoryHarness(monkeypatch, tmp_path)
    harness.problem = "digest" if problem == "digest" else None
    original_state = harness.state

    def state() -> dict[str, Any]:
        value = original_state()
        if harness.rebooted and problem in {"extra-workflow", "failed-workflow"}:
            value["workflows"][0]["status"] = "FAILED"
            if problem == "extra-workflow":
                value["workflows"].append(
                    {
                        **value["workflows"][0],
                        "request_id": "wf-2",
                        "official_steps": [
                            {"operation": "REPLACE_NODE", "node_ids": ["node-a"]}
                        ],
                    }
                )
        value["node_workflow_ids"] = [
            workflow["request_id"] for workflow in value["workflows"]
        ]
        return value

    monkeypatch.setattr(harness, "state", state)
    result = harness.run()

    assert result["verdict"] == "FAIL"
    errors = result["errors"]
    expected_error = {
        "digest": "collector.env differs from the baseline",
        "extra-workflow": "grew 2 node workflows",
        "failed-workflow": "not SUCCEEDED",
    }[problem]
    assert any(expected_error in item for item in errors), errors
    assert harness.restores == 0
    assert "restore-collector-env" not in harness.calls
    if problem == "digest":
        released = harness.cleanup.finish(profile_version="v1", reason="test")
        assert any("configuration changed" in item for item in released["errors"])
        assert "incident-cleanup" not in harness.calls
    else:
        assert result["production_configuration_modified"] is False
    stored = json.loads((tmp_path / "inventory-reboot-progress-a1.json").read_text())
    assert stored["verdict"] == "FAIL"
    assert stored["errors"] == errors
    assert stored["cleanup_errors"] == result["cleanup_errors"]
