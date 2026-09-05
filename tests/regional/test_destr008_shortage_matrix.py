"""DESTR-008's shortage matrix: when each mutation lands, and who cleans up.

Both tested behaviours came from live failures. The service-stop scenarios used
to inject before the workload was submitted, so most of the bounded failsafe
window was spent waiting for a Pod to start. And the fault-node cleanup used to
recognise only its own incident, which the product's own successor escalation
takes over.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr008_warm_spare_shortage as destr008
from scripts.e2e.regional.regional_live_fixture import RegionalLiveSettings

ROOT = Path(__file__).resolve().parents[2]
REGIONAL = ROOT / "scripts/e2e/regional"


def _settings(tmp_path: Path) -> destr008.Settings:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    return destr008.Settings(
        regional=RegionalLiveSettings(
            cpu_kubeconfig=cpu,
            gpu_kubeconfig=gpu,
            gpu_context="gpu-context",
            namespace="gpu-fault-system",
            cluster_id="cluster-a",
            region="us-west-2",
        ),
        site_file=tmp_path / "site.yaml",
        manifest=REGIONAL / "manifests/training/single-node-warm-spare-pytorchjob.yaml",
        hyperpod_cluster="hp-cluster",
        fault_node="node-a",
        spare_node="node-b",
        host_probe_image="registry.example/probe@sha256:" + "a" * 64,
        scenarios=destr008.SCENARIOS,
        predecessor_path=tmp_path / "predecessor.json",
    )


class _RestoreRecorder:
    """The few fixture calls DESTR-008's fault-node cleanup makes."""

    def __init__(self, owner: str | None) -> None:
        self.owner = owner
        self.restored_incidents: list[str] = []
        self.waited: list[str] = []

    def release_spares(self, nodes: list[str], incident_id: str) -> dict[str, Any]:
        return {"nodes": nodes, "incident_id": incident_id}

    def reactivate_agent(self, node: str) -> dict[str, Any]:
        return {"node": node}

    def wait_agent_active(self, node: str) -> dict[str, Any]:
        return {"node": node, "lifecycle_state": "ACTIVE"}

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return {"name": node, "annotations": {"gpu-fault.io/incident-id": self.owner}}

    def wait_incident_idle(self, incident_id: str) -> dict[str, Any]:
        self.waited.append(incident_id)
        return {"incident_id": incident_id}

    def create_restore_workflow(
        self, *, incident_id: str, node: str, profile_version: str, reason: str
    ) -> dict[str, Any]:
        self.restored_incidents.append(incident_id)
        return {"workflow_request_id": f"workflow-validated-restore-{incident_id}"}

    def wait_workflow_id(self, workflow_id: str) -> dict[str, Any]:
        return {"request_id": workflow_id, "status": "SUCCEEDED"}


def test_destr008_cleanup_restores_a_successor_owned_fault_node(tmp_path: Path) -> None:
    # When the shortage blocks recovery the product escalates on its own: a
    # successor support incident re-quarantines the same node under its id.
    # Cleanup that recognised only its own incident left the node cordoned and
    # reported no error, so the case failed in postflight for a condition its
    # own cleanup had already seen.
    settings = _settings(tmp_path)
    successor = "inc-support-after-workflow-a"
    warm = _RestoreRecorder(successor)

    result = destr008.restore_fault_node(
        warm, settings=settings, incident_id="incident-a", profile_version="hyperpod-v1"
    )

    assert result["errors"] == [], result
    assert result["successor_incident"] == successor, result
    assert warm.restored_incidents == [successor], warm.restored_incidents
    assert warm.waited == [successor], warm.waited


def test_destr008_cleanup_restores_nothing_when_the_node_is_unowned(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    warm = _RestoreRecorder(None)

    result = destr008.restore_fault_node(
        warm, settings=settings, incident_id="incident-a", profile_version="hyperpod-v1"
    )

    assert result["errors"] == [], result
    assert result["quarantine_owner"] is None, result
    assert warm.restored_incidents == [], warm.restored_incidents


class _ServiceRecorder:
    """Stands in for the host-probe fixture the service scenarios drive."""

    def __init__(self, warm: Any, *, node: str, image: str, case_id: str, run_id: str):
        self.node = node
        self.created = False
        self.stops: list[dict[str, Any]] = []

    def create(self) -> None:
        self.created = True

    def stop(
        self, service: str, *, restore_seconds: int, delay_seconds: int
    ) -> dict[str, Any]:
        self.stops.append(
            {
                "service": service,
                "restore_seconds": restore_seconds,
                "delay_seconds": delay_seconds,
            }
        )
        return {"service": service, "scheduled": bool(delay_seconds)}


class _WaitRecorder:
    def __init__(self) -> None:
        self.node_ready: list[bool] = []
        self.fleet_ready: list[bool] = []

    def wait_node_ready(self, node: str, *, ready: bool, timeout_seconds: int) -> None:
        self.node_ready.append(ready)

    def wait_fleet_readiness(
        self, node: str, *, ready: bool, timeout_seconds: int
    ) -> None:
        self.fleet_ready.append(ready)


def _fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: str
) -> tuple[destr008.ScenarioFixture, _WaitRecorder, list[_ServiceRecorder]]:
    built: list[_ServiceRecorder] = []

    def build(*args: Any, **kwargs: Any) -> _ServiceRecorder:
        recorder = _ServiceRecorder(*args, **kwargs)
        built.append(recorder)
        return recorder

    monkeypatch.setattr(destr008, "WarmSpareServiceFixture", build)
    warm = _WaitRecorder()
    return (
        destr008.ScenarioFixture(
            _settings(tmp_path),
            warm,
            scenario=scenario,
            run_id=f"destr008-{scenario}-2",
        ),
        warm,
        built,
    )


@pytest.mark.parametrize(
    ("scenario", "service", "delay"),
    [
        ("kubernetes-not-ready", "kubelet.service", 15),
        ("agent-unavailable", "gpu-fault-node-agent.service", 0),
    ],
)
def test_a_service_scenario_stops_the_unit_only_after_the_workload_is_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scenario: str,
    service: str,
    delay: int,
) -> None:
    # The stop is bounded by the failsafe timer that restores it. Injecting it
    # before the workload was submitted spent most of that window waiting for a
    # Pod to start, leaving little of it for the workflow the case is about.
    fixture, warm, built = _fixture(monkeypatch, tmp_path, scenario)

    early = fixture.apply()

    assert built[0].created is True, built
    assert built[0].stops == [], built[0].stops
    assert early == {"service": service, "host_probe": "created"}, early
    assert warm.node_ready == [] and warm.fleet_ready == []

    late = fixture.apply_late()

    assert built[0].stops == [
        {
            "service": service,
            "restore_seconds": destr008.SERVICE_RESTORE_SECONDS,
            "delay_seconds": delay,
        }
    ], built[0].stops
    assert late["service"] == service, late
    if scenario == "kubernetes-not-ready":
        # kubelet carries the probe's own exec channel, so the stop is handed to
        # a systemd timer and the transition is observed from the API instead.
        assert late["stopped"]["scheduled"] is True, late
        assert warm.node_ready == [False], warm.node_ready
    else:
        assert late["stopped"]["scheduled"] is False, late
        assert warm.fleet_ready == [False], warm.fleet_ready


def test_a_label_scenario_has_nothing_left_to_inject_late(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    applied: list[Any] = []

    class _NodeMutation:
        def __init__(self, warm: Any, node: str, **kwargs: Any) -> None:
            self.node = node

        def apply(self, patch: Any) -> None:
            applied.append(patch)

    monkeypatch.setattr(destr008, "NodeMutationFixture", _NodeMutation)
    fixture, warm, built = _fixture(monkeypatch, tmp_path, "no-spare")

    assert fixture.apply() == {"removed_spare_label": True}
    assert fixture.apply_late() == {}
    assert built == [], built
    assert len(applied) == 1, applied


def _shortage_state(
    scenario: str,
    *,
    alert: bool,
    event_id: str = "destr008-event",
    markers: list[str] | None = None,
) -> dict[str, Any]:
    """A store snapshot of a shortage scenario that is otherwise clean.

    ``alert`` is passed explicitly rather than derived from
    ``destr008.ALERT_SCENARIOS`` so that the notification expectation is stated
    twice -- once here as the store's contents, once in the runner -- instead of
    agreeing with itself.
    """

    incident_id = "incident-a"
    if markers is None:
        markers = [f"marker-{event_id}"]
    return {
        "incident": {"incident_id": incident_id},
        "workflow": {
            "status": "FAILED",
            "official_steps": [
                {
                    "operation": "REPLACE_NODE",
                    "parameters": {"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
                }
            ],
            "step_executions": [
                {"operation": "STOP_WORKLOADS", "status": "SUCCEEDED"},
                {
                    "operation": "REPLACE_NODE",
                    "status": "FAILED",
                    "error": destr008.EXPECTED_REASON[scenario],
                },
            ],
        },
        "notifications": (
            [{"deduplication_key": f"{incident_id}/hyperpod-spare-insufficient"}]
            if alert
            else []
        ),
        "markers": [{"marker_id": item} for item in markers],
        "fault_node": {
            "unschedulable": True,
            "taints": [{"key": destr008.QUARANTINE_TAINT}],
        },
        "spare_node": {
            "annotations": {
                "gpu-fault.io/spare-reservation": None,
                "gpu-fault.io/spare-pool-state": "AVAILABLE",
            }
        },
    }


def test_notification_semantics_distinguish_no_pool_from_shortage(
    tmp_path: Path,
) -> None:
    # An empty pool is the operator's own configuration, so it raises nothing;
    # a pool that exists but cannot satisfy the request has to page someone.
    settings = _settings(tmp_path)

    no_pool = destr008.scenario_errors(
        _shortage_state("no-spare", alert=False),
        settings,
        "no-spare",
        event_id="destr008-event",
    )
    shortage = destr008.scenario_errors(
        _shortage_state("topology-mismatch", alert=True),
        settings,
        "topology-mismatch",
        event_id="destr008-event",
    )

    assert no_pool == [], no_pool
    assert shortage == [], shortage


def test_the_findings_own_marker_is_expected_on_the_shortage_incident(
    tmp_path: Path,
) -> None:
    # The injected finding's marker belongs to the incident that owns it. It
    # only looked like "the shortage path created no marker" while the marker
    # pointed at an incident nobody persisted -- the same dangling reference
    # that let a terminal attempt open a second recovery and re-quarantine the
    # fault node minutes into the next scenario.
    event_id = "destr008-no-spare-1788587041"
    state = _shortage_state("no-spare", alert=False, event_id=event_id)

    errors = destr008.scenario_errors(
        state, _settings(tmp_path), "no-spare", event_id=event_id
    )

    assert errors == [], errors


def test_any_other_marker_on_the_shortage_incident_is_an_error(tmp_path: Path) -> None:
    event_id = "destr008-no-spare-1788587041"
    state = _shortage_state(
        "no-spare",
        alert=False,
        event_id=event_id,
        markers=[f"marker-{event_id}", "marker-quick-triage"],
    )

    errors = destr008.scenario_errors(
        state, _settings(tmp_path), "no-spare", event_id=event_id
    )

    assert len(errors) == 1, errors
    assert "marker-quick-triage" in errors[0], errors


def _warm(tmp_path: Path) -> Any:
    from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture
    from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture

    settings = _settings(tmp_path)
    return WarmSpareLiveFixture(RegionalLiveFixture(settings.regional), "hp-cluster")


def test_a_node_whose_kubelet_stopped_reporting_counts_as_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stopping kubelet does not make it report unhealthy, it makes it stop
    # reporting: the node lifecycle controller sets Ready=Unknown and taints the
    # node unreachable. Insisting on the literal "False" waited out the whole
    # window over a snapshot that plainly showed the node was not Ready.
    warm = _warm(tmp_path)
    snapshot = {"name": "node-b", "ready": "Unknown"}
    monkeypatch.setattr(warm, "node_snapshot", lambda node: snapshot)

    assert warm.wait_node_ready("node-b", ready=False, timeout_seconds=30) == snapshot


def test_waiting_for_ready_is_not_satisfied_by_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warm = _warm(tmp_path)
    states = [{"ready": "Unknown"}, {"ready": "False"}, {"ready": "True"}]
    monkeypatch.setattr(warm, "node_snapshot", lambda node: states.pop(0))
    monkeypatch.setattr(
        "scripts.e2e.regional.warm_spare_fixture.time.sleep", lambda _: None
    )

    assert warm.wait_node_ready("node-b", ready=True, timeout_seconds=30) == {
        "ready": "True"
    }
    assert states == []
