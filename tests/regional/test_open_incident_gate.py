"""A drill that injects an XID and waits for a NEW workflow must refuse a node
still owned by an open incident: the processor merges the injected event into
that incident (node-scoped merge; F-N1 once ESCALATED) and plans nothing, so
the runner would spend its whole budget waiting for a workflow that can never
appear, and only find the old FAILED one."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.models import IncidentState
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional.acceptance_runner_common import (
    OPEN_INCIDENTS_PROBE,
    node_open_incidents,
    open_incident_errors,
)
from tests._builders import fault_incident

NODE = "node-a"


def _incident(
    state: str, node: str = NODE, incident_id: str = "inc-old"
) -> dict[str, Any]:
    return {"incident_id": incident_id, "state": state, "node_ids": [node]}


def test_an_escalated_incident_is_refused_with_the_close_incident_lever() -> None:
    errors = open_incident_errors(NODE, [_incident("ESCALATED")])

    assert len(errors) == 1, errors
    assert "inc-old" in errors[0] and "ESCALATED" in errors[0], errors
    assert "workflow-reconcile --close-incident" in errors[0], errors
    assert "--close-quarantined" not in errors[0], errors


def test_a_quarantined_incident_is_refused_with_the_close_quarantined_lever() -> None:
    errors = open_incident_errors(NODE, [_incident("QUARANTINED")])

    assert len(errors) == 1, errors
    assert "workflow-reconcile --close-quarantined" in errors[0], errors


def test_an_in_flight_incident_is_refused_without_a_closure_lever() -> None:
    for state in ("DETECTED", "ACTION_PENDING", "SAFETY_PENDING"):
        errors = open_incident_errors(NODE, [_incident(state)])
        assert len(errors) == 1 and state in errors[0], (state, errors)
        assert "wait for its workflow to end" in errors[0], errors


def test_a_recovered_incident_or_another_nodes_incident_is_not_refused() -> None:
    assert open_incident_errors(NODE, [_incident("RECOVERED")]) == []
    assert open_incident_errors(NODE, [_incident("ESCALATED", node="node-b")]) == []
    assert open_incident_errors(NODE, []) == []


def test_every_incident_state_but_recovered_is_open() -> None:
    for state in IncidentState:
        errors = open_incident_errors(NODE, [_incident(state.value)])
        assert bool(errors) is (state is not IncidentState.RECOVERED), (state, errors)


def test_an_incident_without_node_ids_counts_as_the_nodes_own() -> None:
    """The probe's read is node-scoped; a row that lost the field must not be
    waved through as if it named another node."""
    errors = open_incident_errors(
        NODE, [{"incident_id": "inc-x", "state": "ESCALATED"}]
    )

    assert len(errors) == 1 and "inc-x" in errors[0], errors


def test_the_probe_reads_open_incidents_by_state_and_node_from_the_store(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pinned against the real store and model, not a fake: the read is by
    state and node, not by the node's recent XID events, because the incident a
    rerun trips over is older than any preflight's event lookback."""
    from gpu_fault.app import ApplicationContext

    store = InMemoryStore()
    for incident_id, event_id, node, cluster, state in (
        ("inc-esc", "evt-1", NODE, "cluster-a", IncidentState.ESCALATED),
        ("inc-done", "evt-2", NODE, "cluster-a", IncidentState.RECOVERED),
        ("inc-other", "evt-3", "node-b", "cluster-a", IncidentState.QUARANTINED),
        ("inc-far", "evt-4", NODE, "cluster-b", IncidentState.ESCALATED),
    ):
        store.save_incident(
            fault_incident(
                incident_id, event_id, cluster_id=cluster, node_ids=[node], state=state
            )
        )
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        classmethod(lambda _cls: SimpleNamespace(store=store)),
    )
    monkeypatch.setattr(sys, "argv", ["open-incidents", "cluster-a", NODE])

    exec(
        compile(OPEN_INCIDENTS_PROBE, "<open-incidents>", "exec"),
        {"__name__": "__probe__"},
    )
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert [item["incident_id"] for item in payload["open_incidents"]] == ["inc-esc"], (
        payload
    )
    assert payload["open_incidents"][0]["state"] == "ESCALATED"
    assert payload["open_incidents"][0]["node_ids"] == [NODE]
    assert len(open_incident_errors(NODE, payload["open_incidents"])) == 1


def test_node_open_incidents_runs_the_probe_through_the_cpu_pod() -> None:
    calls: list[tuple[str, ...]] = []

    def cpu_python(script: str, *arguments: str) -> dict[str, Any]:
        calls.append((script, *arguments))
        return {"open_incidents": [_incident("ESCALATED")]}

    assert node_open_incidents(cpu_python, "cluster-a", NODE) == [
        _incident("ESCALATED")
    ]
    assert calls == [(OPEN_INCIDENTS_PROBE, "cluster-a", NODE)]
    assert node_open_incidents(lambda *_: {}, "cluster-a", NODE) == []
