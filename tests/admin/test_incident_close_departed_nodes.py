"""``--close-quarantined`` / ``--close-incident`` for a node that departed.

A QUARANTINED incident whose node a spot reclaim took had no exit: the
evidence pass read the cluster, found no Node and handed the service
``exists: False``, which no close accepts. The pass now reads the provider for
such a node too -- HyperPod's ``list-cluster-nodes`` here, as the
never-changed reconcile does -- and hands the departure proof along; the
service in the Pod clears the node only when that proof shows it gone from
Kubernetes and the provider and its Agent record is not alive.

These run the real in-Pod script against an in-memory control plane, with
``kubectl`` and ``aws`` answered by doubles.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import incident_close
from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.models import IncidentState
from tests._builders import build_store, copy_model
from tests.departed_nodes._support import agent
from tests.orchestration._incident_closure_support import _escalated_reset
from tests.regional._regional_support import TOKEN_A, registration

GONE = "hyperpod-i-0a000001"
PRESENT = "hyperpod-i-0a000002"


def _site(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        release_config={
            "site_name": "staging",
            "aws_region": "us-west-2",
            "namespace": "gpu-fault",
            "cpu_kubeconfig": str(tmp_path / "cpu.kubeconfig"),
            "gpu_kubeconfig": str(tmp_path / "gpu.kubeconfig"),
            "clusters": [
                {
                    "cluster_id": "cluster-a",
                    "context": "cluster-a-context",
                    "hyperpod_cluster_name": "hp-a",
                    "region": "us-west-2",
                }
            ],
        },
        environment={},
        source_sha256="a" * 64,
    )


def _control_plane(monkeypatch: pytest.MonkeyPatch) -> Any:
    """An in-memory control plane with one QUARANTINED incident on ``GONE``."""

    from gpu_fault.app import ApplicationContext

    store = build_store()
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    incident, _ = _escalated_reset(store, incident_id="inc-q", node_ids=(GONE,))
    store.save_incident(
        copy_model(incident, state=IncidentState.QUARANTINED), expected=incident
    )
    context = ApplicationContext(store=store)
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )

    def run(_site: Any, payload: dict[str, Any], *, script: str) -> dict[str, Any]:
        output = io.StringIO()
        with monkeypatch.context() as scoped:
            scoped.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
            scoped.setattr(sys, "stdout", output)
            exec(compile(script, "<pod>", "exec"), {})
        return dict(json.loads(output.getvalue()))

    monkeypatch.setattr(incident_close, "_run_reconcile", run)
    return store


def _cluster(monkeypatch: pytest.MonkeyPatch, *, hyperpod_lists: list[str]) -> None:
    """``kubectl get nodes`` lists only ``PRESENT``; HyperPod lists ``hyperpod_lists``."""

    def run(args: list[str], **_kwargs: Any) -> SimpleNamespace:
        if "list-cluster-nodes" in args:
            body = {
                "ClusterNodeSummaries": [
                    {"InstanceId": item} for item in hyperpod_lists
                ]
            }
        else:
            assert args[-4:] == ["get", "nodes", "-o", "json"], args
            body = {
                "items": [
                    {
                        "metadata": {"name": PRESENT, "annotations": {}},
                        "spec": {"unschedulable": False, "taints": []},
                    }
                ]
            }
        return SimpleNamespace(returncode=0, stdout=json.dumps(body), stderr="")

    monkeypatch.setattr(reconcile, "run_command", run)


def _close(tmp_path: Path) -> dict[str, Any]:
    return incident_close.run_incident_close(
        _site(tmp_path),
        tmp_path,
        selector=incident_close.quarantined_selector(),
        reason="node reclaimed by spot",
        reference="CHG-1",
        dry_run=False,
        actor="ops@example",
    )


def test_a_quarantined_incident_of_a_departed_node_closes_from_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _control_plane(monkeypatch)
    _cluster(monkeypatch, hyperpod_lists=["i-0a000002"])

    result = _close(tmp_path)

    assert incident_close.result_lines(result)[-1] == (
        f"inc-q: closed (isolation absent on {GONE})"
    )
    closed = store.get_incident("inc-q")
    assert closed.state is IncidentState.RECOVERED
    assert closed.reasons[-1] == (
        f"isolation no longer present on node {GONE}: the node departed (gone "
        "from Kubernetes and HyperPod, instance i-0a000001)"
    )
    [evidence] = result["isolation_evidence"]["inc-q"]
    departure = evidence["departure"]
    assert evidence["exists"] is False
    assert departure["absent_from_kubernetes"] is True
    assert departure["absent_from_provider"] is True
    assert departure["sources"]["hyperpod"] == {
        "consulted": True,
        "cluster_name": "hp-a",
        "region": "us-west-2",
        "listed": False,
    }
    assert set(departure["read_at"]) == {"kubernetes", "hyperpod"}


def test_a_node_hyperpod_still_lists_keeps_the_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _control_plane(monkeypatch)
    _cluster(monkeypatch, hyperpod_lists=["i-0a000001", "i-0a000002"])

    result = _close(tmp_path)

    [line] = incident_close.result_lines(result)[-1:]
    assert line.startswith("inc-q: refused("), line
    assert "departure is unproven" in line and "HyperPod still lists" in line, line
    assert store.get_incident("inc-q").state is IncidentState.QUARANTINED


def test_an_agent_that_still_heartbeats_keeps_the_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _control_plane(monkeypatch)
    store.save_agent(
        agent(
            "cluster-a",
            GONE,
            instance_id=None,
            last_seen=datetime.now(timezone.utc) - timedelta(minutes=2),
        )
    )
    _cluster(monkeypatch, hyperpod_lists=["i-0a000002"])

    result = _close(tmp_path)

    assert "its Agent is not departed" in incident_close.result_lines(result)[-1]
    assert store.get_incident("inc-q").state is IncidentState.QUARANTINED
