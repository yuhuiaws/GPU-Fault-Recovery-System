"""``gpu-fault-admin remediation-storm release|abort`` and its in-Pod service.

The breaker never closes by itself. ``release`` closes and re-arms it (the
held workflows run, without counting again); ``abort`` keeps it open and
hands every held workflow to the operator -- SUPERSEDED with an audited
``REMEDIATION_STORM_ABORTED`` event, its incident ESCALATED. Both are audited
with the operator identity, idempotent, and read only under ``--dry-run``.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cli
from gpu_fault.admin import remediation_storm as module
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.models import (
    IncidentState,
    WorkflowEventCode,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.remediation_storm import StormBreakerState, closed_breaker, storm_scope
from gpu_fault.store import InMemoryStore
from gpu_fault.store.shared.remediation_budgets import blocked_by_remediation_budget
from gpu_fault.workflow_reconcile import remediation_storm_action
from tests._builders import fault_incident, workflow_request, workflow_step

NOW = datetime.now(timezone.utc)
CLUSTER = "gpu-a"


def _tripped(store: InMemoryStore, held: int = 2) -> list[str]:
    """A TRIPPED breaker of ``CLUSTER`` and ``held`` workflows it refused."""

    record = closed_breaker(CLUSTER, NOW).model_copy(
        update={
            "state": StormBreakerState.TRIPPED,
            "trip_id": f"{CLUSTER}@{NOW.isoformat()}",
            "tripped_at": NOW - timedelta(minutes=5),
            "trip_node_ids": ["node-0", "node-1", "node-2"],
            "trip_domains": 3,
            "trip_threshold": 3,
        }
    )
    store.replace_remediation_storm_breaker(record, expected=None)
    ids = []
    for index in range(held):
        incident = fault_incident(
            f"incident-{index}",
            f"event-{index}",
            cluster_id=CLUSTER,
            node_ids=[f"node-{index + 2}"],
            state=IncidentState.ACTION_PENDING,
        )
        workflow = blocked_by_remediation_budget(
            workflow_request(
                f"workflow-{index}",
                incident.incident_id,
                official_steps=[
                    workflow_step(
                        WorkflowOperation.RESTART_NODE, node_ids=[f"node-{index + 2}"]
                    )
                ],
            ),
            "held",
            scope=storm_scope(CLUSTER),
            now=NOW,
        )
        store.save_incident_and_workflow(
            incident.model_copy(update={"workflow_request_id": workflow.request_id}),
            workflow,
        )
        ids.append(workflow.request_id)
    return ids


def _act(store, action: str, **values):
    return remediation_storm_action(
        store,
        CLUSTER,
        action,
        actor="arn:aws:iam::123:user/alice",
        reason="driver rollout reverted",
        reference="CHG-1",
        now=NOW,
        **values,
    )


def test_release_closes_and_rearms_it_once() -> None:
    store = InMemoryStore()
    held = _tripped(store)

    planned = _act(store, "release", dry_run=True)
    assert planned["status"] == "plan" and planned["held_workflows"] == held
    assert store.get_remediation_storm_breaker(CLUSTER).open, "dry run changed it"

    released = _act(store, "release")
    record = store.get_remediation_storm_breaker(CLUSTER)
    assert released["status"] == "released"
    assert record.state is StormBreakerState.CLOSED and record.window == []
    assert record.released_by == "arn:aws:iam::123:user/alice"
    assert record.release_reference == "CHG-1"
    assert _act(store, "release")["status"] == "already-closed"
    # The held rows stay PENDING; their next claim is admitted.
    assert {store.get_workflow(i).status for i in held} == {WorkflowStatus.PENDING}


def test_abort_hands_the_held_workflows_to_the_operator() -> None:
    store = InMemoryStore()
    held = _tripped(store)

    result = _act(store, "abort")

    assert result["status"] == "aborted" and result["closed_workflows"] == held
    record = store.get_remediation_storm_breaker(CLUSTER)
    assert record.state is StormBreakerState.ABORTED and record.open
    for request_id in held:
        workflow = store.get_workflow(request_id)
        assert workflow.status is WorkflowStatus.SUPERSEDED
        assert workflow.events[-1].code == (
            WorkflowEventCode.REMEDIATION_STORM_ABORTED.value
        )
        assert workflow.events[-1].actor == "arn:aws:iam::123:user/alice"
        incident = store.get_incident(workflow.incident_id)
        assert incident.state is IncidentState.ESCALATED
        assert "aborted by an operator (CHG-1" in incident.reasons[-1]
    again = _act(store, "abort")
    assert again["status"] == "aborted" and again["closed_workflows"] == []

    with pytest.raises(ValueError, match="not open"):
        _act(InMemoryStore(), "abort")


def test_the_pod_script_runs_the_service_on_the_store(monkeypatch, capsys) -> None:
    from gpu_fault.app import ApplicationContext

    store = InMemoryStore()
    _tripped(store, held=1)
    context = SimpleNamespace(store=store)
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )
    payload = {
        "cluster_id": CLUSTER,
        "action": "release",
        "reason": "done",
        "reference": "CHG-2",
        "actor": "alice",
        "dry_run": False,
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))

    exec(compile(module.SCRIPT, "<remediation-storm>", "exec"), {})

    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["status"] == "released"
    assert not store.get_remediation_storm_breaker(CLUSTER).open, "released"


def _site(tmp_path: Path) -> SimpleNamespace:
    (tmp_path / "site.yaml").write_text("name: staging\n", encoding="utf-8")
    return SimpleNamespace(
        source=tmp_path / "site.yaml",
        repository_root=tmp_path,
        environment={},
        source_sha256="c" * 64,
        release_config={
            "site_name": "staging",
            "cpu_kubeconfig": str(tmp_path / "cpu.kubeconfig"),
            "namespace": "gpu-fault-system",
            "clusters": [{"cluster_id": CLUSTER, "context": "gpu-a-context"}],
        },
    )


def _arguments(tmp_path: Path, action: str, *extra: str):
    return cli.parser().parse_args(
        [
            "remediation-storm",
            action,
            "--state-dir",
            str(tmp_path),
            "--cluster-id",
            CLUSTER,
            "--reason",
            "rollout reverted",
            "--reference",
            "CHG-7",
            *extra,
        ]
    )


def test_the_verb_takes_the_lock_and_records_the_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    calls: list[dict] = []
    locks: list[Path] = []

    @contextmanager
    def lock(path):
        locks.append(path)
        yield

    def run_script(site, payload, *, script):
        assert script is module.SCRIPT
        calls.append(payload)
        return {
            "cluster_id": CLUSTER,
            "action": payload["action"],
            "status": "released",
        }

    monkeypatch.setattr(module, "run_control_plane_script", run_script)
    monkeypatch.setattr(module, "administrator_operation_lock", lock)
    monkeypatch.setattr(module, "reload_site_for_mutation", lambda site: site)
    site = _site(tmp_path)

    monkeypatch.setattr(module, "_operator", lambda: "alice")
    code = module.run_command(_arguments(tmp_path, "release"), site=site)

    assert code == 0 and locks == [tmp_path]
    assert calls == [
        {
            "cluster_id": CLUSTER,
            "action": "release",
            "reason": "rollout reverted",
            "reference": "CHG-7",
            "actor": "alice",
            "dry_run": False,
        }
    ]
    printed = json.loads(capsys.readouterr().out)
    evidence = Path(printed["evidence_path"])
    assert evidence.parent == tmp_path / "remediation-storm" / CLUSTER
    assert json.loads(evidence.read_text())["operator"] == "alice"


def test_dry_run_takes_no_lock_and_bad_input_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        module,
        "administrator_operation_lock",
        lambda _path: pytest.fail("a dry run takes no lock"),
    )
    monkeypatch.setattr(
        module,
        "run_control_plane_script",
        lambda site, payload, *, script: {"status": "plan", **payload},
    )
    monkeypatch.setattr(module, "_operator", lambda: "a")
    site = _site(tmp_path)

    assert (
        module.run_command(_arguments(tmp_path, "abort", "--dry-run"), site=site) == 0
    )
    unknown = _arguments(tmp_path, "release")
    unknown.cluster_id = "gpu-z"
    with pytest.raises(BootstrapError, match="not in the managed site"):
        module.run_command(unknown, site=site)
    bad = _arguments(tmp_path, "release")
    bad.reference = "x"
    with pytest.raises(BootstrapError, match="--reference is invalid"):
        module.run_command(bad, site=site)
