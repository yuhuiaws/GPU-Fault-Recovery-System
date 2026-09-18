from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.models import WorkflowStatus
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import late_ownership_control as control
from tests.regional._late_ownership_runtime import runtime


def inputs():
    context = runtime()[-1]
    workflow = context.workflow
    incident = context.incident.model_copy(update={"drill_id": "owned-run"})
    return workflow, incident, datetime.now(timezone.utc) + timedelta(minutes=10)


def held():
    workflow, incident, end = inputs()
    store = InMemoryStore()
    result = control.hold_workflow(
        store, workflow, incident, run_id="owned-run", window_end=end
    )
    return store, workflow.model_validate(result["workflow"]), incident, end


def test_real_guarded_memory_holder_outlives_mutation_and_finishes_without_restart():
    store, workflow, incident, end = held()
    assert workflow.execution_owner_id == "late-ownership/owned-run"
    assert workflow.execution_lease_expires_at == end + timedelta(minutes=10)
    assert workflow.lifetime_deadline_at == end
    receipt = control.finish_workflow(store, workflow, run_id="owned-run")
    assert receipt == {
        "workflow_id": workflow.request_id,
        "fencing_token": workflow.fencing_token,
        "execution_epoch": workflow.execution_epoch,
        "status": "SUPERSEDED",
    }
    final = store.get_workflow(workflow.request_id)
    assert final.execution_owner_id is None
    assert final.execution_lease_expires_at is None
    assert final.workload_withdrawn_at is not None
    assert store.get_incident(incident.incident_id).state.value == "RECOVERED"
    assert control.finish_workflow(store, workflow, run_id="owned-run") == receipt


@pytest.mark.parametrize(
    "defect", ["window", "incident", "workflow", "run", "fence", "duplicate"]
)
def test_holder_creation_rejects_wrong_identity_before_store_mutation(defect):
    workflow, incident, end = inputs()
    store = InMemoryStore()
    if defect == "window":
        end = end.replace(tzinfo=None)
    elif defect == "incident":
        workflow = workflow.model_copy(update={"incident_id": "other"})
    elif defect == "workflow":
        incident = incident.model_copy(update={"workflow_request_id": "other"})
    elif defect == "run":
        incident = incident.model_copy(update={"drill_id": "other"})
    elif defect == "fence":
        workflow = workflow.model_copy(update={"fencing_token": 7})
    else:
        store.save_incident_and_workflow(incident, workflow)
    with pytest.raises(ValueError):
        control.hold_workflow(
            store, workflow, incident, run_id="owned-run", window_end=end
        )
    if defect == "duplicate":
        assert store.get_workflow(workflow.request_id) == workflow
    else:
        assert not store.list_workflows(), (
            "invalid holder identity must not create a workflow"
        )


def test_unconfirmed_holder_write_is_not_an_arm_receipt(monkeypatch):
    workflow, incident, end = inputs()
    store = InMemoryStore()
    calls = []

    def get(_key):
        if not calls:
            calls.append(True)
            raise KeyError
        return workflow

    monkeypatch.setattr(store, "get_workflow", get)
    with pytest.raises(ValueError, match="confirmed"):
        control.hold_workflow(
            store, workflow, incident, run_id="owned-run", window_end=end
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_owner_id", "replaced-owner"),
        ("execution_epoch", 8),
        ("fencing_token", 9),
        ("incident_id", "other"),
        ("official_steps", []),
        ("status", WorkflowStatus.SUPERSEDED),
    ],
)
def test_replaced_holder_cannot_be_terminalized(field, value):
    store, workflow, _incident, _end = held()
    current = workflow.model_copy(update={field: value})
    store.save_workflow(current, expected=workflow)
    with pytest.raises(ValueError, match="changed"):
        control.finish_workflow(store, workflow, run_id="owned-run")
    assert store.get_workflow(workflow.request_id) == current


@pytest.mark.parametrize("field,value", [("drill_id", "other"), ("fencing_token", 9)])
def test_changed_incident_cannot_be_marked_recovered(field, value):
    store, workflow, incident, _end = held()
    current = incident.model_copy(update={field: value})
    store.save_incident(current)
    with pytest.raises(ValueError, match="incident"):
        control.finish_workflow(store, workflow, run_id="owned-run")
    assert store.get_incident(incident.incident_id) == current


def test_final_readback_must_be_terminal(monkeypatch):
    store, workflow, _incident, _end = held()
    monkeypatch.setattr(store, "get_workflow", lambda _: workflow)
    with pytest.raises(ValueError, match="terminal"):
        control.finish_workflow(store, workflow, run_id="owned-run")


def test_dispatch_and_trusted_cpu_program_use_actual_store_methods():
    workflow, incident, end = inputs()
    store = InMemoryStore()
    value = dict(
        action="hold",
        workflow=workflow.model_dump(mode="json"),
        incident=incident.model_dump(mode="json"),
        run_id="owned-run",
        window_end=end.isoformat(),
    )
    result = control.dispatch(store, value)
    final = control.dispatch(
        store, value | {"action": "finish", "workflow": result["workflow"]}
    )
    assert final["status"] == "SUPERSEDED"
    with pytest.raises(ValueError, match="unsupported"):
        control.dispatch(store, value | {"action": "reset"})
    source = control.cpu_program()
    compile(source, "owned-cpu-probe", "exec")
    assert "def hold_workflow(" in source and source.endswith(
        "raise SystemExit(main())\n"
    )


@pytest.mark.parametrize("failure", [False, True])
def test_cpu_entry_keeps_adapter_error_text_out_of_receipts(
    monkeypatch, capsys, failure
):
    from gpu_fault.app import ApplicationContext

    monkeypatch.setattr(control.sys, "argv", ["owned-test", "{}"])
    monkeypatch.setattr(
        ApplicationContext, "from_environment", lambda: SimpleNamespace(store=object())
    )

    def dispatch(*args):
        if failure:
            raise ValueError("not public")
        return {"status": "SUPERSEDED"}

    monkeypatch.setattr(control, "dispatch", dispatch)
    assert control.main() == int(failure)
    assert json.loads(capsys.readouterr().out) == (
        {"error_kind": "ValueError"} if failure else {"status": "SUPERSEDED"}
    )


def test_cpu_lease_check_is_a_fresh_read_of_the_owned_workflow():
    store, workflow, _incident, _end = held()
    checked = control.dispatch(
        store,
        {
            "action": "check",
            "workflow": workflow.model_dump(mode="json"),
            "run_id": "owned-run",
        },
    )
    assert checked == {
        "workflow_id": workflow.request_id,
        "fencing_token": workflow.fencing_token,
        "execution_epoch": workflow.execution_epoch,
        "holder_valid": True,
        "lifetime_deadline_at": workflow.lifetime_deadline_at.isoformat(),
    }
    assert store.get_workflow(workflow.request_id) == workflow


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_owner_id", "new-owner"),
        ("execution_epoch", 5),
        ("fencing_token", 5),
        ("official_steps", []),
        ("status", WorkflowStatus.FAILED),
        ("execution_lease_expires_at", None),
        ("execution_lease_expires_at", datetime(2000, 1, 1, tzinfo=timezone.utc)),
        ("execution_lease_expires_at", datetime(2100, 1, 1)),
        ("lifetime_deadline_at", None),
        ("lifetime_deadline_at", datetime(2000, 1, 1, tzinfo=timezone.utc)),
        ("lifetime_deadline_at", datetime(2100, 1, 1)),
    ],
)
def test_lost_or_changed_cpu_lease_is_not_permission_to_continue(field, value):
    store, workflow, _incident, _end = held()
    changed = workflow.model_copy(update={field: value})
    store.save_workflow(changed, expected=workflow)
    with pytest.raises(ValueError, match="no longer current"):
        control.check_workflow(store, workflow, run_id="owned-run")
    assert store.get_workflow(workflow.request_id) == changed


@pytest.mark.parametrize(
    "field,value",
    [
        ("drill_id", "other"),
        ("workflow_request_id", "other"),
        ("fencing_token", 7),
        ("state", control.IncidentState.RECOVERED),
    ],
)
def test_cpu_incident_ownership_is_rechecked(field, value):
    store, workflow, incident, _end = held()
    changed = incident.model_copy(update={field: value})
    store.save_incident(changed, expected=incident)
    with pytest.raises(ValueError, match="no longer current"):
        control.check_workflow(store, workflow, run_id="owned-run")


def test_existing_incident_cannot_be_adopted_by_a_new_holder():
    workflow, incident, end = inputs()
    store = InMemoryStore()
    store.save_incident(incident)
    with pytest.raises(ValueError, match="incident already exists"):
        control.hold_workflow(
            store, workflow, incident, run_id="owned-run", window_end=end
        )
    assert not store.list_workflows(), (
        "an existing incident must not authorize a new holder workflow"
    )


def test_partial_terminal_write_can_be_reconciled_without_reopening_workflow(
    monkeypatch,
):
    store, workflow, incident, _end = held()
    save = store.save_incident

    def refused(*args, **kwargs):
        raise OSError("local Store write failure")

    monkeypatch.setattr(store, "save_incident", refused)
    with pytest.raises(OSError):
        control.finish_workflow(store, workflow, run_id="owned-run")
    terminal = store.get_workflow(workflow.request_id)
    assert terminal.status is WorkflowStatus.SUPERSEDED
    assert store.get_incident(incident.incident_id) == incident
    monkeypatch.setattr(store, "save_incident", save)
    assert (
        control.finish_workflow(store, workflow, run_id="owned-run")["status"]
        == "SUPERSEDED"
    )
    assert store.get_workflow(workflow.request_id) == terminal
