from __future__ import annotations

import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from kubernetes.client.exceptions import ApiException

from gpu_fault.models import RecoveryPlan, WorkflowStatus
from gpu_fault.remote_command_models import RemoteCommandStatus
from scripts.e2e.regional.probes import destr008_cancellation_probe as probe
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from tests.regional.test_destr008_cancellation_bridge import arguments, main_bridge
from tests.regional.test_destr008_cancellation_bridge import (
    restore_logging as restore_logging,
)
from tests.regional.test_destr008_cancellation_probe import (
    NOW,
    acknowledge,
    armed_submission,
    claim,
    command,
    plan,
    seed,
    setup,
)


def test_standalone_module_entrypoint_uses_local_helpers_only(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    script = Path(probe.__file__).resolve()
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 0
    assert "--configmap" in capsys.readouterr().out


def test_cli_errors_do_not_echo_unrecognized_credentials(capsys) -> None:
    private = "postgresql://local-user:private-test-value@invalid.example/database"
    assert probe.main([*arguments(plan()), "--unexpected", private]) == 1
    output = capsys.readouterr()
    assert private not in output.out + output.err
    assert json.loads(output.out)["error_code"] == "CLI_ARGUMENTS"


@pytest.mark.parametrize("shape", ["malformed", "revocation_removed"])
def test_damage_after_revocation_never_resets_the_durable_tombstone(shape: str) -> None:
    bound, api, port, _, watchdog = setup()
    watchdog.tick(NOW)
    revoked = watchdog.tick(bound.deadline_at)
    if shape == "malformed":
        api.value.data["control.json"] = '{"unknown":"private-record"}'
    else:
        current = port.read().control
        api.value.data["control.json"] = wire.encode(
            current.model_copy(update={"revocation": None})
        )
    failed = watchdog.tick(bound.deadline_at + 1)
    assert failed.state == "FAILED" and failed.producer_revoked
    assert failed.revocation == revoked.revocation and not failed.monitoring
    assert "private-record" not in wire.encode(failed)


def test_submission_before_armed_is_rejected_and_revoked() -> None:
    bound, api, port, store, watchdog = setup()
    submitted = wire.claim_submission(
        bound, port.read().control, claim_id="claim-a", now=NOW + 1
    )
    api.value.data["control.json"] = wire.encode(submitted)
    failed = watchdog.tick(NOW + 2)
    assert failed.state == "FAILED" and failed.error_code == "UNARMED_SUBMISSION"
    assert failed.producer_revoked and not failed.monitoring
    assert store.list_workflows() == []


@pytest.mark.usefixtures("restore_logging")
def test_main_resumes_monitoring_without_resetting_to_armed(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    bound, api, store, _ = main_bridge(monkeypatch)
    port = probe.KubernetesControlMap(
        api,
        namespace="cpu",
        name="watchdog",
        uid="cm-uid",
        plan_sha256=wire.digest(bound),
        probe_sha256=bound.probe_sha256,
    )
    watchdog = probe.Watchdog(port, store, sleep=lambda _: None)
    watchdog.tick(NOW)
    revoked = watchdog.tick(bound.deadline_at)
    monkeypatch.setattr(probe.time, "time", lambda: float(bound.deadline_at + 1))
    run = probe.run
    times = iter([bound.deadline_at + 1, bound.deadline_at + 2, bound.deadline_at + 5])
    monkeypatch.setattr(
        probe,
        "run",
        lambda port, store: run(
            port, store, clock=lambda: next(times), sleep=lambda _: None
        ),
    )
    assert probe.main(arguments(bound)) == 0
    output = [json.loads(item) for item in capsys.readouterr().out.splitlines()]
    assert [item["state"] for item in output] == ["REVOKED", "QUIESCENT"]
    assert port.read().status.revocation == revoked.revocation and store.closed


def test_initial_control_read_retries_transient_failures_with_a_bound() -> None:
    _, api, port, _, _ = setup()
    api.read_errors = [ApiException(status=503)]
    waits: list[float] = []
    snapshot = probe.read_initial(port, sleep=waits.append)
    assert snapshot.control.producer.state == "NOT_STARTED" and waits == [0.25]
    api.read_errors = [ApiException(status=503)] * probe.RETRY_ATTEMPTS
    with pytest.raises(wire.ProbeError, match="API_UNAVAILABLE"):
        probe.read_initial(port, sleep=waits.append)
    assert len(waits) == 4
    api.read_errors = [ApiException(status=403)]
    with pytest.raises(ApiException) as caught:
        probe.read_initial(port, sleep=waits.append)
    assert caught.value.status == 403 and len(waits) == 4


def test_transient_api_exhaustion_remains_a_monitored_revoked_failure() -> None:
    bound, api, port, _, watchdog = setup()
    watchdog.tick(NOW)
    api.read_errors = [ApiException(status=503)] * probe.RETRY_ATTEMPTS
    failed = watchdog.tick(bound.deadline_at)
    assert failed.state == "FAILED" and failed.error_code == "API_UNAVAILABLE"
    assert failed.monitoring and failed.producer_revoked
    assert watchdog.tick(bound.deadline_at + 1).state == "REVOKED"
    assert watchdog.tick(bound.deadline_at + 6).state == "QUIESCENT"
    assert port.read().control.revocation == failed.revocation


def test_transient_store_exhaustion_keeps_scanning_late_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from psycopg import OperationalError

    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    read = store.list_job_recovery_workflow_incidents
    failures = [OperationalError("local unavailable")] * probe.RETRY_ATTEMPTS

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        if failures:
            raise failures.pop()
        return read(*args, **kwargs)

    monkeypatch.setattr(store, "list_job_recovery_workflow_incidents", unavailable)
    failed = watchdog.tick(bound.deadline_at)
    assert failed.state == "FAILED" and failed.error_code == "STORE_UNAVAILABLE"
    assert failed.monitoring and failed.producer_revoked
    remote = command(store, incident, workflow)
    assert watchdog.tick(bound.deadline_at + 1).state == "REVOKED"
    assert (
        store.get_remote_command(remote.command_id).status is RemoteCommandStatus.FAILED
    )


@pytest.mark.usefixtures("restore_logging")
def test_startup_store_unavailability_can_resume_after_job_restart(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from psycopg import OperationalError

    bound, api, _, _ = main_bridge(monkeypatch)
    monkeypatch.setattr(
        probe,
        "cpu_store",
        lambda: (_ for _ in ()).throw(OperationalError("private-connection-error")),
    )
    assert probe.main(arguments(bound)) == 1
    status = wire.decode(wire.Receipt, api.value.data["status.json"])
    assert status.state == "FAILED" and status.error_code == "CPU_STORE_UNAVAILABLE"
    assert status.monitoring and status.producer_revoked
    assert "private-connection-error" not in capsys.readouterr().out


@pytest.mark.parametrize("record", ["incident", "workflow", "command"])
@pytest.mark.parametrize(
    "created_at",
    [datetime.fromtimestamp(NOW - 1, timezone.utc), datetime.fromtimestamp(NOW)],
)
def test_matching_identity_does_not_authorize_records_older_than_the_run(
    record: str, created_at: datetime
) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    if record == "incident":
        store.save_incident(
            incident.model_copy(update={"created_at": created_at}), expected=incident
        )
    elif record == "workflow":
        store.save_workflow(
            workflow.model_copy(update={"created_at": created_at}), expected=workflow
        )
    else:
        command(store, incident, workflow, created_at=created_at)
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "RECORD_SOURCE"
    assert store.get_workflow(workflow.request_id).workload_withdrawn_at is None


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://localhost/db",
        "postgresql://localhost/db?sslmode=require",
        "postgresql://localhost/db?sslmode=disable",
        "postgresql:///db?sslmode=verify-full",
    ],
)
def test_cpu_store_refuses_unverified_tls_before_connecting(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL", url)
    monkeypatch.setenv("GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT", "false")
    monkeypatch.setattr(
        probe, "PostgresStore", lambda *a, **kw: pytest.fail("must not open unsafe DSN")
    )
    with pytest.raises(wire.ProbeError, match="CPU_STORE_TLS_REQUIRED"):
        probe.cpu_store()


def test_mounted_store_url_is_checked_and_parse_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    secret = tmp_path / "local-store-url"
    monkeypatch.setenv(
        "GPU_FAULT_STORE_URL", "postgresql://localhost/db?sslmode=verify-full"
    )
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(secret))
    monkeypatch.setenv("GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT", "false")
    secret.write_text("postgresql://localhost/db?sslmode=disable")
    with pytest.raises(wire.ProbeError, match="CPU_STORE_TLS_REQUIRED"):
        probe.cpu_store()
    secret.write_text("invalid-private-connection-data")
    with pytest.raises(wire.ProbeError, match="CPU_STORE_CONFIGURATION") as failure:
        probe.cpu_store()
    assert "private-connection-data" not in str(failure.value)


def test_unknown_submission_has_unknown_counts_and_remembers_observed_root() -> None:
    bound, _, port, store, watchdog = setup()
    watchdog.tick(NOW)
    claim(bound, port)
    unknown = watchdog.tick(bound.deadline_at)
    assert unknown.commands_active is unknown.workflows_active is None
    assert unknown.pending_creation is True and unknown.source_complete is False
    seed(store, bound)
    observed = watchdog.tick(bound.deadline_at + 1)
    again = watchdog.tick(bound.deadline_at + 2)
    assert observed.root is not None and again.root == observed.root
    assert again.state == "FAILED" and again.error_code == "SOURCE_UNRESOLVED"


def test_root_with_missing_workflow_pointer_is_unresolved() -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    store.save_incident(
        incident.model_copy(update={"workflow_request_id": None}), expected=incident
    )
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "ROOT_MISSING"
    assert store.get_workflow(workflow.request_id).workload_withdrawn_at is None


def test_unacknowledged_root_cannot_choose_between_disconnected_workflows() -> None:
    bound, _, port, store, watchdog = setup()
    watchdog.tick(NOW)
    claim(bound, port)
    _, workflow = seed(store, bound)
    store.save_workflow(workflow.model_copy(update={"request_id": "other-root"}))
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "ROOT_BINDING"
    assert store.get_workflow(workflow.request_id).workload_withdrawn_at is None


@pytest.mark.parametrize("kind", ["incident", "workflow", "observed_root"])
def test_acknowledgement_must_resolve_the_exact_observed_source(kind: str) -> None:
    bound, _, port, store, watchdog = setup()
    watchdog.tick(NOW)
    claim(bound, port)
    _, workflow = seed(store, bound)
    if kind == "incident":
        acknowledge(bound, port, incident_id="different-incident")
    else:
        if kind == "observed_root":
            observed = watchdog.tick(bound.deadline_at)
            assert observed.root is not None
        seed(
            store,
            bound,
            incident_id="other-incident",
            event_id="other-event",
            workflow_id="other-workflow",
        )
        if kind == "observed_root":
            acknowledge(
                bound, port, now=bound.deadline_at + 1, workflow_id="other-workflow"
            )
        else:
            acknowledge(bound, port, workflow_id="other-workflow")
    result = watchdog.tick(bound.deadline_at + 2)
    assert result.state == "FAILED" and result.error_code == "ROOT_BINDING"
    assert store.get_workflow(workflow.request_id).status is WorkflowStatus.PENDING


def test_proven_descendant_in_another_incident_and_duplicate_edges_are_owned_once() -> (
    None
):
    bound, _, _, store, watchdog, _, workflow = armed_submission(
        status=WorkflowStatus.FAILED
    )
    child_event = "support-after-" + workflow.request_id
    _, child = seed(
        store,
        bound,
        incident_id="inc-" + child_event,
        event_id=child_event,
        workflow_id="workflow-" + child_event,
        predecessor=workflow.request_id,
        status=WorkflowStatus.SUCCEEDED,
    )
    _, grandchild = seed(
        store,
        bound,
        incident_id="grandchild-incident",
        event_id="grandchild-event",
        workflow_id="grandchild-workflow",
        predecessor=child.request_id,
        status=WorkflowStatus.BLOCKED,
    )
    result = watchdog.tick(bound.deadline_at)
    assert set(result.workflow_ids) == {
        workflow.request_id,
        child.request_id,
        grandchild.request_id,
    }
    assert len(store.get_workflow(child.request_id).events) == 1
    assert watchdog.tick(bound.deadline_at + 5).state == "QUIESCENT"


def test_owned_descendant_inventory_is_bounded_even_when_all_children_are_terminal() -> (
    None
):
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    previous_id = workflow.request_id
    for index in range(wire.MAX_OWNED_RECORDS):
        child = workflow.model_copy(
            update={
                "request_id": f"child-{index}",
                "predecessor_workflow_id": previous_id,
                "status": WorkflowStatus.BLOCKED,
            }
        )
        store.save_workflow(child)
        previous_id = child.request_id
    store.save_incident(
        incident.model_copy(update={"workflow_request_id": previous_id}),
        expected=incident,
    )
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "INVENTORY_SIZE"
    assert store.get_workflow(workflow.request_id).workload_withdrawn_at is None


@pytest.mark.parametrize(
    "change",
    [{"event_id": "wrong"}, {"incident_id": "wrong"}, {"workflow_request_id": "wrong"}],
)
def test_escalation_event_lookup_is_not_ancestry_proof_when_identity_disagrees(
    change: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    bound, _, _, store, watchdog, _, workflow = armed_submission()
    event = "support-after-" + workflow.request_id
    child, _ = seed(
        store,
        bound,
        incident_id="inc-" + event,
        event_id=event,
        workflow_id="workflow-" + event,
    )
    lookup = store.get_incident_by_event
    monkeypatch.setattr(
        store,
        "get_incident_by_event",
        lambda key: child.model_copy(update=change) if key == event else lookup(key),
    )
    result = watchdog.tick(bound.deadline_at)
    expected = (
        "INVENTORY_CHANGED" if "workflow_request_id" in change else "DESCENDANT_SOURCE"
    )
    assert result.state == "FAILED" and result.error_code == expected
    assert store.get_workflow(workflow.request_id).workload_withdrawn_at is None


@pytest.mark.parametrize(
    "kind", ["pointer", "disconnected", "preempted", "preemption_pending"]
)
def test_unresolved_lineage_keeps_all_creation_authority_closed(kind: str) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    if kind == "pointer":
        store.save_incident(
            incident.model_copy(update={"workflow_request_id": "missing"}),
            expected=incident,
        )
    elif kind == "disconnected":
        store.save_workflow(workflow.model_copy(update={"request_id": "disconnected"}))
    else:
        field = (
            "preempted_by_workflow_id"
            if kind == "preempted"
            else "preemption_pending_by_workflow_id"
        )
        store.save_workflow(
            workflow.model_copy(update={field: "unresolved"}), expected=workflow
        )
    result = watchdog.tick(bound.deadline_at)
    expected = "DESCENDANT_UNRESOLVED" if kind == "disconnected" else "RECORD_MISSING"
    assert result.state == "FAILED" and result.error_code == expected
    assert result.monitoring and result.producer_revoked


@pytest.mark.parametrize("kind", ["duplicate", "changed"])
def test_active_query_must_agree_with_the_complete_inventory(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    records = (
        [(incident, workflow)] * 2
        if kind == "duplicate"
        else [(incident, workflow.model_copy(update={"merge_revision": 1}))]
    )
    monkeypatch.setattr(
        store, "list_job_recovery_workflow_incidents", lambda *a, **kw: records
    )
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED"
    assert result.error_code == (
        "DUPLICATE_WORKFLOW" if kind == "duplicate" else "INVENTORY_CHANGED"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"plan_id": "wrong"},
        {"incident_id": "wrong"},
        {"attempt_id": "wrong"},
        {"workflow_request_id": "wrong"},
        {"runtime_profile_version": "wrong"},
    ],
)
def test_source_plan_is_identity_checked(
    change: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    recovery = RecoveryPlan(
        plan_id="plan-a",
        incident_id=incident.incident_id,
        attempt_id=bound.attempt_id,
        trigger="local",
        runtime_profile_version=bound.runtime_profile_version,
        workflow_request_id=workflow.request_id,
        steps=[],
    )
    store.save_workflow(
        workflow.model_copy(update={"source_plan_id": "plan-a"}), expected=workflow
    )
    monkeypatch.setattr(store, "get_plan", lambda _: recovery.model_copy(update=change))
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "RECOVERY_PLAN_SCOPE"


def test_duplicate_command_records_are_not_counted_as_two_cancellations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    remote = command(store, incident, workflow)
    monkeypatch.setattr(store, "list_remote_commands", lambda **_: [remote, remote])
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "DUPLICATE_COMMAND"
    assert (
        store.get_remote_command(remote.command_id).status
        is RemoteCommandStatus.PENDING
    )


@pytest.mark.parametrize("kind", ["request_id", "incident_id", "event_id"])
def test_withdrawal_rechecks_identity_immediately_before_amending(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    remote = command(store, incident, workflow)
    get_workflow, get_incident = store.get_workflow, store.get_incident
    reads = 0

    def workflow_at_boundary(key: str):
        nonlocal reads
        reads += 1
        if reads == 2 and kind != "event_id":
            return workflow.model_copy(update={kind: "wrong"})
        return get_workflow(key)

    def incident_at_boundary(key: str):
        if reads == 2 and kind != "request_id":
            return incident.model_copy(update={kind: "wrong"})
        return get_incident(key)

    monkeypatch.setattr(store, "get_workflow", workflow_at_boundary)
    monkeypatch.setattr(store, "get_incident", incident_at_boundary)
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "WORKFLOW_SCOPE"
    assert (
        store.get_remote_command(remote.command_id).status
        is RemoteCommandStatus.PENDING
    )


@pytest.mark.parametrize("kind", ["not_withdrawn", "request_changed"])
def test_store_amendment_ack_is_required_before_cancellation(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    remote = command(store, incident, workflow)
    response = (
        workflow
        if kind == "not_withdrawn"
        else workflow.model_copy(update={"request_id": "wrong"})
    )
    monkeypatch.setattr(store, "amend_workflow", lambda *a, **kw: response)
    result = watchdog.tick(bound.deadline_at)
    assert (
        result.state == "FAILED" and result.error_code == "WITHDRAWAL_NOT_ACKNOWLEDGED"
    )
    assert (
        store.get_remote_command(remote.command_id).status
        is RemoteCommandStatus.PENDING
    )


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"cancelled": 0, "cancellation_requested": 0, "extra": 0},
        {"cancelled": True, "cancellation_requested": 0},
        {"cancelled": -1, "cancellation_requested": 0},
        {"cancelled": 0, "cancellation_requested": wire.MAX_OWNED_RECORDS + 1},
    ],
)
def test_store_cancellation_ack_cannot_be_unknown_or_oversized(
    response: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound, _, _, store, watchdog, _, _ = armed_submission()
    monkeypatch.setattr(
        store, "cancel_remote_commands_for_workflow", lambda *a, **kw: response
    )
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "CANCELLATION_SHAPE"


def test_probe_never_prints_model_serializer_values(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    bound, _, _, store, watchdog, _, workflow = armed_submission()
    private = "private-invalid-model-status"
    monkeypatch.setattr(
        store,
        "list_job_recovery_workflow_incidents",
        lambda *args, **kwargs: [
            (
                store.get_incident(workflow.incident_id),
                workflow.model_copy(update={"status": private}),
            )
        ],
    )
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "STORE_SHAPE"
    captured = capsys.readouterr()
    assert private not in captured.out + captured.err + wire.encode(result)


def test_receipt_maximum_is_enforced_before_api_write() -> None:
    bound = plan()
    control = wire.revoke(wire.initial_control(bound), now=NOW, reason="FAILURE")
    ids = [f"{index:03d}" + "x" * 250 for index in range(wire.MAX_OWNED_RECORDS)]
    with pytest.raises(wire.ProbeError, match="RECEIPT_SIZE"):
        wire.receipt(
            bound,
            control,
            None,
            uid="cm-uid",
            now=NOW,
            state="REVOKED",
            workflow_ids=ids,
            command_ids=ids,
        )


@pytest.mark.parametrize(
    "proof",
    [
        {"producer": None},
        {"fence_release_authorized": 0},
        {
            "producer_revoked": True,
            "revocation": wire.Revocation(
                at=NOW,
                reason="FAILURE",
                producer_sha256="a" * 64,
                producer_state="NOT_STARTED",
            ),
        },
        {"source_complete": True, "workflow_ids": ["wrong"]},
        {"source_complete": True, "command_ids": ["wrong"]},
        {
            "source_complete": True,
            "root": wire.Root(incident_id="wrong", workflow_request_id="wrong"),
        },
    ],
)
def test_structurally_plausible_receipt_cannot_substitute_for_source_proof(
    proof: dict[str, Any],
) -> None:
    bound = plan()
    with pytest.raises(wire.ProbeError, match="RECEIPT_SHAPE"):
        wire.receipt(
            bound,
            wire.initial_control(bound),
            None,
            uid="cm-uid",
            now=NOW,
            state="ARMED",
            **proof,
        )


def test_ack_source_proof_requires_matching_root_and_inventory() -> None:
    bound, _, port, _, _, _, _ = armed_submission()
    control = wire.revoke(port.read().control, now=NOW + 3, reason="FAILURE")
    for proof in [
        {"root": None, "workflow_ids": []},
        {
            "root": wire.Root(incident_id="wrong", workflow_request_id="workflow-a"),
            "workflow_ids": ["workflow-a"],
        },
        {
            "root": wire.Root(incident_id="incident-a", workflow_request_id="wrong"),
            "workflow_ids": ["wrong"],
        },
        {
            "root": wire.Root(
                incident_id="incident-a", workflow_request_id="workflow-a"
            ),
            "workflow_ids": [],
        },
    ]:
        with pytest.raises(wire.ProbeError, match="RECEIPT_SHAPE"):
            wire.receipt(
                bound,
                control,
                None,
                uid=port.uid,
                now=NOW + 3,
                state="REVOKED",
                source_complete=True,
                **proof,
            )
    submitted = wire.claim_submission(
        bound, wire.initial_control(bound), claim_id="claim-a", now=NOW + 1
    )
    with pytest.raises(wire.ProbeError, match="RECEIPT_SHAPE"):
        wire.receipt(
            bound,
            wire.revoke(submitted, now=NOW + 2, reason="FAILURE"),
            None,
            uid=port.uid,
            now=NOW + 2,
            state="REVOKED",
            source_complete=True,
        )
