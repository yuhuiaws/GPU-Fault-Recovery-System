from __future__ import annotations

import json
from typing import Any

import pytest
from kubernetes.client.exceptions import ApiException

from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import RecoveryAction, Severity, WorkflowStatus, WorkloadState
from gpu_fault.remote_command_models import RemoteCommandStatus
from scripts.e2e.regional.destr008_watchdog_control import (
    MAX_CONTROL_BYTES,
    MAX_RECEIPT_AGE,
    CancellationControl,
)
from scripts.e2e.regional.probes import destr008_cancellation_probe as probe
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._destr008_control import build_control
from tests.regional.test_destr008_cancellation_probe import STAMP, seed


def test_complete_parent_protocol_uses_real_watchdog_and_exact_cpu_namespace() -> None:
    h = build_control(namespace="cpu-fixture", name="owned-control", uid="owned-uid")
    with pytest.raises(RegionalFixtureError, match="freshly armed"):
        h.control.assert_armed()
    armed = h.tick()
    assert h.control.assert_armed() == armed
    claim_id = h.submit(command_status=RemoteCommandStatus.PENDING)
    assert h.port.read().control.producer.claim_id == claim_id
    result = h.finish()
    assert h.control.quiescence() == result
    assert result.root.workflow_request_id == "workflow-a"
    assert result.commands_active == result.workflows_active == 0
    assert h.store.get_remote_command("command-a").status is RemoteCommandStatus.FAILED
    assert result.producer_revoked and not result.fence_release_authorized
    assert h.posts == 1
    assert {args[0] for args, _ in h.cpu.calls} == {"get", "patch"}
    identity = h.control.identity()
    identity["plan"]["workload_ids"].append("foreign")
    assert h.control.identity()["plan"]["workload_ids"] == h.plan.workload_ids


def test_a_claim_is_one_shot_across_new_parent_instances() -> None:
    h = build_control()
    h.tick()
    h.submit()
    before = list(h.api.patches)
    with pytest.raises(RegionalFixtureError):
        h.submit()
    resumed = CancellationControl(
        plan=h.plan,
        namespace=h.api.namespace,
        name=h.api.name,
        uid=h.api.uid,
        cpu=h.cpu,
        clock=h.clock,
        monotonic=h.clock.monotonic,
    )
    with pytest.raises(RegionalFixtureError):
        resumed.claim()
    assert h.posts == 1 and h.api.patches == before


def test_unacknowledged_post_is_never_retried_or_closed_from_absence() -> None:
    h = build_control()
    h.tick()
    claim_id = h.control.claim()
    h.posts += 1
    with pytest.raises(RegionalFixtureError, match="acknowledgement"):
        h.control.acknowledge(claim_id, {"status": 504, "body": None})
    with pytest.raises(RegionalFixtureError):
        h.submit()
    with pytest.raises(RegionalFixtureError):
        h.control.request_close()
    h.clock.advance(h.plan.deadline_at - h.clock())
    failure = h.tick()
    assert failure.error_code == "SOURCE_UNRESOLVED"
    assert failure.commands_active is failure.workflows_active is None
    with pytest.raises(RegionalFixtureError, match="quiescence"):
        h.control.quiescence()
    assert h.posts == 1 and h.port.read().control.producer.state == "SUBMITTING"


def test_parent_never_submitted_can_request_an_exact_closed_tombstone() -> None:
    h = build_control()
    h.tick()
    h.control.request_close()
    before = list(h.api.patches)
    h.control.request_close()
    assert h.api.patches == before
    with pytest.raises(RegionalFixtureError):
        h.control.claim()
    h.tick()
    h.clock.advance(5)
    actual = h.tick()
    assert h.control.quiescence() == actual and actual.root is None
    assert actual.producer.state == "NOT_STARTED" and h.posts == 0


def test_revocation_winning_the_claim_cas_prevents_submission() -> None:
    h = build_control()
    h.tick()

    def revoke() -> None:
        h.clock.advance(h.plan.deadline_at - h.clock())
        h.tick()

    h.cpu.before_patch = revoke
    with pytest.raises(RegionalFixtureError):
        h.submit()
    assert h.posts == 0 and h.port.read().control.revocation is not None
    assert h.port.read().control.producer.state == "NOT_STARTED"


def test_revocation_after_claim_ack_is_rechecked_before_post() -> None:
    h = build_control()
    h.tick()
    cpu = h.cpu

    def revoke_after_patch(*args: str, **kwargs: Any) -> str:
        response = cpu(*args, **kwargs)
        if args[0] == "patch":
            h.clock.advance(h.plan.deadline_at - h.clock())
            h.tick()
        return response

    h.control.cpu = revoke_after_patch
    with pytest.raises(RegionalFixtureError):
        h.submit()
    assert h.posts == 0 and h.port.read().control.producer.state == "SUBMITTING"
    assert h.port.read().control.revocation is not None


def test_lost_cas_ack_is_reconciled_by_exact_uid_plan_and_local_claim() -> None:
    h = build_control()
    h.tick()
    h.cpu.patches.append(TimeoutError("private-api-diagnostic"))
    claim_id = h.control.claim()
    assert h.port.read().control.producer.claim_id == claim_id
    assert len([args for args, _ in h.cpu.calls if args[0] == "patch"]) == 1
    with pytest.raises(RegionalFixtureError):
        h.control.claim()


def test_conflicts_are_retried_without_a_second_producer_claim() -> None:
    h = build_control()
    h.tick()
    h.api.patch_errors = [ApiException(status=409), ApiException(status=409)]
    claim_id = h.control.claim()
    assert len([args for args, _ in h.cpu.calls if args[0] == "patch"]) == 3
    assert h.port.read().control.producer.claim_id == claim_id
    assert len(h.api.patches) == 2, "only initial arm and one claim were committed"


def test_cas_retry_exhaustion_and_unknown_readback_never_authorize_post() -> None:
    h = build_control()
    h.tick()
    h.api.patch_errors = [ApiException(status=409)] * 4
    with pytest.raises(RegionalFixtureError, match="exhausted"):
        h.submit()
    assert h.posts == 0 and h.port.read().control.producer.state == "NOT_STARTED"
    h.cpu.patches.append(TimeoutError("private"))
    h.api.after_patch = lambda: setattr(h.api.value.metadata, "uid", "replacement")
    with pytest.raises(RegionalFixtureError):
        h.submit()
    assert h.posts == 0


def test_direct_control_ack_must_not_replace_watchdog_status() -> None:
    # The parent's control CAS replaces only control.json, so it can never
    # overwrite the daemon's status.json. A daemon heartbeat that rewrites
    # status.json concurrently (here, on every patch) must therefore neither
    # block the claim -- that pinned-everything CAS was the live control
    # livelock -- nor lose the daemon's status. The claim converges and the
    # watchdog receipt is preserved and advances monotonically.
    h = build_control()
    h.tick()
    before = h.port.read()
    assert before.status is not None and before.status.state == "ARMED"
    h.api.after_patch = h.tick
    claim_id = h.submit()
    assert claim_id and h.posts == 1
    after = h.port.read()
    assert after.control.producer.state == "ACKNOWLEDGED"
    assert after.status is not None and after.status.state == "ARMED"
    assert after.status.sequence > before.status.sequence


@pytest.mark.parametrize("change", ["version", "control", "plan", "shape"])
def test_unconfirmed_direct_patch_response_is_not_adopted(change: str) -> None:
    h = build_control()
    h.tick()
    document = h.cpu.document()
    if change == "shape":
        raw = "[]"
    else:
        if change == "version":
            document["metadata"]["resourceVersion"] = ""
        elif change == "control":
            document["metadata"]["resourceVersion"] = "999"
        else:
            value = json.loads(document["data"]["plan.json"])
            value["release_id"] = "foreign"
            document["data"]["plan.json"] = wire.encode(value)
        raw = wire.encode(document)
    h.cpu.patches.append(raw)
    with pytest.raises(RegionalFixtureError):
        h.submit()
    assert h.posts == 0 and h.port.read().control.producer.state == "SUBMITTING"


def test_late_ack_after_revocation_closes_source_without_reopening_it() -> None:
    h = build_control()
    h.tick()
    claim_id = h.control.claim()
    seed(h.store, h.plan, status=WorkflowStatus.FAILED)
    h.clock.advance(h.plan.deadline_at - h.clock())
    failure = h.tick()
    assert failure.error_code == "SOURCE_UNRESOLVED"
    h.control.acknowledge(claim_id, h.response())
    revoked = h.port.read().control.revocation
    before = list(h.api.patches)
    h.clock.advance(1)
    h.control.acknowledge(claim_id, h.response())
    assert h.api.patches == before, "same direct source ACK is idempotent"
    with pytest.raises(RegionalFixtureError):
        h.control.claim()
    h.cleanup(attempt_id="cleanup-after-ack")
    h.tick()
    h.clock.advance(5)
    actual = h.tick()
    assert h.control.quiescence() == actual and actual.case_failed
    assert actual.failure == failure.failure and actual.revocation == revoked


def test_ack_cas_can_race_revocation_but_cannot_clear_it() -> None:
    h = build_control()
    h.tick()
    claim_id = h.control.claim()
    seed(h.store, h.plan, status=WorkflowStatus.FAILED)

    def revoke() -> None:
        h.clock.advance(h.plan.deadline_at - h.clock())
        h.tick()

    h.cpu.before_patch = revoke
    h.control.acknowledge(claim_id, h.response())
    current = h.port.read().control
    assert current.producer.state == "ACKNOWLEDGED" and current.revocation is not None
    assert current.revocation.producer_state == "SUBMITTING"


@pytest.mark.parametrize(
    "change",
    [
        {"status": True},
        {"status": 202},
        {"body": None},
        {"body": {"batch_id": "foreign"}},
    ],
)
def test_incomplete_http_acknowledgement_is_not_source_completion(
    change: dict[str, Any],
) -> None:
    h = build_control()
    h.tick()
    claim_id = h.control.claim()
    response = {**h.response(), **change}
    with pytest.raises(RegionalFixtureError, match="acknowledgement"):
        h.control.acknowledge(claim_id, response)
    assert h.port.read().control.producer.state == "SUBMITTING"


@pytest.mark.parametrize(
    "change",
    [
        {"duplicate": True},
        {"duplicate": 0},
        {"batch_id": "foreign"},
        {"incident_ids": []},
        {"incident_ids": ["a", "b"]},
        {"workflow_request_ids": []},
        {"workflow_request_ids": ["a", "b"]},
        {"incident_ids": [None]},
        {"workflow_request_ids": [""]},
        {"unknown": "private-data"},
        {"marker_ids": [None]},
    ],
)
def test_ack_body_shape_and_identity_are_strict(change: dict[str, Any]) -> None:
    h = build_control()
    h.tick()
    claim_id = h.control.claim()
    response = h.response()
    response["body"].update(change)
    with pytest.raises(RegionalFixtureError, match="acknowledgement") as failure:
        h.control.acknowledge(claim_id, response)
    assert "private-data" not in str(failure.value)
    assert h.port.read().control.producer.ack is None


@pytest.mark.parametrize(
    "field",
    [
        "cluster_id",
        "job_id",
        "attempt_id",
        "event_id",
        "node_id",
        "runtime_profile_version",
        "affected_workload_ids",
    ],
)
def test_source_findings_bind_every_target_identity(field: str) -> None:
    h = build_control()
    h.tick()
    claim_id = h.control.claim()
    finding = NodeHealthFinding(
        finding_id="finding-a",
        event_id=h.plan.event_id,
        cluster_id=h.plan.cluster_id,
        job_id=h.plan.job_id,
        attempt_id=h.plan.attempt_id,
        node_id=h.plan.fault_node,
        observed_at=STAMP,
        category=NodeHealthCategory.GPU,
        severity=Severity.CRITICAL,
        reason="local synthetic response",
        recommended_action=RecoveryAction.REPLACE_NODE,
        workload_state=WorkloadState.ACTIVE,
        runtime_profile_version=h.plan.runtime_profile_version,
        affected_workload_ids=h.plan.workload_ids,
    ).model_dump(mode="json")
    finding[field] = ["foreign"] if field == "affected_workload_ids" else "foreign"
    response = h.response()
    response["body"]["findings"] = [finding]
    with pytest.raises(RegionalFixtureError, match="acknowledgement"):
        h.control.acknowledge(claim_id, response)
    assert h.port.read().control.producer.ack is None


def test_changed_ack_or_claim_is_rejected_without_rewriting_completion() -> None:
    h = build_control()
    h.tick()
    claim_id = h.submit()
    original = h.port.read().control
    for changes in [
        {"claim_id": "foreign"},
        {
            "response": {
                "status": 200,
                "body": {**h.response()["body"], "incident_ids": ["foreign"]},
            }
        },
    ]:
        with pytest.raises(RegionalFixtureError):
            h.control.acknowledge(
                changes.get("claim_id", claim_id), changes.get("response", h.response())
            )
    assert h.port.read().control == original


@pytest.mark.parametrize("key", ["namespace", "name", "uid"])
@pytest.mark.parametrize("value", ["", "--foreign", "bad/value", "x" * 254])
def test_controller_identity_is_validated_before_commands(key: str, value: str) -> None:
    h = build_control()
    identity = {
        "namespace": h.api.namespace,
        "name": h.api.name,
        "uid": h.api.uid,
        key: value,
    }
    with pytest.raises(RegionalFixtureError, match="identity"):
        CancellationControl(plan=h.plan, cpu=h.cpu, clock=h.clock, **identity)
    assert h.cpu.calls == []


def test_probe_source_and_local_plan_mutation_are_refused() -> None:
    with pytest.raises(RegionalFixtureError, match="source identity"):
        build_control(probe_sha256="b" * 64)
    h = build_control()
    h.control.plan.workload_ids.append("foreign")
    with pytest.raises(RegionalFixtureError, match="binding changed"):
        h.control.read()
    assert h.cpu.calls == []


@pytest.mark.parametrize("field", ["namespace", "name", "uid"])
def test_local_control_identity_cannot_move_to_another_resource(field: str) -> None:
    h = build_control()
    setattr(h.control, field, "foreign")
    with pytest.raises(RegionalFixtureError, match="binding changed"):
        h.control.identity()
    assert h.cpu.calls == []


@pytest.mark.parametrize(
    "raw",
    [
        "not-json-private-value",
        "null",
        "[]",
        "true",
        '{"kind":"ConfigMap","kind":"ConfigMap"}',
        '{"metadata":{"x":NaN}}',
        "x" * (MAX_CONTROL_BYTES + 1),
    ],
)
def test_malformed_control_response_is_sanitized(raw: str) -> None:
    h = build_control()
    h.cpu.reads.append(raw)
    with pytest.raises(RegionalFixtureError) as failure:
        h.control.read()
    assert "private-value" not in str(failure.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("apiVersion", "v2"),
        ("kind", "Secret"),
        ("metadata", []),
        ("data", {}),
        ("binaryData", {"private": "value"}),
        ("immutable", 0),
        ("metadata.uid", "foreign"),
        ("metadata.name", "foreign"),
        ("metadata.namespace", "foreign"),
        ("metadata.resourceVersion", ""),
        ("metadata.resourceVersion", 1),
        ("metadata.resourceVersion", "bad version"),
        ("metadata.deletionTimestamp", ""),
        ("metadata.deletionTimestamp", False),
    ],
)
def test_configmap_identity_and_shape_changes_are_not_absence(
    field: str, value: Any
) -> None:
    h = build_control()
    document = h.cpu.document()
    if "." in field:
        parent, key = field.split(".")
        document[parent][key] = value
    else:
        document[field] = value
    h.cpu.reads.append(wire.encode(document))
    with pytest.raises(RegionalFixtureError):
        h.control.read()
    assert h.api.patches == []


@pytest.mark.parametrize("key", ["plan.json", "control.json", "status.json"])
def test_owned_data_model_changes_fail_closed(key: str) -> None:
    h = build_control()
    h.tick()
    h.rewrite(key, {"private": "opaque-value"})
    with pytest.raises(RegionalFixtureError) as failure:
        h.control.read()
    assert "opaque-value" not in str(failure.value)


def test_transport_errors_do_not_disclose_diagnostics() -> None:
    h = build_control()
    h.cpu.reads.append(RuntimeError("postgresql://private-diagnostic"))
    with pytest.raises(RegionalFixtureError, match="read failed") as failure:
        h.control.read()
    assert "private-diagnostic" not in str(failure.value)


def test_stale_receipt_requires_explicit_cleanup_observation_not_terminal_replay() -> (
    None
):
    h = build_control()
    h.tick()
    h.submit()
    old = h.finish()
    assert h.control.quiescence() == old
    h.clock.advance(MAX_RECEIPT_AGE + 1)
    assert h.tick() == old, "ordinary completed-job retry keeps its terminal fast path"
    with pytest.raises(RegionalFixtureError, match="quiescence"):
        h.control.quiescence()
    h.cleanup(attempt_id="fresh-verification")
    first = h.tick()
    assert first.state == "REVOKED" and first.quiet_since == int(h.clock())
    with pytest.raises(RegionalFixtureError, match="quiescence"):
        h.control.quiescence()
    h.clock.advance(5)
    current = h.tick()
    assert h.control.quiescence() == current
    assert current.sequence > old.sequence and current.revocation == old.revocation


def test_failed_case_is_preserved_when_cleanup_refreshes_its_receipt() -> None:
    h = build_control()
    h.tick()
    h.submit()
    failed = probe.persist_failure(
        h.port, code="PARENT_LOST", now=int(h.clock()), monitoring=False
    )
    h.control.read()
    h.clock.advance(MAX_RECEIPT_AGE + 1)
    with pytest.raises(RegionalFixtureError, match="quiescence"):
        h.control.quiescence()
    h.cleanup(attempt_id="recovery")
    h.tick()
    h.clock.advance(5)
    actual = h.tick()
    assert h.control.quiescence() == actual
    assert actual.failure == failed.failure and actual.case_failed


def test_real_inflight_command_never_provides_retirement_authority() -> None:
    h = build_control()
    h.tick()
    h.submit(command_status=RemoteCommandStatus.LEASED)
    h.control.request_close()
    current = h.tick()
    assert current.commands_active == 1
    with pytest.raises(RegionalFixtureError, match="quiescence"):
        h.control.quiescence()
    assert h.store.get_remote_command("command-a").status is RemoteCommandStatus.LEASED


def test_wait_quiescence_observes_real_watchdog_progress_with_bounded_reads() -> None:
    h = build_control()
    h.tick()
    h.submit()
    h.control.request_close()
    actual = h.control.wait_quiescence(seconds=10, sleep=h.sleep)
    assert actual == h.port.read().status and actual.state == "QUIESCENT"
    assert all(0 < timeout <= 30 for _, timeout in h.cpu.calls), (
        "CPU requests must have positive timeouts of at most 30 seconds"
    )


@pytest.mark.parametrize(
    "seconds", [0, -1, True, 1.5, float("nan"), float("inf"), 7201]
)
def test_wait_budget_is_strict_and_finite(seconds: Any) -> None:
    h = build_control()
    with pytest.raises(RegionalFixtureError, match="duration"):
        h.control.wait_quiescence(seconds=seconds, sleep=h.sleep)
    assert h.cpu.calls == []


def test_wait_stops_for_terminal_failure_and_expires_without_a_receipt() -> None:
    h = build_control()
    with pytest.raises(RegionalFixtureError, match="deadline expired"):
        h.control.wait_quiescence(seconds=2, sleep=h.clock.advance)
    h.tick()
    probe.persist_failure(
        h.port, code="LOCAL_FAILURE", now=int(h.clock()), monitoring=False
    )
    with pytest.raises(RegionalFixtureError, match="could not prove"):
        h.control.wait_quiescence(seconds=2, sleep=h.clock.advance)


def test_wait_cannot_accept_quiescence_returned_after_its_budget() -> None:
    h = build_control()
    h.tick()
    h.finish()
    cpu = h.cpu

    def slow(*args: str, **kwargs: Any) -> str:
        response = cpu(*args, **kwargs)
        h.clock.advance(6)
        return response

    h.control.cpu = slow
    with pytest.raises(RegionalFixtureError, match="deadline expired"):
        h.control.wait_quiescence(seconds=5, sleep=h.clock.advance)
    assert h.cpu.calls[-1][1] == 5


def test_plan_bytes_and_unversioned_data_are_not_mutable_control() -> None:
    h = build_control()
    h.tick()
    h.control.read()
    h.api.value.data["plan.json"] = json.dumps(h.plan.model_dump(mode="json"), indent=2)
    with pytest.raises(RegionalFixtureError, match="regressed or changed"):
        h.control.read()
    h.api.value.metadata.resource_version = str(
        int(h.api.value.metadata.resource_version) + 1
    )
    with pytest.raises(RegionalFixtureError, match="regressed or changed"):
        h.control.read()


def test_parent_change_cannot_revoke_or_roll_back_submission_authority() -> None:
    h = build_control()
    h.tick()
    before = list(h.api.patches)
    with pytest.raises(RegionalFixtureError, match="revocation"):
        h.control.change(
            lambda control, now: wire.revoke(control, now=now, reason="FAILURE")
        )
    assert h.api.patches == before
    h.control.claim()
    before = list(h.api.patches)
    with pytest.raises(RegionalFixtureError, match="transition"):
        h.control.change(
            lambda control, _: control.model_copy(
                update={"producer": wire.initial_control(h.plan).producer}
            )
        )
    assert h.api.patches == before


@pytest.mark.parametrize(
    "change",
    [{"status": "foreign"}, {"plan_sha256": "b" * 64}, {"revocation": "private"}],
)
def test_invalid_parent_transform_is_refused_before_patch(
    change: dict[str, Any],
) -> None:
    h = build_control()
    h.tick()
    before = list(h.api.patches)
    with pytest.raises(RegionalFixtureError):
        h.control.change(lambda control, _: control.model_copy(update=change))
    assert h.api.patches == before


def test_invalid_plan_and_unserializable_local_binding_are_refused() -> None:
    h = build_control()
    with pytest.raises(RegionalFixtureError, match="plan is invalid"):
        CancellationControl(
            plan=h.plan.model_copy(update={"schema_version": 2}),
            namespace=h.api.namespace,
            name=h.api.name,
            uid=h.api.uid,
            cpu=h.cpu,
        )
    h.control.plan = object()
    with pytest.raises(RegionalFixtureError, match="binding changed"):
        h.control.identity()
    assert not h.cpu.calls, "invalid local bindings must fail before CPU access"


@pytest.mark.parametrize(
    "clock", [False, 0, -1, float("inf"), float("nan"), "private-clock"]
)
def test_control_clock_is_finite_and_typed(clock: Any) -> None:
    h = build_control()
    h.control.clock = lambda: clock
    with pytest.raises(RegionalFixtureError, match="clock") as failure:
        h.control.read()
    assert "private-clock" not in str(failure.value)


@pytest.mark.parametrize("timeout", [False, 0, -1, 31, float("inf"), float("nan")])
def test_per_request_timeout_is_bounded_before_any_command(timeout: Any) -> None:
    h = build_control()
    with pytest.raises(RegionalFixtureError, match="timeout"):
        h.control.read(timeout=timeout)
    assert not h.cpu.calls, "invalid timeouts must fail before CPU access"


def test_noop_and_repeated_close_do_not_change_receipt_or_control() -> None:
    h = build_control()
    h.tick()
    before = h.control.read()
    assert h.control.change(lambda control, _: control) == before
    h.control.request_close()
    closed = h.control.read()
    patches = list(h.api.patches)
    h.clock.advance(1)
    with pytest.raises(RegionalFixtureError, match="immutable"):
        h.control.change(
            lambda control, now: control.model_copy(
                update={
                    "close_request": control.close_request.model_copy(
                        update={"requested_at": now}
                    )
                }
            )
        )
    assert h.control.read() == closed and h.api.patches == patches


def test_oversized_direct_ack_cannot_complete_the_producer() -> None:
    h = build_control()
    h.tick()
    claim_id = h.control.claim()
    response = h.response()
    response["body"]["marker_ids"] = ["x" * MAX_CONTROL_BYTES]
    with pytest.raises(RegionalFixtureError, match="acknowledgement"):
        h.control.acknowledge(claim_id, response)
    assert h.port.read().control.producer.state == "SUBMITTING"


@pytest.mark.parametrize("clock_values", [[float("nan")], [10, float("inf")], [10, 9]])
def test_wait_clock_cannot_be_nonfinite_or_move_backwards(
    clock_values: list[float],
) -> None:
    h = build_control()
    values = iter(clock_values)
    h.control.monotonic = lambda: next(values)
    with pytest.raises(RegionalFixtureError, match="clock"):
        h.control.wait_quiescence(seconds=5, sleep=h.clock.advance)
    assert not h.cpu.calls, "invalid wait clocks must fail before CPU access"


@pytest.mark.parametrize("clock_values", [[0, 0, 0, 6], [0, 0, 0, 1, 6]])
def test_wait_cannot_overrun_between_quiescence_reads(
    clock_values: list[float],
) -> None:
    h = build_control()
    h.tick()
    h.finish()
    values = iter(clock_values)
    h.control.monotonic = lambda: next(values)
    with pytest.raises(RegionalFixtureError, match="deadline expired"):
        h.control.wait_quiescence(seconds=5, sleep=h.clock.advance)


@pytest.mark.parametrize(
    "change", ["missing", "sequence", "same-sequence", "failure", "root", "commands"]
)
def test_receipt_history_cannot_be_removed_or_rewritten(change: str) -> None:
    h = build_control()
    h.tick()
    h.submit(command_status=RemoteCommandStatus.PENDING)
    probe.persist_failure(
        h.port, code="LOCAL_FAILURE", now=int(h.clock()), monitoring=False
    )
    h.cleanup(attempt_id="history-proof")
    h.tick()
    h.clock.advance(5)
    current = h.tick()
    assert h.control.quiescence() == current
    if change == "missing":
        h.rewrite("status.json", None)
    else:
        value = current.model_dump(mode="json")
        if change == "sequence":
            value["sequence"] -= 1
        elif change == "same-sequence":
            value["observed_at"] += 1
            h.clock.advance(1)
        elif change == "failure":
            value["sequence"] += 1
            value["failure"]["code"] = "REPLACED_FAILURE"
        elif change == "root":
            # A failed observation may have unknown root data, but cannot erase a
            # previously proven root for the same parent controller.
            value.update(
                sequence=value["sequence"] + 1,
                state="FAILED",
                error_code="UNKNOWN",
                source_complete=False,
                root=None,
                quiet_since=None,
            )
        else:
            value.update(sequence=value["sequence"] + 1, command_ids=[])
        h.rewrite("status.json", value)
    with pytest.raises(RegionalFixtureError, match="regressed or changed"):
        h.control.read()


def test_producer_claim_and_close_tombstones_cannot_regress() -> None:
    h = build_control()
    h.tick()
    h.control.claim()
    h.rewrite("control.json", wire.initial_control(h.plan))
    with pytest.raises(RegionalFixtureError, match="regressed or changed"):
        h.control.read()
    h = build_control()
    h.tick()
    h.control.request_close()
    old = h.port.read().control
    h.rewrite("control.json", old.model_copy(update={"close_request": None}))
    with pytest.raises(RegionalFixtureError, match="regressed or changed"):
        h.control.read()
