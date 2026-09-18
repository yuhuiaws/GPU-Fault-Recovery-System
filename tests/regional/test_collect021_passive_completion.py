"""The late-XID case waits for the actual source plan and replacement Observation."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    CompletionDecision,
    FaultIncident,
    RecoveryPlan,
    TerminalEvent,
    WorkflowRequest,
)
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.store import InMemoryStore
from gpu_fault.watcher import AttemptObservation
from scripts.e2e.regional import collect021_passive as passive
from tests.regional._collect021_passive_support import completed_passive_state

OPTIONS = {"cluster_id": "cluster-a", "job_id": "job", "attempt_id": "attempt-a"}


def replacement(state):
    return {
        "pods": [
            {"uid": item["pod_uid"], "attempt_id": "attempt-b"}
            for item in state["observations"][0]["containers"]
        ]
    }


@pytest.mark.parametrize("with_predecessor", [False, True])
def test_current_typed_plan_and_optional_containment_chain_are_accepted(
    with_predecessor,
):
    state = completed_passive_state(with_predecessor=with_predecessor)
    assert passive.completion_chain_errors(state, **OPTIONS) == [], (
        "the completed current source plan and its actual predecessor are proved"
    )


@pytest.mark.parametrize(
    ("collection", "field"),
    [
        ("completion_event", "cluster_id"),
        ("completion_event", "job_id"),
        ("completion_event", "attempt_id"),
        ("completion_event", "terminal_status"),
        ("completion_event", "allocation"),
        ("completion_event", "event_key"),
        ("completion_decision", "event_key"),
        ("completion_decision", "cluster_id"),
        ("completion_decision", "attempt_id"),
        ("completion_decision", "status"),
        ("completion_decision", "recovery_plan_id"),
        ("recovery_plan", "plan_id"),
        ("recovery_plan", "attempt_id"),
        ("recovery_plan", "trigger"),
        ("recovery_plan", "steps"),
        ("recovery_plan", "status"),
        ("recovery_plan", "workflow_request_id"),
        ("recovery_plan", "incident_id"),
        ("recovery_workflow", "request_id"),
        ("recovery_workflow", "source_plan_id"),
        ("recovery_workflow", "status"),
        ("recovery_workflow", "incident_id"),
        ("recovery_workflow", "official_steps"),
        ("recovery_workflow", "step_executions"),
        ("recovery_incident", "incident_id"),
        ("recovery_incident", "cluster_id"),
        ("recovery_incident", "job_id"),
        ("recovery_incident", "attempt_id"),
        ("predecessor_workflow", "request_id"),
        ("predecessor_workflow", "status"),
        ("predecessor_workflow", "incident_id"),
        ("predecessor_incident", "incident_id"),
        ("predecessor_incident", "cluster_id"),
        ("predecessor_incident", "job_id"),
        ("predecessor_incident", "state"),
    ],
)
def test_each_required_plan_binding_is_checked(collection, field):
    state = completed_passive_state(with_predecessor=True)
    state[collection][field] = None
    assert passive.completion_chain_errors(state, **OPTIONS), (
        f"missing {collection}/{field} cannot prove this attempt's recovery"
    )


def test_unknown_scope_and_wrong_containment_reference_are_not_inferred():
    state = completed_passive_state(with_predecessor=True)
    assert passive.completion_chain_errors(state, **{**OPTIONS, "job_id": ""}), (
        "anonymous job scope is not accepted"
    )
    state["recovery_plan"]["restart_after_incident_id"] = "unrelated"
    assert passive.completion_chain_errors(state, **OPTIONS), (
        "a completed unrelated predecessor is not the plan's containment"
    )


class Reader:
    settings = SimpleNamespace(cluster_id="cluster-a")

    def __init__(self, states):
        self.states = iter(states)
        self.last = {}
        self.calls = []

    def cpu_python(self, source, *arguments):
        self.calls.append(arguments)
        self.last = next(self.states, self.last)
        return self.last


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        passive,
        "time",
        SimpleNamespace(
            monotonic=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        ),
    )
    return now


def test_replacement_pod_visibility_waits_for_completion_and_watcher_lag(clock):
    complete = completed_passive_state()
    lagging = {**complete, "observations": []}
    pending = {**complete, "recovery_plan": None}
    reader = Reader([lagging, pending, complete])
    result = passive.wait_passive_completion(
        reader, job_id="job", attempt_id="attempt-a", restarted=replacement(complete)
    )
    assert result == complete, "the returned state must contain the converged proof"
    assert len(reader.calls) == 3, "Pod replacement alone must not end the wait"
    assert set(reader.calls) == {("cluster-a", "job", "attempt-a", "attempt-b")}, (
        "polling must stay bound to the original source and actual new attempt"
    )


def test_restart_proof_precedes_and_does_not_require_replacement_pod_visibility(clock):
    complete = {**completed_passive_state(), "observations": []}
    reader = Reader([{**complete, "recovery_plan": None}, complete])
    state = passive.wait_passive_restart(reader, job_id="job", attempt_id="attempt-a")
    assert state is complete
    assert (
        state["commands"][0]["workflow_request_id"]
        == (state["recovery_workflow"]["request_id"])
    )
    assert reader.calls == [("cluster-a", "job", "attempt-a", "")] * 2


@pytest.mark.parametrize("scope", ["cluster_id", "job_id", "attempt_id"])
def test_restart_proof_rejects_unknown_scope_before_reading(clock, scope):
    reader = Reader([])
    options = {"job_id": "job", "attempt_id": "attempt-a"}
    if scope == "cluster_id":
        reader.settings = SimpleNamespace(cluster_id="")
    else:
        options[scope] = ""
    with pytest.raises(
        passive.RegionalFixtureError, match="explicit cluster/job/attempt"
    ):
        passive.wait_passive_restart(reader, **options)
    assert reader.calls == []


@pytest.mark.parametrize("proof", ["monitor-only", "failed", "wrong-job"])
def test_restart_proof_never_promotes_a_different_or_unfinished_chain(clock, proof):
    state = completed_passive_state()
    if proof == "monitor-only":
        state = {"workflow": None, "decision": {"disposition": "MONITOR_ONLY"}}
    elif proof == "failed":
        state["recovery_workflow"]["status"] = "FAILED"
    else:
        state["recovery_incident"]["job_id"] = "foreign"
    with pytest.raises(
        passive.RegionalFixtureError, match="restart evidence did not converge"
    ):
        passive.wait_passive_restart(
            Reader([state]), job_id="job", attempt_id="attempt-a", timeout_seconds=6
        )


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "foreign-job",
        "foreign-cluster",
        "source-attempt",
        "stopped",
        "wrong-pods",
        "duplicate",
        "stale",
        "future",
        "bad-time",
        "naive-time",
        "missing-capture",
        "budget",
        "boolean-budget",
    ],
)
def test_unproven_replacement_does_not_pass_even_when_pods_exist(clock, defect):
    state = completed_passive_state()
    restarted = replacement(state)
    observation = state["observations"][0]
    if defect == "missing":
        state["observations"] = []
    elif defect in {"foreign-job", "foreign-cluster", "source-attempt", "stopped"}:
        key, value = {
            "foreign-job": ("job_id", "other"),
            "foreign-cluster": ("cluster_id", "other"),
            "source-attempt": ("attempt_id", "attempt-a"),
            "stopped": ("workload_phase", "STOPPED"),
        }[defect]
        observation[key] = value
    elif defect == "wrong-pods":
        observation["containers"][0]["pod_uid"] = "unrelated"
    elif defect == "duplicate":
        state["observations"].append(deepcopy(observation))
    elif defect in {"stale", "future"}:
        captured = datetime.fromisoformat(state["captured_at"])
        delta = -121 if defect == "stale" else 6
        observation["observed_at"] = (captured + timedelta(seconds=delta)).isoformat()
    elif defect == "bad-time":
        observation["observed_at"] = "invalid"
    elif defect == "naive-time":
        observation["observed_at"] = "2030-01-01T00:00:00"
    elif defect == "missing-capture":
        state.pop("captured_at")
    else:
        state["restart_budget"]["restart_count"] = (
            True if defect == "boolean-budget" else 0
        )
    with pytest.raises(RuntimeError, match="did not converge"):
        passive.wait_passive_completion(
            Reader([state]),
            job_id="job",
            attempt_id="attempt-a",
            restarted=restarted,
            timeout_seconds=6,
        )


@pytest.mark.parametrize("pods", [[], [{}], [{"uid": "x", "attempt_id": "attempt-a"}]])
def test_unbound_replacement_refuses_before_any_store_read(clock, pods):
    reader = Reader([])
    with pytest.raises(RuntimeError, match="one new attempt"):
        passive.wait_passive_completion(
            reader, job_id="job", attempt_id="attempt-a", restarted={"pods": pods}
        )
    assert reader.calls == [], "invalid Pod evidence cannot select a Store scope"


def seed_store(state):
    store = InMemoryStore()
    event = {
        key: value
        for key, value in state["completion_event"].items()
        if key != "event_key"
    }
    store.save_event_if_absent(TerminalEvent.model_validate(event))
    store.save_decision(CompletionDecision.model_validate(state["completion_decision"]))
    store.save_plan(RecoveryPlan.model_validate(state["recovery_plan"]))
    for key in ("recovery_incident", "predecessor_incident"):
        if state[key] is not None:
            store.save_incident(FaultIncident.model_validate(state[key]))
    for key in ("recovery_workflow", "predecessor_workflow"):
        if state[key] is not None:
            store.save_workflow(WorkflowRequest.model_validate(state[key]))
    for observation in state["observations"]:
        store.save_attempt_observation(AttemptObservation.model_validate(observation))
    for command in state["commands"]:
        store.ensure_remote_command(RemoteActionCommand.model_validate(command))
    store.reserve_job_restart("cluster-a", "job", 1, "one-restart")
    return store


@pytest.mark.parametrize("populated", [False, True])
def test_emitted_probe_reads_the_actual_store_api_and_handles_not_yet_created_rows(
    monkeypatch, populated
):
    expected = completed_passive_state(with_predecessor=True)
    store = seed_store(expected) if populated else InMemoryStore()
    if populated:
        command = store.get_remote_command("recovery-command")
        command.lease_token = "fixture-only-lease"
        store.ensure_remote_command(
            command.model_copy(
                update={
                    "command_id": "unrelated-command",
                    "workflow_request_id": "unrelated-workflow",
                }
            )
        )
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        classmethod(lambda cls: SimpleNamespace(store=store)),
    )
    monkeypatch.setattr(
        sys, "argv", ["probe", "cluster-a", "job", "attempt-a", "attempt-b"]
    )
    output = io.StringIO()
    with redirect_stdout(output):
        exec(passive.PASSIVE_PROBE, {"__name__": "__main__"})
    observed = json.loads(output.getvalue())
    if populated:
        assert passive.completion_chain_errors(observed, **OPTIONS) == [], (
            "the installed probe must consume actual typed Store methods"
        )
        assert observed["observations"] == expected["observations"], (
            "the probe must return only the bound replacement attempt"
        )
        assert observed["commands"] == expected["commands"], (
            "only the actual recovery workflow's commands may authorize custody"
        )
        assert "lease_token" not in observed["commands"][0]
    else:
        assert observed["completion_event"] is None, (
            "a not-yet-created event is a pending proof, not an invented success"
        )
        assert passive.completion_chain_errors(observed, **OPTIONS), (
            "empty Store observations cannot satisfy the live completion proof"
        )
        assert observed["commands"] == []
