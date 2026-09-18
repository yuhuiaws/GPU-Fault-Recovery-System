from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import audit_warm_spare_guardrails as audit
from tests.regional._cov95_warm_guard import configured as configured
from tests.regional._cov95_warm_guard import node
from tests.regional.test_guardrail_audit_evidence import recording
from tests.regional.test_warm_spare_guardrails import EXECUTOR_GUARDS, MANAGED_GUARD

pytestmark = pytest.mark.usefixtures("configured")

IDENTITY = {
    "release_id": "release-a",
    "cluster_id": "cluster-a",
    "eks_cluster_arn": "arn:aws:eks:us-west-2:111122223333:cluster/gpu",
    "registry_generation": "4",
}


def result(run_dir, case_id):
    return json.loads(
        (run_dir / "cases" / case_id / f"{case_id}.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def live_reads(monkeypatch):
    state = {
        "nodes": [node()],
        "after": None,
        "node_calls": 0,
        "environment": [
            {
                "pod": "executor",
                "cluster_id": "cluster-a",
                "spare_failover": "true",
                "remote_state": "true",
                "allow_replace": "false",
            }
        ],
        "environment_after": None,
        "environment_calls": 0,
        "identity": dict(IDENTITY),
        "events": [],
        "tests": True,
        "test_calls": 0,
        "selective": False,
    }

    def snapshot():
        state["node_calls"] += 1
        value = state["nodes"]
        if state["node_calls"] > 1 and state["after"] is not None:
            value = state["after"]
        return deepcopy(value)

    def environment():
        state["environment_calls"] += 1
        value = state["environment"]
        if state["environment_calls"] > 1 and state["environment_after"] is not None:
            value = state["environment_after"]
        return deepcopy(value)

    def tests(_run_dir, definitions):
        state["test_calls"] += 1
        if state["tests"] is None:
            raise RuntimeError("local test process unavailable")
        return {case: state["tests"] for case in definitions}

    monkeypatch.setattr(audit, "node_snapshot", snapshot)
    monkeypatch.setattr(audit, "executor_env", environment)
    monkeypatch.setattr(audit, "audit_identity", lambda: dict(state["identity"]))
    monkeypatch.setattr(audit, "run_focused_pytest", tests)
    monkeypatch.setattr(
        audit,
        "cluster_recovery",
        lambda name: {"cluster_name": name, "node_recovery": "None"},
    )
    monkeypatch.setattr(audit, "replace_events", lambda *_args: state["events"])
    monkeypatch.setattr(
        audit, "deployed_managed_owner_probe", lambda: dict(MANAGED_GUARD)
    )
    monkeypatch.setattr(
        audit, "deployed_executor_guard_probes", lambda: deepcopy(EXECUTOR_GUARDS)
    )
    monkeypatch.setattr(
        audit,
        "record_provider_snapshot",
        lambda _name: recording() | {"recorded_by": "unit-auditor"},
    )
    monkeypatch.setattr(
        audit,
        "deployed_automatic_recovery_probe",
        lambda _snapshot: {
            "observed_node_recovery": "Automatic",
            "warm_spare_guard": {
                "status": "FAILED",
                "error": audit.AUTOMATIC_GUARD_ERROR,
            },
            "control_without_warm_spare_strategy": {
                "status": "FAILED",
                "error": "HyperPod automatic node recovery is enabled",
            },
        },
    )
    monkeypatch.setattr(
        audit,
        "current_acceptance_scope",
        lambda: SimpleNamespace(
            selective=state["selective"],
            result_fields=lambda: {
                "execution_scope": "selective" if state["selective"] else "formal"
            },
        ),
    )
    return state


@pytest.mark.parametrize("case_id", audit.CASE_IDS)
@pytest.mark.parametrize("selective", [False, True])
def test_bound_audit_keeps_scope_and_both_postflight_observations(
    tmp_path, live_reads, case_id, selective
):
    live_reads["selective"] = selective
    code = audit.run_audit(tmp_path, [case_id], identity=IDENTITY)
    observed = result(tmp_path, case_id)
    assert code == 0 and observed["verdict"] == "PASS", (
        "complete mocked evidence must satisfy the selected case",
        observed,
    )
    assert observed["formal_sequence_satisfied"] is (not selective), (
        "a selective audit cannot satisfy the formal predecessor chain"
    )
    assert (live_reads["node_calls"], live_reads["environment_calls"]) == (2, 2), (
        "a passing audit must recheck both node state and executor environment"
    )
    assert all(observed[key] == value for key, value in IDENTITY.items()), (
        "the final report must retain all supplied release and registry bindings"
    )


@pytest.mark.parametrize("existing", ["empty", "quarantine", "ownership"])
def test_bound_preflight_refusal_happens_before_tests_or_deployed_probes(
    tmp_path, live_reads, existing
):
    if existing == "empty":
        live_reads["nodes"] = []
    elif existing == "quarantine":
        live_reads["nodes"][0]["taints"] = [
            {"key": audit.QUARANTINE_TAINT, "effect": "NoSchedule"}
        ]
    else:
        live_reads["nodes"][0]["ownership_annotations"][
            audit.OWNERSHIP_ANNOTATIONS[0]
        ] = "existing-incident"
    code = audit.run_audit(tmp_path, [audit.CASE_IDS[0]], identity=IDENTITY)
    observed = result(tmp_path, audit.CASE_IDS[0])
    assert code == 1 and observed["verdict"] == "FAIL", (
        "unproved clean preflight must not start the acceptance",
        observed,
    )
    assert live_reads["test_calls"] == live_reads["environment_calls"] == 0, (
        "preflight refusal must precede local and deployed probe work"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "wrong-cluster",
        "empty-executors",
        "unsafe-executor",
        "executor-drift",
        "identity-drift",
        "node-drift",
        "provider-mutation",
        "failed-tests",
        "unrun-tests",
    ],
)
def test_audit_never_promotes_missing_or_changed_evidence_to_pass(
    tmp_path, live_reads, defect
):
    if defect == "wrong-cluster":
        live_reads["environment"][0]["cluster_id"] = "foreign"
    elif defect == "empty-executors":
        live_reads["environment"] = []
    elif defect == "unsafe-executor":
        live_reads["environment"][0]["allow_replace"] = "true"
    elif defect == "executor-drift":
        live_reads["environment_after"] = []
    elif defect == "identity-drift":
        live_reads["identity"]["registry_generation"] = "5"
    elif defect == "node-drift":
        live_reads["after"] = []
    elif defect == "provider-mutation":
        live_reads["events"] = [{"event_name": "BatchRebootClusterNodes"}]
    elif defect == "failed-tests":
        live_reads["tests"] = False
    else:
        live_reads["tests"] = None
    code = audit.run_audit(tmp_path, [audit.CASE_IDS[2]], identity=IDENTITY)
    observed = result(tmp_path, audit.CASE_IDS[2])
    assert code == 1 and observed["verdict"] == "FAIL" and observed["errors"], (
        f"{defect} must remain an explicit failing result",
        observed,
    )


@pytest.mark.parametrize(
    "defect",
    ["missing-name", "missing-uid", "blank-uid", "duplicate-name", "duplicate-uid"],
)
def test_unknown_or_ambiguous_node_identity_cannot_prove_unchanged_state(
    tmp_path, live_reads, defect
):
    if defect == "missing-name":
        live_reads["nodes"][0]["name"] = None
    elif defect == "missing-uid":
        live_reads["nodes"][0]["uid"] = None
    elif defect == "blank-uid":
        live_reads["nodes"][0]["uid"] = " "
    elif defect == "duplicate-name":
        live_reads["nodes"].append(deepcopy(live_reads["nodes"][0]))
    else:
        extra = node("node-b")
        extra["uid"] = live_reads["nodes"][0]["uid"]
        live_reads["nodes"].append(extra)
    code = audit.run_audit(tmp_path, [audit.CASE_IDS[0]], identity=IDENTITY)
    observed = result(tmp_path, audit.CASE_IDS[0])
    assert code == 1 and observed["verdict"] == "FAIL", (
        f"{defect} cannot establish the identity of unchanged physical nodes"
    )
    assert live_reads["test_calls"] == 0, (
        "unproved baseline node identity must be refused before audit work"
    )


def test_duplicate_postflight_name_cannot_hide_a_replaced_node(tmp_path, live_reads):
    replacement = node()
    replacement["uid"] = "replacement-uid"
    live_reads["after"] = [replacement, node()]
    code = audit.run_audit(tmp_path, [audit.CASE_IDS[0]], identity=IDENTITY)
    observed = result(tmp_path, audit.CASE_IDS[0])
    assert code == 1 and observed["verdict"] == "FAIL", (
        "a duplicate postflight name must not erase contradictory incarnation evidence"
    )
    assert observed["node_state_identical"] is False, (
        "ambiguous postflight rows cannot prove physical state equality"
    )


@pytest.mark.parametrize("case_id", [audit.CASE_IDS[0], audit.CASE_IDS[2]])
def test_deployed_guard_wrong_result_is_not_evidence_of_the_required_refusal(case_id):
    errors = audit.probe_errors(
        case_id,
        {"status": "SUCCEEDED"},
        {"coordinator_guard": {"status": "SUCCEEDED"}, "startup_guard_error": ""},
    )
    assert errors, "a different or missing refusal cannot establish the guard"


@pytest.fixture
def main_reads(tmp_path, monkeypatch):
    state = {"calls": [], "predecessor": True, "identity_error": None, "prior": True}
    monkeypatch.setattr(
        audit,
        "os",
        SimpleNamespace(
            **(vars(os) | {"umask": lambda mask: state["calls"].append(mask)})
        ),
    )
    monkeypatch.setattr(
        audit,
        "configure",
        lambda _arguments, selected: state["calls"].append(("configure", selected)),
    )

    def identity():
        if state["identity_error"] is not None:
            raise state["identity_error"]
        return dict(IDENTITY)

    monkeypatch.setattr(audit, "audit_identity", identity)
    monkeypatch.setattr(
        audit,
        "predecessor_path",
        lambda *_args: (
            ("previous", tmp_path / "previous.json") if state["prior"] else (None, None)
        ),
    )
    monkeypatch.setattr(
        audit,
        "predecessor_evidence",
        lambda *_args, **_kwargs: {"valid": state["predecessor"]},
    )
    monkeypatch.setattr(
        audit,
        "run_audit",
        lambda run_dir, cases, *, identity: state["calls"].append(
            ("run", run_dir, cases, identity)
        )
        or 0,
    )
    return state


@pytest.mark.parametrize("run_dir_source", ["argument", "environment"])
@pytest.mark.parametrize("prior", [False, True])
def test_main_binds_the_one_selected_case_before_dispatch(
    tmp_path, monkeypatch, main_reads, run_dir_source, prior
):
    main_reads["prior"] = prior
    argv = ["audit", "--case", audit.CASE_IDS[0]]
    if run_dir_source == "argument":
        argv += ["--run-dir", str(tmp_path)]
    else:
        monkeypatch.setenv("GPU_FAULT_ACCEPTANCE_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", argv)
    assert audit.main() == 0, "the bounded mocked entrypoint must dispatch its audit"
    assert main_reads["calls"][-1] == (
        "run",
        tmp_path,
        [audit.CASE_IDS[0]],
        IDENTITY,
    ), "main must carry the exact selected case, directory and live identity"


@pytest.mark.parametrize("failure", ["predecessor", "identity", "interrupt"])
def test_main_failure_replaces_prior_success_with_current_nonpass(
    tmp_path, monkeypatch, main_reads, failure
):
    case_id = audit.CASE_IDS[0]
    path = tmp_path / "cases" / case_id / f"{case_id}.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"verdict":"PASS"}', encoding="ascii")
    if failure == "predecessor":
        main_reads["predecessor"] = False
    else:
        main_reads["identity_error"] = (
            KeyboardInterrupt()
            if failure == "interrupt"
            else RuntimeError("unreadable")
        )
    monkeypatch.setattr(
        sys, "argv", ["audit", "--case", case_id, "--run-dir", str(tmp_path)]
    )
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else RuntimeError):
        audit.main()
    observed = result(tmp_path, case_id)
    assert observed["verdict"] == "FAIL" and observed["status"] == "FAILED", (
        "a fallible preflight cannot leave an earlier canonical PASS"
    )
    assert not any(
        isinstance(call, tuple) and call[0] == "run" for call in main_reads["calls"]
    ), "failed entry preflight must not dispatch deployed work"


def test_main_requires_an_explicit_run_directory(monkeypatch, main_reads):
    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_RUN_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["audit", "--case", audit.CASE_IDS[0]])
    with pytest.raises(RuntimeError, match="explicit audit"):
        audit.main()
    assert main_reads["calls"] == [0o077], (
        "missing run directory must fail before identity reads or configuration"
    )
