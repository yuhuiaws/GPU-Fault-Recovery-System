from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import late_ownership_live as live
from scripts.e2e.regional import run_destr015_parallel_branch_join as normal
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.run_late_ownership_acceptance import RunResult
from tests.regional import test_late_ownership_live as support

assembly = support.assembly


@pytest.mark.parametrize(
    "defect", ["none", "request-id", "nonce", "reply-id", "valid", "lifetime"]
)
def test_cpu_holder_checks_are_scoped_nonce_bound_and_fail_closed(
    assembly, monkeypatch, defect
):
    owned = assembly()
    scope = owned.scope
    request = {
        "workflow_id": scope.workflow_id,
        "fencing_token": scope.fencing_token,
        "execution_epoch": scope.execution_epoch,
        "nonce": "a" * 64,
    }
    reply = {
        **{key: value for key, value in request.items() if key != "nonce"},
        "holder_valid": True,
        "lifetime_deadline_at": scope.maintenance_end.isoformat(),
    }
    if defect == "request-id":
        request["workflow_id"] = "foreign"
    elif defect == "nonce":
        request["nonce"] = "short"
    elif defect == "reply-id":
        reply["execution_epoch"] += 1
    elif defect == "valid":
        reply["holder_valid"] = False
    elif defect == "lifetime":
        reply["lifetime_deadline_at"] = "old"
    calls = []

    def cpu(program, payload, **kwargs):
        calls.append(json.loads(payload))
        assert kwargs == {"attempts": 1}
        return reply

    monkeypatch.setattr(owned.run.regional, "cpu_python", cpu)
    if defect == "none":
        assert owned.io.check_holder(request) == {**reply, "nonce": request["nonce"]}
        assert calls[0]["action"] == "check"
        assert calls[0]["run_id"] == scope.run_id
    else:
        with pytest.raises(BoundaryDenied):
            owned.io.check_holder(request)
    assert len(calls) == (0 if defect in {"request-id", "nonce"} else 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope_sha256", "f" * 64),
        ("executor_uid", "other"),
        ("open_commands", 1),
        ("pending_callbacks", 1),
        ("workflow_terminal", False),
        ("gate_revoked", False),
        ("sequence", 1),
    ],
)
def test_false_quiescence_never_authorizes_physical_cleanup(assembly, field, value):
    owned = assembly()
    owned.io.arm_witnesses(owned.scope)
    owned.io.node_actions_started = True

    def callback(kind, payload):
        return {**payload, field: value} if kind == "confirm-terminal" else payload

    owned.channels[0].callback = callback
    with pytest.raises(BoundaryDenied, match="quiescence acknowledgement"):
        owned.io.quiesce(owned.scope, owned.proof.decision)
    assert not owned.io.quiet, (
        "an invalid terminal receipt cannot establish physical quiescence"
    )
    with pytest.raises(BoundaryDenied):
        owned.io.cleanup(owned.scope)
    assert (
        "mutation-cleanup" not in owned.calls and "workload-delete" not in owned.calls
    )
    assert not owned.channels[0].is_open(), (
        "refused quiescence cleanup must close the probe channel"
    )


@pytest.mark.parametrize("defect", ["executor", "node"])
def test_identity_checks_never_follow_replacement(assembly, monkeypatch, defect):
    owned = assembly()
    if defect == "executor":
        monkeypatch.setattr(live, "executor_identity", lambda regional: {"uid": "new"})
    else:
        monkeypatch.setattr(
            owned.run.regional, "node_snapshot", lambda name: {"uid": "new"}
        )
    with pytest.raises(BoundaryDenied, match="changed"):
        owned.io.check_identity()
    assert not owned.channels, (
        "replacement identity must be refused before opening any probe"
    )


def test_existing_experiment_journal_is_not_overwritten(assembly):
    owned = assembly()
    journal = owned.io.directory / "scope.json"
    before = journal.read_bytes()
    with pytest.raises(BoundaryDenied, match="journal"):
        live.LiveBoundaryIO(owned.run, owned.scope, {}, owned.io.identity)
    assert journal.read_bytes() == before


def test_missing_probe_after_start_cannot_be_a_quiescent_cleanup(assembly):
    owned = assembly()
    owned.io.node_actions_started = True
    owned.io.revoke(owned.scope)
    assert owned.io.revoked, (
        "revocation must remain effective when the started probe is missing"
    )
    with pytest.raises(BoundaryDenied, match="deferred"):
        owned.io.cleanup(owned.scope)
    assert owned.calls == []


def setup_entry(owned, monkeypatch):
    owned.run.settings.predecessor_path = (
        owned.io.directory.parent.parent.parent.parent / "ordinary.json"
    )
    settings = SimpleNamespace(
        base=owned.run.settings,
        case_id=owned.scope.case_id,
        scenario=owned.scope.scenario,
        ordinary_destr015_evidence=(
            owned.io.directory.parent.parent.parent.parent
            / "ordinary"
            / "cases"
            / normal.CASE_ID
            / f"{normal.CASE_ID}.json"
        ),
    )
    cleanup = []
    monkeypatch.setattr(live, "read_only_preflight", lambda *a, **kw: {"errors": []})
    monkeypatch.setattr(normal, "_prepare_live_run", lambda *a, **kw: owned.run)
    monkeypatch.setattr(normal, "_start_job_and_probes", lambda run: None)
    monkeypatch.setattr(normal, "_cleanup", lambda run: cleanup.append(run))
    monkeypatch.setattr(live, "build_scope", lambda *a, **kw: (owned.scope, {}, {}))
    monkeypatch.setattr(live, "LiveBoundaryIO", lambda *a: owned.io)
    return settings, cleanup


def test_live_entry_writes_only_companion_artifacts_after_complete_cleanup(
    assembly, monkeypatch, tmp_path, capsys
):
    owned = assembly("unchanged-owner")
    settings, cleanup = setup_entry(owned, monkeypatch)
    ordinary = owned.run.case_dir / f"{owned.scope.case_id}.json"
    ordinary.write_text('{"ordinary":"preserved"}')
    assert live.execute_case(settings, tmp_path, 1, owned.scope.maintenance_end) == 0
    assert json.loads(capsys.readouterr().out)["evidence_mode"] == "LOCAL_TEST"
    assert (
        json.loads((owned.io.directory / "result.json").read_text())["verdict"]
        == "PASS"
    )
    assert (owned.io.directory / "receipts.json").is_file(), (
        "a successful companion must retain its independently linked receipts"
    )
    assert ordinary.read_text() == '{"ordinary":"preserved"}'
    assert cleanup == [], "the ordinary runner must not clean a second time"


@pytest.mark.parametrize("defect", ["setup", "failed-result"])
def test_entry_setup_failure_or_failed_result_never_promotes_ordinary_case(
    assembly, monkeypatch, tmp_path, capsys, defect
):
    owned = assembly()
    settings, cleanup = setup_entry(owned, monkeypatch)
    if defect == "setup":

        def failed(run):
            raise BoundaryDenied("owned setup failed")

        monkeypatch.setattr(normal, "_start_job_and_probes", failed)
        with pytest.raises(BoundaryDenied):
            live.execute_case(settings, tmp_path, 1, owned.scope.maintenance_end)
        assert capsys.readouterr().out == ""
    else:
        monkeypatch.setattr(
            live,
            "run_case",
            lambda *a: RunResult(
                owned.scope.case_id,
                owned.scope.scenario,
                "LOCAL_TEST",
                owned.scope.digest(),
            ),
        )
        assert (
            live.execute_case(settings, tmp_path, 1, owned.scope.maintenance_end) == 1
        )
        assert json.loads(capsys.readouterr().out)["verdict"] == "FAIL"
        assert not (owned.io.directory / "receipts.json").exists(), (
            "a failed incomplete trial must not manufacture a complete receipt bundle"
        )
    assert cleanup == [owned.run]


@pytest.mark.parametrize("defect", ["inventory", "quiesce", "timers", "services"])
def test_host_baseline_must_return_before_scheduling_is_released(
    assembly, monkeypatch, defect
):
    owned = assembly()
    owned.io.arm_witnesses(owned.scope)
    before = deepcopy(owned.run.baselines["node-a"])
    if defect == "inventory":
        before["gpu_inventory"] = []
    elif defect == "quiesce":
        before["quiesce_states"] = ["pending"]
    elif defect == "timers":
        before["gpu_fault_timers"] = ["pending.timer"]
    else:
        before["services"]["kubelet"]["ActiveState"] = "inactive"
    monkeypatch.setattr(owned.run.probes["node-a"], "execute", lambda *a, **kw: before)
    with pytest.raises(BoundaryDenied, match="baseline"):
        owned.io.quiesce(owned.scope, owned.proof.decision)
    assert "gpu-restore-scheduling" not in owned.calls
    owned.io.gpu.close()
