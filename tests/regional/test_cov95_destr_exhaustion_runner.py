from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.e2e.regional import run_destr014_branch_exhaustion as case
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_exhaustion import (
    RECOVERY_IDENTITY_REFUSAL,
    ExhaustionHarness,
    arbiter_pod,
    dns_pod,
)


def test_real_exhaustion_preflight_still_refuses_without_reboot_surviving_safeguard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ExhaustionHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == [], (
        "bound recovery identity must pass real preflight"
    )
    h.agents[h.settings.sibling_node].pop("artifact_sha256")
    preflight = h.plan(tmp_path)
    assert preflight["errors"] == [RECOVERY_IDENTITY_REFUSAL], preflight
    with pytest.raises(RegionalFixtureError, match="recovery identity"):
        h.execute(tmp_path)
    assert not any(
        name in {"control.open", "executor.open", "probe.create"} for name, _ in h.calls
    ), h.calls


def test_real_exhaustion_preflight_refuses_arbiters_or_all_dns_on_the_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec precondition 2 on the real preflight path: a running arbiter replica
    on the fault node and kube-dns confined to the pair refuse the case before
    any env window opens or a probe lands. A terminating replica on the sibling
    has already left the node's fate and is not a placement."""

    h = ExhaustionHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == [], (
        "arbiters off the pair and one kube-dns endpoint elsewhere must pass"
    )
    fault, sibling = h.settings.fault_node, h.settings.sibling_node
    executor = arbiter_pod("gpu-fault-cluster-executor", "7c9d8b6f5-zzzzz", fault)
    h.arbiters.append(executor)
    h.dns = [dns_pod("ccccc", fault), dns_pod("ddddd", sibling)]
    preflight = h.plan(tmp_path)
    errors = preflight["errors"]
    assert any("arbiter Pods" in error and fault in error for error in errors), errors
    assert not any("arbiter" in error and sibling in error for error in errors), (
        "the terminating replica on the sibling must not count as a placement"
    )
    assert any("every kube-dns endpoint" in error for error in errors), errors
    assert preflight["arbiter_pods"] == {
        fault: [f"gpu-fault-system/{executor['metadata']['name']}"],
        "node-d": [
            "gpu-fault-system/gpu-fault-cluster-executor-7c9d8b6f5-abcde",
            "gpu-fault-system/gpu-fault-completion-watcher-5f6d7c8b9-fghij",
        ],
        "node-e": [
            "gpu-fault-system/gpu-fault-node-installer-reconciler-6d5c4b3a2-klmno"
        ],
    }, preflight["arbiter_pods"]
    assert preflight["dns_nodes"] == sorted([fault, sibling]), preflight["dns_nodes"]
    saved = json.loads(
        (tmp_path / "cases" / case.CASE_ID / "preflight.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["arbiter_pods"] == preflight["arbiter_pods"], saved
    with pytest.raises(RegionalFixtureError, match="arbiter Pods"):
        h.execute(tmp_path)
    assert not any(
        name in {"control.open", "executor.open", "probe.create"} for name, _ in h.calls
    ), h.calls
    # Moving the replica off the pair and one kube-dns endpoint elsewhere is the
    # only way back to a clean plan; the rule cannot be relaxed by the harness.
    executor["spec"]["nodeName"] = "node-d"
    h.dns.append(dns_pod("eeeee", "node-d"))
    assert h.plan(tmp_path)["errors"] == [], h.plan(tmp_path)["arbiter_pods"]


def test_exhaustion_orchestration_under_synthetic_admission_restores_known_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The historical selector now proves Agent recovery with operator debt held."""
    h = ExhaustionHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == [], (
        "valid recovery evidence must pass real preflight"
    )
    h.pending_reads = 1
    h.agent_settle_reads = 3
    code, report = h.execute(tmp_path)
    assert code == 1 and report["errors"] == [], report
    assert report["scenario_verdict"] == "PASS" and report["verdict"] == "BLOCKED"
    assert report["cleanup"]["operator_hold_preserved"]
    assert report["cleanup"]["workload_cleanup_deferred"]
    assert report["cleanup"]["errors"], "operator reconciliation is still owed"
    names = [name for name, _ in h.calls]
    order = [
        "control.open",
        "executor.open",
        "workload.submit",
        "recovery.prepare",
        "recovery.status",
        "recovery.disable",
        "probe.arm-holder",
        "probe.write-xid79",
        "probe.write-xid46",
        "recovery.restore",
        "probe.disarm-holder",
        "recovery.cleanup",
    ]
    assert [names.index(name) for name in order] == sorted(
        names.index(name) for name in order
    ), names
    assert not {"probe.disable-agent-restart", "probe.restore-agent"} & set(names), (
        "persistent recovery must never use a legacy unbound command"
    )
    assert not {
        "workload.delete",
        "agent.reactivate",
        "restore.create",
        "restore.wait",
        "executor.close",
        "control.close",
    } & set(names), (
        "unknown physical outcomes must not restore isolation or configuration"
    )
    saved = json.loads(h.journal_path.read_text())
    assert saved["phase"] == "RECOVERY_REQUIRED"
    assert saved["host_cleanup"]["phase"] == "CLOSED"
    assert saved["run"]["physical_outcome_unknown"] is True
    assert saved["host_cleanup"]["record_kind"] == "FORENSIC_TOMBSTONE"
    assert saved["host_binding"]["node_uid"] == h.nodes[h.settings.sibling_node]["uid"]
    assert h.recovery_transport.commands[:3] == ["prepare", "status", "disable"]
    assert h.host.agent_active and h.host.enable.exists(), (
        "the sibling Agent must be restored"
    )


@pytest.mark.parametrize(
    "phase",
    [
        "control.open",
        "executor.open",
        "workload.submit",
        "probe.create",
        "recovery.probe.create",
        "recovery.prepare",
        "recovery.status",
        "recovery.disable",
        "probe.arm-holder",
        "probe.write-xid79",
        "workflow.wait",
        "support.read",
        "node.ready",
        "recovery.restore",
        "recovery.cleanup",
        "recovery.probe.cleanup",
        "probe.disarm-holder",
        "workload.delete",
        "agent.reactivate",
        "restore.create",
        "restore.wait",
        "executor.close",
        "control.close",
        "prewarm.cleanup",
        "probe.cleanup",
    ],
)
def test_exhaustion_fake_phase_failure_keeps_cleanup_debt_and_blocks_pass(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ExhaustionHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == [], (
        "failure injection requires valid preflight"
    )
    h.failures[phase] = RuntimeError(f"fake failure at {phase}")
    code, report = h.execute(tmp_path)
    if phase in {
        "workload.delete",
        "agent.reactivate",
        "restore.create",
        "restore.wait",
        "executor.close",
        "control.close",
    }:
        assert code == 1 and report["verdict"] == "BLOCKED", report
        assert report["cleanup"]["operator_hold_preserved"]
        assert phase not in {name for name, _ in h.calls}, (
            "a retained hold must prevent even attempting this cleanup phase"
        )
        return
    assert code == 1 and phase in str(report), report
    names = [name for name, _ in h.calls]
    assert "prewarm.cleanup" in names and "probe.cleanup" in names, names
    if report["cleanup"].get("workload_cleanup_deferred"):
        assert not {"workload.delete", "control.close", "executor.close"} & set(
            names
        ), "unproven quiescence must preserve workload and environment windows"
        assert json.loads(h.journal_path.read_text())["phase"] == "RECOVERY_REQUIRED"
    else:
        if "control.open" in names:
            assert "control.close" in names, names
        if "executor.open" in names:
            assert "executor.close" in names, names


@pytest.mark.parametrize("phase", ["before", "control.open", "executor.open"])
def test_exhaustion_maintenance_window_is_checked_between_phases(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ExhaustionHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == [], "deadline checks require valid preflight"
    if phase == "before":
        h.clock.sleep(61)
    else:
        h.advance_at[phase] = 61
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1 and "window ended" in report["error"], report
    assert not h.injected, h.calls


def test_exhaustion_without_quiescence_keeps_workload_cleanup_deferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ExhaustionHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == [], "quiescence checks require valid preflight"
    h.failures["incident.idle"] = TimeoutError("owned commands remain live")
    code, report = h.execute(tmp_path)
    assert code == 1 and report["cleanup"]["workload_cleanup_deferred"] is True, report
    assert not any(name == "workload.delete" for name, _ in h.calls), h.calls
    assert not any(
        name in {"control.close", "executor.close"} for name, _ in h.calls
    ), "live commands must prevent environment window closure"
    assert report["cleanup"]["agent_recovery"]["phase"] == "CLOSED"
    assert json.loads(h.journal_path.read_text())["phase"] == "RECOVERY_REQUIRED"


def test_exhaustion_abort_still_restores_agent_and_closes_both_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort restores the Agent, but unproven quiescence now defers both windows."""
    h = ExhaustionHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == [], "abort injection requires valid preflight"
    h.failures["probe.write-xid79"] = RegionalFixtureAbort(2)
    with pytest.raises(RegionalFixtureAbort):
        h.execute(tmp_path)
    names = [name for name, _ in h.calls]
    assert "recovery.cleanup" in names and "probe.disarm-holder" in names, names
    assert not {"control.close", "executor.close", "workload.delete"} & set(names), (
        "abort before incident discovery cannot prove asynchronous quiescence"
    )
    assert h.host.agent_active and h.host.enable.exists(), (
        "abort must restore Agent boot activation"
    )
    saved = json.loads(h.journal_path.read_text())
    assert saved["host_cleanup"]["phase"] == "CLOSED"
    assert saved["phase"] == "RECOVERY_REQUIRED"


def test_exhaustion_probe_residue_is_not_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ExhaustionHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == [], "residual checks require valid preflight"
    h.probe_residuals["pod"] = True
    code, report = h.execute(tmp_path)
    assert code == 1 and "cleanup left residuals" in str(report["cleanup"]["errors"]), (
        report
    )


def test_exhaustion_missing_independent_arm_ack_never_disables_or_injects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ExhaustionHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == [], (
        "runtime ACK checks require valid preflight"
    )
    h.recovery_transport.arm_ack = False
    code, report = h.execute(tmp_path)
    assert code == 1 and "independent arm ACK was not observed" in report["error"]
    names = {name for name, _ in h.calls}
    assert not {"recovery.disable", "probe.arm-holder"} & names, (
        "unarmed recovery must block disable and holder"
    )
    assert not h.injected, "missing independent ACK must prevent all fault injection"
    assert h.host.enable.exists(), (
        "missing independent ACK must preserve Agent boot activation"
    )
    assert report["cleanup"]["agent_recovery"]["phase"] == "CLOSED"


def test_exhaustion_lost_disable_ack_resumes_real_journal_as_cleanup_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ExhaustionHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == [], "lost-ACK checks require valid preflight"
    h.recovery_transport.lost_ack = "disable"
    h.failures["recovery.cleanup"] = RuntimeError("temporary cleanup outage")
    code, failed = h.execute(tmp_path)
    assert code == 1 and failed["cleanup"]["errors"], failed
    saved = json.loads(h.journal_path.read_text())
    assert saved["run"]["agent_disabled"] is True
    assert saved["host_request"] == "cleanup"
    assert saved["phase"] == "RECOVERY_REQUIRED"
    h.failures.clear()
    before = len(h.calls)
    code, resumed = h.execute(tmp_path)
    assert code == 1 and "cleanup-only" in resumed["error"], resumed
    assert resumed["cleanup"]["errors"] == []
    names = {name for name, _ in h.calls[before:]}
    assert (
        not {
            "control.open",
            "executor.open",
            "workload.submit",
            "recovery.prepare",
            "recovery.disable",
            "probe.write-xid79",
            "probe.write-xid46",
        }
        & names
    ), "journal resumption must never restart the scenario"
    after = json.loads(h.journal_path.read_text())
    assert after["host_binding"] == saved["host_binding"]
    assert after["phase"] == after["host_cleanup"]["phase"] == "CLOSED"
