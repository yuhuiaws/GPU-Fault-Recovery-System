from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import run_destr014_branch_exhaustion as case
from scripts.e2e.regional.probes import destr014_recovery_probe as host_probe
from scripts.e2e.regional.regional_live_fixture import RegionalLiveSettings
from tests.regional import test_destr014_branch_exhaustion as verdicts
from tests.regional.test_destr014_recovery_controller import ProbeTransport
from tests.regional.test_destr014_recovery_probe import HostHarness


class RunnerHarness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.host = HostHarness(tmp_path, monkeypatch)
        self.host.scope.update(
            node=verdicts.SIBLING, node_uid=f"uid-{verdicts.SIBLING}"
        )
        host_probe.AGENT_ENV.write_text(
            "\n".join(
                f"{env}={self.host.scope[key]}"
                for key, env in host_probe.PIN_KEYS.items()
            )
        )
        self.directory = tmp_path / "run-test"
        self.case_dir = self.directory / "cases" / case.CASE_ID
        self.case_dir.mkdir(parents=True)
        self.calls: list[str] = []
        self.failures: dict[str, BaseException] = {}
        self.advance: dict[str, float] = {}
        self.probe_residuals: dict[str, bool] = {}
        self.prewarm_residuals: dict[str, bool] = {}
        self.fault_boot = "fault-before"
        self.terminal = False
        self.isolated = True
        # What the product's reboot does to the sibling host: the reboot lands
        # (new boot id) and, optionally, the node-installer reconciler
        # re-enables the Agent before the runner comes back to it.
        self.sibling_reboots = True
        self.installer_restores = False
        self.pending_reads = 0
        self.restored_incidents: set[str] = set()
        self.live_release = "release-test"
        self.node_uids = {
            node: f"uid-{node}" for node in (verdicts.FAULT, verdicts.SIBLING)
        }
        self.workflow = verdicts.happy_workflow()
        self.incident = verdicts.happy_incident()
        self.recovery_transport = ProbeTransport(self.host)
        transport_execute = self.recovery_transport.execute

        def execute(command: str, flag: str, raw: str, **kwargs: Any) -> dict[str, Any]:
            self.event("recovery." + command)
            self.host.scope = json.loads(raw)
            self.host.recovery = host_probe.Recovery(self.host.scope)
            if command in {"prepare", "disable"}:
                saved = json.loads(self.journal_path.read_text())
                assert saved["host_binding"] == self.host.scope
                if command == "disable":
                    assert saved["run"]["agent_disabled"] is True
            return transport_execute(command, flag, raw, **kwargs)

        self.recovery_transport.execute = execute
        self.preflight = {
            "errors": [],
            "release_id": self.live_release,
            "fault_node": {
                "uid": self.node_uids[verdicts.FAULT],
                "boot_id": self.fault_boot,
            },
            "sibling_node": {
                "uid": self.node_uids[verdicts.SIBLING],
                "boot_id": self.host.boot,
            },
            "recovery_agent": {
                "node_instance_id": self.node_uids[verdicts.SIBLING],
                "artifact_sha256": self.host.scope["artifact_sha256"],
                "installer_bundle_sha256": self.host.scope["bundle_sha256"],
                "runtime_profile_version": self.host.scope["profile_version"],
            },
            "store": {
                "profile": {"profile_version": self.host.scope["profile_version"]}
            },
            "provider_inventory": {"count": 2, "sha256": "inventory"},
            "cpu_blast": {"unchanged": True},
            "control_env": {"poll_interval_seconds": 5},
        }
        kube = tmp_path / "kube"
        kube.write_text("hermetic reference, not a kubeconfig")
        site = tmp_path / "site"
        site.write_text("hermetic site reference")
        self.settings = case.Settings(
            regional=RegionalLiveSettings(
                kube,
                kube,
                "context-test",
                "namespace-test",
                "cluster-test",
                "region-test",
            ),
            site_file=site,
            manifest=case.DEFAULT_MANIFEST,
            hyperpod_cluster="provider-test",
            host_probe_image="image-test",
            fault_node=verdicts.FAULT,
            fault_pci_bdf="0000:01:00",
            fault_device="/dev/nvidia0",
            sibling_node=verdicts.SIBLING,
            sibling_pci_bdf="0000:02:00",
            job_id="job-test",
            attempt_id="attempt-test",
            verify_max_attempts=6,
            managed_recovery_timeout_seconds=600,
            variant="sibling-exhausted",
            predecessor_path=tmp_path / "predecessor",
        )
        self.regional = self.make_regional()
        self.warm = self.make_warm()
        self.workload = SimpleNamespace(
            submit=lambda: self.event("workload.submit"),
            wait_running=lambda **kwargs: self.source(),
            pods=lambda: [],
            delete=lambda: self.event("workload.delete"),
        )
        self.prewarm = SimpleNamespace(
            create=lambda *args: self.event("prewarm.create"),
            cleanup=lambda: self.cleanup_result(
                "prewarm.cleanup", self.prewarm_residuals
            ),
        )

        class Clock(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> datetime:
                return datetime.fromtimestamp(self.host.now, tz=tz)

        monkeypatch.setattr(case, "datetime", Clock)
        monkeypatch.setattr(case, "RegionalLiveFixture", lambda *_args: self.regional)
        monkeypatch.setattr(case, "WarmSpareLiveFixture", lambda *_args: self.warm)
        monkeypatch.setattr(
            case, "ManagedWorkloadFixture", lambda *_args: self.workload
        )
        monkeypatch.setattr(
            case, "ImagePrewarmFixture", lambda *args, **kwargs: self.prewarm
        )
        monkeypatch.setattr(case, "HostProbeFixture", self.make_probe)
        monkeypatch.setattr(case, "read_only_preflight", self.read_preflight)
        monkeypatch.setattr(case.control_window, "survey", lambda *_args: {})
        monkeypatch.setattr(case.env_window, "survey", lambda *_args: {})
        monkeypatch.setattr(case.control_window, "without_survey", lambda value: value)
        for module, prefix in (
            (case.control_window, "control"),
            (case.env_window, "executor"),
        ):
            monkeypatch.setattr(
                module,
                "open_window",
                lambda *args, _prefix=prefix: self.environment_window(_prefix, "open"),
            )
            monkeypatch.setattr(
                module,
                "close_window",
                lambda *args, _prefix=prefix: self.environment_window(_prefix, "close"),
            )
        self.write_plan()

    @property
    def journal_path(self) -> Path:
        return self.case_dir / "destr014-recovery-journal.json"

    def event(self, name: str) -> None:
        self.calls.append(name)
        self.host.now += self.advance.get(name, 0)
        if name in self.failures:
            raise self.failures[name]

    def cleanup_result(self, name: str, value: dict[str, bool]) -> dict[str, bool]:
        self.event(name)
        return dict(value)

    def environment_window(self, name: str, action: str) -> dict[str, str]:
        self.event(f"{name}.{action}")
        if action == "open":
            saved = json.loads(self.journal_path.read_text())
            key = "control_env_opened" if name == "control" else "env_opened"
            assert saved["run"][key] is True
        return {"status": action}

    def source(self) -> dict[str, Any]:
        self.event("workload.running")
        return {"pods": [{"uid": "pod-one"}, {"uid": "pod-two"}]}

    def read_preflight(self, *_args: Any) -> dict[str, Any]:
        self.event("preflight.read")
        return copy.deepcopy(self.preflight)

    def write_plan(self) -> None:
        (self.case_dir / "plan.json").write_text(
            json.dumps({"details": case.plan_details(self.settings, self.preflight)})
        )

    def snapshot(self) -> dict[str, Any]:
        self.event("workflow.snapshot")
        workflow = copy.deepcopy(self.workflow)
        if self.pending_reads:
            self.pending_reads -= 1
            workflow["status"] = "RUNNING"
        return {
            "workflow": workflow,
            "incident": copy.deepcopy(self.incident),
            "commands": [],
            "workflow_failure_reason": f"node branch escalation exhausted: branch:{verdicts.SIBLING}",
            "restart_budget": {"restart_count": 0},
        }

    def make_regional(self) -> Any:
        def wait(**kwargs: Any) -> dict[str, Any]:
            self.event("workflow.wait")
            return self.snapshot()

        def node(node: str) -> dict[str, Any]:
            self.event("node.snapshot")
            isolated = node == verdicts.SIBLING and self.terminal and self.isolated
            return {
                "uid": self.node_uids[node],
                "unschedulable": isolated,
                "taints": [
                    {
                        "key": case.QUARANTINE_TAINT,
                        "value": case.quarantine_taint_value("follow-up"),
                    }
                ]
                if isolated
                else [],
                "ownership_annotations": {"gpu-fault.io/incident-id": "follow-up"}
                if isolated
                else {},
            }

        def provider(*args: Any) -> list[dict[str, str]]:
            self.event("provider.events")
            return [{"event_name": "BatchRebootClusterNodes"}] * 2

        return SimpleNamespace(
            settings=self.settings.regional,
            wait_for_workflow=wait,
            store_snapshot=lambda **kwargs: self.snapshot(),
            cpu_python=lambda *_args: {
                "incident": {
                    "incident_id": "follow-up",
                    "node_ids": [verdicts.SIBLING],
                },
                "workflow": {"official_steps": [{"operation": "ESCALATE_SUPPORT"}]},
            },
            wait_node_ready=lambda *args, **kwargs: self.event("node.ready"),
            node_snapshot=node,
            provider_events=provider,
            cpu_blast_snapshot=lambda: {"unchanged": True},
            release_id=lambda: self.live_release,
        )

    def make_warm(self) -> Any:
        def create(**kwargs: Any) -> dict[str, str]:
            self.event("isolation.restore")
            self.restored_incidents.add(kwargs["incident_id"])
            self.isolated = False
            return {"workflow_request_id": "restore-test"}

        return SimpleNamespace(
            provider_inventory=lambda: {"count": 2},
            wait_incident_idle=lambda *_args: self.event("quiescence"),
            reactivate_agent=lambda *_args: self.event("agent.reactivate"),
            create_restore_workflow=create,
            wait_workflow_id=lambda *_args: {"status": "SUCCEEDED"},
            incident_by_id=lambda incident: {
                "state": "RECOVERED"
                if incident in self.restored_incidents
                else "QUARANTINED"
            },
        )

    def make_probe(self, settings: Any) -> Any:
        if settings.probe_script == case.recovery.PROBE:
            return self.recovery_transport
        node = settings.node

        def execute(command: str, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            self.event("probe." + command)
            if command.startswith("write-xid"):
                saved = json.loads(self.journal_path.read_text())
                assert saved["run"]["injection_started"] is True
                assert saved["host_ack"]["phase"] == "DISABLED"
                if command == "write-xid46":
                    if self.sibling_reboots:
                        self.host.reboot(renumber_devices=True)
                        if self.installer_restores:
                            self.host.installer_reinstall()
                    self.fault_boot = "fault-after"
                    self.terminal = True
            if command == "snapshot":
                fault = node == verdicts.FAULT
                return {
                    "boot_id": self.fault_boot if fault else self.host.boot,
                    "agent_unit": {
                        "UnitFileState": "enabled"
                        if self.host.enable.exists() or fault
                        else "disabled",
                        "ActiveState": "active"
                        if self.host.agent_active or fault
                        else "inactive",
                    },
                    "ledger": [
                        {
                            "command_id": "owned-reset",
                            "operation": "RESET_GPU",
                            "state": "FAILED",
                        }
                    ]
                    if self.terminal and fault
                    else [],
                }
            if command == "holder-status":
                return {
                    "matched_row": {"command_id": "owned-verify"},
                    "hold_started_at": "started",
                }
            assert command not in {"disable-agent-restart", "restore-agent"}, (
                "legacy unbound Agent path"
            )
            return {"ok": True}

        return SimpleNamespace(
            host_script="/run/owned-test-probe.py",
            create=lambda: self.event("probe.create"),
            execute=execute,
            cleanup=lambda: self.cleanup_result("probe.cleanup", self.probe_residuals),
        )

    def execute(self, *, seconds: int = 7000) -> tuple[int, dict[str, Any]]:
        code = case.execute_case(
            self.settings,
            self.directory,
            1,
            datetime.fromtimestamp(1000 + seconds, timezone.utc),
        )
        return code, json.loads((self.case_dir / f"{case.CASE_ID}.json").read_text())


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunnerHarness:
    return RunnerHarness(tmp_path, monkeypatch)


def test_live_runner_preserves_unknown_reboot_hold_after_agent_recovery(
    runner: RunnerHarness,
) -> None:
    from tests.regional.test_acceptance_reboot_hold_alignment import unknown_reboot

    runner.workflow = unknown_reboot()
    runner.pending_reads = 1
    code, report = runner.execute()
    assert code == 1 and report["verdict"] == "BLOCKED", report
    assert report["scenario_verdict"] == "PASS" and report["errors"] == []
    assert report["cleanup"]["operator_hold_preserved"] is True
    names = runner.calls
    assert (
        names.index("recovery.prepare")
        < names.index("recovery.status")
        < names.index("recovery.disable")
    )
    assert (
        names.index("recovery.disable")
        < names.index("probe.arm-holder")
        < names.index("probe.write-xid79")
    )
    assert names.index("recovery.restore") < names.index("recovery.cleanup")
    assert "workload.delete" not in names and "isolation.restore" not in names
    assert names.count("probe.write-xid79") == names.count("probe.write-xid46") == 1
    saved = json.loads(runner.journal_path.read_text())
    assert saved["phase"] == "RECOVERY_REQUIRED"
    assert saved["host_cleanup"]["phase"] == "CLOSED"
    assert saved["run"]["physical_outcome_unknown"] is True
    assert not (host_probe.SYSTEMD / runner.host.recovery.timer).exists(), (
        "the independent Agent recovery can close without clearing the workflow hold"
    )
    assert runner.host.enable.exists() and runner.host.agent_active
    # The hold is the product's: the sibling's reboot itself is proven.
    assert report["sibling_reboot_proof"]["proven"] is True, report
    assert report["sibling_reboot_proof"]["restored_by"] == "probe"
    assert report["hold_reasons"] and all(
        reason.startswith("product:") for reason in report["hold_reasons"]
    ), report["hold_reasons"]
    assert report["cleanup"]["hold_reasons"] == report["hold_reasons"]


def test_installer_restored_sibling_is_accepted_and_the_hold_mirrors_the_product(
    runner: RunnerHarness,
) -> None:
    """The live attempt: the sibling rebooted, the node-installer reconciler
    re-enabled its Agent before the runner returned, the probe refused the new
    boot and the case recorded FAIL without a scenario verdict. The restore
    must be accepted, the probe residue retired, the scenario judged, and the
    remaining hold must be the product's (BLOCKED/NEEDS_OPERATOR), not the
    runner's uncertainty about the sibling."""

    from tests.regional.test_acceptance_reboot_hold_alignment import unknown_reboot

    runner.workflow = unknown_reboot()
    runner.installer_restores = True
    code, report = runner.execute()
    assert code == 1 and report["verdict"] == "BLOCKED", report
    assert report["scenario_verdict"] == "PASS" and report["errors"] == []
    assert report["operator_review_required"] is True
    cleanup = report["cleanup"]
    assert cleanup["operator_hold_preserved"] is True
    assert cleanup["agent_recovery"]["phase"] == "CLOSED"
    assert cleanup["agent_recovery"]["restored_by"] == "reboot-installer"
    assert cleanup["agent_recovery"]["retired"] is True
    proof = report["sibling_reboot_proof"]
    assert proof["proven"] is True and proof["boot_id_observed"] == "boot-after"
    assert proof["restored_by"] == "reboot-installer" and proof["gaps"] == []
    assert report["hold_reasons"] == cleanup["hold_reasons"]
    assert any("NEEDS_OPERATOR" in reason for reason in report["hold_reasons"]), report[
        "hold_reasons"
    ]
    assert all(reason.startswith("product:") for reason in report["hold_reasons"]), (
        "with the sibling proven, only the product's hold may remain"
    )
    assert "workload.delete" not in runner.calls
    assert "isolation.restore" not in runner.calls
    saved = json.loads(runner.journal_path.read_text())
    assert saved["host_recovery_phase"] == "RESTORED_BY_REBOOT"
    assert saved["run"]["sibling_reboot_proof"]["proven"] is True
    assert saved["run"]["hold_reasons"] == report["hold_reasons"]
    assert not runner.host.recovery.dropin.exists(), (
        "the retired probe must not leave the 45 s start bound on the Agent"
    )
    assert not (host_probe.SYSTEMD / runner.host.recovery.timer).exists(), (
        "the retired probe must leave no persistent timer"
    )
    assert runner.host.enable.exists() and runner.host.agent_active
    assert not [
        call
        for call in runner.host.calls
        if call[0] == "start" and call[-1] == host_probe.AGENT
    ], "an installer-restored Agent is never restarted by the probe"


def test_settled_product_rows_with_a_proven_reboot_release_the_hold(
    runner: RunnerHarness,
) -> None:
    """Once the product no longer owns recovery (terminal workflow, settled
    commands) and the sibling is back on a new boot with its Agent restored,
    nothing is left for an operator to confirm: cleanup restores along the
    product path. The scenario still fails on its own terms."""

    code, report = runner.execute()
    assert code == 1 and report["verdict"] == "FAIL", report
    assert report["operator_review_required"] is False
    assert report["hold_reasons"] == [] and report["sibling_reboot_proof"]["proven"]
    assert "operator_hold_preserved" not in report["cleanup"]
    assert report["cleanup"]["errors"] == []
    for name in (
        "quiescence",
        "workload.delete",
        "agent.reactivate",
        "isolation.restore",
        "control.close",
        "executor.close",
    ):
        assert name in runner.calls, name
    saved = json.loads(runner.journal_path.read_text())
    assert saved["phase"] == "CLOSED"
    assert saved["run"]["physical_outcome_unknown"] is False
    assert saved["run"]["sibling_reboot_proof"]["boot_id_observed"] == "boot-after"


def test_a_cleanup_only_rerun_releases_the_hold_once_the_product_settles(
    runner: RunnerHarness,
) -> None:
    from tests.regional.test_acceptance_reboot_hold_alignment import unknown_reboot

    runner.workflow = unknown_reboot()
    runner.installer_restores = True
    code, first = runner.execute()
    assert first["verdict"] == "BLOCKED" and first["cleanup"]["operator_hold_preserved"]
    assert "workload.delete" not in runner.calls
    # The operator reconciled the product side; the host record still holds
    # the sibling's proof. The rerun is cleanup-only and never a second PASS.
    runner.workflow = verdicts.happy_workflow()
    before = len(runner.calls)
    code, second = runner.execute()
    assert code == 1 and "cleanup-only" in second["error"], second
    assert second["cleanup"]["errors"] == [], second["cleanup"]
    assert second["cleanup"]["operator_hold_released_by"]["proven"] is True
    assert second["cleanup"]["operator_hold_released_by"]["restored_by"] == (
        "reboot-installer"
    )
    calls = runner.calls[before:]
    assert "recovery.cleanup" in calls and "recovery.prepare" not in calls
    assert "workload.delete" in calls and "isolation.restore" in calls
    assert "control.close" in calls and "executor.close" in calls
    saved = json.loads(runner.journal_path.read_text())
    assert saved["phase"] == "CLOSED"


def test_a_cleanup_only_rerun_keeps_the_hold_while_the_product_still_owns_recovery(
    runner: RunnerHarness,
) -> None:
    from tests.regional.test_acceptance_reboot_hold_alignment import unknown_reboot

    runner.workflow = unknown_reboot()
    runner.execute()
    before = len(runner.calls)
    code, second = runner.execute()
    assert code == 1 and second["cleanup"]["operator_hold_preserved"] is True
    assert "operator_hold_released_by" not in second["cleanup"]
    assert "workload.delete" not in runner.calls[before:]
    assert json.loads(runner.journal_path.read_text())["phase"] == "RECOVERY_REQUIRED"


@pytest.mark.parametrize("command", ["prepare", "disable"])
def test_unacknowledged_recovery_never_allows_holder_or_injection(
    runner: RunnerHarness, command: str
) -> None:
    runner.recovery_transport.lost_ack = command
    code, report = runner.execute()
    assert code == 1 and report["verdict"] == "FAIL"
    assert "probe.arm-holder" not in runner.calls
    assert "probe.write-xid79" not in runner.calls
    assert report["cleanup"]["agent_recovery"]["phase"] == "CLOSED"
    assert runner.host.enable.exists(), (
        "missing independent ACK must preserve Agent boot activation"
    )


@pytest.mark.parametrize(
    "phase",
    [
        "control.open",
        "executor.open",
        "workload.submit",
        "workload.running",
        "recovery.prepare",
        "recovery.disable",
        "probe.arm-holder",
        "probe.write-xid79",
        "workflow.wait",
        "recovery.restore",
        "provider.events",
        "recovery.cleanup",
        "workload.delete",
        "control.close",
        "executor.close",
        "prewarm.cleanup",
    ],
)
def test_phase_failure_never_leaves_a_passing_case(
    runner: RunnerHarness, phase: str
) -> None:
    if phase in {"workload.delete", "control.close", "executor.close"}:
        runner.failures["workload.submit"] = RuntimeError("pre-injection refusal")
    runner.failures[phase] = RuntimeError(f"fake failure: {phase}")
    code, report = runner.execute()
    assert code == 1 and report["verdict"] == "FAIL", report
    assert phase in str(report)
    assert "prewarm.cleanup" in runner.calls
    assert runner.calls.count("recovery.disable") <= 1
    saved = json.loads(runner.journal_path.read_text())
    assert saved["phase"] == (
        "RECOVERY_REQUIRED" if report["cleanup"]["errors"] else "CLOSED"
    )


def test_interrupted_attempt_restarts_as_cleanup_only_with_the_same_binding(
    runner: RunnerHarness,
) -> None:
    runner.recovery_transport.lost_ack = "disable"
    runner.failures["recovery.cleanup"] = RuntimeError("temporary cleanup outage")
    code, report = runner.execute()
    assert code == 1 and report["cleanup"]["errors"]
    binding_before = json.loads(runner.journal_path.read_text())["host_binding"]
    runner.failures.clear()
    before = len(runner.calls)
    code, report = runner.execute()
    assert code == 1 and report["verdict"] == "FAIL"
    assert "cleanup-only" in report["error"]
    assert "preflight.read" not in runner.calls[before:], (
        "busy-node preflight is not a cleanup gate"
    )
    assert "recovery.prepare" not in runner.calls[before:]
    assert "recovery.disable" not in runner.calls[before:]
    assert "workload.submit" not in runner.calls[before:]
    saved = json.loads(runner.journal_path.read_text())
    assert saved["host_binding"] == binding_before and saved["phase"] == "CLOSED"


@pytest.mark.parametrize("drift", ["release", "node"])
def test_resume_refuses_changed_target_before_any_cleanup_command(
    runner: RunnerHarness, drift: str
) -> None:
    runner.recovery_transport.lost_ack = "disable"
    runner.failures["recovery.cleanup"] = RuntimeError("outage")
    runner.execute()
    runner.failures.clear()
    if drift == "release":
        runner.live_release = "different"
    else:
        runner.node_uids[verdicts.SIBLING] = "recreated"
    before = len(runner.calls)
    code, report = runner.execute()
    assert code == 1 and report["cleanup"]["errors"]
    assert "changed" in report["error"]
    assert "recovery.cleanup" not in runner.calls[before:]
    assert "workload.delete" not in runner.calls[before:]


def test_completed_attempt_is_not_reexecuted_or_relabelled_pass(
    runner: RunnerHarness,
) -> None:
    runner.failures["workload.submit"] = RuntimeError("pre-injection refusal")
    assert runner.execute()[0] == 1, (
        "a confirmed pre-injection failure still fails the case"
    )
    before = len(runner.calls)
    code, report = runner.execute()
    assert code == 1 and report["verdict"] == "FAIL"
    assert report["cleanup"]["already_closed"] is True
    assert "recovery.cleanup" not in runner.calls[before:]
    assert "probe.write-xid79" not in runner.calls[before:]


def test_lost_supervision_never_sends_cleanup_and_persists_refusal(
    runner: RunnerHarness,
) -> None:
    runner.failures["recovery.disable"] = ProcessSupervisionLost(
        "test supervision loss"
    )
    code, report = runner.execute()
    assert code == 1 and report["cleanup"]["errors"]
    assert "recovery.cleanup" not in runner.calls
    assert "workload.delete" not in runner.calls and "control.close" not in runner.calls
    assert json.loads(runner.journal_path.read_text())["supervision_lost"] is True
    runner.failures.clear()
    before = len(runner.calls)
    with pytest.raises(case.RegionalFixtureError, match="supervision"):
        runner.execute()
    assert runner.calls == runner.calls[:before]


@pytest.mark.parametrize(
    "point", ["recovery.cleanup", "executor.close", "control.close"]
)
def test_supervision_loss_during_cleanup_stops_all_remaining_commands(
    runner: RunnerHarness, point: str
) -> None:
    if point in {"executor.close", "control.close"}:
        runner.failures["workload.submit"] = RuntimeError("pre-injection refusal")
    runner.failures[point] = ProcessSupervisionLost("owned test supervision loss")
    code, report = runner.execute()
    assert code == 1 and report["verdict"] == "FAIL"
    assert report["cleanup"]["errors"] == ["cleanup stopped: command supervision lost"]
    assert runner.calls[-1] == point
    saved = json.loads(runner.journal_path.read_text())
    assert saved["phase"] == "RECOVERY_REQUIRED" and saved["supervision_lost"] is True


def test_automatic_restore_before_observation_cannot_be_counted_as_pass(
    runner: RunnerHarness,
) -> None:
    original = runner.snapshot

    def snapshot() -> dict[str, Any]:
        result = original()
        runner.host.now = runner.host.scope["restore_at"]
        runner.host.tick()
        return result

    runner.regional.store_snapshot = lambda **kwargs: snapshot()
    code, report = runner.execute()
    assert code == 1 and "automatic safeguard" in report["error"]
    assert report["cleanup"]["agent_recovery"]["phase"] == "CLOSED"


@pytest.mark.parametrize("residual", ["probe", "prewarm", "recovery", "quiescence"])
def test_unknown_or_residual_cleanup_prevents_pass_and_keeps_journal(
    runner: RunnerHarness, residual: str
) -> None:
    if residual == "probe":
        runner.probe_residuals["pod"] = True
    elif residual == "prewarm":
        runner.prewarm_residuals["pod"] = True
    elif residual == "recovery":
        runner.recovery_transport.residuals = {"pod": True}
    else:
        runner.failures["quiescence"] = RuntimeError("command still pending")
    code, report = runner.execute()
    assert code == 1 and report["verdict"] == "FAIL"
    assert report["cleanup"]["errors"]
    assert json.loads(runner.journal_path.read_text())["phase"] == "RECOVERY_REQUIRED"
    if residual == "quiescence":
        assert "workload.delete" not in runner.calls
        assert (
            "executor.close" not in runner.calls and "control.close" not in runner.calls
        )


def test_window_must_cover_independent_restore_before_disabling(
    runner: RunnerHarness,
) -> None:
    code, report = runner.execute(seconds=1200)
    assert code == 1 and "bounded Agent recovery" in report["error"]
    assert (
        "recovery.prepare" not in runner.calls
        and "recovery.disable" not in runner.calls
    )
    assert "probe.arm-holder" not in runner.calls


@pytest.mark.parametrize("phase", ["control.open", "executor.open", "workload.running"])
def test_window_expiry_before_arm_cannot_disable_agent(
    runner: RunnerHarness, phase: str
) -> None:
    runner.advance[phase] = 7001
    code, report = runner.execute()
    assert code == 1 and "window ended" in report["error"]
    assert "recovery.disable" not in runner.calls
    assert runner.host.enable.exists(), (
        "expired maintenance window must not disable the Agent"
    )


def test_plan_drift_is_refused_without_resource_mutation(runner: RunnerHarness) -> None:
    runner.preflight["sibling_node"]["uid"] = "recreated"
    with pytest.raises(case.RegionalFixtureError, match="identity drifted"):
        runner.execute()
    assert runner.calls == ["preflight.read"]


def test_recovery_plan_binds_helper_and_nonrenewable_hold(
    runner: RunnerHarness,
) -> None:
    details = case.plan_details(runner.settings, runner.preflight)
    window = details["recovery_safeguard"]
    assert window["kind"] == "persistent-host-systemd"
    assert window["requires_independent_arm_ack_before_disable"] is True
    assert window["resume"] == "cleanup-only"
    assert window["hold_seconds"] > runner.settings.managed_recovery_timeout_seconds
    assert window["recovery_seconds"] == host_probe.RECOVERY_SECONDS
    assert len(window["helper_sha256"]) == 64
