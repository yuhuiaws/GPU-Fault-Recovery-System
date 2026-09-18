from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import regional_commands
from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004
from scripts.e2e.regional import run_ha005_rollout_continuity as ha005
from scripts.e2e.regional import run_ha006_executor_takeover as ha006
from scripts.e2e.regional import run_ha007_control_worker_shutdown as ha007
from scripts.e2e.regional import run_ha008_processor_exit_acceptance as ha008
from scripts.e2e.regional import run_ha009_aurora_credential_rotation as ha009
from scripts.e2e.regional.regional_case_contract import (
    case_evidence_path,
    formal_predecessor,
)
from tests.regional._cov95_focused_mock_receipts import write_focused_receipt


@pytest.mark.parametrize("module", [ha003, ha004])
@pytest.mark.parametrize("lost", [False, True])
def test_host_state_directory_and_early_failure_cleanup_are_bound(
    module, lost: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []
    regional_settings = SimpleNamespace(
        cpu_kubeconfig=tmp_path / "cpu",
        gpu_kubeconfig=tmp_path / "gpu",
        gpu_context="unit-context",
        namespace="unit-ns",
        cluster_id="unit-cluster",
    )
    settings = SimpleNamespace(
        regional=regional_settings,
        node="node-a",
        host_probe_image="unit-image",
        rds_cluster_id="unit-aurora",
    )
    monkeypatch.setattr(
        module,
        "read_only_preflight",
        lambda *a: {
            "errors": [],
            "release_id": "unit-release",
            "store": {},
            "aurora_binding": {},
        },
    )
    if module is ha003:
        monkeypatch.setattr(
            module, "regional_binding", lambda *a: SimpleNamespace(read=lambda *a: {})
        )
    monkeypatch.setattr(module, "verify_plan_identity", lambda *a: None)
    monkeypatch.setattr(
        module,
        "RegionalLiveFixture",
        lambda _: SimpleNamespace(
            node_snapshot=lambda _: {"ownership_annotations": {}}
        ),
    )

    def host_settings(*, state_directory: Path, **values):
        assert state_directory == tmp_path / "cases" / module.CASE_ID / "host-probes"
        return SimpleNamespace(state_directory=state_directory, **values)

    class Host:
        host_script = "/unit/probe.py"

        def __init__(self, settings):
            self.settings = settings

        def create(self):
            calls.append("create")
            if lost:
                raise ProcessSupervisionLost("unit supervision loss")

        def execute(self, operation, *args, **kwargs):
            calls.append((operation, args))
            if operation == "snapshot":
                return {
                    "compute_clients": [],
                    "gpu_inventory": [{"pci_bdf": "unit-bdf"}],
                }
            return {}

        def cleanup(self):
            calls.append("cleanup")
            return {}

    monkeypatch.setattr(module, "HostProbeSettings", host_settings)
    monkeypatch.setattr(module, "HostProbeFixture", Host)
    if module is ha004:
        monkeypatch.setattr(
            ha004,
            "ExecutorTimingFixture",
            lambda *a: SimpleNamespace(
                baseline={"replicas": 1, "env": {}},
                apply=lambda: {},
                restore=lambda: calls.append("timing-restore")
                or {"replicas": 1, "env": {}},
            ),
        )

    def interrupted(*args, observe_state, **kwargs):
        observe_state({"incident": {"incident_id": "owned-incident"}})
        raise RuntimeError("unit observation failed")

    monkeypatch.setattr(
        module,
        "wait_reset_claim" if module is ha003 else "command_timeline",
        interrupted,
    )
    assert (
        module.execute_case(
            settings, tmp_path, 1, datetime(2099, 1, 1, tzinfo=timezone.utc)
        )
        == 1
    )
    report = json.loads(case_evidence_path(tmp_path, module.CASE_ID).read_text())
    assert report["verdict"] == "FAIL"
    assert report["release_id"] == "unit-release"
    assert report["cluster_id"] == "unit-cluster"
    if lost:
        assert calls == ["create"]
        assert report["supervision_lost"] is True
        assert report["cleanup_preserved"]
    else:
        assert ("restore-quiesce", ("--incident-id", "owned-incident")) in calls
        assert "cleanup" in calls


@pytest.mark.parametrize("module", [ha005, ha006, ha009])
def test_supervision_loss_stops_cleanup_and_retains_canonical_evidence(
    module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = module.BASE if module is ha009 else module
    monkeypatch.setattr(base, "database_residuals", lambda: {"total": 0})
    monkeypatch.setattr(base, "registry_residuals", lambda: {"count": 0})
    monkeypatch.setattr(base, "kubernetes_residuals", lambda: {"count": 0})
    calls = []

    def register(*a, **kw):
        calls.append("register")
        raise ProcessSupervisionLost("unit supervision loss")

    monkeypatch.setattr(base, "register", register)
    monkeypatch.setattr(
        base, "dataplane", lambda *a, **kw: calls.append("unexpected-kube")
    )
    monkeypatch.setattr(
        base, "teardown", lambda **kw: calls.append("unexpected-teardown")
    )
    if module is ha009:
        monkeypatch.setattr(
            ha009,
            "aurora_guard",
            lambda: SimpleNamespace(
                read=lambda *a: {
                    "identity": {"database": {"master_secret_arn": "unit-resource"}}
                }
            ),
        )
        monkeypatch.setattr(ha009, "pool_max_idle_seconds", lambda: 1)
        monkeypatch.setattr(ha009, "master_secret_arn", lambda: "unit-resource")
        monkeypatch.setattr(
            ha009, "secret_versions", lambda _: {"stages": {"AWSCURRENT": "v1"}}
        )
        monkeypatch.setattr(ha009, "kubernetes_secret_digest", lambda: "unit-digest")
    identity = {"release_id": "unit-release", "cluster_id": "unit-cluster"}
    assert (
        module.run_case(
            tmp_path,
            1,
            datetime(2099, 1, 1, tzinfo=timezone.utc),
            chain={"identity": identity, "predecessor": {"valid": True}},
            **({"all_deployments": True} if module is ha005 else {}),
        )
        == 1
    )
    assert calls == ["register"]
    report = json.loads(case_evidence_path(tmp_path, module.CASE_ID).read_text())
    assert report["verdict"] == "FAIL" and report["supervision_lost"] is True
    assert all(report[key] == value for key, value in identity.items()), {
        "expected_identity": identity,
        "report": report,
    }


@pytest.mark.parametrize(
    "module,probe", [(ha007, "run_probe"), (ha008, "run_acceptance")]
)
@pytest.mark.parametrize("bound", [False, True])
def test_isolated_main_writes_canonical_evidence_without_inventing_live_identity(
    module, probe, bound, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = {"release_id": "unit-release", "cluster_id": "unit-cluster"}
    previous = formal_predecessor(module.CASE_ID)
    path = case_evidence_path(tmp_path, previous)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"case_id": previous, "verdict": "PASS", **identity}))
    seen = []
    monkeypatch.setattr(
        module,
        probe,
        lambda directory, *a: (
            seen.append(directory) or {"case_id": module.CASE_ID, "verdict": "PASS"}
        ),
    )
    argv = [module.__file__, "--run-dir", str(tmp_path)]
    if bound:
        argv += [
            "--release-id",
            identity["release_id"],
            "--cluster-id",
            identity["cluster_id"],
        ]
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    result_path = case_evidence_path(tmp_path, module.CASE_ID)
    assert seen == [result_path.parent]
    result = json.loads(result_path.read_text())
    assert result["validation_scope"] == "isolated-source"
    if bound:
        assert result["predecessor"]["valid"] is True
        assert result["release_id"] == identity["release_id"]
    else:
        assert result["formal_sequence_satisfied"] is False
        assert "release_id" not in result


def test_failover_keeps_unavailable_store_samples_unknown_until_recovery(
    monkeypatch,
) -> None:
    rds = iter(
        [
            {"status": "failing-over", "writer": "old"},
            {"status": "available", "writer": "new"},
        ]
    )
    reads = iter(
        [
            regional_commands.RegionalCommandFailed(1, "unit Store unavailable"),
            {"commands": [{"step": {"operation": "RESET_GPU"}, "status": "WAITING"}]},
        ]
    )
    monkeypatch.setattr(ha003, "rds_snapshot", lambda _: next(rds))

    def observe():
        item = next(reads)
        if isinstance(item, BaseException):
            raise item
        return item

    result, samples = ha003.wait_rds_failover(
        SimpleNamespace(), previous_writer="old", observe=observe, sleep=lambda _: None
    )
    assert result["writer"] == "new"
    assert samples[0]["reset_command_status"] is None
    assert samples[0]["store_observation_error"] == "RegionalCommandFailed"
    assert samples[1]["reset_command_status"] == "WAITING"


def run_focused_boundary(
    module, receipt: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        write_focused_receipt(
            command,
            environment=kwargs["environment"],
            cwd=kwargs["cwd"],
            defect=receipt,
        )
        return subprocess.CompletedProcess(command, 0, "unit focused checks passed", "")

    monkeypatch.setattr(regional_commands, "run_command", run)
    result = module.focused_tests(tmp_path)
    return result, calls


@pytest.mark.parametrize("module", [ha003, ha004])
def test_focused_checks_use_the_public_supervised_command_boundary(
    module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, calls = run_focused_boundary(module, "complete", monkeypatch, tmp_path)
    assert result["passed"] is True, (
        "a complete passing receipt must satisfy the mocked HA prerequisite"
    )
    assert len(calls) == 1, "receipt rejection must not retry the supervised child"
    assert calls[0][0][1:3] == ["-m", "pytest"], (
        "focused checks must remain local pytest invocations"
    )
    assert calls[0][1]["timeout_seconds"] > 0, (
        "receipt verification must preserve the caller's supervised deadline"
    )


@pytest.mark.parametrize("module", [ha003, ha004])
@pytest.mark.parametrize(
    "receipt", ["missing", "failed", "missing-phase", "unexecuted-discovery"]
)
def test_focused_checks_refuse_missing_or_incomplete_child_receipts(
    module, receipt: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, calls = run_focused_boundary(module, receipt, monkeypatch, tmp_path)
    assert result["passed"] is False and result["returncode"] == 1, (
        "missing or failed evidence must override the fake child's zero exit"
    )
    assert len(calls) == 1, "receipt rejection must not retry the supervised child"
    assert (
        "focused pytest evidence rejected"
        in (tmp_path / "focused-tests.log").read_text()
    ), "the private log must retain the verifier's refusal"
