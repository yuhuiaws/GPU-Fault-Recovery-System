from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import live_driver_guard as guard
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr_warm import NOW, regional_settings

MODULES = (
    "run_destr001_gpu_reset",
    "run_destr002_hyperpod_reboot",
    "run_destr003_warm_spare_failover",
    "run_destr008_warm_spare_shortage",
    "run_destr009_workload_restart",
    "run_destr010_fabric_manager_restart",
    "run_destr012_managed_recovery_guard",
    "run_destr014_branch_exhaustion",
    "run_destr015_parallel_branch_join",
    "run_destr016_preempting_reboot",
    "run_destr017_out_of_band_reboot_fence",
    "run_destr018_lifetime_deadline",
    "run_destr019_agent_restart_ledger",
    "run_destr020_identity_mismatch_isolation",
    "run_destr021_adversarial_node_metadata",
    "run_destr022_spare_reservation_reclaim",
    "run_destr023_idle_cluster_reset",
    "run_destr024_watcher_down_fail_closed",
)
REUSE = {"001", "002", "009", "010", "012", "015", "023", "024"}


def cli_arguments(module: Any, tmp_path: Path) -> list[str]:
    regional = regional_settings(tmp_path)
    site = tmp_path / "site.yaml"
    site.write_text("fake site\n", encoding="utf-8")
    common = [
        "--run-dir",
        str(tmp_path),
        "--cpu-kubeconfig",
        str(regional.cpu_kubeconfig),
        "--gpu-kubeconfig",
        str(regional.gpu_kubeconfig),
        "--gpu-context",
        "fake-gpu",
        "--cluster-id",
        "cluster-a",
        "--region",
        "us-west-2",
    ]
    number = module.CASE_ID[-3:]
    if number in {"003", "008", "009", "012", "014", "015", "021"}:
        common += ["--site-file", str(site)]
    if number not in {"003", "009", "012", "020", "022"}:
        common += ["--host-probe-image", "example.test/probe"]
    if number in {"001", "002", "010", "016", "017", "018", "019", "021", "023", "024"}:
        common += ["--node", "node-a"]
    if number in {"002", "003", "008", "014", "016", "022"}:
        common += ["--hyperpod-cluster", "fake-hyperpod"]
    if number in {"002", "016"}:
        common += [
            "--executor-role-arn",
            "arn:aws:iam::123456789012:role/test-executor",
        ]
    if number in {"003", "008"}:
        common += ["--fault-node", "node-a", "--spare-node", "node-b"]
    if number == "014":
        common += [
            "--fault-node",
            "node-a",
            "--sibling-node",
            "node-b",
            "--fault-pci-bdf",
            "0000:01:00.0",
            "--sibling-pci-bdf",
            "0000:02:00.0",
            "--fault-device",
            "/dev/nvidia0",
        ]
    if number == "015":
        common += ["--node-a", "node-a", "--node-b", "node-b"]
    if number == "020":
        common += ["--reference-node", "node-a"]
    if number == "022":
        common += ["--spare-node", "node-b"]
    return common


@pytest.mark.parametrize("name", MODULES)
@pytest.mark.parametrize("explicit_predecessor", [False, True])
def test_configure_retains_local_target_identity_and_predecessor(
    name: str, explicit_predecessor: bool, tmp_path: Path
) -> None:
    module = importlib.import_module(f"scripts.e2e.regional.{name}")
    argv = cli_arguments(module, tmp_path)
    if explicit_predecessor and module.CASE_ID[-3:] != "010":
        argv += ["--predecessor-evidence", str(tmp_path / "previous.json")]
    args = module.parser().parse_args(argv)
    settings = module.configure(args)
    environment = settings.environment()
    assert settings.regional.cluster_id == "cluster-a", settings
    assert settings.regional.gpu_context == "fake-gpu", settings
    assert environment["GPU_FAULT_CLUSTER_ID"] == "cluster-a", environment
    assert environment["CPU_KUBECONFIG"] != environment["GPU_KUBECONFIG"], environment
    if module.CASE_ID[-3:] != "010":
        expected = (
            tmp_path / "previous.json"
            if explicit_predecessor
            else (
                tmp_path
                / "cases"
                / module.PREDECESSOR_CASE_ID
                / f"{module.PREDECESSOR_CASE_ID}.json"
            )
        )
        assert settings.predecessor_path == expected, settings


@pytest.mark.parametrize("name", MODULES)
@pytest.mark.parametrize(
    "mode", ["plan-pass", "plan-fail", "execute-pass", "execute-fail", "unauthorized"]
)
def test_main_routes_plan_verdict_or_authorized_execute_with_bound_arguments(
    name: str, mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(f"scripts.e2e.regional.{name}")
    args = cli_arguments(module, tmp_path)
    if mode.startswith("execute") or mode == "unauthorized":
        args += ["--execute", "--confirm", module.CONFIRMATION]
    monkeypatch.setattr(sys, "argv", ["unit-fake-runner", *args])
    calls: list[str] = []
    captured: dict[str, Any] = {}

    def preflight(
        settings: Any, path: Path, *extra: Any, **kwargs: Any
    ) -> dict[str, Any]:
        calls.append("preflight")
        captured["settings"] = settings
        captured["path"] = path
        captured["extra"] = (extra, kwargs)
        return {"errors": ["fake refusal"] if mode == "plan-fail" else []}

    def details(settings: Any, report: dict[str, Any]) -> dict[str, Any]:
        calls.append("details")
        return {"target": settings.regional.cluster_id, "errors": report["errors"]}

    def build_plan(**kwargs: Any) -> dict[str, Any]:
        calls.append("build")
        captured.update(kwargs)
        return {
            "case_id": kwargs["case_id"],
            "preflight_passed": kwargs["preflight_passed"],
        }

    def authorize(arguments: Any, **kwargs: Any) -> Any:
        calls.append("authorize")
        captured.update(arguments=arguments, **kwargs)
        if mode == "unauthorized":
            raise RegionalFixtureError("unit authorization denied")
        return NOW

    def execute(settings: Any, run_dir: Path, attempt: int, deadline: Any) -> int:
        calls.append("execute")
        captured.update(
            settings=settings, run_dir=run_dir, attempt=attempt, deadline=deadline
        )
        return 1 if mode == "execute-fail" else 0

    owner = module if module.CASE_ID[-3:] in {"020", "022"} else guard
    monkeypatch.setattr(owner, "install_site_profile", lambda: calls.append("profile"))
    monkeypatch.setattr(owner, "install_abort_signals", lambda: calls.append("signals"))
    monkeypatch.setattr(
        owner,
        "os",
        SimpleNamespace(
            umask=lambda mask: calls.append(f"umask:{mask:o}"), getenv=os.getenv
        ),
    )
    monkeypatch.setattr(owner, "build_plan", build_plan)
    monkeypatch.setattr(owner, "authorize_execution", authorize)
    if owner is module:
        monkeypatch.setattr(module, "read_only_preflight", preflight)
        monkeypatch.setattr(module, "plan_details", details)
        monkeypatch.setattr(module, "execute_case", execute)
    else:
        monkeypatch.setattr(
            module,
            "CASE",
            replace(
                module.CASE,
                read_only_preflight=preflight,
                plan_details=details,
                execute_case=execute,
            ),
        )
    if mode == "unauthorized":
        with pytest.raises(RegionalFixtureError, match="unit authorization denied"):
            module.main()
        assert "execute" not in calls, calls
        return
    assert module.main() == (1 if mode.endswith("fail") else 0), calls
    assert calls[:3] == ["profile", "umask:77", "signals"], calls
    assert captured["case_id"] == module.CASE_ID, captured
    assert captured["environment"]["GPU_FAULT_CLUSTER_ID"] == "cluster-a", captured
    if mode.startswith("plan"):
        assert calls[3:] == ["preflight", "details", "build"], calls
        assert captured["preflight_passed"] is (mode == "plan-pass"), captured
        assert captured["arguments"].run_dir == tmp_path, captured
    else:
        assert calls[3:] == ["authorize", "execute"], calls
        assert captured["deadline"] == NOW and captured["run_dir"] == tmp_path, captured


@pytest.mark.parametrize("name", [name for name in MODULES if "014" not in name])
@pytest.mark.parametrize("returncode", [0, 1])
def test_focused_test_wrapper_records_fake_subprocess_outcome_privately(
    name: str, returncode: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(f"scripts.e2e.regional.{name}")
    commands: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(
            command, returncode, "fake-test-output\n", "fake-test-stderr\n"
        )

    monkeypatch.setattr(module, "RegionalLiveFixture", SimpleNamespace(run=run))
    result = module.focused_tests(tmp_path)
    assert result["passed"] is (returncode == 0), result
    assert result["returncode"] == returncode, result
    assert result["command"][:4] == [sys.executable, "-m", "pytest", "-q"], result
    assert len(commands) == 1 and commands[0][1]["check"] is False, commands
    log = tmp_path / "focused-tests.log"
    assert log.read_text() == "fake-test-output\nfake-test-stderr\n", log
    assert log.stat().st_mode & 0o777 == 0o600, log


@pytest.mark.parametrize("name", [name for name in MODULES if name[9:12] in REUSE])
@pytest.mark.parametrize("cached", [False, True])
def test_focused_cache_requires_a_reusable_result(
    name: str, cached: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(f"scripts.e2e.regional.{name}")
    ran: list[list[str]] = []
    record = {"passed": True, "returncode": 0, "command": ["recorded-pytest"]}
    monkeypatch.setattr(
        module, "reusable_focused_tests", lambda path: record if cached else None
    )

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        ran.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "RegionalLiveFixture", SimpleNamespace(run=run))
    result = module.focused_tests(tmp_path, reuse=True)
    assert result["passed"] is True, result
    assert bool(ran) is not cached, ran
    if cached:
        assert result == {**record, "focused_tests_reused": True}, result
