"""The command-line surface of the parallel-domain runners DESTR-027/028/029:
``configure`` from parsed flags, the focused-test gate and its plan cache, and
the script entry under the abort wrapper -- no cluster, no subprocess."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import live_driver_guard, regional_live_fixture
from scripts.e2e.regional import run_destr027_two_idle_node_parallel_reset as destr027
from scripts.e2e.regional import run_destr028_two_job_parallel_reboot as destr028
from scripts.e2e.regional import run_destr029_job_and_idle_parallel as destr029
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_parallel_domains import regional_settings

MODULES = (destr027, destr028, destr029)
IDS = ("destr027", "destr028", "destr029")


def cli(module: ModuleType, tmp_path: Path, *extra: str) -> list[str]:
    regional = regional_settings(tmp_path)
    argv = [
        "--run-dir",
        str(tmp_path / "run-hp"),
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
    if module is not destr027:
        site = tmp_path / "site.yaml"
        site.write_text("fake site\n", encoding="utf-8")
        argv += ["--site-file", str(site)]
    return [*argv, *extra]


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_configure_defaults_the_predecessor_and_the_live_group(
    module: ModuleType, tmp_path: Path
) -> None:
    arguments = module.parser().parse_args(
        cli(module, tmp_path, "--host-probe-image", "example.test/probe")
    )

    settings = module.configure(arguments)

    predecessor = module.PREDECESSOR_CASE_ID
    assert (
        settings.predecessor_path
        == (
            tmp_path / "run-hp" / "cases" / predecessor / f"{predecessor}.json"
        ).resolve()
    ), settings
    assert settings.node_group == "", "the largest live group is the default"
    environment = settings.environment()
    assert environment["GPU_FAULT_CLUSTER_ID"] == "cluster-a", environment
    assert environment["GPU_FAULT_HOST_PROBE_IMAGE"] == "example.test/probe"
    assert environment["GPU_FAULT_NODE_GROUP"] == "", environment
    if module is not destr027:
        assert settings.manifest == module.TRAINING_MANIFEST, settings
        assert environment["GPU_FAULT_SITE_FILE"] == str(tmp_path / "site.yaml")
    if module is destr028:
        assert settings.single_node_manifest == destr028.SINGLE_NODE_TRAINING_MANIFEST
        assert environment["GPU_FAULT_SINGLE_NODE_TRAINING_MANIFEST"] == str(
            destr028.SINGLE_NODE_TRAINING_MANIFEST
        ), environment


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_configure_takes_explicit_inputs_and_the_environment_image(
    module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_HOST_PROBE_IMAGE", "example.test/from-env")
    extra = [
        "--predecessor-evidence",
        str(tmp_path / "previous.json"),
        "--node-group",
        "  p5en  ",
    ]
    if module is not destr027:
        extra += ["--manifest", str(tmp_path / "job.yaml")]
    if module is destr028:
        extra += ["--single-node-manifest", str(tmp_path / "single.yaml")]

    settings = module.configure(
        module.parser().parse_args(cli(module, tmp_path, *extra))
    )

    assert settings.host_probe_image == "example.test/from-env", settings
    assert settings.predecessor_path == (tmp_path / "previous.json").resolve()
    assert settings.node_group == "p5en", "the group name is trimmed"
    environment = settings.environment()
    assert environment["GPU_FAULT_PREDECESSOR_EVIDENCE"] == str(
        (tmp_path / "previous.json").resolve()
    ), environment
    if module is not destr027:
        assert settings.manifest == (tmp_path / "job.yaml").resolve(), settings
        assert environment["GPU_FAULT_TRAINING_MANIFEST"] == str(settings.manifest)
    if module is destr028:
        assert settings.single_node_manifest == (tmp_path / "single.yaml").resolve()


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_configure_requires_a_host_probe_image(
    module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GPU_FAULT_HOST_PROBE_IMAGE", raising=False)
    arguments = module.parser().parse_args(cli(module, tmp_path))

    with pytest.raises(RegionalFixtureError, match="host probe image is required"):
        module.configure(arguments)


@pytest.mark.parametrize("module", MODULES, ids=IDS)
@pytest.mark.parametrize("returncode", [0, 2])
def test_the_focused_tests_run_once_and_keep_a_private_log(
    module: ModuleType, returncode: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, "out\n", "err\n")

    monkeypatch.setattr(module, "RegionalLiveFixture", SimpleNamespace(run=run))

    result = module.focused_tests(tmp_path)

    assert result["passed"] is (returncode == 0), result
    assert result["returncode"] == returncode, result
    assert result["command"] == [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *module.FOCUSED_TESTS,
    ], result
    (command, options) = commands[0]
    assert len(commands) == 1 and options["check"] is False, commands
    assert options["cwd"] == module.ROOT and options["timeout"] == 600, options
    log = tmp_path / "focused-tests.log"
    assert log.read_text(encoding="utf-8") == "out\nerr\n"
    assert log.stat().st_mode & 0o777 == 0o600, oct(log.stat().st_mode)


@pytest.mark.parametrize("module", MODULES, ids=IDS)
@pytest.mark.parametrize("cached", [False, True])
def test_an_execute_reuses_only_a_plan_result_that_still_speaks_for_the_tree(
    module: ModuleType, cached: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = {"passed": True, "returncode": 0, "command": ["recorded"]}
    asked: list[Path] = []
    ran: list[list[str]] = []

    def reusable(path: Path) -> dict[str, Any] | None:
        asked.append(path)
        return dict(record) if cached else None

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        ran.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "reusable_focused_tests", reusable)
    monkeypatch.setattr(module, "RegionalLiveFixture", SimpleNamespace(run=run))

    result = module.focused_tests(tmp_path, reuse=True)

    assert asked == [tmp_path / "plan.json"], asked
    assert bool(ran) is not cached, ran
    if cached:
        assert result == {**record, "focused_tests_reused": True}, result
    else:
        assert result["passed"] is True and "focused_tests_reused" not in result


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_the_script_entry_runs_main_under_the_abort_wrapper(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        sys, "path", [path for path in sys.path if path != str(module.ROOT)]
    )

    def standard(case: Any) -> int:
        calls.append(case.case_id)
        return 3

    def wrapper(entry: Any) -> int:
        calls.append("wrapper")
        return 70 + entry()

    monkeypatch.setattr(live_driver_guard, "run_standard_case", standard)
    monkeypatch.setattr(regional_live_fixture, "run_case_main", wrapper)

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(module.__file__, run_name="__main__")

    assert stopped.value.code == 73, stopped.value
    assert calls == ["wrapper", module.CASE_ID], calls
    assert sys.path[0] == str(module.ROOT), "the script puts its repository first"
