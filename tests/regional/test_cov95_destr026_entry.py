"""GF-REGIONAL-DESTR-026 entry points: configuration, the focused-test gate and
the script entry, each without a cluster."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import live_driver_guard, regional_live_fixture
from scripts.e2e.regional import run_destr026_same_node_two_gpu_reset as destr026
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from scripts.e2e.regional.site_profile import SITE_PROFILE_ENV
from tests.regional._cov95_destr_warm import regional_settings

ENVIRONMENT = (
    "GPU_FAULT_SITE_FILE",
    "GPU_FAULT_HOST_PROBE_IMAGE",
    "GPU_FAULT_FAULT_NODE",
    "GPU_FAULT_NODE_B",
    SITE_PROFILE_ENV,
)


def cli(tmp_path: Path, *extra: str) -> list[str]:
    regional = regional_settings(tmp_path)
    return [
        "--run-dir",
        str(tmp_path / "regional-acceptance-r9-hp"),
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
        *extra,
    ]


@pytest.fixture
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_configure_binds_the_flags_and_derives_the_job_identity(
    tmp_path: Path, clean_environment: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "site.yaml"
    site.write_text("fake site\n", encoding="utf-8")
    arguments = destr026.parser().parse_args(
        cli(
            tmp_path,
            "--site-file",
            str(site),
            "--host-probe-image",
            "example.test/probe",
            "--fault-node",
            "node-a",
            "--node-b",
            "node-b",
            "--pci-bdf-a",
            " 0000:59:00.0 ",
        )
    )

    settings = destr026.configure(arguments)

    run_dir = tmp_path / "regional-acceptance-r9-hp"
    job, attempt = destr026.derived_identity(run_dir, 1)
    assert (settings.job_id, settings.attempt_id) == (job, attempt), settings
    assert settings.nodes == ("node-a", "node-b"), settings
    assert (settings.pci_bdf_a, settings.pci_bdf_b) == ("0000:59:00.0", ""), settings
    assert settings.site_file == site.resolve(), settings
    assert settings.manifest == destr026.join_case.DEFAULT_MANIFEST.resolve(), settings
    predecessor = destr026.PREDECESSOR_CASE_ID
    expected = run_dir / "cases" / predecessor / f"{predecessor}.json"
    assert settings.predecessor_path == expected.resolve(), settings
    environment = settings.environment()
    assert environment["GPU_FAULT_FAULT_NODE"] == "node-a", environment
    assert environment["GPU_FAULT_NODE_B"] == "node-b", environment
    assert environment["GPU_FAULT_TEST_JOB_ID"] == job, environment
    assert environment["GPU_FAULT_TEST_ATTEMPT_ID"] == attempt, environment
    assert environment["GPU_FAULT_CLUSTER_ID"] == "cluster-a", environment


def test_configure_falls_back_to_the_environment_and_keeps_explicit_ids(
    tmp_path: Path, clean_environment: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "from-env.yaml"
    site.write_text("fake site\n", encoding="utf-8")
    clean_environment.setenv("GPU_FAULT_SITE_FILE", str(site))
    clean_environment.setenv("GPU_FAULT_HOST_PROBE_IMAGE", "example.test/env-probe")
    clean_environment.setenv("GPU_FAULT_FAULT_NODE", "env-node-a")
    clean_environment.setenv("GPU_FAULT_NODE_B", "env-node-b")
    arguments = destr026.parser().parse_args(
        cli(
            tmp_path,
            "--job-id",
            " operator-job ",
            "--attempt-id",
            "operator-job-a007",
            "--predecessor-evidence",
            str(tmp_path / "previous.json"),
        )
    )

    settings = destr026.configure(arguments)

    assert settings.site_file == site.resolve(), settings
    assert settings.host_probe_image == "example.test/env-probe", settings
    assert settings.nodes == ("env-node-a", "env-node-b"), settings
    assert (settings.job_id, settings.attempt_id) == (
        "operator-job",
        "operator-job-a007",
    ), settings
    assert settings.predecessor_path == (tmp_path / "previous.json").resolve()


def test_configure_requires_the_fault_node(
    tmp_path: Path, clean_environment: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "site.yaml"
    site.write_text("fake site\n", encoding="utf-8")
    arguments = destr026.parser().parse_args(
        cli(
            tmp_path,
            "--site-file",
            str(site),
            "--host-probe-image",
            "example.test/probe",
            "--node-b",
            "node-b",
        )
    )

    with pytest.raises(RegionalFixtureError, match="fault node is required"):
        destr026.configure(arguments)


@pytest.mark.parametrize(
    ("reuse", "recorded", "returncode"),
    [
        (False, {"passed": True}, 0),
        (False, None, 1),
        (True, None, 0),
        (True, {"passed": True, "returncode": 0, "command": ["recorded"]}, 0),
    ],
    ids=["fresh-ignores-plan", "fresh-fails", "reuse-without-plan", "reuse-plan"],
)
def test_the_focused_tests_run_unless_the_plan_already_passed_them(
    reuse: bool,
    recorded: dict[str, Any] | None,
    returncode: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran: list[tuple[list[str], dict[str, Any]]] = []
    asked: list[Path] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        ran.append((command, kwargs))
        return subprocess.CompletedProcess(
            command, returncode, "fake out\n", "fake err\n"
        )

    def reusable(path: Path) -> dict[str, Any] | None:
        asked.append(path)
        return recorded

    monkeypatch.setattr(destr026, "RegionalLiveFixture", SimpleNamespace(run=run))
    monkeypatch.setattr(destr026, "reusable_focused_tests", reusable)

    result = destr026.focused_tests(tmp_path, reuse=reuse)

    assert asked == ([tmp_path / "plan.json"] if reuse else []), asked
    if reuse and recorded is not None:
        assert result == {**recorded, "focused_tests_reused": True}, result
        assert ran == [], "a reusable plan result is not paid for twice"
        assert not (tmp_path / "focused-tests.log").exists(), "no fresh log"
        return
    (command, kwargs) = ran[0]
    assert len(ran) == 1, ran
    assert command == [sys.executable, "-m", "pytest", "-q", *destr026.FOCUSED_TESTS]
    assert kwargs == {"cwd": destr026.ROOT, "check": False, "timeout": 600}, kwargs
    assert result == {
        "passed": returncode == 0,
        "returncode": returncode,
        "command": command,
    }, result
    log = tmp_path / "focused-tests.log"
    assert log.read_text(encoding="utf-8") == "fake out\nfake err\n", log
    assert log.stat().st_mode & 0o777 == 0o600, oct(log.stat().st_mode)


def test_main_hands_the_case_to_the_shared_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(
        destr026, "run_standard_case", lambda case: seen.append(case) or 3
    )

    assert destr026.main() == 3
    assert seen == [destr026.CASE], seen


def test_the_script_entry_finds_the_repository_and_runs_the_case_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []
    root = str(destr026.ROOT)
    monkeypatch.setattr(sys, "path", [path for path in sys.path if path != root])
    monkeypatch.setattr(
        live_driver_guard, "run_standard_case", lambda case: seen.append(case) or 7
    )
    monkeypatch.setattr(regional_live_fixture, "run_case_main", lambda main: main())

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(destr026.__file__, run_name="__main__")

    assert stopped.value.code == 7, stopped.value
    (case,) = seen
    assert (case.case_id, case.confirmation) == (
        destr026.CASE_ID,
        destr026.CONFIRMATION,
    ), case
    assert sys.path[0] == root, "the script puts the repository first on its path"
