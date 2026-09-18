from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import boot_acceptance_common as common
from scripts.e2e.regional.regional_commands import (
    RegionalCommandTimeout,
    RegionalFixtureError,
)
from tests.regional._cov95_boot_site import BootSite


def test_command_wrapper_preserves_umask_timeout_and_error_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "unit", "")

    monkeypatch.setattr(common, "run_fixture_command", run)
    assert common.run(["unit-command"], umask=0o77, timeout=7).stdout == "unit"
    command, options = calls[0]
    assert command[:2] == ["sh", "-c"]
    assert command[-3:] == ["gpu-fault-umask", "077", "unit-command"]
    assert options["timeout"] == 7

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise RegionalFixtureError("synthetic failure")

    monkeypatch.setattr(common, "run_fixture_command", fail)
    with pytest.raises(common.BootAcceptanceError, match="synthetic failure"):
        common.run(["unit-command"])

    def timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise RegionalCommandTimeout(["unit-command"], 7)

    monkeypatch.setattr(common, "run_fixture_command", timeout)
    with pytest.raises(RegionalCommandTimeout):
        common.run(["unit-command"])


@pytest.mark.parametrize("fault", ["gpu-config", "cluster", "ambiguous"])
def test_site_fixture_requires_explicit_gpu_configuration_and_one_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    model = BootSite(tmp_path, monkeypatch)
    if fault == "gpu-config":
        model.config.pop("gpu_kubeconfig")
    elif fault == "ambiguous":
        model.config["clusters"].append({"cluster_id": "cluster-b", "context": "other"})
    with pytest.raises(common.BootAcceptanceError, match="GPU kubeconfig|exactly one"):
        common.SiteFixture(
            tmp_path / "site.yaml", "unknown" if fault == "cluster" else ""
        )
    assert model.reads == []


def test_site_fixture_can_use_the_site_gpu_environment_without_reading_external_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = BootSite(tmp_path, monkeypatch)
    model.site.environment["KUBECONFIG"] = model.config.pop("gpu_kubeconfig")
    fixture = common.SiteFixture(tmp_path / "site.yaml", "cluster-a")
    assert fixture.gpu_kubeconfig == tmp_path / "unused-kubeconfig"
    assert fixture.regional.settings.gpu_context == "unit-context"
    assert fixture.pods("cpu", "api") == ["other-pod", "test-pod"]


@pytest.mark.parametrize(
    "fault", ["replicas", "inventory", "malformed", "partial", "duplicate-uid"]
)
def test_site_pod_inventory_requires_complete_unique_ready_replicas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    model = BootSite(tmp_path, monkeypatch)
    if fault == "replicas":
        model.replicas = True
    elif fault == "inventory":
        model.pods.pop()
    elif fault == "malformed":
        model.pods[1] = None
    elif fault == "partial":
        model.pods[0]["status"]["containerStatuses"][0]["ready"] = False
    else:
        model.pods[1]["metadata"]["uid"] = model.pods[0]["metadata"]["uid"]
    fixture = common.SiteFixture(tmp_path / "site.yaml", "cluster-a")
    with pytest.raises(common.BootAcceptanceError, match="replica"):
        fixture.pods("cpu", "api")


@pytest.mark.parametrize("output", ['{"ok":true}', 'diagnostic\n{"ok":true}'])
def test_site_probe_uses_installed_interpreter_and_parses_public_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    model = BootSite(tmp_path, monkeypatch)
    model.probe_output = output
    fixture = common.SiteFixture(tmp_path / "site.yaml", "cluster-a")
    assert fixture.pod_json("gpu", "pod", "unit script", "argument", timeout=9) == {
        "ok": True
    }
    plane, args, kwargs = model.reads[-1]
    assert plane == "gpu"
    assert args == (
        "exec",
        "-i",
        "pod",
        "--",
        "/opt/gpu-fault/executor/bin/python",
        "-",
        "argument",
    )
    assert kwargs == {"input_text": "unit script", "timeout": 9}


def test_site_probe_rejects_nonobject_or_missing_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = BootSite(tmp_path, monkeypatch)
    fixture = common.SiteFixture(tmp_path / "site.yaml", "cluster-a")
    model.probe_output = "[]"
    with pytest.raises(common.BootAcceptanceError, match="JSON object"):
        fixture.pod_json("cpu", "pod", "unit script")
    with pytest.raises(json.JSONDecodeError):
        common.parse_probe_json("")


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_site_exec_forwards_scope_and_stays_behind_the_recording_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plane: str
) -> None:
    model = BootSite(tmp_path, monkeypatch)
    monkeypatch.setattr(common, "run", model.run)
    fixture = common.SiteFixture(tmp_path / "site.yaml", "cluster-a")
    fixture.exec(plane, "pod", "unit-command", input_text="unit data", timeout=4)
    args, options = model.commands[-1]
    assert args[0:2] == ["kubectl", "--kubeconfig"]
    assert ("--context" in args) is (plane == "gpu")
    assert args[-5:] == ["exec", "-i", "pod", "--", "unit-command"]
    assert options["input_text"] == "unit data"
    assert options["timeout"] == 4
    with pytest.raises(ValueError, match="unknown plane"):
        fixture.exec("other", "pod", "unit-command")
