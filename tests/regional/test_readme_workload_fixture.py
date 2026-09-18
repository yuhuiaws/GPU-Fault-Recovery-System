"""Public training entrypoints and real apply semantics for baseline fixtures."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.training_submit_cli import parser, render_workload
from scripts.e2e.regional import readme_workload_fixture as module
from scripts.e2e.regional.managed_workload_fixture import (
    OWNER_LABEL,
    ManagedWorkloadSettings,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
)
from tests.regional.test_managed_workload_fixture import Kubernetes


class PublicSubmissionApi(Kubernetes):
    def __init__(self) -> None:
        super().__init__()
        self.public_commands: list[list[str]] = []
        self.apply_commands: list[list[str]] = []
        self.source_documents: list[dict[str, Any]] = []
        self.dry_run_failure = False
        self.late_foreign = False
        self.bad_ack = False
        self.replace_after_apply = False
        self.mutate_after_dry_run = False
        self.cli_failure = False

    def apply(
        self,
        command: list[str],
        document: dict[str, Any],
        *,
        dry_run: bool,
        console: bool,
    ) -> subprocess.CompletedProcess[str]:
        self.apply_commands.append(command.copy())
        key = (document["kind"].lower(), document["metadata"]["name"])
        if dry_run:
            if self.mutate_after_dry_run:
                Path(command[command.index("-f") + 1]).write_text("changed\n")
            return subprocess.CompletedProcess(
                command, int(self.dry_run_failure), json.dumps(document), ""
            )
        if self.late_foreign:
            foreign = copy.deepcopy(document)
            foreign["metadata"]["labels"][OWNER_LABEL] = "foreign-owner"
            self.add(foreign)
        if key in self.objects:
            return subprocess.CompletedProcess(command, 1, "", "patch Forbidden")
        self.created.append(copy.deepcopy(document))
        created = self.add(document)
        output = (
            f"pytorchjob.kubeflow.org/{key[1]} created\n"
            if console
            else json.dumps(created)
        )
        if self.bad_ack:
            output = "not an acknowledgement"
        if self.replace_after_apply:
            self.objects[key]["metadata"]["uid"] = "replacement"
        return subprocess.CompletedProcess(command, 0, output, "")

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if Path(command[0]).name == "gpu-training-submit":
            self.public_commands.append(command.copy())
            assert kwargs["check"] is False, "the fixture must inspect CLI exit status"
            arguments = parser().parse_args(command[1:])
            source = yaml.safe_load(arguments.manifest.read_text())
            self.source_documents.append(source)
            if self.cli_failure:
                return subprocess.CompletedProcess(
                    command, 23, "", "private credential diagnostic must not escape"
                )
            rendered = render_workload(
                arguments.manifest,
                job_id=arguments.job_id,
                attempt_id=arguments.attempt_id,
                attempt_number=arguments.attempt_number,
                runtime_profile_version="profile-test",
                expected_critical_ranks=arguments.expected_critical_ranks,
                training_container=arguments.training_container,
                restart_budget=arguments.restart_budget,
                namespace=arguments.namespace,
            )
            return self.apply(
                ["kubectl", "apply", "-f", "-"],
                yaml.safe_load(rendered.manifest),
                dry_run=False,
                console=True,
            )
        if command[0] == "kubectl" and "apply" in command:
            document = yaml.safe_load(
                Path(command[command.index("-f") + 1]).read_text()
            )
            return self.apply(
                command, document, dry_run="--dry-run=server" in command, console=False
            )
        return super().run(command, **kwargs)


def harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[module.ReadmeWorkloadFixture, PublicSubmissionApi, list[str]]:
    cpu, gpu, site = (tmp_path / name for name in ("cpu.yaml", "gpu.yaml", "site.yaml"))
    for path in (cpu, gpu, site):
        path.write_text("apiVersion: v1\n")
    regional = RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=cpu,
            gpu_kubeconfig=gpu,
            gpu_context="gpu-context",
            namespace="gpu-fault-system",
            cluster_id="cluster-a",
            region="us-west-2",
        )
    )
    api = PublicSubmissionApi()
    monkeypatch.setattr(regional, "run", api.run)
    events: list[str] = []

    class SubmissionIdentity:
        def __init__(self, region: Any, **kwargs: Any) -> None:
            assert region is regional
            assert kwargs["resource"] == "pytorchjob"
            assert kwargs["resource_name"] == "gpu-fault-three-node-pytorch"
            self.kubeconfig = kwargs["directory"] / "restricted.kubeconfig"

        def __enter__(self) -> SubmissionIdentity:
            events.append("identity-enter")
            return self

        def __exit__(self, *args: Any) -> None:
            events.append("identity-cleanup")

    monkeypatch.setattr(module, "CreateOnlySubmissionIdentity", SubmissionIdentity)
    monkeypatch.setattr(module, "training_cli", lambda name: name)
    fixture = module.ReadmeWorkloadFixture(
        regional,
        ManagedWorkloadSettings(
            manifest=module.ROOT / "examples/hyperpod/three-node-pytorchjob.yaml",
            site_file=site,
            job_id="readme-test",
            attempt_id="readme-test-a001",
            restart_budget=1,
            expected_pods=3,
            expected_gpu_count=24,
        ),
        case_dir=tmp_path / "case",
        training_container="pytorch",
    )
    return fixture, api, events


def annotated_path(fixture: module.ReadmeWorkloadFixture) -> Path:
    arguments = parser().parse_args(fixture.submission_arguments())
    rendered = render_workload(
        arguments.manifest,
        job_id=arguments.job_id,
        attempt_id=arguments.attempt_id,
        attempt_number=arguments.attempt_number,
        runtime_profile_version="profile-test",
        expected_critical_ranks=None,
        training_container=arguments.training_container,
        restart_budget=arguments.restart_budget,
        namespace=arguments.namespace,
    )
    path = fixture.directory / "annotated.yaml"
    path.write_text(rendered.manifest)
    return path


def test_submit_runs_public_console_and_preserves_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, events = harness(tmp_path, monkeypatch)
    result = fixture.submit()
    command = api.public_commands[0]
    assert command[0] == "gpu-training-submit"
    assert command[command.index("--attempt-number") + 1] == "1"
    assert "--attempt-id" not in command
    assert command[command.index("--training-container") + 1] == "pytorch"
    assert command[command.index("--kubeconfig") + 1].endswith(
        "restricted.kubeconfig"
    ), "public submission must use the create-only scoped identity"
    assert api.apply_commands == [["kubectl", "apply", "-f", "-"]]
    assert api.source_documents[0]["metadata"]["labels"][OWNER_LABEL] == fixture.owner
    assert result["entrypoint"] == "gpu-training-submit"
    assert result["kubectl_verb"] == "apply"
    assert events == ["identity-enter", "identity-cleanup"]
    fixture.delete()
    assert not api.objects, "successful submission must remain ownership-cleanable"


def test_annotated_path_uses_same_file_for_dry_run_and_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, events = harness(tmp_path, monkeypatch)
    path = annotated_path(fixture)
    original = path.read_bytes()
    result = fixture.apply_rendered(path)
    dry_run, apply = api.apply_commands
    assert dry_run == [*apply, "--dry-run=server"]
    assert apply[apply.index("-f") + 1] == str(path)
    assert path.read_bytes() == original
    assert not api.public_commands, "independent apply must not invoke live submit"
    assert result["entrypoint"] == "kubectl apply"
    assert result["server_dry_run_returncode"] == result["apply_returncode"] == 0
    assert events == ["identity-enter", "identity-cleanup"]
    fixture.delete()
    assert not api.objects, "independent apply must remain ownership-cleanable"


@pytest.mark.parametrize("entrypoint", ["submit", "apply"])
def test_preexisting_workload_is_not_submitted_or_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entrypoint: str
) -> None:
    fixture, api, events = harness(tmp_path, monkeypatch)
    path = annotated_path(fixture)
    foreign = yaml.safe_load(path.read_text())
    foreign["metadata"]["labels"][OWNER_LABEL] = "foreign-owner"
    existing = copy.deepcopy(api.add(foreign))
    with pytest.raises(RegionalFixtureError, match="already exists"):
        fixture.submit() if entrypoint == "submit" else fixture.apply_rendered(path)
    fixture.delete()
    assert api.objects[(fixture.resource, fixture.name)] == existing
    assert not api.public_commands and not api.apply_commands and not api.deletes
    assert events == []


@pytest.mark.parametrize("entrypoint", ["submit", "apply"])
def test_late_foreign_resource_is_not_patched_or_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entrypoint: str
) -> None:
    fixture, api, events = harness(tmp_path, monkeypatch)
    path = annotated_path(fixture)
    api.late_foreign = True
    with pytest.raises(RegionalFixtureError, match="failed with exit status"):
        fixture.submit() if entrypoint == "submit" else fixture.apply_rendered(path)
    with pytest.raises(RegionalFixtureError, match="ownership"):
        fixture.delete()
    assert not api.created and not api.deletes
    assert (
        api.objects[(fixture.resource, fixture.name)]["metadata"]["labels"][OWNER_LABEL]
        == "foreign-owner"
    )
    assert events[-1] == "identity-cleanup"


@pytest.mark.parametrize(
    "defect", ["dry-run", "manifest-drift", "bad-ack", "replacement"]
)
def test_apply_failures_preserve_cleanup_and_incarnation_fences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    fixture, api, events = harness(tmp_path, monkeypatch)
    path = annotated_path(fixture)
    api.dry_run_failure = defect == "dry-run"
    api.mutate_after_dry_run = defect == "manifest-drift"
    api.bad_ack = defect == "bad-ack"
    api.replace_after_apply = defect == "replacement"
    with pytest.raises(RegionalFixtureError):
        fixture.apply_rendered(path)
    assert events[-1] == "identity-cleanup"
    if defect == "replacement":
        with pytest.raises(RegionalFixtureError, match="ownership"):
            fixture.delete()
        assert not api.deletes, "cleanup must never delete a replacement incarnation"
    else:
        fixture.delete()
        assert not api.objects, "a failed own submission must be cleaned safely"
    if defect in {"dry-run", "manifest-drift"}:
        assert len(api.apply_commands) == 1 and not api.created


def test_console_failure_keeps_credential_diagnostics_out_of_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, events = harness(tmp_path, monkeypatch)
    api.cli_failure = True
    with pytest.raises(RegionalFixtureError, match="exit status 23") as failed:
        fixture.submit()
    assert "private credential" not in str(failed.value)
    assert events[-1] == "identity-cleanup"
    fixture.delete()


def test_public_entrypoint_must_come_from_the_current_python_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "bin"
    binary.mkdir()
    monkeypatch.setattr(
        module, "sys", SimpleNamespace(executable=str(binary / "python"))
    )
    with pytest.raises(RegionalFixtureError, match="missing"):
        module.training_cli("gpu-training-submit")
    executable = binary / "gpu-training-submit"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    assert module.training_cli("gpu-training-submit") == str(executable)
    with pytest.raises(ValueError, match="unsupported"):
        module.training_cli("other-command")
