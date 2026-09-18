"""Exercise the public README submission paths without rewriting kubectl apply."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml  # type: ignore[import-untyped]

from scripts.e2e.regional.fixture_ownership import creation_document
from scripts.e2e.regional.managed_workload_fixture import (
    OWNER_LABEL,
    ROOT,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
    read_resource,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
)
from scripts.e2e.regional.workload_submission_identity import (
    CreateOnlySubmissionIdentity,
)


def training_cli(name: str) -> str:
    if name not in {"gpu-training-submit", "gpu-fault-workload-annotate"}:
        raise ValueError("unsupported public training entrypoint")
    executable = Path(sys.executable).parent / name
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RegionalFixtureError(
            f"{name} is missing from the runner's Python environment; "
            "complete README deployment-host setup and activate its environment"
        )
    return str(executable)


def mark_workload_owner(value: Any, owner: str) -> None:
    if isinstance(value, dict):
        if isinstance(value.get("metadata"), dict):
            labels = value["metadata"].setdefault("labels", {})
            if not isinstance(labels, dict) or labels.get(OWNER_LABEL, owner) != owner:
                raise RegionalFixtureError("source workload has another fixture owner")
            labels[OWNER_LABEL] = owner
        for child in value.values():
            mark_workload_owner(child, owner)
    elif isinstance(value, list):
        for child in value:
            mark_workload_owner(child, owner)


class ReadmeWorkloadFixture(ManagedWorkloadFixture):
    def __init__(
        self,
        regional: RegionalLiveFixture,
        settings: ManagedWorkloadSettings,
        *,
        case_dir: Path,
        training_container: str,
    ) -> None:
        super().__init__(regional, settings)
        if not training_container:
            raise ValueError("the baseline training container must be explicit")
        if case_dir.is_symlink():
            raise RegionalFixtureError("workload evidence directory must not be a link")
        case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.directory = case_dir / f"submission-{self.owner}"
        self.directory.mkdir(mode=0o700)
        self.training_container = training_container
        self.source_path = self.directory / "source-workload.yaml"

    def prepare_source(self) -> Path:
        if self.source_path.exists():
            return self.source_path
        document = yaml.safe_load(self.settings.manifest.read_text(encoding="utf-8"))
        mark_workload_owner(document, self.owner)
        with self.source_path.open("x", encoding="utf-8") as stream:
            stream.write(yaml.safe_dump(document, sort_keys=False))
        self.source_path.chmod(0o600)
        return self.source_path

    def submission_arguments(self) -> list[str]:
        arguments = [
            "--site",
            str(self.settings.site_file),
            str(self.prepare_source()),
            "--job-id",
            self.settings.job_id,
            "--attempt-number",
            "1",
            "--training-container",
            self.training_container,
            "--restart-budget",
            str(self.settings.restart_budget),
            "--namespace",
            self.regional.settings.namespace,
        ]
        if self.settings.attempt_id != f"{self.settings.job_id}-a001":
            arguments.extend(("--attempt-id", self.settings.attempt_id))
        return arguments

    def require_empty(self) -> None:
        if self.submission_started:
            raise RegionalFixtureError("managed workload submission already started")
        if (
            read_resource(self.regional, self.resource, self.name) is not None
            or self.inventory()
        ):
            raise RegionalFixtureError(
                "managed workload name or job identity already exists"
            )

    def run_submission(
        self, command: list[str], *, stage: str
    ) -> subprocess.CompletedProcess[str]:
        completed = self.regional.run(
            command,
            cwd=ROOT,
            env={
                **os.environ,
                "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}",
            },
            check=False,
            timeout=300,
        )
        if completed.returncode:
            raise RegionalFixtureError(
                f"{stage} failed with exit status {completed.returncode}"
            )
        return completed

    def observe_submission(self) -> None:
        current = read_resource(self.regional, self.resource, self.name)
        if current is None:
            raise RegionalFixtureError("submitted managed workload is missing")
        self.adopt(current)

    def submit(self) -> dict[str, Any]:
        self.require_empty()
        arguments = self.submission_arguments()
        with CreateOnlySubmissionIdentity(
            self.regional,
            directory=self.directory,
            owner=self.owner,
            resource=self.resource,
            resource_name=self.name,
        ) as identity:
            command = [
                training_cli("gpu-training-submit"),
                *arguments,
                "--kubeconfig",
                str(identity.kubeconfig),
                "--context",
                self.regional.settings.gpu_context,
            ]
            self.submission_started = True
            completed = self.run_submission(command, stage="gpu-training-submit")
            if not completed.stdout.strip().endswith(f"/{self.name} created"):
                raise RegionalFixtureError(
                    "gpu-training-submit did not acknowledge a newly created workload"
                )
            self.observe_submission()
        return {
            "entrypoint": "gpu-training-submit",
            "returncode": 0,
            "kubectl_verb": "apply",
            "create_only_identity": True,
        }

    def apply_rendered(self, path: Path) -> dict[str, Any]:
        self.require_empty()
        if path.is_symlink() or not path.is_file():
            raise RegionalFixtureError("annotated workload file is missing or linked")
        rendered = path.read_text(encoding="utf-8")
        document = yaml.safe_load(rendered)
        metadata = document.get("metadata") if isinstance(document, dict) else None
        labels = metadata.get("labels") if isinstance(metadata, dict) else None
        if (
            not isinstance(metadata, dict)
            or not isinstance(labels, dict)
            or document.get("kind") != self.kind
            or metadata.get("name") != self.name
            or metadata.get("namespace") != self.regional.settings.namespace
            or labels.get("gpu-fault.io/job-id") != self.settings.job_id
            or labels.get("gpu-fault.io/attempt-id") != self.settings.attempt_id
            or labels.get(OWNER_LABEL) != self.owner
        ):
            raise RegionalFixtureError("annotated workload differs from the fixture")
        with CreateOnlySubmissionIdentity(
            self.regional,
            directory=self.directory,
            owner=self.owner,
            resource=self.resource,
            resource_name=self.name,
        ) as identity:
            command = [
                "kubectl",
                "--kubeconfig",
                str(identity.kubeconfig),
                "--context",
                self.regional.settings.gpu_context,
                "--namespace",
                self.regional.settings.namespace,
                "apply",
                "-f",
                str(path),
                "-o",
                "json",
            ]
            self.run_submission(
                [*command, "--dry-run=server"],
                stage="annotated workload server dry-run",
            )
            if path.is_symlink() or path.read_text(encoding="utf-8") != rendered:
                raise RegionalFixtureError("annotated workload changed after dry-run")
            self.submission_started = True
            completed = self.run_submission(command, stage="annotated workload apply")
            try:
                created = creation_document(completed.stdout)
            except RegionalFixtureError:
                raise RegionalFixtureError(
                    "workload apply acknowledgement is invalid"
                ) from None
            self.adopt(created)
            self.observe_submission()
        return {
            "entrypoint": "kubectl apply",
            "returncode": 0,
            "server_dry_run_returncode": 0,
            "apply_returncode": 0,
            "kubectl_verb": "apply",
            "create_only_identity": True,
        }
