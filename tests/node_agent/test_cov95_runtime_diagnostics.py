from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import boto3
import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import NodeActionExecutor, NodeActionStatus
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.node_agent._support import FakeRunner, command, envelope
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


def executable(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "unit-field-diagnostic"
    path.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
    path.chmod(0o755)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "defect",
    ["relative", "placeholder", "no-digest", "missing", "permission", "digest"],
)
def test_field_diagnostic_configuration_requires_a_pinned_executable(
    node_factory: Callable[..., NodeActionExecutor], tmp_path: Path, defect: str
) -> None:
    path, digest = executable(tmp_path)
    arguments = (str(path), "--gpu={gpu_uuid}", "--bdf={pci_bdf}")
    if defect == "relative":
        arguments = ("relative-command",)
    elif defect == "placeholder":
        arguments += ("{unknown}",)
    elif defect == "no-digest":
        digest = ""
    elif defect == "missing":
        arguments = (str(tmp_path / "absent"),)
    elif defect == "permission":
        path.chmod(0o600)
    else:
        digest = "0" * 64
    runner = FakeRunner()
    with pytest.raises(ValueError, match="absolute|placeholders|SHA-256|executable"):
        node_factory(
            allowed_operations={WorkflowOperation.RUN_NVLINK74_WORKFLOW},
            field_diagnostic_enabled=True,
            field_diagnostic_command=arguments,
            field_diagnostic_sha256=digest,
            runner=runner,
        )
    assert runner.commands == []


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("disabled", "disabled"),
        ("missing-memory", "memory Field Diagnostic is not configured"),
        ("missing-link", "explicit NVLink ID"),
        ("negative-link", "explicit NVLink ID"),
        ("memory-negative-link", "NVLink ID is invalid"),
        ("memory-link-template", "cannot diagnose a memory fault"),
        ("zero-gpus", "exactly one GPU UUID"),
        ("two-gpus", "exactly one GPU UUID"),
        ("pci-type", "PCI BDF is invalid"),
    ],
)
def test_field_diagnostic_refuses_unbound_targets_before_any_gpu_probe(
    node_factory: Callable[..., NodeActionExecutor],
    tmp_path: Path,
    defect: str,
    message: str,
) -> None:
    path, digest = executable(tmp_path)
    operation = (
        WorkflowOperation.RUN_FIELD_DIAGNOSTIC
        if defect.startswith("memory-") or defect == "missing-memory"
        else WorkflowOperation.RUN_NVLINK74_WORKFLOW
    )
    arguments = (str(path), "--gpu={gpu_uuid}")
    memory = arguments + (
        ("--link={link_id}",) if defect == "memory-link-template" else ()
    )
    runner = FakeRunner()
    agent = node_factory(
        allowed_operations={operation},
        field_diagnostic_enabled=defect != "disabled",
        field_diagnostic_command=arguments,
        field_diagnostic_sha256=digest,
        memory_field_diagnostic_command=() if defect == "missing-memory" else memory,
        memory_field_diagnostic_sha256=digest,
        runner=runner,
    )
    parameters: dict[str, Any] = {"nvlink_link_id": 1}
    if defect in {"missing-link", "memory-link-template"}:
        parameters = {}
    elif defect in {"negative-link", "memory-negative-link"}:
        parameters["nvlink_link_id"] = -1
    elif defect == "pci-type":
        parameters["pci_bdf"] = 123
    gpus = (
        []
        if defect == "zero-gpus"
        else ["GPU-a", "GPU-b"]
        if defect == "two-gpus"
        else ["GPU-a"]
    )
    result = agent.execute(
        envelope(command(operation, parameters=parameters, gpu_uuids=gpus))
    )
    assert result.status is NodeActionStatus.FAILED
    assert message in (result.error or "")
    assert runner.commands == []


@pytest.mark.parametrize("error_text", [None, "", "x" * 2200])
def test_field_diagnostic_executes_rendered_target_only_after_client_validation(
    node_factory: Callable[..., NodeActionExecutor],
    tmp_path: Path,
    error_text: str | None,
) -> None:
    path, digest = executable(tmp_path)

    class Runner(FakeRunner):
        def __call__(
            self, arguments: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            if arguments[0] != str(path):
                return super().__call__(arguments, **kwargs)
            self.commands.append(arguments)
            if error_text is not None:
                raise subprocess.CalledProcessError(4, arguments, stderr=error_text)
            return subprocess.CompletedProcess(
                arguments, 0, "\nErrors: 0\nOverall Result: PASS\n", "healthy"
            )

    runner = Runner()
    agent = node_factory(
        allowed_operations={WorkflowOperation.RUN_NVLINK74_WORKFLOW},
        field_diagnostic_enabled=True,
        field_diagnostic_command=(
            str(path),
            "--link={link_id}",
            "--gpu={gpu_uuid}",
            "--bdf={pci_bdf}",
        ),
        field_diagnostic_sha256=digest,
        runner=runner,
    )
    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.RUN_NVLINK74_WORKFLOW,
                parameters={"nvlink_link_id": 7, "pci_bdf": "0000:01:00.0"},
            )
        )
    )
    assert runner.commands[-1] == [
        str(path),
        "--link=7",
        "--gpu=GPU-a",
        "--bdf=0000:01:00.0",
    ]
    assert any(arguments[0] == "nvidia-smi" for arguments in runner.commands[:-1]), (
        "field diagnostic must verify GPU clients before the pinned executable"
    )
    if error_text is None:
        assert result.status is NodeActionStatus.SUCCEEDED
        assert result.details["field_diagnostic"] == "PASSED"
        assert result.details["command_sha256"] == digest
        assert result.details["nvlink_link_id"] == 7
    else:
        assert result.status is NodeActionStatus.FAILED
        assert "exited with status 4" in (result.error or "")
        assert len(result.error or "") < 2200


@pytest.mark.parametrize("mode", ["plain", "gzip", "stdout", "timeout", "missing"])
def test_bundle_keeps_partial_capture_evidence_and_cleans_its_staging_directory(
    node_factory: Callable[..., NodeActionExecutor], tmp_path: Path, mode: str
) -> None:
    requests = []

    def runner(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        requests.append(arguments)
        if arguments[0] == "/usr/bin/nvidia-bug-report.sh":
            destination = Path(arguments[arguments.index("--output-file") + 1])
            if mode == "plain":
                destination.write_text("unit bug report", encoding="ascii")
            elif mode == "gzip":
                with gzip.open(str(destination) + ".gz", "wt") as stream:
                    stream.write("unit bug report")
            elif mode == "timeout":
                raise subprocess.TimeoutExpired(arguments, 600)
            elif mode == "missing":
                raise FileNotFoundError("synthetic missing tool")
            return subprocess.CompletedProcess(arguments, 0, "unit bug report", "")
        return subprocess.CompletedProcess(
            arguments,
            1 if arguments[:2] == ["nvidia-smi", "-q"] else 0,
            "capture",
            "unit stderr",
        )

    request = command(
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
        parameters={"diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED"},
    )
    key = hashlib.sha256(request.command_id.encode()).hexdigest()[:24]
    old = tmp_path / "diagnostics" / key
    old.mkdir(parents=True)
    (old / "stale").write_text("owned stale staging file", encoding="ascii")
    agent = node_factory(allowed_operations={request.operation}, runner=runner)
    result = agent.execute(envelope(request))
    assert result.status is NodeActionStatus.SUCCEEDED
    archive = Path(result.details["evidence_ref"].removeprefix("file://"))
    assert result.details["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert not old.exists(), "bundle staging directory must be removed"
    assert list(archive.parent.glob("*.partial")) == []
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        assert "diagnostics/stale" not in names
        stream = bundle.extractfile("diagnostics/manifest.json")
        assert stream is not None
        manifest = json.load(stream)
        report = bundle.extractfile("diagnostics/nvidia-bug-report.log.gz")
        assert report is not None
        text = gzip.decompress(report.read()).decode()
    assert len(manifest["captures"]) == 9
    assert result.details["manifest_summary"]["failed_capture_count"] == (
        2 if mode in {"timeout", "missing"} else 1
    )
    assert (
        "TimeoutExpired" in text
        if mode == "timeout"
        else "FileNotFoundError" in text
        if mode == "missing"
        else "unit bug report" in text
    )
    assert ["rdma", "link", "show"] in requests


@pytest.mark.parametrize(
    "operation",
    [
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
        WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
    ],
)
@pytest.mark.parametrize(
    "destination",
    ["s3://unit-bucket/prefix", "s3://unit-bucket", "https://invalid.local", "s3://"],
)
def test_diagnostic_upload_is_bound_to_valid_s3_bucket_and_node_path(
    node_factory: Callable[..., NodeActionExecutor],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: WorkflowOperation,
    destination: str,
) -> None:
    uploads = []
    monkeypatch.setattr(
        boto3,
        "client",
        lambda service: SimpleNamespace(
            upload_file=lambda *args: uploads.append((service, args))
        ),
    )
    agent = node_factory(
        allowed_operations={operation},
        diagnostic_s3_uri=destination,
        runner=lambda arguments, **kwargs: subprocess.CompletedProcess(
            arguments, 0, '{"status":"PASS"}', ""
        ),
    )
    result = agent.execute(envelope(command(operation)))
    if destination.startswith("s3://unit-bucket"):
        assert result.status is NodeActionStatus.SUCCEEDED
        assert len(uploads) == 1
        service, (local, bucket, key) = uploads[0]
        assert service == "s3" and bucket == "unit-bucket"
        assert Path(local).parent == tmp_path / "diagnostics"
        assert (
            key
            == ("prefix/" if destination.endswith("prefix") else "")
            + "node-a/"
            + Path(local).name
        )
        assert result.details["evidence_ref"] == f"s3://unit-bucket/{key}"
    else:
        assert result.status is NodeActionStatus.FAILED
        assert "invalid diagnostic S3 URI" in (result.error or "")
        assert uploads == []


@pytest.mark.parametrize("output", [None, "not-json", "[]"])
def test_dcgm_unparseable_or_empty_results_remain_inconclusive(
    node_factory: Callable[..., NodeActionExecutor], output: str | None
) -> None:
    agent = node_factory(
        allowed_operations={WorkflowOperation.RUN_DCGM_DIAGNOSTIC},
        runner=lambda arguments, **kwargs: subprocess.CompletedProcess(
            arguments, 0, output, "unit stderr"
        ),
    )
    result = agent.execute(envelope(command(WorkflowOperation.RUN_DCGM_DIAGNOSTIC)))
    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["diagnostic_outcome"] == "INCONCLUSIVE"
    assert bool(result.details["parse_error"]) is (output != "[]")
    assert result.details["status_counts"] == {}


def test_excessive_dcgm_depth_fails_without_dispatching_a_recovery(
    node_factory: Callable[..., NodeActionExecutor],
) -> None:
    payload: dict[str, Any] = {"status": "PASS"}
    for _ in range(66):
        payload = {"child": payload}
    calls = []

    def runner(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, json.dumps(payload), "")

    agent = node_factory(
        allowed_operations={WorkflowOperation.RUN_DCGM_DIAGNOSTIC}, runner=runner
    )
    result = agent.execute(envelope(command(WorkflowOperation.RUN_DCGM_DIAGNOSTIC)))
    assert result.status is NodeActionStatus.FAILED
    assert "maximum depth" in (result.error or "")
    assert calls == [["dcgmi", "diag", "-r", "1", "-j"]]


def test_bundle_archive_failure_cleans_partial_files(
    node_factory: Callable[..., NodeActionExecutor],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tarfile.open

    def failed(path: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if mode == "w:gz":
            Path(path).write_bytes(b"incomplete")
            raise OSError("synthetic archive write failure")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(tarfile, "open", failed)
    agent = node_factory(
        allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE},
        runner=lambda arguments, **kwargs: subprocess.CompletedProcess(
            arguments, 0, "", ""
        ),
    )
    result = agent.execute(
        envelope(command(WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE))
    )
    assert result.status is NodeActionStatus.FAILED
    assert "archive write failure" in (result.error or "")
    assert list((tmp_path / "diagnostics").iterdir()) == []
