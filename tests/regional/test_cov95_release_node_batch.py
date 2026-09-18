from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.node_installer_rendering import InstallerIdentity, InstallerNode
from gpu_fault_release import regional_node_batch as batch
from gpu_fault_release import rollout
from tests.deploy.test_node_preflight_inputs import inputs

SCOPE = batch.NodeScope(
    "gpu-test",
    "gpu-context",
    "gpu-fault-system",
    "gpu-fault-node-action-keys",
    "gpu-fault-regional-connection",
)
IDENTITY = InstallerIdentity(
    SCOPE.namespace,
    "b" * 64,
    "a" * 64,
    "c" * 64,
    "d" * 64,
    SCOPE.node_action_keys_secret,
    60,
    "http://{node_ip}:9400/metrics",
)


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("extra", "undeclared fields"),
        ("name", "identity differs from its key"),
        ("address", "incomplete identity or addressing"),
        ("uid", "duplicate node UIDs"),
        ("empty", "snapshot is empty"),
    ],
)
def test_private_node_snapshot_rejects_ambiguous_or_incomplete_identity(
    tmp_path: Path, fault: str, problem: str
) -> None:
    document = inputs()
    if fault == "extra":
        document["extra"] = "not-authorized"
    elif fault == "name":
        document["nodes"]["node-a"]["metadata"]["name"] = "other"
    elif fault == "address":
        document["nodes"]["node-a"]["status"]["addresses"] = []
    elif fault == "uid":
        document["nodes"]["node-b"] = copy.deepcopy(document["nodes"]["node-a"])
    else:
        document["nodes"] = {}
    path = tmp_path / "nodes.json"
    path.write_text(json.dumps(document))
    path.chmod(0o600)
    with pytest.raises(ValueError, match=problem):
        batch.read_private_snapshot(path, SCOPE)


@pytest.mark.parametrize("fault", ["output-mode", "namespace", "keys"])
def test_node_batch_refuses_output_or_binding_before_reading_inputs(
    tmp_path: Path, fault: str
) -> None:
    output = tmp_path / "batch"
    mode = 0o755 if fault == "output-mode" else 0o700
    output.mkdir(mode=mode)
    # mkdir applies the caller's umask, including the final harness's private mask.
    output.chmod(mode)
    assert output.stat().st_mode & 0o777 == mode, (
        "the negative fixture must retain its intended permissions under any umask"
    )
    identity = IDENTITY
    if fault == "namespace":
        identity = replace(identity, namespace="other")
    elif fault == "keys":
        identity = replace(identity, node_action_keys_secret="other")
    with pytest.raises(
        ValueError, match="private owned directory|differs from its scope"
    ):
        batch.prepare_node_batch(
            tmp_path / "unread", tmp_path / "unread-template", output, SCOPE, identity
        )
    assert list(output.iterdir()) == [], "rejected scope must not write batch outputs"


def prepared_node(tmp_path: Path) -> batch.PreparedNode:
    install = tmp_path / "install.json"
    preflight = tmp_path / "preflight.json"
    for path in (install, preflight):
        path.write_text(
            json.dumps(
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "metadata": {"name": path.stem},
                }
            )
        )
    return batch.PreparedNode(
        InstallerNode("node-a", "uid-a", "192.0.2.1", "ml.p5.48xlarge"),
        install,
        preflight,
        "preflight-example",
    )


class Preflight:
    def __init__(self, node: batch.PreparedNode) -> None:
        self.calls: list[list[str]] = []
        self.node = node
        self.present = False
        self.fail_host = False
        self.fail_cleanup = False
        self.fail_logs = False
        self.foreign = False

    def run(self, arguments: list[str], **kwargs: Any) -> str:
        self.calls.append(arguments)
        if "--dry-run=server" in arguments:
            assert len(json.loads(kwargs["input_text"])["items"]) == 2
        elif "create" in arguments:
            self.present = True
        elif arguments[0] == "bash" and self.fail_host:
            raise RuntimeError("host proof failed")
        elif "logs" in arguments:
            if self.fail_logs:
                raise RuntimeError("logs unavailable")
            return "fixture diagnostic"
        elif "delete" in arguments:
            if self.fail_cleanup:
                raise RuntimeError("cleanup unavailable")
            assert json.loads(kwargs["input_text"])["preconditions"] == {
                "uid": "job-uid"
            }
            self.present = False
        return ""

    def probe_output(
        self, arguments: list[str], **_kwargs: Any
    ) -> tuple[int, str, str]:
        self.calls.append(arguments)
        if not self.present:
            return 0, "", ""
        return (
            0,
            json.dumps(
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "metadata": {
                        "name": self.node.job_name,
                        "namespace": SCOPE.namespace,
                        "uid": "job-uid",
                        "labels": {
                            "gpu-fault.io/node-preflight": "false"
                            if self.foreign
                            else "true",
                            "gpu-fault.io/node-uid": self.node.node.uid,
                        },
                        "annotations": {
                            "gpu-fault.io/installer-artifact-sha256": IDENTITY.artifact_sha256,
                            "gpu-fault.io/installer-template-sha256": IDENTITY.template_sha256,
                        },
                    },
                }
            ),
            "",
        )


@pytest.mark.parametrize(
    "fault", ["foreign", "cleanup", "host-and-cleanup", "host-and-logs"]
)
def test_node_preflight_failure_preserves_ownership_and_diagnostic_context(
    tmp_path: Path, fault: str
) -> None:
    node = prepared_node(tmp_path)
    runner = Preflight(node)
    runner.foreign = fault == "foreign"
    runner.fail_cleanup = fault in {"cleanup", "host-and-cleanup"}
    runner.fail_host = fault.startswith("host")
    runner.fail_logs = fault == "host-and-logs"
    with pytest.raises(
        (RuntimeError, ValueError),
        match="foreign preflight Job|cleanup unavailable|host proof failed",
    ) as caught:
        batch.run_node_preflights([node], SCOPE, IDENTITY, runner, workers=1)
    notes = getattr(caught.value, "__notes__", [])
    assert any("node preflight failed for node-a" in note for note in notes), (
        "preflight failure diagnostics lost the affected node"
    )
    if fault == "host-and-cleanup":
        assert any("cleanup also failed" in note for note in notes), (
            "host failure must retain the subsequent cleanup failure"
        )
        assert any("fixture diagnostic" in note for note in notes), (
            "host diagnostics must remain attached to the original failure"
        )
    if fault == "foreign":
        assert not any("delete" in arguments for arguments in runner.calls), (
            "foreign preflight Job ownership must prevent deletion"
        )
    assert runner.present is (fault != "host-and-logs")


def test_empty_node_preflight_selection_has_no_transport(tmp_path: Path) -> None:
    runner = Preflight(prepared_node(tmp_path))
    batch.run_node_preflights([], SCOPE, IDENTITY, runner)
    assert runner.calls == []


def cli_arguments(tmp_path: Path) -> list[str]:
    values = {
        "inputs": str(tmp_path / "nodes"),
        "template": str(tmp_path / "template"),
        "output": str(tmp_path / "output"),
        "cluster-id": SCOPE.cluster_id,
        "context": SCOPE.context,
        "namespace": SCOPE.namespace,
        "node-action-keys-secret": SCOPE.node_action_keys_secret,
        "connection-secret": SCOPE.connection_secret,
        "config-digest": IDENTITY.config_digest,
        "artifact-sha256": IDENTITY.artifact_sha256,
        "bundle-sha256": IDENTITY.bundle_sha256,
        "template-sha256": IDENTITY.template_sha256,
        "template-content-sha256": "e" * 64,
        "metrics-url-template": IDENTITY.metrics_url_template,
        "deadline-seconds": str(IDENTITY.deadline_seconds),
    }
    return [
        "node-batch",
        *(part for name, value in values.items() for part in (f"--{name}", value)),
    ]


def test_node_batch_cli_runs_only_injected_preflight_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    node = prepared_node(tmp_path)
    runner = Preflight(node)
    monkeypatch.setattr(batch, "prepare_node_batch", lambda *_args: [node])
    monkeypatch.setattr(rollout, "Runner", lambda: runner)
    monkeypatch.setattr(sys, "argv", [*cli_arguments(tmp_path), "--run-preflights"])
    batch.main()
    assert json.loads(capsys.readouterr().out) == {"node_count": 1, "status": "PASSED"}
    assert runner.present is False
    assert runner.calls[0][0] == "kubectl"
    assert "--dry-run=server" in runner.calls[0]


def test_node_batch_cli_failure_emits_notes_without_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def prepare(*_args: Any) -> list[Any]:
        error = ValueError("invalid scoped input")
        error.add_note("fixture diagnostic")
        raise error

    monkeypatch.setattr(batch, "prepare_node_batch", prepare)
    monkeypatch.setattr(sys, "argv", cli_arguments(tmp_path))
    with pytest.raises(SystemExit, match="node preflight batch failed"):
        batch.main()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "fixture diagnostic"
