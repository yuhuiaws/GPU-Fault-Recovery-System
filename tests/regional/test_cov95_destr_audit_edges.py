from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import audit_destr011_provider_replace as retired
from scripts.e2e.regional import audit_destr013_replacement_invariant as audit
from scripts.e2e.regional import regional_commands
from tests.regional.test_destr013_audit_integration import LocalAudit, write_json


class RetiredAudit:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.run_dir = tmp_path
        self.kubeconfig = tmp_path / "gpu.config"
        self.kubeconfig.write_text("unit fake transport\n", encoding="utf-8")
        self.pods = ["executor-a", "executor-b"]
        self.environment = {"allow_replace": "false", "allow_reboot": "true"}
        self.replace_decision = "implicitDeny"
        self.reboot_decision = "allowed"
        self.events: list[dict[str, Any]] = []
        self.inventory_drift = False
        self.inventory_reads = 0
        self.calls: list[list[str]] = []
        self.failure = False
        for name in (
            "AWS_REGION",
            "GPU_CONTEXT",
            "GPU_KUBECONFIG",
            "HYPERPOD_CLUSTER",
            "NAMESPACE",
            "ROLE_ARN",
        ):
            monkeypatch.setattr(retired, name, getattr(retired, name))
        monkeypatch.setattr(
            retired, "subprocess", SimpleNamespace(run=self.run, PIPE=subprocess.PIPE)
        )
        monkeypatch.setattr(
            retired, "os", SimpleNamespace(getenv=os.getenv, umask=lambda _: None)
        )

    def arguments(self) -> argparse.Namespace:
        return argparse.Namespace(
            gpu_kubeconfig=str(self.kubeconfig),
            gpu_context="unit-gpu",
            region="us-west-2",
            executor_role_arn="arn:aws:iam::123456789012:role/executor",
            hyperpod_cluster_name="unit-hp",
            namespace="unit-system",
        )

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        if self.failure:
            return subprocess.CompletedProcess(
                command, 3, "", "fake transport unavailable"
            )
        if "list-cluster-nodes" in command:
            self.inventory_reads += 1
            count = 2 if self.inventory_drift and self.inventory_reads > 1 else 1
            value: Any = {
                "ClusterNodeSummaries": [
                    {
                        "InstanceId": f"i-{index}",
                        "InstanceGroupName": "gpu",
                        "InstanceType": "test",
                    }
                    for index in range(count)
                ]
            }
        elif command[1:3] == ["-m", "pytest"]:
            return subprocess.CompletedProcess(command, 0, "fake tests passed", "")
        elif "get" in command:
            return subprocess.CompletedProcess(command, 0, " ".join(self.pods), "")
        elif "exec" in command:
            value = self.environment
        elif "describe-cluster" in command:
            value = {
                "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/unit"
            }
        elif "simulate-principal-policy" in command:
            value = {
                "EvaluationResults": [
                    {
                        "EvalActionName": "sagemaker:BatchReplaceClusterNodes",
                        "EvalDecision": self.replace_decision,
                    },
                    {
                        "EvalActionName": "sagemaker:BatchRebootClusterNodes",
                        "EvalDecision": self.reboot_decision,
                    },
                ]
            }
        elif "lookup-events" in command:
            value = {"Events": self.events}
        else:
            raise AssertionError(f"unexpected retired audit command: {command[0]}")
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "replicas",
        "replace-env",
        "reboot-env",
        "replace-iam",
        "reboot-iam",
        "event",
        "inventory",
        "transport",
    ],
)
def test_retired_audit_never_satisfies_formal_sequence_and_names_failed_invariants(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = RetiredAudit(tmp_path, monkeypatch)
    if defect == "replicas":
        h.pods = ["executor-a"]
    elif defect == "replace-env":
        h.environment["allow_replace"] = "true"
    elif defect == "reboot-env":
        h.environment["allow_reboot"] = "false"
    elif defect == "replace-iam":
        h.replace_decision = "allowed"
    elif defect == "reboot-iam":
        h.reboot_decision = "implicitDeny"
    elif defect == "event":
        h.events = [
            {
                "EventName": "BatchReplaceClusterNodes",
                "EventTime": "unit",
                "Username": "unit",
            }
        ]
    elif defect == "inventory":
        h.inventory_drift = True
    elif defect == "transport":
        h.failure = True
    retired.configure(h.arguments())
    code = retired.run_case(tmp_path, 1)
    path = tmp_path / "cases" / retired.CASE_ID / f"{retired.CASE_ID}.json"
    report = json.loads(path.read_text())
    assert code == (0 if defect == "none" else 1), report
    assert (
        report["verdict"] == "SUPERSEDED"
        and report["formal_sequence_satisfied"] is False
    ), report
    assert report["diagnostic_verdict"] == ("PASS" if defect == "none" else "FAIL"), (
        report
    )
    if defect != "none":
        assert report.get("errors") or report.get("error"), report


def test_retired_audit_main_binds_explicit_target_and_requires_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = RetiredAudit(tmp_path, monkeypatch)
    args = h.arguments()
    args.gpu_context = ""
    with pytest.raises(RuntimeError, match="required"):
        retired.configure(args)
    args = h.arguments()
    args.gpu_kubeconfig = str(tmp_path / "missing")
    with pytest.raises(RuntimeError, match="does not exist"):
        retired.configure(args)
    args = h.arguments()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unit-retired-audit",
            "--run-dir",
            str(tmp_path),
            "--gpu-kubeconfig",
            str(h.kubeconfig),
            "--gpu-context",
            args.gpu_context,
            "--region",
            args.region,
            "--executor-role-arn",
            args.executor_role_arn,
            "--hyperpod-cluster-name",
            args.hyperpod_cluster_name,
        ],
    )
    assert retired.main() == 0, h.calls
    assert all(
        "unit-gpu" in command for command in h.calls if command[0] == "kubectl"
    ), h.calls


@pytest.mark.parametrize(
    "defect",
    [
        "cluster-state",
        "recovery",
        "replace-iam",
        "reboot-iam",
        "manifest",
        "inventory-empty",
        "inventory-drift",
        "tests",
        "timestamps",
    ],
)
def test_final_audit_refuses_each_unproven_invariant_through_fake_public_transport(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LocalAudit(tmp_path, monkeypatch)
    original = h.command
    inventory_reads = 0
    if defect == "cluster-state":
        h.recovery["ClusterStatus"] = "Creating"
    elif defect == "recovery":
        h.recovery["NodeRecovery"] = "Automatic"
    elif defect == "manifest":
        monkeypatch.setattr(
            audit,
            "manifest_invariants",
            lambda: {"violations": ["unit unsafe manifest"]},
        )
    elif defect == "tests":
        h.pytest_code = 1
    elif defect == "timestamps":

        def append_timestamp(command: list[str]) -> None:
            if command[1:3] == ["-m", "pytest"]:
                write_json(
                    h.run_dir / "cases" / "GF-REGIONAL-DESTR-002" / "new.json",
                    {"started_at": "2026-09-07T10:30:00Z"},
                )

        h.before_command = append_timestamp

    def command(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal inventory_reads
        result = original(argv, **kwargs)
        if argv[1:3] == ["iam", "simulate-principal-policy"] and defect.endswith(
            "-iam"
        ):
            value = json.loads(result.stdout)
            index = 0 if defect == "replace-iam" else 1
            value["EvaluationResults"][index]["EvalDecision"] = (
                "allowed" if index == 0 else "implicitDeny"
            )
            result.stdout = json.dumps(value)
        if argv[1:3] == ["sagemaker", "list-cluster-nodes"]:
            inventory_reads += 1
            if defect == "inventory-empty" or (
                defect == "inventory-drift" and inventory_reads > 1
            ):
                result.stdout = json.dumps({"ClusterNodeSummaries": []})
        return result

    monkeypatch.setattr(regional_commands, "run_command", command)
    code, report = h.run()
    assert code == 1 and report["verdict"] == "FAIL", report
    assert report["formal_sequence_satisfied"] is False, report
    assert report["errors"], report


@pytest.mark.parametrize(
    "defect",
    [
        "inventory",
        "event-type",
        "source-missing",
        "source-other",
        "conflicting-target",
        "bad-page-token",
    ],
)
def test_cloudtrail_pages_require_complete_inventory_and_bound_positive_control(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LocalAudit(tmp_path, monkeypatch)
    name = audit.POSITIVE_CONTROL_EVENT
    if defect == "inventory":
        h.pages = {"": {"Events": None}}
    elif defect == "event-type":
        h.events[name] = ["invalid"]
    elif defect == "source-missing":
        h.events[name] = [h.event(name, eventSource=None)]
    elif defect == "source-other":
        h.events[name].append(h.event(name, eventSource="other.amazonaws.com"))
    elif defect == "conflicting-target":
        h.events[name] = [
            h.event(
                name,
                requestParameters={"clusterName": "managed", "clusterArn": "foreign"},
            )
        ]
    else:
        h.pages = {"": {"Events": [], "NextToken": ""}}
    code, report = h.run()
    if defect == "source-other":
        assert code == 0 and len(report["cloudtrail_events"][name]) == 1, report
    else:
        assert code == 1 and report["verdict"] == "FAIL", report
        assert report["formal_sequence_satisfied"] is False, report


def test_cloudtrail_pagination_is_bounded_even_when_every_token_is_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LocalAudit(tmp_path, monkeypatch)
    original = h.command
    pages = 0

    def command(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal pages
        result = original(argv, **kwargs)
        if argv[1:3] == ["cloudtrail", "lookup-events"]:
            pages += 1
            result.stdout = json.dumps({"Events": [], "NextToken": f"page-{pages}"})
        return result

    monkeypatch.setattr(regional_commands, "run_command", command)
    code, report = h.run()
    assert code == 1 and "pagination exceeds" in report["error"], report
    assert pages == 1000, pages


def test_destructive_timestamp_inventory_rejects_missing_root_and_symlinked_child(
    tmp_path: Path,
) -> None:
    with pytest.raises(audit.AuditError, match="inventory cannot be read"):
        audit.run_timestamps(tmp_path)
    case_dir = tmp_path / "cases" / "GF-REGIONAL-DESTR-002"
    case_dir.mkdir(parents=True)
    target = tmp_path / "other"
    target.mkdir()
    (case_dir / "child").symlink_to(target, target_is_directory=True)
    with pytest.raises(audit.AuditError, match="must not be a symlink"):
        audit.run_timestamps(tmp_path)
