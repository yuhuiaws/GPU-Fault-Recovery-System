from __future__ import annotations

import base64
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import blast_acceptance_base as base
from scripts.e2e.regional import blast_acceptance_cases_1 as one
from scripts.e2e.regional import run_blast_acceptance as entry
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional.test_blast_acceptance_review import evidence, make_runner
from tests.regional.test_multi_cluster_fixture_review import pod_document


@pytest.mark.parametrize("valid_json", [True, False])
def test_blast_command_transport_preserves_check_and_timeout(
    monkeypatch: pytest.MonkeyPatch, valid_json: bool
) -> None:
    calls = []

    def run(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args, 0, '{"ok":true}' if valid_json else "[", ""
        )

    monkeypatch.setattr(base, "run_fixture_command", run)
    assert base.command(["probe"], check=False, input_text="payload").returncode == 0
    assert calls[0] == (
        ["probe"],
        {"check": False, "input_text": "payload", "timeout": 180},
    )
    if valid_json:
        assert base.json_command(["probe"]) == {"ok": True}
    else:
        with pytest.raises(base.CheckError, match="invalid JSON"):
            base.json_command(["probe"])


def test_blast_transport_uses_explicit_cpu_gpu_context_and_region(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    calls = []
    monkeypatch.setattr(
        base, "json_command", lambda args: calls.append(args) or {"ok": True}
    )
    monkeypatch.setattr(
        base,
        "command",
        lambda args, **kwargs: calls.append((args, kwargs))
        or SimpleNamespace(stdout="ok"),
    )
    target = runner.targets[0]
    assert runner.aws("sts", "get-caller-identity") == {"ok": True}
    assert base.BlastRunnerBase.cpu_json(runner, "get", "nodes") == {"ok": True}
    assert runner.gpu_json(target, "get", "nodes") == {"ok": True}
    assert runner.cpu_text("get", "nodes", check=False) == "ok"
    assert runner.gpu_text(target, "get", "nodes", check=False) == "ok"
    assert calls == [
        (
            "aws",
            "sts",
            "get-caller-identity",
            "--region",
            runner.region,
            "--output",
            "json",
        ),
        ("kubectl", "--kubeconfig", runner.cpu_kubeconfig, "get", "nodes"),
        (
            "kubectl",
            "--kubeconfig",
            runner.gpu_kubeconfig,
            "--context",
            target.context,
            "get",
            "nodes",
        ),
        (
            ("kubectl", "--kubeconfig", runner.cpu_kubeconfig, "get", "nodes"),
            {"check": False},
        ),
        (
            (
                "kubectl",
                "--kubeconfig",
                runner.gpu_kubeconfig,
                "--context",
                target.context,
                "get",
                "nodes",
            ),
            {"check": False},
        ),
    ]


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
@pytest.mark.parametrize("ready", [True, False])
def test_blast_pod_selection_requires_a_ready_pod(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plane: str, ready: bool
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    document = pod_document() if ready else {"items": []}
    monkeypatch.setattr(runner, plane + "_json", lambda *args: document)
    select = (
        runner.ready_cpu_pod
        if plane == "cpu"
        else lambda: runner.ready_executor_pod(runner.targets[0])
    )
    if ready:
        assert select() == "api"
    else:
        with pytest.raises(base.CheckError, match="no Ready"):
            select()


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"data": {"state.json": "["}},
        {"data": {"state.json": "[]"}},
        {"data": {"state.json": '{"release_id": 1}'}},
    ],
)
def test_blast_release_read_must_be_valid_and_typed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, document: dict[str, Any]
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    monkeypatch.setattr(runner, "cpu_json", lambda *args: document)
    with pytest.raises(base.CheckError, match="release"):
        runner.evidence_identity()


@pytest.mark.parametrize("defect", ["none", "singular", "base64", "endpoint"])
def test_blast_kubeconfig_proves_ca_endpoint_and_cluster_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    ca = base64.b64encode(b"example-public-ca").decode()
    view = {
        "contexts": [{"name": "context", "context": {"cluster": "eks"}}],
        "clusters": [{"name": "eks", "cluster": {"server": "https://eks.invalid"}}],
    }
    description = {
        "cluster": {
            "arn": runner.cpu_eks_arn,
            "endpoint": "https://eks.invalid",
            "certificateAuthority": {"data": ca},
        }
    }
    if defect == "singular":
        view["contexts"] = []
    elif defect == "endpoint":
        description["cluster"]["endpoint"] = "https://other.invalid"
    monkeypatch.setattr(base, "json_command", lambda *args: view)
    monkeypatch.setattr(
        base,
        "command",
        lambda *args: SimpleNamespace(stdout="!" if defect == "base64" else ca),
    )
    if defect == "none":
        result = runner.kubeconfig_binding(
            kube_args=("--context", "context"),
            expected_arn=runner.cpu_eks_arn,
            eks_description=description,
        )
        assert result == {
            "expected_arn": runner.cpu_eks_arn,
            "context_name": "context",
            "context_cluster_name": "eks",
            "endpoint_matches": True,
            "ca_matches": True,
        }
    else:
        with pytest.raises(base.CheckError):
            runner.kubeconfig_binding(
                kube_args=("--context", "context"),
                expected_arn=runner.cpu_eks_arn,
                eks_description=description,
            )


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "account",
        "cpu-ready",
        "gpu-ready",
        "orchestrator",
        "recovery",
        "site",
        "cpu-kubeconfig",
        "gpu-kubeconfig",
        "source",
    ],
)
def test_blast_preflight_checks_physical_binding_before_publishing_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    calls = []

    def aws(*args: str) -> dict[str, Any]:
        calls.append(args)
        if args[0] == "sts":
            if defect == "source":
                monkeypatch.setattr(base, "source_digest", lambda: "b" * 64)
            elif defect == "site":
                runner.site_path.write_text("changed site", encoding="ascii")
            elif defect.endswith("-kubeconfig"):
                path = Path(getattr(runner, defect.replace("-", "_")))
                path.write_text("changed connection fixture", encoding="ascii")
            return {"Account": "wrong" if defect == "account" else "000000000000"}
        if args[0] == "eks":
            return {"cluster": {"name": args[-1]}}
        assert args[:2] == ("sagemaker", "describe-cluster")
        return {
            "Orchestrator": {
                "Eks": {
                    "ClusterArn": "other"
                    if defect == "orchestrator"
                    else runner.targets[0].eks_cluster_arn
                }
            },
            "NodeRecovery": "Automatic" if defect == "recovery" else "None",
        }

    monkeypatch.setattr(runner, "aws", aws)
    bindings = []
    monkeypatch.setattr(
        runner,
        "kubeconfig_binding",
        lambda **kwargs: bindings.append(kwargs) or {"ca_matches": True},
    )
    monkeypatch.setattr(
        runner,
        "cpu_text",
        lambda *args: "unavailable" if defect == "cpu-ready" else "ok",
    )
    monkeypatch.setattr(
        runner,
        "gpu_text",
        lambda *args: "unavailable" if defect == "gpu-ready" else "ok",
    )
    monkeypatch.setattr(
        runner,
        "gpu_json",
        lambda *args: {"items": [{"metadata": {"name": "gpu-node"}}]},
    )
    original = runner.cpu_json
    monkeypatch.setattr(
        runner,
        "cpu_json",
        lambda *args: {"items": [{"metadata": {"name": "cpu-node"}}]}
        if "nodes" in args
        else original(*args),
    )
    if defect != "none":
        with pytest.raises(base.CheckError):
            base.BlastRunnerBase.preflight(runner)
        assert not (runner.root_run_dir / base.PREFLIGHT_CACHE_NAME).exists(), (
            "failed preflight published a reusable scope"
        )
        return
    base.BlastRunnerBase.preflight(runner)
    cached = json.loads((runner.root_run_dir / base.PREFLIGHT_CACHE_NAME).read_text())
    assert cached["cpu"]["node_names"] == ["cpu-node"]
    assert cached["gpu_clusters"][0]["node_recovery"] == "None"
    assert cached["gpu_clusters"][0]["node_names"] == ["gpu-node"]
    assert bindings[0]["expected_arn"] == runner.cpu_eks_arn
    assert bindings[1]["expected_arn"] == runner.targets[0].eks_cluster_arn
    count = len(calls)
    base.BlastRunnerBase.preflight(runner)
    assert len(calls) == count, "bound recent cache must avoid repeated provider reads"
    assert json.loads((runner.run_dir / "execution-scope.json").read_text())[
        "reused_from"
    ] == str(runner.root_run_dir / base.PREFLIGHT_CACHE_NAME)


@pytest.mark.parametrize(
    "raw",
    [
        "[",
        "[]",
        "{}",
        '{"captured_at":"invalid"}',
        '{"captured_at":"2026-01-01T00:00:00"}',
    ],
)
def test_blast_preflight_cache_rejects_unparseable_and_naive_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: str
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    (runner.root_run_dir / base.PREFLIGHT_CACHE_NAME).write_text(raw)
    assert runner.reusable_preflight() is None


def test_future_blast_preflight_cannot_authorize_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    base.write_json(
        runner.root_run_dir / base.PREFLIGHT_CACHE_NAME,
        {"captured_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()},
    )
    assert runner.reusable_preflight() is None
    runner.predecessor = {"valid": False}
    called = []
    monkeypatch.setattr(runner, "preflight", lambda: called.append("preflight"))
    assert runner.run() == 1
    assert called == []
    assert evidence(runner)["verdict"] == "FAIL"


@pytest.mark.parametrize("namespace", [None, "training"])
@pytest.mark.parametrize(
    ("code", "answer"), [(0, True), (0, False), (0, "no"), (2, False), (0, None)]
)
def test_permission_matrix_accepts_only_explicit_authorization_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    namespace: str | None,
    code: int,
    answer: object,
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    calls = []

    def review(args, **kwargs):
        calls.append((args, kwargs))
        request = json.loads(kwargs["input_text"])
        return SimpleNamespace(
            stdout=json.dumps({**request, "status": {"allowed": answer}}),
            returncode=code,
        )

    monkeypatch.setattr(one, "command", review)
    kwargs = {
        "kube_prefix": ("--context", "context"),
        "service_account": "system:serviceaccount:training:executor",
        "verbs": ("get",),
        "resources": ("pods",),
        "namespace": namespace,
    }
    if code == 0 and type(answer) is bool:
        assert runner.auth_can_i(**kwargs) == {"get": {"pods": answer}}
    else:
        with pytest.raises(base.CheckError, match="authorization review"):
            runner.auth_can_i(**kwargs)
    assert calls[0][1]["check"] is False
    assert "--as=system:serviceaccount:training:executor" in calls[0][0]
    assert calls[0][0][-3:] == (
        "--raw=/apis/authorization.k8s.io/v1/selfsubjectaccessreviews",
        "-f",
        "-",
    )
    assert json.loads(calls[0][1]["input_text"])["spec"]["resourceAttributes"] == {
        "verb": "get",
        "resource": "pods",
        "group": "",
        "namespace": namespace or "",
    }


@pytest.mark.parametrize(
    "items",
    [None, [], [{}], [{"metadata": {"name": "a"}}, {"metadata": {"name": "a"}}]],
)
def test_permission_inventory_cannot_succeed_with_unknown_namespaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, items: Any
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    monkeypatch.setattr(runner, "cpu_json", lambda *args: {"items": items})
    with pytest.raises(base.CheckError, match="namespace permission inventory"):
        runner.namespace_names()


def test_iam_inventory_reads_inline_and_attached_current_policy_versions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    statement = {
        "Effect": "Allow",
        "Action": ["sagemaker:DescribeCluster"],
        "Resource": "bound-resource",
    }
    calls = []

    def aws(*args: str) -> dict[str, Any]:
        calls.append(args)
        return {
            "list-role-policies": {"PolicyNames": ["inline"]},
            "list-attached-role-policies": {
                "AttachedPolicies": [
                    {"PolicyArn": "policy-arn", "PolicyName": "attached"}
                ]
            },
            "get-role-policy": {"PolicyDocument": {"Statement": statement}},
            "get-policy": {"Policy": {"DefaultVersionId": "v3"}},
            "get-policy-version": {
                "PolicyVersion": {"Document": {"Statement": [statement]}}
            },
        }[args[1]]

    monkeypatch.setattr(runner, "aws", aws)
    statements, inventory = runner.iam_role_policies(
        "arn:aws:iam::000000000000:role/path/executor"
    )
    assert statements == [statement, statement]
    assert inventory["role_name"] == "executor"
    assert [item["kind"] for item in inventory["policies"]] == ["inline", "attached"]
    assert inventory["policies"][1]["version_id"] == "v3"
    assert calls[-1][-2:] == ("--version-id", "v3")
    assert "bound-resource" not in json.dumps(inventory), (
        "policy resources are represented by hashes"
    )


@pytest.mark.parametrize(
    "defect",
    ["none", "zero", "two", "role", "namespace", "serviceAccount", "clusterName"],
)
def test_cpu_pod_identity_association_is_unique_and_scope_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    association = {
        "roleArn": "role-arn",
        "namespace": runner.namespace,
        "serviceAccount": "gpu-fault-control-plane",
        "clusterName": runner.cpu_cluster_name,
    }
    if defect == "role":
        association["roleArn"] = ""
    elif defect in association:
        association[defect] = "foreign"
    count = 0 if defect == "zero" else 2 if defect == "two" else 1
    monkeypatch.setattr(
        runner,
        "aws",
        lambda *args: {"associations": [{"associationId": "id"}] * count}
        if args[1] == "list-pod-identity-associations"
        else {"association": association},
    )
    if defect == "none":
        arn, detail = runner.cpu_control_plane_role_arn()
        assert arn == "role-arn"
        assert detail["association_id"] == "id"
    else:
        with pytest.raises(base.CheckError, match="association|ServiceAccount"):
            runner.cpu_control_plane_role_arn()


@pytest.mark.parametrize("predecessor_required", [True, False])
@pytest.mark.parametrize("status", [0, 1])
def test_blast_entry_binds_predecessor_before_case_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    predecessor_required: bool,
    status: int,
) -> None:
    site = tmp_path / "site"
    site.write_text("synthetic", encoding="ascii")
    argv = [
        "blast",
        "--case",
        base.CASE_IDS[1],
        "--site",
        str(site),
        "--run-dir",
        str(tmp_path),
        "--preflight-reuse-seconds",
        "10",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    events = []
    runners = []

    class Runner:
        def __init__(self, **kwargs: Any) -> None:
            events.append("construct")
            self.predecessor = kwargs["predecessor"]
            self.arguments = kwargs
            runners.append(self)

        def evidence_identity(self) -> dict[str, str]:
            events.append("identity")
            return {"release_id": "unit-release"}

        def run(self) -> int:
            events.append("run")
            assert self.predecessor["valid"] is True
            return status

    monkeypatch.setattr(entry, "Runner", Runner)
    monkeypatch.setattr(
        entry,
        "predecessor_path",
        lambda *args: ("previous", tmp_path / "previous")
        if predecessor_required
        else (None, None),
    )

    def predecessor(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("predecessor")
        assert kwargs == {"release_id": "unit-release"}
        return {"valid": True, "case_id": "previous"}

    monkeypatch.setattr(entry, "predecessor_evidence", predecessor)
    assert entry.main() == status
    assert events == [
        "construct",
        "identity",
        *(["predecessor"] if predecessor_required else []),
        "run",
    ]
    assert runners[0].arguments["preflight_reuse_seconds"] == 10


@pytest.mark.parametrize(
    "defect",
    ["missing-options", "missing-site", "missing-evidence", "complete-evidence"],
)
def test_blast_entry_refuses_missing_inputs_before_constructing_a_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    site = tmp_path / "site"
    if defect != "missing-site":
        site.write_text("synthetic", encoding="ascii")
    argv = ["blast", "--case", base.CASE_IDS[0]]
    if defect != "missing-options":
        argv.extend(["--site", str(site), "--run-dir", str(tmp_path)])
    monkeypatch.setattr(sys, "argv", argv)
    constructed = []
    monkeypatch.setattr(entry, "Runner", lambda **kwargs: constructed.append(kwargs))
    if defect == "complete-evidence":
        directory = base.default_e2e_dir(tmp_path)
        directory.mkdir(parents=True)
        for name in (
            base.E2E001_CPU_NODES_BEFORE,
            base.E2E001_EXECUTION_CARD,
            base.E2E001_CONTROL_PLANE_STATE,
        ):
            (directory / name).write_text("{}", encoding="ascii")
        monkeypatch.setattr(
            entry,
            "predecessor_path",
            lambda *args: (_ for _ in ()).throw(
                RuntimeError("stop at bound predecessor")
            ),
        )
        with pytest.raises(RuntimeError, match="bound predecessor"):
            entry.main()
    else:
        with pytest.raises(SystemExit, match="required|does not exist|incomplete"):
            entry.main()
    assert constructed == []


@pytest.mark.parametrize("value", ["broken", "not:an:arn:a:b:c"])
def test_malformed_arn_is_not_a_valid_scope(value: str) -> None:
    with pytest.raises(base.CheckError, match="invalid ARN"):
        base.arn_parts(value)


def test_naive_cpu_event_time_is_not_an_auditable_window() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(base.CheckError, match="no timezone"):
        base.observed_in_windows("2026-01-01T00:00:00", [(now, now)])


@pytest.mark.parametrize(
    ("statement", "action", "expected"),
    [
        ({"Effect": "Deny", "Action": "*"}, "sagemaker:DescribeCluster", False),
        (
            {"Effect": "Allow", "NotAction": "sagemaker:Describe*"},
            "sagemaker:DescribeCluster",
            False,
        ),
        (
            {"Effect": "Allow", "NotAction": ["sagemaker:Describe*"]},
            "sagemaker:BatchReplaceClusterNodes",
            True,
        ),
        ({"Effect": "Allow"}, "sagemaker:DescribeCluster", False),
    ],
)
def test_notaction_and_deny_statements_do_not_hide_mutation_permissions(
    statement: dict[str, Any], action: str, expected: bool
) -> None:
    assert base.allow_statement_matches(statement, action) is expected
