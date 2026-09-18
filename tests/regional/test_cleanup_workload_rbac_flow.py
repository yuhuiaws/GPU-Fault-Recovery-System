from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from gpu_fault_release.regional_deployment_inventory import (
    GPU_EXECUTOR_DEPLOYMENT,
    GPU_WATCHER_DEPLOYMENT,
)
from gpu_fault_release.regional_release_gpu_rollout import (
    EXECUTOR_SERVICE_ACCOUNT,
    WATCHER_SERVICE_ACCOUNT,
    WORKLOAD_NAMESPACE_RBAC_LABEL,
    render_workload_namespace_rbac,
)
from tests.regional.test_clean_redeploy_script import _reset_harness, run_script

NAMESPACE = "gpu-fault-system"
CONTEXT = "gpu-a-context"


def rbac_harness(tmp_path: Path, *, interrupt: bool = False):
    harness = _reset_harness(tmp_path)
    config = json.loads(harness.config.read_text())
    kubeconfig = tmp_path / "gpu.kubeconfig"
    kubeconfig.write_text("fixture", encoding="utf-8")
    config["gpu_kubeconfig"] = str(kubeconfig)
    target = config["clusters"][0]
    target.update(
        eks_cluster_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        executor_irsa_role_arn="arn:aws:iam::123456789012:role/executor",
        allowed_namespaces=["training"],
    )
    harness.config.write_text(json.dumps(config), encoding="utf-8")
    api = json.loads(harness.api.read_text())
    api["interrupt_node"] = False
    api["interrupt_rbac"] = interrupt

    def put(kind, name, namespace="", **fields):
        identity = json.dumps([CONTEXT, namespace, kind.lower(), name])
        item = {
            "apiVersion": "v1",
            "kind": kind,
            "metadata": {
                "name": name,
                "uid": f"uid-{namespace}-{kind}-{name}",
                "resourceVersion": "1",
                **({"namespace": namespace} if namespace else {}),
            },
            **fields,
        }
        api["objects"][identity] = item
        return item

    put("Namespace", "training")
    for account, deployment in (
        (EXECUTOR_SERVICE_ACCOUNT, GPU_EXECUTOR_DEPLOYMENT),
        (WATCHER_SERVICE_ACCOUNT, GPU_WATCHER_DEPLOYMENT),
    ):
        sa = put("ServiceAccount", account, NAMESPACE)
        if account == EXECUTOR_SERVICE_ACCOUNT:
            sa["metadata"]["annotations"] = {
                "eks.amazonaws.com/role-arn": target["executor_irsa_role_arn"]
            }
        key = json.dumps([CONTEXT, NAMESPACE, "deployment", deployment])
        api["objects"][key]["spec"]["template"] = {
            "spec": {"serviceAccountName": account}
        }
    for group in render_workload_namespace_rbac(
        target["allowed_namespaces"], system_namespace=NAMESPACE
    ).values():
        for expected in group:
            item = deepcopy(expected)
            metadata = item["metadata"]
            key = json.dumps(
                [CONTEXT, metadata["namespace"], item["kind"].lower(), metadata["name"]]
            )
            metadata.update(uid="owned-" + key, resourceVersion="1")
            api["objects"][key] = item
    manual = put(
        "Role",
        "operator-custom",
        "training",
        apiVersion="rbac.authorization.k8s.io/v1",
        rules=[{"apiGroups": [""], "resources": ["configmaps"], "verbs": ["get"]}],
    )
    manual["metadata"]["labels"] = {WORKLOAD_NAMESPACE_RBAC_LABEL: "true"}
    harness.api.write_text(json.dumps(api), encoding="utf-8")
    return harness


def invoke(harness, scope):
    if scope == "all":
        return harness.run()
    return run_script(
        "--config",
        str(harness.config),
        "--scope",
        "gpu",
        "--cluster-id",
        "gpu-a",
        "--mode",
        "clean",
        "--node-mode",
        "skip",
        "--state-file",
        str(harness.state),
        "--timeout-seconds",
        "30",
        "--execute",
        env=harness.env,
    )


@pytest.mark.parametrize("scope,interrupt", [("all", False), ("gpu", True)])
def test_guarded_rbac_precedes_anchor_deletion_and_resumes_the_original_journal(
    tmp_path, scope, interrupt
):
    harness = rbac_harness(tmp_path, interrupt=interrupt)
    result = invoke(harness, scope)
    original = None
    if interrupt:
        assert result.returncode != 0, "the RBAC interruption was not exercised"
        original = json.loads(harness.state.read_text())
        assert (original["phase"], original["status"]) == (
            "APPLICATION_OBJECTS_DELETED",
            "FAILED",
        ), "an interrupted guarded deletion must retain its application phase"
        assert len(original["workload_rbac_removed"]["gpu:" + CONTEXT]) == 1, (
            "the first confirmed grant removal must be durable before interruption"
        )
        api = json.loads(harness.api.read_text())
        assert all(
            json.dumps([CONTEXT, NAMESPACE, "serviceaccount", name]) in api["objects"]
            for name in (EXECUTOR_SERVICE_ACCOUNT, WATCHER_SERVICE_ACCOUNT)
        ), "a failed guarded deletion must retain its ownership anchors"
        result = invoke(harness, scope)
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads(harness.state.read_text())
    assert state["schema_version"] == 2, (
        "RBAC integration must preserve schema2 resumption"
    )
    assert (state["phase"], state["status"]) == ("CLEANUP_COMPLETED", "COMPLETED"), (
        "all cleanup phases must finish after guarded removal"
    )
    section = state["inventory_snapshot"]["gpu"]["by_context"][CONTEXT]
    guarded = [row for row in section["resources"] if row.get("guarded_delete")]
    assert guarded and section["workload_rbac"]["removed"] == [], (
        "guarded rows and their original proof must remain in captured evidence"
    )
    assert len(state["workload_rbac_removed"]["gpu:" + CONTEXT]) == len(guarded), (
        "the journal must record every guarded removal"
    )
    if original is not None:
        assert state["inventory_snapshot"] == original["inventory_snapshot"], (
            "resume must not replace its captured ownership proof"
        )
        assert state["run_id"] == original["run_id"], (
            "resume must use the original cleanup run"
        )
    calls = harness.calls()
    rbac = [
        index
        for index, call in enumerate(calls)
        if "delete" in call["args"]
        and "--raw" in call["args"]
        and "/rbac.authorization.k8s.io/"
        in call["args"][call["args"].index("--raw") + 1]
    ]
    ordinary = [
        index
        for index, call in enumerate(calls)
        if call["phase"] == "APPLICATION_OBJECTS_DELETED"
        and "delete" in call["args"]
        and "--raw" not in call["args"]
    ]
    assert rbac and ordinary and max(rbac) < min(ordinary), (
        "all guarded RBAC deletion must precede ordinary application or anchor deletion"
    )
    for index in rbac:
        call = calls[index]
        assert {"QUEUES_DRAINED", "GPU_EXECUTORS_STOPPED"}.issubset(
            call["completed"]
        ), "RBAC deletion requires completed drain and Executor stop checkpoints"
        assert set(call["payload"]["preconditions"]) == {"uid", "resourceVersion"}, (
            "guarded deletion must retain both UID and resourceVersion preconditions"
        )
    for index in ordinary:
        arguments = calls[index]["args"]
        verb = arguments.index("delete")
        assert not any(
            arguments[verb + 1 : verb + 3] == [row["kind"], row["name"]]
            and "-n" in arguments
            and arguments[arguments.index("-n") + 1] == row["namespace"]
            for row in guarded
        ), "a guarded inventory row reached ordinary name-based deletion"
    api = json.loads(harness.api.read_text())
    assert (
        json.dumps([CONTEXT, "training", "role", "operator-custom"]) in api["objects"]
    ), "a matching label alone must never authorize deletion of unrelated RBAC"
    if scope == "all":
        namespaces = [
            index
            for index, call in enumerate(calls)
            if "delete" in call["args"]
            and "--raw" in call["args"]
            and call["args"][call["args"].index("--raw") + 1].startswith(
                "/api/v1/namespaces/"
            )
        ]
        assert namespaces and min(namespaces) > max(rbac), (
            "namespace deletion must wait for all guarded workload grants"
        )
