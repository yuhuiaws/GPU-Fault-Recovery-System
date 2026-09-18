from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError
from tests._script_loader import lazy_script_module
from tests.admin.test_admin_cluster_removal_rbac import RbacApi
from tests.deploy._cov95_tools_support import TOOLS

KUBE = lazy_script_module(TOOLS / "cleanup_kubernetes.py")
STATE = lazy_script_module(TOOLS / "cleanup_state.py")
COLLECT = lazy_script_module(TOOLS / "collect_installed_resource_registry.py")
NAMESPACE = "gpu-fault-system"
CONTEXT = "gpu:gpu-a-context"


@pytest.fixture
def cleanup(tmp_path: Path, monkeypatch, request):
    options = getattr(request, "param", "all")
    if isinstance(options, str):
        options = {"scope": options}
    scope = options["scope"]
    mode = options.get("mode", "clean")
    target = {
        "cluster_id": "gpu-a",
        "context": "gpu-a-context",
        "eks_cluster_arn": "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        "executor_irsa_role_arn": "arn:aws:iam::123456789012:role/executor",
        "allowed_namespaces": ["training"],
    }
    config = {
        "namespace": NAMESPACE,
        "cpu_kubeconfig": str(tmp_path / "cpu.kubeconfig"),
        "gpu_kubeconfig": str(tmp_path / "gpu.kubeconfig"),
        "clusters": [target],
    }
    api = RbacApi(
        SimpleNamespace(release_config=config, environment={}),
        {**target, "expected_namespace_uid": "gpu-namespace"},
    )
    api.put("Namespace", "", "kube-system", apiVersion="v1")
    for item in api.objects.values():
        if item["kind"] == "Namespace":
            item["metadata"].pop("namespace")
        if item["kind"] == "Deployment":
            item["spec"]["replicas"] = 0
            item["status"] = {"replicas": 0, "readyReplicas": 0, "availableReplicas": 0}
    resources = [
        {
            "kind": "deployment",
            "name": item["metadata"]["name"],
            "namespace": NAMESPACE,
            "scope": "namespaced",
            "phase": "executor"
            if "executor" in item["metadata"]["name"]
            else "producer",
            "clean": "delete",
            "order": 10,
        }
        for item in api.objects.values()
        if item["kind"] == "Deployment"
    ]

    class CollectorTransport:
        def run(self, arguments, *, check=False, timeout_seconds=20):
            return api([*api.kubectl, *arguments], timeout_seconds=timeout_seconds)

    section = {"resources": resources}
    COLLECT.attach_workload_rbac(CollectorTransport(), config, target, section)
    inventory = {
        "schema_version": 1,
        "cpu": {"resources": []},
        "gpu": {
            "resources": copy.deepcopy(section["resources"]),
            "by_context": {target["context"]: section},
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    state_path = tmp_path / "cleanup.json"
    state = STATE.initialize(
        state_path,
        config_path=config_path,
        inventory_path=inventory_path,
        scope=scope,
        mode=mode,
        node_mode="uninstall" if mode == "reset" else "skip",
        cluster_ids=["gpu-a"] if scope == "gpu" else [],
    )
    state["cluster_uids"] = {
        "cpu": "cpu-cluster",
        CONTEXT: api.objects[("Namespace", "", "kube-system")]["metadata"]["uid"],
    }
    state["namespace_snapshots"] = {
        "cpu": {"uid": "cpu-namespace", "objects": []},
        CONTEXT: {"uid": "gpu-namespace", "objects": []},
    }
    for phase in STATE.required_phases(state):
        if STATE.PHASE_INDEX[phase] >= STATE.PHASE_INDEX["APPLICATION_OBJECTS_DELETED"]:
            break
        STATE.transition(
            state, phase=phase, status="COMPLETED", message="fixture stopped"
        )
    STATE.transition(
        state,
        phase="APPLICATION_OBJECTS_DELETED",
        status="IN_PROGRESS",
        message="fixture delete",
    )
    STATE.atomic_write(state_path, state)
    calls = []
    remaining_pods = []

    def run(arguments, **kwargs):
        calls.append(list(arguments))
        if "pods" in arguments:
            return subprocess.CompletedProcess(
                arguments, 0, json.dumps({"items": remaining_pods}), ""
            )
        normalized = [
            item for item in arguments if not item.startswith("--request-timeout=")
        ]
        if "--context" not in normalized:
            assert normalized[:3] == [
                "kubectl",
                "--kubeconfig",
                config["cpu_kubeconfig"],
            ], "CPU stop verification must use the source-config kubeconfig"
            name = normalized[normalized.index("get") + 2]
            uid = "cpu-cluster" if name == "kube-system" else "cpu-namespace"
            return subprocess.CompletedProcess(
                arguments,
                0,
                json.dumps(
                    {
                        "apiVersion": "v1",
                        "kind": "Namespace",
                        "metadata": {"name": name, "uid": uid, "resourceVersion": "1"},
                    }
                ),
                "",
            )
        return api(normalized, **kwargs)

    monkeypatch.setattr(KUBE, "run_command", run)
    value = SimpleNamespace(
        api=api,
        config=config,
        config_path=config_path,
        state_path=state_path,
        inventory=inventory,
        calls=calls,
        remaining_pods=remaining_pods,
    )

    def invoke(
        *,
        context=CONTEXT,
        cluster_id="gpu-a",
        action="delete-workload-rbac",
        skip_cpu=False,
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cleanup_kubernetes",
                "--config",
                str(config_path),
                "--state-file",
                str(state_path),
                "--context",
                context,
                "--cluster-id",
                cluster_id,
                *(["--skip-cpu"] if skip_cpu else []),
                action,
            ],
        )
        return KUBE.main()

    value.invoke = invoke
    return value


@pytest.mark.parametrize("cleanup", ["all", "gpu"], indirect=True)
def test_guarded_cleanup_uses_recorded_uid_proof_and_preserves_inventory(cleanup):
    original = STATE.read_state(cleanup.state_path)
    manual = cleanup.api.add_manual_role()

    assert cleanup.invoke() == 0, "the guarded cleanup CLI must complete successfully"

    saved = STATE.read_state(cleanup.state_path)
    proof = saved["inventory_snapshot"]["gpu"]["by_context"]["gpu-a-context"][
        "workload_rbac"
    ]
    assert saved["inventory_snapshot"] == original["inventory_snapshot"], (
        "deletion progress must not rewrite captured resource authority"
    )
    assert proof["removed"] == [], "the original captured inventory is immutable"
    assert set(saved["workload_rbac_removed"][CONTEXT]) == set(proof["resources"]), (
        "every recorded grant needs a durable removal checkpoint"
    )
    assert manual in cleanup.api.objects, (
        "unrelated labelled RBAC is not delete authority"
    )
    assert all(
        options["preconditions"]["uid"] == proof["resources"][cleanup.api.key_name(key)]
        and options["preconditions"]["resourceVersion"] == "1"
        for key, options in cleanup.api.requests
    ), "each deletion must bind the recorded UID and current resourceVersion"
    assert all(key[0] == "RoleBinding" for key in cleanup.api.deleted[:2]), (
        "RoleBindings must be removed before their referenced Roles"
    )
    before = cleanup.state_path.read_bytes()
    assert cleanup.invoke() == 0, "a completed RBAC replay must verify absence"
    assert cleanup.state_path.read_bytes() == before, (
        "completed RBAC replay must not rewrite the journal"
    )
    if saved["scope"] == "gpu":
        assert all("--context" in call for call in cleanup.calls), (
            "GPU-only cleanup must not access the shared CPU cluster"
        )


@pytest.mark.parametrize(
    "phase", ["QUEUES_DRAINED", "CONTROL_CONSUMERS_STOPPED", "GPU_EXECUTORS_STOPPED"]
)
def test_guarded_cleanup_refuses_incomplete_stop_evidence(cleanup, phase):
    state = STATE.read_state(cleanup.state_path)
    for event in state["history"]:
        if event["phase"] == phase:
            event["status"] = "FAILED"
    STATE.atomic_write(cleanup.state_path, state)

    with pytest.raises(KUBE.CleanupStateError, match="completed drain and stop"):
        cleanup.invoke()
    assert cleanup.api.deleted == [], (
        "an incomplete stop barrier must not delete grants"
    )
    assert cleanup.calls == [], (
        "phase prerequisites must be checked before Kubernetes I/O"
    )


@pytest.mark.parametrize(
    "fault", ["source", "cluster", "context", "namespace", "proof-namespace", "rows"]
)
def test_guarded_cleanup_refuses_rebound_scope_before_deleting(cleanup, fault):
    if fault == "source":
        config = copy.deepcopy(cleanup.config)
        config["namespace"] = "other-system"
        cleanup.config_path.write_text(json.dumps(config), encoding="utf-8")
    elif fault == "namespace":
        cleanup.api.objects[("Namespace", "", NAMESPACE)]["metadata"]["uid"] = (
            "replacement"
        )
    else:
        state = STATE.read_state(cleanup.state_path)
        section = state["inventory_snapshot"]["gpu"]["by_context"]["gpu-a-context"]
        if fault == "proof-namespace":
            section["workload_rbac"]["binding"]["system_namespace_uid"] = (
                "foreign-proof"
            )
        elif fault == "rows":
            section["resources"] = [
                row
                for row in section["resources"]
                if row.get("guarded_delete") != "workload-rbac"
            ]
        STATE.atomic_write(cleanup.state_path, state)

    with pytest.raises((KUBE.CleanupStateError, BootstrapError)):
        cleanup.invoke(
            context="gpu:other" if fault == "context" else CONTEXT,
            cluster_id="other" if fault == "cluster" else "gpu-a",
        )
    assert cleanup.api.deleted == [], (
        "rebound source or identity must never authorize deletion"
    )


def test_guarded_cleanup_refuses_a_restarted_executor(cleanup):
    for item in cleanup.api.objects.values():
        if item["kind"] == "Deployment" and "executor" in item["metadata"]["name"]:
            item["spec"]["replicas"] = 1
            item["status"]["readyReplicas"] = 1
    with pytest.raises(KUBE.CleanupStateError, match="reappeared or restarted"):
        cleanup.invoke()
    assert cleanup.api.deleted == [], (
        "a restarted Executor must retain its workload grants"
    )


@pytest.mark.parametrize("terminating", [False, True])
def test_guarded_cleanup_requires_absent_pods_even_when_deployment_reports_zero(
    cleanup, terminating
):
    cleanup.remaining_pods.append(
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": "remaining-executor",
                "namespace": NAMESPACE,
                **(
                    {"deletionTimestamp": "2026-09-14T00:00:00Z"} if terminating else {}
                ),
            },
            "status": {"phase": "Running"},
        }
    )
    with pytest.raises(KUBE.CleanupStateError, match="still has Pod objects"):
        cleanup.invoke()
    assert cleanup.api.deleted == [], (
        "remaining or terminating Pods must block RBAC deletion"
    )
    assert "workload_rbac_removed" not in STATE.read_state(cleanup.state_path), (
        "unproven Pod termination must not create removal checkpoints"
    )


def test_guarded_cleanup_resumes_original_journal_after_checkpoint_interruption(
    cleanup, monkeypatch
):
    original_write = KUBE.atomic_write

    def interrupted(path, document):
        original_write(path, document)
        raise RuntimeError("interrupted after durable checkpoint")

    monkeypatch.setattr(KUBE, "atomic_write", interrupted)
    with pytest.raises(RuntimeError, match="durable checkpoint"):
        cleanup.invoke()
    saved = STATE.read_state(cleanup.state_path)
    assert len(saved["workload_rbac_removed"][CONTEXT]) == 1, (
        "the confirmed first deletion must survive interruption"
    )
    assert saved["inventory_snapshot"] == cleanup.inventory, (
        "interruption must preserve the original per-context inventory"
    )
    monkeypatch.setattr(KUBE, "atomic_write", original_write)

    assert cleanup.invoke() == 0, (
        "the original journal must support guarded cleanup resume"
    )

    saved = STATE.read_state(cleanup.state_path)
    assert len(cleanup.api.deleted) == len(cleanup.api.expected), (
        "resume must not delete a recorded grant twice"
    )
    assert len(saved["workload_rbac_removed"][CONTEXT]) == len(cleanup.api.expected), (
        "resume must checkpoint every confirmed original removal"
    )


def test_completed_application_phase_never_authorizes_unrecorded_rbac_delete(cleanup):
    state = STATE.read_state(cleanup.state_path)
    STATE.transition(
        state,
        phase="APPLICATION_OBJECTS_DELETED",
        status="COMPLETED",
        message="fixture",
    )
    STATE.atomic_write(cleanup.state_path, state)
    with pytest.raises(KUBE.CleanupStateError, match="lacks RBAC deletion checkpoints"):
        cleanup.invoke()
    assert cleanup.api.deleted == [], (
        "application completion is not missing RBAC delete authority"
    )


@pytest.mark.parametrize("cleanup", [{"scope": "all", "mode": "reset"}], indirect=True)
def test_completed_rbac_replay_can_verify_gpu_absence_after_cpu_deletion(
    cleanup, monkeypatch
):
    assert cleanup.invoke() == 0, (
        "the fixture must complete guarded deletion before CPU removal"
    )
    state = STATE.read_state(cleanup.state_path)
    for phase in (
        "APPLICATION_OBJECTS_DELETED",
        "NAMESPACES_DELETED",
        "CLEANUP_COMPLETED",
    ):
        STATE.transition(
            state, phase=phase, status="COMPLETED", message="completed fixture"
        )
    STATE.atomic_write(cleanup.state_path, state)
    for key in list(cleanup.api.objects):
        if key[0] in {"Deployment", "ServiceAccount"} or key == (
            "Namespace",
            "",
            NAMESPACE,
        ):
            cleanup.api.objects.pop(key)
    original_run = KUBE.run_command

    def deleted_cpu(arguments, **kwargs):
        assert "--context" in arguments, (
            "completed replay must not contact the deleted CPU"
        )
        return original_run(arguments, **kwargs)

    monkeypatch.setattr(KUBE, "run_command", deleted_cpu)
    before = cleanup.state_path.read_bytes()
    deletions = list(cleanup.api.deleted)

    assert cleanup.invoke(action="verify-targets", skip_cpu=True) == 0, (
        "completed GPU absence proof must remain verifiable after CPU deletion"
    )

    assert cleanup.api.deleted == deletions, (
        "completed replay must not issue new deletions"
    )
    assert cleanup.state_path.read_bytes() == before, (
        "read-only replay must preserve its journal"
    )


def test_pending_rbac_mutation_cannot_skip_cpu_checks(cleanup):
    with pytest.raises(
        KUBE.CleanupStateError, match="completed read-only cleanup replay"
    ):
        cleanup.invoke(skip_cpu=True)
    assert cleanup.api.deleted == [], "pending deletion cannot bypass CPU verification"
    assert cleanup.calls == [], (
        "an invalid skip-cpu request must fail before Kubernetes I/O"
    )
