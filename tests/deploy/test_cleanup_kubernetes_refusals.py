# ruff: noqa: F401, F811
"""Refusal paths of the Kubernetes cleanup tool that no live run should reach.

Resource discovery that finds nothing to list, guarded workload RBAC rows
without their proof, cleanup requests whose mode, namespace or identities were
rebound, node metadata that cannot be reasoned about, an occupied temporary
DaemonSet name and the entrypoint's drain and verify actions. Every case runs
against the scripted clients of the neighbouring tests; nothing touches a
cluster.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tests._script_loader import lazy_script_module
from tests.deploy._cov95_tools_support import TOOLS
from tests.deploy.test_cleanup_workload_rbac import cleanup
from tests.deploy.test_cov95_tools_cleanup_kubernetes import (
    cli_state,
    install_transport,
)
from tests.regional.test_cleanup_kubernetes import (
    CONTEXT,
    HOST_SCRIPT,
    NAMESPACE,
    Client,
    document,
    resource,
)

KUBE = lazy_script_module(TOOLS / "cleanup_kubernetes.py")
STATE = lazy_script_module(TOOLS / "cleanup_state.py")
RBAC_CONTEXT = "gpu:gpu-a-context"


# --- discovery and guarded RBAC rows ------------------------------------------


class ProjectionOnlyClient(Client):
    def run(self, *arguments: str, payload: dict[str, Any] | None = None) -> str:
        if "api-resources" in arguments:
            return "nodes.metrics.k8s.io\npods.metrics.k8s.io\n"
        return super().run(*arguments, payload=payload)


def test_discovery_of_only_projection_resources_is_incomplete() -> None:
    client = ProjectionOnlyClient()
    with pytest.raises(KUBE.CleanupStateError, match="discovery is incomplete"):
        KUBE.namespace_items(client, NAMESPACE)
    assert client.mutations() == []


def inventory(
    rows: list[dict[str, Any]], *, snapshot: dict[str, Any] | None
) -> dict[str, Any]:
    section: dict[str, Any] = {"resources": rows}
    if snapshot is not None:
        section["by_context"] = {"gpu-a-context": snapshot}
    return {"inventory_snapshot": {"gpu": section}}


GUARDED_ROW = {
    "kind": "role",
    "name": "training-reader",
    "namespace": "training",
    "scope": "namespaced",
    "phase": "support",
    "clean": "delete",
    "guarded_delete": "workload-rbac",
}


def test_guarded_rows_without_a_context_snapshot_are_refused() -> None:
    with pytest.raises(KUBE.CleanupStateError, match="lacks a context-specific"):
        KUBE.workload_rbac_proof(inventory([GUARDED_ROW], snapshot=None), RBAC_CONTEXT)


def test_unguarded_rows_without_a_snapshot_need_no_proof() -> None:
    plain = {**GUARDED_ROW, "guarded_delete": ""}
    assert (
        KUBE.workload_rbac_proof(inventory([plain], snapshot=None), RBAC_CONTEXT)
        is None
    )


def test_unknown_guarded_action_is_refused() -> None:
    row = {**GUARDED_ROW, "guarded_delete": "node-rbac"}
    with pytest.raises(KUBE.CleanupStateError, match="unknown guarded cleanup action"):
        KUBE.workload_rbac_proof(
            inventory([], snapshot={"resources": [row]}), RBAC_CONTEXT
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "deployment"),
        ("scope", "cluster"),
        ("phase", "producer"),
        ("clean", "keep"),
    ],
)
def test_guarded_rows_outside_namespaced_support_deletion_are_refused(
    field: str, value: str
) -> None:
    row = {**GUARDED_ROW, field: value}
    with pytest.raises(KUBE.CleanupStateError, match="invalid scope"):
        KUBE.workload_rbac_proof(
            inventory([], snapshot={"resources": [row]}), RBAC_CONTEXT
        )


def test_snapshot_without_guarded_rows_or_proof_needs_nothing() -> None:
    plain = {**GUARDED_ROW, "guarded_delete": ""}
    assert (
        KUBE.workload_rbac_proof(
            inventory([], snapshot={"resources": [plain]}), RBAC_CONTEXT
        )
        is None
    )


# --- delete_workload_rbac preconditions ---------------------------------------


def delete(cleanup: Any, state: dict[str, Any], **options: Any) -> list[str]:
    return KUBE.delete_workload_rbac(
        cleanup.config,
        state,
        cleanup.state_path,
        context=RBAC_CONTEXT,
        cluster_id="gpu-a",
        **options,
    )


@pytest.mark.parametrize(
    "damage,message",
    [
        ("mode", "requires clean or reset mode"),
        ("namespace", "namespace differs"),
        ("identity", "lacks bound cluster/namespace identities"),
        ("progress", "progress is invalid"),
    ],
)
def test_rebound_cleanup_requests_are_refused_before_any_deletion(
    cleanup: Any, damage: str, message: str
) -> None:
    state = STATE.read_state(cleanup.state_path)
    if damage == "mode":
        state["mode"] = "dry-run"
    elif damage == "namespace":
        state["targets"]["namespace"] = "other-namespace"
    elif damage == "identity":
        state["namespace_snapshots"][RBAC_CONTEXT]["uid"] = ""
    else:
        state["workload_rbac_removed"] = "not-a-mapping"
    with pytest.raises(KUBE.CleanupStateError, match=message):
        delete(cleanup, state)
    assert cleanup.api.deleted == []
    assert cleanup.calls == [], "a refused request performs no Kubernetes I/O"


def test_context_without_guarded_rows_or_proof_deletes_nothing(cleanup: Any) -> None:
    state = STATE.read_state(cleanup.state_path)
    state["inventory_snapshot"]["gpu"]["by_context"]["gpu-a-context"] = {
        "resources": []
    }
    assert delete(cleanup, state) == []
    assert cleanup.api.deleted == []


def test_gpu_scoped_cleanup_never_verifies_the_shared_cpu_cluster(cleanup: Any) -> None:
    state = STATE.read_state(cleanup.state_path)
    state["scope"] = "gpu"
    removed = delete(cleanup, state)
    assert set(removed) == set(
        state["inventory_snapshot"]["gpu"]["by_context"]["gpu-a-context"][
            "workload_rbac"
        ]["resources"]
    )
    assert all("--context" in call for call in cleanup.calls), (
        "GPU-only cleanup must not reach the CPU kubeconfig"
    )


def test_checkpoint_refuses_a_proof_whose_authority_changed(
    cleanup: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def tampering(
        _config: Any, _target: Any, proof: dict[str, Any], *, checkpoint: Any, **_: Any
    ) -> list[str]:
        proof["resources"]["training/Role/injected"] = "uid-injected"
        checkpoint()
        return []

    monkeypatch.setattr(KUBE, "delete_recorded_workload_namespace_rbac", tampering)
    state = STATE.read_state(cleanup.state_path)
    with pytest.raises(KUBE.CleanupStateError, match="changed its captured authority"):
        delete(cleanup, state)
    assert "workload_rbac_removed" not in STATE.read_state(cleanup.state_path), (
        "a refused checkpoint must not be journaled"
    )


def test_deletion_that_confirms_nothing_is_a_failure(
    cleanup: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        KUBE, "delete_recorded_workload_namespace_rbac", lambda *_a, **_k: []
    )
    state = STATE.read_state(cleanup.state_path)
    with pytest.raises(KUBE.CleanupStateError, match="did not confirm every recorded"):
        delete(cleanup, state)


# --- node metadata ----------------------------------------------------------------


def test_spare_with_an_unknown_scheduling_state_is_refused() -> None:
    client, state = Client(), document()
    node = client.nodes[0]
    node["spec"] = {"unschedulable": "yes"}
    node["metadata"]["labels"] = {"gpu-fault.io/spare": "true"}
    node["metadata"]["annotations"] = {"gpu-fault.io/previous-unschedulable": "false"}
    state["inventory_snapshot"]["gpu"]["node_labels"].append("gpu-fault.io/spare")
    with pytest.raises(KUBE.CleanupStateError, match="unknown scheduling state"):
        KUBE.clear_node_metadata(client, state, CONTEXT, "gpu-a")
    assert client.mutations() == []


def test_target_node_without_our_metadata_is_not_patched() -> None:
    client, state = Client(), document()
    before = copy.deepcopy(client.nodes)
    KUBE.clear_node_metadata(client, state, CONTEXT, "gpu-a")
    assert client.mutations() == []
    assert client.nodes == before


def test_orphaned_installer_annotations_are_removed_and_verified() -> None:
    client, state = Client(), document()
    orphan = client.nodes[1]
    orphan["metadata"]["annotations"] = {
        "gpu-fault.io/installer-state": "Failed",
        "gpu-fault.io/installer-attempts": "2",
        "customer.example/owner": "team-a",
    }
    KUBE.clear_node_metadata(client, state, CONTEXT, "gpu-a")
    assert orphan["metadata"]["annotations"] == {"customer.example/owner": "team-a"}
    patches = [args for args in client.mutations() if args[:2] == ("patch", "node")]
    assert [args[2] for args in patches] == ["customer-node"]
    operations = json.loads(patches[0][patches[0].index("-p") + 1])
    assert operations[0] == {
        "op": "test",
        "path": "/metadata/uid",
        "value": "uid-customer-node",
    }
    assert state["orphaned_installer_nodes"] == {
        CONTEXT: {"customer-node": "uid-customer-node"}
    }


# --- the entrypoint -----------------------------------------------------------------


def test_node_cleanup_refuses_an_occupied_temporary_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, path, state = cli_state(tmp_path)
    client = Client()
    suffix = hashlib.sha256(f"{state['run_id']}:{CONTEXT}".encode()).hexdigest()[:16]
    name = f"gpu-fault-node-cleanup-{suffix}"
    client.objects[("daemonset", name, NAMESPACE)] = resource("DaemonSet", name)
    monkeypatch.setattr(KUBE, "CleanupClient", lambda _config, _context: client)
    with pytest.raises(KUBE.CleanupStateError, match="already occupied"):
        KUBE.run_node_cleanup(
            json.loads(config.read_text()),
            STATE.read_state(path),
            path,
            context=CONTEXT,
            cluster_id="gpu-a",
            image="example/image:v1",
            mode="uninstall",
            host_script_b64=HOST_SCRIPT,
        )
    assert client.mutations() == [], "an occupied name creates nothing"


def run_main(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> int:
    monkeypatch.setattr(sys, "argv", ["cleanup_kubernetes", *arguments])
    return KUBE.main()


def test_command_passthrough_strips_a_doubled_separator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _path, _state = cli_state(tmp_path)
    calls = install_transport(monkeypatch, "scoped response\n")
    code = run_main(
        monkeypatch, "--config", str(config), "command", "--", "--", "get", "ns"
    )
    assert code == 0
    assert capsys.readouterr().out == "scoped response\n"
    assert calls[0][0][-2:] == ["get", "ns"], "no separator reaches kubectl"


def test_capture_on_the_cpu_cluster_records_its_identity_without_node_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, path, _state = cli_state(tmp_path)
    client = Client()
    # The scripted client answers for the CPU cluster here, which must not
    # alias the GPU cluster the state already binds.
    client.objects[("namespace", "kube-system", "")]["metadata"]["uid"] = "uid-cpu"
    client.objects[("namespace", NAMESPACE, "")]["metadata"]["uid"] = "uid-cpu-ns"
    monkeypatch.setattr(KUBE, "CleanupClient", lambda _config, _context: client)
    monkeypatch.setattr(
        KUBE,
        "node_targets",
        lambda *_a, **_k: pytest.fail("the CPU cluster has no node targets"),
    )
    assert (
        run_main(
            monkeypatch,
            "--config",
            str(config),
            "--state-file",
            str(path),
            "--context",
            "cpu",
            "capture",
        )
        == 0
    )
    persisted = STATE.read_state(path)
    assert persisted["cluster_uids"]["cpu"] == "uid-cpu"
    assert persisted["namespace_snapshots"]["cpu"]["uid"] == "uid-cpu-ns"


def test_verify_targets_requires_the_cpu_identity_unless_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, path, _state = cli_state(tmp_path)
    client = Client()
    monkeypatch.setattr(KUBE, "CleanupClient", lambda _config, _context: client)
    with pytest.raises(KUBE.CleanupStateError, match="lacks a bound cluster identity"):
        run_main(
            monkeypatch,
            "--config",
            str(config),
            "--state-file",
            str(path),
            "verify-targets",
        )
    assert client.calls == [], "the identity check precedes every cluster read"


def drain_state(tmp_path: Path, *, clusters: bool = True) -> tuple[Path, Path]:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "cpu_kubeconfig": "/dev/null",
                "clusters": [{"cluster_id": "gpu-a", "context": "gpu-a"}]
                if clusters
                else [],
            }
        )
    )
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps({"cpu": {"resources": []}, **document()["inventory_snapshot"]})
    )
    path = tmp_path / "state.json"
    state = STATE.initialize(
        path,
        config_path=config,
        inventory_path=inventory_path,
        scope="all",
        mode="reset",
        node_mode="uninstall",
    )
    STATE.atomic_write(path, state)
    return config, path


def drain(monkeypatch: pytest.MonkeyPatch, config: Path, path: Path) -> int:
    return run_main(
        monkeypatch,
        "--config",
        str(config),
        "--state-file",
        str(path),
        "drain-registry",
    )


def test_registry_drain_requires_a_completed_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, path = drain_state(tmp_path)
    calls = install_transport(monkeypatch, "")
    with pytest.raises(KUBE.CleanupStateError, match="requires completed preflight"):
        drain(monkeypatch, config, path)
    assert calls == []


def start_draining(path: Path) -> None:
    state = STATE.read_state(path)
    STATE.transition(state, phase="PREFLIGHT", status="COMPLETED", message="fixture")
    STATE.transition(
        state, phase="CLUSTERS_DRAINING", status="IN_PROGRESS", message="fixture"
    )
    STATE.atomic_write(path, state)


@pytest.mark.parametrize("code", [0, 1])
def test_registry_drain_runs_the_regional_drain_for_every_cluster(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, code: int
) -> None:
    config, path = drain_state(tmp_path)
    start_draining(path)
    calls = install_transport(monkeypatch, "", code)
    if code:
        with pytest.raises(KUBE.CleanupStateError, match="drain did not converge"):
            drain(monkeypatch, config, path)
    else:
        assert drain(monkeypatch, config, path) == 0
    (arguments, options), *rest = calls
    assert rest == []
    assert Path(arguments[0]).name == "rollout-regional-release.sh"
    assert arguments[1:] == [
        "drain-cluster",
        "--config",
        str(config),
        "--cluster-id",
        "gpu-a",
    ]
    assert options["timeout_seconds"] == 600


def test_registry_drain_without_clusters_has_nothing_to_drain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, path = drain_state(tmp_path, clusters=False)
    start_draining(path)
    calls = install_transport(monkeypatch, "")
    assert drain(monkeypatch, config, path) == 0
    assert calls == []


def test_checkpoint_refuses_deletion_progress_that_moved_backwards(
    cleanup: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forgetting(
        _config: Any, _target: Any, proof: dict[str, Any], *, checkpoint: Any, **_: Any
    ) -> list[str]:
        proof["removed"] = []
        checkpoint()
        return []

    monkeypatch.setattr(KUBE, "delete_recorded_workload_namespace_rbac", forgetting)
    state = STATE.read_state(cleanup.state_path)
    state["workload_rbac_removed"] = {RBAC_CONTEXT: ["training/Role/already-removed"]}
    with pytest.raises(KUBE.CleanupStateError, match="progress moved backwards"):
        delete(cleanup, state)


class IgnoringPatchClient(Client):
    """A cluster whose node patches are acknowledged but change nothing."""

    def run(self, *arguments: str, payload: dict[str, Any] | None = None) -> str:
        if arguments[:2] == ("patch", "node"):
            self.calls.append((arguments, payload))
            return ""
        return super().run(*arguments, payload=payload)


def test_target_metadata_that_survives_its_patch_does_not_converge() -> None:
    client, state = IgnoringPatchClient(), document()
    client.nodes[0]["metadata"]["labels"] = {"gpu-fault.io/managed": "true"}
    with pytest.raises(KUBE.CleanupStateError, match="did not converge"):
        KUBE.clear_node_metadata(client, state, CONTEXT, "gpu-a")
    assert [args[2] for args in client.mutations()] == ["managed-node"]


def test_orphan_annotations_that_survive_their_patch_do_not_converge() -> None:
    client, state = IgnoringPatchClient(), document()
    client.nodes[1]["metadata"]["annotations"] = {
        "gpu-fault.io/installer-state": "WaitingForKey"
    }
    with pytest.raises(KUBE.CleanupStateError, match="did not converge"):
        KUBE.clear_node_metadata(client, state, CONTEXT, "gpu-a")
    assert [args[2] for args in client.mutations()] == ["customer-node"]
