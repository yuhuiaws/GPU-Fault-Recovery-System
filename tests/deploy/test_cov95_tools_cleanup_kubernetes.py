from __future__ import annotations

import base64
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests._script_loader import lazy_script_module
from tests.deploy._cov95_tools_support import TOOLS
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


@pytest.mark.parametrize("context", ["gpu", "gpu:", "context-without-plane"])
def test_cleanup_client_requires_explicit_gpu_plane(context: str) -> None:
    with pytest.raises(KUBE.CleanupStateError, match="explicit plane"):
        KUBE.CleanupClient({}, context)


def test_cleanup_client_scopes_cpu_and_gpu_without_implicit_current_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {"cpu_kubeconfig": "/example/cpu"}
    assert KUBE.CleanupClient(config, "cpu").prefix[-2:] == [
        "--kubeconfig",
        "/example/cpu",
    ]
    monkeypatch.delenv("KUBECONFIG", raising=False)
    assert KUBE.CleanupClient(config, "gpu:local").prefix == [
        "kubectl",
        "--request-timeout=30s",
        "--context",
        "local",
    ]


def install_transport(
    monkeypatch: pytest.MonkeyPatch, value: Any, code: int = 0
) -> list[Any]:
    calls = []

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, kwargs))
        output = value if isinstance(value, str) else json.dumps(value)
        return subprocess.CompletedProcess(
            arguments, code, stdout=output, stderr="fixture transport error"
        )

    monkeypatch.setattr(KUBE, "run_command", run)
    return calls


def test_cleanup_client_serializes_payload_and_never_infers_absence_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = KUBE.CleanupClient({"cpu_kubeconfig": "/dev/null"}, "cpu")
    calls = install_transport(monkeypatch, " acknowledged \n")
    assert (
        client.run("create", "-f", "-", payload={"kind": "Example"}) == "acknowledged"
    )
    assert json.loads(calls[0][1]["input_text"]) == {"kind": "Example"}
    assert 0 < calls[0][1]["timeout_seconds"] <= 45
    install_transport(monkeypatch, "", 1)
    with pytest.raises(KUBE.CleanupStateError, match="command failed"):
        client.get("namespace", NAMESPACE)


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("nonobject", "invalid object metadata"),
        ("metadata", "invalid object metadata"),
        ("uid", "identity is incomplete"),
        ("kind", "different object identity"),
        ("name", "different object identity"),
        ("namespace", "different object identity"),
    ],
)
def test_cleanup_client_validates_returned_object_identity(
    monkeypatch: pytest.MonkeyPatch, fault: str, problem: str
) -> None:
    client = KUBE.CleanupClient({"cpu_kubeconfig": "/dev/null"}, "cpu")
    value: Any = resource("Deployment", "owned")
    if fault == "nonobject":
        value = []
    elif fault == "metadata":
        value["metadata"] = None
    elif fault == "kind":
        value["kind"] = "Secret"
    else:
        value["metadata"][fault] = "" if fault == "uid" else "other"
    install_transport(monkeypatch, value)
    with pytest.raises(KUBE.CleanupStateError, match=problem):
        client.get("deployment", "owned", NAMESPACE)


def test_cleanup_client_returns_only_explicit_absence_or_verified_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = KUBE.CleanupClient({"cpu_kubeconfig": "/dev/null"}, "cpu")
    install_transport(monkeypatch, " ")
    assert client.get("namespace", NAMESPACE) is None
    expected = resource("Namespace", NAMESPACE, namespace=None)
    calls = install_transport(monkeypatch, expected)
    assert client.get("namespace", NAMESPACE) == expected
    assert "--ignore-not-found" in calls[0][0]
    install_transport(monkeypatch, "{")
    with pytest.raises(json.JSONDecodeError):
        client.get("namespace", NAMESPACE)


@pytest.mark.parametrize(
    "value,problem",
    [
        ([], "invalid resource list"),
        ({}, "invalid resource list"),
        ({"items": None}, "invalid resource list"),
        ({"items": [None]}, "invalid list item"),
    ],
)
def test_cleanup_list_read_does_not_drop_invalid_evidence(
    monkeypatch: pytest.MonkeyPatch, value: Any, problem: str
) -> None:
    client = KUBE.CleanupClient({}, "gpu:local")
    install_transport(monkeypatch, value)
    with pytest.raises(KUBE.CleanupStateError, match=problem):
        client.items("get", "nodes")


def test_cleanup_list_read_preserves_all_valid_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = [
        resource("Node", "a", namespace=None),
        resource("Node", "b", namespace=None),
    ]
    install_transport(monkeypatch, {"items": nodes})
    assert KUBE.CleanupClient({}, "gpu:local").items("get", "nodes") == nodes


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("anchor-absent", "identity is unavailable"),
        ("cluster-drift", "different cluster"),
        ("checkpoint", "ownership checkpoint"),
        ("namespace-absent", "namespace is absent"),
        ("terminating", "namespace is terminating"),
    ],
)
def test_cleanup_requires_cluster_and_namespace_ownership(
    fault: str, problem: str
) -> None:
    client, state = Client(), document()
    if fault == "anchor-absent":
        client.objects.pop(("namespace", "kube-system", ""))
    elif fault == "cluster-drift":
        state["cluster_uids"][CONTEXT] = "other"
    elif fault == "checkpoint":
        state["namespace_snapshots"].clear()
    elif fault == "namespace-absent":
        client.objects.pop(("namespace", NAMESPACE, ""))
    else:
        client.objects[("namespace", NAMESPACE, "")]["metadata"][
            "deletionTimestamp"
        ] = "now"
    with pytest.raises(KUBE.CleanupStateError, match=problem):
        if fault in {"anchor-absent", "cluster-drift"}:
            KUBE.verify_cluster(client, state, CONTEXT)
        else:
            KUBE.verify_namespace(client, state, CONTEXT)
    assert client.mutations() == []


def test_missing_namespace_is_allowed_only_after_explicit_absence_permission() -> None:
    client, state = Client(), document()
    client.objects.pop(("namespace", NAMESPACE, ""))
    assert KUBE.verify_namespace(client, state, CONTEXT, allow_absent=True) is None
    KUBE.delete_namespace(client, state, CONTEXT)
    state["history"] = [{"phase": "NAMESPACES_DELETED", "status": "COMPLETED"}]
    KUBE.verify_quiesced(client, state, CONTEXT)
    assert client.mutations() == []


@pytest.mark.parametrize("kind", ["deployment", "cronjob", "daemonset"])
@pytest.mark.parametrize("absent", [True, False])
def test_quiesced_checkpoint_requires_workloads_stopped_or_confirmed_absent(
    kind: str, absent: bool
) -> None:
    client, state = Client(), document()
    state["scope"] = "all"
    state["history"] = [
        {"phase": "GPU_DATA_PLANE_SOURCES_STOPPED", "status": "COMPLETED"}
    ]
    state["inventory_snapshot"]["gpu"]["resources"] = [
        {"kind": "service", "name": "support", "phase": "support"},
        {"kind": kind, "name": "producer", "phase": "producer"},
    ]
    if not absent:
        value = resource(
            {
                "deployment": "Deployment",
                "cronjob": "CronJob",
                "daemonset": "DaemonSet",
            }[kind],
            "producer",
        )
        value["spec"] = {"replicas": 0, "suspend": True}
        client.objects[(kind, "producer", NAMESPACE)] = value
    if kind == "daemonset" and not absent:
        with pytest.raises(KUBE.CleanupStateError, match="reappeared or restarted"):
            KUBE.verify_quiesced(client, state, CONTEXT)
    else:
        KUBE.verify_quiesced(client, state, CONTEXT)
    assert client.mutations() == []


def test_gpu_only_checkpoint_does_not_claim_cpu_consumers_stopped() -> None:
    client, state = Client(), document()
    state["scope"] = "gpu"
    state["history"] = []
    state["namespace_snapshots"]["cpu"] = state["namespace_snapshots"][CONTEXT]
    KUBE.verify_quiesced(client, state, "cpu")
    assert client.calls == []


@pytest.mark.parametrize("count", [0, 513])
def test_namespace_inventory_must_be_bounded_and_complete(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    client = Client()
    monkeypatch.setattr(
        client,
        "run",
        lambda *_args: "\n".join(f"kind-{index}" for index in range(count)),
    )
    with pytest.raises(KUBE.CleanupStateError, match="discovery is incomplete"):
        KUBE.namespace_items(client, NAMESPACE)


def test_namespace_inventory_cannot_cross_scope() -> None:
    client = Client()
    client.namespaced = [resource("Pod", "other", namespace="customer")]
    with pytest.raises(KUBE.CleanupStateError, match="crossed its scope"):
        KUBE.namespace_items(client, NAMESPACE)


@pytest.mark.parametrize(
    "kind,name,allowed",
    [
        ("ConfigMap", "gpu-fault-example", True),
        ("Secret", "gpu-fault-example", True),
        ("ConfigMap", "kube-root-ca.crt", True),
        ("ServiceAccount", "default", True),
        ("Event", "event-1", True),
        ("Secret", "customer", False),
        ("Deployment", "customer", False),
    ],
)
def test_implicit_namespace_objects_do_not_expand_to_customer_resources(
    kind: str, name: str, allowed: bool
) -> None:
    assert KUBE.implicit_namespace_object(resource(kind, name)) is allowed


@pytest.mark.parametrize(
    "namespace", ["default", "kube-system", "kube-public", "kube-node-lease"]
)
def test_shared_namespaces_cannot_be_captured_for_cleanup(namespace: str) -> None:
    client, state = Client(), document()
    state["targets"]["namespace"] = namespace
    with pytest.raises(KUBE.CleanupStateError, match="shared Kubernetes namespace"):
        KUBE.capture_namespace(client, state, CONTEXT)
    assert client.calls == []


def test_absent_namespace_cannot_be_captured_or_deleted_without_checkpoint() -> None:
    client, state = Client(), document()
    client.objects.pop(("namespace", NAMESPACE, ""))
    with pytest.raises(KUBE.CleanupStateError, match="namespace is absent"):
        KUBE.capture_namespace(client, state, CONTEXT)
    state["namespace_snapshots"].clear()
    with pytest.raises(KUBE.CleanupStateError, match="ownership checkpoint"):
        KUBE.delete_namespace(client, state, CONTEXT)


@pytest.mark.parametrize(
    "fault", ["already-absent", "uid-before", "uid-during", "delayed"]
)
def test_owned_delete_is_uid_conditioned_and_waits_for_confirmed_absence(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    client = Client()
    current = resource("Namespace", NAMESPACE, namespace=None)
    replacement = copy.deepcopy(current)
    replacement["metadata"]["uid"] = "foreign"
    values = {
        "already-absent": [None],
        "uid-before": [replacement],
        "uid-during": [current, replacement],
        "delayed": [current, current, None],
    }
    reads = iter(values[fault])
    monkeypatch.setattr(client, "get", lambda *_args: next(reads))
    sleeps = []
    monkeypatch.setattr(KUBE.time, "sleep", sleeps.append)
    if fault.startswith("uid"):
        with pytest.raises(
            KUBE.CleanupStateError, match="UID changed|replaced while deleting"
        ):
            KUBE.delete_owned(
                client, kind="namespace", name=NAMESPACE, uid=current["metadata"]["uid"]
            )
    else:
        KUBE.delete_owned(
            client, kind="namespace", name=NAMESPACE, uid=current["metadata"]["uid"]
        )
    assert len(client.mutations()) == int(fault in {"uid-during", "delayed"})
    assert bool(sleeps) is (fault == "delayed")


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("duplicates", "duplicate Kubernetes node identity"),
        ("no-fleet", "requires captured fleet inventory"),
        ("active-absent", "active fleet node is missing"),
        ("inactive-absent", "no proven fleet targets"),
        ("other-cluster", "no proven fleet targets"),
        ("units", "lacks installed unit inventory"),
    ],
)
def test_node_cleanup_target_set_requires_complete_fleet_proof(
    fault: str, problem: str
) -> None:
    client, state = Client(), document()
    if fault == "duplicates":
        client.nodes.append(copy.deepcopy(client.nodes[0]))
    elif fault == "no-fleet":
        state["fleet_snapshot"] = None
    elif fault.endswith("absent"):
        state["fleet_snapshot"][0]["node_id"] = "missing"
        state["fleet_snapshot"][0]["lifecycle_state"] = (
            "ACTIVE" if fault.startswith("active") else "RETIRED"
        )
    elif fault == "other-cluster":
        state["fleet_snapshot"][0]["cluster_id"] = "other"
    else:
        state["fleet_snapshot"][0]["installed_unit_inventory"] = None
    with pytest.raises(KUBE.CleanupStateError, match=problem):
        KUBE.node_targets(client, state, CONTEXT, "gpu-a")


def test_node_metadata_cleanup_without_owned_keys_does_not_patch_customer_metadata() -> (
    None
):
    client, state = Client(), document()
    client.nodes[0]["metadata"]["labels"] = {"customer": "retained"}
    KUBE.clear_node_metadata(client, state, CONTEXT, "gpu-a")
    assert client.mutations() == []


def manifest(*, image: str = "example/image:v1") -> dict[str, Any]:
    return KUBE.node_manifest(
        name="gpu-fault-cleanup-example",
        namespace=NAMESPACE,
        run_id="run-example",
        nodes=["node-b", "node-a"],
        image=image,
        mode="uninstall",
        host_script_b64=HOST_SCRIPT,
    )


@pytest.mark.parametrize(
    "mode,nodes,script,problem",
    [
        ("unknown", ["node-a"], HOST_SCRIPT, "invalid node cleanup"),
        ("stop", [], HOST_SCRIPT, "invalid node cleanup"),
        ("stop", ["node-a"], base64.b64encode(b" \n").decode(), "host script is empty"),
    ],
)
def test_node_manifest_refuses_unbounded_or_empty_requests(
    mode: str, nodes: list[str], script: str, problem: str
) -> None:
    with pytest.raises(KUBE.CleanupStateError, match=problem):
        KUBE.node_manifest(
            name="owned",
            namespace=NAMESPACE,
            run_id="run",
            nodes=nodes,
            image="example/image:v1",
            mode=mode,
            host_script_b64=script,
        )


@pytest.mark.parametrize(
    "image", ["example/image", "example/image:latest", "example/image:v1"]
)
def test_manifest_verification_accepts_only_known_admission_defaults(
    image: str,
) -> None:
    expected = manifest(image=image)
    live = copy.deepcopy(expected)
    live["spec"].update(
        revisionHistoryLimit=10,
        updateStrategy={
            "type": "RollingUpdate",
            "rollingUpdate": {"maxUnavailable": 1, "maxSurge": 0},
        },
    )
    template = live["spec"]["template"]
    template["metadata"]["creationTimestamp"] = None
    pod = template["spec"]
    pod.update(
        dnsPolicy="ClusterFirst",
        restartPolicy="Always",
        schedulerName="default-scheduler",
        securityContext={},
        terminationGracePeriodSeconds=30,
    )
    container = pod["containers"][0]
    container.update(
        imagePullPolicy="IfNotPresent" if image.endswith(":v1") else "Always",
        resources={},
        terminationMessagePath="/dev/termination-log",
        terminationMessagePolicy="File",
    )
    container["readinessProbe"].update(
        failureThreshold=3, successThreshold=1, timeoutSeconds=1
    )
    before = copy.deepcopy(live)
    KUBE.verify_node_manifest(live, expected)
    assert live == before
    live["spec"]["template"]["spec"]["hostNetwork"] = True
    with pytest.raises(KUBE.CleanupStateError, match="spec differs"):
        KUBE.verify_node_manifest(live, expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "Deployment"),
        ("apiVersion", "v1"),
        ("name", "foreign"),
        ("namespace", "foreign"),
        ("labels", {}),
    ],
)
def test_manifest_identity_is_not_replaced_by_admission(field: str, value: Any) -> None:
    expected = manifest()
    live = copy.deepcopy(expected)
    (live if field in {"kind", "apiVersion"} else live["metadata"])[field] = value
    with pytest.raises(KUBE.CleanupStateError, match="object identity differs"):
        KUBE.verify_node_manifest(live, expected)


@pytest.mark.parametrize("context", ["cpu", "gpu:foreign"])
def test_pending_cleanup_refuses_foreign_plane_without_transport(
    tmp_path: Path, context: str
) -> None:
    state = document()
    state["node_cleanup"] = {context: {"status": "PLANNED"}}
    with pytest.raises(KUBE.CleanupStateError, match="outside the selected scope"):
        KUBE.cleanup_pending({}, state, tmp_path / "state.json")


@pytest.mark.parametrize("missing_identity", [True, False])
def test_pending_cleanup_needs_bound_cluster_and_confirms_absence_before_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_identity: bool
) -> None:
    client, state = Client(), document()
    state["node_cleanup"] = {CONTEXT: {"name": "absent", "status": "PLANNED"}}
    if missing_identity:
        state["cluster_uids"].clear()
    monkeypatch.setattr(KUBE, "CleanupClient", lambda *_args: client)
    if missing_identity:
        with pytest.raises(KUBE.CleanupStateError, match="lacks cluster identity"):
            KUBE.cleanup_pending({}, state, tmp_path / "state.json")
        assert state["node_cleanup"][CONTEXT]["status"] == "PLANNED"
    else:
        path = tmp_path / "state.json"
        KUBE.cleanup_pending({}, state, path)
        assert (
            json.loads(path.read_text())["node_cleanup"][CONTEXT]["status"] == "REMOVED"
        )
        before = path.read_bytes()
        KUBE.cleanup_pending({}, state, path)
        assert path.read_bytes() == before


@pytest.mark.parametrize("primary_failure", [False, True])
def test_failed_cleanup_preserves_unverified_journal_and_original_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, primary_failure: bool
) -> None:
    client, state = Client(), document()
    client.lose_ack = primary_failure
    original = client.run

    def run(*arguments: str, **kwargs: Any) -> str:
        if arguments[0] == "delete":
            raise KUBE.CleanupStateError("cleanup transport unavailable")
        return original(*arguments, **kwargs)

    monkeypatch.setattr(client, "run", run)
    monkeypatch.setattr(KUBE, "CleanupClient", lambda *_args: client)
    with pytest.raises(
        KUBE.CleanupStateError,
        match="lost create acknowledgement"
        if primary_failure
        else "cleanup transport unavailable",
    ) as caught:
        KUBE.run_node_cleanup(
            {},
            state,
            tmp_path / "state.json",
            context=CONTEXT,
            cluster_id="gpu-a",
            image="example/image:v1",
            mode="stop",
            host_script_b64=HOST_SCRIPT,
        )
    assert state["node_cleanup"][CONTEXT]["status"] != "REMOVED"
    if primary_failure:
        assert any("remains unverified" in note for note in caught.value.__notes__), (
            "lost create ACK must retain the subsequent cleanup failure diagnostic"
        )


@pytest.mark.parametrize("fault", ["cluster", "creation", "delayed"])
def test_node_cleanup_checks_cluster_and_observed_creation_before_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    client, state = Client(), document()
    if fault == "cluster":
        state["cluster_uids"].clear()
    original_get = client.get

    def get(kind: str, name: str, namespace: str = "") -> Any:
        result = original_get(kind, name, namespace)
        if kind == "daemonset" and result is not None:
            if fault == "creation":
                return None
            if fault == "delayed" and client.daemonset_reads <= 2:
                result["status"]["numberReady"] = 0
        return result

    monkeypatch.setattr(client, "get", get)
    monkeypatch.setattr(KUBE, "CleanupClient", lambda *_args: client)
    sleeps = []
    monkeypatch.setattr(KUBE.time, "sleep", sleeps.append)
    if fault == "delayed":
        KUBE.run_node_cleanup(
            {},
            state,
            tmp_path / "state.json",
            context=CONTEXT,
            cluster_id="gpu-a",
            image="example/image:v1",
            mode="stop",
            host_script_b64=HOST_SCRIPT,
        )
        assert sleeps, "an unready cleanup DaemonSet must trigger bounded polling"
        assert state["node_cleanup"][CONTEXT]["status"] == "REMOVED"
    else:
        with pytest.raises(
            KUBE.CleanupStateError,
            match="identity checkpoint|creation was not observed",
        ):
            KUBE.run_node_cleanup(
                {},
                state,
                tmp_path / "state.json",
                context=CONTEXT,
                cluster_id="gpu-a",
                image="example/image:v1",
                mode="stop",
                host_script_b64=HOST_SCRIPT,
            )


def cli_state(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "cpu_kubeconfig": "/dev/null",
                "clusters": [{"cluster_id": "gpu-a", "context": "gpu-a"}],
            }
        )
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps({"cpu": {"resources": []}, **document()["inventory_snapshot"]})
    )
    path = tmp_path / "state.json"
    state = STATE.initialize(
        path,
        config_path=config,
        inventory_path=inventory,
        scope="gpu",
        mode="reset",
        node_mode="uninstall",
        cluster_ids=["gpu-a"],
    )
    state.update(
        {
            key: value
            for key, value in document().items()
            if key in {"cluster_uids", "namespace_snapshots", "fleet_snapshot"}
        }
    )
    STATE.atomic_write(path, state)
    return config, path, state


@pytest.mark.parametrize(
    "action",
    [
        "capture",
        "delete-namespace",
        "cleanup-owned",
        "clear-node-metadata",
        "verify-targets",
        "node-cleanup",
    ],
)
def test_cleanup_entrypoint_uses_only_scoped_fake_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, action: str
) -> None:
    config, path, _state = cli_state(tmp_path)
    client = Client()
    contexts = []

    def factory(_config: dict[str, Any], context: str) -> Client:
        contexts.append(context)
        return client

    monkeypatch.setattr(KUBE, "CleanupClient", factory)
    monkeypatch.setenv("GPU_FAULT_CLEANUP_HOST_SCRIPT_B64", HOST_SCRIPT)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cleanup_kubernetes",
            "--config",
            str(config),
            "--state-file",
            str(path),
            "--context",
            CONTEXT,
            "--cluster-id",
            "gpu-a",
            "--skip-cpu",
            "--image",
            "example/image:v1",
            action,
        ],
    )
    assert KUBE.main() == 0
    assert set(contexts) == {CONTEXT}
    if action == "node-cleanup":
        assert STATE.read_state(path)["node_cleanup"][CONTEXT]["status"] == "REMOVED"


def test_cleanup_entrypoint_requires_state_for_noncommand_actions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, _path, _state = cli_state(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["cleanup_kubernetes", "--config", str(config), "capture"]
    )
    with pytest.raises(KUBE.CleanupStateError, match="requires a state file"):
        KUBE.main()


@pytest.mark.parametrize("code", [0, 1])
@pytest.mark.parametrize("separator", [[], ["--"]])
def test_command_passthrough_is_scoped_and_preserves_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    code: int,
    separator: list[str],
) -> None:
    config, _path, _state = cli_state(tmp_path)
    calls = install_transport(monkeypatch, "local fake response\n", code)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cleanup_kubernetes",
            "--config",
            str(config),
            "command",
            *separator,
            "get",
            "namespace",
            NAMESPACE,
        ],
    )
    if code:
        with pytest.raises(KUBE.CleanupStateError, match="command failed"):
            KUBE.main()
        assert capsys.readouterr().out == ""
    else:
        assert KUBE.main() == 0
        assert capsys.readouterr().out == "local fake response\n"
    assert calls[0][0] == [
        "kubectl",
        "--request-timeout=30s",
        "--kubeconfig",
        "/dev/null",
        "get",
        "namespace",
        NAMESPACE,
    ]
