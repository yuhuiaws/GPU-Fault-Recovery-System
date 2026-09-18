from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "control-plane" / "regional" / "prepare-clean-redeploy.sh"
UNINSTALLER = ROOT / "deploy" / "node" / "uninstall-gpu-fault-collector.sh"

FAKE_REGISTRY_ROLLOUT = r"""
import json
import os
import sys
from pathlib import Path

if sys.argv[1:3] != ["-m", "gpu_fault_release.rollout"]:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
args = sys.argv[3:]
if not args or args[0] != "drain-cluster":
    raise SystemExit("unexpected release operation")
config = json.loads(Path(args[args.index("--config") + 1]).read_text())
cluster_ids = [args[index + 1] for index, item in enumerate(args) if item == "--cluster-id"]
if cluster_ids != [item["cluster_id"] for item in config["clusters"]]:
    raise SystemExit("registry drain lost the complete selected cluster set")
api_path = os.environ.get("FAKE_API_STATE")
if not api_path:
    with Path(os.environ["FAKE_KUBECTL_LOG"]).open("a") as stream:
        stream.write("registry-drain " + " ".join(args) + "\n")
    raise SystemExit(0)
api_path = Path(api_path)
api = json.loads(api_path.read_text())
journal = json.loads(Path(os.environ["FAKE_CLEANUP_STATE"]).read_text())
latest = {item["phase"]: item["status"] for item in journal["history"]}
with Path(os.environ["FAKE_KUBECTL_LOG"]).open("a") as stream:
    stream.write(json.dumps({
        "args": ["registry-drain", *args], "phase": journal["phase"],
        "completed": sorted(name for name, status in latest.items() if status == "COMPLETED"),
    }) + "\n")
resumable = (
    latest.get("CLUSTERS_DRAINING") == "COMPLETED"
    and latest.get("CONTROL_CONSUMERS_STOPPED") != "COMPLETED"
    and journal["phase"] in {
        "GPU_DATA_PLANE_SOURCES_STOPPED", "QUEUES_DRAINED", "CONTROL_CONSUMERS_STOPPED",
    }
)
if (
    journal["phase"] != "CLUSTERS_DRAINING" and not resumable
    or latest.get("PREFLIGHT") != "COMPLETED"
):
    raise SystemExit("registry drain preceded the bound preflight")
if api.get("registry_failure") == "before-publish":
    raise SystemExit("injected registry refusal")
api["registry_drains"] = api.get("registry_drains", 0) + 1
api["cluster_states"] = {cluster_id: "DRAINING" for cluster_id in cluster_ids}
api_path.write_text(json.dumps(api))
if api.get("registry_failure") == "after-publish":
    raise SystemExit("injected registry convergence failure")
"""


def _fake_registry_cli(fake_bin: Path) -> None:
    wrapper = fake_bin / "python3"
    wrapper.write_text(f"#!{sys.executable}\n{FAKE_REGISTRY_ROLLOUT}", encoding="utf-8")
    wrapper.chmod(0o755)


def write_config(
    tmp_path: Path,
    *,
    cpu_kubeconfig: str = "/secure/cpu.kubeconfig",
    cluster_ids: tuple[str, ...] = ("gpu-a", "gpu-b"),
) -> Path:
    path = tmp_path / "regional-release.json"
    path.write_text(
        json.dumps(
            {
                "cpu_kubeconfig": cpu_kubeconfig,
                "namespace": "gpu-fault-system",
                "clusters": [
                    {"cluster_id": cluster_id, "context": f"{cluster_id}-context"}
                    for cluster_id in cluster_ids
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def run_script(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), *arguments], check=False, text=True, capture_output=True, env=env
    )


def test_clean_redeploy_supports_explicit_offline_dry_run(tmp_path: Path) -> None:
    result = run_script(
        "--config",
        str(write_config(tmp_path)),
        "--mode",
        "clean",
        "--node-mode",
        "uninstall",
        "--offline-plan",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("DRY RUN:"), result.stdout
    assert "gpu-a-context" in result.stdout, result.stdout
    assert "gpu-b-context" in result.stdout, result.stdout
    assert "Preserve Secrets, release ConfigMaps, Aurora, NLB" in result.stdout


def test_clean_redeploy_can_select_one_gpu_cluster(tmp_path: Path) -> None:
    result = run_script(
        "--config",
        str(write_config(tmp_path)),
        "--scope",
        "gpu",
        "--cluster-id",
        "gpu-b",
        "--offline-plan",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "cluster_id=gpu-b context=gpu-b-context" in result.stdout
    assert "cluster_id=gpu-a" not in result.stdout
    assert "Keep the CPU control plane running" in result.stdout


def test_clean_redeploy_requires_state_file_for_execution(tmp_path: Path) -> None:
    result = run_script("--config", str(write_config(tmp_path)), "--execute")

    assert result.returncode != 0
    assert "--state-file is required with --execute" in result.stderr


def test_reset_requires_uninstall_and_explicit_confirmation(tmp_path: Path) -> None:
    config = str(write_config(tmp_path))
    wrong_node_mode = run_script("--config", config, "--mode", "reset")
    assert wrong_node_mode.returncode != 0
    assert "--mode reset requires --node-mode uninstall" in wrong_node_mode.stderr

    state = tmp_path / "reset.tsv"
    missing_confirmation = run_script(
        "--config",
        config,
        "--mode",
        "reset",
        "--node-mode",
        "uninstall",
        "--state-file",
        str(state),
        "--execute",
    )
    assert missing_confirmation.returncode != 0
    assert "--confirm-reset RESET_GPU_FAULT_INSTALLATION" in missing_confirmation.stderr


def test_reset_plan_preserves_eks_but_requires_external_aurora_cleanup(
    tmp_path: Path,
) -> None:
    result = run_script(
        "--config",
        str(write_config(tmp_path)),
        "--mode",
        "reset",
        "--node-mode",
        "uninstall",
        "--offline-plan",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Preserve all EKS clusters" in result.stdout
    assert "Delete the dedicated Aurora cluster" in result.stdout
    assert "Drop every gpu_fault_*" not in result.stdout


def test_node_uninstaller_removes_gpu_persistence_unit() -> None:
    text = UNINSTALLER.read_text(encoding="utf-8")

    assert (ROOT / "deploy/systemd/gpu-fault-gpu-persistence.service").is_file(), (
        "GPU persistence systemd unit is missing"
    )
    assert "/opt/gpu-fault/installed-units.txt" in text
    assert "gpu-fault-*.service" in text


def _skip_mode_fixture(
    tmp_path: Path, interrupt_worker: bool
) -> tuple[Path, Path, Path, dict[str, str]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _fake_registry_cli(fake_bin)
    log = tmp_path / "kubectl.log"
    fake_kubectl = fake_bin / "kubectl"
    fake_kubectl.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["FAKE_KUBECTL_LOG"], "a", encoding="utf-8") as stream:
    stream.write(" ".join(args) + "\\n")
if (
    os.environ.get("FAKE_INTERRUPT_WORKER") == "true"
    and "scale" in args and "deployment/gpu-fault-control-worker" in args
    and not os.path.exists(os.environ["FAKE_FAILURE_MARKER"])
):
    with open(os.environ["FAKE_FAILURE_MARKER"], "w", encoding="utf-8") as stream:
        stream.write("failed")
    raise SystemExit(1)
if "config" in args and "view" in args:
    print(json.dumps({"users": [{"name": "fixture", "user": {}}]}))
    raise SystemExit(0)
if "exec" in args:
    code_index = args.index("-c")
    code = args[code_index + 1]
    compile(code, "<clean-redeploy-probe>", "exec")
    print("[]" if args[code_index + 2] == "fleet" else "0\\t0\\t0\\t0")
    raise SystemExit(0)
if "get" in args and "--raw=/readyz" in args:
    print("ok")
    raise SystemExit(0)
if (
    "get" in args
    and (
        "gpu-fault-installed-resources" in args
        or "customresourcedefinition" in args
    )
):
    if "--ignore-not-found" in args:
        raise SystemExit(0)
    print("Error from server (NotFound): configmap not found", file=sys.stderr)
    raise SystemExit(1)
context = args[args.index("--context") + 1] if "--context" in args else "cpu"
namespace = args[args.index("-n") + 1] if "-n" in args else "gpu-fault-system"
if (
    "get" in args and "namespace" in args
    and args[args.index("namespace") + 1] in {"kube-system", "gpu-fault-system"}
):
    name = args[args.index("namespace") + 1]
    print(json.dumps({
        "apiVersion": "v1", "kind": "Namespace",
        "metadata": {
            "name": name,
            "uid": ("cluster-" if name == "kube-system" else "namespace-") + context,
        },
    }))
    raise SystemExit(0)
if (
    "get" in args
    and "-o" in args
    and "json" in args
    and (
        args.index("-o") == args.index("get") + 2
        or any("," in argument for argument in args)
        or "namespace" in args
        or "pods" in args
    )
):
    items = []
    kind = args[args.index("get") + 1]
    if "deployment" in kind.split(","):
        names = (
            ["gpu-fault-api-ha", "gpu-fault-control-worker"] if context == "cpu"
            else [
                "gpu-fault-cluster-executor", "gpu-fault-completion-watcher",
                "gpu-fault-kubernetes-node-resource-collector",
                "gpu-fault-node-installer-reconciler",
            ]
        )
        items = [
            {"apiVersion": "apps/v1", "kind": "Deployment",
             "metadata": {"name": name, "namespace": namespace, "uid": "uid-" + name}}
            for name in names
        ]
    print(json.dumps({"items": items}))
    raise SystemExit(0)
if "get" in args and "pod" in args:
    print(json.dumps({"items": [{
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": "fake-database-pod", "namespace": namespace},
        "status": {
            "phase": "Running",
            "containerStatuses": [{"name": "api", "ready": True}],
        },
    }]}))
    raise SystemExit(0)
if "get" in args and "deployment" in args:
    if "-o" in args and args[args.index("-o") + 1] == "json":
        name = args[args.index("deployment") + 1]
        print(json.dumps({
            "apiVersion": "apps/v1", "kind": "Deployment",
            "metadata": {"name": name, "uid": "uid-" + name, "namespace": namespace},
            "spec": {"replicas": 0}, "status": {"replicas": 0},
        }))
    else:
        print("0", end="")
    raise SystemExit(0)
if "get" in args and "nodes" in args:
    print("fake-node Ready")
    raise SystemExit(0)
if "get" in args and ("daemonset" in args or "cronjob" in args):
    if "--ignore-not-found" in args:
        raise SystemExit(0)
    print("Error from server (NotFound): resource not found", file=sys.stderr)
    raise SystemExit(1)
raise SystemExit(0)
""",
        encoding="utf-8",
    )
    fake_kubectl.chmod(0o755)
    kubeconfig = tmp_path / "cpu.kubeconfig"
    kubeconfig.write_text("test", encoding="utf-8")
    config = write_config(
        tmp_path, cpu_kubeconfig=str(kubeconfig), cluster_ids=("gpu-a",)
    )
    state = tmp_path / "state" / "clean.json"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_KUBECTL_LOG": str(log),
        "FAKE_INTERRUPT_WORKER": str(interrupt_worker).lower(),
        "FAKE_FAILURE_MARKER": str(tmp_path / "failed-once"),
    }
    return config, state, log, env


@pytest.mark.parametrize("interrupt_worker", [False, True])
def test_execute_path_compiles_database_probe_and_preserves_stop_order(
    tmp_path: Path, interrupt_worker: bool
) -> None:
    config, state, log, env = _skip_mode_fixture(tmp_path, interrupt_worker)

    result = run_script(
        "--config",
        str(config),
        "--node-mode",
        "skip",
        "--state-file",
        str(state),
        "--execute",
        env=env,
    )
    if interrupt_worker:
        assert result.returncode != 0
        failed = json.loads(state.read_text())
        assert failed["phase"] == "CONTROL_CONSUMERS_STOPPED"
        assert failed["status"] == "FAILED"
        result = run_script(
            "--config",
            str(config),
            "--node-mode",
            "skip",
            "--state-file",
            str(state),
            "--execute",
            env=env,
        )

    assert result.returncode == 0, result.stdout + result.stderr
    assert state.stat().st_mode & 0o777 == 0o600
    cleanup_state = json.loads(state.read_text(encoding="utf-8"))
    assert cleanup_state["phase"] == "CLEANUP_COMPLETED"
    assert cleanup_state["status"] == "COMPLETED"
    assert len(cleanup_state["content_sha256"]) == 64
    assert cleanup_state["inventory_snapshot"]["schema_version"] == 1
    assert cleanup_state["namespace_snapshots"]["cpu"]["uid"] == "namespace-cpu", (
        "the shell must bind the actual CPU solution namespace before mutation"
    )
    assert (
        cleanup_state["namespace_snapshots"]["gpu:gpu-a-context"]["uid"]
        == "namespace-gpu-a-context"
    ), "the shell must bind the actual selected GPU solution namespace"
    calls = log.read_text(encoding="utf-8")
    producer = calls.index(
        "scale deployment/gpu-fault-node-installer-reconciler --replicas=0"
    )
    ingress = calls.index("scale deployment/gpu-fault-api-ha --replicas=0")
    worker = calls.index("scale deployment/gpu-fault-control-worker --replicas=0")
    executor = calls.index("scale deployment/gpu-fault-cluster-executor --replicas=0")
    registry = calls.index("registry-drain drain-cluster")
    assert registry < producer < worker < ingress < executor, calls
    assert (
        calls.count("scale deployment/gpu-fault-completion-watcher --replicas=0") == 1
    )
    before = state.read_bytes()
    repeat = run_script(
        "--config",
        str(config),
        "--node-mode",
        "skip",
        "--state-file",
        str(state),
        "--execute",
        env=env,
    )
    assert repeat.returncode == 0, repeat.stdout + repeat.stderr
    assert state.read_bytes() == before
    repeated_calls = log.read_text(encoding="utf-8")[len(calls) :]
    assert "scale deployment/" not in repeated_calls
    assert "apply -f" not in calls + repeated_calls


def test_node_components_stop_only_after_the_executors_and_the_drain(
    reset_run: ResetRun,
) -> None:
    calls = reset_run.harness.calls()
    consumer_stop = next(
        index
        for index, call in enumerate(calls)
        if "scale" in call["args"]
        and "deployment/gpu-fault-control-worker" in call["args"]
    )
    executor_stop = next(
        index
        for index, call in enumerate(calls)
        if "scale" in call["args"]
        and "deployment/gpu-fault-cluster-executor" in call["args"]
    )
    database_reads = [
        index for index, call in enumerate(calls) if "exec" in call["args"]
    ]
    node_creates = [
        index for index, call in enumerate(calls) if "create" in call["args"]
    ]
    assert database_reads and node_creates, (
        "the fixture skipped the drain or node phase"
    )
    assert max(database_reads) < consumer_stop < executor_stop < min(node_creates), (
        "node cleanup ran while consumers or Executors could still dispatch commands"
    )
    required = {"QUEUES_DRAINED", "CONTROL_CONSUMERS_STOPPED", "GPU_EXECUTORS_STOPPED"}
    for index in node_creates:
        assert calls[index]["phase"] == "NODE_RUNTIMES_STOPPED", (
            "node cleanup ran inside the producer or executor phase"
        )
        assert required.issubset(calls[index]["completed"]), (
            "node cleanup began before all required stop checkpoints"
        )
    rollouts = [index for index, call in enumerate(calls) if "rollout" in call["args"]]
    assert len(rollouts) == 2
    assert max(rollouts) < min(database_reads), (
        "initial preflight selected a database Pod before the CPU rollouts settled"
    )


FAKE_KUBECTL_CLEANUP = r"""#!/usr/bin/env python3
import copy
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
context = args[args.index("--context") + 1] if "--context" in args else "cpu"
namespace = args[args.index("-n") + 1] if "-n" in args else ""
api_path = Path(os.environ["FAKE_API_STATE"])
api = json.loads(api_path.read_text())
state_path = Path(os.environ["FAKE_CLEANUP_STATE"])
journal = json.loads(state_path.read_text()) if state_path.exists() else {}
latest = {item["phase"]: item["status"] for item in journal.get("history", [])}
entry = {
    "args": args, "phase": journal.get("phase"),
    "completed": sorted(name for name, status in latest.items() if status == "COMPLETED"),
}
payload = None
if "create" in args or "delete" in args and "--raw" in args:
    payload = json.load(sys.stdin)
    if "delete" in args:
        entry["payload"] = payload
if "patch" in args:
    entry["patch"] = json.loads(args[args.index("-p") + 1])
with Path(os.environ["FAKE_KUBECTL_LOG"]).open("a") as stream:
    stream.write(json.dumps(entry) + "\n")


def save():
    api_path.write_text(json.dumps(api))


def key(kind, name, selected_namespace=None):
    scope = namespace if selected_namespace is None else selected_namespace
    if kind in {"node", "namespace", "clusterrole", "clusterrolebinding"}:
        scope = ""
    return json.dumps([context, scope, kind, name])


def absent():
    if "--ignore-not-found" not in args:
        print("Error from server (NotFound): object not found", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)


if "config" in args and "view" in args:
    print(json.dumps({"users": [{"name": "fixture", "user": {}}]}))
    raise SystemExit(0)
if "api-resources" in args:
    print("deployment\npod\nconfigmap\nsecret\nserviceaccount")
    raise SystemExit(0)
if "exec" in args:
    if api.get("no_pods") or not any(
        json.loads(identity)[0] == "cpu"
        and item.get("kind") == "Deployment"
        and item.get("spec", {}).get("replicas", 0) > 0
        for identity, item in api["objects"].items()
    ):
        raise SystemExit("database exec after control-plane stop")
    code_index = args.index("-c")
    compile(args[code_index + 1], "<cleanup-database-probe>", "exec")
    action = args[code_index + 2]
    if action == "fleet":
        print(json.dumps(api["fleet"]))
    elif action == "counts":
        counts = api.get("drain_counts", api["counts"]) if journal.get("phase") in {
            "QUEUES_DRAINED", "CONTROL_CONSUMERS_STOPPED",
        } else api["counts"]
        print(*counts, sep="\t")
    else:
        raise SystemExit("only read-only fleet/counts probes are supported")
    raise SystemExit(0)
if "rollout" in args:
    target = args[args.index("rollout") + 2]
    if "\t" in target or not target.startswith("deployment/"):
        raise SystemExit("rollout target is not a deployment name")
    if key("deployment", target.split("/", 1)[1]) not in api["objects"]:
        raise SystemExit("rollout changed the captured namespace or deployment")
    raise SystemExit(0)
if "scale" in args:
    name = args[args.index("scale") + 1].split("/", 1)[1]
    if name == "gpu-fault-control-worker" and api.get("interrupt_consumer"):
        api["interrupt_consumer"] = False
        save()
        raise SystemExit("injected consumer stop failure")
    item = api["objects"][key("deployment", name)]
    item["spec"]["replicas"] = 0
    item["status"] = {"replicas": 0, "readyReplicas": 0, "availableReplicas": 0}
    save()
    raise SystemExit(0)
if "wait" in args:
    name = args[args.index("-l") + 1].removeprefix("app=")
    if api["objects"][key("deployment", name)]["spec"]["replicas"] != 0:
        raise SystemExit("consumer Pods remain")
    raise SystemExit(0)
if "get" in args and "--raw=/readyz" in args:
    print("ok")
    raise SystemExit(0)
if "get" in args:
    index = args.index("get")
    aliases = {
        "nodes": "node", "namespaces": "namespace", "pods": "pod",
        "deployments": "deployment", "daemonsets": "daemonset",
        "configmaps": "configmap", "cronjobs": "cronjob",
        "secrets": "secret", "serviceaccounts": "serviceaccount",
        "roles": "role", "rolebindings": "rolebinding",
    }
    kinds = [
        aliases.get(value.split(".")[0], value.split(".")[0])
        for value in args[index + 1].split(",")
    ]
    name = (
        args[index + 2]
        if len(args) > index + 2 and not args[index + 2].startswith("-")
        else None
    )
    output = args[args.index("-o") + 1] if "-o" in args else "json"
    if kinds == ["pod"] and "-l" in args:
        label = args[args.index("-l") + 1].removeprefix("app=")
        deployment = api["objects"].get(key("deployment", label))
        items = [
            copy.deepcopy(item)
            for identity, item in api["objects"].items()
            if not api.get("no_pods")
            and json.loads(identity)[:3] == [context, namespace, "pod"]
            and item.get("metadata", {}).get("labels", {}).get("app") == label
        ]
        if not api.get("no_pods") and deployment and deployment["spec"]["replicas"] > 0:
            items.append({
                "apiVersion": "v1", "kind": "Pod",
                "metadata": {"name": "database-" + label, "namespace": namespace},
                "status": {
                    "phase": "Running", "containerStatuses": [{"name": "api", "ready": True}],
                },
            })
        print(json.dumps({"items": items}))
        raise SystemExit(0)
    if name is not None:
        item = api["objects"].get(key(kinds[0], name))
        if item is None:
            absent()
        if "spec.replicas" in output:
            print(item["spec"]["replicas"])
        elif output == "name":
            print(kinds[0] + "/" + name)
        else:
            print(json.dumps(item))
        raise SystemExit(0)
    items = []
    for identity, item in api["objects"].items():
        selected_context, selected_namespace, kind, _name = json.loads(identity)
        if selected_context == context and kind in kinds and (
            not namespace or selected_namespace == namespace
        ):
            items.append(item)
    print(json.dumps({"items": items}))
    raise SystemExit(0)
if "create" in args:
    required = {"QUEUES_DRAINED", "CONTROL_CONSUMERS_STOPPED", "GPU_EXECUTORS_STOPPED"}
    if journal.get("phase") != "NODE_RUNTIMES_STOPPED" or not required.issubset(
        entry["completed"]
    ):
        raise SystemExit("node creation preceded the stop barrier")
    if "--dry-run=server" in args:
        print(json.dumps(payload))
        raise SystemExit(0)
    if api.get("interrupt_node"):
        api["interrupt_node"] = False
        save()
        raise SystemExit("injected node cleanup create failure")
    item = copy.deepcopy(payload)
    name = item["metadata"]["name"]
    item["metadata"].update(uid="cleanup-uid-" + context, generation=1)
    item["status"] = {
        "observedGeneration": 1, "desiredNumberScheduled": 1,
        "updatedNumberScheduled": 1, "numberReady": 1, "numberMisscheduled": 0,
    }
    api["objects"][key("daemonset", name, item["metadata"]["namespace"])] = item
    save()
    raise SystemExit(0)
if "delete" in args:
    if "--raw" in args:
        path = args[args.index("--raw") + 1]
        parts = path.split("/")
        kind = next(
            (singular for plural, singular in (
                ("daemonsets", "daemonset"), ("roles", "role"),
                ("rolebindings", "rolebinding"),
            ) if plural in parts), "namespace"
        )
        name = parts[-1]
        selected_namespace = (
            parts[parts.index("namespaces") + 1] if kind != "namespace" else ""
        )
        identity = key(kind, name, selected_namespace)
        item = api["objects"].get(identity)
        if item is not None and payload["preconditions"]["uid"] != item["metadata"]["uid"]:
            raise SystemExit("delete UID precondition failed")
        if payload["propagationPolicy"] != "Foreground":
            raise SystemExit("deletion lost foreground confirmation")
        if kind in {"role", "rolebinding"}:
            required = {"QUEUES_DRAINED", "GPU_EXECUTORS_STOPPED", "NODE_RUNTIMES_STOPPED"}
            if journal.get("scope") == "all":
                required.add("CONTROL_CONSUMERS_STOPPED")
            if journal.get("phase") != "APPLICATION_OBJECTS_DELETED" or not required.issubset(
                entry["completed"]
            ):
                raise SystemExit("workload RBAC deletion preceded the stop barrier")
            if api.get("interrupt_rbac") and api.get("rbac_deleted"):
                api["interrupt_rbac"] = False
                save()
                raise SystemExit("injected workload RBAC deletion failure")
            if item is not None and payload["preconditions"].get("resourceVersion") != (
                item["metadata"]["resourceVersion"]
            ):
                raise SystemExit("RBAC delete resourceVersion precondition failed")
            api.setdefault("rbac_deleted", []).append(identity)
    else:
        index = args.index("delete")
        kind, name = args[index + 1 : index + 3]
        identity = key(kind, name)
    api["objects"].pop(identity, None)
    save()
    raise SystemExit(0)
if "patch" in args and "node" in args:
    name = args[args.index("node") + 1]
    item = api["objects"][key("node", name)]
    for operation in entry["patch"]:
        parts = [
            part.replace("~1", "/").replace("~0", "~")
            for part in operation["path"].split("/")[1:]
        ]
        parent = item
        for part in parts[:-1]:
            parent = parent[part]
        field = parts[-1]
        if operation["op"] == "test":
            if parent[field] != operation["value"]:
                raise SystemExit("node patch precondition failed")
        elif operation["op"] == "remove":
            del parent[field]
        elif operation["op"] == "replace":
            parent[field] = operation["value"]
        else:
            raise SystemExit("unexpected node patch operation")
    save()
    raise SystemExit(0)
raise SystemExit("unexpected fake Kubernetes command")
"""


@dataclass(frozen=True)
class CleanupHarness:
    config: Path
    state: Path
    log: Path
    api: Path
    env: dict[str, str]

    def run(self, *, timeout_seconds: int = 30) -> subprocess.CompletedProcess[str]:
        return run_script(
            "--config",
            str(self.config),
            "--mode",
            "reset",
            "--node-mode",
            "uninstall",
            "--state-file",
            str(self.state),
            "--confirm-reset",
            "RESET_GPU_FAULT_INSTALLATION",
            "--timeout-seconds",
            str(timeout_seconds),
            "--execute",
            env=self.env,
        )

    def calls(self) -> list[dict[str, Any]]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]


def _reset_harness(tmp_path: Path, *, stopped: bool = False) -> CleanupHarness:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _fake_registry_cli(fake_bin)
    log = tmp_path / "kubectl.jsonl"
    fake_kubectl = fake_bin / "kubectl"
    fake_kubectl.write_text(FAKE_KUBECTL_CLEANUP, encoding="utf-8")
    fake_kubectl.chmod(0o755)
    kubeconfig = tmp_path / "cpu.kubeconfig"
    kubeconfig.write_text("test", encoding="utf-8")
    config = write_config(
        tmp_path, cpu_kubeconfig=str(kubeconfig), cluster_ids=("gpu-a",)
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)
    state = state_dir / "clean.json"
    objects: dict[str, dict[str, Any]] = {}

    def add(context: str, kind: str, name: str, namespace: str = "") -> dict[str, Any]:
        item: dict[str, Any] = {
            "apiVersion": "apps/v1" if kind == "Deployment" else "v1",
            "kind": kind,
            "metadata": {
                "name": name,
                "uid": f"uid-{context}-{name}",
                "resourceVersion": "1",
                **({"namespace": namespace} if namespace else {}),
            },
        }
        objects[json.dumps([context, namespace, kind.lower(), name])] = item
        return item

    for context in ("cpu", "gpu-a-context"):
        for namespace_name, uid in (
            ("kube-system", f"cluster-{context}"),
            ("gpu-fault-system", f"namespace-{context}"),
        ):
            add(context, "Namespace", namespace_name)["metadata"]["uid"] = uid
        deployments = (
            ["gpu-fault-api-ha", "gpu-fault-control-worker"]
            if context == "cpu"
            else [
                "gpu-fault-node-installer-reconciler",
                "gpu-fault-completion-watcher",
                "gpu-fault-kubernetes-node-resource-collector",
                "gpu-fault-cluster-executor",
            ]
        )
        for name in deployments:
            item = add(context, "Deployment", name, "gpu-fault-system")
            replicas = 0 if stopped else 1
            item["spec"] = {"replicas": replicas}
            item["status"] = {
                "replicas": replicas,
                "readyReplicas": replicas,
                "availableReplicas": replicas,
            }
    node = add("gpu-a-context", "Node", "node-a")
    node["metadata"]["labels"] = {
        "gpu-fault.io/spare": "true",
        "customer.example/retained": "yes",
    }
    node["metadata"]["annotations"] = {"gpu-fault.io/previous-unschedulable": "false"}
    node["spec"] = {"unschedulable": True}
    customer = add("gpu-a-context", "Node", "customer-node")
    customer["metadata"]["labels"] = {"customer.example/retained": "yes"}
    customer["spec"] = {"unschedulable": True}
    fleet = [
        {
            "cluster_id": "gpu-a",
            "node_id": "node-a",
            "lifecycle_state": "ACTIVE",
            "installed_unit_inventory": {"units": ["gpu-fault-node-agent.service"]},
        }
    ]
    api = tmp_path / "api.json"
    api.write_text(
        json.dumps(
            {
                "objects": objects,
                "fleet": fleet,
                "counts": [0, 0, 0, 0],
                "no_pods": stopped,
                "interrupt_node": True,
            }
        ),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_KUBECTL_LOG": str(log),
        "FAKE_API_STATE": str(api),
        "FAKE_CLEANUP_STATE": str(state),
    }
    return CleanupHarness(config, state, log, api, env)


@dataclass(frozen=True)
class ResetRun:
    harness: CleanupHarness
    failed: dict[str, Any]
    restart_at: int


@pytest.fixture
def interrupted_reset(tmp_path: Path) -> ResetRun:
    harness = _reset_harness(tmp_path)
    first = harness.run()
    assert first.returncode != 0, "the node cleanup interruption was not exercised"
    failed = json.loads(harness.state.read_text())
    assert (failed["phase"], failed["status"]) == ("NODE_RUNTIMES_STOPPED", "FAILED"), (
        first.stdout + first.stderr
    )
    assert failed["schema_version"] == 2
    return ResetRun(harness, failed, len(harness.calls()))


@pytest.fixture
def reset_run(interrupted_reset: ResetRun) -> ResetRun:
    result = interrupted_reset.harness.run()
    assert result.returncode == 0, result.stdout + result.stderr
    return interrupted_reset


def test_a_reset_resumes_over_a_stopped_control_plane_with_the_earlier_fleet_snapshot(
    reset_run: ResetRun,
) -> None:
    """Only the same bound journal can retain the drain and fleet proof."""

    document = json.loads(reset_run.harness.state.read_text(encoding="utf-8"))
    assert document["phase"] == "CLEANUP_COMPLETED", document["phase"]
    assert document["status"] == "COMPLETED"
    for field in (
        "run_id",
        "config_sha256",
        "inventory_sha256",
        "fleet_snapshot",
        "fleet_snapshot_sha256",
        "cluster_uids",
        "namespace_snapshots",
        "node_targets",
    ):
        assert document[field] == reset_run.failed[field], (
            f"restart changed the original {field} authority"
        )
    assert (
        document["history"][: len(reset_run.failed["history"])]
        == reset_run.failed["history"]
    ), "restart discarded the interrupted journal prefix"
    assert document["node_targets"] == {
        "gpu:gpu-a-context": {"node-a": "uid-gpu-a-context-node-a"}
    }
    assert document["node_cleanup"]["gpu:gpu-a-context"]["status"] == "REMOVED"
    calls = reset_run.harness.calls()[reset_run.restart_at :]
    assert not any("exec" in call["args"] for call in calls), (
        "restart attempted to export fleet or drain after the CPU plane stopped"
    )
    assert not any(
        "scale" in call["args"] or "rollout" in call["args"] for call in calls
    ), "restart repeated an already completed stop or preflight phase"


@pytest.mark.parametrize("location", ["adjacent-failed-record", "requested-state"])
def test_a_reset_refuses_an_unbound_phase_and_fleet_snapshot(
    tmp_path: Path, location: str
) -> None:
    harness = _reset_harness(tmp_path, stopped=True)
    api_before = harness.api.read_bytes()
    fleet = json.loads(api_before)["fleet"]
    unbound = (
        harness.state.with_name("clean.failed-20260912T115617.json")
        if location == "adjacent-failed-record"
        else harness.state
    )
    unbound.write_text(
        json.dumps(
            {"phase": "QUEUES_DRAINED", "status": "FAILED", "fleet_snapshot": fleet}
        ),
        encoding="utf-8",
    )
    unbound.chmod(0o600)
    before = unbound.read_bytes()

    result = harness.run()

    assert result.returncode != 0, (
        "an unbound snapshot authorized stopped-plane cleanup"
    )
    assert unbound.read_bytes() == before, (
        "cleanup rewrote the untrusted prior evidence"
    )
    assert harness.api.read_bytes() == api_before, (
        "an unbound snapshot authorized a Kubernetes mutation"
    )
    assert not any("exec" in call["args"] for call in harness.calls()), (
        "cleanup attempted a database read without a Ready Pod"
    )
    if location == "adjacent-failed-record":
        assert "no running CPU pod" in result.stderr
        current = json.loads(harness.state.read_text())
        assert current["fleet_snapshot"] is None
        assert (current["phase"], current["status"]) == ("PREFLIGHT", "FAILED")
    else:
        assert "unsupported cleanup state schema" in result.stderr
        assert harness.calls() == [], "invalid state reached the Kubernetes API"


@pytest.mark.parametrize(
    ("drift", "error"),
    [
        ("config", "config_sha256"),
        ("cluster", "different cluster"),
        ("namespace", "namespace was replaced"),
        ("node", "changed UID"),
        ("legacy-schema", "legacy phase order requires reconciliation"),
    ],
)
def test_same_journal_restart_refuses_identity_or_legacy_order_drift(
    interrupted_reset: ResetRun, drift: str, error: str
) -> None:
    harness = interrupted_reset.harness
    if drift == "config":
        harness.config.write_text(harness.config.read_text() + "\n")
    elif drift == "legacy-schema":
        document = json.loads(harness.state.read_text())
        document["schema_version"] = 1
        harness.state.write_text(json.dumps(document))
    else:
        api = json.loads(harness.api.read_text())
        kind, name = {
            "cluster": ("namespace", "kube-system"),
            "namespace": ("namespace", "gpu-fault-system"),
            "node": ("node", "node-a"),
        }[drift]
        item = api["objects"][json.dumps(["gpu-a-context", "", kind, name])]
        item["metadata"]["uid"] += "-replacement"
        harness.api.write_text(json.dumps(api))
    api_before = harness.api.read_bytes()

    result = harness.run()

    assert result.returncode != 0, f"{drift} authorized cleanup from stale proof"
    assert error in result.stderr, result.stderr
    assert harness.api.read_bytes() == api_before, (
        "restart mutated Kubernetes after its original identity or ordering changed"
    )
    calls = harness.calls()[interrupted_reset.restart_at :]
    assert not any(
        verb in call["args"]
        for call in calls
        for verb in ("exec", "scale", "create", "delete", "patch")
    ), "invalid restart proof reached database or mutation commands"


def test_open_remote_commands_block_reset_without_orphan_settlement(
    tmp_path: Path,
) -> None:
    harness = _reset_harness(tmp_path)
    api = json.loads(harness.api.read_text())
    api["counts"] = [0, 1, 0, 0]
    harness.api.write_text(json.dumps(api))
    before = harness.api.read_bytes()

    result = harness.run()

    assert result.returncode != 0
    assert "open remote commands=1" in result.stderr, result.stderr
    assert harness.api.read_bytes() == before, "an open command authorized cleanup"
    assert not any(
        verb in call["args"]
        for call in harness.calls()
        for verb in ("scale", "create", "delete", "patch")
    ), "cleanup bypassed the outstanding-command barrier"
    state = json.loads(harness.state.read_text())
    assert (state["phase"], state["status"]) == ("PREFLIGHT", "FAILED")


@pytest.mark.parametrize("failure", ["before-publish", "after-publish"])
def test_registry_drain_failure_stops_before_producers_and_resumes_the_same_journal(
    tmp_path: Path, failure: str
) -> None:
    harness = _reset_harness(tmp_path)
    api = json.loads(harness.api.read_text())
    api["registry_failure"] = failure
    api["interrupt_node"] = False
    harness.api.write_text(json.dumps(api))

    first = harness.run()

    assert first.returncode != 0, "a failed registry publish authorized shutdown"
    failed = json.loads(harness.state.read_text())
    assert (failed["phase"], failed["status"]) == ("CLUSTERS_DRAINING", "FAILED")
    assert not any(
        verb in call["args"]
        for call in harness.calls()
        for verb in ("scale", "create", "delete", "patch")
    ), "a producer stopped without fleet convergence"
    api = json.loads(harness.api.read_text())
    api.pop("registry_failure")
    harness.api.write_text(json.dumps(api))

    resumed = harness.run()

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    completed = json.loads(harness.state.read_text())
    assert completed["run_id"] == failed["run_id"]
    assert completed["fleet_snapshot_sha256"] == failed["fleet_snapshot_sha256"]
    assert completed["history"][: len(failed["history"])] == failed["history"]
    assert completed["phase"] == "CLEANUP_COMPLETED"


@pytest.mark.parametrize(
    "counts", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)
def test_drain_leftovers_keep_ingress_consumers_executors_and_nodes_running(
    tmp_path: Path, counts: list[int]
) -> None:
    harness = _reset_harness(tmp_path)
    api = json.loads(harness.api.read_text())
    api["drain_counts"] = counts
    harness.api.write_text(json.dumps(api))

    result = harness.run(timeout_seconds=1)

    assert result.returncode != 0, "leftover records were treated as completion"
    state = json.loads(harness.state.read_text())
    assert (state["phase"], state["status"]) == ("QUEUES_DRAINED", "FAILED")
    assert "did not drain" in result.stderr, result.stderr
    calls = harness.calls()
    assert any("registry-drain" in call["args"] for call in calls), (
        "cleanup waited for queues without invoking the registry drain"
    )
    assert not any(
        "scale" in call["args"]
        and any(
            f"deployment/{name}" in call["args"]
            for name in (
                "gpu-fault-api-ha",
                "gpu-fault-control-worker",
                "gpu-fault-cluster-executor",
            )
        )
        or "create" in call["args"]
        for call in calls
    ), "unfinished or unknown work authorized runtime shutdown"


def test_resume_reasserts_registry_drain_and_does_not_reuse_stale_queue_evidence(
    tmp_path: Path,
) -> None:
    harness = _reset_harness(tmp_path)
    api = json.loads(harness.api.read_text())
    api["interrupt_consumer"] = True
    api["interrupt_node"] = False
    harness.api.write_text(json.dumps(api))
    first = harness.run()
    assert first.returncode != 0
    failed = json.loads(harness.state.read_text())
    assert (failed["phase"], failed["status"]) == (
        "CONTROL_CONSUMERS_STOPPED",
        "FAILED",
    )
    start = len(harness.calls())
    api = json.loads(harness.api.read_text())
    api["cluster_states"]["gpu-a"] = "ACTIVE"
    api["drain_counts"] = ["unknown", 0, 0, 0]
    harness.api.write_text(json.dumps(api))

    resumed = harness.run()

    assert resumed.returncode != 0, "old queue evidence authorized consumer shutdown"
    assert "could not parse Aurora drain snapshot" in resumed.stderr, resumed.stderr
    api = json.loads(harness.api.read_text())
    assert api["registry_drains"] == 2, "the old registry barrier was trusted on resume"
    assert api["cluster_states"]["gpu-a"] == "DRAINING"
    assert not any(
        verb in call["args"]
        for call in harness.calls()[start:]
        for verb in ("scale", "create", "delete", "patch")
    ), "unknown current queues authorized additional shutdown"


def test_namespace_and_resource_deletes_preserve_stop_order_and_identity(
    reset_run: ResetRun,
) -> None:
    calls = reset_run.harness.calls()

    def first(*arguments: str) -> int:
        return next(
            index
            for index, call in enumerate(calls)
            if all(argument in call["args"] for argument in arguments)
        )

    namespace_deletes = [
        index
        for index, call in enumerate(calls)
        if "delete" in call["args"]
        and "/api/v1/namespaces/gpu-fault-system" in call["args"]
    ]
    assert len(namespace_deletes) == 2, "one namespace delete per plane"
    contexts = set()
    for index in namespace_deletes:
        call = calls[index]
        args = call["args"]
        context = args[args.index("--context") + 1] if "--context" in args else "cpu"
        contexts.add(context)
        assert call["payload"]["preconditions"] == {"uid": f"namespace-{context}"}
        assert call["payload"]["propagationPolicy"] == "Foreground"
        assert "NODE_RUNTIMES_STOPPED" in call["completed"]
        assert "APPLICATION_OBJECTS_DELETED" in call["completed"]
    assert contexts == {"cpu", "gpu-a-context"}
    producers = (
        "gpu-fault-node-installer-reconciler",
        "gpu-fault-completion-watcher",
        "gpu-fault-kubernetes-node-resource-collector",
    )
    producer_deletes = [
        first("--context", "gpu-a-context", "delete", "deployment", name)
        for name in producers
    ]
    executor_delete = first(
        "--context",
        "gpu-a-context",
        "delete",
        "deployment",
        "gpu-fault-cluster-executor",
    )
    assert max(producer_deletes) < executor_delete, (
        "executor deletion overtook the producer resource wave"
    )
    ingress_delete = first("--kubeconfig", "delete", "deployment", "gpu-fault-api-ha")
    consumer_delete = first(
        "--kubeconfig", "delete", "deployment", "gpu-fault-control-worker"
    )
    assert ingress_delete < consumer_delete < min(namespace_deletes), (
        "namespace deletion preceded the CPU ingress/consumer resource waves"
    )
    assert executor_delete < min(namespace_deletes)


def test_node_cleanup_uncordons_declared_spares_before_stripping_their_label(
    reset_run: ResetRun,
) -> None:
    calls = reset_run.harness.calls()
    patches = [call for call in calls if "patch" in call["args"]]
    assert len(patches) == 1, "cleanup touched a node outside the captured fleet"
    call = patches[0]
    assert call["phase"] == "NAMESPACES_DELETED"
    assert "gpu-a-context" in call["args"] and "node-a" in call["args"]
    operations = call["patch"]
    assert operations[:2] == [
        {"op": "test", "path": "/metadata/uid", "value": "uid-gpu-a-context-node-a"},
        {"op": "test", "path": "/metadata/resourceVersion", "value": "1"},
    ], "spare restoration lost its Node UID/resourceVersion guards"
    restore = operations.index(
        {"op": "replace", "path": "/spec/unschedulable", "value": False}
    )
    strip = operations.index(
        {"op": "remove", "path": "/metadata/labels/gpu-fault.io~1spare"}
    )
    assert restore < strip, "the spare label was stripped before restoring scheduling"
    api = json.loads(reset_run.harness.api.read_text())
    node = api["objects"][json.dumps(["gpu-a-context", "", "node", "node-a"])]
    customer = api["objects"][
        json.dumps(["gpu-a-context", "", "node", "customer-node"])
    ]
    assert node["spec"]["unschedulable"] is False
    assert node["metadata"]["labels"] == {"customer.example/retained": "yes"}
    assert customer["spec"]["unschedulable"] is True, (
        "cleanup removed an unrelated node's cordon"
    )
