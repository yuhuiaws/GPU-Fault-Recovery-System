from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "control-plane" / "regional" / "prepare-clean-redeploy.sh"
UNINSTALLER = ROOT / "deploy" / "node" / "uninstall-gpu-fault-collector.sh"


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / "regional-release.json"
    path.write_text(
        json.dumps(
            {
                "cpu_kubeconfig": "/secure/cpu.kubeconfig",
                "namespace": "gpu-fault-system",
                "clusters": [
                    {"cluster_id": "gpu-a", "context": "gpu-a-context"},
                    {"cluster_id": "gpu-b", "context": "gpu-b-context"},
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


def test_execute_path_compiles_database_probe_and_preserves_stop_order(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
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
if "exec" in args:
    code_index = args.index("-c")
    code = args[code_index + 1]
    compile(code, "<clean-redeploy-probe>", "exec")
    print("[]" if "WHERE kind='agent'" in code else "0\\t0\\t0\\t0")
    raise SystemExit(0)
if "get" in args and "--raw=/readyz" in args:
    print("ok")
    raise SystemExit(0)
if (
    "get" in args
    and "configmap" in args
    and "gpu-fault-installed-resources" in args
):
    print("Error from server (NotFound): configmap not found", file=sys.stderr)
    raise SystemExit(1)
if (
    "get" in args
    and "-o" in args
    and "json" in args
    and (
        args.index("-o") == args.index("get") + 2
        or any("," in argument for argument in args)
        or "namespace" in args
    )
):
    items = []
    kind = args[args.index("get") + 1]
    if "," not in kind and kind == "deployment":
        items = [
            {"metadata": {"name": "gpu-fault-api-ha"}},
            {"metadata": {"name": "gpu-fault-control-worker"}},
            {"metadata": {"name": "gpu-fault-cluster-executor"}},
            {"metadata": {"name": "gpu-fault-completion-watcher"}},
            {
                "metadata": {
                    "name": "gpu-fault-kubernetes-node-resource-collector"
                }
            },
            {"metadata": {"name": "gpu-fault-node-installer-reconciler"}},
        ]
    print(json.dumps({"items": items}))
    raise SystemExit(0)
if "get" in args and "pod" in args:
    print("fake-database-pod", end="")
    raise SystemExit(0)
if "get" in args and "deployment" in args:
    print("0", end="")
    raise SystemExit(0)
if "get" in args and "nodes" in args:
    print("fake-node Ready")
    raise SystemExit(0)
if "get" in args and ("daemonset" in args or "cronjob" in args):
    print("Error from server (NotFound): resource not found", file=sys.stderr)
    raise SystemExit(1)
raise SystemExit(0)
""",
        encoding="utf-8",
    )
    fake_kubectl.chmod(0o755)
    kubeconfig = tmp_path / "cpu.kubeconfig"
    kubeconfig.write_text("test", encoding="utf-8")
    config = tmp_path / "regional-release.json"
    config.write_text(
        json.dumps(
            {
                "cpu_kubeconfig": str(kubeconfig),
                "namespace": "gpu-fault-system",
                "clusters": [{"cluster_id": "gpu-a", "context": "gpu-a-context"}],
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "state" / "clean.json"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_KUBECTL_LOG": str(log),
    }

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
    calls = log.read_text(encoding="utf-8")
    producer = calls.index(
        "scale deployment/gpu-fault-node-installer-reconciler --replicas=0"
    )
    ingress = calls.index("scale deployment/gpu-fault-api-ha --replicas=0")
    worker = calls.index("scale deployment/gpu-fault-control-worker --replicas=0")
    executor = calls.index("scale deployment/gpu-fault-cluster-executor --replicas=0")
    assert producer < ingress < worker < executor, calls


def test_node_components_stop_only_after_the_executors_and_the_drain() -> None:
    """Live uninstall, 2026-09-12: the node cleanup ran inside
    ``stop_gpu_producers``, before ingress went down and while the executors
    were still claiming. Uninstalling the agents produced incidents, the
    executors leased their commands, ingress was then stopped, and the drain
    waited on three LEASED commands nobody could ever complete. Node
    components now stop after the drain and after the executors, and a reset
    fails the orphaned leases before the safety assertion and before the
    node cleanup."""

    text = SCRIPT.read_text(encoding="utf-8")
    producers_start = text.index("stop_gpu_producers() {")
    producers_end = text.index("\n}\n", producers_start)
    assert "run_node_cleanup" not in text[producers_start:producers_end], (
        "node cleanup must not run while the control plane can still react"
    )
    executors_stopped = text.index("GPU_EXECUTORS_STOPPED IN_PROGRESS")
    executors_done = text.index("GPU_EXECUTORS_STOPPED COMPLETED")
    node_cleanup_call = text.index('run_node_cleanup "${CLUSTER_IDS[index]}"')
    assert executors_stopped < node_cleanup_call < executors_done, (
        "node cleanup must run inside the executor stop phase, after the scale"
    )
    scale = text.index(
        'scale_gpu_deployment_zero "${context}" "${deployment}"', executors_stopped
    )
    assert scale < node_cleanup_call
    orphan_before_assert = text.index("fail_orphaned_remote_commands")
    first_assert = text.index('assert_no_active_work "${DATABASE_POD}"')
    assert (
        text.index("fail_orphaned_remote_commands", first_assert - 400) < first_assert
    ), "a resumed reset must fail orphaned leases before the safety assertion"
    assert orphan_before_assert < first_assert
    assert "payload->>'\\''status'\\'' = '\\''LEASED'\\''" in text, (
        "orphan handling must target LEASED commands only"
    )
    assert "(payload->>'\\''lease_expires_at'\\'')::timestamptz <" in text, (
        "orphan handling must require a lapsed lease"
    )
    assert "clean-redeploy-orphan" in text, (
        "orphaned commands must carry an audit source"
    )


FAKE_KUBECTL_STOPPED_PLANE = """#!/usr/bin/env python3
import json
import os
import re
import sys

args = sys.argv[1:]
with open(os.environ["FAKE_KUBECTL_LOG"], "a", encoding="utf-8") as stream:
    stream.write(" ".join(args) + "\\n")

# A minimal API server memory: an object deleted here is NotFound afterwards
# (silently absent under --ignore-not-found), so the script's own polls for
# disappearance terminate the way they do against a real cluster.
context = args[args.index("--context") + 1] if "--context" in args else "cpu"
deleted_path = os.environ["FAKE_KUBECTL_LOG"] + ".deleted"
deleted = set()
if os.path.exists(deleted_path):
    with open(deleted_path, encoding="utf-8") as stream:
        deleted = set(stream.read().split())


def object_key(verb):
    rest = [item for item in args[args.index(verb) + 1 :] if not item.startswith("-")]
    return f"{context}:{rest[0]}:{rest[1]}" if len(rest) >= 2 else None


if "delete" in args:
    key = object_key("delete")
    if key is not None:
        with open(deleted_path, "a", encoding="utf-8") as stream:
            stream.write(key + "\\n")
    raise SystemExit(0)
if "apply" in args:
    manifest = sys.stdin.read()
    kind = re.search(r"^kind: (\\S+)", manifest, re.M)
    name = re.search(r"^  name: (\\S+)", manifest, re.M)
    if kind and name:
        key = f"{context}:{kind.group(1).lower()}:{name.group(1)}"
        deleted.discard(key)
        with open(deleted_path, "w", encoding="utf-8") as stream:
            stream.write("\\n".join(sorted(deleted)) + "\\n")
    raise SystemExit(0)
if "get" in args and object_key("get") in deleted:
    if "--ignore-not-found" in args:
        raise SystemExit(0)
    print("Error from server (NotFound): object not found", file=sys.stderr)
    raise SystemExit(1)
if "exec" in args:
    raise SystemExit("no Pod is running: exec must not be attempted")
if "get" in args and "--raw=/readyz" in args:
    print("ok")
    raise SystemExit(0)
if "get" in args and "pod" in args:
    print("", end="")
    raise SystemExit(0)
if (
    "get" in args
    and "configmap" in args
    and "gpu-fault-installed-resources" in args
):
    print("Error from server (NotFound): configmap not found", file=sys.stderr)
    raise SystemExit(1)
if (
    "get" in args
    and "daemonset" in args
    and "-o" in args
    and "jsonpath" in " ".join(args)
):
    print("1", end="")
    raise SystemExit(0)
if (
    "get" in args
    and "-o" in args
    and "json" in args
    and (
        args.index("-o") == args.index("get") + 2
        or any("," in argument for argument in args)
        or "namespace" in args
    )
):
    items = []
    kind = args[args.index("get") + 1]
    if "," not in kind and kind == "deployment":
        items = [
            {"metadata": {"name": name}}
            for name in (
                "gpu-fault-api-ha",
                "gpu-fault-control-worker",
                "gpu-fault-cluster-executor",
                "gpu-fault-completion-watcher",
                "gpu-fault-kubernetes-node-resource-collector",
                "gpu-fault-node-installer-reconciler",
            )
        ]
    print(json.dumps({"items": items}))
    raise SystemExit(0)
if "get" in args and "deployment" in args:
    print("0", end="")
    raise SystemExit(0)
if "get" in args and "nodes" in args:
    print("fake-node Ready")
    raise SystemExit(0)
if "get" in args and ("daemonset" in args or "cronjob" in args):
    print("Error from server (NotFound): resource not found", file=sys.stderr)
    raise SystemExit(1)
raise SystemExit(0)
"""


def _run_reset_over_stopped_plane(
    tmp_path: Path,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, list[dict[str, object]]]:
    """Execute a reset against the stopped-plane fake with one GPU cluster.

    Returns the process, the state file, the kubectl call log, and the fleet
    snapshot the earlier (failed) record carried.
    """

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "kubectl.log"
    fake_kubectl = fake_bin / "kubectl"
    fake_kubectl.write_text(FAKE_KUBECTL_STOPPED_PLANE, encoding="utf-8")
    fake_kubectl.chmod(0o755)
    kubeconfig = tmp_path / "cpu.kubeconfig"
    kubeconfig.write_text("test", encoding="utf-8")
    config = tmp_path / "regional-release.json"
    config.write_text(
        json.dumps(
            {
                "cpu_kubeconfig": str(kubeconfig),
                "namespace": "gpu-fault-system",
                "clusters": [{"cluster_id": "gpu-a", "context": "gpu-a-context"}],
            }
        ),
        encoding="utf-8",
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    fleet = [
        {
            "cluster_id": "gpu-a",
            "node_id": "node-a",
            "lifecycle_state": "ACTIVE",
            "installed_unit_inventory": {"units": ["gpu-fault-node-agent.service"]},
        }
    ]
    (state_dir / "clean.failed-20260912T115617.json").write_text(
        json.dumps(
            {"phase": "QUEUES_DRAINED", "status": "FAILED", "fleet_snapshot": fleet}
        ),
        encoding="utf-8",
    )
    state = state_dir / "clean.json"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_KUBECTL_LOG": str(log),
    }

    result = run_script(
        "--config",
        str(config),
        "--mode",
        "reset",
        "--node-mode",
        "uninstall",
        "--state-file",
        str(state),
        "--confirm-reset",
        "RESET_GPU_FAULT_INSTALLATION",
        "--execute",
        env=env,
    )
    return result, state, log, fleet


def test_a_reset_resumes_over_a_stopped_control_plane_with_the_earlier_fleet_snapshot(
    tmp_path: Path,
) -> None:
    """Live uninstall, 2026-09-12: the first reset stopped ingress and the
    consumers, then failed; the rerun died with "reset requires a running
    control-plane pod to export fleet inventory" because no Pod was left to
    export it, and it would have died again waiting for a drain no Pod can
    report. The failed record the caller moved aside still carries the
    export: a reset over a stopped control plane reuses it, skips the
    Aurora checks nothing can invalidate, and completes."""

    result, state, log, fleet = _run_reset_over_stopped_plane(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "reused its fleet inventory (1 agent record(s))"
        in result.stderr + result.stdout
    )
    document = json.loads(state.read_text(encoding="utf-8"))
    assert document["phase"] == "CLEANUP_COMPLETED", document["phase"]
    assert document["fleet_snapshot"] == fleet, (
        "the earlier export must be carried over"
    )
    calls = log.read_text(encoding="utf-8")
    assert " exec " not in f" {calls} ", "no drain probe may be attempted without a Pod"
    assert "delete namespace gpu-fault-system" in calls


def test_namespace_and_wave_deletes_are_issued_before_their_waits(
    tmp_path: Path,
) -> None:
    """Live uninstall, 2026-09-12: deleting the application objects one by one
    with kubectl's own wait cost ~50 s, and the two namespaces ~49 s more. A
    wave's deletes are all issued with --wait=false before the wave is polled
    once, a later wave starts only after the earlier one is gone, and both
    planes' namespace deletes go out before either namespace is waited on."""

    result, _state, log, _fleet = _run_reset_over_stopped_plane(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()

    def first(*fragments: str) -> int:
        return next(
            index
            for index, call in enumerate(calls)
            if all(fragment in call for fragment in fragments)
        )

    namespace_deletes = [
        index
        for index, call in enumerate(calls)
        if "delete namespace gpu-fault-system" in call
    ]
    assert len(namespace_deletes) == 2, "one namespace delete per plane"
    assert all(
        "--wait=false" in calls[index] and "--timeout" not in calls[index]
        for index in namespace_deletes
    ), "namespace deletes must not wait one at a time"
    namespace_poll = first("get namespace gpu-fault-system", "--ignore-not-found")
    assert max(namespace_deletes) < namespace_poll, (
        "both planes' namespace deletes go out before either is waited on"
    )

    # The fake reports three GPU producer Deployments and one executor, so the
    # producer wave has three members and the executor wave follows it.
    producers = (
        "gpu-fault-node-installer-reconciler",
        "gpu-fault-completion-watcher",
        "gpu-fault-kubernetes-node-resource-collector",
    )
    producer_deletes = [
        first("--context gpu-a-context", f"delete deployment {name}", "--wait=false")
        for name in producers
    ]
    producer_polls = [
        first("--context gpu-a-context", f"get deployment {name}", "--ignore-not-found")
        for name in producers
    ]
    assert max(producer_deletes) < min(producer_polls), (
        "every delete of a wave is issued before the wave is polled once"
    )
    executor_delete = first(
        "--context gpu-a-context",
        "delete deployment gpu-fault-cluster-executor",
        "--wait=false",
    )
    assert max(producer_polls) < executor_delete, (
        "a later wave starts only after the earlier wave is gone"
    )
    ingress_delete = first("--kubeconfig", "delete deployment gpu-fault-api-ha")
    ingress_poll = first(
        "--kubeconfig", "get deployment gpu-fault-api-ha", "--ignore-not-found"
    )
    consumer_delete = first(
        "--kubeconfig", "delete deployment gpu-fault-control-worker"
    )
    assert ingress_delete < ingress_poll < consumer_delete, (
        "the CPU ingress wave is gone before the consumer wave starts"
    )
