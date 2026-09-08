"""Data-plane manifest contracts from the 2026-09-08 review (task 16).

Every check names the outage it exists to stop, because a manifest scan that
proves nothing looks exactly like a fixed manifest:

* The DCGM exporter DaemonSet pinned ``nodeSelector`` to ``ml.p5en.48xlarge``
  and regional rendering substituted only namespace and image, so a fleet on
  any other supported GPU type got zero exporter Pods: ``rollout status``
  passed trivially, ``dcgm_ready`` failed inside the node installer, the
  install Job failed and the reconciler recreated it every 300 s forever
  (``gpu-metrics-collectors §F4``, ``hma-installer-deploy §F1``). The allowed
  list is rendered from ``node_installer_reconciler._INVENTORY`` -- the one
  table the installer already sizes GPU/EFA counts from -- so a new supported
  type cannot be added to the fleet without the exporter following, and an
  unsupported type still gets no Pod at all.
* The exporter bound ``0.0.0.0:9400`` under ``hostNetwork`` while every
  consumer dials loopback, publishing GPU UUIDs and hostnames to the VPC
  unauthenticated (``§F8``), and collected on DCGM's 30 s default while the
  collector scrapes every 15 s, which makes real power throttling
  un-gradable (owner decision 3: ``-c 15000`` on *both* launch paths).
* A half-open apiserver connection wedges a controller loop while the Pod
  stays Running and Ready (``hma-installer-deploy §F6``). Liveness therefore
  reads the age of a file the loop refreshes every cycle, not a start-up
  breadcrumb, and the thresholds are wide enough that a healthy idle
  component is never restarted.
* The reconciler tolerated every taint with no ``tolerationSeconds``
  (``§F7``), so taint-based eviction never moved the singleton off a dead
  node.
* The optional HMA manifests still posted to ``gpu-fault-api-canary`` with no
  token, which the regional API answers with 401 -- a non-retryable error on
  the very first post (``§F11``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import yaml

from gpu_fault.node_installer_reconciler import _INVENTORY as RECONCILER_INVENTORY
from gpu_fault_release import regional_gpu_bootstrap as BOOTSTRAP
from tests.regional._release_orchestrator_support import RELEASE_MODULE as MODULE
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]
DATAPLANE = ROOT / "deploy" / "dataplane"
SYSTEMD = ROOT / "deploy" / "systemd"
#: Liveness must never fire on a component that is merely idle: the heartbeat
#: threshold is the review's floor, and the probe period the review's cadence.
MINIMUM_HEARTBEAT_SECONDS = 300
LIVENESS_PERIOD_SECONDS = 30


class _RecordingRunner:
    """Captures every manifest the release hands to ``kubectl apply``."""

    dry_run = False

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(self, arguments: Any, **kwargs: Any) -> str:
        self.calls.append((list(arguments), dict(kwargs)))
        if "create" in arguments and "configmap" in arguments:
            return (
                "apiVersion: v1\nkind: ConfigMap\nmetadata:\n"
                "  name: gpu-fault-dcgm-counters\ndata:\n"
                "  gpu-fault-counters.csv: test\n"
            )
        return ""


def _documents(path: Path) -> list[dict[str, Any]]:
    return [
        document
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if isinstance(document, dict)
    ]


def _workloads(paths: Iterator[Path], kind: str) -> list[tuple[Path, dict[str, Any]]]:
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        for document in _documents(path):
            if document.get("kind") == kind:
                found.append((path, document))
    return found


def _pod_spec(workload: dict[str, Any]) -> dict[str, Any]:
    return ((workload.get("spec") or {}).get("template") or {}).get("spec") or {}


def _rendered_exporter(tmp_path: Path) -> dict[str, Any]:
    """The DaemonSet exactly as the regional deploy applies it.

    Rendered through the public preflight entry point rather than read off
    disk, because the whole defect was that rendering dropped the part of the
    manifest that decides where the Pods land.
    """

    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    runner = _RecordingRunner()
    release = MODULE.RegionalRelease(config, runner)
    release.dcgm_exporter_image = "registry.example/dcgm@sha256:" + "b" * 64

    BOOTSTRAP.preflight_gpu_dcgm_exporter(release, config.clusters[0])

    daemon_sets = [
        document
        for _arguments, kwargs in runner.calls
        for document in yaml.safe_load_all(kwargs.get("input_text") or "")
        if isinstance(document, dict) and document.get("kind") == "DaemonSet"
    ]
    assert len(daemon_sets) == 1, (
        f"expected exactly one rendered exporter DaemonSet, got {len(daemon_sets)}"
    )
    return daemon_sets[0]


def _instance_type_values(spec: dict[str, Any]) -> list[str]:
    affinity = spec.get("affinity") or {}
    node_affinity = affinity.get("nodeAffinity") or {}
    required = node_affinity.get("requiredDuringSchedulingIgnoredDuringExecution") or {}
    values: list[str] = []
    for term in required.get("nodeSelectorTerms") or []:
        for expression in term.get("matchExpressions") or []:
            if expression.get("key") != "node.kubernetes.io/instance-type":
                continue
            assert expression.get("operator") == "In", (
                "an exporter node affinity that is not an In list is not "
                f"fail-closed: {expression}"
            )
            values.extend(str(value) for value in expression.get("values") or [])
    return values


def test_dcgm_exporter_daemonset_matches_every_reconciler_instance_type(
    tmp_path: Path,
) -> None:
    daemon_set = _rendered_exporter(tmp_path)
    spec = _pod_spec(daemon_set)

    assert "nodeSelector" not in spec, (
        "a single-value nodeSelector is what pinned the exporter to one "
        "instance type; the affinity list replaces it"
    )
    values = set(_instance_type_values(spec))
    assert values, "the rendered DaemonSet has no instance-type node affinity"
    expected = {f"ml.{name}" for name in RECONCILER_INVENTORY} | set(
        RECONCILER_INVENTORY
    )
    assert values == expected, (
        "the exporter must schedule on exactly the instance types the node "
        "installer supports -- a missing type gets no exporter and can never "
        "finish an install, an extra one would schedule where no GPU exists: "
        f"missing={sorted(expected - values)} extra={sorted(values - expected)}"
    )
    assert daemon_set["spec"]["template"]["spec"]["priorityClassName"] == (
        "system-node-critical"
    ), "the exporter must not be preempted off a GPU node by training work"


def test_dcgm_exporter_collects_every_15_seconds_on_both_launch_paths(
    tmp_path: Path,
) -> None:
    daemon_set = _rendered_exporter(tmp_path)
    container = _pod_spec(daemon_set)["containers"][0]
    unit = (SYSTEMD / "gpu-fault-dcgm-exporter.service").read_text(encoding="utf-8")

    arguments = [str(value) for value in container.get("args") or []]
    assert "-c" in arguments, (
        "without -c the exporter keeps DCGM's 30 s default while the collector "
        "scrapes every 15 s, so half of every throttling episode is invisible"
    )
    assert arguments[arguments.index("-c") + 1] == "15000", (
        f"the DaemonSet must collect every 15 000 ms, got {arguments}"
    )
    assert "-c 15000" in unit, (
        "the systemd launch path must collect on the same cadence as the "
        "DaemonSet, or a node's grading depends on how its exporter started"
    )


def test_exporter_binds_loopback(tmp_path: Path) -> None:
    daemon_set = _rendered_exporter(tmp_path)
    container = _pod_spec(daemon_set)["containers"][0]

    arguments = [str(value) for value in container.get("args") or []]
    assert "-a" in arguments, f"the exporter must bind explicitly, got {arguments}"
    assert arguments[arguments.index("-a") + 1] == "127.0.0.1:9400", (
        "under hostNetwork a 0.0.0.0 bind publishes GPU UUIDs and hostnames to "
        f"the VPC unauthenticated, and no consumer needs it: {arguments}"
    )
    assert _pod_spec(daemon_set).get("hostNetwork") is True, (
        "the loopback bind is only reachable by the node's collector because "
        "the Pod shares the host network namespace"
    )


def _liveness_heartbeat_seconds(probe: dict[str, Any]) -> int:
    command = " ".join(str(value) for value in (probe.get("exec") or {})["command"])
    digits = [
        int(word)
        for word in command.replace("<", " ").replace(")", " ").split()
        if word.isdigit()
    ]
    assert digits, f"the liveness probe reads no age threshold: {command}"
    return max(digits)


def _assert_liveness_is_local(where: str, probe: dict[str, Any]) -> None:
    """A liveness probe must answer from the Pod, never from the control plane.

    A regional outage that failed every probe would restart both replicas of
    every cluster at once, which is the one moment the data plane must not
    lose its claim loops.
    """

    http = probe.get("httpGet")
    if http:
        assert not http.get("host"), (
            f"{where}: liveness must query the Pod itself, not {http['host']}"
        )
        return
    assert _liveness_heartbeat_seconds(probe) >= MINIMUM_HEARTBEAT_SECONDS, (
        f"{where}: a threshold below {MINIMUM_HEARTBEAT_SECONDS}s restarts a "
        "component that is merely idle"
    )


def test_every_data_plane_deployment_has_a_liveness_probe() -> None:
    """Readiness cannot answer "is the loop turning".

    ``optional/`` is deliberately out of scope: those components are absent
    from the regional inventory and are never deployed by the release.
    """

    deployments = _workloads(iter(sorted(DATAPLANE.glob("*.yaml"))), "Deployment")

    assert len(deployments) == 4, (
        f"expected the four regional data-plane Deployments, got {deployments}"
    )
    for path, deployment in deployments:
        for container in _pod_spec(deployment).get("containers") or []:
            where = f"{path.name}:{container.get('name')}"
            probe = container.get("livenessProbe")
            assert probe, (
                f"{where}: a wedged loop keeps the Pod Running and Ready "
                "forever without a liveness probe"
            )
            assert probe.get("periodSeconds") == LIVENESS_PERIOD_SECONDS, (
                f"{where}: liveness must be probed every {LIVENESS_PERIOD_SECONDS}s"
            )
            _assert_liveness_is_local(where, probe)


def test_liveness_heartbeat_directory_is_writable() -> None:
    """A read-only root filesystem turns a heartbeat probe into a restart loop.

    The node-resources collector runs with ``readOnlyRootFilesystem: true``, so
    the directory its heartbeat lives in has to come from a writable volume;
    otherwise the write fails with EROFS, the probe never sees a fresh file and
    the Pod is restarted every few minutes forever.
    """

    checked = 0
    for path, deployment in _workloads(
        iter(sorted(DATAPLANE.glob("*.yaml"))), "Deployment"
    ):
        spec = _pod_spec(deployment)
        for container in spec.get("containers") or []:
            probe = container.get("livenessProbe") or {}
            if not probe.get("exec"):
                continue
            security = container.get("securityContext") or {}
            if not security.get("readOnlyRootFilesystem"):
                continue
            command = " ".join(str(value) for value in probe["exec"]["command"])
            mounts = {
                str(mount["mountPath"])
                for mount in container.get("volumeMounts") or []
                if not mount.get("readOnly")
            }
            assert any(f'"{mount}/' in command for mount in mounts), (
                f"{path.name}:{container.get('name')}: the heartbeat path in "
                f"{command} is not under a writable mount ({sorted(mounts)})"
            )
            checked += 1
    assert checked >= 1, "no read-only-rootfs Deployment with an exec probe was seen"


def test_reconciler_reads_pods_through_a_namespaced_role_only() -> None:
    """Classifying a never-started Pod needs ``pods`` read, and nothing more.

    Without it the reconciler silently falls back to plain backoff and the
    never-started verdict is dead code; with ``watch`` or ``delete`` the
    identity would gain a lever it has no use for.
    """

    path = DATAPLANE / "node-installer-reconciler.yaml"
    roles = {
        (document["kind"], (document.get("metadata") or {}).get("namespace")): document
        for document in _documents(path)
        if document.get("kind") in {"Role", "ClusterRole"}
    }
    namespaced = roles[("Role", "gpu-fault-system")]
    cluster = roles[("ClusterRole", None)]

    pod_rules = [
        rule for rule in namespaced["rules"] if "pods" in (rule.get("resources") or [])
    ]
    assert len(pod_rules) == 1, (
        f"expected exactly one namespaced pods rule, got {pod_rules}"
    )
    assert sorted(pod_rules[0]["verbs"]) == ["get", "list"], (
        f"the reconciler only reads Pod waiting reasons: {pod_rules[0]}"
    )
    assert all(
        "pods" not in (rule.get("resources") or []) for rule in cluster["rules"]
    ), "reading Pods cluster-wide is not needed; the installer Jobs are namespaced"


def test_reconciler_tolerations_are_bounded() -> None:
    path = DATAPLANE / "node-installer-reconciler.yaml"
    ((_path, deployment),) = _workloads(iter([path]), "Deployment")
    tolerations = _pod_spec(deployment).get("tolerations") or []

    assert not any(set(toleration) == {"operator"} for toleration in tolerations), (
        "a bare `operator: Exists` tolerates not-ready and unreachable "
        "forever, so the singleton never leaves a dead node"
    )
    keys = {toleration.get("key"): toleration for toleration in tolerations}
    assert set(keys) == {
        "gpu-fault.io/quarantined",
        "node.kubernetes.io/unschedulable",
        "node.kubernetes.io/not-ready",
        "node.kubernetes.io/unreachable",
    }, f"unexpected reconciler tolerations: {sorted(keys)}"
    for key in ("node.kubernetes.io/not-ready", "node.kubernetes.io/unreachable"):
        assert keys[key].get("tolerationSeconds") == 60, (
            f"{key} must be tolerated only briefly: {keys[key]}"
        )


#: ``completion-watcher.yaml`` is a singleton too, but its manifest belongs to
#: another work package in this review; its missing priority class is reported
#: rather than changed here.
PRIORITY_CLASS_SINGLETONS = (
    "kubernetes-node-resource-collector.yaml",
    "node-installer-reconciler.yaml",
)


def test_singleton_deployments_declare_a_priority_class() -> None:
    """A preempted singleton is a silent data-plane outage."""

    checked = 0
    for path, deployment in _workloads(
        iter(sorted(DATAPLANE.glob("*.yaml"))), "Deployment"
    ):
        if path.name not in PRIORITY_CLASS_SINGLETONS:
            continue
        assert (deployment.get("spec") or {}).get("replicas") == 1, (
            f"{path.name} is no longer a singleton; revisit its priority class"
        )
        priority = _pod_spec(deployment).get("priorityClassName")
        assert priority == "system-cluster-critical", (
            f"{path.name}: singleton without a critical priority class, got "
            f"{priority!r}"
        )
        checked += 1
    assert checked == len(PRIORITY_CLASS_SINGLETONS), (
        f"expected {len(PRIORITY_CLASS_SINGLETONS)} singletons, saw {checked}"
    )


def test_executor_finishes_its_command_before_the_kubelet_kills_it() -> None:
    """The executor handles SIGTERM itself; the default 30 s grace cuts it off."""

    path = DATAPLANE / "cluster-action-executor.yaml"
    ((_path, deployment),) = _workloads(iter([path]), "Deployment")

    assert _pod_spec(deployment).get("terminationGracePeriodSeconds") == 60, (
        "the executor needs longer than the 30 s default to finish the batch "
        "it already claimed"
    )


def _optional_containers() -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    found = []
    for path, deployment in _workloads(
        iter(sorted((DATAPLANE / "optional").glob("*.yaml"))), "Deployment"
    ):
        spec = _pod_spec(deployment)
        for container in spec.get("containers") or []:
            found.append((path, spec, container))
    return found


def test_optional_hma_manifests_use_the_regional_connection_secret() -> None:
    containers = _optional_containers()

    assert len(containers) == 2, f"expected two optional collectors, got {containers}"
    for path, spec, container in containers:
        where = f"{path.name}:{container.get('name')}"
        env = {item["name"]: item for item in container.get("env") or []}
        for name in (
            "GPU_FAULT_CONTROL_PLANE_URL",
            "GPU_FAULT_CONTROL_PLANE_TOKEN",
            "GPU_FAULT_CLUSTER_ID",
        ):
            source = (env.get(name) or {}).get("valueFrom") or {}
            secret = source.get("secretKeyRef") or {}
            assert secret.get("name") == "gpu-fault-regional-connection", (
                f"{where}: {name} must come from the regional connection "
                f"Secret, got {env.get(name)!r} -- the canary URL answers 401 "
                "and a 401 is not retried"
            )
        assert (env.get("SSL_CERT_FILE") or {}).get("value") == (
            "/etc/gpu-fault/tls/ca.crt"
        ), f"{where}: the control-plane CA is not trusted"
        outbox = (env.get("GPU_FAULT_COLLECTOR_OUTBOX_PATH") or {}).get("value")
        assert outbox and outbox.startswith("/var/lib/gpu-fault/"), (
            f"{where}: without an outbox path every buffered event is dropped, "
            f"got {outbox!r}"
        )
        mounts = {
            str(mount["mountPath"]) for mount in container.get("volumeMounts") or []
        }
        assert any(outbox.startswith(f"{mount}/") for mount in mounts), (
            f"{where}: the outbox path {outbox} is not under a mounted volume"
        )
        assert "/etc/gpu-fault/tls" in mounts, f"{where}: the CA volume is not mounted"
        declared = {volume["name"] for volume in spec.get("volumes") or []}
        mounted = {str(mount["name"]) for mount in container.get("volumeMounts") or []}
        assert declared == mounted, (
            f"{where}: declared volumes {sorted(declared)} do not match the "
            f"mounted ones {sorted(mounted)}"
        )
    texts = [path.read_text(encoding="utf-8") for path, _spec, _c in containers]
    assert not any("gpu-fault-api-canary" in text for text in texts), (
        "the canary Service does not exist in the regional architecture"
    )
