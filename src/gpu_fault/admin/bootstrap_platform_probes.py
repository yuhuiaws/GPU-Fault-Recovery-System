"""Read-only probes for the platform-prerequisite bootstrap tasks.

The monitoring installer, the Aurora credential refresh CronJob and the Node
Action keys converge live state that no static digest can fully describe, so
their checkpoints are re-proved on every deploy instead of being trusted. Each
probe reads the facts its ensure step would converge and raises
`BootstrapMutationRequired` on the first difference, which is what
`run_parallel` turns into a real ensure run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapMutationRequired,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault.admin.node_key_proof import read_node_key_proof
from gpu_fault_release.regional_admin_commands import BOOTSTRAP_PHASES


def kubectl_projection(
    runner: CommandRunner,
    *,
    kubeconfig: Path,
    namespace: str,
    arguments: list[str],
    context: str | None = None,
) -> str:
    """Read one projection of a live object, treating absence as an empty value.

    Probes must never fail on a missing object: `run_parallel` only understands
    `BootstrapMutationRequired`, so the caller decides that an empty read means
    the ensure step still has work to do.
    """

    command = ["kubectl", "--kubeconfig", str(kubeconfig)]
    if context:
        command += ["--context", context]
    command += ["-n", namespace, *arguments, "--ignore-not-found"]
    return runner.run(command, capture=True).strip()


AMP_RULE_NAMESPACE = "gpu-fault-control-plane-capacity"


def release_owns_monitoring(
    runner: CommandRunner, *, cpu_kubeconfig: Path, namespace: str
) -> bool:
    raw = kubectl_projection(
        runner,
        kubeconfig=cpu_kubeconfig,
        namespace=namespace,
        arguments=[
            "get",
            "configmap",
            "gpu-fault-regional-release-state",
            "-o",
            "json",
        ],
    )
    if not raw:
        return False
    try:
        document = json.loads(raw)
        state = json.loads(document["data"]["state.json"])
    except (ValueError, TypeError, KeyError) as exc:
        raise BootstrapError(
            "cannot determine monitoring owner: invalid release state"
        ) from exc
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("phase"), str)
        or not state["phase"]
        or not isinstance(state.get("release_id"), str)
        or not state["release_id"]
    ):
        raise BootstrapError(
            "cannot determine monitoring owner: incomplete release state"
        )
    return state["phase"] not in BOOTSTRAP_PHASES


def assert_monitoring_install_current(
    runner: CommandRunner,
    *,
    cpu_kubeconfig: Path,
    namespace: str,
    region: str,
    adot_image: str,
    workspace_id: str,
) -> None:
    """Assert the installer's converged resources still match this release.

    The installer renders the ADOT collector from the workspace ID, the Region
    and the collector image, and it publishes the AMP rule groups namespace and
    alert manager definition. The probe reads those four facts back; the rendered
    file contents themselves are covered by the task digest, so what is left to
    detect here is a recreated workspace, a rolled-back image, a scaled-down
    collector or a deleted AMP definition.
    """

    image = kubectl_projection(
        runner,
        kubeconfig=cpu_kubeconfig,
        namespace=namespace,
        arguments=[
            "get",
            "deployment",
            "gpu-fault-adot",
            "-o",
            "jsonpath={.spec.template.spec.containers[0].image}",
        ],
    )
    if image != adot_image:
        raise BootstrapMutationRequired("gpu-fault-adot image")
    available = kubectl_projection(
        runner,
        kubeconfig=cpu_kubeconfig,
        namespace=namespace,
        arguments=[
            "get",
            "deployment",
            "gpu-fault-adot",
            "-o",
            "jsonpath={.status.availableReplicas}",
        ],
    )
    if int(available or 0) < 1:
        raise BootstrapMutationRequired("gpu-fault-adot replicas")
    collector_config = kubectl_projection(
        runner,
        kubeconfig=cpu_kubeconfig,
        namespace=namespace,
        arguments=[
            "get",
            "configmap",
            "gpu-fault-adot",
            "-o",
            "jsonpath={.data}",
        ],
    )
    endpoint = (
        f"https://aps-workspaces.{region}.amazonaws.com/"
        f"workspaces/{workspace_id}/api/v1/remote_write"
    )
    if endpoint not in collector_config:
        raise BootstrapMutationRequired("gpu-fault-adot remote write endpoint")
    rule_namespaces = runner.aws_json(
        region,
        "amp",
        "list-rule-groups-namespaces",
        "--workspace-id",
        workspace_id,
        "--name",
        AMP_RULE_NAMESPACE,
    )
    if not rule_namespaces.get("ruleGroupsNamespaces"):
        raise BootstrapMutationRequired(AMP_RULE_NAMESPACE)
    try:
        runner.aws_json(
            region,
            "amp",
            "describe-alert-manager-definition",
            "--workspace-id",
            workspace_id,
        )
    except BootstrapError:
        # There is no list form for the alert manager definition, so an absent
        # definition surfaces as a failed read. Treating any failed read as work
        # for the ensure step keeps this fail-closed: the installer re-runs and
        # reports the real error if the read failed for any other reason.
        raise BootstrapMutationRequired("amp alert manager definition") from None


AURORA_REFRESH_CRONJOB = "gpu-fault-aurora-credential-refresh"
# One kubectl jsonpath expression, deliberately left unterminated so each
# projection below closes it. `{a}{b}` is not "b inside a": kubectl evaluates
# both groups against the root document and concatenates the results, so the
# earlier `"{...spec}" + "{.containers[0].image}"` printed the whole pod spec
# and every comparison in this probe failed. A failing probe is not visible as
# an error -- it just means the apply, the verify Job and its 420s wait ran on
# every deploy.
_AURORA_REFRESH_POD = "{.spec.jobTemplate.spec.template.spec"


def assert_aurora_refresh_current(
    runner: CommandRunner,
    *,
    cpu_kubeconfig: Path,
    namespace: str,
    wheel_configmap: str,
    runtime_image: str,
    master_secret_arn: str,
) -> None:
    """Assert the live refresh CronJob is the one this release would render.

    The rendered manifest only varies in three places, so reading those three
    projections back is a complete equivalence check: a different wheel
    ConfigMap, runtime image or master Secret ARN means the apply still has work
    to do. Nothing here reads Secret material.
    """

    def projection(expression: str) -> str:
        """Read one field of the CronJob's pod spec.

        `expression` continues `_AURORA_REFRESH_POD`: it starts with a `.` and
        closes the single jsonpath group with `}`.
        """

        return kubectl_projection(
            runner,
            kubeconfig=cpu_kubeconfig,
            namespace=namespace,
            arguments=[
                "get",
                "cronjob",
                AURORA_REFRESH_CRONJOB,
                "-o",
                f"jsonpath={_AURORA_REFRESH_POD}{expression}",
            ],
        )

    image = projection(".containers[0].image}")
    if image != runtime_image:
        raise BootstrapMutationRequired(f"{AURORA_REFRESH_CRONJOB} image")
    artifact = projection('.volumes[?(@.name=="artifact")].configMap.name}')
    if artifact != wheel_configmap:
        raise BootstrapMutationRequired(f"{AURORA_REFRESH_CRONJOB} wheel")
    secret_arn = projection(
        '.containers[0].env[?(@.name=="GPU_FAULT_AURORA_MASTER_SECRET_ARN")].value}'
    )
    if secret_arn != master_secret_arn:
        raise BootstrapMutationRequired(f"{AURORA_REFRESH_CRONJOB} master secret")


def _node_action_keys_secret() -> str:
    return os.environ.get("GPU_FAULT_NODE_ACTION_KEYS_SECRET") or (
        "gpu-fault-node-action-keys"
    )


def assert_node_action_keys_current(
    runner: CommandRunner,
    *,
    cpu_kubeconfig: Path,
    gpu_kubeconfig: Path,
    namespace: str,
    cluster: ClusterIdentity,
) -> None:
    """Assert every current HyperPod node already has a scoped key on both sides.

    The GPU map must match this cluster; the CPU map is a site-wide union.
    Private digest comparison also detects divergence after a partial rotation.
    """

    if os.environ.get("GPU_FAULT_ROTATE_NODE_ACTION_KEY"):
        raise BootstrapMutationRequired("node action key rotation requested")
    secret_name = _node_action_keys_secret()
    listed = kubectl_projection(
        runner,
        kubeconfig=gpu_kubeconfig,
        namespace=namespace,
        context=cluster.context,
        arguments=[
            "get",
            "nodes",
            "-l",
            f"sagemaker.amazonaws.com/cluster-name={cluster.hyperpod_name}",
            "-o",
            'jsonpath={range .items[*]}{.metadata.name}{"\\n"}{end}',
        ],
    )
    nodes = tuple(sorted(line.strip() for line in listed.splitlines() if line.strip()))
    if not nodes:
        # An empty inventory is either a read that raced a cluster scale event or
        # a cluster with no nodes; either way the ensure step is the component
        # that knows how to fail closed on it.
        raise BootstrapMutationRequired(f"{cluster.hyperpod_name} node inventory")
    on_gpu = read_node_key_proof(
        runner,
        ["kubectl", "--kubeconfig", str(gpu_kubeconfig), "--context", cluster.context],
        namespace,
        secret_name=secret_name,
    )
    if on_gpu is None or set(on_gpu.digests) != set(nodes) or on_gpu.rotation_pending:
        raise BootstrapMutationRequired(f"{cluster.hyperpod_name} node action keys")
    on_cpu = read_node_key_proof(
        runner,
        ["kubectl", "--kubeconfig", str(cpu_kubeconfig)],
        namespace,
        secret_name=secret_name,
    )
    if on_cpu is not None and on_cpu.uid == on_gpu.uid:
        raise BootstrapError("CPU and GPU node-key Secrets have the same identity")
    if (
        on_cpu is None
        or on_cpu.rotation_pending
        or any(on_cpu.digests.get(node) != on_gpu.digests[node] for node in nodes)
    ):
        raise BootstrapMutationRequired(
            f"{cluster.hyperpod_name} node action key mirror"
        )
