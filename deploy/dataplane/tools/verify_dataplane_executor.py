#!/usr/bin/env python3
"""Post-deploy self-check for the data-plane cluster executor.

The control plane has verify_control_plane_role_split.py; the data plane
had nothing, and every gap found on real hardware so far shared one
shape: the Pod was Running, the logs were clean, and the executor was
still unable to do its job. This checks the properties that decide that,
none of which appear in pod status:

  * the ServiceAccount carries a real ``eks.amazonaws.com/role-arn``.
    Without it the pod has no AWS credentials at all, and a HyperPod
    reboot dies on NoCredentialsError at the moment a GPU breaks. The
    annotation is also not enough on its own: the projected token volume
    is injected by the admission webhook at pod *creation*, so a pod that
    predates the annotation keeps running without credentials. Comparing
    the live pod's AWS_ROLE_ARN against the ServiceAccount catches both
    "never annotated" and "annotated but never restarted";
  * the wheel ConfigMap matches the control plane's. The two clusters are
    rolled by hand and the data plane has silently lagged behind before,
    which produces a fleet running code the control plane's digest pin
    rejects -- every node action then fails closed;
  * the three ENABLE_* switches are on. The regional architecture
    delegates all mutation to this executor, and
    ENABLE_NODE_ACTION_ADAPTER=false is not a safe default: with no
    node-action adapter the fleet agents report no product, and the
    policy quietly downgrades "restart the application" to "isolate the
    whole node";
  * the Secret keys those switches require exist. The HyperPod adapter
    reads GPU_FAULT_HYPERPOD_CONFIRM_CLUSTER from an ``optional: true``
    key, so a missing key is a startup crash loop rather than a
    manifest error.

Exits non-zero with the reason so a deploy fails loudly.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys

NAMESPACE = os.getenv("GPU_FAULT_NAMESPACE", "gpu-fault-system")
# The data-plane cluster. The deployment manual addresses it as
# --context "${GPU_EKS_CONTEXT}" rather than by switching the current
# context, so this has to be selectable; empty means the current context.
KUBE_CONTEXT = os.getenv("GPU_FAULT_KUBE_CONTEXT", "")
DEPLOYMENT = "gpu-fault-cluster-executor"
SERVICE_ACCOUNT = "gpu-fault-cluster-executor"
SECRET = "gpu-fault-regional-connection"
NODE_KEYS_SECRET = "gpu-fault-node-action-keys"
REGISTRY_SECRET = "gpu-fault-regional-clusters"
IRSA_ANNOTATION = "eks.amazonaws.com/role-arn"
# The tiers that carry the current release on the control plane. The
# control plane also owns a gpu-fault-cluster-executor Deployment scaled
# to zero (BOOT-004/005 keep mutation off that cluster), and it is not
# rolled, so it must never be used as the reference.
CONTROL_PLANE_REFERENCE_DEPLOYMENTS = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
)


def kubectl(args: list[str], *, control_plane: bool = False) -> tuple[int, str, str]:
    env = dict(os.environ)
    context = KUBE_CONTEXT
    if control_plane:
        kubeconfig = os.getenv("GPU_FAULT_CONTROL_PLANE_KUBECONFIG")
        context = os.getenv("GPU_FAULT_CONTROL_PLANE_CONTEXT", "")
        if kubeconfig:
            env["KUBECONFIG"] = kubeconfig
    command = ["kubectl"]
    if context:
        command += ["--context", context]
    command += ["-n", NAMESPACE, *args]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    return result.returncode, result.stdout, result.stderr


def get_json(args: list[str], *, control_plane: bool = False) -> dict | None:
    code, out, _ = kubectl([*args, "-o", "json"], control_plane=control_plane)
    if code != 0:
        return None
    return json.loads(out)


def container(pod_spec: dict, name: str) -> dict | None:
    for candidate in pod_spec.get("containers", []):
        if candidate["name"] == name:
            return candidate
    return None


def env_entry(item: dict, name: str) -> dict | None:
    for entry in item.get("env", []):
        if entry.get("name") == name:
            return entry
    return None


def artifact_configmap(pod_spec: dict) -> str | None:
    for volume in pod_spec.get("volumes", []):
        if volume["name"] == "artifact":
            return volume.get("configMap", {}).get("name")
    return None


def expected_wheel_configmap(
    problems: list[str],
) -> str | None:
    """The wheel ConfigMap name this cluster should be running.

    Explicit override first, then the control plane's serving tiers. If
    neither is reachable this reports a problem rather than skipping:
    a silent skip is how the data plane fell a release behind without
    anyone noticing.
    """

    override = os.getenv("GPU_FAULT_EXPECTED_WHEEL_CONFIGMAP")
    if override:
        return override.strip()
    if not (
        os.getenv("GPU_FAULT_CONTROL_PLANE_KUBECONFIG")
        or os.getenv("GPU_FAULT_CONTROL_PLANE_CONTEXT")
    ):
        problems.append(
            "cannot verify the wheel ConfigMap: set "
            "GPU_FAULT_EXPECTED_WHEEL_CONFIGMAP, or "
            "GPU_FAULT_CONTROL_PLANE_KUBECONFIG/"
            "GPU_FAULT_CONTROL_PLANE_CONTEXT so the control plane's "
            "current wheel can be read"
        )
        return None
    metadata = get_json(
        ["get", "configmap", "gpu-fault-release-metadata"],
        control_plane=True,
    )
    if metadata is not None:
        executor_sha = (metadata.get("data") or {}).get(
            "required-regional-executor-artifact-sha256"
        )
        if executor_sha:
            return "gpu-fault-executor-wheel-0100-" + executor_sha[:12]

    # Legacy releases did not publish an independent Executor artifact.
    names = set()
    for name in CONTROL_PLANE_REFERENCE_DEPLOYMENTS:
        item = get_json(["get", "deployment", name], control_plane=True)
        if item is None:
            continue
        if not item["spec"].get("replicas"):
            continue
        wheel = artifact_configmap(item["spec"]["template"]["spec"])
        if wheel:
            names.add(wheel)
    if not names:
        problems.append(
            "cannot verify the wheel ConfigMap: no scaled-up "
            + "/".join(CONTROL_PLANE_REFERENCE_DEPLOYMENTS)
            + " on the control plane to compare against"
        )
        return None
    if len(names) > 1:
        problems.append(
            "the control plane's own tiers disagree on the wheel "
            f"ConfigMap ({', '.join(sorted(names))}): finish the roll "
            "there before rolling the data plane"
        )
        return None
    return names.pop()


def check_service_account(
    problems: list[str],
) -> str | None:
    item = get_json(["get", "serviceaccount", SERVICE_ACCOUNT])
    if item is None:
        problems.append(f"serviceaccount {SERVICE_ACCOUNT} is missing")
        return None
    role_arn = item["metadata"].get("annotations", {}).get(IRSA_ANNOTATION)
    if not role_arn:
        problems.append(
            f"serviceaccount {SERVICE_ACCOUNT} has no "
            f"{IRSA_ANNOTATION} annotation: the pod gets no AWS "
            "credentials, so every HyperPod call fails at the moment a "
            "real fault arrives"
        )
        return None
    if role_arn.startswith("REPLACE_WITH"):
        problems.append(
            f"serviceaccount {SERVICE_ACCOUNT} still carries the manifest placeholder {role_arn}"
        )
        return None
    if not role_arn.startswith("arn:") or ":role/" not in role_arn:
        problems.append(f"{IRSA_ANNOTATION}={role_arn} is not an IAM role ARN")
        return None
    return role_arn


def check_secret_keys(problems: list[str], container_spec: dict) -> None:
    item = get_json(["get", "secret", SECRET])
    if item is None:
        problems.append(f"secret {SECRET} is missing")
        return
    keys = set(item.get("data", {}))
    required = {
        "control-plane-url",
        "cluster-token",
        "cluster-id",
        "allowed-namespaces",
        "ca.crt",
    }
    if "node-action-secret" in keys:
        problems.append(
            f"secret {SECRET} still contains the fleet master "
            "node-action-secret; provision node-scoped keys and remove it"
        )
    if flag_is_true(container_spec, "GPU_FAULT_ENABLE_HYPERPOD_ADAPTER"):
        # Both are optional: true in the manifest, so a missing key is a
        # startup crash loop instead of an apply-time error. The adapter
        # refuses to run without an independent confirmation of which
        # cluster it is about to mutate.
        required.update(
            {
                "hyperpod-cluster-name",
                "hyperpod-confirm-cluster-name",
            }
        )
    missing = sorted(required - keys)
    if missing:
        problems.append(
            f"secret {SECRET} is missing keys the enabled adapters require: {', '.join(missing)}"
        )
    check_allowed_namespaces(problems, item)
    if flag_is_true(
        container_spec,
        "GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER",
    ):
        node_keys = get_json(["get", "secret", NODE_KEYS_SECRET])
        if node_keys is None or not node_keys.get("data"):
            problems.append(f"secret {NODE_KEYS_SECRET} is missing or empty")
        directory = env_entry(
            container_spec,
            "GPU_FAULT_NODE_ACTION_KEYS_DIR",
        )
        if not directory or not directory.get("value"):
            problems.append("GPU_FAULT_NODE_ACTION_KEYS_DIR is unset")
        volume = next(
            (
                item
                for item in container_spec.get("_pod_volumes", [])
                if item.get("name") == "node-action-keys"
            ),
            None,
        )
        if (
            volume is None
            or volume.get("secret", {}).get("secretName") != NODE_KEYS_SECRET
        ):
            problems.append(f"Deployment does not mount {NODE_KEYS_SECRET}")


def _decode_secret(data: dict, key: str) -> str:
    encoded = data.get(key)
    if not encoded:
        return ""
    try:
        return base64.b64decode(encoded).decode().strip()
    except (ValueError, UnicodeDecodeError):
        return ""


def check_allowed_namespaces(problems: list[str], connection_secret: dict) -> None:
    if not (
        os.getenv("GPU_FAULT_CONTROL_PLANE_KUBECONFIG")
        or os.getenv("GPU_FAULT_CONTROL_PLANE_CONTEXT")
    ):
        return
    data = connection_secret.get("data", {})
    cluster_id = _decode_secret(data, "cluster-id")
    actual = {
        item.strip()
        for item in _decode_secret(data, "allowed-namespaces").split(",")
        if item.strip()
    }
    registry = get_json(
        ["get", "secret", REGISTRY_SECRET],
        control_plane=True,
    )
    if registry is None:
        problems.append(f"control-plane secret {REGISTRY_SECRET} is missing")
        return
    raw = _decode_secret(registry.get("data", {}), "clusters.json")
    try:
        registrations = json.loads(raw)
    except json.JSONDecodeError:
        problems.append(
            f"control-plane secret {REGISTRY_SECRET}/clusters.json is invalid JSON"
        )
        return
    registration = next(
        (item for item in registrations if item.get("cluster_id") == cluster_id),
        None,
    )
    if registration is None:
        problems.append(f"cluster {cluster_id!r} is absent from {REGISTRY_SECRET}")
        return
    expected = set(registration.get("allowed_namespaces", []))
    if actual != expected:
        problems.append(
            "allowed namespace drift: data-plane connection Secret has "
            f"{sorted(actual)}, control-plane registry has "
            f"{sorted(expected)}"
        )


def flag_is_true(container_spec: dict, name: str) -> bool:
    entry = env_entry(container_spec, name)
    return bool(entry) and str(entry.get("value", "")).lower() == "true"


def check_flags(problems: list[str], container_spec: dict) -> None:
    reasons = {
        "GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER": (
            "without it the fleet agents report no product and the "
            "policy silently downgrades 'restart the application' to "
            "'isolate the whole node'"
        ),
        "GPU_FAULT_ENABLE_HYPERPOD_ADAPTER": (
            "without it no reboot ever reaches HyperPod"
        ),
        "GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER": (
            "it is the guard that refuses to fall back to the provider node replacement API"
        ),
    }
    for name, reason in reasons.items():
        entry = env_entry(container_spec, name)
        if entry is None:
            problems.append(f"{name} is unset: {reason}")
            continue
        if "valueFrom" in entry:
            problems.append(
                f"{name} comes from valueFrom, so this check cannot "
                "read it; set it inline in the Deployment"
            )
            continue
        if str(entry.get("value", "")).lower() != "true":
            problems.append(f"{name}={entry.get('value')!r}: {reason}")
    if flag_is_true(
        container_spec,
        "GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER",
    ) and not flag_is_true(
        container_spec,
        "GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE",
    ):
        problems.append(
            "spare failover requires "
            "GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE=true; the "
            "executor refuses to start otherwise"
        )


def check_pods(
    problems: list[str],
    role_arn: str | None,
    wheel: str | None,
) -> int:
    listing = get_json(
        [
            "get",
            "pods",
            "-l",
            f"app={DEPLOYMENT}",
            "--field-selector",
            "status.phase=Running",
        ]
    )
    pods = (listing or {}).get("items", [])
    if not pods:
        problems.append(f"no Running {DEPLOYMENT} pod")
        return 0
    for pod in pods:
        name = pod["metadata"]["name"]
        spec = pod["spec"]
        executor = container(spec, "executor")
        if executor is None:
            problems.append(f"{name} has no executor container")
            continue
        pod_role = env_entry(executor, "AWS_ROLE_ARN")
        token_file = env_entry(executor, "AWS_WEB_IDENTITY_TOKEN_FILE")
        if pod_role is None or token_file is None:
            problems.append(
                f"{name} has no AWS_ROLE_ARN/"
                "AWS_WEB_IDENTITY_TOKEN_FILE: it was created before the "
                f"{IRSA_ANNOTATION} annotation existed. The projected "
                "token volume is only injected at pod creation, so this "
                "pod has no credentials until it is restarted: kubectl "
                f"-n {NAMESPACE} rollout restart deploy/{DEPLOYMENT}"
            )
            continue
        if role_arn and pod_role.get("value") != role_arn:
            problems.append(
                f"{name} assumes {pod_role.get('value')} but the "
                f"ServiceAccount now says {role_arn}: restart the "
                "deployment"
            )
        path = token_file.get("value") or ""
        code, _, err = kubectl(
            [
                "exec",
                name,
                "-c",
                "executor",
                "--",
                "/bin/sh",
                "-c",
                f"test -s {path}",
            ]
        )
        if code != 0:
            problems.append(
                f"{name} has no projected identity token at {path} "
                f"({err.strip() or 'test -s failed'})"
            )
        pod_wheel = artifact_configmap(spec)
        if wheel and pod_wheel != wheel:
            problems.append(
                f"{name} still mounts {pod_wheel}, not {wheel}: the rollout did not finish"
            )
    return len(pods)


def main() -> int:
    problems: list[str] = []

    deployment = get_json(["get", "deployment", DEPLOYMENT])
    if deployment is None:
        print(f"executor check failed: {DEPLOYMENT} is missing")
        return 1
    pod_spec = deployment["spec"]["template"]["spec"]
    executor = container(pod_spec, "executor")
    if executor is None:
        print(f"executor check failed: {DEPLOYMENT} has no executor container")
        return 1
    executor["_pod_volumes"] = pod_spec.get("volumes", [])
    if pod_spec.get("serviceAccountName") != SERVICE_ACCOUNT:
        problems.append(
            f"{DEPLOYMENT} does not run as {SERVICE_ACCOUNT}, so the "
            "IRSA annotation on that ServiceAccount does nothing"
        )

    replicas = deployment["spec"].get("replicas", 0)
    ready = deployment.get("status", {}).get("readyReplicas", 0)
    if not replicas:
        problems.append(
            f"{DEPLOYMENT} is scaled to zero: the control plane queues "
            "node actions that nothing claims"
        )
    elif ready != replicas:
        problems.append(f"{DEPLOYMENT} has {ready}/{replicas} ready")

    role_arn = check_service_account(problems)
    check_flags(problems, executor)
    check_secret_keys(problems, executor)

    wheel = artifact_configmap(pod_spec)
    if wheel is None:
        problems.append(f"{DEPLOYMENT} has no artifact volume")
    else:
        if get_json(["get", "configmap", wheel]) is None:
            problems.append(
                f"the wheel ConfigMap {wheel} does not exist in this cluster: the pod cannot start"
            )
        expected = expected_wheel_configmap(problems)
        if expected and expected != wheel:
            problems.append(
                f"{DEPLOYMENT} mounts {wheel} but the control plane "
                f"runs {expected}: the data plane is running different "
                "code, and the control plane's agent digest pin will "
                "fail node actions closed"
            )

    running = check_pods(problems, role_arn, wheel)

    for problem in problems:
        print(f"executor check failed: {problem}")
    if problems:
        return 1
    print(
        f"executor check passed: {running} Running pods on {wheel}, "
        f"assuming {role_arn}, node action + HyperPod + spare failover "
        "all enabled"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
