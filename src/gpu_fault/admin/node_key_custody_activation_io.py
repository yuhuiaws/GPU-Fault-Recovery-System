"""Existing Kubernetes/Fleet delivery primitives for custody activation."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from urllib.parse import quote

from gpu_fault.admin.bootstrap_common import CommandRunner
from gpu_fault.admin.execution import current_deadline
from gpu_fault.admin.node_key_custody_activation_state import (
    owned_wave_data,
    require_owned_wave as validate_owned_wave,
)
from gpu_fault.admin.node_key_custody_admin_probe import (
    AdminNodeKeyContext,
    CustodyReadRunner,
    cluster_namespace_anchors,
    live_node_uids,
    private_json,
)
from gpu_fault.admin.node_key_custody_chain import authorize_now
from gpu_fault.admin.node_key_custody_crypto import (
    private_command_environment,
    read_regular,
)
from gpu_fault.admin.node_key_custody_models import (
    Authorization,
    Completed,
    CustodyError,
)
from gpu_fault.admin.node_key_custody_pods import (
    bound_consumer_pods,
    cpu_role_container,
)
from gpu_fault.admin.node_key_custody_preflight import preflight_activation
from gpu_fault.admin.node_key_custody_process import (
    PROJECTED_KEY_PROBE as PROJECTED_KEY_PROBE,
    projected_key_precedes_process as verify_projected_process,
)
from gpu_fault.admin.node_key_proof import read_node_key_proof
from gpu_fault.node_installer_reconciler import (
    INSTALLER_ACTIVATION_ANNOTATION,
    INSTALLER_ACTIVATED_ANNOTATION,
)
from gpu_fault_release.regional_deployment_inventory import (
    CPU_INGRESS_DEPLOYMENT,
    CPU_RUNTIME_DEPLOYMENTS,
    GPU_EXECUTOR_DEPLOYMENT,
    GPU_RECONCILER_DEPLOYMENT,
)
from gpu_fault_release.regional_release_config import ClusterTarget
from gpu_fault_release.regional_release_gpu_rollout import agents_converged
from gpu_fault_release.regional_release_node_preflight import (
    validate_target_node_state,
)
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_fleet_rollout import (
    INSTALLER_WAVE_CONFIG_MAP_ENV,
    reconciler_container_env,
    reconciler_installer_identity,
)

ACTIVATION_ANNOTATION = "gpu-fault.io/node-key-activation"
MAX_CONSUMER_RELOADS = 3

AGENTS_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
store = ApplicationContext.from_environment().store
expected = json.load(sys.stdin)
print(json.dumps({
    item.node_id: item.model_dump(mode="json")
    for item in store.list_agents(expected["cluster_id"])
}))
"""

TRANSITION_PROBE = r"""
import json
import os
import sys
import urllib.request
request = json.load(sys.stdin)
call = urllib.request.Request(
    "http://127.0.0.1:8080" + request["path"],
    data=json.dumps(request["payload"]).encode(),
    method="POST",
    headers={
        "Content-Type": "application/json",
        "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
    },
)
with urllib.request.urlopen(call, timeout=20) as response:
    value = json.loads(response.read())
print(json.dumps({
    key: value.get(key) for key in
    ("cluster_id", "node_id", "generation", "lifecycle_state", "transition_id")
}))
"""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _time(value: Any) -> datetime:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError
        return result
    except ValueError:
        raise CustodyError("node key activation has an invalid time binding") from None


class CustodyActivationIO:
    def __init__(
        self,
        runner: CommandRunner,
        context: AdminNodeKeyContext,
        authorization: Authorization,
    ) -> None:
        self.runner = runner
        self.context = context
        self.authorization = authorization
        self.completed: Completed | None = None
        self.node = authorization.rotate_node or ""
        if (
            authorization.purpose != "rotate"
            or self.node not in authorization.binding.nodes
        ):
            raise CustodyError(
                "node key activation requires single-node rotation authorization"
            )
        self.target = ClusterTarget(
            context.cluster_id,
            context.cluster.context,
            "",
            context.cluster.region,
            context.cluster.hyperpod_name,
            context.cluster.eks_arn,
        )

    def bind_completed(self, completed: Completed) -> None:
        # Only provision_admin_custody calls this, after verifying the signed
        # chain against this authorization and the actual Secret byte maps.
        self.completed = completed

    def verify_expected_keys(self) -> None:
        if self.completed is None:
            raise CustodyError(
                "node key activation lacks its verified signed completion"
            )
        for plane in ("cpu", "gpu"):
            expected = getattr(self.completed, plane)
            current = read_node_key_proof(
                CustodyReadRunner(self.runner),
                self.context.kubectl(plane),
                self.context.namespace,
            )
            if (
                current is None
                or current.uid != expected.uid
                or current.rotation_pending
                or any(
                    current.digests.get(node) != key.sha256
                    for node, key in expected.keys.items()
                )
            ):
                raise CustodyError(
                    "node key activation source differs from signed completion"
                )

    def keys_provisioned(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        self.verify_expected_keys()
        return {"signed_completion_verified": True}

    def _gpu(self, _target: ClusterTarget, *arguments: str) -> list[str]:
        return [*self.context.kubectl("gpu"), *arguments]

    def _get_json(self, command: list[str]) -> dict[str, Any]:
        return private_json(self.runner, [*command, "-o", "json"])

    def read(self, plane: str, kind: str, name: str) -> dict[str, Any]:
        args = [*self.context.kubectl(plane)]
        if kind != "node":
            args.extend(["-n", self.context.namespace])
        return private_json(self.runner, [*args, "get", kind, name, "-o", "json"])

    def execute(
        self,
        command: list[str],
        *,
        data: str | None = None,
        seconds: float = 30,
        snapshot: dict[str, Any] | None = None,
    ) -> str:
        if snapshot is not None:
            self.require_owned_wave(snapshot)
        authorize_now(self.authorization, datetime.now(timezone.utc))
        return self.runner.run(
            command,
            input_text=data,
            capture=True,
            sensitive=True,
            mutate=True,
            timeout_seconds=seconds,
            env=private_command_environment(),
        )

    def host_preflight(self) -> None:
        preflight_activation(self.context, self.authorization, self.runner, self._gpu)

    def cpu_probe(
        self, script: str, payload: dict[str, Any], *, mutate: bool = False
    ) -> dict[str, Any]:
        pods = self.consumer_pods("cpu", CPU_INGRESS_DEPLOYMENT)
        if not pods:
            raise CustodyError("node key activation has no Ready CPU ingress")
        command = [
            *self.context.kubectl("cpu"),
            "-n",
            self.context.namespace,
            "exec",
            "-i",
            pods[0]["metadata"]["name"],
            "-c",
            "api",
            "--",
            "/opt/gpu-fault/control-plane/bin/python",
            "-c",
            script,
        ]
        output = self.runner.run(
            command,
            input_text=json.dumps(payload),
            capture=True,
            sensitive=True,
            mutate=mutate,
            timeout_seconds=30,
            env=private_command_environment(),
        )
        try:
            result = json.loads(output)
            if not isinstance(result, dict):
                raise ValueError
            return result
        except (ValueError, TypeError):
            raise CustodyError(
                "node key activation probe returned invalid evidence"
            ) from None

    def agents(self) -> dict[str, Any]:
        return self.cpu_probe(AGENTS_PROBE, {"cluster_id": self.context.cluster_id})

    def consumer_pods(self, plane: str, name: str) -> list[dict[str, Any]]:
        deployment = self.read(plane, "deployment", name)
        documents = [
            private_json(
                self.runner,
                [
                    *self.context.kubectl(plane),
                    "-n",
                    self.context.namespace,
                    "get",
                    kind,
                    "-l",
                    "app=" + name,
                    "-o",
                    "json",
                ],
            )
            for kind in ("replicasets", "pods")
        ]
        return bound_consumer_pods(
            deployment, *documents, namespace=self.context.namespace
        )

    def verify_anchors(self) -> None:
        binding = self.authorization.binding.site
        if any(
            uid != getattr(binding, field)
            for field, uid in cluster_namespace_anchors(
                self.runner, self.context
            ).items()
        ):
            raise CustodyError(
                "node key activation cluster or namespace anchor changed"
            )

    def safety(self) -> None:
        result = self.cpu_probe(
            probe_source("rollout_wave_safety"),
            {
                "cluster_id": self.context.cluster_id,
                "nodes": sorted(self.authorization.binding.nodes),
                "wave": [self.node],
                "minimum_lease_remaining_seconds": 30,
            },
        )
        if (
            set(result.get("open_remote", {})) != {"PENDING", "LEASED", "WAITING"}
            or any(
                type(value) is not int or value != 0
                for value in result["open_remote"].values()
            )
            or type(result.get("destructive_workflow_count")) is not int
            or result["destructive_workflow_count"] != 0
            or type(result.get("agent_blocker_count")) is not int
            or result["agent_blocker_count"] != 0
        ):
            raise CustodyError("node key activation is blocked by fleet safety state")

    def _resource(self, plane: str, name: str) -> dict[str, Any]:
        value = self.read(plane, "deployment", name)
        metadata, spec = value.get("metadata") or {}, value.get("spec") or {}
        replicas = spec.get("replicas")
        if (
            not metadata.get("uid")
            or not metadata.get("resourceVersion")
            or type(replicas) is not int
            or replicas < 0
        ):
            raise CustodyError("node key activation Deployment identity is incomplete")
        spec = copy.deepcopy(spec)
        annotations = spec["template"]["metadata"].setdefault("annotations", {})
        previous = annotations.pop(ACTIVATION_ANNOTATION, None)
        if previous is not None and re.fullmatch(r"[0-9a-f]{64}", previous) is None:
            raise CustodyError(
                "node key activation annotation belongs to another writer"
            )
        return {
            "uid": metadata["uid"],
            "spec_sha256": _digest(spec),
            "replicas": replicas,
            "activation": previous,
        }

    def markers(self, plane: str, name: str) -> tuple[str, ...]:
        return tuple(
            hashlib.sha256(
                f"{self.authorization.transaction_id}:{plane}:{name}:{attempt}".encode()
            ).hexdigest()
            for attempt in range(MAX_CONSUMER_RELOADS)
        )

    def sibling_keys(self) -> dict[str, str]:
        result = {}
        for plane in ("cpu", "gpu"):
            proof = read_node_key_proof(
                CustodyReadRunner(self.runner),
                self.context.kubectl(plane),
                self.context.namespace,
            )
            if proof is None:
                raise CustodyError("node key activation has no key source")
            result[plane] = _digest(
                {
                    "uid": proof.uid,
                    "keys": {
                        node: digest
                        for node, digest in proof.digests.items()
                        if node != self.node
                    },
                }
            )
        return result

    def release_state(self) -> dict[str, Any]:
        value = self.read("cpu", "configmap", "gpu-fault-regional-release-state")
        try:
            state = json.loads(value["data"]["state.json"])
            expected = self.authorization.binding.release
            if (
                state["release_id"] != expected.release_id
                or state["phase"] != "complete"
                or state["transaction_committed"] is not True
                or state["release_delivery_sha256"] != expected.delivery_sha256
                or state["bundle_sha256"] != expected.bundle_sha256
                or state["agent_config_digest"] != expected.config_digest
                or state["runtime_profile_version"] != expected.runtime_profile_version
            ):
                raise ValueError
            return {
                "uid": value["metadata"]["uid"],
                "release_id": state["release_id"],
                "template_sha256": state["node_template_sha256"],
                "sha256": _digest(state),
            }
        except (KeyError, TypeError, ValueError):
            raise CustodyError(
                "node key activation requires the committed authorized release"
            ) from None

    def capture(self) -> dict[str, Any]:
        self.verify_anchors()
        self.safety()
        validate_target_node_state(
            cast(Any, self), self.target, tuple(self.authorization.binding.nodes)
        )
        agents = self.agents()
        if set(agents) != set(self.authorization.binding.nodes):
            raise CustodyError(
                "node key activation Agent inventory differs from its authorization"
            )
        now = datetime.now(timezone.utc)
        if any(
            value.get("lifecycle_state") != "ACTIVE"
            or not value.get("agent_incarnation_id")
            or _time(value.get("lease_expires_at")) <= now + timedelta(seconds=30)
            for value in agents.values()
        ):
            raise CustodyError("node key activation requires current ACTIVE Agents")
        resources = {
            "cpu/" + name: self._resource("cpu", name)
            for name in CPU_RUNTIME_DEPLOYMENTS
        }
        resources.update(
            {
                "gpu/" + name: self._resource("gpu", name)
                for name in (GPU_EXECUTOR_DEPLOYMENT, GPU_RECONCILER_DEPLOYMENT)
            }
        )
        manifest = json.loads(read_regular(self.context.release_manifest))
        for resource in resources:
            plane, name = resource.split("/", 1)
            deployment = self.read(plane, "deployment", name)
            container_name = (
                cpu_role_container(name, deployment["spec"]["template"]["spec"])
                if plane == "cpu"
                else ("executor" if name == GPU_EXECUTOR_DEPLOYMENT else "reconciler")
            )
            containers = [
                item
                for item in deployment["spec"]["template"]["spec"]["containers"]
                if item["name"] == container_name
            ]
            image = manifest["delivery"]["images"][
                "runtime" if plane == "cpu" else "executor"
            ]["reference"]
            if len(containers) != 1 or containers[0].get("image") != image:
                raise CustodyError(
                    "node key activation consumer is not the signed runtime image"
                )
        jobs = private_json(
            self.runner,
            [
                *self.context.kubectl("gpu"),
                "-n",
                self.context.namespace,
                "get",
                "jobs",
                "-o",
                "json",
            ],
        )
        if not isinstance(jobs.get("items"), list) or (jobs.get("metadata") or {}).get(
            "continue"
        ):
            raise CustodyError(
                "node key activation installer Job inventory is incomplete"
            )
        if any(
            str(item.get("metadata", {}).get("name", "")).startswith(
                "gpu-fault-install-"
            )
            and not any(
                condition.get("type") in {"Complete", "Failed"}
                and condition.get("status") == "True"
                for condition in item.get("status", {}).get("conditions", [])
            )
            for item in jobs["items"]
        ):
            raise CustodyError(
                "node key activation cannot overlap an in-flight installer Job"
            )
        reconciler = self.read("gpu", "deployment", GPU_RECONCILER_DEPLOYMENT)
        environment = reconciler_container_env(self, self.target, deployment=reconciler)
        identity = reconciler_installer_identity(environment)
        wave_name = environment.get(INSTALLER_WAVE_CONFIG_MAP_ENV)
        if not wave_name or identity != (
            self.authorization.binding.release.bundle_sha256,
            self.release_state()["template_sha256"],
        ):
            raise CustodyError("node key activation Reconciler identity differs")
        wave = self.read("gpu", "configmap", wave_name)
        data = wave.get("data") or {}
        if (
            set(data) != {"allowed-nodes", "max-unavailable", "generation"}
            or data.get("allowed-nodes") != "*"
            or not isinstance(data.get("max-unavailable"), str)
            or not data["max-unavailable"].isdigit()
            or not 1 <= int(data["max-unavailable"]) <= 64
            or not isinstance(data.get("generation"), str)
            or not re.fullmatch(r"[A-Za-z0-9.-]{1,128}", data["generation"])
        ):
            raise CustodyError(
                "node key activation cannot take over an active installer wave"
            )
        return {
            "captured_at": now.isoformat(),
            "release": self.release_state(),
            "resources": resources,
            "agents": agents,
            "sibling_keys": self.sibling_keys(),
            "node_uids": live_node_uids(self.runner, self.context),
            "wave": {"name": wave_name, "uid": wave["metadata"]["uid"], "data": data},
        }

    def verify(self, snapshot: dict[str, Any]) -> None:
        self.verify_anchors()
        if self.completed is not None:
            self.verify_expected_keys()
        if (
            self.release_state() != snapshot["release"]
            or live_node_uids(self.runner, self.context)
            != self.authorization.binding.nodes
            or snapshot["node_uids"] != self.authorization.binding.nodes
            or self.sibling_keys() != snapshot["sibling_keys"]
        ):
            raise CustodyError("node key activation release or node binding changed")
        for resource, original in snapshot["resources"].items():
            plane, name = resource.split("/", 1)
            observed = self._resource(plane, name)
            if any(
                observed[key] != original[key]
                for key in ("uid", "spec_sha256", "replicas")
            ) or observed["activation"] not in {
                original["activation"],
                *self.markers(plane, name),
            }:
                raise CustodyError("node key activation consumer identity changed")

    def require_owned_wave(self, snapshot: dict[str, Any]) -> None:
        wave = snapshot["wave"]
        validate_owned_wave(
            self.read("gpu", "configmap", wave["name"]), wave, self.authorization
        )

    def guard(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        self.safety()
        current = self.agents()[self.node]
        transition = "node-key-" + self.authorization.transaction_id
        if (
            current.get("lifecycle_state") == "DRAINING"
            and current.get("transition_id") == transition
        ):
            return {"transition_id": transition}
        self.host_preflight()
        validate_target_node_state(
            cast(Any, self), self.target, tuple(self.authorization.binding.nodes)
        )
        if current != snapshot["agents"][self.node]:
            # A routine heartbeat may advance freshness; only stable identity
            # and generation, not timestamp bytes, authorizes the transition.
            for field in ("generation", "agent_incarnation_id", "boot_id", "endpoint"):
                if current.get(field) != snapshot["agents"][self.node].get(field):
                    raise CustodyError("node key activation Agent changed before drain")
        result = self.cpu_probe(
            TRANSITION_PROBE,
            {
                "path": "/v1/fleet/agents/"
                + quote(self.context.cluster_id, safe="")
                + "/"
                + quote(str(self.node), safe="")
                + "/drain",
                "payload": {
                    "expected_generation": current["generation"],
                    "transition_id": transition,
                    "reason": "authorized same-release node key activation",
                },
            },
            mutate=True,
        )
        if (
            result.get("lifecycle_state") != "DRAINING"
            or result.get("transition_id") != transition
        ):
            raise CustodyError("node key activation drain was not acknowledged")
        self.safety()
        return {"transition_id": transition}

    def _wave(self, snapshot: dict[str, Any], *, restore: bool) -> dict[str, Any]:
        original = snapshot["wave"]
        current = self.read("gpu", "configmap", original["name"])
        owned = owned_wave_data(original, self.authorization)
        wanted = original["data"] if restore else owned
        if current["metadata"]["uid"] != original["uid"] or current.get("data") not in (
            original["data"],
            owned,
        ):
            raise CustodyError("node key activation installer wave ownership changed")
        if current.get("data") != wanted:
            patch = [
                {"op": "test", "path": "/metadata/uid", "value": original["uid"]},
                {
                    "op": "test",
                    "path": "/metadata/resourceVersion",
                    "value": current["metadata"]["resourceVersion"],
                },
                {"op": "test", "path": "/data", "value": current["data"]},
                {"op": "replace", "path": "/data", "value": wanted},
            ]
            self.execute(
                [
                    *self.context.kubectl("gpu"),
                    "-n",
                    self.context.namespace,
                    "patch",
                    "configmap",
                    original["name"],
                    "--type=json",
                    "--patch-file=/dev/stdin",
                ],
                data=json.dumps(patch),
            )
        if self.read("gpu", "configmap", original["name"]).get("data") != wanted:
            raise CustodyError("node key activation installer wave did not converge")
        return {"restored": restore, "max_unavailable": 1 if not restore else None}

    def fence(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        self.safety()
        return self._wave(snapshot, restore=False)

    def projected_key_precedes_process(
        self, plane: str, pod: dict[str, Any], expected_digest: str
    ) -> bool:
        return verify_projected_process(
            plane=plane,
            pod=pod,
            namespace=self.context.namespace,
            node_id=self.node,
            expected_digest=expected_digest,
            kubectl=self.context.kubectl(plane),
            run=self.runner.run,
            read_pod=lambda name: self.read(plane, "pod", name),
        )

    def _roll(self, snapshot: dict[str, Any], plane: str, name: str) -> str | None:
        self.require_owned_wave(snapshot)
        self.safety()
        self.verify_expected_keys()
        original = snapshot["resources"][plane + "/" + name]
        if original["replicas"] == 0:
            return None
        proof = read_node_key_proof(
            CustodyReadRunner(self.runner),
            self.context.kubectl(plane),
            self.context.namespace,
        )
        if proof is None or self.node not in proof.digests:
            raise CustodyError("node key activation projected source is missing")
        markers = self.markers(plane, name)
        value = self.read(plane, "deployment", name)
        actual = (
            value["spec"]["template"]["metadata"]
            .get("annotations", {})
            .get(ACTIVATION_ANNOTATION)
        )
        first = markers.index(actual) if actual in markers else 0
        for marker in markers[first:]:
            self.verify(snapshot)
            value = self.read(plane, "deployment", name)
            metadata = value["metadata"]
            annotations = value["spec"]["template"]["metadata"].get("annotations") or {}
            if annotations.get(ACTIVATION_ANNOTATION) not in {
                original["activation"],
                *markers,
            }:
                raise CustodyError(
                    "node key activation consumer restart has another owner"
                )
            if annotations.get(ACTIVATION_ANNOTATION) != marker:
                self.execute(
                    [
                        *self.context.kubectl(plane),
                        "-n",
                        self.context.namespace,
                        "patch",
                        "deployment",
                        name,
                        "--type=json",
                        "--patch-file=/dev/stdin",
                    ],
                    data=json.dumps(
                        [
                            {
                                "op": "test",
                                "path": "/metadata/uid",
                                "value": original["uid"],
                            },
                            {
                                "op": "test",
                                "path": "/metadata/resourceVersion",
                                "value": metadata["resourceVersion"],
                            },
                            {
                                "op": "add",
                                "path": "/spec/template/metadata/annotations",
                                "value": {
                                    **annotations,
                                    ACTIVATION_ANNOTATION: marker,
                                },
                            },
                        ]
                    ),
                    snapshot=snapshot,
                )
            self.runner.run(
                [
                    *self.context.kubectl(plane),
                    "-n",
                    self.context.namespace,
                    "rollout",
                    "status",
                    "deployment/" + name,
                    "--timeout=600s",
                ],
                capture=True,
                sensitive=True,
                timeout_seconds=610,
                env=private_command_environment(),
            )
            pods = self.consumer_pods(plane, name)
            if len(pods) != original["replicas"] or any(
                row["metadata"].get("annotations", {}).get(ACTIVATION_ANNOTATION)
                != marker
                for row in pods
            ):
                raise CustodyError(
                    "node key activation still has an old consumer process"
                )
            # A projection updated after process startup does not prove that
            # cached signers loaded it. Retry only this bounded owned rollout.
            if all(
                self.projected_key_precedes_process(
                    plane, pod, proof.digests[self.node]
                )
                for pod in pods
            ):
                return marker
        raise CustodyError("node key projection did not precede consumer startup")

    def refresh_executor(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        marker = self._roll(snapshot, "gpu", GPU_EXECUTOR_DEPLOYMENT)
        return {"deployment": GPU_EXECUTOR_DEPLOYMENT, "activation_marker": marker}

    def install(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        self.require_owned_wave(snapshot)
        self.safety()
        node = self.read("gpu", "node", str(self.node))
        metadata = node["metadata"]
        annotations = metadata.get("annotations") or {}
        if (
            annotations.get(INSTALLER_ACTIVATION_ANNOTATION)
            != self.authorization.transaction_id
        ):
            validate_target_node_state(
                cast(Any, self), self.target, tuple(self.authorization.binding.nodes)
            )
            self.execute(
                [
                    *self.context.kubectl("gpu"),
                    "patch",
                    "node",
                    str(self.node),
                    "--type=json",
                    "--patch-file=/dev/stdin",
                ],
                data=json.dumps(
                    [
                        {
                            "op": "test",
                            "path": "/metadata/uid",
                            "value": self.authorization.binding.nodes[str(self.node)],
                        },
                        {
                            "op": "test",
                            "path": "/metadata/resourceVersion",
                            "value": metadata["resourceVersion"],
                        },
                        {
                            "op": "add",
                            "path": "/metadata/annotations",
                            "value": {
                                **annotations,
                                INSTALLER_ACTIVATION_ANNOTATION: self.authorization.transaction_id,
                            },
                        },
                    ]
                ),
                snapshot=snapshot,
            )
        deadline = time.monotonic() + 900
        while True:
            self.verify(snapshot)
            self.require_owned_wave(snapshot)
            node = self.read("gpu", "node", str(self.node))
            annotations = node["metadata"].get("annotations") or {}
            if annotations.get(
                INSTALLER_ACTIVATED_ANNOTATION
            ) == self.authorization.transaction_id and agents_converged(
                [node],
                self.target,
                self.authorization.binding.release.node_wheel_sha256,
                bundle_sha=self.authorization.binding.release.bundle_sha256,
                template_sha=snapshot["release"]["template_sha256"],
                config_digest=self.authorization.binding.release.config_digest,
                require_node_uid=True,
                node_names=frozenset({str(self.node)}),
            ):
                return {
                    "node_id": self.node,
                    "activation_id": self.authorization.transaction_id,
                }
            if (
                annotations.get("gpu-fault.io/installer-state")
                in {"Failed", "Unsupported"}
                or time.monotonic() >= deadline
            ):
                raise CustodyError("node key activation installer did not converge")
            remaining = current_deadline()
            time.sleep(min(5, remaining.remaining()) if remaining is not None else 5)

    def refresh_cpu(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        ordered = sorted(
            CPU_RUNTIME_DEPLOYMENTS, key=lambda name: name == CPU_INGRESS_DEPLOYMENT
        )
        markers = {name: self._roll(snapshot, "cpu", name) for name in ordered}
        return {"deployments": ordered, "activation_markers": markers}

    def observe(
        self, snapshot: dict[str, Any], *, preserve_siblings: bool = True
    ) -> dict[str, Any]:
        deadline = time.monotonic() + 120
        while True:
            agents = self.agents()
            if set(agents) != set(snapshot["agents"]):
                raise CustodyError("node key activation Agent membership changed")
            for name, previous in snapshot["agents"].items():
                if name == self.node or not preserve_siblings:
                    continue
                if any(
                    agents[name].get(key) != previous.get(key)
                    for key in (
                        "generation",
                        "agent_incarnation_id",
                        "boot_id",
                        "endpoint",
                        "artifact_sha256",
                        "config_digest",
                        "tls_certificate_pem",
                    )
                ):
                    raise CustodyError("node key activation changed a sibling identity")
            current = agents[str(self.node)]
            previous = snapshot["agents"][str(self.node)]
            if (
                current.get("lifecycle_state") == "ACTIVE"
                and current.get("agent_incarnation_id")
                != previous.get("agent_incarnation_id")
                and current.get("generation", 0) > previous.get("generation", 0)
                and (
                    not preserve_siblings
                    or current.get("boot_id") == previous.get("boot_id")
                )
                and current.get("node_action_key_version") == 2
                and current.get("artifact_sha256")
                == self.authorization.binding.release.node_wheel_sha256
                and current.get("config_digest")
                == self.authorization.binding.release.config_digest
                and current.get("compatibility_digest")
                == self.authorization.binding.release.node_digest
                and current.get("runtime_profile_version")
                == self.authorization.binding.release.runtime_profile_version
                and current.get("installer_bundle_sha256")
                == self.authorization.binding.release.bundle_sha256
                and current.get("installer_template_sha256")
                == snapshot["release"]["template_sha256"]
                and _time(current.get("last_seen_at")) > _time(snapshot["captured_at"])
                and _time(current.get("last_seen_at"))
                <= datetime.now(timezone.utc) + timedelta(seconds=5)
                and _time(current.get("lease_expires_at")) > datetime.now(timezone.utc)
            ):
                return {
                    "node_id": self.node,
                    "agent_generation": current["generation"],
                    "agent_incarnation_id": current["agent_incarnation_id"],
                    "independent_witness": "NOT_PROVED",
                }
            if time.monotonic() >= deadline:
                raise CustodyError("node key activation has no fresh new Agent")
            remaining = current_deadline()
            time.sleep(min(5, remaining.remaining()) if remaining is not None else 5)

    def unfence(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        self.safety()
        return self._wave(snapshot, restore=True)

    def current(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        self.verify_anchors()
        self.verify_expected_keys()
        state = self.release_state()
        if (
            state["uid"] != snapshot["release"]["uid"]
            or state["template_sha256"] != snapshot["release"]["template_sha256"]
            or live_node_uids(self.runner, self.context)
            != self.authorization.binding.nodes
        ):
            raise CustodyError(
                "completed node key activation no longer has its release/node binding"
            )
        return self.observe(snapshot, preserve_siblings=False)
