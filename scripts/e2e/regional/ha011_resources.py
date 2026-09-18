"""Read-only deployment binding and UID-owned isolated resource lifecycle."""

from __future__ import annotations

import base64
import copy
import json
import time
from datetime import datetime, timezone
from typing import Any, Literal, overload
from urllib.parse import quote

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional.ha011_contracts import (
    CASE_ID,
    DEPLOYMENT,
    FAILURE_STAGES,
    FAILURE_TYPES,
    IMAGE,
    LABEL,
    POD_NAME,
    PYTHON,
    ProofError,
    Settings,
    digest,
    image_digest,
    require_owned,
    require_subset,
)
from scripts.e2e.regional.ha011_cpu_nodes import cpu_node
from scripts.e2e.regional.ha011_manifests import (
    namespace_manifest,
    priority_class_manifest,
)
from scripts.e2e.regional.ha_plan_preflight import require_window
from scripts.e2e.regional.regional_commands import run_fixture_command

INTENT = "gpu-fault.nvidia.com/ha011-intent"


def canonical_resource(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    spec = result.get("spec", {})
    if result["kind"] == "Pod":
        for key in ("hostNetwork", "hostPID", "hostIPC"):
            spec.setdefault(key, False)
        spec.setdefault("initContainers", [])
        spec.setdefault("imagePullSecrets", [])
        spec.setdefault("schedulingGates", [])
        for container in spec.get("containers", []):
            container.setdefault("envFrom", [])
            container.setdefault("securityContext", {}).setdefault("privileged", False)
    elif result["kind"] == "NetworkPolicy":
        spec.setdefault("ingress", [])
        spec.setdefault("egress", [])
    elif result["kind"] == "PriorityClass":
        result.setdefault("globalDefault", False)
    return result


def verify_pod(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    node_name: str,
    scheduled: bool,
) -> None:
    actual = canonical_resource(observed)
    spec = actual["spec"]
    assigned = spec.pop("nodeName", None)
    if (scheduled and assigned != node_name) or assigned not in (None, node_name):
        raise ProofError("probe Pod was not placed on the bound CPU node")
    if spec.pop("serviceAccount", "default") != "default":
        raise ProofError("probe Pod acquired a different service account")
    if observed["metadata"].get("annotations", {}) != expected["metadata"].get(
        "annotations", {}
    ):
        raise ProofError("probe Pod acquired unapproved admission annotations")
    if spec != expected["spec"]:
        raise ProofError("admitted Pod spec contains changed or unapproved fields")


class CpuKubernetes:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def call(
        self,
        args: list[str],
        *,
        namespace: str | None = None,
        input_text: str | None = None,
        timeout: float = 30,
    ) -> str:
        command = [
            "kubectl",
            "--kubeconfig",
            str(self.settings.cpu_kubeconfig),
            "--context",
            self.settings.cpu_context,
        ]
        if namespace is not None:
            command.extend(["--namespace", namespace])
        return run_fixture_command(
            [*command, *args], input_text=input_text, timeout=timeout
        ).stdout

    @overload
    def read(
        self,
        kind: str,
        name: str,
        *,
        namespace: str | None = None,
        optional: Literal[False] = False,
    ) -> dict[str, Any]: ...

    @overload
    def read(
        self,
        kind: str,
        name: str,
        *,
        namespace: str | None = None,
        optional: Literal[True],
    ) -> dict[str, Any] | None: ...

    def read(
        self,
        kind: str,
        name: str,
        *,
        namespace: str | None = None,
        optional: bool = False,
    ) -> dict[str, Any] | None:
        args = ["get", kind, name, "-o", "json"]
        if optional:
            args.append("--ignore-not-found=true")
        output = self.call(args, namespace=namespace)
        if optional and not output.strip():
            return None
        try:
            value = json.loads(output)
        except ValueError:
            raise ProofError("Kubernetes read did not produce a JSON object") from None
        if (
            not isinstance(value, dict)
            or value.get("kind") != kind
            or value.get("metadata", {}).get("name") != name
            or not isinstance(value["metadata"].get("uid"), str)
            or not value["metadata"]["uid"]
            or (
                namespace is not None
                and value["metadata"].get("namespace") != namespace
            )
        ):
            raise ProofError("Kubernetes read returned a different resource")
        return value

    def identity(self) -> dict[str, Any]:
        namespace = self.settings.namespace
        scope = self.read("Namespace", namespace)
        release = self.read(
            "ConfigMap", "gpu-fault-regional-release-state", namespace=namespace
        )
        deployment = self.read("Deployment", DEPLOYMENT, namespace=namespace)
        state = json.loads(release["data"]["state.json"])
        release_id = state.get("release_id")
        containers = deployment["spec"]["template"]["spec"]["containers"]
        if not isinstance(release_id, str) or not release_id or len(containers) != 1:
            raise ProofError(
                "deployed release or runtime container identity is ambiguous"
            )
        image = containers[0]["image"]
        if not IMAGE.fullmatch(image):
            raise ProofError(
                "deployed control-plane runtime image is not digest-pinned"
            )
        status = deployment.get("status", {})
        replicas = deployment["spec"].get("replicas", 1)
        if (
            type(replicas) is not int
            or replicas <= 0
            or status.get("observedGeneration")
            != deployment["metadata"].get("generation")
            or status.get("updatedReplicas") != replicas
            or status.get("readyReplicas") != replicas
        ):
            raise ProofError(
                "the deployed control worker is not fully rolled out and Ready"
            )
        pods = json.loads(
            self.call(
                ["get", "pods", "-l", f"app={DEPLOYMENT}", "-o", "json"],
                namespace=namespace,
            )
        )["items"]
        image_ids = set()
        pod_uids = []
        for pod in pods:
            pod_containers = pod["spec"]["containers"]
            statuses = pod.get("status", {}).get("containerStatuses", [])
            if (
                len(pod_containers) != 1
                or pod_containers[0]["image"] != image
                or len(statuses) != 1
                or statuses[0].get("name") != pod_containers[0]["name"]
                or statuses[0].get("ready") is not True
                or pod["metadata"].get("deletionTimestamp")
                or pod["metadata"].get("namespace") != namespace
                or not isinstance(pod["metadata"].get("uid"), str)
                or not pod["metadata"]["uid"]
            ):
                raise ProofError("business worker Pod readiness or image drifted")
            image_ids.add(statuses[0].get("imageID"))
            pod_uids.append(pod["metadata"]["uid"])
        if len(pods) != replicas or len(image_ids) != 1:
            raise ProofError("running business worker image identity is not unique")
        image_id = image_digest(image_ids.pop())
        selected_node = sorted(pod["spec"]["nodeName"] for pod in pods)[0]
        return {
            "release_id": release_id,
            "cluster_id": self.settings.cluster_id,
            "region": self.settings.region,
            "namespace_uid": scope["metadata"]["uid"],
            "deployment_uid": deployment["metadata"]["uid"],
            "deployment_spec_sha256": digest(deployment["spec"]),
            "release_state_sha256": digest(state),
            "runtime_image": image,
            "runtime_image_id": image_id,
            "business_pod_uids": sorted(pod_uids),
            "cpu_node": cpu_node(self, selected_node),
        }


class IsolatedResources:
    def __init__(
        self,
        kubernetes: CpuKubernetes,
        *,
        intent_sha256: str,
        deadline: datetime,
        cpu_node_identity: dict[str, Any],
    ) -> None:
        self.kubernetes = kubernetes
        self.settings = kubernetes.settings
        self.deadline = deadline
        self.cpu_node = cpu_node_identity
        self.intent_sha256 = intent_sha256
        self.namespace = namespace_manifest(self.settings)
        self.namespace["metadata"]["annotations"] = {INTENT: intent_sha256}
        self.namespace_attempted = False
        self.namespace_uid: str | None = None
        self.receipts: dict[str, str] = {}
        self.expected: dict[str, dict[str, Any]] = {}
        self.pod: dict[str, Any] | None = None
        self.failure_details: dict[str, Any] | None = None

    def create(self, value: dict[str, Any]) -> dict[str, Any]:
        kind, metadata = value["kind"], value["metadata"]
        names = {
            "Namespace": self.settings.isolated_namespace,
            "PriorityClass": self.settings.priority_class_name,
            "Pod": POD_NAME,
            "Secret": "private-postgres",
            "ConfigMap": "probe-source",
            "NetworkPolicy": "loopback-only",
        }
        namespace = (
            None
            if kind in {"Namespace", "PriorityClass"}
            else self.settings.isolated_namespace
        )
        if (
            kind not in names
            or metadata.get("name") != names[kind]
            or metadata.get("namespace") != namespace
            or metadata.get("labels", {}).get(LABEL) != self.settings.isolation_id
        ):
            raise ProofError("creation target is outside the owned isolation boundary")
        if kind == "PriorityClass" and (
            not self.namespace_uid
            or value != priority_class_manifest(self.settings, self.namespace_uid)
        ):
            raise ProofError("PriorityClass intent or namespace owner is unproven")
        require_window(self.deadline, required_seconds=35)
        output = self.kubernetes.call(
            ["create", "-f", "-", "-o", "jsonpath={.metadata.uid}"],
            namespace=metadata.get("namespace"),
            input_text=json.dumps(value),
        )
        uid = output.strip()
        if not uid:
            raise ProofError("resource creation returned no UID")
        key = f"{kind}/{metadata['name']}"
        # An acknowledged identity survives any subsequent read or validation failure.
        self.receipts[key] = uid
        if kind == "Namespace":
            self.namespace_uid = uid
        observed = self.kubernetes.read(
            kind, metadata["name"], namespace=metadata.get("namespace")
        )
        require_owned(observed, self.settings.isolation_id, uid)
        expected = dict(value)
        if kind == "Secret":
            expected.pop("stringData")
            expected["data"] = {
                key: base64.b64encode(text.encode()).decode()
                for key, text in value["stringData"].items()
            }
        require_subset(canonical_resource(observed), expected)
        if kind == "PriorityClass":
            self.verify_priority_class(observed)
        if kind == "Pod":
            verify_pod(
                observed, expected, node_name=self.cpu_node["name"], scheduled=False
            )
        if (
            kind == "NetworkPolicy"
            and canonical_resource(observed)["spec"] != expected["spec"]
        ):
            raise ProofError("network isolation policy changed during admission")
        self.expected[key] = expected
        return observed

    def start(self, resources: list[dict[str, Any]]) -> None:
        if cpu_node(self.kubernetes, self.cpu_node["name"]) != self.cpu_node:
            raise ProofError("CPU node inventory changed before resource placement")
        name = self.settings.isolated_namespace
        if self.kubernetes.read("Namespace", name, optional=True) is not None:
            raise ProofError("refusing to adopt an existing isolated namespace")
        if (
            self.kubernetes.read(
                "PriorityClass", self.settings.priority_class_name, optional=True
            )
            is not None
        ):
            raise ProofError("refusing to adopt an existing isolated PriorityClass")
        self.namespace_attempted = True
        namespace = self.create(self.namespace)
        self.namespace_uid = namespace["metadata"]["uid"]
        for resource in resources:
            self.require_namespace()
            if resource["kind"] == "PriorityClass":
                if resource != priority_class_manifest(self.settings):
                    raise ProofError("unapproved PriorityClass intent")
                resource = priority_class_manifest(self.settings, self.namespace_uid)
            observed = self.create(resource)
            if resource["kind"] == "Pod":
                self.pod = resource
                self.receipts["Pod/" + POD_NAME] = observed["metadata"]["uid"]

    def require_namespace(self) -> None:
        current = self.kubernetes.read("Namespace", self.settings.isolated_namespace)
        require_owned(current, self.settings.isolation_id, self.namespace_uid)
        require_subset(current, self.namespace)

    def verify_priority_class(
        self, observed: dict[str, Any], *, allow_deleting: bool = False
    ) -> None:
        key = "PriorityClass/" + self.settings.priority_class_name
        uid = self.receipts.get(key)
        if not uid or not self.namespace_uid:
            raise ProofError(
                "PriorityClass creation UID or namespace custody is missing"
            )
        expected = priority_class_manifest(self.settings, self.namespace_uid)
        ownership = observed
        if allow_deleting and observed["metadata"].get("deletionTimestamp"):
            ownership = copy.deepcopy(observed)
            ownership["metadata"].pop("deletionTimestamp")
        require_owned(ownership, self.settings.isolation_id, uid)
        metadata = observed["metadata"]
        if (
            observed.get("apiVersion") != expected["apiVersion"]
            or observed.get("kind") != "PriorityClass"
            or metadata.get("name") != self.settings.priority_class_name
            or metadata.get("namespace") not in (None, "")
            or metadata.get("ownerReferences")
            != expected["metadata"]["ownerReferences"]
            or metadata.get("annotations", {})
            != expected["metadata"].get("annotations", {})
            or metadata.get("finalizers")
            or type(observed.get("value")) is not int
            or observed["value"] != 0
            or observed.get("globalDefault", False) is not False
            or observed.get("preemptionPolicy") != "Never"
        ):
            raise ProofError("owned PriorityClass policy or namespace owner changed")

    def read_probe_pod(self, *, scheduled: bool) -> dict[str, Any]:
        if self.pod is None:
            raise ProofError("no owned probe Pod was created")
        self.require_namespace()
        pod = self.kubernetes.read(
            "Pod", POD_NAME, namespace=self.settings.isolated_namespace
        )
        require_owned(pod, self.settings.isolation_id, self.receipts["Pod/" + POD_NAME])
        verify_pod(pod, self.pod, node_name=self.cpu_node["name"], scheduled=scheduled)
        return pod

    def arm(self, runtime_image_id: str) -> None:
        expected_image_id = image_digest(runtime_image_id)
        self.release_scheduling_gate()
        expected_pod = self.pod
        if expected_pod is None:
            raise ProofError("no owned probe Pod was created")
        deadline = time.monotonic() + 50
        while time.monotonic() < deadline:
            require_window(self.deadline, required_seconds=35)
            pod = self.read_probe_pod(scheduled=False)
            by_name = {
                entry["name"]: entry
                for entry in pod.get("status", {}).get("containerStatuses", [])
            }
            if any(
                entry.get("state", {}).get("terminated") for entry in by_name.values()
            ):
                self.capture_failure(pod, stage="before-arm")
                raise ProofError("isolated container terminated before arm")
            if set(by_name) == {"runtime", "postgres"} and all(
                entry.get("state", {}).get("running") for entry in by_name.values()
            ):
                verify_pod(
                    pod, expected_pod, node_name=self.cpu_node["name"], scheduled=True
                )
                if (
                    image_digest(by_name["runtime"].get("imageID")) != expected_image_id
                    or any(entry.get("restartCount", 0) for entry in by_name.values())
                    or cpu_node(self.kubernetes, self.cpu_node["name"]) != self.cpu_node
                ):
                    raise ProofError(
                        "live CPU node or container identity changed before arm"
                    )
                self.verify_created_resources()
                image_digest(by_name["postgres"].get("imageID"))
                output = self.kubernetes.call(
                    [
                        "exec",
                        POD_NAME,
                        "-c",
                        "runtime",
                        "--",
                        PYTHON,
                        "-c",
                        "import json,sys; "
                        "from scripts.e2e.regional.probes.ha011_probe import arm; "
                        "print(json.dumps(arm(*sys.argv[1:])))",
                        pod["metadata"]["uid"],
                        self.settings.isolation_id,
                        self.intent_sha256,
                    ],
                    namespace=self.settings.isolated_namespace,
                )
                expected = {
                    "armed": True,
                    "pod_uid": pod["metadata"]["uid"],
                    "isolation_id": self.settings.isolation_id,
                    "intent_sha256": self.intent_sha256,
                }
                receipt = json.loads(output)
                if receipt != expected or receipt.get("armed") is not True:
                    raise ProofError("owned start barrier was not acknowledged")
                return
            time.sleep(0.1)
        raise ProofError("admitted probe Pod did not reach the bounded arm barrier")

    def release_scheduling_gate(self) -> None:
        expected_pod = self.pod
        if expected_pod is None:
            raise ProofError("no owned probe Pod was created")
        pod = self.read_probe_pod(scheduled=False)
        if pod["spec"].get("nodeName"):
            raise ProofError("probe Pod was scheduled before admission verification")
        version = pod["metadata"].get("resourceVersion")
        if not isinstance(version, str) or not version:
            raise ProofError(
                "admitted Pod has no resource version for conditional activation"
            )
        self.verify_created_resources()
        if cpu_node(self.kubernetes, self.cpu_node["name"]) != self.cpu_node:
            raise ProofError("CPU inventory changed before scheduling activation")
        require_window(self.deadline, required_seconds=35)
        operations = [
            {
                "op": "test",
                "path": "/metadata/uid",
                "value": self.receipts["Pod/" + POD_NAME],
            },
            {"op": "test", "path": "/metadata/resourceVersion", "value": version},
            {
                "op": "test",
                "path": "/spec/schedulingGates",
                "value": expected_pod["spec"]["schedulingGates"],
            },
            {"op": "remove", "path": "/spec/schedulingGates"},
        ]
        uid = self.kubernetes.call(
            [
                "patch",
                "Pod",
                POD_NAME,
                "--type=json",
                "--patch-file=/dev/stdin",
                "-o",
                "jsonpath={.metadata.uid}",
            ],
            namespace=self.settings.isolated_namespace,
            input_text=json.dumps(operations),
        ).strip()
        if uid != self.receipts["Pod/" + POD_NAME]:
            raise ProofError("scheduling activation did not acknowledge the owned Pod")
        self.pod = copy.deepcopy(expected_pod)
        self.pod["spec"]["schedulingGates"] = []
        self.read_probe_pod(scheduled=False)

    def verify_created_resources(self) -> None:
        for key, expected in self.expected.items():
            kind, name = key.split("/", 1)
            if kind in {"Namespace", "Pod"}:
                continue
            observed = self.kubernetes.read(
                kind,
                name,
                namespace=None
                if kind == "PriorityClass"
                else self.settings.isolated_namespace,
            )
            require_owned(observed, self.settings.isolation_id, self.receipts[key])
            require_subset(canonical_resource(observed), expected)
            if kind == "PriorityClass":
                self.verify_priority_class(observed)
            if (
                kind == "NetworkPolicy"
                and canonical_resource(observed)["spec"] != expected["spec"]
            ):
                raise ProofError("network isolation policy changed before arm")

    def collect(self, runtime_image_id: str) -> tuple[dict[str, Any], str]:
        expected_image_id = image_digest(runtime_image_id)
        if self.pod is None:
            raise ProofError("no owned probe Pod was created")
        deadline = min(
            time.monotonic() + 270,
            time.monotonic()
            + (self.deadline - datetime.now(timezone.utc)).total_seconds()
            - 35,
        )
        while time.monotonic() < deadline:
            pod = self.read_probe_pod(scheduled=True)
            uid = require_owned(
                pod, self.settings.isolation_id, self.receipts["Pod/" + POD_NAME]
            )
            statuses = pod.get("status", {}).get("containerStatuses", [])
            by_name = {entry["name"]: entry for entry in statuses}
            runtime = by_name.get("runtime", {})
            if by_name.get("postgres", {}).get("state", {}).get("terminated"):
                self.capture_failure(pod, stage="collect")
                raise ProofError("isolated database container terminated")
            if runtime.get("restartCount", 0):
                raise ProofError("probe runtime container restarted")
            terminated = runtime.get("state", {}).get("terminated")
            if terminated is not None:
                if (
                    terminated.get("exitCode") != 0
                    or image_digest(runtime.get("imageID")) != expected_image_id
                ):
                    self.capture_failure(pod, stage="collect")
                    raise ProofError(
                        "isolated runtime failed or ran a different deployed image"
                    )
                if not by_name.get("postgres", {}).get("state", {}).get("running"):
                    raise ProofError(
                        "isolated database did not survive the owned worker crash"
                    )
                output = self.kubernetes.call(
                    ["logs", POD_NAME, "-c", "runtime", "--limit-bytes=65536"],
                    namespace=self.settings.isolated_namespace,
                )
                result = json.loads(output)
                if not isinstance(result, dict):
                    raise ProofError("probe returned malformed evidence")
                return result, uid
            time.sleep(0.2)
        raise ProofError("isolated runtime observation deadline expired")

    def capture_failure(self, pod: dict[str, Any], *, stage: str) -> None:
        if stage not in {"before-arm", "collect"}:
            raise ProofError("unknown failure capture stage")
        statuses = pod.get("status", {}).get("containerStatuses", [])
        self.failure_details = {
            "stage": stage,
            "container_exits": {
                item["name"]: ended["exitCode"]
                for item in statuses
                if item.get("name") in {"runtime", "postgres"}
                and isinstance(
                    ended := (item.get("state") or {}).get("terminated"), dict
                )
                and type(ended.get("exitCode")) is int
            },
        }
        runtime: dict[str, Any] = next(
            (item for item in statuses if item.get("name") == "runtime"), {}
        )
        if not runtime.get("state", {}).get("terminated"):
            return
        try:
            raw = self.kubernetes.call(
                ["logs", POD_NAME, "-c", "runtime", "--limit-bytes=65536"],
                namespace=self.settings.isolated_namespace,
            )
            value = json.loads(raw)
        except ProcessSupervisionLost:
            raise
        except Exception:
            self.failure_details["probe_report"] = "unavailable"
            return
        if (
            isinstance(value, dict)
            and set(value) == {"case_id", "verdict", "stage", "error_type"}
            and value.get("case_id") == CASE_ID
            and value.get("verdict") == "FAIL"
            and isinstance(value.get("stage"), str)
            and isinstance(value.get("error_type"), str)
            and value.get("stage") in FAILURE_STAGES
            and value.get("error_type") in FAILURE_TYPES | {"OtherError"}
        ):
            self.failure_details["probe_stage"] = value["stage"]
            self.failure_details["probe_error_type"] = value["error_type"]
        else:
            self.failure_details["probe_report"] = "unrecognized"

    def cleanup(self) -> dict[str, Any]:
        if not self.namespace_attempted:
            return {"namespace_absent": True, "resource_uids": dict(self.receipts)}
        name = self.settings.isolated_namespace
        current = self.kubernetes.read("Namespace", name, optional=True)
        if current is None:
            return self._finish_cleanup()
        uid = require_owned(current, self.settings.isolation_id, self.namespace_uid)
        require_subset(current, self.namespace)
        for resource, expected_uid in self.receipts.items():
            kind, resource_name = resource.split("/", 1)
            if kind == "Namespace":
                continue
            observed = self.kubernetes.read(
                kind,
                resource_name,
                namespace=None if kind == "PriorityClass" else name,
                optional=True,
            )
            if observed is not None:
                if kind == "PriorityClass":
                    self.verify_priority_class(observed, allow_deleting=True)
                else:
                    require_owned(observed, self.settings.isolation_id, expected_uid)
        priority = self.kubernetes.read(
            "PriorityClass", self.settings.priority_class_name, optional=True
        )
        if priority is not None:
            self.verify_priority_class(priority, allow_deleting=True)
        options = {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "preconditions": {"uid": uid},
            "propagationPolicy": "Foreground",
        }
        self.kubernetes.call(
            [
                "delete",
                "--raw",
                f"/api/v1/namespaces/{quote(name, safe='')}",
                "-f",
                "-",
            ],
            input_text=json.dumps(options),
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            current = self.kubernetes.read("Namespace", name, optional=True)
            if current is None:
                return self._finish_cleanup()
            metadata = current["metadata"]
            if (
                metadata.get("uid") != uid
                or metadata.get("labels", {}).get(LABEL) != self.settings.isolation_id
            ):
                raise ProofError("isolated namespace was recreated during cleanup")
            time.sleep(0.2)
        raise ProofError("owned namespace cleanup did not finish")

    def _finish_cleanup(self) -> dict[str, Any]:
        name = self.settings.priority_class_name
        current = self.kubernetes.read("PriorityClass", name, optional=True)
        if current is not None:
            self.verify_priority_class(current, allow_deleting=True)
            uid = self.receipts["PriorityClass/" + name]
            version = current["metadata"].get("resourceVersion")
            if not isinstance(version, str) or not version:
                raise ProofError("PriorityClass deletion requires a resource version")
            if not current["metadata"].get("deletionTimestamp"):
                self.kubernetes.call(
                    [
                        "delete",
                        "--raw",
                        f"/apis/scheduling.k8s.io/v1/priorityclasses/{quote(name, safe='')}",
                        "-f",
                        "-",
                    ],
                    input_text=json.dumps(
                        {
                            "apiVersion": "v1",
                            "kind": "DeleteOptions",
                            "preconditions": {"uid": uid, "resourceVersion": version},
                            "propagationPolicy": "Foreground",
                        }
                    ),
                )
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                current = self.kubernetes.read("PriorityClass", name, optional=True)
                if current is None:
                    break
                if (
                    current["metadata"].get("uid") != uid
                    or current["metadata"].get("labels", {}).get(LABEL)
                    != self.settings.isolation_id
                ):
                    raise ProofError(
                        "isolated PriorityClass was recreated during cleanup"
                    )
                time.sleep(0.2)
            else:
                raise ProofError("owned PriorityClass cleanup did not finish")
        return {
            "namespace_absent": True,
            "priority_class_absent": True,
            "resource_uids": dict(self.receipts),
        }
