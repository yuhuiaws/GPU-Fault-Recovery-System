"""An in-memory one-cluster site for the synthetic-secondary tests.

The real ``IdentitySite`` and ``RegionalLiveFixture`` run unchanged; only the
process boundary is replaced: ``RegionalLiveFixture.run`` becomes a kubectl
interpreter over an in-memory GPU plane (namespaces, Secrets, ConfigMaps, Pods,
the primary's executor Deployment) and control plane (the API Pod, the durable
registry behind ``REGISTRY_API_PROBE`` / ``REGISTRY_REVISION_PROBE``, the
release-state ConfigMap). Claim probes exec'd into a Pod answer the way the
deployed middleware does -- from that Pod's own connection environment against
the registry rows -- so a disabled or removed registration is observed exactly
where the runner looks for it.
"""

from __future__ import annotations

import base64
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from gpu_fault.regional import RegionalClusterRegistration
from gpu_fault.regional_compatibility import CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
from scripts.e2e.regional import identity_acceptance_common as common
from scripts.e2e.regional import identity_synthetic_secondary as synthetic
from scripts.e2e.regional.identity_auth_probes import AUTH008_EXECUTOR_IDENTITY_PROBE
from scripts.e2e.regional.regional_commands import RegionalCommandFailed
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture
from tests.regional._regional_support import TOKEN_A, registration

SITE_NAMESPACE = "gpu-system"
PRIMARY = "cluster-a"
SECONDARY = "auth-logical-b"
RELEASE_ID = "release-test"
PRIMARY_EXECUTOR_POD = "gpu-fault-cluster-executor-primary"
API_POD = "gpu-fault-api-ha-0"
IMAGE = "registry.example/executor@sha256:" + "e" * 64
ARTIFACT = "a" * 64
COMPATIBILITY = "b" * 64
# What the CPU Pods export and gpu-fault-release-metadata declares; the AUTH-008
# CPU probe binds its receipt to these, never to GPU_FAULT_RELEASE_ID.
RELEASE_PINS = {
    "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": ARTIFACT,
    "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "c" * 64,
    "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST": "d" * 64,
}
# The in-process API derives its executor policy from the same environment.
POD_ENVIRONMENT = {
    **RELEASE_PINS,
    "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST": COMPATIBILITY,
}
RELEASE_METADATA_DATA = {
    "required-regional-executor-artifact-sha256": ARTIFACT,
    "required-agent-artifact-sha256": "c" * 64,
    "required-agent-compatibility-digest": "d" * 64,
}
CONTROL_PLANE_URL = "https://control.unit.invalid"
CA_PEM = "-----BEGIN CERTIFICATE-----\nunit\n-----END CERTIFICATE-----\n"
NOT_REGISTERED = json.dumps({"detail": "regional cluster is not registered"})
AUTH_FAILED = json.dumps({"detail": "regional cluster authentication failed"})

ExecHook = Callable[[dict[str, str], list[str]], dict[str, Any]]


def encode(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


def decode(value: str) -> str:
    return base64.b64decode(value).decode()


class FakeSite:
    """State of one site plus the kubectl interpreter that serves it."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.cpu_kubeconfig = tmp_path / "cpu"
        self.gpu_kubeconfig = tmp_path / "gpu"
        for path in (self.cpu_kubeconfig, self.gpu_kubeconfig, tmp_path / "site"):
            path.write_text("unit connection fixture\n", encoding="utf-8")
        self.generation = 3
        self.registry: list[dict[str, Any]] = [
            registration(PRIMARY, TOKEN_A).model_dump(mode="json")
        ]
        self.revisions: list[dict[str, Any]] = []
        self.namespaces: dict[str, dict[str, Any]] = {
            SITE_NAMESPACE: {
                "name": SITE_NAMESPACE,
                "uid": "uid-site",
                "resourceVersion": "1",
                "labels": {},
            }
        }
        self.resources: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.calls: list[list[str]] = []
        self.created: list[str] = []
        self.deleted: list[str] = []
        self.claims: list[dict[str, Any]] = []
        self.exec_hooks: dict[str, ExecHook] = {}
        self.on_publish: Callable[[list[dict[str, Any]]], None] | None = None
        self.pods_never_ready = False
        self.uid_counter = 0
        self._seed_primary()

    # -- setup -----------------------------------------------------------------

    def _seed_primary(self) -> None:
        pod = synthetic.secondary_pod_manifest(
            cluster_id=PRIMARY,
            namespace=SITE_NAMESPACE,
            run_id="release",
            case_id="release",
            image=IMAGE,
            pins={"artifact": ARTIFACT, "compatibility": COMPATIBILITY},
            lifetime_seconds=3600,
        )
        pod["metadata"]["name"] = PRIMARY_EXECUTOR_POD
        pod["metadata"]["labels"] = {"app": common.EXECUTOR_APP}
        self._put(
            SITE_NAMESPACE,
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": common.EXECUTOR_APP, "namespace": SITE_NAMESPACE},
                "spec": {"template": {"spec": pod["spec"]}},
            },
        )
        self._put(
            SITE_NAMESPACE,
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": common.CONNECTION_SECRET,
                    "namespace": SITE_NAMESPACE,
                },
                "data": {
                    "control-plane-url": encode(CONTROL_PLANE_URL),
                    "ca.crt": encode(CA_PEM),
                    "cluster-id": encode(PRIMARY),
                    "cluster-token": encode(TOKEN_A),
                    "allowed-namespaces": encode("training"),
                },
            },
        )
        self._put(SITE_NAMESPACE, pod)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> common.IdentitySite:
        """The real IdentitySite over this fake's kubectl."""

        config = {
            "namespace": SITE_NAMESPACE,
            "aws_region": "us-west-2",
            "cpu_kubeconfig": str(self.cpu_kubeconfig),
            "gpu_kubeconfig": str(self.gpu_kubeconfig),
            "clusters": [
                {
                    "cluster_id": PRIMARY,
                    "context": "context-a",
                    "region": "us-west-2",
                    "hyperpod_cluster_name": "hp-a",
                    "eks_cluster_arn": "arn:aws:eks:us-west-2:000000000000:cluster/a",
                    "executor_irsa_role_arn": "arn:aws:iam::000000000000:role/a",
                    "control_plane_url": CONTROL_PLANE_URL,
                    "ca_file": str(self.tmp_path / "public-ca"),
                }
            ],
        }
        monkeypatch.setattr(
            common,
            "load_site",
            lambda *args, **kwargs: SimpleNamespace(
                release_config=config,
                environment={"KUBECONFIG": str(self.gpu_kubeconfig)},
            ),
        )
        monkeypatch.setattr(RegionalLiveFixture, "run", staticmethod(self.run))
        return common.IdentitySite(self.tmp_path / "site")

    # -- state helpers -----------------------------------------------------------

    def _next_uid(self) -> str:
        self.uid_counter += 1
        return f"uid-{self.uid_counter}"

    def _put(self, namespace: str, manifest: dict[str, Any]) -> dict[str, Any]:
        kind = str(manifest["kind"]).lower()
        name = str(manifest["metadata"]["name"])
        metadata = {
            "name": name,
            "namespace": namespace,
            "uid": self._next_uid(),
            "resourceVersion": "1",
            "labels": dict(manifest["metadata"].get("labels") or {}),
        }
        self.resources[(namespace, kind, name)] = {
            "manifest": copy.deepcopy(manifest),
            "metadata": metadata,
        }
        return metadata

    def registration_of(self, cluster_id: str) -> dict[str, Any] | None:
        return next(
            (row for row in self.registry if row["cluster_id"] == cluster_id), None
        )

    def pod_environment(self, namespace: str, pod: str) -> dict[str, str]:
        record = self.resources[(namespace, "pod", pod)]
        environment: dict[str, str] = {}
        for item in record["manifest"]["spec"]["containers"][0]["env"]:
            if "value" in item:
                environment[item["name"]] = str(item["value"])
                continue
            reference = item["valueFrom"]["secretKeyRef"]
            secret = self.resources[(namespace, "secret", reference["name"])]
            environment[item["name"]] = decode(
                secret["manifest"]["data"][reference["key"]]
            )
        return environment

    def secondary_token(self) -> str:
        secret = self.resources[(SECONDARY, "secret", common.CONNECTION_SECRET)]
        return decode(secret["manifest"]["data"]["cluster-token"])

    # -- kubectl ---------------------------------------------------------------

    def run(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
        timeout: int = 300,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del timeout, cwd, env
        self.calls.append(list(command))
        assert command[0] == "kubectl", command
        plane = "gpu" if "--context" in command else "cpu"
        namespace = None
        arguments: list[str] = []
        index = 1
        while index < len(command):
            token = command[index]
            if token in {"--kubeconfig", "--context"}:
                index += 2
                continue
            if token == "-n":
                namespace = command[index + 1]
                index += 2
                continue
            arguments.append(token)
            index += 1
        try:
            output = self._dispatch(plane, namespace, arguments, input_text)
        except RegionalCommandFailed as exc:
            if check:
                raise
            return subprocess.CompletedProcess(command, 1, "", str(exc))
        return subprocess.CompletedProcess(command, 0, output, "")

    def _dispatch(
        self,
        plane: str,
        namespace: str | None,
        arguments: list[str],
        input_text: str | None,
    ) -> str:
        verb = arguments[0]
        if verb == "get":
            return self._get(plane, namespace, arguments[1:])
        if verb == "create":
            return self._create(namespace, input_text)
        if verb == "wait":
            return self._wait(namespace, arguments[1:])
        if verb == "exec":
            return self._exec(plane, namespace, arguments[1:], input_text)
        if verb == "delete":
            return self._delete(arguments[1:], input_text)
        raise RegionalCommandFailed(1, f"unconfigured kubectl verb: {arguments}")

    def _pod_document(self, namespace: str, app: str) -> dict[str, Any]:
        items = []
        for (item_namespace, kind, _name), record in sorted(self.resources.items()):
            if item_namespace != namespace or kind != "pod":
                continue
            if record["metadata"]["labels"].get("app") != app:
                continue
            containers = record["manifest"]["spec"]["containers"]
            ready = not self.pods_never_ready
            items.append(
                {
                    "metadata": record["metadata"],
                    "spec": {
                        "containers": [{"name": c["name"]} for c in containers],
                        "nodeName": "node-1",
                    },
                    "status": {
                        "phase": "Running",
                        "conditions": [
                            {"type": "Ready", "status": "True" if ready else "False"}
                        ],
                        "containerStatuses": [
                            {"name": c["name"], "ready": ready} for c in containers
                        ],
                    },
                }
            )
        return {"items": items}

    def _get(self, plane: str, namespace: str | None, arguments: list[str]) -> str:
        kind = arguments[0]
        ignore_not_found = "--ignore-not-found" in arguments
        output = arguments[arguments.index("-o") + 1] if "-o" in arguments else ""
        if kind == "pod" and "-l" in arguments:
            app = arguments[arguments.index("-l") + 1].removeprefix("app=")
            if plane == "cpu":
                return json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {"name": API_POD, "uid": "uid-api"},
                                "spec": {"containers": [{"name": "api"}]},
                                "status": {
                                    "phase": "Running",
                                    "conditions": [{"type": "Ready", "status": "True"}],
                                    "containerStatuses": [
                                        {"name": "api", "ready": True}
                                    ],
                                },
                            }
                        ]
                    }
                )
            return json.dumps(self._pod_document(namespace or "", app))
        if kind == "configmap" and plane == "cpu":
            if arguments[1] == "gpu-fault-release-metadata":
                return json.dumps({"data": dict(RELEASE_METADATA_DATA)})
            return json.dumps(
                {"data": {"state.json": json.dumps({"release_id": RELEASE_ID})}}
            )
        if (
            kind == "secret"
            and plane == "cpu"
            and arguments[1] == common.REGISTRY_SECRET
        ):
            # The bootstrap copy a site without a durable head still reads.
            return json.dumps(
                {"data": {"clusters.json": encode(json.dumps(self.registry))}}
            )
        name = arguments[1]
        if kind == "namespace":
            record = self.namespaces.get(name)
            if record is None:
                if ignore_not_found:
                    return ""
                raise RegionalCommandFailed(1, f'namespaces "{name}" not found')
            return json.dumps(record)
        record = self.resources.get((namespace or "", kind, name))
        if record is None:
            if ignore_not_found:
                return ""
            raise RegionalCommandFailed(1, f'{kind} "{name}" not found')
        if output == "jsonpath={.metadata}":
            return json.dumps(record["metadata"])
        if output == "jsonpath={.spec.nodeName}":
            return "node-1"
        return json.dumps(record["manifest"])

    def _create(self, namespace: str | None, input_text: str | None) -> str:
        manifest = json.loads(input_text or "")
        kind = str(manifest["kind"]).lower()
        name = str(manifest["metadata"]["name"])
        if kind == "namespace":
            if name in self.namespaces:
                raise RegionalCommandFailed(1, f'namespaces "{name}" already exists')
            metadata = {
                "name": name,
                "uid": self._next_uid(),
                "resourceVersion": "1",
                "labels": dict(manifest["metadata"].get("labels") or {}),
            }
            self.namespaces[name] = metadata
            self.created.append(f"namespace/{name}")
            return json.dumps(metadata)
        target = str(manifest["metadata"].get("namespace") or namespace or "")
        if target not in self.namespaces:
            raise RegionalCommandFailed(1, f'namespaces "{target}" not found')
        if (target, kind, name) in self.resources:
            raise RegionalCommandFailed(1, f'{kind} "{name}" already exists')
        metadata = self._put(target, manifest)
        self.created.append(f"{kind}/{target}/{name}")
        return json.dumps(metadata)

    def _wait(self, namespace: str | None, arguments: list[str]) -> str:
        name = arguments[1].removeprefix("pod/")
        if (
            namespace or "",
            "pod",
            name,
        ) not in self.resources or self.pods_never_ready:
            raise RegionalCommandFailed(1, "timed out waiting for the condition")
        return f"pod/{name} condition met"

    def _claim(self, environment: dict[str, str]) -> dict[str, Any]:
        cluster_id = environment["GPU_FAULT_CLUSTER_ID"]
        token = environment["GPU_FAULT_CONTROL_PLANE_TOKEN"]
        self.claims.append(
            {"cluster_id": cluster_id, "token_sha256": common.secret_digest(token)}
        )
        if (
            environment.get("GPU_FAULT_EXECUTOR_ARTIFACT_SHA256") != ARTIFACT
            or environment.get("GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST")
            != COMPATIBILITY
        ):
            return {"status": 503, "latency_seconds": 0.01, "detail": "pin mismatch"}
        row = self.registration_of(cluster_id)
        if row is None:
            return {"status": 403, "latency_seconds": 0.01, "detail": NOT_REGISTERED}
        model = RegionalClusterRegistration.model_validate(row)
        if model.matched_token_slot(token) is None:
            return {"status": 403, "latency_seconds": 0.01, "detail": AUTH_FAILED}
        return {"status": 200, "latency_seconds": 0.01, "command_count": 0}

    def _registry_api(self, arguments: list[str]) -> dict[str, Any]:
        method, path = arguments[0], arguments[1]
        if method == "GET" and path.endswith("/status"):
            return {
                "status": 200,
                "body": {
                    "generation": self.generation,
                    "content_sha256": "0" * 64,
                    "missing_member_ids": [],
                    "converged": True,
                },
            }
        if method == "POST" and path.endswith("/revisions"):
            payload = json.loads(arguments[2])
            if payload["expected_generation"] != self.generation:
                return {"status": 409, "body": {"detail": "generation conflict"}}
            try:
                rows = [
                    RegionalClusterRegistration.model_validate(item).model_dump(
                        mode="json"
                    )
                    for item in payload["registrations"]
                ]
            except ValueError as exc:
                return {"status": 422, "body": {"detail": str(exc)}}
            self.registry = rows
            self.generation += 1
            self.revisions.append(
                {
                    "generation": self.generation,
                    "reason": payload["reason"],
                    "cluster_ids": [row["cluster_id"] for row in rows],
                }
            )
            if self.on_publish is not None:
                self.on_publish(rows)
            return {
                "status": 200,
                "body": {"generation": self.generation, "content_sha256": "0" * 64},
            }
        raise RegionalCommandFailed(1, f"unconfigured registry call: {arguments}")

    def _exec(
        self,
        plane: str,
        namespace: str | None,
        arguments: list[str],
        input_text: str | None,
    ) -> str:
        pod = arguments[1] if arguments[0] == "-i" else arguments[0]
        separator = arguments.index("--")
        script_arguments = arguments[separator + 3 :]
        script = input_text or ""
        if plane == "cpu":
            assert pod == API_POD, pod
            if script == common.REGISTRY_API_PROBE:
                return json.dumps(self._registry_api(script_arguments))
            if script == common.REGISTRY_REVISION_PROBE:
                return json.dumps(
                    {
                        "generation": self.generation,
                        "registrations": copy.deepcopy(self.registry),
                    }
                )
            hook = self.exec_hooks.get(script)
            if hook is not None:
                return json.dumps(hook({}, script_arguments))
            raise RegionalCommandFailed(1, "unconfigured control-plane probe")
        if (namespace or "", "pod", pod) not in self.resources:
            raise RegionalCommandFailed(1, f'pods "{pod}" not found')
        environment = self.pod_environment(namespace or "", pod)
        if script == common.CLAIM_PROBE:
            return json.dumps(self._claim(environment))
        if script == AUTH008_EXECUTOR_IDENTITY_PROBE:
            return json.dumps(
                {
                    "cluster_id": environment["GPU_FAULT_CLUSTER_ID"],
                    "executor_id": environment.get(
                        "GPU_FAULT_CLUSTER_EXECUTOR_ID",
                        environment["GPU_FAULT_CLUSTER_ID"] + "/" + pod,
                    ),
                    "artifact": environment["GPU_FAULT_EXECUTOR_ARTIFACT_SHA256"],
                    "compatibility": environment[
                        "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST"
                    ],
                    "protocol": CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
                    "probe_owner_configured": False,
                }
            )
        hook = self.exec_hooks.get(script)
        if hook is not None:
            return json.dumps(hook(environment, script_arguments))
        raise RegionalCommandFailed(1, "unconfigured data-plane probe")

    def _delete(self, arguments: list[str], input_text: str | None) -> str:
        assert arguments[0] == "--raw", arguments
        path = arguments[1].strip("/").split("/")
        options = json.loads(input_text or "{}")
        preconditions = options.get("preconditions") or {}
        if path[:3] == ["api", "v1", "namespaces"] and len(path) == 4:
            name = path[3]
            record = self.namespaces.get(name)
            if record is None:
                raise RegionalCommandFailed(1, f'namespaces "{name}" not found')
            self._check_preconditions(record, preconditions)
            del self.namespaces[name]
            for key in [key for key in self.resources if key[0] == name]:
                del self.resources[key]
            self.deleted.append(f"namespace/{name}")
            return ""
        namespace, plural, name = path[3], path[4], path[5]
        kind = plural.removesuffix("s")
        record = self.resources.get((namespace, kind, name))
        if record is None:
            raise RegionalCommandFailed(1, f'{kind} "{name}" not found')
        self._check_preconditions(record["metadata"], preconditions)
        del self.resources[(namespace, kind, name)]
        self.deleted.append(f"{kind}/{namespace}/{name}")
        return ""

    @staticmethod
    def _check_preconditions(
        metadata: dict[str, Any], preconditions: dict[str, Any]
    ) -> None:
        if preconditions.get("uid") != metadata["uid"] or (
            preconditions.get("resourceVersion") != metadata["resourceVersion"]
        ):
            raise RegionalCommandFailed(
                1, "the object has been modified; preconditions failed"
            )

    # -- readings for assertions ---------------------------------------------------

    def owned_resources(self) -> list[str]:
        return sorted(
            f"{kind}/{namespace}/{name}"
            for (namespace, kind, name) in self.resources
            if namespace != SITE_NAMESPACE
        )
