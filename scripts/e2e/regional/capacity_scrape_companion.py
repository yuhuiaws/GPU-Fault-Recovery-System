"""Owned, probe-token-only ADOT companion for native CAP002 acceptance."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import yaml  # type: ignore[import-untyped]

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.capacity_scrape_config import (
    ScrapeCompanionError as ScrapeCompanionError,
    digest,
    name,
    pipeline,
    require,
    workload_identity,
)
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional.fixture_ownership import FixtureOwnership, creation_document
from scripts.e2e.regional.regional_commands import run_fixture_command
from scripts.e2e.regional.regional_pod_inventory import ready_pod_records

ADOT = "gpu-fault-adot"
GATE = "gpu-fault.io/capacity-scrape-verified"
OWNER = "gpu-fault.io/acceptance-run"
CLEANUP_SECONDS = 120
STARTUP_SECONDS = 180
MAX_LIFETIME_SECONDS = 1800
POLL_SECONDS = 1.0
METRICS = "up|gpu_fault_store_io_in_flight|gpu_fault_store_io_max_in_flight"


def build_scrape_config(
    *, source: dict[str, Any], service: str, namespace: str, pod: str, run_id: str
) -> dict[str, Any]:
    """Pure builder for local validation with the actual pinned collector image."""
    service, namespace, pod, run_id = (
        name(service),
        name(namespace),
        name(pod, limit=253),
        name(run_id),
    )
    try:
        original = source["pipeline"]
        labels = source["labels"]
        require(
            isinstance(original, dict)
            and set(original) == {"processors", "exporters", "extensions", "service"}
            and isinstance(labels, dict)
            and set(labels) == {"region", "control_plane_cluster"},
            "scrape builder source is incomplete",
        )
        endpoint = original["exporters"]["prometheusremotewrite"]["endpoint"]
        match = re.fullmatch(
            r"/workspaces/(ws-[a-z0-9-]+)/api/v1/remote_write",
            urlsplit(endpoint).path,
        )
        require(match is not None, "scrape builder AMP endpoint is invalid")
        assert match is not None
        validation = copy.deepcopy(original)
        validation["receivers"] = {
            "prometheus": {
                "config": {
                    "scrape_configs": [
                        {
                            "job_name": "gpu-fault-control-plane",
                            "relabel_configs": [
                                {
                                    "action": "replace",
                                    "target_label": key,
                                    "replacement": value,
                                }
                                for key, value in labels.items()
                            ],
                        }
                    ]
                }
            }
        }
        config, labels = pipeline(
            yaml.safe_dump(validation),
            region=labels["region"],
            workspace_id=match.group(1),
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError):
        raise ScrapeCompanionError("scrape builder source is invalid") from None
    config["receivers"] = {
        "prometheus": {
            "config": {
                "scrape_configs": [
                    {
                        "job_name": f"capacity-{run_id}",
                        "scrape_interval": "15s",
                        "scrape_timeout": "10s",
                        "metrics_path": "/metrics",
                        "scheme": "http",
                        "honor_labels": False,
                        "authorization": {
                            "type": "Bearer",
                            "credentials_file": "/etc/capacity-token/execution-token",
                        },
                        "static_configs": [
                            {
                                "targets": [f"{service}.{namespace}.svc:18080"],
                                "labels": {
                                    **labels,
                                    "pod": pod,
                                    "capacity_run": run_id,
                                },
                            }
                        ],
                        "metric_relabel_configs": [
                            {
                                "action": "keep",
                                "source_labels": ["__name__"],
                                "regex": METRICS,
                            }
                        ],
                    }
                ]
            }
        }
    }
    # Prometheus adds scrape_* metrics after metric relabeling. Filter again
    # after reception, and disable the exporter's synthesized target_info.
    config["processors"]["filter/capacity"] = {
        "metrics": {
            "include": {
                "match_type": "strict",
                "metric_names": METRICS.split("|"),
            }
        },
    }
    config["service"]["pipelines"]["metrics"]["processors"] = [
        "filter/capacity",
        "batch",
    ]
    config["exporters"]["prometheusremotewrite"]["target_info"] = {"enabled": False}
    return config


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _metadata(
    value: Any, kind: str, resource: str, namespace: str | None
) -> dict[str, Any]:
    require(
        isinstance(value, dict)
        and value.get("kind") == kind
        and isinstance(value.get("metadata"), dict),
        "scrape resource response is invalid",
    )
    meta = value["metadata"]
    require(
        meta.get("name") == resource
        and (namespace is None or meta.get("namespace") == namespace)
        and all(
            isinstance(meta.get(key), str) and meta[key]
            for key in ("uid", "resourceVersion")
        ),
        "scrape resource identity is incomplete",
    )
    return dict(meta)


def _file_identity(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    try:
        with path.open("rb") as handle:
            sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError:
        raise ScrapeCompanionError("CPU kubeconfig identity is unavailable") from None
    return {"path": str(path), "sha256": sha256}


def _image(value: Any) -> str:
    require(isinstance(value, str), "ADOT running image digest is missing")
    value = re.sub(r"^(?:docker-pullable|containerd)://", "", value)
    require(
        re.fullmatch(r"[a-z0-9][a-z0-9./:_-]*@sha256:[0-9a-f]{64}", value) is not None,
        "ADOT running image is not a pullable digest reference",
    )
    return str(value)


def _node(value: Any, node_name: str) -> dict[str, str]:
    meta = _metadata(value, "Node", node_name, None)
    status = value.get("status") or {}
    conditions = status.get("conditions") or []
    ready = [
        entry.get("status")
        for entry in conditions
        if isinstance(entry, dict) and entry.get("type") == "Ready"
    ]
    info = status.get("nodeInfo") or {}
    require(
        ready == ["True"]
        and info.get("operatingSystem") == "linux"
        and isinstance(info.get("bootID"), str)
        and bool(info["bootID"])
        and str((status.get("allocatable") or {}).get("nvidia.com/gpu", "0")) == "0"
        and not meta.get("deletionTimestamp"),
        "scrape node is not a known Ready CPU node",
    )
    return {"name": node_name, "uid": meta["uid"], "boot_id": info["bootID"]}


def _owned_source_pod(
    pod: dict[str, Any],
    deployment: dict[str, Any],
    read: Callable[..., dict[str, Any]],
    namespace: str,
) -> None:
    owners = pod["metadata"].get("ownerReferences") or []
    require(
        len(owners) == 1
        and isinstance(owners[0], dict)
        and owners[0].get("kind") == "ReplicaSet"
        and owners[0].get("apiVersion") == "apps/v1"
        and owners[0].get("controller") is True,
        "source Pod has no unique ReplicaSet owner",
    )
    owner = owners[0]
    replica_name = name(owner.get("name"), limit=253)
    replica = read("replicaset", replica_name)
    meta = _metadata(replica, "ReplicaSet", replica_name, namespace)
    parents = meta.get("ownerReferences") or []
    require(
        meta["uid"] == owner.get("uid")
        and not meta.get("deletionTimestamp")
        and len(parents) == 1
        and isinstance(parents[0], dict)
        and parents[0].get("kind") == "Deployment"
        and parents[0].get("apiVersion") == "apps/v1"
        and parents[0].get("controller") is True
        and parents[0].get("name") == deployment["metadata"]["name"]
        and parents[0].get("uid") == deployment["metadata"]["uid"],
        "source Pod does not belong to the bound Deployment",
    )


def capture_scrape_source(
    *, kubeconfig: Path, namespace: str, region: str, workspace_id: str
) -> dict[str, Any]:
    """Read and bind the installed collector; never read a Kubernetes Secret."""
    namespace = name(namespace)
    connection = _file_identity(kubeconfig)

    def read(kind: str, resource: str) -> dict[str, Any]:
        response = run_fixture_command(
            [
                "kubectl",
                "--kubeconfig",
                connection["path"],
                "-n",
                namespace,
                "get",
                kind,
                resource,
                "-o",
                "json",
            ],
            timeout=30,
        )
        return creation_document(response.stdout)

    try:
        scope = _metadata(read("namespace", namespace), "Namespace", namespace, None)
        require(not scope.get("deletionTimestamp"), "ADOT namespace is terminating")
        deployment = read("deployment", ADOT)
        meta = _metadata(deployment, "Deployment", ADOT, namespace)
        spec, status = deployment["spec"], deployment["status"]
        require(
            not meta.get("deletionTimestamp")
            and type(meta.get("generation")) is int
            and type(spec.get("replicas")) is int
            and spec.get("replicas") == 1
            and all(
                type(status.get(key)) is int and status[key] == 1
                for key in (
                    "replicas",
                    "updatedReplicas",
                    "readyReplicas",
                    "availableReplicas",
                )
            )
            and status.get("observedGeneration") == meta["generation"],
            "ADOT must have one fully rolled out Ready replica",
        )
        template = spec["template"]["spec"]
        require(
            isinstance(template.get("containers"), list)
            and len(template["containers"]) == 1,
            "ADOT source container is ambiguous",
        )
        container = template["containers"][0]
        require(
            container.get("args") == ["--config=/etc/otel/collector.yaml"]
            and not container.get("command"),
            "ADOT source command is unsupported",
        )
        response = run_fixture_command(
            [
                "kubectl",
                "--kubeconfig",
                connection["path"],
                "-n",
                namespace,
                "get",
                "pods",
                "-l",
                f"app={ADOT}",
                "-o",
                "json",
            ],
            timeout=30,
        )
        inventory = creation_document(response.stdout)
        ready = ready_pod_records(inventory)
        require(
            len(ready) == len(inventory["items"]) == 1,
            "ADOT Ready Pod inventory is incomplete",
        )
        pod = inventory["items"][0]
        pod_name = name(ready[0]["name"], limit=253)
        pod_meta = _metadata(pod, "Pod", pod_name, namespace)
        require(
            pod["spec"]["containers"][0]["image"] == container["image"]
            and len(pod["spec"]["containers"]) == 1
            and pod["spec"]["containers"][0].get("args") == container["args"]
            and not pod["spec"]["containers"][0].get("command")
            and not pod["spec"].get("initContainers")
            and not pod["spec"].get("ephemeralContainers"),
            "ADOT deployed and running containers differ",
        )
        _owned_source_pod(pod, deployment, read, namespace)
        running = pod["status"]["containerStatuses"][0]
        image = _image(running.get("imageID"))
        require(
            isinstance(running.get("containerID"), str)
            and bool(running["containerID"])
            and isinstance((running.get("state") or {}).get("running"), dict)
            and type(running.get("restartCount")) is int
            and running["restartCount"] >= 0,
            "ADOT running container identity is incomplete",
        )
        service_account = name(template.get("serviceAccountName"))
        require(
            pod["spec"].get("serviceAccountName") == service_account,
            "ADOT running ServiceAccount differs",
        )
        account = read("serviceaccount", service_account)
        account_meta = _metadata(account, "ServiceAccount", service_account, namespace)
        require(
            not account_meta.get("deletionTimestamp")
            and not account.get("imagePullSecrets"),
            "ADOT ServiceAccount is terminating or requires unsupported pull credentials",
        )
        identity = workload_identity(pod, account, region=region)
        mounts = [
            item
            for item in container.get("volumeMounts", [])
            if item.get("mountPath") == "/etc/otel" and item.get("readOnly") is True
        ]
        require(len(mounts) == 1, "ADOT configuration mount is ambiguous")
        volumes = [
            item
            for item in template.get("volumes", [])
            if item.get("name") == mounts[0]["name"]
        ]
        require(
            len(volumes) == 1 and set(volumes[0]) == {"name", "configMap"},
            "ADOT configuration must come from a ConfigMap",
        )
        require(
            [
                item
                for item in pod["spec"]["containers"][0].get("volumeMounts", [])
                if item.get("mountPath") == "/etc/otel"
            ]
            == mounts
            and [
                item
                for item in pod["spec"].get("volumes", [])
                if item.get("name") == mounts[0]["name"]
            ]
            == volumes
            and not mounts[0].get("subPath")
            and not mounts[0].get("subPathExpr")
            and not volumes[0]["configMap"].get("items"),
            "ADOT running configuration projection differs from its Deployment",
        )
        config_name = name(volumes[0]["configMap"].get("name"))
        configmap = read("configmap", config_name)
        config_meta = _metadata(configmap, "ConfigMap", config_name, namespace)
        raw = configmap.get("data", {}).get("collector.yaml")
        safe_pipeline, labels = pipeline(raw, region=region, workspace_id=workspace_id)
        node_name = name(pod["spec"].get("nodeName"), limit=253)
        source_node = _node(read("node", node_name), node_name)
        require(
            _file_identity(kubeconfig) == connection,
            "CPU kubeconfig changed during ADOT capture",
        )
        return {
            "schema_version": 1,
            "connection": connection,
            "namespace": namespace,
            "namespace_uid": scope["uid"],
            "region": region,
            "workspace_id": workspace_id,
            "deployment_uid": meta["uid"],
            "deployment_generation": meta["generation"],
            "deployment_spec_sha256": digest(spec),
            "source_pod": {
                "name": pod_name,
                "uid": pod_meta["uid"],
                "spec_sha256": digest(pod["spec"]),
                "container_id": running["containerID"],
                "restart_count": running["restartCount"],
            },
            "image": image,
            "service_account": service_account,
            "service_account_uid": account_meta["uid"],
            "service_account_config_sha256": digest(
                {
                    key: value
                    for key, value in account.items()
                    if key not in {"metadata", "status"}
                }
                | {
                    "metadata": {
                        key: value
                        for key, value in account_meta.items()
                        if key
                        not in {"resourceVersion", "managedFields", "creationTimestamp"}
                    }
                }
            ),
            "configmap": {
                "name": config_name,
                "uid": config_meta["uid"],
                "sha256": hashlib.sha256(raw.encode()).hexdigest(),
            },
            "pipeline": safe_pipeline,
            "pipeline_yaml": yaml.safe_dump(safe_pipeline, sort_keys=True),
            "labels": labels,
            "identity": identity,
            "node": source_node,
        }
    except (KeyError, TypeError, ValueError, IndexError, RecursionError):
        raise ScrapeCompanionError("ADOT source configuration is incomplete") from None


def _canonical_pod(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    spec = result["spec"]
    for key in ("hostNetwork", "hostPID", "hostIPC"):
        spec.setdefault(key, False)
    spec.setdefault("schedulingGates", [])
    spec.setdefault("preemptionPolicy", "PreemptLowerPriority")
    for container in spec["containers"]:
        container.setdefault("terminationMessagePath", "/dev/termination-log")
        container.setdefault("terminationMessagePolicy", "File")
    return result


class ScrapeCompanion:
    """A single run's standalone scraper. Construction performs no mutations."""

    def __init__(
        self,
        harness: Any,
        probe: Any,
        *,
        expected_source: dict[str, Any],
        deadline: datetime,
    ) -> None:
        require(
            isinstance(deadline, datetime) and deadline.tzinfo is not None,
            "scrape deadline must include a timezone",
        )
        self.harness, self.probe = harness, probe
        self.namespace = name(harness.namespace)
        self.run_id = name(harness.run_id)
        self.resource_name = name(f"gpu-fault-{self.run_id}-scrape")
        self.probe_pod = name(probe.pod, limit=253)
        self.probe_service = name(probe.service)
        self.probe_deployment = name(probe.deployment)
        self.secret_name = name(harness.secret_name)
        require(
            self.secret_name == f"gpu-fault-{self.run_id}-registry",
            "scrape token Secret does not belong to this capacity run",
        )
        require(isinstance(expected_source, dict), "ADOT source binding is missing")
        self.source = copy.deepcopy(expected_source)
        self.deadline = deadline.astimezone(timezone.utc)
        self.path = Path(harness.run_dir) / "capacity-scrape-ownership.json"
        self.evidence_path = Path(harness.run_dir) / "capacity-scrape-lifecycle.json"
        self.custody: FixtureOwnership | None = None
        self.expected: dict[str, dict[str, Any]] = {}
        self.evidence: dict[str, Any] = {"schema_version": 1, "run_id": self.run_id}
        self.target: dict[str, Any] = {}
        self.cleanup_until: float | None = None
        self.started = False

    @property
    def selector(self) -> str:
        return f'pod="{self.probe_pod}",capacity_run="{self.run_id}"'

    def _save(self) -> None:
        write_json_atomic(self.evidence_path, self.evidence)

    def _call(self, *args: str, body: dict[str, Any] | None = None) -> str:
        timeout = 30.0
        if self.cleanup_until is not None:
            timeout = min(timeout, self.cleanup_until - time.monotonic())
        else:
            timeout = min(timeout, (self.deadline - _now()).total_seconds())
        require(timeout > 0, "scrape command budget has expired")
        return str(
            self.harness.kubectl(
                *args,
                input_text=json.dumps(body) if body is not None else None,
                timeout=timeout,
            ).stdout
        )

    def _read(self, kind: str, resource: str) -> dict[str, Any] | None:
        raw = self._call("get", kind, resource, "--ignore-not-found", "-o", "json")
        if not raw.strip():
            return None
        value = creation_document(raw)
        _metadata(
            value,
            kind,
            resource,
            None if kind in {"Node", "Namespace"} else self.namespace,
        )
        return value

    def _scope(self) -> None:
        require(
            _file_identity(Path(self.harness.cpu_kubeconfig))
            == self.source["connection"],
            "scrape CPU kubeconfig changed",
        )
        scope = self._read("Namespace", self.namespace)
        require(
            scope is not None
            and scope["metadata"]["uid"] == self.source["namespace_uid"],
            "scrape namespace identity changed",
        )

    def _target(self) -> dict[str, Any]:
        deployment = self._read("Deployment", self.probe_deployment)
        service = self._read("Service", self.probe_service)
        pod = self._read("Pod", self.probe_pod)
        require(
            deployment is not None and service is not None and pod is not None,
            "capacity probe target is incomplete",
        )
        assert deployment is not None and service is not None and pod is not None
        require(
            len(ready_pod_records({"items": [pod]})) == 1,
            "capacity probe target is not Ready",
        )
        selector = service.get("spec", {}).get("selector")
        require(
            selector == {"gpu-fault.io/capacity-probe": f"{self.run_id}-cap002"}
            and all(
                pod["metadata"].get("labels", {}).get(key) == value
                for key, value in selector.items()
            )
            and service["spec"].get("type", "ClusterIP") == "ClusterIP"
            and not service["spec"].get("externalIPs")
            and not service["spec"].get("externalName")
            and len(service["spec"].get("ports", [])) == 1
            and service["spec"]["ports"][0].get("port") == 18080
            and service["spec"]["ports"][0].get("targetPort") == "http",
            "capacity Service is not isolated to this probe",
        )
        _owned_source_pod(
            pod,
            deployment,
            lambda kind, resource: self._read(
                {"replicaset": "ReplicaSet"}[kind], resource
            )
            or {},
            self.namespace,
        )
        return {
            "deployment_uid": deployment["metadata"]["uid"],
            "service_uid": service["metadata"]["uid"],
            "service_spec_sha256": digest(service["spec"]),
            "pod_uid": pod["metadata"]["uid"],
            "pod_spec_sha256": digest(pod["spec"]),
        }

    def _manifests(self, lifetime: int) -> dict[str, dict[str, Any]]:
        config = build_scrape_config(
            source=self.source,
            service=self.probe_service,
            namespace=self.namespace,
            pod=self.probe_pod,
            run_id=self.run_id,
        )
        metadata = {
            "name": self.resource_name,
            "namespace": self.namespace,
            "labels": {OWNER: self.run_id, "app": "gpu-fault-capacity-scraper"},
        }
        identity = self.source["identity"]
        require(
            identity["volume"]["name"] not in {"config", "probe-token"},
            "ADOT identity volume collides with companion volumes",
        )
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": metadata,
            "immutable": True,
            "data": {"collector.yaml": yaml.safe_dump(config, sort_keys=True)},
        }
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": copy.deepcopy(metadata),
            "spec": {
                "serviceAccountName": self.source["service_account"],
                "automountServiceAccountToken": False,
                "enableServiceLinks": False,
                "restartPolicy": "Never",
                "activeDeadlineSeconds": lifetime,
                "terminationGracePeriodSeconds": 30,
                "dnsPolicy": "ClusterFirst",
                "schedulerName": "default-scheduler",
                "schedulingGates": [{"name": GATE}],
                "hostNetwork": False,
                "hostPID": False,
                "hostIPC": False,
                "affinity": {
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchFields": [
                                        {
                                            "key": "metadata.name",
                                            "operator": "In",
                                            "values": [self.source["node"]["name"]],
                                        }
                                    ]
                                }
                            ],
                        },
                    }
                },
                "securityContext": {
                    "runAsUser": 65534,
                    "runAsGroup": 65534,
                    "runAsNonRoot": True,
                    "fsGroup": 65534,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "containers": [
                    {
                        "name": "collector",
                        "image": self.source["image"],
                        "imagePullPolicy": "IfNotPresent",
                        "args": ["--config=/etc/otel/collector.yaml"],
                        "env": copy.deepcopy(identity["env"]),
                        "ports": [
                            {"name": "health", "containerPort": 8090, "protocol": "TCP"}
                        ],
                        "readinessProbe": {
                            "httpGet": {
                                "path": "/health",
                                "port": "health",
                                "scheme": "HTTP",
                            },
                            "initialDelaySeconds": 2,
                            "periodSeconds": 2,
                            "timeoutSeconds": 2,
                            "successThreshold": 1,
                            "failureThreshold": 3,
                        },
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "128Mi"},
                            "limits": {"cpu": "500m", "memory": "256Mi"},
                        },
                        "securityContext": {
                            "allowPrivilegeEscalation": False,
                            "privileged": False,
                            "readOnlyRootFilesystem": True,
                            "capabilities": {"drop": ["ALL"]},
                        },
                        "volumeMounts": [
                            {
                                "name": "config",
                                "mountPath": "/etc/otel",
                                "readOnly": True,
                            },
                            {
                                "name": "probe-token",
                                "mountPath": "/etc/capacity-token",
                                "readOnly": True,
                            },
                            copy.deepcopy(identity["mount"]),
                        ],
                    }
                ],
                "volumes": [
                    {
                        "name": "config",
                        "configMap": {
                            "name": self.resource_name,
                            "defaultMode": 292,
                        },
                    },
                    {
                        "name": "probe-token",
                        "secret": {
                            "secretName": self.secret_name,
                            "defaultMode": 288,
                            "items": [
                                {"key": "execution-token", "path": "execution-token"}
                            ],
                        },
                    },
                    copy.deepcopy(identity["volume"]),
                ],
            },
        }
        return {"ConfigMap": configmap, "Pod": _canonical_pod(pod)}

    def _verify(self, kind: str, value: dict[str, Any]) -> dict[str, Any]:
        require(self.custody is not None, "scrape creation custody is unavailable")
        assert self.custody is not None
        current = _canonical_pod(value) if kind == "Pod" else value
        meta = _metadata(current, kind, self.resource_name, self.namespace)
        creation = self.custody.record.creations.get(
            f"{kind.lower()}/{self.resource_name}"
        )
        require(
            creation is not None and creation.approved and creation.uid == meta["uid"],
            "scrape resource has no matching creation acknowledgement",
        )
        expected = self.expected[kind]
        labels = copy.deepcopy(current["metadata"].get("labels"))
        if (
            kind == "Pod"
            and self.source["identity"]["mode"] == "pod-identity"
            and isinstance(labels, dict)
            and labels.get("eks.amazonaws.com/pod-identity") == "enabled"
        ):
            # EKS adds this marker after applying the already-bound IAM projection.
            labels.pop("eks.amazonaws.com/pod-identity")
        require(
            labels == expected["metadata"]["labels"]
            and not current["metadata"].get("ownerReferences")
            and current["metadata"].get("annotations", {})
            == expected["metadata"].get("annotations", {}),
            "scrape resource ownership metadata changed",
        )
        if kind == "ConfigMap":
            require(
                current.get("data") == expected["data"]
                and current.get("immutable") is True
                and not current.get("binaryData"),
                "scrape ConfigMap content changed",
            )
        else:
            spec = copy.deepcopy(current["spec"])
            gates = spec.get("schedulingGates")
            require(
                gates == [{"name": GATE}]
                or (gates == [] and self.evidence.get("activation_started") is True),
                "scrape scheduling authorization changed",
            )
            spec["schedulingGates"] = [{"name": GATE}]
            require(
                spec.pop("nodeName", self.source["node"]["name"])
                == self.source["node"]["name"]
                and spec.pop("serviceAccount", self.source["service_account"])
                == self.source["service_account"]
                and spec.pop("priority", 0) == 0
                and not spec.pop("imagePullSecrets", [])
                and not spec.pop("initContainers", [])
                and not spec.pop("ephemeralContainers", []),
                "scrape Pod acquired unapproved authority",
            )
            tolerations = spec.pop("tolerations", [])
            require(
                isinstance(tolerations, list)
                and all(
                    item
                    in [
                        {
                            "key": key,
                            "operator": "Exists",
                            "effect": "NoExecute",
                            "tolerationSeconds": 300,
                        }
                        for key in (
                            "node.kubernetes.io/not-ready",
                            "node.kubernetes.io/unreachable",
                        )
                    ]
                    for item in tolerations
                ),
                "scrape Pod acquired unapproved tolerations",
            )
            require(
                spec == expected["spec"], "scrape admitted Pod specification changed"
            )
        return current

    def _create(self, kind: str) -> None:
        assert self.custody is not None
        expected = self.expected[kind]
        self.custody.intend(expected)
        raw = self._call("create", "-f", "-", "-o", "json", body=expected)
        ack = creation_document(raw)
        if kind == "Pod":
            ack = _canonical_pod(ack)
        self.custody.acknowledge(ack)
        self._verify(kind, ack)

    def start(self) -> dict[str, Any]:
        require(
            not self.started and self.custody is None, "scrape start cannot be repeated"
        )
        require(
            not self.path.exists() and not self.path.is_symlink(),
            "scrape journal already exists",
        )
        actual = capture_scrape_source(
            kubeconfig=Path(self.harness.cpu_kubeconfig),
            namespace=self.namespace,
            region=self.harness.region,
            workspace_id=self.harness.workspace_id,
        )
        require(actual == self.source, "ADOT scrape source drifted before start")
        self._scope()
        self.target = self._target()
        lifetime = min(
            MAX_LIFETIME_SECONDS,
            math.floor((self.deadline - _now()).total_seconds()) - CLEANUP_SECONDS,
        )
        require(
            lifetime >= STARTUP_SECONDS,
            "scrape window lacks startup and cleanup budget",
        )
        for kind in ("ConfigMap", "Pod"):
            require(
                self._read(kind, self.resource_name) is None,
                "scrape resource name already exists",
            )
        self.expected = self._manifests(lifetime)
        self.custody = FixtureOwnership(
            self.path,
            {
                "namespace": self.namespace,
                "namespace_uid": self.source["namespace_uid"],
                "run_id": self.run_id,
                "source_sha256": digest(self.source),
                "connection": self.source["connection"],
                "target": self.target,
                "deadline": self.deadline.isoformat(),
            },
        )
        self.custody.begin()
        try:
            self._create("ConfigMap")
            self._create("Pod")
            pod = self._read("Pod", self.resource_name)
            require(pod is not None, "scrape Pod disappeared before activation")
            assert pod is not None
            self._verify("Pod", pod)
            require(
                not pod["spec"].get("nodeName"),
                "scrape Pod scheduled before verification",
            )
            require(
                self._target() == self.target,
                "capacity target changed before scrape activation",
            )
            self.evidence["activation_started"] = True
            self._save()
            patch = [
                {
                    "op": "test",
                    "path": "/metadata/uid",
                    "value": pod["metadata"]["uid"],
                },
                {
                    "op": "test",
                    "path": "/metadata/resourceVersion",
                    "value": pod["metadata"]["resourceVersion"],
                },
                {
                    "op": "test",
                    "path": "/spec/schedulingGates",
                    "value": [{"name": GATE}],
                },
                {"op": "remove", "path": "/spec/schedulingGates"},
            ]
            self.harness.kubectl(
                "patch",
                "Pod",
                self.resource_name,
                "--type=json",
                "--patch-file=/dev/stdin",
                input_text=json.dumps(patch),
                timeout=30,
            )
            until = min(time.monotonic() + STARTUP_SECONDS, time.monotonic() + lifetime)
            while time.monotonic() < until:
                pod = self._read("Pod", self.resource_name)
                require(pod is not None, "scrape Pod disappeared during startup")
                assert pod is not None
                self._verify("Pod", pod)
                if ready_pod_records({"items": [pod]}):
                    running = pod["status"]["containerStatuses"][0]
                    require(
                        _image(running.get("imageID")) == self.source["image"]
                        and not pod["spec"].get("schedulingGates")
                        and pod["spec"].get("nodeName") == self.source["node"]["name"]
                        and running.get("restartCount") == 0
                        and isinstance(running.get("containerID"), str)
                        and bool(running["containerID"]),
                        "scrape running container identity differs",
                    )
                    self.evidence.update(
                        {
                            "ready": True,
                            "pod_uid": pod["metadata"]["uid"],
                            "container_id": running["containerID"],
                            "image": self.source["image"],
                            "selector": self.selector,
                            "active_deadline_seconds": lifetime,
                            "target": self.target,
                        }
                    )
                    self._save()
                    self.started = True
                    return copy.deepcopy(self.evidence)
                require(
                    pod.get("status", {}).get("phase") not in {"Failed", "Succeeded"},
                    "scrape Pod exited before readiness",
                )
                time.sleep(POLL_SECONDS)
            raise ScrapeCompanionError("scrape Pod readiness timed out")
        except BaseException as exc:
            self.evidence["start_error_type"] = type(exc).__name__
            self._save()
            if isinstance(exc, Exception) and not isinstance(
                exc, (ScrapeCompanionError, ProcessSupervisionLost)
            ):
                raise ScrapeCompanionError(
                    "scrape startup failed; creation custody was retained"
                ) from None
            raise

    def _current_node(self) -> dict[str, str]:
        node_name = self.source["node"]["name"]
        value = self._read("Node", node_name)
        require(value is not None, "scrape CPU node disappeared")
        assert value is not None
        current = _node(value, node_name)
        require(current == self.source["node"], "scrape CPU node identity changed")
        return current

    def _termination(
        self, pod: dict[str, Any] | None, *, kubelet: bool = True
    ) -> dict[str, Any] | None:
        assert self.custody is not None
        uid = self.custody.record.creations[f"pod/{self.resource_name}"].uid
        if pod is not None:
            states = pod.get("status", {}).get("containerStatuses")
            if isinstance(states, list) and len(states) == 1:
                state = states[0]
                terminated = state.get("state", {}).get("terminated")
                container_id = state.get("containerID")
                finished: datetime | None = None
                if isinstance(terminated, dict):
                    try:
                        finished = datetime.fromisoformat(
                            str(terminated.get("finishedAt", "")).replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass
                if (
                    state.get("name") == "collector"
                    and isinstance(container_id, str)
                    and bool(container_id)
                    and container_id == self.evidence.get("container_id", container_id)
                    and isinstance(terminated, dict)
                    and type(terminated.get("exitCode")) is int
                    and 0 <= terminated["exitCode"] <= 255
                    and finished is not None
                    and finished.tzinfo is not None
                    and finished <= _now()
                ):
                    return {
                        "source": "container-status",
                        "pod_uid": uid,
                        "container_id": container_id,
                        "finished_at": finished.isoformat(),
                    }
        if not kubelet:
            return None
        before = self._current_node()
        raw = self._call(
            "get",
            "--raw",
            f"/api/v1/nodes/{before['name']}/proxy/runningpods/",
        )
        value = creation_document(raw)
        require(
            value.get("kind") == "PodList"
            and value.get("apiVersion") == "v1"
            and isinstance(value.get("items"), list),
            "CPU kubelet running Pod inventory is invalid",
        )
        identities: set[str] = set()
        present = False
        for item in value["items"]:
            require(
                isinstance(item, dict) and isinstance(item.get("metadata"), dict),
                "CPU kubelet Pod identity is incomplete",
            )
            meta = item["metadata"]
            require(
                all(
                    isinstance(meta.get(key), str) and meta[key]
                    for key in ("uid", "name", "namespace")
                )
                and meta["uid"] not in identities,
                "CPU kubelet Pod identity is incomplete or duplicated",
            )
            identities.add(meta["uid"])
            if (
                meta["name"] == self.resource_name
                and meta["namespace"] == self.namespace
            ):
                require(
                    meta["uid"] == uid, "scrape Pod name was reused on the CPU node"
                )
            present = present or meta["uid"] == uid
        require(
            self._current_node() == before,
            "CPU node changed during termination observation",
        )
        if present:
            return None
        return {"source": "kubelet-runningpods", "node": before, "pod_uid": uid}

    def _delete(self, kind: str, current: dict[str, Any]) -> None:
        self.evidence[f"{kind.lower()}_delete_started"] = True
        self._save()
        meta = current["metadata"]
        plural = {"Pod": "pods", "ConfigMap": "configmaps"}[kind]
        try:
            self._call(
                "delete",
                "--raw",
                f"/api/v1/namespaces/{self.namespace}/{plural}/{self.resource_name}",
                "-f",
                "-",
                body={
                    "apiVersion": "v1",
                    "kind": "DeleteOptions",
                    "preconditions": {
                        "uid": meta["uid"],
                        "resourceVersion": meta["resourceVersion"],
                    },
                    "propagationPolicy": "Foreground",
                },
            )
        except ProcessSupervisionLost:
            raise
        except Exception:
            self.evidence[f"{kind.lower()}_delete_ack_lost"] = True
            self._save()

    def stop(self) -> dict[str, Any]:
        if self.custody is None:
            require(not self.path.exists(), "existing scrape custody requires recovery")
            return {
                "cleanup_complete": True,
                "process_termination_proven": True,
                "not_created": True,
            }
        if self.cleanup_until is None:
            self.cleanup_until = time.monotonic() + CLEANUP_SECONDS
        try:
            self._scope()
            self.custody.require_acknowledged()
            proof: dict[str, Any] | None = None
            pod_key = f"pod/{self.resource_name}"
            if pod_key in self.custody.record.creations:
                pod = self._read("Pod", self.resource_name)
                if pod is not None:
                    self._verify("Pod", pod)
                    proof = self._termination(pod, kubelet=False)
                    if not pod["metadata"].get("deletionTimestamp"):
                        self._delete("Pod", pod)
                while time.monotonic() < self.cleanup_until:
                    pod = self._read("Pod", self.resource_name)
                    if pod is not None:
                        self._verify("Pod", pod)
                    proof = proof or self._termination(pod, kubelet=pod is None)
                    if pod is None and proof is not None:
                        break
                    time.sleep(POLL_SECONDS)
                require(
                    pod is None and proof is not None,
                    "scrape Pod termination or absence is unproven",
                )
            else:
                proof = {"source": "pod-not-created"}
            self.evidence["termination_proof"] = proof
            self._save()
            if f"configmap/{self.resource_name}" in self.custody.record.creations:
                config = self._read("ConfigMap", self.resource_name)
                if config is not None:
                    self._verify("ConfigMap", config)
                    self._delete("ConfigMap", config)
                while (
                    remaining := self._read("ConfigMap", self.resource_name)
                ) is not None:
                    self._verify("ConfigMap", remaining)
                    require(
                        time.monotonic() < self.cleanup_until,
                        "scrape ConfigMap removal was not confirmed",
                    )
                    time.sleep(POLL_SECONDS)
            self.custody.complete()
            result = {
                "cleanup_complete": True,
                "process_termination_proven": True,
                "pod_absent": True,
                "configmap_absent": True,
                "termination_proof": proof,
            }
            self.evidence["cleanup"] = result
            self._save()
            return result
        except BaseException as exc:
            self.evidence["cleanup"] = {
                "cleanup_complete": False,
                "process_termination_proven": self.evidence.get("termination_proof")
                is not None,
                "error_type": type(exc).__name__,
            }
            self._save()
            if isinstance(exc, Exception) and not isinstance(
                exc, (ScrapeCompanionError, ProcessSupervisionLost)
            ):
                raise ScrapeCompanionError(
                    "scrape cleanup failed; dependent resources must be retained"
                ) from None
            raise
