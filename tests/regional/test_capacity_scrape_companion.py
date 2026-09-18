from __future__ import annotations

import copy
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import capacity_scrape_companion as module
from scripts.e2e.regional.fixture_ownership import FixtureOwnership

REGION = "us-west-2"
NAMESPACE = "gpu-fault-system"
WORKSPACE = "ws-unit"
RUN = "cap-unit"
IMAGE = "registry.example/adot@sha256:" + "1" * 64
NODE = "cpu.example.internal"
PROBE = f"gpu-fault-{RUN}-cap002"
COMPANION = f"gpu-fault-{RUN}-scrape"
SYNTHETIC = "synthetic-value-must-never-appear-in-evidence"


def document(kind: str, resource: str, **values: Any) -> dict[str, Any]:
    return {
        "apiVersion": "apps/v1" if kind in {"Deployment", "ReplicaSet"} else "v1",
        "kind": kind,
        "metadata": {
            "name": resource,
            "uid": resource + "-uid",
            "resourceVersion": "1",
            **({"namespace": NAMESPACE} if kind not in {"Node", "Namespace"} else {}),
        },
        **values,
    }


def owner(kind: str, resource: str) -> dict[str, Any]:
    return {
        "apiVersion": "apps/v1",
        "kind": kind,
        "name": resource,
        "uid": resource + "-uid",
        "controller": True,
    }


def config() -> dict[str, Any]:
    return {
        "receivers": {
            "prometheus": {
                "config": {
                    "scrape_configs": [
                        {
                            "job_name": "gpu-fault-control-plane",
                            "authorization": {
                                "type": "Bearer",
                                "credentials_file": "/etc/gpu-fault/execution-token",
                            },
                            "kubernetes_sd_configs": [{"role": "pod"}],
                            "relabel_configs": [
                                {
                                    "action": "replace",
                                    "target_label": key,
                                    "replacement": value,
                                }
                                for key, value in {
                                    "region": REGION,
                                    "control_plane_cluster": "cpu-unit",
                                }.items()
                            ],
                        }
                    ]
                }
            }
        },
        "processors": {
            "batch": {
                "timeout": "5s",
                "send_batch_size": 1000,
                "send_batch_max_size": 2000,
            }
        },
        "exporters": {
            "prometheusremotewrite": {
                "endpoint": (
                    f"https://aps-workspaces.{REGION}.amazonaws.com/workspaces/"
                    f"{WORKSPACE}/api/v1/remote_write"
                ),
                "add_metric_suffixes": False,
                "auth": {"authenticator": "sigv4auth"},
                "retry_on_failure": {
                    "enabled": True,
                    "initial_interval": "1s",
                    "max_interval": "10s",
                    "max_elapsed_time": "120s",
                },
            }
        },
        "extensions": {
            "sigv4auth": {"region": REGION, "service": "aps"},
            "health_check": {"endpoint": "0.0.0.0:8090", "path": "/health"},
        },
        "service": {
            "extensions": ["sigv4auth", "health_check"],
            "pipelines": {
                "metrics": {
                    "receivers": ["prometheus"],
                    "processors": ["batch"],
                    "exporters": ["prometheusremotewrite"],
                }
            },
            "telemetry": {
                "logs": {"level": "info"},
                "metrics": {
                    "level": "detailed",
                    "readers": [
                        {
                            "pull": {
                                "exporter": {
                                    "prometheus": {"host": "127.0.0.1", "port": 8889}
                                }
                            }
                        }
                    ],
                },
            },
        },
    }


def ready_status(image: str = IMAGE) -> dict[str, Any]:
    return {
        "phase": "Running",
        "conditions": [{"type": "Ready", "status": "True"}],
        "containerStatuses": [
            {
                "name": "collector",
                "ready": True,
                "restartCount": 0,
                "containerID": "containerd://unit-container",
                "imageID": "docker-pullable://" + image,
                "state": {"running": {"startedAt": "2026-09-17T00:00:00Z"}},
            }
        ],
    }


class Clock:
    def __init__(self) -> None:
        self.elapsed = 0.0

    def monotonic(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += max(0.1, seconds)

    def now(self) -> datetime:
        return datetime(2026, 9, 17, tzinfo=timezone.utc) + timedelta(
            seconds=self.elapsed
        )


class Cluster:
    def __init__(self, tmp_path: Path, *, irsa: bool = False) -> None:
        self.irsa = irsa
        self.resources: dict[tuple[str, str], dict[str, Any]] = {}
        self.calls: list[tuple[str, ...]] = []
        self.creates: list[dict[str, Any]] = []
        self.deletes: list[tuple[str, dict[str, Any]]] = []
        self.lost_ack = ""
        self.admission = ""
        self.runtime_present = False
        self.runtime_stuck = False
        self.kubelet_error = False
        self.bad_kubelet = False
        self.node_drift = False
        self.delete_ack_lost = False
        self.config_delete_stuck = False
        self.terminated_on_delete = False
        self.terminal_reads = 0
        self.clock = Clock()
        self.kubeconfig = tmp_path / "cpu.kubeconfig"
        self.kubeconfig.write_text("synthetic kubeconfig, never used for network\n")
        self.kubeconfig.chmod(0o600)
        self.harness = SimpleNamespace(
            kubectl=self.kubectl,
            kubectl_json=self.kubectl_json,
            cpu_kubeconfig=str(self.kubeconfig),
            namespace=NAMESPACE,
            region=REGION,
            workspace_id=WORKSPACE,
            run_dir=tmp_path,
            run_id=RUN,
            secret_name=f"gpu-fault-{RUN}-registry",
        )
        self.probe = SimpleNamespace(
            deployment=PROBE, service=PROBE, pod=PROBE + "-pod"
        )
        self.put(document("Namespace", NAMESPACE))
        self.put(
            document(
                "Node",
                NODE,
                status={
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "allocatable": {"cpu": "8", "nvidia.com/gpu": "0"},
                    "nodeInfo": {"operatingSystem": "linux", "bootID": "unit-boot"},
                },
            )
        )
        account = document("ServiceAccount", module.ADOT)
        container = {
            "name": "collector",
            "image": "registry.example/adot:source-tag",
            "args": ["--config=/etc/otel/collector.yaml"],
            "env": [
                {"name": "GOGC", "value": "50"},
                {"name": "GOMEMLIMIT", "value": "220MiB"},
            ],
            "volumeMounts": [
                {"name": "config", "mountPath": "/etc/otel", "readOnly": True},
                {
                    "name": "execution-token",
                    "mountPath": "/etc/gpu-fault",
                    "readOnly": True,
                },
            ],
        }
        pod_spec = {
            "serviceAccountName": module.ADOT,
            "containers": [container],
            "volumes": [
                {"name": "config", "configMap": {"name": module.ADOT}},
                {
                    "name": "execution-token",
                    "secret": {
                        "secretName": "gpu-fault-control-plane-active",
                        "items": [
                            {"key": "execution-token", "path": "execution-token"}
                        ],
                    },
                },
            ],
        }
        self.add_deployment(module.ADOT, pod_spec, {"app": module.ADOT})
        running = self.get("Pod", module.ADOT + "-pod")["spec"]
        if irsa:
            role = "arn:aws:iam::000000000000:role/unit-adot-writer"
            account["metadata"]["annotations"] = {"eks.amazonaws.com/role-arn": role}
            directory = "/var/run/secrets/eks.amazonaws.com/serviceaccount"
            path, audience = "token", "sts.amazonaws.com"
            identity_env = {
                "AWS_ROLE_ARN": role,
                "AWS_WEB_IDENTITY_TOKEN_FILE": directory + "/" + path,
            }
        else:
            directory = "/var/run/secrets/pods.eks.amazonaws.com/serviceaccount"
            path, audience = "eks-pod-identity-token", "pods.eks.amazonaws.com"
            identity_env = {
                "AWS_CONTAINER_CREDENTIALS_FULL_URI": "http://169.254.170.23/v1/credentials",
                "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE": directory + "/" + path,
            }
        running["containers"][0]["env"].extend(
            {"name": key, "value": value}
            for key, value in {
                **identity_env,
                "AWS_REGION": REGION,
                "AWS_DEFAULT_REGION": REGION,
            }.items()
        )
        running["containers"][0]["volumeMounts"].append(
            {"name": "identity", "mountPath": directory, "readOnly": True}
        )
        running["volumes"].append(
            {
                "name": "identity",
                "projected": {
                    "defaultMode": 420,
                    "sources": [
                        {
                            "serviceAccountToken": {
                                "audience": audience,
                                "expirationSeconds": 86400,
                                "path": path,
                            }
                        }
                    ],
                },
            }
        )
        self.put(account)
        self.put(
            document(
                "ConfigMap",
                module.ADOT,
                data={"collector.yaml": yaml.safe_dump(config())},
            )
        )
        self.add_deployment(
            PROBE,
            {"containers": [{"name": "collector", "image": IMAGE}]},
            {"gpu-fault.io/capacity-probe": f"{RUN}-cap002"},
        )
        self.put(
            document(
                "Service",
                PROBE,
                spec={
                    "type": "ClusterIP",
                    "selector": {"gpu-fault.io/capacity-probe": f"{RUN}-cap002"},
                    "ports": [{"port": 18080, "targetPort": "http"}],
                },
            )
        )

    def add_deployment(
        self, resource: str, spec: dict[str, Any], labels: dict[str, str]
    ) -> None:
        deployment = document(
            "Deployment",
            resource,
            spec={
                "replicas": 1,
                "selector": {"matchLabels": labels},
                "template": {
                    "metadata": {"labels": labels},
                    "spec": copy.deepcopy(spec),
                },
            },
            status={
                "replicas": 1,
                "readyReplicas": 1,
                "availableReplicas": 1,
                "updatedReplicas": 1,
                "observedGeneration": 1,
            },
        )
        deployment["metadata"]["generation"] = 1
        replica = document("ReplicaSet", resource + "-rs")
        replica["metadata"]["ownerReferences"] = [owner("Deployment", resource)]
        pod = document(
            "Pod", resource + "-pod", spec=copy.deepcopy(spec), status=ready_status()
        )
        pod["metadata"].update(
            labels=labels, ownerReferences=[owner("ReplicaSet", resource + "-rs")]
        )
        pod["spec"]["nodeName"] = NODE
        for value in (deployment, replica, pod):
            self.put(value)

    def put(self, value: dict[str, Any]) -> None:
        self.resources[(value["kind"].lower(), value["metadata"]["name"])] = value

    def get(self, kind: str, resource: str) -> dict[str, Any]:
        return self.resources[(kind.lower(), resource)]

    def capture(self) -> dict[str, Any]:
        return module.capture_scrape_source(
            kubeconfig=self.kubeconfig,
            namespace=NAMESPACE,
            region=REGION,
            workspace_id=WORKSPACE,
        )

    def companion(self) -> module.ScrapeCompanion:
        return module.ScrapeCompanion(
            self.harness,
            self.probe,
            expected_source=self.capture(),
            deadline=self.clock.now() + timedelta(seconds=1200),
        )

    def run(
        self, args: list[str], *, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        assert args[:2] == ["kubectl", "--kubeconfig"], (
            "capture must use its explicit CPU connection"
        )
        assert args[2] == str(self.kubeconfig), "capture must not inherit a GPU context"
        assert args[5] == "get", "source capture must be read-only"
        return self.kubectl(*args[5:], timeout=timeout)

    def kubectl_json(self, *args: str) -> Any:
        return json.loads(self.kubectl(*args).stdout)

    def kubectl(
        self, *args: str, input_text: str | None = None, timeout: float = 30
    ) -> subprocess.CompletedProcess[str]:
        assert timeout > 0, "all operations need a bounded timeout"
        self.calls.append(args)
        operation = args[0]
        body: Any = json.loads(input_text) if input_text else None
        result: Any = None
        if operation == "get":
            if args[1] == "--raw":
                assert args[2] == f"/api/v1/nodes/{NODE}/proxy/runningpods/", (
                    "only the bound CPU kubelet is queried"
                )
                if self.kubelet_error:
                    raise RuntimeError(SYNTHETIC)
                if self.node_drift:
                    self.get("Node", NODE)["metadata"]["uid"] = "recreated-node"
                result = (
                    {}
                    if self.bad_kubelet
                    else {
                        "apiVersion": "v1",
                        "kind": "PodList",
                        "items": [document("Pod", COMPANION)]
                        if self.runtime_present
                        else [],
                    }
                )
            elif args[1] == "pods" and args[2] == "-l":
                result = {
                    "items": [copy.deepcopy(self.get("Pod", module.ADOT + "-pod"))]
                }
            else:
                assert args[1].lower() != "secret", (
                    "neither capture nor companion may read Secret contents"
                )
                key = (args[1].lower(), args[2])
                result = copy.deepcopy(self.resources.get(key))
                if key == ("pod", COMPANION) and self.terminal_reads:
                    self.terminal_reads -= 1
                    if not self.terminal_reads:
                        self.resources.pop(key, None)
                if result is None:
                    return subprocess.CompletedProcess(args, 0, "", "")
        elif operation == "create":
            result = copy.deepcopy(body)
            self.creates.append(copy.deepcopy(body))
            meta = result["metadata"]
            key = (result["kind"].lower(), meta["name"])
            assert key not in self.resources, (
                "creation may never adopt existing resources"
            )
            meta.update(uid=meta["name"] + "-uid", resourceVersion="1")
            if result["kind"] == "Pod":
                if not self.irsa:
                    meta["labels"]["eks.amazonaws.com/pod-identity"] = "enabled"
                spec = result["spec"]
                assert (
                    spec.get("preemptionPolicy", "PreemptLowerPriority")
                    == "PreemptLowerPriority"
                ), "default PriorityClass admission rejects Never"
                spec["priority"] = 0
                spec["serviceAccount"] = spec["serviceAccountName"]
                spec["tolerations"] = [
                    {
                        "key": "node.kubernetes.io/not-ready",
                        "effect": "NoExecute",
                        "operator": "Exists",
                        "tolerationSeconds": 300,
                    }
                ]
                result["status"] = {"phase": "Pending"}
                if self.admission == "privileged":
                    spec["containers"][0]["securityContext"]["privileged"] = True
                elif self.admission == "env":
                    spec["containers"][0]["env"].append(
                        {"name": "AWS_SECRET_ACCESS_KEY", "value": SYNTHETIC}
                    )
                elif self.admission == "host":
                    spec["hostNetwork"] = True
                elif self.admission == "account":
                    spec["serviceAccountName"] = "foreign-account"
                elif self.admission == "sidecar":
                    spec["containers"].append({"name": "foreign", "image": IMAGE})
                elif self.admission == "gates":
                    spec["schedulingGates"] = []
                elif self.admission == "pod-identity-label-value":
                    meta["labels"]["eks.amazonaws.com/pod-identity"] = "disabled"
                elif self.admission == "extra-label":
                    meta["labels"]["unapproved-owner"] = "another-run"
                elif self.admission == "without-pod-identity-label":
                    meta["labels"].pop("eks.amazonaws.com/pod-identity", None)
                elif self.admission == "unexpected-pod-identity":
                    meta["labels"]["eks.amazonaws.com/pod-identity"] = "enabled"
            self.put(result)
            if self.lost_ack == result["kind"]:
                raise RuntimeError(SYNTHETIC)
        elif operation == "patch":
            assert args[1:3] == ("Pod", COMPANION), (
                "activation can only patch its own standalone Pod"
            )
            result = self.get("Pod", COMPANION)
            assert body[:2] == [
                {
                    "op": "test",
                    "path": "/metadata/uid",
                    "value": result["metadata"]["uid"],
                },
                {
                    "op": "test",
                    "path": "/metadata/resourceVersion",
                    "value": result["metadata"]["resourceVersion"],
                },
            ], "activation must be UID/version conditional"
            result["spec"].pop("schedulingGates")
            result["spec"]["nodeName"] = NODE
            result["metadata"]["resourceVersion"] = "2"
            result["status"] = ready_status()
            self.runtime_present = True
        elif operation == "delete":
            assert args[1] == "--raw", (
                "cleanup must use explicit DeleteOptions preconditions"
            )
            plural = args[2].split("/")[-2]
            kind = {"pods": "Pod", "configmaps": "ConfigMap"}[plural]
            current = self.get(kind, COMPANION)
            assert body["preconditions"] == {
                "uid": current["metadata"]["uid"],
                "resourceVersion": current["metadata"]["resourceVersion"],
            }, "deletion must bind acknowledged UID and current version"
            self.deletes.append((kind, body))
            if kind == "ConfigMap" and self.config_delete_stuck:
                return subprocess.CompletedProcess(args, 0, "{}", "")
            if kind == "Pod":
                self.runtime_present = self.runtime_stuck
                if self.terminated_on_delete:
                    current["metadata"]["deletionTimestamp"] = (
                        self.clock.now().isoformat()
                    )
                    current["status"]["containerStatuses"][0]["state"] = {
                        "terminated": {
                            "exitCode": 0,
                            "finishedAt": self.clock.now().isoformat(),
                        }
                    }
                    self.terminal_reads = 1
                else:
                    self.resources.pop((kind.lower(), COMPANION))
            else:
                self.resources.pop((kind.lower(), COMPANION))
            result = {"kind": "Status", "status": "Success"}
            if self.delete_ack_lost:
                raise RuntimeError(SYNTHETIC)
        else:
            raise AssertionError("unexpected mutation outside the companion API")
        return subprocess.CompletedProcess(args, 0, json.dumps(result), "")


@pytest.fixture
def cluster(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Cluster:
    value = Cluster(tmp_path)
    monkeypatch.setattr(module, "run_fixture_command", value.run)
    monkeypatch.setattr(module, "time", value.clock)
    monkeypatch.setattr(module, "_now", value.clock.now)
    return value


@pytest.mark.parametrize("irsa", [False, True])
def test_source_capture_binds_actual_ready_owner_image_and_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, irsa: bool
) -> None:
    cluster = Cluster(tmp_path, irsa=irsa)
    monkeypatch.setattr(module, "run_fixture_command", cluster.run)
    result = cluster.capture()
    assert result["image"] == IMAGE, (
        "the running digest, not a template tag, must be pinned"
    )
    assert result["deployment_uid"] == module.ADOT + "-uid", (
        "the Ready Pod must lead to this Deployment"
    )
    assert result["service_account_uid"] == module.ADOT + "-uid", (
        "the actual ServiceAccount UID is bound"
    )
    assert result["identity"]["mode"] == ("irsa" if irsa else "pod-identity"), (
        "only the installed identity mode is reused"
    )
    assert result["node"]["uid"] == NODE + "-uid", "termination uses a bound CPU node"
    assert all(
        call[0] == "get" and call[1].lower() != "secret" for call in cluster.calls
    ), "capture is read-only and never reads Secrets"
    assert "gpu-fault-control-plane-active" not in json.dumps(result), (
        "production token configuration must not be copied"
    )


def test_constructor_and_builder_are_pure_and_export_only_bound_families(
    cluster: Cluster,
) -> None:
    source = cluster.capture()
    before = copy.deepcopy(source)
    calls = list(cluster.calls)
    value = module.ScrapeCompanion(
        cluster.harness,
        cluster.probe,
        expected_source=source,
        deadline=cluster.clock.now() + timedelta(seconds=1200),
    )
    result = module.build_scrape_config(
        source=source,
        service=PROBE,
        namespace=NAMESPACE,
        pod=cluster.probe.pod,
        run_id=RUN,
    )
    assert cluster.calls == calls and source == before, (
        "construction/rendering must not perform I/O or mutate source inputs"
    )
    assert value.selector == f'pod="{cluster.probe.pod}",capacity_run="{RUN}"', (
        "AMP matching must name only this probe and run"
    )
    job = result["receivers"]["prometheus"]["config"]["scrape_configs"][0]
    assert job["static_configs"] == [
        {
            "targets": [f"{PROBE}.{NAMESPACE}.svc:18080"],
            "labels": {
                "region": REGION,
                "control_plane_cluster": "cpu-unit",
                "pod": cluster.probe.pod,
                "capacity_run": RUN,
            },
        }
    ], "the private receiver must not discover other Kubernetes Pods"
    assert job["authorization"] == {
        "type": "Bearer",
        "credentials_file": "/etc/capacity-token/execution-token",
    }, "only the run token file authorizes scraping"
    assert result["processors"]["filter/capacity"]["metrics"]["include"] == {
        "match_type": "strict",
        "metric_names": [
            "up",
            "gpu_fault_store_io_in_flight",
            "gpu_fault_store_io_max_in_flight",
        ],
    }, "post-receiver filtering must remove automatically added scrape metrics"
    assert result["exporters"]["prometheusremotewrite"]["target_info"] == {
        "enabled": False
    }, "the exporter cannot synthesize a fourth family"
    assert result["service"]["pipelines"]["metrics"]["processors"] == [
        "filter/capacity",
        "batch",
    ], "filtering must precede export batching"
    assert "kubernetes_sd_configs" not in yaml.safe_dump(result), (
        "there is no production discovery path"
    )


def test_start_and_stop_only_own_pod_and_configmap(cluster: Cluster) -> None:
    value = cluster.companion()
    result = value.start()
    assert result["ready"] is True and result["image"] == IMAGE, (
        "readiness must bind the actual digest"
    )
    configmap, pod = cluster.creates
    assert configmap["immutable"] is True, "the generated config is immutable"
    assert pod["spec"]["restartPolicy"] == "Never" and not pod["metadata"].get(
        "ownerReferences"
    ), "standalone Pods cannot resurrect through a controller"
    assert pod["spec"]["activeDeadlineSeconds"] <= 1200 - module.CLEANUP_SECONDS, (
        "cleanup time must remain outside the Pod lifetime"
    )
    secrets = [
        volume["secret"] for volume in pod["spec"]["volumes"] if "secret" in volume
    ]
    assert secrets == [
        {
            "secretName": cluster.harness.secret_name,
            "defaultMode": 288,
            "items": [{"key": "execution-token", "path": "execution-token"}],
        }
    ], "only one key from the existing run Secret may be projected"
    assert pod["metadata"]["labels"]["app"] != module.ADOT, (
        "production ADOT inventory must exclude the companion"
    )
    cleaned = value.stop()
    assert cleaned["cleanup_complete"] is True, (
        "cleanup needs resource and process proofs"
    )
    assert cleaned["termination_proof"]["source"] == "kubelet-runningpods", (
        "API disappearance alone is insufficient"
    )
    assert [kind for kind, _ in cluster.deletes] == ["Pod", "ConfigMap"], (
        "token/config dependencies must outlive the scraper process"
    )
    assert not cluster.runtime_present, "the kubelet must report process termination"
    evidence = "".join(
        path.read_text() for path in Path(cluster.harness.run_dir).glob("*.json")
    )
    assert (
        SYNTHETIC not in evidence and "gpu-fault-control-plane-active" not in evidence
    ), "journals contain only safe identities and own references"
    assert value.stop()["cleanup_complete"] is True, (
        "verified cleanup may be repeated without creating resources"
    )


@pytest.mark.parametrize(
    "problem",
    ["deployment", "pod", "account", "config", "connection", "namespace", "node"],
)
def test_source_drift_refuses_all_creation(cluster: Cluster, problem: str) -> None:
    value = cluster.companion()
    if problem == "deployment":
        cluster.get("Deployment", module.ADOT)["metadata"]["generation"] += 1
    elif problem == "pod":
        cluster.get("Pod", module.ADOT + "-pod")["metadata"]["uid"] = "recreated-pod"
    elif problem == "account":
        cluster.get("ServiceAccount", module.ADOT)["metadata"]["uid"] = (
            "recreated-account"
        )
    elif problem == "config":
        current = config()
        current["processors"]["batch"]["timeout"] = "4s"
        cluster.get("ConfigMap", module.ADOT)["data"]["collector.yaml"] = (
            yaml.safe_dump(current)
        )
    elif problem == "connection":
        cluster.kubeconfig.write_text("different synthetic kubeconfig\n")
    elif problem == "namespace":
        cluster.get("Namespace", NAMESPACE)["metadata"]["uid"] = "recreated-namespace"
    else:
        cluster.get("Node", NODE)["metadata"]["uid"] = "recreated-node"
    with pytest.raises(module.ScrapeCompanionError):
        value.start()
    assert cluster.creates == [], "source drift must be rejected before mutation"


@pytest.mark.parametrize(
    "problem",
    [
        "pod-owner",
        "replica-owner",
        "not-ready",
        "image",
        "identity",
        "credentials",
        "endpoint",
        "debug",
        "duplicate-yaml",
    ],
)
def test_capture_rejects_untrusted_source_without_serializing_it(
    cluster: Cluster, problem: str
) -> None:
    pod = cluster.get("Pod", module.ADOT + "-pod")
    current = config()
    if problem == "pod-owner":
        pod["metadata"]["ownerReferences"][0]["uid"] = "foreign-replica"
    elif problem == "replica-owner":
        cluster.get("ReplicaSet", module.ADOT + "-rs")["metadata"]["ownerReferences"][
            0
        ]["uid"] = "foreign-deployment"
    elif problem == "not-ready":
        pod["status"]["conditions"][0]["status"] = "False"
    elif problem == "image":
        pod["status"]["containerStatuses"][0]["imageID"] = (
            "registry.example/adot:mutable"
        )
    elif problem == "identity":
        pod["spec"]["volumes"][-1]["projected"]["sources"][0]["serviceAccountToken"][
            "audience"
        ] = "foreign"
    elif problem == "credentials":
        pod["spec"]["containers"][0]["env"].append(
            {"name": "AWS_SECRET_ACCESS_KEY", "value": SYNTHETIC}
        )
    elif problem == "endpoint":
        current["exporters"]["prometheusremotewrite"]["endpoint"] = (
            "https://foreign.example/" + SYNTHETIC
        )
    elif problem == "debug":
        current["service"]["telemetry"]["logs"]["level"] = "debug"
    raw = yaml.safe_dump(current)
    if problem == "duplicate-yaml":
        raw += "\nexporters:\n  unexpected: " + SYNTHETIC + "\n"
    cluster.get("ConfigMap", module.ADOT)["data"]["collector.yaml"] = raw
    with pytest.raises(module.ScrapeCompanionError) as error:
        cluster.capture()
    assert SYNTHETIC not in str(error.value), (
        "invalid source values must never be printed"
    )
    assert cluster.creates == [], "source inspection cannot create resources"


@pytest.mark.parametrize("kind", ["Pod", "ConfigMap"])
def test_lost_creation_ack_is_not_adopted_or_cleaned(
    cluster: Cluster, kind: str
) -> None:
    value = cluster.companion()
    cluster.lost_ack = kind
    with pytest.raises(module.ScrapeCompanionError) as error:
        value.start()
    assert SYNTHETIC not in str(error.value), (
        "creation diagnostics must not expose untrusted output"
    )
    with pytest.raises(module.ScrapeCompanionError):
        value.stop()
    assert cluster.deletes == [], (
        "a matching name or label cannot replace a direct creation ACK"
    )
    journal = json.loads(
        (Path(cluster.harness.run_dir) / "capacity-scrape-ownership.json").read_text()
    )
    assert journal["creations"][f"{kind.lower()}/{COMPANION}"]["uid"] is None, (
        "unknown creation authority remains unresolved"
    )


@pytest.mark.parametrize(
    "problem",
    [
        "privileged",
        "env",
        "host",
        "account",
        "sidecar",
        "gates",
        "pod-identity-label-value",
        "extra-label",
    ],
)
def test_admitted_authority_drift_never_activates(
    cluster: Cluster, problem: str
) -> None:
    value = cluster.companion()
    cluster.admission = problem
    with pytest.raises(module.ScrapeCompanionError):
        value.start()
    assert not any(call[0] == "patch" for call in cluster.calls), (
        "unapproved admission cannot release the scheduling gate"
    )
    assert not cluster.runtime_present, (
        "no scraper process is authorized before admission validation"
    )


def test_bound_projection_without_optional_admission_marker_is_accepted(
    cluster: Cluster,
) -> None:
    cluster.admission = "without-pod-identity-label"
    value = cluster.companion()
    value.start()
    assert value.stop()["process_termination_proven"] is True


def test_irsa_cannot_adopt_a_pod_identity_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = Cluster(tmp_path, irsa=True)
    monkeypatch.setattr(module, "run_fixture_command", cluster.run)
    monkeypatch.setattr(module, "time", cluster.clock)
    monkeypatch.setattr(module, "_now", cluster.clock.now)
    cluster.admission = "unexpected-pod-identity"
    with pytest.raises(module.ScrapeCompanionError, match="metadata changed"):
        cluster.companion().start()
    assert not any(call[0] == "patch" for call in cluster.calls), (
        'test_irsa_cannot_adopt_a_pod_identity_marker: expected no any(call[0] == "patch" for call in cluster.calls)'
    )
    assert not cluster.runtime_present, (
        "test_irsa_cannot_adopt_a_pod_identity_marker: expected no cluster.runtime_present"
    )


@pytest.mark.parametrize("kind", ["Pod", "ConfigMap"])
def test_preexisting_resources_are_not_adopted(cluster: Cluster, kind: str) -> None:
    value = cluster.companion()
    cluster.put(document(kind, COMPANION))
    with pytest.raises(module.ScrapeCompanionError, match="already exists"):
        value.start()
    assert cluster.creates == [] and cluster.deletes == [], (
        "name collision grants no ownership"
    )


def test_replaced_pod_uid_blocks_cleanup(cluster: Cluster) -> None:
    value = cluster.companion()
    value.start()
    cluster.get("Pod", COMPANION)["metadata"]["uid"] = "foreign-pod"
    with pytest.raises(module.ScrapeCompanionError):
        value.stop()
    assert cluster.deletes == [], (
        "cleanup must not delete a replacement with identical labels"
    )


@pytest.mark.parametrize(
    "problem",
    [
        "still-running",
        "node-replaced",
        "kubelet-unreadable",
        "kubelet-malformed",
        "namespace-gone",
    ],
)
def test_api_absence_is_not_process_termination(cluster: Cluster, problem: str) -> None:
    value = cluster.companion()
    value.start()
    if problem == "still-running":
        cluster.runtime_stuck = True
    elif problem == "node-replaced":
        cluster.node_drift = True
    elif problem == "kubelet-unreadable":
        cluster.kubelet_error = True
    elif problem == "kubelet-malformed":
        cluster.bad_kubelet = True
    else:
        cluster.resources.pop(("namespace", NAMESPACE))
    with pytest.raises(module.ScrapeCompanionError) as error:
        value.stop()
    assert SYNTHETIC not in str(error.value), (
        "cleanup errors may not print arbitrary command output"
    )
    assert not any(kind == "ConfigMap" for kind, _ in cluster.deletes), (
        "config/token dependencies survive an unproven process stop"
    )
    assert cluster.get("ConfigMap", COMPANION), (
        "failed shutdown retains its immutable configuration"
    )


def test_terminal_container_status_can_prove_stop_without_kubelet_access(
    cluster: Cluster,
) -> None:
    value = cluster.companion()
    value.start()
    cluster.terminated_on_delete = True
    cluster.kubelet_error = True
    result = value.stop()
    assert result["termination_proof"]["source"] == "container-status", (
        "a bound terminal container is an independent stop proof"
    )
    assert result["cleanup_complete"] is True, (
        "valid terminal status does not require node proxy permission"
    )


def test_lost_delete_ack_requires_independent_absence_and_process_proof(
    cluster: Cluster,
) -> None:
    value = cluster.companion()
    value.start()
    cluster.delete_ack_lost = True
    result = value.stop()
    assert result["cleanup_complete"] is True, (
        "known creation UIDs may be observed after a lost deletion ACK"
    )
    assert result["termination_proof"]["source"] == "kubelet-runningpods", (
        "lost ACK does not waive the process proof"
    )


def test_failed_configmap_cleanup_is_not_success(cluster: Cluster) -> None:
    value = cluster.companion()
    value.start()
    cluster.config_delete_stuck = True
    with pytest.raises(module.ScrapeCompanionError):
        value.stop()
    assert [kind for kind, _ in cluster.deletes] == ["Pod", "ConfigMap"], (
        "cleanup must attempt only its own dependencies in order"
    )


@pytest.mark.parametrize(
    "bad", ['pod"escape', "pod\\escape", "pod\nescape", "../pod", ""]
)
def test_selector_and_resource_names_are_strict(cluster: Cluster, bad: str) -> None:
    source = cluster.capture()
    cluster.probe.pod = bad
    before = list(cluster.calls)
    with pytest.raises(module.ScrapeCompanionError):
        module.ScrapeCompanion(
            cluster.harness,
            cluster.probe,
            expected_source=source,
            deadline=cluster.clock.now() + timedelta(seconds=1200),
        )
    assert cluster.calls == before, "invalid names are rejected without cluster calls"


def test_insufficient_window_and_production_secret_are_rejected(
    cluster: Cluster,
) -> None:
    source = cluster.capture()
    value = module.ScrapeCompanion(
        cluster.harness,
        cluster.probe,
        expected_source=source,
        deadline=cluster.clock.now() + timedelta(seconds=60),
    )
    with pytest.raises(module.ScrapeCompanionError, match="budget"):
        value.start()
    cluster.harness.secret_name = "gpu-fault-control-plane-active"
    with pytest.raises(module.ScrapeCompanionError, match="Secret"):
        module.ScrapeCompanion(
            cluster.harness,
            cluster.probe,
            expected_source=source,
            deadline=cluster.clock.now() + timedelta(seconds=1200),
        )
    assert cluster.creates == [], (
        "neither a short window nor a production token grants startup"
    )


def test_pure_builder_rejects_credential_extensions(cluster: Cluster) -> None:
    source = cluster.capture()
    source["pipeline"]["exporters"]["prometheusremotewrite"]["headers"] = {
        "Authorization": SYNTHETIC
    }
    with pytest.raises(module.ScrapeCompanionError) as error:
        module.build_scrape_config(
            source=source,
            service=PROBE,
            namespace=NAMESPACE,
            pod=cluster.probe.pod,
            run_id=RUN,
        )
    assert SYNTHETIC not in str(error.value), (
        "the local validation builder must not copy arbitrary credential-bearing extensions"
    )


@pytest.mark.parametrize("problem", ["mount", "configmap", "args", "env-from"])
def test_source_pod_must_run_its_deployments_actual_config(
    cluster: Cluster, problem: str
) -> None:
    pod = cluster.get("Pod", module.ADOT + "-pod")
    if problem == "mount":
        pod["spec"]["containers"][0]["volumeMounts"][0]["subPath"] = "foreign.yaml"
    elif problem == "configmap":
        pod["spec"]["volumes"][0]["configMap"]["name"] = "foreign-config"
    elif problem == "args":
        pod["spec"]["containers"][0]["args"] = ["--config=/foreign/config.yaml"]
    else:
        pod["spec"]["containers"][0]["envFrom"] = [{"secretRef": {"name": "foreign"}}]
    with pytest.raises(module.ScrapeCompanionError):
        cluster.capture()
    assert cluster.creates == [], (
        "a label and owner chain cannot authorize different running configuration"
    )


@pytest.mark.parametrize(
    "finished", ["", "invalid", "2026-09-17T00:00:00", "2099-01-01T00:00:00Z"]
)
def test_malformed_terminal_status_does_not_replace_kubelet_proof(
    cluster: Cluster, finished: str
) -> None:
    value = cluster.companion()
    value.start()
    current = cluster.get("Pod", COMPANION)
    current["status"]["containerStatuses"][0]["state"] = {
        "terminated": {"exitCode": 0, "finishedAt": finished}
    }
    cluster.kubelet_error = True
    with pytest.raises(module.ScrapeCompanionError):
        value.stop()
    assert not any(kind == "ConfigMap" for kind, _ in cluster.deletes), (
        "invalid termination timestamps cannot authorize dependency cleanup"
    )


def test_startup_timeout_retains_acknowledged_resources_for_cleanup(
    cluster: Cluster, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = cluster.companion()

    def pending(*args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        result = cluster.kubectl(*args, **kwargs)
        if args[:3] == ("patch", "Pod", COMPANION):
            cluster.get("Pod", COMPANION)["status"] = {"phase": "Pending"}
            cluster.runtime_present = False
        return result

    monkeypatch.setattr(cluster.harness, "kubectl", pending)
    with pytest.raises(module.ScrapeCompanionError, match="readiness timed out"):
        value.start()
    assert cluster.clock.elapsed == module.STARTUP_SECONDS, (
        "an unready Pod must exhaust only the bounded startup window"
    )
    assert cluster.get("Pod", COMPANION)["status"] == {"phase": "Pending"}, (
        "readiness cannot be inferred from successful resource creation"
    )
    lifecycle = json.loads(
        (Path(cluster.harness.run_dir) / "capacity-scrape-lifecycle.json").read_text()
    )
    assert lifecycle["start_error_type"] == "ScrapeCompanionError", (
        "startup failure must remain observable before cleanup"
    )
    assert lifecycle.get("ready") is not True, "timeout must never leave Ready evidence"
    result = value.stop()
    assert result["process_termination_proven"] is True, (
        "missing container status requires an independent process-stop proof"
    )
    assert result["termination_proof"]["source"] == "kubelet-runningpods", (
        "the bound kubelet must confirm the unready Pod is not running"
    )
    assert [kind for kind, _ in cluster.deletes] == ["Pod", "ConfigMap"], (
        "startup timeout must not lose acknowledged resource custody"
    )


@pytest.mark.parametrize("stage", ["ConfigMap", "Pod"])
def test_interruption_before_creation_intent_cleans_only_acknowledged_resources(
    cluster: Cluster, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    value = cluster.companion()
    original = FixtureOwnership.intend
    interrupted = KeyboardInterrupt()

    def interrupt(custody: FixtureOwnership, manifest: dict[str, Any]) -> None:
        if manifest["kind"] == stage:
            raise interrupted
        original(custody, manifest)

    monkeypatch.setattr(FixtureOwnership, "intend", interrupt)
    with pytest.raises(KeyboardInterrupt) as error:
        value.start()
    assert error.value is interrupted, (
        "an operator interruption must propagate without conversion or retry"
    )
    result = value.stop()
    assert result["cleanup_complete"] is True, (
        "cleanup may finish when no unacknowledged creation was attempted"
    )
    assert result["termination_proof"] == {"source": "pod-not-created"}, (
        "the journal must distinguish never-created from a disappeared process"
    )
    expected = ["ConfigMap"] if stage == "Pod" else []
    assert [item["kind"] for item in cluster.creates] == expected, (
        "interruption must prevent the selected creation and every successor"
    )
    assert [kind for kind, _ in cluster.deletes] == expected, (
        "cleanup cannot invent custody of resources that were never created"
    )
    assert not any(call[:2] == ("get", "--raw") for call in cluster.calls), (
        "a never-created Pod does not require or authorize runtime interrogation"
    )
    journal = json.loads(
        (Path(cluster.harness.run_dir) / "capacity-scrape-ownership.json").read_text()
    )
    assert set(journal["creations"]) == {
        f"{kind.lower()}/{COMPANION}" for kind in expected
    }, "the completed journal must preserve the exact acknowledged creation set"
    assert journal["completed"] is True, "verified partial cleanup must be durable"


def test_stop_before_start_is_a_noop_without_cluster_io(cluster: Cluster) -> None:
    value = cluster.companion()
    calls = list(cluster.calls)
    result = value.stop()
    assert result == {
        "cleanup_complete": True,
        "process_termination_proven": True,
        "not_created": True,
    }, "a fresh companion must explicitly report that nothing was created"
    assert cluster.calls == calls, "stop before start must not contact the cluster"
    assert not list(Path(cluster.harness.run_dir).glob("capacity-scrape-*.json")), (
        "a no-op stop must not manufacture creation or cleanup custody"
    )


def test_kubelet_other_pods_do_not_block_owned_process_stop(
    cluster: Cluster, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = cluster.companion()
    value.start()
    source_pod = copy.deepcopy(cluster.get("Pod", module.ADOT + "-pod"))
    namesake = document("Pod", COMPANION)
    namesake["metadata"].update(namespace="other-namespace", uid="other-namespace-uid")

    def other_pods(*args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        result = cluster.kubectl(*args, **kwargs)
        if args[:2] == ("get", "--raw"):
            inventory = json.loads(result.stdout)
            inventory["items"].extend([source_pod, namesake])
            result.stdout = json.dumps(inventory)
        return result

    monkeypatch.setattr(cluster.harness, "kubectl", other_pods)
    result = value.stop()
    assert result["termination_proof"]["source"] == "kubelet-runningpods", (
        "only the owned Pod UID determines the process-stop verdict"
    )
    assert result["cleanup_complete"] is True, (
        "other Pods and cross-namespace namesakes are not owned-process residue"
    )
    assert cluster.get("Pod", module.ADOT + "-pod") == source_pod, (
        "the production collector must remain untouched"
    )
    assert [kind for kind, _ in cluster.deletes] == ["Pod", "ConfigMap"], (
        "kubelet inventory grants no deletion authority over unrelated Pods"
    )


def test_already_deleting_pod_is_observed_without_duplicate_delete(
    cluster: Cluster,
) -> None:
    value = cluster.companion()
    value.start()
    pod = cluster.get("Pod", COMPANION)
    pod["metadata"]["deletionTimestamp"] = cluster.clock.now().isoformat()
    pod["status"]["containerStatuses"][0]["state"] = {
        "terminated": {"exitCode": 0, "finishedAt": cluster.clock.now().isoformat()}
    }
    cluster.runtime_present = False
    cluster.terminal_reads = 1
    cluster.kubelet_error = True
    result = value.stop()
    assert result["termination_proof"]["source"] == "container-status", (
        "cleanup must retain the bound terminal status observed during deletion"
    )
    assert result["pod_absent"] is True, (
        "a deletion timestamp alone cannot substitute for confirmed disappearance"
    )
    assert [kind for kind, _ in cluster.deletes] == ["ConfigMap"], (
        "an existing Pod deletion must not be resubmitted or forced"
    )
    assert not any(call[:2] == ("get", "--raw") for call in cluster.calls), (
        "a complete terminal-status proof does not require a kubelet fallback"
    )


def test_delete_supervision_loss_propagates_and_retains_dependencies(
    cluster: Cluster, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = cluster.companion()
    value.start()
    lost = ProcessSupervisionLost("synthetic supervision loss")

    def lose_supervision(*args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if args[:2] == ("delete", "--raw"):
            cluster.calls.append(args)
            raise lost
        return cluster.kubectl(*args, **kwargs)

    monkeypatch.setattr(cluster.harness, "kubectl", lose_supervision)
    with pytest.raises(ProcessSupervisionLost) as error:
        value.stop()
    assert error.value is lost, (
        "supervision loss must not become an ordinary lost deletion acknowledgement"
    )
    assert cluster.calls[-1][:2] == ("delete", "--raw"), (
        "no further cluster commands are authorized after supervision loss"
    )
    assert cluster.deletes == [], "unknown deletion results are not successful cleanup"
    assert cluster.get("ConfigMap", COMPANION), (
        "configuration and token dependencies must survive an unproven shutdown"
    )
    lifecycle = json.loads(
        (Path(cluster.harness.run_dir) / "capacity-scrape-lifecycle.json").read_text()
    )
    assert lifecycle["cleanup"] == {
        "cleanup_complete": False,
        "process_termination_proven": False,
        "error_type": "ProcessSupervisionLost",
    }, "the durable result must preserve the supervision-loss refusal"
    assert "pod_delete_ack_lost" not in lifecycle, (
        "ordinary readback recovery must not be offered for supervision loss"
    )


def test_unreadable_kubeconfig_refuses_capture_before_cluster_io(
    cluster: Cluster,
) -> None:
    missing = cluster.kubeconfig.with_name("missing.kubeconfig")
    calls = list(cluster.calls)
    with pytest.raises(module.ScrapeCompanionError, match="kubeconfig identity"):
        module.capture_scrape_source(
            kubeconfig=missing,
            namespace=NAMESPACE,
            region=REGION,
            workspace_id=WORKSPACE,
        )
    assert cluster.calls == calls, (
        "an unavailable CPU connection must not fall back to ambient cluster credentials"
    )


def test_incomplete_source_response_is_reported_without_raw_values(
    cluster: Cluster,
) -> None:
    deployment = cluster.get("Deployment", module.ADOT)
    deployment.pop("status")
    deployment["untrusted"] = SYNTHETIC
    with pytest.raises(
        module.ScrapeCompanionError, match="source configuration is incomplete"
    ) as error:
        cluster.capture()
    assert SYNTHETIC not in str(error.value), (
        "source-shape errors must remain sanitized"
    )
    assert cluster.creates == [], (
        "incomplete source responses cannot authorize creation"
    )


def test_builder_missing_pipeline_is_a_sanitized_input_error(cluster: Cluster) -> None:
    calls = list(cluster.calls)
    with pytest.raises(
        module.ScrapeCompanionError, match="builder source is invalid"
    ) as error:
        module.build_scrape_config(
            source={"untrusted": SYNTHETIC},
            service=PROBE,
            namespace=NAMESPACE,
            pod=cluster.probe.pod,
            run_id=RUN,
        )
    assert SYNTHETIC not in str(error.value), (
        "malformed builder inputs must not be echoed"
    )
    assert cluster.calls == calls, (
        "malformed pure-builder inputs cannot cause cluster I/O"
    )


def test_invalid_yaml_is_rejected_without_echoing_the_document(
    cluster: Cluster,
) -> None:
    cluster.get("ConfigMap", module.ADOT)["data"]["collector.yaml"] = (
        "exporters: [" + SYNTHETIC
    )
    with pytest.raises(
        module.ScrapeCompanionError, match="ADOT YAML is invalid"
    ) as error:
        cluster.capture()
    assert SYNTHETIC not in str(error.value), (
        "parser errors must not expose YAML excerpts"
    )
    assert cluster.creates == [], "invalid configuration cannot authorize a companion"


def test_real_utc_clock_keeps_the_declared_cleanup_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = Cluster(tmp_path)
    monkeypatch.setattr(module, "run_fixture_command", cluster.run)
    monkeypatch.setattr(module, "time", cluster.clock)
    value = module.ScrapeCompanion(
        cluster.harness,
        cluster.probe,
        expected_source=cluster.capture(),
        deadline=datetime.now(timezone.utc) + timedelta(seconds=1200),
    )
    result = value.start()
    assert (
        module.STARTUP_SECONDS
        <= result["active_deadline_seconds"]
        <= 1200 - module.CLEANUP_SECONDS
    ), "the real UTC clock must reserve cleanup time before setting the Pod lifetime"
    assert value.stop()["cleanup_complete"] is True, (
        "the normal clock path must preserve the same verified cleanup contract"
    )
