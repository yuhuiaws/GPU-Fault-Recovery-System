"""Minimal CPU watchdog resources, without execution tokens or GPU credentials."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import posixpath
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from scripts.e2e.regional.destr008_parallel import gather
from scripts.e2e.regional.probes.destr008_cancellation_protocol import (
    DRAIN_SECONDS,
    SOURCE_FILES,
    Plan,
    digest,
    initial_data,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
    component_python,
)
from scripts.e2e.regional.seeded_command_fixture import RUN_LABEL
from scripts.e2e.regional.destr008_watchdog_fields import (  # noqa: E402
    _defaults,
    _image,
    _image_id,
    _items,
    _object,
    _same,
    _status_image,
    _text,
)

DEPLOYMENT = "gpu-fault-control-worker"
STORE_SECRET = "gpu-fault-aurora"
STORE_KEY = "postgres-url"
STORE_DIRECTORY = "/etc/gpu-fault/aurora"
CA_CONFIGMAP = "gpu-fault-rds-ca-bundle"
CA_KEY = "ca-bundle.pem"
CA_DIRECTORY = "/etc/gpu-fault/rds"
CODE_DIRECTORY = "/opt/gpu-fault/acceptance-watchdog"
CODE_SOURCE = Path(__file__).with_name("probes")
SCHEDULING_GATE = "gpu-fault.io/acceptance-watchdog-verified"
MODE_KEYS = (
    "GPU_FAULT_POSTGRES_HOT_STATE_MODE",
    "GPU_FAULT_PROCESSOR_QUEUE_STATE_MODE",
)
MODES = {"legacy", "dual", "dedicated"}
SAFE_CONFIG_KEYS = {
    *MODE_KEYS,
    "GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT",
    "GPU_FAULT_STORE_URL_FILE",
}
WatchdogPhase = Literal["gated", "running", "succeeded"]
HISTORY_LOADER = (
    "import hashlib,sys\n"
    "source=sys.stdin.buffer.read(65537)\n"
    "expected=sys.argv[1]\n"
    "actual=hashlib.sha256(source).hexdigest()\n"
    "if len(source)>65536 or actual!=expected:\n"
    " raise SystemExit('CPU history probe source differs')\n"
    "exec(compile(source,'<cpu-history-capability>','exec'),"
    "{'__name__':'__main__','_PROBE_SHA256':actual})\n"
)


@dataclass(frozen=True)
class CpuRuntime:
    namespace: str
    namespace_uid: str
    deployment_uid: str
    generation: int
    image: str
    modes: dict[str, str]
    config_identities: dict[str, dict[str, str]]

    def identity(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "namespace": self.namespace,
                "namespace_uid": self.namespace_uid,
                "deployment_uid": self.deployment_uid,
                "generation": self.generation,
                "image": self.image,
                "modes": self.modes,
                "config_identities": self.config_identities,
            }
        )


def read_object(regional: RegionalLiveFixture, kind: str, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            regional.kubectl("cpu", "get", kind, name, "-o", "json", timeout=30)
        )
    except (ValueError, TypeError):
        raise RegionalFixtureError("CPU watchdog source response is invalid") from None
    if not isinstance(value, dict):
        raise RegionalFixtureError("CPU watchdog source resource is not an object")
    metadata = value.get("metadata")
    if (
        not isinstance(metadata, dict)
        or value.get("kind")
        != {
            "namespace": "Namespace",
            "configmap": "ConfigMap",
            "deployment": "Deployment",
        }[kind]
        or value.get("apiVersion") != ("apps/v1" if kind == "deployment" else "v1")
        or metadata.get("name") != name
        or not _text(metadata.get("uid"))
        or not _text(metadata.get("resourceVersion"))
        or metadata.get("deletionTimestamp") is not None
        or (
            kind != "namespace"
            and metadata.get("namespace") != regional.settings.namespace
        )
    ):
        raise RegionalFixtureError(
            "CPU watchdog source resource identity is incomplete"
        )
    return value


def effective_store_configuration(
    regional: RegionalLiveFixture, container: dict[str, Any]
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    effective: dict[str, str] = {}
    identities: dict[str, dict[str, str]] = {}
    names: list[str] = []
    for source in _items(container.get("envFrom", [])):
        if (
            not isinstance(source, dict)
            or set(source) - {"configMapRef", "prefix"}
            or source.get("prefix", "") != ""
            or not isinstance(source.get("configMapRef"), dict)
        ):
            raise RegionalFixtureError(
                "watchdog cannot inherit an ambiguous environment"
            )
        reference = source["configMapRef"]
        name = reference.get("name")
        if (
            not _text(name)
            or set(reference) - {"name", "optional"}
            or reference.get("optional", False) is not False
            or name in names
        ):
            raise RegionalFixtureError("watchdog configuration source must be required")
        names.append(name)
    configmaps = gather(
        [lambda name=name: read_object(regional, "configmap", name) for name in names]
    )
    for name, configmap in zip(names, configmaps, strict=True):
        data = configmap.get("data")
        if not isinstance(data, dict):
            raise RegionalFixtureError("watchdog configuration source is incomplete")
        identities[name] = {
            "uid": configmap["metadata"]["uid"],
            "resource_version": configmap["metadata"]["resourceVersion"],
        }
        for key in SAFE_CONFIG_KEYS & data.keys():
            if not isinstance(data[key], str) or key in effective:
                raise RegionalFixtureError("watchdog configuration value is invalid")
            effective[key] = data[key]
    variables = _items(container.get("env", []))
    names = [item.get("name") for item in variables]
    if any(not _text(name) for name in names) or len(set(names)) != len(names):
        raise RegionalFixtureError("CPU environment variable identity is ambiguous")
    url = [item for item in variables if item["name"] == "GPU_FAULT_STORE_URL"]
    expected_url = {
        "name": "GPU_FAULT_STORE_URL",
        "valueFrom": {"secretKeyRef": {"name": STORE_SECRET, "key": STORE_KEY}},
    }
    observed_url = copy.deepcopy(url[0]) if len(url) == 1 else {}
    secret_reference = _object(
        _object(observed_url.get("valueFrom", {})).get("secretKeyRef", {})
    )
    if secret_reference.get("optional", False) is False:
        secret_reference.pop("optional", None)
    _defaults(observed_url, {"value": ""})
    if not _same(observed_url, expected_url):
        raise RegionalFixtureError(
            "watchdog requires the existing CPU-only Store Secret reference"
        )
    for variable in variables:
        if variable["name"] in SAFE_CONFIG_KEYS:
            if set(variable) != {"name", "value"} or not isinstance(
                variable["value"], str
            ):
                raise RegionalFixtureError(
                    "watchdog configuration cannot resolve indirect overrides"
                )
            effective[variable["name"]] = variable["value"]
    if (
        any(effective.get(key) not in MODES for key in MODE_KEYS)
        or effective.get("GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT") != "false"
        or effective.get("GPU_FAULT_STORE_URL_FILE")
        != STORE_DIRECTORY + "/" + STORE_KEY
    ):
        raise RegionalFixtureError(
            "CPU Store mode, credential projection or schema guard is unsupported"
        )
    return {key: effective[key] for key in MODE_KEYS}, identities


def read_runtime(regional: RegionalLiveFixture) -> CpuRuntime:
    namespace, deployment = gather(
        [
            lambda: read_object(regional, "namespace", regional.settings.namespace),
            lambda: read_object(regional, "deployment", DEPLOYMENT),
        ]
    )
    spec = _object(deployment.get("spec"))
    status = _object(deployment.get("status"))
    generation = deployment["metadata"].get("generation")
    replicas = spec.get("replicas")
    conditions = _conditions(status)
    if (
        type(generation) is not int
        or generation < 1
        or type(replicas) is not int
        or replicas < 1
        or type(status.get("observedGeneration")) is not int
        or status["observedGeneration"] != generation
        or any(
            type(status.get(key)) is not int or status[key] != replicas
            for key in (
                "replicas",
                "updatedReplicas",
                "readyReplicas",
                "availableReplicas",
            )
        )
        or not _same(status.get("unavailableReplicas", 0), 0)
        or not _same(status.get("terminatingReplicas", 0), 0)
        or spec.get("paused", False) is not False
        or conditions.get("Available") != "True"
        or conditions.get("Progressing") != "True"
        or conditions.get("ReplicaFailure") == "True"
    ):
        raise RegionalFixtureError(
            "CPU watchdog requires a complete Ready control-worker baseline"
        )
    pod = _object(_object(spec.get("template")).get("spec"))
    containers = _items(pod.get("containers"))
    selected = [
        item
        for item in containers
        if isinstance(item, dict) and item.get("name") == "control-worker"
    ]
    if len(selected) != 1 or len(containers) != 1:
        raise RegionalFixtureError("control-worker runtime container is ambiguous")
    container = selected[0]
    image = container.get("image")
    if not _image(image):
        raise RegionalFixtureError("watchdog runtime image must be immutable")
    modes, sources = effective_store_configuration(regional, container)
    mounts = _items(container.get("volumeMounts", []))
    volumes = _items(pod.get("volumes", []))
    volume_names = [item.get("name") for item in volumes]
    mount_paths = [item.get("mountPath") for item in mounts]
    if (
        any(not _text(value) for value in volume_names)
        or any(
            not _text(path)
            or not path.startswith("/")
            or posixpath.normpath(path) != path
            for path in mount_paths
        )
        or len(set(volume_names)) != len(volume_names)
        or len(set(mount_paths)) != len(mount_paths)
    ):
        raise RegionalFixtureError("CPU credential mount identity is ambiguous")
    for path, source in (
        (STORE_DIRECTORY, {"secret": {"secretName": STORE_SECRET}}),
        (CA_DIRECTORY, {"configMap": {"name": CA_CONFIGMAP}}),
    ):
        matching = [
            item
            for item in mounts
            if isinstance(item, dict) and item.get("mountPath") == path
        ]
        if (
            len(matching) != 1
            or matching[0].get("readOnly") is not True
            or "subPath" in matching[0]
            or "subPathExpr" in matching[0]
            or set(matching[0]) - {"name", "mountPath", "readOnly", "mountPropagation"}
            or matching[0].get("mountPropagation", "None") != "None"
            or any(mount["mountPath"].startswith(path + "/") for mount in mounts)
        ):
            raise RegionalFixtureError(
                "CPU credential or CA mount is not a complete read-only projection"
            )
        actual = [
            item
            for item in volumes
            if isinstance(item, dict) and item.get("name") == matching[0].get("name")
        ]
        source_kind = next(iter(source))
        projection = actual[0].get(source_kind) if len(actual) == 1 else None
        if (
            not isinstance(projection, dict)
            or set(actual[0]) != {"name", source_kind}
            or set(projection)
            - {*source[source_kind], "defaultMode", "optional", "items"}
            or any(
                projection.get(key) != value
                for key, value in source[source_kind].items()
            )
            or projection.get("optional", False) is not False
            or projection.get("items") not in (None, [])
            or (
                "defaultMode" in projection
                and (
                    type(projection["defaultMode"]) is not int
                    or projection["defaultMode"]
                    not in {0o400, 0o440, 0o444, 0o600, 0o640, 0o644}
                )
            )
        ):
            raise RegionalFixtureError("CPU credential or CA volume source differs")
    ca = read_object(regional, "configmap", CA_CONFIGMAP)
    ca_data = _object(ca.get("data")).get(CA_KEY)
    if not isinstance(ca_data, str) or not ca_data.strip():
        raise RegionalFixtureError("CPU RDS CA projection is incomplete")
    sources[CA_CONFIGMAP] = {
        "uid": ca["metadata"]["uid"],
        "resource_version": ca["metadata"]["resourceVersion"],
    }
    fresh_namespace, fresh_deployment, *current_sources = gather(
        [
            lambda: read_object(regional, "namespace", regional.settings.namespace),
            lambda: read_object(regional, "deployment", DEPLOYMENT),
            *[
                lambda name=name: read_object(regional, "configmap", name)
                for name in sources
            ],
        ]
    )
    if (
        fresh_namespace["metadata"]["uid"] != namespace["metadata"]["uid"]
        or fresh_deployment["metadata"]["uid"] != deployment["metadata"]["uid"]
        or not _same(fresh_deployment["metadata"].get("generation"), generation)
        or not _same(fresh_deployment.get("spec"), spec)
        or not _same(fresh_deployment.get("status"), status)
    ):
        raise RegionalFixtureError("CPU runtime changed during watchdog discovery")
    for (name, identity), fresh in zip(sources.items(), current_sources, strict=True):
        current = fresh["metadata"]
        if (
            current["uid"] != identity["uid"]
            or current["resourceVersion"] != identity["resource_version"]
        ):
            raise RegionalFixtureError(
                "CPU configuration changed during watchdog discovery"
            )
    return CpuRuntime(
        regional.settings.namespace,
        namespace["metadata"]["uid"],
        deployment["metadata"]["uid"],
        generation,
        image,
        modes,
        sources,
    )


def history_api_identity() -> dict[str, Any]:
    """Inspect the installed PostgreSQL API without constructing a Store or reading env."""
    import hashlib
    import importlib
    import inspect
    from pathlib import Path

    from gpu_fault.store import PostgresStore

    method = PostgresStore.list_job_recovery_workflow_incidents
    signature = inspect.signature(method)
    parameters = list(signature.parameters.values())
    if (
        PostgresStore.__module__ != "gpu_fault.store.postgres.store"
        or method.__module__ != "gpu_fault.store.postgres.workflows"
        or method.__qualname__
        != "PostgresWorkflowMixin.list_job_recovery_workflow_incidents"
        or [item.name for item in parameters]
        != ["self", "cluster_id", "job_id", "attempt_id", "limit", "include_terminal"]
        or any(
            item.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
            or item.default is not inspect.Parameter.empty
            for item in parameters[:4]
        )
        or any(item.kind != inspect.Parameter.KEYWORD_ONLY for item in parameters[4:])
        or type(parameters[4].default) is not int
        or parameters[4].default != 100
        or parameters[5].default is not False
        or parameters[5].annotation != "bool"
    ):
        raise ValueError("CPU_HISTORY_SIGNATURE")
    sources = {}
    for name in (
        "gpu_fault.store.contracts",
        "gpu_fault.store.postgres.store",
        "gpu_fault.store.postgres.workflows",
    ):
        module = importlib.import_module(name)
        filename = inspect.getsourcefile(module)
        if filename is None:
            raise ValueError("CPU_HISTORY_SOURCE")
        sources[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "method": method.__module__ + "." + method.__qualname__,
        "signature": str(signature),
        "sources": sources,
    }


def history_capability_script() -> str:
    return (
        "from __future__ import annotations\n"
        "import json,logging\nlogging.disable(logging.CRITICAL)\n"
        + inspect.getsource(history_api_identity)
        + "\ntry:\n"
        " result=history_api_identity()\n"
        " result['probe_sha256']=_PROBE_SHA256\n"
        "except Exception:\n"
        " result={'error':'CPU_HISTORY_CAPABILITY_UNAVAILABLE'}\n"
        "print(json.dumps(result,sort_keys=True))\n"
    )


RAW_LIST_PATHS = {
    ("replicaset", "apps/v1"): "/apis/apps/v1/namespaces/{namespace}/replicasets",
    ("pod", "v1"): "/api/v1/namespaces/{namespace}/pods",
}


def raw_list_path(kind: str, api_version: str, namespace: str) -> str:
    """The server's own typed list for ``kind``.

    ``kubectl get <kind> -o json`` (client v1.35 here) always prints a
    client-side ``v1/List`` with an empty ``resourceVersion``, so the
    population check below -- typed list kind, a resource version to bind the
    survey to, no ``continue`` -- can only be satisfied by the API's list
    itself, read through ``kubectl get --raw``.
    """

    try:
        return RAW_LIST_PATHS[kind, api_version].format(namespace=namespace)
    except KeyError:
        raise RegionalFixtureError(
            f"CPU capability population has no raw list for {kind}"
        ) from None


def _runtime_list(
    regional: RegionalLiveFixture, kind: str, api_version: str, list_kind: str
) -> list[dict[str, Any]]:
    path = raw_list_path(kind, api_version, regional.settings.namespace)
    try:
        value = _object(
            json.loads(regional.kubectl("cpu", "get", "--raw", path, timeout=30))
        )
    except (TypeError, ValueError):
        raise RegionalFixtureError(
            "CPU capability population response is invalid"
        ) from None
    metadata = _object(value.get("metadata"))
    if (
        value.get("apiVersion") != api_version
        or value.get("kind") != list_kind
        or not _text(metadata.get("resourceVersion"))
        or metadata.get("continue", "") != ""
    ):
        raise RegionalFixtureError("CPU capability population is incomplete")
    items = _items(value.get("items"))
    # The server's typed list carries bare items; only kubectl stamps
    # apiVersion/kind on each one. The per-item identity checks below were
    # written against kubectl output and refused every live ReplicaSet as
    # "owner is unproven" (DESTR-008 attempt 9), so give each item the identity
    # the list envelope already proved.
    for item in items:
        item.setdefault("apiVersion", api_version)
        item.setdefault("kind", list_kind.removesuffix("List"))
    return items


def _worker_population(
    regional: RegionalLiveFixture, runtime: CpuRuntime
) -> dict[str, dict[str, Any]]:
    deployment = read_object(regional, "deployment", DEPLOYMENT)
    if deployment["metadata"]["uid"] != runtime.deployment_uid or not _same(
        deployment["metadata"].get("generation"), runtime.generation
    ):
        raise RegionalFixtureError("CPU capability Deployment identity changed")
    replicas = _object(deployment.get("spec")).get("replicas")
    expected_owner = [
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "name": DEPLOYMENT,
            "uid": runtime.deployment_uid,
            "controller": True,
            "blockOwnerDeletion": True,
        }
    ]
    replica_sets: dict[str, str] = {}
    for value in _runtime_list(regional, "replicaset", "apps/v1", "ReplicaSetList"):
        metadata = _object(value.get("metadata"))
        owners = _items(metadata.get("ownerReferences", []))
        if any(
            owner.get("uid") == runtime.deployment_uid
            or (owner.get("kind") == "Deployment" and owner.get("name") == DEPLOYMENT)
            for owner in owners
        ):
            name, uid = metadata.get("name"), metadata.get("uid")
            if (
                value.get("apiVersion") != "apps/v1"
                or value.get("kind") != "ReplicaSet"
                or metadata.get("namespace") != runtime.namespace
                or metadata.get("deletionTimestamp") is not None
                or not _text(name)
                or not _text(uid)
                or uid in replica_sets
                or name in replica_sets.values()
                or not _same(owners, expected_owner)
            ):
                raise RegionalFixtureError(
                    "CPU capability ReplicaSet owner is unproven"
                )
            replica_sets[uid] = name
    population: dict[str, dict[str, Any]] = {}
    for value in _runtime_list(regional, "pod", "v1", "PodList"):
        metadata = _object(value.get("metadata"))
        owners = _items(metadata.get("ownerReferences", []))
        labels = _object(metadata.get("labels", {}))
        if not (
            labels.get("app") == DEPLOYMENT
            or any(
                owner.get("uid") in replica_sets
                or (
                    owner.get("kind") == "ReplicaSet"
                    and owner.get("name") in replica_sets.values()
                )
                for owner in owners
            )
        ):
            continue
        name, uid = metadata.get("name"), metadata.get("uid")
        if (
            value.get("apiVersion") != "v1"
            or value.get("kind") != "Pod"
            or metadata.get("namespace") != runtime.namespace
            or metadata.get("deletionTimestamp") is not None
            or not _text(name)
            or not _text(uid)
            or name in population
            or any(item["uid"] == uid for item in population.values())
            or len(owners) != 1
            or owners[0].get("uid") not in replica_sets
            or not _same(
                owners,
                [
                    {
                        "apiVersion": "apps/v1",
                        "kind": "ReplicaSet",
                        "name": replica_sets[owners[0]["uid"]],
                        "uid": owners[0]["uid"],
                        "controller": True,
                        "blockOwnerDeletion": True,
                    }
                ],
            )
        ):
            raise RegionalFixtureError("CPU capability worker Pod owner is unproven")
        spec, status = _object(value.get("spec")), _object(value.get("status"))
        containers = _items(spec.get("containers"))
        states = _items(status.get("containerStatuses", []))
        conditions = _conditions(status)
        if (
            len(containers) != 1
            or containers[0].get("name") != "control-worker"
            or containers[0].get("image") != runtime.image
            or len(states) != 1
            or states[0].get("name") != "control-worker"
            or not _status_image(states[0].get("image"), runtime.image)
            or not _image_id(states[0].get("imageID"), runtime.image)
            or not _text(states[0].get("containerID"))
            or type(states[0].get("restartCount")) is not int
            or states[0]["restartCount"] < 0
            or states[0].get("ready") is not True
            or set(_object(states[0].get("state"))) != {"running"}
            or status.get("phase") != "Running"
            or conditions.get("Ready") != "True"
            or conditions.get("ContainersReady") != "True"
        ):
            raise RegionalFixtureError(
                "CPU capability requires a complete Ready worker population"
            )
        _timestamp(_object(states[0]["state"]["running"]).get("startedAt"))
        population[name] = {
            "uid": uid,
            "replica_set_uid": owners[0]["uid"],
            "image_id": states[0]["imageID"],
            "container_id": states[0]["containerID"],
            "restart_count": states[0]["restartCount"],
        }
    if type(replicas) is not int or replicas < 1 or len(population) != replicas:
        raise RegionalFixtureError("CPU capability worker population is incomplete")
    return population


def require_cpu_history_capability(
    regional: RegionalLiveFixture, runtime: CpuRuntime
) -> None:
    """Prove the scoped-history keyword/source in every real Ready CPU worker."""
    if read_runtime(regional).identity() != runtime.identity():
        raise RegionalFixtureError("CPU capability runtime binding changed")
    before = _worker_population(regional, runtime)
    source = history_capability_script()
    expected = {
        **history_api_identity(),
        "probe_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }

    def probe(name: str) -> Any:
        return json.loads(
            regional.kubectl(
                "cpu",
                "exec",
                "-i",
                name,
                "-c",
                "control-worker",
                "--",
                component_python("cpu"),
                "-I",
                "-B",
                "-c",
                HISTORY_LOADER,
                expected["probe_sha256"],
                input_text=source,
                timeout=30,
            )
        )

    try:
        results = gather([lambda name=name: probe(name) for name in sorted(before)])
    except Exception:
        raise RegionalFixtureError(
            "CPU scoped-history capability probe failed"
        ) from None
    if any(not _same(result, expected) for result in results):
        raise RegionalFixtureError(
            "CPU scoped-history API or implementation source differs"
        )
    if (
        _worker_population(regional, runtime) != before
        or read_runtime(regional).identity() != runtime.identity()
    ):
        raise RegionalFixtureError(
            "CPU worker identity changed during capability verification"
        )


def control_manifest(plan: Plan, runtime: CpuRuntime, name: str) -> dict[str, Any]:
    if (
        not _text(name)
        or len(name) > 58
        or not re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", name)
        or not all(
            _text(value)
            for value in (
                runtime.namespace,
                runtime.namespace_uid,
                runtime.deployment_uid,
            )
        )
        or type(runtime.generation) is not int
        or runtime.generation < 1
        or not _image(runtime.image)
        or set(runtime.modes) != set(MODE_KEYS)
        or any(value not in MODES for value in runtime.modes.values())
    ):
        raise RegionalFixtureError("CPU watchdog runtime binding is incomplete")
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "namespace": runtime.namespace,
            "labels": {
                RUN_LABEL: plan.run_id,
                "gpu-fault.io/acceptance-case": "GF-REGIONAL-DESTR-008",
            },
            "annotations": {
                "gpu-fault.io/acceptance-plan-sha256": digest(plan),
                "gpu-fault.io/acceptance-runtime-sha256": digest(runtime.identity()),
                "gpu-fault.io/acceptance-source-sha256": plan.probe_sha256,
            },
        },
        "data": initial_data(plan),
    }


def watchdog_source(
    plan: Plan,
    *,
    source_data: dict[str, str] | None = None,
) -> dict[str, str]:
    """Capture or verify the exact nonsecret source bytes bound by the plan."""
    try:
        if source_data is None:
            paths = [CODE_SOURCE / file for file in SOURCE_FILES]
            if CODE_SOURCE.is_symlink() or any(
                path.is_symlink() or not path.is_file() for path in paths
            ):
                raise ValueError("source file shape")
            source = {path.name: path.read_bytes() for path in paths}
        else:
            if set(source_data) != set(SOURCE_FILES) or any(
                not isinstance(value, str) for value in source_data.values()
            ):
                raise ValueError("captured source shape")
            source = {
                file: value.encode("utf-8") for file, value in source_data.items()
            }
        code_data = {file: value.decode("utf-8") for file, value in source.items()}
    except (OSError, ValueError):
        raise RegionalFixtureError(
            "CPU watchdog source files are unavailable"
        ) from None
    if (
        digest(
            {file: hashlib.sha256(value).hexdigest() for file, value in source.items()}
        )
        != plan.probe_sha256
    ):
        raise RegionalFixtureError(
            "CPU watchdog source or control-map identity differs"
        )
    return code_data


def _watchdog_pod_spec(
    plan: Plan, runtime: CpuRuntime, *, name: str, control_uid: str
) -> dict[str, Any]:
    token_path = "/var/run/secrets/kubernetes.io/serviceaccount"
    container = {
        "name": "cancellation-watchdog",
        "image": runtime.image,
        "imagePullPolicy": "IfNotPresent",
        "command": [
            component_python("cpu"),
            "-s",
            "-B",
            CODE_DIRECTORY + "/destr008_cancellation_probe.py",
            "--namespace",
            runtime.namespace,
            "--configmap",
            name,
            "--uid",
            control_uid,
            "--plan-sha256",
            digest(plan),
        ],
        "env": [
            {
                "name": "GPU_FAULT_STORE_URL",
                "valueFrom": {"secretKeyRef": {"name": STORE_SECRET, "key": STORE_KEY}},
            },
            {
                "name": "GPU_FAULT_STORE_URL_FILE",
                "value": STORE_DIRECTORY + "/" + STORE_KEY,
            },
            {"name": "GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT", "value": "false"},
            {"name": "GPU_FAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS", "value": "10"},
            *[
                {"name": key, "value": value}
                for key, value in sorted(runtime.modes.items())
            ],
        ],
        "envFrom": [],
        "volumeMounts": [
            {"name": "code", "mountPath": CODE_DIRECTORY, "readOnly": True},
            {"name": "store", "mountPath": STORE_DIRECTORY, "readOnly": True},
            {"name": "ca", "mountPath": CA_DIRECTORY, "readOnly": True},
            {"name": "api-access", "mountPath": token_path, "readOnly": True},
            {"name": "scratch", "mountPath": "/tmp"},
        ],
        "securityContext": {
            "privileged": False,
            "allowPrivilegeEscalation": False,
            "readOnlyRootFilesystem": True,
            "capabilities": {"drop": ["ALL"]},
        },
        "resources": {
            "requests": {"cpu": "50m", "memory": "128Mi"},
            "limits": {"cpu": "500m", "memory": "256Mi"},
        },
    }
    pod = {
        "serviceAccountName": name,
        "automountServiceAccountToken": False,
        "restartPolicy": "Never",
        "enableServiceLinks": False,
        "hostNetwork": False,
        "hostPID": False,
        "hostIPC": False,
        "terminationGracePeriodSeconds": 30,
        "dnsPolicy": "ClusterFirst",
        "schedulerName": "default-scheduler",
        # No preemptionPolicy: Priority admission refuses an explicit value
        # without a PriorityClass; at priority 0 there is nothing to preempt.
        "tolerations": [
            {
                "key": "node.kubernetes.io/" + key,
                "operator": "Exists",
                "effect": "NoExecute",
                "tolerationSeconds": 300,
            }
            for key in ("not-ready", "unreachable")
        ],
        "schedulingGates": [{"name": SCHEDULING_GATE}],
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 10001,
            "runAsGroup": 10001,
            "fsGroup": 10001,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "containers": [container],
        "initContainers": [],
        "volumes": [
            {
                "name": "code",
                "configMap": {
                    "name": name + "-code",
                    "defaultMode": 0o444,
                    "items": [{"key": file, "path": file} for file in SOURCE_FILES],
                },
            },
            {
                "name": "store",
                "secret": {
                    "secretName": STORE_SECRET,
                    "defaultMode": 0o440,
                    "items": [{"key": STORE_KEY, "path": STORE_KEY}],
                },
            },
            {
                "name": "ca",
                "configMap": {
                    "name": CA_CONFIGMAP,
                    "defaultMode": 0o444,
                    "items": [{"key": CA_KEY, "path": CA_KEY}],
                },
            },
            {"name": "scratch", "emptyDir": {"sizeLimit": "16Mi"}},
            {
                "name": "api-access",
                "projected": {
                    "defaultMode": 0o440,
                    "sources": [
                        {
                            "serviceAccountToken": {
                                "path": "token",
                                "expirationSeconds": 600,
                            }
                        },
                        {
                            "configMap": {
                                "name": "kube-root-ca.crt",
                                "items": [{"key": "ca.crt", "path": "ca.crt"}],
                            }
                        },
                        {
                            "downwardAPI": {
                                "items": [
                                    {
                                        "path": "namespace",
                                        "fieldRef": {
                                            "apiVersion": "v1",
                                            "fieldPath": "metadata.namespace",
                                        },
                                    }
                                ]
                            }
                        },
                    ],
                },
            },
        ],
    }
    return pod


def supporting_manifests(
    plan: Plan,
    runtime: CpuRuntime,
    *,
    name: str,
    control_uid: str,
    source_data: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if not _text(control_uid):
        raise RegionalFixtureError("CPU watchdog control-map identity is incomplete")
    code_data = watchdog_source(plan, source_data=source_data)
    metadata = {
        **control_manifest(plan, runtime, name)["metadata"],
        "ownerReferences": [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "name": name,
                "uid": control_uid,
                "controller": False,
                "blockOwnerDeletion": False,
            }
        ],
    }
    account = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": metadata,
        "automountServiceAccountToken": False,
    }
    role = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": metadata,
        "rules": [
            {
                "apiGroups": [""],
                "resources": ["configmaps"],
                "resourceNames": [name],
                "verbs": ["get", "patch"],
            }
        ],
    }
    binding = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": metadata,
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "Role",
            "name": name,
        },
        "subjects": [
            {"kind": "ServiceAccount", "name": name, "namespace": runtime.namespace}
        ],
    }
    code = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {**metadata, "name": name + "-code"},
        "immutable": True,
        "data": code_data,
    }
    pod = _watchdog_pod_spec(plan, runtime, name=name, control_uid=control_uid)
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": metadata,
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": plan.deadline_at
            - plan.created_at
            + DRAIN_SECONDS
            + 180,
            "ttlSecondsAfterFinished": 600,
            "template": {
                "metadata": {
                    key: copy.deepcopy(metadata[key])
                    for key in ("labels", "annotations")
                },
                "spec": pod,
            },
        },
    }
    return [copy.deepcopy(document) for document in (account, role, binding, code, job)]


def _owners(value: Any) -> list[dict[str, Any]]:
    result = copy.deepcopy(_items(value))
    for owner in result:
        _defaults(owner, {"controller": False, "blockOwnerDeletion": False})
    return result


def _metadata(actual: dict[str, Any], expected: dict[str, Any], *, uid: str) -> None:
    metadata = _object(actual.get("metadata"))
    wanted = _object(expected.get("metadata"))
    if (
        not _text(uid)
        or not _text(wanted.get("name"))
        or not _text(wanted.get("namespace"))
        or actual.get("apiVersion") != expected.get("apiVersion")
        or actual.get("kind") != expected.get("kind")
        or metadata.get("uid") != uid
        or metadata.get("name") != wanted.get("name")
        or metadata.get("namespace") != wanted.get("namespace")
        or not _text(metadata.get("resourceVersion"))
        or metadata.get("deletionTimestamp") is not None
        or not _same(
            _owners(metadata.get("ownerReferences", [])),
            _owners(wanted.get("ownerReferences", [])),
        )
    ):
        raise RegionalFixtureError("CPU watchdog admitted resource identity differs")
    for field in ("labels", "annotations"):
        observed = _object(metadata.get(field, {}))
        required = _object(wanted.get(field, {}))
        if any(observed.get(key) != value for key, value in required.items()):
            raise RegionalFixtureError("CPU watchdog admitted source binding differs")


def validate_supporting_resource(
    actual: dict[str, Any], expected: dict[str, Any], *, uid: str
) -> None:
    """Validate a read-back SA, Role, RoleBinding or code/initial control ConfigMap.

    ``expected`` must be a trusted builder output, never an admission response.
    Mutable control documents need the protocol validator after initialization.
    """
    actual, expected = _object(actual), _object(expected)
    _metadata(actual, expected, uid=uid)
    kind = expected.get("kind")
    if kind not in {"ServiceAccount", "Role", "RoleBinding", "ConfigMap"}:
        raise RegionalFixtureError(
            "CPU watchdog supporting resource kind is unsupported"
        )
    bodies = []
    for document in (actual, expected):
        body = copy.deepcopy(
            {key: value for key, value in document.items() if key != "metadata"}
        )
        if kind == "ServiceAccount":
            _defaults(body, {"secrets": [], "imagePullSecrets": []})
            if not _same(
                _object(document["metadata"]).get("annotations", {}),
                _object(expected["metadata"]).get("annotations", {}),
            ):
                raise RegionalFixtureError(
                    "CPU watchdog service account annotations differ"
                )
        elif kind == "RoleBinding":
            for subject in _items(body.get("subjects")):
                _defaults(subject, {"apiGroup": ""})
        elif kind == "ConfigMap":
            _defaults(body, {"binaryData": {}, "immutable": False})
        bodies.append(body)
    if not _same(*bodies):
        raise RegionalFixtureError("CPU watchdog admitted supporting resource differs")


def _normal_pod(spec: Any) -> dict[str, Any]:
    pod = copy.deepcopy(_object(spec))
    containers = _items(pod.get("containers"))
    if (
        len(containers) != 1
        or containers[0].get("name") != "cancellation-watchdog"
        or not _image(containers[0].get("image"))
    ):
        raise RegionalFixtureError(
            "CPU watchdog requires one immutable runtime container"
        )
    if "serviceAccount" in pod:
        if pod["serviceAccount"] != pod.get("serviceAccountName"):
            raise RegionalFixtureError("CPU watchdog service account alias differs")
        del pod["serviceAccount"]
    _defaults(
        pod,
        {
            "dnsPolicy": "ClusterFirst",
            "schedulerName": "default-scheduler",
            "terminationGracePeriodSeconds": 30,
            "hostNetwork": False,
            "hostPID": False,
            "hostIPC": False,
            "priority": 0,
            "preemptionPolicy": "PreemptLowerPriority",
            "nodeName": "",
            "initContainers": [],
            "ephemeralContainers": [],
            "imagePullSecrets": [],
            "nodeSelector": {},
            "schedulingGates": [],
        },
    )
    for container in containers:
        _defaults(
            container,
            {
                "imagePullPolicy": "IfNotPresent",
                "terminationMessagePath": "/dev/termination-log",
                "terminationMessagePolicy": "File",
                "envFrom": [],
                "args": [],
                "workingDir": "",
                "stdin": False,
                "stdinOnce": False,
                "tty": False,
            },
        )
        for variable in _items(container.get("env", [])):
            if "valueFrom" in variable:
                _defaults(variable, {"value": ""})
                reference = _object(variable["valueFrom"])
                for key in ("secretKeyRef", "configMapKeyRef"):
                    if key in reference:
                        _defaults(_object(reference[key]), {"optional": False})
        for mount in _items(container.get("volumeMounts", [])):
            _defaults(mount, {"readOnly": False, "mountPropagation": "None"})
    for volume in _items(pod.get("volumes")):
        for key in ("secret", "configMap", "projected", "downwardAPI"):
            if key not in volume:
                continue
            projection = _object(volume[key])
            _defaults(projection, {"defaultMode": 0o644})
            if key in {"secret", "configMap"}:
                _defaults(projection, {"optional": False, "items": []})
            if key == "projected":
                for source in _items(projection.get("sources")):
                    for reference_kind in ("secret", "configMap"):
                        if reference_kind in source:
                            _defaults(
                                _object(source[reference_kind]), {"optional": False}
                            )
        if "emptyDir" in volume:
            _defaults(_object(volume["emptyDir"]), {"medium": ""})
    return pod


def _controller_labels(labels: dict[str, Any], *, uid: str, name: str) -> None:
    for key, value in (
        ("controller-uid", uid),
        ("batch.kubernetes.io/controller-uid", uid),
        ("job-name", name),
        ("batch.kubernetes.io/job-name", name),
    ):
        if key in labels:
            if labels[key] != value:
                raise RegionalFixtureError("CPU watchdog Job controller labels differ")
            del labels[key]


def _job_spec(document: dict[str, Any], *, uid: str, admitted: bool) -> dict[str, Any]:
    spec = copy.deepcopy(_object(document.get("spec")))
    name = _object(document.get("metadata"))["name"]
    if admitted:
        selector = _object(spec.pop("selector", None))
        _defaults(selector, {"matchExpressions": []})
        if not any(
            _same(selector, {"matchLabels": {key: uid}})
            for key in ("batch.kubernetes.io/controller-uid", "controller-uid")
        ):
            raise RegionalFixtureError("CPU watchdog Job selector differs")
    _defaults(
        spec,
        {
            "parallelism": 1,
            "completions": 1,
            "completionMode": "NonIndexed",
            "manualSelector": False,
            "suspend": False,
            "podReplacementPolicy": "TerminatingOrFailed",
            "managedBy": "kubernetes.io/job-controller",
        },
    )
    template = _object(spec.get("template"))
    metadata = _object(template.get("metadata"))
    _defaults(metadata, {"creationTimestamp": None})
    _controller_labels(_object(metadata.get("labels")), uid=uid, name=name)
    template["spec"] = _normal_pod(template.get("spec"))
    return spec


def _conditions(status: dict[str, Any]) -> dict[str, str]:
    conditions: dict[str, str] = {}
    for item in _items(status.get("conditions", [])):
        kind, value = item.get("type"), item.get("status")
        if (
            not _text(kind)
            or kind in conditions
            or value not in ("True", "False", "Unknown")
        ):
            raise RegionalFixtureError("CPU watchdog status conditions are ambiguous")
        conditions[kind] = value
    return conditions


def _phase(phase: WatchdogPhase | None) -> None:
    if phase is not None and phase not in ("gated", "running", "succeeded"):
        raise RegionalFixtureError("CPU watchdog validation phase is unsupported")


def validate_job(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    uid: str,
    phase: WatchdogPhase | None = None,
) -> None:
    """Check an actual Job against the planned Job, including exact control owner.

    None permits an initial, not-yet-counted Job, never an injection proof. Use
    ``gated`` before gate release, ``running`` before submission, and ``succeeded``
    for successful termination. A failed Job is rejected in every phase.
    """
    actual, expected = _object(actual), _object(expected)
    _phase(phase)
    _metadata(actual, expected, uid=uid)
    if expected.get("kind") != "Job" or not _same(
        _job_spec(actual, uid=uid, admitted=True),
        _job_spec(expected, uid=uid, admitted=False),
    ):
        raise RegionalFixtureError("CPU watchdog admitted Job spec differs")
    status = _object(actual.get("status", {}))
    counts = {
        key: status.get(key, 0)
        for key in ("active", "ready", "succeeded", "failed", "terminating")
    }
    conditions = _conditions(status)
    uncounted = _object(status.get("uncountedTerminatedPods", {}))
    if (
        any(type(value) is not int or not 0 <= value <= 1 for value in counts.values())
        or counts["failed"] != 0
        or counts["terminating"] != 0
        or counts["ready"] > counts["active"]
        or counts["active"] + counts["succeeded"] > 1
        or set(conditions)
        - {"Complete", "Failed", "FailureTarget", "Suspended", "SuccessCriteriaMet"}
        or "Unknown" in conditions.values()
        or any(
            conditions.get(key) == "True"
            for key in ("Failed", "FailureTarget", "Suspended")
        )
        or set(uncounted) - {"failed", "succeeded"}
        or any(not _same(value, []) for value in uncounted.values())
    ):
        raise RegionalFixtureError("CPU watchdog Job failed or has ambiguous counts")
    complete = conditions.get("Complete") == "True"
    if (complete or conditions.get("SuccessCriteriaMet") == "True") and (
        counts["succeeded"] != 1 or counts["active"] != 0
    ):
        raise RegionalFixtureError("CPU watchdog Job completion counts differ")
    if phase is not None:
        active, ready, succeeded = {
            "gated": (1, 0, 0),
            "running": (1, 1, 0),
            "succeeded": (0, 0, 1),
        }[phase]
        if (
            counts["active"] != active
            or ("ready" in status and counts["ready"] != ready)
            or counts["succeeded"] != succeeded
            or complete != (phase == "succeeded")
        ):
            raise RegionalFixtureError("CPU watchdog Job is not in the required phase")


def validate_job_for_cleanup(
    actual: dict[str, Any], expected: dict[str, Any], *, uid: str
) -> None:
    """Check ownership and the complete Job spec without granting execution authority."""
    actual = _cleanup_object(actual)
    expected = _object(expected)
    _metadata(actual, expected, uid=uid)
    if expected.get("kind") != "Job" or not _same(
        _job_spec(actual, uid=uid, admitted=True),
        _job_spec(expected, uid=uid, admitted=False),
    ):
        raise RegionalFixtureError("CPU watchdog cleanup Job spec differs")


def _cleanup_object(actual: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(_object(actual))
    metadata = _object(result.get("metadata"))
    deleting = metadata.pop("deletionTimestamp", None)
    if deleting is not None:
        _timestamp(deleting)
    return result


def _timestamp(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is not None:
            return parsed
    except (ValueError, TypeError):
        pass
    raise RegionalFixtureError("CPU watchdog process timestamp is incomplete")


def _process_status(
    status: dict[str, Any], *, image: str, phase: WatchdogPhase
) -> None:
    containers = _items(status.get("containerStatuses", []))
    if len(containers) != 1:
        raise RegionalFixtureError("CPU watchdog process status is incomplete")
    container = containers[0]
    container_id = container.get("containerID")
    if (
        container.get("name") != "cancellation-watchdog"
        or not _status_image(container.get("image"), image)
        or not _image_id(container.get("imageID"), image)
        or not _text(container_id)
        or not _same(container.get("restartCount"), 0)
        or not _same(container.get("lastState", {}), {})
        or container.get("ready") is not (phase == "running")
    ):
        raise RegionalFixtureError(
            "CPU watchdog process identity or restart state differs"
        )
    state = _object(container.get("state"))
    if phase == "running":
        if set(state) != {"running"} or container.get("started") is not True:
            raise RegionalFixtureError("CPU watchdog process is not running")
        _timestamp(_object(state["running"]).get("startedAt"))
    else:
        if set(state) != {"terminated"} or container.get("started", False) is not False:
            raise RegionalFixtureError("CPU watchdog process is not terminated")
        terminated = _object(state["terminated"])
        if (
            not _same(terminated.get("exitCode"), 0)
            or not _same(terminated.get("signal", 0), 0)
            or terminated.get("reason") != "Completed"
            or terminated.get("containerID") != container_id
            or _timestamp(terminated.get("finishedAt"))
            < _timestamp(terminated.get("startedAt"))
        ):
            raise RegionalFixtureError(
                "CPU watchdog process did not finish successfully"
            )


def validate_pod(
    actual: dict[str, Any],
    expected_job: dict[str, Any],
    *,
    job_uid: str,
    pod_name: str,
    pod_uid: str,
    phase: WatchdogPhase,
) -> None:
    """Validate the actual Pod, not just its Job template or identifying labels.

    The parent must first validate ``gated`` and persist this Pod's UID before
    removing the gate with UID/resourceVersion CAS. Later phases must use that
    same recorded UID; they do not authorize adopting an already-running Pod.
    """
    actual, expected_job = _object(actual), _object(expected_job)
    _phase(phase)
    if phase is None:
        raise RegionalFixtureError("CPU watchdog Pod binding is incomplete")
    template = _object(_object(expected_job.get("spec")).get("template"))
    expected = _pod_identity(expected_job, job_uid=job_uid, pod_name=pod_name)
    _metadata(actual, expected, uid=pod_uid)
    _controller_labels(
        copy.deepcopy(_object(actual["metadata"].get("labels", {}))),
        uid=job_uid,
        name=expected_job["metadata"]["name"],
    )
    observed = copy.deepcopy(_object(actual.get("spec")))
    planned = copy.deepcopy(_object(template.get("spec")))
    status = _object(actual.get("status"))
    conditions = _conditions(status)
    if (
        not _same(status.get("initContainerStatuses", []), [])
        or not _same(status.get("ephemeralContainerStatuses", []), [])
        or conditions.get("DisruptionTarget") == "True"
    ):
        raise RegionalFixtureError(
            "CPU watchdog Pod is disrupted or has extra processes"
        )
    if phase == "gated":
        if (
            observed.get("nodeName", "") != ""
            or not _same(observed.get("schedulingGates"), [{"name": SCHEDULING_GATE}])
            or status.get("phase") != "Pending"
            or not _same(status.get("containerStatuses", []), [])
            or status.get("startTime") is not None
            or status.get("podIP", "") != ""
            or not _same(status.get("podIPs", []), [])
            or any(
                conditions.get(key) == "True"
                for key in ("Ready", "ContainersReady", "PodScheduled")
            )
        ):
            raise RegionalFixtureError(
                "CPU watchdog Pod was not held unstarted by its gate"
            )
    else:
        if (
            not _text(observed.pop("nodeName", None))
            or not _same(observed.get("schedulingGates", []), [])
            or status.get("phase") != ("Running" if phase == "running" else "Succeeded")
            or conditions.get("PodScheduled") != "True"
            or conditions.get("Initialized") != "True"
            or any(
                conditions.get(key) != ("True" if phase == "running" else "False")
                for key in ("Ready", "ContainersReady")
            )
        ):
            raise RegionalFixtureError("CPU watchdog Pod is not in the required phase")
        del planned["schedulingGates"]
    if not _same(_normal_pod(observed), _normal_pod(planned)):
        raise RegionalFixtureError("CPU watchdog admitted Pod spec differs")
    if phase != "gated":
        _process_status(status, image=planned["containers"][0]["image"], phase=phase)


def _pod_identity(
    expected_job: dict[str, Any], *, job_uid: str, pod_name: str
) -> dict[str, Any]:
    if not _text(job_uid) or not _text(pod_name) or expected_job.get("kind") != "Job":
        raise RegionalFixtureError("CPU watchdog Pod binding is incomplete")
    job_metadata = _object(expected_job.get("metadata"))
    template = _object(_object(expected_job.get("spec")).get("template"))
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            **copy.deepcopy(_object(template.get("metadata"))),
            "name": pod_name,
            "namespace": job_metadata["namespace"],
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": job_metadata["name"],
                    "uid": job_uid,
                    "controller": True,
                    "blockOwnerDeletion": True,
                }
            ],
        },
    }


def validate_pod_for_cleanup(
    actual: dict[str, Any],
    expected_job: dict[str, Any],
    *,
    job_uid: str,
    pod_name: str,
    pod_uid: str,
    release_requested: bool,
) -> None:
    """Validate only a recorded Pod's ownership/spec for stopping, never for injection."""
    actual = _cleanup_object(actual)
    expected_job = _object(expected_job)
    expected = _pod_identity(expected_job, job_uid=job_uid, pod_name=pod_name)
    _metadata(actual, expected, uid=pod_uid)
    observed = copy.deepcopy(_object(actual.get("spec")))
    planned = copy.deepcopy(expected_job["spec"]["template"]["spec"])
    if release_requested and _same(observed.get("schedulingGates", []), []):
        node = observed.pop("nodeName", "")
        if node != "" and not _text(node):
            raise RegionalFixtureError("CPU watchdog cleanup Pod node is malformed")
        del planned["schedulingGates"]
    if not _same(_normal_pod(observed), _normal_pod(planned)):
        raise RegionalFixtureError("CPU watchdog cleanup Pod spec differs")
