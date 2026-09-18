from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, cast
from urllib.parse import quote
import uuid

yaml = importlib.import_module("yaml")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.regional_commands import (  # noqa: E402
    RegionalCommandFailed,
    RegionalFixtureError,
)
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture  # noqa: E402
from scripts.e2e.regional.regional_pod_inventory import ready_pod_records  # noqa: E402
from scripts.e2e.regional.fixture_ownership import (  # noqa: E402
    CleanupCustody,
    FixtureOwnership,
    capture_cleanup_custody,
    creation_document,
    file_digest,
    fixture_binding,
    verify_cleanup_custody,
)
from scripts.e2e.regional.workload_restart_custody import (  # noqa: E402
    RestartCustody,
    WorkloadOwnershipError,
)


TRAINING_IMAGE = (
    "public.ecr.aws/deep-learning-containers/"
    "pytorch-training@sha256:"
    "ff2c928a2e7b3b290b7c3e353085a373b3cfc7f6b165e338b9be4df439feae38"
)
# The training fixtures print `HEARTBEAT ... all_reduce=<value>` every step and
# one `SUCCESS rank=<r>/<world>` line at the end; the loss lines
# (`rank=<r> step=<s> loss=<l>`) are what a per-rank loss check reads.
HEARTBEAT_LOG_MARKERS = ("HEARTBEAT", "SUCCESS", "loss=")
ALL_REDUCE_PATTERN = re.compile(r"all_reduce=([-+0-9.eE]+)")
SUCCESS_RANK_PATTERN = re.compile(r"SUCCESS rank=(\d+)/(\d+)")
OWNER_LABEL = "gpu-fault.io/acceptance-owner"
RESOURCE_APIS = {
    "pod": ("api/v1", "pods"),
    "job": ("apis/batch/v1", "jobs"),
    "pytorchjob": ("apis/kubeflow.org/v1", "pytorchjobs"),
    "jobset": ("apis/jobset.x-k8s.io/v1alpha2", "jobsets"),
}


def resource_identity(
    regional: RegionalLiveFixture, document: dict[str, Any]
) -> tuple[str, str, str]:
    metadata = document.get("metadata") or {}
    kind = str(document.get("kind") or "").lower()
    name, uid = metadata.get("name"), metadata.get("uid")
    if (
        kind not in RESOURCE_APIS
        or metadata.get("namespace") != regional.settings.namespace
        or not isinstance(name, str)
        or not name
        or not isinstance(uid, str)
        or not uid
        or not isinstance(metadata.get("resourceVersion"), str)
        or not metadata["resourceVersion"]
    ):
        raise RegionalFixtureError("fixture resource identity is incomplete or foreign")
    return kind, name, uid


def read_resource(
    regional: RegionalLiveFixture, kind: str, name: str
) -> dict[str, Any] | None:
    raw = regional.kubectl(
        "gpu", "get", kind, name, "--ignore-not-found", "-o", "json", timeout=60
    )
    if not raw.strip():
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RegionalFixtureError("fixture resource query is not an object")
    actual_kind, actual_name, _uid = resource_identity(regional, value)
    if (actual_kind, actual_name) != (kind, name):
        raise RegionalFixtureError("fixture resource query returned another identity")
    return value


def delete_resource(
    regional: RegionalLiveFixture,
    document: dict[str, Any],
    *,
    propagation_policy: str = "Foreground",
) -> None:
    """Delete exactly the observed incarnation, including a lost DELETE receipt."""

    if propagation_policy not in {"Foreground", "Orphan"}:
        raise RegionalFixtureError("unsupported fixture deletion propagation policy")
    kind, name, uid = resource_identity(regional, document)
    api, plural = RESOURCE_APIS[kind]
    namespace = quote(regional.settings.namespace, safe="")
    options: dict[str, Any] = {
        "apiVersion": "v1",
        "kind": "DeleteOptions",
        "preconditions": {
            "uid": uid,
            "resourceVersion": document["metadata"]["resourceVersion"],
        },
        "propagationPolicy": propagation_policy,
    }
    if kind == "pod":
        options["gracePeriodSeconds"] = 0
    try:
        regional.kubectl(
            "gpu",
            "delete",
            "--raw",
            f"/{api}/namespaces/{namespace}/{plural}/{quote(name, safe='')}",
            "-f",
            "-",
            input_text=json.dumps(options),
            timeout=120,
        )
    except Exception:
        if read_resource(regional, kind, name) is not None:
            raise
        return
    deadline = time.monotonic() + 120
    while True:
        current = read_resource(regional, kind, name)
        if current is None:
            return
        if resource_identity(regional, current)[2] != uid:
            raise RegionalFixtureError("fixture resource was replaced during deletion")
        if time.monotonic() >= deadline:
            raise RegionalFixtureError("fixture resource deletion was not confirmed")
        time.sleep(1)


def expected_all_reduce(world_size: int) -> float:
    """The all-reduce every fixture rank checks: sum of (rank + 1) over ranks."""

    if world_size < 1:
        raise ValueError("world size must be positive")
    return world_size * (world_size + 1) / 2


def heartbeat_healthy(log_text: str, *, world_size: int) -> bool:
    """Whether ``log_text`` proves the collective ran at ``world_size``.

    A substring test on ``all_reduce=`` passed for any Pod that had printed
    one heartbeat, whatever the value -- a job that came up at the wrong world
    size prints the same prefix. The value has to parse and equal the sum the
    fixture asserts, or a SUCCESS line has to name the same world size.
    """

    expected = expected_all_reduce(world_size)
    for match in ALL_REDUCE_PATTERN.finditer(log_text):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if value == expected:
            return True
    return any(
        int(match.group(2)) == world_size
        for match in SUCCESS_RANK_PATTERN.finditer(log_text)
    )


def render_node_pinned_manifest(
    source: Path,
    destination: Path,
    *,
    node: str,
) -> Path:
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("node-pinned workload manifest is not a mapping")
    kind = str(document.get("kind") or "").lower()
    spec = document.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("node-pinned workload manifest has no spec")
    templates: list[dict[str, Any]] = []
    if kind == "job":
        template = spec.get("template")
        if isinstance(template, dict):
            templates.append(template)
    elif kind == "pytorchjob":
        replicas = spec.get("pytorchReplicaSpecs")
        if isinstance(replicas, dict):
            templates.extend(
                replica["template"]
                for replica in replicas.values()
                if isinstance(replica, dict)
                and isinstance(replica.get("template"), dict)
            )
    else:
        raise ValueError("node-pinned fixture supports Job or PyTorchJob")
    if not templates:
        raise ValueError("node-pinned workload has no Pod templates")
    for template in templates:
        pod_spec = template.get("spec")
        if not isinstance(pod_spec, dict):
            raise ValueError("node-pinned workload Pod template has no spec")
        pod_spec["nodeName"] = node
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    destination.chmod(0o600)
    return destination


@dataclass(frozen=True)
class ManagedWorkloadSettings:
    manifest: Path
    site_file: Path
    job_id: str
    attempt_id: str
    restart_budget: int
    expected_pods: int
    expected_gpu_count: int

    def __post_init__(self) -> None:
        if not self.manifest.is_file():
            raise ValueError("managed workload manifest does not exist")
        if not self.site_file.is_file():
            raise ValueError("regional site file does not exist")
        if not self.job_id or not self.attempt_id:
            raise ValueError("managed workload identity is empty")
        if self.restart_budget < 0:
            raise ValueError("restart budget cannot be negative")
        if self.expected_pods < 1 or self.expected_gpu_count < 1:
            raise ValueError("managed workload expectations must be positive")


class ImagePrewarmFixture:
    def __init__(
        self,
        regional: RegionalLiveFixture,
        *,
        case_id: str,
        run_id: str,
        image: str = TRAINING_IMAGE,
        state_path: Path | None = None,
    ) -> None:
        self.regional = regional
        self.image = image
        suffix = hashlib.sha256(f"{case_id}\0{run_id}\0{image}".encode()).hexdigest()[
            :10
        ]
        self.prefix = f"gpu-fault-image-prewarm-{suffix}"
        self.pods: list[str] = []
        self.owner = uuid.uuid4().hex
        self.created: dict[str, str | None] = {}
        self.expected: dict[str, dict[str, Any]] = {}
        self.ownership: FixtureOwnership | None = None
        if state_path is not None:

            def binding() -> dict[str, Any]:
                return fixture_binding(
                    regional,
                    purpose="image-prewarm",
                    inputs={
                        "case_id": case_id,
                        "run_id": run_id,
                        "image": image,
                        "fixture_sha256": file_digest(Path(__file__)),
                    },
                )

            self.ownership = FixtureOwnership(
                state_path, binding(), current_binding=binding
            )
            self.owner = self.ownership.record.owner
            for creation in self.ownership.record.creations.values():
                name = str(creation.expected["metadata"]["name"])
                self.expected[name] = creation.expected
                self.pods.append(name)
                self.created[name] = creation.uid

    def manifest(self, node: str, index: int) -> dict[str, Any]:
        name = f"{self.prefix}-{index}"
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": name,
                "namespace": self.regional.settings.namespace,
                "labels": {
                    "app": "gpu-fault-image-prewarm",
                    "gpu-fault.io/acceptance-run": self.prefix,
                    OWNER_LABEL: self.owner,
                },
            },
            "spec": {
                "nodeName": node,
                "restartPolicy": "Never",
                "activeDeadlineSeconds": 1800,
                "terminationGracePeriodSeconds": 0,
                "tolerations": [{"operator": "Exists"}],
                "containers": [
                    {
                        "name": "prewarm",
                        "image": self.image,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["/bin/bash", "-ceu", "echo cached"],
                        "resources": {
                            "requests": {"cpu": "10m", "memory": "32Mi"},
                            "limits": {"cpu": "100m", "memory": "128Mi"},
                        },
                    }
                ],
            },
        }

    def create(self, nodes: list[str]) -> dict[str, list[str]]:
        """Pull the image onto ``nodes`` that do not already hold it.

        A node that reports the digest in ``status.images`` needs no Pod; the
        pull was the whole point, and the prewarm Pod's schedule/pull/exit
        cycle is the slowest step of the case when it runs on every candidate.
        Returns which nodes were skipped and which received a Pod.
        """

        if self.ownership is not None:
            self.ownership.begin()
        if self.pods:
            raise RegionalFixtureError("image prewarm fixture is already created")
        if (
            not nodes
            or len(set(nodes)) != len(nodes)
            or any(not node for node in nodes)
        ):
            raise RegionalFixtureError(
                "image prewarm requires a nonempty unique node set"
            )
        cached = set(self.cached_nodes())
        skipped = [node for node in nodes if node in cached]
        pending = [node for node in nodes if node not in cached]
        for index, node in enumerate(pending):
            manifest = self.manifest(node, index)
            name = manifest["metadata"]["name"]
            if read_resource(self.regional, "pod", name) is not None:
                raise RegionalFixtureError("image prewarm name is already occupied")
            self.expected[name] = manifest
            self.pods.append(name)
            # Register intent before create: a lost ACK must still be cleaned up.
            self.created[name] = None
            if self.ownership is not None:
                self.ownership.intend(manifest)
            output = self.regional.kubectl(
                "gpu",
                "create",
                "-f",
                "-",
                *(("-o", "json") if self.ownership is not None else ()),
                input_text=json.dumps(manifest),
            )
            if self.ownership is not None:
                self.created[name] = self.ownership.acknowledge(
                    creation_document(output)
                )
            current = read_resource(self.regional, "pod", name)
            if current is None:
                raise RegionalFixtureError("created image prewarm Pod is missing")
            self.created[name] = self.owned_pod(current)
        for pod in self.pods:
            self.regional.kubectl(
                "gpu",
                "wait",
                "--for=jsonpath={.status.phase}=Succeeded",
                f"pod/{pod}",
                "--timeout=1800s",
                timeout=1830,
            )
        return {"skipped": skipped, "created": pending}

    def cached_nodes(self) -> list[str]:
        value = json.loads(self.regional.kubectl("gpu", "get", "node", "-o", "json"))
        if not isinstance(value, dict) or not isinstance(value.get("items"), list):
            raise RegionalFixtureError("image prewarm node inventory is incomplete")
        digest = self.image.rsplit("@", 1)[-1]
        return sorted(
            item["metadata"]["name"]
            for item in value.get("items", [])
            if any(
                str(name).rsplit("@", 1)[-1] == digest
                for image in item.get("status", {}).get("images", [])
                for name in image.get("names", [])
            )
        )

    def owned_pod(self, document: dict[str, Any]) -> str:
        _kind, name, uid = resource_identity(self.regional, document)
        expected = self.expected.get(name) or {}
        spec = document.get("spec") or {}
        containers = spec.get("containers") or []
        intended = (expected.get("spec") or {}).get("containers") or []
        if (
            (document["metadata"].get("labels") or {}).get(OWNER_LABEL) != self.owner
            or spec.get("nodeName") != (expected.get("spec") or {}).get("nodeName")
            or len(containers) != len(intended)
            or any(
                any(
                    actual.get(key) != wanted.get(key)
                    for key in ("name", "image", "command")
                )
                for actual, wanted in zip(containers, intended, strict=True)
            )
            or self.created.get(name) not in (None, uid)
        ):
            raise RegionalFixtureError("image prewarm Pod ownership changed")
        if self.ownership is not None:
            self.ownership.observe(
                document, lambda kind, name: read_resource(self.regional, kind, name)
            )
        return uid

    def cleanup(self) -> dict[str, bool]:
        if self.ownership is not None:
            self.ownership.require_acknowledged()
            documents = self.ownership.rejected_resources(
                lambda kind, name: read_resource(self.regional, kind, name)
            )
            by_name = {item["metadata"]["name"]: item for item in documents}
            for name in self.created:
                current = read_resource(self.regional, "pod", name)
                if current is None:
                    continue
                if not self.ownership.deletion_only(current):
                    self.owned_pod(current)
                by_name[name] = current
            for current in by_name.values():
                delete_resource(self.regional, current)
            self.ownership.complete()
            return {name: False for name in self.created}
        residuals = {}
        errors = []
        for name in self.created:
            try:
                current = read_resource(self.regional, "pod", name)
                if current is not None:
                    self.created[name] = self.owned_pod(current)
                    delete_resource(self.regional, current)
                residuals[name] = False
            except Exception:
                residuals[name] = True
                errors.append(name)
        if errors:
            raise RegionalFixtureError(
                "image prewarm cleanup is unproven: " + ", ".join(errors)
            )
        return residuals


class ManagedWorkloadFixture:
    def __init__(
        self,
        regional: RegionalLiveFixture,
        settings: ManagedWorkloadSettings,
        *,
        state_path: Path | None = None,
    ) -> None:
        self.regional = regional
        self.settings = settings
        document = yaml.safe_load(settings.manifest.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("managed workload manifest is not a mapping")
        self.kind = str(document.get("kind") or "")
        self.name = str((document.get("metadata") or {}).get("name") or "")
        if (
            self.kind.lower() not in RESOURCE_APIS
            or self.kind.lower() == "pod"
            or not self.name
        ):
            raise ValueError("managed workload manifest has no kind/name")
        namespace = (document.get("metadata") or {}).get("namespace")
        if namespace is not None and namespace != regional.settings.namespace:
            raise ValueError("workload namespace contradicts the shared site namespace")
        self.owner = uuid.uuid4().hex
        self.submission_started = False
        self.owned_uids: dict[tuple[str, str], str] = {}
        self.restart_custody: RestartCustody | None = None
        self.retired_pod_uids: set[str] = set()
        self.cleanup_snapshots: dict[tuple[str, str], CleanupCustody] = {}
        self.ownership: FixtureOwnership | None = None
        if state_path is not None:

            def binding() -> dict[str, Any]:
                return fixture_binding(
                    regional,
                    purpose="managed-workload",
                    inputs={
                        "manifest_sha256": file_digest(settings.manifest),
                        "site_sha256": file_digest(settings.site_file),
                        "job_id": settings.job_id,
                        "attempt_id": settings.attempt_id,
                        "restart_budget": settings.restart_budget,
                        "expected_pods": settings.expected_pods,
                        "expected_gpu_count": settings.expected_gpu_count,
                        "fixture_sha256": file_digest(Path(__file__)),
                    },
                )

            self.ownership = FixtureOwnership(
                state_path, binding(), current_binding=binding
            )
            self.owner = self.ownership.record.owner
            self.submission_started = self.ownership.record.started
            self.owned_uids = {
                (key.split("/", 1)[0], key.split("/", 1)[1]): uid
                for key, uid in self.ownership.record.observed.items()
            }

    def authorize_restart(self, state: dict[str, Any]) -> None:
        """Accept replacements only after the matching product restart completed."""

        if not self.submission_started or self.ownership is not None:
            raise WorkloadOwnershipError(
                "restart custody requires a submitted, non-resuming workload fixture"
            )
        custody = RestartCustody.from_state(
            state,
            cluster_id=self.regional.settings.cluster_id,
            namespace=self.regional.settings.namespace,
            job_id=self.settings.job_id,
            source_attempt_id=self.settings.attempt_id,
            source_workload_id=f"{self.regional.settings.namespace}/{self.resource}/{self.name}",
            source_uid=self.owned_uids.get((self.resource, self.name), ""),
            gpu_count=self.settings.expected_gpu_count,
            restart_budget=self.settings.restart_budget,
        )
        if self.restart_custody is not None and self.restart_custody != custody:
            raise WorkloadOwnershipError(
                "managed workload restart authorization changed"
            )
        if self.restart_custody is None:
            # Keep the first, equal custody: it carries the adoption receipts.
            self.restart_custody = custody

    @property
    def resource(self) -> str:
        return self.kind.lower()

    def deletion_only(self, document: dict[str, Any]) -> bool:
        if self.ownership is not None:
            return self.ownership.deletion_only(document)
        kind, name, _uid = resource_identity(self.regional, document)
        custody = self.cleanup_snapshots.get((kind, name))
        if custody is None:
            return False
        verify_cleanup_custody(
            custody,
            document,
            observed_uid=self.owned_uids.get((kind, name)),
            namespace=self.regional.settings.namespace,
        )
        return True

    def delete(self) -> None:
        if self.ownership is not None:
            self.ownership.require_acknowledged()
        if not self.submission_started:
            return
        if self.ownership is not None:
            # A rejected controller ACK grants rollback, not descendant adoption.
            for rejected in self.ownership.rejected_resources(
                lambda kind, name: read_resource(self.regional, kind, name)
            ):
                delete_resource(self.regional, rejected)
        documents = self.inventory()
        source = read_resource(self.regional, self.resource, self.name)
        if source is not None and source not in documents:
            documents.append(source)
        # Validate the entire set before the first delete. Recovery copies retain
        # our nonce in top-level and Pod-template labels; a job-id is not ownership.
        for document in documents:
            if self.deletion_only(document):
                kind, name, uid = resource_identity(self.regional, document)
                self.owned_uids[(kind, name)] = uid
            else:
                self.adopt(document)
        for (kind, name), uid in self.owned_uids.copy().items():
            current = read_resource(self.regional, kind, name)
            if current is not None:
                if not self.deletion_only(current):
                    self.adopt(current)
                if resource_identity(self.regional, current)[2] != uid:
                    raise RegionalFixtureError("managed workload UID changed")
                if current not in documents:
                    documents.append(current)
        documents = list(
            {
                resource_identity(self.regional, item)[:2]: item for item in documents
            }.values()
        )
        if self.ownership is not None:
            for document in documents:
                if not self.ownership.deletion_only(document):
                    self.ownership.retain_for_cleanup(document)
        else:
            for document in documents:
                kind, name, _uid = resource_identity(self.regional, document)
                self.cleanup_snapshots.setdefault(
                    (kind, name), capture_cleanup_custody(document)
                )
        # Orphan controllers first so they cannot replace Pods during cleanup.
        # Their Pods are then deleted explicitly with zero grace.
        priority = {"jobset": 0, "pytorchjob": 0, "job": 1, "pod": 2}
        for document in sorted(
            documents,
            key=lambda item: priority[resource_identity(self.regional, item)[0]],
        ):
            self._delete_owned_resource(document)
        # An already-dispatched controller request can create a final child.
        remaining = self.inventory()
        for document in remaining:
            self.adopt(document)
        for document in sorted(
            remaining,
            key=lambda item: priority[resource_identity(self.regional, item)[0]],
        ):
            self._delete_owned_resource(document)
        if self.inventory():
            raise RegionalFixtureError(
                "managed workload resources remain after cleanup"
            )
        if self.ownership is not None:
            self.ownership.complete()

    def _delete_owned_resource(self, document: dict[str, Any]) -> None:
        kind, name, uid = resource_identity(self.regional, document)
        for attempt in range(3):
            current = read_resource(self.regional, kind, name)
            if current is None:
                return
            if resource_identity(self.regional, current)[2] != uid:
                raise RegionalFixtureError(
                    "managed workload UID changed before deletion"
                )
            if not self.deletion_only(current):
                self.adopt(current)
            try:
                delete_resource(
                    self.regional,
                    current,
                    propagation_policy="Foreground" if kind == "pod" else "Orphan",
                )
                return
            except RegionalCommandFailed:
                latest = read_resource(self.regional, kind, name)
                if latest is None:
                    return
                if resource_identity(self.regional, latest)[2] != uid:
                    raise RegionalFixtureError(
                        "managed workload UID changed during deletion"
                    )
                if not self.deletion_only(latest):
                    self.adopt(latest)
                if (
                    attempt == 2
                    or latest["metadata"]["resourceVersion"]
                    == current["metadata"]["resourceVersion"]
                ):
                    raise

    def inventory(self) -> list[dict[str, Any]]:
        resources = f"pod,{self.resource}" + (
            ",job" if self.resource == "jobset" else ""
        )
        value = json.loads(
            self.regional.kubectl(
                "gpu",
                "get",
                resources,
                "-l",
                f"gpu-fault.io/job-id={self.settings.job_id}",
                "-o",
                "json",
            )
        )
        if not isinstance(value, dict) or not isinstance(value.get("items"), list):
            raise RegionalFixtureError("managed workload inventory is incomplete")
        if any(not isinstance(item, dict) for item in value["items"]):
            raise RegionalFixtureError(
                "managed workload inventory contains an invalid object"
            )
        return cast(list[dict[str, Any]], value["items"])

    def adopt(self, document: dict[str, Any]) -> None:
        kind, name, uid = resource_identity(self.regional, document)
        labels = document["metadata"].get("labels") or {}
        if (
            not self.submission_started
            or labels.get(OWNER_LABEL) != self.owner
            or labels.get("gpu-fault.io/job-id") != self.settings.job_id
            or not labels.get("gpu-fault.io/attempt-id")
        ):
            raise WorkloadOwnershipError("managed workload ownership or UID changed")
        previous_uid = self.owned_uids.get((kind, name))
        if uid in self.retired_pod_uids:
            raise WorkloadOwnershipError(
                "a retired managed workload Pod UID reappeared"
            )
        if self.restart_custody is None:
            if (
                previous_uid not in (None, uid)
                or labels["gpu-fault.io/attempt-id"] != self.settings.attempt_id
            ):
                raise WorkloadOwnershipError(
                    "managed workload ownership or UID changed"
                )
        elif not (
            previous_uid == uid
            and labels["gpu-fault.io/attempt-id"]
            == self.restart_custody.source_attempt_id
        ):
            root_kind, root_name, root_uid = self.restart_custody.validate(
                document,
                owner=self.owner,
                known_uids=self.owned_uids,
                read=lambda kind, name: read_resource(self.regional, kind, name),
            )
            if previous_uid is not None and previous_uid != uid:
                if kind != "pod":
                    raise WorkloadOwnershipError(
                        "managed workload controller UID changed"
                    )
                self.retired_pod_uids.add(previous_uid)
            self.owned_uids[(root_kind, root_name)] = root_uid
        if self.ownership is not None:
            self.ownership.observe(
                document, lambda kind, name: read_resource(self.regional, kind, name)
            )
        self.owned_uids[(kind, name)] = uid

    def submit_rendered(self, rendered: str) -> dict[str, Any]:
        if self.ownership is not None and self.ownership.resuming:
            raise RegionalFixtureError("an existing managed workload is cleanup-only")
        if self.submission_started:
            raise RegionalFixtureError("managed workload submission already started")
        if (
            read_resource(self.regional, self.resource, self.name) is not None
            or self.inventory()
        ):
            raise RegionalFixtureError(
                "managed workload name or job identity already exists"
            )
        document = yaml.safe_load(rendered)
        metadata = document.get("metadata") if isinstance(document, dict) else None
        if (
            not isinstance(metadata, dict)
            or document.get("kind") != self.kind
            or metadata.get("name") != self.name
            or metadata.get("namespace") != self.regional.settings.namespace
            or (metadata.get("labels") or {}).get("gpu-fault.io/job-id")
            != self.settings.job_id
            or (metadata.get("labels") or {}).get("gpu-fault.io/attempt-id")
            != self.settings.attempt_id
        ):
            raise RegionalFixtureError(
                "rendered workload identity differs from the fixture"
            )

        def mark(value: Any) -> None:
            if isinstance(value, dict):
                if isinstance(value.get("metadata"), dict):
                    value["metadata"].setdefault("labels", {})[OWNER_LABEL] = self.owner
                    if value["metadata"].get("annotations") == {}:
                        value["metadata"].pop("annotations")
                for child in value.values():
                    mark(child)
            elif isinstance(value, list):
                for child in value:
                    mark(child)

        mark(document)
        if self.ownership is not None:
            self.ownership.begin()
            self.ownership.intend(document)
        self.submission_started = True
        output = self.regional.kubectl(
            "gpu",
            "create",
            "-f",
            "-",
            *(("-o", "json") if self.ownership is not None else ()),
            input_text=json.dumps(document),
            timeout=300,
        )
        if self.ownership is not None:
            uid = self.ownership.acknowledge(creation_document(output))
            self.owned_uids[(self.resource, self.name)] = uid
        current = read_resource(self.regional, self.resource, self.name)
        if current is None:
            raise RegionalFixtureError("created managed workload is missing")
        self.adopt(current)
        return {
            "stdout": output if self.ownership is None else "created",
            "stderr": "",
            "create_only": True,
        }

    def submit(self) -> dict[str, Any]:
        from gpu_fault import training_submit_cli

        arguments = [
            str(self.settings.manifest),
            "--site",
            str(self.settings.site_file),
            "--job-id",
            self.settings.job_id,
            "--attempt-id",
            self.settings.attempt_id,
            "--restart-budget",
            str(self.settings.restart_budget),
            "--kubeconfig",
            str(self.regional.settings.gpu_kubeconfig),
            "--context",
            self.regional.settings.gpu_context,
            "--namespace",
            self.regional.settings.namespace,
        ]
        submission: dict[str, Any] = {}

        def create_only(
            command: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            submission.update(self.submit_rendered(kwargs["input"]))
            return subprocess.CompletedProcess(command, 0, submission["stdout"], "")

        training_submit_cli.run(
            training_submit_cli.parser().parse_args(arguments), runner=create_only
        )
        return {"entrypoint": "gpu_fault.training_submit_cli.run", **submission}

    def annotate_auto_resume(self, value: str | None) -> None:
        current = self.workload()
        self.adopt(current)
        annotation = "sagemaker.amazonaws.com/enable-job-auto-resume"
        annotations = dict(current["metadata"].get("annotations") or {})
        if value is None:
            annotations.pop(annotation, None)
        else:
            annotations[annotation] = value
        self.regional.kubectl(
            "gpu",
            "patch",
            str(current["kind"]).lower(),
            str(current["metadata"]["name"]),
            "--type=json",
            "--patch-file=/dev/stdin",
            input_text=json.dumps(
                [
                    {
                        "op": "test",
                        "path": "/metadata/uid",
                        "value": current["metadata"]["uid"],
                    },
                    {
                        "op": "test",
                        "path": "/metadata/resourceVersion",
                        "value": current["metadata"]["resourceVersion"],
                    },
                    {
                        "op": "add",
                        "path": "/metadata/annotations",
                        "value": annotations,
                    },
                ]
            ),
        )

    def workload(self) -> dict[str, Any]:
        kind, name = self.resource, self.name
        if self.restart_custody is not None:
            _namespace, kind, name = next(
                iter(self.restart_custody.target_workload_ids)
            ).split("/")
        value = json.loads(
            self.regional.kubectl(
                "gpu",
                "get",
                kind,
                name,
                "-o",
                "json",
            )
        )
        if not isinstance(value, dict):
            raise RegionalFixtureError("workload query did not return a JSON object")
        self.adopt(value)
        return cast(dict[str, Any], value)

    def pods(self) -> list[dict[str, Any]]:
        value = json.loads(
            self.regional.kubectl(
                "gpu",
                "get",
                "pod",
                "-l",
                f"gpu-fault.io/job-id={self.settings.job_id}",
                "-o",
                "json",
            )
        )
        ready = {item["uid"] for item in ready_pod_records(value)}
        result = []
        for item in value["items"]:
            self.adopt(item)
            result.append(
                {
                    "name": item["metadata"]["name"],
                    "uid": item["metadata"]["uid"],
                    "node": item["spec"].get("nodeName"),
                    "phase": item.get("status", {}).get("phase"),
                    "ready": item["metadata"]["uid"] in ready,
                    "attempt_id": item["metadata"]
                    .get("labels", {})
                    .get("gpu-fault.io/attempt-id"),
                }
            )
        return sorted(result, key=lambda item: str(item["name"]))

    def heartbeat_logs(self, pods: list[dict[str, Any]]) -> dict[str, str]:
        logs = {}
        for pod in pods:
            completed = self.regional.kubectl(
                "gpu",
                "logs",
                str(pod["name"]),
                "--tail=200",
                timeout=60,
            )
            lines = completed.splitlines()
            # The heartbeat/SUCCESS tail and the loss tail are kept separately
            # so a step that prints a loss line per rank cannot push the last
            # heartbeats out of the window a consumer substring-checks.
            heartbeats = [
                line for line in lines if "HEARTBEAT" in line or "SUCCESS" in line
            ]
            losses = [line for line in lines if "loss=" in line]
            logs[str(pod["name"])] = "\n".join([*heartbeats[-20:], *losses[-20:]])
        return logs

    def pods_healthy(self, pods: list[dict[str, Any]]) -> bool:
        """The Pod-level half of ``wait_running``: count, phase, spread."""

        return bool(
            len(pods) == self.settings.expected_pods
            and all(item["phase"] == "Running" and item["ready"] for item in pods)
            and all(item.get("node") and item.get("uid") for item in pods)
            and len({item["uid"] for item in pods}) == len(pods)
            and len({item["node"] for item in pods}) == self.settings.expected_pods
        )

    def logs_healthy(self, pods: list[dict[str, Any]], logs: dict[str, str]) -> bool:
        """The log half: every Pod heartbeats at the expected world size."""

        return all(
            "HEARTBEAT" in logs.get(str(item["name"]), "")
            and heartbeat_healthy(
                logs.get(str(item["name"]), ""),
                world_size=self.settings.expected_gpu_count,
            )
            for item in pods
        )

    def snapshot(self) -> dict[str, Any]:
        workload = self.workload()
        pods = self.pods()
        return {
            "workload": {
                "kind": workload["kind"],
                "name": workload["metadata"]["name"],
                "uid": workload["metadata"]["uid"],
                "annotations": workload["metadata"].get("annotations", {}),
                "labels": workload["metadata"].get("labels", {}),
                "suspend": workload.get("spec", {}).get("suspend"),
            },
            "pods": pods,
            "heartbeat_logs": self.heartbeat_logs(pods),
        }

    def wait_running(self, timeout_seconds: int = 900) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                last = self.snapshot()
            except WorkloadOwnershipError:
                raise
            except Exception:
                time.sleep(5)
                continue
            pods = last["pods"]
            if self.pods_healthy(pods) and self.logs_healthy(
                pods, last["heartbeat_logs"]
            ):
                return last
            time.sleep(10)
        raise RegionalFixtureError(f"managed workload did not become healthy: {last}")

    def wait_restarted(
        self,
        old_uids: set[str],
        *,
        timeout_seconds: int = 900,
        poll_seconds: int = 10,
    ) -> dict[str, Any]:
        """Wait for the workload's Pods to be replaced and heartbeat again.

        Replacement is detected from ``pods()`` alone -- one ``kubectl get``
        per poll -- because the old shape ran ``wait_running`` in the loop,
        which pulled every Pod's logs every ten seconds for up to fifteen
        minutes while the old Pods were still terminating. Logs are read only
        once the new Pods are Running, and then only until they heartbeat.
        """

        deadline = time.monotonic() + timeout_seconds
        last_pods: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            try:
                last_pods = self.pods()
            except WorkloadOwnershipError:
                raise
            except Exception:
                time.sleep(poll_seconds)
                continue
            current_uids = {str(item["uid"]) for item in last_pods}
            if (
                current_uids
                and current_uids.isdisjoint(old_uids)
                and self.pods_healthy(last_pods)
            ):
                break
            time.sleep(poll_seconds)
        else:
            raise RegionalFixtureError(
                f"managed workload Pods were not replaced: {last_pods}"
            )
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                last = self.snapshot()
            except WorkloadOwnershipError:
                raise
            except Exception:
                time.sleep(poll_seconds)
                continue
            pods = last["pods"]
            if {str(item["uid"]) for item in pods} & old_uids:
                raise RegionalFixtureError(
                    f"an old managed workload Pod reappeared: {pods}"
                )
            if self.pods_healthy(pods) and self.logs_healthy(
                pods, last["heartbeat_logs"]
            ):
                return last
            time.sleep(poll_seconds)
        raise RegionalFixtureError(
            f"replaced managed workload Pods did not heartbeat: {last or last_pods}"
        )

    def wait_pod_uids_unchanged(
        self,
        expected_uids: set[str],
        *,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self.snapshot()
            current_uids = {str(item["uid"]) for item in last["pods"]}
            if current_uids != expected_uids:
                raise RegionalFixtureError("managed workload Pod UIDs changed")
            time.sleep(5)
        return last
