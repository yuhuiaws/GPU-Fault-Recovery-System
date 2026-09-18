"""UID-conditional mutations of only the live late-ownership drill's objects.

Resource journal entries carry Creation's expected/uid/ack_sha256/approved fields.
A valid-identity but unapproved ACK grants DELETE-ONLY custody of that unchanged
object, never an owner patch or Ready wait. Unknown ACKs cannot be adopted by GET.
Custody is not a no-execution attestation for an object already accepted by Kubernetes.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kubernetes.utils.quantity import parse_quantity

from gpu_fault.adapters.common import (
    ANNOTATION_EXECUTION_EPOCH,
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    ANNOTATION_OPERATION,
    ANNOTATION_STEP_INDEX,
    ANNOTATION_TERMINATION_INCIDENT,
    ANNOTATION_WORKFLOW,
)
from scripts.e2e.regional.fixture_ownership import Creation, creation_document
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied, check_stop
from scripts.e2e.regional.late_ownership_contract import AcceptanceScope, StopReceipt
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.managed_workload_fixture import (
    OWNER_LABEL,
    delete_resource,
    read_resource,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture

SPEC_ANNOTATION = "gpu-fault.io/late-ownership-spec"
GVKS = {"pod": ("v1", "Pod"), "job": ("batch/v1", "Job")}
SERVER_METADATA = {
    "resourceVersion",
    "generation",
    "creationTimestamp",
    "deletionTimestamp",
    "deletionGracePeriodSeconds",
    "managedFields",
}
POD_DEFAULTS: dict[str, Any] = {
    "dnsPolicy": "ClusterFirst",
    "schedulerName": "default-scheduler",
    "terminationGracePeriodSeconds": 30,
    "enableServiceLinks": True,
    "securityContext": {},
    "serviceAccountName": "default",
    "serviceAccount": "default",
    "hostNetwork": False,
    "hostPID": False,
    "hostIPC": False,
    "shareProcessNamespace": False,
    "preemptionPolicy": "PreemptLowerPriority",
    "priority": 0,
    "nodeSelector": {},
    "affinity": {},
    "schedulingGates": [],
    "tolerations": [],
}
CONTAINER_DEFAULTS: dict[str, Any] = {
    "imagePullPolicy": "IfNotPresent",
    "terminationMessagePath": "/dev/termination-log",
    "terminationMessagePolicy": "File",
    "securityContext": {},
    "stdin": False,
    "stdinOnce": False,
    "tty": False,
}
JOB_DEFAULTS: dict[str, Any] = {
    "parallelism": 1,
    "completions": 1,
    "manualSelector": False,
    "completionMode": "NonIndexed",
    "podReplacementPolicy": "TerminatingOrFailed",
    "managedBy": "kubernetes.io/job-controller",
}

CUDA_HOLDER = (
    "import pathlib,threading,torch;"
    "held=torch.ones(1,device='cuda');"
    "torch.cuda.synchronize();"
    "pathlib.Path('/tmp/late-ownership-ready').write_text('ready');"
    "threading.Event().wait(240)"
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stable(document: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(document)
    value.pop("status", None)
    for key in SERVER_METADATA:
        value["metadata"].pop(key, None)
    return value


def _fingerprint(document: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(_stable(document)).encode()).hexdigest()


def _defaults(value: dict[str, Any], defaults: dict[str, Any]) -> None:
    for key, default in defaults.items():
        if key in value and _canonical(value[key]) == _canonical(default):
            del value[key]


def _pod_spec(spec: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(spec)
    _defaults(value, POD_DEFAULTS)
    tolerations = [
        {
            "key": "node.kubernetes.io/" + state,
            "operator": "Exists",
            "effect": "NoExecute",
            "tolerationSeconds": 300,
        }
        for state in ("not-ready", "unreachable")
    ]
    if "tolerations" in value and _canonical(value["tolerations"]) in {
        _canonical(tolerations),
        _canonical(tolerations[::-1]),
    }:
        del value["tolerations"]
    containers = value.get("containers")
    if not isinstance(containers, list):
        raise BoundaryDenied("late-ownership Pod containers are unavailable")
    for container in containers:
        if not isinstance(container, dict):
            raise BoundaryDenied("late-ownership container is not an object")
        _defaults(container, CONTAINER_DEFAULTS)
        if "resources" in container:
            requests = container["resources"]
            if not isinstance(requests, dict):
                raise BoundaryDenied("late-ownership resource quantities are invalid")
            for name in ("limits", "requests"):
                if name not in requests:
                    continue
                quantities = requests[name]
                if not isinstance(quantities, dict):
                    raise BoundaryDenied(
                        "late-ownership resource quantities are invalid"
                    )
                for key, quantity in quantities.items():
                    if type(quantity) not in {str, int, float}:
                        raise BoundaryDenied(
                            "late-ownership resource quantity is invalid"
                        )
                    try:
                        number = parse_quantity(str(quantity))
                        if not number.is_finite():
                            raise ValueError("non-finite quantity")
                    except (ValueError, TypeError, ArithmeticError):
                        raise BoundaryDenied(
                            "late-ownership resource quantity is invalid"
                        ) from None
                    quantities[key] = str(number.normalize())
                if not quantities:
                    del requests[name]
            if not requests:
                del container["resources"]
        if "readinessProbe" in container:
            readiness = container["readinessProbe"]
            if not isinstance(readiness, dict):
                raise BoundaryDenied("late-ownership readiness probe is invalid")
            _defaults(
                readiness,
                {
                    "initialDelaySeconds": 0,
                    "timeoutSeconds": 1,
                    "successThreshold": 1,
                },
            )
    return value


def _spec(document: dict[str, Any], *, uid: str) -> dict[str, Any]:
    spec = document.get("spec")
    if not isinstance(spec, dict):
        raise BoundaryDenied("late-ownership resource spec is unavailable")
    if document["kind"] == "Pod":
        return _pod_spec(spec)
    value = deepcopy(spec)
    _defaults(value, JOB_DEFAULTS)
    selector = value.get("selector")
    if selector is not None:
        permitted = {
            _canonical({"matchLabels": {key: uid}})
            for key in ("controller-uid", "batch.kubernetes.io/controller-uid")
        }
        if isinstance(selector, dict):
            _defaults(selector, {"matchExpressions": []})
        if _canonical(selector) in permitted:
            del value["selector"]
    template = value.get("template")
    if not isinstance(template, dict) or not isinstance(template.get("spec"), dict):
        raise BoundaryDenied("late-ownership anchor template is unavailable")
    if "metadata" in template:
        metadata = template["metadata"]
        if not isinstance(metadata, dict):
            raise BoundaryDenied("late-ownership anchor metadata is invalid")
        _defaults(metadata, {"creationTimestamp": None, "annotations": {}})
        if "labels" in metadata:
            labels = metadata["labels"]
            if not isinstance(labels, dict):
                raise BoundaryDenied("late-ownership anchor labels are invalid")
            _defaults(
                labels,
                {
                    "controller-uid": uid,
                    "batch.kubernetes.io/controller-uid": uid,
                    "job-name": document["metadata"]["name"],
                    "batch.kubernetes.io/job-name": document["metadata"]["name"],
                },
            )
            if not labels:
                del metadata["labels"]
        if not metadata:
            del template["metadata"]
    template["spec"] = _pod_spec(template["spec"])
    return value


def _owners(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    original = metadata.get("ownerReferences")
    owners = [] if original is None else deepcopy(original)
    if not isinstance(owners, list) or any(
        not isinstance(owner, dict) for owner in owners
    ):
        raise BoundaryDenied("late-ownership owner references are invalid")
    for owner in owners:
        _defaults(owner, {"controller": False, "blockOwnerDeletion": False})
    return owners


@dataclass
class OwnedMutation:
    regional: RegionalLiveFixture
    scope: AcceptanceScope
    source: dict[str, Any]
    image: str
    resources: list[dict[str, Any]] = field(default_factory=list)
    mutation_started: bool = False
    injected_owners: list[dict[str, Any]] | None = None
    journal_path: Path | None = None
    stop_receipt: StopReceipt | None = field(default=None, init=False)

    def _save(self) -> None:
        if self.journal_path is not None:
            write_json_atomic(
                self.journal_path,
                {
                    "scope_sha256": self.scope.digest(),
                    "workload": self.scope.workload.model_dump(mode="json"),
                    "original_owners": self.source["metadata"].get("ownerReferences"),
                    "injected_owners": self.injected_owners,
                    "mutation_started": self.mutation_started,
                    "resources": self.resources,
                    "stop_receipt": (
                        self.stop_receipt.model_dump(mode="json")
                        if self.stop_receipt is not None
                        else None
                    ),
                },
            )

    def _stopped_source(self, stop: StopReceipt) -> dict[str, Any]:
        expected = deepcopy(self.source)
        if (
            (expected.get("apiVersion"), expected.get("kind"))
            != ("kubeflow.org/v1", "PyTorchJob")
            or expected["metadata"].get("name") != self.scope.workload.name
            or expected["metadata"].get("namespace") != self.scope.workload.namespace
            or expected["metadata"].get("uid") != self.scope.workload.uid
        ):
            raise BoundaryDenied("STOP source identity differs from the approved scope")
        annotations = expected["metadata"].setdefault("annotations", {})
        spec = expected.get("spec")
        if not isinstance(annotations, dict) or not isinstance(spec, dict):
            raise BoundaryDenied("STOP source declared state is unavailable")
        policy = spec.setdefault("runPolicy", {})
        if not isinstance(policy, dict):
            raise BoundaryDenied("STOP source run policy is unavailable")
        policy["suspend"] = True
        annotations.update(
            {
                ANNOTATION_INCIDENT: self.scope.incident_id,
                ANNOTATION_FENCING: str(self.scope.fencing_token),
                ANNOTATION_WORKFLOW: self.scope.workflow_id,
                ANNOTATION_OPERATION: stop.stop_command_id,
                ANNOTATION_EXECUTION_EPOCH: str(self.scope.execution_epoch),
                ANNOTATION_STEP_INDEX: "0",
                ANNOTATION_TERMINATION_INCIDENT: self.scope.incident_id,
            }
        )
        return expected

    def _read_source(self, expected: dict[str, Any]) -> dict[str, Any]:
        value = read_resource(self.regional, "pytorchjob", self.scope.workload.name)
        if (
            value is None
            or value["metadata"]["uid"] != self.scope.workload.uid
            or value["metadata"].get("deletionTimestamp") is not None
        ):
            raise BoundaryDenied("owned workload disappeared or its UID changed")
        original, current = _stable(expected), _stable(value)
        for document in (original, current):
            document["metadata"].pop("ownerReferences", None)
        if _canonical(original) != _canonical(current):
            raise BoundaryDenied("owned workload declared state changed")
        return value

    def acknowledge_stop(self, stop: StopReceipt) -> None:
        check_stop(self.scope, stop)
        if (
            self.stop_receipt is not None
            or self.resources
            or self.mutation_started
            or stop.stop_command_id != f"{self.scope.workflow_id}/0/STOP_WORKLOADS"
        ):
            raise BoundaryDenied("STOP transition is late, repeated or unbound")
        current = self._read_source(self._stopped_source(stop))
        if _canonical(_owners(current["metadata"])) != _canonical(
            _owners(self.source["metadata"])
        ):
            raise BoundaryDenied("source owner drifted before STOP acknowledgement")
        self.stop_receipt = stop
        self._save()

    def source_object(self) -> dict[str, Any]:
        expected = (
            self.source
            if self.stop_receipt is None
            else self._stopped_source(self.stop_receipt)
        )
        return self._read_source(expected)

    def _create(self, document: dict[str, Any]) -> dict[str, Any]:
        kind = str(document["kind"]).lower()
        name = str(document["metadata"]["name"])
        if read_resource(self.regional, kind, name) is not None:
            raise BoundaryDenied("late-ownership resource already exists")
        body = deepcopy(document)
        metadata = body["metadata"]
        metadata["namespace"] = self.scope.workload.namespace
        metadata.setdefault("labels", {})[OWNER_LABEL] = self.scope.challenge
        metadata.setdefault("annotations", {})
        digest = hashlib.sha256(_canonical(body).encode()).hexdigest()
        metadata["annotations"][SPEC_ANNOTATION] = digest
        entry = {
            "kind": kind,
            "name": name,
            "owners": deepcopy(metadata.get("ownerReferences", [])),
            **Creation(expected=deepcopy(body)).model_dump(mode="json"),
        }
        self.resources.append(entry)
        self._save()
        raw = self.regional.kubectl(
            "gpu",
            "create",
            "-f",
            "-",
            "-o",
            "json",
            input_text=json.dumps(body),
            timeout=60,
        )
        try:
            created = creation_document(raw)
        except RegionalFixtureError:
            raise BoundaryDenied(
                "late-ownership creation was not acknowledged"
            ) from None
        self._identity(entry, created)
        entry["uid"] = created["metadata"]["uid"]
        entry["ack_sha256"] = _fingerprint(created)
        self._save()  # A valid identity ACK has custody even if its spec is rejected.
        self._declared(entry, created)
        entry["approved"] = True
        self._save()
        current = read_resource(self.regional, kind, name)
        if current is None:
            raise BoundaryDenied("late-ownership resource disappeared after creation")
        self._owned(entry, current)
        return current

    def _identity(self, entry: dict[str, Any], current: dict[str, Any]) -> None:
        metadata = current.get("metadata")
        expected = GVKS.get(entry["kind"])
        if (
            expected is None
            or (current.get("apiVersion"), current.get("kind")) != expected
            or not isinstance(metadata, dict)
            or metadata.get("name") != entry["name"]
            or metadata.get("namespace") != self.scope.workload.namespace
            or any(
                not isinstance(metadata.get(key), str) or not metadata[key]
                for key in ("uid", "resourceVersion")
            )
            or metadata.get("deletionTimestamp") is not None
            or not isinstance(metadata.get("labels"), dict)
            or metadata["labels"].get(OWNER_LABEL) != self.scope.challenge
        ):
            raise BoundaryDenied(
                "late-ownership creation owner or UID identity was not acknowledged"
            )

    def _declared(self, entry: dict[str, Any], current: dict[str, Any]) -> None:
        expected = entry["expected"]
        for section in ("labels", "annotations"):
            values = current["metadata"].get(section)
            if not isinstance(values, dict) or any(
                _canonical(values.get(key)) != _canonical(value)
                for key, value in expected["metadata"].get(section, {}).items()
            ):
                raise BoundaryDenied("late-ownership declared metadata changed")
        if _canonical(_owners(current["metadata"])) != _canonical(
            _owners(expected["metadata"])
        ):
            raise BoundaryDenied("late-ownership cleanup owner or UID changed")
        if _canonical(_spec(expected, uid=entry["uid"])) != _canonical(
            _spec(current, uid=entry["uid"])
        ):
            raise BoundaryDenied("late-ownership declared spec changed")

    def _owned(
        self, entry: dict[str, Any], current: dict[str, Any], *, cleanup: bool = False
    ) -> None:
        if not entry.get("uid") or current["metadata"]["uid"] != entry["uid"]:
            raise BoundaryDenied("late-ownership cleanup owner or UID changed")
        self._custody(entry)
        self._identity(entry, current)
        if entry["approved"]:
            self._declared(entry, current)
        elif not cleanup or _fingerprint(current) != entry["ack_sha256"]:
            raise BoundaryDenied(
                "late-ownership deletion-only resource changed or is unapproved"
            )

    def _custody(self, entry: dict[str, Any]) -> None:
        if not entry.get("uid"):
            raise BoundaryDenied("resource creation UID was never acknowledged")
        try:
            custody = Creation.model_validate(
                {key: entry[key] for key in Creation.model_fields}
            )
            expected = deepcopy(custody.expected)
            metadata = expected["metadata"]
            declared = metadata["annotations"].pop(SPEC_ANNOTATION)
            if (
                (expected["apiVersion"], expected["kind"]) != GVKS[entry["kind"]]
                or metadata["name"] != entry["name"]
                or metadata["namespace"] != self.scope.workload.namespace
                or metadata["labels"][OWNER_LABEL] != self.scope.challenge
                or hashlib.sha256(_canonical(expected).encode()).hexdigest() != declared
            ):
                raise ValueError("custody intent binding")
        except (KeyError, ValueError, TypeError):
            raise BoundaryDenied(
                "late-ownership custody record is incomplete"
            ) from None

    def change_owner(self) -> None:
        current = self.source_object()
        original = self.source["metadata"].get("ownerReferences", [])
        if current["metadata"].get("ownerReferences", []) != original:
            raise BoundaryDenied("source owner drifted before the controlled mutation")
        anchor = self._create(
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": f"late-owner-{self.scope.challenge[:12]}"},
                "spec": {
                    "suspend": True,
                    "backoffLimit": 0,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "automountServiceAccountToken": False,
                            "containers": [
                                {
                                    "name": "anchor",
                                    "image": self.image,
                                    "command": ["true"],
                                }
                            ],
                        }
                    },
                },
            }
        )
        current = self.source_object()
        if current["metadata"].get("ownerReferences", []) != original:
            raise BoundaryDenied("source owner drifted before the controlled mutation")
        self.injected_owners = [
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "name": anchor["metadata"]["name"],
                "uid": anchor["metadata"]["uid"],
                "controller": False,
                "blockOwnerDeletion": False,
            }
        ]
        self.mutation_started = True
        self._save()
        patch = [
            {"op": "test", "path": "/metadata/uid", "value": self.scope.workload.uid},
            {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": current["metadata"]["resourceVersion"],
            },
            {
                "op": "add",
                "path": "/metadata/ownerReferences",
                "value": self.injected_owners,
            },
        ]
        self.regional.kubectl(
            "gpu",
            "patch",
            "pytorchjob",
            self.scope.workload.name,
            "--type=json",
            "-p",
            json.dumps(patch),
            timeout=60,
        )
        if (
            self.source_object()["metadata"].get("ownerReferences")
            != self.injected_owners
        ):
            raise BoundaryDenied("controlled owner change was not observed")

    def late_sibling(self) -> str:
        current = self.source_object()
        if current["metadata"].get("ownerReferences", []) != self.source[
            "metadata"
        ].get("ownerReferences", []):
            raise BoundaryDenied("late sibling source owner changed")
        created = self._create(
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": f"late-sibling-{self.scope.challenge[:12]}",
                    "labels": {
                        "gpu-fault.io/managed": "true",
                        "gpu-fault.io/attempt-id": self.scope.workload.attempt_id,
                    },
                    "annotations": {
                        "gpu-fault.io/runtime-profile-version": self.scope.runtime_profile,
                        "gpu-fault.io/training-container": "training",
                    },
                    "ownerReferences": [
                        {
                            "apiVersion": "kubeflow.org/v1",
                            "kind": "PyTorchJob",
                            "name": self.scope.workload.name,
                            "uid": self.scope.workload.uid,
                            "controller": True,
                            "blockOwnerDeletion": False,
                        }
                    ],
                },
                "spec": {
                    "nodeName": self.scope.nodes[1].name,
                    "restartPolicy": "Never",
                    "activeDeadlineSeconds": 240,
                    "automountServiceAccountToken": False,
                    "containers": [
                        {
                            "name": "training",
                            "image": self.image,
                            "command": ["python3", "-u", "-c", CUDA_HOLDER],
                            "resources": {
                                "requests": {
                                    "cpu": "100m",
                                    "memory": "512Mi",
                                    "nvidia.com/gpu": "1",
                                },
                                "limits": {
                                    "cpu": "1",
                                    "memory": "2Gi",
                                    "nvidia.com/gpu": "1",
                                },
                            },
                            "readinessProbe": {
                                "exec": {
                                    "command": [
                                        "test",
                                        "-f",
                                        "/tmp/late-ownership-ready",
                                    ]
                                },
                                "periodSeconds": 1,
                                "failureThreshold": 60,
                            },
                        }
                    ],
                },
            }
        )
        self.regional.kubectl(
            "gpu",
            "wait",
            "--for=condition=Ready",
            f"pod/{created['metadata']['name']}",
            "--timeout=30s",
            timeout=40,
        )
        pod_current = read_resource(self.regional, "pod", created["metadata"]["name"])
        if pod_current is None:
            raise BoundaryDenied("late GPU sibling disappeared before observation")
        self._owned(self.resources[-1], pod_current)
        return str(created["metadata"]["uid"])

    def cleanup(self) -> None:
        for entry in self.resources:
            self._custody(entry)
        if self.mutation_started:
            current = self.source_object()
            original = self.source["metadata"].get("ownerReferences")
            owners = current["metadata"].get("ownerReferences")
            if owners == original:
                self.mutation_started = False
                self._save()
            elif owners != self.injected_owners:
                raise BoundaryDenied(
                    "source owner was replaced after the controlled mutation"
                )
        if self.mutation_started:
            patch = [
                {
                    "op": "test",
                    "path": "/metadata/uid",
                    "value": self.scope.workload.uid,
                },
                {
                    "op": "test",
                    "path": "/metadata/resourceVersion",
                    "value": current["metadata"]["resourceVersion"],
                },
                {
                    "op": "test",
                    "path": "/metadata/ownerReferences",
                    "value": self.injected_owners,
                },
                (
                    {
                        "op": "replace",
                        "path": "/metadata/ownerReferences",
                        "value": original,
                    }
                    if original is not None
                    else {"op": "remove", "path": "/metadata/ownerReferences"}
                ),
            ]
            self.regional.kubectl(
                "gpu",
                "patch",
                "pytorchjob",
                self.scope.workload.name,
                "--type=json",
                "-p",
                json.dumps(patch),
                timeout=60,
            )
            if self.source_object()["metadata"].get("ownerReferences") != original:
                raise BoundaryDenied("original source owner was not restored")
            self.mutation_started = False
            self._save()
        for entry in reversed(self.resources):
            resource = read_resource(self.regional, entry["kind"], entry["name"])
            if resource is None:
                continue
            self._owned(entry, resource, cleanup=True)
            delete_resource(self.regional, resource)
            if read_resource(self.regional, entry["kind"], entry["name"]) is not None:
                raise BoundaryDenied(
                    "owned late-ownership resource remains after cleanup"
                )
