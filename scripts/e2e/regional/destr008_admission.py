"""Run-owned, server-enforced spare activation inhibition for expiring fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.e2e.regional import destr008_controller_lock as locking
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.destr008_watchdog_control import CancellationControl
from scripts.e2e.regional.live_driver_guard import connection_identity
from scripts.e2e.regional.probes.destr008_admission_probe import (
    DENIAL_PREFIX,
    validate_identity,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
)
from scripts.e2e.regional.seeded_command_fixture import (
    RUN_LABEL,
    delete_owned_resource,
)

CASE_ID = "GF-REGIONAL-DESTR-008"
API_VERSION = "admissionregistration.k8s.io/v1"
RESOURCES = {
    "validatingadmissionpolicy": "ValidatingAdmissionPolicy",
    "validatingadmissionpolicybinding": "ValidatingAdmissionPolicyBinding",
}
PROBE = Path(__file__).with_name("probes") / "destr008_admission_probe.py"
SOURCE_LOADER = (
    "import hashlib,sys\n"
    "source=sys.stdin.buffer.read(65537)\n"
    "expected=sys.argv.pop(1)\n"
    "actual=hashlib.sha256(source).hexdigest()\n"
    "if len(source)>65536 or actual!=expected:\n"
    " raise SystemExit('pinned admission probe source differs')\n"
    "exec(compile(source,'<pinned-admission-probe>','exec'),"
    "{'__name__':'__main__','__file__':'<pinned-admission-probe>',"
    "'_SOURCE_SHA256':actual})\n"
)
RESERVATION = "gpu-fault.io/spare-reservation"
POOL_STATE = "gpu-fault.io/spare-pool-state"


@dataclass(frozen=True)
class FenceBinding:
    run_id: str
    cluster_id: str
    node: str
    node_uid: str
    release_id: str

    def __post_init__(self) -> None:
        validate_identity(*asdict(self).values())
        if len(self.run_id) > 63:
            raise RegionalFixtureError("the activation fence run label is too long")


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def configuration_identity(regional: RegionalLiveFixture) -> dict[str, Any]:
    settings = regional.settings
    connections = connection_identity(argparse.Namespace(**asdict(settings)), {})
    if len(connections) != 2 or any(
        set(item) != {"path", "sha256"} for item in connections.values()
    ):
        raise RegionalFixtureError("activation fence kubeconfig identity is incomplete")
    return {
        "connections": connections,
        "gpu_context": settings.gpu_context,
        "namespace": settings.namespace,
        "region": settings.region,
        "cluster_id": settings.cluster_id,
    }


def manifests(binding: FenceBinding, marker: str) -> list[dict[str, Any]]:
    validate_identity(marker)
    name = "gpu-fault-d008-" + marker[:20]
    metadata = {
        "name": name,
        "labels": {RUN_LABEL: binding.run_id, "gpu-fault.io/acceptance-case": CASE_ID},
        "annotations": {"gpu-fault.io/acceptance-scope-sha256": marker},
    }
    matching = {
        "matchPolicy": "Exact",
        "namespaceSelector": {},
        "objectSelector": {},
        "resourceRules": [
            {
                "apiGroups": [""],
                "apiVersions": ["v1"],
                "operations": ["CREATE", "UPDATE"],
                "resources": ["nodes"],
                "resourceNames": [binding.node],
                "scope": "Cluster",
            }
        ],
    }
    reservation = json.dumps(RESERVATION)
    pool = json.dumps(POOL_STATE)
    condition = (
        f"object.metadata.uid == {json.dumps(binding.node_uid)}"
        " && has(object.spec.unschedulable) && object.spec.unschedulable"
        " && (!has(object.metadata.annotations) || ("
        f"(!({reservation} in object.metadata.annotations)"
        f" || object.metadata.annotations[{reservation}] == '')"
        f" && (!({pool} in object.metadata.annotations)"
        f" || object.metadata.annotations[{pool}] != 'ALLOCATED')))"
    )
    return [
        {
            "apiVersion": API_VERSION,
            "kind": RESOURCES["validatingadmissionpolicy"],
            "metadata": metadata,
            "spec": {
                "failurePolicy": "Fail",
                "matchConstraints": matching,
                "validations": [
                    {
                        "expression": condition,
                        "message": DENIAL_PREFIX + marker,
                        "reason": "Forbidden",
                    }
                ],
            },
        },
        {
            "apiVersion": API_VERSION,
            "kind": RESOURCES["validatingadmissionpolicybinding"],
            "metadata": metadata,
            "spec": {
                "policyName": name,
                "validationActions": ["Deny"],
                "matchResources": matching,
            },
        },
    ]


def require_admission_api(regional: RegionalLiveFixture) -> None:
    value = json.loads(
        regional.kubectl("gpu", "get", "--raw", "/apis/" + API_VERSION, timeout=30)
    )
    if (
        not isinstance(value, dict)
        or value.get("kind") != "APIResourceList"
        or value.get("groupVersion") != API_VERSION
        or not isinstance(value.get("resources"), list)
    ):
        raise RegionalFixtureError("activation-fence API discovery is incomplete")
    found = {
        item.get("name"): item for item in value["resources"] if isinstance(item, dict)
    }
    for name in ("validatingadmissionpolicies", "validatingadmissionpolicybindings"):
        resource = found.get(name)
        if (
            not isinstance(resource, dict)
            or resource.get("namespaced") is not False
            or not {"get", "create", "delete"}.issubset(resource.get("verbs") or [])
        ):
            raise RegionalFixtureError("the GPU API does not support activation fences")


class ActivationFence:
    def __init__(
        self,
        regional: RegionalLiveFixture,
        binding: FenceBinding,
        directory: Path,
    ) -> None:
        self.regional = regional
        self.binding = binding
        self.probe_source = PROBE.read_bytes()
        if len(self.probe_source) > 65536:
            raise RegionalFixtureError("admission probe source exceeds its size limit")
        self.probe_sha256 = hashlib.sha256(self.probe_source).hexdigest()
        if binding.cluster_id != regional.settings.cluster_id:
            raise RegionalFixtureError(
                "activation fence cluster binding is inconsistent"
            )
        configuration = configuration_identity(regional)
        self.path = directory / (
            "fence-"
            + hashlib.sha256(binding.run_id.encode()).hexdigest()[:20]
            + ".json"
        )
        local_pins = {
            "schema_version": 1,
            "binding": asdict(binding),
            "configuration": configuration,
            "controller_host_sha256": locking.host_identity(),
            "locking_sha256": hashlib.sha256(
                Path(locking.__file__).read_bytes()
            ).hexdigest(),
            "controller_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "probe_sha256": self.probe_sha256,
        }
        saved = None
        if self.path.exists():
            if self.path.is_symlink():
                raise RegionalFixtureError("activation fence journal cannot be a link")
            saved = locking.read_private_document(self.path)
            if (
                not isinstance(saved, dict)
                or type(saved.get("schema_version")) is not int
                or any(saved.get(key) != value for key, value in local_pins.items())
            ):
                raise RegionalFixtureError(
                    "activation fence journal identity is invalid"
                )
        scope = regional.evidence_identity()
        if (
            scope.get("release_id") != binding.release_id
            or scope.get("cluster_id") != binding.cluster_id
        ):
            raise RegionalFixtureError(
                "activation fence live release or cluster differs"
            )
        self.pins = {**local_pins, "scope": scope}
        self.marker = digest(self.pins)
        self.objects = manifests(binding, self.marker)
        self.name = str(self.objects[0]["metadata"]["name"])
        self.record: dict[str, Any] = {
            **self.pins,
            "phase": "PREPARING",
            "action_started": False,
            "attempted": [],
            "resources": {},
        }
        if saved is not None:
            if (
                not isinstance(saved, dict)
                or type(saved.get("schema_version")) is not int
                or any(saved.get(key) != value for key, value in self.pins.items())
                or saved.get("phase") not in {"PREPARING", "ARMED", "CLOSING", "CLOSED"}
                or type(saved.get("action_started")) is not bool
                or not isinstance(saved.get("attempted"), list)
                or not isinstance(saved.get("resources"), dict)
                or any(kind not in RESOURCES for kind in saved["attempted"])
                or len(saved["attempted"]) != len(set(saved["attempted"]))
                or saved["attempted"] != list(RESOURCES)[: len(saved["attempted"])]
                or (
                    (saved["action_started"] or saved["phase"] == "ARMED")
                    and set(saved["resources"]) != set(RESOURCES)
                )
                or (saved["action_started"] and saved["phase"] == "PREPARING")
                or any(
                    kind not in saved["attempted"]
                    or not isinstance(uid, str)
                    or not uid
                    for kind, uid in saved["resources"].items()
                )
            ):
                raise RegionalFixtureError(
                    "activation fence journal identity is invalid"
                )
            self.record = saved

    def save(self) -> None:
        if not locking.ownership_held(self.path):
            raise RegionalFixtureError(
                "controller ownership is required before journal writes"
            )
        write_json_atomic(self.path, self.record)

    @contextmanager
    def operation(self) -> Iterator[None]:
        with locking.controller_ownership(self.path):
            self.verify_scope()
            current = ActivationFence(self.regional, self.binding, self.path.parent)
            if current.pins != self.pins:
                raise RegionalFixtureError(
                    "activation fence controller identity changed"
                )
            self.record = current.record
            yield

    def verify_scope(self) -> None:
        if configuration_identity(self.regional) != self.pins["configuration"]:
            raise RegionalFixtureError("activation fence connection identity changed")
        if self.regional.evidence_identity() != self.pins["scope"]:
            raise RegionalFixtureError("activation fence cluster identity changed")

    def gpu(self, *args: str, stdin: bytes | None = None, **kwargs: Any) -> str:
        self.verify_scope()
        return self.regional.kubectl(
            "gpu",
            *args,
            input_text=None if stdin is None else stdin.decode(),
            **kwargs,
        )

    def read(self, kind: str) -> dict[str, Any] | None:
        text = self.gpu("get", kind, self.name, "--ignore-not-found", "-o", "json")
        if not text.strip():
            return None
        value = json.loads(text)
        if not isinstance(value, dict):
            raise RegionalFixtureError("activation fence API response is not an object")
        return value

    def owned(self, kind: str, *, required: bool = True) -> dict[str, Any] | None:
        value = self.read(kind)
        if value is None:
            if required:
                raise RegionalFixtureError("activation fence resource is missing")
            return None
        return self.validate_resource(kind, value)

    def validate_resource(self, kind: str, value: dict[str, Any]) -> dict[str, Any]:
        expected = self.objects[list(RESOURCES).index(kind)]
        metadata = value.get("metadata") or {}
        expected_uid = self.record["resources"].get(kind)
        if (
            value.get("apiVersion") != expected["apiVersion"]
            or value.get("kind") != expected["kind"]
            or value.get("spec") != expected["spec"]
            or metadata.get("name") != self.name
            or not isinstance(metadata.get("uid"), str)
            or not metadata["uid"]
            or not isinstance(metadata.get("resourceVersion"), str)
            or not metadata["resourceVersion"]
            or metadata.get("deletionTimestamp")
            or (expected_uid is not None and metadata["uid"] != expected_uid)
            or any(
                (metadata.get("labels") or {}).get(key) != item
                for key, item in expected["metadata"]["labels"].items()
            )
            or any(
                (metadata.get("annotations") or {}).get(key) != item
                for key, item in expected["metadata"]["annotations"].items()
            )
        ):
            raise RegionalFixtureError(
                "activation fence resource ownership or spec changed"
            )
        return value

    def target(self) -> None:
        value = json.loads(self.gpu("get", "node", self.binding.node, "-o", "json"))
        metadata = value.get("metadata") or {}
        annotations = metadata.get("annotations") or {}
        if (
            metadata.get("uid") != self.binding.node_uid
            or metadata.get("name") != self.binding.node
            or (value.get("spec") or {}).get("unschedulable") is not True
            or annotations.get(RESERVATION)
            or annotations.get(POOL_STATE) == "ALLOCATED"
        ):
            raise RegionalFixtureError("protected spare identity or allocation changed")

    def probe(self) -> dict[str, Any]:
        for kind in RESOURCES:
            self.owned(kind)
        self.target()
        completed = self.regional.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                SOURCE_LOADER,
                self.probe_sha256,
                "--kubeconfig",
                str(self.regional.settings.gpu_kubeconfig),
                "--context",
                self.regional.settings.gpu_context,
                "--node",
                self.binding.node,
                "--uid",
                self.binding.node_uid,
                "--policy",
                self.name,
                "--binding",
                self.name,
                "--marker",
                self.marker,
            ],
            input_text=self.probe_source.decode("utf-8"),
            timeout=60,
        )
        expected = {
            "state": "DENYING_ACTIVATION",
            "node": self.binding.node,
            "node_uid": self.binding.node_uid,
            "policy": self.name,
            "binding": self.name,
            "marker": self.marker,
            "safe_dry_run_acknowledged": True,
            "activation_dry_run_denied": True,
            "probe_not_persisted": True,
            "source_sha256": self.probe_sha256,
        }
        observed = json.loads(completed.stdout)
        if (
            completed.returncode != 0
            or observed != expected
            or any(
                type(observed.get(key)) is not bool
                for key in (
                    "safe_dry_run_acknowledged",
                    "activation_dry_run_denied",
                    "probe_not_persisted",
                )
            )
        ):
            raise RegionalFixtureError("independent activation-fence probe failed")
        for kind in RESOURCES:
            self.owned(kind)
        self.target()
        return expected

    def arm(self) -> dict[str, Any]:
        with self.operation():
            return self._arm()

    def _arm(self) -> dict[str, Any]:
        if self.record["phase"] in {"CLOSED", "CLOSING"}:
            raise RegionalFixtureError(
                "a closing activation fence cannot authorize actions"
            )
        self.target()
        require_admission_api(self.regional)
        self.save()
        for kind, manifest in zip(RESOURCES, self.objects, strict=True):
            if kind not in self.record["attempted"]:
                if self.read(kind) is not None:
                    raise RegionalFixtureError(
                        "activation fence name is already in use"
                    )
                self.record["attempted"].append(kind)
                self.save()
                response = self.gpu(
                    "create",
                    "-f",
                    "-",
                    "-o",
                    "json",
                    stdin=json.dumps(manifest).encode(),
                    timeout=60,
                )
                created = json.loads(response)
                if not isinstance(created, dict):
                    raise RegionalFixtureError(
                        "activation fence create ACK is incomplete"
                    )
                acknowledged = self.validate_resource(kind, created)
                self.record["resources"][kind] = acknowledged["metadata"]["uid"]
                self.save()
            elif kind not in self.record["resources"]:
                raise RegionalFixtureError(
                    "activation fence creation outcome is still unknown"
                )
            resource = self.owned(kind)
            if resource is None:
                raise RegionalFixtureError("activation fence creation is unconfirmed")
            self.record["resources"][kind] = resource["metadata"]["uid"]
            self.save()
        proof = self.probe()
        self.record.update(phase="ARMED", admission_proof=proof)
        self.save()
        return self.identity()

    def identity(self) -> dict[str, str]:
        resources = self.record["resources"]
        if set(resources) != set(RESOURCES):
            raise RegionalFixtureError(
                "activation fence has incomplete resource identities"
            )
        return {
            "policy": self.name,
            "policy_uid": resources["validatingadmissionpolicy"],
            "binding": self.name,
            "binding_uid": resources["validatingadmissionpolicybinding"],
            "marker": self.marker,
            "node": self.binding.node,
            "node_uid": self.binding.node_uid,
        }

    def bind_watchdog(self, control: CancellationControl) -> None:
        with self.operation():
            if self.record["action_started"]:
                raise RegionalFixtureError(
                    "watchdog identity cannot change after action admission"
                )
            if (
                control.plan.run_id != self.binding.run_id
                or control.plan.cluster_id != self.binding.cluster_id
                or control.plan.release_id != self.binding.release_id
                or control.plan.fence.model_dump(mode="json") != self.identity()
                or control.namespace != self.regional.settings.namespace
            ):
                raise RegionalFixtureError(
                    "watchdog plan does not bind this activation fence"
                )
            identity = control.identity()
            if self.record.get("watchdog") not in (None, identity):
                raise RegionalFixtureError("a different watchdog is already bound")
            control.assert_armed()
            self.record["watchdog"] = identity
            self.save()

    def require_watchdog(
        self, control: CancellationControl | None
    ) -> CancellationControl:
        if (
            not isinstance(control, CancellationControl)
            or self.record.get("watchdog") != control.identity()
        ):
            raise RegionalFixtureError(
                "the bound complete watchdog control is required"
            )
        return control

    def protect_action(self, control: CancellationControl) -> None:
        with self.operation():
            self._protect_action(control)

    def _protect_action(self, control: CancellationControl) -> None:
        if self.record["phase"] != "ARMED" or self.record["action_started"]:
            raise RegionalFixtureError("the activation fence has not been armed")
        self.require_watchdog(control).assert_armed()
        self.probe()
        self.record["action_started"] = True
        self.save()

    def close(self, control: CancellationControl | None = None) -> None:
        with self.operation():
            self._close(control)

    def _close(self, control: CancellationControl | None = None) -> None:
        if self.record["phase"] == "CLOSED":
            if any(self.read(kind) is not None for kind in RESOURCES):
                raise RegionalFixtureError(
                    "a closed activation fence name was recreated"
                )
            return
        receipt = None
        if self.record["action_started"] or self.record.get("watchdog") is not None:
            receipt = self.require_watchdog(control).quiescence()
            if (
                receipt.run_id != self.binding.run_id
                or receipt.fence.model_dump(mode="json") != self.identity()
            ):
                raise RegionalFixtureError(
                    "watchdog quiescence belongs to another fence"
                )
        if self.record["action_started"]:
            self.target()
        if any(
            kind not in self.record["resources"] for kind in self.record["attempted"]
        ):
            raise RegionalFixtureError(
                "activation fence creation outcome is still unknown"
            )
        for kind in self.record["attempted"]:
            self.owned(kind, required=False)
        self.record["phase"] = "CLOSING"
        if receipt is not None:
            self.record["retirement_receipt"] = receipt.model_dump(mode="json")
        self.save()
        for kind in reversed(list(RESOURCES)):
            if kind not in self.record["attempted"]:
                continue
            resource = self.owned(kind, required=False)
            if resource is not None:
                uid = self.record["resources"].get(kind)
                if not uid:
                    raise RegionalFixtureError("activation fence UID was not recorded")
                delete_owned_resource(
                    kind,
                    self.name,
                    self.binding.run_id,
                    client=self.gpu,
                    expected_uid=uid,
                    expected_resource_version=resource["metadata"]["resourceVersion"],
                    require_uid=True,
                )
        self.record["phase"] = "CLOSED"
        self.save()
