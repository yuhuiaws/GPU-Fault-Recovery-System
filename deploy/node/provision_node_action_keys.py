"""Deploy-host node-key synchronization; Secret material stays in private stdio."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from gpu_fault.admin.execution import deadline_scope, run_command
from gpu_fault.admin.node_key_custody import ProvisionCustody
from gpu_fault.admin.node_key_custody_crypto import (
    private_command_environment,
    read_regular,
)
from gpu_fault.admin.node_key_custody_models import CustodyError
from gpu_fault.admin.node_key_proof import load_node_key_custody_request
from gpu_fault.node_action_keys import derive_node_action_secret
from gpu_fault_release.regional_resource_probe import ResourceRef, probe_resource

ROTATION_ANNOTATION = "gpu-fault.io/node-action-key-rotation"
HYPERPOD_LABEL = "sagemaker.amazonaws.com/cluster-name"
ATTEMPTS = 3
COMMAND_SECONDS = 20


class ProvisionError(RuntimeError):
    pass


class Runner(Protocol):
    def __call__(
        self,
        arguments: list[str],
        *,
        input_text: str | None = None,
        capture: bool = True,
        timeout_seconds: float | None = None,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True)
class Scope:
    command: tuple[str, ...]
    namespace: str
    secret_name: str = "gpu-fault-node-action-keys"

    def reference(self) -> ResourceRef:
        return ResourceRef("secret", "Secret", self.secret_name, self.namespace)


@dataclass(frozen=True)
class Snapshot:
    document: dict[str, Any] = field(repr=False)
    data: dict[str, str] = field(repr=False)
    uid: str
    version: str


@dataclass(frozen=True)
class KeyPlan:
    data: dict[str, str] = field(repr=False)
    previous: dict[str, str] = field(repr=False)
    derived: dict[str, str] = field(repr=False)
    rotation: dict[str, Any] | None = field(default=None, repr=False)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProvisionError("Kubernetes JSON contains duplicate fields")
        result[key] = value
    return result


def _object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text, object_pairs_hook=_object_pairs)
    except (ValueError, TypeError):
        raise ProvisionError("invalid Kubernetes JSON") from None
    if not isinstance(value, dict):
        raise ProvisionError("expected a Kubernetes JSON object")
    return cast(dict[str, Any], value)


def _name(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", value) is not None
    )


def _expected_nodes(raw: str | None) -> dict[str, str] | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw, object_pairs_hook=_object_pairs)
    except (ValueError, TypeError):
        raise ProvisionError("expected node inventory is not valid JSON") from None
    if isinstance(value, list):
        if (
            not value
            or any(not _name(name) for name in value)
            or len(set(value)) != len(value)
        ):
            raise ProvisionError("expected node names are empty or ambiguous")
        # The join caller historically supplied only names. A UID map also
        # closes the discovery-to-provisioning same-name reincarnation gap.
        return dict.fromkeys(value, "")
    if (
        not isinstance(value, dict)
        or not value
        or any(
            not _name(name) or not isinstance(uid, str) or not uid
            for name, uid in value.items()
        )
        or len(set(value.values())) != len(value)
    ):
        raise ProvisionError("expected node identities are empty or ambiguous")
    return cast(dict[str, str], value)


def _key_bytes(value: object) -> bytes:
    try:
        if not isinstance(value, str):
            raise ValueError
        decoded = base64.b64decode(value, validate=True)
        if (
            base64.b64encode(decoded).decode("ascii") != value
            or len(decoded.decode("utf-8").strip()) < 32
        ):
            raise ValueError
    except (ValueError, UnicodeError):
        raise ProvisionError("node action key data has an invalid shape") from None
    return decoded


def _key_digest(value: str) -> str:
    return hashlib.sha256(_key_bytes(value)).hexdigest()


def _marker(document: dict[str, Any]) -> str | None:
    raw = document["metadata"].get("annotations")
    annotations = {} if raw is None else raw
    if not isinstance(annotations, dict) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in annotations.items()
    ):
        raise ProvisionError("Secret annotations have an invalid shape")
    return cast(str | None, annotations.get(ROTATION_ANNOTATION))


class SecretClient:
    def __init__(self, scope: Scope, runner: Runner) -> None:
        self.scope = scope
        self.runner = runner
        self.reference = scope.reference()
        self.uid: str | None = None
        self.custody: ProvisionCustody | None = None
        self.custody_gpu = False

    def execute(
        self,
        arguments: list[str],
        *,
        input_text: str | None = None,
        mutating: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        try:
            if self.custody is not None:
                seconds = self.custody.command_seconds(
                    COMMAND_SECONDS, mutating=mutating
                )
                return self.runner(
                    arguments,
                    input_text=input_text,
                    capture=True,
                    timeout_seconds=seconds,
                    environment=private_command_environment(),
                )
            return self.runner(
                arguments,
                input_text=input_text,
                capture=True,
                timeout_seconds=COMMAND_SECONDS,
            )
        except (subprocess.SubprocessError, TimeoutError, OSError):
            raise ProvisionError("node-key kubectl command failed") from None

    def probe_output(
        self, args: list[str], *, timeout_seconds: float | None = None
    ) -> tuple[int, str, str]:
        del timeout_seconds
        result = self.execute(args)
        # Authentication helpers and API diagnostics can echo Secret payloads.
        # The shared tri-state probe receives only a safe error category.
        if result.returncode:
            return result.returncode, "", "node-key Secret read failed"
        if result.stdout.strip():
            _object(result.stdout)
        return 0, result.stdout, ""

    def read(self) -> Snapshot | None:
        observation = probe_resource(self, list(self.scope.command), self.reference)
        document = observation.require_readable()
        if document is None:
            if self.uid is not None:
                raise ProvisionError(
                    "node-key Secret disappeared during synchronization"
                )
            return None
        metadata = cast(dict[str, Any], document["metadata"])
        version = metadata.get("resourceVersion")
        raw_data = document.get("data")
        data = {} if raw_data is None else raw_data
        if (
            document.get("apiVersion") != "v1"
            or document.get("type") != "Opaque"
            or "stringData" in document
            or not isinstance(version, str)
            or not version
            or metadata.get("deletionTimestamp")
            or not isinstance(data, dict)
            or ("immutable" in document and type(document["immutable"]) is not bool)
        ):
            raise ProvisionError("node-key Secret has an invalid shape or identity")
        for name, value in data.items():
            if not _name(name):
                raise ProvisionError("node-key Secret contains an invalid NodeName")
            _key_bytes(value)
        uid = cast(str, metadata["uid"])
        if self.uid is not None and self.uid != uid:
            raise ProvisionError("node-key Secret UID changed during synchronization")
        self.uid = uid
        snapshot = Snapshot(dict(document), dict(data), uid, version)
        _marker(snapshot.document)
        return snapshot

    def nodes(self, hyperpod_cluster: str) -> dict[str, str]:
        result = self.execute(
            [
                *self.scope.command,
                "get",
                "nodes",
                "-l",
                f"{HYPERPOD_LABEL}={hyperpod_cluster}",
                "-o",
                "json",
                "--request-timeout=15s",
            ]
        )
        if result.returncode:
            raise ProvisionError("HyperPod node inventory could not be read")
        document = _object(result.stdout)
        items = document.get("items")
        metadata = document.get("metadata", {})
        if (
            document.get("apiVersion") != "v1"
            or document.get("kind") not in {"NodeList", "List"}
            or not isinstance(metadata, dict)
            or metadata.get("continue")
            or not isinstance(items, list)
            or not items
        ):
            raise ProvisionError("HyperPod node inventory is empty or invalid")
        nodes: dict[str, str] = {}
        for item in items:
            if not isinstance(item, dict):
                raise ProvisionError("HyperPod node identity is invalid")
            node = item.get("metadata")
            if not isinstance(node, dict):
                raise ProvisionError("HyperPod node identity is invalid")
            name, uid = node.get("name"), node.get("uid")
            labels = node.get("labels")
            if (
                item.get("apiVersion") != "v1"
                or item.get("kind") != "Node"
                or not _name(name)
                or not isinstance(uid, str)
                or not uid
                or name in nodes
                or uid in nodes.values()
                or node.get("deletionTimestamp")
                or not isinstance(labels, dict)
                or labels.get(HYPERPOD_LABEL) != hyperpod_cluster
            ):
                raise ProvisionError("HyperPod node identity is invalid or ambiguous")
            nodes[cast(str, name)] = uid
        return dict(sorted(nodes.items()))

    def write(
        self, previous: Snapshot | None, desired: dict[str, Any]
    ) -> tuple[Snapshot, bool]:
        if previous is not None and previous.document.get("immutable"):
            raise ProvisionError("node-key Secret is immutable")
        if self.custody is not None:
            self.custody.transfer(desired, gpu=self.custody_gpu)
        result = self.execute(
            [
                *self.scope.command,
                "-n",
                self.scope.namespace,
                "replace" if previous else "create",
                "-f",
                "-",
                "-o",
                "json",
                "--request-timeout=15s",
            ],
            input_text=json.dumps(desired, separators=(",", ":")),
            mutating=True,
        )
        if self.custody is not None and previous is None and result.returncode:
            raise CustodyError(
                "custody create lacks direct acknowledgement; reconciliation required"
            )
        if not result.returncode:
            self.confirm_acknowledgement(result.stdout, desired)
        current = self.read()
        if current is None:
            raise ProvisionError("node-key Secret write was not confirmed")
        complete = current.data == desired["data"] and _marker(
            current.document
        ) == _marker(desired)
        if (
            not complete
            and previous is not None
            and current.version == previous.version
        ):
            raise ProvisionError(
                "node-key Secret write failed without confirmed progress"
            )
        if result.returncode and previous is None and not complete:
            # A concurrent creator is not absence. Its live object must pass a
            # new ownership/conflict check before any replacement is attempted.
            return current, False
        return current, complete

    def confirm_acknowledgement(self, output: str, desired: dict[str, Any]) -> None:
        document = _object(output)
        metadata = document.get("metadata")
        if (
            document.get("apiVersion") != "v1"
            or document.get("kind") != "Secret"
            or document.get("type") != "Opaque"
            or not isinstance(metadata, dict)
            or metadata.get("name") != self.scope.secret_name
            or metadata.get("namespace") != self.scope.namespace
            or not isinstance(metadata.get("uid"), str)
            or not metadata["uid"]
            or self.uid is not None
            and metadata["uid"] != self.uid
            or not isinstance(metadata.get("resourceVersion"), str)
            or not metadata["resourceVersion"]
            or document.get("data") != desired["data"]
            or _marker(document) != _marker(desired)
        ):
            raise ProvisionError("node-key Secret write acknowledgement differs")
        self.uid = metadata["uid"]


def _document(
    scope: Scope,
    snapshot: Snapshot | None,
    data: dict[str, str],
) -> dict[str, Any]:
    result = (
        copy.deepcopy(snapshot.document)
        if snapshot is not None
        else {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": scope.secret_name, "namespace": scope.namespace},
            "type": "Opaque",
        }
    )
    result["data"] = dict(data)
    result["metadata"].pop("managedFields", None)
    return result


def _pending_rotation(
    gpu: Snapshot | None,
    cpu: Snapshot | None,
    cpu_scope: Scope | None,
    cluster_id: str,
    nodes: dict[str, str],
    rotate_node: str,
) -> dict[str, Any] | None:
    raw = _marker(gpu.document) if gpu else None
    if raw is None:
        return None
    value = _object(raw)
    node = value.get("node")
    if (
        set(value)
        != {
            "schema",
            "cluster_id",
            "node",
            "node_uid",
            "cpu_namespace",
            "cpu_secret",
            "cpu_uid",
            "before_sha256",
            "after_sha256",
        }
        or type(value["schema"]) is not int
        or value["schema"] != 1
        or value["cluster_id"] != cluster_id
        or not isinstance(node, str)
        or node not in nodes
        or value["node_uid"] != nodes[node]
        or cpu_scope is None
        or value["cpu_namespace"] != cpu_scope.namespace
        or value["cpu_secret"] != cpu_scope.secret_name
        or not isinstance(value["cpu_uid"], str)
        or value["cpu_uid"]
        and (cpu is None or value["cpu_uid"] != cpu.uid)
        or rotate_node
        and rotate_node != node
        or any(
            not isinstance(value[key], str)
            or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None
            for key in ("before_sha256", "after_sha256")
        )
        or gpu is None
        or node not in gpu.data
        or _key_digest(gpu.data[node]) != value["after_sha256"]
    ):
        raise ProvisionError("pending node-key rotation binding differs")
    return value


def _plan(
    gpu: Snapshot | None,
    cpu: Snapshot | None,
    cpu_scope: Scope | None,
    *,
    nodes: dict[str, str],
    cluster_id: str,
    derived: dict[str, str],
    rotate_node: str,
    replacement: str,
) -> KeyPlan:
    previous = gpu.data if gpu else {}
    data = {node: previous.get(node, derived[node]) for node in nodes}
    rotation = _pending_rotation(gpu, cpu, cpu_scope, cluster_id, nodes, rotate_node)
    if rotate_node and rotation is None:
        before = data[rotate_node]
        data[rotate_node] = replacement
        if cpu_scope is not None:
            rotation = {
                "schema": 1,
                "cluster_id": cluster_id,
                "node": rotate_node,
                "node_uid": nodes[rotate_node],
                "cpu_namespace": cpu_scope.namespace,
                "cpu_secret": cpu_scope.secret_name,
                "cpu_uid": cpu.uid if cpu else "",
                "before_sha256": _key_digest(before),
                "after_sha256": _key_digest(replacement),
            }
    return KeyPlan(data, dict(previous), derived, rotation)


def _check_cpu(cpu: Snapshot | None, plan: KeyPlan) -> None:
    if cpu is None:
        return
    if cpu.document.get("immutable") and any(
        cpu.data.get(node) != desired for node, desired in plan.data.items()
    ):
        raise ProvisionError("CPU node-key Secret is immutable")
    for node, desired in plan.data.items():
        current = cpu.data.get(node)
        if current is None or current in {
            desired,
            plan.derived[node],
            plan.previous.get(node),
        }:
            continue
        pending = plan.rotation
        if (
            pending is not None
            and pending["node"] == node
            and pending["cpu_uid"] == cpu.uid
            and _key_digest(current) == pending["before_sha256"]
        ):
            continue
        raise ProvisionError(
            "CPU NodeName key conflict; exclusive cluster ownership requires reconciliation"
        )


def _gpu_document(
    scope: Scope, snapshot: Snapshot | None, plan: KeyPlan
) -> dict[str, Any]:
    result = _document(scope, snapshot, plan.data)
    if plan.rotation is not None:
        annotations = dict(result["metadata"].get("annotations") or {})
        annotations[ROTATION_ANNOTATION] = json.dumps(
            plan.rotation, sort_keys=True, separators=(",", ":")
        )
        result["metadata"]["annotations"] = annotations
    return result


def _assert_gpu(current: Snapshot | None, expected: Snapshot) -> Snapshot:
    if (
        current is None
        or current.uid != expected.uid
        or current.data != expected.data
        or _marker(current.document) != _marker(expected.document)
    ):
        raise ProvisionError("GPU node keys changed during CPU synchronization")
    return current


def _finish_rotation(
    gpu: SecretClient,
    cpu: SecretClient,
    expected: Snapshot,
    *,
    hyperpod_cluster: str,
    nodes: dict[str, str],
) -> None:
    if _marker(expected.document) is None:
        return
    for _attempt in range(ATTEMPTS):
        current = gpu.read()
        mirror = cpu.read()
        if mirror is None or any(
            mirror.data.get(node) != value for node, value in expected.data.items()
        ):
            raise ProvisionError("CPU node keys do not confirm the pending rotation")
        if gpu.nodes(hyperpod_cluster) != nodes:
            raise ProvisionError("HyperPod node identities changed")
        if (
            current is not None
            and current.data == expected.data
            and _marker(current.document) is None
        ):
            return
        current = _assert_gpu(current, expected)
        desired = _document(gpu.scope, current, current.data)
        desired["metadata"]["annotations"].pop(ROTATION_ANNOTATION)
        _fresh, complete = gpu.write(current, desired)
        if complete:
            return
    raise ProvisionError("GPU node-key rotation completion changed repeatedly")


def provision(
    gpu_scope: Scope,
    cpu_scope: Scope | None,
    *,
    cluster_id: str,
    hyperpod_cluster: str,
    master_file: Path,
    rotate_node: str = "",
    expected_nodes: dict[str, str] | None = None,
    runner: Runner = run_command,
    custody: ProvisionCustody | None = None,
) -> int:
    if (
        not cluster_id
        or any(ord(character) < 32 for character in cluster_id)
        or re.fullmatch(
            r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,61}[A-Za-z0-9])?", hyperpod_cluster
        )
        is None
    ):
        raise ProvisionError("node-key cluster identity is invalid")
    with deadline_scope("node action key synchronization", 300):
        if custody is not None and (
            gpu_scope.secret_name != "gpu-fault-node-action-keys"
            or cpu_scope is None
            or cpu_scope.secret_name != gpu_scope.secret_name
        ):
            raise CustodyError("custody requires the CPU/GPU node-action key sources")
        gpu = SecretClient(gpu_scope, runner)
        gpu.custody = custody
        gpu.custody_gpu = True
        if custody is not None:
            custody.activation_wave_reader = lambda name: _read_activation_wave(
                gpu, name
            )
        cpu = SecretClient(cpu_scope, runner) if cpu_scope else None
        if cpu is not None:
            cpu.custody = custody
        nodes = gpu.nodes(hyperpod_cluster)
        if expected_nodes is not None and (
            set(nodes) != set(expected_nodes)
            or any(uid and nodes[node] != uid for node, uid in expected_nodes.items())
        ):
            raise ProvisionError(
                "current nodes differ from the verified node inventory"
            )
        if rotate_node and rotate_node not in nodes:
            raise ProvisionError("rotation target is not a current HyperPod node")
        if custody is not None:
            custody.bind(
                script=Path(__file__),
                cluster_id=cluster_id,
                hyperpod_cluster=hyperpod_cluster,
                nodes=nodes,
                gpu_namespace=gpu_scope.namespace,
                cpu_namespace=cpu_scope.namespace if cpu_scope else None,
                rotate_node=rotate_node,
            )
            _custody_anchors(custody, gpu, cpu)
        try:
            master = (
                read_regular(master_file, private=True, limit=4096).decode("utf-8")
                if custody is not None
                else master_file.read_text(encoding="utf-8").strip()
            )
            if custody is not None:
                _custody_master(custody, cpu, master)
            derived = {
                node: base64.b64encode(
                    derive_node_action_secret(master, cluster_id, node).encode()
                ).decode("ascii")
                for node in nodes
            }
        except (OSError, UnicodeError, ValueError):
            raise ProvisionError(
                "fleet master could not be read or validated"
            ) from None
        replacement = (
            base64.b64encode(secrets.token_hex(32).encode()).decode("ascii")
            if rotate_node
            else ""
        )
        current = gpu.read()
        mirror = cpu.read() if cpu else None
        for _attempt in range(ATTEMPTS):
            if current is not None and mirror is not None and current.uid == mirror.uid:
                raise ProvisionError("CPU and GPU node-key targets must be distinct")
            if gpu.nodes(hyperpod_cluster) != nodes:
                raise ProvisionError("HyperPod node identities changed")
            plan = _plan(
                current,
                mirror,
                cpu_scope,
                nodes=nodes,
                cluster_id=cluster_id,
                derived=derived,
                rotate_node=rotate_node,
                replacement=replacement,
            )
            _check_cpu(mirror, plan)
            if custody is not None:
                custody.begin(current, mirror, plan.data)
            desired = _gpu_document(gpu_scope, current, plan)
            if (
                current is None
                or current.data != plan.data
                or _marker(current.document) != _marker(desired)
            ):
                current, complete = gpu.write(current, desired)
                if not complete:
                    mirror = cpu.read() if cpu else None
                    continue
            assert current is not None
            if cpu is not None:
                _synchronize_cpu(gpu, cpu, current, plan, hyperpod_cluster, nodes)
                _finish_rotation(
                    gpu,
                    cpu,
                    current,
                    hyperpod_cluster=hyperpod_cluster,
                    nodes=nodes,
                )
            if gpu.nodes(hyperpod_cluster) != nodes:
                raise ProvisionError("HyperPod node identities changed")
            final = gpu.read()
            if (
                final is None
                or final.data != plan.data
                or _marker(final.document) is not None
            ):
                raise ProvisionError("GPU node keys did not converge")
            if cpu is not None:
                final_cpu = cpu.read()
                if final_cpu is None or any(
                    final_cpu.data.get(node) != value
                    for node, value in plan.data.items()
                ):
                    raise ProvisionError("CPU node keys did not converge")
                if custody is not None:
                    _custody_anchors(custody, gpu, cpu)
                    _custody_master(custody, cpu, master)
                    custody.finish(final, final_cpu)
            return len(nodes)
    raise ProvisionError("GPU node-key Secret changed repeatedly")


def _synchronize_cpu(
    gpu: SecretClient,
    cpu: SecretClient,
    expected: Snapshot,
    plan: KeyPlan,
    hyperpod_cluster: str,
    nodes: dict[str, str],
) -> None:
    for _attempt in range(ATTEMPTS):
        mirror = cpu.read()
        if mirror is not None and mirror.uid == expected.uid:
            raise ProvisionError("CPU and GPU node-key targets must be distinct")
        _assert_gpu(gpu.read(), expected)
        if gpu.nodes(hyperpod_cluster) != nodes:
            raise ProvisionError("HyperPod node identities changed")
        _check_cpu(mirror, plan)
        data = {**(mirror.data if mirror else {}), **plan.data}
        if mirror is not None and data == mirror.data:
            return
        _fresh, complete = cpu.write(mirror, _document(cpu.scope, mirror, data))
        if complete:
            return
    raise ProvisionError("CPU node-key Secret changed repeatedly")


def _read_activation_wave(gpu: SecretClient, name: str) -> dict[str, Any]:
    result = gpu.execute(
        [
            *gpu.scope.command,
            "-n",
            gpu.scope.namespace,
            "get",
            "configmap",
            name,
            "-o",
            "json",
            "--request-timeout=15s",
        ]
    )
    if result.returncode:
        raise CustodyError("custody writer could not verify the owned installer wave")
    return _object(result.stdout)


def _custody_anchors(
    custody: ProvisionCustody, gpu: SecretClient, cpu: SecretClient | None
) -> None:
    if cpu is None:
        raise CustodyError("custody requires a distinct CPU scope")

    def read(plane: str, namespace: str) -> dict[str, Any]:
        client = cpu if plane == "cpu" else gpu
        result = client.execute(
            [
                *client.scope.command,
                "get",
                "namespace",
                namespace,
                "-o",
                "json",
                "--request-timeout=15s",
            ]
        )
        if result.returncode:
            raise CustodyError("custody cluster anchor could not be read")
        return _object(result.stdout)

    custody.anchors(read)


def _custody_master(
    custody: ProvisionCustody, cpu: SecretClient | None, master: str
) -> None:
    if cpu is None:
        raise CustodyError("custody master source requires the CPU scope")
    result = cpu.execute(
        [
            *cpu.scope.command,
            "-n",
            cpu.scope.namespace,
            "get",
            "secret",
            custody.binding.master.secret_name,
            "-o",
            "json",
            "--request-timeout=15s",
        ]
    )
    if result.returncode:
        raise CustodyError("custody master source could not be read")
    custody.master(_object(result.stdout), master.encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-context", default="")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--hyperpod-cluster", required=True)
    parser.add_argument("--master-file", type=Path, required=True)
    parser.add_argument("--secret-name", required=True)
    parser.add_argument("--rotate-node", default="")
    parser.add_argument("--cpu-kubeconfig", default="")
    parser.add_argument("--cpu-context", default="")
    parser.add_argument("--cpu-namespace", required=True)
    parser.add_argument("--custody-request", type=Path)
    parser.add_argument("--custody-trust-sha256", default="")
    parser.add_argument("--custody-input-sha256", default="")
    parser.add_argument("--custody-activation-state", type=Path)
    parser.add_argument("--custody-activation-sha256", default="")
    arguments = parser.parse_args()
    gpu_command = ["kubectl"]
    if arguments.gpu_context:
        gpu_command += ["--context", arguments.gpu_context]
    cpu_command = ["kubectl"]
    if arguments.cpu_kubeconfig:
        cpu_command += ["--kubeconfig", arguments.cpu_kubeconfig]
    if arguments.cpu_context:
        cpu_command += ["--context", arguments.cpu_context]
    cpu = (
        Scope(tuple(cpu_command), arguments.cpu_namespace, arguments.secret_name)
        if arguments.cpu_kubeconfig or arguments.cpu_context
        else None
    )
    try:
        if bool(arguments.custody_request) != bool(arguments.custody_trust_sha256):
            raise CustodyError("custody request requires an external trust pin")
        if arguments.custody_input_sha256 and arguments.custody_request is None:
            raise CustodyError("custody input pin requires an explicit request")
        if bool(arguments.custody_activation_state) != bool(
            arguments.custody_activation_sha256
        ):
            raise CustodyError("custody activation requires a complete state binding")
        if (
            arguments.custody_activation_state is not None
            and arguments.custody_request is None
        ):
            raise CustodyError("custody activation requires an explicit request")
        custody = (
            load_node_key_custody_request(
                arguments.custody_request,
                arguments.custody_trust_sha256,
                **(
                    {
                        "activation_state": arguments.custody_activation_state,
                        "activation_state_sha256": arguments.custody_activation_sha256,
                    }
                    if arguments.custody_activation_state is not None
                    else {}
                ),
                **(
                    {"expected_input_sha256": arguments.custody_input_sha256}
                    if arguments.custody_input_sha256
                    else {}
                ),
            )
            if arguments.custody_request is not None
            else None
        )
        count = provision(
            Scope(tuple(gpu_command), arguments.namespace, arguments.secret_name),
            cpu,
            cluster_id=arguments.cluster_id,
            hyperpod_cluster=arguments.hyperpod_cluster,
            master_file=arguments.master_file,
            rotate_node=arguments.rotate_node,
            expected_nodes=_expected_nodes(
                os.environ.get("GPU_FAULT_NODE_KEY_EXPECTED_NODES_JSON")
            ),
            custody=custody,
        )
    except (ProvisionError, CustodyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"ERROR: node-key synchronization failed ({type(exc).__name__})",
            file=sys.stderr,
        )
        return 1
    print(f"provisioned {count} node action key(s)")
    if cpu is not None:
        print("synchronized node action keys on the control plane")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
