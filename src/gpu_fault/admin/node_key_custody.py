"""Prospective custody capture called by the real node-key provisioning path."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from gpu_fault.admin.deadlines import remaining_timeout
from gpu_fault.admin.node_key_custody_chain import (
    authorize_now,
    planned_keys,
    validate_transition,
    verify_chain,
)
from gpu_fault.admin.node_key_custody_crypto import (
    CustodyCrypto,
    parse,
    private_directory,
    read_regular,
    write_once,
)
from gpu_fault.admin.node_key_custody_models import (
    Authorization,
    Chain,
    Completed,
    CustodyError,
    Identity,
    KeyMap,
    Signed,
    Started,
    Statement,
    Transaction,
    Trust,
    canonical,
    statement_sha256,
)
from gpu_fault.admin.node_key_custody_writer import ActivationWriterGuard


class ProvisionInputs(Statement):
    trust: Identity
    authorization: Identity
    release_manifest: Identity
    previous_chain: Identity | None
    state_directory: Identity
    retired_key_file: Identity | None


class Snapshot(Protocol):
    @property
    def document(self) -> dict[str, Any]: ...
    @property
    def data(self) -> dict[str, str]: ...
    @property
    def uid(self) -> str: ...
    @property
    def version(self) -> str: ...


def provisioning_input_identity(path: Path, trust_sha256: str) -> str:
    request = parse(ProvisionInputs, read_regular(path))
    trust_path = Path(request.trust)
    trust_raw = read_regular(trust_path)
    if hashlib.sha256(trust_raw).hexdigest() != trust_sha256:
        raise CustodyError("custody trust differs from the external input pin")
    trust = parse(Trust, trust_raw)
    files = {
        "request": path,
        "trust": trust_path,
        "authorization": Path(request.authorization),
        "release_manifest": Path(request.release_manifest),
        **{
            role: trust_path.parent / getattr(trust, role).public_key
            for role in ("approval", "provisioner", "witness")
        },
    }
    if request.previous_chain is not None:
        files["previous_chain"] = Path(request.previous_chain)
    values = {
        name: hashlib.sha256(read_regular(value)).hexdigest()
        for name, value in files.items()
    }
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def producer_identity(script: Path) -> str:
    admin = Path(__file__).parent
    files = [
        script,
        script.with_name("provision-node-action-keys.sh"),
        admin / "node_key_proof.py",
        admin / "release_engine.py",
        admin.parent / "node_action_keys.py",
        *sorted(admin.glob("node_key_custody*.py")),
    ]
    values = {
        path.name: hashlib.sha256(read_regular(path)).hexdigest() for path in files
    }
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def witness_source_identity(repository_root: Path) -> str:
    directory = repository_root / "scripts/e2e/regional"
    files = [
        *sorted(directory.glob("auth015_*.py")),
        directory / "identity_acceptance_auth.py",
        directory / "identity_acceptance_common.py",
        directory / "run_identity_acceptance.py",
    ]
    values = {
        path.name: hashlib.sha256(read_regular(path)).hexdigest() for path in files
    }
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def key_digests(data: dict[str, str]) -> dict[str, str]:
    try:
        result = {}
        for node, encoded in data.items():
            value = base64.b64decode(encoded, validate=True)
            if len(value) < 32 or value.strip() != value:
                raise ValueError
            result[node] = hashlib.sha256(value).hexdigest()
        return result
    except (TypeError, ValueError):
        raise CustodyError(
            "custody key bytes are invalid or whitespace-normalized"
        ) from None


def other_keys_sha256(data: dict[str, str], nodes: dict[str, str]) -> str:
    values = key_digests(
        {node: value for node, value in data.items() if node not in nodes}
    )
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def reject_material_in_process(values: list[bytes]) -> None:
    destinations = [*sys.argv, *os.environ.values()]
    for value in values:
        forms = (value.decode("utf-8"), base64.b64encode(value).decode("ascii"))
        if any(form in destination for form in forms for destination in destinations):
            raise CustodyError(
                "custody forbids credential material in environment or argv"
            )


def namespace_uid(document: dict[str, Any], name: str) -> str:
    try:
        metadata = document["metadata"]
        uid = metadata["uid"]
        if (
            document.get("apiVersion") != "v1"
            or document.get("kind") != "Namespace"
            or metadata.get("name") != name
            or metadata.get("deletionTimestamp")
            or not isinstance(uid, str)
            or not uid
        ):
            raise ValueError
        return uid
    except (KeyError, TypeError, ValueError):
        raise CustodyError("custody namespace or cluster anchor is invalid") from None


class ProvisionCustody:
    def __init__(
        self,
        inputs: ProvisionInputs,
        crypto: CustodyCrypto,
        authorization: Signed[Authorization],
        previous: Chain | None,
    ) -> None:
        self.inputs = inputs
        self.crypto = crypto
        self.authorization = authorization
        self.previous = previous
        self.statement = crypto.verify(authorization, "approval")
        authorize_now(self.statement, datetime.now(timezone.utc))
        self.binding = self.statement.binding
        self.predecessor = (
            verify_chain(
                previous, crypto, binding=self.binding, require_activation=True
            )
            if previous is not None
            else None
        )
        if (
            (self.predecessor is None) != (self.statement.purpose == "install")
            or self.predecessor is not None
            and self.statement.previous_receipt_sha256
            != statement_sha256(self.predecessor)
        ):
            raise CustodyError("custody request lacks its signed predecessor")
        self.directory = Path(inputs.state_directory)
        private_directory(self.directory)
        self.prefix = self.directory / self.statement.transaction_id
        if any(self.directory.glob(self.statement.transaction_id + ".*")):
            raise CustodyError(
                "custody transaction already started; reconciliation required"
            )
        if (
            hashlib.sha256(read_regular(Path(inputs.release_manifest))).hexdigest()
            != self.binding.release.manifest_sha256
        ):
            raise CustodyError("custody release differs from the authorized manifest")
        if (inputs.retired_key_file is not None) != (
            self.statement.purpose == "rotate"
        ):
            raise CustodyError(
                "rotation requires an explicit private retired-key destination"
            )
        if inputs.retired_key_file is not None:
            retired = Path(inputs.retired_key_file).absolute()
            private_directory(retired.parent)
            if retired.is_relative_to(self.directory.absolute()) or {
                "artifacts",
                "dist",
            } & set(retired.parts):
                raise CustodyError(
                    "retired key must be outside custody evidence and artifacts"
                )
        self.started: Signed[Started] | None = None
        self.master_verified = False
        self._master_material = b""
        self.activation_guard: ActivationWriterGuard | None = None
        self.activation_wave_reader: Callable[[str], dict[str, Any]] | None = None

    @classmethod
    def load(
        cls,
        path: Path,
        trust_sha256: str,
        *,
        crypto_factory: Callable[..., CustodyCrypto] = CustodyCrypto,
        expected_input_sha256: str | None = None,
        activation_state: Path | None = None,
        activation_state_sha256: str | None = None,
    ) -> ProvisionCustody:
        identity = provisioning_input_identity(path, trust_sha256)
        if expected_input_sha256 is not None and identity != expected_input_sha256:
            raise CustodyError(
                "custody inputs differ from the administrator's pinned request"
            )
        inputs = parse(ProvisionInputs, read_regular(path))
        crypto = crypto_factory(Path(inputs.trust), trust_sha256)
        authorization = parse(
            Signed[Authorization], read_regular(Path(inputs.authorization))
        )
        previous = (
            parse(Chain, read_regular(Path(inputs.previous_chain)))
            if inputs.previous_chain is not None
            else None
        )
        result = cls(inputs, crypto, authorization, previous)
        if (activation_state is None) != (activation_state_sha256 is None):
            raise CustodyError("custody activation requires its complete state binding")
        if activation_state is not None and activation_state_sha256 is not None:
            result.activation_guard = ActivationWriterGuard(
                activation_state,
                activation_state_sha256,
                authorization,
                now=lambda: datetime.now(timezone.utc),
            )
        if provisioning_input_identity(path, trust_sha256) != identity:
            raise CustodyError("custody inputs changed while loading")
        return result

    def command_seconds(self, maximum: float, *, mutating: bool = False) -> float:
        authorize_now(self.statement, datetime.now(timezone.utc))
        seconds = remaining_timeout(maximum)
        if self.activation_guard is not None:
            if mutating:
                if self.activation_wave_reader is None:
                    raise CustodyError("custody writer has no owned wave reader")
                remaining = self.activation_guard.before_write(
                    self.activation_wave_reader
                )
            else:
                remaining = self.activation_guard.check()
            seconds = min(seconds, remaining)
        return seconds

    def bind(
        self,
        *,
        script: Path,
        cluster_id: str,
        hyperpod_cluster: str,
        nodes: dict[str, str],
        gpu_namespace: str,
        cpu_namespace: str | None,
        rotate_node: str,
    ) -> None:
        site = self.binding.site
        if (
            self.statement.producer_sha256 != producer_identity(script)
            or site.cluster_id != cluster_id
            or site.hyperpod_cluster != hyperpod_cluster
            or site.namespace != gpu_namespace
            or site.namespace != cpu_namespace
            or self.binding.nodes != nodes
            or (self.statement.rotate_node or "") != rotate_node
        ):
            raise CustodyError("custody provisioning inputs differ from authorization")

    def anchors(self, read: Callable[[str, str], dict[str, Any]]) -> None:
        site = self.binding.site
        for plane in ("cpu", "gpu"):
            if namespace_uid(read(plane, "kube-system"), "kube-system") != getattr(
                site, plane + "_cluster_uid"
            ) or namespace_uid(read(plane, site.namespace), site.namespace) != getattr(
                site, plane + "_namespace_uid"
            ):
                raise CustodyError("custody live cluster or namespace identity differs")

    def master(self, document: dict[str, Any], material: bytes) -> None:
        source = self.binding.master
        try:
            metadata = document["metadata"]
            encoded = document["data"][source.data_key]
            value = base64.b64decode(encoded, validate=True)
            if (
                document.get("apiVersion") != "v1"
                or document.get("kind") != "Secret"
                or document.get("type") != "Opaque"
                or metadata.get("namespace") != self.binding.site.namespace
                or metadata.get("name") != source.secret_name
                or metadata.get("uid") != source.secret_uid
                or not metadata.get("resourceVersion")
                or metadata.get("deletionTimestamp")
                or "stringData" in document
                or value != material
                or hashlib.sha256(value).hexdigest() != source.sha256
                or len(value) < 32
                or value.strip() != value
            ):
                raise ValueError
            reject_material_in_process([material])
        except (KeyError, TypeError, ValueError, UnicodeError):
            raise CustodyError(
                "custody master source identity or bytes differ"
            ) from None
        self.master_verified = True
        self._master_material = material

    def transfer(self, document: dict[str, Any], *, gpu: bool) -> None:
        self.command_seconds(300)
        if self.started is None:
            raise CustodyError("custody transfer lacks its signed start")
        expected = {
            node: state.sha256
            for node, state in self.started.statement.planned_keys.items()
        }
        actual = key_digests(document["data"])
        if (
            gpu
            and actual != expected
            or any(actual.get(node) != value for node, value in expected.items())
        ):
            raise CustodyError("custody transfer differs from the signed plan")
        if not gpu:
            return
        metadata = json.dumps(document["metadata"]).encode()
        material = [
            self._master_material,
            *(base64.b64decode(value) for value in document["data"].values()),
        ]
        if any(
            value in metadata or base64.b64encode(value) in metadata
            for value in material
        ):
            raise CustodyError("GPU custody forbids credentials in metadata")

    def begin(
        self,
        gpu: Snapshot | None,
        cpu: Snapshot | None,
        desired: dict[str, str],
    ) -> None:
        self.command_seconds(300, mutating=True)
        if not self.master_verified:
            raise CustodyError("custody master source has not been verified")
        if self.started is not None:
            if key_digests(desired) != {
                node: state.sha256
                for node, state in self.started.statement.planned_keys.items()
            }:
                raise CustodyError("custody plan changed after the signed start")
            return
        nodes = self.binding.nodes
        before = None
        if self.predecessor is None:
            if gpu is not None or cpu is not None and set(cpu.data) & set(nodes):
                raise CustodyError(
                    "pre-existing installation requires new authorized provenance"
                )
        else:
            previous = self.predecessor.completed.statement
            if (
                gpu is None
                or cpu is None
                or gpu.uid != previous.gpu.uid
                or cpu.uid != previous.cpu.uid
                or key_digests(gpu.data)
                != {node: state.sha256 for node, state in previous.gpu.keys.items()}
                or any(cpu.data.get(node) != value for node, value in gpu.data.items())
            ):
                raise CustodyError(
                    "custody live keys differ from the signed predecessor"
                )
            before = previous.gpu.model_copy(update={"resource_version": gpu.version})
        planned = planned_keys(
            self.binding,
            key_digests(desired),
            self.predecessor,
            self.statement.rotate_node,
        )
        if len({state.sha256 for state in planned.values()}) != len(planned) or any(
            state.sha256 == self.binding.master.sha256 for state in planned.values()
        ):
            raise CustodyError(
                "custody requires distinct node-scoped keys, never the master"
            )
        if self.predecessor is not None and {
            node
            for node, state in planned.items()
            if state.sha256
            != self.predecessor.completed.statement.gpu.keys[node].sha256
        } != {self.statement.rotate_node}:
            raise CustodyError(
                "custody rotation must change exactly its authorized node"
            )
        reject_material_in_process(
            [base64.b64decode(value) for value in desired.values()]
        )
        started = Started(
            authorization_sha256=statement_sha256(self.statement),
            observed_at=datetime.now(timezone.utc),
            before_gpu=before,
            before_cpu_uid=cpu.uid if cpu is not None else None,
            cpu_other_keys_sha256=other_keys_sha256(
                cpu.data if cpu is not None else {}, nodes
            ),
            planned_keys=planned,
        )
        material = [
            self._master_material,
            *(base64.b64decode(value) for value in desired.values()),
            *(
                base64.b64decode(value)
                for value in (gpu.data if gpu is not None else {}).values()
            ),
        ]
        receipt_bytes = canonical(self.authorization) + canonical(started)
        if any(
            value in receipt_bytes or base64.b64encode(value) in receipt_bytes
            for value in material
        ):
            raise CustodyError("custody receipt identity contains credential material")
        # Persist independently verifiable intent before either Secret can change.
        self.started = self.crypto.sign(started, "provisioner")
        self.command_seconds(300, mutating=True)
        write_once(Path(str(self.prefix) + ".started.json"), canonical(self.started))
        if self.inputs.retired_key_file is not None:
            if gpu is None or self.statement.rotate_node is None:
                raise CustodyError("retired-key capture lacks the original node")
            write_once(
                Path(self.inputs.retired_key_file),
                base64.b64decode(gpu.data[self.statement.rotate_node], validate=True),
            )

    def finish(self, gpu: Snapshot, cpu: Snapshot) -> None:
        if self.started is None:
            raise CustodyError("custody completion lacks a prospective signed start")
        self.command_seconds(300, mutating=True)
        planned = self.started.statement.planned_keys
        expected = {node: state.sha256 for node, state in planned.items()}
        if (
            key_digests(gpu.data) != expected
            or {node: key_digests(cpu.data).get(node) for node in planned} != expected
        ):
            raise CustodyError(
                "custody final readback differs from the signed key plan"
            )
        complete = Completed(
            started_sha256=statement_sha256(self.started.statement),
            observed_at=datetime.now(timezone.utc),
            gpu=KeyMap(uid=gpu.uid, resource_version=gpu.version, keys=planned),
            cpu=KeyMap(uid=cpu.uid, resource_version=cpu.version, keys=planned),
            cpu_other_keys_sha256=other_keys_sha256(cpu.data, self.binding.nodes),
        )
        validate_transition(
            self.statement, self.started.statement, complete, self.predecessor
        )
        signed = self.crypto.sign(complete, "provisioner")
        self.command_seconds(300, mutating=True)
        transaction = Transaction(
            authorization=self.authorization, started=self.started, completed=signed
        )
        chain = Chain(
            transactions=[
                *(self.previous.transactions if self.previous else []),
                transaction,
            ]
        )
        write_once(Path(str(self.prefix) + ".chain.json"), canonical(chain))
