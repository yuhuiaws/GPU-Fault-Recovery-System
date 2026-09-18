"""Plan-bound custody inputs; the authority pin must come from outside evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from gpu_fault.admin.node_key_custody_chain import verify_chain
from gpu_fault.admin.node_key_custody_crypto import CustodyCrypto, parse, read_regular
from gpu_fault.admin.node_key_custody_models import (
    Chain,
    CustodyError,
    Identity,
    Statement,
    Transaction,
    Trust,
)


class CustodyDescriptor(Statement):
    trust: Identity
    chain: Identity
    retired_key_file: Identity | None


@dataclass(frozen=True)
class Auth015CustodyInputs:
    descriptor_path: Path
    trust_sha256: str
    expected_identity: dict[str, str] | None = None


def input_files(inputs: Auth015CustodyInputs) -> tuple[CustodyDescriptor, Path, Path]:
    descriptor = parse(CustodyDescriptor, read_regular(inputs.descriptor_path))
    root = inputs.descriptor_path.parent
    return descriptor, root / descriptor.trust, root / descriptor.chain


def custody_input_identity(inputs: Auth015CustodyInputs) -> dict[str, str]:
    descriptor, trust_path, chain_path = input_files(inputs)
    raw = read_regular(trust_path)
    if hashlib.sha256(raw).hexdigest() != inputs.trust_sha256:
        raise CustodyError("AUTH015 custody trust differs from the external pin")
    trust = parse(Trust, raw)
    paths = {
        "descriptor": inputs.descriptor_path,
        "trust": trust_path,
        "chain": chain_path,
        **{
            role: trust_path.parent / getattr(trust, role).public_key
            for role in ("approval", "provisioner", "witness")
        },
    }
    result = {
        name: hashlib.sha256(read_regular(path)).hexdigest()
        for name, path in paths.items()
    }
    if descriptor.retired_key_file is not None:
        result["retired_key"] = hashlib.sha256(
            read_regular(
                inputs.descriptor_path.parent / descriptor.retired_key_file,
                private=True,
                limit=4096,
            )
        ).hexdigest()
    return result


def verify_custody_inputs(
    inputs: Auth015CustodyInputs,
) -> tuple[Chain, Transaction, CustodyCrypto]:
    identity = custody_input_identity(inputs)
    if inputs.expected_identity is not None and identity != inputs.expected_identity:
        raise CustodyError("AUTH015 custody inputs differ from the approved plan")
    _descriptor, trust_path, chain_path = input_files(inputs)
    crypto = CustodyCrypto(trust_path, inputs.trust_sha256)
    chain = parse(Chain, read_regular(chain_path))
    head = verify_chain(chain, crypto)
    if custody_input_identity(inputs) != identity:
        raise CustodyError("AUTH015 custody inputs changed during verification")
    return chain, head, crypto


def retired_key(inputs: Auth015CustodyInputs, chain: Chain) -> str | None:
    descriptor, _trust, _chain = input_files(inputs)
    head = chain.transactions[-1]
    if head.authorization.statement.purpose == "install":
        if descriptor.retired_key_file is not None:
            raise CustodyError("installation proof must not supply a retired key")
        return None
    if descriptor.retired_key_file is None or len(chain.transactions) < 2:
        raise CustodyError("rotation activation requires the controlled retired key")
    value = read_regular(
        inputs.descriptor_path.parent / descriptor.retired_key_file,
        private=True,
        limit=4096,
    )
    node = head.authorization.statement.rotate_node
    previous = chain.transactions[-2]
    if (
        node is None
        or value.strip() != value
        or hashlib.sha256(value).hexdigest()
        != previous.completed.statement.gpu.keys[node].sha256
    ):
        raise CustodyError(
            "retired key does not match the signed activated predecessor"
        )
    try:
        return value.decode("utf-8")
    except UnicodeError:
        raise CustodyError("retired key encoding is invalid") from None
