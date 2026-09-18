"""Pinned offline ECDSA verification and KMS signing on the trusted deploy host."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from gpu_fault.admin.execution import run_command
from gpu_fault.admin.node_key_custody_models import (
    CustodyError,
    PublicAuthority,
    Signed,
    Statement,
    Trust,
    canonical,
)

MAX_FILE_BYTES = 4 * 1024 * 1024
_T = TypeVar("_T", bound=Statement)


def private_command_environment() -> dict[str, str]:
    # Authentication uses role/config-file references, never inherited raw keys.
    names = (
        "HOME",
        "PATH",
        "KUBECONFIG",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_PROFILE",
        "AWS_EC2_METADATA_DISABLED",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def read_regular(
    path: Path, *, private: bool = False, limit: int = MAX_FILE_BYTES
) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            with os.fdopen(fd, "rb", closefd=False) as handle:
                before = os.fstat(handle.fileno())
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or private
                    and (
                        stat.S_IMODE(before.st_mode) != 0o600
                        or before.st_uid != os.geteuid()
                    )
                ):
                    raise CustodyError("custody input is not a controlled regular file")
                value = handle.read(limit + 1)
                after = os.fstat(handle.fileno())
                if (
                    not value
                    or len(value) > limit
                    or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                ):
                    raise CustodyError("custody input is empty, oversized or changed")
                return value
        finally:
            os.close(fd)
    except OSError:
        raise CustodyError("custody input cannot be read safely") from None


def private_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.geteuid()
        ):
            raise CustodyError("custody state requires an owned mode-0700 directory")
    except OSError:
        raise CustodyError("custody state directory is unavailable") from None


def write_once(path: Path, content: bytes) -> None:
    private_directory(path.parent)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        # An existing or partial record is never overwritten or silently adopted.
        raise CustodyError(
            "custody record already exists or could not be sealed"
        ) from None


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for name, value in pairs:
        if name in document:
            raise CustodyError("custody JSON contains duplicate fields")
        document[name] = value
    return document


def parse(model: type[_T], raw: bytes) -> _T:
    try:
        json.loads(raw, object_pairs_hook=_pairs)
        return model.model_validate_json(raw)
    except (ValueError, TypeError):
        raise CustodyError("custody document has an invalid schema") from None


class CustodyCrypto:
    def __init__(
        self,
        trust_path: Path,
        expected_sha256: str,
        *,
        runner: Callable[..., Any] = run_command,
    ) -> None:
        raw = read_regular(trust_path)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise CustodyError("custody trust differs from the external trust pin")
        self.trust = parse(Trust, raw)
        self.trust_sha256 = expected_sha256
        self.runner = runner
        self.public_keys: dict[str, bytes] = {}
        self.authorities: dict[str, PublicAuthority] = {
            "approval": self.trust.approval,
            "provisioner": self.trust.provisioner,
            "witness": self.trust.witness,
        }
        for authority in self.authorities.values():
            key = read_regular(trust_path.parent / authority.public_key, limit=16384)
            identity = self.public_identity(key)
            if identity != authority.public_key_sha256:
                raise CustodyError("custody public key differs from its pin")
            self.public_keys[identity] = key
        if (
            len(self.public_keys) != 3
            or len({item.kms_key_arn for item in self.authorities.values()}) != 3
        ):
            raise CustodyError("custody authorities must use three independent keys")

    def command(self, arguments: list[str]) -> Any:
        try:
            result = self.runner(
                arguments,
                capture=True,
                timeout_seconds=20,
                environment=private_command_environment(),
            )
        except Exception:
            raise CustodyError("custody signature command could not complete") from None
        if result.returncode:
            raise CustodyError("custody signature command failed")
        return result

    def public_identity(self, pem: bytes) -> str:
        with tempfile.TemporaryDirectory(prefix="node-key-public-") as directory:
            root = Path(directory)
            public, der = root / "public.pem", root / "public.der"
            write_once(public, pem)
            self.command(
                [
                    "openssl",
                    "pkey",
                    "-pubin",
                    "-in",
                    str(public),
                    "-outform",
                    "DER",
                    "-out",
                    str(der),
                ]
            )
            # Canonical SPKI, not PEM spelling, determines independent authority.
            value = read_regular(der, limit=16384)
            if len(value) != 91 or not value.startswith(
                bytes.fromhex("3059301306072a8648ce3d020106082a8648ce3d03010703420004")
            ):
                raise CustodyError("custody requires an ECDSA P-256 public key")
            return hashlib.sha256(value).hexdigest()

    def authority(self, role: str) -> PublicAuthority:
        try:
            return self.authorities[role]
        except KeyError:
            raise CustodyError("unknown custody authority role") from None

    def verify(self, envelope: Signed[_T], role: str) -> _T:
        authority = self.authority(role)
        if envelope.signer_sha256 != authority.public_key_sha256:
            raise CustodyError("custody receipt is signed by the wrong authority")
        try:
            signature = base64.b64decode(envelope.signature, validate=True)
            if not 64 <= len(signature) <= 256:
                raise ValueError
        except ValueError:
            raise CustodyError("custody signature encoding is invalid") from None
        with tempfile.TemporaryDirectory(prefix="node-key-verify-") as directory:
            root = Path(directory)
            public, content, signed = (
                root / "public.pem",
                root / "statement",
                root / "signature",
            )
            write_once(public, self.public_keys[authority.public_key_sha256])
            write_once(content, canonical(envelope.statement))
            write_once(signed, signature)
            self.command(
                [
                    "openssl",
                    "dgst",
                    "-sha256",
                    "-verify",
                    str(public),
                    "-signature",
                    str(signed),
                    str(content),
                ]
            )
        return envelope.statement

    def sign(self, statement: _T, role: str) -> Signed[_T]:
        if role not in {"provisioner", "witness"}:
            raise CustodyError("provisioning cannot issue its own authorization")
        authority = self.authority(role)
        with tempfile.TemporaryDirectory(prefix="node-key-sign-") as directory:
            digest = Path(directory) / "message.sha256"
            write_once(digest, hashlib.sha256(canonical(statement)).digest())
            result = self.command(
                [
                    "aws",
                    "kms",
                    "sign",
                    "--key-id",
                    authority.kms_key_arn,
                    "--region",
                    authority.kms_key_arn.split(":")[3],
                    "--signing-algorithm",
                    "ECDSA_SHA_256",
                    "--message-type",
                    "DIGEST",
                    "--message",
                    "fileb://" + str(digest),
                    "--output",
                    "json",
                    "--no-cli-pager",
                ]
            )
            try:
                value = json.loads(result.stdout, object_pairs_hook=_pairs)
                if (
                    value["KeyId"] != authority.kms_key_arn
                    or value["SigningAlgorithm"] != "ECDSA_SHA_256"
                ):
                    raise ValueError
                envelope = Signed[_T](
                    statement=statement,
                    signer_sha256=authority.public_key_sha256,
                    signature=value["Signature"],
                )
            except (KeyError, TypeError, ValueError):
                raise CustodyError("custody signing response is invalid") from None
        self.verify(envelope, role)
        return envelope
