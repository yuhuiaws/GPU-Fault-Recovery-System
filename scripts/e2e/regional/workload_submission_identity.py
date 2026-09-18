"""Temporary create-only credentials for one named public WORKLOAD submission."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast
from urllib.parse import quote, urlsplit
from uuid import uuid4

from scripts.e2e.regional.fixture_ownership import creation_document
from scripts.e2e.regional.managed_workload_fixture import OWNER_LABEL, RESOURCE_APIS
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
)

_RBAC_API = "rbac.authorization.k8s.io"
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_CONFIG_NAME = "gpu.kubeconfig.json"


def _open_directory(path: Path) -> int:
    """Resolve directory components without traversing even an ancestor symlink."""
    descriptor = os.open(path.anchor, _DIRECTORY_FLAGS)
    try:
        for part in path.parts[1:]:
            if part == "..":
                raise RegionalFixtureError("submission directory cannot contain '..'")
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


@dataclass
class _Creation:
    document: dict[str, Any]
    uid: str | None = None


class CreateOnlySubmissionIdentity:
    """One-shot, GPU-only impersonation; the existing case directory must exist.

    The underlying deployment credential is preserved, not delegated to a new
    ServiceAccount. Callers must pass this kubeconfig to the unmodified public
    CLI, bind its manifest to resource_name, and must not archive the kubeconfig.
    Permission changes during use remain outside this context manager's authority.
    """

    def __init__(
        self,
        regional: RegionalLiveFixture,
        *,
        directory: Path,
        owner: str,
        resource: str,
        resource_name: str,
    ) -> None:
        if (
            resource not in {"job", "pytorchjob", "jobset"}
            or re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,61}[A-Za-z0-9])?", owner)
            is None
            or not isinstance(resource_name, str)
            or len(resource_name) > 253
            or re.fullmatch(
                r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?"
                r"(?:\.[a-z0-9](?:[-a-z0-9]*[a-z0-9])?)*",
                resource_name,
            )
            is None
        ):
            raise RegionalFixtureError("submission resource, name or owner is invalid")
        self.regional = regional
        self._namespace = regional.settings.namespace
        self._context = regional.settings.gpu_context
        self._original_config = regional.settings.gpu_kubeconfig
        self._owner = owner
        self._resource_name = resource_name
        api, self._plural = RESOURCE_APIS[resource]
        self._group = api.split("/")[1]
        nonce = uuid4().hex
        self.subject = f"gpu-fault-acceptance-submit-{owner}-{nonce}"
        self._name = f"gpu-fault-submit-{nonce}"
        self._directory = directory.absolute()
        self._private_name = f".gpu-fault-submit-{nonce}"
        self.kubeconfig = self._directory / self._private_name / _CONFIG_NAME
        self._parent_fd: int | None = None
        self._private_fd: int | None = None
        self._config_fd: int | None = None
        self._private_stat: os.stat_result | None = None
        self._config_stat: os.stat_result | None = None
        self._private_created = False
        self._creations: list[_Creation] = []
        self._entered = False

    def __enter__(self) -> Self:
        if self._entered:
            raise RegionalFixtureError("submission identity cannot be reused")
        self._entered = True
        try:
            self._prepare_config()
            metadata = {
                "name": self._name,
                "namespace": self._namespace,
                "labels": {OWNER_LABEL: self._owner},
            }
            self._create(
                {
                    "apiVersion": f"{_RBAC_API}/v1",
                    "kind": "Role",
                    "metadata": metadata,
                    "rules": [
                        {
                            "apiGroups": [self._group],
                            "resources": [self._plural],
                            "verbs": ["get", "create"],
                        }
                    ],
                }
            )
            self._create(
                {
                    "apiVersion": f"{_RBAC_API}/v1",
                    "kind": "RoleBinding",
                    "metadata": metadata,
                    "roleRef": {
                        "apiGroup": _RBAC_API,
                        "kind": "Role",
                        "name": self._name,
                    },
                    "subjects": [
                        {"kind": "User", "apiGroup": _RBAC_API, "name": self.subject}
                    ],
                }
            )
            self._verify_permissions()
            return self
        except BaseException:
            self._cleanup()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._cleanup()

    def _command(
        self,
        *arguments: str,
        impersonated: bool = False,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self.regional.run(
                [
                    "kubectl",
                    "--kubeconfig",
                    str(self.kubeconfig if impersonated else self._original_config),
                    "--context",
                    self._context,
                    "-n",
                    self._namespace,
                    *arguments,
                ],
                input_text=input_text,
                check=False,
                timeout=60,
            )
        except Exception:
            # Exec plugins and transport exceptions can contain raw credentials.
            raise RegionalFixtureError("submission identity transport failed") from None

    def _output(
        self,
        *arguments: str,
        impersonated: bool = False,
        input_text: str | None = None,
    ) -> str:
        result = self._command(
            *arguments, impersonated=impersonated, input_text=input_text
        )
        if result.returncode != 0 or result.stderr:
            raise RegionalFixtureError("submission identity command was not confirmed")
        return result.stdout

    def _prepare_config(self) -> None:
        try:
            self._parent_fd = _open_directory(self._directory)
            os.mkdir(self._private_name, mode=0o700, dir_fd=self._parent_fd)
            self._private_created = True
            self._private_fd = os.open(
                self._private_name, _DIRECTORY_FLAGS, dir_fd=self._parent_fd
            )
            self._private_stat = os.fstat(self._private_fd)
            if stat.S_IMODE(self._private_stat.st_mode) != 0o700:
                raise RegionalFixtureError("submission directory is not private")
            value = creation_document(
                self._output(
                    "config", "view", "--raw", "--minify", "--flatten", "-o", "json"
                )
            )
            self._impersonate(value)
            descriptor = os.open(
                _CONFIG_NAME,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=self._private_fd,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                # Pin the inode through cleanup, including unlink/recreate races.
                self._config_fd = os.dup(stream.fileno())
                self._config_stat = os.fstat(self._config_fd)
                json.dump(value, stream, allow_nan=False)
        except Exception:
            raise RegionalFixtureError(
                "private GPU submission kubeconfig could not be prepared"
            ) from None

    def _impersonate(self, value: dict[str, Any]) -> None:
        entries = [value.get(key) for key in ("contexts", "clusters", "users")]
        if (
            value.get("apiVersion") != "v1"
            or value.get("kind") != "Config"
            or value.get("current-context") != self._context
            or any(
                not isinstance(items, list)
                or len(items) != 1
                or not isinstance(items[0], dict)
                for items in entries
            )
        ):
            raise RegionalFixtureError("GPU kubeconfig is not a singular context")
        context, cluster, user = (
            items[0] for items in cast(list[list[dict[str, Any]]], entries)
        )
        scope, endpoint, auth = (
            context.get("context"),
            cluster.get("cluster"),
            user.get("user"),
        )
        if (
            context.get("name") != self._context
            or not isinstance(scope, dict)
            or not isinstance(endpoint, dict)
            or not isinstance(auth, dict)
            or not cluster.get("name")
            or not user.get("name")
            or scope.get("cluster") != cluster["name"]
            or scope.get("user") != user["name"]
            or endpoint.get("insecure-skip-tls-verify") not in (None, False)
            or urlsplit(endpoint.get("server", "")).scheme != "https"
            or not endpoint.get("certificate-authority-data")
            or endpoint.get("certificate-authority")
            or auth.get("client-certificate")
            or auth.get("client-key")
        ):
            raise RegionalFixtureError("GPU kubeconfig identity or TLS is invalid")
        if any(key in auth for key in ("as", "as-groups", "as-uid", "as-user-extra")):
            raise RegionalFixtureError("GPU kubeconfig already contains impersonation")
        auth["as"] = self.subject
        auth["as-groups"] = ["system:authenticated"]

    def _validate(self, creation: _Creation, value: dict[str, Any]) -> tuple[str, str]:
        expected = creation.document
        metadata = value.get("metadata")
        if (
            value.get("apiVersion") != expected["apiVersion"]
            or value.get("kind") != expected["kind"]
            or not isinstance(metadata, dict)
            or metadata.get("name") != self._name
            or metadata.get("namespace") != self._namespace
            or not isinstance(metadata.get("labels"), dict)
            or metadata["labels"].get(OWNER_LABEL) != self._owner
            or any(
                not isinstance(metadata.get(key), str) or not metadata[key]
                for key in ("uid", "resourceVersion")
            )
            or creation.uid is not None
            and metadata["uid"] != creation.uid
        ):
            raise RegionalFixtureError("RBAC identity, owner or UID changed")
        return metadata["uid"], metadata["resourceVersion"]

    def _create(self, document: dict[str, Any]) -> None:
        creation = _Creation(document)
        self._creations.append(creation)
        value = creation_document(
            self._output(
                "create", "-f", "-", "-o", "json", input_text=json.dumps(document)
            )
        )
        creation.uid, _version = self._validate(creation, value)
        if value["metadata"].get("deletionTimestamp") or any(
            value.get(key) != document[key]
            for key in ("rules", "subjects", "roleRef")
            if key in document
        ):
            raise RegionalFixtureError("RBAC creation changed its declared intent")

    def _verify_permissions(self) -> None:
        for verb in ("get", "create", "patch", "update", "delete"):
            target = f"{self._plural}.{self._group}"
            if verb != "create":
                # An unnamed denial misses resourceNames-scoped grants.
                target += f"/{self._resource_name}"
            result = self._command(
                "auth",
                "can-i",
                verb,
                target,
                impersonated=True,
            )
            expected = (0, "yes\n") if verb in {"get", "create"} else (1, "no\n")
            if (result.returncode, result.stdout) != expected or result.stderr:
                raise RegionalFixtureError(
                    f"submission identity permission {verb} was not proven"
                )

    def _read(self, creation: _Creation) -> dict[str, Any] | None:
        output = self._output(
            "get",
            f"{creation.document['kind'].lower()}s.{_RBAC_API}",
            self._name,
            "--ignore-not-found",
            "-o",
            "json",
        )
        return creation_document(output) if output else None

    def _cleanup(self) -> None:
        errors: list[str] = []
        try:
            self._cleanup_rbac(errors)
        except BaseException as error:
            error.add_note(
                "submission identity cleanup interrupted; RBAC cleanup uncertain"
            )
            raise
        finally:
            self._cleanup_files(errors)
        if errors:
            raise RegionalFixtureError(
                "submission identity cleanup uncertain: " + "; ".join(errors)
            ) from None

    def _cleanup_rbac(self, errors: list[str]) -> None:
        for creation in reversed(self._creations):
            kind = creation.document["kind"]
            if creation.uid is None:
                errors.append(f"{kind}/{self._name}: creation ACK missing")
                continue
            try:
                current = self._read(creation)
                if current is None:
                    continue
                uid, version = self._validate(creation, current)
                self._output(
                    "delete",
                    "--raw",
                    f"/apis/{_RBAC_API}/v1/namespaces/"
                    f"{quote(self._namespace, safe='')}/{kind.lower()}s/{self._name}",
                    "-f",
                    "-",
                    input_text=json.dumps(
                        {
                            "apiVersion": "v1",
                            "kind": "DeleteOptions",
                            "preconditions": {"uid": uid, "resourceVersion": version},
                        }
                    ),
                )
                if self._read(creation) is not None:
                    raise RegionalFixtureError("RBAC deletion was not confirmed")
            except Exception:
                errors.append(f"{kind}/{self._name}: owned deletion unproven")

    def _cleanup_files(self, errors: list[str]) -> None:
        if self._private_fd is not None:
            try:
                if self._config_stat is not None:
                    current = os.stat(
                        _CONFIG_NAME, dir_fd=self._private_fd, follow_symlinks=False
                    )
                    if not stat.S_ISREG(current.st_mode) or not _same_inode(
                        current, self._config_stat
                    ):
                        raise RegionalFixtureError("private kubeconfig was replaced")
                    os.unlink(_CONFIG_NAME, dir_fd=self._private_fd)
            except FileNotFoundError:
                pass
            except Exception:
                errors.append("private kubeconfig removal unproven")
            finally:
                if self._config_fd is not None:
                    os.close(self._config_fd)
                    self._config_fd = None
                os.close(self._private_fd)
                self._private_fd = None
        if self._parent_fd is not None:
            try:
                if self._private_stat is not None:
                    current = os.stat(
                        self._private_name,
                        dir_fd=self._parent_fd,
                        follow_symlinks=False,
                    )
                    if not _same_inode(current, self._private_stat):
                        raise RegionalFixtureError("private directory was replaced")
                    os.rmdir(self._private_name, dir_fd=self._parent_fd)
                elif self._private_created:
                    errors.append("private directory creation cleanup unproven")
            except FileNotFoundError:
                pass
            except Exception:
                errors.append("private directory removal unproven")
            finally:
                os.close(self._parent_fd)
                self._parent_fd = None
