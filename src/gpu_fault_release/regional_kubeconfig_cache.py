"""Private, expiring AWS EKS token copies for bounded direct kubectl calls.

Generic exec plugins retain kubectl's native protocol. Child scripts and calls
that can outlive the token use the original kubeconfig; process globals and
source files are never rewritten. Credential fetches use the shared supervised
command runner and never include plugin output in diagnostics.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, MutableMapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.execution import run_command
from gpu_fault_release.regional_release_config import ReleaseConfig, ReleaseError

TOKEN_CACHE_ENV = "GPU_FAULT_KUBECONFIG_TOKEN_CACHE"
DEFAULT_LEAD_SECONDS = 180.0
EXEC_TIMEOUT_SECONDS = 60.0
MAX_CREDENTIAL_OUTPUT_BYTES = 8 * 1024 * 1024
# kubeconfig file references are relative to the file's own directory; the copy
# lives elsewhere, so they are made absolute against the source directory.
RELATIVE_PATH_KEYS = (
    "certificate-authority",
    "client-certificate",
    "client-key",
    "tokenFile",
)
_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})


def token_cache_enabled(environ: MutableMapping[str, str] | None = None) -> bool:
    value = (os.environ if environ is None else environ).get(TOKEN_CACHE_ENV, "")
    return value.strip().lower() not in _DISABLED_VALUES


def _parse_expiry(value: object, now: float) -> float | None:
    if value is None:
        return None
    try:
        if not isinstance(value, str):
            raise ValueError
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError
        expires = parsed.timestamp()
        if expires <= now + EXEC_TIMEOUT_SECONDS:
            raise ValueError
    except (ValueError, OverflowError):
        raise ReleaseError("exec credential expiry is invalid or too soon") from None
    return expires


def _standard_eks_exec(user: dict[str, Any]) -> bool:
    spec = user.get("exec")
    if set(user) != {"exec"} or not isinstance(spec, dict):
        return False
    arguments = spec.get("args")
    return (
        # Bare aws is the generated EKS form and passes through the budget
        # shim. Absolute/custom executables retain kubectl's native protocol.
        spec.get("command") == "aws"
        and isinstance(arguments, list)
        and all(isinstance(item, str) for item in arguments)
        and any(
            arguments[index : index + 2] == ["eks", "get-token"]
            for index in range(len(arguments) - 1)
        )
        and spec.get("provideClusterInfo", False) is False
        and spec.get("interactiveMode") in (None, "Never", "IfAvailable")
        and spec.get("apiVersion")
        in (
            "client.authentication.k8s.io/v1",
            "client.authentication.k8s.io/v1beta1",
        )
        and not set(spec).difference(
            {
                "command",
                "args",
                "env",
                "apiVersion",
                "interactiveMode",
                "provideClusterInfo",
            }
        )
    )


def _absolutize_paths(section: dict[str, Any], base: Path) -> None:
    for key in RELATIVE_PATH_KEYS:
        value = section.get(key)
        if isinstance(value, str) and value and not os.path.isabs(value):
            section[key] = str((base / value).resolve())


class KubeconfigTokenCache:
    """One kubeconfig file's token-cached copy."""

    def __init__(
        self,
        source: Path,
        *,
        directory: Path,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.source = source.resolve()
        self._now = now
        self._lock = threading.Lock()
        try:
            text = self.source.read_text(encoding="utf-8")
            document = yaml.safe_load(text)
        except (OSError, yaml.YAMLError):
            raise ReleaseError(f"cannot read kubeconfig {self.source}") from None
        if not isinstance(document, dict):
            raise ReleaseError(f"kubeconfig {self.source} is not a mapping")
        self._source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self._document = document
        self._exec_users: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for entry in document.get("users") or []:
            user = entry.get("user") if isinstance(entry, dict) else None
            if isinstance(user, dict):
                _absolutize_paths(user, self.source.parent)
                exec_spec = user.get("exec")
                if isinstance(exec_spec, dict):
                    command = exec_spec.get("command")
                    if isinstance(command, str) and "/" in command:
                        exec_spec["command"] = str(
                            (self.source.parent / command).resolve()
                        )
                    if _standard_eks_exec(user):
                        self._exec_users.append((user, exec_spec))
        for entry in document.get("clusters") or []:
            cluster = entry.get("cluster") if isinstance(entry, dict) else None
            if isinstance(cluster, dict):
                _absolutize_paths(cluster, self.source.parent)
        digest = hashlib.sha256(str(self.source).encode("utf-8")).hexdigest()[:8]
        self.path = directory / f"{digest}-{self.source.name}"
        self.earliest_expiry: float | None = None
        if self._exec_users:
            self._refresh()

    @property
    def has_exec_users(self) -> bool:
        return bool(self._exec_users)

    def refresh_if_needed(self, lead_seconds: float = DEFAULT_LEAD_SECONDS) -> bool:
        """Re-fetch every token when one expires within ``lead_seconds``."""

        with self._lock:
            try:
                current = hashlib.sha256(self.source.read_bytes()).hexdigest()
            except OSError:
                raise ReleaseError("source kubeconfig is no longer readable") from None
            if current != self._source_sha256:
                raise ReleaseError("source kubeconfig changed during release")
            if self.earliest_expiry is None:
                return False
            if self.earliest_expiry - self._now() > lead_seconds:
                return False
            self._refresh()
            return True

    def _refresh(self) -> None:
        # Publish all users together. A failed refresh never leaves the old
        # token eligible for reuse on a later command.
        self.earliest_expiry = 0.0
        earliest: float | None = None
        updated: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for user, exec_spec in self._exec_users:
            token, expiry = _fetch_exec_credential(exec_spec, now=self._now)
            updated.append(
                (user, {"token": token} if expiry is not None else {"exec": exec_spec})
            )
            if expiry is not None:
                earliest = expiry if earliest is None else min(earliest, expiry)
        for user, replacement in updated:
            user.clear()
            user.update(replacement)
        self._write(yaml.safe_dump(self._document, sort_keys=False))
        self.earliest_expiry = earliest

    def _write(self, text: str) -> None:
        handle, temp = tempfile.mkstemp(dir=self.path.parent, prefix=".kubeconfig-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(text)
            os.chmod(temp, 0o600)
            os.replace(temp, self.path)
        except OSError:
            Path(temp).unlink(missing_ok=True)
            raise


def _fetch_exec_credential(
    exec_spec: dict[str, Any], *, now: Callable[[], float]
) -> tuple[str, float | None]:
    command = exec_spec.get("command")
    if not isinstance(command, str) or not command:
        raise ReleaseError("kubeconfig exec entry has no command")
    args = exec_spec.get("args") or []
    if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
        raise ReleaseError("kubeconfig exec arguments are invalid")
    environment = dict(os.environ)
    entries = exec_spec.get("env") or []
    if not isinstance(entries, list):
        raise ReleaseError("kubeconfig exec environment is invalid")
    for item in entries:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item["name"]
            or not isinstance(item.get("value"), str)
        ):
            raise ReleaseError("kubeconfig exec environment is invalid")
        environment[item["name"]] = item["value"]
    environment["KUBERNETES_EXEC_INFO"] = json.dumps(
        {
            "apiVersion": exec_spec.get(
                "apiVersion", "client.authentication.k8s.io/v1beta1"
            ),
            "kind": "ExecCredential",
            "spec": {"interactive": False},
        }
    )
    try:
        completed = run_command(
            [command, *args],
            capture=True,
            environment=environment,
            timeout_seconds=EXEC_TIMEOUT_SECONDS,
        )
    except (OSError, TimeoutError, subprocess.TimeoutExpired):
        raise ReleaseError("exec credential plugin failed") from None
    if completed.returncode:
        raise ReleaseError(f"exec credential plugin exited {completed.returncode}")
    if (
        sum(
            len(value.encode("utf-8"))
            for value in (completed.stdout or "", completed.stderr or "")
        )
        > MAX_CREDENTIAL_OUTPUT_BYTES
    ):
        raise ReleaseError("exec credential output exceeds its size limit")
    try:
        credential = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise ReleaseError("exec credential plugin did not return JSON") from None
    if (
        not isinstance(credential, dict)
        or credential.get("kind") != "ExecCredential"
        or credential.get("apiVersion") != exec_spec.get("apiVersion")
    ):
        raise ReleaseError("exec credential response identity is invalid")
    status = credential.get("status") if isinstance(credential, dict) else None
    if not isinstance(status, dict):
        raise ReleaseError("exec credential plugin returned no status")
    if set(status).difference({"token", "expirationTimestamp"}):
        raise ReleaseError("exec credential returned unsupported status fields")
    token = status.get("token")
    if not isinstance(token, str) or not token:
        raise ReleaseError("exec credential plugin returned no status.token")
    return token, _parse_expiry(status.get("expirationTimestamp"), now())


class ReleaseKubeconfigCache:
    """The engine process's cached kubeconfigs: CPU path plus ``KUBECONFIG``.

    ``command_inputs`` selects copies only for direct kubectl commands whose
    full budget fits the credentials' explicit lifetime. Other commands keep
    native authentication, including scripts that start their own kubectl.
    """

    def __init__(
        self,
        cpu_kubeconfig: str,
        *,
        dry_run: bool = False,
        environ: MutableMapping[str, str] | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._environ = os.environ if environ is None else environ
        self._original_kubeconfig = self._environ.get("KUBECONFIG")
        self._directory: Path | None = None
        self._caches: dict[Path, KubeconfigTokenCache] = {}
        self._now = now
        self._original_cpu_kubeconfig = cpu_kubeconfig
        self._cached_gpu_kubeconfig = self._original_kubeconfig
        self.cpu_kubeconfig = cpu_kubeconfig
        self.active = not dry_run and token_cache_enabled(self._environ)
        if not self.active:
            return
        self._directory = Path(tempfile.mkdtemp(prefix="gpu-fault-kubeconfig-"))
        self._directory.chmod(0o700)
        try:
            self.cpu_kubeconfig = self._cached_path(cpu_kubeconfig)
            if self._original_kubeconfig:
                self._cached_gpu_kubeconfig = os.pathsep.join(
                    self._cached_path(item)
                    for item in self._original_kubeconfig.split(os.pathsep)
                )
        except BaseException:
            self.cleanup()
            raise

    def _cached_path(self, original: str) -> str:
        """The cached copy of ``original``, or ``original`` when nothing to cache."""

        if self._directory is None or not original:
            return original
        source = Path(original)
        if not source.is_file():
            # kubectl reports the missing file exactly as before.
            return original
        key = source.resolve()
        cache = self._caches.get(key)
        if cache is None:
            cache = KubeconfigTokenCache(
                source, directory=self._directory, now=self._now
            )
            if not cache.has_exec_users or cache.earliest_expiry is None:
                return original
            self._caches[key] = cache
        return str(cache.path)

    def rewrite_config(self, config: ReleaseConfig) -> ReleaseConfig:
        """``config`` with the CPU kubeconfig pointing at the cached copy."""

        if config.cpu_kubeconfig == self.cpu_kubeconfig:
            return config
        return dataclasses.replace(config, cpu_kubeconfig=self.cpu_kubeconfig)

    def refresh_if_needed(self, lead_seconds: float = DEFAULT_LEAD_SECONDS) -> None:
        for cache in self._caches.values():
            cache.refresh_if_needed(lead_seconds)

    def command_inputs(
        self,
        arguments: list[str],
        environment: dict[str, str] | None,
        *,
        timeout_seconds: float,
    ) -> tuple[list[str], dict[str, str] | None]:
        if not self.active or not arguments or Path(arguments[0]).name != "kubectl":
            return arguments, environment
        values = dict(self._environ if environment is None else environment)
        if values.get("KUBECONFIG") != self._original_kubeconfig:
            return arguments, environment
        self.refresh_if_needed()
        if any(
            cache.earliest_expiry is not None
            and cache.earliest_expiry - self._now()
            <= timeout_seconds + DEFAULT_LEAD_SECONDS
            for cache in self._caches.values()
        ):
            return arguments, environment
        prepared = list(arguments)
        for index, word in enumerate(prepared):
            if word == "--kubeconfig" and index + 1 < len(prepared):
                if prepared[index + 1] == self._original_cpu_kubeconfig:
                    prepared[index + 1] = self.cpu_kubeconfig
            elif word == f"--kubeconfig={self._original_cpu_kubeconfig}":
                prepared[index] = f"--kubeconfig={self.cpu_kubeconfig}"
        if self._cached_gpu_kubeconfig is not None:
            values["KUBECONFIG"] = self._cached_gpu_kubeconfig
        return prepared, values

    def cleanup(self) -> None:
        if self._directory is not None:
            shutil.rmtree(self._directory, ignore_errors=True)
            self._directory = None
        self._caches = {}

    def __enter__(self) -> ReleaseKubeconfigCache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.cleanup()
