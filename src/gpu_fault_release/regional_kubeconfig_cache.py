"""Per-process kubeconfig copies whose exec credentials are pre-fetched tokens.

Every ``kubectl`` call costs ~1.2 s on the deploy host, ~0.8 s of which is the
kubeconfig's exec credential plugin (``aws eks get-token``) that kubectl re-runs
on EVERY invocation. A CONTROL_PLANE_ONLY ``config`` release issues ~250 calls,
a full deploy far more, so the plugin alone is minutes of a release.

The engine therefore runs each exec plugin ONCE, writes a private ``0600`` copy
of the kubeconfig in which the ``exec`` entry is replaced by the static
``token`` it returned, and points kubectl (``--kubeconfig`` for the CPU
cluster, ``KUBECONFIG`` for the GPU ``--context`` calls and every child
script) at the copy. EKS tokens live ~15 min and releases run 7-40 min, so the
runner asks :meth:`ReleaseKubeconfigCache.refresh_if_needed` before each
command: when any token is inside the lead window the tokens are fetched again
and the copy is replaced atomically (``os.replace``), which a child script's
next kubectl call -- kubectl reads the file per call -- picks up unchanged.

Fail closed: a plugin that fails raises :class:`ReleaseError`; the engine never
proceeds with a kubeconfig that has no credentials. Users without ``exec`` and
files without any exec user are left as they are. ``--dry-run`` fetches
nothing, and ``GPU_FAULT_KUBECONFIG_TOKEN_CACHE=false`` restores the per-call
plugin behaviour.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault_release.regional_release_config import ReleaseConfig, ReleaseError

TOKEN_CACHE_ENV = "GPU_FAULT_KUBECONFIG_TOKEN_CACHE"
DEFAULT_LEAD_SECONDS = 180.0
# A plugin that omits ``expirationTimestamp`` gets a short lifetime, never a
# permanent one: refreshing an unexpired token is cheap, a 401 mid-release is not.
DEFAULT_TOKEN_LIFETIME_SECONDS = 600.0
EXEC_TIMEOUT_SECONDS = 60.0
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


def _parse_expiry(value: object, now: float) -> float:
    if not isinstance(value, str) or not value.strip():
        return now + DEFAULT_TOKEN_LIFETIME_SECONDS
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ReleaseError(
            f"exec credential plugin returned an unreadable expirationTimestamp: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


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
            document = yaml.safe_load(self.source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ReleaseError(f"cannot read kubeconfig {self.source}: {exc}") from exc
        if not isinstance(document, dict):
            raise ReleaseError(f"kubeconfig {self.source} is not a mapping")
        self._document = document
        self._exec_users: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for entry in document.get("users") or []:
            user = entry.get("user") if isinstance(entry, dict) else None
            if isinstance(user, dict):
                _absolutize_paths(user, self.source.parent)
                exec_spec = user.get("exec")
                if isinstance(exec_spec, dict):
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
            if self.earliest_expiry is None:
                return False
            if self.earliest_expiry - self._now() > lead_seconds:
                return False
            self._refresh()
            return True

    def _refresh(self) -> None:
        earliest: float | None = None
        for user, exec_spec in self._exec_users:
            token, expiry = _fetch_exec_credential(exec_spec, now=self._now)
            user.pop("exec", None)
            user["token"] = token
            earliest = expiry if earliest is None else min(earliest, expiry)
        self.earliest_expiry = earliest
        self._write(yaml.safe_dump(self._document, sort_keys=False))

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
) -> tuple[str, float]:
    command = exec_spec.get("command")
    if not isinstance(command, str) or not command:
        raise ReleaseError("kubeconfig exec entry has no command")
    args = [str(item) for item in exec_spec.get("args") or []]
    environment = dict(os.environ)
    for item in exec_spec.get("env") or []:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            environment[item["name"]] = str(item.get("value", ""))
    environment["KUBERNETES_EXEC_INFO"] = json.dumps(
        {
            "apiVersion": exec_spec.get(
                "apiVersion", "client.authentication.k8s.io/v1beta1"
            ),
            "kind": "ExecCredential",
            "spec": {"interactive": False},
        }
    )
    label = " ".join([command, *args])
    try:
        completed = subprocess.run(
            [command, *args],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=EXEC_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseError(f"exec credential plugin failed: {label}: {exc}") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        raise ReleaseError(
            f"exec credential plugin exited {completed.returncode}: {label}"
            + (f": {detail[-1]}" if detail else "")
        )
    try:
        credential = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ReleaseError(
            f"exec credential plugin did not return JSON: {label}"
        ) from exc
    status = credential.get("status") if isinstance(credential, dict) else None
    if not isinstance(status, dict):
        raise ReleaseError(f"exec credential plugin returned no status: {label}")
    token = status.get("token")
    if not isinstance(token, str) or not token:
        raise ReleaseError(f"exec credential plugin returned no status.token: {label}")
    return token, _parse_expiry(status.get("expirationTimestamp"), now())


class ReleaseKubeconfigCache:
    """The engine process's cached kubeconfigs: CPU path plus ``KUBECONFIG``.

    ``cpu_kubeconfig`` is the path to hand kubectl for the CPU cluster (the
    cached copy, or the original when the cache is off, the release is a dry
    run, or the file has no exec user). ``KUBECONFIG`` in ``environ`` is
    rewritten in place -- the GPU ``--context`` calls and every child script
    read it from the process environment -- and restored by :meth:`cleanup`.
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
        self.cpu_kubeconfig = cpu_kubeconfig
        self.active = not dry_run and token_cache_enabled(self._environ)
        if not self.active:
            return
        self._directory = Path(tempfile.mkdtemp(prefix="gpu-fault-kubeconfig-"))
        self._directory.chmod(0o700)
        try:
            self.cpu_kubeconfig = self._cached_path(cpu_kubeconfig)
            if self._original_kubeconfig:
                self._environ["KUBECONFIG"] = os.pathsep.join(
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
            if not cache.has_exec_users:
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

    def cleanup(self) -> None:
        if self._original_kubeconfig is None:
            self._environ.pop("KUBECONFIG", None)
        else:
            self._environ["KUBECONFIG"] = self._original_kubeconfig
        if self._directory is not None:
            shutil.rmtree(self._directory, ignore_errors=True)
            self._directory = None
        self._caches = {}

    def __enter__(self) -> ReleaseKubeconfigCache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.cleanup()
