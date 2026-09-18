"""Private references for local native-test PostgreSQL credentials."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

ALLOCATION_ENV = "PYTEST_GPU_FAULT_POSTGRES_ALLOCATION_DIR"
POSTGRES_URL_ENV = "GPU_FAULT_TEST_POSTGRES_URL"
OWNER_FILE = "cov95-postgres-owner.json"
URL_FILE = "cov95-postgres-url"
OWNER_LABEL = "gpu-fault.test-owner"
LOCAL_DOCKER_HOST = "unix:///var/run/docker.sock"
CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
MAX_GRANT_BYTES = 64 * 1024


class PostgresGrantError(ValueError):
    pass


@dataclass(frozen=True)
class PostgresTestAllocation:
    url: str = field(repr=False)
    directory: Path | None = None

    def build_environment(self, parent: Mapping[str, str]) -> dict[str, str]:
        """Pass references to the build without giving its other tools a new HOME."""
        result = {**parent, POSTGRES_URL_ENV: self.url}
        if self.directory is not None:
            result[ALLOCATION_ENV] = str(self.directory)
        return result


def private_directory(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
        if (
            not path.is_absolute()
            or path.resolve() != path
            or not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700
            or info.st_uid != os.geteuid()
        ):
            raise PostgresGrantError(
                "PostgreSQL grant directory is not private and owned"
            )
        return info
    except OSError:
        raise PostgresGrantError("PostgreSQL grant directory is unavailable") from None


def private_write(path: Path, content: str) -> None:
    private_directory(path.parent)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _private_file(path: Path) -> int:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_uid != os.geteuid()
        or info.st_nlink != 1
        or not 0 < info.st_size <= MAX_GRANT_BYTES
    ):
        os.close(descriptor)
        raise PostgresGrantError("PostgreSQL grant file is not private and owned")
    return descriptor


def _read_private(path: Path) -> str:
    with os.fdopen(_private_file(path), "r", encoding="utf-8") as source:
        before = os.fstat(source.fileno())
        value = source.read(MAX_GRANT_BYTES + 1)
        after = os.fstat(source.fileno())
        if len(value) > MAX_GRANT_BYTES or (
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise PostgresGrantError("PostgreSQL grant file changed while reading")
        return value


def postgres_test_environment(parent: Mapping[str, str]) -> dict[str, str]:
    """Select an explicit grant for a PG test child; never mint or adopt one."""
    if ALLOCATION_ENV not in parent:
        return dict(parent)
    try:
        reference = parent[ALLOCATION_ENV]
        if not reference.strip():
            raise PostgresGrantError("PostgreSQL allocation reference is empty")
        directory = Path(reference)
        private_directory(directory)
        owner = json.loads(_read_private(directory / OWNER_FILE))
        url = _read_private(directory / URL_FILE).strip()
        parsed = urlsplit(url)
        if not isinstance(owner, dict):
            raise PostgresGrantError("PostgreSQL grant metadata is not an object")
        home = Path(str(owner.get("pgpass_home", "")))
        private_directory(home)
        descriptor = _private_file(home / ".pgpass")
        os.close(descriptor)
        port = str(owner.get("port", ""))
        if (
            parsed.scheme != "postgresql"
            or parsed.hostname != "127.0.0.1"
            or not parsed.username
            or parsed.password is not None
            or not re.fullmatch(r"[1-9][0-9]{0,4}", port)
            or not 1 <= int(port) <= 65535
            or parsed.port != int(port)
            or not parsed.path.startswith("/")
            or len(parsed.path) < 2
            or parsed.query
            or parsed.fragment
            or url != parent.get(POSTGRES_URL_ENV)
            or not isinstance(owner.get("owner"), str)
            or not owner["owner"].strip()
            or not isinstance(owner.get("container"), str)
            or CONTAINER_ID.fullmatch(owner["container"]) is None
        ):
            raise PostgresGrantError(
                "PostgreSQL child environment differs from its grant"
            )
    except (OSError, ValueError):
        raise PostgresGrantError(
            "PostgreSQL child requires a matching private allocation grant"
        ) from None
    result = {
        name: value for name, value in parent.items() if not name.startswith("PG")
    }
    result.pop("DOCKER_CONTEXT", None)
    result.pop("DOCKER_CONFIG", None)
    result.pop("DOCKER_TLS_VERIFY", None)
    result.pop("DOCKER_CERT_PATH", None)
    result.pop("COSIGN_PASSWORD", None)
    result.update(
        HOME=str(home),
        PGPASSFILE=str(home / ".pgpass"),
        DOCKER_HOST=LOCAL_DOCKER_HOST,
    )
    return result
