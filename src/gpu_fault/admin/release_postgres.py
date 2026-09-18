"""Supervised lifetime and private grants for one local release-test database."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.command_log import FAILURE_EXIT_CODE_ATTRIBUTE
from gpu_fault.admin.execution import cleanup_deadline, deadline_scope
from gpu_fault.admin.postgres_grant import (
    CONTAINER_ID,
    LOCAL_DOCKER_HOST,
    OWNER_FILE,
    OWNER_LABEL,
    POSTGRES_URL_ENV,
    URL_FILE,
    PostgresTestAllocation,
    private_directory,
    private_write,
)
from gpu_fault.admin.process_supervisor import (
    ProcessSupervisionLost,
    ensure_supervision_safe,
)

POSTGRES_IMAGE = "postgres:16"
ALLOCATION_DIRECTORY = "release-postgres"
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
IDENTITY_FORMAT = (
    '{"id":{{json .Id}},"name":{{json .Name}},"image":{{json .Image}},'
    '"reference":{{json .Config.Image}},"labels":{{json .Config.Labels}},'
    '"running":{{json .State.Running}},"ports":{{json .NetworkSettings.Ports}},'
    '"bindings":{{json .HostConfig.PortBindings}}}'
)


class PostgresCleanupError(BootstrapError):
    """The private ownership record must remain available for reconciliation."""


def require_postgres_cleanup(state_dir: Path) -> None:
    directory = state_dir.resolve() / ALLOCATION_DIRECTORY
    if directory.exists() or directory.is_symlink():
        raise PostgresCleanupError(
            "a local PostgreSQL allocation requires explicit cleanup reconciliation; "
            f"private ownership location: {directory}"
        )


def _docker_environment() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if name
        not in {
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_TLS_VERIFY",
            "DOCKER_CERT_PATH",
            "COSIGN_PASSWORD",
        }
        and not name.startswith("PG")
    }


def _docker(
    runner: CommandRunner,
    *arguments: str,
    mutate: bool = False,
    timeout: float = 20,
) -> str:
    return runner.run(
        ["docker", "--host", LOCAL_DOCKER_HOST, *arguments],
        env=_docker_environment(),
        capture=True,
        sensitive=True,
        mutate=mutate,
        timeout_seconds=timeout,
    )


def _image(runner: CommandRunner) -> str:
    def listed() -> list[str]:
        values = _docker(
            runner,
            "image",
            "ls",
            "--quiet",
            "--no-trunc",
            "--filter",
            f"reference={POSTGRES_IMAGE}",
        ).split()
        if len(values) > 1 or any(
            IMAGE_ID.fullmatch(value) is None for value in values
        ):
            raise BootstrapError("local PostgreSQL image identity is ambiguous")
        return values

    values = listed()
    if not values:
        _docker(runner, "pull", POSTGRES_IMAGE, mutate=True, timeout=600)
        values = listed()
    if len(values) != 1:
        raise BootstrapError("local PostgreSQL image is unavailable after pull")
    observed = json.loads(
        _docker(runner, "image", "inspect", "--format", "{{json .Id}}", values[0])
    )
    if observed != values[0]:
        raise BootstrapError("local PostgreSQL image identity changed")
    return values[0]


@dataclass
class OwnedPostgres:
    runner: CommandRunner
    directory: Path
    image: str
    owner: str
    directory_identity: tuple[int, int]
    container: str | None = None
    port: int | None = None
    create_attempted: bool = False
    supervision_lost: bool = False
    phase: str = "PREPARED"
    password: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)

    @property
    def name(self) -> str:
        return "gpu-fault-release-postgres-" + self.owner

    @property
    def cidfile(self) -> Path:
        return self.directory / "container.cid"

    def record(self, phase: str) -> None:
        self.phase = phase
        info = private_directory(self.directory)
        if (info.st_dev, info.st_ino) != self.directory_identity:
            raise BootstrapError("local PostgreSQL ownership directory was replaced")
        write_json_atomic(
            self.directory / "ownership.json",
            {
                "schema_version": 1,
                "owner": self.owner,
                "name": self.name,
                "image_id": self.image,
                "image_reference": POSTGRES_IMAGE,
                "docker_host": LOCAL_DOCKER_HOST,
                "container_id": self.container,
                "port": self.port,
                "create_attempted": self.create_attempted,
                "supervision_lost": self.supervision_lost,
                "phase": phase,
                "directory_device": info.st_dev,
                "directory_inode": info.st_ino,
            },
        )

    def read_cid(self) -> str | None:
        try:
            descriptor = os.open(
                self.cidfile, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            )
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "r", encoding="ascii") as source:
            before = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or before.st_size > 65
            ):
                raise BootstrapError("local PostgreSQL CID file is unsafe")
            value = source.read(66).strip()
            after = os.fstat(source.fileno())
            if (
                (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or value
                and CONTAINER_ID.fullmatch(value) is None
                or self.container is not None
                and value
                and value != self.container
            ):
                raise BootstrapError("local PostgreSQL CID file differs")
        return value or None

    def listed(self, selector: str) -> list[str]:
        values = _docker(
            self.runner,
            "container",
            "ls",
            "--all",
            "--no-trunc",
            "--filter",
            selector,
            "--format",
            "{{.ID}}",
        ).split()
        if len(values) > 1 or any(
            CONTAINER_ID.fullmatch(value) is None for value in values
        ):
            raise BootstrapError("local PostgreSQL container inventory is ambiguous")
        return values

    def inspect(self, identifier: str, *, running: bool | None = None) -> None:
        value = json.loads(
            _docker(
                self.runner,
                "container",
                "inspect",
                "--format",
                IDENTITY_FORMAT,
                identifier,
            )
        )
        labels = value.get("labels") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or value.get("id") != identifier
            or self.container not in (None, identifier)
            or value.get("name") != "/" + self.name
            or value.get("image") != self.image
            or value.get("reference") != self.image
            or not isinstance(labels, dict)
            or labels.get(OWNER_LABEL) != self.owner
            or not isinstance(value.get("running"), bool)
            or running is not None
            and value["running"] is not running
        ):
            raise BootstrapError("local PostgreSQL container ownership changed")
        bindings = value.get("bindings")
        if (
            not isinstance(bindings, dict)
            or set(bindings) != {"5432/tcp"}
            or not isinstance(bindings["5432/tcp"], list)
            or len(bindings["5432/tcp"]) != 1
            or not isinstance(bindings["5432/tcp"][0], dict)
            or bindings["5432/tcp"][0].get("HostIp") != "127.0.0.1"
        ):
            raise BootstrapError("local PostgreSQL listener is not loopback-only")
        if value["running"]:
            ports = value.get("ports")
            if (
                not isinstance(ports, dict)
                or set(ports) != {"5432/tcp"}
                or not isinstance(ports["5432/tcp"], list)
                or len(ports["5432/tcp"]) != 1
                or not isinstance(ports["5432/tcp"][0], dict)
                or ports["5432/tcp"][0].get("HostIp") != "127.0.0.1"
            ):
                raise BootstrapError("local PostgreSQL port binding is incomplete")
            port = ports["5432/tcp"][0].get("HostPort")
            if (
                not isinstance(port, str)
                or re.fullmatch(r"[1-9][0-9]{0,4}", port) is None
                or not 1 <= int(port) <= 65535
                or self.port not in (None, int(port))
            ):
                raise BootstrapError("local PostgreSQL published port changed")
            self.port = int(port)
        self.container = identifier
        self.record(self.phase)

    def start(self) -> PostgresTestAllocation:
        private_write(
            self.directory / "postgres.env",
            "POSTGRES_USER=postgres\nPOSTGRES_DB=postgres\n"
            "POSTGRES_HOST_AUTH_METHOD=scram-sha-256\n"
            f"POSTGRES_PASSWORD={self.password}\n",
        )
        self.create_attempted = True
        self.record("CREATE_STARTED")
        created = _docker(
            self.runner,
            "create",
            "--name",
            self.name,
            "--label",
            f"{OWNER_LABEL}={self.owner}",
            "--cidfile",
            str(self.cidfile),
            "--publish",
            "127.0.0.1::5432",
            "--env-file",
            str(self.directory / "postgres.env"),
            "--",
            self.image,
            mutate=True,
        ).strip()
        if CONTAINER_ID.fullmatch(created) is None:
            raise BootstrapError("local PostgreSQL creation returned an invalid CID")
        self.container = created
        self.record("CREATED")
        if self.read_cid() != created:
            raise BootstrapError("local PostgreSQL creation lacks a matching CID file")
        self.inspect(created, running=False)
        _docker(self.runner, "start", created, mutate=True)
        self.record("STARTED")
        with deadline_scope("local release PostgreSQL readiness", 120):
            for _attempt in range(60):
                self.inspect(created, running=True)
                try:
                    _docker(
                        self.runner,
                        "exec",
                        created,
                        "pg_isready",
                        "-h",
                        "127.0.0.1",
                        "-U",
                        "postgres",
                        "-d",
                        "postgres",
                        timeout=10,
                    )
                except BootstrapError as error:
                    if getattr(error, FAILURE_EXIT_CODE_ATTRIBUTE, None) not in {1, 2}:
                        raise
                else:
                    self.inspect(created, running=True)
                    return self.grant()
                time.sleep(1)
        raise BootstrapError("temporary PostgreSQL 16 did not become ready")

    def grant(self) -> PostgresTestAllocation:
        if self.container is None or self.port is None:
            raise BootstrapError("local PostgreSQL grant has no complete identity")
        home = self.directory / "pgpass-home"
        home.mkdir(mode=0o700)
        private_write(
            home / ".pgpass", f"127.0.0.1:{self.port}:*:postgres:{self.password}\n"
        )
        url = f"postgresql://postgres@127.0.0.1:{self.port}/postgres"
        private_write(
            self.directory / OWNER_FILE,
            json.dumps(
                {
                    "schema_version": 1,
                    "owner": self.owner,
                    "container": self.container,
                    "port": self.port,
                    "pgpass_home": str(home),
                    "image_id": self.image,
                    "allocation_device": self.directory_identity[0],
                    "allocation_inode": self.directory_identity[1],
                },
                sort_keys=True,
            )
            + "\n",
        )
        private_write(self.directory / URL_FILE, url + "\n")
        self.record("READY")
        return PostgresTestAllocation(url, self.directory)

    def cleanup(self) -> None:
        if not self.create_attempted:
            return
        ensure_supervision_safe(allow_interrupted=True)
        identifier = self.read_cid() or self.container
        if identifier is None:
            raise BootstrapError("local PostgreSQL creation outcome is unknown")
        named = self.listed(f"name=^/{self.name}$")
        if named and named != [identifier]:
            raise BootstrapError("local PostgreSQL container name was replaced")
        present = self.listed(f"id={identifier}")
        if present and present != [identifier]:
            raise BootstrapError("local PostgreSQL container ID differs")
        if present:
            self.inspect(identifier)
            try:
                _docker(
                    self.runner,
                    "container",
                    "rm",
                    "--force",
                    "--volumes",
                    identifier,
                    mutate=True,
                )
            except (BootstrapError, OSError, subprocess.TimeoutExpired, TimeoutError):
                # Only fresh, successful absence reads can resolve a lost delete ACK.
                pass
        if self.listed(f"id={identifier}") or self.listed(f"name=^/{self.name}$"):
            raise BootstrapError("local PostgreSQL removal is unconfirmed")
        self.record("REMOVED")

    def remove_directory(self) -> None:
        info = private_directory(self.directory)
        if (
            info.st_dev,
            info.st_ino,
        ) != self.directory_identity or not shutil.rmtree.avoids_symlink_attacks:
            raise BootstrapError("local PostgreSQL private directory ownership changed")
        shutil.rmtree(self.directory)

    def retain(self, *, supervision_lost: bool = False) -> bool:
        self.supervision_lost = self.supervision_lost or supervision_lost
        try:
            self.record("SUPERVISION_LOST" if self.supervision_lost else "UNCONFIRMED")
        except Exception:
            return False
        return True


@contextmanager
def isolated_postgres_allocation(
    runner: CommandRunner,
    *,
    repository_root: Path | None = None,
    state_dir: Path | None = None,
) -> Iterator[PostgresTestAllocation]:
    if state_dir is not None:
        require_postgres_cleanup(state_dir)
    configured = os.environ.get(POSTGRES_URL_ENV, "").strip()
    if configured:
        yield PostgresTestAllocation(configured)
        return
    ensure_supervision_safe()
    image = _image(runner)
    if state_dir is None:
        directory = Path(
            tempfile.mkdtemp(prefix="gpu-fault-release-postgres-")
        ).resolve()
    else:
        private_directory(state_dir.resolve())
        directory = state_dir.resolve() / ALLOCATION_DIRECTORY
        directory.mkdir(mode=0o700)
    info = private_directory(directory)
    if directory.is_relative_to((repository_root or Path.cwd()).resolve()):
        directory.rmdir()
        raise BootstrapError(
            "PostgreSQL grant credentials must stay outside the checkout"
        )
    owned = OwnedPostgres(
        runner, directory, image, uuid4().hex, (info.st_dev, info.st_ino)
    )
    try:
        owned.record("PREPARED")
        yield owned.start()
    except ProcessSupervisionLost as error:
        if not owned.retain(supervision_lost=True):
            error.add_note(f"could not update private PostgreSQL intent at {directory}")
        raise
    finally:
        if not owned.supervision_lost:
            try:
                with cleanup_deadline("local release PostgreSQL cleanup"):
                    owned.cleanup()
                    owned.remove_directory()
            except ProcessSupervisionLost as error:
                if not owned.retain(supervision_lost=True):
                    error.add_note(
                        f"could not update private PostgreSQL intent at {directory}"
                    )
                raise
            except BaseException as error:
                retained = owned.retain()
                raise PostgresCleanupError(
                    "local PostgreSQL cleanup unconfirmed; private ownership location: "
                    f"{directory}"
                    + ("" if retained else "; ownership update also failed")
                ) from error
