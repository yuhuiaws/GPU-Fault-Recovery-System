"""Bind native PostgreSQL process tests to the Actions-owned service."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
OWNER_LABEL = "gpu-fault.test-owner"
OWNER_FILE = "cov95-postgres-owner.json"
URL_FILE = "cov95-postgres-url"


class PostgresGrantError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConnectionTarget:
    user: str
    database: str
    port: int
    password: str = field(repr=False)

    @property
    def url(self) -> str:
        return urlunsplit(
            (
                "postgresql",
                f"{quote(self.user, safe='')}@127.0.0.1:{self.port}",
                "/" + quote(self.database, safe=""),
                "",
                "",
            )
        )


def job_owner(environment: Mapping[str, str]) -> str:
    names = ("GITHUB_REPOSITORY_ID", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
    if (
        environment.get("GITHUB_ACTIONS") != "true"
        or environment.get("GITHUB_JOB") != "postgres"
        or any(
            re.fullmatch(r"[1-9][0-9]*", environment.get(name, "")) is None
            for name in names
        )
    ):
        raise PostgresGrantError("a complete Actions postgres job identity is required")
    return "gpu-fault-ci-" + "-".join(environment[name] for name in names) + "-postgres"


def _canonical_path(value: str, *, directory: bool) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(character in value for character in "\r\n\x00")
        or path.is_symlink()
        or path.resolve() != path
        or (not path.is_dir() if directory else not path.is_file())
        or path.stat().st_uid != os.geteuid()
    ):
        raise PostgresGrantError(
            "grant paths must be canonical, existing and runner-owned"
        )
    return path


def grant_paths(environment: Mapping[str, str]) -> tuple[Path, Path, Path]:
    temporary = _canonical_path(environment.get("RUNNER_TEMP", ""), directory=True)
    workspace = _canonical_path(environment.get("GITHUB_WORKSPACE", ""), directory=True)
    if temporary.is_relative_to(workspace) or temporary.is_relative_to(ROOT):
        raise PostgresGrantError("grant credentials must stay outside the checkout")
    env_file = _canonical_path(environment.get("GITHUB_ENV", ""), directory=False)
    if not env_file.is_relative_to(temporary):
        raise PostgresGrantError("Actions environment file is outside RUNNER_TEMP")
    return temporary, temporary / job_owner(environment), env_file


def _container_id(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PostgresGrantError(
            "the complete Actions service container ID is required"
        )


def inspect_container(container: str) -> Mapping[str, Any]:
    result = subprocess.run(
        [
            "docker",
            "--host",
            "unix:///var/run/docker.sock",
            "inspect",
            "--format",
            '{"id":{{json .Id}},"running":{{json .State.Running}},'
            '"labels":{{json .Config.Labels}},"ports":{{json .NetworkSettings.Ports}},'
            '"environment":{{json .Config.Env}}}',
            container,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode:
        raise PostgresGrantError(
            "the Actions PostgreSQL service could not be inspected"
        )
    try:
        value = json.loads(result.stdout)
    except ValueError:
        raise PostgresGrantError("the Actions service inspection is invalid") from None
    if not isinstance(value, dict):
        raise PostgresGrantError("the Actions service inspection is invalid")
    return value


def connection_target(
    raw_url: str, *, container: str, owner: str, inspected: Mapping[str, Any]
) -> ConnectionTarget:
    from psycopg.conninfo import conninfo_to_dict

    _container_id(container)
    try:
        values = conninfo_to_dict(raw_url)
    except Exception:
        raise PostgresGrantError(
            "the test PostgreSQL connection input is invalid"
        ) from None
    port_text = values.get("port")
    if (
        not values
        or set(values) - {"host", "port", "user", "dbname", "password"}
        or values.get("host") != "127.0.0.1"
        or not isinstance(port_text, str)
        or re.fullmatch(r"[1-9][0-9]{0,4}", port_text) is None
        or not 1 <= int(port_text) <= 65535
    ):
        raise PostgresGrantError(
            "the test PostgreSQL target must be explicit loopback TCP"
        )
    labels, ports = inspected.get("labels"), inspected.get("ports")
    if (
        inspected.get("id") != container
        or inspected.get("running") is not True
        or not isinstance(labels, dict)
        or labels.get(OWNER_LABEL) != owner
        or not isinstance(ports, dict)
        or ports.get("5432/tcp") != [{"HostIp": "127.0.0.1", "HostPort": port_text}]
    ):
        raise PostgresGrantError(
            "the PostgreSQL service ownership or port binding differs"
        )
    settings: dict[str, str] = {}
    entries = inspected.get("environment")
    if not isinstance(entries, list):
        raise PostgresGrantError("the PostgreSQL service credentials are unavailable")
    for entry in entries:
        if not isinstance(entry, str):
            raise PostgresGrantError("the PostgreSQL service environment is invalid")
        name, separator, value = entry.partition("=")
        if name not in {"POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"}:
            continue
        if (
            not separator
            or name in settings
            or not value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise PostgresGrantError("the PostgreSQL service credentials are ambiguous")
        settings[name] = value
    if (
        set(settings) != {"POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"}
        or values.get("user") != settings["POSTGRES_USER"]
        or values.get("dbname") != settings["POSTGRES_DB"]
        or (
            "password" in values and values["password"] != settings["POSTGRES_PASSWORD"]
        )
    ):
        raise PostgresGrantError(
            "the PostgreSQL connection and service credentials differ"
        )
    return ConnectionTarget(
        settings["POSTGRES_USER"],
        settings["POSTGRES_DB"],
        int(port_text),
        settings["POSTGRES_PASSWORD"],
    )


def _private_write(path: Path, value: str) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())


def _append_environment(path: Path, values: Mapping[str, str]) -> None:
    if any(
        any(character in value for character in "\r\n\x00") for value in values.values()
    ):
        raise PostgresGrantError(
            "Actions environment values must be single-line references"
        )
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        if not stat.S_ISREG(os.fstat(output.fileno()).st_mode):
            raise PostgresGrantError("Actions environment output is not a regular file")
        output.writelines(f"{name}={value}\n" for name, value in values.items())
        output.flush()
        os.fsync(output.fileno())


def _remove_owned(path: Path, identity: tuple[int, int]) -> None:
    current = path.lstat()
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != identity
        or current.st_uid != os.geteuid()
        or not shutil.rmtree.avoids_symlink_attacks
    ):
        raise PostgresGrantError("grant cleanup refused a replaced or unsafe directory")
    shutil.rmtree(path)


def prepare_grant(environment: Mapping[str, str], container: str) -> None:
    _container_id(container)
    owner = job_owner(environment)
    _temporary, allocation, env_file = grant_paths(environment)
    original_home = _canonical_path(environment.get("HOME", ""), directory=True)
    if any(
        environment.get(name)
        for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG")
    ):
        raise PostgresGrantError(
            "native process tests require the local Actions Docker context"
        )
    if any(
        value
        for name, value in environment.items()
        if name.startswith("PG") and name != "PGPASSFILE"
    ):
        raise PostgresGrantError("ambient libpq settings cannot override the job grant")
    if allocation.exists() or allocation.is_symlink():
        raise PostgresGrantError(
            "the job grant directory already exists; refusing adoption"
        )
    target = connection_target(
        environment.get("GPU_FAULT_TEST_POSTGRES_URL", ""),
        container=container,
        owner=owner,
        inspected=inspect_container(container),
    )
    allocation.mkdir(mode=0o700)
    info = allocation.stat()
    identity = (info.st_dev, info.st_ino)
    home = allocation / "home"
    try:
        metadata = {
            "schema_version": 1,
            "owner": owner,
            "container": container,
            "port": target.port,
            "pgpass_home": str(home),
            "original_home": str(original_home),
            "allocation_device": info.st_dev,
            "allocation_inode": info.st_ino,
        }
        _private_write(
            allocation / OWNER_FILE, json.dumps(metadata, sort_keys=True) + "\n"
        )
        home.mkdir(mode=0o700)
        # Native fixtures create OID-owned child databases on this same service.
        fields = ("127.0.0.1", str(target.port), "*", target.user, target.password)
        _private_write(
            home / ".pgpass",
            ":".join(
                value.replace("\\", "\\\\").replace(":", "\\:") for value in fields
            )
            + "\n",
        )
        _private_write(allocation / URL_FILE, target.url + "\n")
        _append_environment(
            env_file,
            {
                "HOME": str(home),
                "PGPASSFILE": str(home / ".pgpass"),
                "GPU_FAULT_TEST_POSTGRES_URL": target.url,
                "PYTEST_GPU_FAULT_POSTGRES_ALLOCATION_DIR": str(allocation),
            },
        )
    except BaseException:
        _remove_owned(allocation, identity)
        raise


def cleanup_grant(environment: Mapping[str, str], container: str) -> None:
    _container_id(container)
    _temporary, allocation, env_file = grant_paths(environment)
    if not allocation.exists() and not allocation.is_symlink():
        return
    info = allocation.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        raise PostgresGrantError("grant cleanup requires the private job directory")
    descriptor = os.open(allocation / OWNER_FILE, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, encoding="utf-8") as source:
        record_info = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(record_info.st_mode)
            or record_info.st_uid != os.geteuid()
            or stat.S_IMODE(record_info.st_mode) != 0o600
        ):
            raise PostgresGrantError("grant cleanup owner record is unsafe")
        metadata = json.load(source)
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != 1
        or metadata.get("owner") != job_owner(environment)
        or metadata.get("container") != container
        or metadata.get("pgpass_home") != str(allocation / "home")
        or (metadata.get("allocation_device"), metadata.get("allocation_inode"))
        != (info.st_dev, info.st_ino)
    ):
        raise PostgresGrantError("grant cleanup ownership differs from the current job")
    try:
        original_home = _canonical_path(
            str(metadata.get("original_home", "")), directory=True
        )
        _append_environment(
            env_file,
            {
                "HOME": str(original_home),
                "PGPASSFILE": "",
                "GPU_FAULT_TEST_POSTGRES_URL": "",
                "PYTEST_GPU_FAULT_POSTGRES_ALLOCATION_DIR": "",
            },
        )
    finally:
        _remove_owned(allocation, (info.st_dev, info.st_ino))


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "cleanup"))
    parser.add_argument("--container", required=True)
    options = parser.parse_args(arguments)
    try:
        action = prepare_grant if options.command == "prepare" else cleanup_grant
        action(os.environ, options.container)
    except (PostgresGrantError, OSError, ValueError, subprocess.SubprocessError):
        print("ci-postgres-grant: refused unsafe or incomplete job grant")
        return 2
    print(f"ci-postgres-grant: {options.command} complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
