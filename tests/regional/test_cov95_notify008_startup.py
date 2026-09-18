from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.e2e.regional.notify008_resources import POSTGRES_START, pod_spec
from tests.regional._cov95_notify008_lifecycle import target, uid
from tests.regional._cov95_notify008_support import RUN_ID


def test_postgres_manifest_path_resolves_debian_versioned_tools(tmp_path):
    binary = tmp_path / "usr/lib/postgresql/16/bin/initdb"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    database = next(
        item for item in pod_spec(target())["containers"] if item["name"] == "database"
    )
    configured = next(
        item["value"] for item in database["env"] if item["name"] == "PATH"
    )
    directories = configured.split(os.pathsep)
    assert all(Path(item).is_absolute() for item in directories), (
        "database tools must not resolve through a writable working directory"
    )
    image_path = os.pathsep.join(
        str(tmp_path / item.removeprefix("/")) for item in directories
    )
    assert shutil.which("initdb", path=image_path) == str(binary), (
        "the PostgreSQL 16 image keeps initdb outside the generic system bin paths"
    )


@pytest.mark.parametrize(
    ("mode", "status", "started"),
    [
        ("unarmed", 75, False),
        ("stopped", 0, False),
        ("foreign-arm", 76, False),
        ("armed-and-stopped", 0, False),
        ("expired-and-armed", 75, False),
        ("armed", 0, True),
    ],
)
def test_postgres_start_barrier_executes_only_after_owned_arm_with_live_budget(
    tmp_path, mode, status, started
):
    control = tmp_path / "control"
    control.mkdir()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    recorder = binaries / "docker-entrypoint.sh"
    recorder.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$CAPTURE_FILE"\n')
    recorder.chmod(0o700)
    capture = tmp_path / "arguments"
    script = tmp_path / "postgres-start"
    script.write_text(POSTGRES_START)
    if mode in {"foreign-arm", "armed-and-stopped", "expired-and-armed", "armed"}:
        (control / f"arm-{uid(30)}").write_text(
            "foreign" if mode == "foreign-arm" else RUN_ID
        )
    if mode in {"stopped", "armed-and-stopped"}:
        (control / f"stop-{uid(30)}").write_text(RUN_ID)
    environment = {
        **os.environ,
        "PATH": f"{binaries}:/usr/bin:/bin",
        "NOTIFY008_RUN_ID": RUN_ID,
        "NOTIFY008_POD_UID": uid(30),
        "NOTIFY008_SECONDS": "0" if mode == "expired-and-armed" else "2",
        "CAPTURE_FILE": str(capture),
    }

    result = subprocess.run(
        ["/bin/sh", str(script), str(control)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )

    assert result.returncode == status, (
        f"startup barrier returned the wrong state for {mode}"
    )
    assert capture.exists() is started, (
        f"database initialization must not cross an unarmed, stopped or expired barrier: {mode}"
    )
    if started:
        assert capture.read_text().splitlines() == [
            "postgres",
            "-c",
            "listen_addresses=",
            "-c",
            "unix_socket_directories=/socket,/var/run/postgresql",
            "-c",
            "unix_socket_permissions=0700",
        ], "the accepted startup must remain Unix-socket-only"
