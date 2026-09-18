"""Verify local or published image contents without network or application actions."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from typing import Any
from uuid import uuid4


DockerRunner = Callable[..., subprocess.CompletedProcess[str]]
CONTAINER_OWNER_LABEL = "gpu-fault.image-validation-owner"
CONTROL_TIMEOUT_SECONDS = 10.0
CLEANUP_TIMEOUT_SECONDS = 30.0
CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
IDENTITY_FORMAT = (
    '{"id":{{json .Id}},"name":{{json .Name}},"image":{{json .Image}},'
    '"reference":{{json .Config.Image}},"owner":'
    '{{json (index .Config.Labels "gpu-fault.image-validation-owner")}}}'
)
# Signed split images built before the single-environment migration remain valid.
LEGACY_COMPONENT_DOCKERFILE_SHA256 = (
    "ccb51594d93bdde17e314d5a971f16a3c5ea19e1786bd75722c23d78f5de2b04"
)
SINGLE_ENVIRONMENT_CHECK = """\
import importlib
import importlib.metadata as metadata
from pathlib import Path
import shutil
import site
import sys

runtime = Path(sys.argv[1]).resolve()
assert Path(sys.prefix).resolve() == runtime, "incorrect application interpreter"
assert Path(sys.base_prefix).resolve() != runtime, "application venv is missing"
assert site.ENABLE_USER_SITE is False, "user packages are enabled"
configuration = dict(
    line.split(" = ", 1)
    for line in (runtime / "pyvenv.cfg").read_text().splitlines()
    if " = " in line
)
assert configuration.get("include-system-site-packages") == "false", "shared packages enabled"
for entry in sys.path:
    path = Path(entry).resolve()
    if path.name in {"site-packages", "dist-packages"}:
        assert path.is_relative_to(runtime), "external package search path"
for distribution in metadata.distributions():
    assert Path(distribution.locate_file("")).resolve().is_relative_to(runtime), "external distribution"
for name in ("gpu_fault", "pydantic", "psycopg"):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(runtime), "external module"
assert set(metadata.packages_distributions()["gpu_fault"]) == {sys.argv[2]}, "component distribution differs"
for entry in metadata.distribution(sys.argv[2]).entry_points:
    if entry.group == "console_scripts":
        command = Path(shutil.which(entry.name) or "")
        assert command.is_file() and command.samefile(runtime / "bin" / entry.name), "incorrect CLI path"
        assert command.read_text().splitlines()[0] == "#!" + str(runtime / "bin/python"), "incorrect CLI interpreter"
        entry.load()
"""


class ImageValidationCleanupError(RuntimeError):
    """An owned validation container could not be proven removed."""


@contextmanager
def _cleanup_interrupts() -> Iterator[Callable[[], None]]:
    cleaning = False
    interrupted = False

    def begin_cleanup() -> None:
        nonlocal cleaning
        cleaning = True

    def interrupt(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        if not cleaning:
            raise KeyboardInterrupt("image validation interrupted")

    previous = {}
    if threading.current_thread() is threading.main_thread():
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, interrupt)
    try:
        yield begin_cleanup
        if interrupted:
            raise KeyboardInterrupt("image validation interrupted")
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


@dataclass
class _ImageCheckContainer:
    reference: str
    directory: Path
    owner: str
    runner: DockerRunner
    container_id: str | None = None
    image_id: str | None = None
    create_attempted: bool = False

    @property
    def name(self) -> str:
        return f"gpu-fault-image-check-{self.owner}"

    @property
    def cidfile(self) -> Path:
        return self.directory / "container.cid"

    def record(self) -> None:
        path = self.directory / "ownership.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "name": self.name,
                    "owner": self.owner,
                    "reference": self.reference,
                    "container_id": self.container_id,
                    "image_id": self.image_id,
                },
                sort_keys=True,
            )
            + "\n"
        )
        temporary.chmod(0o600)
        temporary.replace(path)

    def execute(
        self,
        arguments: list[str],
        deadline: float,
        *,
        control: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("image validation deadline exceeded")
        return self.runner(
            ["docker", *arguments],
            text=True,
            capture_output=True,
            check=False,
            timeout=min(remaining, CONTROL_TIMEOUT_SECONDS) if control else remaining,
            start_new_session=True,
        )

    def read_cid(self) -> str | None:
        try:
            info = self.cidfile.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(info.st_mode) or info.st_size > 65:
            raise ValueError("image validation CID file is unsafe")
        value = self.cidfile.read_text(encoding="ascii").strip()
        if not value:
            return None
        if CONTAINER_ID.fullmatch(value) is None:
            raise ValueError("image validation CID file is malformed")
        if self.container_id is not None and self.container_id != value:
            raise ValueError("image validation CID changed")
        return value

    def listed(self, selector: str, deadline: float) -> list[str]:
        result = self.execute(
            [
                "container",
                "ls",
                "--all",
                "--no-trunc",
                "--filter",
                selector,
                "--format",
                "{{.ID}}",
            ],
            deadline,
        )
        if result.returncode:
            raise RuntimeError("image validation container listing failed")
        identifiers = result.stdout.split()
        if len(identifiers) > 1 or any(
            CONTAINER_ID.fullmatch(value) is None for value in identifiers
        ):
            raise ValueError("image validation container identity is ambiguous")
        return identifiers

    def bind(self, identifier: str, deadline: float) -> None:
        result = self.execute(
            ["container", "inspect", "--format", IDENTITY_FORMAT, identifier], deadline
        )
        if result.returncode:
            raise RuntimeError("image validation container inspection failed")
        value = json.loads(result.stdout)
        if (
            not isinstance(value, dict)
            or value.get("id") != identifier
            or value.get("name") != f"/{self.name}"
            or value.get("owner") != self.owner
            or value.get("reference") != self.reference
            or not isinstance(value.get("image"), str)
            or IMAGE_ID.fullmatch(value["image"]) is None
            or self.container_id not in (None, identifier)
            or self.image_id not in (None, value["image"])
        ):
            raise ValueError("image validation container ownership changed")
        self.container_id = identifier
        self.image_id = value["image"]
        self.record()

    def run(
        self, command: Sequence[str], deadline: float
    ) -> subprocess.CompletedProcess[str]:
        self.create_attempted = True
        created = self.execute(
            [
                "create",
                "--name",
                self.name,
                "--label",
                f"{CONTAINER_OWNER_LABEL}={self.owner}",
                "--cidfile",
                str(self.cidfile),
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--",
                self.reference,
                *command,
            ],
            deadline,
            control=False,
        )
        if created.returncode:
            raise ValueError("image validation container creation failed")
        identifier = created.stdout.strip()
        if CONTAINER_ID.fullmatch(identifier) is None:
            raise ValueError("image validation creation returned an invalid CID")
        self.container_id = identifier
        if self.read_cid() != identifier:
            raise ValueError("image validation creation lacks a matching CID file")
        self.bind(identifier, deadline)
        return self.execute(["start", "--attach", identifier], deadline, control=False)

    def cleanup(self) -> None:
        if not self.create_attempted:
            return
        deadline = time.monotonic() + CLEANUP_TIMEOUT_SECONDS
        identifier = self.read_cid() or self.container_id
        named = self.listed(f"name=^/{self.name}$", deadline)
        if identifier is None:
            # Name/label reuse cannot identify the original creation. Absence
            # also cannot rule out a timed-out create finishing in the daemon.
            raise RuntimeError("image validation creation outcome is unknown")
        if named and named != [identifier]:
            raise ValueError("image validation container name was replaced")
        present = self.listed(f"id={identifier}", deadline)
        if not present:
            return
        if present != [identifier]:
            raise ValueError("image validation container ID differs")
        self.bind(identifier, deadline)
        try:
            self.execute(
                ["container", "rm", "--force", "--volumes", identifier], deadline
            )
        except (OSError, subprocess.TimeoutExpired):
            # An ACK can be lost after the daemon removed the exact owned ID.
            # Only successful absence queries can resolve that uncertainty.
            pass
        remaining = self.listed(f"id={identifier}", deadline)
        named = self.listed(f"name=^/{self.name}$", deadline)
        if remaining or named:
            raise RuntimeError("image validation container removal is unconfirmed")


def run_image_check(
    reference: str,
    command: Sequence[str],
    *,
    runner: DockerRunner = subprocess.run,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    """Run one sandboxed image command and verify its owned container is gone."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("image validation timeout must be positive and finite")
    deadline = time.monotonic() + timeout
    directory = Path(tempfile.mkdtemp(prefix="gf-image-check-"))
    container = _ImageCheckContainer(reference, directory, uuid4().hex, runner)
    with _cleanup_interrupts() as begin_cleanup:
        try:
            container.record()
            return container.run(command, deadline)
        finally:
            begin_cleanup()
            try:
                container.cleanup()
            except BaseException as error:
                raise ImageValidationCleanupError(
                    f"image validation cleanup unconfirmed for {container.name}; "
                    f"ownership record retained at {directory}"
                ) from error
            shutil.rmtree(directory)


def verify_images(
    descriptor: dict[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    images = descriptor.get("images")
    if descriptor.get("schema_version") != 3 or not isinstance(images, dict):
        raise ValueError("image check requires the split image-set descriptor")
    if set(images) != {"control_plane", "executor", "node_dependencies"}:
        raise ValueError("image set is incomplete")

    def run(name: str, *command: str) -> str:
        image = images[name]
        reference = image.get("reference") or image["tag"]
        result = run_image_check(str(reference), command, runner=runner)
        if result.returncode:
            raise ValueError(f"{name} image content check failed ({result.returncode})")
        return result.stdout

    for name, directory, other, commands in (
        (
            "control_plane",
            "control-plane",
            "executor",
            ("gpu-fault-api", "gpu-fault-store-migrate"),
        ),
        (
            "executor",
            "executor",
            "control-plane",
            (
                "gpu-fault-cluster-executor",
                "gpu-fault-completion-watcher",
                "gpu-fault-collector",
                "gpu-fault-node-installer-reconciler",
            ),
        ),
    ):
        path = f"/opt/gpu-fault/{directory}/bin"
        dockerfile_sha = images[name].get("dockerfile_sha256")
        if (
            not isinstance(dockerfile_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", dockerfile_sha) is None
        ):
            raise ValueError(f"{name} image has no valid Dockerfile identity")
        if dockerfile_sha == LEGACY_COMPONENT_DOCKERFILE_SHA256:
            environment_checks = [
                '! python -c "import gpu_fault" >/dev/null 2>&1',
                f"{path}/python -c 'import pydantic'",
            ]
        else:
            distribution = (
                "gpu-fault-control-plane"
                if name == "control_plane"
                else "gpu-fault-cluster-executor"
            )
            environment_checks = [
                "! test -L /opt/gpu-fault/runtime",
                f"test -L /opt/gpu-fault/{directory}",
                *(
                    shlex.join(
                        [
                            python,
                            "-I",
                            "-B",
                            "-c",
                            SINGLE_ENVIRONMENT_CHECK,
                            "/opt/gpu-fault/runtime",
                            distribution,
                        ]
                    )
                    for python in ("python", f"{path}/python")
                ),
                "python -I -B -m pip check",
                (
                    "gpu-fault-store-migrate --help >/dev/null"
                    if name == "control_plane"
                    else "gpu-fault-collector --help >/dev/null"
                ),
            ]
        checks = [
            *environment_checks,
            f"! test -e {path}/gpu-fault-admin",
            f"! test -d /opt/gpu-fault/{other}",
            *(f"test -x {path}/{command}" for command in commands),
        ]
        run(name, "/bin/sh", "-ec", " && ".join(checks))
        digest = run(
            name,
            f"{path}/python",
            "-I",
            "-B",
            "-c",
            "from gpu_fault import module_digest; print(module_digest())",
        ).strip()
        if digest != descriptor["components"][name]["module_digest"]:
            raise ValueError(f"{name} image module digest differs from its component")
        if name == "executor":
            run(
                name,
                f"{path}/python",
                "-I",
                "-B",
                "-m",
                "gpu_fault.collectors_cli",
                "validate-plugins",
            )
    raw = run(
        "node_dependencies", "/bin/cat", "/opt/gpu-fault/wheelhouse/inventory.json"
    )
    if (
        hashlib.sha256(raw.encode()).hexdigest()
        != images["node_dependencies"]["wheelhouse_sha256"]
    ):
        raise ValueError("node image inventory digest differs")
    inventory = json.loads(raw)
    checksums = run(
        "node_dependencies",
        "/bin/sh",
        "-ec",
        "cd /opt/gpu-fault/wheelhouse; sha256sum -- *",
    )
    observed = {}
    for line in checksums.splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or name in observed:
            raise ValueError("node image checksum listing is malformed")
        observed[name] = digest
    expected = {name: item["sha256"] for name, item in inventory["files"].items()}
    expected["inventory.json"] = images["node_dependencies"]["wheelhouse_sha256"]
    if observed != expected:
        raise ValueError("node image wheelhouse files differ from its inventory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", required=True, type=Path)
    args = parser.parse_args()
    verify_images(json.loads(args.descriptor.read_bytes()))
    print("CPU, Executor and offline node dependency image contents: PASSED")


if __name__ == "__main__":
    main()
