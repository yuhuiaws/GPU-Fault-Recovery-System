from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


MAX_SCAN_BYTES = 1024 * 1024
SAFE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# HostProbeFixture executes this script after chroot into the node root.
SCAN_ROOTS = (Path("/tmp"), Path("/etc/gpu-fault"), Path("/var/lib/kubelet/pods"))
PROC_ROOT = Path("/proc")


class ProbeError(RuntimeError):
    pass


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def candidate_files() -> Iterable[Path]:
    def unreadable(error: OSError) -> None:
        raise ProbeError("a host scan directory could not be inspected") from error

    for root in SCAN_ROOTS:
        if not root.is_dir():
            raise ProbeError("a required host scan directory is missing")
        for directory, _subdirs, filenames in os.walk(root, onerror=unreadable):
            for filename in filenames:
                path = Path(directory) / filename
                try:
                    if path.is_file():
                        if path.stat().st_size > MAX_SCAN_BYTES:
                            raise ProbeError(
                                "a host scan file exceeds the bounded scan size"
                            )
                        yield path
                except OSError as exc:
                    raise ProbeError("a host scan file could not be inspected") from exc


def value_tokens(path: Path, payload: bytes) -> Iterable[tuple[str, bytes]]:
    yield str(path), payload.strip()
    if path.name.endswith(".env"):
        for line in payload.splitlines():
            _key, separator, value = line.partition(b"=")
            if separator:
                yield f"{path}:env", value.strip().strip(b"'\"")
    if path.name == "environ":
        for item in payload.split(b"\0"):
            _key, separator, value = item.partition(b"=")
            if separator:
                yield f"{path}:environ", value


def process_environments() -> Iterable[tuple[str, bytes]]:
    proc = PROC_ROOT
    if not proc.is_dir():
        raise ProbeError("host proc directory is missing")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        path = entry / "environ"
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            # A process may exit between listing its PID and reading environ.
            continue
        except OSError as exc:
            raise ProbeError("a process environment could not be inspected") from exc
        yield from value_tokens(path, payload)


def scan(master_sha256: str) -> dict[str, Any]:
    if SAFE_SHA256.fullmatch(master_sha256) is None:
        raise ProbeError("master SHA-256 is invalid")
    matches = []
    scanned = 0
    for path in candidate_files():
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ProbeError("a host scan file could not be read") from exc
        for label, value in value_tokens(path, payload):
            if not value:
                continue
            scanned += 1
            if digest(value) == master_sha256:
                matches.append(label)
    for label, value in process_environments():
        if not value:
            continue
        scanned += 1
        if digest(value) == master_sha256:
            matches.append(label)
    if scanned == 0:
        raise ProbeError("host scan inspected no nonempty values")
    return {
        "master_matches": sorted(set(matches)),
        "values_scanned": scanned,
        "host_tmp_exists": SCAN_ROOTS[0].is_dir(),
        "systemd_environment_exists": SCAN_ROOTS[1].is_dir(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-sha256", required=True)
    arguments = parser.parse_args()
    try:
        result = scan(arguments.master_sha256)
    except (OSError, ProbeError) as exc:
        result = {"error": str(exc)}
        print(json.dumps(result, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())
