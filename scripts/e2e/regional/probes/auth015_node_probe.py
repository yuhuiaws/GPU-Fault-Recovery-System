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


class ProbeError(RuntimeError):
    pass


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def candidate_files() -> Iterable[Path]:
    roots = (
        Path("/host/tmp"),
        Path("/host/etc/gpu-fault"),
        Path("/host/var/lib/kubelet/pods"),
    )
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            try:
                if path.is_file() and path.stat().st_size <= MAX_SCAN_BYTES:
                    yield path
            except OSError:
                continue


def value_tokens(path: Path, payload: bytes) -> Iterable[tuple[str, bytes]]:
    yield str(path), payload.strip()
    if path.name.endswith(".env"):
        for line in payload.splitlines():
            _key, separator, value = line.partition(b"=")
            if separator:
                yield f"{path}:env", value.strip().strip(b"'\"")
    if "/proc/" in str(path) and path.name == "environ":
        for item in payload.split(b"\0"):
            _key, separator, value = item.partition(b"=")
            if separator:
                yield f"{path}:environ", value


def process_environments() -> Iterable[tuple[str, bytes]]:
    proc = Path("/host/proc")
    if not proc.is_dir():
        return
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        path = entry / "environ"
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        yield from value_tokens(path, payload)


def scan(master_sha256: str) -> dict[str, Any]:
    if SAFE_SHA256.fullmatch(master_sha256) is None:
        raise ProbeError("master SHA-256 is invalid")
    matches = []
    scanned = 0
    for path in candidate_files():
        try:
            payload = path.read_bytes()
        except OSError:
            continue
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
    return {
        "master_matches": sorted(set(matches)),
        "values_scanned": scanned,
        "host_tmp_exists": Path("/host/tmp").is_dir(),
        "systemd_environment_exists": Path("/host/etc/gpu-fault").is_dir(),
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
