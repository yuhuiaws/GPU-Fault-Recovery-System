"""Read only the installed control-plane component identity; standalone payload."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from importlib import metadata
from pathlib import Path
from typing import Any

import gpu_fault

SHARED_ADMIN_FILES = frozenset(
    {
        "__init__.py",
        "atomic_json.py",
        "capacity_evidence.py",
        "config.py",
        "config_parser.py",
        "operation_lock.py",
        "python_environment.py",
        "site.py",
    }
)


def runtime_identity() -> dict[str, Any]:
    distribution = metadata.distribution("gpu-fault-control-plane")
    root = Path(gpu_fault.__file__).resolve().parent
    if root != Path(str(distribution.locate_file("gpu_fault"))).resolve():
        raise RuntimeError(
            "imported runtime is not the installed control-plane distribution"
        )
    admin_files = {
        path.relative_to(root / "admin").as_posix()
        for path in (root / "admin").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if admin_files - SHARED_ADMIN_FILES or (root.parent / "gpu_fault_release").exists():
        raise RuntimeError(
            "control-plane runtime contains forbidden admin or release surfaces"
        )
    record = distribution.read_text("RECORD")
    if not record:
        raise RuntimeError("control-plane distribution has no installed file inventory")
    # Distribution.files may filter absent paths; the wheel RECORD must not.
    for row in csv.reader(io.StringIO(record), strict=True):
        if len(row) != 3:
            raise RuntimeError("control-plane installed file inventory is malformed")
        name, recorded_hash, _size = row
        if not name.startswith("gpu_fault/") or Path(name).suffix not in {
            ".py",
            ".yaml",
            ".yml",
            ".json",
        }:
            continue
        path = Path(str(distribution.locate_file(name))).resolve()
        algorithm, separator, expected_hash = recorded_hash.partition("=")
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or not separator
            or algorithm != "sha256"
        ):
            raise RuntimeError("control-plane installed file identity is unavailable")
        actual = (
            base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
            .decode()
            .rstrip("=")
        )
        if actual != expected_hash:
            raise RuntimeError(
                "control-plane installed files differ from wheel metadata"
            )
    return {
        "distribution": "gpu-fault-control-plane",
        "version": distribution.version,
        "module_digest": gpu_fault.module_digest(),
    }


def main() -> int:
    try:
        result = runtime_identity()
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
