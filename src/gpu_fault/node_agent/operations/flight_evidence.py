"""Immutable, bounded FR evidence, separate from the application's rotating file."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

MAX_DUMP_BYTES = 8 * 1024 * 1024
MAX_NODE_EVIDENCE_BYTES = 64 * 1024 * 1024


def request_flight(
    agent: Any, command: Any, process: dict[str, Any], sample: str
) -> dict[str, Any]:
    request = agent._trigger_flight_recorder(process)
    root = getattr(agent, "diagnostic_output_dir", None)
    if root is not None:
        key = hashlib.sha256(command.command_id.encode()).hexdigest()[:24]
        request["evidence_path"] = str(
            Path(root) / f"nccl-flight-{key}-{process['pid']}-{sample}.dump"
        )
    return dict(request)


def preserve_dump(
    source: Path, target: Path, expected: os.stat_result
) -> dict[str, Any]:
    try:
        if expected.st_size > MAX_DUMP_BYTES:
            return {
                "status": "BYTE_BUDGET_EXCEEDED",
                "source_size_bytes": expected.st_size,
            }
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        retained = sum(
            item.stat().st_size for item in target.parent.glob("nccl-flight-*.dump")
        )
        if retained + expected.st_size > MAX_NODE_EVIDENCE_BYTES:
            return {"status": "NODE_BYTE_BUDGET_EXCEEDED"}
        with source.open("rb") as stream:
            before = os.fstat(stream.fileno())
            raw = stream.read(MAX_DUMP_BYTES + 1)
            after = os.fstat(stream.fileno())

        def identity(value: os.stat_result) -> tuple[int, int, int]:
            return value.st_ino, value.st_size, value.st_mtime_ns

        if identity(before) != identity(expected) or identity(after) != identity(
            expected
        ):
            return {"status": "SOURCE_CHANGED"}
        if len(raw) != expected.st_size:
            return {"status": "SOURCE_CHANGED"}
        digest = hashlib.sha256(raw).hexdigest()
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as target_stream:
            target_stream.write(raw)
        return {
            "status": "PRESERVED",
            "path": str(target),
            "sha256": digest,
            "bytes": len(raw),
        }
    except OSError as exc:
        return {"status": "UNAVAILABLE", "error": str(exc)[:256]}


def bundle_raw_flights(
    agent: Any, command: Any, work_dir: Path
) -> list[dict[str, Any]]:
    records = command.parameters.get("raw_flights_by_node", {}).get(command.node_id, [])
    result = []
    total = 0
    for index, record in enumerate(records[:64]):
        try:
            source = Path(record["path"]).resolve()
            if source.parent != Path(
                agent.diagnostic_output_dir
            ).resolve() or not source.name.startswith("nccl-flight-"):
                raise ValueError("raw FR reference is outside the evidence directory")
            info = source.stat()
            total += info.st_size
            if total > MAX_NODE_EVIDENCE_BYTES or info.st_size > MAX_DUMP_BYTES:
                raise ValueError("raw FR bundle budget exceeded")
            with source.open("rb") as stream:
                raw = stream.read(MAX_DUMP_BYTES + 1)
            if len(raw) != info.st_size:
                raise ValueError("raw FR evidence changed while bundling")
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("raw FR digest mismatch")
            name = f"flight-{index:02d}-{record['sample']}.dump"
            (work_dir / name).write_bytes(raw)
            result.append(
                {
                    **record,
                    "path": name,
                    "node_id": command.node_id,
                    "status": "BUNDLED",
                }
            )
        except (OSError, ValueError, KeyError) as exc:
            result.append({"status": "UNAVAILABLE", "error": str(exc)[:256]})
    return result
