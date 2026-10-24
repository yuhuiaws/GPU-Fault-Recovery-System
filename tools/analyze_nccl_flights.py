"""Offline adapter for the official PyTorch 2.10 FR analyzer, never run on a trainer."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import resource
import os
from pathlib import Path
from typing import Any

from gpu_fault.node_agent.operations.flight_recorder import (
    decode_flight_dump,
)


def analyze(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return {
            "status": "UNAVAILABLE",
            "reason": "PyTorch 2.10 analyzer is not installed",
        }
    if not version.startswith("2.10."):
        return {
            "status": "UNAVAILABLE",
            "reason": "unsupported analyzer version",
            "version": version,
        }
    records = manifest.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= 65536:
        raise ValueError("a nonempty process-bound rank manifest is required")
    details: dict[str, Any] = {}
    ranks, total = set(), 0
    for record in records:
        rank = record.get("rank")
        if type(rank) is not int or rank < 0 or rank in ranks:
            raise ValueError("unknown or duplicate rank")
        if (
            type(record.get("pid")) is not int
            or record["pid"] <= 0
            or type(record.get("process_start_ticks")) is not int
            or record["process_start_ticks"] <= 0
            or not record.get("node_id")
        ):
            raise ValueError("process generation metadata is required")
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("archive path escapes manifest root")
        total += path.stat().st_size
        if total > 256 * 1024 * 1024:
            raise ValueError("offline input exceeds 256 MiB")
        with path.open("rb") as stream:
            raw = stream.read(64 * 1024 * 1024 + 1)
        if len(raw) > 64 * 1024 * 1024:
            raise ValueError("individual dump exceeds 64 MiB")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record.get("sha256"):
            raise ValueError("raw evidence digest mismatch")
        payload = decode_flight_dump(raw)
        if not isinstance(payload, dict) or payload.get("version") != "2.10":
            raise ValueError("mixed or unsupported dump versions")
        details[str(rank)] = {**payload, "rank": rank}
        ranks.add(rank)
    builder = importlib.import_module(
        "torch.distributed.flight_recorder.components.builder"
    )
    database = builder.build_db(
        details,
        argparse.Namespace(
            verbose=False,
            just_print_entries=False,
            allow_incomplete_ranks=False,
            mismatch_cap=100,
        ),
        "2.10",
    )
    return {
        "status": "ANALYZED",
        "analyzer": f"pytorch/{version}",
        "rank_count": len(ranks),
        "root_cause_confirmed": False,
        "collective_count": len(database.collectives),
        "collectives_truncated": len(database.collectives) > 4096,
        "collectives": [
            collective_result(item) for item in database.collectives[:4096]
        ],
        "scope": "provided_archived_dump_set_only",
    }


def collective_result(item: Any) -> dict[str, Any]:
    mismatches = item.mismatch_collectives or {}
    return {
        "sequence": item.collective_seq_id,
        "group": item.group_id,
        "operation": item.collective_name,
        "pass_check": item.pass_check,
        "expected_rank_count": len(item.expected_ranks),
        "missing_ranks": sorted(item.missing_ranks or [])[:128],
        "missing_rank_count": len(item.missing_ranks or []),
        "mismatches": {
            str(rank): {
                "operation": record.collective_name,
                "reason": str(record.type_of_mismatch),
                "input_sizes": record.input_sizes,
                "output_sizes": record.output_sizes,
            }
            for rank, record in list(mismatches.items())[:128]
        },
        "mismatched_rank_count": len(mismatches),
        "mismatch": str(item.type_of_mismatch),
    }


def main() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 61))
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 * 1024 * 1024,) * 2)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(json.loads(args.manifest.read_text()), args.manifest.parent)
    except (OSError, ValueError, AssertionError, ImportError, KeyError) as exc:
        result = {"status": "UNDETERMINED", "error": str(exc)[:1024]}
    raw = json.dumps(result)
    if len(raw.encode()) > 8 * 1024 * 1024:
        result = {"status": "UNDETERMINED", "error": "analyzer output exceeds 8 MiB"}
        raw = json.dumps(result)
    print(raw)
    if result["status"] != "ANALYZED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
