"""Reviewed pass-through ``nvidia-smi`` wrapper for the GPU inventory replay.

GF-GENERAL-LIVE-026 (GPU identity / expected count) replays the metrics
collector's inventory reading, never the host binary. A ``/run`` drop-in puts
the directory holding a one-line shim first on the PATH of exactly one unit,
``gpu-fault-metrics-collector.service``; the shim runs this file. Every
invocation is forwarded to the real ``nvidia-smi`` unchanged -- product
discovery, driver/CUDA version, the ``-q -x`` limits, everything -- except the
one query the inventory delivery makes
(``--query-gpu=index,uuid,pci.bus_id,name --format=csv,noheader``,
``gpu_fault.collectors.gpu.discovery.query_gpu_inventory``), whose rows it
rewrites per a reviewed plan:

* ``replace_uuid`` -- one GPU row reports a replay UUID instead of its own;
* ``swap_bdf`` -- two GPU rows report each other's PCI bus id.

It never adds or drops a row, never changes an index or the product name and
never invents a successful answer: a failing real query is passed through as
it failed, and anything it cannot apply exactly (an unknown index, a replay
UUID that names a real GPU, a past deadline, an unreadable plan) passes the
real output through unchanged. Each inventory call is journaled with the
sha256 of the real and the served output. The host collector, the Node Agent
and every other unit keep reading the real binary.

Standard library only; runs as ``python -I`` from the unit's private PATH.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

INVENTORY_ARGV = ("--query-gpu=index,uuid,pci.bus_id,name", "--format=csv,noheader")
PLAN_ENV = "GFRS_LIVE_INVENTORY_PLAN"
AUDIT_ENV = "GFRS_LIVE_INVENTORY_AUDIT"
#: Where replay fixtures live; a PATH entry below it is never the real binary.
REPLAY_ROOT = "/run/gfrs-live"
GPU_UUID = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
MAX_OPERATIONS = 4
PROOF_SCOPE = "live-replay"


def validate_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"operations", "not_after"}:
        raise ValueError("plan must contain exactly operations and not_after")
    deadline = value["not_after"]
    if type(deadline) not in (int, float) or deadline <= 0:
        raise ValueError("not_after must be a positive epoch time")
    operations = value["operations"]
    if not isinstance(operations, list) or not 1 <= len(operations) <= MAX_OPERATIONS:
        raise ValueError(f"plan requires 1..{MAX_OPERATIONS} operations")
    touched: set[int] = set()
    for item in operations:
        if not isinstance(item, dict):
            raise ValueError("each operation must be an object")
        kind = item.get("op")
        if kind == "replace_uuid":
            if set(item) != {"op", "index", "uuid"}:
                raise ValueError("replace_uuid needs exactly op, index and uuid")
            indexes = [item["index"]]
            if (
                not isinstance(item["uuid"], str)
                or GPU_UUID.fullmatch(item["uuid"]) is None
            ):
                raise ValueError("replace_uuid needs a well-formed GPU UUID")
        elif kind == "swap_bdf":
            if set(item) != {"op", "indexes"}:
                raise ValueError("swap_bdf needs exactly op and indexes")
            indexes = item["indexes"]
            if (
                not isinstance(indexes, list)
                or len(indexes) != 2
                or len(set(indexes)) != 2
            ):
                raise ValueError("swap_bdf needs two distinct indexes")
        else:
            raise ValueError(f"operation {kind!r} is not reviewed")
        for index in indexes:
            if type(index) is not int or not 0 <= index <= 63:
                raise ValueError("a GPU index must be an integer in 0..63")
            if index in touched:
                raise ValueError("a GPU row may be rewritten by one operation only")
            touched.add(index)
    return {"operations": [dict(item) for item in operations], "not_after": deadline}


def rewrite(text: str, plan: dict[str, Any], *, now: float) -> tuple[str, list[str]]:
    """The served inventory text, and why it is unchanged (empty: rewritten)."""

    if now >= float(plan["not_after"]):
        return text, ["past the plan deadline"]
    lines = text.splitlines(keepends=True)
    rows: dict[int, tuple[int, list[str]]] = {}
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        parsed = next(csv.reader(io.StringIO(line)))
        if len(parsed) != 4:
            return text, [f"row {number} is not index,uuid,pci.bus_id,name"]
        index = parsed[0].strip()
        if not index.isdigit():
            return text, [f"row {number} has no numeric index"]
        rows[int(index)] = (number, [item.strip() for item in parsed])
    real_uuids = {fields[1] for _number, fields in rows.values()}
    served = list(lines)
    for item in plan["operations"]:
        if item["op"] == "replace_uuid":
            target = rows.get(item["index"])
            if target is None:
                return text, [f"GPU index {item['index']} is not in the real inventory"]
            if item["uuid"] in real_uuids:
                return text, ["the replay UUID names a real GPU"]
            number, fields = target
            served[number] = served[number].replace(fields[1], item["uuid"], 1)
        else:
            first, second = (rows.get(index) for index in item["indexes"])
            if first is None or second is None:
                return text, [f"GPU indexes {item['indexes']} are not both present"]
            (left, left_fields), (right, right_fields) = first, second
            if left_fields[2].lower() == right_fields[2].lower():
                return text, ["the two rows share one PCI bus id"]
            served[left] = served[left].replace(left_fields[2], right_fields[2], 1)
            served[right] = served[right].replace(right_fields[2], left_fields[2], 1)
    return "".join(served), []


def real_binary(path_value: str) -> str | None:
    """The first ``nvidia-smi`` on PATH outside the replay directory."""

    for entry in path_value.split(os.pathsep):
        if not entry or os.path.realpath(entry).startswith(REPLAY_ROOT):
            continue
        candidate = os.path.join(entry, "nvidia-smi")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            if not os.path.realpath(candidate).startswith(REPLAY_ROOT):
                return candidate
    return None


def load_plan() -> dict[str, Any] | None:
    path = os.environ.get(PLAN_ENV, "")
    if not path:
        return None
    try:
        return validate_plan(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def journal(record: dict[str, Any]) -> None:
    path = os.environ.get(AUDIT_ENV, "")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    real = real_binary(os.environ.get("PATH", ""))
    if real is None:
        sys.stderr.write("nvidia-smi replay wrapper: no real nvidia-smi on PATH\n")
        return 127
    if tuple(arguments) != INVENTORY_ARGV:
        os.execv(real, [real, *arguments])
    completed = subprocess.run(
        [real, *arguments], capture_output=True, text=True, timeout=14, check=False
    )
    served, unchanged = completed.stdout, ["the real query failed"]
    plan = load_plan()
    if completed.returncode == 0:
        if plan is None:
            unchanged = ["no reviewed plan"]
        else:
            served, unchanged = rewrite(completed.stdout, plan, now=time.time())
    journal(
        {
            "at": time.time(),
            "proof_scope": PROOF_SCOPE,
            "physical_fault_injected": False,
            "returncode": completed.returncode,
            "rewritten": not unchanged,
            "unchanged_reason": unchanged,
            "operations": plan["operations"]
            if plan is not None and not unchanged
            else [],
            "real_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            "served_sha256": hashlib.sha256(served.encode()).hexdigest(),
            "served": served if not unchanged else None,
        }
    )
    sys.stdout.write(served)
    sys.stderr.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
