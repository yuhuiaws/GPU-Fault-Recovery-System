from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ordered_aurora_instances(
    database: dict[str, Any],
    instances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    writers = {
        str(member.get("DBInstanceIdentifier") or "")
        for member in database.get("DBClusterMembers", [])
        if member.get("IsClusterWriter")
    }
    return sorted(
        instances,
        key=lambda instance: (
            str(instance.get("DBInstanceIdentifier") or "") in writers,
            str(instance.get("DBInstanceIdentifier") or ""),
        ),
    )


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)
