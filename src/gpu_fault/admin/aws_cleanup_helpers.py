from __future__ import annotations

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
