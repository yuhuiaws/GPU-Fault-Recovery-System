from __future__ import annotations

from typing import Any

from gpu_fault.admin.aws_commands import json_command


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


def aurora_instance_deleting_or_absent(cleaner: Any, instance_id: str) -> bool:
    """Whether ``instance_id`` is gone or already in ``deleting``.

    That is all ``delete-db-cluster`` needs from the instances, so the cluster
    deletion does not wait for each instance to disappear in turn.
    """

    document = json_command(
        cleaner._aws(
            "rds", "describe-db-instances", "--db-instance-identifier", instance_id
        ),
        not_found=("DBInstanceNotFound",),
    )
    if document is None:
        return True
    instances = document.get("DBInstances", [])
    status = str((instances[0] if instances else {}).get("DBInstanceStatus") or "")
    return not instances or status.lower() == "deleting"
