from __future__ import annotations

from typing import Any

from gpu_fault.admin.aws_commands import (
    json_command,
    matches_not_found,
    run_command,
    wait_until,
)
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.installation_resources import InstallationResource


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


def delete_certificate_once_released(
    cleaner: Any, resource: InstallationResource, arn: str
) -> None:
    """Delete the certificate when ACM no longer sees its listener.

    The NLB is deleted and gone before this runs, but ACM keeps reporting the
    certificate as in use by the vanished listener for minutes afterwards (live
    2026-09-13: ``ResourceInUseException`` ten minutes after
    ``delete-load-balancer``, and the uninstall stopped there). Retry that one
    refusal until the association clears; every other failure is raised as
    before.
    """

    def attempt() -> bool:
        result = run_command(
            cleaner._aws("acm", "delete-certificate", "--certificate-arn", arn)
        )
        if result.returncode == 0:
            return True
        if matches_not_found(result, ("ResourceNotFoundException",)):
            return True
        if "ResourceInUseException" in result.stderr:
            return False
        raise BootstrapError(
            f"command failed ({result.returncode}): aws acm delete-certificate: "
            f"{result.stderr.strip()}"
        )

    wait_until(
        attempt,
        description=f"{resource.resource_key} release by its listener",
        timeout_seconds=900,
        interval_seconds=15,
    )
