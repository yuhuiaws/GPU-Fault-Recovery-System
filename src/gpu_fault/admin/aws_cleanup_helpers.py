from __future__ import annotations

import json
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.aws_commands import matches_not_found, run_command, wait_until
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.installation_resources import InstallationResource

if TYPE_CHECKING:
    from gpu_fault.admin.aws_cleanup import ResourceDeletion


def objects(
    document: dict[str, Any], key: str, *, required: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    values = document.get(key)
    if not isinstance(values, list) or any(
        not isinstance(value, dict) for value in values
    ):
        raise BootstrapError(f"cleanup response has no valid {key} list")
    if any(
        not isinstance(item.get(field), str) or not item[field]
        for item in values
        for field in required
    ):
        raise BootstrapError(f"cleanup response has incomplete {key} identities")
    return values


def object_field(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise BootstrapError(f"cleanup response has no valid {key} object")
    return value


def strings(document: dict[str, Any], key: str) -> list[str]:
    value = document.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise BootstrapError(f"cleanup response has no valid {key} list")
    if len(set(value)) != len(value):
        raise BootstrapError(f"cleanup response has duplicate {key} identities")
    return value


def single_object(document: dict[str, Any], key: str) -> dict[str, Any]:
    values = objects(document, key)
    if len(values) != 1:
        raise BootstrapError(f"cleanup response has no unique {key} identity")
    return values[0]


def strict_tags(value: object) -> dict[str, str]:
    if isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        return cast(dict[str, str], value)
    if not isinstance(value, list):
        raise BootstrapError("cleanup ownership tags are unavailable")
    result: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict):
            raise BootstrapError("cleanup ownership tags are malformed")
        key = item.get("Key", item.get("key"))
        content = item.get("Value", item.get("value"))
        if not isinstance(key, str) or not isinstance(content, str) or key in result:
            raise BootstrapError("cleanup ownership tags are malformed or ambiguous")
        result[key] = content
    return result


def sqs_policy(raw: object) -> dict[str, Any]:
    if raw in (None, ""):
        return {"Statement": []}
    if not isinstance(raw, str):
        raise BootstrapError("SQS queue policy is not a JSON string")
    try:
        policy = json.loads(raw)
    except ValueError:
        raise BootstrapError("SQS queue policy is invalid JSON") from None
    if not isinstance(policy, dict):
        raise BootstrapError("SQS queue policy is not a JSON object")
    statements = policy.get("Statement")
    if isinstance(statements, dict):
        policy["Statement"] = [statements]
    objects(policy, "Statement")
    return policy


def sqs_topic_statements(
    policy: dict[str, Any], topic_arn: str
) -> list[dict[str, Any]]:
    matched = []
    for statement in objects(policy, "Statement"):
        conditions = statement.get("Condition", {})
        if not isinstance(conditions, dict):
            raise BootstrapError("SQS queue policy has malformed conditions")
        for operator, condition in conditions.items():
            if not isinstance(condition, dict):
                raise BootstrapError("SQS queue policy has malformed conditions")
            for key, value in condition.items():
                if str(key).casefold() != "aws:sourcearn":
                    continue
                values = value if isinstance(value, list) else [value]
                if any(
                    isinstance(item, str)
                    and item != topic_arn
                    and fnmatchcase(topic_arn, item)
                    for item in values
                ):
                    raise BootstrapError(
                        "SQS topic binding is covered by a shared wildcard"
                    )
                if topic_arn not in values:
                    continue
                if (
                    operator not in {"ArnEquals", "ArnLike", "StringEquals"}
                    or values != [topic_arn]
                    or statement.get("Effect") != "Allow"
                ):
                    raise BootstrapError(
                        "SQS topic binding shares or ambiguously restricts a statement"
                    )
                matched.append(statement)
    return matched


def oidc_provider_users(document: dict[str, Any], provider: str) -> list[str]:
    users = []
    for role in objects(document, "Roles"):
        policy = role.get("AssumeRolePolicyDocument")
        if isinstance(policy, str):
            try:
                policy = json.loads(policy)
            except ValueError:
                try:
                    policy = json.loads(unquote(policy))
                except ValueError:
                    raise BootstrapError(
                        "IAM role trust document is unavailable"
                    ) from None
        if not isinstance(policy, dict):
            raise BootstrapError("IAM role trust document is unavailable")
        statements = policy.get("Statement")
        if isinstance(statements, dict):
            statements = [statements]
        for statement in objects({"Statement": statements}, "Statement"):
            principal = object_field(statement, "Principal")
            federated = principal.get("Federated", [])
            if isinstance(federated, str):
                federated = [federated]
            if provider in strings({"Federated": federated}, "Federated"):
                users.append(str(role.get("RoleName") or "unknown"))
    return users


def ordered_aurora_instances(
    database: dict[str, Any],
    instances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    members = objects(database, "DBClusterMembers")
    member_ids = [member.get("DBInstanceIdentifier") for member in members]
    instance_ids = [instance.get("DBInstanceIdentifier") for instance in instances]
    if (
        any(
            not isinstance(value, str) or not value
            for value in member_ids + instance_ids
        )
        or len(set(member_ids)) != len(member_ids)
        or len(set(instance_ids)) != len(instance_ids)
        or any(
            not isinstance(member.get("IsClusterWriter"), bool) for member in members
        )
        or set(member_ids) - set(instance_ids)
        or any(
            instance["DBInstanceIdentifier"] not in member_ids
            and instance.get("DBInstanceStatus") != "deleting"
            for instance in instances
        )
    ):
        raise BootstrapError("Aurora membership is incomplete or ambiguous")
    writers = {
        str(member["DBInstanceIdentifier"])
        for member in members
        if member["IsClusterWriter"]
    }
    if len(writers) != 1 and any(
        instance.get("DBInstanceStatus") != "deleting" for instance in instances
    ):
        raise BootstrapError("Aurora writer identity is unavailable")
    return sorted(
        instances,
        key=lambda instance: (
            str(instance.get("DBInstanceIdentifier") or "") in writers,
            str(instance.get("DBInstanceIdentifier") or ""),
        ),
    )


def delete_certificate_once_released(
    cleaner: ResourceDeletion, resource: InstallationResource, arn: str
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
        if cleaner._ownership.before_delete(resource, cleaner.exists) is None:
            return True
        result = run_command(
            cleaner._aws("acm", "delete-certificate", "--certificate-arn", arn)
        )
        if result.returncode == 0:
            return True
        if matches_not_found(result, ("ResourceNotFoundException",)):
            return True
        if matches_not_found(result, ("ResourceInUseException",)):
            return False
        raise BootstrapError(
            f"command failed ({result.returncode}): aws acm delete-certificate: "
            f"{diagnostic_text(result.stderr.strip())}"
        )

    wait_until(
        attempt,
        description=f"{resource.resource_key} release by its listener",
        timeout_seconds=900,
        interval_seconds=15,
    )
