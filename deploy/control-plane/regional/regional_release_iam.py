from __future__ import annotations

import json
from typing import Any

from regional_release_config import ClusterTarget, ReleaseError

EXECUTOR_SAGEMAKER_ACTIONS = frozenset(
    {
        "sagemaker:describecluster",
        "sagemaker:listclusternodes",
        "sagemaker:describeclusternode",
        "sagemaker:batchrebootclusternodes",
    }
)


def validate_executor_iam_documents(
    role_arn: str,
    documents: list[dict[str, Any]],
) -> None:
    forbidden = []
    for document in documents:
        for statement in document.get("Statement", []):
            if statement.get("Effect") != "Allow":
                continue
            if statement.get("NotAction") is not None:
                forbidden.append("Allow/NotAction")
                continue
            raw = statement.get("Action", [])
            actions = raw if isinstance(raw, list) else [raw]
            for action in actions:
                normalized = str(action).lower()
                if normalized.startswith("ses:"):
                    forbidden.append(str(action))
                elif (
                    normalized.startswith("sagemaker:")
                    and normalized not in EXECUTOR_SAGEMAKER_ACTIONS
                ):
                    forbidden.append(str(action))
    if forbidden:
        raise ReleaseError(
            f"executor role {role_arn} exceeds the regional "
            "data-plane boundary: " + ", ".join(sorted(set(forbidden)))
        )


def validate_executor_iam_role(release: Any, target: ClusterTarget) -> None:
    role_name = target.executor_irsa_role_arn.rsplit("/", 1)[-1]
    inline = json.loads(
        release.runner.run(
            [
                "aws",
                "iam",
                "list-role-policies",
                "--role-name",
                role_name,
                "--output",
                "json",
            ],
            capture=True,
        )
    )
    documents = []
    for name in inline.get("PolicyNames", []):
        value = json.loads(
            release.runner.run(
                [
                    "aws",
                    "iam",
                    "get-role-policy",
                    "--role-name",
                    role_name,
                    "--policy-name",
                    name,
                    "--output",
                    "json",
                ],
                capture=True,
            )
        )
        documents.append(value["PolicyDocument"])
    attached = json.loads(
        release.runner.run(
            [
                "aws",
                "iam",
                "list-attached-role-policies",
                "--role-name",
                role_name,
                "--output",
                "json",
            ],
            capture=True,
        )
    )
    for policy in attached.get("AttachedPolicies", []):
        metadata = json.loads(
            release.runner.run(
                [
                    "aws",
                    "iam",
                    "get-policy",
                    "--policy-arn",
                    policy["PolicyArn"],
                    "--output",
                    "json",
                ],
                capture=True,
            )
        )
        version = metadata["Policy"]["DefaultVersionId"]
        value = json.loads(
            release.runner.run(
                [
                    "aws",
                    "iam",
                    "get-policy-version",
                    "--policy-arn",
                    policy["PolicyArn"],
                    "--version-id",
                    version,
                    "--output",
                    "json",
                ],
                capture=True,
            )
        )
        documents.append(value["PolicyVersion"]["Document"])
    validate_executor_iam_documents(target.executor_irsa_role_arn, documents)
