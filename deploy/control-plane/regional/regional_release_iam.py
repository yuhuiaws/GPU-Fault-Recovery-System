from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

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


def _iam_json(release: Any, arguments: list[str]) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(
        release.runner.run(["aws", "iam", *arguments, "--output", "json"], capture=True)
    )
    return value


def _fan_out(release: Any, fetches: list[Callable[[], dict[str, Any]]]) -> list[Any]:
    """Run independent IAM reads at once, keeping the caller's order.

    Expanding one role is a wide tree of small reads -- every inline policy, and
    two reads for every attached one -- and each is its own `aws` process, so the
    cost is almost entirely round trips that do not depend on each other. `status`
    expands one role per GPU cluster, which is why this is worth widening rather
    than leaving as the obvious loop.

    `map` keeps the results in submission order: the documents feed an error
    message, and an administrator comparing two runs of the same check should not
    see the reasons reordered by whichever read finished first.
    """

    if not fetches:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(fetches))) as executor:
        return list(executor.map(lambda fetch: fetch(), fetches))


def _attached_policy_document(release: Any, policy_arn: str) -> dict[str, Any]:
    # The version has to be read before the document can be, so these two stay
    # sequential inside one worker rather than becoming two rounds of fan-out.
    metadata = _iam_json(release, ["get-policy", "--policy-arn", policy_arn])
    version = metadata["Policy"]["DefaultVersionId"]
    value = _iam_json(
        release,
        ["get-policy-version", "--policy-arn", policy_arn, "--version-id", version],
    )
    document: dict[str, Any] = value["PolicyVersion"]["Document"]
    return document


def validate_executor_iam_role(release: Any, target: ClusterTarget) -> None:
    role_name = target.executor_irsa_role_arn.rsplit("/", 1)[-1]
    inline, attached = _fan_out(
        release,
        [
            lambda: _iam_json(
                release, ["list-role-policies", "--role-name", role_name]
            ),
            lambda: _iam_json(
                release, ["list-attached-role-policies", "--role-name", role_name]
            ),
        ],
    )
    fetches: list[Callable[[], dict[str, Any]]] = [
        lambda name=name: _iam_json(  # type: ignore[misc]
            release,
            ["get-role-policy", "--role-name", role_name, "--policy-name", name],
        )["PolicyDocument"]
        for name in inline.get("PolicyNames", [])
    ]
    fetches.extend(
        lambda arn=policy["PolicyArn"]: _attached_policy_document(release, arn)  # type: ignore[misc]
        for policy in attached.get("AttachedPolicies", [])
    )
    documents = _fan_out(release, fetches)
    validate_executor_iam_documents(target.executor_irsa_role_arn, documents)
