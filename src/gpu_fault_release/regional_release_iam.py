from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from gpu_fault.cluster_scope import ClusterProvider, cluster_provider
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from gpu_fault_release.regional_release_state import aws_json

EXECUTOR_SAGEMAKER_ACTIONS = frozenset(
    {
        "sagemaker:describecluster",
        "sagemaker:listclusternodes",
        "sagemaker:describeclusternode",
        "sagemaker:batchrebootclusternodes",
    }
)
#: What a plain EKS (``eks-managed``) cluster's executor may hold on EC2/EKS:
#: read the instances behind the nodes, reboot them, and nothing that stops,
#: terminates or launches -- a replacement never comes from the data plane.
EXECUTOR_EKS_MANAGED_ACTIONS = frozenset(
    {
        "ec2:describeinstances",
        "ec2:describeinstancestatus",
        "ec2:rebootinstances",
        "eks:describecluster",
        "eks:describenodegroup",
        "eks:listnodegroups",
    }
)
#: Reads EC2 only grants on ``Resource: "*"``; the one place the executor
#: boundary tolerates the wildcard resource, and only for these actions.
WILDCARD_RESOURCE_READS = frozenset(
    {"ec2:describeinstances", "ec2:describeinstancestatus"}
)


def validate_executor_iam_documents(
    role_arn: str,
    documents: list[dict[str, Any]],
    *,
    provider: str = ClusterProvider.HYPERPOD.value,
) -> None:
    """Refuse an executor role that exceeds the data-plane boundary.

    HyperPod roles are judged exactly as before. An ``eks-managed`` role may
    hold no ``sagemaker:`` action at all and only the EC2/EKS actions in
    ``EXECUTOR_EKS_MANAGED_ACTIONS``; ``ec2:DescribeInstances`` may use
    ``Resource: "*"`` because EC2 offers no narrower resource for it.
    """

    eks_managed = cluster_provider(provider) is ClusterProvider.EKS_MANAGED
    forbidden = []
    for document in documents:
        for statement in document.get("Statement", []):
            if statement.get("Effect") != "Allow":
                continue
            if statement.get("NotAction") is not None:
                forbidden.append("Allow/NotAction")
                continue
            if statement.get("NotResource") is not None:
                forbidden.append("Allow/NotResource")
                continue
            raw = statement.get("Action", [])
            actions = raw if isinstance(raw, list) else [raw]
            normalized_actions = {str(action).lower() for action in actions}
            raw_resource = statement.get("Resource", [])
            resources = (
                raw_resource if isinstance(raw_resource, list) else [raw_resource]
            )
            for resource in resources:
                if str(resource).strip() == "*" and not (
                    eks_managed
                    and normalized_actions
                    and normalized_actions <= WILDCARD_RESOURCE_READS
                ):
                    forbidden.append('Resource "*"')
            for action in actions:
                normalized = str(action).lower()
                # Least privilege: reject the account-wide wildcard and any
                # service-wide wildcard (`iam:*`, `sts:*`, `sagemaker:*`, ...).
                if normalized == "*" or normalized.endswith(":*"):
                    forbidden.append(str(action))
                elif normalized.startswith("ses:"):
                    forbidden.append(str(action))
                elif eks_managed and normalized.startswith("sagemaker:"):
                    forbidden.append(str(action))
                elif (
                    eks_managed
                    and normalized.startswith(("ec2:", "eks:"))
                    and normalized not in EXECUTOR_EKS_MANAGED_ACTIONS
                ):
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
    # No `--region`: IAM is global, and passing one would make the cache key
    # differ from the command actually issued.
    #
    # Through the snapshot cache because a fleet's roles overlap. Every cluster's
    # executor role tends to attach the same managed policies, so expanding role
    # N repeats the `get-policy` and `get-policy-version` pair that role N-1
    # already resolved -- and the expansions run concurrently, on a thread pool
    # per cluster inside a thread pool over clusters.
    return aws_json(release, ["iam", *arguments], region=False)


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
    validate_executor_iam_documents(
        target.executor_irsa_role_arn,
        documents,
        provider=cluster_provider(getattr(target, "provider", None)).value,
    )
