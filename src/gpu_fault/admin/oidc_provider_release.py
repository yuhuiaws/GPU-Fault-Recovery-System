"""Let a site-created executor OIDC provider go, on a double proof only.

``remove-cluster`` used to leave in place the IAM OIDC identity provider the site
had created for the removed GPU cluster (its registry row DETACHED): the
provider anchors every IRSA role in that EKS cluster, and the customer's own
roles may have come to trust it. Now ``remove-cluster`` -- and ``uninstall``,
for a provider an earlier removal left behind -- deletes it when both proofs
hold, and keeps it, as before, when either does not:

(a) ownership: the registry row is site-created (CREATED, DETACH) and the live
    provider carries the site tag (the cleaner's own ownership proof);
(b) no IAM role in the account names it in its trust policy -- one listing of
    every role (``aws iam list-roles``; the CLI follows every page) and every
    role's ``AssumeRolePolicyDocument`` read, matching the provider ARN or its
    issuer anywhere, a federated principal or a condition key alike. A failed
    read (permissions, a throttle the CLI gave up on), a role whose trust
    policy cannot be read, or a pagination marker left in the result is no
    proof.

A kept provider is one finding on stderr naming its ARN, the roles that trust it
and the command that deletes it by hand. Every verdict is returned for the
caller's journal and output. ``REQUIRED_PERMISSIONS`` is what the deploy-host
identity running the deletion verbs needs for it.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import unquote

from gpu_fault.admin.aws_cleanup_helpers import objects
from gpu_fault.admin.aws_cleanup_ownership import CleanupOwnership
from gpu_fault.admin.aws_commands import checked_command, json_command
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
)

REQUIRED_PERMISSIONS = (
    "iam:GetOpenIDConnectProvider",
    "iam:ListOpenIDConnectProviderTags",
    "iam:ListRoles",
    "iam:DeleteOpenIDConnectProvider",
)


class TrustNotProven(BootstrapError):
    """The role listing cannot prove that no role trusts the provider."""


def releasable_provider(resource: InstallationResource) -> bool:
    """A site-created executor OIDC provider row, the only kind released here."""

    return (
        resource.resource_type == "iam_oidc_provider"
        and resource.ownership is InstallationResourceOwnership.CREATED
        and resource.delete_policy is InstallationResourceDeletePolicy.DETACH
    )


def _trust_document(policy: object) -> dict[str, Any]:
    if isinstance(policy, str):
        for text in (policy, unquote(policy)):
            try:
                policy = json.loads(text)
                break
            except ValueError:
                continue
    if not isinstance(policy, dict):
        raise TrustNotProven("IAM role trust document is unavailable")
    return policy


def provider_trusting_roles(provider_arn: str) -> list[str]:
    """Every IAM role whose trust policy names ``provider_arn`` or its issuer."""

    document = json_command(["aws", "iam", "list-roles"])
    if document is None:
        raise TrustNotProven("the IAM role listing returned nothing")
    if any(document.get(key) for key in ("IsTruncated", "Marker", "NextToken")):
        raise TrustNotProven("the IAM role listing is incomplete (a page is missing)")
    issuer = provider_arn.split(":oidc-provider/", 1)[-1]
    trusting = set()
    for role in objects(document, "Roles", required=("RoleName",)):
        text = json.dumps(_trust_document(role.get("AssumeRolePolicyDocument")))
        if provider_arn in text or issuer in text:
            trusting.add(str(role["RoleName"]))
    return sorted(trusting)


def _finding(verdict: dict[str, Any]) -> None:
    roles = ", ".join(verdict["trusting_roles"]) or "none read"
    print(
        f"finding: OIDC provider {verdict['provider_arn']} "
        f"({verdict['resource_key']}) kept: {verdict['reason']}; trusted by: "
        f"{roles}; once nothing trusts it, delete it by hand: "
        f"{verdict['manual_command']}",
        file=sys.stderr,
    )


def release_oidc_provider(
    resource: InstallationResource,
    *,
    ownership: CleanupOwnership,
    exists: Callable[[InstallationResource], bool],
    wait_absent: Callable[[InstallationResource], None],
    member: bool,
    recorded_elsewhere: Sequence[str] = (),
) -> dict[str, Any]:
    """Delete the provider if both proofs hold; the verdict, either way.

    ``member`` is whether the provider's cluster is still a site cluster (the
    cleaner's scope check names it); a provider an earlier removal left behind
    belongs to a cluster the site no longer lists. ``recorded_elsewhere`` are
    other live registry rows naming the same provider: it stays while any does.
    """

    if not releasable_provider(resource):
        raise BootstrapError(
            f"{resource.resource_key} is not a site-created OIDC provider row"
        )
    arn = resource.resource_arn or resource.resource_id
    verdict: dict[str, Any] = {
        "resource_key": resource.resource_key,
        "provider_arn": arn,
        "outcome": "kept",
        "site_tagged": None,
        "trusting_roles": [],
        "reason": None,
        "manual_command": (
            f"aws iam delete-open-id-connect-provider --open-id-connect-provider-arn {arn}"
        ),
        "decided_at": datetime.now(UTC).isoformat(),
    }

    def keep(reason: str) -> dict[str, Any]:
        verdict["reason"] = reason
        _finding(verdict)
        return verdict

    try:
        ownership.validate_scope(resource, member=member)
        present = exists(resource)
    except BootstrapError as exc:
        return keep(f"scope or existence not proven: {exc}")
    if not present:
        verdict.update(outcome="absent", manual_command=None)
        return verdict
    if recorded_elsewhere:
        return keep(
            "the registry still records it for " + ", ".join(recorded_elsewhere)
        )
    try:
        ownership.prove_owned(resource, member=member)
    except BootstrapError as exc:
        verdict["site_tagged"] = False
        return keep(f"ownership not proven: {exc}")
    verdict["site_tagged"] = True
    try:
        verdict["trusting_roles"] = provider_trusting_roles(arn)
    except BootstrapError as exc:
        return keep(f"unused not proven: {exc}")
    if verdict["trusting_roles"]:
        return keep("IAM roles still trust it")
    try:
        checked_command(
            [
                "aws",
                "iam",
                "delete-open-id-connect-provider",
                "--open-id-connect-provider-arn",
                arn,
            ],
            not_found=("NoSuchEntity",),
        )
    except BootstrapError as exc:
        return keep(f"the deletion was refused: {exc}")
    wait_absent(resource)
    verdict.update(outcome="deleted", manual_command=None)
    return verdict
