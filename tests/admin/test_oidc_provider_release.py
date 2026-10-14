"""remove-cluster lets a site-created executor OIDC provider go on a double proof.

(a) the registry row is site-created and the live provider carries the site
tag; (b) one complete listing of every IAM role names it in no trust policy.
Anything less keeps the provider (its row DETACHED, as before) with one finding
naming the ARN, the trusting roles and the manual command. These run the real
removal step and the real cleaner against IAM fakes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from gpu_fault.admin import aws_commands
from gpu_fault.admin.aws_cleanup import ResourceCleaner
from gpu_fault.admin.cluster_removal_resources import remove_target_resources
from gpu_fault.admin.oidc_provider_release import REQUIRED_PERMISSIONS
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceSnapshot,
)
from gpu_fault.installation_resources import InstallationResourceDeletePolicy as Policy
from gpu_fault.installation_resources import InstallationResourceOwnership as Ownership
from gpu_fault.installation_resources import InstallationResourceStatus as Status
from tests.admin._aws_cleanup_support import ACCOUNT, TAGS, Aws, absent
from tests.admin.test_admin_site import site_file

ISSUER = "oidc.eks.us-east-1.amazonaws.com/id/GPUA0001"
PROVIDER = f"arn:aws:iam::{ACCOUNT}:oidc-provider/{ISSUER}"
EXECUTOR = "gpu-fault-test-site-hp-gpu-a-executor"
PROVIDER_KEY = "aws/iam/executor/gpu-a/oidc-provider"
ROLE_KEY = "aws/iam/executor/gpu-a/role"


def _row(key: str, kind: str, identifier: str, **updates: Any) -> InstallationResource:
    return InstallationResource(
        site_id="test-site",
        resource_key=key,
        resource_type=kind,
        resource_id=identifier,
        region="us-east-1",
        account_id=ACCOUNT,
        ownership=Ownership.CREATED,
        delete_policy=Policy.DELETE,
    ).model_copy(update=updates)


def _registry(**provider: Any) -> InstallationResourceSnapshot:
    rows = [
        _row(
            ROLE_KEY,
            "iam_role",
            EXECUTOR,
            resource_arn=f"arn:aws:iam::{ACCOUNT}:role/{EXECUTOR}",
            attributes={"inline_policy_name": "GPUFaultRegionalExecutor"},
        ),
        _row(
            PROVIDER_KEY,
            "iam_oidc_provider",
            PROVIDER,
            **{
                "resource_arn": PROVIDER,
                "delete_policy": Policy.DETACH,
                "attributes": {"cluster_name": "gpu-a"},
                **provider,
            },
        ),
    ]
    value = InstallationResourceSnapshot(site_id="test-site", resources=rows)
    return value.model_copy(update={"source_sha256": value.digest()})


def _trust(*principals: str, condition: str | None = None) -> dict[str, Any]:
    statement: dict[str, Any] = {"Principal": {"Federated": list(principals)}}
    if condition is not None:
        statement["Condition"] = {"StringEquals": {f"{condition}:aud": "sts"}}
    return {"Statement": [statement]}


class Iam:
    """IAM as the removal sees it: the executor role, the provider, the rest."""

    def __init__(self) -> None:
        self.roles: dict[str, Any] = {EXECUTOR: _trust(PROVIDER)}
        self.provider_tags: list[dict[str, str]] | None = TAGS
        self.listing: Any = None
        self.calls: list[str] = []

    def __call__(self, arguments: list[str], **_kwargs: Any) -> Any:
        self.calls.append(arguments[2])
        return self.aws(arguments)

    def _role(self, arguments: list[str]) -> Any:
        name = arguments[arguments.index("--role-name") + 1]
        if name not in self.roles:
            return absent("NoSuchEntity")
        return {
            "Role": {"RoleName": name, "Arn": f"arn:aws:iam::{ACCOUNT}:role/{name}"}
        }

    def _delete_role(self, arguments: list[str]) -> Any:
        self.roles.pop(arguments[arguments.index("--role-name") + 1], None)
        return {}

    def _provider(self, _arguments: list[str]) -> Any:
        return (
            {"Url": ISSUER}
            if self.provider_tags is not None
            else absent("NoSuchEntity")
        )

    def _delete_provider(self, _arguments: list[str]) -> Any:
        self.provider_tags = None
        return {}

    def _list_roles(self, _arguments: list[str]) -> Any:
        if self.listing is not None:
            return self.listing
        return {
            "Roles": [
                {"RoleName": name, "AssumeRolePolicyDocument": document}
                for name, document in self.roles.items()
            ]
        }

    @property
    def aws(self) -> Aws:
        return Aws(
            {
                ("iam", "get-role"): self._role,
                ("iam", "list-role-tags"): {"Tags": TAGS},
                ("iam", "list-instance-profiles-for-role"): {"InstanceProfiles": []},
                ("iam", "list-role-policies"): {"PolicyNames": []},
                ("iam", "list-attached-role-policies"): {"AttachedPolicies": []},
                ("iam", "delete-role"): self._delete_role,
                ("iam", "get-open-id-connect-provider"): self._provider,
                ("iam", "list-open-id-connect-provider-tags"): lambda _a: {
                    "Tags": self.provider_tags or []
                },
                ("iam", "list-roles"): self._list_roles,
                ("iam", "delete-open-id-connect-provider"): self._delete_provider,
            }
        )


@pytest.fixture
def iam(monkeypatch: pytest.MonkeyPatch) -> Iam:
    fake = Iam()
    monkeypatch.setattr(aws_commands, "bounded_command", fake)
    return fake


def _remove(tmp_path: Path, registry: InstallationResourceSnapshot):
    updated, evidence = remove_target_resources(
        load_site(site_file(tmp_path)),
        "gpu-a",
        registry,
        cleaner_factory=ResourceCleaner,
    )
    statuses = {row.resource_key: row.status for row in updated.resources}
    (verdict,) = evidence["oidc_providers"]
    return statuses, evidence, verdict


def test_a_provider_nothing_else_trusts_is_deleted_after_the_roles(
    tmp_path: Path, iam: Iam
) -> None:
    statuses, evidence, verdict = _remove(tmp_path, _registry())

    assert iam.calls.index("delete-role") < iam.calls.index("list-roles"), (
        "the trust listing must come after the cluster's own roles are gone"
    )
    assert iam.provider_tags is None, "the provider was not deleted"
    assert verdict["outcome"] == "deleted" and verdict["site_tagged"] is True
    assert verdict["trusting_roles"] == [] and verdict["manual_command"] is None
    assert statuses[PROVIDER_KEY] is Status.DELETED
    assert evidence["deleted"] == [PROVIDER_KEY, ROLE_KEY]


@pytest.mark.parametrize(
    "trust",
    [
        _trust(PROVIDER),
        quote(str(_trust(PROVIDER)).replace("'", '"')),
        _trust("arn:aws:iam::123456789012:oidc-provider/other", condition=ISSUER),
    ],
    ids=["federated", "url-encoded", "condition-only"],
)
def test_a_provider_a_customer_role_trusts_is_kept_with_one_finding(
    tmp_path: Path, iam: Iam, capsys: pytest.CaptureFixture[str], trust: Any
) -> None:
    iam.roles["customer-irsa"] = trust

    statuses, evidence, verdict = _remove(tmp_path, _registry())

    assert iam.provider_tags == TAGS, "a trusted provider was deleted"
    assert (verdict["outcome"], verdict["trusting_roles"]) == (
        "kept",
        ["customer-irsa"],
    )
    assert statuses[PROVIDER_KEY] is Status.DETACHED, "kept rows stay DETACHED"
    assert evidence["detached"] == [PROVIDER_KEY]
    (line,) = capsys.readouterr().err.strip().splitlines()
    assert line.startswith(
        f"finding: OIDC provider {PROVIDER} ({PROVIDER_KEY}) kept"
    ), line
    assert "trusted by: customer-irsa" in line
    assert line.endswith(
        f"aws iam delete-open-id-connect-provider --open-id-connect-provider-arn {PROVIDER}"
    ), line


@pytest.mark.parametrize(
    ("listing", "reason"),
    [
        (
            subprocess.CompletedProcess(
                ["aws"],
                254,
                "",
                "An error occurred (Throttling) when calling ListRoles",
            ),
            "unused not proven",
        ),
        ({"Roles": [], "IsTruncated": True, "Marker": "next"}, "a page is missing"),
        ({"Roles": [{"RoleName": "opaque"}]}, "trust document is unavailable"),
        (
            {"Roles": [{"RoleName": "garbled", "AssumeRolePolicyDocument": "%7Bno"}]},
            "trust document is unavailable",
        ),
    ],
    ids=["throttled", "paginated", "no-trust-policy", "unreadable-trust-policy"],
)
def test_an_incomplete_role_listing_is_no_proof(
    tmp_path: Path, iam: Iam, listing: Any, reason: str
) -> None:
    iam.listing = listing

    statuses, _, verdict = _remove(tmp_path, _registry())

    assert verdict["outcome"] == "kept" and reason in verdict["reason"]
    assert "delete-open-id-connect-provider" not in iam.calls
    assert statuses[PROVIDER_KEY] is Status.DETACHED


def test_a_provider_without_the_site_tag_is_kept(tmp_path: Path, iam: Iam) -> None:
    iam.provider_tags = []

    _, _, verdict = _remove(tmp_path, _registry())

    assert (verdict["outcome"], verdict["site_tagged"]) == ("kept", False)
    assert "without gpu-fault:site-id" in verdict["reason"]
    assert "list-roles" not in iam.calls, "proof (b) runs only after (a) holds"
    assert "delete-open-id-connect-provider" not in iam.calls


def test_a_provider_another_live_row_records_is_kept(tmp_path: Path, iam: Iam) -> None:
    registry = _registry()
    other = _row(
        "aws/iam/executor/gpu-b/oidc-provider",
        "iam_oidc_provider",
        PROVIDER,
        resource_arn=PROVIDER,
        delete_policy=Policy.DETACH,
    )
    shared = registry.model_copy(update={"resources": [*registry.resources, other]})
    shared = shared.model_copy(update={"source_sha256": shared.digest()})

    _, _, verdict = _remove(tmp_path, shared)

    assert verdict["outcome"] == "kept"
    assert verdict["reason"] == (
        "the registry still records it for aws/iam/executor/gpu-b/oidc-provider"
    )
    assert "delete-open-id-connect-provider" not in iam.calls


def test_an_already_absent_provider_is_recorded_deleted(
    tmp_path: Path, iam: Iam
) -> None:
    iam.provider_tags = None

    statuses, _, verdict = _remove(tmp_path, _registry())

    assert verdict["outcome"] == "absent" and statuses[PROVIDER_KEY] is Status.DELETED
    assert "delete-open-id-connect-provider" not in iam.calls


def test_a_provider_the_site_did_not_create_is_left_alone(
    tmp_path: Path, iam: Iam
) -> None:
    registry = _registry(ownership=Ownership.EXTERNAL, delete_policy=Policy.PRESERVE)

    updated, evidence = remove_target_resources(
        load_site(site_file(tmp_path)),
        "gpu-a",
        registry,
        cleaner_factory=ResourceCleaner,
    )

    assert evidence["oidc_providers"] == []
    assert not {"get-open-id-connect-provider", "list-roles"} & set(iam.calls)
    statuses = {row.resource_key: row.status for row in updated.resources}
    assert statuses[PROVIDER_KEY] is Status.DETACHED


def test_the_required_permissions_are_named() -> None:
    assert REQUIRED_PERMISSIONS == (
        "iam:GetOpenIDConnectProvider",
        "iam:ListOpenIDConnectProviderTags",
        "iam:ListRoles",
        "iam:DeleteOpenIDConnectProvider",
    )
