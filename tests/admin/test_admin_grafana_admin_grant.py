"""The automatic Grafana ADMIN grant for the site administrator.

After the dashboards are imported the deploy derives the operator from the
administrator email: ``sso-admin list-instances`` names the identity store,
``identitystore get-user-id`` (``emails.value``, then ``userName``) names the
user, ``grafana list-permissions`` makes the grant idempotent and
``update-permissions`` grants ADMIN. Nothing here may fail the deploy: every
way the derivation can miss ends in a ``not-derivable`` record, the printed
``update-permissions`` command and one sentence naming what was missing. Every
AWS call goes through the recorded fake runner of ``test_admin_grafana``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapMutationRequired,
    BootstrapState,
    ReadOnlyProbeRunner,
)
from gpu_fault.admin.grafana import (
    GrafanaSettings,
    ensure_grafana_dashboards,
    grafana_access_lines,
)
from tests.admin._bootstrap_support import _cluster
from tests.admin.test_admin_grafana import (
    AMP,
    HYPERPOD_WORKSPACE,
    REGION,
    SITE,
    Http,
    Runner,
    _dashboards,
)

ADMIN_EMAIL = "ops@example.com"
INSTANCE = {
    "InstanceArn": "arn:aws:sso:::instance/ssoins-1",
    "IdentityStoreId": "d-1234567890",
}
SECOND_INSTANCE = {
    "InstanceArn": "arn:aws:sso:::instance/ssoins-2",
    "IdentityStoreId": "d-0987654321",
}
USER_ID = "9a8b7c6d-1234-5678-9abc-def012345678"


class IdentityRunner(Runner):
    """The Grafana runner plus IAM Identity Center and the permission listing.

    ``users`` maps ``(attribute_path, value)`` to a user id, so a test decides
    whether ``emails.value`` or ``userName`` finds the administrator.
    """

    def __init__(
        self,
        workspaces: Sequence[Mapping[str, Any]] = (HYPERPOD_WORKSPACE,),
        *,
        instances: Sequence[Mapping[str, Any]] = (INSTANCE,),
        users: Mapping[tuple[str, str], str] | None = None,
        permissions: Sequence[Mapping[str, Any]] = (),
        identitystore_denied: bool = False,
        **keywords: Any,
    ) -> None:
        super().__init__(workspaces, **keywords)
        self.instances = [dict(item) for item in instances]
        self.users = dict(users or {})
        self.permissions = [dict(item) for item in permissions]
        self.identitystore_denied = identitystore_denied

    def run(self, arguments: Sequence[str], **keywords: Any) -> str:
        argv = list(arguments)
        if argv[1] == "sso-admin":
            self.calls.append((argv, keywords))
            assert argv[2] == "list-instances", argv
            assert not keywords.get("mutate"), "listing instances is a read"
            return json.dumps({"Instances": self.instances})
        if argv[1] == "identitystore":
            self.calls.append((argv, keywords))
            return self._identitystore(argv, keywords)
        if argv[1] == "grafana" and argv[2] == "list-permissions":
            self.calls.append((argv, keywords))
            assert not keywords.get("mutate"), "listing permissions is a read"
            assert argv[argv.index("--user-type") + 1] == "SSO_USER"
            return json.dumps({"permissions": self.permissions})
        return super().run(arguments, **keywords)

    def _identitystore(self, argv: list[str], keywords: Any) -> str:
        assert argv[2] == "get-user-id", argv
        assert not keywords.get("mutate"), "looking a user up is a read"
        if self.identitystore_denied:
            raise BootstrapError(
                "command failed (254): aws identitystore get-user-id: An error "
                "occurred (AccessDeniedException) when calling the GetUserId "
                "operation: User: arn:aws:iam::123456789012:user/deployer is not "
                "authorized to perform: identitystore:GetUserId"
            )
        assert (
            argv[argv.index("--identity-store-id") + 1] == INSTANCE["IdentityStoreId"]
        )
        identifier = json.loads(argv[argv.index("--alternate-identifier") + 1])
        attribute = identifier["UniqueAttribute"]
        key = (attribute["AttributePath"], attribute["AttributeValue"])
        if key not in self.users:
            raise BootstrapError(
                "command failed (254): aws identitystore get-user-id: An error "
                "occurred (ResourceNotFoundException) when calling the GetUserId "
                "operation: USER not found."
            )
        return json.dumps(
            {"IdentityStoreId": INSTANCE["IdentityStoreId"], "UserId": self.users[key]}
        )

    def lookups(self) -> list[tuple[str, str]]:
        """``(AttributePath, AttributeValue)`` of every get-user-id, in order."""

        found = []
        for argv, _keywords in self.calls:
            if argv[1] == "identitystore":
                attribute = json.loads(argv[argv.index("--alternate-identifier") + 1])
                attribute = attribute["UniqueAttribute"]
                found.append((attribute["AttributePath"], attribute["AttributeValue"]))
        return found

    def grants(self) -> list[dict[str, Any]]:
        """Every instruction sent through update-permissions, in order."""

        batches = []
        for argv, _keywords in self.calls:
            if argv[1] == "grafana" and argv[2] == "update-permissions":
                batches.extend(
                    json.loads(argv[argv.index("--update-instruction-batch") + 1])
                )
        return batches


def _ensure(
    runner: Any,
    tmp_path: Path,
    *,
    admin_email: str | None = ADMIN_EMAIL,
    settings: GrafanaSettings | None = None,
    probe_only: bool = False,
) -> dict[str, Any]:
    return ensure_grafana_dashboards(
        runner,
        settings=settings or GrafanaSettings(),
        cpu=_cluster(),
        site_id=SITE,
        amp_workspace_id=AMP,
        repository_root=tmp_path,
        probe_only=probe_only,
        http=Http(),
        admin_email=admin_email,
    )


def _admin_permission(role: str, user_id: str = USER_ID) -> dict[str, Any]:
    return {"role": role, "user": {"id": user_id, "type": "SSO_USER"}}


# --- (a) the derived grant -----------------------------------------------------------


def test_the_administrator_is_derived_from_the_email_and_granted_admin(
    tmp_path: Path,
) -> None:
    runner = IdentityRunner(users={("emails.value", ADMIN_EMAIL): USER_ID})
    _dashboards(tmp_path, "gpu-fault-overview")

    result = _ensure(runner, tmp_path)

    assert result["status"] == "PROVISIONED"
    assert result["admin_grant"] == {
        "email": ADMIN_EMAIL,
        "sso_user_id": USER_ID,
        "role": "ADMIN",
        "status": "granted",
    }
    operations = runner.operations()
    assert operations.index("list-instances") > operations.index(
        "delete-workspace-service-account-token"
    ), "the grant started before the import finished"
    assert operations.index("list-instances") < operations.index("get-user-id")
    assert operations.index("get-user-id") < operations.index("list-permissions")
    assert operations.index("list-permissions") < operations.index(
        "update-permissions"
    ), "the grant did not read the current permissions first"
    assert runner.lookups() == [("emails.value", ADMIN_EMAIL)]
    assert runner.grants() == [
        {
            "action": "ADD",
            "role": "ADMIN",
            "users": [{"id": USER_ID, "type": "SSO_USER"}],
        }
    ]
    grant = next(argv for argv, _k in runner.calls if argv[2] == "update-permissions")
    assert grant[grant.index("--workspace-id") + 1] == "g-5b81a13d97"
    store = next(argv for argv, _k in runner.calls if argv[2] == "get-user-id")
    assert store[store.index("--identity-store-id") + 1] == "d-1234567890"
    assert store[store.index("--region") + 1] == REGION


# --- (b) idempotent -------------------------------------------------------------------


def test_a_user_who_already_holds_admin_is_not_granted_again(tmp_path: Path) -> None:
    runner = IdentityRunner(
        users={("emails.value", ADMIN_EMAIL): USER_ID},
        permissions=[
            _admin_permission("VIEWER", "other-user"),
            _admin_permission("ADMIN"),
        ],
    )

    result = _ensure(runner, tmp_path)

    assert result["admin_grant"]["status"] == "already"
    assert result["admin_grant"]["sso_user_id"] == USER_ID
    assert "update-permissions" not in runner.operations(), (
        "the deploy re-granted a role the user already holds"
    )


def test_a_user_who_only_holds_viewer_is_raised_to_admin(tmp_path: Path) -> None:
    runner = IdentityRunner(
        users={("emails.value", ADMIN_EMAIL): USER_ID},
        permissions=[_admin_permission("VIEWER")],
    )

    result = _ensure(runner, tmp_path)

    assert result["admin_grant"]["status"] == "granted"
    assert [grant["role"] for grant in runner.grants()] == ["ADMIN"]


# --- (c) the userName fallback ------------------------------------------------------


def test_the_lookup_falls_back_to_the_user_name_when_no_email_matches(
    tmp_path: Path,
) -> None:
    runner = IdentityRunner(users={("userName", ADMIN_EMAIL): USER_ID})

    result = _ensure(runner, tmp_path)

    assert result["admin_grant"]["status"] == "granted"
    assert result["admin_grant"]["sso_user_id"] == USER_ID
    assert runner.lookups() == [
        ("emails.value", ADMIN_EMAIL),
        ("userName", ADMIN_EMAIL),
    ]


# --- (d) no such user -----------------------------------------------------------------


def test_no_matching_user_keeps_the_printed_command_and_names_the_missing_user(
    tmp_path: Path,
) -> None:
    runner = IdentityRunner(users={})

    result = _ensure(runner, tmp_path)

    assert result["status"] == "PROVISIONED", "a missing user failed the deploy step"
    grant = result["admin_grant"]
    assert grant["status"] == "not-derivable"
    assert grant["email"] == ADMIN_EMAIL
    assert grant["role"] == "ADMIN"
    assert "sso_user_id" not in grant or grant["sso_user_id"] is None
    assert f"no Identity Center user has email {ADMIN_EMAIL}" in grant["reason"]
    assert "IAM Identity Center" in grant["reason"]
    assert runner.lookups() == [
        ("emails.value", ADMIN_EMAIL),
        ("userName", ADMIN_EMAIL),
    ]
    assert "update-permissions" not in runner.operations()
    assert "list-permissions" not in runner.operations(), (
        "permissions were listed for a user that does not exist"
    )

    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id=SITE)
    state.record("monitoring_install", {"grafana": result})
    lines = grafana_access_lines(state)
    assert lines[0].startswith("Grafana dashboards: ")
    command_line = next(line for line in lines if "update-permissions" in line)
    assert "--workspace-id g-5b81a13d97" in command_line
    assert "<sso-user-id>" in command_line, "the ready command lost its placeholder"
    sentence = next(line for line in lines if "no Identity Center user" in line)
    assert ADMIN_EMAIL in sentence
    assert "re-run deploy" in sentence and "command above" in sentence
    assert lines.index(command_line) < lines.index(sentence), (
        "the sentence points at 'the command above' but is printed before it"
    )


# --- (e) ambiguous or absent Identity Center -------------------------------------------


def test_two_identity_center_instances_are_never_guessed_between(
    tmp_path: Path,
) -> None:
    runner = IdentityRunner(
        instances=[INSTANCE, SECOND_INSTANCE],
        users={("emails.value", ADMIN_EMAIL): USER_ID},
    )

    result = _ensure(runner, tmp_path)

    grant = result["admin_grant"]
    assert grant["status"] == "not-derivable"
    assert "2" in grant["reason"] and "Identity Center instance" in grant["reason"]
    assert "ssoins-1" in grant["reason"] and "ssoins-2" in grant["reason"]
    assert "get-user-id" not in runner.operations(), (
        "a user was looked up in a store the deploy could not choose"
    )
    assert "update-permissions" not in runner.operations()


def test_no_identity_center_instance_is_not_derivable(tmp_path: Path) -> None:
    runner = IdentityRunner(
        instances=[], users={("emails.value", ADMIN_EMAIL): USER_ID}
    )

    result = _ensure(runner, tmp_path)

    grant = result["admin_grant"]
    assert grant["status"] == "not-derivable"
    assert "no IAM Identity Center instance" in grant["reason"]
    assert REGION in grant["reason"], "the reason does not say which region was asked"
    assert "get-user-id" not in runner.operations()


def test_a_deploy_without_an_administrator_email_asks_nothing(tmp_path: Path) -> None:
    runner = IdentityRunner(users={("emails.value", ADMIN_EMAIL): USER_ID})

    result = _ensure(runner, tmp_path, admin_email=None)

    assert result["status"] == "PROVISIONED"
    assert result["admin_grant"]["status"] == "not-derivable"
    assert "administrator email" in result["admin_grant"]["reason"]
    assert "list-instances" not in runner.operations()


# --- (f) the explicit viewer still works, next to the admin grant -------------------


def test_an_explicit_viewer_is_granted_viewer_and_the_administrator_admin(
    tmp_path: Path,
) -> None:
    runner = IdentityRunner(users={("emails.value", ADMIN_EMAIL): USER_ID})

    result = _ensure(
        runner, tmp_path, settings=GrafanaSettings(viewer_sso_user_id="u-42")
    )

    assert result["viewer_sso_user_id"] == "u-42"
    assert result["admin_grant"]["status"] == "granted"
    assert runner.grants() == [
        {
            "action": "ADD",
            "role": "VIEWER",
            "users": [{"id": "u-42", "type": "SSO_USER"}],
        },
        {
            "action": "ADD",
            "role": "ADMIN",
            "users": [{"id": USER_ID, "type": "SSO_USER"}],
        },
    ]


def test_a_refused_explicit_viewer_still_fails_the_deploy(tmp_path: Path) -> None:
    runner = IdentityRunner(
        users={("emails.value", ADMIN_EMAIL): USER_ID},
        permission_errors=[{"code": 1, "message": "user u-typo not found"}],
    )

    with pytest.raises(BootstrapError, match="u-typo"):
        _ensure(runner, tmp_path, settings=GrafanaSettings(viewer_sso_user_id="u-typo"))


# --- (g) identitystore access denied -----------------------------------------------


def test_a_denied_identitystore_lookup_is_not_derivable_with_the_reason(
    tmp_path: Path,
) -> None:
    runner = IdentityRunner(
        users={("emails.value", ADMIN_EMAIL): USER_ID}, identitystore_denied=True
    )

    result = _ensure(runner, tmp_path)

    assert result["status"] == "PROVISIONED", "a denied lookup failed the deploy step"
    grant = result["admin_grant"]
    assert grant["status"] == "not-derivable"
    assert "AccessDeniedException" in grant["reason"]
    assert "identitystore:GetUserId" in grant["reason"]
    assert "update-permissions" not in runner.operations()


def test_a_refused_admin_grant_is_recorded_not_raised(tmp_path: Path) -> None:
    """The derived id is not operator input, so a refusal from the workspace
    degrades like every other miss instead of failing the deploy."""

    runner = IdentityRunner(
        users={("emails.value", ADMIN_EMAIL): USER_ID},
        permission_errors=[{"code": 1, "message": "workspace refused"}],
    )

    result = _ensure(runner, tmp_path)

    assert result["status"] == "PROVISIONED"
    assert result["admin_grant"]["status"] == "not-derivable"
    assert "workspace refused" in result["admin_grant"]["reason"]


# --- the printed lines --------------------------------------------------------------


def _state_with(tmp_path: Path, **grafana: Any) -> BootstrapState:
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id=SITE)
    state.record(
        "monitoring_install",
        {
            "grafana": {
                "status": "PROVISIONED",
                "workspace_id": "g-5b81a13d97",
                "region": REGION,
                "dashboards_url": "https://g-5b81a13d97.grafana-workspace/x",
                **grafana,
            }
        },
    )
    return state


def test_the_access_lines_say_who_was_granted_admin(tmp_path: Path) -> None:
    granted = grafana_access_lines(
        _state_with(
            tmp_path,
            admin_grant={
                "email": ADMIN_EMAIL,
                "sso_user_id": USER_ID,
                "role": "ADMIN",
                "status": "granted",
            },
        )
    )
    already = grafana_access_lines(
        _state_with(
            tmp_path,
            admin_grant={
                "email": ADMIN_EMAIL,
                "sso_user_id": USER_ID,
                "role": "ADMIN",
                "status": "already",
            },
        )
    )

    assert granted[1] == (
        f"Grafana ADMIN granted to {ADMIN_EMAIL} (Identity Center user {USER_ID})"
    )
    assert not any("update-permissions" in line for line in granted), (
        "the manual command was printed although the administrator was granted"
    )
    assert ADMIN_EMAIL in already[1] and USER_ID in already[1]
    assert "already" in already[1]
    assert not any("update-permissions" in line for line in already)


def test_the_access_lines_keep_the_command_when_the_grant_was_not_derivable(
    tmp_path: Path,
) -> None:
    lines = grafana_access_lines(
        _state_with(
            tmp_path,
            admin_grant={
                "email": ADMIN_EMAIL,
                "role": "ADMIN",
                "status": "not-derivable",
                "reason": f"no IAM Identity Center instance is visible from {REGION}",
            },
        )
    )

    assert any(
        "aws grafana update-permissions --region us-east-1 --workspace-id g-5b81a13d97"
        in line
        for line in lines
    )
    assert any("no IAM Identity Center instance" in line for line in lines), (
        "the reason the grant was skipped is not printed"
    )


# --- the read-only probe ------------------------------------------------------------


def _previous(**admin_grant: Any) -> dict[str, Any]:
    previous: dict[str, Any] = {"status": "PROVISIONED", "workspace_id": "g-5b81a13d97"}
    if admin_grant:
        previous["admin_grant"] = admin_grant
    return previous


def test_a_probe_asks_for_ensure_once_the_missing_user_exists(tmp_path: Path) -> None:
    """The user was created after the last deploy: the read-only derivation now
    finds them without ADMIN, and the write it needs re-runs the task."""

    tagged = {**HYPERPOD_WORKSPACE, "tags": {"gpu-fault:site-id": SITE}}
    runner = IdentityRunner([tagged], users={("emails.value", ADMIN_EMAIL): USER_ID})
    previous = _previous(
        email=ADMIN_EMAIL, role="ADMIN", status="not-derivable", reason="no user"
    )

    with pytest.raises(BootstrapMutationRequired):
        _ensure(
            ReadOnlyProbeRunner(runner),
            tmp_path,
            settings=GrafanaSettings(previous=previous),
            probe_only=True,
        )

    assert "update-permissions" not in runner.operations(), "the probe wrote"
    assert "create-workspace-service-account-token" not in runner.operations()


def test_a_probe_passes_while_the_user_is_still_missing_or_already_granted(
    tmp_path: Path,
) -> None:
    tagged = {**HYPERPOD_WORKSPACE, "tags": {"gpu-fault:site-id": SITE}}
    still_missing = IdentityRunner([tagged], users={})
    result = _ensure(
        ReadOnlyProbeRunner(still_missing),
        tmp_path,
        settings=GrafanaSettings(
            previous=_previous(
                email=ADMIN_EMAIL, role="ADMIN", status="not-derivable", reason="x"
            )
        ),
        probe_only=True,
    )
    assert result == {"status": "PROBED"}
    assert "get-user-id" in still_missing.operations(), (
        "the probe did not retry the derivation"
    )

    granted = IdentityRunner([tagged], users={("emails.value", ADMIN_EMAIL): USER_ID})
    result = _ensure(
        ReadOnlyProbeRunner(granted),
        tmp_path,
        settings=GrafanaSettings(
            previous=_previous(
                email=ADMIN_EMAIL, sso_user_id=USER_ID, role="ADMIN", status="granted"
            )
        ),
        probe_only=True,
    )
    assert result == {"status": "PROBED"}
    assert "list-instances" not in granted.operations(), (
        "a granted administrator was derived again on every deploy"
    )
