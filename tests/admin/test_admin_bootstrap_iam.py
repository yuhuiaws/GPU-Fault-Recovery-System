"""The IAM identities bootstrap creates, and how narrow they are.

These two roles are the entire AWS blast radius of the solution: the control
plane may read SageMaker and send one kind of e-mail, and one executor per GPU
cluster may reboot nodes in that cluster and nothing else. The documents are
asserted literally, and the ensure paths are driven with a fake account so that
reuse, drift repair and refusal to share another site's resources are all proved
without an AWS call.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import pytest

from gpu_fault.admin import bootstrap_services as services
from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    BootstrapError,
    ClusterIdentity,
)
from gpu_fault.admin.bootstrap_services import (
    control_plane_policy_document,
    ensure_control_plane_role,
    ensure_executor_role,
    executor_policy_document,
    pod_identity_trust,
)
from tests.admin._bootstrap_support import _cluster

SITE = "site-a"
REGION = "us-east-1"
ACCOUNT = "123456789012"
ISSUER = "https://oidc.eks.us-east-1.amazonaws.com/id/EXAMPLE"
CHAIN = (
    "-----BEGIN CERTIFICATE-----\nleaf\n-----END CERTIFICATE-----\n"
    "-----BEGIN CERTIFICATE-----\nroot\n-----END CERTIFICATE-----\n"
)
# ``CommandRunner.run`` strips captured output, so the fake returns it stripped too.
FINGERPRINT = "sha1 Fingerprint=AB:CD:EF:01"
FORBIDDEN_ACTIONS = (
    "sagemaker:BatchReplaceClusterNodes",
    "sagemaker:DeleteCluster",
    "sagemaker:UpdateCluster",
    "iam:PassRole",
    "eks:*",
)


def _role_name(argv: list[str]) -> str:
    return argv[argv.index("--role-name") + 1]


def _gpu() -> ClusterIdentity:
    return replace(_cluster(), role="gpu", hyperpod_name="hp-gpu-a", eks_name="gpu-a")


def _actions(document: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    for statement in document["Statement"]:
        value = statement["Action"]
        actions.extend([value] if isinstance(value, str) else value)
    return actions


class Account:
    """A fake AWS account and CPU cluster for the ensure paths.

    Every ``exists`` flag is a separate attribute because the interesting cases
    are the mixtures: a role that exists but is untagged, an add-on installed by
    someone else, an association pointing at a role from a previous attempt.

    Absence is expressed the way AWS expresses it: the read fails with the error
    the CLI prints. The ensure paths make one read per resource and decide from
    its outcome, so a fake that answered a "does it exist" question separately
    from the "what does it look like" question could no longer be wrong in the
    way the real account can.
    """

    def __init__(self) -> None:
        self.addon_installed = False
        self.addon_site: str | None = SITE
        self.service_account_exists = False
        self.role_exists = False
        self.role_read_error: str | None = None
        self.role_tags: list[dict[str, str]] | None = None
        self.role_trust: dict[str, Any] | None = None
        self.role_policy: dict[str, Any] | None = None
        self.policy_lookup: tuple[int, str] = (0, "")
        self.associations: list[dict[str, str]] = []
        self.association_role: str | None = None
        self.provider_exists = False
        self.provider_tags: list[dict[str, str]] = []
        self.issuer = ISSUER
        self.chain = CHAIN
        self.calls: list[list[str]] = []

    # -- CommandRunner -----------------------------------------------------
    def run(self, arguments: Sequence[Any], **_keywords: Any) -> str:
        argv = [str(item) for item in arguments]
        self.calls.append(argv)
        line = " ".join(argv)
        if argv[0] == "kubectl":
            if "get serviceaccount" in line and not self.service_account_exists:
                raise BootstrapError(
                    'Error from server (NotFound): serviceaccounts "x" not found'
                )
            return "apiVersion: v1\nkind: ServiceAccount\n"
        if "get-role-policy" in argv:
            returncode, stderr = self.policy_lookup
            if returncode:
                raise BootstrapError(f"command failed ({returncode}): aws: {stderr}")
            return json.dumps({"PolicyDocument": self.role_policy})
        if "iam get-role" in line:
            if self.role_read_error is not None:
                raise BootstrapError(
                    f"command failed (254): aws: {self.role_read_error}"
                )
            if not self.role_exists:
                raise BootstrapError(
                    "command failed (254): aws: An error occurred (NoSuchEntity) "
                    f"when calling the GetRole operation: Role {_role_name(argv)} "
                    "cannot be found."
                )
            return json.dumps(
                {
                    "Role": {
                        "Arn": f"arn:aws:iam::{ACCOUNT}:role/{argv[argv.index('--role-name') + 1]}",
                        "Tags": self.role_tags,
                        "AssumeRolePolicyDocument": self.role_trust,
                    }
                }
            )
        if argv[0] == "openssl" and "s_client" in argv:
            return self.chain
        if argv[0] == "openssl":
            return FINGERPRINT
        return ""

    def aws_json(self, _region: str, *arguments: str, **_keywords: Any) -> Any:
        self.calls.append(["aws", *arguments])
        operation = arguments[1]
        if operation == "describe-addon":
            if not self.addon_installed:
                raise BootstrapError(
                    "command failed (254): aws: An error occurred "
                    "(ResourceNotFoundException) when calling the DescribeAddon "
                    "operation: No addon: eks-pod-identity-agent found in cluster"
                )
            return {"addon": {"addonArn": f"arn:aws:eks:{REGION}:{ACCOUNT}:addon/a"}}
        if operation == "list-tags-for-resource":
            return {
                "tags": {}
                if self.addon_site is None
                else {SITE_TAG_KEY: self.addon_site}
            }
        if operation == "list-pod-identity-associations":
            return {"associations": list(self.associations)}
        if operation == "describe-pod-identity-association":
            return {"association": {"roleArn": self.association_role}}
        if operation == "create-pod-identity-association":
            return {"association": {"associationId": "assoc-new"}}
        if operation == "get-open-id-connect-provider":
            if not self.provider_exists:
                raise BootstrapError(
                    "command failed (254): aws: An error occurred (NoSuchEntity) "
                    "when calling the GetOpenIDConnectProvider operation: "
                    "OpenIDConnect Provider not found"
                )
            # The provider read carries its tags, so ownership needs no second call.
            return {"Url": self.issuer, "Tags": list(self.provider_tags)}
        raise AssertionError(f"unexpected aws call: {arguments}")

    def aws_text(self, _region: str, *arguments: str, **_keywords: Any) -> str:
        self.calls.append(["aws", *arguments])
        return self.issuer

    # -- process boundary --------------------------------------------------
    def process(self, arguments: Sequence[Any], **_keywords: Any) -> Any:
        """No existence probe bypasses the runner any more.

        IAM roles, inline policies, the Pod Identity add-on and the account-wide
        OIDC provider are all read through ``run``/``aws_json`` above, once each,
        so anything arriving here is a read that was meant to be deduplicated.
        """

        raise AssertionError(f"unexpected process: {[str(item) for item in arguments]}")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(services.subprocess, "run", self.process)

    def mutations(self, fragment: str) -> list[list[str]]:
        return [argv for argv in self.calls if fragment in " ".join(argv)]

    def control_plane(self, tmp_path: Path, **overrides: Any) -> dict[str, str]:
        return ensure_control_plane_role(
            self,
            cpu=_cluster(),
            cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
            namespace="gpu-fault-system",
            site_id=SITE,
            **overrides,
        )

    def executor(self) -> dict[str, str]:
        return ensure_executor_role(
            self, cluster=_gpu(), namespace="gpu-fault-system", site_id=SITE
        )


def test_the_control_plane_policy_grants_no_node_mutation() -> None:
    """Read-only on SageMaker is the CPU side's whole safety argument.

    Node reboot and replacement live in the per-cluster executor role so that a
    compromised control plane cannot touch training capacity, and there is no
    account-wide or wildcard-service grant to work around that.
    """

    document = control_plane_policy_document(region=REGION, account_id=ACCOUNT)

    assert _actions(document) == [
        "sagemaker:DescribeCluster",
        "sagemaker:ListClusterNodes",
        "sagemaker:DescribeClusterNode",
    ]
    assert document["Statement"][0]["Resource"] == (
        f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/*"
    )
    for action in FORBIDDEN_ACTIONS:
        assert action not in _actions(document), f"{action} reached the control plane"


def test_the_control_plane_policy_pins_email_to_the_verified_sender() -> None:
    """SES is granted for one identity and one From address.

    Without the condition the control-plane role could send as any verified
    identity in the account, which is the account's whole SES reputation.
    """

    document = control_plane_policy_document(
        region=REGION, account_id=ACCOUNT, email_sender="alerts@example.com"
    )
    email = document["Statement"][1]

    assert email["Action"] == "ses:SendEmail"
    assert email["Resource"] == (
        f"arn:aws:ses:{REGION}:{ACCOUNT}:identity/alerts@example.com"
    )
    assert email["Condition"] == {
        "StringEquals": {"ses:FromAddress": "alerts@example.com"}
    }


def test_the_pod_identity_trust_is_scoped_to_one_cluster() -> None:
    """The EKS Auth service can act as a confused deputy without a Condition.

    ``pods.eks.amazonaws.com`` fronts every cluster in the account; with no
    Condition any of them could be pointed at this role. Pinning the assuming
    cluster's ARN and account restricts the trust to exactly this cluster.
    """

    cluster = _cluster()
    trust = pod_identity_trust(cluster)
    statement = trust["Statement"][0]

    assert statement["Principal"] == {"Service": "pods.eks.amazonaws.com"}
    assert statement["Condition"] == {
        "StringEquals": {"aws:SourceAccount": cluster.account_id},
        "ArnEquals": {"aws:SourceArn": cluster.eks_arn},
    }
    # A different cluster in the same account produces a different trust, so the
    # scope really is per-cluster and not merely per-account.
    other = replace(cluster, eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/other")
    assert pod_identity_trust(other) != trust


def test_the_executor_policy_allows_reboot_only_on_its_own_cluster() -> None:
    """One executor cannot act on another cluster, and cannot send e-mail.

    Reboot is the only mutation the design permits; replacement is what the
    standing constraint on this repository forbids outright, and notification
    stays a control-plane concern so deduplication cannot be bypassed.
    """

    document = executor_policy_document(hyperpod_arn=_gpu().hyperpod_arn)
    statement = document["Statement"][0]

    assert statement["Resource"] == _gpu().hyperpod_arn
    assert "sagemaker:BatchRebootClusterNodes" in _actions(document)
    assert "sagemaker:BatchReplaceClusterNodes" not in _actions(document)
    assert not any(action.startswith("ses:") for action in _actions(document)), (
        "the executor role can send mail as the administrator identity"
    )
    assert "*" not in _actions(document)


def test_a_fresh_account_gets_the_addon_role_and_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control-plane identity is four resources that must all appear.

    Pod Identity needs the add-on, the ServiceAccount, the role and the
    association; any one of them missing leaves the control plane with no
    credentials and no way to say so other than an AccessDenied at runtime.
    """

    account = Account()
    account.install(monkeypatch)

    result = account.control_plane(tmp_path)

    assert account.mutations("create-addon"), (
        "the Pod Identity add-on was not installed"
    )
    assert account.mutations("wait addon-active"), (
        "bootstrap continued without waiting for the add-on to become active"
    )
    assert account.mutations("create serviceaccount"), (
        "the control-plane ServiceAccount was not created"
    )
    assert account.mutations("apply -f -"), (
        "the ServiceAccount annotations were never applied"
    )
    create = account.mutations("create-role")[0]
    assert json.loads(create[create.index("--assume-role-policy-document") + 1]) == (
        pod_identity_trust(_cluster())
    )
    assert f"Key={SITE_TAG_KEY},Value={SITE}" in create
    assert account.mutations("put-role-policy"), (
        "the role was created without its inline policy"
    )
    assert account.mutations("create-pod-identity-association"), (
        "the role was never bound to the ServiceAccount"
    )
    assert result["role_arn"] == (
        f"arn:aws:iam::{ACCOUNT}:role/gpu-fault-{SITE}-control"
    )
    assert result["association_id"] == "assoc-new"
    assert result["service_account"] == "gpu-fault-control-plane"


def test_an_existing_identity_that_already_matches_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun must not rewrite IAM or re-create the association.

    ``put-role-policy`` and ``update-assume-role-policy`` are audited mutations;
    issuing them on every deploy makes the audit trail useless for spotting a real
    change.
    """

    account = Account()
    account.addon_installed = True
    account.service_account_exists = True
    account.role_exists = True
    account.role_tags = [{"Key": SITE_TAG_KEY, "Value": SITE}]
    account.role_trust = pod_identity_trust(_cluster())
    account.role_policy = control_plane_policy_document(
        region=REGION, account_id=ACCOUNT
    )
    account.associations = [{"associationId": "assoc-a"}]
    account.association_role = f"arn:aws:iam::{ACCOUNT}:role/gpu-fault-{SITE}-control"
    account.install(monkeypatch)

    result = account.control_plane(tmp_path)

    assert result["association_id"] == "assoc-a"
    for fragment in (
        "create-addon",
        "create-role",
        "tag-role",
        "update-assume-role-policy",
        "put-role-policy",
        "create-pod-identity-association",
        "update-pod-identity-association",
        "create serviceaccount",
    ):
        assert not account.mutations(fragment), (
            f"a matching identity was rewritten: {fragment}"
        )


def test_drift_on_an_existing_identity_is_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trust, inline policy, site tag and binding are each repaired in place.

    A role left with an edited trust policy or a stale association is how a
    console change silently survives a redeploy; deleting and re-creating the role
    instead would break every association that references it.
    """

    account = Account()
    account.addon_installed = True
    account.service_account_exists = True
    account.role_exists = True
    account.role_tags = []
    account.role_trust = {"Version": "2012-10-17", "Statement": []}
    account.role_policy = {"Version": "2012-10-17", "Statement": []}
    account.associations = [{"associationId": "assoc-a"}]
    account.association_role = f"arn:aws:iam::{ACCOUNT}:role/somebody-elses-role"
    account.install(monkeypatch)

    account.control_plane(tmp_path, email_sender="alerts@example.com")

    assert account.mutations("tag-role"), (
        "the untagged role was not claimed for this site"
    )
    assert account.mutations("update-assume-role-policy"), (
        "the wrong trust policy was left in place"
    )
    policy = account.mutations("put-role-policy")[0]
    document = json.loads(policy[policy.index("--policy-document") + 1])
    assert "ses:SendEmail" in _actions(document)
    update = account.mutations("update-pod-identity-association")[0]
    assert "assoc-a" in update
    assert not account.mutations("create-role"), (
        "an existing role was recreated instead of repaired in place"
    )


def test_an_inline_policy_that_cannot_be_read_stops_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only NoSuchEntity means the policy is absent.

    On any other error the current policy is unknown, and writing over it could
    replace a narrower document with a wider one without anybody noticing.
    """

    account = Account()
    account.role_exists = True
    account.role_tags = [{"Key": SITE_TAG_KEY, "Value": SITE}]
    account.role_trust = pod_identity_trust(_cluster())
    account.policy_lookup = (
        254,
        "AccessDenied: not authorized to perform iam:GetRolePolicy",
    )
    account.install(monkeypatch)

    with pytest.raises(BootstrapError, match="cannot inspect inline policy"):
        account.control_plane(tmp_path)


def test_a_role_that_cannot_be_read_stops_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only NoSuchEntity means the role is absent.

    One read now answers both whether the role exists and whether it drifted, so
    the error handling around that read is the whole existence decision. Reading
    an AccessDenied as "absent" would send the run into ``create-role`` against a
    role that is already there, and the failure would name the wrong problem.
    """

    account = Account()
    account.addon_installed = True
    account.service_account_exists = True
    account.role_read_error = (
        "An error occurred (AccessDenied) when calling the GetRole operation"
    )
    account.install(monkeypatch)

    with pytest.raises(BootstrapError, match="AccessDenied"):
        account.control_plane(tmp_path)

    assert not account.mutations("create-role"), (
        "an unreadable role was recreated instead of stopping the bootstrap"
    )


def test_a_role_belonging_to_another_site_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two sites must never share one role.

    Adopting it would let this site's bootstrap rewrite the other site's trust
    policy, and uninstall would later delete a role the other site still needs.
    """

    account = Account()
    account.role_exists = True
    account.role_tags = [{"Key": SITE_TAG_KEY, "Value": "site-b"}]
    account.install(monkeypatch)

    with pytest.raises(BootstrapError, match="belongs to site 'site-b'"):
        account.control_plane(tmp_path)


def test_two_associations_for_one_service_account_stop_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ambiguous binding cannot be reconciled.

    Updating one of two associations leaves the other pointing at a different
    role, and which one the pod gets is not something bootstrap can decide.
    """

    account = Account()
    account.addon_installed = True
    account.service_account_exists = True
    account.role_exists = True
    account.role_tags = [{"Key": SITE_TAG_KEY, "Value": SITE}]
    account.role_trust = pod_identity_trust(_cluster())
    account.role_policy = control_plane_policy_document(
        region=REGION, account_id=ACCOUNT
    )
    account.associations = [{"associationId": "a"}, {"associationId": "b"}]
    account.install(monkeypatch)

    with pytest.raises(BootstrapError, match="multiple Pod Identity associations"):
        account.control_plane(tmp_path)


def test_an_addon_installed_by_someone_else_is_not_recreated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The add-on may predate this site, and is still waited for.

    ``create-addon`` on an existing add-on fails the bootstrap, and skipping the
    wait would let the first association be created before the agent can serve it.
    """

    account = Account()
    account.addon_installed = True
    account.addon_site = None
    account.install(monkeypatch)

    account.control_plane(tmp_path)

    assert not account.mutations("create-addon"), (
        "an add-on installed by someone else was recreated"
    )
    assert account.mutations("wait addon-active"), (
        "bootstrap did not wait for the borrowed add-on to become active"
    )


def test_the_executor_role_trusts_only_its_own_service_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trust policy is what stops any other pod from assuming the role.

    Both conditions matter: the audience keeps it to STS, and the subject keeps it
    to one namespace and one ServiceAccount in one cluster.
    """

    account = Account()
    account.install(monkeypatch)

    result = account.executor()

    create = account.mutations("create-role")[0]
    trust = json.loads(create[create.index("--assume-role-policy-document") + 1])
    issuer_host = ISSUER.removeprefix("https://")
    condition = trust["Statement"][0]["Condition"]["StringEquals"]
    assert trust["Statement"][0]["Principal"]["Federated"] == (
        f"arn:aws:iam::{ACCOUNT}:oidc-provider/{issuer_host}"
    )
    assert condition[f"{issuer_host}:aud"] == "sts.amazonaws.com"
    assert condition[f"{issuer_host}:sub"] == (
        "system:serviceaccount:gpu-fault-system:gpu-fault-cluster-executor"
    )
    assert result["inline_policy_name"] == "GPUFaultRegionalExecutor"
    assert result["cluster_name"] == "gpu-a"


def test_a_missing_oidc_provider_is_created_from_the_chain_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thumbprint has to come from the root of the served chain.

    Using the leaf certificate makes the provider stop working the next time the
    endpoint's certificate is rotated, which breaks every executor at once.
    """

    account = Account()
    account.install(monkeypatch)

    account.executor()

    create = account.mutations("create-open-id-connect-provider")[0]
    assert create[create.index("--thumbprint-list") + 1] == "abcdef01"
    assert create[create.index("--client-id-list") + 1] == "sts.amazonaws.com"
    fingerprint = account.mutations("x509")[0]
    assert fingerprint[fingerprint.index("-in") + 1].startswith("/"), (
        "the thumbprint was read from a relative path instead of the fetched chain"
    )


def test_an_existing_oidc_provider_is_reused_rather_than_recreated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider is account-wide and shared by every site in the cluster.

    Re-creating it is an error, and its ownership decides whether uninstall may
    delete it: only a provider this site created carries the site tag.
    """

    account = Account()
    account.provider_exists = True
    account.provider_tags = [{"Key": SITE_TAG_KEY, "Value": SITE}]
    account.install(monkeypatch)

    owned = account.executor()

    other = Account()
    other.provider_exists = True
    other.install(monkeypatch)

    external = other.executor()

    assert owned["oidc_provider_ownership"] == "CREATED"
    assert external["oidc_provider_ownership"] == "EXTERNAL"
    assert not account.mutations("create-open-id-connect-provider"), (
        "the site's own provider was recreated"
    )
    assert not other.mutations("create-open-id-connect-provider"), (
        "an externally owned provider was recreated"
    )


@pytest.mark.parametrize(
    ("issuer", "chain", "message"),
    [
        ("", CHAIN, "has no OIDC issuer"),
        ("not-a-url", CHAIN, "has no OIDC issuer"),
        (ISSUER, "no certificates here", "cannot read the EKS OIDC certificate chain"),
    ],
)
def test_an_unusable_oidc_endpoint_is_refused(
    monkeypatch: pytest.MonkeyPatch, issuer: str, chain: str, message: str
) -> None:
    """A guessed thumbprint or issuer would produce a role nothing can assume.

    Failing here names the cluster; failing later shows up as an AccessDenied in
    the executor with no indication that the provider is at fault.
    """

    account = Account()
    account.issuer = issuer
    account.chain = chain
    account.install(monkeypatch)

    with pytest.raises(BootstrapError, match=message):
        account.executor()
