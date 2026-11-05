"""The regional PKI and a first deploy's Aurora cluster, from bootstrap's side.

``_ensure_pki`` either adopts the site's Secrets Manager PKI (same hostname,
site tag added when it is missing on the secret or the ACM certificate) or,
on a fresh site, generates a private CA and server certificate with openssl,
imports it into ACM and stores the whole bundle in one secret written from a
0600 temporary file that never outlives the call. ``_ensure_aurora`` on a site
without a cluster creates the parameter group at the engine version the
product pins and the cluster with the shared ``create-db-cluster`` argument
list, and reports the master secret only when RDS returned one. An ingress
rule RDS refuses for any reason other than "already there" stops the task.
"""

from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import bootstrap as admin_bootstrap
from gpu_fault.admin.aurora_capacity import AURORA_ENGINE_VERSION
from gpu_fault.admin.bootstrap import (
    RDS_CA_BUNDLE_ENVIRONMENT,
    _ensure_aurora,
    _ensure_pki,
    _generate_pki,
)
from gpu_fault.admin.bootstrap_common import SITE_TAG_KEY, BootstrapError, CommandRunner
from gpu_fault.admin.config import AuroraCapacityConfig
from tests.admin._bootstrap_support import _cluster

SITE = "site-a"
HOSTNAME = f"api.{SITE}.gpu-fault.internal"
SECRET = f"gpu-fault/{SITE}/regional-pki"
IMPORTED_ARN = "arn:aws:acm:us-east-1:123456789012:certificate/imported"
EXISTING_ARN = "arn:aws:acm:us-east-1:123456789012:certificate/existing"
SITE_TAGS = [{"Key": SITE_TAG_KEY, "Value": SITE}]
OPENSSL_SEQUENCE = ["genrsa", "req", "genrsa", "req", "x509"]


class PkiAccount(CommandRunner):
    """Route53, ACM, Secrets Manager and openssl as the PKI task reaches them.

    The private zone already exists, tagged and associated with the CPU VPC.
    ``secret`` is the stored PKI bundle (``None``: never created); openssl
    writes a recognisable placeholder to each ``-out`` file unless
    ``write_openssl`` is off (a runner that records without executing).
    """

    def __init__(
        self,
        *,
        secret: dict[str, Any] | None = None,
        secret_tags: list[dict[str, str]] | None = None,
        certificate_tags: list[dict[str, str]] | None = None,
        write_openssl: bool = True,
    ) -> None:
        super().__init__()
        self.cpu = _cluster()
        self.secret = secret
        self.secret_tags = [] if secret_tags is None else secret_tags
        self.certificate_tags = [] if certificate_tags is None else certificate_tags
        self.write_openssl = write_openssl
        self.openssl: list[list[str]] = []
        self.tag_writes: list[list[str]] = []
        self.imported: dict[str, str] | None = None
        self.created: list[str] | None = None
        self.stored: dict[str, Any] | None = None
        self.stored_mode: int | None = None
        self.stored_file: Path | None = None

    def aws_json(self, region, service, operation, *arguments, **_options):
        assert region == self.cpu.region, "the PKI left the CPU Region"
        if (service, operation) == ("route53", "list-hosted-zones-by-name"):
            return {
                "HostedZones": [
                    {
                        "Id": "/hostedzone/ZSITE",
                        "Name": f"{SITE}.gpu-fault.internal.",
                        "Config": {"PrivateZone": True},
                    }
                ]
            }
        if (service, operation) == ("route53", "list-tags-for-resource"):
            return {"ResourceTagSet": {"Tags": SITE_TAGS}}
        if (service, operation) == ("route53", "get-hosted-zone"):
            return {"VPCs": [{"VPCRegion": region, "VPCId": self.cpu.vpc_id}]}
        if (service, operation) == ("secretsmanager", "describe-secret"):
            assert arguments == ("--secret-id", SECRET), arguments
            if self.secret is None:
                raise BootstrapError(
                    "An error occurred (ResourceNotFoundException) when calling "
                    "the DescribeSecret operation"
                )
            return {"Name": SECRET, "Tags": self.secret_tags}
        if (service, operation) == ("acm", "list-tags-for-certificate"):
            assert arguments == ("--certificate-arn", EXISTING_ARN), arguments
            return {"Tags": self.certificate_tags}
        pytest.fail(f"unexpected AWS read: {service} {operation}")

    def aws_text(self, region, service, operation, *arguments, **options):
        if (service, operation) == ("secretsmanager", "get-secret-value"):
            assert options.get("sensitive"), "the PKI secret was read in the clear"
            return json.dumps(self.secret)
        if (service, operation) == ("acm", "import-certificate"):
            assert options.get("mutate") and options.get("sensitive"), options
            self.imported = dict(zip(arguments[::2], arguments[1::2], strict=True))
            return IMPORTED_ARN
        pytest.fail(f"unexpected AWS text call: {service} {operation}")

    def run(self, arguments, *, mutate=False, sensitive=False, **_options) -> str:
        argv = [str(item) for item in arguments]
        assert mutate, f"read-only command routed through run: {argv[:3]}"
        if argv[0] == "openssl":
            assert sensitive, "openssl key material was not marked sensitive"
            self.openssl.append(argv)
            if self.write_openssl:
                output = Path(argv[argv.index("-out") + 1])
                output.write_text(f"{argv[1]} wrote {output.name}\n", encoding="utf-8")
            return ""
        if argv[:3] == ["aws", "secretsmanager", "create-secret"]:
            source = argv[argv.index("--secret-string") + 1]
            self.stored_file = Path(source.removeprefix("file://"))
            self.stored_mode = stat.S_IMODE(self.stored_file.stat().st_mode)
            self.stored = json.loads(self.stored_file.read_text(encoding="utf-8"))
            self.created = argv
            return ""
        if argv[:3] in (
            ["aws", "secretsmanager", "tag-resource"],
            ["aws", "acm", "add-tags-to-certificate"],
        ):
            self.tag_writes.append(argv)
            return ""
        pytest.fail(f"unexpected command: {argv[:3]}")


def ensure(account: PkiAccount, state_dir: Path) -> dict[str, Any]:
    return _ensure_pki(
        account, cpu=account.cpu, gpu_clusters=[], state_dir=state_dir, site_id=SITE
    )


def existing_secret(**overrides: Any) -> dict[str, Any]:
    return {
        "hostname": HOSTNAME,
        "certificate_arn": EXISTING_ARN,
        "ca_certificate": "-----BEGIN CERTIFICATE-----\nexisting\n",
        **overrides,
    }


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# ---------------------------------------------------------------- fresh PKI


def test_a_fresh_site_generates_imports_and_stores_its_private_pki(
    tmp_path: Path,
) -> None:
    account = PkiAccount()

    result = ensure(account, tmp_path)

    pki = tmp_path / "pki"
    assert [argv[1] for argv in account.openssl] == OPENSSL_SEQUENCE
    ca_request, server_signing = account.openssl[1], account.openssl[4]
    assert ca_request[ca_request.index("-days") + 1] == "3650"
    assert server_signing[server_signing.index("-CA") + 1] == str(pki / "ca.crt")
    assert server_signing[server_signing.index("-CAkey") + 1] == str(pki / "ca.key")
    assert server_signing[server_signing.index("-days") + 1] == "825"
    assert f"subjectAltName=DNS:{HOSTNAME}\n" in (pki / "server.ext").read_text(), (
        "the server certificate must name the private API hostname"
    )
    assert mode(pki) == 0o700
    assert mode(pki / "ca.key") == mode(pki / "server.key") == 0o600
    assert account.imported == {
        "--certificate": f"fileb://{pki / 'server.crt'}",
        "--private-key": f"fileb://{pki / 'server.key'}",
        "--certificate-chain": f"fileb://{pki / 'ca.crt'}",
        "--tags": f"Key=gpu-fault:site-id,Value={SITE}",
        "--query": "CertificateArn",
    }
    ca_file = tmp_path / "control-plane-ca.crt"
    assert ca_file.read_text() == (pki / "ca.crt").read_text()
    assert mode(ca_file) == 0o644
    assert account.stored == {
        "ca_certificate": (pki / "ca.crt").read_text(),
        "ca_private_key": (pki / "ca.key").read_text(),
        "server_certificate": (pki / "server.crt").read_text(),
        "server_private_key": (pki / "server.key").read_text(),
        "hostname": HOSTNAME,
        "certificate_arn": IMPORTED_ARN,
    }
    assert account.stored_mode == 0o600
    assert account.stored_file is not None and not account.stored_file.exists(), (
        "the temporary secret file outlived the create-secret call"
    )
    assert account.created is not None
    assert account.created[account.created.index("--name") + 1] == SECRET
    assert result["hostname"] == HOSTNAME
    assert {
        key: result[key]
        for key in (
            "certificate_arn",
            "certificate_ownership",
            "ca_file",
            "pki_secret_id",
            "pki_secret_ownership",
        )
    } == {
        "certificate_arn": IMPORTED_ARN,
        "certificate_ownership": "CREATED",
        "ca_file": str(ca_file),
        "pki_secret_id": SECRET,
        "pki_secret_ownership": "CREATED",
    }
    assert account.tag_writes == [], "a created secret is tagged at creation"


def test_key_material_left_by_an_interrupted_run_is_imported_not_regenerated(
    tmp_path: Path,
) -> None:
    pki = tmp_path / "pki"
    pki.mkdir()
    for name in ("ca.key", "ca.crt", "server.key", "server.crt"):
        (pki / name).write_text(f"kept {name}\n", encoding="utf-8")
    account = PkiAccount()

    result = ensure(account, tmp_path)

    assert account.openssl == [], "existing key material was regenerated"
    assert account.stored is not None
    assert account.stored["ca_private_key"] == "kept ca.key\n"
    assert account.stored["server_certificate"] == "kept server.crt\n"
    assert (tmp_path / "control-plane-ca.crt").read_text() == "kept ca.crt\n"
    assert result["certificate_arn"] == IMPORTED_ARN


def test_a_runner_that_records_without_executing_leaves_no_key_to_tighten(
    tmp_path: Path,
) -> None:
    account = PkiAccount(write_openssl=False)
    pki = tmp_path / "pki"

    paths = _generate_pki(account, directory=pki, hostname="api.example.internal")

    assert paths == (
        pki / "ca.key",
        pki / "ca.crt",
        pki / "server.key",
        pki / "server.crt",
    )
    assert not any(path.exists() for path in paths), (
        "the recording runner wrote no key or certificate"
    )
    assert [argv[1] for argv in account.openssl] == OPENSSL_SEQUENCE
    assert (pki / "server.ext").read_text() == (
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
        "subjectAltName=DNS:api.example.internal\n"
    )


# ----------------------------------------------------------- adopted PKI


def test_an_untagged_site_pki_is_adopted_and_tagged_for_the_site(
    tmp_path: Path,
) -> None:
    account = PkiAccount(secret=existing_secret())

    result = ensure(account, tmp_path)

    assert account.tag_writes == [
        [
            "aws",
            "secretsmanager",
            "tag-resource",
            "--region",
            account.cpu.region,
            "--secret-id",
            SECRET,
            "--tags",
            f"Key={SITE_TAG_KEY},Value={SITE}",
        ],
        [
            "aws",
            "acm",
            "add-tags-to-certificate",
            "--region",
            account.cpu.region,
            "--certificate-arn",
            EXISTING_ARN,
            "--tags",
            f"Key={SITE_TAG_KEY},Value={SITE}",
        ],
    ]
    ca_file = tmp_path / "control-plane-ca.crt"
    assert ca_file.read_text() == existing_secret()["ca_certificate"]
    assert mode(ca_file) == 0o644
    assert result["certificate_arn"] == EXISTING_ARN
    assert result["pki_secret_id"] == SECRET
    assert account.openssl == [] and account.imported is None, (
        "an adopted PKI was generated or imported again"
    )
    assert account.created is None, "the existing secret was created a second time"


def test_a_site_pki_already_tagged_is_adopted_without_any_write(tmp_path: Path) -> None:
    account = PkiAccount(
        secret=existing_secret(), secret_tags=SITE_TAGS, certificate_tags=SITE_TAGS
    )

    result = ensure(account, tmp_path)

    assert account.tag_writes == []
    assert result["certificate_ownership"] == "CREATED"
    assert result["ca_file"] == str(tmp_path / "control-plane-ca.crt")


def test_a_site_pki_for_another_hostname_is_refused_before_any_write(
    tmp_path: Path,
) -> None:
    account = PkiAccount(
        secret=existing_secret(hostname="api.other.gpu-fault.internal")
    )

    with pytest.raises(BootstrapError, match="belongs to another hostname"):
        ensure(account, tmp_path)

    assert account.tag_writes == [] and account.openssl == []
    assert not (tmp_path / "control-plane-ca.crt").exists(), (
        "a foreign CA was handed to the control plane"
    )


# ------------------------------------------------------------ first Aurora


class NewAuroraAccount(CommandRunner):
    """RDS and EC2 of a site without any Aurora resource yet."""

    def __init__(self, *, created: dict[str, Any]) -> None:
        super().__init__()
        self.created = created
        self.reads: list[str] = []
        self.writes: list[list[str]] = []
        self.authorized: list[list[str]] = []
        self.engine_versions: list[str] = []

    def run(self, arguments, *, mutate=False, **_options) -> str:
        argv = [str(item) for item in arguments]
        if mutate:
            self.writes.append(argv)
            return ""
        assert argv[:2] == ["kubectl", "--kubeconfig"], argv[:3]
        node = {
            "metadata": {"labels": {"topology.kubernetes.io/zone": "us-east-1b"}},
            "status": {"addresses": [{"type": "InternalIP", "address": "10.0.1.5"}]},
        }
        return json.dumps({"items": [node]})

    def aws_text(self, region, *arguments, mutate=False, **_options) -> str:
        assert arguments[:2] == ("ec2", "create-security-group") and mutate, arguments
        self.writes.append(["aws", *arguments])
        return "sg-aurora-new"

    def aws_json(self, region, *arguments, mutate=False, **_options) -> dict:
        operation = arguments[1]
        self.reads.append(operation)
        if operation == "create-db-cluster":
            assert mutate, "the cluster was created outside the mutation guard"
            self.writes.append(["aws", *arguments])
            return self.created
        if operation in {"describe-db-subnet-groups", "describe-db-clusters"}:
            fault = (
                "DBSubnetGroupNotFoundFault"
                if operation == "describe-db-subnet-groups"
                else "DBClusterNotFoundFault"
            )
            raise BootstrapError(f"An error occurred ({fault}) when calling it")
        answers: dict[str, dict[str, Any]] = {
            "describe-subnets": {
                "Subnets": [
                    {"SubnetId": "subnet-private-a", "AvailabilityZone": "us-east-1a"},
                    {"SubnetId": "subnet-private-b", "AvailabilityZone": "us-east-1b"},
                ]
            },
            "describe-route-tables": {
                "RouteTables": [
                    {
                        "RouteTableId": "rtb-main",
                        "Associations": [{"Main": True}],
                        "Routes": [
                            {"DestinationCidrBlock": "0.0.0.0/0", "NatGatewayId": "n"}
                        ],
                    }
                ]
            },
            "describe-security-groups": {"SecurityGroups": []},
            "describe-network-interfaces": {
                "NetworkInterfaces": [{"Groups": [{"GroupId": "sg-cpu-nodes"}]}]
            },
        }
        assert operation in answers, f"unexpected AWS read: {operation}"
        return answers[operation]


def first_aurora(
    account: NewAuroraAccount,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    ingress: subprocess.CompletedProcess[str] | None = None,
) -> dict[str, Any]:
    monkeypatch.setenv(RDS_CA_BUNDLE_ENVIRONMENT, "/etc/rds/ca.pem")

    def authorize(arguments, **_options):
        account.authorized.append(list(arguments))
        return ingress or subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(admin_bootstrap.subprocess, "run", authorize)

    def parameter_group(_runner, *, engine_version, **_kwargs):
        account.engine_versions.append(engine_version)
        return "gpu-fault-site-a-aurora-params"

    monkeypatch.setattr(
        admin_bootstrap, "ensure_cluster_parameter_group", parameter_group
    )

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("an existing-cluster reconciler ran for a new cluster")

    for name in ("reconcile_cluster_diagnostics", "reconcile_existing_capacity"):
        monkeypatch.setattr(admin_bootstrap, name, must_not_run)
    monkeypatch.setattr(
        admin_bootstrap,
        "ensure_serverless_instances",
        lambda _runner, **_kwargs: ["writer-1", "reader-1"],
    )
    return _ensure_aurora(
        account,
        cpu=_cluster(),
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id=SITE,
        capacity=AuroraCapacityConfig(min_acu=0.5, max_acu=8.0),
    )


def test_a_first_deploy_creates_the_cluster_and_keeps_its_master_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    account = NewAuroraAccount(
        created={
            "DBCluster": {
                "MasterUserSecret": {
                    "SecretArn": "arn:aws:secretsmanager:us-east-1:1:secret:new",
                    "KmsKeyId": "kms-new",
                }
            }
        }
    )

    result = first_aurora(account, monkeypatch, tmp_path)

    created = next(argv for argv in account.writes if "create-db-cluster" in argv)
    assert created[created.index("--db-cluster-identifier") + 1] == (
        "gpu-fault-site-a-aurora"
    )
    assert created[created.index("--db-subnet-group-name") + 1] == (
        "gpu-fault-site-a-aurora"
    )
    assert created[created.index("--vpc-security-group-ids") + 1] == "sg-aurora-new"
    assert created[created.index("--db-cluster-parameter-group-name") + 1] == (
        "gpu-fault-site-a-aurora-params"
    )
    assert account.engine_versions == [AURORA_ENGINE_VERSION], (
        "a new cluster's parameter group must follow the pinned engine version"
    )
    assert [argv[2] for argv in account.writes if argv[0] == "aws"][:3] == [
        "create-db-subnet-group",
        "create-security-group",
        "create-tags",
    ]
    assert account.authorized[0][-2:] == ["--source-group", "sg-cpu-nodes"]
    assert (
        result["master_secret_arn"] == "arn:aws:secretsmanager:us-east-1:1:secret:new"
    )
    assert result["master_secret_kms_key_arn"] == "kms-new"
    assert result["instance_ids"] == ["writer-1", "reader-1"]
    assert result["availability_zones"] == ["us-east-1b", "us-east-1a"]
    assert result["security_group"] == "sg-aurora-new"
    assert result["parameter_group"] == "gpu-fault-site-a-aurora-params"


def test_a_created_cluster_without_a_master_secret_reports_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    account = NewAuroraAccount(created={"DBCluster": None})

    result = first_aurora(account, monkeypatch, tmp_path)

    assert "master_secret_arn" not in result
    assert "master_secret_kms_key_arn" not in result
    assert result["cluster_ownership"] == "CREATED"


@pytest.mark.parametrize(
    ("stderr", "refused"),
    [
        ("An error occurred (InvalidPermission.Duplicate): rule exists", False),
        ("An error occurred (UnauthorizedOperation): not allowed", True),
    ],
    ids=["duplicate-rule-is-fine", "other-error-stops"],
)
def test_only_an_existing_ingress_rule_is_an_acceptable_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stderr: str, refused: bool
) -> None:
    account = NewAuroraAccount(created={"DBCluster": {}})
    ingress = subprocess.CompletedProcess([], 254, "", f"  {stderr}\n")

    if refused:
        with pytest.raises(BootstrapError) as raised:
            first_aurora(account, monkeypatch, tmp_path, ingress=ingress)
        assert str(raised.value) == stderr
        assert "describe-db-clusters" not in account.reads, (
            "the cluster was looked up after the ingress rule failed"
        )
    else:
        result = first_aurora(account, monkeypatch, tmp_path, ingress=ingress)
        assert result["cluster_id"] == "gpu-fault-site-a-aurora"
