"""Discovery of a deployment that predates the managed site file.

``gpu-fault-admin uninstall`` is the only caller: given the cluster ARNs it
reconstructs a site document and a bootstrap state from what is actually running,
so that uninstall deletes what this solution created and leaves everything else
alone. Every ownership label it writes here decides whether a live AWS resource
is later deleted, which is why the failure paths are asserted as carefully as the
happy one.
"""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import pytest
import yaml

from gpu_fault.admin import legacy_site
from gpu_fault.admin.bootstrap_common import BootstrapError, ClusterIdentity
from tests.admin._bootstrap_support import _cluster

REGION = "us-east-1"
ACCOUNT = "123456789012"
CERTIFICATE_ARN = f"arn:aws:acm:{REGION}:{ACCOUNT}:certificate/legacy"
MASTER_SECRET_ARN = f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:aurora-master"
CONTROL_ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/gpu-fault-control-plane"
EXECUTOR_ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/gpu-fault-executor"
COLLECTOR = (
    "exporters:\n"
    "  prometheusremotewrite:\n"
    f"    endpoint: https://aps-workspaces.{REGION}.amazonaws.com"
    "/workspaces/ws-legacy-0001/api/v1/remote_write\n"
)


def _b64(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


def _repository_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    rollout = root / "deploy/control-plane/regional/rollout-regional-release.sh"
    rollout.parent.mkdir(parents=True)
    rollout.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    rollout.chmod(0o755)
    (root / "dist").mkdir()
    (root / "dist/current-release.json").write_text("{}", encoding="utf-8")
    (root / "config").mkdir()
    (root / "config/runtime-profile.regional-hyperpod-safe.example.yaml").write_text(
        "cluster_id: placeholder\n"
        "environment: hyperpod-eks\n"
        "profile_version: hyperpod-v1\n"
        "claims: []\n"
        "observed: []\n",
        encoding="utf-8",
    )
    return root


class Legacy:
    """A live legacy control plane, as ``discover_legacy_site`` sees it.

    Everything is an attribute so a test can change exactly one fact -- an empty
    Pod Identity association, a CloudFormation-tagged workspace, a missing Secret
    key -- and assert what the discovered state says about it.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.repository_root = _repository_root(tmp_path)
        self.state_dir = tmp_path / "state"
        self.release_state: dict[str, Any] = {
            "runtime_profile_version": "hyperpod-v1",
            "runtime_profile_registration_cluster_id": "gpu-a",
        }
        self.release_metadata = {"required-agent-config-digest": "a" * 64}
        self.allowed_namespaces = json.dumps(["training", "gpu-fault-system"])
        # When set, overrides the ``cluster-id`` the connection Secret decodes to,
        # so a test can feed discovery a malformed or path-escaping identity.
        self.connection_cluster_id: str | None = None
        self.connection_keys = (
            "cluster-id",
            "cluster-token",
            "ca.crt",
            "control-plane-url",
            "hyperpod-cluster-name",
            "allowed-namespaces",
        )
        self.nlb_annotations = {
            "service.beta.kubernetes.io/aws-load-balancer-name": "gpu-fault-legacy",
            "service.beta.kubernetes.io/aws-load-balancer-subnets": (
                "subnet-public-a,subnet-public-b"
            ),
            "service.beta.kubernetes.io/aws-load-balancer-security-groups": "sg-nlb",
            "service.beta.kubernetes.io/aws-load-balancer-ssl-cert": CERTIFICATE_ARN,
        }
        self.executor_annotations = {"eks.amazonaws.com/role-arn": EXECUTOR_ROLE_ARN}
        self.collector = COLLECTOR
        self.workspace_tags: dict[str, str] = {"Name": "gpu-fault"}
        self.db_clusters: list[dict[str, Any]] = [
            {
                "DBClusterIdentifier": "gpu-fault-aurora",
                "MasterUserSecret": {
                    "SecretArn": MASTER_SECRET_ARN,
                    "KmsKeyId": f"arn:aws:kms:{REGION}:{ACCOUNT}:key/legacy",
                },
                "DBClusterMembers": [
                    {"DBInstanceIdentifier": "gpu-fault-aurora-writer"},
                    {"DBInstanceIdentifier": "gpu-fault-aurora-reader"},
                ],
                "DBSubnetGroup": "gpu-fault-aurora-subnets",
                "VpcSecurityGroups": [{"VpcSecurityGroupId": "sg-aurora"}],
                "DBClusterParameterGroup": "gpu-fault-aurora-pg",
            }
        ]
        self.pki_secret_names = ["gpu-fault-nlb-private-pki-legacy"]
        self.pki_certificate_arn = CERTIFICATE_ARN
        self.associations = {
            "gpu-fault-control-plane": "assoc-control",
            "gpu-fault-aurora-credential-refresh": "assoc-refresh",
            "gpu-fault-adot": "assoc-adot",
        }
        self.addon_installed = True
        self.kubectl_failure: str | None = None

    # -- kubectl -----------------------------------------------------------
    def _document(self, command: Sequence[str]) -> dict[str, Any]:
        name = command[-3]
        if name == "gpu-fault-regional-release-state":
            return {"data": {"state.json": json.dumps(self.release_state)}}
        if name == "gpu-fault-release-metadata":
            return {"data": dict(self.release_metadata)}
        if name == "gpu-fault-control-plane-active":
            return {"data": {"node-action-secret": _b64("f" * 64)}}
        if name == "gpu-fault-regional-connection":
            context = command[command.index("--context") + 1]
            values = {
                "cluster-id": (
                    self.connection_cluster_id
                    if self.connection_cluster_id is not None
                    else context
                ),
                "cluster-token": "t" * 64,
                "ca.crt": "certificate",
                "control-plane-url": "https://control.internal",
                "hyperpod-cluster-name": f"hp-{context}",
                "allowed-namespaces": self.allowed_namespaces,
            }
            return {
                "data": {
                    key: _b64(values[key])
                    for key in self.connection_keys
                    if key in values
                }
            }
        if name == "gpu-fault-api-nlb":
            return {"metadata": {"annotations": dict(self.nlb_annotations)}}
        if name == "gpu-fault-aurora":
            return {"data": {"master-secret-arn": _b64(MASTER_SECRET_ARN)}}
        if name == "gpu-fault-adot":
            return {"data": {"collector.yaml": self.collector}}
        if name == "gpu-fault-cluster-executor":
            return {"metadata": {"annotations": dict(self.executor_annotations)}}
        raise AssertionError(f"unexpected kubectl request: {list(command)}")

    def process(self, command: Sequence[str], **_keywords: Any) -> Any:
        class Result:
            def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        if command[0] == "aws":
            return Result(0 if self.addon_installed else 254, "", "")
        if self.kubectl_failure is not None:
            return Result(1, "", self.kubectl_failure)
        return Result(0, json.dumps(self._document(command)), "")

    # -- aws ---------------------------------------------------------------
    def aws_json(self, _region: str, *arguments: str, **_keywords: Any) -> Any:
        operation = arguments[1]
        if operation == "describe-db-clusters":
            return {"DBClusters": list(self.db_clusters)}
        if operation == "describe-workspace":
            return {"workspace": {"tags": dict(self.workspace_tags)}}
        if operation == "get-topic-attributes":
            return {}
        if operation == "list-pod-identity-associations":
            service_account = arguments[arguments.index("--service-account") + 1]
            association_id = self.associations.get(service_account)
            if association_id is None:
                return {"associations": []}
            return {"associations": [{"associationId": association_id}]}
        if operation == "describe-pod-identity-association":
            return {"association": {"roleArn": CONTROL_ROLE_ARN}}
        if operation == "describe-security-groups":
            return {"SecurityGroups": [{"VpcId": "vpc-legacy"}]}
        if operation == "list-secrets":
            return {"SecretList": [{"Name": name} for name in self.pki_secret_names]}
        raise AssertionError(f"unexpected aws call: {arguments}")

    def aws_text(self, _region: str, *arguments: str, **_keywords: Any) -> str:
        operation = arguments[1]
        if operation == "describe-cluster":
            return f"https://oidc.eks.{REGION}.amazonaws.com/id/LEGACY"
        if operation == "get-secret-value":
            return json.dumps({"certificate_arn": self.pki_certificate_arn})
        raise AssertionError(f"unexpected aws call: {arguments}")

    # -- driver ------------------------------------------------------------
    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def discover(_runner: Any, *, cluster_arn: str, role: str, context: str):
            # The kubeconfig context is the cluster's own name here rather than the
            # generated alias, so a fixture can address one GPU cluster by name.
            name = cluster_arn.rsplit("/", 1)[-1]
            assert context, "discovery must name the kubeconfig context it reads"
            return replace(
                _cluster(),
                input_arn=cluster_arn,
                role=role,
                region=REGION,
                account_id=ACCOUNT,
                hyperpod_arn=f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/{name}",
                hyperpod_name=name,
                eks_arn=cluster_arn,
                eks_name=name,
                context=name,
            )

        def kubeconfigs(
            _runner: Any,
            *,
            cpu: ClusterIdentity,
            gpu_clusters: Sequence[ClusterIdentity],
            state_dir: Path,
        ) -> tuple[Path, Path]:
            for name in ("cpu.kubeconfig", "gpu.kubeconfig"):
                path = state_dir / name
                path.write_text("kubeconfig", encoding="utf-8")
                path.chmod(0o600)
            return state_dir / "cpu.kubeconfig", state_dir / "gpu.kubeconfig"

        monkeypatch.setattr(legacy_site, "discover_cluster", discover)
        monkeypatch.setattr(legacy_site, "_ensure_kubeconfigs", kubeconfigs)
        monkeypatch.setattr(
            legacy_site, "_require_same_scope", lambda *_args, **_keywords: None
        )
        monkeypatch.setattr(legacy_site.subprocess, "run", self.process)

    def discover(self, monkeypatch: pytest.MonkeyPatch, *, gpu_clusters: int = 1):
        self.install(monkeypatch)
        return legacy_site.discover_legacy_site(
            legacy_site.LegacySiteRequest(
                cpu_cluster_arn=f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/control",
                gpu_cluster_arns=tuple(
                    f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-{index}"
                    for index in range(1, gpu_clusters + 1)
                ),
                repository_root=self.repository_root,
                state_dir=self.state_dir,
            ),
            runner=self,
        )

    def bootstrap_state(self) -> dict[str, Any]:
        return json.loads(
            (self.state_dir / "bootstrap-state.json").read_text(encoding="utf-8")
        )

    def site_document(self) -> dict[str, Any]:
        return yaml.safe_load(
            (self.state_dir / "site.yaml").read_text(encoding="utf-8")
        )


def test_discovery_renders_a_site_that_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The discovered document has to survive ``load_site`` unchanged.

    Uninstall consumes the ``RenderedSite``, not the dict written here, so a
    document that renders but does not load turns "uninstall a legacy site" into
    an error after the state directory has already been written.
    """

    legacy = Legacy(tmp_path)

    site = legacy.discover(monkeypatch, gpu_clusters=2)

    assert site.release_config["aws_region"] == REGION
    assert [item["cluster_id"] for item in site.release_config["clusters"]] == [
        "gpu-1",
        "gpu-2",
    ]
    assert site.release_config["nlb"]["public_subnets"] == (
        "subnet-public-a,subnet-public-b"
    )
    assert site.release_config["health"]["aurora_cluster_id"] == "gpu-fault-aurora"
    assert site.release_config["health"]["amp_workspace_id"] == "ws-legacy-0001"
    assert site.release_config["health"]["sns_topic_arn"] == (
        f"arn:aws:sns:{REGION}:{ACCOUNT}:gpu-fault-control-plane-alerts-{REGION}"
    )
    assert legacy.site_document()["spec"]["notifications"] == {
        "allowEmail": False,
        "acknowledgeExternalAlertChannel": True,
    }


def test_discovery_writes_the_live_secrets_with_private_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The token, CA and fleet master key are written to disk from live Secrets.

    They are credentials for the running fleet, so a state directory readable by
    anyone else on the admin host would leak them; the site file that references
    them is checked for the same thing by ``load_site``.
    """

    legacy = Legacy(tmp_path)

    site = legacy.discover(monkeypatch)

    cluster = site.release_config["clusters"][0]
    written = [
        Path(cluster["token_file"]),
        Path(cluster["ca_file"]),
        Path(cluster["fleet_master_file"]),
    ]
    for path in written:
        assert path.stat().st_mode & 0o077 == 0, f"{path} is not private"
    assert Path(cluster["token_file"]).read_text(encoding="utf-8") == "t" * 64
    assert legacy.state_dir.stat().st_mode & 0o777 == 0o700
    assert (legacy.state_dir / "bootstrap-state.json").stat().st_mode & 0o077 == 0
    assert not any(
        path.with_suffix(path.suffix + ".tmp").exists() for path in written
    ), "a temporary copy of a secret was left next to the secret itself"


def test_discovered_bootstrap_state_marks_what_uninstall_may_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ownership labels are the whole point of the discovered state.

    ``CREATED`` means uninstall deletes the resource; ``EXTERNAL`` means it was
    there first and must survive. The OIDC provider and the Pod Identity add-on
    are the two that are always external, and the load balancer controller is
    recorded as external so uninstall never removes a cluster-wide controller.
    """

    legacy = Legacy(tmp_path)

    legacy.discover(monkeypatch)
    resources = legacy.bootstrap_state()["resources"]

    assert legacy.bootstrap_state()["completed_tasks"] == []
    assert resources["aurora"]["cluster_ownership"] == "CREATED"
    assert resources["aurora"]["master_secret_arn"] == MASTER_SECRET_ARN
    assert resources["aurora"]["instance_ids"] == [
        "gpu-fault-aurora-writer",
        "gpu-fault-aurora-reader",
    ]
    # deploy.sh names the diagnostics group ``<cluster>-pg`` exactly as bootstrap
    # does; it is ours only while it is the group the cluster actually runs on.
    assert resources["aurora"]["parameter_group"] == "gpu-fault-aurora-pg"
    assert resources["aurora"]["parameter_group_ownership"] == "CREATED"
    assert resources["nlb_network"]["vpc_id"] == "vpc-legacy"
    assert resources["monitoring_resources"]["workspace_ownership"] == "CREATED"
    assert resources["load_balancer_controller"] == {"external": True}
    assert resources["pod_identity_agent"]["ownership"] == "EXTERNAL"
    assert resources["pki"]["pki_secret_ownership"] == "CREATED"
    assert resources["control_plane_role"]["role_arn"] == CONTROL_ROLE_ARN
    executor = resources["executor_role:gpu-1"]
    assert executor["oidc_provider_ownership"] == "EXTERNAL"
    assert executor["oidc_provider_arn"] == (
        f"arn:aws:iam::{ACCOUNT}:oidc-provider/oidc.eks.{REGION}"
        ".amazonaws.com/id/LEGACY"
    )


def test_a_cluster_on_the_engine_default_group_registers_no_parameter_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deploy.sh run from before the diagnostics group left the cluster on
    ``default.aurora-postgresql16``. That group is AWS's, not ours: recording the
    derived name would have uninstall delete a group that may not exist, or
    worse, one another team created under the same name."""

    legacy = Legacy(tmp_path)
    legacy.db_clusters[0]["DBClusterParameterGroup"] = "default.aurora-postgresql16"

    legacy.discover(monkeypatch)
    aurora = legacy.bootstrap_state()["resources"]["aurora"]

    assert "parameter_group" not in aurora
    assert "parameter_group_ownership" not in aurora


def test_a_cloudformation_managed_workspace_is_recorded_as_external(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An AMP workspace owned by a stack must outlive uninstall.

    Deleting it would take the metrics history of whatever created it, and the
    stack would recreate it on the next deploy anyway.
    """

    legacy = Legacy(tmp_path)
    legacy.workspace_tags = {"aws:cloudformation:stack-name": "observability"}

    legacy.discover(monkeypatch)

    assert (
        legacy.bootstrap_state()["resources"]["monitoring_resources"][
            "workspace_ownership"
        ]
        == "EXTERNAL"
    )


def test_absent_optional_resources_are_left_out_of_the_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resource that does not exist must not be recorded at all.

    Recording an empty Pod Identity association or a missing PKI Secret would
    make uninstall try to delete something that was never created, and the run
    fails on the first such call.
    """

    legacy = Legacy(tmp_path)
    legacy.associations = {}
    legacy.pki_secret_names = []
    legacy.addon_installed = False

    legacy.discover(monkeypatch)
    resources = legacy.bootstrap_state()["resources"]

    assert set(resources) == {
        "nlb_network",
        "pki",
        "aurora",
        "monitoring_resources",
        "load_balancer_controller",
        "executor_role:gpu-1",
    }
    assert "pki_secret_id" not in resources["pki"]


def test_a_pki_secret_for_another_certificate_is_not_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Name matching alone is not ownership.

    The Secret name is a prefix shared by every site in the account, so the
    certificate ARN inside it is what ties one to this NLB. Adopting another
    site's Secret would delete its private key on uninstall.
    """

    legacy = Legacy(tmp_path)
    legacy.pki_secret_names = [
        "gpu-fault-nlb-private-pki-other",
        "gpu-fault-unrelated-secret",
    ]
    legacy.pki_certificate_arn = f"arn:aws:acm:{REGION}:{ACCOUNT}:certificate/other"

    legacy.discover(monkeypatch)

    assert "pki_secret_id" not in legacy.bootstrap_state()["resources"]["pki"]


@pytest.mark.parametrize(
    "value",
    [
        json.dumps(["training", "gpu-fault-system"]),
        "gpu-fault-system, training",
        "training,,gpu-fault-system,training",
    ],
)
def test_allowed_namespaces_are_normalized_from_either_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Both encodings appear in deployments this command has to migrate.

    The list ends up in the site file the executor authorizes against, so an
    accidental empty entry or a duplicate would widen or corrupt that boundary.
    """

    legacy = Legacy(tmp_path)
    legacy.allowed_namespaces = value

    site = legacy.discover(monkeypatch)

    assert site.release_config["clusters"][0]["allowed_namespaces"] == [
        "gpu-fault-system",
        "training",
    ]


@pytest.mark.parametrize(
    ("value", "message"),
    [("{}", "invalid format"), ("[]", "is empty"), (" , ", "is empty")],
)
def test_unusable_allowed_namespaces_stop_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str, message: str
) -> None:
    """An empty authorization list is not a safe default.

    Rendering it would produce a site whose executor accepts nothing, and the
    failure would only surface at the next deploy rather than here.
    """

    legacy = Legacy(tmp_path)
    legacy.allowed_namespaces = value

    with pytest.raises(BootstrapError, match=message):
        legacy.discover(monkeypatch)


def test_a_missing_secret_key_stops_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial connection Secret cannot be migrated silently.

    Every key is required to reach the GPU cluster; writing a site with an empty
    token would leave uninstall unable to reach the cluster it is cleaning up.
    """

    legacy = Legacy(tmp_path)
    legacy.connection_keys = tuple(
        key for key in legacy.connection_keys if key != "ca.crt"
    )

    with pytest.raises(BootstrapError, match="missing ca.crt"):
        legacy.discover(monkeypatch)


def test_an_executor_service_account_without_a_role_stops_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = Legacy(tmp_path)
    legacy.executor_annotations = {}

    with pytest.raises(BootstrapError, match="executor ServiceAccount has no IAM role"):
        legacy.discover(monkeypatch)


@pytest.mark.parametrize(
    ("clusters", "message"),
    [
        ([], "exactly one DB cluster"),
        (
            [
                {
                    "DBClusterIdentifier": "gpu-fault-aurora",
                    "MasterUserSecret": {"SecretArn": MASTER_SECRET_ARN},
                    "DBClusterMembers": [],
                    "DBSubnetGroup": "subnets",
                    "VpcSecurityGroups": [
                        {"VpcSecurityGroupId": "sg-a"},
                        {"VpcSecurityGroupId": "sg-b"},
                    ],
                }
            ],
            "exactly one solution SG",
        ),
    ],
)
def test_an_ambiguous_aurora_topology_stops_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clusters: list[dict[str, Any]],
    message: str,
) -> None:
    """Aurora is identified by its master Secret, and must resolve to one cluster.

    Guessing here would point uninstall's ``delete-db-cluster`` at a database
    this solution never created.
    """

    legacy = Legacy(tmp_path)
    legacy.db_clusters = clusters

    with pytest.raises(BootstrapError, match=message):
        legacy.discover(monkeypatch)


def test_two_pki_secrets_for_one_certificate_stop_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = Legacy(tmp_path)
    legacy.pki_secret_names = [
        "gpu-fault-nlb-private-pki-a",
        "gpu-fault-nlb-private-pki-b",
    ]

    with pytest.raises(BootstrapError, match="multiple legacy GPU fault PKI Secrets"):
        legacy.discover(monkeypatch)


def test_an_adot_config_without_a_workspace_stops_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The AMP workspace is only discoverable from the collector config.

    Without it there is no way to tell which workspace belongs to this site, and
    a guess would delete another team's metrics.
    """

    legacy = Legacy(tmp_path)
    legacy.collector = "exporters:\n  logging: {}\n"

    with pytest.raises(BootstrapError, match="cannot discover AMP workspace"):
        legacy.discover(monkeypatch)


def test_an_invalid_previous_release_snapshot_stops_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The release state carries the previous snapshot uninstall may need.

    Migrating a state whose snapshot cannot be hydrated would produce a site that
    looks complete but cannot describe what it would roll back to.
    """

    legacy = Legacy(tmp_path)
    legacy.release_state = {**legacy.release_state, "previous": "not-a-mapping"}

    with pytest.raises(BootstrapError, match="previous snapshot is invalid"):
        legacy.discover(monkeypatch)


def test_a_failed_kubectl_read_names_the_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery reads a dozen objects; the error has to say which one failed.

    An operator running uninstall against the wrong context sees a permission
    error, and without the query in the message there is nothing to act on.
    """

    legacy = Legacy(tmp_path)
    legacy.kubectl_failure = "Error from server (Forbidden): configmaps is forbidden"

    with pytest.raises(BootstrapError, match="legacy discovery failed: -n"):
        legacy.discover(monkeypatch)


def test_the_legit_executor_role_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H-12: an IAM role in the cluster's own account passes validation.

    The default fixture role lives in ``ACCOUNT``, so discovery renders it into
    the site unchanged -- the new check must not reject the legitimate value.
    """

    legacy = Legacy(tmp_path)

    site = legacy.discover(monkeypatch)

    assert (
        site.release_config["clusters"][0]["executor_irsa_role_arn"]
        == EXECUTOR_ROLE_ARN
    )


def test_an_executor_role_in_a_foreign_account_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H-12: the role ARN comes from an editable ServiceAccount annotation.

    A well-formed ARN in someone else's account would otherwise have the CLI
    trust -- and later assume -- a role outside the cluster's account, so it is
    refused before it can reach the site.
    """

    legacy = Legacy(tmp_path)
    foreign = "arn:aws:iam::111122223333:role/attacker"
    legacy.executor_annotations = {"eks.amazonaws.com/role-arn": foreign}

    with pytest.raises(BootstrapError, match="is not an IAM role in account"):
        legacy.discover(monkeypatch)


def test_a_malformed_executor_role_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H-12: a non-ARN annotation value is not a role at all."""

    legacy = Legacy(tmp_path)
    legacy.executor_annotations = {"eks.amazonaws.com/role-arn": "not-an-arn"}

    with pytest.raises(BootstrapError, match="is not a valid ARN"):
        legacy.discover(monkeypatch)


def test_a_non_iam_executor_role_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H-12: only an ``iam`` ``role/`` resource is an assumable role.

    A syntactically valid ARN for a different service (or a user) in the right
    account is still not a role the executor may become.
    """

    legacy = Legacy(tmp_path)
    legacy.executor_annotations = {
        "eks.amazonaws.com/role-arn": f"arn:aws:iam::{ACCOUNT}:user/someone"
    }

    with pytest.raises(BootstrapError, match="is not an IAM role in account"):
        legacy.discover(monkeypatch)


@pytest.mark.parametrize(
    "cluster_id",
    [
        "../../etc/cron.d/evil",
        "a/b",
        "with space",
        "..",
        "x" * 129,
        "-leading-dash-ok-but-slash/",
    ],
)
def test_a_malformed_connection_cluster_id_stops_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cluster_id: str
) -> None:
    """M-13: the cluster id is decoded from a Secret and names on-disk files.

    An id with a path separator, a dot segment, an illegal character, or an
    oversized length could escape the private state directory or be trusted as a
    destructive target, so it is validated against the site identifier shape
    before any file is written for it.
    """

    legacy = Legacy(tmp_path)
    legacy.connection_cluster_id = cluster_id

    with pytest.raises(BootstrapError, match="cluster-id"):
        legacy.discover(monkeypatch)
