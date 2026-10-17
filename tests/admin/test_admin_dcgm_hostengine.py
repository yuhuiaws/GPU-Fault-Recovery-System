"""The DCGM host engine mode a plain EKS cluster declares on its node groups.

``gpu-fault.io/dcgm-hostengine=host`` on every GPU node group the cluster's
selector selects declares a host DCGM (``nvidia-dcgm`` runs on each node);
none declares the exporter's embedded hostengine (what every plain EKS cluster
ran before); anything else is refused before any change, naming each group.
The mode is node-image capability, re-derived by every deploy and join and
recorded on the cluster's site entry (``dcgmHostEngine`` /
``dcgmHostEngineSource``); a HyperPod cluster is always ``host`` and records
nothing. The release-side half (rendering, diff, waits, status) lives in
``tests/regional/test_release_dcgm_hostengine.py``.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import bootstrap as admin_bootstrap
from gpu_fault.admin import bootstrap_ownership as ownership
from gpu_fault.admin import cluster_join as join
from gpu_fault.admin.bootstrap_common import BootstrapError, ClusterIdentity
from gpu_fault.admin.bootstrap_discovery import cluster_provider_entries
from gpu_fault.admin.bootstrap_site import preserve_existing_site_contract
from gpu_fault.admin.node_scope_command import dcgm_hostengine_report
from gpu_fault.admin.site import SiteConfigError, load_site
from gpu_fault.cluster_scope import (
    DCGM_HOSTENGINE_LABEL,
    OWNERSHIP_LABEL,
    DcgmHostEngine,
    DcgmHostEngineSource,
    dcgm_hostengine,
    dcgm_hostengine_environment,
    dcgm_hostengine_source,
    validate_dcgm_hostengine,
)
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from tests.admin.test_admin_site import site_file

REGION = "us-east-1"
ACCOUNT = "123456789012"
EKS_NAME = "gpu-eks-cluster"
EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{EKS_NAME}"
SITE_ID = "us-east-1-control-985fd370"
CPU_GROUP = "cpu-node-group"
CIDRS = {"subnet-gpu-a": "10.0.0.0/20", "subnet-gpu-b": "10.0.16.0/20"}


class NodeGroupRunner:
    """``aws`` answers for a plain EKS cluster with one CPU and N GPU node groups.

    ``groups`` maps each GPU node group to the value of its
    ``gpu-fault.io/dcgm-hostengine`` label (``None``: absent); every GPU group
    carries the site's ownership label unless named in ``unowned``.
    """

    def __init__(
        self, groups: dict[str, str | None], *, unowned: tuple[str, ...] = ()
    ) -> None:
        self.groups = groups
        self.unowned = unowned
        self.calls: list[tuple[str, str]] = []

    def _labels(self, name: str) -> dict[str, str]:
        labels = {"node-type": "gpu"}
        if name not in self.unowned:
            labels[OWNERSHIP_LABEL] = SITE_ID
        value = self.groups[name]
        if value is not None:
            labels[DCGM_HOSTENGINE_LABEL] = value
        return labels

    def aws_json(self, region, service, operation, *arguments, **_options):
        self.calls.append((service, operation))
        assert region == REGION, "discovery crossed the Region boundary"
        if (service, operation) == ("sagemaker", "list-clusters"):
            return {"ClusterSummaries": []}
        if (service, operation) == ("eks", "describe-cluster"):
            return {
                "cluster": {
                    "arn": EKS_ARN,
                    "status": "ACTIVE",
                    "resourcesVpcConfig": {
                        "vpcId": "vpc-gpu-eks",
                        "subnetIds": ["subnet-cp-a"],
                    },
                }
            }
        if (service, operation) == ("eks", "list-nodegroups"):
            return {"nodegroups": [CPU_GROUP, *self.groups]}
        if (service, operation) == ("eks", "describe-nodegroup"):
            name = arguments[arguments.index("--nodegroup-name") + 1]
            if name == CPU_GROUP:
                return {
                    "nodegroup": {
                        "nodegroupName": name,
                        "status": "ACTIVE",
                        "instanceTypes": ["m7i-flex.xlarge"],
                        "subnets": ["subnet-cpu-a"],
                        "labels": {"node-type": "cpu"},
                    }
                }
            return {
                "nodegroup": {
                    "nodegroupName": name,
                    "status": "ACTIVE",
                    "instanceTypes": ["p5en.48xlarge"],
                    "subnets": ["subnet-gpu-a", "subnet-gpu-b"],
                    "labels": self._labels(name),
                }
            }
        if (service, operation) == ("ec2", "describe-subnets"):
            ids = arguments[arguments.index("--subnet-ids") + 1 :]
            return {
                "Subnets": [
                    {"SubnetId": item, "CidrBlock": CIDRS[item]} for item in ids
                ]
            }
        raise AssertionError(f"unexpected discovery call {service}/{operation}")


def _discover(runner: NodeGroupRunner, **overrides: Any) -> ClusterIdentity:
    arguments: dict[str, Any] = dict(
        cluster_arn=EKS_ARN, role="gpu", context="gpu-eks", site_id=SITE_ID
    )
    arguments.update(overrides)
    return admin_bootstrap.discover_cluster(runner, **arguments)


def _facts(name: str, value: str | None) -> ownership.NodeGroupFacts:
    labels = {OWNERSHIP_LABEL: SITE_ID}
    if value is not None:
        labels[DCGM_HOSTENGINE_LABEL] = value
    return ownership.eks_nodegroup_facts(
        {"nodegroupName": name, "instanceTypes": ["p5en.48xlarge"], "labels": labels}
    )


# --- resolution ---------------------------------------------------------------


def test_every_selected_group_labelled_host_declares_the_host_engine() -> None:
    mode, source = ownership.resolve_dcgm_hostengine(
        [_facts("gpu-a", "host"), _facts("gpu-b", "host")], cluster=EKS_NAME
    )
    assert (mode, source) == (
        DcgmHostEngine.HOST,
        DcgmHostEngineSource.NODE_GROUP_LABEL,
    )


def test_no_selected_group_labelled_is_the_embedded_engine() -> None:
    mode, source = ownership.resolve_dcgm_hostengine(
        [_facts("gpu-a", None), _facts("gpu-b", None)], cluster=EKS_NAME
    )
    assert (mode, source) == (DcgmHostEngine.EMBEDDED, DcgmHostEngineSource.DEFAULT)


def test_a_mixed_cluster_is_refused_naming_every_group_and_the_remedy() -> None:
    with pytest.raises(BootstrapError) as refusal:
        ownership.resolve_dcgm_hostengine(
            [_facts("gpu-a", "host"), _facts("gpu-b", None)], cluster=EKS_NAME
        )
    message = str(refusal.value)
    assert message.startswith(
        f"managed node groups of EKS cluster {EKS_NAME} disagree on "
        f"{DCGM_HOSTENGINE_LABEL}: gpu-a {DCGM_HOSTENGINE_LABEL}=host; "
        f"gpu-b {DCGM_HOSTENGINE_LABEL}=absent."
    ), message
    assert (
        f"aws eks update-nodegroup-config --cluster-name {EKS_NAME} "
        f"--nodegroup-name gpu-b --labels "
        f"addOrUpdateLabels={{{DCGM_HOSTENGINE_LABEL}=host}}"
    ) in message, "the remedy is the command for the unlabelled group"
    assert "§5.0.1" in message, "the refusal names the launch template section"


@pytest.mark.parametrize("value", ["embedded", "Host", "true", ""])
def test_any_other_label_value_is_refused_rather_than_read_as_embedded(
    value: str,
) -> None:
    with pytest.raises(BootstrapError, match="values other than host") as refusal:
        ownership.resolve_dcgm_hostengine(
            [_facts("gpu-a", "host"), _facts("gpu-b", value)], cluster=EKS_NAME
        )
    assert f"gpu-b {DCGM_HOSTENGINE_LABEL}={value}" in str(refusal.value), (
        "the evidence quotes the value the group carries"
    )


# --- discovery ------------------------------------------------------------------


def test_joining_discovery_records_the_mode_of_the_owned_groups() -> None:
    host = _discover(NodeGroupRunner({"gpu-a": "host", "gpu-b": "host"}))
    assert (host.dcgm_hostengine, host.dcgm_hostengine_source) == (
        "host",
        "node-group-label",
    )
    embedded = _discover(NodeGroupRunner({"gpu-a": None}))
    assert (embedded.dcgm_hostengine, embedded.dcgm_hostengine_source) == (
        "embedded",
        "default",
    )
    # A GPU group the site does not own does not vote.
    unowned = _discover(
        NodeGroupRunner({"gpu-a": "host", "gpu-b": None}, unowned=("gpu-b",))
    )
    assert unowned.dcgm_hostengine == "host", "only the selected groups declare"


def test_a_mixed_declaration_refuses_discovery_before_anything_is_recorded() -> None:
    runner = NodeGroupRunner({"gpu-a": "host", "gpu-b": None})
    with pytest.raises(
        BootstrapError, match="disagree on gpu-fault.io/dcgm-hostengine"
    ):
        _discover(runner)
    assert ("ec2", "describe-subnets") not in runner.calls, (
        "the refusal comes from the node groups, before the subnet reads"
    )


def test_a_managed_cluster_re_derives_the_mode_on_every_deploy() -> None:
    """Unlike the recorded selector, the mode is re-read from the groups the
    recorded selector covers: flipping the label flips the identity."""

    recorded = (
        {
            "clusterId": EKS_NAME,
            "provider": "eks-managed",
            "eksClusterArn": EKS_ARN,
            "nodeSelector": "eks.amazonaws.com/nodegroup=gpu-a",
            "dcgmHostEngine": "embedded",
            "dcgmHostEngineSource": "default",
        },
    )
    flipped = _discover(
        NodeGroupRunner({"gpu-a": "host", "gpu-b": None}), existing_clusters=recorded
    )
    assert flipped.node_selector == "eks.amazonaws.com/nodegroup=gpu-a", (
        "membership is kept as recorded"
    )
    assert flipped.dcgm_hostengine == "host", (
        "gpu-b is not covered by the recorded selector, so only gpu-a declares"
    )


def test_identity_only_reads_and_hyperpod_identities_record_no_mode() -> None:
    identity = _discover(
        NodeGroupRunner({"gpu-a": "host", "gpu-b": None}),
        site_id=None,
        resolve_node_scope=False,
    )
    assert (identity.dcgm_hostengine, identity.dcgm_hostengine_source) == ("", ""), (
        "an identity-only read (remove-cluster, rotate-token) resolves no mode, "
        "so a mixed declaration does not block it"
    )
    hyperpod = ClusterIdentity(
        input_arn="arn:aws:sagemaker:us-east-1:123456789012:cluster/hp",
        role="gpu",
        region=REGION,
        account_id=ACCOUNT,
        hyperpod_arn="arn:aws:sagemaker:us-east-1:123456789012:cluster/hp",
        hyperpod_name="hp",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/hp-eks",
        eks_name="hp-eks",
        vpc_id="vpc-a",
        subnet_ids=(),
        node_recovery="None",
        context="hp",
    )
    entries = cluster_provider_entries(hyperpod)
    assert "dcgmHostEngine" not in entries and "dcgmHostEngineSource" not in entries, (
        "a HyperPod entry never records a mode: it is always host"
    )
    assert dcgm_hostengine(hyperpod) is DcgmHostEngine.HOST
    assert dcgm_hostengine_source(hyperpod) is DcgmHostEngineSource.PROVIDER
    assert dcgm_hostengine_environment(hyperpod) == "", (
        "nothing reaches a HyperPod node's environment"
    )


def test_the_eks_site_entry_records_the_discovered_mode() -> None:
    identity = _discover(NodeGroupRunner({"gpu-a": "host"}))
    entries = cluster_provider_entries(identity)
    assert entries["dcgmHostEngine"] == "host"
    assert entries["dcgmHostEngineSource"] == "node-group-label"
    embedded = cluster_provider_entries(_discover(NodeGroupRunner({"gpu-a": None})))
    assert (embedded["dcgmHostEngine"], embedded["dcgmHostEngineSource"]) == (
        "embedded",
        "default",
    )


def test_a_regenerated_site_keeps_the_re_derived_mode_not_the_previous_one() -> None:
    previous = {
        "clusterId": EKS_NAME,
        "context": "gpu-eks",
        "provider": "eks-managed",
        "nodeSelector": f"{OWNERSHIP_LABEL}={SITE_ID}",
        "nodeSelectorSource": "label-convention",
        "dcgmHostEngine": "embedded",
        "dcgmHostEngineSource": "default",
    }
    regenerated = {
        **previous,
        "dcgmHostEngine": "host",
        "dcgmHostEngineSource": "node-group-label",
    }
    merged = preserve_existing_site_contract(
        {"spec": {"clusters": [dict(regenerated)]}},
        {"spec": {"clusters": [dict(previous)]}},
    )
    (entry,) = merged["spec"]["clusters"]
    assert (entry["dcgmHostEngine"], entry["dcgmHostEngineSource"]) == (
        "host",
        "node-group-label",
    ), "the mode is node-image capability: the deploy's discovery wins"
    assert entry["nodeSelector"] == previous["nodeSelector"]


def test_a_resumed_join_recorded_before_the_mode_existed_is_not_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpu_fault.admin.cluster_join_types import JoinClusterRequest

    site = load_site(site_file(tmp_path))
    request = JoinClusterRequest(
        site=site, gpu_cluster_arn=EKS_ARN, state_dir=tmp_path / "join"
    )
    current = _discover(NodeGroupRunner({"gpu-a": "host"}))
    monkeypatch.setattr(join, "discover_cluster", lambda *_args, **_kwargs: current)
    recorded = asdict(current)
    for key in ("dcgm_hostengine", "dcgm_hostengine_source"):
        recorded.pop(key)
    join.validate_saved_join_target(
        request,
        {"target": recorded, "cluster_id": join.default_cluster_id(current)},
        NodeGroupRunner({}),
    )
    drifted = {**asdict(current), "dcgm_hostengine": "embedded"}
    with pytest.raises(join.JoinTargetIdentityError):
        join.validate_saved_join_target(
            request,
            {"target": drifted, "cluster_id": join.default_cluster_id(current)},
            NodeGroupRunner({}),
        )


# --- site.yaml and the release config -----------------------------------------


def _site_with_eks(tmp_path: Path, **overrides: Any) -> Path:
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    eks = dict(document["spec"]["clusters"][0])
    eks.pop("hyperpodClusterName", None)
    eks.update(
        {
            "clusterId": EKS_NAME,
            "context": EKS_NAME,
            "provider": "eks-managed",
            "nodeSelector": f"{OWNERSHIP_LABEL}={SITE_ID}",
            "nodeSelectorSource": "label-convention",
            "eksClusterArn": EKS_ARN,
            **overrides,
        }
    )
    document["spec"]["clusters"].append(
        {key: value for key, value in eks.items() if value is not None}
    )
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_the_recorded_mode_round_trips_into_the_release_target(tmp_path: Path) -> None:
    site = load_site(
        _site_with_eks(
            tmp_path, dcgmHostEngine="host", dcgmHostEngineSource="node-group-label"
        )
    )
    hyperpod, eks = site.release_config["clusters"]
    assert (eks["dcgm_hostengine"], eks["dcgm_hostengine_source"]) == (
        "host",
        "node-group-label",
    )
    assert "dcgm_hostengine" not in hyperpod, "HyperPod renders exactly as before"
    target = ClusterTarget.from_mapping(eks, expected_region=REGION)
    assert (target.dcgm_hostengine, target.dcgm_hostengine_source) == (
        "host",
        "node-group-label",
    )
    assert dcgm_hostengine_environment(target) == "host"
    legacy = load_site(_site_with_eks(tmp_path / "legacy"))
    _hp, legacy_eks = legacy.release_config["clusters"]
    assert "dcgm_hostengine" not in legacy_eks, (
        "an entry written before the field renders byte for byte as before"
    )
    legacy_target = ClusterTarget.from_mapping(legacy_eks, expected_region=REGION)
    assert dcgm_hostengine(legacy_target) is DcgmHostEngine.EMBEDDED, (
        "and runs what it ran: the embedded engine"
    )


@pytest.mark.parametrize(
    ("mode", "source", "message"),
    [
        ("host", "default", "is recorded with source node-group-label"),
        ("embedded", "node-group-label", "is recorded with source default"),
        ("host", None, "got none"),
        ("hosted", "node-group-label", "must be one of host, embedded"),
        ("host", "provider", "source must be one of"),
    ],
)
def test_an_inconsistent_recorded_mode_fails_closed(
    tmp_path: Path, mode: str, source: str | None, message: str
) -> None:
    with pytest.raises(SiteConfigError, match="dcgmHostEngine") as refusal:
        load_site(
            _site_with_eks(tmp_path, dcgmHostEngine=mode, dcgmHostEngineSource=source)
        )
    assert message in str(refusal.value), str(refusal.value)
    with pytest.raises(ValueError, match="DCGM host engine"):
        validate_dcgm_hostengine("eks-managed", mode, source)


def test_a_hyperpod_entry_cannot_record_a_mode(tmp_path: Path) -> None:
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["spec"]["clusters"][0]["dcgmHostEngine"] = "host"
    document["spec"]["clusters"][0]["dcgmHostEngineSource"] = "node-group-label"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(SiteConfigError, match="runs DCGM as a host service"):
        load_site(path)
    with pytest.raises(ReleaseError, match="runs DCGM as a host service"):
        ClusterTarget.from_mapping(
            {
                "cluster_id": "hp-gpu-a",
                "context": "hp",
                "executor_irsa_role_arn": "arn:aws:iam::123456789012:role/x",
                "region": REGION,
                "hyperpod_cluster_name": "hp-gpu-a",
                "eks_cluster_arn": f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/hp-eks",
                "dcgm_hostengine": "embedded",
                "dcgm_hostengine_source": "default",
            },
            expected_region=REGION,
        )


# --- node-scope ------------------------------------------------------------------


def _entry(**overrides: Any) -> dict[str, Any]:
    return {
        "cluster_id": EKS_NAME,
        "provider": "eks-managed",
        "eks_cluster_arn": EKS_ARN,
        "node_selector": f"{OWNERSHIP_LABEL}={SITE_ID}",
        **overrides,
    }


def test_node_scope_reports_a_declared_flip_as_pending_deploy() -> None:
    facts = [_facts("gpu-a", "host"), _facts("gpu-b", "host")]
    pending = dcgm_hostengine_report(
        _entry(), facts, selector=f"{OWNERSHIP_LABEL}={SITE_ID}", cluster=EKS_NAME
    )
    assert pending == {
        "recorded": "embedded",
        "source": "default",
        "declared": "host",
        "refusal": None,
        "state": "pending deploy",
    }
    converged = dcgm_hostengine_report(
        _entry(dcgm_hostengine="host", dcgm_hostengine_source="node-group-label"),
        facts,
        selector=f"{OWNERSHIP_LABEL}={SITE_ID}",
        cluster=EKS_NAME,
    )
    assert converged["state"] == "converged" and converged["declared"] == "host", (
        converged
    )


def test_node_scope_reports_a_mixed_declaration_as_blocked_with_the_refusal() -> None:
    report = dcgm_hostengine_report(
        _entry(dcgm_hostengine="host", dcgm_hostengine_source="node-group-label"),
        [_facts("gpu-a", "host"), _facts("gpu-b", None)],
        selector=f"{OWNERSHIP_LABEL}={SITE_ID}",
        cluster=EKS_NAME,
    )
    assert report["state"] == "blocked" and report["declared"] is None, report
    assert "disagree on gpu-fault.io/dcgm-hostengine" in str(report["refusal"])


def test_node_scope_reports_hyperpod_as_host_from_the_provider() -> None:
    report = dcgm_hostengine_report(
        {"cluster_id": "hp-gpu-a", "hyperpod_cluster_name": "hp-gpu-a"},
        [],
        selector="sagemaker.amazonaws.com/cluster-name=hp-gpu-a",
        cluster="hp-gpu-a",
    )
    assert report == {
        "recorded": "host",
        "source": "provider",
        "declared": "host",
        "refusal": None,
        "state": "converged",
    }


def test_a_replaced_identity_keeps_the_new_fields_comparable() -> None:
    """``ClusterIdentity`` equality now covers the mode: a join resumed after
    the label changed is drift (fail closed), the fields default to empty."""

    identity = _discover(NodeGroupRunner({"gpu-a": "host"}))
    assert replace(identity, dcgm_hostengine="embedded") != identity
    assert ClusterIdentity(**{**asdict(identity), "dcgm_hostengine": ""}) != identity
