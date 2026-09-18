from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from gpu_fault.admin.bootstrap import _ensure_pki, _ensure_private_zone
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapState,
    CommandRunner,
)
from gpu_fault.admin.bootstrap_tasks import foundation_task_graph
from gpu_fault.admin.notifications import NotificationRouting
from gpu_fault.admin.resource_registry_dns import zone_vpc_associations
from tests.admin._bootstrap_support import _cluster


def gpu(name: str, vpc: str):
    cpu = _cluster()
    return replace(
        cpu,
        role="gpu",
        input_arn=cpu.input_arn.replace("/control", f"/{name}"),
        eks_arn=cpu.eks_arn.replace("/control", f"/{name}"),
        eks_name=name,
        hyperpod_arn=cpu.hyperpod_arn.replace("/control", f"/{name}"),
        hyperpod_name=name,
        context=name,
        vpc_id=vpc,
    )


class Route53(CommandRunner):
    def __init__(self, clusters, *, exists=True, tagged=True):
        super().__init__()
        self.cpu = _cluster()
        self.clusters = clusters
        self.exists = exists
        self.tagged = tagged
        self.zone_id = "ZTEST"
        self.vpcs = [{"VPCRegion": self.cpu.region, "VPCId": self.cpu.vpc_id}]
        self.details = None
        self.change = {"ChangeInfo": {"Id": "/change/created", "Status": "PENDING"}}
        self.failure = ""
        self.lose_reply = False
        self.mutations = []
        self.calls = []

    def aws_json(self, region, service, operation, *arguments, mutate=False, **_kwargs):
        assert region == self.cpu.region
        self.calls.append((service, operation, arguments))
        if service == "secretsmanager":
            raise BootstrapError("PKI intentionally stopped after DNS")
        assert service == "route53", "the DNS caller used an unrelated service"
        if mutate:
            self.mutations.append((operation, arguments))
        if operation == "list-hosted-zones-by-name":
            return {
                "HostedZones": (
                    [
                        {
                            "Id": f"/hostedzone/{self.zone_id}",
                            "Name": "test.gpu-fault.internal.",
                            "Config": {"PrivateZone": True},
                        }
                    ]
                    if self.exists
                    else []
                )
            }
        if operation == "list-tags-for-resource":
            return {
                "ResourceTagSet": {
                    "Tags": (
                        [{"Key": "gpu-fault:site-id", "Value": "test"}]
                        if self.tagged
                        else []
                    )
                }
            }
        if operation == "get-hosted-zone":
            return self.details if self.details is not None else {"VPCs": self.vpcs}
        if operation == "create-hosted-zone":
            assert mutate and not self.exists
            self.exists = True
            return {"HostedZone": {"Id": f"/hostedzone/{self.zone_id}"}}
        if operation == "associate-vpc-with-hosted-zone":
            assert mutate, "association mutation bypassed the CommandRunner guard"
            if self.failure and not self.lose_reply:
                raise BootstrapError(self.failure)
            vpc = arguments[arguments.index("--vpc") + 1]
            target = next(
                cluster
                for cluster in self.clusters
                if vpc == f"VPCRegion={cluster.region},VPCId={cluster.vpc_id}"
            )
            self.vpcs.append({"VPCRegion": target.region, "VPCId": target.vpc_id})
            if self.failure:
                raise BootstrapError(self.failure)
            return self.change
        pytest.fail(f"unexpected DNS operation: {operation}")

    def run(self, arguments, *, mutate=False, **_kwargs):
        assert list(arguments[:3]) == ["aws", "route53", "change-tags-for-resource"], (
            "DNS called an unexpected command"
        )
        assert mutate, "tag mutation bypassed the CommandRunner guard"
        self.mutations.append(("change-tags-for-resource", tuple(arguments[3:])))
        self.tagged = True
        return ""


def ensure(runner, state=None):
    return _ensure_private_zone(
        runner,
        cpu=runner.cpu,
        gpu_clusters=runner.clusters,
        site_id="test",
        state=state,
    )


def checkpoint(state, runner, ownership):
    previous = {
        "hosted_zone_id": runner.zone_id,
        "vpc_associations": zone_vpc_associations(
            runner.cpu, runner.clusters, ownership_by_vpc=ownership
        ),
        "certificate_arn": "arn:aws:acm:us-east-1:123456789012:certificate/retained",
    }
    state.record("pki", previous)
    return previous


def associations(result):
    return {
        (item["vpc_region"], item["vpc_id"]): item
        for item in result["vpc_associations"]
    }


def test_created_zone_records_native_and_shared_gpu_ownership_once(tmp_path):
    clusters = [
        gpu("gpu-b", "vpc-shared"),
        gpu("gpu-a", "vpc-shared"),
        gpu("gpu-native", _cluster().vpc_id),
    ]
    runner = Route53(clusters, exists=False)
    state = BootstrapState(tmp_path / "state.json", site_id="test")
    result = ensure(runner, state)
    entries = associations(result)
    native = entries[("us-east-1", "vpc-control")]
    shared = entries[("us-east-1", "vpc-shared")]
    assert len(entries) == 2, "bootstrap duplicated a shared VPC association"
    assert native["native"] is True and native["cluster_ids"] == ["gpu-native"]
    assert shared["native"] is False and shared["cluster_ids"] == ["gpu-a", "gpu-b"]
    assert shared["resource_key"] == (
        "aws/route53/vpc-association/us-east-1/vpc-shared"
    )
    assert {item["ownership"] for item in entries.values()} == {"CREATED"}
    assert [operation for operation, _ in runner.mutations] == [
        "create-hosted-zone",
        "change-tags-for-resource",
        "associate-vpc-with-hosted-zone",
    ]
    assert state.result("pki") == result
    assert not state.is_complete("pki"), "DNS progress marked unfinished PKI complete"


@pytest.mark.parametrize("tagged", [False, True])
def test_existing_association_presence_does_not_grant_detach_ownership(tagged):
    runner = Route53([gpu("gpu-a", "vpc-a")], tagged=tagged)
    runner.vpcs.append({"VPCRegion": "us-east-1", "VPCId": "vpc-a"})
    result = ensure(runner)
    assert result["zone_ownership"] == "CREATED"
    assert {item["ownership"] for item in associations(result).values()} == {
        "EXTERNAL"
    }, "zone ownership or live presence silently granted association ownership"
    assert [operation for operation, _ in runner.mutations] == (
        [] if tagged else ["change-tags-for-resource"]
    )


@pytest.mark.parametrize("owner", ["CREATED", "EXTERNAL", "REUSED"])
def test_same_zone_explicit_ownership_survives_reordered_checkpoint(tmp_path, owner):
    runner = Route53([gpu("gpu-a", "vpc-a")])
    runner.vpcs.append({"VPCRegion": "us-east-1", "VPCId": "vpc-a"})
    state = BootstrapState(tmp_path / "state.json", site_id="test")
    previous = checkpoint(
        state,
        runner,
        {("us-east-1", "vpc-control"): "CREATED", ("us-east-1", "vpc-a"): owner},
    )
    previous["vpc_associations"].reverse()
    state.record("pki", previous)
    result = ensure(runner, state)
    assert associations(result)[("us-east-1", "vpc-a")]["ownership"] == owner
    assert state.result("pki")["certificate_arn"] == previous["certificate_arn"]
    assert runner.mutations == [], "known association ownership caused a mutation"


@pytest.mark.parametrize("owner", ["EXTERNAL", "REUSED"])
def test_missing_external_association_is_not_recreated_as_owned(tmp_path, owner):
    runner = Route53([gpu("gpu-a", "vpc-a")], tagged=False)
    state = BootstrapState(tmp_path / "state.json", site_id="test")
    previous = checkpoint(
        state,
        runner,
        {("us-east-1", "vpc-control"): "CREATED", ("us-east-1", "vpc-a"): owner},
    )
    with pytest.raises(BootstrapError, match="external Route53 association is absent"):
        ensure(runner, state)
    assert runner.mutations == [], "absence promoted external ownership to CREATED"
    assert state.result("pki") == previous, "refusal destroyed recorded provenance"


@pytest.mark.parametrize(
    "damage",
    ["zone", "ownership", "native", "duplicate", "invalid-checkpoint", "absent"],
)
def test_conflicting_checkpoint_stops_before_dns_mutations(tmp_path, damage):
    runner = Route53([gpu("gpu-a", "vpc-a")], tagged=False)
    state = BootstrapState(tmp_path / "state.json", site_id="test")
    previous = checkpoint(
        state,
        runner,
        {("us-east-1", "vpc-control"): "CREATED", ("us-east-1", "vpc-a"): "CREATED"},
    )
    if damage == "zone":
        previous["hosted_zone_id"] = "ZOTHER"
    elif damage == "ownership":
        previous["vpc_associations"][1]["ownership"] = "UNKNOWN"
    elif damage == "native":
        previous["vpc_associations"][1]["native"] = True
    elif damage == "duplicate":
        duplicate = {**previous["vpc_associations"][1], "ownership": "EXTERNAL"}
        previous["vpc_associations"].append(duplicate)
    elif damage == "invalid-checkpoint":
        previous = []
    else:
        runner.exists = False
    state.record("pki", previous)
    before = deepcopy(previous)
    with pytest.raises(BootstrapError, match="Route53"):
        ensure(runner, state)
    assert runner.mutations == [], "ambiguous provenance reached a DNS mutation"
    assert state.result("pki") == before


@pytest.mark.parametrize(
    "inventory",
    [
        None,
        {},
        "unavailable",
        [None],
        [{}],
        [],
        [{"VPCRegion": "us-east-1", "VPCId": "vpc-a"}],
    ],
)
def test_unknown_or_missing_cpu_associations_are_not_invented(inventory):
    runner = Route53([gpu("gpu-a", "vpc-a")], tagged=False)
    runner.details = {"VPCs": inventory}
    with pytest.raises(BootstrapError, match="Route53"):
        ensure(runner)
    assert runner.mutations == [], "unknown VPC state allowed adoption or association"


def test_invalid_hosted_zone_response_stops_before_any_mutation():
    runner = Route53([gpu("gpu-a", "vpc-a")], tagged=False)
    runner.details = []
    with pytest.raises(BootstrapError, match="invalid VPC association inventory"):
        ensure(runner)
    assert runner.mutations == [], "invalid Route53 JSON allowed a mutation"


@pytest.mark.parametrize(
    "change",
    [
        None,
        [],
        {},
        {"ChangeInfo": None},
        {"ChangeInfo": {}},
        {"ChangeInfo": {"Id": 1, "Status": "PENDING"}},
        {"ChangeInfo": {"Id": "", "Status": "PENDING"}},
        {"ChangeInfo": {"Id": " ", "Status": "PENDING"}},
        {"ChangeInfo": {"Id": "/change/", "Status": "PENDING"}},
        {"ChangeInfo": {"Id": "not-a-change-id", "Status": "PENDING"}},
        {"ChangeInfo": {"Id": "/change/test", "Status": "FAILED"}},
    ],
)
def test_invalid_association_response_does_not_record_creation_proof(tmp_path, change):
    runner = Route53([gpu("gpu-a", "vpc-a")])
    runner.change = change
    state = BootstrapState(tmp_path / "state.json", site_id="test")
    with pytest.raises(BootstrapError, match="invalid ChangeInfo"):
        ensure(runner, state)
    assert set(associations(state.result("pki"))) == {("us-east-1", "vpc-control")}
    assert not state.is_complete("pki"), "an invalid response completed the PKI task"


@pytest.mark.parametrize("lose_reply", [False, True])
def test_failed_association_never_becomes_creation_proof_on_resume(
    tmp_path, lose_reply
):
    runner = Route53([gpu("gpu-a", "vpc-a")])
    runner.failure = "PriorRequestNotComplete"
    runner.lose_reply = lose_reply
    state = BootstrapState(tmp_path / "state.json", site_id="test")
    with pytest.raises(BootstrapError, match="PriorRequestNotComplete"):
        ensure(runner, state)
    assert set(associations(state.result("pki"))) == {("us-east-1", "vpc-control")}
    runner.failure = ""
    runner.mutations.clear()
    result = ensure(runner, BootstrapState(state.path, site_id="test"))
    gpu_association = associations(result)[("us-east-1", "vpc-a")]
    assert gpu_association["ownership"] == ("EXTERNAL" if lose_reply else "CREATED")
    assert [operation for operation, _ in runner.mutations] == (
        [] if lose_reply else ["associate-vpc-with-hosted-zone"]
    )


def test_pki_failure_preserves_confirmed_dns_provenance_for_crash_resume(tmp_path):
    runner = Route53([gpu("gpu-a", "vpc-a")], exists=False)
    state = BootstrapState(tmp_path / "state.json", site_id="test")
    graph = foundation_task_graph(
        runner=runner,
        state=state,
        cpu=runner.cpu,
        gpu_clusters=runner.clusters,
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id="test",
        state_dir=tmp_path,
        admin_email="admin@example.com",
        routing=NotificationRouting(
            sender="admin@example.com",
            recipients=("admin@example.com",),
            subject_prefix="test",
        ),
        aurora_capacity=None,
        ensure_nlb_network=lambda *_, **__: {},
        ensure_pki=_ensure_pki,
        ensure_aurora=lambda *_, **__: {},
    )
    with pytest.raises(BootstrapError, match="PKI intentionally stopped after DNS"):
        graph.tasks["pki"]()
    assert not state.is_complete("pki"), "certificate preparation never completed"
    runner.mutations.clear()
    restored = BootstrapState(state.path, site_id="test")
    result = ensure(runner, restored)
    assert {item["ownership"] for item in associations(result).values()} == {"CREATED"}
    assert runner.mutations == [], "retry did not preserve already-created associations"


def test_second_association_failure_preserves_first_creation_proof(
    tmp_path, monkeypatch
):
    runner = Route53([gpu("gpu-a", "vpc-a"), gpu("gpu-b", "vpc-b")])
    state = BootstrapState(tmp_path / "state.json", site_id="test")
    original = runner.aws_json

    def fail_second(region, service, operation, *arguments, **kwargs):
        if operation == "associate-vpc-with-hosted-zone" and (
            "VPCRegion=us-east-1,VPCId=vpc-b" in arguments
        ):
            raise BootstrapError("second association refused")
        return original(region, service, operation, *arguments, **kwargs)

    monkeypatch.setattr(runner, "aws_json", fail_second)
    with pytest.raises(BootstrapError, match="second association refused"):
        ensure(runner, state)
    retained = associations(state.result("pki"))
    assert set(retained) == {("us-east-1", "vpc-control"), ("us-east-1", "vpc-a")}
    assert retained[("us-east-1", "vpc-a")]["ownership"] == "CREATED"
    monkeypatch.setattr(runner, "aws_json", original)
    runner.mutations.clear()
    result = ensure(runner, BootstrapState(state.path, site_id="test"))
    assert associations(result)[("us-east-1", "vpc-a")]["ownership"] == "CREATED"
    assert associations(result)[("us-east-1", "vpc-b")]["ownership"] == "CREATED"
    assert [
        arguments[arguments.index("--vpc") + 1] for _, arguments in runner.mutations
    ] == ["VPCRegion=us-east-1,VPCId=vpc-b"], (
        "retry replayed or adopted an already-proven association"
    )
