"""The control plane's own provider reads for the departed-node proof.

Both reads answer one question -- does the provider still have this instance?
-- and fail closed: a throttle, a denied call, an answer that does not parse
or more pages than the limit is a ``ProviderLookupError``, never "absent".
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.departed_node_provider import (
    EC2_FILTER_BATCH,
    PROVIDER_PAGE_LIMIT,
    Boto3ProviderMembership,
    ProviderLookupError,
)


class ClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": "secret detail"}}


class FakeSageMaker:
    def __init__(self, *pages: Any, error: Exception | None = None) -> None:
        self.pages = list(pages)
        self.error = error
        self.requests: list[dict[str, Any]] = []

    def list_cluster_nodes(self, **request: Any) -> Any:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.pages[(len(self.requests) - 1) % len(self.pages)]


class FakeEc2:
    def __init__(self, *pages: Any, error: Exception | None = None) -> None:
        self.pages = list(pages)
        self.error = error
        self.requests: list[dict[str, Any]] = []

    def describe_instances(self, **request: Any) -> Any:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.pages[min(len(self.requests), len(self.pages)) - 1]


def _membership(
    **clients: Any,
) -> tuple[Boto3ProviderMembership, list[tuple[str, str]]]:
    created: list[tuple[str, str]] = []

    def factory(service: str, region: str) -> Any:
        created.append((service, region))
        return clients[service]

    return Boto3ProviderMembership(factory), created


def _instance(identifier: str, state: str) -> dict[str, Any]:
    return {"InstanceId": identifier, "State": {"Name": state}}


def test_hyperpod_lists_every_page_of_the_registered_cluster() -> None:
    sagemaker = FakeSageMaker(
        {"ClusterNodeSummaries": [{"InstanceId": "i-0a"}], "NextToken": "t1"},
        {"ClusterNodeSummaries": [{"InstanceId": "i-0b"}, {"NodeLogicalId": "x"}]},
    )
    membership, created = _membership(sagemaker=sagemaker)

    listed = membership.hyperpod_instance_ids("hp-a", "us-west-2")

    assert listed == frozenset({"i-0a", "i-0b"})
    assert sagemaker.requests == [
        {"ClusterName": "hp-a", "MaxResults": 100},
        {"ClusterName": "hp-a", "MaxResults": 100, "NextToken": "t1"},
    ]
    assert membership.hyperpod_instance_ids("hp-a", "us-west-2") == listed
    assert created == [("sagemaker", "us-west-2")], "one client per region"


@pytest.mark.parametrize(
    "answer",
    [
        {"ClusterNodeSummaries": None},
        "not a mapping",
        {"ClusterNodeSummaries": [], "NextToken": 7},
    ],
)
def test_an_unreadable_hyperpod_answer_is_a_failed_lookup(answer: Any) -> None:
    membership, _ = _membership(sagemaker=FakeSageMaker(answer))

    with pytest.raises(ProviderLookupError):
        membership.hyperpod_instance_ids("hp-a", "us-west-2")


def test_a_repeating_or_endless_page_token_is_a_failed_lookup() -> None:
    repeating = FakeSageMaker(
        {"ClusterNodeSummaries": [], "NextToken": "same"},
        {"ClusterNodeSummaries": [], "NextToken": "same"},
    )
    membership, _ = _membership(sagemaker=repeating)
    with pytest.raises(ProviderLookupError, match="repeats its page token"):
        membership.hyperpod_instance_ids("hp-a", "us-west-2")

    endless = FakeSageMaker(
        *(
            {"ClusterNodeSummaries": [], "NextToken": f"t{index}"}
            for index in range(PROVIDER_PAGE_LIMIT + 1)
        )
    )
    membership, _ = _membership(sagemaker=endless)
    with pytest.raises(ProviderLookupError, match="pages"):
        membership.hyperpod_instance_ids("hp-a", "us-west-2")


@pytest.mark.parametrize("code", ["ThrottlingException", "AccessDeniedException"])
def test_a_throttled_or_denied_call_names_its_code_and_nothing_else(code: str) -> None:
    membership, _ = _membership(
        sagemaker=FakeSageMaker(error=ClientError(code)),
        ec2=FakeEc2(error=ClientError(code)),
    )

    with pytest.raises(ProviderLookupError) as hyperpod:
        membership.hyperpod_instance_ids("hp-a", "us-west-2")
    with pytest.raises(ProviderLookupError) as ec2:
        membership.ec2_instance_states("us-west-2", ["i-0a"])

    for error in (hyperpod.value, ec2.value):
        assert code in str(error)
        assert "secret detail" not in str(error)


def test_a_cluster_without_a_name_or_a_region_is_never_read() -> None:
    membership, created = _membership(sagemaker=FakeSageMaker(), ec2=FakeEc2())

    with pytest.raises(ProviderLookupError, match="HyperPod cluster name"):
        membership.hyperpod_instance_ids(" ", "us-west-2")
    with pytest.raises(ProviderLookupError, match="region"):
        membership.hyperpod_instance_ids("hp-a", "")
    assert created == []


def test_ec2_filters_by_id_so_an_unknown_id_is_simply_absent() -> None:
    ec2 = FakeEc2(
        {
            "Reservations": [
                {"Instances": [_instance("i-0a", "Terminated")]},
                {"Instances": [_instance("i-0b", "running")]},
            ],
            "NextToken": "n1",
        },
        {"Reservations": [{"Instances": [_instance("i-0c", "stopped")]}]},
    )
    membership, created = _membership(ec2=ec2)

    states = membership.ec2_instance_states(
        "us-west-2", ["i-0b", "i-0a", "i-0d", "i-0a"]
    )

    assert states == {"i-0a": "terminated", "i-0b": "running", "i-0c": "stopped"}
    assert "i-0d" not in states, "an id EC2 no longer knows is absent, not an error"
    assert ec2.requests[0] == {
        "Filters": [{"Name": "instance-id", "Values": ["i-0a", "i-0b", "i-0d"]}]
    }
    assert ec2.requests[1]["NextToken"] == "n1"
    assert created == [("ec2", "us-west-2")]
    assert membership.ec2_instance_states("us-west-2", []) == {}


def test_more_ids_than_one_filter_takes_are_asked_in_batches() -> None:
    ec2 = FakeEc2({"Reservations": []})
    membership, _ = _membership(ec2=ec2)
    ids = [f"i-{index:08x}" for index in range(EC2_FILTER_BATCH + 1)]

    assert membership.ec2_instance_states("us-west-2", ids) == {}
    assert [len(item["Filters"][0]["Values"]) for item in ec2.requests] == [
        EC2_FILTER_BATCH,
        1,
    ]


@pytest.mark.parametrize(
    "answer",
    [
        {"Reservations": None},
        {"Reservations": [{"Instances": None}]},
        {"Reservations": [{"Instances": [{"InstanceId": "i-0a"}]}]},
    ],
)
def test_an_unreadable_ec2_answer_is_a_failed_lookup(answer: Any) -> None:
    membership, _ = _membership(ec2=FakeEc2(answer))

    with pytest.raises(ProviderLookupError, match="invalid"):
        membership.ec2_instance_states("us-west-2", ["i-0a"])
