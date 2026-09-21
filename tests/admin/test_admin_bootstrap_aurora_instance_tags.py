"""Every Aurora instance carries the site tag the uninstall proves ownership by."""

from __future__ import annotations

import json
from typing import Any, Sequence

import pytest

from gpu_fault.admin import bootstrap_aurora as aurora
from gpu_fault.admin.bootstrap_common import BootstrapError
from tests.admin.test_admin_bootstrap_aurora import InstanceCreationRunner

SITE = "site-1"
ARGUMENTS: dict[str, Any] = {
    "aws_region": "us-east-1",
    "cluster_id": "aurora-a",
    "availability_zones": ["us-east-1a", "us-east-1b"],
    "safe_name": lambda value, maximum: value[:maximum],
    "wait": False,
    "site_id": SITE,
}


class TaggedRunner(InstanceCreationRunner):
    """The creation fake plus instance ARNs, tag lists and ``add-tags-to-resource``."""

    def __init__(self, tags: dict[str, list[dict[str, str]]], **keywords: Any) -> None:
        super().__init__(**keywords)
        self.tags = tags

    def aws_json(self, region: str, *arguments: str, **keywords: Any) -> dict:
        document = super().aws_json(region, *arguments, **keywords)
        for instance in document.get("DBInstances", []):
            name = instance["DBInstanceIdentifier"]
            instance["DBInstanceArn"] = f"arn:aws:rds:us-east-1:123456789012:db:{name}"
            instance["TagList"] = json.loads(json.dumps(self.tags.get(name, [])))
        return document

    def run(self, arguments: Sequence[str], **keywords: Any) -> str:
        if arguments[2] == "add-tags-to-resource":
            self.calls.append(tuple(arguments))
            return ""
        return super().run(arguments, **keywords)


def _tag_calls(runner: TaggedRunner) -> list[tuple[str, ...]]:
    return [call for call in runner.calls if call[2] == "add-tags-to-resource"]


def test_existing_untagged_instances_receive_the_site_tag_once() -> None:
    runner = TaggedRunner(
        {},
        instances={"aurora-a-writer": "available", "aurora-a-reader": "available"},
        primary="aurora-a-writer",
        cluster_status="available",
    )

    aurora.ensure_serverless_instances(runner, **ARGUMENTS)

    tagged = sorted(
        call[call.index("--resource-name") + 1] for call in _tag_calls(runner)
    )
    assert tagged == [
        "arn:aws:rds:us-east-1:123456789012:db:aurora-a-reader",
        "arn:aws:rds:us-east-1:123456789012:db:aurora-a-writer",
    ], "both members are tagged by ARN"
    assert all(
        f"Key=gpu-fault:site-id,Value={SITE}" in call for call in _tag_calls(runner)
    ), "the site tag is the one the uninstall proves ownership by"
    assert "create-db-instance" not in {call[2] for call in runner.calls}, (
        "existing instances are tagged, not recreated"
    )


def test_tagged_instances_are_left_alone_and_foreign_tags_fail_closed() -> None:
    own = [{"Key": "gpu-fault:site-id", "Value": SITE}]
    runner = TaggedRunner(
        {"aurora-a-writer": own, "aurora-a-reader": own},
        instances={"aurora-a-writer": "available", "aurora-a-reader": "available"},
        primary="aurora-a-writer",
        cluster_status="available",
    )
    aurora.ensure_serverless_instances(runner, **ARGUMENTS)
    assert _tag_calls(runner) == [], "an owned instance is not re-tagged"

    foreign = TaggedRunner(
        {"aurora-a-writer": [{"Key": "gpu-fault:site-id", "Value": "other-site"}]},
        instances={"aurora-a-writer": "available", "aurora-a-reader": "available"},
        primary="aurora-a-writer",
        cluster_status="available",
    )
    with pytest.raises(BootstrapError, match="belongs to site 'other-site'"):
        aurora.ensure_serverless_instances(foreign, **ARGUMENTS)
    assert _tag_calls(foreign) == [], "a foreign instance is never claimed"


def test_new_instances_are_created_with_the_site_tag() -> None:
    runner = TaggedRunner({})

    aurora.ensure_serverless_instances(runner, **ARGUMENTS)

    creates = [call for call in runner.calls if call[2] == "create-db-instance"]
    assert len(creates) == 2, "writer and reader are created"
    for call in creates:
        assert (
            call[call.index("--tags") + 1] == f"Key=gpu-fault:site-id,Value={SITE}"
        ), "a new instance is born with the site tag"


def test_without_a_site_id_the_legacy_call_shape_is_unchanged() -> None:
    runner = TaggedRunner(
        {},
        instances={"aurora-a-writer": "available", "aurora-a-reader": "available"},
        primary="aurora-a-writer",
        cluster_status="available",
    )
    aurora.ensure_serverless_instances(
        runner, **{key: value for key, value in ARGUMENTS.items() if key != "site_id"}
    )
    assert _tag_calls(runner) == [], "callers without a site id tag nothing"


class GroupRunner:
    """A parameter-group fake: existing or absent group, its tags, and the calls."""

    def __init__(self, *, exists: bool, tags: list[dict[str, str]]) -> None:
        self.exists = exists
        self.tags = tags
        self.calls: list[tuple[str, ...]] = []

    def aws_json(self, _region: str, *arguments: str, **_keywords: Any) -> dict:
        self.calls.append(("aws", *arguments))
        operation = arguments[1]
        if operation == "describe-db-cluster-parameter-groups":
            if not self.exists:
                raise BootstrapError(
                    "command failed (254): aws: An error occurred "
                    "(DBParameterGroupNotFound) when calling the operation"
                )
            return {
                "DBClusterParameterGroups": [
                    {
                        "DBClusterParameterGroupName": "aurora-a-pg",
                        "DBClusterParameterGroupArn": (
                            "arn:aws:rds:us-east-1:123456789012:cluster-pg:aurora-a-pg"
                        ),
                    }
                ]
            }
        if operation == "list-tags-for-resource":
            return {"TagList": json.loads(json.dumps(self.tags))}
        if operation == "describe-db-engine-versions":
            return {
                "DBEngineVersions": [{"DBParameterGroupFamily": "aurora-postgresql16"}]
            }
        if operation == "describe-db-cluster-parameters":
            return {"Parameters": []}
        raise AssertionError(f"unexpected read {arguments}")

    def run(self, arguments: Sequence[str], **_keywords: Any) -> str:
        self.calls.append(tuple(arguments))
        return ""


def _group(runner: GroupRunner) -> str:
    return aurora.ensure_cluster_parameter_group(
        runner,
        aws_region="us-east-1",
        cluster_id="aurora-a",
        engine_version="16.4",
        safe_name=lambda value, maximum: value[:maximum],
        site_id=SITE,
    )


def test_an_existing_untagged_parameter_group_receives_the_site_tag() -> None:
    runner = GroupRunner(exists=True, tags=[])
    _group(runner)
    tag_calls = [call for call in runner.calls if call[2] == "add-tags-to-resource"]
    assert len(tag_calls) == 1, "tagged exactly once"
    assert tag_calls[0][tag_calls[0].index("--resource-name") + 1].endswith(
        ":cluster-pg:aurora-a-pg"
    ), "tagged by its ARN"
    assert f"Key=gpu-fault:site-id,Value={SITE}" in tag_calls[0], "with the site tag"

    owned = GroupRunner(exists=True, tags=[{"Key": "gpu-fault:site-id", "Value": SITE}])
    _group(owned)
    assert not [call for call in owned.calls if call[2] == "add-tags-to-resource"], (
        "an owned group is left alone"
    )


def test_a_new_parameter_group_is_created_with_the_site_tag() -> None:
    runner = GroupRunner(exists=False, tags=[])
    _group(runner)
    creates = [
        call for call in runner.calls if call[2] == "create-db-cluster-parameter-group"
    ]
    assert len(creates) == 1, "created once"
    assert creates[0][creates[0].index("--tags") + 1] == (
        f"Key=gpu-fault:site-id,Value={SITE}"
    ), "born with the site tag"
