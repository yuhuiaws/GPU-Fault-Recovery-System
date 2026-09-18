from __future__ import annotations

from dataclasses import replace

import pytest

from gpu_fault.admin import monitoring_policy as policy
from gpu_fault.admin.bootstrap_common import SITE_TAG_KEY, BootstrapError
from tests.admin._cov95_join_support import target
from tests.admin.test_monitoring_policy import GENERATION, MonitoringAWS


@pytest.fixture
def context():
    cpu = replace(target(), role="cpu")
    runner = MonitoringAWS(cpu)
    return cpu, runner


def converge(context, *, generation=GENERATION):
    cpu, runner = context
    policy.ensure_amp_sns_publish_policy(
        runner,
        cpu=cpu,
        site_id=runner.site_id,
        workspace_id=runner.workspace["workspaceId"],
        topic_arn=runner.topic_arn,
        topic_generation=generation,
    )


@pytest.mark.parametrize(
    "tags",
    [
        [],
        [{"Key": SITE_TAG_KEY, "Value": "policy-test"}],
        [{"Key": policy.SNS_TOPIC_GENERATION_TAG, "Value": GENERATION}],
    ],
)
def test_topic_adoption_records_each_missing_site_or_generation_tag(
    context, monkeypatch, tags
):
    cpu, runner = context
    runner.sns_tags = tags
    run = runner.run

    def tag(arguments, **options):
        if arguments[:3] == ["aws", "sns", "tag-resource"]:
            runner.calls.append(("sns", "tag-resource", bool(options.get("mutate"))))
            return ""
        return run(arguments, **options)

    monkeypatch.setattr(runner, "run", tag)
    arn, _reused, generation = policy.ensure_sns_topic(
        runner, cpu=cpu, site_id=runner.site_id
    )
    assert arn == runner.topic_arn
    assert len(generation) == 32
    assert runner.mutations() == [("sns", "tag-resource")]


def test_topic_invalid_generation_is_refused_before_tagging(context):
    cpu, runner = context
    runner.sns_tags[-1]["Value"] = "invalid"
    with pytest.raises(BootstrapError, match="invalid generation tag"):
        policy.ensure_sns_topic(runner, cpu=cpu, site_id=runner.site_id)
    assert runner.mutations() == []


def test_created_topic_must_match_the_exact_site_identity(context, monkeypatch):
    cpu, runner = context
    runner.topic_exists = False
    run = runner.run

    def create(arguments, **options):
        if arguments[:3] == ["aws", "sns", "create-topic"]:
            runner.calls.append(("sns", "create-topic", True))
            return runner.topic_arn + "-other"
        return run(arguments, **options)

    monkeypatch.setattr(runner, "run", create)
    with pytest.raises(BootstrapError, match="created SNS topic identity differs"):
        policy.ensure_sns_topic(runner, cpu=cpu, site_id=runner.site_id)
    assert runner.mutations() == [("sns", "create-topic")]


@pytest.mark.parametrize("tags", [None, [], {"example": None}])
def test_publish_policy_requires_observable_amp_ownership_tags(context, tags):
    runner = context[1]
    runner.amp_tags = tags
    with pytest.raises(BootstrapError, match="AMP workspace tags are unavailable"):
        converge(context)
    assert runner.mutations() == []


@pytest.mark.parametrize(
    "tags",
    [
        None,
        [None],
        [{"Key": "example", "Value": 1}],
        [{"Key": "example", "Value": "a"}, {"Key": "example", "Value": "b"}],
    ],
)
def test_publish_policy_requires_unambiguous_sns_ownership_tags(context, tags):
    runner = context[1]
    runner.sns_tags = tags
    with pytest.raises(BootstrapError, match="SNS topic tags are"):
        converge(context)
    assert runner.mutations() == []


def test_publish_policy_invalid_generation_stops_before_transport(context):
    with pytest.raises(BootstrapError, match="generation is invalid"):
        converge(context, generation="invalid")
    assert context[1].calls == []


def test_publish_policy_requires_an_active_workspace(context):
    context[1].workspace["status"] = {"statusCode": "CREATING"}
    with pytest.raises(BootstrapError, match="not ACTIVE"):
        converge(context)
    assert context[1].mutations() == []


def test_publish_statement_asset_must_be_available_before_policy_writes(
    context, tmp_path
):
    cpu, runner = context
    with pytest.raises(BootstrapError, match="asset is unavailable"):
        policy.amp_sns_publish_statement(
            cpu=cpu,
            site_id=runner.site_id,
            workspace_id=runner.workspace["workspaceId"],
            topic_arn=runner.topic_arn,
            root=tmp_path,
        )
    assert runner.calls == []
