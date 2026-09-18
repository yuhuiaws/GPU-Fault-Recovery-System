from __future__ import annotations

import json

import pytest

from gpu_fault.admin import aws_cleanup_helpers as shapes
from gpu_fault.admin.bootstrap_common import BootstrapError


@pytest.mark.parametrize("values", [None, "", [1], [""], ["same", "same"]])
def test_cleanup_string_inventory_requires_unique_nonempty_identities(values):
    with pytest.raises(BootstrapError, match="valid|duplicate"):
        shapes.strings({"items": values}, "items")


@pytest.mark.parametrize("items", [[], [{}, {}]])
def test_cleanup_single_resource_cannot_guess_between_missing_or_duplicate_rows(items):
    with pytest.raises(BootstrapError, match="no unique"):
        shapes.single_object({"items": items}, "items")


@pytest.mark.parametrize(
    "tags",
    [
        None,
        {"name": 3},
        [None],
        [{"Key": 1, "Value": "x"}],
        [{"Key": "k", "Value": "a"}, {"Key": "k", "Value": "b"}],
    ],
)
def test_cleanup_ownership_tags_cannot_be_missing_malformed_or_ambiguous(tags):
    with pytest.raises(BootstrapError, match="tags are"):
        shapes.strict_tags(tags)


def test_cleanup_accepts_both_native_tag_spellings_without_losing_identity():
    assert shapes.strict_tags([{"key": "example", "value": "owned"}]) == {
        "example": "owned"
    }


@pytest.mark.parametrize("value", [[], "invalid", "[]"])
def test_sqs_cleanup_policy_must_be_json_object(value):
    with pytest.raises(BootstrapError, match="JSON string|invalid JSON|JSON object"):
        shapes.sqs_policy(value)


def test_sqs_single_statement_shape_is_normalized_without_changing_it():
    statement = {"Effect": "Allow", "Action": "sqs:SendMessage"}
    assert shapes.sqs_policy(json.dumps({"Statement": statement})) == {
        "Statement": [statement]
    }
    assert shapes.sqs_policy(None) == {"Statement": []}


@pytest.mark.parametrize("conditions", [[], {"ArnEquals": []}])
def test_sqs_topic_cleanup_refuses_malformed_condition_shapes(conditions):
    with pytest.raises(BootstrapError, match="malformed conditions"):
        shapes.sqs_topic_statements(
            {"Statement": [{"Condition": conditions}]}, "arn:example:topic"
        )


def test_sqs_topic_cleanup_ignores_unrelated_condition_keys():
    policy = {
        "Statement": [
            {"Condition": {"ArnEquals": {"aws:OtherArn": "arn:example:topic"}}}
        ]
    }
    assert shapes.sqs_topic_statements(policy, "arn:example:topic") == []


@pytest.mark.parametrize("trust", ["invalid", "%7Binvalid", None])
def test_oidc_cleanup_requires_a_readable_role_trust_document(trust):
    with pytest.raises(BootstrapError, match="trust document is unavailable"):
        shapes.oidc_provider_users(
            {"Roles": [{"RoleName": "example", "AssumeRolePolicyDocument": trust}]},
            "arn:example:provider",
        )


def test_oidc_user_inventory_handles_single_statement_and_single_federated_principal():
    provider = "arn:example:provider"
    assert shapes.oidc_provider_users(
        {
            "Roles": [
                {
                    "RoleName": "example",
                    "AssumeRolePolicyDocument": {
                        "Statement": {"Principal": {"Federated": provider}}
                    },
                }
            ]
        },
        provider,
    ) == ["example"]
