"""Bootstrap-owned SNS permission for one verified AMP workspace."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
    assert_site_tag,
    describe_or_absent,
    safe_name,
    tag_map,
)
from gpu_fault_release import repository_root

AMP_SNS_POLICY_ASSET = "deploy/observability/amp-sns-publish-policy.json"
SNS_TOPIC_GENERATION_TAG = "gpu-fault:topic-generation"


def site_sns_topic_arn(cpu: ClusterIdentity, site_id: str) -> str:
    """Bind parallel IAM preparation and topic creation to the same site ARN."""
    topic_name = safe_name(f"gpu-fault-{site_id}-alerts", maximum=256)
    return f"arn:aws:sns:{cpu.region}:{cpu.account_id}:{topic_name}"


def ensure_sns_topic(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    site_id: str,
) -> tuple[str, bool, str]:
    expected_topic_arn = site_sns_topic_arn(cpu, site_id)
    topic_name = expected_topic_arn.rsplit(":", 1)[1]
    existing = describe_or_absent(
        runner,
        cpu.region,
        "sns",
        "get-topic-attributes",
        "--topic-arn",
        expected_topic_arn,
        not_found=("NotFound",),
    )
    if existing is not None:
        topic_arn = expected_topic_arn
        validate_sns_attributes(existing, cpu=cpu, topic_arn=topic_arn)
        tags = runner.aws_json(
            cpu.region,
            "sns",
            "list-tags-for-resource",
            "--resource-arn",
            topic_arn,
        ).get("Tags")
        if not isinstance(tags, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("Key"), str)
            or not isinstance(item.get("Value"), str)
            for item in tags
        ):
            raise BootstrapError("SNS topic tags are unavailable")
        tags_by_key = tag_map(tags)
        if len(tags_by_key) != len(tags):
            raise BootstrapError("SNS topic tags are ambiguous")
        tagged = assert_site_tag(
            tags,
            site_id=site_id,
            description=f"SNS topic {topic_arn}",
            allow_missing=True,
        )
        generation = str(tags_by_key.get(SNS_TOPIC_GENERATION_TAG) or "")
        if generation and not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise BootstrapError(f"SNS topic {topic_arn} has an invalid generation tag")
        tag_values = []
        if not tagged:
            tag_values.append(f"Key={SITE_TAG_KEY},Value={site_id}")
        if not generation:
            generation = uuid4().hex
            tag_values.append(f"Key={SNS_TOPIC_GENERATION_TAG},Value={generation}")
        if tag_values:
            runner.run(
                [
                    "aws",
                    "sns",
                    "tag-resource",
                    "--region",
                    cpu.region,
                    "--resource-arn",
                    topic_arn,
                    "--tags",
                    *tag_values,
                ],
                mutate=True,
                capture=False,
            )
    else:
        generation = uuid4().hex
        topic_arn = runner.aws_text(
            cpu.region,
            "sns",
            "create-topic",
            "--name",
            topic_name,
            "--tags",
            f"Key=gpu-fault:site-id,Value={site_id}",
            f"Key={SNS_TOPIC_GENERATION_TAG},Value={generation}",
            "--query",
            "TopicArn",
            mutate=True,
        )
        if topic_arn != expected_topic_arn:
            raise BootstrapError("created SNS topic identity differs from the site")
    return topic_arn, False, generation


def amp_workspace_arn(cpu: ClusterIdentity, workspace_id: str) -> str:
    if (
        re.fullmatch(r"[0-9]{12}", cpu.account_id) is None
        or re.fullmatch(r"[a-z]{2}(?:-[a-z]+)+-[0-9]+", cpu.region) is None
        or re.fullmatch(r"ws-[A-Za-z0-9-]+", workspace_id) is None
    ):
        raise BootstrapError("AMP workspace binding is invalid")
    return f"arn:aws:aps:{cpu.region}:{cpu.account_id}:workspace/{workspace_id}"


def validate_amp_workspace(
    value: object, *, cpu: ClusterIdentity, site_id: str, workspace_id: str
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("workspaceId") != workspace_id
        or value.get("arn") != amp_workspace_arn(cpu, workspace_id)
        or value.get("alias") != safe_name(f"gpu-fault-{site_id}", maximum=100)
    ):
        raise BootstrapError("AMP workspace identity differs from the site")
    return cast(dict[str, Any], value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BootstrapError("monitoring policy contains a duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(_value: str) -> object:
    raise BootstrapError("monitoring policy contains a non-JSON constant")


def _json_document(raw: object) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw or len(raw.encode("utf-8")) > 256 * 1024:
        raise BootstrapError("monitoring policy is unavailable or oversized")
    try:
        value = json.loads(
            raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        )
    except (ValueError, RecursionError):
        raise BootstrapError("monitoring policy is not valid JSON") from None
    if not isinstance(value, dict):
        raise BootstrapError("monitoring policy must be a JSON object")
    return cast(dict[str, Any], value)


def _topic_policy(raw: object) -> dict[str, Any]:
    value = _json_document(raw)
    statements = value.get("Statement")
    version = value.get("Version", "2008-10-17")
    if (
        not isinstance(version, str)
        or version not in {"2008-10-17", "2012-10-17"}
        or not isinstance(statements, list)
        or not all(isinstance(statement, dict) for statement in statements)
    ):
        raise BootstrapError("SNS topic policy has an invalid document shape")
    for statement in statements:
        effect = statement.get("Effect")
        action = statement.get("Action", statement.get("NotAction"))
        if (
            not isinstance(effect, str)
            or effect not in {"Allow", "Deny"}
            or ("Action" in statement) == ("NotAction" in statement)
            or not (
                isinstance(action, str)
                and bool(action)
                or isinstance(action, list)
                and bool(action)
                and all(isinstance(item, str) and item for item in action)
            )
        ):
            raise BootstrapError("SNS topic policy contains an invalid statement")
    return value


def validate_sns_attributes(
    value: object, *, cpu: ClusterIdentity, topic_arn: str
) -> dict[str, Any]:
    attributes = value.get("Attributes") if isinstance(value, dict) else None
    if (
        not isinstance(attributes, dict)
        or attributes.get("TopicArn") != topic_arn
        or attributes.get("Owner") != cpu.account_id
    ):
        raise BootstrapError("SNS topic identity differs from the site")
    _topic_policy(attributes.get("Policy"))
    return cast(dict[str, Any], attributes)


def _owned_tags(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    site_id: str,
    service: str,
    arn: str,
    generation: str | None = None,
) -> None:
    document = runner.aws_json(
        cpu.region, service, "list-tags-for-resource", "--resource-arn", arn
    )
    if service == "amp":
        tags = document.get("tags")
        if not isinstance(tags, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in tags.items()
        ):
            raise BootstrapError("AMP workspace tags are unavailable")
    else:
        values = document.get("Tags")
        if not isinstance(values, list):
            raise BootstrapError("SNS topic tags are unavailable")
        tags = {}
        for item in values:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("Key"), str)
                or not isinstance(item.get("Value"), str)
                or item["Key"] in tags
            ):
                raise BootstrapError("SNS topic tags are invalid or ambiguous")
            tags[item["Key"]] = item["Value"]
        if tags.get(SNS_TOPIC_GENERATION_TAG) != generation:
            raise BootstrapError(
                "SNS topic generation changed during policy convergence"
            )
    assert_site_tag(tags, site_id=site_id, description="monitoring resource")


def amp_sns_publish_statement(
    *,
    cpu: ClusterIdentity,
    site_id: str,
    workspace_id: str,
    topic_arn: str,
    root: Path | None = None,
) -> dict[str, Any]:
    expected = site_sns_topic_arn(cpu, site_id)
    if topic_arn != expected:
        raise BootstrapError("SNS topic ARN differs from the site")
    workspace_arn = amp_workspace_arn(cpu, workspace_id)
    try:
        statement = _json_document(
            ((root or repository_root()) / AMP_SNS_POLICY_ASSET).read_text("utf-8")
        )
    except OSError:
        raise BootstrapError("AMP SNS publish statement asset is unavailable") from None
    condition = statement.get("Condition")
    if (
        set(statement)
        != {"Sid", "Effect", "Principal", "Action", "Resource", "Condition"}
        or statement["Sid"] != "AllowAmpAlertmanagerPublish"
        or statement["Effect"] != "Allow"
        or statement["Principal"] != {"Service": "aps.amazonaws.com"}
        or statement["Action"] != "sns:Publish"
        or statement["Resource"] != "REPLACE_WITH_SNS_TOPIC_ARN"
        or condition
        != {
            "StringEquals": {"AWS:SourceAccount": "REPLACE_WITH_AWS_ACCOUNT_ID"},
            "ArnEquals": {"AWS:SourceArn": "REPLACE_WITH_AMP_WORKSPACE_ARN"},
        }
    ):
        raise BootstrapError("AMP SNS publish statement asset has an invalid shape")
    statement["Resource"] = topic_arn
    statement["Condition"]["StringEquals"]["AWS:SourceAccount"] = cpu.account_id
    statement["Condition"]["ArnEquals"]["AWS:SourceArn"] = workspace_arn
    return statement


def _canonical_policy(value: dict[str, Any]) -> str:
    return json.dumps(
        {
            **value,
            "Statement": sorted(
                value["Statement"],
                key=lambda statement: json.dumps(statement, sort_keys=True),
            ),
        },
        sort_keys=True,
    )


def ensure_amp_sns_publish_policy(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    site_id: str,
    workspace_id: str,
    topic_arn: str,
    topic_generation: str,
) -> None:
    statement = amp_sns_publish_statement(
        cpu=cpu, site_id=site_id, workspace_id=workspace_id, topic_arn=topic_arn
    )
    if re.fullmatch(r"[0-9a-f]{32}", topic_generation) is None:
        raise BootstrapError("SNS topic generation is invalid")
    workspace = validate_amp_workspace(
        runner.aws_json(
            cpu.region, "amp", "describe-workspace", "--workspace-id", workspace_id
        ).get("workspace"),
        cpu=cpu,
        site_id=site_id,
        workspace_id=workspace_id,
    )
    if workspace.get("status") != {"statusCode": "ACTIVE"}:
        raise BootstrapError("AMP workspace is not ACTIVE for SNS publication")
    _owned_tags(
        runner,
        cpu=cpu,
        site_id=site_id,
        service="amp",
        arn=amp_workspace_arn(cpu, workspace_id),
    )

    def read_policy() -> dict[str, Any]:
        attributes = validate_sns_attributes(
            runner.aws_json(
                cpu.region, "sns", "get-topic-attributes", "--topic-arn", topic_arn
            ),
            cpu=cpu,
            topic_arn=topic_arn,
        )
        _owned_tags(
            runner,
            cpu=cpu,
            site_id=site_id,
            service="sns",
            arn=topic_arn,
            generation=topic_generation,
        )
        return _topic_policy(attributes["Policy"])

    current = read_policy()
    desired = {
        **current,
        "Statement": [
            item for item in current["Statement"] if item.get("Sid") != statement["Sid"]
        ]
        + [statement],
    }
    if _canonical_policy(current) == _canonical_policy(desired):
        return
    if _canonical_policy(read_policy()) != _canonical_policy(current):
        raise BootstrapError("SNS topic policy changed during convergence")
    runner.run(
        [
            "aws",
            "sns",
            "set-topic-attributes",
            "--region",
            cpu.region,
            "--topic-arn",
            topic_arn,
            "--attribute-name",
            "Policy",
            "--attribute-value",
            json.dumps(desired, separators=(",", ":")),
        ],
        mutate=True,
        capture=False,
        sensitive=True,
    )
    if _canonical_policy(read_policy()) != _canonical_policy(desired):
        raise BootstrapError("SNS AMP publish policy did not converge")
