"""CloudTrail's standard dryRun flag cannot turn a rehearsal into physical proof."""

from __future__ import annotations

import json
from typing import Any

import pytest

from scripts.e2e.regional import collector_reboot_evidence as evidence
from tests.regional.test_collector_reboot_evidence import cloudtrail_item, inputs


@pytest.mark.parametrize("key", ["dryRun", "DryRun"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_cloudtrail_default_flag_is_typed_and_dry_run_never_proves_reboot(
    key: str, dry_run: bool
) -> None:
    item = cloudtrail_item()
    detail = json.loads(item["CloudTrailEvent"])
    detail["requestParameters"][key] = dry_run
    detail["responseElements"].pop("failed")
    item["CloudTrailEvent"] = json.dumps(detail)
    event = evidence.normalize_provider_event(item)
    assert event["normalization_errors"] == []
    assert event["request_dry_run"] is dry_run
    assert event["response_failed_node_count"] == 0
    data = inputs()
    data["events"] = [event]
    proof = evidence.prove_reboot_scope(**data)
    assert proof["valid"] is (not dry_run)
    if dry_run:
        assert "dry-run" in proof["errors"][0]


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true", [], {}])
def test_non_boolean_dry_run_is_not_a_reboot_receipt(value: Any) -> None:
    item = cloudtrail_item()
    detail = json.loads(item["CloudTrailEvent"])
    detail["requestParameters"]["dryRun"] = value
    item["CloudTrailEvent"] = json.dumps(detail)
    event = evidence.normalize_provider_event(item)
    assert event["normalization_errors"]
    data = inputs()
    data["events"] = [event]
    assert evidence.prove_reboot_scope(**data)["valid"] is False


def test_duplicate_dry_run_aliases_are_not_silently_selected() -> None:
    item = cloudtrail_item()
    detail = json.loads(item["CloudTrailEvent"])
    detail["requestParameters"].update(dryRun=False, DryRun=False)
    item["CloudTrailEvent"] = json.dumps(detail)
    assert evidence.normalize_provider_event(item)["normalization_errors"]


@pytest.mark.parametrize("value", [True, 0, 1, "false"])
def test_final_proof_also_rejects_invalid_pre_normalized_flags(value: Any) -> None:
    data = inputs()
    data["events"][0]["request_dry_run"] = value
    assert evidence.prove_reboot_scope(**data)["valid"] is False
