from __future__ import annotations

import copy

import pytest
import yaml

from gpu_fault.admin.site import (
    RegionalSite,
    RetentionSiteConfig,
    SiteConfigError,
    load_site,
    site_retention,
)
from tests.admin.test_admin_site import site_file


@pytest.fixture
def document(tmp_path):
    return yaml.safe_load(site_file(tmp_path).read_text())


@pytest.mark.parametrize(
    "path,value,message",
    [
        (("spec",), [], "spec must be a mapping"),
        (("metadata", "name"), "REPLACE_SITE", "placeholder"),
        (("metadata", "name"), "invalid name", "unsupported characters"),
        (("apiVersion",), "example/v9", "apiVersion"),
        (("kind",), "OtherSite", "kind must be"),
        (("spec", "awsRegion"), "invalid", "valid AWS Region"),
        (("spec", "clusters"), {}, "clusters must be a list"),
        (("spec", "clusters", 0, "clusterId"), "bad id", "unsupported characters"),
        (("spec", "clusters", 0, "controlPlaneUrl"), "http://example.invalid", "https"),
        (("spec", "clusters", 0, "region"), "us-west-2", "Region does not match"),
        (
            ("spec", "nlb", "certificateArn"),
            "arn:aws:acm:us-west-2:123456789012:certificate/example",
            "Region does not match",
        ),
        (("spec", "nlb", "publicSubnets"), "subnet-example", "must be a list"),
        (("spec", "nlb", "publicSubnets"), ["subnet-one"], "at least 2"),
        (("spec", "release", "agentConfigDigest"), "not-a-digest", "SHA-256"),
        (("spec", "release", "upgradeMaxUnavailable"), True, "must be an integer"),
        (
            ("spec", "runtimeProfile", "version"),
            "bad profile",
            "unsupported characters",
        ),
        (
            ("spec", "dns"),
            {"hostedZoneId": "ZEXAMPLE", "hostname": "localhost"},
            "DNS name",
        ),
        (("spec", "images"), {"runtime": "example/image#tag"}, "whitespace or #"),
        (("spec", "notifications"), {"adminEmail": "invalid"}, "valid email"),
        (
            ("spec", "notifications"),
            {"emailRecipients": "person@example.invalid"},
            "must be a list",
        ),
        (("spec", "notifications"), {"emailRecipients": []}, "at least one valid"),
        (("spec", "notifications"), {"emailRecipients": [None]}, "at least one valid"),
        (("spec", "notifications"), {"emailSubjectPrefix": 17}, "must be a string"),
        (
            ("spec", "notifications"),
            {"emailSubjectPrefix": "first\nsecond"},
            "single line",
        ),
        (("spec", "notifications"), {"emailSubjectPrefix": "x" * 65}, "at most 64"),
        (
            ("spec", "notifications"),
            {"allowEmail": False, "acknowledgeExternalAlertChannel": False},
            "must enable email",
        ),
    ],
)
def test_site_contract_rejects_invalid_fields_at_public_model_boundary(
    document, path, value, message
):
    cursor = document
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(SiteConfigError, match=message):
        RegionalSite.from_value(document)


def test_site_contract_rejects_duplicate_cluster_ids(document):
    document["spec"]["clusters"].append(copy.deepcopy(document["spec"]["clusters"][0]))
    with pytest.raises(SiteConfigError, match="clusterId values must be unique"):
        RegionalSite.from_value(document)


@pytest.mark.parametrize("days,uri", [(0, None), (30, "s3://example-archive/records")])
def test_explicit_retention_policy_does_not_get_replaced_by_defaults(days, uri):
    policy = RetentionSiteConfig(control_record_retention_days=days, archive_s3_uri=uri)
    assert (
        policy.resolved(
            account_id="123456789012", region="us-east-1", site_name="example"
        )
        is policy
    )


def test_unresolved_retention_exports_no_archive_environment():
    assert RetentionSiteConfig().environment() == {}


def test_raw_site_retention_handles_missing_spec_without_cloud_reads():
    assert site_retention({"spec": []}) == RetentionSiteConfig()


@pytest.mark.parametrize("kind", ["missing", "invalid-yaml", "missing-entrypoint"])
def test_site_loading_reports_filesystem_and_parser_failures(tmp_path, kind):
    path = site_file(tmp_path)
    if kind == "missing":
        path.unlink()
    elif kind == "invalid-yaml":
        path.write_text("spec: [invalid")
    else:
        document = yaml.safe_load(path.read_text())
        document["spec"]["repositoryRoot"] = str(tmp_path / "not-a-repository")
        path.write_text(yaml.safe_dump(document))
    with pytest.raises(SiteConfigError, match="invalid site config|rollout entrypoint"):
        load_site(path)
