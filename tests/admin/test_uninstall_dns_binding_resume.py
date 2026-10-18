"""Resuming an uninstall whose saved DNS target binding is gone or differs.

The DNS record deletion is bound to the NLB it currently points at before the
Service is removed, and the binding is journaled. A resume after
``KUBERNETES_VERIFIED`` cannot re-derive that binding (the NLB may already be
gone), so a journal without it, or with a binding for other records, stops
the uninstall instead of guessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import aws_commands
from gpu_fault.admin import uninstall as lifecycle
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.uninstall import uninstall
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceSnapshot,
)
from gpu_fault.installation_resources import InstallationResourceDeletePolicy as Policy
from gpu_fault.installation_resources import InstallationResourceOwnership as Ownership
from tests.admin._aws_cleanup_support import Aws
from tests.admin.test_uninstall_lifecycle import Harness

RECORD_KEY = "aws/dns/record"


def interrupted_dns_uninstall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Harness:
    """Run an uninstall up to the first DNS record deletion and stop there."""
    harness = Harness(tmp_path, monkeypatch)
    harness.site.release_config["dns"] = {
        "hosted_zone_id": "Z123",
        "hostname": "control.example",
    }
    monkeypatch.setattr(lifecycle, "reload_site_for_mutation", lambda site: site)
    nlb_name = harness.site.release_config["nlb"]["name"]
    nlb_arn = (
        "arn:aws:elasticloadbalancing:us-east-1:123456789012:"
        f"loadbalancer/net/{nlb_name}/original"
    )
    resources = [
        item.model_copy(
            update={
                "resource_id": nlb_name,
                "resource_arn": nlb_arn,
                "attributes": {"dns_name": "original.elb.example"},
            }
        )
        if item.resource_key == "aws/nlb"
        else item
        for item in harness.snapshot.resources
    ]
    resources.append(
        InstallationResource(
            site_id="test-site",
            resource_key="aws/dns/zone",
            resource_type="route53_zone",
            resource_id="Z123",
            region="us-east-1",
            account_id="123456789012",
            ownership=Ownership.EXTERNAL,
            delete_policy=Policy.PRESERVE,
        )
    )
    resources.append(
        InstallationResource(
            site_id="test-site",
            resource_key=RECORD_KEY,
            resource_type="route53_record",
            resource_id="control.example",
            region="us-east-1",
            account_id="123456789012",
            ownership=Ownership.CREATED,
            delete_policy=Policy.DELETE,
            dependencies=["aws/dns/zone", "aws/nlb"],
            attributes={"hosted_zone_id": "Z123", "record_type": "CNAME"},
        )
    )
    snapshot = InstallationResourceSnapshot(site_id="test-site", resources=resources)
    harness.snapshot = snapshot.model_copy(update={"source_sha256": snapshot.digest()})
    harness.existing = {item.resource_key for item in resources}
    aws = Aws(
        {
            ("elbv2", "describe-load-balancers"): {
                "LoadBalancers": [
                    {
                        "LoadBalancerName": nlb_name,
                        "LoadBalancerArn": nlb_arn,
                        "DNSName": "original.elb.example",
                    }
                ]
            },
            ("elbv2", "describe-tags"): {
                "TagDescriptions": [
                    {
                        "ResourceArn": nlb_arn,
                        "Tags": [{"Key": "gpu-fault:site-id", "Value": "test-site"}],
                    }
                ]
            },
            ("route53", "list-resource-record-sets"): {
                "ResourceRecordSets": [
                    {
                        "Name": "control.example.",
                        "Type": "CNAME",
                        "TTL": 60,
                        "ResourceRecords": [{"Value": "original.elb.example"}],
                    }
                ]
            },
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    original_delete = harness.delete

    def delete(resource: InstallationResource) -> None:
        if resource.resource_type == "route53_record":
            raise BootstrapError("injected interruption before DNS deletion")
        original_delete(resource)

    monkeypatch.setattr(harness, "delete", delete)
    with pytest.raises(BootstrapError, match="interruption before DNS"):
        uninstall(harness.request(), runner=harness)
    return harness


def rewrite_state(harness: Harness, **changes: Any) -> None:
    path = harness.site.source.parent / "uninstall/state.json"
    state = json.loads(path.read_text())
    for key, value in changes.items():
        if value is None:
            state.pop(key)
        else:
            state[key] = value
    path.write_text(json.dumps(state))


def test_resume_without_the_saved_dns_binding_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = interrupted_dns_uninstall(tmp_path, monkeypatch)
    assert set(harness.state()["dns_bindings"]) == {RECORD_KEY}
    rewrite_state(harness, dns_bindings=None)
    events = list(harness.events)
    with pytest.raises(BootstrapError, match="lacks an NLB target binding"):
        uninstall(harness.request(), runner=harness)
    assert harness.events == events
    assert RECORD_KEY in harness.existing


def test_resume_with_a_binding_for_other_records_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = interrupted_dns_uninstall(tmp_path, monkeypatch)
    binding = harness.state()["dns_bindings"][RECORD_KEY]
    rewrite_state(harness, dns_bindings={"aws/dns/other": binding})
    events = list(harness.events)
    with pytest.raises(BootstrapError, match="binding differs from its registry"):
        uninstall(harness.request(), runner=harness)
    assert harness.events == events
    assert RECORD_KEY in harness.existing
