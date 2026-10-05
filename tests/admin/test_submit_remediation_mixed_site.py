"""``submit-remediation`` names the target cluster's Runtime Profile version.

Live 2026-10-05: on a mixed-provider site a validated restore submitted for
the eks-managed cluster (pinned to ``eks-v1``) was stamped with the site
profile, ``hyperpod-v1``. The control plane compiles step owners, escalation
rungs and provider correlation against a workflow's ``runtime_profile_version``,
so an operator submission for a cluster must carry that cluster's own version
-- the override the site declares for it -- and the site's only for a cluster
without one.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import submit_remediation as module
from gpu_fault.admin.site_cluster_profile import cluster_runtime_profile_version
from gpu_fault.admin.submit_remediation import submit_restore as real_submit_restore
from tests.admin.test_admin_submit_remediation import (
    INCIDENT,
    QUARANTINED,
    Harness,
    RestoreHarness,
    _quarantined_inspection,
    _quarantined_nodes,
)

EKS_CLUSTER = "gpu-eks-cluster"


def _mixed_site(site: SimpleNamespace) -> SimpleNamespace:
    site.release_config["clusters"] = [
        {"cluster_id": "gpu-a", "context": "gpu-a-context"},
        {
            "cluster_id": EKS_CLUSTER,
            "context": "gpu-eks-context",
            "provider": "eks-managed",
            "runtime_profile": {
                "source": "/state/profiles/eks-v1.yaml",
                "template_source": "config/runtime-profile.regional-eks-safe.example.yaml",
                "version": "eks-v1",
                "agent_config_digest": "e" * 64,
            },
        },
    ]
    return site


def test_the_cluster_version_is_the_override_else_the_site_profile() -> None:
    config = _mixed_site(
        SimpleNamespace(release_config={"runtime_profile": {"version": "hyperpod-v1"}})
    ).release_config
    assert cluster_runtime_profile_version(config, EKS_CLUSTER) == "eks-v1"
    assert cluster_runtime_profile_version(config, "gpu-a") == "hyperpod-v1", (
        "a cluster without an override runs the site profile"
    )
    assert cluster_runtime_profile_version(config, "unknown") == "hyperpod-v1", (
        "an unknown cluster falls back to the site profile rather than failing"
    )
    config["clusters"][1]["runtime_profile"] = {"version": ""}
    assert cluster_runtime_profile_version(config, EKS_CLUSTER) == "hyperpod-v1"


@pytest.mark.parametrize(
    ("cluster_id", "expected_version"),
    [(EKS_CLUSTER, "eks-v1"), ("gpu-a", "hyperpod-v1")],
)
def test_hardware_dispositions_stamp_the_target_clusters_profile_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cluster_id: str,
    expected_version: str,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    _mixed_site(harness.site)
    harness.inspection["incident"]["cluster_id"] = cluster_id
    harness.submit_result = harness.decision()

    result = harness.submit("reset-gpu")

    assert result.plan is not None and result.plan.cluster_id == cluster_id
    assert result.plan.terminal is not None
    assert result.plan.terminal["runtime_profile_version"] == expected_version, (
        "the terminal attempt the control plane compiles against names the "
        "cluster's own profile"
    )
    (submission,) = harness.submissions
    assert submission["terminal"]["runtime_profile_version"] == expected_version


def test_the_restore_payload_names_the_clusters_profile_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = RestoreHarness(tmp_path, monkeypatch)
    _mixed_site(harness.site)
    inspection = _quarantined_inspection()
    inspection["incident"]["cluster_id"] = EKS_CLUSTER
    plan = module.build_restore_plan(
        inspection,
        incident_id=QUARANTINED,
        operator="arn:aws:sts::123456789012:assumed-role/ops/alice",
        reference="CHG-R1",
        live_nodes=_quarantined_nodes(),
        now=harness.now(),
    )
    assert plan.cluster_id == EKS_CLUSTER
    payloads: list[dict] = []

    def command(site, payload, *, script):
        payloads.append(json.loads(json.dumps(payload)))
        return {"no_op": False, "workflow": {"request_id": "wf-r"}, "incident": {}}

    monkeypatch.setattr(module, "run_control_plane_script", command)

    real_submit_restore(harness.site, plan, operator="ops@example", reference="CHG-R1")

    (payload,) = payloads
    assert payload["mode"] == "restore"
    assert payload["runtime_profile_version"] == "eks-v1", (
        "the validated restore is compiled and escalated against eks-v1"
    )
    # The same submission for the HyperPod cluster keeps the site profile.
    hyperpod = module.build_restore_plan(
        _quarantined_inspection(),
        incident_id=QUARANTINED,
        operator="arn:aws:sts::123456789012:assumed-role/ops/alice",
        reference="CHG-R1",
        live_nodes=_quarantined_nodes(),
        now=harness.now(),
    )
    real_submit_restore(harness.site, hyperpod, operator="ops@example", reference="CHG")
    assert payloads[-1]["runtime_profile_version"] == "hyperpod-v1"
    assert INCIDENT != QUARANTINED, "the fixtures name two distinct incidents"
