from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.e2e.regional import notify008_resources as resources
from tests.regional._cov95_notify008_resource_extra_support import JOB_UID, admitted_pod
from tests.regional._cov95_notify008_resource_extra_support import (
    resource_isolation as resource_isolation,
)
from tests.regional._cov95_notify008_resource_extra_support import (
    resource_target_fixture as resource_target_fixture,
)


@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("defaults", [False, True])
def test_expected_admitted_pod_accepts_only_documented_server_defaults(
    resource_target, gated, defaults
):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=gated)
    if defaults:
        pod["spec"].update(
            dnsPolicy="ClusterFirst",
            schedulerName="default-scheduler",
            serviceAccount="notify008",
            tolerations=[
                {
                    "key": key,
                    "operator": "Exists",
                    "effect": "NoExecute",
                    "tolerationSeconds": seconds,
                }
                for key, seconds in (
                    ("node.kubernetes.io/not-ready", 0),
                    ("node.kubernetes.io/unreachable", 300),
                )
            ],
        )
        for container in pod["spec"]["containers"]:
            container.update(
                terminationMessagePath="/dev/termination-log",
                terminationMessagePolicy="File",
            )
        for name in (
            "hostNetwork",
            "hostPID",
            "hostIPC",
            "shareProcessNamespace",
            "priority",
        ):
            pod["spec"].pop(name)
    before = deepcopy(pod)
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=gated
    )
    assert errors == [], errors
    assert pod == before, "validation must not normalize or mutate the observed Pod"


@pytest.mark.parametrize(
    "defect",
    [
        "namespace",
        "uid-empty",
        "uid-type",
        "deleting",
        "run-label",
        "case-label",
        "no-owner",
        "two-owners",
        "kind",
        "owner-uid",
        "controller",
    ],
)
def test_admission_refuses_foreign_or_unproven_pod_ownership(resource_target, defect):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=True)
    meta = pod["metadata"]
    if defect == "namespace":
        meta["namespace"] = "another-run"
    elif defect.startswith("uid"):
        meta["uid"] = "" if defect == "uid-empty" else 7
    elif defect == "deleting":
        meta["deletionTimestamp"] = "2026-09-12T00:00:00Z"
    elif defect in {"run-label", "case-label"}:
        key = (
            "gpu-fault.io/acceptance-run"
            if defect == "run-label"
            else "gpu-fault.io/acceptance-case"
        )
        meta["labels"][key] = "another-owner"
    elif defect == "no-owner":
        meta["ownerReferences"] = []
    elif defect == "two-owners":
        meta["ownerReferences"] *= 2
    else:
        key = {"kind": "kind", "owner-uid": "uid", "controller": "controller"}[defect]
        meta["ownerReferences"][0][key] = False if defect == "controller" else "foreign"
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=True
    )
    assert "admitted Pod ownership differs" in errors, (defect, errors)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hostNetwork", True),
        ("automountServiceAccountToken", True),
        ("serviceAccountName", "other-account"),
        ("activeDeadlineSeconds", 9999),
        ("terminationGracePeriodSeconds", 60),
        ("volumes", [{"name": "host", "hostPath": {"path": "/"}}]),
        ("priorityClassName", ""),
        ("priorityClassName", "mount-s3-critical"),
        ("priorityClassName", "notify008-fedcba9876543210"),
        ("priority", 1000000000),
        ("priority", True),
        ("preemptionPolicy", "PreemptLowerPriority"),
    ],
)
def test_admission_refuses_changed_security_lifetime_and_storage(
    resource_target, field, value
):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=True)
    pod["spec"][field] = value
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=True
    )
    assert f"admitted Pod differs at {field}" in errors, (field, errors)


@pytest.mark.parametrize(
    "field", ["initContainers", "ephemeralContainers", "hostAliases"]
)
def test_admission_refuses_unexpected_pod_spec_fields(resource_target, field):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=True)
    pod["spec"][field] = []
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=True
    )
    assert "admitted Pod contains unknown fields" in errors, errors


@pytest.mark.parametrize("field", ["dnsPolicy", "schedulerName", "serviceAccount"])
def test_only_exact_default_values_are_allowed(resource_target, field):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=True)
    pod["spec"][field] = "foreign"
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=True
    )
    assert f"admitted Pod default differs at {field}" in errors, errors


@pytest.mark.parametrize("gated", [True, False])
def test_scheduling_gate_must_match_the_current_phase(resource_target, gated):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=gated)
    pod["spec"]["schedulingGates"] = [] if gated else expected["schedulingGates"]
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=gated
    )
    assert "admitted Pod scheduling gate differs" in errors, errors


def test_gated_admission_cannot_accept_a_pod_already_bound_to_another_node(
    resource_target,
):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=True)
    pod["spec"]["nodeName"] = "unapproved-accelerator-node"
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=True
    )
    assert errors, (
        "a nodeName binding bypasses scheduler affinity and must be checked before ARM"
    )


@pytest.mark.parametrize(
    "defect", ["node", "phase", "ready", "restarts", "not-running", "image"]
)
def test_running_pod_requires_approved_placement_health_and_resolved_images(
    resource_target, defect
):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=False)
    status = pod["status"]
    if defect == "node":
        pod["spec"]["nodeName"] = "other-node"
    elif defect == "phase":
        status["phase"] = "Pending"
    elif defect == "ready":
        status["containerStatuses"][0]["ready"] = False
    elif defect == "restarts":
        status["containerStatuses"][0]["restartCount"] = 1
    elif defect == "not-running":
        status["containerStatuses"][0]["state"] = {"terminated": {"exitCode": 0}}
    else:
        status["containerStatuses"][0]["imageID"] = "registry.invalid/unapproved:latest"
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=False
    )
    assert errors, ("an unproved running Pod must not be armed", defect)
