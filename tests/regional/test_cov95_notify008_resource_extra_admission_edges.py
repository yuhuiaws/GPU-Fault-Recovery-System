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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("key", "node.kubernetes.io/accelerator"),
        ("operator", "Equal"),
        ("effect", "NoSchedule"),
        ("tolerationSeconds", -1),
        ("tolerationSeconds", 301),
        ("tolerationSeconds", True),
        ("tolerationSeconds", 300.0),
        ("tolerationSeconds", "300"),
        ("extra", "unapproved"),
        ("missing", None),
        ("scalar", "not-a-toleration"),
    ],
)
def test_admission_rejects_unbounded_or_unapproved_tolerations(
    resource_target, field, value
):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=True)
    toleration = {
        "key": "node.kubernetes.io/not-ready",
        "operator": "Exists",
        "effect": "NoExecute",
        "tolerationSeconds": 300,
    }
    if field == "missing":
        toleration.pop("tolerationSeconds")
    elif field != "scalar":
        toleration[field] = value
    pod["spec"]["tolerations"] = [value if field == "scalar" else toleration]
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=True
    )
    assert "admitted Pod tolerations differ" in errors, (field, value, errors)


@pytest.mark.parametrize(
    "inventory", [None, [], {}, {"runtime": {}}, [{}], [{}, {}, {}]]
)
def test_bad_container_inventory_preserves_prior_identity_refusal(
    resource_target, inventory
):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=True)
    pod["metadata"]["uid"] = ""
    pod["spec"]["containers"] = inventory
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=True
    )
    assert "admitted Pod ownership differs" in errors, errors
    assert "admitted Pod container inventory differs" in errors, (inventory, errors)


@pytest.mark.parametrize("container", [None, "unexpected"])
def test_nonobject_container_is_rejected_without_reading_its_fields(
    resource_target, container
):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=True)
    pod["spec"]["containers"][0] = container
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=True
    )
    assert "admitted container contains unknown fields" in errors, errors


@pytest.mark.parametrize(
    "field", ["lifecycle", "startupProbe", "ports", "volumeDevices"]
)
def test_admission_rejects_injected_container_capabilities(resource_target, field):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=True)
    pod["spec"]["containers"][0][field] = {}
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=True
    )
    assert "admitted container contains unknown fields" in errors, (field, errors)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image", "registry.invalid/other:latest"),
        ("command", ["/bin/sh"]),
        ("args", ["unapproved"]),
        ("env", []),
        ("volumeMounts", []),
        ("securityContext", {"privileged": True}),
        ("resources", {"limits": {"nvidia.com/gpu": "1"}}),
    ],
)
def test_container_content_cannot_drift_from_the_approved_spec(
    resource_target, field, value
):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=True)
    pod["spec"]["containers"][0][field] = value
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=True
    )
    assert f"admitted runtime differs at {field}" in errors, (field, errors)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("terminationMessagePath", "/case/other-log"),
        ("terminationMessagePolicy", "FallbackToLogsOnError"),
    ],
)
def test_container_termination_fields_accept_no_nondefault_override(
    resource_target, field, value
):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=True)
    pod["spec"]["containers"][1][field] = value
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=True
    )
    assert "admitted container termination fields differ" in errors, errors


def test_pinned_pullable_image_ids_and_omitted_released_gate_are_accepted(
    resource_target,
):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=False)
    pod["spec"].pop("schedulingGates")
    for status in pod["status"]["containerStatuses"]:
        status["imageID"] = "docker-pullable://" + status["imageID"]
    before = deepcopy(pod)
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=False
    )
    assert errors == [], errors
    assert pod == before, "image normalization must not rewrite the observed Pod"


@pytest.mark.parametrize("history", [True, False, 0.0, "0", None, -1])
def test_restart_history_requires_an_integer_zero(resource_target, history):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=False)
    pod["status"]["containerStatuses"][1]["restartCount"] = history
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=False
    )
    assert "admitted container readiness or restart history differs" in errors, (
        history,
        errors,
    )


@pytest.mark.parametrize(
    "shape", ["absent", "one", "duplicate", "mapping", "scalar-entry"]
)
def test_malformed_status_inventory_returns_a_refusal_instead_of_a_runtime_error(
    resource_target, shape
):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=False)
    states = pod["status"]["containerStatuses"]
    if shape == "absent":
        states = []
    elif shape == "one":
        states = states[:1]
    elif shape == "duplicate":
        states = [states[0], deepcopy(states[0])]
    elif shape == "mapping":
        states = {"runtime": states[0], "database": states[1]}
    else:
        states = [None, states[1]]
    pod["status"]["containerStatuses"] = states
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=False
    )
    assert "admitted container readiness or restart history differs" in errors, (
        shape,
        errors,
    )


def test_null_running_state_is_not_evidence_of_a_running_container(resource_target):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=False)
    pod["status"]["containerStatuses"][0]["state"] = {"running": None}
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=False
    )
    assert "admitted container readiness or restart history differs" in errors, (
        "a null running observation cannot authorize the active phase",
        errors,
    )


def test_gated_pod_cannot_bypass_the_gate_even_on_the_approved_cpu_node(
    resource_target,
):
    expected = resources.pod_spec(resource_target)
    pod = admitted_pod(expected, resource_target, gated=True)
    pod["spec"]["nodeName"] = resource_target.node
    errors = resources.admitted_pod_errors(
        pod, expected, resource_target, JOB_UID, gated=True
    )
    assert errors, "the admission gate must still be effective before inert startup"
