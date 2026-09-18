from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.e2e.regional import notify008_resources as resources
from tests.regional._cov95_notify008_resource_extra_support import (
    cpu_node_fixture as cpu_node_fixture,
)
from tests.regional._cov95_notify008_resource_extra_support import (
    resource_isolation as resource_isolation,
)
from tests.regional._cov95_notify008_resource_extra_support import (
    resource_target_fixture as resource_target_fixture,
)


@pytest.mark.parametrize("value", ["1", "250m", "0.25", "1e2", "0.001m"])
def test_positive_cpu_accepts_finite_nonzero_quantities(value):
    assert resources.positive_cpu(value) is True, value


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        0,
        True,
        [],
        {},
        "",
        "0",
        "0m",
        "-1",
        "-1m",
        "NaN",
        "Infinity",
        "-Infinity",
        "many",
        "2mm",
    ],
)
def test_positive_cpu_refuses_unknown_nonfinite_and_nonpositive_quantities(value):
    assert resources.positive_cpu(value) is False, value


@pytest.mark.parametrize(
    "instance",
    [
        "c5.metal",
        "c7i.4xlarge",
        "c8g.large",
        "m6i.large",
        "r7a.8xlarge",
        "t3.small",
        "t4g.medium",
        "ml.c5.4xlarge",
        "ml.c7i.4xlarge",
        "ml.m6i.large",
        "ml.r7a.8xlarge",
    ],
)
def test_matching_ready_cpu_instance_is_accepted_without_mutating_node_data(
    resource_target, cpu_node, instance
):
    cpu_node["metadata"]["labels"]["node.kubernetes.io/instance-type"] = instance
    before = deepcopy(cpu_node)
    assert resources.cpu_node_errors(cpu_node, resource_target) == [], cpu_node
    assert cpu_node == before, "CPU admission checks must remain read-only"


@pytest.mark.parametrize(
    "instance",
    [
        "p5.48xlarge",
        "g6.2xlarge",
        "inf2.xlarge",
        "trn1.32xlarge",
        "ml.p5.48xlarge",
        "ml.g6.2xlarge",
        "ml.inf2.xlarge",
        "ml.trn1.32xlarge",
        "ml.ml.c5.4xlarge",
        "mac2.metal",
        "c9i.large",
        "unknown",
        "",
        None,
        4,
    ],
)
def test_accelerator_or_unknown_instance_class_is_not_cpu_proof(
    resource_target, cpu_node, instance
):
    cpu_node["metadata"]["labels"]["node.kubernetes.io/instance-type"] = instance
    assert resources.cpu_node_errors(cpu_node, resource_target), (
        "CPU placement must refuse accelerator or unrecognized instance classes",
        instance,
    )


@pytest.mark.parametrize(
    "defect",
    [
        "name",
        "uid",
        "deleted",
        "os",
        "capacity",
        "allocatable",
        "not-ready",
        "no-ready-condition",
    ],
)
def test_cpu_node_requires_all_identity_capacity_and_readiness_evidence(
    resource_target, cpu_node, defect
):
    if defect in {"name", "uid"}:
        cpu_node["metadata"][defect] = "different"
    elif defect == "deleted":
        cpu_node["metadata"]["deletionTimestamp"] = "2026-09-12T00:00:00Z"
    elif defect == "os":
        cpu_node["metadata"]["labels"]["kubernetes.io/os"] = "windows"
    elif defect in {"capacity", "allocatable"}:
        cpu_node["status"][defect]["cpu"] = "0"
    elif defect == "not-ready":
        cpu_node["status"]["conditions"][0]["status"] = "False"
    else:
        cpu_node["status"]["conditions"] = [{"type": "DiskPressure", "status": "True"}]
    assert resources.cpu_node_errors(cpu_node, resource_target), (defect, cpu_node)


@pytest.mark.parametrize("section", ["capacity", "allocatable"])
@pytest.mark.parametrize("instance", ["c5.4xlarge", "ml.c5.4xlarge"])
@pytest.mark.parametrize(
    "key",
    [
        "nvidia.com/gpu",
        "nvidia.com/mig-1g.5gb",
        "amd.com/gpu",
        "habana.ai/gaudi",
        "example.com/GPU",
    ],
)
def test_any_reported_accelerator_blocks_cpu_admission(
    resource_target, cpu_node, section, instance, key
):
    cpu_node["metadata"]["labels"]["node.kubernetes.io/instance-type"] = instance
    cpu_node["status"][section][key] = "1"
    assert resources.cpu_node_errors(cpu_node, resource_target), (
        "accelerator inventory cannot be ignored because the instance label looks CPU-only",
        section,
        key,
    )


def test_explicit_zero_accelerator_quantities_are_accepted(resource_target, cpu_node):
    for section in ("capacity", "allocatable"):
        cpu_node["status"][section].update({"nvidia.com/gpu": "0", "amd.com/gpu": "0"})
    assert resources.cpu_node_errors(cpu_node, resource_target) == [], cpu_node


@pytest.mark.parametrize("key", ["nvidia.com/gpu", "habana.ai/gaudi"])
def test_zero_allocatable_cannot_hide_physical_accelerator_capacity(
    resource_target, cpu_node, key
):
    cpu_node["status"]["capacity"][key] = "8"
    cpu_node["status"]["allocatable"][key] = "0"
    assert resources.cpu_node_errors(cpu_node, resource_target), (
        "an accelerator node remains non-CPU even when no accelerator is allocatable",
        cpu_node["status"],
    )
