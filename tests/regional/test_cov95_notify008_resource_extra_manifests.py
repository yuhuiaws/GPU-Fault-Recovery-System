from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from scripts.e2e.regional import notify008_resources as resources
from scripts.e2e.regional.probes.notify008_protocol import ProbeError, digest
from tests.regional._cov95_notify008_resource_extra_support import (
    resource_isolation as resource_isolation,
)
from tests.regional._cov95_notify008_resource_extra_support import (
    resource_target_fixture as resource_target_fixture,
)


@pytest.mark.parametrize("namespaced", [True, False])
def test_resource_metadata_binds_the_run_and_namespace_security(
    resource_target, namespaced
):
    actual = resources.metadata(resource_target, namespaced=namespaced)
    assert actual["labels"]["gpu-fault.io/acceptance-run"] == resource_target.run_id, (
        actual
    )
    assert actual["labels"]["gpu-fault.io/acceptance-case"] == (
        "GF-REGIONAL-NOTIFY-008"
    ), actual
    if namespaced:
        assert actual["name"] == "notify008", actual
        assert actual["namespace"] == resource_target.run_id, actual
    else:
        assert actual["name"] == resource_target.run_id and "namespace" not in actual, (
            actual
        )
        assert actual["labels"]["pod-security.kubernetes.io/enforce"] == "restricted", (
            actual
        )
        assert actual["labels"]["pod-security.kubernetes.io/enforce-version"] == (
            "v1.30"
        ), actual


def test_container_environment_has_no_credentials_and_binds_the_actual_pod_uid(
    resource_target,
):
    entries = resources.environment(resource_target)
    values = {entry["name"]: entry.get("value") for entry in entries}
    assert len(entries) == len(values), "environment variable names must be unique"
    assert values == {
        "HOME": "/work",
        "PYTHONPATH": "/case",
        "PYTHONDONTWRITEBYTECODE": "1",
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "AWS_EC2_METADATA_DISABLED": "true",
        "KUBECONFIG": "/dev/null",
        "GPU_FAULT_ALLOW_HYPERPOD_REPLACE": "false",
        "NOTIFY008_RUN_ID": resource_target.run_id,
        "NOTIFY008_SECONDS": str(resource_target.seconds),
        "NOTIFY008_POD_UID": None,
    }, values
    uid = next(entry for entry in entries if entry["name"] == "NOTIFY008_POD_UID")
    assert uid["valueFrom"] == {
        "fieldRef": {"apiVersion": "v1", "fieldPath": "metadata.uid"}
    }, uid


@pytest.mark.parametrize("seconds", [300, 600])
def test_pod_is_cpu_pinned_bounded_nonprivileged_and_initially_gated(
    resource_target, seconds
):
    target = replace(resource_target, seconds=seconds)
    spec = resources.pod_spec(target)
    for name in (
        "automountServiceAccountToken",
        "enableServiceLinks",
        "hostNetwork",
        "hostPID",
        "hostIPC",
        "shareProcessNamespace",
    ):
        assert spec[name] is False, (name, spec)
    assert spec["serviceAccountName"] == "notify008", spec
    assert spec["restartPolicy"] == spec["preemptionPolicy"] == "Never", spec
    assert spec["priorityClassName"] == target.run_id and spec["priority"] == 0, spec
    assert spec["activeDeadlineSeconds"] == seconds, spec
    assert spec["terminationGracePeriodSeconds"] == 15, spec
    assert spec["schedulingGates"] == [{"name": "gpu-fault.io/notify008-admission"}], (
        spec
    )
    assert "nodeName" not in spec, "a pre-bound Pod could bypass scheduler gates"
    assert spec["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ] == {
        "nodeSelectorTerms": [
            {
                "matchFields": [
                    {"key": "metadata.name", "operator": "In", "values": [target.node]}
                ]
            }
        ]
    }, spec["affinity"]
    assert spec["securityContext"] == {
        "runAsNonRoot": True,
        "runAsUser": 999,
        "runAsGroup": 999,
        "fsGroup": 999,
    }, spec["securityContext"]
    containers = {item["name"]: item for item in spec["containers"]}
    assert set(containers) == {"runtime", "database"}, containers
    assert containers["runtime"]["image"] == target.runtime_image, containers["runtime"]
    assert containers["database"]["image"] == target.postgres_image, containers[
        "database"
    ]
    assert containers["runtime"]["command"] == [
        "/opt/gpu-fault/control-plane/bin/python"
    ], containers["runtime"]
    assert containers["runtime"]["args"] == [
        "-m",
        "scripts.e2e.regional.probes.notify008_probe",
        "idle",
    ], containers["runtime"]
    assert containers["database"]["args"] == ["/case/postgres-start", "/control"], (
        containers["database"]
    )
    for item in containers.values():
        assert item["securityContext"] == {
            "allowPrivilegeEscalation": False,
            "readOnlyRootFilesystem": True,
            "runAsNonRoot": True,
            "runAsUser": 999,
            "runAsGroup": 999,
            "capabilities": {"drop": ["ALL"]},
            "seccompProfile": {"type": "RuntimeDefault"},
        }, item
        assert item["resources"] == {
            "requests": {"cpu": "100m", "memory": "256Mi"},
            "limits": {"cpu": "1", "memory": "512Mi"},
        }, item
        assert "ports" not in item, "the isolated database must not expose a Pod port"


def test_volumes_are_disposable_and_database_cannot_write_the_control_channel(
    resource_target,
):
    spec = resources.pod_spec(resource_target)
    volumes = {item["name"]: item for item in spec["volumes"]}
    assert volumes.pop("scripts") == {
        "name": "scripts",
        "configMap": {"name": "notify008", "defaultMode": 0o444},
    }, spec["volumes"]
    assert {name: item["emptyDir"] for name, item in volumes.items()} == {
        name: {"medium": "Memory", "sizeLimit": size}
        for name, size in (
            ("control", "1Mi"),
            ("socket", "1Mi"),
            ("work", "32Mi"),
            ("database", "512Mi"),
            ("pg-run", "1Mi"),
        )
    }, volumes
    assert all(set(item) == {"name", "emptyDir"} for item in volumes.values()), volumes
    runtime, database = spec["containers"]
    runtime_mounts = {item["name"]: item for item in runtime["volumeMounts"]}
    database_mounts = {item["name"]: item for item in database["volumeMounts"]}
    assert "database" not in runtime_mounts, runtime_mounts
    assert runtime_mounts["scripts"]["readOnly"] is True, runtime_mounts
    assert database_mounts["control"] == {
        "name": "control",
        "mountPath": "/control",
        "readOnly": True,
    }, database_mounts
    assert runtime_mounts["control"] == {"name": "control", "mountPath": "/control"}, (
        runtime_mounts
    )
    assert database_mounts["pg-run"] == {
        "name": "pg-run",
        "mountPath": "/var/run/postgresql",
    }, "the image initializer needs its private default Unix socket"
    assert "pg-run" not in runtime_mounts, (
        "only the database container may mount its initialization socket directory"
    )


@pytest.mark.parametrize("namespace_uid", [None, "", 0, True])
def test_missing_namespace_identity_cannot_produce_resources(
    resource_target, namespace_uid
):
    with pytest.raises(ProbeError, match="namespace identity"):
        resources.manifests(
            resource_target, namespace_uid, {"notify008_probe.py": "pass\n"}
        )


def test_empty_probe_bundle_cannot_produce_resources(resource_target):
    with pytest.raises(ProbeError, match="complete probe bundle"):
        resources.manifests(resource_target, "namespace-uid", {})


def test_manifest_bundle_is_suspended_isolated_content_bound_and_does_not_mutate_inputs(
    resource_target,
):
    sources = {
        "notify008_probe.py": "PROBE_UNIT = True\n",
        "notify008_protocol.py": "PROTOCOL_UNIT = True\n",
        "unit-data.txt": "opaque unit payload",
    }
    before = deepcopy(sources)
    output = resources.manifests(resource_target, "namespace-uid", sources)
    assert set(output) == {
        "priorityclass",
        "serviceaccount",
        "networkpolicy",
        "configmap",
        "job",
    }, output
    assert output["serviceaccount"]["automountServiceAccountToken"] is False, output
    assert output["networkpolicy"]["spec"] == {
        "podSelector": {},
        "policyTypes": ["Ingress", "Egress"],
        "ingress": [],
        "egress": [],
    }, output["networkpolicy"]
    assert output["configmap"]["immutable"] is True, output["configmap"]
    data = output["configmap"]["data"]
    config = json.loads(data["config.json"])
    assert config["target"] == asdict(resource_target), config
    assert config["namespace_uid"] == "namespace-uid", config
    assert config["source_sha256"] == digest(
        {key: value for key, value in data.items() if key != "config.json"}
    ), "the configuration must bind every mounted source and startup payload"
    assert sources == before, "rendering must not overwrite the caller's source bundle"
    job = output["job"]["spec"]
    assert (
        job["suspend"] is True
        and job["parallelism"] == job["completions"] == 1
        and job["backoffLimit"] == 0
    ), job
    assert job["activeDeadlineSeconds"] == resource_target.seconds, job
    assert job["ttlSecondsAfterFinished"] == 120, job
    items = job["template"]["spec"]["volumes"][0]["configMap"]["items"]
    assert {item["key"]: item["path"] for item in items} == {
        "notify008_probe.py": "scripts/e2e/regional/probes/notify008_probe.py",
        "notify008_protocol.py": "scripts/e2e/regional/probes/notify008_protocol.py",
        "unit-data.txt": "unit-data.txt",
        "postgres-start": "postgres-start",
        "config.json": "config.json",
    }, items
    for kind, resource in output.items():
        if kind == "priorityclass":
            assert "namespace" not in resource["metadata"], resource
            assert resource["metadata"]["name"] == resource_target.run_id, resource
        else:
            assert resource["metadata"]["namespace"] == resource_target.run_id, resource
            assert resource["metadata"]["name"] == "notify008", resource


def test_bundle_digest_is_order_independent_but_changes_with_source_bytes(
    resource_target,
):
    source = {"a.py": "first\n", "b.py": "second\n"}
    same = dict(reversed(list(source.items())))
    changed = {**source, "a.py": "changed\n"}
    hashes = [
        json.loads(
            resources.manifests(resource_target, "namespace-uid", data)["configmap"][
                "data"
            ]["config.json"]
        )["source_sha256"]
        for data in (source, same, changed)
    ]
    assert hashes[0] == hashes[1], "source dictionary order must not change identity"
    assert hashes[0] != hashes[2], (
        "modified payload bytes must invalidate the source pin"
    )
