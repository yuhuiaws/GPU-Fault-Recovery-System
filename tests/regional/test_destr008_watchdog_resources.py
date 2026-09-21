"""CPU resource provenance and actual admitted-Pod safety, without cluster I/O."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr008_watchdog_resources as resources
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._destr008_watchdog_resources import (
    CONTROL_UID,
    FINISHED,
    IMAGE,
    JOB_UID,
    NAME,
    NAMESPACE,
    POD_NAME,
    POD_UID,
    STARTED,
    Harness,
    admitted,
    edit,
    harness,
    metadata,
    pod,
    validate_pod,
)


@pytest.fixture
def cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    return harness(tmp_path, monkeypatch)


def test_runtime_binds_ready_deployment_namespace_image_and_config_sources(
    cpu: Harness,
) -> None:
    runtime = resources.read_runtime(cpu.regional)
    assert runtime.identity() == {
        "namespace": NAMESPACE,
        "namespace_uid": "namespace-uid",
        "deployment_uid": "deployment-uid",
        "generation": 7,
        "image": IMAGE,
        "modes": dict(zip(resources.MODE_KEYS, ("dedicated", "legacy"), strict=True)),
        "config_identities": {
            "worker-postgres": {"uid": "config-uid", "resource_version": "10"},
            resources.CA_CONFIGMAP: {"uid": "ca-uid", "resource_version": "10"},
        },
    }, "the runtime must bind actual CPU resources, not labels or local defaults"
    identity = runtime.identity()
    identity["modes"][resources.MODE_KEYS[0]] = "legacy"
    identity["config_identities"]["worker-postgres"]["uid"] = "changed"
    assert runtime.modes[resources.MODE_KEYS[0]] == "dedicated", (
        "identity callers must not mutate the trusted runtime"
    )
    assert runtime.config_identities["worker-postgres"]["uid"] == "config-uid", (
        "nested source identities must also be copied"
    )
    assert all(call[0] == "cpu" and call[1] == "get" for call in cpu.calls), (
        "discovery must remain CPU-only and read-only"
    )


def test_runtime_accepts_real_serialized_projection_defaults(cpu: Harness) -> None:
    spec = cpu.deployment["spec"]["template"]["spec"]
    for volume in spec["volumes"]:
        source = volume.get("secret", volume.get("configMap"))
        source.update(defaultMode=420, optional=False, items=[])
    for mount in cpu.container["volumeMounts"]:
        mount["mountPropagation"] = "None"
    cpu.container["envFrom"][0]["configMapRef"]["optional"] = False
    cpu.container["env"][0]["value"] = ""
    cpu.container["env"][0]["valueFrom"]["secretKeyRef"]["optional"] = False
    assert resources.read_runtime(cpu.regional).image == IMAGE, (
        "defaultMode=0644 and required-reference defaults are normal API output"
    )


def test_effective_modes_honor_explicit_env_precedence_without_inheriting_sources(
    cpu: Harness,
) -> None:
    cpu.container["env"].extend(
        {"name": key, "value": "dual"} for key in resources.MODE_KEYS
    )
    runtime = resources.read_runtime(cpu.regional)
    assert set(runtime.modes.values()) == {"dual"}, "explicit env overrides envFrom"
    variables = cpu.manifests()[-1]["spec"]["template"]["spec"]["containers"][0]["env"]
    assert {
        item["name"]: item["value"]
        for item in variables
        if item["name"] in resources.MODE_KEYS
    } == runtime.modes, "the watchdog must use the effective CPU Store modes"


def test_multiple_disjoint_required_configmaps_and_literal_only_source(
    cpu: Harness,
) -> None:
    config = cpu.objects["configmap", "worker-postgres"]
    queue_mode = config["data"].pop(resources.MODE_KEYS[1])
    cpu.objects["configmap", "worker-queue"] = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": metadata("worker-queue", "queue-uid"),
        "data": {resources.MODE_KEYS[1]: queue_mode},
    }
    cpu.container["envFrom"].append({"configMapRef": {"name": "worker-queue"}})
    assert "worker-queue" in resources.read_runtime(cpu.regional).config_identities, (
        "each nonsecret source must be individually bound"
    )
    cpu.container["envFrom"] = []
    cpu.container["env"].extend(
        {"name": key, "value": value}
        for key, value in {**config["data"], resources.MODE_KEYS[1]: queue_mode}.items()
    )
    assert (
        resources.read_runtime(cpu.regional).modes[resources.MODE_KEYS[1]] == "legacy"
    ), "required settings may also be explicit literal env values"


@pytest.mark.parametrize(
    "response", ["not-json-fixture", "[]", "null", "42", '{"kind": "Namespace"}']
)
def test_source_response_errors_are_sanitized(cpu: Harness, response: str) -> None:
    cpu.response = response
    with pytest.raises(RegionalFixtureError) as caught:
        resources.read_runtime(cpu.regional)
    assert response not in str(caught.value), "source payloads must not enter errors"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("apiVersion",), "v2"),
        (("kind",), "Secret"),
        (("metadata",), None),
        (("metadata", "name"), "foreign"),
        (("metadata", "namespace"), "foreign"),
        (("metadata", "uid"), ""),
        (("metadata", "uid"), True),
        (("metadata", "resourceVersion"), ""),
        (("metadata", "resourceVersion"), 1),
        (("metadata", "deletionTimestamp"), STARTED),
    ],
)
def test_source_identity_is_not_inferred_from_labels(
    cpu: Harness, path: tuple[str, ...], value: Any
) -> None:
    edit(cpu.deployment, path, value)
    with pytest.raises(RegionalFixtureError):
        resources.read_runtime(cpu.regional)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("metadata", "generation"), None),
        (("metadata", "generation"), True),
        (("metadata", "generation"), 0),
        (("spec", "replicas"), None),
        (("spec", "replicas"), True),
        (("spec", "replicas"), 0),
        (("status", "observedGeneration"), "7"),
        (("status", "observedGeneration"), True),
        (("status", "observedGeneration"), 6),
        (("status", "observedGeneration"), 8),
        (("status", "replicas"), 3),
        (("status", "updatedReplicas"), 1),
        (("status", "updatedReplicas"), 2.0),
        (("status", "readyReplicas"), 1),
        (("status", "availableReplicas"), 1),
        (("status", "availableReplicas"), True),
        (("status", "unavailableReplicas"), 1),
        (("status", "unavailableReplicas"), False),
        (("status", "terminatingReplicas"), 1),
        (("spec", "paused"), True),
        (("status", "conditions"), []),
        (("status", "conditions"), [{"type": "Available", "status": "False"}]),
        (
            ("status", "conditions"),
            [
                {"type": "Available", "status": "True"},
                {"type": "Progressing", "status": "False"},
            ],
        ),
        (
            ("status", "conditions"),
            [
                {"type": "Available", "status": "True"},
                {"type": "Progressing", "status": "True"},
                {"type": "ReplicaFailure", "status": "True"},
            ],
        ),
        (("spec",), []),
        (("status",), None),
        (("spec", "template"), None),
        (("spec", "template", "spec"), None),
    ],
)
def test_unready_mixed_or_ambiguous_rollout_is_rejected(
    cpu: Harness, path: tuple[str, ...], value: Any
) -> None:
    edit(cpu.deployment, path, value)
    with pytest.raises(RegionalFixtureError):
        resources.read_runtime(cpu.regional)


@pytest.mark.parametrize(
    "containers",
    [
        None,
        [],
        [None],
        [{"name": "foreign"}],
        [{"name": "control-worker"}, {"name": "control-worker"}],
    ],
)
def test_runtime_container_must_be_unique(cpu: Harness, containers: Any) -> None:
    cpu.deployment["spec"]["template"]["spec"]["containers"] = containers
    with pytest.raises(RegionalFixtureError):
        resources.read_runtime(cpu.regional)


@pytest.mark.parametrize(
    "image",
    [None, 7, "cpu:latest", "cpu@sha256:" + "A" * 64, "cpu @sha256:" + "a" * 64],
)
def test_runtime_image_requires_an_exact_immutable_digest(
    cpu: Harness, image: Any
) -> None:
    cpu.container["image"] = image
    with pytest.raises(RegionalFixtureError, match="immutable"):
        resources.read_runtime(cpu.regional)


@pytest.mark.parametrize(
    "source",
    [
        None,
        {"secretRef": {"name": "not-readable"}},
        {"configMapRef": {"name": "worker-postgres"}, "prefix": "OTHER_"},
        {"configMapRef": {"name": "worker-postgres"}, "unexpected": True},
        {"configMapRef": None},
        {"configMapRef": {"name": ""}},
        {"configMapRef": {"name": "worker-postgres", "optional": True}},
        {"configMapRef": {"name": "worker-postgres", "optional": 0}},
        {"configMapRef": {"name": "worker-postgres", "extra": "value"}},
    ],
)
def test_ambiguous_environment_sources_are_not_resolved(
    cpu: Harness, source: Any
) -> None:
    cpu.container["envFrom"] = [source]
    with pytest.raises(RegionalFixtureError):
        resources.read_runtime(cpu.regional)


@pytest.mark.parametrize("duplicate", ["source", "key"])
def test_duplicate_sources_or_cross_config_keys_are_ambiguous(
    cpu: Harness, duplicate: str
) -> None:
    if duplicate == "source":
        cpu.container["envFrom"] *= 2
    else:
        cpu.objects["configmap", "other"] = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": metadata("other", "other-uid"),
            "data": {resources.MODE_KEYS[0]: "dedicated"},
        }
        cpu.container["envFrom"].append({"configMapRef": {"name": "other"}})
    with pytest.raises(RegionalFixtureError):
        resources.read_runtime(cpu.regional)


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {},
        {resources.MODE_KEYS[0]: False},
        {resources.MODE_KEYS[0]: "unknown"},
    ],
)
def test_missing_or_invalid_configuration_is_not_a_default(
    cpu: Harness, data: Any
) -> None:
    cpu.objects["configmap", "worker-postgres"]["data"] = data
    with pytest.raises(RegionalFixtureError):
        resources.read_runtime(cpu.regional)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (resources.MODE_KEYS[0], "unknown"),
        (resources.MODE_KEYS[1], "true"),
        ("GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT", "true"),
        ("GPU_FAULT_STORE_URL_FILE", "/elsewhere"),
    ],
)
def test_unsafe_store_configuration_is_rejected(
    cpu: Harness, key: str, value: str
) -> None:
    cpu.objects["configmap", "worker-postgres"]["data"][key] = value
    with pytest.raises(RegionalFixtureError):
        resources.read_runtime(cpu.regional)


@pytest.mark.parametrize(
    "variables",
    [
        None,
        {},
        [None],
        [{"name": []}],
        [{"name": ""}],
        [],
        [{"name": "duplicate"}, {"name": "duplicate"}],
        [{"name": "GPU_FAULT_STORE_URL", "value": "not-a-secret"}],
        [{"name": "GPU_FAULT_STORE_URL", "valueFrom": None}],
        [{"name": "GPU_FAULT_STORE_URL", "valueFrom": {"secretKeyRef": None}}],
    ],
)
def test_ambiguous_or_missing_dsn_environment_is_rejected(
    cpu: Harness, variables: Any
) -> None:
    cpu.container["env"] = variables
    with pytest.raises(RegionalFixtureError):
        resources.read_runtime(cpu.regional)


@pytest.mark.parametrize(
    "change",
    [
        {"name": "foreign"},
        {"key": "execution-token"},
        {"optional": True},
        {"optional": 0},
        {"extra": "value"},
    ],
)
def test_dsn_must_remain_the_exact_required_cpu_secret_key(
    cpu: Harness, change: dict[str, Any]
) -> None:
    cpu.container["env"][0]["valueFrom"]["secretKeyRef"].update(change)
    with pytest.raises(RegionalFixtureError):
        resources.read_runtime(cpu.regional)


@pytest.mark.parametrize(
    "variable",
    [
        {
            "name": resources.MODE_KEYS[0],
            "valueFrom": {
                "configMapKeyRef": {
                    "name": "worker-postgres",
                    "key": resources.MODE_KEYS[0],
                }
            },
        },
        {"name": resources.MODE_KEYS[0], "value": "dual", "valueFrom": {}},
        {"name": resources.MODE_KEYS[0], "value": True},
    ],
)
def test_indirect_or_mixed_mode_overrides_are_rejected(
    cpu: Harness, variable: dict[str, Any]
) -> None:
    cpu.container["env"].append(variable)
    with pytest.raises(RegionalFixtureError):
        resources.read_runtime(cpu.regional)


@pytest.mark.parametrize(
    "change",
    [
        {"readOnly": False},
        {"subPath": "postgres-url"},
        {"subPathExpr": "$(KEY)"},
        {"mountPropagation": "Bidirectional"},
        {"extra": True},
        {"name": "missing"},
    ],
)
def test_source_credential_mount_is_a_complete_read_only_projection(
    cpu: Harness, change: dict[str, Any]
) -> None:
    cpu.container["volumeMounts"][0].update(change)
    with pytest.raises(RegionalFixtureError):
        resources.read_runtime(cpu.regional)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "shadow",
        "missing",
        "bad-name",
        "duplicate-volume",
        "bad-volume-name",
        "foreign-source",
        "mixed-source",
    ],
)
def test_source_mount_or_volume_identity_cannot_be_ambiguous(
    cpu: Harness, case: str
) -> None:
    spec = cpu.deployment["spec"]["template"]["spec"]
    mounts, volumes = cpu.container["volumeMounts"], spec["volumes"]
    if case == "duplicate":
        mounts.append(copy.deepcopy(mounts[0]))
    elif case == "shadow":
        mounts.append(
            {"name": "ca", "mountPath": resources.STORE_DIRECTORY + "/postgres-url"}
        )
    elif case == "missing":
        mounts.pop(0)
    elif case == "bad-name":
        mounts[0]["mountPath"] = None
    elif case == "duplicate-volume":
        volumes.append(copy.deepcopy(volumes[0]))
    elif case == "bad-volume-name":
        volumes[0]["name"] = []
    elif case == "foreign-source":
        volumes[0] = {"name": "store", "projected": {"sources": []}}
    else:
        volumes[0]["hostPath"] = {"path": "/"}
    with pytest.raises(RegionalFixtureError):
        resources.read_runtime(cpu.regional)


@pytest.mark.parametrize(
    "change",
    [
        {"secretName": "foreign"},
        {"optional": True},
        {"optional": 0},
        {"items": [{"key": "postgres-url", "path": "other"}]},
        {"defaultMode": True},
        {"defaultMode": "420"},
        {"defaultMode": 0o777},
        {"defaultMode": -1},
        {"defaultMode": 0},
        {"extra": "value"},
    ],
)
def test_source_projection_permissions_and_keys_are_constrained(
    cpu: Harness, change: dict[str, Any]
) -> None:
    cpu.deployment["spec"]["template"]["spec"]["volumes"][0]["secret"].update(change)
    with pytest.raises(RegionalFixtureError):
        resources.read_runtime(cpu.regional)


@pytest.mark.parametrize(
    "data", [None, {}, {resources.CA_KEY: False}, {resources.CA_KEY: "\n"}]
)
def test_source_rds_ca_must_exist(cpu: Harness, data: Any) -> None:
    cpu.objects["configmap", resources.CA_CONFIGMAP]["data"] = data
    with pytest.raises(RegionalFixtureError):
        resources.read_runtime(cpu.regional)


def test_builder_projects_only_planned_cpu_inputs_and_scoped_rbac(cpu: Harness) -> None:
    runtime, plan = resources.read_runtime(cpu.regional), cpu.plan()
    control = resources.control_manifest(plan, runtime, NAME)
    account, role, binding, code, job = resources.supporting_manifests(
        plan, runtime, name=NAME, control_uid=CONTROL_UID
    )
    assert control["data"] == wire.initial_data(plan), (
        "control protocol must start unclaimed"
    )
    assert control["metadata"]["annotations"] == {
        "gpu-fault.io/acceptance-plan-sha256": wire.digest(plan),
        "gpu-fault.io/acceptance-runtime-sha256": wire.digest(runtime.identity()),
        "gpu-fault.io/acceptance-source-sha256": plan.probe_sha256,
    }, "namespace/deployment/config/source bindings must travel with the resources"
    assert account["automountServiceAccountToken"] is False, (
        "no ambient SA mount is allowed"
    )
    assert role["rules"] == [
        {
            "apiGroups": [""],
            "resources": ["configmaps"],
            "resourceNames": [NAME],
            "verbs": ["get", "patch"],
        }
    ], "the watchdog may only read and patch its own control map"
    assert binding["subjects"] == [
        {"kind": "ServiceAccount", "name": NAME, "namespace": NAMESPACE}
    ], "RBAC must refer only to the run-scoped CPU service account"
    assert binding["roleRef"]["name"] == NAME, (
        "the scoped Role must be the sole binding"
    )
    assert set(code["data"]) == set(wire.SOURCE_FILES), (
        "all split probe files must be shipped"
    )
    assert (
        wire.digest(
            {
                name: hashlib.sha256(text.encode()).hexdigest()
                for name, text in code["data"].items()
            }
        )
        == plan.probe_sha256
    ), "the emitted bytes, including newlines, must match the plan"
    assert code["immutable"] is True, (
        "the admitted probe source may not change in place"
    )
    spec = job["spec"]["template"]["spec"]
    assert spec["restartPolicy"] == "Never" and job["spec"]["backoffLimit"] == 0, (
        "automatic retries cannot replace the originally admitted watchdog"
    )
    assert spec["schedulingGates"] == [{"name": resources.SCHEDULING_GATE}], (
        "the real Pod must not run until admission verification and CAS release"
    )
    assert spec["automountServiceAccountToken"] is False, (
        "only the explicit token may be projected"
    )
    container = spec["containers"][0]
    assert container["image"] == runtime.image, (
        "the watchdog uses the bound CPU runtime image"
    )
    assert container["command"] == [
        "/opt/gpu-fault/control-plane/bin/python",
        "-s",
        "-B",
        resources.CODE_DIRECTORY + "/destr008_cancellation_probe.py",
        "--namespace",
        NAMESPACE,
        "--configmap",
        NAME,
        "--uid",
        CONTROL_UID,
        "--plan-sha256",
        wire.digest(plan),
    ], "the command must execute only the pinned probe and exact control identity"
    assert container["envFrom"] == [], (
        "the source deployment's environment is not inherited"
    )
    assert {entry["name"] for entry in container["env"]} == {
        "GPU_FAULT_STORE_URL",
        "GPU_FAULT_STORE_URL_FILE",
        "GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT",
        "GPU_FAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS",
        *resources.MODE_KEYS,
    }, "execution tokens, masters, GPU kubeconfig, and ambient settings are absent"
    volumes = {volume["name"]: volume for volume in spec["volumes"]}
    assert set(volumes) == {"code", "store", "ca", "scratch", "api-access"}, (
        "there must be no extra credential or host volumes"
    )
    assert volumes["store"]["secret"]["items"] == [
        {"key": resources.STORE_KEY, "path": resources.STORE_KEY}
    ], "only the CPU Aurora DSN key is projected"
    assert volumes["ca"]["configMap"]["items"] == [
        {"key": resources.CA_KEY, "path": resources.CA_KEY}
    ], "only the public RDS CA key is projected"
    assert volumes["api-access"]["projected"]["sources"][0] == {
        "serviceAccountToken": {"path": "token", "expirationSeconds": 600}
    }, "the only API token is an explicit bounded projection"
    assert job["spec"]["activeDeadlineSeconds"] == (
        plan.deadline_at - plan.created_at + wire.DRAIN_SECONDS + 180
    ), "the Job has its own bounded lifetime"


@pytest.mark.parametrize(
    "change",
    [
        {"namespace": ""},
        {"namespace_uid": ""},
        {"deployment_uid": ""},
        {"generation": True},
        {"generation": 0},
        {"image": "cpu:latest"},
        {"modes": {}},
        {
            "modes": {
                resources.MODE_KEYS[0]: "foreign",
                resources.MODE_KEYS[1]: "legacy",
            }
        },
        {
            "modes": {
                resources.MODE_KEYS[0]: "legacy",
                resources.MODE_KEYS[1]: "legacy",
                "EXTRA_ENV": "unsafe",
            }
        },
    ],
)
def test_builders_reject_unbound_runtime(cpu: Harness, change: dict[str, Any]) -> None:
    runtime = replace(resources.read_runtime(cpu.regional), **change)
    with pytest.raises(RegionalFixtureError):
        resources.control_manifest(cpu.plan(), runtime, NAME)


@pytest.mark.parametrize("name", ["", "a" * 59, "UPPER", "wrong/name"])
def test_builder_names_are_bounded_kubernetes_names(cpu: Harness, name: str) -> None:
    with pytest.raises(RegionalFixtureError):
        resources.control_manifest(
            cpu.plan(), resources.read_runtime(cpu.regional), name
        )


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "directory",
        "symlink",
        "invalid-utf8",
        "drift",
        "control-uid",
        "root-symlink",
    ],
)
def test_missing_unsafe_or_changed_script_cannot_be_shipped(
    cpu: Harness, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    runtime, plan = resources.read_runtime(cpu.regional), cpu.plan()
    file = cpu.source / wire.SOURCE_FILES[0]
    control_uid = CONTROL_UID
    if case in {"missing", "directory", "symlink"}:
        file.unlink()
        if case == "directory":
            file.mkdir()
        elif case == "symlink":
            file.symlink_to(cpu.source / wire.SOURCE_FILES[1])
    elif case == "invalid-utf8":
        file.write_bytes(b"\xff")
    elif case == "drift":
        file.write_bytes(b"# Different source\n")
    elif case == "control-uid":
        control_uid = ""
    else:
        link = cpu.source.parent / "linked-source"
        link.symlink_to(cpu.source, target_is_directory=True)
        monkeypatch.setattr(resources, "CODE_SOURCE", link)
    with pytest.raises(RegionalFixtureError):
        resources.supporting_manifests(
            plan, runtime, name=NAME, control_uid=control_uid
        )


def test_read_failure_does_not_echo_paths_or_payloads(
    cpu: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, plan = resources.read_runtime(cpu.regional), cpu.plan()

    def fail(_path: Path) -> bytes:
        raise PermissionError("private path fixture")

    monkeypatch.setattr(Path, "read_bytes", fail)
    with pytest.raises(RegionalFixtureError) as caught:
        resources.supporting_manifests(
            plan, runtime, name=NAME, control_uid=CONTROL_UID
        )
    assert "private path" not in str(caught.value), (
        "local read errors must be sanitized"
    )


def test_hash_binds_emitted_bytes_even_when_file_changes_after_read(
    cpu: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, plan = resources.read_runtime(cpu.regional), cpu.plan()
    original_read = Path.read_bytes

    def read_then_change(path: Path) -> bytes:
        content = original_read(path)
        path.write_bytes(b"# Changed after capture\n")
        return content

    monkeypatch.setattr(Path, "read_bytes", read_then_change)
    code = resources.supporting_manifests(
        plan, runtime, name=NAME, control_uid=CONTROL_UID
    )[-2]
    assert (
        wire.digest(
            {
                key: hashlib.sha256(value.encode()).hexdigest()
                for key, value in code["data"].items()
            }
        )
        == plan.probe_sha256
    ), "hashing and projection must use the same captured bytes"


@pytest.mark.parametrize("index", range(4))
def test_supporting_resources_accept_real_defaults_without_mutating_inputs(
    cpu: Harness, index: int
) -> None:
    expected = cpu.manifests()[index]
    actual = admitted(expected, uid="support-uid")
    before = copy.deepcopy((actual, expected))
    resources.validate_supporting_resource(actual, expected, uid="support-uid")
    assert (actual, expected) == before, "validation must not rewrite either object"


def test_initial_control_configmap_accepts_empty_default_fields(cpu: Harness) -> None:
    expected = resources.control_manifest(
        cpu.plan(), resources.read_runtime(cpu.regional), NAME
    )
    actual = admitted(expected, uid=CONTROL_UID)
    actual["immutable"] = False
    resources.validate_supporting_resource(actual, expected, uid=CONTROL_UID)


@pytest.mark.parametrize(
    ("index", "path", "value"),
    [
        (0, ("automountServiceAccountToken",), True),
        (0, ("secrets",), [{"name": "extra"}]),
        (0, ("imagePullSecrets",), [{"name": "extra"}]),
        (0, ("metadata", "annotations", "eks.amazonaws.com/role-arn"), "not-approved"),
        (1, ("rules", 0, "verbs"), ["get", "patch", "create"]),
        (1, ("rules", 0, "resources"), ["configmaps", "secrets"]),
        (1, ("rules", 0, "resourceNames"), ["*"]),
        (1, ("rules", 0, "apiGroups"), ["*"]),
        (1, ("aggregationRule",), {"clusterRoleSelectors": [{}]}),
        (2, ("subjects", 0, "namespace"), "foreign"),
        (2, ("subjects", 0, "name"), "default"),
        (2, ("subjects", 0, "apiGroup"), "rbac.authorization.k8s.io"),
        (2, ("roleRef", "kind"), "ClusterRole"),
        (2, ("roleRef", "name"), "cluster-admin"),
        (3, ("data", wire.SOURCE_FILES[0]), "# Replaced probe\n"),
        (3, ("data", "extra.py"), "# Unexpected executable\n"),
        (3, ("binaryData",), {"extra": "fixture"}),
        (3, ("immutable",), False),
    ],
)
def test_supporting_resources_reject_source_rbac_or_credential_drift(
    cpu: Harness, index: int, path: tuple[str | int, ...], value: Any
) -> None:
    expected = cpu.manifests()[index]
    actual = admitted(expected, uid="support-uid")
    edit(actual, path, value)
    with pytest.raises(RegionalFixtureError):
        resources.validate_supporting_resource(actual, expected, uid="support-uid")


@pytest.mark.parametrize("phase", ["gated", "running", "succeeded"])
def test_job_and_actual_pod_validate_each_lifecycle_phase(
    cpu: Harness, phase: resources.WatchdogPhase
) -> None:
    expected = cpu.manifests()[-1]
    actual_job = admitted(expected, uid=JOB_UID)
    if phase == "running":
        actual_job["status"]["ready"] = 1
    elif phase == "succeeded":
        actual_job["status"] = {
            "succeeded": 1,
            "conditions": [{"type": "Complete", "status": "True"}],
            "uncountedTerminatedPods": {"failed": [], "succeeded": []},
        }
    actual_pod = pod(expected, phase)
    before = copy.deepcopy((actual_job, actual_pod, expected))
    resources.validate_job(actual_job, expected, uid=JOB_UID, phase=phase)
    validate_pod(actual_pod, expected, phase)
    assert (actual_job, actual_pod, expected) == before, (
        "admission validation must be read-only"
    )


def test_job_initial_read_allows_unpopulated_counts_but_not_injection(
    cpu: Harness,
) -> None:
    expected = cpu.manifests()[-1]
    actual = admitted(expected, uid=JOB_UID)
    actual.pop("status")
    resources.validate_job(actual, expected, uid=JOB_UID)
    with pytest.raises(RegionalFixtureError):
        resources.validate_job(actual, expected, uid=JOB_UID, phase="gated")


def test_legacy_job_controller_default_selector_is_uid_bound(cpu: Harness) -> None:
    expected = cpu.manifests()[-1]
    actual = admitted(expected, uid=JOB_UID)
    actual["spec"]["selector"] = {
        "matchLabels": {"controller-uid": JOB_UID},
        "matchExpressions": [],
    }
    resources.validate_job(actual, expected, uid=JOB_UID, phase="gated")


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_resource_owner_false_default_may_be_omitted(cpu: Harness, index: int) -> None:
    expected = cpu.manifests()[index]
    uid = JOB_UID if index == 4 else "support-uid"
    actual = admitted(expected, uid=uid)
    actual["metadata"]["ownerReferences"][0].pop("blockOwnerDeletion")
    actual["metadata"]["ownerReferences"][0].pop("controller")
    if index == 4:
        resources.validate_job(actual, expected, uid=uid)
    else:
        resources.validate_supporting_resource(actual, expected, uid=uid)


@pytest.mark.parametrize("target", ["job", "pod", "code"])
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("apiVersion",), "foreign/v1"),
        (("kind",), "Foreign"),
        (("metadata", "namespace"), "gpu-namespace"),
        (("metadata", "name"), "same-labels-different-name"),
        (("metadata", "uid"), "replacement"),
        (("metadata", "resourceVersion"), ""),
        (("metadata", "deletionTimestamp"), STARTED),
        (("metadata", "ownerReferences"), []),
        (("metadata", "ownerReferences", 0, "uid"), "replaced-owner"),
        (("metadata", "ownerReferences", 0, "name"), "replaced-owner-name"),
        (("metadata", "ownerReferences", 0, "apiVersion"), "foreign/v1"),
        (("metadata", "ownerReferences", 0, "extra"), True),
        (
            ("metadata", "annotations", "gpu-fault.io/acceptance-source-sha256"),
            "0" * 64,
        ),
        (
            ("metadata", "annotations", "gpu-fault.io/acceptance-runtime-sha256"),
            "0" * 64,
        ),
        (("metadata", "annotations", "gpu-fault.io/acceptance-plan-sha256"), "0" * 64),
        (("metadata", "labels", "gpu-fault.io/acceptance-case"), "FOREIGN"),
    ],
)
def test_actual_resource_identity_is_exact_despite_copied_metadata(
    cpu: Harness, target: str, path: tuple[str | int, ...], value: Any
) -> None:
    manifests = cpu.manifests()
    expected = manifests[-2] if target == "code" else manifests[-1]
    actual = pod(expected) if target == "pod" else admitted(expected, uid=JOB_UID)
    edit(actual, path, value)
    with pytest.raises(RegionalFixtureError):
        if target == "job":
            resources.validate_job(actual, expected, uid=JOB_UID)
        elif target == "pod":
            validate_pod(actual, expected)
        else:
            resources.validate_supporting_resource(actual, expected, uid=JOB_UID)


CONTAINER = ("spec", "containers", 0)
VOLUME = ("spec", "volumes")


@pytest.mark.parametrize("target", ["job", "pod"])
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("spec", "containers"), []),
        (("spec", "containers"), [None]),
        (("spec", "initContainers"), [{"name": "injected", "image": IMAGE}]),
        (("spec", "ephemeralContainers"), [{"name": "injected", "image": IMAGE}]),
        (("spec", "serviceAccountName"), "default"),
        (("spec", "serviceAccount"), "foreign"),
        (("spec", "automountServiceAccountToken"), True),
        (("spec", "hostNetwork"), True),
        (("spec", "hostPID"), True),
        (("spec", "hostIPC"), True),
        (("spec", "shareProcessNamespace"), True),
        (("spec", "enableServiceLinks"), True),
        (("spec", "restartPolicy"), "OnFailure"),
        (("spec", "nodeName"), "already-bound"),
        (("spec", "schedulingGates"), []),
        (("spec", "dnsPolicy"), "Default"),
        (("spec", "schedulerName"), "foreign"),
        (("spec", "priority"), 1000),
        (("spec", "preemptionPolicy"), "Never"),
        (("spec", "tolerations"), [{"operator": "Exists"}]),
        (("spec", "securityContext", "runAsUser"), 0),
        (("spec", "securityContext", "runAsNonRoot"), False),
        (("spec", "securityContext", "seccompProfile"), {"type": "Unconfined"}),
        ((*CONTAINER, "name"), "foreign"),
        ((*CONTAINER, "image"), "cpu:latest"),
        ((*CONTAINER, "image"), "registry.example/cpu@sha256:" + "b" * 64),
        ((*CONTAINER, "command"), ["/bin/sh"]),
        ((*CONTAINER, "args"), ["--cleanup-only"]),
        ((*CONTAINER, "workingDir"), "/tmp"),
        ((*CONTAINER, "envFrom"), [{"secretRef": {"name": "not-approved"}}]),
        ((*CONTAINER, "env", 0, "valueFrom", "secretKeyRef", "name"), "not-approved"),
        ((*CONTAINER, "env", 0, "valueFrom", "secretKeyRef", "optional"), True),
        ((*CONTAINER, "env", 0, "value"), "not-an-approved-dsn"),
        ((*CONTAINER, "env", 2, "value"), "true"),
        ((*CONTAINER, "env", 3, "value"), "0"),
        ((*CONTAINER, "env"), [{"name": "KUBECONFIG", "value": "/gpu/config"}]),
        (
            (*CONTAINER, "env"),
            [
                {
                    "name": "GPU_FAULT_EXECUTION_TOKEN",
                    "valueFrom": {
                        "secretKeyRef": {"name": "not-approved", "key": "token"}
                    },
                }
            ],
        ),
        (
            (*CONTAINER, "env"),
            [
                {
                    "name": "GPU_FAULT_FLEET_MASTER_KEY",
                    "valueFrom": {
                        "secretKeyRef": {"name": "not-approved", "key": "master"}
                    },
                }
            ],
        ),
        ((*CONTAINER, "env"), None),
        ((*CONTAINER, "securityContext", "privileged"), True),
        ((*CONTAINER, "securityContext", "allowPrivilegeEscalation"), True),
        ((*CONTAINER, "securityContext", "readOnlyRootFilesystem"), False),
        ((*CONTAINER, "securityContext", "capabilities", "add"), ["SYS_ADMIN"]),
        ((*CONTAINER, "resources", "limits", "cpu"), "10"),
        ((*CONTAINER, "lifecycle"), {"postStart": {"exec": {"command": ["unsafe"]}}}),
        ((*CONTAINER, "volumeMounts", 0, "readOnly"), False),
        ((*CONTAINER, "volumeMounts", 1, "subPath"), "postgres-url"),
        ((*CONTAINER, "volumeMounts", 1, "mountPath"), "/elsewhere"),
        ((*VOLUME, 0, "configMap", "name"), "foreign-code"),
        ((*VOLUME, 0, "configMap", "optional"), True),
        ((*VOLUME, 0, "configMap", "defaultMode"), 0o777),
        ((*VOLUME, 0, "configMap", "items"), []),
        ((*VOLUME, 1, "secret", "items"), []),
        (
            (*VOLUME, 1, "secret", "items"),
            [{"key": "execution-token", "path": "postgres-url"}],
        ),
        ((*VOLUME, 1, "secret", "defaultMode"), 0o644),
        ((*VOLUME, 1, "secret", "optional"), True),
        ((*VOLUME, 2, "configMap", "name"), "foreign-ca"),
        ((*VOLUME, 2, "configMap", "items"), []),
        ((*VOLUME, 3, "emptyDir", "sizeLimit"), "1Gi"),
        ((*VOLUME, 3, "hostPath"), {"path": "/"}),
        ((*VOLUME, 4, "projected", "defaultMode"), 0o777),
        (
            (
                *VOLUME,
                4,
                "projected",
                "sources",
                0,
                "serviceAccountToken",
                "expirationSeconds",
            ),
            3600,
        ),
        (
            (*VOLUME, 4, "projected", "sources", 0, "serviceAccountToken", "audience"),
            "sts.amazonaws.com",
        ),
        ((*VOLUME, 4, "projected", "sources", 1, "configMap", "optional"), True),
        (
            (*VOLUME, 4, "projected", "sources", 1, "configMap", "name"),
            "foreign-root-ca",
        ),
        (
            (
                *VOLUME,
                4,
                "projected",
                "sources",
                2,
                "downwardAPI",
                "items",
                0,
                "fieldRef",
                "fieldPath",
            ),
            "metadata.name",
        ),
    ],
)
def test_admission_cannot_change_any_planned_execution_or_security_field(
    cpu: Harness, target: str, path: tuple[str | int, ...], value: Any
) -> None:
    expected = cpu.manifests()[-1]
    actual = pod(expected) if target == "pod" else admitted(expected, uid=JOB_UID)
    edit(actual if target == "pod" else actual["spec"]["template"], path, value)
    with pytest.raises(RegionalFixtureError):
        if target == "pod":
            validate_pod(actual, expected)
        else:
            resources.validate_job(actual, expected, uid=JOB_UID, phase="gated")


@pytest.mark.parametrize("field", ["containers", "volumes"])
def test_injected_sidecar_and_extra_volume_cannot_hide_behind_original_entries(
    cpu: Harness, field: str
) -> None:
    expected = cpu.manifests()[-1]
    actual = pod(expected)
    injected: dict[str, Any] = (
        {"name": "injected", "image": IMAGE}
        if field == "containers"
        else {"name": "host", "hostPath": {"path": "/"}}
    )
    actual["spec"][field].append(injected)
    with pytest.raises(RegionalFixtureError):
        validate_pod(actual, expected)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("spec", "backoffLimit"), 1),
        (("spec", "activeDeadlineSeconds"), 1),
        (("spec", "ttlSecondsAfterFinished"), 1),
        (("spec", "parallelism"), True),
        (("spec", "parallelism"), 2),
        (("spec", "completions"), 2),
        (("spec", "manualSelector"), True),
        (("spec", "suspend"), True),
        (("spec", "podReplacementPolicy"), "Failed"),
        (("spec", "managedBy"), "foreign"),
        (("spec", "selector"), {"matchLabels": {"job-name": NAME}}),
        (
            ("spec", "selector"),
            {"matchLabels": {"batch.kubernetes.io/controller-uid": "foreign"}},
        ),
        (("spec", "selector"), None),
        (("spec", "template", "metadata", "labels", "job-name"), "foreign"),
        (("spec", "template", "metadata", "labels", "controller-uid"), "foreign"),
        (("spec", "template", "metadata", "annotations", "source"), "foreign"),
    ],
)
def test_job_default_normalization_never_ignores_nondefault_values(
    cpu: Harness, path: tuple[str | int, ...], value: Any
) -> None:
    expected = cpu.manifests()[-1]
    actual = admitted(expected, uid=JOB_UID)
    edit(actual, path, value)
    with pytest.raises(RegionalFixtureError):
        resources.validate_job(actual, expected, uid=JOB_UID)


@pytest.mark.parametrize(
    "status",
    [
        None,
        {"failed": 1},
        {"failed": True},
        {"failed": False},
        {"active": True},
        {"active": -1},
        {"active": 2},
        {"active": 1.0},
        {"succeeded": True},
        {"succeeded": 2},
        {"terminating": 1},
        {"conditions": [{"type": "Failed", "status": "True"}]},
        {"conditions": [{"type": "FailureTarget", "status": "True"}]},
        {"conditions": [{"type": "Suspended", "status": "True"}]},
        {"conditions": [{"type": "Complete", "status": "True"}]},
        {
            "conditions": [{"type": "Complete", "status": "True"}],
            "active": 1,
            "succeeded": 1,
        },
        {"conditions": [{"type": "Complete", "status": True}]},
        {
            "conditions": [
                {"type": "Complete", "status": "Unknown"},
                {"type": "Complete", "status": "False"},
            ]
        },
        {"conditions": [{"type": None, "status": "True"}]},
        {"conditions": [None]},
        {"uncountedTerminatedPods": {"failed": ["old-pod"]}},
        {"uncountedTerminatedPods": {"succeeded": ["old-pod"]}},
        {"uncountedTerminatedPods": {"extra": []}},
    ],
)
def test_job_failure_false_counts_and_incomplete_termination_are_never_proof(
    cpu: Harness, status: Any
) -> None:
    expected = cpu.manifests()[-1]
    actual = admitted(expected, uid=JOB_UID)
    actual["status"] = status
    with pytest.raises(RegionalFixtureError):
        resources.validate_job(actual, expected, uid=JOB_UID)


@pytest.mark.parametrize("phase", ["gated", "running", "succeeded"])
def test_job_lifecycle_requires_phase_specific_counts(
    cpu: Harness, phase: resources.WatchdogPhase
) -> None:
    expected = cpu.manifests()[-1]
    actual = admitted(expected, uid=JOB_UID)
    actual["status"] = {
        "active": 0,
        "ready": 1,
        "succeeded": 0,
        "conditions": [{"type": "Complete", "status": "False"}],
    }
    with pytest.raises(RegionalFixtureError):
        resources.validate_job(actual, expected, uid=JOB_UID, phase=phase)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("spec", "nodeName"), "prebound"),
        (("spec", "schedulingGates"), []),
        (("spec", "schedulingGates"), [{"name": "unrelated"}]),
        (
            ("spec", "schedulingGates"),
            [{"name": resources.SCHEDULING_GATE}, {"name": "extra"}],
        ),
        (("status", "phase"), "Running"),
        (("status", "phase"), "Failed"),
        (("status", "startTime"), STARTED),
        (("status", "podIP"), "127.0.0.1"),
        (("status", "podIPs"), [{"ip": "127.0.0.1"}]),
        (
            ("status", "containerStatuses"),
            [{"name": "cancellation-watchdog", "state": {"waiting": {}}}],
        ),
        (("status", "conditions"), [{"type": "PodScheduled", "status": "True"}]),
        (("status", "conditions"), [{"type": "Ready", "status": "True"}]),
        (("status", "conditions"), [{"type": "DisruptionTarget", "status": "True"}]),
        (("status", "initContainerStatuses"), [{"name": "injected"}]),
        (("status", "ephemeralContainerStatuses"), [{"name": "injected"}]),
    ],
)
def test_gated_pod_must_be_actually_unbound_and_unstarted(
    cpu: Harness, path: tuple[str | int, ...], value: Any
) -> None:
    expected = cpu.manifests()[-1]
    actual = pod(expected)
    edit(actual, path, value)
    with pytest.raises(RegionalFixtureError):
        validate_pod(actual, expected)


def test_missing_gate_and_replaced_controller_are_not_authorized_by_labels(
    cpu: Harness,
) -> None:
    expected = cpu.manifests()[-1]
    actual = pod(expected)
    actual["spec"].pop("schedulingGates")
    with pytest.raises(RegionalFixtureError):
        validate_pod(actual, expected)
    actual = pod(expected)
    actual["metadata"]["ownerReferences"][0]["controller"] = False
    with pytest.raises(RegionalFixtureError):
        validate_pod(actual, expected)
    actual = pod(expected)
    actual["metadata"]["ownerReferences"].append(
        copy.deepcopy(actual["metadata"]["ownerReferences"][0])
    )
    with pytest.raises(RegionalFixtureError):
        validate_pod(actual, expected)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("spec", "nodeName"), ""),
        (("spec", "schedulingGates"), [{"name": resources.SCHEDULING_GATE}]),
        (("metadata", "uid"), "replacement-pod"),
        (
            ("metadata", "labels", "batch.kubernetes.io/controller-uid"),
            "replacement-job",
        ),
        (("status", "phase"), "Unknown"),
        (("status", "conditions"), [{"type": "PodScheduled", "status": "True"}]),
        (("status", "containerStatuses"), []),
        (("status", "containerStatuses", 0, "name"), "foreign"),
        (("status", "containerStatuses", 0, "image"), "cpu:latest"),
        (("status", "containerStatuses", 0, "image"), "sha256:" + "9" * 63),
        (("status", "containerStatuses", 0, "image"), "registry.example/cpu:latest"),
        (("status", "containerStatuses", 0, "imageID"), None),
        (("status", "containerStatuses", 0, "imageID"), "sha256:" + "b" * 64),
        (("status", "containerStatuses", 0, "containerID"), ""),
        (("status", "containerStatuses", 0, "restartCount"), 1),
        (("status", "containerStatuses", 0, "restartCount"), False),
        (
            ("status", "containerStatuses", 0, "lastState"),
            {"terminated": {"exitCode": 0}},
        ),
        (("status", "containerStatuses", 0, "ready"), False),
        (("status", "containerStatuses", 0, "ready"), 1),
        (("status", "containerStatuses", 0, "started"), False),
        (
            ("status", "containerStatuses", 0, "state"),
            {"waiting": {"reason": "CrashLoopBackOff"}},
        ),
        (
            ("status", "containerStatuses", 0, "state", "running", "startedAt"),
            "invalid",
        ),
        (("status", "containerStatuses", 0, "state", "running", "startedAt"), None),
        (
            ("status", "containerStatuses", 0, "state", "running", "startedAt"),
            "2026-09-13T00:00:00",
        ),
    ],
)
def test_running_requires_same_unrestarted_ready_process(
    cpu: Harness, path: tuple[str | int, ...], value: Any
) -> None:
    expected = cpu.manifests()[-1]
    actual = pod(expected, "running")
    edit(actual, path, value)
    with pytest.raises(RegionalFixtureError):
        validate_pod(actual, expected, "running")


@pytest.mark.parametrize(
    "image_id",
    [
        IMAGE,
        "sha256:" + "a" * 64,
        "containerd://sha256:" + "a" * 64,
        "cri-o://sha256:" + "a" * 64,
    ],
)
def test_standard_cri_image_id_forms_keep_the_exact_digest(
    cpu: Harness, image_id: str
) -> None:
    expected = cpu.manifests()[-1]
    actual = pod(expected, "running")
    actual["status"]["containerStatuses"][0]["imageID"] = image_id
    validate_pod(actual, expected, "running")


@pytest.mark.parametrize("status_image", [IMAGE, "sha256:" + "9" * 64])
def test_kubelet_status_image_forms_keep_the_planned_process(
    cpu: Harness, status_image: str
) -> None:
    # containerd 2.x reports a digest-pinned pull without a local tag as the
    # bare image config id (live 2026-09-19); the pulled identity stays proven
    # by ``imageID``, so either form names the planned process.
    expected = cpu.manifests()[-1]
    actual = pod(expected, "running")
    actual["status"]["containerStatuses"][0]["image"] = status_image
    validate_pod(actual, expected, "running")


@pytest.mark.parametrize(
    "change",
    [
        {"exitCode": 1},
        {"exitCode": False},
        {"exitCode": 0.0},
        {"signal": 9},
        {"signal": False},
        {"reason": "Error"},
        {"containerID": "foreign"},
        {"startedAt": None},
        {"finishedAt": None},
        {"startedAt": FINISHED, "finishedAt": STARTED},
    ],
)
def test_successful_termination_requires_real_zero_exit_and_ordered_times(
    cpu: Harness, change: dict[str, Any]
) -> None:
    expected = cpu.manifests()[-1]
    actual = pod(expected, "succeeded")
    actual["status"]["containerStatuses"][0]["state"]["terminated"].update(change)
    with pytest.raises(RegionalFixtureError):
        validate_pod(actual, expected, "succeeded")


@pytest.mark.parametrize(
    "change",
    [
        {"state": {"running": {"startedAt": STARTED}}},
        {"started": True},
        {"ready": True},
    ],
)
def test_succeeded_pod_phase_does_not_replace_process_termination(
    cpu: Harness, change: dict[str, Any]
) -> None:
    expected = cpu.manifests()[-1]
    actual = pod(expected, "succeeded")
    actual["status"]["containerStatuses"][0].update(change)
    with pytest.raises(RegionalFixtureError):
        validate_pod(actual, expected, "succeeded")


@pytest.mark.parametrize("phase", [None, "unknown", True])
def test_pod_phase_must_be_explicit_and_known(cpu: Harness, phase: Any) -> None:
    expected = cpu.manifests()[-1]
    with pytest.raises(RegionalFixtureError):
        validate_pod(pod(expected), expected, phase)


def test_job_rejects_unknown_validation_phase(cpu: Harness) -> None:
    expected = cpu.manifests()[-1]
    unknown: Any = "unknown"
    with pytest.raises(RegionalFixtureError):
        resources.validate_job(
            admitted(expected, uid=JOB_UID), expected, uid=JOB_UID, phase=unknown
        )


@pytest.mark.parametrize("missing", ["job_uid", "pod_uid", "pod_name"])
def test_pod_validation_requires_previously_recorded_identity(
    cpu: Harness, missing: str
) -> None:
    expected = cpu.manifests()[-1]
    arguments = {"job_uid": JOB_UID, "pod_uid": POD_UID, "pod_name": POD_NAME}
    arguments[missing] = ""
    with pytest.raises(RegionalFixtureError):
        resources.validate_pod(pod(expected), expected, phase="gated", **arguments)


def test_unrelated_metadata_never_substitutes_for_spec_validation(cpu: Harness) -> None:
    expected = cpu.manifests()[-1]
    actual = pod(expected)
    actual["metadata"]["annotations"]["example.org/trace"] = "unrelated"
    actual["metadata"]["labels"]["example.org/trace"] = "unrelated"
    validate_pod(actual, expected)
    actual["spec"]["containers"][0]["command"] = ["/bin/sh"]
    with pytest.raises(RegionalFixtureError):
        validate_pod(actual, expected)


@pytest.mark.parametrize(
    "mount_path",
    [
        "etc/gpu-fault/aurora",
        resources.STORE_DIRECTORY + "/",
        "/etc/gpu-fault/other/../aurora",
        "/etc//gpu-fault/aurora",
    ],
)
def test_source_mount_paths_must_be_canonical(cpu: Harness, mount_path: str) -> None:
    cpu.container["volumeMounts"][0]["mountPath"] = mount_path
    with pytest.raises(RegionalFixtureError):
        resources.read_runtime(cpu.regional)


@pytest.mark.parametrize(
    ("resource", "path", "value"),
    [
        (("namespace", NAMESPACE), ("metadata", "uid"), "new-namespace"),
        (("deployment", resources.DEPLOYMENT), ("metadata", "uid"), "new-deployment"),
        (("deployment", resources.DEPLOYMENT), ("metadata", "generation"), 8),
        (("deployment", resources.DEPLOYMENT), ("spec", "replicas"), 1),
        (("deployment", resources.DEPLOYMENT), ("status", "readyReplicas"), 1),
        (("configmap", "worker-postgres"), ("metadata", "uid"), "new-config"),
        (("configmap", "worker-postgres"), ("metadata", "resourceVersion"), "11"),
        (("configmap", resources.CA_CONFIGMAP), ("metadata", "resourceVersion"), "11"),
    ],
)
def test_source_drift_during_discovery_is_not_published_as_a_baseline(
    cpu: Harness,
    monkeypatch: pytest.MonkeyPatch,
    resource: tuple[str, str],
    path: tuple[str, ...],
    value: Any,
) -> None:
    original = cpu.kube

    def kube(plane: str, *args: str, **kwargs: Any) -> str:
        result = original(plane, *args, **kwargs)
        if args[1:3] == ("configmap", resources.CA_CONFIGMAP):
            edit(cpu.objects[resource], path, value)
        return result

    monkeypatch.setattr(cpu.regional, "kubectl", kube)
    with pytest.raises(RegionalFixtureError, match="changed during"):
        resources.read_runtime(cpu.regional)


def test_manifest_metadata_is_not_shared_across_independent_resources(
    cpu: Harness,
) -> None:
    account, role, binding, code, job = cpu.manifests()
    unchanged = copy.deepcopy((role, binding, code, job))
    account["metadata"]["ownerReferences"][0]["uid"] = "changed"
    account["metadata"]["labels"]["new"] = "changed"
    account["metadata"]["annotations"]["new"] = "changed"
    assert (role, binding, code, job) == unchanged, (
        "recording one resource must not mutate another resource's planned binding"
    )


@pytest.mark.parametrize("invalid", [float("nan"), object()])
def test_non_json_admission_fields_fail_closed(cpu: Harness, invalid: Any) -> None:
    expected = cpu.manifests()[-1]
    actual = pod(expected)
    actual["spec"]["priority"] = invalid
    with pytest.raises(RegionalFixtureError):
        validate_pod(actual, expected)


def test_supporting_validator_does_not_accept_a_job(cpu: Harness) -> None:
    expected = cpu.manifests()[-1]
    with pytest.raises(RegionalFixtureError, match="kind is unsupported"):
        resources.validate_supporting_resource(
            admitted(expected, uid=JOB_UID), expected, uid=JOB_UID
        )


@pytest.mark.parametrize(
    "container_change", [{"image": "cpu:latest"}, {"name": "other"}]
)
def test_public_validators_require_immutable_single_watchdog_even_in_expected_shape(
    cpu: Harness, container_change: dict[str, str]
) -> None:
    expected = cpu.manifests()[-1]
    expected["spec"]["template"]["spec"]["containers"][0].update(container_change)
    with pytest.raises(RegionalFixtureError):
        validate_pod(pod(expected), expected)
    with pytest.raises(RegionalFixtureError):
        resources.validate_job(admitted(expected, uid=JOB_UID), expected, uid=JOB_UID)


def test_actual_pod_omitted_empty_defaults_and_job_tracking_metadata(
    cpu: Harness,
) -> None:
    expected = cpu.manifests()[-1]
    actual = pod(expected, "running")
    actual["metadata"]["finalizers"] = ["batch.kubernetes.io/job-tracking"]
    actual["spec"].update(
        nodeSelector={}, imagePullSecrets=[], ephemeralContainers=[], schedulingGates=[]
    )
    container = actual["spec"]["containers"][0]
    container.update(args=[], workingDir="", stdin=False, stdinOnce=False, tty=False)
    validate_pod(actual, expected, "running")


@pytest.mark.parametrize(
    "status",
    [
        {"active": 0, "ready": 1},
        {"active": 1, "succeeded": 1},
        {"conditions": [{"type": "UnknownTerminal", "status": "True"}]},
        {"conditions": [{"type": "Failed", "status": "Unknown"}]},
        {"conditions": [{"type": "SuccessCriteriaMet", "status": "True"}]},
    ],
)
def test_job_unknown_or_contradictory_status_is_not_a_healthy_snapshot(
    cpu: Harness, status: dict[str, Any]
) -> None:
    expected = cpu.manifests()[-1]
    actual = admitted(expected, uid=JOB_UID)
    actual["status"] = status
    with pytest.raises(RegionalFixtureError):
        resources.validate_job(actual, expected, uid=JOB_UID)


def test_success_criteria_condition_is_compatible_with_real_job_completion(
    cpu: Harness,
) -> None:
    expected = cpu.manifests()[-1]
    actual = admitted(expected, uid=JOB_UID)
    actual["status"] = {
        "succeeded": 1,
        "conditions": [
            {"type": "SuccessCriteriaMet", "status": "True"},
            {"type": "Complete", "status": "True"},
        ],
    }
    resources.validate_job(actual, expected, uid=JOB_UID, phase="succeeded")
