from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.e2e.regional import ha011_contracts as contracts
from scripts.e2e.regional import ha011_manifests as manifests
from scripts.e2e.regional import ha011_resources as resources
from scripts.e2e.regional import run_ha011_busy_cpu_takeover as runner
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.regional_commands import RegionalCommandTimeout
from tests.regional._cov95_ha011_support import (
    IMAGE_ID,
    INTENT,
    POD_UID,
    Clock,
    deadline,
    install_kubernetes,
    settings_at,
)
from tests.regional._cov95_ha011_support import (
    blocked_external_transports as blocked_external_transports,
)


def prepare(monkeypatch, tmp_path: Path):
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    api = resources.CpuKubernetes(settings)
    identity = api.identity()
    lifetime = resources.IsolatedResources(
        api,
        intent_sha256=INTENT,
        deadline=deadline(),
        cpu_node_identity=identity["cpu_node"],
    )
    objects = manifests.manifests(
        settings,
        runtime_image=identity["runtime_image"],
        bundle="public-fake-bundle",
        password="public-fake-private-database-password",
        cpu_node=identity["cpu_node"],
    )
    return settings, fake, api, lifetime, objects


def test_readback_precedes_arm_then_owned_cleanup_is_uid_conditional(
    monkeypatch, tmp_path: Path
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    assert fake.armed is False, (
        "creating the Pod must not arm either runtime or database"
    )
    lifetime.arm(IMAGE_ID)
    proof, uid = lifetime.collect(IMAGE_ID)
    assert uid == POD_UID and proof["arm_intent_sha256"] == INTENT, (
        "the armed proof must bind the actual Pod UID and resource intent"
    )
    commands = [args[0] for _, args in fake.calls]
    arm_index = commands.index("exec")
    assert all(args[0] != "delete" for _, args in fake.calls[:arm_index]), (
        "pre-arm checks must not touch business or isolated deletion paths"
    )
    assert any(args[:2] == ["get", "Node"] for _, args in fake.calls[:arm_index]), (
        "CPU inventory must be read before arming"
    )
    cleanup = lifetime.cleanup()
    assert cleanup["namespace_absent"] is True and fake.deleted, (
        "owned resources must be confirmed absent"
    )
    assert (
        fake.object("Namespace", settings.namespace)["metadata"]["uid"] == "business-ns"
    ), "business namespace identity must remain untouched"


@pytest.mark.parametrize(
    "change",
    [
        "privileged",
        "host-pid",
        "service-account",
        "extra-env-from",
        "extra-env",
        "extra-volume",
        "extra-mount",
        "extra-init",
        "extra-container",
        "args",
        "security-extra",
        "pod-extra",
        "annotations",
        "wrong-node",
        "namespace",
    ],
)
def test_admission_changes_cannot_reach_arm(
    monkeypatch, tmp_path: Path, change: str
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    modified = False

    def inject(args):
        nonlocal modified
        key = ("Pod", settings.isolated_namespace, contracts.POD_NAME)
        if modified or args[:2] != ["get", "Pod"] or key not in fake.objects:
            return
        modified = True
        pod = fake.objects[key]
        spec = pod["spec"]
        container = spec["containers"][0]
        if change == "privileged":
            container["securityContext"]["privileged"] = True
        elif change == "host-pid":
            spec["hostPID"] = True
        elif change == "service-account":
            spec["serviceAccountName"] = "business-worker"
        elif change == "extra-env-from":
            container["envFrom"] = [{"secretRef": {"name": "production"}}]
        elif change == "extra-env":
            container["env"].append(
                {"name": "GPU_FAULT_STORE_URL", "value": "forbidden-target"}
            )
        elif change == "extra-volume":
            spec["volumes"].append({"name": "host", "hostPath": {"path": "/"}})
        elif change == "extra-mount":
            container["volumeMounts"].append({"name": "host", "mountPath": "/host"})
        elif change == "extra-init":
            spec["initContainers"] = [{"name": "injected", "image": "unexpected"}]
        elif change == "extra-container":
            spec["containers"].append(copy.deepcopy(container))
        elif change == "args":
            container["args"] = ["unapproved"]
        elif change == "security-extra":
            container["securityContext"]["procMount"] = "Unmasked"
        elif change == "pod-extra":
            spec["runtimeClassName"] = "unapproved"
        elif change == "annotations":
            pod["metadata"]["annotations"] = {
                "container.apparmor.security.beta.kubernetes.io/runtime": "unconfined"
            }
        elif change == "wrong-node":
            spec["nodeName"] = "gpu-node"
        else:
            pod["metadata"]["namespace"] = settings.namespace

    fake.before = inject
    with pytest.raises(contracts.ProofError):
        lifetime.start(objects)
    assert not fake.armed, (
        "an altered admitted Pod must never receive the arm operation"
    )
    assert not any(args[0] == "exec" for _, args in fake.calls), (
        "no arm transport may follow failed readback"
    )


@pytest.mark.parametrize("change", ["namespace", "name", "kind", "label"])
def test_public_create_refuses_out_of_scope_targets(
    monkeypatch, tmp_path: Path, change: str
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    pod = copy.deepcopy(objects[-1])
    if change == "namespace":
        pod["metadata"]["namespace"] = settings.namespace
    elif change == "name":
        pod["metadata"]["name"] = "business-worker"
    elif change == "kind":
        pod["kind"] = "Deployment"
    else:
        pod["metadata"]["labels"][contracts.LABEL] = "foreign"
    with pytest.raises(contracts.ProofError, match="boundary"):
        lifetime.create(pod)
    assert not fake.created, "an out-of-bound resource must be refused before transport"


@pytest.mark.parametrize(
    "boundary", ["Namespace", "ConfigMap", "Deployment", "Node", "Lease"]
)
def test_deployment_read_failures_are_not_absence(
    monkeypatch, tmp_path: Path, boundary: str
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)

    def fail(args):
        if args[:2] == ["get", boundary]:
            raise RegionalCommandTimeout(args, 1)

    fake.before = fail
    with pytest.raises(RegionalCommandTimeout):
        runner.read_only_preflight(settings, tmp_path)
    assert not fake.created and not fake.deleted, (
        "read failure must not authorize a resource mutation"
    )


@pytest.mark.parametrize(
    "change",
    [
        "release",
        "containers",
        "image",
        "replicas",
        "readiness",
        "pods",
        "pod-ready",
        "pod-image",
        "pod-deleting",
        "pod-namespace",
        "pod-uid",
        "image-id",
    ],
)
def test_ambiguous_deployed_runtime_is_refused(
    monkeypatch, tmp_path: Path, change: str
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    deployment = fake.object("Deployment", contracts.DEPLOYMENT, business=True)
    if change == "release":
        fake.object("ConfigMap", "gpu-fault-regional-release-state", business=True)[
            "data"
        ]["state.json"] = "{}"
    elif change == "containers":
        deployment["spec"]["template"]["spec"]["containers"] = []
    elif change == "image":
        deployment["spec"]["template"]["spec"]["containers"][0]["image"] = (
            "image:unbound-tag"
        )
    elif change == "replicas":
        deployment["spec"]["replicas"] = 0
    elif change == "readiness":
        deployment["status"]["readyReplicas"] = 0
    elif change == "pods":
        fake.business_pods.clear()
    elif change == "pod-ready":
        fake.business_pods[0]["status"]["containerStatuses"][0]["ready"] = False
    elif change == "pod-image":
        fake.business_pods[0]["spec"]["containers"][0]["image"] = "other"
    elif change == "pod-deleting":
        fake.business_pods[0]["metadata"]["deletionTimestamp"] = "deleting"
    elif change == "pod-namespace":
        fake.business_pods[0]["metadata"]["namespace"] = "foreign"
    elif change == "pod-uid":
        fake.business_pods[0]["metadata"]["uid"] = ""
    else:
        fake.business_pods[0]["status"]["containerStatuses"][0]["imageID"] = ""
    with pytest.raises(contracts.ProofError):
        resources.CpuKubernetes(settings).identity()


@pytest.mark.parametrize(
    "change",
    [
        "node-before-placement",
        "node-before-arm",
        "pod-before-arm",
        "policy",
        "image",
        "restart",
    ],
)
def test_identity_drift_before_arm_preserves_barrier(
    monkeypatch, tmp_path: Path, change: str
) -> None:
    _settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    if change == "node-before-placement":
        fake.object("Node", "cpu-node")["status"]["capacity"]["cpu"] = "16"
        with pytest.raises(contracts.ProofError, match="CPU"):
            lifetime.start(objects)
        assert not fake.created, (
            "a replaced placement inventory must not create a namespace"
        )
        return
    lifetime.start(objects)
    if change == "node-before-arm":
        fake.object("Node", "cpu-node")["status"]["capacity"]["cpu"] = "16"
    elif change == "pod-before-arm":
        fake.object("Pod", contracts.POD_NAME)["metadata"]["uid"] = "new-pod"
    elif change == "policy":
        fake.object("NetworkPolicy", "loopback-only")["spec"]["podSelector"] = {
            "matchLabels": {"unrelated": "true"}
        }
    elif change == "image":
        fake.object("Pod", contracts.POD_NAME)["status"]["containerStatuses"][0][
            "imageID"
        ] = "sha256:other"
    else:
        fake.object("Pod", contracts.POD_NAME)["status"]["containerStatuses"][1][
            "restartCount"
        ] = 1
    with pytest.raises(contracts.ProofError):
        lifetime.arm(IMAGE_ID)
    assert not fake.armed, "pre-arm drift must leave the workload barrier closed"


def test_arm_wait_is_bounded_without_starting_any_work(
    monkeypatch, tmp_path: Path
) -> None:
    _settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    fake.object("Pod", contracts.POD_NAME)["status"] = {}
    monkeypatch.setattr(resources, "time", Clock(step=25))
    with pytest.raises(contracts.ProofError, match="barrier"):
        lifetime.arm(IMAGE_ID)
    assert not fake.armed, (
        "a Pod that never reaches a verified barrier must not execute"
    )


@pytest.mark.parametrize(
    "change", ["wrong-uid", "malformed-ack", "integer-armed", "timeout"]
)
def test_arm_acknowledgement_loss_never_claims_no_work_or_pass(
    monkeypatch, tmp_path: Path, change: str
) -> None:
    _settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    original = fake.dispatch

    def response(args, namespace, text):
        if args[0] == "exec":
            if change == "timeout":
                raise RegionalCommandTimeout(args, 1)
            ack = json.loads(original(args, namespace, text))
            if change == "wrong-uid":
                ack["pod_uid"] = "new-pod"
            elif change == "integer-armed":
                ack["armed"] = 1
            else:
                ack = {}
            return json.dumps(ack)
        return original(args, namespace, text)

    fake.dispatch = response
    with pytest.raises((contracts.ProofError, RegionalCommandTimeout)):
        lifetime.arm(IMAGE_ID)
    assert sum(args[0] == "exec" for _, args in fake.calls) == 1, (
        "an ambiguous arm must never be retried"
    )


@pytest.mark.parametrize("change", ["namespace-uid", "pod-uid", "foreign-label"])
def test_cleanup_does_not_delete_changed_ownership(
    monkeypatch, tmp_path: Path, change: str
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    namespace = fake.object("Namespace", settings.isolated_namespace)
    if change == "namespace-uid":
        namespace["metadata"]["uid"] = "replacement"
    elif change == "pod-uid":
        fake.object("Pod", contracts.POD_NAME)["metadata"]["uid"] = "replacement"
    else:
        namespace["metadata"]["labels"][contracts.LABEL] = "foreign"
    with pytest.raises(contracts.ProofError):
        lifetime.cleanup()
    assert not fake.deleted, (
        "cleanup must not delete a resource whose UID or ownership changed"
    )


def test_uncertain_namespace_create_can_only_cleanup_matching_owned_intent(
    monkeypatch, tmp_path: Path
) -> None:
    _settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    original = fake.dispatch

    def lost_ack(args, namespace, text):
        result = original(args, namespace, text)
        if args[0] == "create":
            raise RegionalCommandTimeout(args, 1)
        return result

    fake.dispatch = lost_ack
    with pytest.raises(RegionalCommandTimeout):
        lifetime.start(objects)
    assert lifetime.namespace_uid is None, (
        "the timeout must not manufacture a create receipt"
    )
    assert lifetime.cleanup()["namespace_absent"] is True, (
        "readback of the fresh run-owned namespace may compensate an uncertain create"
    )
    assert fake.deleted, (
        "verified compensation must use the observed owned namespace UID"
    )


def test_full_runner_public_boundary_succeeds_only_after_verification_and_cleanup(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    case_dir = tmp_path / "cases" / contracts.CASE_ID
    preflight = runner.read_only_preflight(settings, case_dir)
    write_json_atomic(
        case_dir / "plan.json", {"details": runner.plan_details(settings, preflight)}
    )
    assert runner.execute_case(settings, tmp_path, 1, deadline()) == 0, (
        "the fake transport path requires all proof and cleanup guards to pass"
    )
    result = json.loads(capsys.readouterr().out)
    assert result["verdict"] == "PASS" and fake.armed and fake.deleted, (
        "the fake case must cover guarded arm, evidence validation and owned cleanup"
    )
    assert result["probe"]["roles"]["spool"]["replacement_live_before_late"] is True, (
        "the evidence contract must retain the live-replacement interval"
    )
