"""Read-only capability inspection of the actual bound CPU worker population."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.store import PostgresStore
from scripts.e2e.regional import destr008_watchdog_resources as resources
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._destr008_cancellation_controller import CpuApi, build_api, set_path

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def source_environment() -> dict[str, str]:
    """Inspect this checkout even when only the deploy-host wheel is installed."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT)))}


@pytest.fixture
def setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[CpuApi, wire.Plan, resources.CpuRuntime]:
    return build_api(tmp_path, monkeypatch)


def test_all_real_workers_are_inspected_without_constructing_a_store(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, _, runtime = setup

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("feature inspection must never construct a Store")

    monkeypatch.setattr(PostgresStore, "__init__", forbidden)
    resources.require_cpu_history_capability(api.regional, runtime)
    assert {args[2] for verb, args, _ in api.calls if verb == "exec"} == {
        "cpu-worker-0",
        "cpu-worker-1",
    }, "the entire Ready worker population must prove the API"
    assert {verb for verb, _, _ in api.calls} == {"get", "exec"}, (
        "capability verification is read-only and cannot create resources"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"signature": "(self, cluster_id, job_id, attempt_id, *, limit=100)"},
        {"method": "unrelated.Store.list_job_recovery_workflow_incidents"},
        {"probe_sha256": "0" * 64},
        {"sources/gpu_fault.store.postgres.workflows": "0" * 64},
        {"sources/gpu_fault.store.postgres.store": "0" * 64},
        {"sources/gpu_fault.store.contracts": "0" * 64},
    ],
)
def test_old_keyword_or_any_production_source_drift_blocks_arming_before_mutation(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], change: dict[str, Any]
) -> None:
    api, plan, runtime = setup
    api.capability_change = change
    with pytest.raises(RegionalFixtureError, match="API or implementation"):
        api.watchdog(plan, runtime).arm()
    assert not [
        call for call in api.calls if call[0] in {"create", "patch", "delete"}
    ], (
        "an old or drifted CPU worker must fail before any watchdog or GPU fixture action"
    )


@pytest.mark.parametrize(
    ("target", "path", "value"),
    [
        ("pod", "metadata/uid", ""),
        ("pod", "metadata/namespace", "foreign"),
        ("pod", "metadata/deletionTimestamp", "2026-09-13T00:00:00Z"),
        ("pod", "metadata/ownerReferences/0/uid", "replaced-rs"),
        ("pod", "metadata/ownerReferences/0/controller", False),
        ("pod", "spec/containers/0/image", "cpu:old"),
        ("pod", "status/containerStatuses/0/image", "cpu:old"),
        ("pod", "status/containerStatuses/0/image", "sha256:" + "9" * 63),
        ("pod", "status/phase", "Pending"),
        ("pod", "status/conditions/0/status", "False"),
        ("pod", "status/containerStatuses/0/ready", False),
        ("pod", "status/containerStatuses/0/restartCount", False),
        ("pod", "status/containerStatuses/0/restartCount", -1),
        ("pod", "status/containerStatuses/0/imageID", "sha256:" + "b" * 64),
        ("pod", "status/containerStatuses/0/state", {"terminated": {}}),
        ("replicaset", "metadata/ownerReferences/0/uid", "replaced-deployment"),
        ("replicaset", "metadata/ownerReferences/0/controller", False),
        ("replicaset", "metadata/deletionTimestamp", "2026-09-13T00:00:00Z"),
        ("replicaset", "metadata/namespace", "foreign"),
    ],
)
def test_mixed_unready_or_unowned_population_is_not_a_capability_proof(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    target: str,
    path: str,
    value: Any,
) -> None:
    api, _, runtime = setup
    name = "cpu-worker-1" if target == "pod" else resources.DEPLOYMENT + "-rs"
    set_path(api.objects[target, name], path, value)
    with pytest.raises(RegionalFixtureError):
        resources.require_cpu_history_capability(api.regional, runtime)
    assert not [
        call for call in api.calls if call[0] in {"create", "patch", "delete"}
    ], "unknown worker ownership cannot be repaired during capability verification"


@pytest.mark.parametrize("status_image", ["sha256:" + "9" * 64, None])
def test_kubelet_status_image_forms_keep_the_worker_in_the_population(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], status_image: str | None
) -> None:
    # containerd 2.x reports ``containerStatuses[].image`` as the bare image
    # config id for a digest-pinned pull (live 2026-09-19); older kubelets echo
    # the spec reference. Both are the same Ready worker: ``imageID`` proves it.
    api, _, runtime = setup
    pod = api.objects["pod", "cpu-worker-1"]
    if status_image is None:
        status_image = pod["spec"]["containers"][0]["image"]
    set_path(pod, "status/containerStatuses/0/image", status_image)
    resources.require_cpu_history_capability(api.regional, runtime)
    assert {args[2] for verb, args, _ in api.calls if verb == "exec"} == {
        "cpu-worker-0",
        "cpu-worker-1",
    }, "both kubelet image forms must stay in the inspected population"


def test_missing_population_and_duplicate_pod_uids_are_rejected(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, _, runtime = setup
    pod = api.objects.pop(("pod", "cpu-worker-1"))
    with pytest.raises(RegionalFixtureError, match="population is incomplete"):
        resources.require_cpu_history_capability(api.regional, runtime)
    api.objects["pod", "cpu-worker-1"] = pod
    pod["metadata"]["uid"] = "cpu-worker-0-uid"
    with pytest.raises(RegionalFixtureError, match="owner is unproven"):
        resources.require_cpu_history_capability(api.regional, runtime)


@pytest.mark.parametrize("field", ["uid", "restart", "deployment"])
def test_population_is_rechecked_after_exec(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    api, _, runtime = setup
    original = api.kube

    def changed(plane: str, *args: str, **kwargs: Any) -> str:
        response = original(plane, *args, **kwargs)
        if args[0] == "exec":
            if field == "uid":
                api.objects["pod", "cpu-worker-1"]["metadata"]["uid"] = "new-worker-uid"
            elif field == "restart":
                api.objects["pod", "cpu-worker-1"]["status"]["containerStatuses"][0][
                    "restartCount"
                ] += 1
            else:
                api.objects["deployment", resources.DEPLOYMENT]["metadata"][
                    "generation"
                ] += 1
        return response

    monkeypatch.setattr(api.regional, "kubectl", changed)
    with pytest.raises(RegionalFixtureError, match="changed"):
        resources.require_cpu_history_capability(api.regional, runtime)


def test_named_exec_failure_is_sanitized_and_not_retried_in_another_pod(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, _, runtime = setup
    original = api.kube
    attempts = []

    def failed(plane: str, *args: str, **kwargs: Any) -> str:
        if args[0] == "exec":
            attempts.append(args[2])
            raise RuntimeError("private execution diagnostic")
        return original(plane, *args, **kwargs)

    monkeypatch.setattr(api.regional, "kubectl", failed)
    with pytest.raises(RegionalFixtureError) as caught:
        resources.require_cpu_history_capability(api.regional, runtime)
    assert len(attempts) == 1 and "private execution" not in str(caught.value), (
        "exec failure must neither leak diagnostics nor select an unrelated worker"
    )


def test_real_inspection_program_is_hash_guarded_and_matches_local_implementation(
    source_environment: dict[str, str],
) -> None:
    source = resources.history_capability_script()
    digest = hashlib.sha256(source.encode()).hexdigest()
    result = subprocess.run(
        [sys.executable, "-B", "-c", resources.HISTORY_LOADER, digest],
        cwd=ROOT,
        env=source_environment,
        input=source,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == {
        **resources.history_api_identity(),
        "probe_sha256": digest,
    }, (
        "the actual inspection program must report the concrete installed class and source"
    )
    refused = subprocess.run(
        [sys.executable, "-B", "-c", resources.HISTORY_LOADER, "0" * 64],
        cwd=ROOT,
        env=source_environment,
        input=source,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert refused.returncode != 0 and not refused.stdout, (
        "changed inspection code must not run before its source hash is verified"
    )


def test_actual_program_rejects_an_old_method_without_printing_exception_values(
    source_environment: dict[str, str],
) -> None:
    original = resources.history_capability_script()
    prefix = (
        "from gpu_fault.store import PostgresStore\n"
        "def old(self,cluster_id,job_id,attempt_id,*,limit=100):\n"
        " raise AssertionError('must never call Store')\n"
        "old.__module__='gpu_fault.store.postgres.workflows'\n"
        "old.__qualname__='PostgresWorkflowMixin.list_job_recovery_workflow_incidents'\n"
        "PostgresStore.list_job_recovery_workflow_incidents=old\n"
    )
    source = original.replace("try:\n", prefix + "try:\n", 1)
    digest = hashlib.sha256(source.encode()).hexdigest()
    result = subprocess.run(
        [sys.executable, "-B", "-c", resources.HISTORY_LOADER, digest],
        cwd=ROOT,
        env=source_environment,
        input=source,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == {
        "error": "CPU_HISTORY_CAPABILITY_UNAVAILABLE"
    }, "the exact old CPU API must fail without constructing or calling a Store"
