from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from gpu_fault.admin import bootstrap_common, bootstrap_services, bootstrap_tasks
from gpu_fault.admin.bootstrap_checkpoint import bind_bootstrap_inputs
from gpu_fault.admin.bootstrap_common import BootstrapRequest, BootstrapState
from gpu_fault.admin.node_key_custody_admin import (
    CustodyPreparationRequired,
    bootstrap_custody_profile,
)
from gpu_fault.admin.node_key_custody_admin_config import load_admin_custody
from tests.admin._node_key_custody_admin_support import AdminWorld
from tests.regional._cov95_identity_support import offline_guard as offline_guard


def state_for(world):
    state = BootstrapState(
        world.state_dir / "bootstrap-state.json", site_id=world.site_id
    )
    for name, value in (
        ("monitoring_resources", {"workspace_id": "workspace"}),
        ("aurora", {}),
    ):
        state.record(name, value)
        state.complete(name)
    return state


@pytest.mark.parametrize("profile_version", ["hyperpod-v1", "existing-profile-v2"])
def test_configured_key_tasks_wait_for_release_and_run_serially_without_blocking_iam(
    tmp_path, monkeypatch, profile_version
):
    world = AdminWorld(tmp_path, monkeypatch)
    world.configure()
    state = state_for(world)
    release_started, release_allowed = threading.Event(), threading.Event()
    first_started, first_allowed, second_started = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    iam_started = threading.Event()
    order = []
    digests = []

    def compute_digest(_runner, *, repository_root, runtime_profile_version):
        assert repository_root == world.repo
        assert runtime_profile_version == profile_version
        digests.append(runtime_profile_version)
        return "f" * 64

    monkeypatch.setattr(bootstrap_common, "compute_agent_config_digest", compute_digest)
    other = replace(
        world.gpu, hyperpod_name="hyperpod-b", eks_arn=world.gpu.eks_arn + "-b"
    )
    monkeypatch.setattr(
        bootstrap_services,
        "ensure_grafana_dashboards",
        lambda *a, **k: {"status": "SKIPPED"},
    )
    monkeypatch.setattr(
        bootstrap_services,
        "install_monitoring",
        lambda *a, **k: iam_started.set() or {},
    )
    monkeypatch.setattr(
        bootstrap_services, "install_aurora_refresh", lambda *a, **k: {}
    )

    def release():
        release_started.set()
        assert release_allowed.wait(5), "test did not release the blocked release task"
        return world.release

    def keys(_runner, **kwargs):
        assert release_allowed.is_set(), "custody ran before the release completed"
        assert kwargs["custody_context"].release_manifest == world.manifest_path
        assert kwargs["custody_context"].agent_config_digest == (
            "e" * 64 if profile_version == "hyperpod-v1" else "f" * 64
        )
        if kwargs["cluster_id"] == world.cluster_id:
            first_started.set()
            assert first_allowed.wait(5), (
                "test did not release the first custody key task"
            )
        else:
            second_started.set()
        order.append(kwargs["cluster_id"])
        return {"cluster_id": kwargs["cluster_id"]}

    monkeypatch.setattr(bootstrap_services, "provision_node_action_keys", keys)
    graph = bootstrap_tasks.platform_task_graph(
        runner=world,
        state=state,
        repository_root=world.repo,
        cpu=world.cpu,
        gpu_clusters=[world.gpu, other],
        cpu_kubeconfig=world.cpu_kubeconfig,
        gpu_kubeconfig=world.gpu_kubeconfig,
        namespace=world.context().namespace,
        site_id=world.site_id,
        alert_email=None,
        release=release,
        fleet_master_file=world.master_file,
        ensure_aurora_ready=lambda *a, **k: {},
        custody_runtime_profile=profile_version,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(graph.run, state=state)
        try:
            assert release_started.wait(5), (
                "release task did not start within the test timeout"
            )
            assert iam_started.wait(5), "unrelated IAM work waited for custody/release"
            assert not first_started.is_set() and not second_started.is_set()
            release_allowed.set()
            assert first_started.wait(5), (
                "first custody key task did not start after release"
            )
            assert not second_started.is_set(), "two custody key maps ran concurrently"
        finally:
            release_allowed.set()
            first_allowed.set()
        running.result(timeout=10)
    assert order == ["hyperpod-a", "hyperpod-b"]
    assert bool(digests) == (profile_version != "hyperpod-v1")


def test_custody_inputs_and_source_closure_invalidate_only_owned_key_checkpoints(
    tmp_path, monkeypatch
):
    world = AdminWorld(tmp_path, monkeypatch)
    world.configure()
    state = state_for(world)
    request = BootstrapRequest(
        cpu_cluster_arn=world.cpu.eks_arn,
        gpu_cluster_arns=(world.gpu.eks_arn,),
        repository_root=world.repo,
        state_dir=world.state_dir,
        alert_email="admin@example.com",
    )
    bind_bootstrap_inputs(
        state, request=request, cpu=world.cpu, gpu_clusters=(world.gpu,)
    )
    before = dict(state.value["task_input_sha256"])
    bind_bootstrap_inputs(
        state,
        request=request,
        cpu=world.cpu,
        gpu_clusters=(world.gpu,),
        release=world.release,
    )
    released = dict(state.value["task_input_sha256"])
    assert released["node_keys:hyperpod-a"] != before["node_keys:hyperpod-a"]
    assert released["aurora"] == before["aurora"]
    source = world.repo / "src/gpu_fault/admin/node_key_custody_admin.py"
    source.write_bytes(source.read_bytes() + b"\n")
    bind_bootstrap_inputs(
        state,
        request=request,
        cpu=world.cpu,
        gpu_clusters=(world.gpu,),
        release=world.release,
    )
    changed = state.value["task_input_sha256"]
    assert changed["node_keys:hyperpod-a"] != released["node_keys:hyperpod-a"]
    assert changed["aurora"] == released["aurora"]
    assert load_admin_custody(world.state_dir) is not None


def test_existing_profile_drift_is_not_pre_authorized_as_a_known_version(
    tmp_path, monkeypatch
):
    world = AdminWorld(tmp_path, monkeypatch)
    world.configure()
    source = world.repo / "current-profile.yaml"
    template = world.repo / "next-profile.yaml"
    source.write_text("claims: []\n")
    template.write_text("claims: [different]\n")
    existing = {
        "spec": {
            "runtimeProfile": {
                "source": str(source),
                "templateSource": str(template),
                "version": "profile-v1",
            }
        }
    }
    with pytest.raises(CustodyPreparationRequired, match="unresolved profile"):
        bootstrap_custody_profile(world.state_dir, world.repo, existing)
    template.write_bytes(source.read_bytes())
    assert (
        bootstrap_custody_profile(world.state_dir, world.repo, existing) == "profile-v1"
    )
    assert bootstrap_custody_profile(world.state_dir, world.repo, None) == "hyperpod-v1"
