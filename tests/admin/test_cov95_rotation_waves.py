from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import release_engine
from gpu_fault.admin import rotate_token as rotation
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from gpu_fault_release import repository_root
from gpu_fault_release.regional_release_config import ReleaseError
from tests.admin.test_admin_rotate_token import Harness
from tests.admin.test_admin_site import site_file
from tests.admin.test_cov95_rotate_token import pause


class WaveTransport:
    dry_run = False

    def __init__(self):
        self.calls = []
        self.data = {"allowed-nodes": "*", "generation": "steady"}
        self.ignore_patch = False

    def run(self, arguments, **_options):
        self.calls.append(list(arguments))
        if "patch" in arguments and "configmap" in arguments and not self.ignore_patch:
            self.data = json.loads(arguments[arguments.index("-p") + 1])["data"]
        return json.dumps({"data": self.data})


@pytest.fixture
def context(tmp_path, monkeypatch):
    site = load_site(site_file(tmp_path))
    transport = WaveTransport()
    monkeypatch.setattr(release_engine, "Runner", lambda: transport)
    release = rotation.build_release(site)
    assert release.runner is transport, (
        "the real neutral release factory must use the offline wave transport"
    )
    return SimpleNamespace(
        site=site,
        release=release,
        target=release.config.clusters[0],
        transport=transport,
        events=[],
        names=("node-a", "node-b"),
        environment={rotation.INSTALLER_WAVE_CONFIG_MAP_ENV: "wave-example"},
        handoff=True,
    )


def install_wave_reads(context, monkeypatch):
    monkeypatch.setattr(rotation, "target_node_names", lambda *_args: context.names)
    monkeypatch.setattr(
        rotation,
        "validate_target_node_state",
        lambda *_args: context.events.append("validate"),
    )
    monkeypatch.setattr(
        rotation, "reconciler_container_env", lambda *_args: context.environment
    )
    monkeypatch.setattr(
        rotation, "reconciler_installer_identity", lambda _env: ("b" * 64, "c" * 64)
    )
    monkeypatch.setattr(
        rotation,
        "target_node_failure_domains",
        lambda _release, _target, names: {name: "rack-example" for name in names},
    )
    monkeypatch.setattr(
        rotation,
        "node_rollout_policy",
        lambda *_args, **_options: rotation.NodeRolloutPolicy(
            max_unavailable=1,
            first_wave_max_unavailable=1,
            max_unavailable_per_failure_domain=1,
        ),
    )
    monkeypatch.setattr(
        rotation,
        "ensure_rollout_wave_safe",
        lambda *_args, **_options: context.events.append("safe"),
    )
    monkeypatch.setattr(
        rotation,
        "wait_agents",
        lambda *_args, **_options: context.events.append("agents"),
    )

    def handoff(_release, _target, _wave_context, wave):
        context.events.append("handoff")
        context.transport.data["allowed-nodes"] = ",".join(wave)
        return ("b" * 64, "c" * 64) if context.handoff else None

    monkeypatch.setattr(rotation, "hand_wave_to_reconciler", handoff)


def test_rotation_restarts_every_secret_consumer_before_waiting(context, monkeypatch):
    waits = []
    monkeypatch.setattr(
        rotation,
        "wait_deployment_rollout",
        lambda _release, _target, name, **options: waits.append(
            (name, len(context.transport.calls), options["timeout_seconds"])
        ),
    )
    result = rotation.restart_data_plane(context.release, context.target)
    assert result["deployments"] == list(rotation.inventory.DEPLOYMENTS)
    assert [name for name, _count, _timeout in waits] == list(
        rotation.inventory.DEPLOYMENTS
    )
    assert all(count == len(waits) for _name, count, _timeout in waits), (
        "rotation waited before all connection-secret consumers restarted"
    )


def test_control_plane_secret_consumers_are_the_manifests_that_mount_it():
    """The rolled set is exactly the CPU Deployments that read the registry Secret."""

    import yaml

    generated = repository_root() / "deploy/control-plane/regional/generated"
    consumers = set()
    for manifest in sorted(generated.glob("*.yaml")):
        for document in yaml.safe_load_all(manifest.read_text(encoding="utf-8")):
            if not isinstance(document, dict) or document.get("kind") != "Deployment":
                continue
            if "gpu-fault-regional-clusters" in json.dumps(document):
                consumers.add(document["metadata"]["name"])
    assert consumers == set(rotation.CONTROL_PLANE_SECRET_CONSUMERS), (
        "rotate-token must roll every CPU Deployment that loads the registry Secret"
    )
    assert consumers == {
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
    }


def test_rotation_restarts_each_control_plane_consumer_then_waits_for_it(
    context, monkeypatch
):
    waits = []

    def wait(release, *, kubectl, scope, deployment_name, timeout_seconds):
        waits.append(
            (
                deployment_name,
                len(context.transport.calls),
                timeout_seconds,
                scope,
                kubectl("-n", "gpu-fault-system")[:3],
            )
        )
        return {
            "deployment": deployment_name,
            "duration_seconds": 1.5,
            "progress": [2, 1, 1, 1],
            "object": {"kind": "Deployment"},
        }

    monkeypatch.setattr(rotation, "wait_deployment_rollout_with", wait)
    result = rotation.restart_control_plane(context.release)

    names = list(rotation.CONTROL_PLANE_SECRET_CONSUMERS)
    restarts = [call for call in context.transport.calls if "restart" in call]
    assert [call[-1] for call in restarts] == [f"deployment/{name}" for name in names]
    cpu_prefix = context.release._cpu()
    assert all(call[: len(cpu_prefix)] == cpu_prefix for call in restarts), (
        "control-plane restarts go through the CPU kubeconfig, not a GPU context"
    )
    assert all("--context" not in call for call in restarts), (
        "a GPU context must never be used for the control plane"
    )
    assert [name for name, *_rest in waits] == names
    assert [count for _name, count, *_rest in waits] == list(
        range(1, len(names) + 1)
    ), "each Deployment is restarted and waited for before the next one restarts"
    assert all(
        timeout == rotation.DEPLOYMENT_ROLLOUT_TIMEOUT_SECONDS
        and scope == "control plane"
        and prefix == cpu_prefix
        for _name, _count, timeout, scope, prefix in waits
    ), (
        "every wait uses the rotation timeout, the control-plane scope and the CPU kubectl"
    )
    assert result["deployments"] == names
    assert result["rollouts"][names[0]] == {
        "duration_seconds": 1.5,
        "progress": [2, 1, 1, 1],
    }, "the journal keeps the rollout outcome, not the Deployment object"


def test_control_plane_rollout_failure_stops_before_the_next_restart(
    context, monkeypatch
):
    def wait(release, *, kubectl, scope, deployment_name, timeout_seconds):
        raise ReleaseError(f"{scope} Deployment {deployment_name} exceeded 600 seconds")

    monkeypatch.setattr(rotation, "wait_deployment_rollout_with", wait)
    with pytest.raises(ReleaseError, match="control plane Deployment .* exceeded"):
        rotation.restart_control_plane(context.release)
    restarts = [call for call in context.transport.calls if "restart" in call]
    assert len(restarts) == 1, (
        "a rollout that did not complete leaves the remaining Deployments alone"
    )


@pytest.mark.parametrize("drift", [False, True])
def test_wave_restore_requires_readback_of_every_steady_field(context, drift):
    context.transport.ignore_patch = drift
    steady = {"allowed-nodes": "*", "generation": "saved"}
    if drift:
        with pytest.raises(BootstrapError, match="did not return to its steady state"):
            rotation.restore_wave_config(
                context.release, context.target, "wave-example", steady
            )
    else:
        rotation.restore_wave_config(
            context.release, context.target, "wave-example", steady
        )
        assert (
            rotation.read_wave_config(context.release, context.target, "wave-example")
            == steady
        )
    assert any("patch" in call for call in context.transport.calls), (
        "wave restoration never reached the fake ConfigMap transport"
    )


@pytest.mark.parametrize("failure", ["inventory-drift", "missing-wave", "handoff"])
def test_rotation_wave_refusals_never_mark_nodes_completed(
    context, monkeypatch, failure
):
    install_wave_reads(context, monkeypatch)
    progress = {"completed_nodes": []}
    if failure == "inventory-drift":
        progress["node_names"] = ["different-node"]
    elif failure == "missing-wave":
        context.environment.clear()
    else:
        context.handoff = False
    with pytest.raises(
        BootstrapError, match="inventory changed|predates|did not accept"
    ):
        rotation.roll_node_tokens(
            context.release,
            context.target,
            rotation_id="example",
            progress=progress,
            record=lambda: None,
        )
    assert progress["completed_nodes"] == []
    assert not any("annotate" in call for call in context.transport.calls), (
        "refused wave marked nodes for reinstallation"
    )


@pytest.mark.parametrize(
    "completed,only_nodes", [(["node-a", "node-b"], None), ([], frozenset({"node-b"}))]
)
def test_rotation_reuses_steady_snapshot_and_respects_rollback_node_scope(
    context, monkeypatch, completed, only_nodes
):
    install_wave_reads(context, monkeypatch)
    progress = {
        "node_names": list(context.names),
        "completed_nodes": completed,
        "steady_wave_config": dict(context.transport.data),
    }
    result = rotation.roll_node_tokens(
        context.release,
        context.target,
        rotation_id="example",
        progress=progress,
        record=lambda: None,
        only_nodes=only_nodes,
    )
    assert result["reinstalled_nodes"] == (
        ["node-a", "node-b"] if completed else ["node-b"]
    )
    assert result["waves"] == ([] if completed else [["node-b"]])
    assert context.transport.data == {"allowed-nodes": "*", "generation": "steady"}
    assert context.events[-1] == "agents"


@pytest.mark.parametrize(
    "existing", ["stale-temporary", "matching-retired", "foreign-retired"]
)
def test_atomic_token_replace_handles_only_owned_matching_retry_files(
    tmp_path, existing
):
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    path = tmp_path / "token"
    path.write_text("a" * 64)
    path.chmod(0o600)
    retired = tmp_path / "token.retired-20260101T000000Z"
    temporary = tmp_path / ".token.rotating"
    if existing == "stale-temporary":
        temporary.write_text("incomplete")
    elif existing == "matching-retired":
        os.link(path, retired)
    else:
        retired.write_text("c" * 64)
    if existing == "foreign-retired":
        with pytest.raises(BootstrapError, match="retired token file conflicts"):
            rotation.write_token_file(path, "b" * 64, now=stamp)
        assert path.read_text() == "a" * 64
        assert retired.read_text() == "c" * 64
    else:
        assert rotation.write_token_file(path, "b" * 64, now=stamp) == retired
        assert retired.read_text() == "a" * 64
        assert path.read_text() == "b" * 64
        assert not temporary.exists(), (
            "atomic token replacement left its temporary file"
        )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    return Harness(tmp_path, monkeypatch)


def test_rotation_records_growing_terminal_command_counters(harness, monkeypatch):
    monkeypatch.setattr(
        rotation,
        "remote_command_stats",
        lambda _release: {
            "by_status": {
                "SUCCEEDED": 4,
                "FAILED": 2
                if any(call[0] == "acceptance" for call in harness.calls)
                else 1,
            }
        },
    )
    harness.rotate()
    assert any(
        "terminal counters grew" in warning for warning in harness.state["warnings"]
    ), "rotation suppressed newly failed remote commands from its audit"


def test_early_unprepared_rotation_can_rollback_without_any_data_plane_action(
    harness, monkeypatch
):
    def unavailable(_release):
        raise BootstrapError("example initial command status unavailable")

    with monkeypatch.context() as failure:
        failure.setattr(rotation, "remote_command_stats", unavailable)
        with pytest.raises(BootstrapError, match="initial command status unavailable"):
            harness.rotate()
    assert rotation.STEP_PREPARED not in harness.state["steps"]
    result = harness.rotate(rollback=True)
    assert result["status"] == rotation.STATUS_ROLLED_BACK
    assert harness.calls == []


def test_terminal_rollback_resumes_only_pending_local_cleanup(harness):
    pause(harness)
    harness.rotate(rollback=True)
    state = harness.state
    state["pending_token_cleanup_completed"] = False
    rotation.rotation_state_path(harness.site, "gpu-a").write_text(json.dumps(state))
    before = list(harness.calls)
    assert harness.rotate(rollback=True)["status"] == rotation.STATUS_ROLLED_BACK
    assert harness.calls == before
    assert harness.state["pending_token_cleanup_completed"] is True


def test_token_write_ack_loss_cannot_resume_with_replaced_retired_evidence(harness):
    state = pause(harness)
    stamp = harness.now()
    pending = Path(state["pending_token_file"]).read_text()
    retired = rotation.write_token_file(harness.token_file, pending, now=stamp)
    retired.write_text("e" * 64)
    state.setdefault("started_steps", {})[rotation.STEP_TOKEN_FILE_WRITTEN] = (
        stamp.isoformat()
    )
    rotation.rotation_state_path(harness.site, "gpu-a").write_text(json.dumps(state))
    before = list(harness.calls)
    with pytest.raises(BootstrapError, match="retired token file no longer matches"):
        harness.rotate()
    assert harness.calls == before
