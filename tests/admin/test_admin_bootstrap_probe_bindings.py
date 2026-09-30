"""Bootstrap input-digest and checkpoint binding probes (split from test_admin_bootstrap_probes.py).

These cover bind_bootstrap_inputs, task digests and the prebuild/full bind order; the
probe-vs-ensure tests for individual bootstrap tasks stay in the original module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpu_fault.admin.bootstrap_checkpoint import bind_bootstrap_inputs
from gpu_fault.admin.bootstrap_common import (
    BootstrapRequest,
    BootstrapState,
    run_parallel,
)
from tests.admin.test_admin_bootstrap_probes import (
    _cluster,
    _gpu_cluster,
    _mirror_repository_root,
)


def test_bootstrap_input_digest_invalidates_only_stale_task_checkpoints(
    tmp_path: Path,
) -> None:
    """Each simulated deploy reloads the state from disk, as an earlier process
    looks: the stale-checkpoint rule this test pins is about checkpoints
    inherited from an earlier process."""

    path = tmp_path / "state.json"
    state = BootstrapState(path, site_id="test")
    state.record("pki", {"certificate_arn": "arn:certificate"})
    state.complete("pki")

    task_digests = {"pki": "1" * 64, "aurora": "2" * 64}
    state = BootstrapState(path, site_id="test")
    state.bind_inputs("a" * 64, task_digests)
    assert state.value["completed_tasks"] == [], (
        "a checkpoint with no recorded digest was trusted"
    )
    state.complete("pki")
    state.complete("aurora")

    state = BootstrapState(path, site_id="test")
    state.bind_inputs("a" * 64, task_digests)
    assert state.value["completed_tasks"] == ["aurora", "pki"]

    state = BootstrapState(path, site_id="test")
    state.bind_inputs("b" * 64, {"pki": "1" * 64, "aurora": "3" * 64})
    assert state.value["completed_tasks"] == ["pki"]
    assert state.value["resources"]["pki"] == {"certificate_arn": "arn:certificate"}


def test_a_task_completed_after_the_prebuild_bind_survives_the_full_bind(
    tmp_path: Path,
) -> None:
    """The bind that follows the release build must not erase this run's work.

    ``nlb_network`` and ``pki`` finish minutes before the image build does. The
    pre-build bind records their digests before the graph starts, and the
    post-build bind carries byte-identical digests for them, so a task
    completed between the two binds stays completed. Completion by this
    process is not an exemption of its own: a digest that did change drops the
    checkpoint whoever completed it (a custody fingerprint is provisional until
    the candidate is known; see ``test_admin_bootstrap_early_binding``).
    """

    state = BootstrapState(tmp_path / "state.json", site_id="test")
    state.bind_inputs(None, {"nlb_network": "1" * 64})
    state.record("nlb_network", {"load_balancer_arn": "arn:nlb"})
    state.complete("nlb_network")

    state.bind_inputs("a" * 64, {"nlb_network": "1" * 64, "release": "3" * 64})
    assert state.value["completed_tasks"] == ["nlb_network"], (
        "the full bind erased a task completed against the same digest"
    )
    state.bind_inputs("b" * 64, {"nlb_network": "2" * 64, "release": "3" * 64})
    assert state.value["completed_tasks"] == [], (
        "a changed digest kept a checkpoint because this process completed it"
    )


def test_a_checkpoint_from_an_earlier_process_is_dropped_when_its_digest_changes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    state = BootstrapState(path, site_id="test")
    state.bind_inputs("a" * 64, {"nlb_network": "1" * 64})
    state.record("nlb_network", {"load_balancer_arn": "arn:nlb"})
    state.complete("nlb_network")

    reloaded = BootstrapState(path, site_id="test")
    reloaded.bind_inputs("a" * 64, {"nlb_network": "1" * 64})
    assert reloaded.value["completed_tasks"] == ["nlb_network"], (
        "an inherited checkpoint with an unchanged digest was dropped"
    )
    reloaded.bind_inputs("b" * 64, {"nlb_network": "2" * 64})
    assert reloaded.value["completed_tasks"] == [], (
        "an inherited checkpoint with a changed digest was trusted"
    )


def test_bind_inputs_judges_only_the_tasks_it_names_and_merges_their_digests(
    tmp_path: Path,
) -> None:
    """A partial bind (``digest=None``) leaves the tasks it does not name, and
    their recorded digests, alone -- that is what lets the release-independent
    tasks be bound before the graph starts and the release-dependent ones after
    the build without the second bind undoing the first. ``digest=None``
    likewise leaves the whole-input digest as it was."""

    path = tmp_path / "state.json"
    state = BootstrapState(path, site_id="test")
    state.bind_inputs(
        "a" * 64, {"nlb_network": "1" * 64, "monitoring_install": "2" * 64}
    )
    for name in ("nlb_network", "monitoring_install"):
        state.record(name, {})
        state.complete(name)

    reloaded = BootstrapState(path, site_id="test")
    reloaded.bind_inputs(None, {"nlb_network": "9" * 64})

    assert reloaded.value["completed_tasks"] == ["monitoring_install"], (
        "a task the bind did not name was judged by it"
    )
    assert reloaded.value["task_input_sha256"] == {
        "monitoring_install": "2" * 64,
        "nlb_network": "9" * 64,
    }, "the recorded digest of a task the bind did not name was lost"
    assert reloaded.value["input_sha256"] == "a" * 64, (
        "a bind without a whole-input digest overwrote the recorded one"
    )

    fresh = BootstrapState(tmp_path / "fresh.json", site_id="test")
    fresh.bind_inputs(None, {"nlb_network": "9" * 64})
    assert "input_sha256" not in fresh.value, (
        "a bind without a whole-input digest invented one"
    )


def test_the_legacy_whole_input_bind_clears_every_checkpoint_on_a_changed_digest(
    tmp_path: Path,
) -> None:
    """The pre-v3 shape has no per-task digests, so nothing can prove which
    checkpoint is still current: a changed whole-input digest clears them all,
    this process's own included."""

    path = tmp_path / "state.json"
    state = BootstrapState(path, site_id="test")
    state.bind_inputs("a" * 64)
    state.record("pki", {})
    state.complete("pki")

    reloaded = BootstrapState(path, site_id="test")
    reloaded.record("aurora", {})
    reloaded.complete("aurora")
    reloaded.bind_inputs("a" * 64)
    assert reloaded.value["completed_tasks"] == ["aurora", "pki"], (
        "an unchanged whole-input digest dropped checkpoints"
    )
    reloaded.bind_inputs("b" * 64)
    assert reloaded.value["completed_tasks"] == [], (
        "the legacy bind kept a checkpoint across a changed whole-input digest"
    )
    assert reloaded.value["input_sha256"] == "b" * 64


# What the pre-build bind settles: every task whose digest is fixed before the
# release is built. ``release`` needs the manifest and is left to the full bind.
FOUNDATION_BOUND_TASKS = (
    "release_repositories",
    "pod_identity_agent",
    "nlb_network",
    "pki",
    "aurora",
    "aurora_ready",
    "load_balancer_controller",
    "control_plane_role",
    "email_notifications",
    "monitoring_resources",
    "node_keys:gpu-a",
    "executor_role:gpu-a",
    "adot_writer_role:gpu-a",
)


def _foundation_request(tmp_path: Path) -> tuple[BootstrapRequest, dict[str, Any]]:
    """A request over a mirrored repository root and a state directory that
    already holds the admin config, plus the release the full bind digests."""

    root = _mirror_repository_root(tmp_path, "repository")
    state_dir = tmp_path / "state"
    (state_dir / "admin-config").mkdir(parents=True)
    (state_dir / "admin-config/desired.json").write_text(
        json.dumps({"schema_version": 1, "config": {}}), encoding="utf-8"
    )
    wheel = tmp_path / "gpu_fault-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel-bytes")
    manifest = state_dir / "release.json"
    manifest.write_text(
        json.dumps({"release_id": "0100-a", "wheel": str(wheel)}), encoding="utf-8"
    )
    request = BootstrapRequest(
        cpu_cluster_arn="arn:aws:eks:us-east-1:123456789012:cluster/control",
        gpu_cluster_arns=("arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",),
        repository_root=root,
        state_dir=state_dir,
        alert_email="ops@example.com",
    )
    release = {
        "release_id": "0100-a",
        "manifest": str(manifest),
        "agent_config_digest": "c" * 64,
        "images": {"runtime": "runtime@sha256:aaa", "adot": "adot@sha256:bbb"},
    }
    return request, release


def test_the_prebuild_bind_settles_the_release_independent_digests_first(
    tmp_path: Path,
) -> None:
    """Before the build: every digest that does not need the release, computed
    exactly as the full bind computes it (``release`` itself stays unbound);
    after the build: the full bind adds the release digest and keeps what the
    graph completed in between."""

    request, release = _foundation_request(tmp_path)
    state = BootstrapState(request.state_dir / "bootstrap-state.json", site_id="site-a")

    bind_bootstrap_inputs(
        state, request=request, cpu=_cluster(), gpu_clusters=(_gpu_cluster(),)
    )
    foundation = dict(state.value["task_input_sha256"])

    assert set(FOUNDATION_BOUND_TASKS) <= set(foundation), (
        "a release-independent task was left unbound before the graph"
    )
    assert "release" not in foundation, (
        "the pre-build bind invented a digest for the unbuilt release"
    )

    state.record("nlb_network", {"load_balancer_arn": "arn:nlb"})
    state.complete("nlb_network")
    bind_bootstrap_inputs(
        state,
        request=request,
        cpu=_cluster(),
        gpu_clusters=(_gpu_cluster(),),
        release=release,
    )
    full = dict(state.value["task_input_sha256"])

    assert {name: full[name] for name in FOUNDATION_BOUND_TASKS} == {
        name: foundation[name] for name in FOUNDATION_BOUND_TASKS
    }, "the two binds digest a release-independent task differently"
    assert "release" in full
    assert state.value["completed_tasks"] == ["nlb_network"], (
        "the full bind erased a task completed between the two binds"
    )


def test_a_stale_foundation_checkpoint_reruns_in_the_deploy_that_changed_its_inputs(
    tmp_path: Path,
) -> None:
    """``run_parallel`` reads ``completed_tasks`` once, when the graph starts.
    Bound only after the image build, a changed digest reached the state
    minutes too late and the stale checkpoint was trusted for one more deploy;
    bound before the graph, the task runs again now."""

    request, _release = _foundation_request(tmp_path)
    path = request.state_dir / "bootstrap-state.json"
    previous = BootstrapState(path, site_id="site-a")
    previous.bind_inputs("a" * 64, {"nlb_network": "0" * 64})
    previous.record("nlb_network", {"load_balancer_arn": "arn:previous"})
    previous.complete("nlb_network")
    ran: list[str] = []

    def ensure_nlb_network() -> dict[str, str]:
        ran.append("nlb_network")
        return {"load_balancer_arn": "arn:new"}

    trusted = BootstrapState(path, site_id="site-a")
    assert run_parallel({"nlb_network": ensure_nlb_network}, state=trusted) == {
        "nlb_network": {"load_balancer_arn": "arn:previous"}
    }
    assert ran == [], "the unbound graph did not trust the checkpoint"

    state = BootstrapState(path, site_id="site-a")
    bind_bootstrap_inputs(
        state, request=request, cpu=_cluster(), gpu_clusters=(_gpu_cluster(),)
    )
    assert run_parallel({"nlb_network": ensure_nlb_network}, state=state) == {
        "nlb_network": {"load_balancer_arn": "arn:new"}
    }
    assert ran == ["nlb_network"], (
        "a checkpoint whose inputs changed was trusted after the pre-build bind"
    )
