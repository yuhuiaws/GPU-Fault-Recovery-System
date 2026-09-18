from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.admin import deploy_host_binding
from gpu_fault.admin.operation_lock import (
    SITE_OPERATION_LOCK_FD_ENV,
    inherited_site_operation_lock_fd,
)
from gpu_fault.admin.site import SiteConfigError
from scripts.e2e.regional import boot032_contract as contract
from scripts.e2e.regional import boot032_lifecycle as lifecycle
from scripts.e2e.regional import boot032_native as adapter
from scripts.e2e.regional import regional_case_contract as cases
from scripts.e2e.regional import run_boot032_full_uninstall as entry
from scripts.e2e.regional.boot032_lifecycle import (
    process_identity as real_process_identity,
)
from tests.regional._cov95_boot032_approval import approve, execute
from tests.regional._cov95_boot032_native import NativeHarness
from tests.regional._cov95_boot032_world import World


def test_accepted_site_bound_interpreter_cannot_plan_a_different_uninstall(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    monkeypatch.setattr(
        deploy_host_binding,
        "bound_deploy_host_state_dir",
        lambda: world.protected.source.parent,
    )
    with pytest.raises(SiteConfigError, match="bound"):
        adapter.NativeBackend(world.settings).initial()
    assert not world.settings.native_dir.exists(), (
        "native installed-state binding must gate planning"
    )


def test_native_state_binding_change_after_approval_blocks_entry(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    harness = NativeHarness(world, monkeypatch)
    _settings, _plan, deadline = approve(world)
    monkeypatch.setattr(
        deploy_host_binding,
        "bound_deploy_host_state_dir",
        lambda: world.protected.source.parent,
    )
    with pytest.raises(SiteConfigError, match="bound"):
        execute(world, deadline)
    assert harness.events == [], (
        "accepted-site-bound interpreter must never enter target native cleanup"
    )


@pytest.mark.parametrize("identity", ["correct", "target-release", "target-cluster"])
def test_canonical_predecessor_is_bound_to_the_protected_site(
    tmp_path, monkeypatch, identity
):
    world = World(tmp_path, monkeypatch)
    predecessor_id = "GF-REGIONAL-BOOT-031"
    order = tmp_path / "formal-order.yaml"
    order.write_text(
        "phases:\n- sequence: 1\n  entries:\n"
        f"  - case: {predecessor_id}\n"
        f"  - case: {contract.CASE_ID}\n    predecessor: {predecessor_id}\n"
    )
    monkeypatch.setattr(cases, "ORDER_PATH", order)
    monkeypatch.setattr(lifecycle, "predecessor_path", cases.predecessor_path)
    cases.expanded_order.cache_clear()
    try:
        binding = adapter.NativeBackend(world.settings).initial()
        cluster = contract.cluster_specs(world.protected)[1]["cluster_id"]
        release = binding["protected"]["runtime"][cluster]["release_state"][
            "release_id"
        ]
        evidence = cases.case_evidence_path(world.root, predecessor_id)
        evidence.parent.mkdir(parents=True)
        evidence.write_text(
            json.dumps(
                {
                    "case_id": predecessor_id,
                    "verdict": "PASS",
                    "status": "COMPLETED",
                    "cluster_id": contract.cluster_specs(world.target)[1]["cluster_id"]
                    if identity == "target-cluster"
                    else cluster,
                    "release_id": "target-release"
                    if identity == "target-release"
                    else release,
                }
            )
        )
        value = lifecycle.predecessor(world.settings, binding)
        assert value["path"] == str(evidence.resolve()), (
            "predecessor path must come only from canonical order"
        )
        assert value["expected_case_id"] == predecessor_id, (
            "operator cannot choose an arbitrary predecessor"
        )
        assert value["valid"] is (identity == "correct"), (
            "sacrificial-site release or cluster evidence cannot stand in for accepted-site proof"
        )
        preflight = lifecycle.read_only_preflight(
            world.settings, world.settings.case_dir
        )
        assert bool(preflight["errors"]) is (identity != "correct"), (
            "canonical predecessor mismatch must block planning"
        )
    finally:
        cases.expanded_order.cache_clear()


def test_parser_has_no_predecessor_or_arbitrary_target_site_override(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    required = [
        "--run-dir",
        str(tmp_path),
        "--fixture-id",
        world.settings.fixture_id,
        "--protected-site",
        str(world.protected.source),
        "--protected-cluster-id",
        world.settings.protected_cluster_id,
    ]
    for extra in (
        ["--predecessor", "another.json"],
        ["--site", str(world.protected.source)],
    ):
        with pytest.raises(SystemExit) as failure:
            entry.parser().parse_args([*required, *extra])
        assert failure.value.code == 2, (
            "case scope and predecessor cannot be selected by extra CLI flags"
        )


def test_entrypoint_delegates_to_the_existing_case_runner(monkeypatch):
    called = []

    def run(case):
        called.append(case)
        return 75

    monkeypatch.setattr(entry, "run_standard_case", run)
    assert entry.main() == 75, (
        "entrypoint must preserve the shared runner's exit status"
    )
    assert (
        called == [entry.CASE] and entry.CASE.execute_case is lifecycle.execute_case
    ), "the case must retain the existing approval/configure/execute wrapper"


def test_execute_without_checked_approval_arguments_is_refused(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    with pytest.raises(contract.UninstallCaseError, match="approval arguments"):
        lifecycle.execute_case(
            world.settings, world.root, 1, datetime.now(timezone.utc)
        )
    assert not world.settings.native_dir.exists(), (
        "direct unapproved execution must not start native work"
    )


def test_actual_local_process_incarnations_differ_after_restart():
    current = real_process_identity()
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from scripts.e2e.regional.boot032_lifecycle import process_identity; print(json.dumps(process_identity()))",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    other = json.loads(child.stdout)
    assert current["pid"] == os.getpid(), (
        "process proof must describe this actual process"
    )
    assert current["boot_id"] == other["boot_id"] and current != other, (
        "a real local child must have a distinct incarnation on the same kernel boot"
    )
    assert current["start_ticks"].isdigit() and other["start_ticks"].isdigit(), (
        "PID reuse protection must use kernel start ticks"
    )


def test_existing_native_lock_fd_is_restored_and_expired_window_refuses(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    monkeypatch.setenv(SITE_OPERATION_LOCK_FD_ENV, "prior-invalid-fd")
    with adapter.checked_locks(
        world.settings, datetime.now(timezone.utc) + timedelta(minutes=1)
    ):
        assert (
            inherited_site_operation_lock_fd(world.target.source.parent) is not None
        ), "native entry must inherit an actually held target lock"
    assert os.environ[SITE_OPERATION_LOCK_FD_ENV] == "prior-invalid-fd", (
        "temporary target FD binding must not leak beyond the case scope"
    )
    with pytest.raises(contract.UninstallCaseError, match="maintenance window"):
        with adapter.checked_locks(
            world.settings, datetime.now(timezone.utc) - timedelta(seconds=1)
        ):
            pytest.fail(
                "expired maintenance window must not enter locked mutation scope"
            )


def test_unbound_checkout_is_not_silently_rebound_by_the_case(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    monkeypatch.setattr(
        deploy_host_binding, "bound_deploy_host_state_dir", lambda: None
    )
    binding = world.settings.native_binding()
    assert binding["installed_state_dir"] is None, (
        "native unbound-checkout policy must remain explicit in approval"
    )
    assert binding["state_dir"] == str(world.target.source.parent), (
        "required target state directory is still canonical"
    )


def test_registered_case_remains_manual_and_uses_canonical_collector_predecessor(
    tmp_path,
):
    metadata = cases.case_metadata(contract.CASE_ID)
    assert metadata.automation == "manual" and metadata.risk == "destructive", (
        "local verification cannot turn the isolated maintenance case into an automated PASS"
    )
    assert metadata.predecessor == "GF-REGIONAL-COLLECT-015", (
        "the reviewed canonical predecessor must remain the accepted-site Collector case"
    )
    predecessor_id, path = cases.predecessor_path(tmp_path, contract.CASE_ID, "")
    assert path == cases.case_evidence_path(tmp_path, predecessor_id).resolve(), (
        "formal predecessor evidence must use the canonical acceptance-run path"
    )
