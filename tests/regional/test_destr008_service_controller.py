from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import destr008_service_window as controller
from scripts.e2e.regional import warm_spare_fixture as fixture
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture, HostProbeSettings
from scripts.e2e.regional.probes import warm_spare_node_probe as probe
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._destr008_service_window import CASE, PINS, Host, Transport, write


def build(transport: Transport, **overrides: Any) -> controller.ServiceWindowController:
    kwargs = {
        "cluster_id": PINS["cluster_id"],
        "node_uid": PINS["node_uid"],
        "plan_sha256": "d" * 64,
        "release_id": "test-release",
        "maintenance_expires_at": datetime.fromtimestamp(2_000_001_200, timezone.utc),
        "read_node_uid": lambda: PINS["node_uid"],
        **overrides,
    }
    return controller.ServiceWindowController(
        cast(HostProbeFixture, transport), **kwargs
    )


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    host = Host(tmp_path, monkeypatch)
    transport = Transport(host, tmp_path / "controller")
    window = build(transport)
    yield SimpleNamespace(host=host, transport=transport, window=window)
    window.close()


def test_controller_persists_binding_before_stop_and_closes_only_after_both_cleanups(
    setup: Any,
) -> None:
    window = setup.window
    assert window.service == "" and window.failsafe_at is None
    window.create()
    assert window.restore()["phase"] == "NOT_STOPPED"
    result = window.stop("kubelet.service", delay_seconds=15)
    assert result["phase"] == "SCHEDULED"
    assert window.service == "kubelet.service"
    assert window.failsafe_at == datetime.fromtimestamp(2_000_000_195, timezone.utc)
    saved = probe.private_read(window.path)
    assert saved["binding"]["expires_at"] == 2_000_000_375
    assert (
        saved["phase"] == "STOP_REQUESTED" and saved["host_proof"]["owner"] == "e" * 32
    )
    assert window.path.stat().st_mode & 0o077 == 0
    setup.host.fire_stop()
    assert window.restore()["phase"] == "RESTORED"
    residuals = window.cleanup()
    assert all(v is False for v in residuals.values()), residuals
    saved = probe.private_read(window.path)
    assert saved["phase"] == "CLOSED" and saved["host_cleanup"]["phase"] == "CLOSED"
    assert setup.transport.commands == [
        "snapshot",
        "prepare",
        "status",
        "stop-with-failsafe",
        "restore-service",
        "cleanup",
    ]
    assert setup.transport.creates == setup.transport.cleanups == 1
    before = list(setup.transport.commands)
    fresh = build(setup.transport)
    assert fresh.resume_cleanup()["phase"] == "CLOSED"
    assert setup.transport.commands == before and setup.transport.creates == 1


@pytest.mark.parametrize(
    "lost",
    [
        "prepare",
        "stop-with-failsafe",
        "restore-service",
        "cleanup",
        "transport-cleanup",
    ],
)
def test_lost_reply_fresh_controller_only_restores_original_owned_window(
    setup: Any, lost: str
) -> None:
    window = setup.window
    window.create()
    if lost not in {"prepare", "stop-with-failsafe"}:
        window.stop("gpu-fault-node-agent.service")
        setup.host.fire_stop()
    setup.transport.lost_ack = lost
    with pytest.raises(TimeoutError):
        if lost in {"prepare", "stop-with-failsafe"}:
            window.stop("gpu-fault-node-agent.service")
        elif lost == "restore-service":
            window.restore()
        else:
            window.cleanup()
    original = copy.deepcopy(window.binding)
    window.close()
    fresh = build(setup.transport)
    try:
        result = fresh.resume_cleanup()
        assert result["phase"] == "CLOSED" and result["resumed"] is True
        assert result["host"]["binding_sha256"] == probe.binding_key(original)
        assert fresh.binding == original
        with pytest.raises(RegionalFixtureError, match="cleanup only"):
            fresh.stop("kubelet.service")
        with pytest.raises(RegionalFixtureError, match="cleanup only"):
            fresh.create()
    finally:
        fresh.close()
    assert setup.transport.creates == 1
    assert setup.transport.commands.count("prepare") == 1
    assert setup.transport.commands.count("stop-with-failsafe") <= 1
    assert setup.host.units[original["service"]]["ActiveState"] == "active"


def test_missing_independent_ack_does_not_authorize_stop(setup: Any) -> None:
    setup.window.create()
    setup.transport.arm_ack = False
    with pytest.raises(RegionalFixtureError, match="ACK was not observed"):
        setup.window.stop("kubelet.service")
    assert "stop-with-failsafe" not in setup.transport.commands
    assert setup.window.cleanup() == setup.transport.residual


@pytest.mark.parametrize("operation", ["create", "stop", "cleanup"])
def test_supervision_loss_is_durable_and_fresh_process_cannot_resume_commands(
    setup: Any, operation: str
) -> None:
    if operation != "create":
        setup.window.create()
    if operation == "cleanup":
        setup.window.stop("kubelet.service")

    def lost(*_args: Any, **_kwargs: Any) -> Any:
        raise ProcessSupervisionLost("owned test supervision lost")

    if operation == "create":
        setup.transport.create = lost
    else:
        setup.transport.execute = lost
    with pytest.raises(ProcessSupervisionLost):
        if operation == "stop":
            setup.window.stop("kubelet.service")
        else:
            getattr(setup.window, operation)()
    assert probe.private_read(setup.window.path)["supervision_lost"] is True
    with pytest.raises(RegionalFixtureError, match="supervision"):
        setup.window.resume_cleanup()
    setup.window.close()
    before = list(setup.transport.commands)
    fresh = build(setup.transport)
    with pytest.raises(RegionalFixtureError, match="supervision"):
        fresh.resume_cleanup()
    assert setup.transport.commands == before


@pytest.mark.parametrize(
    "field", ["scope", "owner", "schema_version", "phase", "supervision_lost"]
)
def test_corrupted_controller_proof_refuses_before_any_remote_request(
    setup: Any, field: str
) -> None:
    setup.window.create()
    data = probe.private_read(setup.window.path)
    data[field] = {}
    probe.write_record(setup.window.path, data)
    setup.window.close()
    fresh = build(setup.transport)
    with pytest.raises(RegionalFixtureError, match="identity or state"):
        fresh.resume_cleanup()
    assert setup.transport.commands == []


@pytest.mark.parametrize(
    "field", ["owner", "node_uid", "scope", "resources", "closed", "script_may_exist"]
)
def test_changed_host_proof_cannot_be_reconstructed_from_service_name(
    setup: Any, field: str
) -> None:
    setup.window.create()
    setup.window.stop("kubelet.service")
    data = probe.private_read(setup.transport.state_path)
    data[field] = "unknown"
    probe.write_record(setup.transport.state_path, data)
    setup.window.close()
    before = list(setup.transport.commands)
    fresh = build(setup.transport)
    try:
        with pytest.raises(RegionalFixtureError, match="host transport|owner|receipt"):
            fresh.resume_cleanup()
    finally:
        fresh.close()
    assert setup.transport.commands == before and setup.transport.cleanups == 0


@pytest.mark.parametrize(
    "changed", ["source", "kubeconfig", "release", "plan", "node", "deadline"]
)
def test_mutable_identity_cannot_select_a_new_journal(setup: Any, changed: str) -> None:
    setup.window.create()
    setup.window.close()
    kwargs: dict[str, Any] = {}
    if changed == "source":
        setup.window.scope["controller_sha256"] = "f" * 64
        data = probe.private_read(setup.window.path)
        data["scope"] = setup.window.scope
        probe.write_record(setup.window.path, data)
    elif changed == "kubeconfig":
        setup.transport.settings.kubeconfig.write_text("changed test config")
    elif changed == "release":
        kwargs["release_id"] = "other-release"
    elif changed == "plan":
        kwargs["plan_sha256"] = "f" * 64
    elif changed == "node":
        kwargs["node_uid"] = "replacement-uid"
    else:
        kwargs["maintenance_expires_at"] = datetime.fromtimestamp(
            2_000_001_201, timezone.utc
        )
    fresh = build(setup.transport, **kwargs)
    assert fresh.path == setup.window.path
    with pytest.raises(RegionalFixtureError, match="identity"):
        fresh.resume_cleanup()
    assert setup.transport.commands == []


@pytest.mark.parametrize(
    "residuals",
    [
        None,
        {},
        {"pod/test-owned-pod": False},
        {"pod/test-owned-pod": False, "configmap/test-owned-configmap": "false"},
        {"pod/test-owned-pod": True, "configmap/test-owned-configmap": False},
    ],
)
def test_unknown_or_nonempty_transport_cleanup_never_closes(
    setup: Any, residuals: Any
) -> None:
    setup.window.create()
    setup.window.stop("kubelet.service")
    setup.transport.cleanup = lambda: residuals
    with pytest.raises(RegionalFixtureError, match="unknown or remaining"):
        setup.window.cleanup()
    assert probe.private_read(setup.window.path)["phase"] == "CLEANUP_REQUIRED"


@pytest.mark.parametrize(
    "defect",
    ["phase", "binding", "service", "timer", "start-intent", "generic-boolean"],
)
def test_controller_requires_structured_bound_ack(setup: Any, defect: str) -> None:
    setup.window.create()
    original = setup.transport.execute

    def execute(command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = original(command, *args, **kwargs)
        if command != "prepare":
            return result
        if defect == "generic-boolean":
            return {"ready": True}
        key = {
            "phase": "phase",
            "binding": "binding_sha256",
            "service": "service",
            "timer": "stop_unit",
            "start-intent": "start_intent",
        }[defect]
        result[key] = "foreign"
        return result

    setup.transport.execute = execute
    with pytest.raises(RegionalFixtureError, match="unbound or incomplete"):
        setup.window.stop("kubelet.service")
    assert "stop-with-failsafe" not in setup.transport.commands


@pytest.mark.parametrize("stage", ["create", "before-create", "missing-proof"])
def test_cleanup_without_binding_never_guesses_a_service(
    setup: Any, stage: str
) -> None:
    if stage == "before-create":
        with pytest.raises(RegionalFixtureError, match="controller journal"):
            setup.window.resume_cleanup()
        assert setup.transport.creates == setup.transport.cleanups == 0
        return
    setup.transport.lost_ack = "create"
    with pytest.raises(TimeoutError):
        setup.window.create()
    setup.transport.lost_ack = ""
    if stage == "missing-proof":
        setup.transport.state_path.unlink()
    setup.window.close()
    fresh = build(setup.transport)
    assert fresh.resume_cleanup()["host"] == {"phase": "NOT_REQUESTED"}
    assert setup.transport.commands == [] and setup.transport.creates == 1


def test_fixture_facade_resume_uses_journals_and_never_sets_service_name(
    setup: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = setup.transport
    settings = transport.settings
    warm = SimpleNamespace(
        regional=SimpleNamespace(
            settings=SimpleNamespace(
                gpu_kubeconfig=settings.kubeconfig,
                gpu_context=settings.context,
                namespace=settings.namespace,
                cluster_id=PINS["cluster_id"],
            )
        ),
        node_snapshot=lambda _node: {"uid": PINS["node_uid"]},
    )
    monkeypatch.setattr(fixture, "HostProbeFixture", lambda _settings: transport)
    kwargs = {
        "node": settings.node,
        "image": settings.image,
        "case_id": CASE,
        "run_id": settings.run_id,
        "state_directory": settings.state_directory,
        "node_uid": PINS["node_uid"],
        "plan_sha256": "d" * 64,
        "release_id": "test-release",
        "maintenance_expires_at": datetime.fromtimestamp(2_000_001_200, timezone.utc),
    }
    first = fixture.WarmSpareServiceFixture(
        cast(fixture.WarmSpareLiveFixture, warm), **kwargs
    )
    assert first.service == "" and first.failsafe_at is None
    first.create()
    first.stop("kubelet.service", delay_seconds=15)
    assert first.journal_path.exists() and first.failsafe_at is not None
    setup.host.fire_stop()
    assert first.restore()["phase"] == "RESTORED"
    first.close()
    fresh = fixture.WarmSpareServiceFixture(
        cast(fixture.WarmSpareLiveFixture, warm), **kwargs
    )
    assert fresh.resume_cleanup()["phase"] == "CLOSED"
    assert fresh.service == "kubelet.service"
    assert fresh.cleanup() == transport.residual
    fresh.close()
    assert (
        transport.creates == 1 and transport.commands.count("stop-with-failsafe") == 1
    )


def test_owned_child_crash_releases_private_controller_lock_without_erasing_intent(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "controller"
    config = write(directory / "kubeconfig", "test-only config")
    settings = HostProbeSettings(
        kubeconfig=config,
        context="test-context",
        namespace="test-namespace",
        node="test-node",
        image="test-image",
        case_id=CASE,
        run_id="test-child-run",
        probe_script=Path(probe.__file__),
        state_directory=directory,
    )
    host = HostProbeFixture(settings)
    code = """
import os, sys
from datetime import datetime, timezone
from pathlib import Path
from scripts.e2e.regional.destr008_service_window import ServiceWindowController
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture, HostProbeSettings
settings = HostProbeSettings(kubeconfig=Path(sys.argv[1]), context="test-context",
 namespace="test-namespace", node="test-node", image="test-image",
 case_id="GF-REGIONAL-DESTR-008", run_id="test-child-run",
 probe_script=Path(sys.argv[2]), state_directory=Path(sys.argv[1]).parent)
window = ServiceWindowController(HostProbeFixture(settings), cluster_id="test-cluster",
 node_uid="test-node-uid", plan_sha256="d"*64, release_id="test-release",
 maintenance_expires_at=datetime.fromtimestamp(2000001200, timezone.utc),
 read_node_uid=lambda: "test-node-uid")
window.acquire(existing=False)
window.data = {"schema_version": 1, "scope": window.scope, "owner": "e"*32,
 "phase": "CREATING", "supervision_lost": False}
window.save()
os._exit(17)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(config), str(settings.probe_script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 17, result.stderr
    window = controller.ServiceWindowController(
        host,
        cluster_id="test-cluster",
        node_uid="test-node-uid",
        plan_sha256="d" * 64,
        release_id="test-release",
        maintenance_expires_at=datetime.fromtimestamp(2_000_001_200, timezone.utc),
        read_node_uid=lambda: "test-node-uid",
    )
    window.acquire()
    try:
        assert window.resumed and window.data["phase"] == "CREATING"
        with pytest.raises(RegionalFixtureError, match="cleanup only"):
            window.stop("kubelet.service")
        second = controller.ServiceWindowController(
            host,
            cluster_id="test-cluster",
            node_uid="test-node-uid",
            plan_sha256="d" * 64,
            release_id="test-release",
            maintenance_expires_at=datetime.fromtimestamp(2_000_001_200, timezone.utc),
            read_node_uid=lambda: "test-node-uid",
        )
        with pytest.raises(BlockingIOError):
            second.acquire()
    finally:
        window.close()
    assert json.loads(window.path.read_text())["phase"] == "CREATING"


@pytest.mark.parametrize("stage", ["stop", "restore"])
def test_finally_close_after_failure_never_disarms_guard(
    setup: Any, stage: str
) -> None:
    setup.window.create()
    if stage == "stop":
        setup.transport.lost_ack = "stop-with-failsafe"
        with pytest.raises(TimeoutError):
            try:
                setup.window.stop("kubelet.service")
            finally:
                setup.window.close()
    else:
        setup.window.stop("kubelet.service")
        setup.host.pending_stop = True
        setup.host.fire_stop()
        with pytest.raises(probe.ProbeError, match="quiescent"):
            try:
                setup.window.restore()
            finally:
                setup.window.close()
    assert setup.window.fd is None
    name = setup.host.window.units["restore-service"]
    assert setup.host.units[name]["ActiveState"] == "active"
    assert setup.host.window.targets["restore-service"].exists(), (
        setup.host.window.targets
    )
    assert setup.host.read()["phase"] != "CLOSED"
    before = list(setup.host.calls)
    fresh = build(setup.transport)
    fresh.acquire()
    assert fresh.resumed and fresh.binding == setup.window.binding
    fresh.close()
    assert setup.host.calls == before


@pytest.mark.parametrize(
    ("argument", "value"),
    [("maintenance_expires_at", datetime(2033, 1, 1)), ("plan_sha256", "not-a-plan")],
)
def test_constructor_requires_explicit_bounded_approval(
    setup: Any, argument: str, value: Any
) -> None:
    with pytest.raises(RegionalFixtureError):
        build(setup.transport, **{argument: value})
    assert setup.transport.creates == 0 and setup.transport.commands == []


@pytest.mark.parametrize("defect", ["node", "expired", "pre-existing"])
def test_create_refuses_unapproved_or_preexisting_target(
    setup: Any, defect: str
) -> None:
    if defect == "node":
        setup.window.read_node_uid = lambda: "replacement"
    elif defect == "expired":
        setup.host.now = setup.window.scope["maintenance_expires_at"]
    else:
        write(setup.transport.state_path, "{}")
    with pytest.raises(RegionalFixtureError):
        setup.window.create()
    assert setup.transport.creates == 0 and not setup.window.path.exists()


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("restore_seconds", 59),
        ("restore_seconds", True),
        ("delay_seconds", 121),
        ("delay_seconds", False),
    ],
)
def test_stop_bounds_fail_before_even_reading_host_baseline(
    setup: Any, argument: str, value: Any
) -> None:
    setup.window.create()
    with pytest.raises(RegionalFixtureError, match="outside"):
        setup.window.stop("kubelet.service", **{argument: value})
    assert setup.transport.commands == [] and "binding" not in setup.window.data


def test_approved_deadline_cannot_be_exceeded_by_a_new_window(setup: Any) -> None:
    setup.window = build(
        setup.transport,
        maintenance_expires_at=datetime.fromtimestamp(2_000_000_200, timezone.utc),
    )
    setup.window.create()
    with pytest.raises(RegionalFixtureError, match="exceeds"):
        setup.window.stop("kubelet.service")
    assert setup.transport.commands == ["snapshot"]
    assert "binding" not in setup.window.data


@pytest.mark.parametrize("defect", ["pins", "state", "service"])
def test_malformed_baseline_cannot_arm_independent_service(
    setup: Any, defect: str
) -> None:
    setup.window.create()
    snapshot = probe.capture("kubelet.service")
    snapshot[defect] = {}
    setup.transport.reports["snapshot"] = snapshot
    with pytest.raises(RegionalFixtureError, match="approved target"):
        setup.window.stop("kubelet.service")
    assert setup.transport.commands == ["snapshot"]


@pytest.mark.parametrize(
    "defect", ["missing", "time", "future", "boot", "pid", "invocation"]
)
def test_controller_rechecks_independent_arm_receipt(setup: Any, defect: str) -> None:
    setup.window.create()
    execute = setup.transport.execute

    def altered(command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = execute(command, *args, **kwargs)
        if command == "status":
            if defect == "missing":
                result["ack"] = None
            else:
                key, value = {
                    "time": ("at", setup.host.now - 11),
                    "future": ("at", setup.host.now + 1),
                    "boot": ("boot_id", "another-boot"),
                    "pid": ("pid", 0),
                    "invocation": ("invocation_id", "unknown"),
                }[defect]
                result["ack"][key] = value
        return result

    setup.transport.execute = altered
    with pytest.raises(RegionalFixtureError, match="stale or malformed"):
        setup.window.stop("kubelet.service")
    assert "stop-with-failsafe" not in setup.transport.commands


@pytest.mark.parametrize(
    "defect",
    [
        "no-after",
        "missing-after-field",
        "wrong-service",
        "not-running",
        "no-quiescence",
    ],
)
def test_restoration_ack_requires_actual_service_and_stop_proof(
    setup: Any, defect: str
) -> None:
    setup.window.create()
    setup.window.stop("kubelet.service")
    report = setup.host.window.restore()
    if defect == "no-after":
        report["after"] = None
    elif defect == "missing-after-field":
        report["after"].pop("Job")
    elif defect == "wrong-service":
        report["after"]["Id"] = "another.service"
    elif defect == "not-running":
        report["after"]["ActiveState"] = "inactive"
    else:
        report["stop_quiescence"] = {"ready": True}
    with pytest.raises(RegionalFixtureError, match="restoration proof"):
        controller.require_report(report, setup.window.binding, {"RESTORED"})


@pytest.mark.parametrize("role", ["timer", "service", "restore"])
@pytest.mark.parametrize(
    "field",
    ["unit", "active_state", "job_id", "main_pid", "control_pid", "cgroup_empty"],
)
def test_no_generic_quiet_ack_can_close_window(
    setup: Any, role: str, field: str
) -> None:
    setup.window.create()
    setup.window.stop("kubelet.service")
    report = setup.host.window.cleanup()
    quiet = (
        report["restore_quiescence"]
        if role == "restore"
        else report["stop_quiescence"][role]
    )
    quiet[field] = (
        "unknown" if field not in {"job_id", "main_pid", "control_pid"} else 1
    )
    with pytest.raises(RegionalFixtureError, match="quiescence"):
        controller.require_report(report, setup.window.binding, {"CLOSED"})


@pytest.mark.parametrize("defect", ["resource", "new-owner", "missing-journal"])
def test_durable_transport_proof_cannot_be_replaced(setup: Any, defect: str) -> None:
    setup.window.create()
    if defect == "missing-journal":
        setup.transport.state_path.unlink()
        with pytest.raises(RegionalFixtureError, match="journal disappeared"):
            setup.window.cleanup()
        return
    value = probe.private_read(setup.transport.state_path)
    if defect == "resource":
        value["resources"]["pod"]["uid"] = None
    else:
        value["owner"] = "f" * 32
    probe.write_record(setup.transport.state_path, value)
    with pytest.raises(RegionalFixtureError, match="receipt|owner changed"):
        setup.window.stop("kubelet.service")
    assert setup.transport.commands == []


@pytest.mark.parametrize(
    "defect", ["binding", "owner", "helper", "deadline", "host-proof"]
)
def test_controller_binding_drift_refuses_before_remote_restore(
    setup: Any, defect: str
) -> None:
    setup.window.create()
    setup.window.stop("kubelet.service")
    if defect == "binding":
        setup.window.data["binding"] = None
    elif defect == "host-proof":
        setup.window.data.pop("host_proof")
    else:
        binding = setup.window.data["binding"]
        if defect == "owner":
            binding["owner"] = "f" * 32
        elif defect == "helper":
            binding["helper_sha256"] = "f" * 64
        else:
            binding["restore_at"] = setup.window.scope["maintenance_expires_at"]
            binding["expires_at"] = binding["restore_at"] + probe.RECOVERY_SECONDS
    setup.window.save()
    before = list(setup.transport.commands)
    with pytest.raises(RegionalFixtureError):
        setup.window.request("restore-service", {"RESTORED"})
    assert setup.transport.commands == before


def test_closed_unstarted_fixture_rejects_new_unrecorded_transport(setup: Any) -> None:
    setup.transport.lost_ack = "create"
    with pytest.raises(TimeoutError):
        setup.window.create()
    setup.transport.lost_ack = ""
    setup.transport.state_path.unlink()
    assert setup.window.cleanup() == setup.transport.residual
    fresh = build(setup.transport)
    assert fresh.resume_cleanup()["phase"] == "CLOSED"
    write(setup.transport.state_path, "{}")
    with pytest.raises(RegionalFixtureError, match="unrecorded host owner"):
        fresh.resume_cleanup()
    fresh.close()


def test_closed_fixture_cannot_reopen_restore_and_unlocked_save_refuses(
    setup: Any,
) -> None:
    with pytest.raises(RegionalFixtureError, match="not locked"):
        setup.window.save()
    setup.window.create()
    setup.window.stop("kubelet.service")
    setup.window.cleanup()
    before = list(setup.transport.commands)
    with pytest.raises(RegionalFixtureError, match="closed"):
        setup.window.restore()
    assert setup.transport.commands == before
    assert probe.private_read(setup.window.path)["phase"] == "CLOSED"


def test_controller_lock_must_be_private(setup: Any) -> None:
    path = setup.window.path.with_suffix(".lock")
    write(path, "").chmod(0o644)
    with pytest.raises(RegionalFixtureError, match="lock is not private"):
        setup.window.create()
    assert setup.transport.creates == 0
