from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from scripts.e2e.regional.probes import warm_spare_node_probe as probe
from tests.regional._destr008_service_window import Host, write


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Host:
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    return Host(tmp_path, monkeypatch)


@pytest.mark.parametrize("service", sorted(probe.ALLOWED_SERVICES))
@pytest.mark.parametrize("delay", [0, 15, 120])
def test_owned_roundtrip_requires_ack_and_quiescent_cleanup(
    host: Host, service: str, delay: int
) -> None:
    host.binding = host.make_binding(service, delay)
    host.window = probe.ServiceWindow(host.binding)
    prepared = host.window.prepare()
    assert prepared["phase"] == "INSTALLED"
    before = host.mutations()
    with pytest.raises(probe.ProbeError, match="ARMED"):
        host.window.schedule_stop()
    assert host.mutations() == before
    assert host.tick()["phase"] == "ARMED"
    scheduled = host.window.schedule_stop()
    assert scheduled["phase"] == "SCHEDULED" and scheduled["stop_requested"] is False
    assert ("stop", "--no-block", "--job-mode=fail", service) not in host.calls
    stopped = host.fire_stop()
    assert stopped["phase"] == "STOPPING" and stopped["stop_requested"] is True
    restored = host.window.restore()
    assert (
        restored["phase"] == "RESTORED" and restored["after"]["ActiveState"] == "active"
    )
    assert restored["stop_quiescence"]["service"]["cgroup_empty"] is True
    mutations = host.mutations()
    start = mutations.index(("start", "--no-block", "--job-mode=fail", service))
    cancel = mutations.index(("stop", host.window.units["stop-timer"]))
    assert cancel < start
    closed = host.window.cleanup()
    assert closed["phase"] == "CLOSED" and closed["record_kind"] == "TOMBSTONE"
    assert closed["restore_quiescence"]["job_id"] == 0
    assert not any(
        (probe.SYSTEMD / name).exists() for name in host.window.units.values()
    ), host.window.targets
    assert not host.window.dropin.exists(), host.window.dropin
    assert host.window.targets["helper"].is_file(), host.window.targets
    assert not host.window.targets["claim"].exists(), host.window.targets
    assert host.read()["binding"] == host.binding
    assert host.window.cleanup()["phase"] == "CLOSED"
    with pytest.raises(probe.ProbeError, match="already exists"):
        host.window.prepare()
    before = host.mutations()
    assert host.window.stop_owned()["phase"] == "CLOSED"
    assert host.mutations() == before


def test_independent_recovery_runs_without_controller_or_transport(host: Host) -> None:
    host.stopped()
    host.now = host.binding["restore_at"]
    host.elapsed = host.read()["clock"]["restore"]
    result = host.tick()
    assert result["phase"] == "RESTORED" and result["restore_reason"] == "deadline"
    assert host.units[host.binding["service"]]["ActiveState"] == "active"
    assert result["record_kind"] == "RECOVERY_REQUIRED"
    assert host.window.cleanup()["phase"] == "CLOSED"


def test_restore_seals_scheduled_stop_before_it_fires(host: Host) -> None:
    host.arm()
    host.window.schedule_stop()
    result = host.window.restore()
    assert result["phase"] == "RESTORED" and result["start_intent"] is False
    before = host.mutations()
    assert host.window.stop_owned()["phase"] == "RESTORED"
    assert host.mutations() == before
    assert not any(c[-1] in probe.ALLOWED_SERVICES for c in before), before


@pytest.mark.parametrize("role", ["stop-timer", "stop-service"])
@pytest.mark.parametrize("failure", ["exit", "stuck", "pending-job", "child"])
def test_restore_refuses_nonquiescent_delayed_stop(
    host: Host, role: str, failure: str
) -> None:
    host.stopped()
    name = host.window.units[role]
    unit = host.units[name]
    if failure == "exit":
        unit.update(ActiveState="active", SubState="running")
        host.fail.add(("stop", name))
    elif failure == "stuck":
        unit.update(ActiveState="active", SubState="running")
        host.stuck.add(name)
    elif failure == "pending-job":
        host.overrides[name] = {"Job": "41"}
    else:
        if role.endswith("timer"):
            host.overrides[name] = {"SubState": "unknown"}
        else:
            unit["ControlGroup"] = f"/system.slice/{name}"
            write(
                probe.CGROUP_ROOT / "system.slice" / name / "cgroup.events",
                "populated 1\n",
            )
    before = list(host.mutations())
    with pytest.raises(probe.ProbeError):
        host.window.restore()
    after = host.mutations()[len(before) :]
    assert not any(
        c[0] == "start" and c[-1] == host.binding["service"] for c in after
    ), after
    assert ("stop", host.window.units["restore-service"]) not in after
    assert host.read()["phase"] == "RESTORING"
    assert host.window.targets["restore-service"].exists(), host.window.targets


def test_target_stop_job_must_finish_before_start(host: Host) -> None:
    host.pending_stop = True
    host.stopped()
    host.complete_on_sleep = host.binding["service"]
    before = host.elapsed
    assert host.window.restore()["phase"] == "RESTORED"
    assert host.elapsed > before
    assert host.units[host.binding["service"]]["Job"] == "0"


def test_pending_target_stop_never_allows_early_restore_or_disarm(host: Host) -> None:
    host.pending_stop = True
    host.stopped()
    with pytest.raises(probe.ProbeError, match="quiescent"):
        host.window.cleanup()
    assert host.units[host.binding["service"]]["Job"] == "81"
    assert host.window.targets["restore-service"].exists(), host.window.targets
    assert not any(
        c[0] == "start" and c[-1] == host.binding["service"] for c in host.mutations()
    ), host.mutations()


@pytest.mark.parametrize(
    "resource",
    ["claim", "helper", "bound", "stop-service", "stop-timer", "restore-service"],
)
def test_identical_replacement_resource_is_not_owned(host: Host, resource: str) -> None:
    host.stopped()
    path = host.window.targets[resource]
    replacement = write(path.with_name(path.name + ".replacement"), path.read_text())
    os.replace(replacement, path)
    before = host.mutations()
    with pytest.raises(probe.ProbeError, match="replaced"):
        host.window.cleanup()
    assert host.mutations() == before
    assert path.exists(), path


@pytest.mark.parametrize(
    "changed",
    ["unit", "environment", "runtime", "host", "boot", "invocation", "enabled"],
)
def test_foreign_target_is_never_started_or_rewritten(host: Host, changed: str) -> None:
    host.stopped()
    unit = host.units[host.binding["service"]]
    if changed == "unit":
        write(Path(unit["FragmentPath"]), "foreign unit")
    elif changed == "environment":
        probe.AGENT_ENV.write_text(probe.AGENT_ENV.read_text() + "\nFOREIGN=yes\n")
    elif changed == "runtime":
        write(probe.CURRENT / "venv/bin/gpu-fault-node-agent", "new runtime")
    elif changed == "host":
        write(probe.MACHINE_ID, "another host")
    elif changed == "boot":
        write(probe.BOOT_ID, "another boot")
    elif changed == "invocation":
        unit.update(
            ActiveState="active",
            SubState="running",
            MainPID="300",
            InvocationID="f" * 32,
        )
    else:
        unit["UnitFileState"] = "disabled"
    before = host.mutations()
    with pytest.raises(probe.ProbeError):
        host.window.restore()
    added = host.mutations()[len(before) :]
    assert not any(
        c[0] == "start" and c[-1] == host.binding["service"] for c in added
    ), added
    assert ("stop", host.window.units["restore-service"]) not in added


@pytest.mark.parametrize("clock", ["wall-forward", "wall-backward", "elapsed-expired"])
def test_absolute_and_same_boot_expiry_cannot_be_renewed(
    host: Host, clock: str
) -> None:
    host.stopped()
    if clock == "wall-forward":
        host.now = host.binding["expires_at"] + 1
    elif clock == "wall-backward":
        host.now -= 10000
        host.elapsed = host.read()["clock"]["restore"]
        assert host.tick()["phase"] == "RESTORED"
        assert host.read()["binding"]["restore_at"] == host.binding["restore_at"]
        return
    else:
        host.elapsed = host.read()["clock"]["expires"] + 1
    with pytest.raises(probe.ProbeError, match="expired"):
        host.window.restore()
    assert host.read()["phase"] == "EXPIRED"
    assert not any(
        c[0] == "start" and c[-1] == host.binding["service"] for c in host.mutations()
    ), host.mutations()


def test_lost_target_start_ack_is_not_adopted_as_ownership(host: Host) -> None:
    host.stopped()
    host.fail_after.add(
        ("start", "--no-block", "--job-mode=fail", host.binding["service"])
    )
    with pytest.raises(probe.ProbeError, match="systemctl failed"):
        host.window.restore()
    assert host.read()["start_intent"] is True
    host.fail_after.clear()
    before = host.mutations()
    with pytest.raises(probe.ProbeError, match="unknown owner"):
        probe.ServiceWindow(host.binding).cleanup()
    assert not any(c[0] == "start" for c in host.mutations()[len(before) :]), (
        host.mutations()
    )
    assert host.window.targets["restore-service"].exists(), host.window.targets


def test_unsubmitted_prepare_can_only_be_fenced_when_baseline_and_absence_are_proven(
    host: Host,
) -> None:
    assert host.window.cleanup()["phase"] == "CLOSED"
    assert host.read()["resources"] == {}
    assert not any(c[0] in {"start", "stop"} for c in host.mutations()), (
        host.mutations()
    )
    with pytest.raises(probe.ProbeError, match="cleanup only"):
        host.window.prepare()


@pytest.mark.parametrize(
    "field", ["binding", "phase", "baseline", "clock", "stop_requested", "resources"]
)
def test_corrupt_journal_never_authorizes_commands(host: Host, field: str) -> None:
    host.stopped()
    record = host.read()
    if field == "binding":
        record["binding"] = {**record["binding"], "owner": "f" * 32}
    elif field == "clock":
        record["clock"]["expires"] += 1
    elif field == "stop_requested":
        record[field] = "false"
    else:
        record[field] = {}
    host.save(record)
    before = host.mutations()
    with pytest.raises(probe.ProbeError):
        host.window.cleanup()
    assert host.mutations() == before


def test_independent_ack_requires_its_real_owned_service_identity(host: Host) -> None:
    host.window.prepare()
    before = copy.deepcopy(host.read())
    with pytest.raises(probe.ProbeError, match="independent"):
        host.window.tick()
    assert host.read() == before


def test_unconfirmed_stop_rpc_cannot_be_inferred_from_an_exited_helper(
    host: Host,
) -> None:
    host.arm()
    host.window.schedule_stop()
    command = ("stop", "--no-block", "--job-mode=fail", "kubelet.service")
    host.fail_after.add(command)
    with pytest.raises(probe.ProbeError, match="systemctl failed"):
        host.fire_stop()
    assert host.read()["stop_requested"] is True
    assert host.read()["stop_acknowledged"] is False
    host.fail_after.clear()
    before = host.mutations()
    with pytest.raises(probe.ProbeError, match="no durable ACK"):
        probe.ServiceWindow(host.binding).cleanup()
    assert host.units[host.window.units["stop-service"]]["MainPID"] == "0"
    assert host.units[host.window.units["stop-service"]]["Job"] == "0"
    added = host.mutations()[len(before) :]
    assert not any(c[0] == "start" for c in added), added
    assert ("stop", host.window.units["restore-service"]) not in added
    assert host.window.targets["restore-service"].exists(), host.window.targets


def test_ack_without_scoped_stop_observation_is_not_recovery_proof(host: Host) -> None:
    host.stopped()
    record = host.read()
    record["stop_observation"] = {"ready": True}
    host.save(record)
    before = host.mutations()
    with pytest.raises(probe.ProbeError, match="scoped observation"):
        host.window.cleanup()
    assert host.mutations() == before


def test_stop_delay_is_not_counted_twice_after_the_timer_fires(host: Host) -> None:
    host.binding = host.make_binding(delay=120)
    host.binding["restore_at"] = int(host.now) + 180
    host.binding["expires_at"] = host.binding["restore_at"] + probe.RECOVERY_SECONDS
    host.window = probe.ServiceWindow(host.binding)
    result = host.stopped()
    assert result["stop_acknowledged"] is True
    assert host.units["kubelet.service"]["ActiveState"] == "inactive"
    assert host.window.restore()["phase"] == "RESTORED"
