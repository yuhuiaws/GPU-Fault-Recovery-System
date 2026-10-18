"""Collector-env recovery guards of the node probe that the owned cycle never hits.

The COLLECT-004 fixtures walk the happy override/restore cycle and a handful of
tampering cases. These tests pin the remaining refusals: untrusted state and lock
files, drifted recovery materials, stale drop-ins, half-finished cleanups and
the automatic restore of a record that never mutated the node. Every host
interaction goes through the private service-manager fake from
``test_collector_env_safety``; nothing here reaches a real systemd or socket.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import collector_node_probe as probe
from tests.regional.test_collector_env_safety import ENV_TEXT, OWNER_NONCE, EnvHost
from tests.regional.test_collector_env_safety import env_host as env_host_fixture

env_host = env_host_fixture

UNIT_PREFIX = "gpu-fault-collector-env-restore-"


def _write_private(path: Path, contents: bytes) -> None:
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(contents)
    path.chmod(0o600)


def _rewrite_unit(path: Path, contents: bytes) -> None:
    """Replace an owned unit file in place, keeping the 0o600 owner-only mode."""
    path.write_bytes(contents)
    path.chmod(0o600)


def test_parse_env_refuses_a_missing_collector_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(probe, "COLLECTOR_ENV", tmp_path / "absent.env")
    with pytest.raises(probe.ProbeError, match="collector env file does not exist"):
        probe.parse_env()


def test_require_root_passes_for_the_root_uid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    assert probe.collector_require_root() is None


def test_env_lock_refuses_a_group_readable_state_directory(env_host: EnvHost) -> None:
    probe.ACCEPTANCE_STATE.mkdir(mode=0o750)
    probe.ACCEPTANCE_STATE.chmod(0o750)
    with pytest.raises(
        probe.CollectorEnvGuardError, match="recovery directory is not private"
    ):
        with probe.collector_env_lock():
            raise AssertionError("the lock must not be granted")


def test_env_lock_refuses_a_group_readable_lock_file(env_host: EnvHost) -> None:
    probe.ACCEPTANCE_STATE.mkdir(mode=0o700)
    lock = probe.ACCEPTANCE_STATE / "collector-env.lock"
    lock.write_bytes(b"")
    lock.chmod(0o640)
    with pytest.raises(probe.CollectorEnvGuardError, match="lock is not private"):
        with probe.collector_env_lock():
            raise AssertionError("the lock must not be granted")


def test_nonce_file_capability_rejects_a_foreign_path_or_an_inline_nonce(
    env_host: EnvHost,
) -> None:
    expected = probe.collector_restore_paths(env_host.run_id)[0].with_suffix(".nonce")
    with pytest.raises(probe.CollectorEnvGuardError, match="capability path differs"):
        probe.collector_nonce(
            env_host.arguments(owner_nonce="", owner_nonce_file="/tmp/other.nonce")
        )
    with pytest.raises(probe.CollectorEnvGuardError, match="capability path differs"):
        probe.collector_nonce(
            env_host.arguments(owner_nonce=OWNER_NONCE, owner_nonce_file=str(expected))
        )


def test_env_values_reject_duplicates_and_non_integer_counts() -> None:
    duplicated = (
        b"GPU_FAULT_EXPECTED_GPU_COUNT=8\nGPU_FAULT_EXPECTED_GPU_COUNT=9\n"
        b"GPU_FAULT_CLUSTER_ID=c\nNODE_NAME=n\n"
    )
    with pytest.raises(probe.CollectorEnvGuardError, match="key is duplicated"):
        probe.collector_env_values(duplicated)
    textual = (
        b"GPU_FAULT_EXPECTED_GPU_COUNT=eight\nGPU_FAULT_CLUSTER_ID=c\nNODE_NAME=n\n"
    )
    with pytest.raises(probe.CollectorEnvGuardError, match="not an integer"):
        probe.collector_env_values(textual)
    with pytest.raises(probe.CollectorEnvGuardError, match="value is invalid"):
        probe.collector_env_values(b"GPU_FAULT_EXPECTED_GPU_COUNT=8\nNODE_NAME=a b\n")


def test_env_values_tolerate_a_missing_optional_instance_id() -> None:
    values = probe.collector_env_values(
        b"GPU_FAULT_EXPECTED_GPU_COUNT=8\nGPU_FAULT_CLUSTER_ID=c\nNODE_NAME=n\n"
    )
    assert values == {
        "GPU_FAULT_EXPECTED_GPU_COUNT": "8",
        "GPU_FAULT_CLUSTER_ID": "c",
        "NODE_NAME": "n",
    }


def test_runtime_identity_refuses_a_group_writable_runtime(env_host: EnvHost) -> None:
    probe.COLLECTOR_RUNTIME.resolve().chmod(0o775)
    with pytest.raises(
        probe.CollectorEnvGuardError, match="runtime identity is unavailable"
    ):
        probe.collector_runtime_identity()


def test_guard_failure_path_rejects_unknown_commands_and_codes() -> None:
    with pytest.raises(probe.CollectorEnvGuardError, match="receipt identity"):
        probe.collector_guard_failure_path("c004-1", "throttle-gpu", "0" * 16)
    with pytest.raises(probe.CollectorEnvGuardError, match="receipt identity"):
        probe.collector_guard_failure_path("c004-1", "restore-collector-env", "zz")


def test_guard_failure_refuses_to_report_over_a_malformed_previous_receipt(
    env_host: EnvHost,
) -> None:
    env_host.override()
    code = "a" * 16
    path = probe.collector_guard_failure_path(
        env_host.run_id, "restore-collector-env", code
    )
    _write_private(path, json.dumps({"unexpected": True}).encode())
    arguments = env_host.arguments(command="restore-collector-env")
    result = probe.collector_guard_failure(arguments, error_code=code, error_site=None)
    assert result == {"status": "LOGGING_UNAVAILABLE"}
    assert json.loads(path.read_bytes()) == {"unexpected": True}


def test_guard_failure_refuses_an_unrecordable_error_site(env_host: EnvHost) -> None:
    env_host.override()
    code = "b" * 16
    path = probe.collector_guard_failure_path(
        env_host.run_id, "restore-collector-env", code
    )
    arguments = env_host.arguments(command="restore-collector-env")
    result = probe.collector_guard_failure(
        arguments, error_code=code, error_site="not a site"
    )
    assert result == {"status": "LOGGING_UNAVAILABLE"}
    assert path.exists() is False


def test_verify_identity_detects_a_changed_dropin(env_host: EnvHost) -> None:
    env_host.override()
    record = env_host.record()
    dropin = next(
        path for path in probe.collector_unit_contents(record) if path.suffix == ".conf"
    )
    _write_private(dropin, b"[Unit]\nRequires=other.service\n")
    with pytest.raises(
        probe.CollectorEnvGuardError, match="recovery resource identity changed"
    ):
        probe.collector_verify_identity(record)


def test_recovery_command_refuses_unsafe_interpreter_paths(env_host: EnvHost) -> None:
    env_host.override()
    record = env_host.record()
    record["python"] = "/opt/gpu fault/python"
    with pytest.raises(probe.CollectorEnvGuardError, match="unsafe characters"):
        probe.collector_command(record)


def test_verify_identity_detects_changed_recovery_materials(env_host: EnvHost) -> None:
    env_host.override()
    record = env_host.record()
    backup, _ = env_host.paths()
    _write_private(backup.with_suffix(".probe.py"), b"# tampered probe copy\n")
    with pytest.raises(
        probe.CollectorEnvGuardError, match="recovery material identity changed"
    ):
        probe.collector_verify_identity(record)


def test_verify_identity_detects_a_changed_host_unit(env_host: EnvHost) -> None:
    env_host.override()
    record = env_host.record()
    host_unit = Path(record["host_unit_path"])
    host_unit.write_text(host_unit.read_text() + "# edited by an operator\n")
    with pytest.raises(
        probe.CollectorEnvGuardError, match="host collector unit identity changed"
    ):
        probe.collector_verify_identity(record)


def test_verify_identity_requires_every_resource_once_mutation_started(
    env_host: EnvHost,
) -> None:
    env_host.override()
    record = env_host.record()
    first = next(iter(probe.collector_unit_contents(record)))
    first.unlink()
    with pytest.raises(
        probe.CollectorEnvGuardError, match="recovery resource is missing"
    ):
        probe.collector_verify_identity(record)


def test_unit_state_is_unknown_when_systemd_answers_for_another_unit(
    env_host: EnvHost,
) -> None:
    env_host.overrides[probe.HOST_COLLECTOR_UNIT] = {"Id": "other.service"}
    with pytest.raises(probe.CollectorEnvGuardError, match="unit state is unknown"):
        probe.collector_unit_state(probe.HOST_COLLECTOR_UNIT)


def test_owned_unit_refuses_an_unloaded_file_with_foreign_contents(
    env_host: EnvHost,
) -> None:
    env_host.override()
    record = env_host.record()
    _, unit = env_host.paths()
    timer = probe.SYSTEMD_UNIT_DIR / f"{unit}.timer"
    env_host.overrides[timer.name] = {"LoadState": "not-found"}
    _rewrite_unit(timer, b"[Unit]\nDescription=foreign\n")
    with pytest.raises(probe.CollectorEnvGuardError, match="is not owned"):
        probe.collector_owned_unit(record, timer, absent=True)


def test_verify_armed_detects_a_unit_changed_before_arming(env_host: EnvHost) -> None:
    env_host.override()
    record = env_host.record()
    _, unit = env_host.paths()
    _rewrite_unit(
        probe.SYSTEMD_UNIT_DIR / f"{unit}.timer", b"[Unit]\nDescription=changed\n"
    )
    with pytest.raises(probe.CollectorEnvGuardError, match="changed before arming"):
        probe.collector_verify_armed(record)


def test_check_start_requires_the_restoring_boot_to_match(env_host: EnvHost) -> None:
    env_host.override()
    env_host.restore()
    record = env_host.record()
    assert record["state"] == "CLEANED"
    record["restored_boot_id"] = "boot-fixture-b"
    with pytest.raises(
        probe.CollectorEnvGuardError, match="boot restoration is not proven"
    ):
        probe.collector_check_start(record)


def test_start_check_refuses_without_a_recovery_intent(env_host: EnvHost) -> None:
    with pytest.raises(probe.CollectorEnvGuardError, match="no recovery intent"):
        probe.collector_env_start_check(env_host.arguments())
    assert env_host.emitted == []


def test_replace_env_rejects_a_drifted_compare_and_swap_digest(
    env_host: EnvHost,
) -> None:
    with pytest.raises(probe.CollectorEnvGuardError, match="rejected drift"):
        probe.replace_collector_env(b"X=1\n", expected_sha256="0" * 64)
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT


def test_override_requires_a_unique_anchored_count_line(env_host: EnvHost) -> None:
    indented = ENV_TEXT.replace(
        "GPU_FAULT_EXPECTED_GPU_COUNT=8\n", " GPU_FAULT_EXPECTED_GPU_COUNT=8\n"
    )
    probe.COLLECTOR_ENV.write_text(indented)
    probe.COLLECTOR_ENV.chmod(0o600)
    digest = hashlib.sha256(indented.encode()).hexdigest()
    with pytest.raises(
        probe.CollectorEnvGuardError, match="no unique expected GPU count"
    ):
        env_host.override(expected_env_sha256=digest)
    assert env_host.mutations() == []


def test_override_refuses_a_host_collector_with_a_pending_job(
    env_host: EnvHost,
) -> None:
    env_host.overrides[probe.HOST_COLLECTOR_UNIT] = {"Job": "4711"}
    with pytest.raises(
        probe.CollectorEnvGuardError, match="host collector baseline is not stable"
    ):
        env_host.override()
    assert env_host.mutations() == []


def test_override_refuses_a_leftover_armed_receipt(env_host: EnvHost) -> None:
    probe.ACCEPTANCE_STATE.mkdir(mode=0o700)
    backup, _ = env_host.paths()
    _write_private(backup.with_suffix(".armed.json"), b"{}")
    with pytest.raises(
        probe.CollectorEnvGuardError, match="recovery path already exists"
    ):
        env_host.override()
    assert env_host.mutations() == []


def test_override_refuses_a_recovery_unit_systemd_already_loaded(
    env_host: EnvHost,
) -> None:
    _, unit = env_host.paths()
    env_host.overrides[f"{unit}.service"] = {"LoadState": "loaded"}
    with pytest.raises(
        probe.CollectorEnvGuardError, match="recovery unit already exists"
    ):
        env_host.override()
    assert env_host.mutations() == []


def test_override_refuses_a_group_writable_dropin_directory(env_host: EnvHost) -> None:
    dropin_dir = probe.SYSTEMD_UNIT_DIR / f"{probe.HOST_COLLECTOR_UNIT}.d"
    dropin_dir.mkdir(mode=0o775)
    dropin_dir.chmod(0o775)
    with pytest.raises(
        probe.CollectorEnvGuardError, match="drop-in directory is not trusted"
    ):
        env_host.override()
    assert env_host.mutations() == []
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT


def test_override_requires_the_recovery_service_to_acknowledge_armed(
    env_host: EnvHost,
) -> None:
    env_host.acknowledge_start = False
    with pytest.raises(probe.CollectorEnvGuardError, match="did not acknowledge ARMED"):
        env_host.override()
    assert env_host.record()["state"] == "PREPARED"
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT


def test_override_refuses_an_activation_past_its_deadline(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow_restart(command: list[str], **kwargs: Any) -> Any:
        completed = env_host.run(command, **kwargs)
        if command[1:] == ["restart", probe.HOST_COLLECTOR_UNIT]:
            env_host.now = 10_000.0
        return completed

    monkeypatch.setattr(probe, "run", slow_restart)
    with pytest.raises(
        probe.CollectorEnvGuardError, match="activation exceeded its deadline"
    ):
        env_host.override()
    record = env_host.record()
    assert record["state"] == "MUTATING"
    assert record["mutation_started"] is True


def test_override_after_a_cleaned_run_starts_a_new_record(env_host: EnvHost) -> None:
    env_host.override()
    env_host.restore()
    first_run = env_host.run_id
    assert env_host.record()["state"] == "CLEANED"
    env_host.run_id = "c004-2"
    second = env_host.override()
    assert second["run_id"] == "c004-2"
    assert env_host.record("c004-2")["state"] == "ACTIVE"
    assert env_host.record(first_run)["state"] == "CLEANED"


def test_finish_units_refuses_to_remove_a_foreign_recovery_file(
    env_host: EnvHost,
) -> None:
    env_host.override()
    record = env_host.record()
    _, unit = env_host.paths()
    _rewrite_unit(
        probe.SYSTEMD_UNIT_DIR / f"{unit}.service", b"[Service]\nExecStart=/bin/true\n"
    )
    with pytest.raises(probe.CollectorEnvGuardError, match="refusing to remove"):
        probe.collector_finish_units(record)


def test_finish_units_completes_after_units_were_already_removed(
    env_host: EnvHost,
) -> None:
    env_host.override()
    env_host.failures[("systemctl", "daemon-reload")] = [
        probe.ProbeError("daemon-reload failed")
    ]
    with pytest.raises(probe.ProbeError, match="daemon-reload failed"):
        env_host.restore()
    record = env_host.record()
    assert record["state"] == "CLEANING"
    units = probe.collector_unit_contents(record)
    for path in units:
        path.unlink(missing_ok=True)
    dropin = next(path for path in units if path.suffix == ".conf")
    dropin.parent.rmdir()
    for wants in probe.SYSTEMD_UNIT_DIR.glob("*.wants/*"):
        wants.unlink()
    env_host.reload()
    env_host.calls.clear()
    probe.collector_finish_units(record)
    assert [call[1] for call in env_host.mutations()] == [
        "daemon-reload",
        "daemon-reload",
    ]


def test_verify_removed_reports_each_leftover(env_host: EnvHost) -> None:
    env_host.override()
    record = env_host.record()
    _, unit = env_host.paths()
    units = probe.collector_unit_contents(record)
    with pytest.raises(
        probe.CollectorEnvGuardError, match="enablement link remains installed"
    ):
        probe.collector_verify_removed(record)
    for wants in probe.SYSTEMD_UNIT_DIR.glob("*.wants/*"):
        wants.unlink()
    with pytest.raises(
        probe.CollectorEnvGuardError, match="unit file remains installed"
    ):
        probe.collector_verify_removed(record)
    for path in units:
        path.unlink()
    env_host.overrides[f"{unit}.service"] = {"LoadState": "loaded"}
    with pytest.raises(probe.CollectorEnvGuardError, match="unit remains loaded"):
        probe.collector_verify_removed(record)


def test_remove_materials_refuses_changed_backups_and_receipts(
    env_host: EnvHost,
) -> None:
    env_host.override()
    record = env_host.record()
    backup, _ = env_host.paths()
    original = backup.read_bytes()
    _write_private(backup, original + b"# drift\n")
    with pytest.raises(
        probe.CollectorEnvGuardError, match="recovery material identity changed"
    ):
        probe.collector_remove_materials(record)
    _write_private(backup, original)
    armed = backup.with_suffix(".armed.json")
    proof = json.loads(armed.read_bytes())
    proof["intent_sha256"] = "f" * 64
    _write_private(armed, json.dumps(proof).encode())
    with pytest.raises(
        probe.CollectorEnvGuardError, match="ARMED receipt identity changed"
    ):
        probe.collector_remove_materials(record)
    assert backup.exists() is False
    assert armed.exists() is True


def test_manual_restore_rebinds_the_boot_of_a_deferred_restore(
    env_host: EnvHost,
) -> None:
    env_host.override()
    env_host.now = 10_000.0
    deferred = env_host.restore(automatic=True)
    assert deferred["cleanup_deferred"] is True
    assert env_host.record()["restored_boot_id"] == "boot-fixture-a"
    probe.BOOT_ID_FILE.write_text("boot-fixture-b")
    env_host.active.discard(probe.HOST_COLLECTOR_UNIT)
    final = env_host.restore()
    assert final["cleanup_verified"] is True
    record = env_host.record()
    assert record["state"] == "CLEANED"
    assert record["restored_boot_id"] == "boot-fixture-b"


def test_automatic_restore_of_an_armed_record_never_restarts_the_collector(
    env_host: EnvHost,
) -> None:
    started = False

    def expire_after_recovery_start(command: list[str]) -> None:
        nonlocal started
        if started:
            env_host.now = 10_000.0
        if command[1] == "start":
            started = True

    env_host.before_call = expire_after_recovery_start
    with pytest.raises(probe.CollectorEnvGuardError, match="deadline elapsed"):
        env_host.override()
    record = env_host.record()
    assert record["state"] == "ARMED"
    assert record["mutation_started"] is False
    env_host.before_call = None
    env_host.calls.clear()
    receipt = env_host.restore(automatic=True)
    assert receipt["restored"] is True
    assert receipt["cleanup_deferred"] is True
    assert env_host.record()["state"] == "RESTORED"
    assert "restart" not in {call[1] for call in env_host.mutations()}
    assert "--no-block" not in {call[1] for call in env_host.mutations()}


def test_restore_resumes_a_cleanup_that_stopped_mid_way(env_host: EnvHost) -> None:
    env_host.override()
    env_host.failures[("systemctl", "daemon-reload")] = [
        probe.ProbeError("daemon-reload failed")
    ]
    with pytest.raises(probe.ProbeError, match="daemon-reload failed"):
        env_host.restore()
    assert env_host.record()["state"] == "CLEANING"
    env_host.calls.clear()
    receipt = env_host.restore()
    assert receipt["cleanup_verified"] is True
    assert env_host.record()["state"] == "CLEANED"
    assert "restart" not in {call[1] for call in env_host.mutations()}


def test_inventory_sampling_identity_requires_a_profile_version(
    env_host: EnvHost,
) -> None:
    pid = str(os.getpid())
    env_host.overrides[probe.HOST_COLLECTOR_UNIT] = {"MainPID": pid}
    process = probe.PROC_ROOT / pid
    process.mkdir()
    (process / "stat").write_text(f"{pid} (collector) S " + " ".join(["0"] * 30))
    (process / "environ").write_bytes(
        b"GPU_FAULT_EXPECTED_GPU_COUNT=8\0GPU_FAULT_HOST_INTERVAL_SECONDS=15\0"
        b"GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES=2\0"
    )
    with pytest.raises(probe.ProbeError, match="sampling identity is incomplete"):
        probe.inventory_sampling_identity()
