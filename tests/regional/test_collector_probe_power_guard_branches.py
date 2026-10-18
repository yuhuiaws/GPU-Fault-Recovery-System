"""Power, FM-journal and inventory guards of the node probe off the owned cycle.

``test_collector_power_safety`` drives the complete throttle/restore cycle on a
private service-manager fake. These tests pin the refusals that cycle never
reaches: untrusted state directories and lock files, recovery records owned by
another run, units that systemd still reports after cleanup, load-start receipts
that arrive late or name a stopped service, and the FM/inventory probes'
argument and journal shape checks. No GPU, systemd, journal or network is ever
touched; every host read is a fixture file or an injected fake.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import collector_node_probe as probe
from tests.regional.test_collector_env_safety import EnvHost
from tests.regional.test_collector_env_safety import env_host as env_host_fixture
from tests.regional.test_collector_power_safety import PowerHost
from tests.regional.test_collector_power_safety import power_host as power_host_fixture

env_host = env_host_fixture
power_host = power_host_fixture

GPU_A = "GPU-0000000a-0000-0000-0000-000000000000"
GPU_B = "GPU-0000000b-0000-0000-0000-000000000000"


def _completed(
    command: list[str], stdout: str, returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, "")


def _cycle(host: PowerHost) -> dict[str, Any]:
    """Run the owned throttle and restore so every unit is cleaned up again."""
    probe.throttle_gpu(host.arguments())
    probe.restore_gpu_power_limit(host.arguments())
    record = host.record()
    assert record["phase"] == "CLEANED"
    return record


def test_strict_gpu_inventory_refuses_failures_and_incomplete_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers: list[tuple[int, str]] = [(1, ""), (0, f"0,{GPU_A},0000:01:00.0\n")]
    monkeypatch.setattr(
        probe,
        "run",
        lambda command, **kwargs: _completed(command, *reversed(answers.pop(0))),
    )
    with pytest.raises(probe.ProbeError, match="GPU identity query failed"):
        probe.gpu_inventory(strict=True)
    with pytest.raises(probe.ProbeError, match="incomplete row"):
        probe.gpu_inventory(strict=True)


def test_lenient_gpu_inventory_skips_incomplete_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = f"0,{GPU_A},00000000:01:00.0,NVIDIA H100\nbroken row\n"
    monkeypatch.setattr(
        probe, "run", lambda command, **kwargs: _completed(command, stdout)
    )
    assert probe.gpu_inventory() == [
        {"index": "0", "uuid": GPU_A, "pci_bdf": "0000:01:00.0", "name": "NVIDIA H100"}
    ]


def test_power_file_reads_take_a_shared_lock_and_reject_non_objects(
    tmp_path: Path,
) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(b"[1, 2]")
    path.chmod(0o600)
    assert probe.power_read_file(path, shared_lock=True) == b"[1, 2]"
    with pytest.raises(probe.ProbeError, match="record is not an object"):
        probe.power_read_json(path)


def test_power_file_reads_refuse_a_group_writable_file(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(b"{}")
    path.chmod(0o664)
    with pytest.raises(probe.ProbeError, match="not a trusted regular file"):
        probe.power_read_file(path)


def test_power_json_writes_fail_cleanly_without_a_state_directory(
    tmp_path: Path,
) -> None:
    target = tmp_path / "absent" / "record.json"
    with pytest.raises(FileNotFoundError):
        probe.power_write_json(target, {"run_id": "x"})
    assert list(tmp_path.iterdir()) == []


def test_load_start_receipt_fails_cleanly_without_a_state_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(probe, "ACCEPTANCE_STATE", tmp_path / "absent")
    record = {
        "run_id": "power-a1",
        "boot_id": "boot",
        "intent_sha256": "0" * 64,
        "gpu_index": 0,
        "baseline": [{"index": 0, "uuid": GPU_A}],
    }
    with pytest.raises(FileNotFoundError):
        probe.power_create_load_start(record)
    assert list(tmp_path.iterdir()) == []


def test_power_lock_refuses_untrusted_directories_and_lock_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = tmp_path / "state"
    monkeypatch.setattr(probe, "ACCEPTANCE_STATE", state)
    state.mkdir(mode=0o722)
    state.chmod(0o722)
    with pytest.raises(probe.ProbeError, match="directory is not trusted"):
        with probe.power_operation_lock():
            raise AssertionError("the lock must not be granted")
    state.chmod(0o700)
    lock = state / "gpu-power.lock"
    lock.write_bytes(b"")
    lock.chmod(0o640)
    with pytest.raises(probe.ProbeError, match="lock is not a regular file"):
        with probe.power_operation_lock():
            raise AssertionError("the lock must not be granted")


def test_checked_power_state_rejects_a_uuid_rebinding_between_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe, "parse_env", lambda: {"GPU_FAULT_EXPECTED_GPU_COUNT": "1"}
    )
    monkeypatch.setattr(
        probe,
        "gpu_power_state",
        lambda *, strict=False: [
            {
                "index": 0,
                "uuid": GPU_A,
                "power_draw_w": 100.0,
                "power_limit_w": 700.0,
                "power_min_limit_w": 200.0,
                "power_default_limit_w": 700.0,
                "utilization_percent": 0.0,
            }
        ],
    )
    monkeypatch.setattr(
        probe,
        "gpu_inventory",
        lambda *, strict=False: [
            {"index": "0", "uuid": GPU_B, "pci_bdf": "0000:01:00.0", "name": "H100"}
        ],
    )
    with pytest.raises(probe.ProbeError, match="binding changed"):
        probe.checked_power_state()


def test_power_unit_contents_reject_unsupported_command_characters() -> None:
    record = {
        "run_id": "power-a1",
        "python": "/opt/gpu fault/python",
        "restore_deadline_monotonic": 1100.0,
        "load_seconds": 40,
    }
    with pytest.raises(probe.ProbeError, match="unsupported path characters"):
        probe.power_unit_contents(record)


def test_power_unit_state_rejects_foreign_or_unloadable_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = [
        "Id=other.service\nLoadState=loaded\n",
        "Id=a.service\nLoadState=masked\n",
    ]
    monkeypatch.setattr(
        probe, "run", lambda command, **kwargs: _completed(command, answers.pop(0))
    )
    with pytest.raises(probe.ProbeError, match="unit state is unknown"):
        probe.power_unit_state("a.service")
    with pytest.raises(probe.ProbeError, match="could not be loaded"):
        probe.power_unit_state("a.service")


def test_systemd_seconds_reject_a_non_finite_total() -> None:
    with pytest.raises(probe.ProbeError, match="not finite"):
        probe.power_systemd_seconds("9" * 400 + "y")
    assert probe.power_systemd_seconds("1min 30s") == 90


def test_owned_unit_handles_every_absent_unit_shape(power_host: PowerHost) -> None:
    host = power_host
    record = _cycle(host)
    timer, service, _ = host.units()
    contents = probe.power_unit_contents(record)
    assert probe.power_owned_unit(record, timer, allow_absent=True)["LoadState"] == (
        "not-found"
    )
    with pytest.raises(probe.ProbeError, match="owned power unit is missing"):
        probe.power_owned_unit(record, timer)
    path = probe.SYSTEMD_UNIT_DIR / service
    path.write_bytes(contents[service])
    path.chmod(0o644)
    host.overrides[service] = {"LoadState": "not-found"}
    assert probe.power_owned_unit(record, service, allow_absent=True)["LoadState"] == (
        "not-found"
    )
    path.write_bytes(b"[Unit]\nDescription=foreign\n")
    with pytest.raises(probe.ProbeError, match="owned power unit is missing"):
        probe.power_owned_unit(record, service, allow_absent=True)


def test_power_load_record_refuses_another_runs_ownership(
    power_host: PowerHost,
) -> None:
    probe.ACCEPTANCE_STATE.mkdir(mode=0o700)
    owner = probe.ACCEPTANCE_STATE / "gpu-power-owner.json"
    owner.write_bytes(json.dumps({"run_id": "other-run"}).encode())
    owner.chmod(0o600)
    with pytest.raises(probe.ProbeError, match="another power operation owns"):
        probe.power_load_record("power-safety-a1")
    with pytest.raises(probe.ProbeError, match="recovery record is missing"):
        probe.power_load_record("other-run")


def test_stop_and_finish_are_idempotent_after_cleanup(power_host: PowerHost) -> None:
    host = power_host
    record = _cycle(host)
    host.calls.clear()
    probe.power_stop_load(record)
    probe.power_finish_units(record)
    assert [call[1] for call in host.calls if call[0] == "systemctl"] == [
        "show",
        "show",
        "show",
        "daemon-reload",
        "show",
        "show",
        "show",
    ]


def test_finish_units_refuses_to_delete_a_foreign_unit_file(
    power_host: PowerHost,
) -> None:
    host = power_host
    record = _cycle(host)
    _, _, load = host.units()
    path = probe.SYSTEMD_UNIT_DIR / load
    path.write_bytes(b"[Unit]\nDescription=foreign\n")
    path.chmod(0o644)
    with pytest.raises(probe.ProbeError, match="refusing to remove an unowned"):
        probe.power_finish_units(record)
    assert path.exists() is True


def test_verify_removed_units_reports_a_unit_systemd_still_loads(
    power_host: PowerHost,
) -> None:
    host = power_host
    record = _cycle(host)
    timer, _, _ = host.units()
    path = probe.SYSTEMD_UNIT_DIR / timer
    path.write_bytes(probe.power_unit_contents(record)[timer])
    path.chmod(0o644)
    with pytest.raises(probe.ProbeError, match="remains installed after cleanup"):
        probe.power_verify_removed_units(record)


def test_wait_load_start_stops_polling_once_the_deadline_passes(
    power_host: PowerHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    record = host.record()
    probe.power_load_start_path(record["run_id"]).unlink()

    def ticking() -> float:
        host.now += 10.0
        return host.now

    def never_sleep(seconds: float) -> None:
        raise AssertionError("no budget remained, so sleeping is wrong")

    monkeypatch.setattr(
        probe, "time", SimpleNamespace(monotonic=ticking, sleep=never_sleep)
    )
    host.now = 1000.0
    with pytest.raises(probe.ProbeError, match="cannot finish before"):
        probe.power_wait_load_start(
            record, requested_at=host.utc_now(), requested_monotonic=1000.0
        )


def test_wait_load_start_requires_the_load_service_to_own_the_receipt(
    power_host: PowerHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    record = host.record()
    monkeypatch.setattr(
        probe,
        "power_owned_unit",
        lambda record, name, **kwargs: {"ActiveState": "inactive", "MainPID": "0"},
    )
    receipt = json.loads(probe.power_load_start_path(record["run_id"]).read_bytes())
    host.now = receipt["started_monotonic"] + 1.0
    with pytest.raises(probe.ProbeError, match="does not match its start receipt"):
        probe.power_wait_load_start(
            record,
            requested_at=host.epoch,
            requested_monotonic=receipt["started_monotonic"],
        )


def test_load_only_requires_the_verified_power_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = [
        {"index": 0, "uuid": GPU_A, "power_limit_w": 700.0, "power_min_limit_w": 200.0}
    ]
    monkeypatch.setattr(
        probe,
        "power_load_record",
        lambda run_id: {
            "run_id": run_id,
            "load_intended": True,
            "load_closed": False,
            "baseline": baseline,
        },
    )
    monkeypatch.setattr(probe, "checked_power_state", lambda baseline: baseline)
    with pytest.raises(probe.ProbeError, match="requires the verified power cap"):
        probe.power_load_only(argparse.Namespace(run_id="power-a1"))


def test_throttle_refuses_a_leftover_load_start_receipt(power_host: PowerHost) -> None:
    host = power_host
    probe.ACCEPTANCE_STATE.mkdir(mode=0o700)
    receipt = probe.power_load_start_path(host.arguments().run_id)
    receipt.write_bytes(b"{}")
    with pytest.raises(probe.ProbeError, match="receipt already exists"):
        probe.throttle_gpu(host.arguments())
    assert host.power_writes() == []


def test_throttle_requires_cgroup_v2_for_load_cleanup(power_host: PowerHost) -> None:
    host = power_host
    (probe.POWER_CGROUP_ROOT / "cgroup.controllers").unlink()
    with pytest.raises(probe.ProbeError, match="requires cgroup v2"):
        probe.throttle_gpu(host.arguments())
    assert host.power_writes() == []


def test_throttle_refuses_a_recovery_unit_systemd_already_loads(
    power_host: PowerHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = power_host
    # systemd keeps a unit loaded after its file was removed until daemon-reload.
    monkeypatch.setattr(
        probe, "power_unit_state", lambda name: {"Id": name, "LoadState": "loaded"}
    )
    with pytest.raises(probe.ProbeError, match="recovery unit already exists"):
        probe.throttle_gpu(host.arguments())
    assert host.power_writes() == []
    assert list(probe.SYSTEMD_UNIT_DIR.iterdir()) == []


def test_throttle_refuses_a_cap_the_gpus_did_not_take(power_host: PowerHost) -> None:
    host = power_host
    host.ignore_writes = True
    with pytest.raises(probe.ProbeError, match="cap readback does not match"):
        probe.throttle_gpu(host.arguments())
    assert [call[-1] for call in host.power_writes()] == ["200.0", "250.0"]
    assert host.record()["mutation_started"] is True
    assert host.record()["load_intended"] is False


def test_throttle_requires_a_boot_identity(power_host: PowerHost) -> None:
    host = power_host
    probe.BOOT_ID_FILE.write_text("")
    with pytest.raises(probe.ProbeError, match="no boot identity"):
        probe.throttle_gpu(host.arguments())
    assert host.power_writes() == []


def _fm_arguments(cursor: str = "") -> argparse.Namespace:
    return argparse.Namespace(cursor=cursor)


def test_fm_delivery_evidence_rejects_an_unsafe_cursor() -> None:
    with pytest.raises(probe.ProbeError, match="cursor is invalid"):
        probe.fm_delivery_evidence(_fm_arguments("s=1;i=2 --flag"))


@pytest.fixture
def fm_journal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    boot = tmp_path / "boot-id"
    boot.write_text("11111111-2222-3333-4444-555555555555\n")
    monkeypatch.setattr(probe, "BOOT_ID_FILE", boot)
    monkeypatch.setattr(
        probe,
        "active_unit_identity",
        lambda unit, what: ({"InvocationID": "c" * 32}, "4242"),
    )
    journal: dict[str, Any] = {"stdout": "", "returncode": 0, "commands": []}

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        journal["commands"].append(command)
        return _completed(command, journal["stdout"], journal["returncode"])

    monkeypatch.setattr(probe, "run", run)
    return journal


def test_fm_delivery_evidence_requires_a_hex_producer_invocation(
    fm_journal: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        probe,
        "active_unit_identity",
        lambda unit, what: ({"InvocationID": "not-an-invocation"}, "4242"),
    )
    with pytest.raises(probe.ProbeError, match="producer process identity"):
        probe.fm_delivery_evidence(_fm_arguments())
    assert fm_journal["commands"] == []


def test_fm_delivery_evidence_reports_a_failed_journal_read(
    fm_journal: dict[str, Any],
) -> None:
    fm_journal["returncode"] = 1
    with pytest.raises(probe.ProbeError, match="journal receipt read failed"):
        probe.fm_delivery_evidence(_fm_arguments())
    assert fm_journal["commands"][0][0] == "journalctl"
    assert "--boot=11111111222233334444555555555555" in fm_journal["commands"][0]


def test_fm_delivery_evidence_skips_blank_lines_and_rejects_double_receipts(
    fm_journal: dict[str, Any],
) -> None:
    entry = {
        "_BOOT_ID": "11111111222233334444555555555555",
        "MESSAGE": f"{probe.FM_RECEIPT_PREFIX}{{}} {probe.FM_RECEIPT_PREFIX}{{}}",
        "__CURSOR": "s=1",
        "_SYSTEMD_INVOCATION_ID": "c" * 32,
        "_PID": "4242",
        "__MONOTONIC_TIMESTAMP": "1",
    }
    fm_journal["stdout"] = "\n\n" + json.dumps(entry) + "\n   \n"
    with pytest.raises(probe.ProbeError, match="no unique receipt payload"):
        probe.fm_delivery_evidence(_fm_arguments())


def test_inventory_sink_rejects_a_third_delivery_and_foreign_paths() -> None:
    sink = probe.InventorySampleSink("/host-telemetry")
    sink.post("/host-telemetry", {"batch": 1})
    sink.post("/host-telemetry", {"batch": 2})
    with pytest.raises(probe.ProbeError, match="rejected an extra delivery"):
        sink.post("/host-telemetry", {"batch": 3})
    with pytest.raises(probe.ProbeError, match="rejected an extra delivery"):
        probe.InventorySampleSink("/host-telemetry").post("/other", {})
    assert sink.batches == [{"batch": 1}, {"batch": 2}]


def _publish_arguments(**changes: Any) -> argparse.Namespace:
    return argparse.Namespace(
        **{
            "confirm": "PUBLISH_GPU_INVENTORY",
            "receipt_json": "{}",
            "run_id": "collect004-campaign-fixture-a1",
            "expected_sha256": "0" * 64,
            **changes,
        }
    )


def test_publish_inventory_requires_the_confirmation_and_an_object_receipt() -> None:
    with pytest.raises(probe.ProbeError, match="not authorized"):
        probe.publish_gpu_inventory(_publish_arguments(confirm="yes"))
    with pytest.raises(probe.ProbeError, match="not authorized"):
        probe.publish_gpu_inventory(
            _publish_arguments(receipt_json="[" + "1," * 70_000 + "1]")
        )
    with pytest.raises(probe.ProbeError, match="receipt is malformed"):
        probe.publish_gpu_inventory(_publish_arguments(receipt_json="[]"))


def test_publish_inventory_requires_a_configured_https_endpoint(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        probe, "inventory_publication_batches", lambda receipt, **kwargs: []
    )
    with pytest.raises(probe.ProbeError, match="requires the configured HTTPS"):
        probe.publish_gpu_inventory(_publish_arguments())
    assert env_host.emitted == []
