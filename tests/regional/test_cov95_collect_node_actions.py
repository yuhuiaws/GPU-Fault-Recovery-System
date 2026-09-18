"""Reversible node operations with fake syscalls and local recovery receipts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import collector_node_probe as probe
from tests.regional import _cov95_collect_node as node_support
from tests.regional._cov95_collect_net import (
    StopLoop,
    no_external_effects,  # noqa: F401
)
from tests.regional.test_collector_env_safety import EnvHost
from tests.regional.test_collector_env_safety import env_host as env_host_fixture
from tests.regional.test_collector_power_safety import PowerHost
from tests.regional.test_collector_power_safety import power_host as power_host_fixture

power_host = power_host_fixture
env_host = env_host_fixture
node_host = node_support.node_host


@pytest.fixture
def collector_env_host(env_host: EnvHost) -> EnvHost:
    env_host.run_id = "env-a"
    return env_host


def test_power_load_binary_discovery_is_versioned(node_host: Any) -> None:
    host = node_host
    for name in ("dcgmproftester12", "dcgmproftester13"):
        host.paths("/usr/bin").joinpath(name).touch()
    host.paths("/usr/bin/dcgmproftester14").mkdir()
    assert probe.proftester_binary() == str(host.paths("/usr/bin/dcgmproftester13"))


def test_power_throttle_arms_then_loads_and_restores_before_disarming(
    power_host: PowerHost,
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    assert host.emitted[-1]["gpu_power"][0]["power_limit_w"] == 200
    arm_index = next(
        i
        for i, call in enumerate(host.calls)
        if call[:3] == ["systemctl", "enable", "--now"]
    )
    assert arm_index < next(i for i, call in enumerate(host.calls) if "-pl" in call)
    host.calls.clear()
    probe.restore_gpu_power_limit(host.arguments())
    assert host.emitted[-1]["gpu_power"][0]["power_limit_w"] == 700
    assert next(i for i, call in enumerate(host.calls) if "-pl" in call) < next(
        i
        for i, call in enumerate(host.calls)
        if call[:3] == ["systemctl", "disable", "--now"] and call[-1].endswith(".timer")
    )


@pytest.mark.parametrize(
    "failure",
    [
        "no-gpus",
        "index",
        "min-unknown",
        "default-unknown",
        "min-high",
        "nondefault",
        "load-zero",
        "load-too-long",
        "no-binary",
    ],
)
def test_throttle_rejects_unsafe_power_or_deadline_before_writes(
    power_host: PowerHost, failure: str
) -> None:
    host = power_host
    args = host.arguments()
    if failure == "no-gpus":
        host.power_output = ""
    elif failure == "index":
        args.gpu_index = 99
    elif failure == "min-unknown":
        host.gpus[0]["power_min_limit_w"] = "N/A"
    elif failure == "default-unknown":
        host.gpus[0]["power_default_limit_w"] = "N/A"
    elif failure == "min-high":
        host.gpus[0]["power_min_limit_w"] = 800
    elif failure == "nondefault":
        host.gpus[0]["power_limit_w"] = 250
    elif failure == "load-zero":
        args.load_seconds = 0
    elif failure == "load-too-long":
        args.load_seconds = 600
    elif failure == "no-binary":
        (host.root / "bin/dcgmproftester13").unlink()
    with pytest.raises(probe.ProbeError):
        probe.throttle_gpu(args)
    assert not any(
        call[:2] == ["systemctl", "enable"] or "-pl" in call for call in host.calls
    ), "invalid power premise cannot arm or change GPU state"


@pytest.mark.parametrize(
    "failure", ["no-default", "after-empty", "after-identity", "after-limit"]
)
def test_power_restore_keeps_timer_when_readback_is_unproven(
    power_host: PowerHost, failure: str
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    host.calls.clear()
    if failure == "no-default":
        host.gpus[0]["power_default_limit_w"] = "N/A"
    elif failure == "after-limit":
        host.ignore_writes = True
    else:

        def invalidate(command: list[str]) -> None:
            if "-pl" in command:
                if failure == "after-empty":
                    host.power_output = ""
                else:
                    host.gpus[0]["uuid"] = "GPU-ffffffff-0000-0000-0000-000000000000"

        host.after_call = invalidate
    with pytest.raises(probe.ProbeError):
        probe.restore_gpu_power_limit(host.arguments())
    assert not any(
        call[:2] == ["systemctl", "disable"] and call[-1].endswith(".timer")
        for call in host.calls
    ), "unverified restoration must leave the independent timer armed"
    assert host.units()[0] in host.active


@pytest.mark.parametrize("fail_write", [False, True])
def test_xid_protocol_closes_descriptor_on_all_write_outcomes(
    node_host: Any, monkeypatch: Any, fail_write: bool
) -> None:
    host = node_host
    arguments = probe.parser().parse_args(
        ["write-xid", "--xid", "63", "--marker", "marker-a", "--pci-bdf", "0000:af:00"]
    )
    if fail_write:

        def write(*a: Any, **k: Any) -> None:
            raise OSError("write failure")

        monkeypatch.setattr(probe.os, "write", write)
        with pytest.raises(OSError):
            probe.write_xid(arguments)
    else:
        probe.write_xid(arguments)
        assert host.emitted[-1]["pci_bdf"] == "0000:af:00.0"
        assert b"Xid (PCI:0000:af:00): 63," in host.writes[-1][1]
    assert host.closed == [42]
    arguments.xid = 999999
    with pytest.raises(probe.ProbeError, match="not allowlisted"):
        probe.write_xid(arguments)
    with pytest.raises(probe.ProbeError, match="unsafe"):
        probe.safe_id("bad marker", "marker")


@pytest.mark.parametrize("initially_bound", [False, True])
@pytest.mark.parametrize("timer_fired", [False, True])
def test_efa_restore_reports_preexisting_binding_and_timer_provenance(
    node_host: Any, monkeypatch: Any, initially_bound: bool, timer_fired: bool
) -> None:
    host = node_host
    driver = host.paths("/sys/bus/pci/drivers/efa")
    bdf = "0000:af:00.0"
    if initially_bound:
        (driver / bdf).touch()
    args = SimpleNamespace(run_id="efa-a", pci_bdf=bdf, restore_seconds=300)
    original = Path.write_text

    def write(path: Path, text: str, *a: Any, **k: Any) -> int:
        count = original(path, text, *a, **k)
        if path == driver / "bind":
            (driver / bdf).touch()
        return count

    monkeypatch.setattr(Path, "write_text", write)
    host.started = "fixture-timestamp" if timer_fired else ""
    probe.restore_efa(args)
    assert host.emitted[-1]["already_bound"] is initially_bound
    assert host.emitted[-1]["timer_fired"] is timer_fired
    assert host.emitted[-1]["bound"] is True
    probe.unbind_efa(args)
    assert (driver / "unbind").read_text() == bdf + "\n"
    (driver / bdf).unlink()
    with pytest.raises(probe.ProbeError, match="not currently bound"):
        probe.unbind_efa(args)


def override(host: EnvHost) -> None:
    host.override()


@pytest.mark.parametrize("trigger", ["manual", "new-boot", "elapsed", "defer"])
def test_collector_env_restore_preserves_exact_bytes_and_private_mode(
    collector_env_host: EnvHost, trigger: str
) -> None:
    host = collector_env_host
    original = probe.COLLECTOR_ENV.read_bytes()
    override(host)
    assert probe.parse_env()["GPU_FAULT_EXPECTED_GPU_COUNT"] == "9"
    assert probe.COLLECTOR_ENV.stat().st_mode & 0o777 == 0o600
    backup, unit = probe.collector_restore_paths("env-a")
    assert backup.read_bytes() == original
    recovery_paths = (
        backup,
        backup.with_suffix(".probe.py"),
        probe.SYSTEMD_UNIT_DIR / f"{unit}.service",
        probe.SYSTEMD_UNIT_DIR / f"{unit}.timer",
    )
    if trigger == "new-boot":
        probe.BOOT_ID_FILE.write_text("boot-b")
    if trigger == "elapsed":
        host.now += 601
    host.calls.clear()
    probe.restore_collector_env(host.arguments(automatic=trigger != "manual"))
    if trigger == "defer":
        assert host.emitted[-1]["deferred"] is True, (
            "the original boot must retain its unexpired override window"
        )
        assert host.emitted[-1]["restored"] is False, (
            "deferral cannot claim that the override was restored"
        )
        assert all(path.exists() for path in recovery_paths), (
            "same-boot premature callback must retain all recovery evidence"
        )
        assert probe.parse_env()["GPU_FAULT_EXPECTED_GPU_COUNT"] == "9", (
            "a premature callback must not change the override"
        )
        assert host.mutations() == [], "deferral must not restart or disarm any service"
        return
    assert host.emitted[-1]["restored"] is True, (
        "an eligible restore must confirm the original environment"
    )
    assert probe.COLLECTOR_ENV.read_bytes() == original, (
        "restore must preserve every original byte"
    )
    assert probe.COLLECTOR_ENV.stat().st_mode & 0o777 == 0o600, (
        "restored configuration must retain private permissions"
    )
    if trigger != "manual":
        assert host.emitted[-1]["cleanup_deferred"] is True, (
            "automatic restoration cannot claim independent service verification"
        )
        assert all(path.exists() for path in recovery_paths), (
            "automatic restoration must retain the backup, record, probe and units"
        )
        assert backup.read_bytes() == original, (
            "automatic restoration must leave trustworthy original recovery bytes"
        )
        assert host.mutations() == [
            ["systemctl", "--no-block", "restart", probe.HOST_COLLECTOR_UNIT]
        ], "automatic restore must not block its boot dependency or disarm recovery"
        host.calls.clear()
        probe.restore_collector_env(host.arguments())
    assert host.emitted[-1]["restored"] is True, (
        "manual cleanup must report verified restoration, not just retired files"
    )
    assert host.emitted[-1]["cleanup_verified"] is True, (
        "manual restore must complete service verification before retiring recovery"
    )
    assert host.calls.index(
        ["systemctl", "restart", probe.HOST_COLLECTOR_UNIT]
    ) < host.calls.index(["systemctl", "disable", "--now", f"{unit}.timer"]), (
        "collector health must be verified before its recovery timer is disarmed"
    )
    assert not any(path.exists() for path in recovery_paths), (
        "verified manual restore must retire all owned recovery state"
    )
    assert host.record()["state"] == "CLEANED", "keep the nonsecret retry receipt"


def test_manual_restore_keeps_recovery_evidence_when_service_health_is_unknown(
    collector_env_host: EnvHost,
) -> None:
    host = collector_env_host
    original = probe.COLLECTOR_ENV.read_bytes()
    override(host)
    backup, unit = probe.collector_restore_paths("env-a")
    host.calls.clear()
    host.overrides[probe.HOST_COLLECTOR_UNIT] = {"InvocationID": ""}

    with pytest.raises(probe.ProbeError, match="health or job completion is unknown"):
        probe.restore_collector_env(host.arguments())

    assert probe.COLLECTOR_ENV.read_bytes() == original, (
        "environment restoration alone does not prove service recovery"
    )
    assert backup.read_bytes() == original, (
        "unverified service recovery must preserve trustworthy original bytes"
    )
    assert all(
        path.exists()
        for path in (
            probe.collector_override_record(backup),
            backup.with_suffix(".probe.py"),
            probe.SYSTEMD_UNIT_DIR / f"{unit}.service",
            probe.SYSTEMD_UNIT_DIR / f"{unit}.timer",
        )
    ), "unknown service health must retain the complete independent recovery path"
    assert not any(call[:2] == ["systemctl", "disable"] for call in host.calls), (
        "unknown service health cannot authorize disarming recovery"
    )


@pytest.mark.parametrize(
    "bad", ["no-env", "wrong-target", "duplicate-key", "existing-recovery"]
)
def test_env_override_rejects_invalid_baseline_or_existing_receipt(
    collector_env_host: EnvHost, bad: str
) -> None:
    if bad == "no-env":
        probe.COLLECTOR_ENV.unlink()
    elif bad == "wrong-target":
        probe.COLLECTOR_ENV.write_text("GPU_FAULT_EXPECTED_GPU_COUNT=10\n")
    elif bad == "duplicate-key":
        probe.COLLECTOR_ENV.write_text(
            "GPU_FAULT_EXPECTED_GPU_COUNT=2\nGPU_FAULT_EXPECTED_GPU_COUNT=2\n"
        )
    else:
        backup, _unit = probe.collector_restore_paths("env-a")
        backup.parent.mkdir(mode=0o700, parents=True)
        backup.write_text("preserve")
    with pytest.raises(probe.ProbeError):
        override(collector_env_host)
    assert collector_env_host.mutations() == [], (
        "refused override must not touch collector or timer"
    )


@pytest.mark.parametrize("when", ["arm", "restart", "abort", "rollback"])
def test_failed_override_recovers_or_retains_retry_receipt(
    collector_env_host: EnvHost, when: str
) -> None:
    host = collector_env_host
    original = probe.COLLECTOR_ENV.read_bytes()
    error: BaseException = (
        StopLoop() if when == "abort" else RuntimeError("fixture failure")
    )
    key = ("systemctl", "enable") if when == "arm" else ("systemctl", "restart")
    host.failures[key] = [error]
    with pytest.raises(type(error)) as caught:
        override(host)
    assert caught.value is error
    backup, _ = probe.collector_restore_paths("env-a")
    assert backup.exists(), "failed setup keeps its nonce-bound recovery intent"
    if when == "rollback":
        host.failures[("systemctl", "restart")] = [RuntimeError("restore failed")]
        with pytest.raises(RuntimeError, match="restore failed"):
            probe.restore_collector_env(host.arguments())
        assert backup.exists(), (
            "test_failed_override_recovers_or_retains_retry_receipt: expected backup.exists()"
        )
    probe.restore_collector_env(host.arguments())
    assert probe.COLLECTOR_ENV.read_bytes() == original
    assert not backup.exists(), "matching-owner cleanup retires the sensitive backup"
    assert host.record()["state"] == "CLEANED"


@pytest.mark.parametrize(
    "record",
    [
        None,
        [],
        "{bad",
        {"restore_after_monotonic": True},
        {"boot_id": "boot-a", "restore_after_monotonic": float("inf")},
    ],
)
def test_invalid_automatic_recovery_identity_never_changes_env(
    collector_env_host: EnvHost, record: Any
) -> None:
    backup, _ = probe.collector_restore_paths("env-a")
    backup.parent.mkdir(mode=0o700, parents=True)
    if record is not None:
        probe.collector_override_record(backup).write_text(
            record if isinstance(record, str) else json.dumps(record)
        )
        probe.collector_override_record(backup).chmod(0o600)
    original = probe.COLLECTOR_ENV.read_bytes()
    with pytest.raises(probe.ProbeError):
        probe.restore_collector_env(collector_env_host.arguments(automatic=True))
    assert probe.COLLECTOR_ENV.read_bytes() == original
    assert collector_env_host.mutations() == []
    if record is None:
        probe.restore_collector_env(collector_env_host.arguments())
        assert collector_env_host.emitted[-1]["state"] == "NOT_STARTED"
    else:
        with pytest.raises(probe.ProbeError):
            probe.restore_collector_env(collector_env_host.arguments())


@pytest.mark.parametrize(
    "failure",
    ["missing-backup", "corrupt-backup", "write-not-applied", "missing-but-restored"],
)
def test_restore_requires_exact_original_digest(
    collector_env_host: EnvHost, monkeypatch: Any, failure: str
) -> None:
    original = probe.COLLECTOR_ENV.read_bytes()
    override(collector_env_host)
    backup, _ = probe.collector_restore_paths("env-a")
    if failure in {"missing-backup", "missing-but-restored"}:
        backup.unlink()
        if failure == "missing-but-restored":
            probe.COLLECTOR_ENV.write_bytes(original)
    elif failure == "corrupt-backup":
        backup.write_bytes(b"corrupt")
    else:
        monkeypatch.setattr(probe, "replace_collector_env", lambda *a, **k: None)
    collector_env_host.calls.clear()
    with pytest.raises(probe.ProbeError):
        probe.restore_collector_env(collector_env_host.arguments())
    assert not any(call[1] == "disable" for call in collector_env_host.calls), (
        "missing recovery bytes or a failed write cannot authorize disarming"
    )
    collector_env_host.now += 601
    with pytest.raises(probe.ProbeError):
        probe.restore_collector_env(collector_env_host.arguments(automatic=True))


def test_atomic_env_replacement_copies_owner_and_cleans_failed_tempfile(
    collector_env_host: EnvHost, monkeypatch: Any
) -> None:
    owner = probe.COLLECTOR_ENV.stat()
    changes: list[Any] = []
    original_fstat = probe.os.fstat
    original_create = probe.tempfile.NamedTemporaryFile
    temporaries: list[Any] = []

    def create(*args: Any, **kwargs: Any) -> Any:
        result = original_create(*args, **kwargs)
        temporaries.append(result)
        return result

    def fstat(fd: int) -> os.stat_result:
        current = original_fstat(fd)
        if (
            temporaries
            and not temporaries[-1].file.closed
            and fd == temporaries[-1].fileno()
        ):
            fields = list(current)
            fields[4], fields[5] = owner.st_uid + 1, owner.st_gid + 1
            return os.stat_result(fields)
        return current

    monkeypatch.setattr(probe.tempfile, "NamedTemporaryFile", create)
    monkeypatch.setattr(probe.os, "fstat", fstat)
    monkeypatch.setattr(
        probe.os, "fchown", lambda fd, uid, gid: changes.append((uid, gid))
    )
    digest = hashlib.sha256(probe.COLLECTOR_ENV.read_bytes()).hexdigest()
    probe.replace_collector_env(
        b"GPU_FAULT_EXPECTED_GPU_COUNT=2\n", expected_sha256=digest
    )
    assert changes == [(owner.st_uid, owner.st_gid)]
    assert not list(probe.COLLECTOR_ENV.parent.glob(".collector-acceptance-*")), (
        "temporary file must be retired"
    )

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("creation refused")

    monkeypatch.setattr(probe.tempfile, "NamedTemporaryFile", fail)
    with pytest.raises(OSError, match="creation refused"):
        probe.replace_collector_env(
            b"unused",
            expected_sha256=hashlib.sha256(
                probe.COLLECTOR_ENV.read_bytes()
            ).hexdigest(),
        )


def test_main_success_dispatches_fake_host_command(
    node_host: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "write-xid",
            "--xid",
            "63",
            "--marker",
            "m",
            "--pci-bdf",
            "0000:af:00",
        ],
    )
    assert probe.main() == 0
    assert node_host.emitted[-1]["xid"] == 63
