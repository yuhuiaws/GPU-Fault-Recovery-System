"""Contract tests for the on-node EFA driver fixture probe.

Nothing here runs systemd or touches a real device: ``run`` (systemctl /
systemd-run) is a fake, and ``/sys`` is a small tree under ``tmp_path`` the
module's path constants are pointed at. The tests cover the parsing of the EFA
inventory, the safety preconditions that refuse to touch the node unless
exactly one spare function is safe, the deadman arming that fails closed, the
release that restores the baseline or fails loudly, the kernel's own
``driver_override`` read-back that both variants must accept, the trap that
waits for the fixture's unbind before the product's bind and then re-unbinds on
the product's post-repair snapshot (the host collector's restart, read-only;
never before its floor, on the grace when no restart shows), and the failure
line the host fixture turns into the case's ``probe-error.json``.
"""

from __future__ import annotations

import filecmp
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import host_probe_fixture as host
from scripts.e2e.regional.probes import general_live_efa_probe as probe
from tests.regional._host_probe_support import ProbeApi, host_probe

BDF_A = "0000:30:00.0"
BDF_B = "0000:31:00.0"
RUN = "gl013-x-a1-dead"
#: What ``cat /sys/bus/pci/devices/<bdf>/driver_override`` prints for a function
#: without an override: ``driver_override_show()`` (drivers/pci/pci-sysfs.c)
#: formats the NULL ``pdev->driver_override`` with ``"%s\n"`` and vsprintf
#: renders a NULL ``%s`` as ``(null)`` (lib/vsprintf.c ``check_pointer_msg()``
#: at v6.1, ``string()`` at v4.19). The 2026-10-27 live survey of the HyperPod
#: test node (kernel 6.12.103) read exactly this on all 16 EFA functions; the
#: fixture compared it against ``"(none)"`` and refused.
KERNEL_UNSET = "(null)\n"


class FakeRun:
    """systemctl / systemd-run without systemd; records the commands."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, command, *, check=True, timeout=120, input_text=None):
        self.calls.append(list(command))
        text = ""
        if command[:2] == ["systemctl", "show"]:
            # A deadman/trap query: report the timer active and loaded.
            text = "LoadState=loaded\nActiveState=active\nSubState=waiting\nMainPID=0\n"

        class Completed:
            returncode = 0
            stdout = text
            stderr = ""

        return Completed()


def _tree(
    tmp_path: Path,
    monkeypatch,
    *,
    functions=((BDF_A, True, None), (BDF_B, True, None)),
    module: bool = True,
    allow_remediation: bool = True,
    remediation_value: str | None = None,
) -> FakeRun:
    """A fake ``/sys``: ``override`` ``None`` is the kernel's unset read-back
    (``(null)\\n``); a string is a stored name (``""`` writes a bare newline)."""

    sysfs = tmp_path / "sys"
    devices = sysfs / "bus" / "pci" / "devices"
    drivers = sysfs / "bus" / "pci" / "drivers" / "efa"
    ib = sysfs / "class" / "infiniband"
    for path in (devices, drivers, ib):
        path.mkdir(parents=True, exist_ok=True)
    for index, (bdf, bound, override) in enumerate(functions):
        device = devices / bdf
        device.mkdir(parents=True, exist_ok=True)
        (device / "vendor").write_text("0x1d0f\n")
        (device / "device").write_text("0xefa1\n")
        (device / "driver_override").write_text(
            KERNEL_UNSET if override is None else override + "\n"
        )
        if bound:
            (device / "driver").symlink_to(drivers)
            (drivers / bdf).symlink_to(device)
        rdma = ib / f"rdmap{index}"
        rdma.mkdir(parents=True, exist_ok=True)
        (rdma / "device").symlink_to(device)
    if module:
        (sysfs / "module" / "efa").mkdir(parents=True, exist_ok=True)
    env_dir = tmp_path / "etc"
    env_dir.mkdir(exist_ok=True)
    collector_env = env_dir / "collector.env"
    if remediation_value is None:
        remediation_value = "true" if allow_remediation else "false"
    collector_env.write_text(
        "GPU_FAULT_NODE_ALLOW_EFA_DRIVER_REMEDIATION="
        + remediation_value
        + "\nGPU_FAULT_EXPECTED_EFA_DEVICE_COUNT=16\n"
    )
    boot = tmp_path / "boot_id"
    boot.write_text("boot-xyz\n")
    monkeypatch.setattr(probe, "SYSFS", sysfs)
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run" / "efa-driver")
    monkeypatch.setattr(probe, "COLLECTOR_ENV", collector_env)
    monkeypatch.setattr(probe, "NODE_AGENT_ENV", env_dir / "node-agent.env")
    monkeypatch.setattr(probe, "BOOT_ID_FILE", boot)
    fake = FakeRun()
    monkeypatch.setattr(probe, "run", fake)
    return fake


def _last_json(capsys) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def _ns(**values: Any):
    return type("NS", (), values)()


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0000:30:00.0", "0000:30:00.0"),
        ("00000000:30:00.0", "0000:30:00.0"),
        ("0000:30:00", "0000:30:00.0"),
        ("0000:AB:1F.3", "0000:ab:1f.3"),
    ],
)
def test_normalize_bdf(raw: str, expected: str) -> None:
    assert probe.normalize_bdf(raw) == expected


@pytest.mark.parametrize("bad", ["", "lol", "30:00.0", "0000:30:00.8", "../etc"])
def test_normalize_bdf_rejects_unsafe(bad: str) -> None:
    with pytest.raises(probe.ProbeError):
        probe.normalize_bdf(bad)


def test_bounded_and_run_id() -> None:
    assert probe.bounded(300, 60, 5400, "x") == 300
    with pytest.raises(probe.ProbeError):
        probe.bounded(10, 60, 5400, "x")
    with pytest.raises(probe.ProbeError):
        probe.checked_run_id("../nope")
    assert probe.unit_name("gl013-x-a1-dead", "deadman").startswith("gfrs-"), (
        "transient unit names must carry the gfrs- prefix"
    )


# --------------------------------------------------------------------------- #
# The kernel's unset-override read-back
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (KERNEL_UNSET, None),
        ("(null)", None),
        ("", None),
        ("\n", None),
        ("efa\n", "efa"),
        ("vfio-pci\n", "vfio-pci"),
        (probe.HOLD_OVERRIDE + "\n", probe.HOLD_OVERRIDE),
        ("none\n", "none"),
        ("(none)\n", "(none)"),
    ],
    ids=[
        "kernel-null-line",
        "kernel-null",
        "empty",
        "newline",
        "efa",
        "vfio-pci",
        "hold",
        "abi-opt-out-none",
        "old-harness-constant",
    ],
)
def test_override_name_reads_the_kernel_s_rendering(
    text: str, expected: str | None
) -> None:
    """``driver_override_show()`` prints the ``pdev->driver_override`` pointer
    with ``"%s\\n"`` (``sprintf`` at v3.16, ``sysfs_emit`` at v5.18/v6.1) and a
    NULL ``%s`` renders as ``(null)`` (``lib/vsprintf.c`` ``string()`` v4.19,
    ``check_pointer_msg()`` v6.1), so an unset override reads ``(null)``; the
    store clears on ``""``/``"\\n"`` instead of storing an empty name
    (``driver_set_override()``, v6.12), so an empty read-back is unset too. Any
    stored name -- including the ABI's opt-out example ``none`` and the old
    harness constant ``(none)`` -- is a real override."""

    assert probe.KERNEL_NO_OVERRIDE == "(null)"
    assert probe.override_name(text) == expected


def test_read_override_refuses_a_function_without_the_attribute(
    tmp_path: Path, monkeypatch
) -> None:
    _tree(tmp_path, monkeypatch)
    (probe.device_dir(BDF_A) / "driver_override").unlink()
    with pytest.raises(probe.ProbeError, match="no driver_override attribute") as got:
        probe.read_override(BDF_A)
    assert got.value.reason == "no_override_attribute"


def test_summary_keeps_the_kernel_s_words_beside_the_parsed_override(
    tmp_path: Path, monkeypatch
) -> None:
    _tree(tmp_path, monkeypatch)
    value = probe.summary(RUN, BDF_A)
    assert value["driver_override"] is None
    assert value["driver_override_sysfs"] == "(null)"
    assert [item["driver_override"] for item in value["functions"]] == [None, None]


def test_hold_unbind_accepts_the_kernel_s_null_read_back(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """bind_failure on a real node: ``(null)`` is no override, so the hold
    proceeds -- deadman armed, state saved, the hold name written, the
    function's unbind receipt written to the driver's ``unbind``."""

    fake = _tree(tmp_path, monkeypatch)
    probe.hold_unbind(_ns(run_id=RUN, pci_bdf=BDF_A, restore_seconds=600))
    receipt = _last_json(capsys)
    assert receipt["mode"] == "hold"
    assert receipt["driver_override"] == probe.HOLD_OVERRIDE
    device = probe.device_dir(BDF_A)
    assert (device / "driver_override").read_text() == probe.HOLD_OVERRIDE + "\n"
    assert (probe.driver_root() / "unbind").read_text() == BDF_A + "\n"
    armed = [call for call in fake.calls if call[0] == "systemd-run"]
    assert [call[1] for call in armed] == [f"--unit={probe.unit_name(RUN, 'deadman')}"]
    state = json.loads((probe.state_dir(RUN) / "state.json").read_text())
    assert state["override_value"] == probe.HOLD_OVERRIDE


def test_trap_unbind_accepts_the_kernel_s_null_read_back(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """post_repair_failure shares the precondition; its trap unit must not be
    ``Type=oneshot`` (systemd-run would block until the trap exits and
    ``RuntimeMaxSec`` would not apply, systemd-run(1) / systemd.service(5))."""

    fake = _tree(tmp_path, monkeypatch)
    probe.trap_unbind(
        _ns(
            run_id=RUN,
            pci_bdf=BDF_A,
            restore_seconds=900,
            grace_seconds=5,
            trap_window_seconds=600,
        )
    )
    receipt = _last_json(capsys)
    assert receipt["mode"] == "trap"
    # This variant sets no override: the kernel's read-back stays untouched.
    assert (probe.device_dir(BDF_A) / "driver_override").read_text() == KERNEL_UNSET
    assert (probe.driver_root() / "unbind").read_text() == BDF_A + "\n"
    traps = [c for c in fake.calls if c[0] == "systemd-run" and "trap-run" in c]
    assert len(traps) == 1
    (trap,) = traps
    assert "--property=Type=oneshot" not in trap
    assert [arg for arg in trap if arg.startswith("--property=RuntimeMaxSec=")] == [
        f"--property=RuntimeMaxSec={600 + 5 + 30}"
    ]


@pytest.mark.parametrize("command", ["hold-unbind", "trap-unbind"])
@pytest.mark.parametrize("override", ["efa", "vfio-pci", "none"])
def test_both_variants_refuse_a_real_override(
    tmp_path: Path, monkeypatch, capsys, command: str, override: str
) -> None:
    """A stored name means another actor holds the function: both variants
    refuse before any mutation, and the failure line names the reason and
    carries the message whose digest the exception text shows (live
    2026-10-27 the digest ``ce15ef168e94b634`` was sha256 of
    ``"0000:55:00.0 already has a driver_override: '(null)'"``)."""

    fake = _tree(
        tmp_path, monkeypatch, functions=((BDF_A, True, override), (BDF_B, True, None))
    )
    assert probe.main([command, "--run-id", RUN, "--pci-bdf", BDF_A]) == 1
    line = _last_json(capsys)
    assert line["error_class"] == "ProbeError"
    assert line["error_kind"] == "probe"
    assert line["error_reason"] == "driver_override_set"
    assert (
        line["error_message"] == f"{BDF_A} already has a driver_override: {override!r}"
    )
    expected_code = hashlib.sha256(line["error_message"].encode()).hexdigest()[:16]
    assert line["error_code"] == expected_code
    assert line["error_site"].split(":")[0] == "_preconditions"
    assert line["command"] == command
    # Nothing on the node changed: no override write, no unbind, no unit.
    assert (probe.device_dir(BDF_A) / "driver_override").read_text() == override + "\n"
    assert not (probe.driver_root() / "unbind").exists(), "refused before the unbind"
    assert [call for call in fake.calls if call[0] == "systemd-run"] == []


@pytest.mark.parametrize(
    "value",
    ['"true"', "1", "yes", "ON"],
    ids=["installer-quoted-true", "one", "yes", "upper-on"],
)
def test_remediation_flag_accepts_the_product_s_boolean_tokens(
    tmp_path: Path, monkeypatch, value: str
) -> None:
    """The installer's ``write_env`` writes ``KEY="true"`` (``systemd_quote``)
    and the product's ``env_bool`` accepts 1/true/yes/on case-insensitively;
    the probe must not refuse a node the product itself has enabled."""

    _tree(tmp_path, monkeypatch, remediation_value=value)
    assert probe.main(["hold-unbind", "--run-id", RUN, "--pci-bdf", BDF_A]) == 0


# --------------------------------------------------------------------------- #
# Inventory parsing
# --------------------------------------------------------------------------- #
def test_efa_functions_lists_bound_functions(tmp_path: Path, monkeypatch) -> None:
    _tree(tmp_path, monkeypatch)
    functions = probe.efa_functions()
    assert {item["pci_bdf"] for item in functions} == {BDF_A, BDF_B}
    assert all(item["driver"] == "efa" and item["bound"] for item in functions), (
        functions
    )
    assert all(item["rdma_device"] for item in functions), functions
    assert [item["driver_override"] for item in functions] == [None, None]


def test_list_functions_emits_json(tmp_path: Path, monkeypatch, capsys) -> None:
    _tree(tmp_path, monkeypatch)
    probe.list_functions(_ns(run_id="gl013-x-a1-dead"))
    payload = _last_json(capsys)
    assert payload["bound_count"] == 2
    assert payload["module_loaded"] is True


# --------------------------------------------------------------------------- #
# Safety preconditions
# --------------------------------------------------------------------------- #
def test_preconditions_pass_for_one_spare_function(tmp_path: Path, monkeypatch) -> None:
    _tree(tmp_path, monkeypatch)
    facts = probe._preconditions(BDF_A)
    assert facts["baseline_bound_count"] == 2


def test_preconditions_refuse_the_last_bound_function(
    tmp_path: Path, monkeypatch
) -> None:
    _tree(tmp_path, monkeypatch, functions=((BDF_A, True, None), (BDF_B, False, None)))
    with pytest.raises(probe.ProbeError, match="last one"):
        probe._preconditions(BDF_A)


def test_preconditions_refuse_an_existing_override(tmp_path: Path, monkeypatch) -> None:
    _tree(
        tmp_path,
        monkeypatch,
        functions=((BDF_A, True, "vfio-pci"), (BDF_B, True, None)),
    )
    with pytest.raises(probe.ProbeError, match="driver_override"):
        probe._preconditions(BDF_A)


def test_preconditions_refuse_when_module_missing(tmp_path: Path, monkeypatch) -> None:
    _tree(tmp_path, monkeypatch, module=False)
    with pytest.raises(probe.ProbeError, match="module is not loaded"):
        probe._preconditions(BDF_A)


def test_preconditions_refuse_when_remediation_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    _tree(tmp_path, monkeypatch, allow_remediation=False)
    with pytest.raises(probe.ProbeError, match="REMEDIATE_EFA_DRIVER"):
        probe._preconditions(BDF_A)


def test_preconditions_refuse_a_non_efa_function(tmp_path: Path, monkeypatch) -> None:
    _tree(tmp_path, monkeypatch)
    device = probe.device_dir(BDF_A)
    (device / "vendor").write_text("0x10de\n")  # an NVIDIA GPU, not EFA
    with pytest.raises(probe.ProbeError, match="not an EFA"):
        probe._preconditions(BDF_A)


# --------------------------------------------------------------------------- #
# The trap waits for the fixture's unbind before the product's bind
# --------------------------------------------------------------------------- #
class ScriptedBinding:
    """``is_bound`` as the trap sees it poll by poll; the last reading repeats.

    Records at which polls the fixture's unbind write already existed, so a
    test can tell a re-unbind after the product's bind from one fired on the
    baseline binding.
    """

    def __init__(self, readings: list[bool], unbind_file: Path) -> None:
        self._readings = iter(readings)
        self.last = readings[0]
        self.unbind_file = unbind_file
        self.polls = 0
        self.unbind_seen_at: list[int] = []

    def is_bound(self, bdf: str) -> bool:
        self.polls += 1
        if self.unbind_file.exists():
            self.unbind_seen_at.append(self.polls)
        try:
            self.last = next(self._readings)
        except StopIteration:
            pass
        return self.last

    def bound_driver(self, bdf: str) -> str | None:
        return "efa" if self.last else None


def _scripted_trap(
    tmp_path: Path, monkeypatch, readings: list[bool]
) -> tuple[ScriptedBinding, list[float]]:
    _tree(tmp_path, monkeypatch)
    script = ScriptedBinding(readings, probe.driver_root() / "unbind")
    monkeypatch.setattr(probe, "is_bound", script.is_bound)
    monkeypatch.setattr(probe, "bound_driver", script.bound_driver)
    clock = {"now": 0.0}

    def monotonic_clock() -> float:
        clock["now"] += 1.0
        return clock["now"]

    pauses: list[float] = []
    monkeypatch.setattr(probe, "monotonic_clock", monotonic_clock)
    monkeypatch.setattr(probe, "pause", pauses.append)
    return script, pauses


def test_trap_run_waits_for_the_fixture_s_unbind_before_the_product_s_bind(
    tmp_path: Path, monkeypatch
) -> None:
    """The trap is armed while the baseline binding still stands (state first,
    then the unbind): bound, bound, unbound (the fixture), unbound, bound (the
    product). It fires once, after the product's bind -- here on the grace
    fallback, the scripted ``systemctl`` showing a collector that never
    restarts."""

    script, pauses = _scripted_trap(
        tmp_path, monkeypatch, [True, True, False, False, True]
    )
    probe.trap_run(
        _ns(run_id=RUN, pci_bdf=BDF_A, grace_seconds=5, trap_window_seconds=60)
    )
    assert (probe.driver_root() / "unbind").read_text() == BDF_A + "\n"
    # Polls 1-5 saw no unbind write of ours; only the receipt's poll (6) does.
    assert script.unbind_seen_at == [6]
    assert pauses[:3] == [1, 1, probe.BIND_POLL_SECONDS], (
        "two baseline polls, then the short poll for the product's bind"
    )
    assert set(pauses[3:]) == {probe.COLLECTOR_POLL_SECONDS}, (
        "then only the collector watch's short polls, no fixed sleep"
    )
    fired = json.loads((probe.state_dir(RUN) / "trap-fired").read_text())
    assert fired["fired"] is True
    assert fired["trigger"] == probe.TRIGGER_GRACE_FALLBACK


def test_trap_run_never_unbinds_a_function_that_stayed_bound(
    tmp_path: Path, monkeypatch
) -> None:
    """A baseline binding that never went away is not the product's bind: the
    trap times out without touching the function and says which wait ended."""

    _scripted_trap(tmp_path, monkeypatch, [True])
    probe.trap_run(
        _ns(run_id=RUN, pci_bdf=BDF_A, grace_seconds=5, trap_window_seconds=30)
    )
    assert not (probe.driver_root() / "unbind").exists(), (
        "the baseline binding is never the product's bind"
    )
    fired = json.loads((probe.state_dir(RUN) / "trap-fired").read_text())
    assert fired["fired"] is False
    assert fired["reason"] == "unbind not observed"


# --------------------------------------------------------------------------- #
# The re-unbind fires on the product's post-repair snapshot
# --------------------------------------------------------------------------- #
#: ``systemctl show`` of the host collector before and after the restart the
#: product's TRIGGER_HEALTH_SNAPSHOT runs: a new invocation, main process,
#: main-process start and activation.
COLLECTOR_RUNNING = {
    "LoadState": "loaded",
    "ActiveState": "active",
    "SubState": "running",
    "InvocationID": "a" * 32,
    "MainPID": "4242",
    "ExecMainStartTimestampMonotonic": "1000000",
    "ActiveEnterTimestampMonotonic": "1900000",
}
COLLECTOR_RESTARTED = {
    **COLLECTOR_RUNNING,
    "InvocationID": "b" * 32,
    "MainPID": "4343",
    "ExecMainStartTimestampMonotonic": "9000000",
    "ActiveEnterTimestampMonotonic": "9800000",
}
#: The scripted run below: the fixture's unbind lands at t=1, the product's
#: bind at t=3 (the trap's 1 s unbind poll and 0.25 s bind poll see each one
#: exactly then).
BIND_AT = 3.0


class SnapshotWorld:
    """One trap run on a clock only ``pause`` advances.

    The function is bound until t=1 (the fixture's own unbind), bound again
    from ``BIND_AT`` (the product's REMEDIATE_EFA_DRIVER) and unbound for good
    once the trap's unbind write exists; ``reunbound_at`` is when the trap
    first read its own unbind back (the record's ``bound``, right after the
    write). ``systemctl show`` of the host collector answers
    ``COLLECTOR_RUNNING`` until ``restart_at`` and ``COLLECTOR_RESTARTED``
    from then on, or times out on every read (``collector_fails``). Every
    command is kept with its time.
    """

    def __init__(
        self,
        unbind_file: Path,
        *,
        restart_at: float | None = None,
        collector_fails: bool = False,
    ) -> None:
        self.now = 0.0
        self.pauses: list[float] = []
        self.commands: list[tuple[float, list[str]]] = []
        self.unbind_file = unbind_file
        self.restart_at = restart_at
        self.collector_fails = collector_fails
        self.reunbound_at: float | None = None

    def monotonic_clock(self) -> float:
        return self.now

    def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)
        self.now = round(self.now + seconds, 6)

    def bound_now(self) -> bool:
        if self.unbind_file.exists():
            if self.reunbound_at is None:
                self.reunbound_at = self.now
            return False
        return self.now < 1.0 or self.now >= BIND_AT

    def is_bound(self, bdf: str) -> bool:
        return self.bound_now()

    def bound_driver(self, bdf: str) -> str | None:
        return "efa" if self.bound_now() else None

    def run(
        self, command: list[str], *, check: bool = True, timeout: int = 120
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append((self.now, list(command)))
        stdout = ""
        if command[:3] == ["systemctl", "show", probe.SNAPSHOT_COLLECTOR_UNIT]:
            if self.collector_fails:
                raise subprocess.TimeoutExpired(command, timeout)
            restarted = self.restart_at is not None and self.now >= self.restart_at
            state = COLLECTOR_RESTARTED if restarted else COLLECTOR_RUNNING
            stdout = "".join(f"{key}={value}\n" for key, value in state.items())
        return subprocess.CompletedProcess(command, 0, stdout, "")

    def collector_commands(self) -> list[tuple[float, list[str]]]:
        return [
            (at, argv)
            for at, argv in self.commands
            if probe.SNAPSHOT_COLLECTOR_UNIT in argv
        ]


def _snapshot_trap(
    tmp_path: Path,
    monkeypatch,
    *,
    grace: int = 5,
    floor: int = 1,
    restart_at: float | None = None,
    collector_fails: bool = False,
) -> tuple[SnapshotWorld, dict[str, Any]]:
    _tree(tmp_path, monkeypatch)
    world = SnapshotWorld(
        probe.driver_root() / "unbind",
        restart_at=restart_at,
        collector_fails=collector_fails,
    )
    for name in ("monotonic_clock", "pause", "is_bound", "bound_driver", "run"):
        monkeypatch.setattr(probe, name, getattr(world, name))
    probe.trap_run(
        _ns(
            run_id=RUN,
            pci_bdf=BDF_A,
            grace_seconds=grace,
            floor_seconds=floor,
            trap_window_seconds=60,
        )
    )
    record = json.loads((probe.state_dir(RUN) / "trap-fired").read_text())
    assert (probe.driver_root() / "unbind").read_text() == BDF_A + "\n"
    return world, record


def test_the_trap_re_unbinds_the_moment_the_snapshot_collector_restarts(
    tmp_path: Path, monkeypatch
) -> None:
    """The live order of 2026-11-06: the product's bind, its SUCCEEDED receipt,
    then TRIGGER_HEALTH_SNAPSHOT restarting the host collector 1.4 s after the
    bind -- and that collector's first tick about two seconds later, sooner than
    the old fixed 5 s grace. The trap re-unbinds at the restart itself, so that
    first tick reads the function unbound, and says so in its record."""

    world, record = _snapshot_trap(tmp_path, monkeypatch, restart_at=BIND_AT + 1.4)
    assert world.reunbound_at == BIND_AT + 1.4, world.pauses
    assert record["fired"] is True
    assert record["trigger"] == probe.TRIGGER_COLLECTOR_RESTART
    assert record["seconds_after_bind"] == 1.4
    assert record["collector_unit"] == "gpu-fault-host-collector.service"
    assert record["collector_before"]["InvocationID"] == "a" * 32
    assert record["collector_restart"]["property"] == "InvocationID"
    assert record["collector_restart"]["state"]["MainPID"] == "4343"
    assert (record["floor_seconds"], record["grace_seconds"]) == (1, 5)
    # The product's unit is only ever read, and only between the bind and the
    # re-unbind: no restart, stop or any other verb, no read before the bind.
    reads = world.collector_commands()
    assert {tuple(argv[:2]) for _, argv in reads} == {("systemctl", "show")}
    assert [at for at, _ in reads][0] == BIND_AT
    assert max(at for at, _ in reads) == world.reunbound_at
    assert [
        argv for _, argv in world.commands if argv[:2] != ["systemctl", "show"]
    ] == []


def test_the_trap_falls_back_to_the_grace_when_no_restart_shows(
    tmp_path: Path, monkeypatch
) -> None:
    """No restart of the snapshot collector within the grace: the old fixed
    delay is the bounded fallback, recorded as such."""

    world, record = _snapshot_trap(tmp_path, monkeypatch, restart_at=None)
    assert world.reunbound_at == BIND_AT + 5, world.pauses
    assert record["trigger"] == probe.TRIGGER_GRACE_FALLBACK
    assert record["seconds_after_bind"] == 5.0
    assert record["collector_restart"] is None
    assert record["collector_before"] == COLLECTOR_RUNNING
    assert record["collector_reads"] == 1 + round(5 / probe.COLLECTOR_POLL_SECONDS)


def test_the_trap_never_re_unbinds_before_its_floor(
    tmp_path: Path, monkeypatch
) -> None:
    """A restart nobody planned (the collector's watchdog) right after the
    bind is seen at once, but the re-unbind waits for the floor: the Node
    Agent's own bind read-back must have recorded SUCCEEDED first."""

    world, record = _snapshot_trap(
        tmp_path, monkeypatch, floor=2, restart_at=BIND_AT + 0.2
    )
    assert record["trigger"] == probe.TRIGGER_COLLECTOR_RESTART
    assert record["collector_restart"]["seconds_after_bind"] == 0.2
    assert world.reunbound_at == BIND_AT + 2, world.pauses
    assert record["seconds_after_bind"] == 2.0
    assert world.pauses[-1] == pytest.approx(1.8), "topped up to the floor"


def test_a_collector_restart_before_the_product_s_bind_is_not_the_trigger(
    tmp_path: Path, monkeypatch
) -> None:
    """The watch starts at the product's bind: a restart while the function is
    still unbound is not the post-repair snapshot, and the collector is not
    read at all before the bind."""

    world, record = _snapshot_trap(tmp_path, monkeypatch, restart_at=BIND_AT - 1)
    assert min(at for at, _ in world.collector_commands()) == BIND_AT
    assert record["collector_before"] == COLLECTOR_RESTARTED
    assert record["trigger"] == probe.TRIGGER_GRACE_FALLBACK
    assert world.reunbound_at == BIND_AT + 5, world.pauses


def test_an_unreadable_collector_still_ends_on_the_grace_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    """``systemctl show`` hanging on every read is a reading without identity,
    not a crash of the trap: the one re-unbind still lands, on the grace."""

    world, record = _snapshot_trap(tmp_path, monkeypatch, collector_fails=True)
    assert record["fired"] is True
    assert record["trigger"] == probe.TRIGGER_GRACE_FALLBACK
    assert record["collector_before"] == {"error": "TimeoutExpired"}
    assert world.reunbound_at == BIND_AT + 5, world.pauses


@pytest.mark.parametrize(
    ("after", "expected"),
    [
        ({**COLLECTOR_RUNNING, "InvocationID": "c" * 32}, "InvocationID"),
        (
            {**COLLECTOR_RUNNING, "MainPID": "0", "ActiveState": "deactivating"},
            "MainPID",
        ),
        (
            {**COLLECTOR_RUNNING, "ExecMainStartTimestampMonotonic": "5"},
            "ExecMainStartTimestampMonotonic",
        ),
        (
            {**COLLECTOR_RUNNING, "ActiveEnterTimestampMonotonic": "7"},
            "ActiveEnterTimestampMonotonic",
        ),
        ({**COLLECTOR_RUNNING, "ActiveState": "deactivating"}, "ActiveState"),
        ({**COLLECTOR_RUNNING, "SubState": "reload"}, None),
        (dict(COLLECTOR_RUNNING), None),
        ({"error": "TimeoutExpired"}, None),
    ],
    ids=[
        "new-invocation",
        "main-process-gone",
        "new-main-process-start",
        "new-activation",
        "stopping",
        "substate-only",
        "unchanged",
        "unreadable",
    ],
)
def test_collector_restart_reads_the_unit_s_identity(
    after: dict[str, str], expected: str | None
) -> None:
    assert probe.collector_restart(COLLECTOR_RUNNING, after) == expected
    assert probe.collector_restart({"error": "x"}, COLLECTOR_RESTARTED) is None


@pytest.mark.parametrize(
    ("returncode", "stdout", "error"),
    [
        (0, "LoadState=loaded\nMainPID=7\nnoise\nOther=1\n", None),
        (1, "", "systemctl show exited 1"),
        (0, "LoadState=not-found\nMainPID=0\n", "LoadState=not-found"),
    ],
    ids=["loaded", "failed", "not-found"],
)
def test_collector_state_is_one_read_only_show(
    monkeypatch, returncode: int, stdout: str, error: str | None
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, *, check=True, timeout=120):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, returncode, stdout, "")

    monkeypatch.setattr(probe, "run", fake_run)
    state = probe.collector_state()
    assert state.get("error") == error
    assert "Other" not in state, "only the identity and state properties are kept"
    (command,) = calls
    assert command[:3] == ["systemctl", "show", probe.SNAPSHOT_COLLECTOR_UNIT]
    assert command[3] == "--property=" + ",".join(probe.COLLECTOR_STATE_PROPERTIES)


def test_trap_unbind_arms_the_floor_and_refuses_one_above_the_grace(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    fake = _tree(tmp_path, monkeypatch)
    code = probe.main(
        ["trap-unbind", "--run-id", RUN, "--pci-bdf", BDF_A, "--grace-seconds", "5"]
    )
    receipt = _last_json(capsys)
    assert code == 0, receipt
    assert (receipt["floor_seconds"], receipt["grace_seconds"]) == (1, 5)
    assert receipt["collector_unit"] == probe.SNAPSHOT_COLLECTOR_UNIT
    (trap,) = [c for c in fake.calls if c[0] == "systemd-run" and "trap-run" in c]
    assert trap[trap.index("--floor-seconds") + 1] == "1"
    assert trap[trap.index("--grace-seconds") + 1] == "5"

    refused_tree = tmp_path / "refused"
    refused_tree.mkdir()
    fake = _tree(refused_tree, monkeypatch)
    code = probe.main(
        [
            "trap-unbind",
            "--run-id",
            RUN,
            "--pci-bdf",
            BDF_A,
            "--grace-seconds",
            "2",
            "--floor-seconds",
            "3",
        ]
    )
    error = _last_json(capsys)
    assert code == 1, error
    assert error["error_message"] == "floor is outside 1..2"
    assert [call for call in fake.calls if call[0] == "systemd-run"] == []
    assert not (probe.driver_root() / "unbind").exists(), "refused before the unbind"


def test_a_re_arm_of_the_same_run_runs_the_shipped_trap_and_reads_its_own_record(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """An earlier arm of this run id on this boot left its probe copy and its
    trap record in ``/run`` (a run the product never rebooted): the deadman and
    the trap unit run the shipped probe, not that copy, and ``summary`` reads
    nothing until the new trap writes its own record."""

    _tree(tmp_path, monkeypatch)
    directory = probe.state_dir(RUN)
    directory.mkdir(parents=True)
    (directory / "probe.py").write_text("# an older probe: a fixed-grace trap\n")
    (directory / "trap-fired").write_text(json.dumps({"fired": True, "at": "old"}))
    assert probe.main(["trap-unbind", "--run-id", RUN, "--pci-bdf", BDF_A]) == 0
    capsys.readouterr()
    assert filecmp.cmp(directory / "probe.py", probe.__file__, shallow=False), (
        "the deadman and the trap unit run the shipped probe"
    )
    assert not (directory / "trap-fired").exists(), "the old record is not this trap's"
    assert probe.summary(RUN, BDF_A)["trap_fired_at"] is None


# --------------------------------------------------------------------------- #
# Deadman fails closed; release restores or fails loudly
# --------------------------------------------------------------------------- #
def test_arm_deadman_requires_an_active_timer(tmp_path: Path, monkeypatch) -> None:
    _tree(tmp_path, monkeypatch)
    # A run that reports the timer inactive must fail closed.
    monkeypatch.setattr(
        probe,
        "run",
        lambda *a, **k: type("C", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )
    with pytest.raises(probe.ProbeError, match="not armed"):
        probe.arm_deadman("gl013-x-a1-dead", BDF_A, 600)


def test_release_is_clean_when_bound(tmp_path: Path, monkeypatch, capsys) -> None:
    """The kernel's ``(null)`` read-back needs no clearing write and an already
    bound function no rebind; live 2026-10-27 this release called the same
    read-back "not restored" (``release:733``) and failed the cleanup."""

    _tree(tmp_path, monkeypatch)
    probe.release(_ns(run_id=RUN, pci_bdf=BDF_A, automatic=False))
    receipt = _last_json(capsys)
    assert receipt["clean"] is True
    assert receipt["bound"] is True
    assert receipt["driver_override"] is None
    assert receipt["override_before"] is None
    assert receipt["driver_override_sysfs"] == "(null)"
    assert (probe.device_dir(BDF_A) / "driver_override").read_text() == KERNEL_UNSET
    assert not (probe.driver_root() / "bind").exists(), "already bound: no rebind"


def test_release_clears_a_real_override_before_rebinding(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _tree(
        tmp_path,
        monkeypatch,
        functions=((BDF_A, True, probe.HOLD_OVERRIDE), (BDF_B, True, None)),
    )
    probe.release(_ns(run_id=RUN, pci_bdf=BDF_A, automatic=False))
    receipt = _last_json(capsys)
    assert receipt["override_before"] == probe.HOLD_OVERRIDE
    # The ABI's clearing write: "may be cleared with an empty string
    # (echo > driver_override)" -- a bare newline, which the store maps to NULL.
    assert (probe.device_dir(BDF_A) / "driver_override").read_text() == "\n"
    assert receipt["driver_override"] is None
    assert receipt["clean"] is True


def test_release_fails_loudly_when_unbound(tmp_path: Path, monkeypatch, capsys) -> None:
    # The function is left unbound; _bind writes to the (fake) bind file, which
    # does not re-attach it, so the end-state check must refuse.
    _tree(tmp_path, monkeypatch, functions=((BDF_A, False, None), (BDF_B, True, None)))
    with pytest.raises(probe.ProbeError, match="did not restore") as refused:
        probe.release(_ns(run_id=RUN, pci_bdf=BDF_A, automatic=False))
    receipt = _last_json(capsys)
    assert receipt["clean"] is False
    assert refused.value.reason == "not_restored"
    assert refused.value.detail == receipt, "the receipt travels as the detail"


# --------------------------------------------------------------------------- #
# The host fixture keeps the refusal in the case directory
# --------------------------------------------------------------------------- #
def test_the_host_fixture_records_the_refusal_and_keeps_the_earlier_one(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The probe's failure line carries ``error_message``/``error_detail`` like
    the source-replay probe's, so ``HostProbeFixture.execute`` writes the
    case's ``probe-error.json`` and says so in its exception text; a second
    failure of the same case (the cleanup's own refusal) keeps the first
    record under ``previous`` instead of erasing it."""

    _tree(
        tmp_path,
        monkeypatch,
        functions=((BDF_A, True, "vfio-pci"), (BDF_B, True, None)),
    )
    assert probe.main(["hold-unbind", "--run-id", RUN, "--pci-bdf", BDF_A]) == 1
    refusal = _last_json(capsys)
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    api = ProbeApi()
    monkeypatch.setattr(host, "run_fixture_command", api.run)
    fixture = host_probe(
        case_dir, case_id="GF-GENERAL-LIVE-013.bind_failure", run_id=RUN
    )
    fixture.create()
    api.probe_returncode = 1
    api.probe_stdout = json.dumps(refusal) + "\n"
    with pytest.raises(host.HostProbeError) as caught:
        fixture.execute("hold-unbind", "--pci-bdf", BDF_A, "--run-id", RUN)
    text = str(caught.value)
    assert re.fullmatch(
        r"host probe failed \(exit 1; error ProbeError [0-9a-f]{16} at "
        r"_preconditions:[0-9]+; reason driver_override_set; "
        r"detail in probe-error\.json\); output withheld",
        text,
    ), text
    assert caught.value.reason == "driver_override_set"
    record = json.loads((case_dir / host.PROBE_ERROR_RECORD).read_text())
    assert record["message"] == f"{BDF_A} already has a driver_override: 'vfio-pci'"
    assert record["command"] == "hold-unbind"
    assert record["error_reason"] == "driver_override_set"
    assert record["case_id"] == "GF-GENERAL-LIVE-013.bind_failure"
    assert "previous" not in record

    # The cleanup's release on a function that stayed unbound refuses too.
    (probe.device_dir(BDF_A) / "driver").unlink()
    (probe.driver_root() / BDF_A).unlink()
    assert probe.main(["release", "--run-id", RUN, "--pci-bdf", BDF_A]) == 1
    cleanup = _last_json(capsys)
    assert cleanup["error_reason"] == "not_restored"
    assert cleanup["error_detail"]["clean"] is False
    api.probe_stdout = json.dumps(cleanup) + "\n"
    with pytest.raises(
        host.HostProbeError, match=r"reason not_restored; detail in probe-error\.json"
    ):
        fixture.execute("release", "--pci-bdf", BDF_A, "--run-id", RUN)
    record = json.loads((case_dir / host.PROBE_ERROR_RECORD).read_text())
    assert record["message"].startswith(
        f"EFA driver fixture did not restore {BDF_A}: bound=False override=None"
    ), record["message"]
    assert record["detail"]["bound"] is False
    assert record["detail"]["override_before"] == "vfio-pci"
    assert [item["message"] for item in record["previous"]] == [
        f"{BDF_A} already has a driver_override: 'vfio-pci'"
    ]
    assert record["previous"][0]["command"] == "hold-unbind"
