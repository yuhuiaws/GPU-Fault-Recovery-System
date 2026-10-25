#!/usr/bin/env python3
"""Host probe of the kernel-log source replay helper (GF-GENERAL-LIVE 016/032).

Runs on the target GPU node through ``host_probe_fixture.py`` (a privileged
probe Pod that ``chroot``s into the host and runs this file with the node's own
Python). Standard library only, Python 3.9+, one JSON object on the last stdout
line. It replays *one* reviewed set of kernel-transport records for one node and
boot through the product's own readers, two ways, and nothing else:

* **journal mode** -- the node log collector (``gpu-fault-log-collector.service``,
  installed but disabled by default) reads ``journalctl --output=json``. A
  reviewed pass-through wrapper is placed first on that one unit's ``PATH`` with
  a ``/run`` systemd drop-in: it forwards every invocation to the real
  ``journalctl`` and, only after ``arm``, appends the reviewed records as extra
  JSON lines so they travel through the real ``_parse_journal_line`` ->
  ``_TRANSPORT`` -> ``dmesg``/``journal`` origin -> trusted-source gate
  (``log_rules``/``log_recovery_context``) exactly as a real kernel line would.
  A ``_TRANSPORT=kernel`` record is the trusted (dmesg) origin; a non-kernel
  transport stays the untrusted ``journal`` origin, so a reviewed user-space
  line can never be graded as a trusted kernel fault.

* **kmsg mode** -- the kernel collector (``gpu-fault-kernel-collector.service``)
  reads ``GPU_FAULT_KMSG_PATH`` (``/dev/kmsg`` by default). A drop-in repoints
  that one unit at a reviewed fixture file under ``/run``; ``arm`` appends the
  reviewed records in ``/dev/kmsg`` record syntax, so they travel through the
  real ``KernelLogCollector`` parser, dedup and NVIDIA_KERNEL delivery into the
  real XID/SXID catalog. The real ``/dev/kmsg`` is never written.

Everything is bounded and reversible two ways -- the runner's cleanup and a
``systemd-run --on-active`` deadman that restores the baseline from a copy of
this file even if the runner, the probe Pod and the control plane all vanish.
Nothing here forges a finding, writes a real ``/proc``/sysfs/``/dev/kmsg``
source, touches the Node Agent, or starts a second producer of any channel.
``proof_scope`` is always ``live-replay``: the reviewed source proves the
processing chain, never a physical fault.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_ROOT = Path("/run/gpu-fault-kernel-replay")
DROPIN_ROOT = Path("/run/systemd/system")
LOG_UNIT = "gpu-fault-log-collector.service"
KERNEL_UNIT = "gpu-fault-kernel-collector.service"
KMSG_ENV = "GPU_FAULT_KMSG_PATH"
#: The two source readers this probe may replay, and the one unit each lives in.
MODES = {"journal": LOG_UNIT, "kmsg": KERNEL_UNIT}
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MIN_RESTORE_SECONDS = 60
MAX_RESTORE_SECONDS = 5400
#: Reviewed transports a record may carry. ``kernel`` is the trusted (dmesg)
#: origin; the rest stay the untrusted ``journal`` origin on purpose.
ALLOWED_TRANSPORTS = ("kernel", "syslog", "stdout", "audit")
MAX_RECORDS = 32
MAX_EMISSIONS = 16
#: A record message is carried in every batch until the node recovers, so it is
#: bounded here the way the collector bounds one entry.
MAX_MESSAGE_BYTES = 16 * 1024


class ProbeError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def run(
    command: list[str], *, check: bool = True, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode:
        raise ProbeError(
            f"command failed ({completed.returncode}): {' '.join(command)}: "
            f"{completed.stderr.strip()[:300]}"
        )
    return completed


def safe_id(value: str) -> str:
    if SAFE_ID.fullmatch(value) is None:
        raise ProbeError("unsafe run ID")
    return value


def digest(run_id: str) -> str:
    return hashlib.sha256(safe_id(run_id).encode()).hexdigest()[:16]


def state_dir(run_id: str) -> Path:
    return STATE_ROOT / digest(run_id)


def dropin_path(unit: str, run_id: str) -> Path:
    return DROPIN_ROOT / f"{unit}.d" / f"gpu-fault-kernel-replay-{digest(run_id)}.conf"


def deadman_unit(run_id: str) -> str:
    return f"gpu-fault-kernel-replay-{digest(run_id)}"


def checked_restore_seconds(value: int) -> int:
    if not MIN_RESTORE_SECONDS <= int(value) <= MAX_RESTORE_SECONDS:
        raise ProbeError(
            f"restore seconds is outside {MIN_RESTORE_SECONDS}..{MAX_RESTORE_SECONDS}"
        )
    return int(value)


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def unit_state(unit: str) -> dict[str, str]:
    completed = run(
        [
            "systemctl",
            "show",
            unit,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=UnitFileState",
            "--property=MainPID",
            "--property=NRestarts",
            "--property=InvocationID",
        ],
        check=False,
    )
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    values["returncode"] = str(completed.returncode)
    return values


def load_state(run_id: str) -> dict[str, Any]:
    path = state_dir(run_id) / "window.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        raise ProbeError("kernel replay window identity mismatch")
    return value


def save_state(run_id: str, state: dict[str, Any]) -> None:
    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "window.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def probe_copy(run_id: str) -> Path:
    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = directory / "probe.py"
    if not target.is_file():
        shutil.copyfile(Path(__file__).resolve(), target)
        target.chmod(0o700)
    return target


def records_path(run_id: str) -> Path:
    return state_dir(run_id) / "records.json"


def kmsg_fixture_path(run_id: str) -> Path:
    return state_dir(run_id) / "kmsg.fixture"


def armed_path(run_id: str) -> Path:
    return state_dir(run_id) / "armed"


def emission_log_path(run_id: str) -> Path:
    return state_dir(run_id) / "emissions.jsonl"


def shadow_dir(run_id: str) -> Path:
    return state_dir(run_id) / "bin"


# --------------------------------------------------------------------------- #
# Reviewed records
# --------------------------------------------------------------------------- #
def validate_records(records: Any) -> list[dict[str, Any]]:
    """Check the reviewed record set a caller installed; never mutate it."""

    if not isinstance(records, list) or not records:
        raise ProbeError("reviewed records must be a non-empty list")
    if len(records) > MAX_RECORDS:
        raise ProbeError(f"at most {MAX_RECORDS} reviewed records may be installed")
    result = []
    for item in records:
        if not isinstance(item, dict):
            raise ProbeError("each reviewed record must be an object")
        message = item.get("message")
        if not isinstance(message, str) or not message:
            raise ProbeError("a reviewed record needs a non-empty message")
        if len(message.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise ProbeError("reviewed record message is too large")
        transport = str(item.get("transport") or "kernel")
        if transport not in ALLOWED_TRANSPORTS:
            raise ProbeError(f"transport {transport!r} is not reviewed")
        emissions = int(item.get("emissions", 4))
        if not 1 <= emissions <= MAX_EMISSIONS:
            raise ProbeError(f"emissions must be 1..{MAX_EMISSIONS}")
        priority = int(item.get("priority", 3))
        if not 0 <= priority <= 7:
            raise ProbeError("priority must be 0..7")
        result.append(
            {
                "message": message,
                "transport": transport,
                "priority": priority,
                "emissions": emissions,
                "boot_id": str(item.get("boot_id") or ""),
                "unit": str(item.get("unit") or ""),
                "label": str(item.get("label") or ""),
            }
        )
    return result


def install_records(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    if SHA256.fullmatch(arguments.sha256) is None:
        raise ProbeError("a sha256 is required for the reviewed record set")
    try:
        content = base64.b64decode(arguments.b64, validate=True)
    except (ValueError, TypeError) as error:
        raise ProbeError(f"records are not valid base64: {error}") from error
    if hashlib.sha256(content).hexdigest() != arguments.sha256:
        raise ProbeError("installed record set sha256 does not match")
    records = validate_records(json.loads(content.decode("utf-8")))
    path = records_path(run_id)
    if path.exists():
        raise ProbeError("a record set is already installed for this run")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(records, sort_keys=True), encoding="utf-8")
    path.chmod(0o600)
    emit(
        {
            "run_id": run_id,
            "records": len(records),
            "sha256": arguments.sha256,
            "transports": sorted({item["transport"] for item in records}),
        }
    )


# --------------------------------------------------------------------------- #
# The journalctl pass-through wrapper (journal mode)
# --------------------------------------------------------------------------- #
WRAPPER_SCRIPT = r'''#!/usr/bin/env python3
"""Reviewed journalctl pass-through of the GF-GENERAL-LIVE kernel replay.

Forwards every call to the real journalctl and, only when armed, appends the
reviewed records as extra --output=json lines inside the asked-for window. See
general_live_kernel_source_probe.py. It never removes or edits a real line.
"""
import fcntl
import json
import os
import subprocess
import sys
import time

REAL = %(real)r
RECORDS = %(records)r
ARMED = %(armed)r
STATE = %(state)r
EMISSIONS = %(emissions)r
BOOT_ID = %(boot_id)r
CURSOR_PREFIX = "gfrs-replay-"


def _forward_args(argv):
    # A synthetic cursor this wrapper emitted must never reach the real
    # journalctl (it would fail to seek); drop only our own after-cursor, in
    # either --after-cursor=<c> or --after-cursor <c> form, and forward a real
    # journal cursor untouched.
    argv = list(argv)
    kept = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item.startswith("--after-cursor=" + CURSOR_PREFIX):
            index += 1
            continue
        if (
            item == "--after-cursor"
            and index + 1 < len(argv)
            and argv[index + 1].startswith(CURSOR_PREFIX)
        ):
            index += 2
            continue
        kept.append(item)
        index += 1
    return kept


def _emit_reviewed(out):
    if not os.path.exists(ARMED) or "--output=json" not in sys.argv[1:]:
        return
    try:
        records = json.load(open(RECORDS, encoding="utf-8"))
    except (OSError, ValueError):
        return
    try:
        with open(STATE, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            text = handle.read().strip()
            counts = json.loads(text) if text else {}
            now_us = int(time.time() * 1_000_000)
            for index, record in enumerate(records):
                key = str(index)
                done = int(counts.get(key, 0))
                if done >= int(record.get("emissions", EMISSIONS)):
                    continue
                counts[key] = done + 1
                transport = record.get("transport") or "kernel"
                entry = {
                    "__CURSOR": CURSOR_PREFIX + str(index) + "-" + BOOT_ID,
                    "__REALTIME_TIMESTAMP": str(now_us),
                    "__MONOTONIC_TIMESTAMP": str(now_us),
                    "MESSAGE": record["message"],
                    "PRIORITY": str(record.get("priority", 3)),
                    "_TRANSPORT": transport,
                    "_BOOT_ID": record.get("boot_id") or BOOT_ID,
                }
                if record.get("unit"):
                    entry["_SYSTEMD_UNIT"] = record["unit"]
                out.write(json.dumps(entry) + "\n")
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(counts))
    except OSError:
        return


def main():
    completed = subprocess.run([REAL, *_forward_args(sys.argv[1:])], capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    _emit_reviewed(sys.stdout)
    sys.stderr.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_wrapper(run_id: str) -> str:
    real = shutil.which("journalctl", path="/usr/bin:/usr/local/bin:/bin:/sbin")
    if real is None:
        raise ProbeError("journalctl is not installed on the host")
    directory = shadow_dir(run_id)
    directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    script = directory / "journalctl"
    script.write_text(
        WRAPPER_SCRIPT
        % {
            "real": os.path.realpath(real),
            "records": str(records_path(run_id)),
            "armed": str(armed_path(run_id)),
            "state": str(emission_log_path(run_id)),
            "emissions": 4,
            "boot_id": boot_id(),
        },
        encoding="utf-8",
    )
    script.chmod(0o755)
    return str(directory)


# --------------------------------------------------------------------------- #
# Deadman
# --------------------------------------------------------------------------- #
def arm_deadman(run_id: str, mode: str, seconds: int) -> str:
    name = deadman_unit(run_id)
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service"], check=False)
    run(
        [
            "systemd-run",
            f"--unit={name}",
            f"--on-active={int(seconds)}s",
            "--timer-property=AccuracySec=1s",
            "--property=Restart=on-failure",
            "--property=RestartSec=30s",
            "--property=StartLimitIntervalSec=0",
            sys.executable,
            str(probe_copy(run_id)),
            "close-window",
            "--run-id",
            run_id,
            "--mode",
            mode,
        ]
    )
    if unit_state(f"{name}.timer").get("ActiveState") != "active":
        raise ProbeError("kernel replay deadman timer is not active")
    return name


def disarm_deadman(run_id: str) -> None:
    name = deadman_unit(run_id)
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    if unit_state(f"{name}.service").get("MainPID") != str(os.getpid()):
        run(["systemctl", "stop", f"{name}.service"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service", f"{name}.timer"], check=False)


# --------------------------------------------------------------------------- #
# preflight / open-window / arm / status / close-window / snapshot
# --------------------------------------------------------------------------- #
def preflight(arguments: argparse.Namespace) -> None:
    emit(
        {
            "captured_at": utc_now(),
            "boot_id": boot_id(),
            "journalctl": shutil.which("journalctl") is not None,
            "units": {unit: unit_state(unit) for unit in (LOG_UNIT, KERNEL_UNIT)},
        }
    )


def _log_collector_runtime_enabled(state: dict[str, str]) -> bool:
    return state.get("UnitFileState") == "enabled-runtime"


def open_window(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    mode = arguments.mode
    if mode not in MODES:
        raise ProbeError("mode must be journal or kmsg")
    unit = MODES[mode]
    restore_seconds = checked_restore_seconds(arguments.restore_seconds)
    if not records_path(run_id).is_file():
        raise ProbeError("install the reviewed records before opening the window")
    if load_state(run_id):
        raise ProbeError("a kernel replay window for this run is already open")
    dropin = dropin_path(unit, run_id)
    if dropin.exists():
        raise ProbeError("an untracked kernel replay drop-in already exists")

    enabled_log_collector = False
    before = unit_state(unit)
    if mode == "journal":
        if not arguments.approved:
            raise ProbeError(
                "the node log collector is disabled in production; journal mode "
                "needs the runner's ENABLE_NODE_LOG_COLLECTOR approval"
            )
        if before.get("LoadState") != "loaded":
            raise ProbeError("the node log collector unit is not installed")
    elif before.get("ActiveState") != "active":
        raise ProbeError(f"{unit} is not active at baseline")

    probe_copy(run_id)
    armed_path(run_id).unlink(missing_ok=True)
    emission_log_path(run_id).unlink(missing_ok=True)
    state = {
        "run_id": run_id,
        "mode": mode,
        "unit": unit,
        "dropin": str(dropin),
        "before": before,
        "opened_at": utc_now(),
        "restore_seconds": restore_seconds,
        "enabled_log_collector": False,
        "boot_id": boot_id(),
    }
    save_state(run_id, state)
    deadman = arm_deadman(run_id, mode, restore_seconds)

    lines = ["[Service]"]
    if mode == "journal":
        shadow = write_wrapper(run_id)
        lines.append(
            f"Environment=PATH={shadow}:/usr/local/sbin:/usr/local/bin:"
            "/usr/sbin:/usr/bin:/sbin:/bin"
        )
        # The wrapper writes its bounded emission counts here; /run is read-only
        # for the unit under ProtectSystem=strict without this.
        lines.append(f"ReadWritePaths={state_dir(run_id)}")
    else:
        fixture = kmsg_fixture_path(run_id)
        fixture.write_text("", encoding="utf-8")
        fixture.chmod(0o600)
        lines.append(f"Environment={KMSG_ENV}={fixture}")
    dropin.parent.mkdir(parents=True, exist_ok=True)
    dropin.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if mode == "journal" and not _log_collector_runtime_enabled(before):
        run(["systemctl", "enable", "--runtime", LOG_UNIT], timeout=120)
        enabled_log_collector = True
        state["enabled_log_collector"] = True
        save_state(run_id, state)
    run(["systemctl", "daemon-reload"])
    if mode == "journal":
        run(["systemctl", "restart", unit], timeout=120)
    else:
        run(["systemctl", "restart", unit], timeout=120)
    after = unit_state(unit)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{unit} is not active after the window opened")
    emit(
        {
            **state,
            "after": after,
            "deadman_timer": unit_state(f"{deadman}.timer"),
            "enabled_log_collector": enabled_log_collector,
            "proof_scope": "live-replay",
        }
    )


#: Reviewed kmsg records carry sequence numbers far above anything a real boot
#: reaches, in a band of its own per run: the kernel collector's record id is
#: ``kmsg-<boot id>-<sequence>`` and the control plane dedups the XID event id
#: derived from it, so two runs on one boot that reused the same sequence made
#: the second run's event a duplicate of the first run's (never a fresh
#: decision). The band is stable within a run (copies of one record still dedup
#: to one event) and ``KMSG_SEQUENCE_STRIDE`` keeps a run's records apart.
KMSG_SEQUENCE_BASE = 900_000_000
KMSG_SEQUENCE_STRIDE = 64
KMSG_SEQUENCE_SLOTS = 1_000_000


def kmsg_sequence(run_id: str, index: int) -> int:
    """The /dev/kmsg sequence number of reviewed record ``index`` of ``run_id``."""

    if not 0 <= index < KMSG_SEQUENCE_STRIDE:
        raise ProbeError("reviewed record index is outside the sequence stride")
    slot = int(digest(run_id)[:6], 16) % KMSG_SEQUENCE_SLOTS
    return KMSG_SEQUENCE_BASE + slot * KMSG_SEQUENCE_STRIDE + index


def _kmsg_line(index: int, record: dict[str, Any], run_id: str) -> str:
    """One reviewed record in /dev/kmsg syntax: prio,seq,monotonic,flags;message."""

    monotonic_us = int(time.monotonic() * 1_000_000)
    priority = int(record.get("priority", 3))
    return (
        f"{priority},{kmsg_sequence(run_id, index)},{monotonic_us},-;"
        f"{record['message']}\n"
    )


def arm(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    if not state:
        raise ProbeError("no kernel replay window is open for this run")
    records = json.loads(records_path(run_id).read_text(encoding="utf-8"))
    if state["mode"] == "journal":
        armed_path(run_id).write_text(utc_now(), encoding="utf-8")
        emit(
            {
                "run_id": run_id,
                "mode": "journal",
                "armed": True,
                "records": len(records),
                "proof_scope": "live-replay",
            }
        )
        return
    # kmsg mode: append the reviewed records to the fixture the collector tails.
    # The kernel collector reopens at the live tail on EOF, so each record is
    # appended a bounded number of times; a sequence number stable within the
    # run makes the parser dedup them to one event however many copies it
    # reads, and distinct from every other run's (``kmsg_sequence``).
    fixture = kmsg_fixture_path(run_id)
    passes = max(int(record.get("emissions", 4)) for record in records)
    written = 0
    for _attempt in range(passes):
        with fixture.open("a", encoding="utf-8") as handle:
            for index, record in enumerate(records):
                handle.write(_kmsg_line(index, record, run_id))
                written += 1
            handle.flush()
            os.fsync(handle.fileno())
        time.sleep(1.0)
    emit(
        {
            "run_id": run_id,
            "mode": "kmsg",
            "armed": True,
            "records": len(records),
            "lines_written": written,
            "fixture": str(fixture),
            "proof_scope": "live-replay",
        }
    )


def status(arguments: argparse.Namespace) -> None:
    """Read back what the reviewed source produced; never consume a step."""

    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    if not state:
        raise ProbeError("no kernel replay window is open for this run")
    injected: dict[str, Any] = {
        "mode": state["mode"],
        "armed": armed_path(run_id).is_file(),
    }
    if state["mode"] == "journal":
        log = emission_log_path(run_id)
        injected["emission_counts"] = (
            json.loads(log.read_text(encoding="utf-8"))
            if log.is_file() and log.read_text(encoding="utf-8").strip()
            else {}
        )
    else:
        fixture = kmsg_fixture_path(run_id)
        injected["fixture_lines"] = (
            len(fixture.read_text(encoding="utf-8").splitlines())
            if fixture.is_file()
            else 0
        )
    emit({"run_id": run_id, "unit": unit_state(state["unit"]), "injected": injected})


def close_window(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    if not state:
        # Nothing tracked: make sure no stray drop-in is left for either unit.
        for unit in MODES.values():
            dropin_path(unit, run_id).unlink(missing_ok=True)
        if state_dir(run_id).is_dir():
            shutil.rmtree(state_dir(run_id))
        emit({"run_id": run_id, "already_closed": True, "clean": True})
        return
    unit = str(state["unit"])
    dropin = Path(state["dropin"])
    dropin.unlink(missing_ok=True)
    if state.get("enabled_log_collector"):
        run(["systemctl", "disable", "--runtime", LOG_UNIT], check=False, timeout=120)
    run(["systemctl", "daemon-reload"])
    if unit == LOG_UNIT and state.get("enabled_log_collector"):
        run(["systemctl", "stop", unit], check=False, timeout=120)
    else:
        run(["systemctl", "restart", unit], timeout=120)
    after = unit_state(unit)
    expected_active = not (unit == LOG_UNIT and state.get("enabled_log_collector"))
    if expected_active and after.get("ActiveState") != "active":
        raise ProbeError(f"{unit} is not active after the window closed")
    disarm_deadman(run_id)
    residue = _residue(run_id, state)
    if state_dir(run_id).is_dir():
        shutil.rmtree(state_dir(run_id))
    emit(
        {
            "run_id": run_id,
            "unit": unit,
            "mode": state["mode"],
            "after": after,
            "dropin_removed": not dropin.exists(),
            "residue": residue,
            "clean": not any(residue.values()),
        }
    )


def _residue(run_id: str, state: dict[str, Any]) -> dict[str, bool]:
    unit = str(state.get("unit") or "")
    return {
        "dropin": dropin_path(unit, run_id).exists() if unit else False,
        "log_collector_enabled_by_run": bool(state.get("enabled_log_collector"))
        and _log_collector_runtime_enabled(unit_state(LOG_UNIT)),
        "deadman": unit_state(f"{deadman_unit(run_id)}.timer").get("ActiveState")
        == "active",
    }


def snapshot(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    residue = _residue(run_id, state) if state else {"window": False}
    emit({"run_id": run_id, "window_open": bool(state), "residue": residue})


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    commands.add_parser("preflight").set_defaults(handler=preflight)

    item = commands.add_parser("install-records")
    item.add_argument("--run-id", required=True)
    item.add_argument("--sha256", required=True)
    item.add_argument("--b64", required=True)
    item.set_defaults(handler=install_records)

    item = commands.add_parser("open-window")
    item.add_argument("--run-id", required=True)
    item.add_argument("--mode", choices=sorted(MODES), required=True)
    item.add_argument("--restore-seconds", type=int, default=900)
    item.add_argument("--approved", action="store_true")
    item.set_defaults(handler=open_window)

    item = commands.add_parser("arm")
    item.add_argument("--run-id", required=True)
    item.set_defaults(handler=arm)

    item = commands.add_parser("status")
    item.add_argument("--run-id", required=True)
    item.set_defaults(handler=status)

    item = commands.add_parser("close-window")
    item.add_argument("--run-id", required=True)
    item.add_argument("--mode", choices=sorted(MODES), default="")
    item.set_defaults(handler=close_window)

    item = commands.add_parser("snapshot")
    item.add_argument("--run-id", required=True)
    item.set_defaults(handler=snapshot)
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        emit(
            {
                "error": type(exc).__name__,
                "error_kind": "probe",
                "command": getattr(arguments, "command", None),
            }
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
