#!/usr/bin/env python3
"""EFA driver-binding fixture of GF-GENERAL-LIVE-013 ``bind_failure`` and
``post_repair_failure``.

Runs on the target GPU node through ``host_probe_fixture.py`` (a privileged
probe Pod that ``chroot``s into the host and runs this file with the node's own
Python). Standard library only, Python 3.9+, one JSON object on the last stdout
line. It is the native sibling of ``collector_node_probe.py``'s ``unbind-efa``
(COLLECT-017 A / DESTR-021): the same real unbind of ONE allow-listed EFA data
function through ``/sys/bus/pci/drivers/efa/unbind`` and the same
``systemd-run --on-active`` rebind deadman, plus the two negative branches the
spec requires:

* ``hold-unbind`` (``bind_failure``): before the unbind the function's own
  ``driver_override`` is set to a driver name no module provides
  (``gfrs-live-hold``). The kernel then refuses every bind of *this function*
  to ``efa`` with ENODEV (``pci_match_device``: with ``driver_override`` set,
  only the named driver may bind), so the product's ``REMEDIATE_EFA_DRIVER``
  (``modprobe efa`` + ``printf BDF > .../efa/bind``) fails on exactly this
  BDF and records ``EFA driver bind verification failed: 15/16;
  bind_errors=<BDF>=...``. The other functions keep their binding, the
  module stays loaded and nothing is unloaded; the override is a per-device
  kernel attribute that one write (``printf '\\n' > driver_override``) or a
  reboot clears.
* ``trap-unbind`` (``post_repair_failure``): the unbind, plus a bounded
  transient unit that waits for this unbind and then for the product's bind
  of the same function, then watches -- by reading its ``systemctl show``
  state, nothing else -- the host collector the product restarts for its
  post-repair snapshot (``TRIGGER_HEALTH_SNAPSHOT``), unbinds the function
  once more the moment that restart shows (never sooner than a floor after the
  bind, so the Node Agent's own bind read-back has finished; at the old grace
  after the bind at the latest when no restart shows) and records when and on
  which trigger. The action's receipt therefore says SUCCEEDED (bound 16/16,
  ``rebound_pci_bdfs`` = [BDF]) while the restarted collector's first reading
  -- the sample the post-action validation judges -- and every later one show
  the shortfall again, so the validation fails on fresh evidence.

Before any of these a deadman timer is armed that stops the trap, clears the
override and rebinds the function, so the node returns to its baseline even
if the runner, the probe Pod and the control plane all disappear; everything
the run created is recorded under ``/run/gpu-fault-acceptance/efa-driver/
<digest>/state.json`` first. ``release`` is the cleanup: it undoes exactly
that, verifies the function is bound to ``efa`` again and fails loudly when
it is not. Nothing here unloads a module, touches another function, writes a
counter or runs ``modprobe``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SYSFS = Path("/sys")
BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")
NODE_AGENT_ENV = Path("/etc/gpu-fault/node-agent.env")
COLLECTOR_ENV = Path("/etc/gpu-fault/collector.env")
STATE_ROOT = Path("/run/gpu-fault-acceptance/efa-driver")
UNIT_PREFIX = "gfrs-"
DRIVER = "efa"
EFA_VENDOR = "0x1d0f"
EFA_DEVICE_PREFIX = "0xefa"
#: The ``driver_override`` value of a held function: a driver no module
#: provides, so no bind can satisfy it until the override is cleared.
HOLD_OVERRIDE = "gfrs-live-hold"
#: What ``driver_override`` reads back when NO override is set. The attribute
#: formats the ``pdev->driver_override`` pointer with ``"%s\n"`` --
#: ``drivers/pci/pci-sysfs.c`` ``driver_override_show()``: ``sprintf(buf,
#: "%s\n", pdev->driver_override)`` at v3.16 (the attribute's first release),
#: ``sysfs_emit(buf, "%s\n", pdev->driver_override)`` at v5.18 and v6.1 -- and
#: vsprintf renders a NULL ``%s`` argument as the literal ``(null)``
#: (``lib/vsprintf.c``: ``string()`` at v4.19, ``if ((unsigned long)s <
#: PAGE_SIZE) s = "(null)";``; ``check_pointer_msg()`` at v6.1, ``if (!ptr)
#: return "(null)";``). So an unset override reads ``"(null)\n"``, never an
#: empty string: the store side turns ``""`` and ``"\n"`` into a NULL pointer
#: (``driver_override_store()`` in pci-sysfs.c through v5.18 -- ``if
#: (strlen(driver_override)) ... else { kfree(driver_override);
#: pdev->driver_override = NULL; }`` -- and ``drivers/base/driver.c``
#: ``driver_set_override()`` from v5.19, read at v6.12: ``/* Empty string
#: passed - clear override */`` and ``/* "\n" passed - clear override */``),
#: as ``Documentation/ABI/testing/sysfs-bus-pci`` promises ("may be cleared
#: with an empty string (echo > driver_override)"). The attribute is
#: unconditional (``static DEVICE_ATTR_RW(driver_override)`` in
#: ``pci_dev_attrs[]``), so a function without it is not a PCI device of a
#: kernel this fixture knows. Live read-backs: ``(null)`` on every EFA
#: function of the HyperPod test node (kernel 6.12.103, AL2023, 2026-10-27
#: efa-driver-survey.json) and on every PCI function of the controller host
#: (kernel 7.0.0-1012-aws). The first live run compared this against a
#: made-up ``"(none)"`` and refused every real node.
KERNEL_NO_OVERRIDE = "(null)"
#: The product's own boolean tokens (``gpu_fault.env.TRUE_TOKENS``; the probe
#: is stdlib-only and cannot import them): the installer writes ``"true"``.
TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
SAFE_BDF = re.compile(r"^0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
MIN_RESTORE_SECONDS = 60
MAX_RESTORE_SECONDS = 5400
MIN_GRACE_SECONDS = 1
MAX_GRACE_SECONDS = 60
#: The trap's re-unbind never lands sooner than this after the product's bind
#: was seen, whatever restart it sees: the Node Agent reads every function back
#: right after its bind write returns (``REMEDIATE_EFA_DRIVER``'s
#: ``bind_and_verify``), and the product's own restart only follows that
#: SUCCEEDED receipt, so the floor only ever guards against a restart of the
#: collector nobody planned (its watchdog, ``Restart=always``).
MIN_FLOOR_SECONDS = 1
#: The unit whose restart IS the product's post-repair snapshot. The plan runs
#: TRIGGER_HEALTH_SNAPSHOT after RESTART_EFA_DEVICE_PLUGIN; the Node Agent
#: writes the gpu/host snapshot requests, ``systemctl restart``s
#: gpu-fault-metrics-collector.service and then this unit
#: (``node_agent/operations/diagnostics.py`` ``_trigger_health_snapshot``). The
#: restarted host collector sends READY=1 and, with the ``host.request`` force
#: snapshot pending, runs its first tick at once without the start-up spread
#: (``collectors/host/collector.py`` ``run``): that tick's EFA inventory, about
#: two seconds after the start, is the sample VALIDATE_FABRIC judges -- the
#: first newer than the last recovery step (``adapters/gpu_validation.py``
#: ``inventory_observation_window``). The trap only ever READS this unit's
#: state; it never restarts, stops or changes it.
SNAPSHOT_COLLECTOR_UNIT = "gpu-fault-host-collector.service"
#: The ``systemctl show`` properties that change when that unit (re)starts: a
#: new invocation (systemd >= 232), a new or departed main process, a new
#: start of the main process, a new activation (READY=1 of ``Type=notify``).
COLLECTOR_IDENTITY_PROPERTIES = (
    "InvocationID",
    "MainPID",
    "ExecMainStartTimestampMonotonic",
    "ActiveEnterTimestampMonotonic",
)
COLLECTOR_STATE_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    *COLLECTOR_IDENTITY_PROPERTIES,
)
#: The ActiveState of a unit that was active and is now being stopped or
#: started again: a restart in progress, its next first tick still ahead.
RESTARTING_STATES = frozenset({"deactivating", "inactive", "failed", "activating"})
#: How often the trap looks for the product's bind: the collector's state is
#: first read when the bind shows, so a slower poll would let a restart that
#: follows the bind closely land before that first reading and go unseen.
BIND_POLL_SECONDS = 0.25
#: How often the trap reads the collector between the product's bind and the
#: restart (the restart-to-first-tick margin is about two seconds), and how
#: long one read may take before it counts as a reading without identity.
COLLECTOR_POLL_SECONDS = 0.2
COLLECTOR_READ_TIMEOUT_SECONDS = 10
#: ``trigger`` of the trap's record: the collector's restart was seen, or none
#: was within the grace and the trap fell back to the old fixed delay.
TRIGGER_COLLECTOR_RESTART = "collector_restart"
TRIGGER_GRACE_FALLBACK = "grace_fallback"
#: Node facts the runner's preflight reads back (values are shell-quoted by
#: the installer's ``write_env``; the quotes are stripped).
ENV_KEYS = (
    "GPU_FAULT_NODE_ALLOW_EFA_DRIVER_REMEDIATION",
    "GPU_FAULT_EFA_DRIVER_BIND_PATH",
    "GPU_FAULT_EFA_DRIVER_MODULE",
    "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT",
    "GPU_FAULT_PCI_DEVICES_ROOT",
)
MODES = ("hold", "trap")


class ProbeError(RuntimeError):
    """A refused or failed step.

    ``reason`` is one token of a closed vocabulary (``[a-z][a-z0-9_]{2,40}``)
    the host fixture shows in its exception text (``; reason
    driver_override_set``); ``detail`` is a JSON object that travels with the
    message into the case's ``probe-error.json``.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.detail = detail


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def monotonic_clock() -> float:
    """``time.monotonic`` behind a name the tests can script."""

    return time.monotonic()


def pause(seconds: float) -> None:
    """``time.sleep`` behind a name the tests can script."""

    time.sleep(seconds)


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
            f"command failed ({completed.returncode}): {shlex.join(command)}: "
            f"{completed.stderr.strip()[:300]}"
        )
    return completed


def checked_run_id(value: str) -> str:
    if SAFE_RUN_ID.fullmatch(value) is None:
        raise ProbeError("unsafe run ID")
    return value


def normalize_bdf(value: str) -> str:
    """The one ``0000:bb:dd.f`` spelling (sysfs, nvidia-smi's eight-digit
    domain, a BDF without its function)."""

    text = value.strip().lower()
    parts = text.split(":")
    if len(parts) == 3 and len(parts[0]) == 8 and parts[0].startswith("0000"):
        text = ":".join([parts[0][4:], parts[1], parts[2]])
    if re.fullmatch(r"0000:[0-9a-f]{2}:[0-9a-f]{2}", text):
        text += ".0"
    if SAFE_BDF.fullmatch(text) is None:
        raise ProbeError("unsafe PCI BDF")
    return text


def bounded(value: int, low: int, high: int, label: str) -> int:
    number = int(value)
    if not low <= number <= high:
        raise ProbeError(f"{label} is outside {low}..{high}")
    return number


def run_digest(run_id: str) -> str:
    return hashlib.sha256(checked_run_id(run_id).encode()).hexdigest()[:12]


def state_dir(run_id: str) -> Path:
    return STATE_ROOT / run_digest(run_id)


def unit_name(run_id: str, kind: str) -> str:
    return f"{UNIT_PREFIX}{run_digest(run_id)}-efa-{kind}"


def load_state(run_id: str) -> dict[str, Any] | None:
    path = state_dir(run_id) / "state.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        raise ProbeError("EFA driver fixture state identity mismatch")
    return value


def save_state(run_id: str, state: dict[str, Any]) -> None:
    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "state.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


# --------------------------------------------------------------------------- #
# Read-only facts
# --------------------------------------------------------------------------- #
def pci_devices_root() -> Path:
    return SYSFS / "bus" / "pci" / "devices"


def driver_root() -> Path:
    return SYSFS / "bus" / "pci" / "drivers" / DRIVER


def infiniband_root() -> Path:
    return SYSFS / "class" / "infiniband"


def device_dir(bdf: str) -> Path:
    return pci_devices_root() / bdf


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def is_efa_function(bdf: str) -> bool:
    directory = device_dir(bdf)
    vendor = read_text(directory / "vendor").lower()
    device = read_text(directory / "device").lower()
    return vendor == EFA_VENDOR and device.startswith(EFA_DEVICE_PREFIX)


def bound_driver(bdf: str) -> str | None:
    try:
        return (device_dir(bdf) / "driver").resolve(strict=True).name
    except OSError:
        return None


def is_bound(bdf: str) -> bool:
    return (driver_root() / bdf).exists()


def override_name(text: str) -> str | None:
    """The driver name a ``driver_override`` read-back carries, or ``None``.

    ``(null)`` is the kernel's rendering of the unset (NULL) override, see
    ``KERNEL_NO_OVERRIDE``; an empty read-back counts as unset too because the
    kernel never stores an empty name (the store clears instead), so nothing
    real is hidden by accepting it. Anything else -- ``efa``, ``vfio-pci``,
    the ABI's own opt-out example ``none``, the old harness constant
    ``(none)`` -- is a stored override. (A name spelled literally ``(null)``
    would read the same as unset; the kernel ABI itself cannot tell them
    apart, and this fixture's own hold value never collides.)
    """

    value = text.strip()
    if value in {"", KERNEL_NO_OVERRIDE}:
        return None
    return value


def override_sysfs_text(bdf: str) -> str:
    """The raw ``driver_override`` read-back, for the evidence receipts."""

    return read_text(device_dir(bdf) / "driver_override")


def read_override(bdf: str) -> str | None:
    """The function's stored override name, ``None`` when none is set."""

    path = device_dir(bdf) / "driver_override"
    if not path.exists():
        raise ProbeError(
            f"{bdf} has no driver_override attribute", reason="no_override_attribute"
        )
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise ProbeError(
            f"{bdf} driver_override is unreadable: {error.strerror}",
            reason="override_unreadable",
        ) from error
    return override_name(text)


def infiniband_map() -> dict[str, str]:
    """``{pci_bdf: rdma device name}`` of the node's RDMA devices."""

    result: dict[str, str] = {}
    root = infiniband_root()
    if not root.is_dir():
        return result
    for device in sorted(root.iterdir()):
        try:
            bdf = (device / "device").resolve().name.lower()
        except OSError:
            continue
        result[bdf] = device.name
    return result


def efa_functions() -> list[dict[str, Any]]:
    """Every EFA PCI function with its binding, override and RDMA device."""

    rdma = infiniband_map()
    result = []
    root = pci_devices_root()
    if not root.is_dir():
        raise ProbeError("the PCI device tree is not readable")
    for entry in sorted(root.iterdir()):
        bdf = entry.name.lower()
        if SAFE_BDF.fullmatch(bdf) is None or not is_efa_function(bdf):
            continue
        override_path = entry / "driver_override"
        result.append(
            {
                "pci_bdf": bdf,
                "driver": bound_driver(bdf),
                "bound": is_bound(bdf),
                "driver_override": (
                    override_name(read_text(override_path))
                    if override_path.exists()
                    else None
                ),
                "rdma_device": rdma.get(bdf),
            }
        )
    return result


def efa_module_loaded() -> bool:
    return (SYSFS / "module" / DRIVER).is_dir()


def boot_id() -> str:
    return read_text(BOOT_ID_FILE)


def _unquote(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def node_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in (COLLECTOR_ENV, NODE_AGENT_ENV):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() in ENV_KEYS:
                values[key.strip()] = _unquote(value)
    return values


def unit_state(unit: str) -> dict[str, str]:
    completed = run(
        [
            "systemctl",
            "show",
            unit,
            "--property=LoadState,ActiveState,SubState,MainPID,ExecMainStartTimestamp,"
            "Result,ExecMainStatus",
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


def collector_state() -> dict[str, str]:
    """The snapshot collector's (re)start identity, read-only.

    One ``systemctl show`` of ``SNAPSHOT_COLLECTOR_UNIT`` -- never a restart,
    stop or any other verb on the product's unit. A read that fails or hangs is
    a reading carrying ``error`` instead of an exception: the trap must still
    reach its grace fallback.
    """

    try:
        completed = run(
            [
                "systemctl",
                "show",
                SNAPSHOT_COLLECTOR_UNIT,
                "--property=" + ",".join(COLLECTOR_STATE_PROPERTIES),
            ],
            check=False,
            timeout=COLLECTOR_READ_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": type(error).__name__}
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in COLLECTOR_STATE_PROPERTIES:
            values[key] = value.strip()
    if completed.returncode:
        values["error"] = f"systemctl show exited {completed.returncode}"
    elif values.get("LoadState") != "loaded":
        values["error"] = f"LoadState={values.get('LoadState') or '?'}"
    return values


def collector_restart(before: dict[str, str], now: dict[str, str]) -> str | None:
    """The property saying the snapshot collector restarted between two
    readings, ``None`` when none does (or either reading is unusable).

    A changed InvocationID, main PID, main-process start or activation is a new
    -- or departing -- process; an ``active`` unit that now reads stopping or
    starting is a restart under way. Either way the restarted collector's first
    tick, the sample the post-repair validation judges, is still ahead.
    """

    if before.get("error") or now.get("error"):
        return None
    for key in COLLECTOR_IDENTITY_PROPERTIES:
        if key in before and key in now and before[key] != now[key]:
            return key
    if (
        before.get("ActiveState") == "active"
        and now.get("ActiveState") in RESTARTING_STATES
    ):
        return "ActiveState"
    return None


def run_units(run_id: str) -> list[str]:
    """The transient units of this run systemd still knows."""

    prefix = unit_name(run_id, "")
    listing = run(
        ["systemctl", "list-units", "--all", "--no-legend", "--plain", f"{prefix}*"],
        check=False,
    ).stdout
    names = []
    for line in listing.splitlines():
        name = line.split()[0] if line.strip() else ""
        if name.startswith(prefix):
            names.append(name)
    return sorted(names)


def summary(run_id: str, bdf: str) -> dict[str, Any]:
    """The read-only state receipt: this run's units, the function's binding."""

    functions = efa_functions()
    deadman = unit_name(run_id, "deadman")
    trap = unit_name(run_id, "trap")
    return {
        "run_id": run_id,
        "pci_bdf": bdf,
        "bound": is_bound(bdf),
        "driver": bound_driver(bdf),
        "driver_override": read_override(bdf),
        # The kernel's own words next to the parsed value: the read-back that
        # proved the "(none)" comparison wrong lives in the survey evidence.
        "driver_override_sysfs": override_sysfs_text(bdf),
        "rdma_device": infiniband_map().get(bdf),
        "module_loaded": efa_module_loaded(),
        "efa_function_count": len(functions),
        "bound_count": sum(1 for item in functions if item["driver"] == DRIVER),
        "functions": functions,
        "boot_id": boot_id(),
        "deadman": {
            "timer": unit_state(f"{deadman}.timer"),
            "service": unit_state(f"{deadman}.service"),
        },
        "trap": unit_state(f"{trap}.service"),
        "trap_fired_at": read_text(state_dir(run_id) / "trap-fired") or None,
        "run_units": run_units(run_id),
        "observed_at": utc_now(),
    }


# --------------------------------------------------------------------------- #
# Deadman: a copy of this file the timer runs after the Pod is gone
# --------------------------------------------------------------------------- #
def probe_copy(run_id: str) -> Path:
    """The copy of this file the deadman and the trap unit run.

    Always the shipped version: a copy an earlier arm of the same run id left
    in ``/run`` on this boot (an older probe -- one whose trap slept a fixed
    grace) is replaced, atomically, rather than run again.
    """

    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = directory / "probe.py"
    source = Path(__file__).resolve().read_bytes()
    if not target.is_file() or target.read_bytes() != source:
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(source)
        temporary.chmod(0o700)
        os.replace(temporary, target)
    return target


def arm_deadman(run_id: str, bdf: str, seconds: int) -> str:
    """Arm ``release --automatic`` after ``seconds``; raise if not armed.

    The timer restarts on failure like the node probe's deadman, so a restore
    that loses a race with the trap unit is retried until the function is bound
    to efa with no override.
    """

    bounded(seconds, MIN_RESTORE_SECONDS, MAX_RESTORE_SECONDS, "restore seconds")
    name = unit_name(run_id, "deadman")
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service"], check=False)
    run(
        [
            "systemd-run",
            f"--unit={name}",
            f"--on-active={int(seconds)}s",
            "--timer-property=AccuracySec=1s",
            "--property=Restart=on-failure",
            "--property=RestartSec=15s",
            "--property=StartLimitIntervalSec=0",
            sys.executable,
            str(probe_copy(run_id)),
            "release",
            "--run-id",
            run_id,
            "--pci-bdf",
            bdf,
            "--automatic",
        ]
    )
    if unit_state(f"{name}.timer").get("ActiveState") != "active":
        raise ProbeError(f"deadman {name} is not armed")
    return name


# --------------------------------------------------------------------------- #
# Injection primitives (one allow-listed function only)
# --------------------------------------------------------------------------- #
def _preconditions(bdf: str) -> dict[str, Any]:
    """Refuse to touch the node unless exactly this one function is safe to.

    The function must be a real EFA data function bound to ``efa`` now, the
    node must carry more than one EFA function (so one stays a spare and the
    module is never left with nothing bound), the ``efa`` module must be
    loaded (we never ``modprobe``), and the product's own EFA driver
    remediation must be enabled or the product cannot take its branch.
    """

    if not device_dir(bdf).is_dir():
        raise ProbeError(
            f"{bdf} is not a PCI device on this node", reason="not_a_pci_device"
        )
    if not is_efa_function(bdf):
        raise ProbeError(
            f"{bdf} is not an EFA data function (vendor/device)",
            reason="not_an_efa_function",
        )
    if bound_driver(bdf) != DRIVER:
        raise ProbeError(
            f"{bdf} is not bound to {DRIVER} at baseline", reason="not_bound_to_efa"
        )
    if not efa_module_loaded():
        raise ProbeError(
            f"the {DRIVER} module is not loaded; refusing to inject",
            reason="module_not_loaded",
        )
    functions = efa_functions()
    bound = [item for item in functions if item["driver"] == DRIVER]
    if len(bound) < 2:
        raise ProbeError(
            f"only {len(bound)} EFA function(s) bound; refusing to unbind the last one",
            reason="last_bound_function",
        )
    # ``None`` is the kernel's "(null)" read-back (KERNEL_NO_OVERRIDE); any
    # stored name -- efa, vfio-pci, the ABI's opt-out "none" -- means another
    # actor holds this function and the fixture must not touch it.
    override = read_override(bdf)
    if override is not None:
        raise ProbeError(
            f"{bdf} already has a driver_override: {override!r}",
            reason="driver_override_set",
        )
    env = node_env()
    enabled = env.get("GPU_FAULT_NODE_ALLOW_EFA_DRIVER_REMEDIATION", "")
    if enabled.strip().lower() not in TRUE_TOKENS:
        raise ProbeError(
            "GPU_FAULT_NODE_ALLOW_EFA_DRIVER_REMEDIATION is not enabled; the "
            "product cannot run its REMEDIATE_EFA_DRIVER branch",
            reason="remediation_disabled",
        )
    return {
        "baseline_bound_count": len(bound),
        "efa_function_count": len(functions),
        "node_env": env,
        "boot_id": boot_id(),
    }


def _set_override(bdf: str, value: str) -> None:
    """Write the function's ``driver_override`` (``value`` empty clears it)."""

    path = device_dir(bdf) / "driver_override"
    if not path.exists():
        raise ProbeError(f"{bdf} has no driver_override attribute")
    # An empty write clears the override; a name restricts binding to it.
    path.write_text((value + "\n") if value else "\n", encoding="ascii")


def _unbind(bdf: str) -> None:
    root = driver_root()
    if not root.joinpath(bdf).exists():
        return
    root.joinpath("unbind").write_text(bdf + "\n", encoding="ascii")


def _bind(bdf: str) -> None:
    root = driver_root()
    if root.joinpath(bdf).exists():
        return
    root.joinpath("bind").write_text(bdf + "\n", encoding="ascii")


def hold_unbind(arguments: argparse.Namespace) -> None:
    """bind_failure: hold the function un-rebindable, then unbind it.

    ``driver_override`` is set to a name no module provides, so the kernel
    refuses every bind of this one function to ``efa`` (ENODEV) and the
    product's REMEDIATE_EFA_DRIVER records the bind failure for this BDF.
    Reversible by one write to ``driver_override`` or a reboot; the module and
    every other function are untouched.
    """

    run_id = checked_run_id(arguments.run_id)
    bdf = normalize_bdf(arguments.pci_bdf)
    restore_seconds = bounded(
        arguments.restore_seconds, MIN_RESTORE_SECONDS, MAX_RESTORE_SECONDS, "restore"
    )
    facts = _preconditions(bdf)
    deadman = arm_deadman(run_id, bdf, restore_seconds)
    save_state(
        run_id,
        {
            "run_id": run_id,
            "mode": "hold",
            "pci_bdf": bdf,
            "deadman_unit": deadman,
            "restore_seconds": restore_seconds,
            "override_value": HOLD_OVERRIDE,
            **facts,
        },
    )
    _set_override(bdf, HOLD_OVERRIDE)
    _unbind(bdf)
    emit(
        {
            "run_id": run_id,
            "mode": "hold",
            "pci_bdf": bdf,
            "driver_override": read_override(bdf),
            "bound": is_bound(bdf),
            "deadman_timer": deadman + ".timer",
            "restore_seconds": restore_seconds,
            "baseline_bound_count": facts["baseline_bound_count"],
            "boot_id": facts["boot_id"],
        }
    )


def trap_timing(arguments: argparse.Namespace) -> tuple[int, int]:
    """``(grace, floor)`` of the trap: the latest and the earliest re-unbind
    after the product's bind, in seconds; the floor may not exceed the grace.
    A caller without ``floor_seconds`` gets ``MIN_FLOOR_SECONDS``."""

    grace = bounded(
        arguments.grace_seconds, MIN_GRACE_SECONDS, MAX_GRACE_SECONDS, "grace"
    )
    floor = bounded(
        getattr(arguments, "floor_seconds", MIN_FLOOR_SECONDS),
        MIN_FLOOR_SECONDS,
        grace,
        "floor",
    )
    return grace, floor


def trap_unbind(arguments: argparse.Namespace) -> None:
    """post_repair_failure: unbind now, and re-unbind right after the product
    binds it back -- on its post-repair snapshot -- so the action's receipt
    succeeds and the post-action validation fails on fresh evidence.

    A bounded transient unit waits for this unbind and then for the function to
    be bound again (the product's REMEDIATE_EFA_DRIVER), then watches the
    product's host collector restart for TRIGGER_HEALTH_SNAPSHOT
    (``trap_run``) and unbinds the function once more when it shows -- never
    sooner than ``floor`` seconds after the bind, at ``grace`` seconds at the
    latest -- and records when and why. No ``driver_override`` is used, so the
    second unbind is the only change and the deadman's plain rebind restores
    the baseline.
    """

    run_id = checked_run_id(arguments.run_id)
    bdf = normalize_bdf(arguments.pci_bdf)
    restore_seconds = bounded(
        arguments.restore_seconds, MIN_RESTORE_SECONDS, MAX_RESTORE_SECONDS, "restore"
    )
    grace, floor = trap_timing(arguments)
    trap_window = bounded(
        arguments.trap_window_seconds, 30, MAX_RESTORE_SECONDS, "trap window"
    )
    facts = _preconditions(bdf)
    deadman = arm_deadman(run_id, bdf, restore_seconds)
    trap = _arm_trap(run_id, bdf, grace=grace, floor=floor, window=trap_window)
    # The record an earlier arm of this run id left on this boot is not this
    # trap's (``_arm_trap`` stopped that arm's unit, and this one writes
    # nothing before the unbind below): ``summary`` must read nothing until
    # this unit writes its own.
    (state_dir(run_id) / "trap-fired").unlink(missing_ok=True)
    save_state(
        run_id,
        {
            "run_id": run_id,
            "mode": "trap",
            "pci_bdf": bdf,
            "deadman_unit": deadman,
            "trap_unit": trap,
            "restore_seconds": restore_seconds,
            "grace_seconds": grace,
            "floor_seconds": floor,
            "collector_unit": SNAPSHOT_COLLECTOR_UNIT,
            "override_value": None,
            **facts,
        },
    )
    _unbind(bdf)
    emit(
        {
            "run_id": run_id,
            "mode": "trap",
            "pci_bdf": bdf,
            "bound": is_bound(bdf),
            "trap_unit": trap + ".service",
            "grace_seconds": grace,
            "floor_seconds": floor,
            "collector_unit": SNAPSHOT_COLLECTOR_UNIT,
            "deadman_timer": deadman + ".timer",
            "restore_seconds": restore_seconds,
            "baseline_bound_count": facts["baseline_bound_count"],
            "boot_id": facts["boot_id"],
        }
    )


def _arm_trap(run_id: str, bdf: str, *, grace: int, floor: int, window: int) -> str:
    """A bounded transient unit that re-unbinds the function once, later.

    The unit keeps systemd-run's default ``Type=simple``: ``systemd-run`` waits
    for the start job unless ``--no-block`` ("systemd-run will wait until the
    unit's start-up is completed", systemd-run(1)), and a ``Type=oneshot``
    service completes its start job only when its process exits ("the service
    manager will consider the unit up after the main process exits",
    systemd.service(5)), so ``Type=oneshot`` would block this call for the
    trap's whole wait, past ``run()``'s timeout; and ``RuntimeMaxSec`` "does
    not have any effect on Type=oneshot services" (systemd.service(5)) -- it
    is the kill switch only for a simple service.
    """

    name = unit_name(run_id, "trap")
    run(["systemctl", "stop", f"{name}.service"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service"], check=False)
    run(
        [
            "systemd-run",
            f"--unit={name}",
            f"--property=RuntimeMaxSec={int(window) + grace + 30}",
            sys.executable,
            str(probe_copy(run_id)),
            "trap-run",
            "--run-id",
            run_id,
            "--pci-bdf",
            bdf,
            "--grace-seconds",
            str(grace),
            "--floor-seconds",
            str(floor),
            "--trap-window-seconds",
            str(window),
        ]
    )
    if unit_state(f"{name}.service").get("LoadState") == "not-found":
        raise ProbeError(f"trap {name} was not started")
    return name


def trap_run(arguments: argparse.Namespace) -> None:
    """Inside the trap unit: wait for the fixture's unbind, then the product's
    bind, then re-unbind once on the product's post-repair snapshot.

    After the bind the unit reads the snapshot collector's state every
    ``COLLECTOR_POLL_SECONDS`` (``collector_state``, read-only) and re-unbinds
    as soon as a reading differs from the one taken when the bind was seen
    (``collector_restart``): the restart's first tick comes about two seconds
    later and must already read the function unbound. Never sooner than
    ``floor`` seconds after the bind; when no restart shows within ``grace``
    seconds, at ``grace`` -- the old fixed delay, kept as the bounded fallback.
    Live 2026-11-06 (HyperPod, R23 validation) that fixed 5 s grace alone lost
    the race: TRIGGER_HEALTH_SNAPSHOT restarted the collectors right after the
    bind receipt, the restarted host collector's first sample (16/16) came
    about 3.6 s after the bind the trap had seen and VALIDATE_FABRIC SUCCEEDED
    on it, before the re-unbind 5 s after that bind; earlier runs had had the
    opposite order.

    Prints nothing the runner reads (it runs detached); it records its own
    outcome in ``trap-fired`` so ``summary`` can report it: ``trigger``
    (``collector_restart`` or ``grace_fallback``), the seconds from the bind it
    saw to the re-unbind, and the collector readings the trigger rests on.
    """

    run_id = checked_run_id(arguments.run_id)
    bdf = normalize_bdf(arguments.pci_bdf)
    grace, floor = trap_timing(arguments)
    window = bounded(
        arguments.trap_window_seconds, 30, MAX_RESTORE_SECONDS, "trap window"
    )
    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    deadline = monotonic_clock() + window

    def not_fired(reason: str) -> None:
        (directory / "trap-fired").write_text(
            json.dumps({"fired": False, "reason": reason, "at": utc_now()}),
            encoding="utf-8",
        )

    # The trap is armed BEFORE trap_unbind's own unbind (state first, then the
    # mutation), so the baseline binding still stands when this starts: wait
    # for the fixture's unbind first, or the baseline would count as the
    # product's bind, the function would be unbound twice at once, the
    # product's later bind would stand, and the variant would fail the product
    # for the harness's own race.
    unbound_seen = False
    while monotonic_clock() < deadline:
        if not is_bound(bdf):
            unbound_seen = True
            break
        pause(1)
    if not unbound_seen:
        not_fired("unbind not observed")
        return
    bound_seen = False
    while monotonic_clock() < deadline:
        if is_bound(bdf) and bound_driver(bdf) == DRIVER:
            bound_seen = True
            break
        pause(BIND_POLL_SECONDS)
    if not bound_seen:
        not_fired("bind not observed")
        return
    bind_seen = monotonic_clock()
    bind_seen_at = utc_now()
    trigger, before, restart, reads = _await_snapshot_restart(bind_seen, grace)
    # The Node Agent's synchronous bind read-back records SUCCEEDED before the
    # product restarts anything; the floor covers a restart nobody planned.
    elapsed = monotonic_clock() - bind_seen
    if elapsed < floor:
        pause(floor - elapsed)
    _unbind(bdf)
    fired_after = monotonic_clock() - bind_seen
    (directory / "trap-fired").write_text(
        json.dumps(
            {
                "fired": True,
                "bound": is_bound(bdf),
                "at": utc_now(),
                "trigger": trigger,
                "bind_seen_at": bind_seen_at,
                "seconds_after_bind": round(fired_after, 3),
                "floor_seconds": floor,
                "grace_seconds": grace,
                "collector_unit": SNAPSHOT_COLLECTOR_UNIT,
                "collector_before": before,
                "collector_restart": restart,
                "collector_reads": reads,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _await_snapshot_restart(
    bind_seen: float, grace: int
) -> tuple[str, dict[str, str], dict[str, Any] | None, int]:
    """Watch the snapshot collector from the product's bind on, read-only.

    ``(trigger, reading at the bind, the restart seen or None, reads)``: the
    first usable reading is the baseline, a later one ``collector_restart``
    tells apart ends the watch on ``TRIGGER_COLLECTOR_RESTART``, and the grace
    ends it on ``TRIGGER_GRACE_FALLBACK``.
    """

    before = collector_state()
    reads = 1
    while monotonic_clock() - bind_seen < grace:
        pause(COLLECTOR_POLL_SECONDS)
        now = collector_state()
        reads += 1
        if before.get("error"):
            before = now
            continue
        changed = collector_restart(before, now)
        if changed is not None:
            restart = {
                "property": changed,
                "state": now,
                "seconds_after_bind": round(monotonic_clock() - bind_seen, 3),
            }
            return TRIGGER_COLLECTOR_RESTART, before, restart, reads
    return TRIGGER_GRACE_FALLBACK, before, None, reads


# --------------------------------------------------------------------------- #
# Cleanup / deadman restore
# --------------------------------------------------------------------------- #
def _disarm(run_id: str) -> list[dict[str, str]]:
    """Stop the trap and the deadman of this run (idempotent)."""

    actions = []
    for kind in ("trap", "deadman"):
        name = unit_name(run_id, kind)
        run(["systemctl", "stop", f"{name}.timer"], check=False)
        if unit_state(f"{name}.service").get("MainPID") != str(os.getpid()):
            run(["systemctl", "stop", f"{name}.service"], check=False)
        run(
            ["systemctl", "reset-failed", f"{name}.service", f"{name}.timer"],
            check=False,
        )
        actions.append({kind: name})
    return actions


def release(arguments: argparse.Namespace) -> None:
    """Restore the baseline: stop the trap, clear any override, rebind.

    Called by the harness cleanup and, with ``--automatic``, by the deadman.
    A run must leave the function bound to ``efa`` with no override and the
    module loaded, or this fails loudly so the variant's cleanup fails.
    """

    run_id = checked_run_id(arguments.run_id)
    bdf = normalize_bdf(arguments.pci_bdf)
    state = load_state(run_id)
    # Stop the trap first so it cannot re-unbind after we rebind.
    disarmed = _disarm(run_id)
    present = device_dir(bdf).is_dir()
    override_before = read_override(bdf) if present else None
    if override_before is not None:
        # The ABI's clearing write ("echo > driver_override"); the kernel's
        # "(null)" read-back needs none and is never a stored name.
        _set_override(bdf, "")
    _bind(bdf)
    # The end state the deadman and the harness both guarantee.
    bound = is_bound(bdf) and bound_driver(bdf) == DRIVER
    override = read_override(bdf) if present else None
    module = efa_module_loaded()
    remaining = run_units(run_id)
    # Only the units that are still loaded block "clean"; a .timer/.service
    # systemd has forgotten does not.
    stuck = [
        name for name in remaining if unit_state(name).get("LoadState") != "not-found"
    ]
    clean = bound and override is None and module and not stuck
    receipt = {
        "run_id": run_id,
        "pci_bdf": bdf,
        "mode": (state or {}).get("mode"),
        "automatic": bool(getattr(arguments, "automatic", False)),
        "override_before": override_before,
        "bound": bound,
        "driver": bound_driver(bdf),
        "driver_override": override,
        "driver_override_sysfs": override_sysfs_text(bdf) if present else None,
        "module_loaded": module,
        "trap_fired": read_text(state_dir(run_id) / "trap-fired") or None,
        "disarmed": disarmed,
        "remaining_units": stuck,
        "boot_id": boot_id(),
        "clean": clean,
    }
    if not getattr(arguments, "automatic", False):
        # Forget the state only on a clean manual release; the deadman never
        # removes state (its next retry needs it).
        if clean:
            path = state_dir(run_id) / "state.json"
            if path.exists():
                path.unlink()
        emit(receipt)
        if not clean:
            # The receipt line above is not what the host fixture reads (it
            # keeps the last line only): carry it as the error's detail.
            raise ProbeError(
                f"EFA driver fixture did not restore {bdf}: "
                f"bound={bound} override={override!r} module_loaded={module} "
                f"units={stuck}",
                reason="not_restored",
                detail=receipt,
            )
    else:
        emit(receipt)
        if not clean:
            # A nonzero exit makes the deadman's Restart=on-failure retry.
            raise ProbeError(f"deadman could not restore {bdf}", reason="not_restored")


def show_summary(arguments: argparse.Namespace) -> None:
    emit(summary(checked_run_id(arguments.run_id), normalize_bdf(arguments.pci_bdf)))


def list_functions(arguments: argparse.Namespace) -> None:
    """Enumerate the node's EFA functions (read-only, no target needed)."""

    functions = efa_functions()
    emit(
        {
            "run_id": checked_run_id(arguments.run_id),
            "functions": functions,
            "module_loaded": efa_module_loaded(),
            "bound_count": sum(1 for item in functions if item["driver"] == DRIVER),
            "boot_id": boot_id(),
            "observed_at": utc_now(),
        }
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="GF-GENERAL-LIVE-013 EFA driver fixture"
    )
    commands = value.add_subparsers(dest="command", required=True)

    def command(name: str, handler: Any) -> argparse.ArgumentParser:
        item = commands.add_parser(name)
        item.add_argument("--run-id", required=True)
        item.add_argument("--pci-bdf", required=True)
        item.set_defaults(handler=handler)
        return item

    command("summary", show_summary)

    functions = commands.add_parser("functions")
    functions.add_argument("--run-id", required=True)
    functions.set_defaults(handler=list_functions)

    hold = command("hold-unbind", hold_unbind)
    hold.add_argument("--restore-seconds", type=int, default=600)

    trap = command("trap-unbind", trap_unbind)
    trap.add_argument("--restore-seconds", type=int, default=900)
    trap.add_argument("--grace-seconds", type=int, default=5)
    trap.add_argument("--floor-seconds", type=int, default=MIN_FLOOR_SECONDS)
    trap.add_argument("--trap-window-seconds", type=int, default=600)

    runner = command("trap-run", trap_run)
    runner.add_argument("--grace-seconds", type=int, default=5)
    runner.add_argument("--floor-seconds", type=int, default=MIN_FLOOR_SECONDS)
    runner.add_argument("--trap-window-seconds", type=int, default=600)

    rel = command("release", release)
    rel.add_argument("--automatic", action="store_true")
    return value


#: This file's own helpers: the failing *step* is the frame above them.
HELPER_FRAMES = frozenset(
    {
        "run",
        "bounded",
        "checked_run_id",
        "normalize_bdf",
        "emit",
        "read_text",
        "read_override",
        "override_name",
    }
)


def exception_site(exc: BaseException) -> str | None:
    site, frame = None, exc.__traceback__
    while frame is not None:
        code = frame.tb_frame.f_code
        if code.co_filename == __file__ and code.co_name not in HELPER_FRAMES:
            site = f"{code.co_name}:{frame.tb_lineno}"
        frame = frame.tb_next
    return site


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        # The host fixture's exception text shows only whitelisted tokens (the
        # class, the message digest, the site, a closed-vocabulary reason);
        # the message and the structured detail travel on the same line and
        # the fixture keeps them, redacted, as the case's ``probe-error.json``
        # -- the lane the source-replay probe uses. The first live refusal
        # (2026-10-27) carried only the digest, and the message ("... already
        # has a driver_override: '(null)'") had to be rebuilt from the survey.
        emit(
            {
                "error": type(exc).__name__,
                "error_kind": "probe",
                "error_class": type(exc).__name__,
                "error_code": hashlib.sha256(str(exc).encode()).hexdigest()[:16],
                "error_site": exception_site(exc),
                "error_reason": getattr(exc, "reason", None),
                "error_message": str(exc)[:2000],
                "error_detail": getattr(exc, "detail", None),
                "command": arguments.command,
            }
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
