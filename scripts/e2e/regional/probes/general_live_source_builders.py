"""Fixture builders of the GF-GENERAL-LIVE host-collector source replay.

The second node-side file of ``probes/general_live_source_probe.py``: the
reviewed tree and file builders (``build-pci-fixture``,
``build-infiniband-fixture``, ``build-nvidia-proc-fixture``,
``build-meminfo-fixture``) that copy a real sysfs/procfs reading into the run's
fixture directory with the variant's mutations. A probe ships to the node as one
self-contained file (``host_probe_fixture.py``), so this module travels like the
command wrapper does: the controller installs it through ``install-source``
(base64 + sha256-checked, under ``/run/gfrs-live/source-<digest>/fixtures``)
before the first build and the probe imports the installed copy by path. The
window, deadman, verification and cleanup commands never need it, so a window
closes with the probe alone. Standard library only, Python 3.9+.

All allow-lists stay in the probe; the builders take them from the probe module
the loader registered (``PROBE_ALIAS``) or, in a checkout, the sibling module.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

#: ``sys.modules`` name the probe registers itself under before loading this file.
PROBE_ALIAS = "gfrs_general_live_source_probe"

if TYPE_CHECKING:
    from scripts.e2e.regional.probes import general_live_source_probe as probe
else:
    probe = sys.modules.get(PROBE_ALIAS) or importlib.import_module(
        "scripts.e2e.regional.probes.general_live_source_probe"
    )

ProbeError = probe.ProbeError
emit = probe.emit
safe_id = probe.safe_id
fixtures_dir = probe.fixtures_dir
write_atomic = probe.write_atomic
DEVICE_NAME = probe.DEVICE_NAME
PORT_NAME = probe.PORT_NAME
PORT_STATE = probe.PORT_STATE
COUNTER_NAME = probe.COUNTER_NAME
COUNTER_VALUE = probe.COUNTER_VALUE
DRIVER_NAME = probe.DRIVER_NAME
NETDEV_NAME = probe.NETDEV_NAME
BDF = probe.BDF
EFA_VENDOR = probe.EFA_VENDOR
MAX_TREE_DEVICES = probe.MAX_TREE_DEVICES
MEMINFO_FIELDS = probe.MEMINFO_FIELDS
MEMINFO_FIXTURE = probe.MEMINFO_FIXTURE
MAX_MEMINFO_KB = probe.MAX_MEMINFO_KB


#: Entries a real PCI root may list (a p5en has a few hundred functions).
MAX_PCI_ENTRIES = 4096
#: The regular file an ``--unreadable`` PCI root fixture is: every attempt of
#: the product to list it fails with ENOTDIR, whatever the unit's privileges
#: (a mode-0000 directory is readable to a CAP_DAC_OVERRIDE process).
UNREADABLE_PCI_ROOT = b"gfrs-live: unreadable PCI root fixture (not a directory)\n"


def _pci_identity(entry: Path) -> tuple[str, str]:
    """``(vendor, device)`` of one PCI function as ``is_efa_pci_device`` reads them."""

    try:
        vendor = (entry / "vendor").read_text(encoding="ascii").strip()
        device = (entry / "device").read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise ProbeError(f"{entry.name}: vendor/device unreadable: {error}") from error
    return vendor, device


def _is_efa_function(vendor: str, device: str) -> bool:
    return vendor.lower() == EFA_VENDOR and device.lower().startswith("0xefa")


def _synthetic_pci_root(root: Path, efa_count: int, malformed: bool) -> list[str]:
    """Two plausible non-EFA devices plus ``efa_count`` EFA functions."""

    if not 0 <= efa_count <= 32:
        raise ProbeError("efa-count is outside 0..32")
    for index, (vendor, device) in enumerate(
        (("0x10de", "0x2330"), ("0x1d0f", "0x8061"))
    ):
        slot = root / f"0000:00:{index:02x}.0"
        slot.mkdir(mode=0o755)
        (slot / "vendor").write_text(vendor + "\n", encoding="ascii")
        (slot / "device").write_text(device + "\n", encoding="ascii")
    functions = []
    for index in range(efa_count):
        slot = root / f"0000:53:{index:02x}.0"
        slot.mkdir(mode=0o755)
        (slot / "vendor").write_text(EFA_VENDOR + "\n", encoding="ascii")
        device_id = "0xzz99" if malformed and index == 0 else "0xefa2"
        (slot / "device").write_text(device_id + "\n", encoding="ascii")
        functions.append(slot.name)
    return functions


def _copied_pci_root(root: Path, source: Path, drop: set[str]) -> dict[str, Any]:
    """The node's real PCI root (vendor/device of every function) minus ``drop``.

    013 ``pci_missing``: the product counts EFA functions by vendor/device, so
    the copy carries exactly those two files per function and the dropped
    functions -- which must be EFA functions the source really lists -- are
    simply absent, the way a function that fell off the bus is.
    """

    if not source.is_dir():
        raise ProbeError(f"{source} is not a directory")
    for name in drop:
        if BDF.fullmatch(name) is None:
            raise ProbeError(f"--drop names no PCI function: {name!r}")
    entries = sorted(path for path in source.iterdir() if BDF.fullmatch(path.name))
    if len(entries) > MAX_PCI_ENTRIES:
        raise ProbeError(f"{source} lists more than {MAX_PCI_ENTRIES} functions")
    functions: list[str] = []
    dropped: list[str] = []
    for entry in entries:
        vendor, device = _pci_identity(entry)
        efa = _is_efa_function(vendor, device)
        if entry.name in drop:
            if not efa:
                raise ProbeError(
                    f"--drop {entry.name} is not an EFA function ({vendor} {device})"
                )
            dropped.append(entry.name)
            continue
        slot = root / entry.name
        slot.mkdir(mode=0o755)
        (slot / "vendor").write_text(vendor + "\n", encoding="ascii")
        (slot / "device").write_text(device + "\n", encoding="ascii")
        if efa:
            functions.append(entry.name)
    missing = sorted(drop - set(dropped))
    if missing:
        raise ProbeError(f"--drop names no PCI function of {source}: {missing}")
    return {
        "efa_devices": len(functions),
        "efa_functions": functions,
        "dropped": dropped,
        "entries": len(entries) - len(dropped),
    }


def build_pci_fixture(arguments: argparse.Namespace) -> None:
    """A reviewed PCI root for the ``pci-root`` repoint, in one of three shapes.

    ``--efa-count N [--malformed]``: a synthetic ``/sys/bus/pci/devices`` of two
    non-EFA devices plus ``N`` EFA functions (one bad device id when malformed;
    013 ``unreadable_state``). ``--source ROOT [--drop BDF ...]``: a copy of the
    node's real root with the named EFA functions absent (013 ``pci_missing``:
    the real BDFs, so the evidence names the function that "fell off").
    ``--unreadable``: the fixture path is a regular file, so the product's
    listing of the root fails (ENOTDIR) on every tick -- the read-failure
    negative control that must be UNKNOWN, never zero functions.
    """

    run_id = safe_id(arguments.run_id)
    source = getattr(arguments, "source", None)
    drop = list(getattr(arguments, "drop", None) or [])
    unreadable = bool(getattr(arguments, "unreadable", False))
    modes = [arguments.efa_count is not None, source is not None, unreadable]
    if sum(modes) != 1:
        raise ProbeError("pass exactly one of --efa-count, --source, --unreadable")
    if arguments.malformed and arguments.efa_count is None:
        raise ProbeError("--malformed needs --efa-count")
    if drop and source is None:
        raise ProbeError("--drop needs --source")
    root = fixtures_dir(run_id) / "pci"
    if root.is_dir() and not root.is_symlink():
        shutil.rmtree(root)
    elif root.exists() or root.is_symlink():
        root.unlink()
    receipt: dict[str, Any] = {
        "run_id": run_id,
        "path": str(root),
        "efa_devices": 0,
        "malformed": bool(arguments.malformed),
        "source": source,
        "dropped": [],
        "efa_functions": [],
        "unreadable": unreadable,
    }
    if unreadable:
        root.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        write_atomic(root, UNREADABLE_PCI_ROOT, 0o644)
        emit(receipt)
        return
    root.mkdir(mode=0o755, parents=True, exist_ok=True)
    try:
        if source is not None:
            receipt.update(_copied_pci_root(root, Path(source), set(drop)))
        else:
            functions = _synthetic_pci_root(
                root, int(arguments.efa_count), bool(arguments.malformed)
            )
            receipt.update({"efa_devices": len(functions), "efa_functions": functions})
    except ProbeError:
        # A refused build leaves no half-built root a later window could repoint
        # the unit at (fewer functions than the node has would read as missing).
        shutil.rmtree(root, ignore_errors=True)
        raise
    emit(receipt)


def _copy_text(source: Path, target: Path) -> bool:
    try:
        content = source.read_bytes()
    except OSError:
        return False
    target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    target.write_bytes(content)
    target.chmod(0o644)
    return True


def _split(value: str, separator: str, count: int, label: str) -> list[str]:
    parts = value.split(separator, count - 1)
    if len(parts) != count:
        raise ProbeError(f"{label} must have {count} '{separator}'-separated parts")
    return parts


def _device_dir(root: Path, device: str) -> Path:
    if DEVICE_NAME.fullmatch(device) is None:
        raise ProbeError(f"unsafe RDMA device name {device!r}")
    path = root / device
    if not path.is_dir():
        raise ProbeError(f"the fixture tree has no device {device!r}")
    return path


def _port_dir(root: Path, device: str, port: str) -> Path:
    if PORT_NAME.fullmatch(port) is None:
        raise ProbeError(f"unsafe port name {port!r}")
    path = _device_dir(root, device) / "ports" / port
    if not path.is_dir():
        raise ProbeError(f"device {device!r} has no port {port!r}")
    return path


def _copy_rdma_device(source: Path, target: Path, drivers: Path) -> dict[str, Any]:
    """One real RDMA device as the product reads it, into the fixture tree."""

    real = source.resolve()
    summary: dict[str, Any] = {
        "ports": {},
        "counters": 0,
        "driver": None,
        "netdevs": [],
    }
    for port in sorted((real / "ports").iterdir()) if (real / "ports").is_dir() else []:
        port_target = target / "ports" / port.name
        port_target.mkdir(mode=0o755, parents=True, exist_ok=True)
        states = {}
        for name in ("state", "phys_state"):
            if _copy_text(port / name, port_target / name):
                states[name] = (port_target / name).read_text(encoding="utf-8").strip()
        for group in ("counters", "hw_counters"):
            group_dir = port / group
            if group_dir.is_dir():
                for counter in sorted(group_dir.iterdir()):
                    if counter.is_file() and _copy_text(
                        counter, port_target / group / counter.name
                    ):
                        summary["counters"] += 1
        summary["ports"][port.name] = states
    device_dir = real / "device"
    (target / "device").mkdir(mode=0o755, parents=True, exist_ok=True)
    driver = device_dir / "driver"
    if driver.is_symlink() or driver.is_dir():
        name = driver.resolve().name
        summary["driver"] = name
        (drivers / name).mkdir(mode=0o755, parents=True, exist_ok=True)
        (target / "device" / "driver").symlink_to(drivers / name)
    for name in ("uevent", "vendor", "device"):
        _copy_text(device_dir / name, target / "device" / name)
    net = device_dir / "net"
    if net.is_dir():
        for netdev in sorted(net.iterdir()):
            (target / "device" / "net" / netdev.name).mkdir(
                mode=0o755, parents=True, exist_ok=True
            )
            summary["netdevs"].append(netdev.name)
    return summary


def _set_driver(device_dir: Path, drivers: Path, mode: str) -> None:
    link = device_dir / "device" / "driver"
    if link.is_symlink() or link.exists():
        link.unlink()
    uevent = device_dir / "device" / "uevent"
    lines = (
        [
            line
            for line in uevent.read_text(encoding="utf-8").splitlines()
            if not line.startswith("DRIVER=")
        ]
        if uevent.is_file()
        else []
    )
    if mode == "ambiguous":
        lines += ["DRIVER=efa", "DRIVER=ena"]
    elif mode != "none":
        if DRIVER_NAME.fullmatch(mode) is None:
            raise ProbeError(f"unsafe driver name {mode!r}")
        (drivers / mode).mkdir(mode=0o755, parents=True, exist_ok=True)
        link.symlink_to(drivers / mode)
        lines.append(f"DRIVER={mode}")
    uevent.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    uevent.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_infiniband_fixture(arguments: argparse.Namespace) -> None:
    """A copy of the node's /sys/class/infiniband with reviewed mutations.

    Mutations, applied in this order: ``--drop DEV`` (device absent),
    ``--duplicate NEW=FROM`` (excess device), ``--port-state DEV:PORT:TEXT`` /
    ``--phys-state DEV:PORT:TEXT`` (link inactive), ``--driver
    DEV:none|ambiguous|NAME`` (unknown or non-EFA identity), ``--set
    DEV:PORT:GROUP/COUNTER=VALUE`` / ``--remove-counter DEV:PORT:GROUP/COUNTER``
    (counter replay, a broken pair), ``--no-ports DEV`` (no observable ports).
    The driver directories the ``device/driver`` symlinks resolve to live beside
    the tree, never inside it, so the product's device walk sees devices only.
    """

    run_id = safe_id(arguments.run_id)
    source = Path(arguments.source)
    if not source.is_dir():
        raise ProbeError(f"{source} is not a directory")
    root = fixtures_dir(run_id) / "infiniband"
    drivers = fixtures_dir(run_id) / "infiniband-drivers"
    for path in (root, drivers):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(mode=0o755, parents=True, exist_ok=True)
    for name in arguments.drop:
        if DEVICE_NAME.fullmatch(name) is None or not (source / name).exists():
            raise ProbeError(f"--drop names no device of {source}: {name!r}")
    devices: dict[str, Any] = {}
    for device in sorted(source.iterdir()):
        if DEVICE_NAME.fullmatch(device.name) is None or device.name in arguments.drop:
            continue
        devices[device.name] = _copy_rdma_device(device, root / device.name, drivers)
    if not devices and not arguments.allow_empty:
        raise ProbeError(
            f"{source} lists no RDMA device; pass --allow-empty to build an empty tree"
        )
    for item in arguments.duplicate:
        new, origin = _split(item, "=", 2, "--duplicate")
        if DEVICE_NAME.fullmatch(new) is None or new in devices:
            raise ProbeError(f"duplicate target {new!r} is unsafe or already present")
        shutil.copytree(_device_dir(root, origin), root / new, symlinks=True)
        devices[new] = {**devices[origin], "duplicate_of": origin}
    if len(devices) > MAX_TREE_DEVICES:
        raise ProbeError(f"the fixture tree holds more than {MAX_TREE_DEVICES} devices")
    mutations: list[str] = []
    for flag, name in (("port_state", "state"), ("phys_state", "phys_state")):
        for item in getattr(arguments, flag):
            device, port, text = _split(item, ":", 3, f"--{flag.replace('_', '-')}")
            if PORT_STATE.fullmatch(text) is None:
                raise ProbeError(f"port state text {text!r} is not allow-listed")
            path = _port_dir(root, device, port) / name
            path.write_text(text + "\n", encoding="utf-8")
            mutations.append(f"{device}/ports/{port}/{name}")
    for item in arguments.driver:
        device, mode = _split(item, ":", 2, "--driver")
        _set_driver(_device_dir(root, device), drivers, mode)
        mutations.append(f"{device}/device/driver")
    for item in arguments.set:
        target, value = _split(item, "=", 2, "--set")
        device, port, counter = _split(target, ":", 3, "--set")
        group, _, name = counter.partition("/")
        if (
            group not in ("counters", "hw_counters")
            or COUNTER_NAME.fullmatch(name) is None
        ):
            raise ProbeError(f"counter {counter!r} is not allow-listed")
        if COUNTER_VALUE.fullmatch(value) is None:
            raise ProbeError(f"counter value {value!r} is not allow-listed")
        path = _port_dir(root, device, port) / group / name
        path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        path.write_text(value + "\n", encoding="utf-8")
        mutations.append(f"{device}/ports/{port}/{group}/{name}")
    for item in arguments.remove_counter:
        device, port, counter = _split(item, ":", 3, "--remove-counter")
        group, _, name = counter.partition("/")
        if (
            group not in ("counters", "hw_counters")
            or COUNTER_NAME.fullmatch(name) is None
        ):
            raise ProbeError(f"counter {counter!r} is not allow-listed")
        path = _port_dir(root, device, port) / group / name
        if not path.is_file():
            raise ProbeError(f"{device}:{port}:{counter} is not in the tree")
        path.unlink()
        mutations.append(f"{device}/ports/{port}/{group}/{name} (removed)")
    for device in arguments.no_ports:
        ports = _device_dir(root, device) / "ports"
        shutil.rmtree(ports)
        ports.mkdir(mode=0o755)
        mutations.append(f"{device}/ports (emptied)")
    for item in arguments.netdev:
        # 010: the host collector runs ``ethtool -S`` only on the netdevs an
        # EFA function owns (``device/net/*``); on a node whose EFA functions
        # own none, a real physical netdev of the node is attached to one so
        # the ethtool reading can be replayed. The netdev itself is untouched.
        device, iface = _split(item, ":", 2, "--netdev")
        if (
            NETDEV_NAME.fullmatch(iface) is None
            or not (Path(arguments.netdev_root) / iface / "device").exists()
        ):
            raise ProbeError(
                f"--netdev names no physical netdev of {arguments.netdev_root}: {iface!r}"
            )
        (_device_dir(root, device) / "device" / "net" / iface).mkdir(
            mode=0o755, parents=True, exist_ok=True
        )
        devices[device]["netdevs"] = sorted({*devices[device]["netdevs"], iface})
        mutations.append(f"{device}/device/net/{iface} (attached)")
    summary = {
        "run_id": run_id,
        "path": str(root),
        "source": str(source),
        "devices": devices,
        "dropped": list(arguments.drop),
        "mutations": mutations,
    }
    (fixtures_dir(run_id) / "infiniband.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    emit(summary)


def build_nvidia_proc_fixture(arguments: argparse.Namespace) -> None:
    """A copy of /proc/driver/nvidia (version + gpus/<bdf>) with reviewed gaps."""

    run_id = safe_id(arguments.run_id)
    source = Path(arguments.source)
    root = fixtures_dir(run_id) / "nvidia-proc"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(mode=0o755, parents=True, exist_ok=True)
    version: str | None = None
    if arguments.version_text is not None:
        if len(arguments.version_text.encode("utf-8")) > 1024:
            raise ProbeError("version text exceeds 1 KiB")
        version = arguments.version_text + "\n"
    elif not arguments.omit_version:
        try:
            version = (source / "version").read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise ProbeError(f"{source}/version is unreadable: {error}") from error
    if version is not None:
        (root / "version").write_text(version, encoding="utf-8")
    gpus: list[str] = []
    if not arguments.omit_gpus:
        gpus_dir = root / "gpus"
        gpus_dir.mkdir(mode=0o755)
        try:
            entries = sorted(
                path.name
                for path in (source / "gpus").iterdir()
                if path.is_dir() and BDF.fullmatch(path.name)
            )
        except OSError as error:
            raise ProbeError(f"{source}/gpus is unreadable: {error}") from error
        if arguments.gpu_count is not None:
            if not 0 <= int(arguments.gpu_count) <= len(entries):
                raise ProbeError("gpu-count is outside 0..<real GPU count>")
            entries = entries[: int(arguments.gpu_count)]
        for name in entries:
            (gpus_dir / name).mkdir(mode=0o755)
            _copy_text(
                source / "gpus" / name / "information", gpus_dir / name / "information"
            )
            gpus.append(name)
    emit(
        {
            "run_id": run_id,
            "path": str(root),
            "source": str(source),
            "version": version.strip()[:120] if version is not None else None,
            "gpus": gpus,
            "omit_gpus": bool(arguments.omit_gpus),
        }
    )


def _meminfo_kb(values: dict[str, float], key: str) -> float:
    return float(values.get(key, 0.0))


def meminfo_percentages(values: dict[str, float]) -> dict[str, float | None]:
    """The two 002 rule inputs as the host collector computes them (percent)."""

    total = _meminfo_kb(values, "MemTotal")
    swap_total = _meminfo_kb(values, "SwapTotal")
    available = values.get("MemAvailable", values.get("MemFree", 0.0))
    return {
        "memory_used_percent": (
            100.0 * (total - float(available)) / total if total else None
        ),
        "swap_used_percent": (
            100.0 * (swap_total - _meminfo_kb(values, "SwapFree")) / swap_total
            if swap_total
            else 0.0
        ),
    }


def build_meminfo_fixture(arguments: argparse.Namespace) -> None:
    """A copy of ``/proc/meminfo`` with reviewed memory/swap fields rewritten.

    Every other line is the node's real reading at build time, so the memory,
    page-cache and available-memory rules keep seeing the host as it is; only
    ``--set FIELD=KB`` for ``MEMINFO_FIELDS`` changes. ``SwapTotal`` may be set
    on a swapless node: the product then grades ``swap_used_percent`` from the
    replayed pair exactly as it would from a real swap device.
    """

    run_id = safe_id(arguments.run_id)
    source = Path(arguments.source)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProbeError(f"{source} is unreadable: {error}") from error
    overrides: dict[str, int] = {}
    for item in arguments.set:
        key, _, raw = item.partition("=")
        if key not in MEMINFO_FIELDS:
            raise ProbeError(f"meminfo field {key!r} is not allow-listed")
        if not raw.isdigit() or int(raw) > MAX_MEMINFO_KB:
            raise ProbeError(f"meminfo value for {key} must be an integer in kB")
        overrides[key] = int(raw)
    if not overrides:
        raise ProbeError("a meminfo fixture needs at least one --set FIELD=KB")
    rewritten: list[str] = []
    values: dict[str, float] = {}
    seen: set[str] = set()
    for line in lines:
        key, separator, rest = line.partition(":")
        if not separator or not rest.strip():
            rewritten.append(line)
            continue
        key = key.strip()
        if key in overrides:
            line = f"{key}:{overrides[key]:>15} kB"
        seen.add(key)
        try:
            values[key] = float(line.split(":", 1)[1].strip().split()[0])
        except (IndexError, ValueError):
            pass
        rewritten.append(line)
    for key in MEMINFO_FIELDS:
        if key in overrides and key not in seen:
            # A swapless kernel still prints SwapTotal/SwapFree as 0, but a
            # field the source lacks is appended rather than silently dropped.
            rewritten.append(f"{key}:{overrides[key]:>15} kB")
            values[key] = float(overrides[key])
    directory = fixtures_dir(run_id)
    directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    target = directory / MEMINFO_FIXTURE
    if target.is_dir():
        raise ProbeError("a fixture tree of that name exists")
    write_atomic(target, ("\n".join(rewritten) + "\n").encode("utf-8"), 0o644)
    emit(
        {
            "run_id": run_id,
            "name": MEMINFO_FIXTURE,
            "path": str(target),
            "source": str(source),
            "overrides_kb": overrides,
            "values_kb": {key: values[key] for key in sorted(values)},
            **meminfo_percentages(values),
        }
    )
