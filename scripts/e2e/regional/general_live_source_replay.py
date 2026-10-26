"""Runner-side controller for the host-collector source replay (shared fixture).

Owns the host probe (``probes/general_live_source_probe.py``; its fixture
builders ``probes/general_live_source_builders.py`` are installed sha256-checked
through ``install-source`` before the first build, like the command wrapper)
that replaces allow-listed raw readings in the host collector unit's private
environment and nothing else: a procfs/sysfs *file* bound read-only into the unit's mount
namespace (001 invalid_delta, 008 statistics_failed, 009 errors_delta, 002
meminfo -- ``build_meminfo`` copies the real file with reviewed swap/available
fields --, 010/011 EFA counters, 032 driver version), a whole reviewed *tree*
(``/sys/class/infiniband`` for the 013 inventory variants, ``/proc/driver/
nvidia`` for 012/032), a reviewed pass-through *command wrapper* first on the
unit's PATH (``tools/host_command_replay.py``: smartctl 007, ipmitool 015,
ethtool 010, the host collector's nvidia-smi queries 012), a procfs file or
tool binary made *inaccessible* inside the unit (the probe's unopenable node
bound over it: the unit runs as root, which reads an ``InaccessiblePaths=``
node as an empty file -- 002 read_failure, 007 tool_missing), an allow-listed
expected-count *environment* override, or the 013 PCI-root repoint. The API mirrors the kernel-log replay: a
small install / build / open / (step) / read-back / cleanup surface, with the
window on the variant's LIFO cleanup stack so a window that will not close
clean fails the variant. ``proof_scope`` is always ``live-replay``.

Everything beyond the phase-1 binds (``/proc/stat``, ``gfrs*`` statistics) and
the PCI-root repoint needs the ``HOST_SOURCE_REPLAY`` approval on the plan;
``host_source_replay(ctx)`` carries the variant's approvals in and refuses the
extended modes without it.

The same controller opens a *command* window on the kernel collector unit
(``kernel_collector_source_replay(ctx)``, ``unit=KERNEL_UNIT``) under its own
approval, ``KERNEL_COLLECTOR_SOURCE_REPLAY``: only the ``nvidia-smi``
``driver_version`` query the unit's ``CollectorContext`` runs at start may be
answered there (``tools/host_command_replay.py`` ``UNIT_QUERIES``), the probe
unsets ``GPU_FAULT_DRIVER_BRANCH`` in that unit's environment for the window,
and every file, tree, inaccessible-path, count-override and PCI-root mode is
refused for it. ``HOST_SOURCE_REPLAY`` never unlocks the kernel collector and
``KERNEL_COLLECTOR_SOURCE_REPLAY`` never unlocks the host collector's extended
modes (GF-GENERAL-LIVE-032 ``missing_version``).

Plans for the command wrapper are built here and validated against the
wrapper's own per-tool, per-field allow-list before anything is sent::

    plan = command_plan("smartctl", [
        rule("health", "set_json", device="/dev/nvme0", field="smart_status.passed", value=False),
    ])
    replay.install_wrapper("smartctl", plan)
    replay.open_command(["smartctl"], restore_seconds=ctx.spec.max_minutes * 60)
    ...
    replay.audit("smartctl")   # what the product actually asked and was served
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture, HostProbeSettings
from scripts.e2e.regional.probes import general_live_source_probe as probe
from tools import host_command_replay as wrapper

if TYPE_CHECKING:
    from scripts.e2e.regional.general_live_fixture import VariantContext

PROBE_SCRIPT = Path(__file__).with_name("probes") / "general_live_source_probe.py"
WRAPPER_TOOL = Path(__file__).resolve().parents[3] / "tools" / "host_command_replay.py"
#: The probe's fixture builders, shipped like the wrapper (``install-source``).
BUILDERS_MODULE = Path(__file__).with_name("probes") / "general_live_source_builders.py"
PROOF_SCOPE = "live-replay"
#: The ``--approve`` token the extended modes need (``general_live_registry.APPROVALS``).
HOST_SOURCE_REPLAY = "HOST_SOURCE_REPLAY"
#: The token a command window on the kernel collector unit needs instead.
KERNEL_COLLECTOR_SOURCE_REPLAY = "KERNEL_COLLECTOR_SOURCE_REPLAY"
HOST_UNIT = probe.HOST_UNIT
KERNEL_UNIT = probe.KERNEL_UNIT
#: The one token that unlocks each allow-listed unit.
UNIT_TOKENS = {
    HOST_UNIT: HOST_SOURCE_REPLAY,
    KERNEL_UNIT: KERNEL_COLLECTOR_SOURCE_REPLAY,
}
#: Bind destinations the phase-1 variants used before the token existed.
LEGACY_BIND_DESTS = ("/proc/stat",)
#: Fixture tree names the builders produce (``--tree <name>:<dest>``).
INFINIBAND_TREE = "infiniband"
NVIDIA_PROC_TREE = "nvidia-proc"
#: The fixture file ``build_meminfo`` writes (``--bind meminfo:/proc/meminfo``).
MEMINFO_FIXTURE = probe.MEMINFO_FIXTURE
MEMINFO_FIELDS = probe.MEMINFO_FIELDS
TREE_DESTS = probe.TREE_DESTS
TOOLS = tuple(wrapper.TOOLS)
DEFAULT_PLAN_SECONDS = 3600


#: A bounded exposure (010/013 ``link_inactive``) ends when the collector has
#: read the replaced source on the ticks the product needs to confirm it
#: (``GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES``, 2 by default) and
#: before it reads it once more: ``observed_exposure`` lets the probe watch the
#: unit's own ticks on the node (``await-ticks``: the ``WATCHDOG=1`` stamps the
#: collector leaves on its unit) and close right after the required count.
#: The window's deadman is only the fallback for a runner that never gets
#: there, sized by ``exposure_fallback_seconds`` so it never fires first: the
#: collector's restart phase (``GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS`` plus
#: ``GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS``, 15 + 20 s by default -- a
#: fixed 1.7-interval deadman measured from the restart closed the live 013
#: window 2026-10-26 after one tick, 13 s in), the required ticks each with
#: their own duration, and the probe's close.
STARTUP_SPREAD_ENV = "GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS"
STARTUP_JITTER_ENV = "GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS"
CONSECUTIVE_SAMPLES_ENV = "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES"
TICK_DURATION_MARGIN_SECONDS = 10
CLOSE_MARGIN_SECONDS = 20


class HostSourceReplayError(RuntimeError):
    """A refused or malformed replay request (never a cluster error)."""


def rule(query: str, action: str, **fields: Any) -> dict[str, Any]:
    """One wrapper rule: ``{"query", "action", **fields}`` (validated in ``command_plan``)."""

    return {"query": query, "action": action, **fields}


def command_plan(
    tool: str,
    rules: Sequence[Mapping[str, Any]],
    *,
    not_after_seconds: int = DEFAULT_PLAN_SECONDS,
    now: float | None = None,
) -> dict[str, Any]:
    """A validated wrapper plan for ``tool`` that passes through after the deadline."""

    if not 60 <= int(not_after_seconds) <= wrapper.MAX_PLAN_SECONDS:
        raise HostSourceReplayError(
            f"not_after_seconds is outside 60..{wrapper.MAX_PLAN_SECONDS}"
        )
    deadline = (time.time() if now is None else now) + int(not_after_seconds)
    try:
        return wrapper.build_plan(
            tool, [dict(item) for item in rules], not_after=deadline
        )
    except wrapper.PlanError as error:
        raise HostSourceReplayError(f"{tool} plan rejected: {error}") from error


def confirmation_ticks(ctx: VariantContext) -> int:
    """The consecutive mismatch samples the host collector needs before it mints
    an inventory finding (``collectors/host/inventory.py``), from the survey."""

    return max(1, min(ctx.setting(CONSECUTIVE_SAMPLES_ENV, 2), probe.MAX_TICKS))


def exposure_fallback_seconds(ctx: VariantContext, ticks: int) -> int:
    """The deadman of a bounded-exposure window: after the collector's restart
    phase and ``ticks`` ticks, with margins, so the probe's own close comes first."""

    seconds = (
        ctx.setting(STARTUP_SPREAD_ENV, 15)
        + ctx.setting(STARTUP_JITTER_ENV, 20)
        + ticks * (ctx.host_interval + TICK_DURATION_MARGIN_SECONDS)
        + CLOSE_MARGIN_SECONDS
    )
    return max(probe.MIN_RESTORE_SECONDS, min(probe.MAX_RESTORE_SECONDS, int(seconds)))


def _moment(value: Any) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def observed_exposure(
    ctx: VariantContext,
    replay: HostSourceReplay,
    *,
    ticks: int,
    name: str,
    opened: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Close the open window right after the collector's ``ticks`` ticks, as the
    probe observes them on the node, and record the exposure that really was.

    Blocks for the probe's wait (bounded by the window's deadman). The record
    names when each observed tick ended, when the real source was read again
    (``closed_at``), who closed the window (``probe``, or ``deadman`` when the
    fallback fired first) and whether the ticks were seen at all
    (``completed``); the probe's read-back of the real configuration rides
    along as ``restore``. The caller judges the product's records against
    ``closed_at`` and the tick count, never against a fixed bound. When the
    deadman closed first, ``closed_at`` is its schedule read off the open
    receipt (``opened``: ``opened_at`` + ``restore_seconds``).
    """

    fallback = exposure_fallback_seconds(ctx, ticks)
    receipt = replay.await_ticks(ticks, interval=ctx.host_interval, timeout=fallback)
    close = receipt.get("close") or {}
    closed_at = _moment(close.get("closed_at"))
    if closed_at is None and receipt.get("already_closed") and opened is not None:
        opened_at = _moment(opened.get("opened_at"))
        if opened_at is not None:
            closed_at = opened_at + timedelta(
                seconds=int(opened.get("restore_seconds") or fallback)
            )
    closed_at = closed_at or datetime.now(timezone.utc)
    observed = list(receipt.get("ticks") or [])
    record = {
        "ticks_required": ticks,
        "ticks_observed": len(observed),
        "ticks": observed,
        "completed": bool(receipt.get("completed")),
        "closed_by": "deadman" if receipt.get("already_closed") else "probe",
        "closed_at": closed_at.isoformat(),
        "fallback_seconds": fallback,
        "restore": close.get("restore"),
        "clean": close.get("clean", receipt.get("already_closed", False)),
    }
    ctx.evidence.write(f"{name}-exposure", {**record, "receipt": receipt})
    if not record["clean"]:
        raise HostSourceReplayError(f"the exposure window did not close clean: {close}")
    return record


def legacy_bind(dest: str) -> bool:
    """Whether ``dest`` is a phase-1 bind that needs no HOST_SOURCE_REPLAY token."""

    return (
        dest in LEGACY_BIND_DESTS or probe.TEST_IFACE_STAT.fullmatch(dest) is not None
    )


class HostSourceReplay:
    """One reviewed host-collector source window for one node."""

    def __init__(
        self,
        regional: Any,
        *,
        node: str,
        image: str,
        case_id: str,
        run_id: str,
        case_dir: Path,
        active_deadline_seconds: int = 3600,
        host: Any | None = None,
        approvals: Iterable[str] = (),
        unit: str = HOST_UNIT,
    ) -> None:
        self.run_id = run_id
        self.approvals = frozenset(approvals)
        try:
            self.unit = wrapper.checked_unit(unit)
        except wrapper.PlanError as error:
            raise HostSourceReplayError(str(error)) from error
        self._opened = False
        self._wrapper_installed = False
        self._builders_installed = False
        self.host = host or HostProbeFixture(
            HostProbeSettings(
                kubeconfig=regional.settings.gpu_kubeconfig,
                context=regional.settings.gpu_context,
                namespace=regional.settings.namespace,
                node=node,
                image=image,
                case_id=case_id,
                run_id=run_id,
                probe_script=PROBE_SCRIPT,
                state_directory=case_dir / "host-probes",
                active_deadline_seconds=active_deadline_seconds,
            )
        )

    # ------------------------------------------------------------------ guards
    @property
    def token(self) -> str:
        """The ``--approve`` token this controller's unit needs."""

        return UNIT_TOKENS[self.unit]

    @property
    def approved(self) -> bool:
        return self.token in self.approvals

    def _require_approval(self, what: str) -> None:
        if not self.approved:
            raise HostSourceReplayError(
                f"{what} needs the {self.token} approval on the plan"
            )

    def _require_host_unit(self, what: str) -> None:
        if self.unit != HOST_UNIT:
            raise HostSourceReplayError(
                f"{what} is a host collector mode; {self.unit} carries a "
                "command-output replay only"
            )

    def create(self) -> None:
        if not getattr(self.host, "created", False):
            require_action_time(180)
            self.host.create()

    def _execute(self, *arguments: str, timeout: int) -> dict[str, Any]:
        return self.host.execute(*arguments, "--run-id", self.run_id, timeout=timeout)

    # ---------------------------------------------------------------- fixtures
    def install(self, name: str, content: bytes) -> dict[str, Any]:
        """Install one fixture file (sha256-checked) under the run's fixtures."""

        self.create()
        return self._execute(
            "install-source",
            "--name",
            name,
            "--sha256",
            hashlib.sha256(content).hexdigest(),
            "--b64",
            base64.b64encode(content).decode("ascii"),
            timeout=120,
        )

    def _install_builders(self) -> None:
        """Ship the sha256-checked fixture builders once per run, before a build.

        A probe is one self-contained file on the node, so the builders travel
        as a second installed file that the probe imports by path; the window,
        deadman and cleanup commands never need it.
        """

        if not self._builders_installed:
            self.install(probe.BUILDERS_NAME, BUILDERS_MODULE.read_bytes())
            self._builders_installed = True

    def build_pci(
        self,
        *,
        efa_count: int | None = None,
        malformed: bool = False,
        source: str | None = None,
        drop: Iterable[str] = (),
        unreadable: bool = False,
    ) -> dict[str, Any]:
        """A reviewed PCI root for ``open_pci_root`` (``build-pci-fixture``).

        One shape per call: ``efa_count`` synthetic EFA functions (``malformed``
        gives one a bad device id; 013 ``unreadable_state``), a copy of the
        node's real ``source`` root with the ``drop`` BDFs absent (013
        ``pci_missing``), or ``unreadable`` -- a regular file at the fixture
        path, so every listing by the product fails (ENOTDIR) and the reading
        must be UNKNOWN, never zero functions. The PCI-root repoint is a
        phase-1 mode and needs no approval; the receipt names the EFA
        functions kept and dropped.
        """

        self._require_host_unit("a PCI-root replay")
        if sum((efa_count is not None, source is not None, unreadable)) != 1:
            raise HostSourceReplayError(
                "build_pci takes exactly one of efa_count, source, unreadable"
            )
        self._install_builders()
        arguments = ["build-pci-fixture"]
        if efa_count is not None:
            arguments += ["--efa-count", str(int(efa_count))]
        if malformed:
            arguments.append("--malformed")
        if source is not None:
            arguments += ["--source", source]
        for bdf in drop:
            arguments += ["--drop", bdf]
        if unreadable:
            arguments.append("--unreadable")
        return self._execute(*arguments, timeout=120)

    def build_infiniband(
        self,
        *,
        drop: Iterable[str] = (),
        duplicate: Iterable[tuple[str, str]] = (),
        port_state: Iterable[tuple[str, str, str]] = (),
        phys_state: Iterable[tuple[str, str, str]] = (),
        driver: Iterable[tuple[str, str]] = (),
        counters: Iterable[tuple[str, str, str, str]] = (),
        remove_counters: Iterable[tuple[str, str, str]] = (),
        no_ports: Iterable[str] = (),
        netdev: Iterable[tuple[str, str]] = (),
        source: str = "/sys/class/infiniband",
    ) -> dict[str, Any]:
        """Copy the node's RDMA tree with reviewed mutations (``--tree infiniband:...``).

        ``duplicate=[(new, origin)]``, ``port_state=[(dev, port, "1: DOWN")]``,
        ``phys_state=[(dev, port, "3: Disabled")]``, ``driver=[(dev, "none" |
        "ambiguous" | <name>)]``, ``counters=[(dev, port, "hw_counters/rx_drops",
        "5000")]``, ``remove_counters=[(dev, port, "hw_counters/tx_pkts")]``,
        ``netdev=[(dev, <physical netdev>)]`` (010: an EFA function that owns
        no netdev is given one so ``ethtool -S`` is run for it).
        """

        self._require_host_unit("an infiniband tree replay")
        self._require_approval("an infiniband tree replay")
        self._install_builders()
        arguments = ["build-infiniband-fixture", "--source", source]
        for device in drop:
            arguments += ["--drop", device]
        for new, origin in duplicate:
            arguments += ["--duplicate", f"{new}={origin}"]
        for device, port, text in port_state:
            arguments += ["--port-state", f"{device}:{port}:{text}"]
        for device, port, text in phys_state:
            arguments += ["--phys-state", f"{device}:{port}:{text}"]
        for device, mode in driver:
            arguments += ["--driver", f"{device}:{mode}"]
        for device, port, counter, value in counters:
            arguments += ["--set", f"{device}:{port}:{counter}={value}"]
        for device, port, counter in remove_counters:
            arguments += ["--remove-counter", f"{device}:{port}:{counter}"]
        for device in no_ports:
            arguments += ["--no-ports", device]
        for device, interface in netdev:
            arguments += ["--netdev", f"{device}:{interface}"]
        return self._execute(*arguments, timeout=180)

    # ---------------------------------------------------------------- read-only
    def inventory(self, *, source: str = "/sys/class/infiniband") -> dict[str, Any]:
        """The node's RDMA tree as the product reads it (no window, no approval)."""

        self._install_builders()
        return self._execute("inventory", "--source", source, timeout=120)

    def read_back(self, dest: str) -> dict[str, Any]:
        """``dest`` as the unit reads it next to PID 1's view (``same`` when equal).

        After ``close`` this is the proof that the real source is read again.
        """

        self._require_host_unit("a source file read-back")
        try:
            probe.checked_source_path(dest)
        except probe.ProbeError as error:
            raise HostSourceReplayError(str(error)) from error
        self.create()
        return self._execute("read-source", "--path", dest, timeout=60)

    def journal(self, *, since: Any, tail: int = 200) -> dict[str, Any]:
        """The unit's own journal since ``since`` (a datetime): bounded, read-only."""

        self._install_builders()
        moment = since.isoformat() if hasattr(since, "isoformat") else str(since)
        return self._execute(
            "journal-read",
            "--unit",
            self.unit,
            "--since",
            moment,
            "--tail",
            str(int(tail)),
            timeout=60,
        )

    def build_nvidia_proc(
        self,
        *,
        omit_version: bool = False,
        version_text: str | None = None,
        gpu_count: int | None = None,
        omit_gpus: bool = False,
        source: str = "/proc/driver/nvidia",
    ) -> dict[str, Any]:
        """Copy ``/proc/driver/nvidia`` with reviewed gaps (``--tree nvidia-proc:...``)."""

        self._require_host_unit("a /proc/driver/nvidia tree replay")
        self._require_approval("a /proc/driver/nvidia tree replay")
        self._install_builders()
        arguments = ["build-nvidia-proc-fixture", "--source", source]
        if omit_version:
            arguments.append("--omit-version")
        if version_text is not None:
            arguments += ["--version-text", version_text]
        if gpu_count is not None:
            arguments += ["--gpu-count", str(int(gpu_count))]
        if omit_gpus:
            arguments.append("--omit-gpus")
        return self._execute(*arguments, timeout=120)

    def build_meminfo(
        self, *, source: str = "/proc/meminfo", **fields_kb: int
    ) -> dict[str, Any]:
        """Copy the node's ``/proc/meminfo`` with reviewed fields rewritten (kB).

        ``fields_kb`` names ``MEMINFO_FIELDS`` only (``SwapTotal=8388608,
        SwapFree=1048576`` for 002 ``swap_pressure``); the receipt carries the
        resulting ``swap_used_percent`` / ``memory_used_percent`` as the oracle
        of what the product's ``_memory`` contributor will compute.
        """

        self._require_host_unit("a /proc/meminfo replay")
        self._require_approval("a /proc/meminfo replay")
        unknown = sorted(set(fields_kb) - set(MEMINFO_FIELDS))
        if unknown or not fields_kb:
            raise HostSourceReplayError(
                f"meminfo fields must be one or more of {list(MEMINFO_FIELDS)}; "
                f"got {sorted(fields_kb)}"
            )
        self._install_builders()
        arguments = ["build-meminfo-fixture", "--source", source]
        for name in MEMINFO_FIELDS:
            if name in fields_kb:
                arguments += ["--set", f"{name}={int(fields_kb[name])}"]
        return self._execute(*arguments, timeout=120)

    # ---------------------------------------------------------------- wrappers
    def install_plan(self, tool: str, plan: Mapping[str, Any]) -> dict[str, Any]:
        """Install (or atomically replace) ``plan-<tool>.json``; validated first."""

        self._require_approval("a command-output replay")
        try:
            checked = wrapper.checked_unit_plan(self.unit, tool, dict(plan))
        except wrapper.PlanError as error:
            raise HostSourceReplayError(f"{tool} plan rejected: {error}") from error
        content = json.dumps(checked, sort_keys=True).encode("utf-8")
        return self.install(f"plan-{tool}.json", content)

    def _unit_arguments(self) -> list[str]:
        """``--unit`` for a window off the host collector (the probe's default)."""

        return [] if self.unit == HOST_UNIT else ["--unit", self.unit]

    def install_wrapper(
        self, tool: str, plan: Mapping[str, Any], *, allow_missing_real: bool = False
    ) -> dict[str, Any]:
        """Ship the reviewed wrapper (once) and ``tool``'s plan; write the shim.

        ``allow_missing_real`` is for a tool the host lacks (ipmitool on EC2):
        the shim then serves only the plan's whole-output rules and answers
        "command not found" to everything else.
        """

        self._require_approval("a command-output replay")
        if not self._wrapper_installed:
            self.install(probe.WRAPPER_NAME, WRAPPER_TOOL.read_bytes())
            self._wrapper_installed = True
        receipts = {"plan": self.install_plan(tool, plan)}
        arguments = ["install-wrapper", "--tool", tool, *self._unit_arguments()]
        if allow_missing_real:
            arguments.append("--allow-missing-real")
        receipts["wrapper"] = self._execute(*arguments, timeout=120)
        return receipts

    def replan(self, tool: str, plan: Mapping[str, Any]) -> dict[str, Any]:
        """Replace an open window's plan; the wrapper re-reads it on its next call."""

        return self.install_plan(tool, plan)

    def audit(self, tool: str, *, tail: int = 100) -> dict[str, Any]:
        """The wrapper's journal for ``tool``: what the product asked and was served."""

        self._install_builders()
        return self._execute(
            "audit-read", "--tool", tool, "--tail", str(int(tail)), timeout=60
        )

    # ------------------------------------------------------------------ window
    def open_window(
        self,
        *,
        binds: Sequence[tuple[str, str]] = (),
        trees: Sequence[tuple[str, str]] = (),
        wrappers: Sequence[str] = (),
        inaccessible: Sequence[str] = (),
        env: Mapping[str, str] | Sequence[tuple[str, str]] = (),
        pci_root: bool = False,
        restore_seconds: int = 900,
        mode: str | None = None,
        verify: bool = True,
    ) -> dict[str, Any]:
        """Open one window combining any of the modes; deadman first, then verify.

        ``binds``/``trees`` are ``(fixture name, destination)`` pairs,
        ``wrappers`` tool names already installed with ``install_wrapper``,
        ``inaccessible`` allow-listed procfs files or ``tool:<name>`` (made
        unopenable in the unit; never a directory -- replay a tree without the
        entry, ``build_nvidia_proc(omit_gpus=True)``), ``env`` the allow-listed
        expected-count overrides.
        """

        env_pairs = list(env.items()) if isinstance(env, Mapping) else list(env)
        if binds or trees or inaccessible or env_pairs or pci_root:
            self._require_host_unit(
                "a file, tree, inaccessible-path, override or PCI-root window"
            )
        extended = (
            any(not legacy_bind(dest) for _name, dest in binds)
            or bool(trees)
            or bool(wrappers)
            or bool(inaccessible)
            or bool(env_pairs)
        )
        if extended:
            self._require_approval("this source window")
        for _name, dest in binds:
            try:
                probe.checked_bind_dest(dest)
            except probe.ProbeError as error:
                raise HostSourceReplayError(str(error)) from error
        for _name, dest in trees:
            try:
                probe.checked_tree_dest(dest)
            except probe.ProbeError as error:
                raise HostSourceReplayError(str(error)) from error
        for tool in wrappers:
            try:
                wrapper.checked_tool(tool)
            except wrapper.PlanError as error:
                raise HostSourceReplayError(str(error)) from error
        for name, value in env_pairs:
            try:
                probe.checked_env_override(f"{name}={value}")
            except probe.ProbeError as error:
                raise HostSourceReplayError(str(error)) from error
        if mode is None:
            kinds = [
                kind
                for kind, present in (
                    ("bind", bool(binds)),
                    ("tree", bool(trees)),
                    ("command", bool(wrappers)),
                    ("pci-root", pci_root),
                )
                if present
            ]
            mode = (
                kinds[0]
                if len(kinds) == 1 and not (inaccessible or env_pairs)
                else "mixed"
            )
        self.create()
        require_action_time(180)
        arguments = [
            "open-window",
            "--mode",
            mode,
            "--restore-seconds",
            str(int(restore_seconds)),
            *self._unit_arguments(),
        ]
        for name, dest in binds:
            arguments += ["--bind", f"{name}:{dest}"]
        for name, dest in trees:
            arguments += ["--tree", f"{name}:{dest}"]
        for tool in wrappers:
            arguments += ["--wrapper", tool]
        for item in inaccessible:
            arguments += ["--inaccessible", item]
        for name, value in env_pairs:
            arguments += ["--env", f"{name}={value}"]
        if pci_root:
            arguments.append("--pci-root")
        if not verify:
            arguments.append("--skip-verify")
        receipt = self._execute(*arguments, timeout=240)
        self._opened = True
        return receipt

    def open_bind(
        self, binds: list[tuple[str, str]], *, restore_seconds: int = 900
    ) -> dict[str, Any]:
        return self.open_window(
            binds=binds, restore_seconds=restore_seconds, mode="bind"
        )

    def open_pci_root(self, *, restore_seconds: int = 900) -> dict[str, Any]:
        return self.open_window(
            pci_root=True, restore_seconds=restore_seconds, mode="pci-root"
        )

    def open_command(
        self, tools: Sequence[str], *, restore_seconds: int = 900, **extra: Any
    ) -> dict[str, Any]:
        """A PATH-wrapper window for ``tools`` (each installed with ``install_wrapper``)."""

        return self.open_window(
            wrappers=list(tools),
            restore_seconds=restore_seconds,
            mode="command",
            **extra,
        )

    def open_tree(
        self,
        trees: Sequence[tuple[str, str]],
        *,
        restore_seconds: int = 900,
        **extra: Any,
    ) -> dict[str, Any]:
        """A whole-tree bind window, e.g. ``[(INFINIBAND_TREE, "/sys/class/infiniband")]``."""

        return self.open_window(
            trees=list(trees), restore_seconds=restore_seconds, mode="tree", **extra
        )

    def set_value(self, name: str, value: int | str) -> dict[str, Any]:
        self._require_host_unit("a fixture value step")
        require_action_time(60)
        return self._execute(
            "set-value", "--name", name, "--value", str(value), timeout=60
        )

    def set_tree_value(self, name: str, path: str, value: int | str) -> dict[str, Any]:
        """Rewrite one file of an open fixture tree (``name`` = tree, ``path`` inside it)."""

        self._require_host_unit("a fixture tree step")
        require_action_time(60)
        return self._execute(
            "set-value",
            "--name",
            name,
            "--path",
            path,
            "--value",
            str(value),
            timeout=60,
        )

    def status(self) -> dict[str, Any]:
        self._install_builders()
        return self._execute("status", timeout=60)

    def verify(self) -> dict[str, Any]:
        """Re-check that every replacement is visible inside the unit and not to PID 1."""

        return self._execute("verify", timeout=120)

    def close(self) -> dict[str, Any]:
        receipt = self._execute("close-window", timeout=150)
        self._opened = False
        return receipt

    def await_ticks(
        self, count: int, *, interval: int, timeout: int, close: bool = True
    ) -> dict[str, Any]:
        """Block until the unit completed ``count`` collector ticks since the
        window opened (the probe watches ``WatchdogTimestampMonotonic`` on the
        node), then close the window in the same call (``close``).

        ``timeout`` is the longest the probe waits before it closes anyway (the
        probe caps it at the window's deadman); the exec round trip gets the
        close's own time on top. A window the deadman closed meanwhile answers
        ``already_closed``.
        """

        arguments = [
            "await-ticks",
            "--count",
            str(int(count)),
            "--interval",
            str(int(interval)),
            "--timeout",
            str(int(timeout)),
        ]
        if close:
            arguments.append("--close")
        receipt = self._execute(*arguments, timeout=int(timeout) + 120)
        if close and (
            receipt.get("close") is not None or receipt.get("already_closed")
        ):
            self._opened = False
        return receipt

    def snapshot(self) -> dict[str, Any]:
        """What of this run is on the node: window, drop-in, state dir, deadman,
        and the host collector unit's own PCI root (``unit_pci_root``)."""

        self.create()
        return self._execute("snapshot", timeout=60)

    def recreate(self) -> dict[str, Any]:
        """Replace the probe Pod after the node it ran on rebooted.

        A host probe is a plain Pod pinned to one node (``restartPolicy:
        Never``): a RESTART_NODE takes it down with the node and ``kubectl
        exec`` into the Failed Pod is refused. The reboot also wiped ``/run``
        -- the drop-in, the deadman's transient units, the fixture directory
        and the probe copy -- so there is no window left to close; the caller
        proves that with :meth:`snapshot` on the new boot. The host fixture's
        own cleanup re-establishes the scoped channel on the same node UID,
        removes what may be left and deletes the terminated Pod before a new
        one is created; the installed files are gone, so they are shipped
        again before the next build.
        """

        require_action_time(180)
        residuals = self.host.cleanup()
        self._opened = False
        self._wrapper_installed = False
        self._builders_installed = False
        self.host.create()
        return {"previous_residuals": residuals, "recreated": True}

    def cleanup(self) -> dict[str, Any]:
        receipts: dict[str, Any] = {}
        if self._opened:
            receipts["close"] = self.close()
            if not receipts["close"].get("clean", True):
                raise RuntimeError(
                    f"host source window did not close clean: {receipts['close']}"
                )
        receipts["probe_residuals"] = self.host.cleanup()
        if any(receipts["probe_residuals"].values()):
            raise RuntimeError(
                f"host source probe left residue: {receipts['probe_residuals']}"
            )
        return receipts

    def evidence(self, **extra: Any) -> dict[str, Any]:
        """The evidence header every replay variant should write."""

        return {
            "proof_scope": PROOF_SCOPE,
            "physical_fault_injected": False,
            "run_id": self.run_id,
            "unit": self.unit,
            "approved": self.approved,
            **extra,
        }


#: The probe handle key and cleanup name per unit (tests pre-register fakes
#: under the key; phase-1 variants keep the host collector's names).
UNIT_HANDLES = {
    HOST_UNIT: "source-replay",
    KERNEL_UNIT: "kernel-collector-replay",
}


def host_source_replay(
    ctx: VariantContext, *, unit: str = HOST_UNIT
) -> HostSourceReplay:
    """Build a ``HostSourceReplay`` from a variant context and register cleanup."""

    handle = UNIT_HANDLES.get(unit, "source-replay")
    host = ctx.probes.get(f"{handle}@{ctx.settings.node}")
    replay = HostSourceReplay(
        ctx.regional,
        node=ctx.settings.node,
        image=ctx.settings.host_probe_image,
        case_id=ctx.spec.key,
        run_id=ctx.run_id,
        case_dir=ctx.case_dir,
        active_deadline_seconds=ctx.spec.max_minutes * 60 + 1200,
        host=host,
        approvals=ctx.settings.approvals,
        unit=unit,
    )
    ctx.cleanup.push(handle, replay.cleanup)
    return replay


def kernel_collector_source_replay(ctx: VariantContext) -> HostSourceReplay:
    """The kernel collector's command-only window (``KERNEL_COLLECTOR_SOURCE_REPLAY``)."""

    return host_source_replay(ctx, unit=KERNEL_UNIT)


__all__ = [
    "CLOSE_MARGIN_SECONDS",
    "CONSECUTIVE_SAMPLES_ENV",
    "DEFAULT_PLAN_SECONDS",
    "HOST_SOURCE_REPLAY",
    "HOST_UNIT",
    "INFINIBAND_TREE",
    "KERNEL_COLLECTOR_SOURCE_REPLAY",
    "KERNEL_UNIT",
    "MEMINFO_FIELDS",
    "MEMINFO_FIXTURE",
    "NVIDIA_PROC_TREE",
    "PROOF_SCOPE",
    "TOOLS",
    "TREE_DESTS",
    "UNIT_HANDLES",
    "UNIT_TOKENS",
    "HostSourceReplay",
    "HostSourceReplayError",
    "STARTUP_JITTER_ENV",
    "STARTUP_SPREAD_ENV",
    "TICK_DURATION_MARGIN_SECONDS",
    "command_plan",
    "confirmation_ticks",
    "exposure_fallback_seconds",
    "host_source_replay",
    "kernel_collector_source_replay",
    "legacy_bind",
    "observed_exposure",
    "rule",
]
