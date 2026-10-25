"""Runner-side controller for the host-collector source replay (shared fixture).

Owns the host probe (``probes/general_live_source_probe.py``) that replaces
allow-listed raw readings in the host collector unit's private environment and
nothing else: a procfs/sysfs *file* bound read-only into the unit's mount
namespace (001 invalid_delta, 008 statistics_failed, 009 errors_delta, 002
meminfo, 010/011 EFA counters, 032 driver version), a whole reviewed *tree*
(``/sys/class/infiniband`` for the 013 inventory variants, ``/proc/driver/
nvidia`` for 012/032), a reviewed pass-through *command wrapper* first on the
unit's PATH (``tools/host_command_replay.py``: smartctl 007, ipmitool 015,
ethtool 010, the host collector's nvidia-smi queries 012), a path made
*inaccessible* inside the unit, an allow-listed expected-count *environment*
override, or the 013 PCI-root repoint. The API mirrors the kernel-log replay: a
small install / build / open / (step) / read-back / cleanup surface, with the
window on the variant's LIFO cleanup stack so a window that will not close
clean fails the variant. ``proof_scope`` is always ``live-replay``.

Everything beyond the phase-1 binds (``/proc/stat``, ``gfrs*`` statistics) and
the PCI-root repoint needs the ``HOST_SOURCE_REPLAY`` approval on the plan;
``host_source_replay(ctx)`` carries the variant's approvals in and refuses the
extended modes without it.

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
PROOF_SCOPE = "live-replay"
#: The ``--approve`` token the extended modes need (``general_live_registry.APPROVALS``).
HOST_SOURCE_REPLAY = "HOST_SOURCE_REPLAY"
#: Bind destinations the phase-1 variants used before the token existed.
LEGACY_BIND_DESTS = ("/proc/stat",)
#: Fixture tree names the builders produce (``--tree <name>:<dest>``).
INFINIBAND_TREE = "infiniband"
NVIDIA_PROC_TREE = "nvidia-proc"
TREE_DESTS = probe.TREE_DESTS
TOOLS = tuple(wrapper.TOOLS)
DEFAULT_PLAN_SECONDS = 3600


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
    ) -> None:
        self.run_id = run_id
        self.approvals = frozenset(approvals)
        self._opened = False
        self._wrapper_installed = False
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
    def approved(self) -> bool:
        return HOST_SOURCE_REPLAY in self.approvals

    def _require_approval(self, what: str) -> None:
        if not self.approved:
            raise HostSourceReplayError(
                f"{what} needs the {HOST_SOURCE_REPLAY} approval on the plan"
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

    def build_pci(self, *, efa_count: int, malformed: bool = False) -> dict[str, Any]:
        self.create()
        arguments = ["build-pci-fixture", "--efa-count", str(int(efa_count))]
        if malformed:
            arguments.append("--malformed")
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

        self._require_approval("an infiniband tree replay")
        self.create()
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

        self.create()
        return self._execute("inventory", "--source", source, timeout=120)

    def read_back(self, dest: str) -> dict[str, Any]:
        """``dest`` as the unit reads it next to PID 1's view (``same`` when equal).

        After ``close`` this is the proof that the real source is read again.
        """

        try:
            probe.checked_source_path(dest)
        except probe.ProbeError as error:
            raise HostSourceReplayError(str(error)) from error
        self.create()
        return self._execute("read-source", "--path", dest, timeout=60)

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

        self._require_approval("a /proc/driver/nvidia tree replay")
        self.create()
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

    # ---------------------------------------------------------------- wrappers
    def install_plan(self, tool: str, plan: Mapping[str, Any]) -> dict[str, Any]:
        """Install (or atomically replace) ``plan-<tool>.json``; validated first."""

        self._require_approval("a command-output replay")
        try:
            checked = wrapper.validate_plan(tool, dict(plan))
        except wrapper.PlanError as error:
            raise HostSourceReplayError(f"{tool} plan rejected: {error}") from error
        content = json.dumps(checked, sort_keys=True).encode("utf-8")
        return self.install(f"plan-{tool}.json", content)

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
        arguments = ["install-wrapper", "--tool", tool]
        if allow_missing_real:
            arguments.append("--allow-missing-real")
        receipts["wrapper"] = self._execute(*arguments, timeout=120)
        return receipts

    def replan(self, tool: str, plan: Mapping[str, Any]) -> dict[str, Any]:
        """Replace an open window's plan; the wrapper re-reads it on its next call."""

        return self.install_plan(tool, plan)

    def audit(self, tool: str, *, tail: int = 100) -> dict[str, Any]:
        """The wrapper's journal for ``tool``: what the product asked and was served."""

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
        ``inaccessible`` allow-listed paths or ``tool:<name>``, ``env`` the
        allow-listed expected-count overrides.
        """

        env_pairs = list(env.items()) if isinstance(env, Mapping) else list(env)
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
        require_action_time(60)
        return self._execute(
            "set-value", "--name", name, "--value", str(value), timeout=60
        )

    def set_tree_value(self, name: str, path: str, value: int | str) -> dict[str, Any]:
        """Rewrite one file of an open fixture tree (``name`` = tree, ``path`` inside it)."""

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
        return self._execute("status", timeout=60)

    def verify(self) -> dict[str, Any]:
        """Re-check that every replacement is visible inside the unit and not to PID 1."""

        return self._execute("verify", timeout=120)

    def close(self) -> dict[str, Any]:
        receipt = self._execute("close-window", timeout=150)
        self._opened = False
        return receipt

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
            "approved": self.approved,
            **extra,
        }


def host_source_replay(ctx: VariantContext) -> HostSourceReplay:
    """Build a ``HostSourceReplay`` from a variant context and register cleanup."""

    host = ctx.probes.get(f"source-replay@{ctx.settings.node}")
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
    )
    ctx.cleanup.push("source-replay", replay.cleanup)
    return replay


__all__ = [
    "DEFAULT_PLAN_SECONDS",
    "HOST_SOURCE_REPLAY",
    "INFINIBAND_TREE",
    "NVIDIA_PROC_TREE",
    "PROOF_SCOPE",
    "TOOLS",
    "TREE_DESTS",
    "HostSourceReplay",
    "HostSourceReplayError",
    "command_plan",
    "host_source_replay",
    "legacy_bind",
    "rule",
]
