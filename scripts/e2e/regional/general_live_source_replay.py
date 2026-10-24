"""Runner-side controller for the host-collector file-source replay.

Owns the host probe (``probes/general_live_source_probe.py``) that replaces one
allow-listed procfs/sysfs reading in the host collector unit's private mount
namespace (001 invalid_delta, 008 statistics_failed, 009 errors_delta) or
repoints its PCI devices root (013). The API mirrors the kernel-log replay: a
small install / open / (step) / read-back / cleanup surface, with the window on
the variant's LIFO cleanup stack so a window that will not close clean fails the
variant. ``proof_scope`` is always ``live-replay``.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture, HostProbeSettings

if TYPE_CHECKING:
    from scripts.e2e.regional.general_live_fixture import VariantContext

PROBE_SCRIPT = Path(__file__).with_name("probes") / "general_live_source_probe.py"
PROOF_SCOPE = "live-replay"


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
    ) -> None:
        self.run_id = run_id
        self._opened = False
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

    def create(self) -> None:
        if not getattr(self.host, "created", False):
            require_action_time(180)
            self.host.create()

    def install(self, name: str, content: bytes) -> dict[str, Any]:
        self.create()
        return self.host.execute(
            "install-source",
            "--run-id",
            self.run_id,
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
        arguments = ["build-pci-fixture", "--run-id", self.run_id, "--efa-count", str(int(efa_count))]
        if malformed:
            arguments.append("--malformed")
        return self.host.execute(*arguments, timeout=120)

    def open_bind(self, binds: list[tuple[str, str]], *, restore_seconds: int = 900) -> dict[str, Any]:
        self.create()
        require_action_time(180)
        arguments = ["open-window", "--run-id", self.run_id, "--mode", "bind",
                     "--restore-seconds", str(int(restore_seconds))]
        for name, dest in binds:
            arguments += ["--bind", f"{name}:{dest}"]
        receipt = self.host.execute(*arguments, timeout=150)
        self._opened = True
        return receipt

    def open_pci_root(self, *, restore_seconds: int = 900) -> dict[str, Any]:
        self.create()
        require_action_time(180)
        receipt = self.host.execute(
            "open-window", "--run-id", self.run_id, "--mode", "pci-root",
            "--restore-seconds", str(int(restore_seconds)), timeout=150,
        )
        self._opened = True
        return receipt

    def set_value(self, name: str, value: int | str) -> dict[str, Any]:
        require_action_time(60)
        return self.host.execute(
            "set-value", "--run-id", self.run_id, "--name", name, "--value", str(value), timeout=60
        )

    def status(self) -> dict[str, Any]:
        return self.host.execute("status", "--run-id", self.run_id, timeout=60)

    def close(self) -> dict[str, Any]:
        receipt = self.host.execute("close-window", "--run-id", self.run_id, timeout=150)
        self._opened = False
        return receipt

    def cleanup(self) -> dict[str, Any]:
        receipts: dict[str, Any] = {}
        if self._opened:
            receipts["close"] = self.close()
            if not receipts["close"].get("clean", True):
                raise RuntimeError(f"host source window did not close clean: {receipts['close']}")
        receipts["probe_residuals"] = self.host.cleanup()
        if any(receipts["probe_residuals"].values()):
            raise RuntimeError(f"host source probe left residue: {receipts['probe_residuals']}")
        return receipts


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
    )
    ctx.cleanup.push("source-replay", replay.cleanup)
    return replay


__all__ = ["HostSourceReplay", "PROOF_SCOPE", "host_source_replay"]
