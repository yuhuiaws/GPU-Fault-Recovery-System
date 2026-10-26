"""Reusable kernel-log source replay for GF-GENERAL-LIVE (and lane R later).

One small, documented controller -- ``KernelLogReplay`` -- installs a reviewed
set of kernel-transport records for *one node and one boot*, arms them, reads
back what the product ingested, and cleans up. The injection travels through the
product's own readers so only the raw reading is replaced and the parsing,
trust classification, dedup, delivery and grading stay real
(``docs/components/general-fault-live-acceptance.md`` ``live-replay``):

* ``mode="journal"`` repoints one reviewed pass-through wrapper onto the node
  log collector's ``journalctl`` (``gpu-fault-log-collector.service``), so the
  records reach ``collectors/logs/node.py`` ``_parse_journal_line`` and the
  ``log_rules``/``log_recovery_context`` trusted-source gate. A
  ``transport="kernel"`` record is the trusted ``dmesg`` origin; any other
  transport stays the untrusted ``journal`` origin and cannot earn a
  hardware-fatal (QUARANTINE) rule -- that is what keeps a reviewed user-space
  line from being graded as a trusted kernel fault.
* ``mode="kmsg"`` repoints the kernel collector's ``GPU_FAULT_KMSG_PATH``
  (``gpu-fault-kernel-collector.service``) at a reviewed fixture file, so the
  records reach ``collectors/logs/kernel.py`` and the real NVIDIA_KERNEL XID/SXID
  catalog. The real ``/dev/kmsg`` is never written.

API (all through the one host-probe Pod, ``probes/general_live_kernel_source_probe.py``):

    replay = kernel_replay(ctx, mode="journal")   # or construct KernelLogReplay
    replay.install(records)      # reviewed list[dict]; sha256+base64 to the node
    replay.open(approved=True)   # drop-in + deadman; journal mode needs approval
    replay.arm()                 # start emitting the reviewed records
    ingested = replay.read_back()  # the runner's oracle of what was injected
    ...                          # the caller reads the product's own records
    replay.cleanup()             # remove the wrapper/drop-in, restore, verify

A reviewed record is ``{"message", "transport"?, "priority"?, "emissions"?,
"boot_id"?, "unit"?, "label"?}``; ``transport`` defaults to ``kernel`` and
``emissions`` bounds how many polls re-post it (dedup collapses them to one
incident). ``reviewed_record`` builds one and ``records_sha256`` is the digest a
caller records in evidence. Nothing here posts a finding; the caller's verdict
reads the product's own records plus ``read_back`` as the raw-source oracle.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture, HostProbeSettings

if TYPE_CHECKING:
    from scripts.e2e.regional.general_live_fixture import VariantContext

PROBE_SCRIPT = (
    Path(__file__).with_name("probes") / "general_live_kernel_source_probe.py"
)
PROOF_SCOPE = "live-replay"
MODES = ("journal", "kmsg")


def reviewed_record(
    message: str,
    *,
    transport: str = "kernel",
    priority: int = 3,
    emissions: int = 4,
    boot_id: str = "",
    unit: str = "",
    label: str = "",
) -> dict[str, Any]:
    """One reviewed kernel-transport record for ``install``.

    ``transport="kernel"`` is the trusted dmesg origin; a non-kernel transport
    is the deliberately untrusted journal origin (the ``user_text`` case).
    """

    return {
        "message": message,
        "transport": transport,
        "priority": int(priority),
        "emissions": int(emissions),
        "boot_id": boot_id,
        "unit": unit,
        "label": label,
    }


def records_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_encode(records)).hexdigest()


def kmsg_follow_error(injected: Mapping[str, Any]) -> str | None:
    """Why the kmsg-mode oracle says the collector never read the fixture (or None).

    ``status`` in kmsg mode reports, beside ``fixture_lines``, whether the
    kernel collector holds the fixture open at the end of what ``arm`` appended
    (``consumed``) -- read from the collector's own descriptor and journal, not
    assumed. A collector that takes a regular file's EOF for an ended stream
    reopens it at its end every second and skips everything appended in
    between; live (release R13g) that left every kmsg-mode variant with a full
    fixture and no XID/SXID event in the store. A reviewed line that never left
    the node proves nothing about the product's grading, so the caller raises
    INCONCLUSIVE with this text instead of FAIL. A probe that does not report
    ``consumed`` at all is not judged here.
    """

    if injected.get("consumed") is not False:
        return None
    collector = dict(injected.get("collector") or {})
    return (
        "the kernel collector did not follow the reviewed kmsg fixture: stream "
        f"reopened at EOF {injected.get('stream_reopens')} time(s) in this invocation, "
        f"fixture open in the collector={collector.get('fixture_open')}, read offset "
        f"{collector.get('offset')} of {injected.get('fixture_bytes')} bytes; a kernel "
        "collector that reopens a regular GPU_FAULT_KMSG_PATH at its end instead of "
        "following it never reads what is appended, so the reviewed line never "
        "reached the control plane (needs the release whose collector follows the file)"
    )


def _encode(records: list[dict[str, Any]]) -> bytes:
    return json.dumps(records, sort_keys=True).encode("utf-8")


class KernelLogReplay:
    """Owns the host probe and the reviewed-source window for one node."""

    def __init__(
        self,
        regional: Any,
        *,
        node: str,
        image: str,
        case_id: str,
        run_id: str,
        case_dir: Path,
        mode: str = "journal",
        active_deadline_seconds: int = 3600,
        host: Any | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.run_id = run_id
        self.case_dir = case_dir
        self._opened = False
        self._armed = False
        self._installed: list[dict[str, Any]] = []
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

    # -- lifecycle ------------------------------------------------------- #
    def create(self) -> None:
        if not getattr(self.host, "created", False):
            require_action_time(180)
            self.host.create()

    def preflight(self) -> dict[str, Any]:
        return self.host.execute("preflight", timeout=120)

    def install(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Install the reviewed record set; sha256 is checked on the node."""

        self.create()
        self._installed = list(records)
        receipt = self.host.execute(
            "install-records",
            "--run-id",
            self.run_id,
            "--sha256",
            records_sha256(records),
            "--b64",
            base64.b64encode(_encode(records)).decode("ascii"),
            timeout=120,
        )
        return receipt

    def open(
        self, *, restore_seconds: int = 900, approved: bool = False
    ) -> dict[str, Any]:
        self.create()
        require_action_time(180)
        arguments = [
            "open-window",
            "--run-id",
            self.run_id,
            "--mode",
            self.mode,
            "--restore-seconds",
            str(int(restore_seconds)),
        ]
        if approved:
            arguments.append("--approved")
        receipt = self.host.execute(*arguments, timeout=150)
        self._opened = True
        return receipt

    def arm(self) -> dict[str, Any]:
        require_action_time(60)
        receipt = self.host.execute("arm", "--run-id", self.run_id, timeout=120)
        self._armed = True
        return receipt

    def read_back(self) -> dict[str, Any]:
        """What the reviewed source produced on the node (the runner's oracle)."""

        return self.host.execute("status", "--run-id", self.run_id, timeout=60)

    def close(self) -> dict[str, Any]:
        receipt = self.host.execute(
            "close-window", "--run-id", self.run_id, "--mode", self.mode, timeout=150
        )
        self._opened = False
        self._armed = False
        return receipt

    def cleanup(self) -> dict[str, Any]:
        """Close the window and delete the probe; raise if anything is left.

        Pushed onto the variant's LIFO cleanup stack, so a window that cannot be
        proved closed fails the variant (``run_general_live`` ``_teardown``).
        """

        receipts: dict[str, Any] = {}
        if self._opened:
            receipts["close"] = self.close()
            if not receipts["close"].get("clean", True):
                raise RuntimeError(
                    f"kernel replay window did not close clean: {receipts['close']}"
                )
        receipts["probe_residuals"] = self.host.cleanup()
        if any(receipts["probe_residuals"].values()):
            raise RuntimeError(
                f"kernel replay probe left residue: {receipts['probe_residuals']}"
            )
        return receipts

    # -- evidence -------------------------------------------------------- #
    def evidence(self) -> dict[str, Any]:
        return {
            "proof_scope": PROOF_SCOPE,
            "mode": self.mode,
            "records_sha256": records_sha256(self._installed)
            if self._installed
            else None,
            "record_count": len(self._installed),
            "armed": self._armed,
        }


def kernel_replay(ctx: VariantContext, *, mode: str = "journal") -> KernelLogReplay:
    """Build a ``KernelLogReplay`` from a variant context and register cleanup.

    The cleanup goes on ``ctx.cleanup`` (LIFO), so the runner tears the window
    down in its ``finally`` and a window that will not close clean fails the
    variant.
    """

    # The context's handle under ``kernel-replay-<mode>@<node>`` (a test
    # pre-registers a fake there): torn down by the runner whatever happens
    # to the cleanup pushed below.
    host = ctx.host_probe(
        f"kernel-replay-{mode}",
        script=PROBE_SCRIPT,
        deadline_seconds=ctx.spec.max_minutes * 60 + 1200,
    )
    replay = KernelLogReplay(
        ctx.regional,
        node=ctx.settings.node,
        image=ctx.settings.host_probe_image,
        case_id=ctx.spec.key,
        run_id=ctx.run_id,
        case_dir=ctx.case_dir,
        mode=mode,
        active_deadline_seconds=ctx.spec.max_minutes * 60 + 1200,
        host=host,
    )
    ctx.cleanup.push(f"kernel-replay-{mode}", replay.cleanup)
    return replay


__all__ = [
    "KernelLogReplay",
    "MODES",
    "PROOF_SCOPE",
    "kernel_replay",
    "kmsg_follow_error",
    "records_sha256",
    "reviewed_record",
]
