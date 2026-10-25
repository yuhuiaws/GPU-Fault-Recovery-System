"""The GF-ENV-LIVE telemetry replay inside a GF-GENERAL-LIVE variant (ENV-reuse).

GF-GENERAL-LIVE-024 ``thermal_warning`` and ``power_performance`` drive the
THERMAL_STRESS and POWER_LIMIT_THROTTLING composites. Their components --
temperature, T.Limit, thermal/power violation durations, clock reasons, power
draw/limit, utilization -- are exactly the fields the reviewed environment
replay tool ``tools/gpu_environment_replay.py`` may change (its
``ALLOWED_FIELDS``; the DCGM field tool of the other 024 variants deliberately
does not carry them). This module runs that tool the way ``run_env_live.py``
does -- the audited loopback proxy served by ``probes/env_replay_proxy.py`` on
the node, the one metrics collector unit repointed through a ``/run`` drop-in
with a ``systemd-run`` deadman armed first, pass-through until ``arm`` -- but
inside a variant's ``VariantContext``: cleanup on the LIFO stack, receipts and
the proxy's ``/status`` readings in the evidence, and the general-live error
semantics (``VariantInconclusive`` when the injection cannot be shown).

Only the reviewed plan is new per variant (``env_live_plans.thermal_stress_plan``,
``power_limit_throttling_plan``); the tool, its allow-list, the probe and the
controller (``env_live_fixture.EnvReplayController``) are the GF-ENV-LIVE ones
unchanged. Nothing here decides a verdict; the 024 module judges the product's
own records.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from scripts.e2e.regional.env_live_fixture import (
    PLAN_FILE,
    EnvReplayController,
    replay_tool_sha256,
    write_plan_file,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tools.gpu_environment_replay import ALLOWED_FIELDS, validate_plan

#: The environment proxy's loopback port (the DCGM field tool uses 9412).
PORT = 9411
#: The proxy's replay window (``run_env_live.WINDOW_SECONDS``; the probe caps
#: ``--max-seconds`` there): the tool passes through by itself when the window
#: ends and its unit stops 60 s later, so every observation of a variant must
#: finish, and the collector endpoint be restored, inside it.
PROXY_WINDOW_SECONDS = 600
#: The collector deadman restores this much after the proxy window ends.
DEADMAN_MARGIN_SECONDS = 600
#: The probe's upper bound on the deadman (``MAX_RESTORE_SECONDS``).
MAX_RESTORE_SECONDS = 1800
#: Left for the restore itself when a wait is bounded by the window.
WINDOW_CLOSE_MARGIN_SECONDS = 90
#: ``ctx.probes`` key under which a test pre-registers a fake host probe.
PROBE_KEY = "env-replay"
PROOF_SCOPE = "live-replay"
CLEANUP_NAME = "env-replay"


def plan_sha256(plan: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def plan_reads(plan: dict[str, Any]) -> int:
    """How many collector reads the plan consumes once armed."""

    return sum(int(step.get("reads") or 0) for step in plan.get("steps") or [])


def plan_fields(plan: dict[str, Any]) -> set[str]:
    return {
        str(name)
        for step in plan.get("steps") or []
        for name in (step.get("values") or {})
    }


def checked_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """The plan the reviewed tool itself accepts (allow-list, bounds, numbers)."""

    try:
        validate_plan(plan)
    except ValueError as exc:
        raise RegionalFixtureError(f"environment replay plan rejected: {exc}") from exc
    foreign = sorted(plan_fields(plan) - ALLOWED_FIELDS)
    if foreign:  # pragma: no cover - validate_plan refuses these first
        raise RegionalFixtureError(
            f"plan names fields outside ALLOWED_FIELDS: {foreign}"
        )
    return plan


@dataclass
class EnvReplay:
    """One reviewed environment replay on the node, cleanup registered.

    ``deploy`` ships the audited tool and the plan, starts the proxy pass-through,
    repoints the collector (deadman first, inside the probe) and proves the
    proxy is still unarmed; ``arm`` starts the plan. ``wait_served`` is the raw
    -source oracle: the proxy's ``/status`` must show the collector really
    consumed the plan's reads and the tool never aborted on a real unsafe
    reading. ``close`` restores the endpoint before it stops the proxy (one
    producer at every moment) and verifies both.
    """

    ctx: VariantContext
    plan: dict[str, Any]
    gpu_uuid: str
    seconds: int
    controller: EnvReplayController = field(init=False)
    receipts: dict[str, Any] = field(default_factory=dict)
    statuses: list[dict[str, Any]] = field(default_factory=list)
    reads_at_arm: int | None = None
    armed_at: Any = None
    deployed: bool = False
    armed: bool = False
    closed: bool = False

    @property
    def window_seconds(self) -> int:
        """The proxy's replay window: the variant's hold, capped by the probe."""

        return max(60, min(PROXY_WINDOW_SECONDS, int(self.seconds)))

    def remaining_seconds(self, floor: float = 30.0) -> float:
        """Seconds the armed window still allows for a wait, keeping the close margin."""

        if self.armed_at is None:
            return float(self.window_seconds - WINDOW_CLOSE_MARGIN_SECONDS)
        elapsed = (self.ctx.clock() - self.armed_at).total_seconds()
        return max(floor, self.window_seconds - WINDOW_CLOSE_MARGIN_SECONDS - elapsed)

    def __post_init__(self) -> None:
        checked_plan(self.plan)
        if not self.gpu_uuid.startswith("GPU-"):
            raise RegionalFixtureError(
                "the environment replay needs a physical GPU UUID"
            )
        ctx = self.ctx
        host = ctx.probes.get(f"{PROBE_KEY}@{ctx.settings.node}")
        self.controller = EnvReplayController(
            ctx.regional,
            node=ctx.settings.node,
            image=ctx.settings.host_probe_image,
            case_id=ctx.spec.key,
            run_id=ctx.run_id,
            case_dir=ctx.case_dir,
            active_deadline_seconds=max(
                600, min(7200, ctx.spec.max_minutes * 60 + 1200)
            ),
            host=host,
        )

    # -- lifecycle ------------------------------------------------------- #
    def deploy(self) -> dict[str, Any]:
        """Install, serve pass-through, repoint; the plan is not armed yet."""

        self.ctx.cleanup.push(CLEANUP_NAME, self.cleanup)
        self.controller.create()
        plan_path = write_plan_file(self.ctx.case_dir, PLAN_FILE, self.plan)
        self.receipts["preflight"] = self.controller.preflight()
        self.receipts["deploy"] = self.controller.deploy_proxy(
            plan_path,
            self.gpu_uuid,
            port=PORT,
            max_seconds=self.window_seconds,
        )
        self.receipts["repoint"] = self.controller.repoint(
            port=PORT,
            restore_seconds=min(
                MAX_RESTORE_SECONDS, self.window_seconds + DEADMAN_MARGIN_SECONDS
            ),
        )
        self.deployed = True
        status = self.status()
        if (status.get("status") or {}).get("armed") is not False:
            raise VariantInconclusive(
                "the environment replay proxy was armed before the baseline read"
            )
        self.receipts["passthrough_status"] = status
        self.ctx.evidence.write("env-replay-deploy", self.receipts)
        return self.receipts

    def arm(self) -> dict[str, Any]:
        receipt = self.controller.arm(port=PORT)
        self.armed = True
        self.armed_at = self.ctx.clock()
        reads = (receipt.get("status") or {}).get("reads")
        self.reads_at_arm = int(reads) if isinstance(reads, (int, float)) else 0
        self.receipts["arm"] = receipt
        self.ctx.evidence.write("env-replay-arm", receipt)
        return receipt

    def status(self) -> dict[str, Any]:
        """One ``/status`` reading (never consumes a plan step), kept as oracle."""

        value = self.controller.status(port=PORT)
        self.statuses.append(value)
        self.ctx.evidence.append_oracle(
            [
                {
                    "status": value.get("status"),
                    "endpoint": value.get("effective_endpoint"),
                }
            ],
            source="env-replay",
        )
        return value

    def served_reads(self, status: dict[str, Any]) -> int:
        reads = (status.get("status") or {}).get("reads")
        if not isinstance(reads, (int, float)) or self.reads_at_arm is None:
            return 0
        return max(0, int(reads) - self.reads_at_arm)

    def require_not_aborted(self, status: dict[str, Any]) -> None:
        aborted = (status.get("status") or {}).get("aborted")
        if aborted:
            raise VariantInconclusive(
                f"the environment replay aborted on a real unsafe reading: {aborted}"
            )

    def wait_served(
        self, reads: int, *, timeout_seconds: float, name: str
    ) -> dict[str, Any]:
        """The oracle gate: the collector consumed ``reads`` armed reads."""

        last: dict[str, Any] = {}

        def accept() -> dict[str, Any] | None:
            nonlocal last
            last = self.status()
            self.require_not_aborted(last)
            return last if self.served_reads(last) >= reads else None

        value = self.ctx.wait_until(
            accept, timeout_seconds=timeout_seconds, poll_seconds=15, name=name
        )
        if value is None:
            raise VariantInconclusive(
                f"the collector consumed {self.served_reads(last)} of the {reads} "
                "armed replay reads inside the budget"
            )
        return value

    def close(self) -> dict[str, Any]:
        """Restore the collector endpoint, then stop the proxy; verify both."""

        if self.closed:
            return dict(self.receipts.get("close") or {})
        receipts: dict[str, Any] = {}
        if self.deployed:
            receipts["restore"] = self.controller.restore()
            self.deployed = False
        receipts["stop_proxy"] = self.controller.stop_proxy(port=PORT)
        self.closed = True
        self.armed = False
        self.receipts["close"] = receipts
        self.ctx.evidence.write("env-replay-close", receipts)
        restore = receipts.get("restore") or {}
        if restore:
            if (restore.get("after") or {}).get("ActiveState") != "active":
                raise VariantInconclusive(
                    "the metrics collector is not active after the endpoint restore"
                )
            endpoint = restore.get("restored_endpoint")
            if endpoint not in (None, restore.get("original_endpoint")):
                raise VariantInconclusive(
                    f"the collector endpoint {endpoint} was not restored to "
                    f"{restore.get('original_endpoint')}"
                )
        return receipts

    def cleanup(self) -> dict[str, Any]:
        """Close if still open and delete the probe; raise on any residue."""

        receipts: dict[str, Any] = {}
        if not self.closed:
            receipts["close"] = self.close()
        receipts["probe_residuals"] = self.controller.host.cleanup()
        if any(receipts["probe_residuals"].values()):
            raise RuntimeError(
                f"environment replay probe left residue: {receipts['probe_residuals']}"
            )
        return receipts

    # -- evidence -------------------------------------------------------- #
    def evidence(self) -> dict[str, Any]:
        return {
            "proof_scope": PROOF_SCOPE,
            "tool": "tools/gpu_environment_replay.py",
            "tool_sha256": replay_tool_sha256(),
            "controller": "scripts/e2e/regional/probes/env_replay_proxy.py",
            "plan_sha256": plan_sha256(self.plan),
            "plan_reads": plan_reads(self.plan),
            "plan_fields": sorted(plan_fields(self.plan)),
            "gpu_uuid": self.gpu_uuid,
            "window_seconds": self.window_seconds,
            "armed": self.armed,
            "armed_at": self.armed_at.isoformat() if self.armed_at else None,
            "reads_at_arm": self.reads_at_arm,
            "status_readings": len(self.statuses),
        }


def env_replay(
    ctx: VariantContext, plan: dict[str, Any], gpu_uuid: str, seconds: int
) -> EnvReplay:
    """Build an ``EnvReplay`` for the variant (cleanup is pushed by ``deploy``)."""

    return EnvReplay(ctx=ctx, plan=plan, gpu_uuid=gpu_uuid, seconds=int(seconds))


__all__ = [
    "CLEANUP_NAME",
    "DEADMAN_MARGIN_SECONDS",
    "MAX_RESTORE_SECONDS",
    "PORT",
    "PROBE_KEY",
    "PROOF_SCOPE",
    "PROXY_WINDOW_SECONDS",
    "WINDOW_CLOSE_MARGIN_SECONDS",
    "EnvReplay",
    "checked_plan",
    "env_replay",
    "plan_fields",
    "plan_reads",
    "plan_sha256",
]
