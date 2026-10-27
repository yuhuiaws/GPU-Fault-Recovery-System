"""Which product the pending-remap reset chain faces, decided before the reset.

GF-GENERAL-LIVE-024 ``wrong_scope_source`` and 019 ``pending_supported`` /
``post_reset`` mint a ``row_remap_pending`` (``retired_pages_pending`` on older
parts) WARNING finding whose policy is the product's single-GPU service-window
RESET_GPU chain, and the chain's post-reset VALIDATE_GPU decides the record.
The replay is bounded to the minting scrape, so the first batch after
RESTORE_GPU_SERVICES is the device's real reading -- but whether that batch
re-judges the flag is the exporter's and the collector's business, not the
harness's, and on the product generations seen so far the verdict does not
wait for it:

* ``adapters/gpu_validation.GpuValidationAdapter._validate_active_findings``
  grants a WARNING-only finding a grace, and (b202, gap 26) past the grace a
  wait for a GPU-metrics batch newer than the finding, for the thermal,
  correctable-memory and power families only (``thermal_names`` /
  ``memory_names`` / ``power_names``). A ``row_remap_pending`` finding is in
  none of them: ``warning_grace`` stays ``None`` and the step fails at once
  with ``active_gpu_health_findings`` on the state the minting batch left.
  R13g (live 2026-10-26 14:54Z and 2026-10-27 14:36Z, HyperPod) and the R13h
  candidate's ``gpu_validation.py`` (b202 applied) read identically here.
* ``orchestration/escalation.next_rung`` turns a validation failed after a
  reset into RESTART_NODE where the ladder climbs; where it does not, the
  branch is exhausted and the incident ends QUARANTINED (2026-10-27: no
  reboot, boot id unchanged, record settled QUARANTINED 2 min 20 s after
  the mint).

So the variant's PASS depends on a product property, not on the fixture:
``PENDING_FLAG_VERDICT_PROBE`` asks the deployed control plane (read-only, in
an API Pod, the installed interpreter) what its own adapter answers for a
WARNING pending-flag finding inside and past the grace. A product whose
VALIDATE_GPU keeps WAITING past the grace for newer data re-judges the flag on
the real post-reset batch (``rejudges``); the variants run their full oracle
against it. A product that fails at once cannot PASS these variants by design:
they end INCONCLUSIVE before anything is deployed, with the mechanism and the
generation cited (the 014 freeze path's rule, S47), the node untouched. A
probe the product's code shape defeats is "undecided" and refuses the same
way, naming the error -- never a FAIL, never a reset on a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
)

#: The pending-repair flags ``gpu_memory_policy.memory_decision`` grades
#: WARNING RESET_GPU on a modern GPU.
PENDING_FLAGS = ("row_remap_pending", "retired_pages_pending")
#: Evidence file of the probe's answer (the 014 fixture's name).
GENERATION_EVIDENCE = "product-generation"

#: Read-only, in an API Pod: what the deployed VALIDATE_GPU adapter answers
#: for an active WARNING pending-flag finding of one GPU, 30 s after the
#: workflow was created (inside any grace) and 10 min after it (past every
#: grace, inside the executor's 600 s per-step cap measured from the step's own
#: start). ``_validate_active_findings`` reads the findings through the metrics
#: service it is given and, when it waits for newer data, that service's
#: ``latest``; both come from this script, nothing touches the store. A dict
#: is the step still pending (grace cooldown / awaiting newer GPU metrics);
#: ``None`` with ``active_gpu_health_findings`` is the verdict at once.
PENDING_FLAG_VERDICT_PROBE = r"""
import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

flag = sys.argv[1]
result = {"flag": flag, "present": False, "grace": None, "rejudges": None}
try:
    from gpu_fault.adapters import gpu_validation
    from gpu_fault.gpu_metric_models import GpuHealthFinding, GpuHealthSeverity
    from gpu_fault.models import WorkflowOperation
except ImportError as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"[:300]
    print(json.dumps(result))
    sys.exit(0)

created = datetime(2026, 1, 1, tzinfo=timezone.utc)
gpu = "GPU-00000000-0000-4000-8000-00000000abcd"
node = "generation-probe-node"
finding = GpuHealthFinding(
    finding_id=f"generation-probe-{flag}",
    cluster_id="generation-probe-cluster",
    node_id=node,
    observed_at=created,
    severity=GpuHealthSeverity.WARNING,
    reason="product generation probe",
    canonical_name=flag,
    value=1.0,
    gpu_uuid=gpu,
    automatic_action="RESET_GPU",
)


class Metrics:
    store = None

    def findings(self, cluster_id, node_id, *, active_only=True):
        return [finding]

    def latest(self, cluster_id, node_id):
        return []


context = SimpleNamespace(
    incident=SimpleNamespace(cluster_id="generation-probe-cluster"),
    step=SimpleNamespace(
        operation=WorkflowOperation.VALIDATE_GPU,
        gpu_uuids=[gpu],
        node_ids=[node],
        parameters={},
    ),
    step_index=7,
    workflow=SimpleNamespace(
        request_id="workflow-generation-probe",
        created_at=created,
        step_executions=[],
    ),
)


def judge(now):
    adapter = gpu_validation.GpuValidationAdapter(Metrics())
    failures = {}
    pending = adapter._validate_active_findings(
        context, node, WorkflowOperation.VALIDATE_GPU, now, failures, {}
    )
    return (
        sorted(str(key) for key in pending) if isinstance(pending, dict) else None,
        [str(item) for item in failures.get(node, [])],
    )


try:
    inside, inside_failures = judge(created + timedelta(seconds=30))
    past, past_failures = judge(created + timedelta(minutes=10))
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"[:300]
else:
    result.update(
        present=True,
        grace=inside is not None,
        rejudges=past is not None,
        inside=inside,
        inside_failures=inside_failures,
        past=past,
        past_failures=past_failures,
        newer_metrics_wait=hasattr(gpu_validation, "awaiting_newer_gpu_metrics"),
    )
print(json.dumps(result))
"""


@dataclass(frozen=True)
class PendingFlagGeneration:
    """The control plane's answer to ``PENDING_FLAG_VERDICT_PROBE`` for ``flag``."""

    flag: str
    control_plane: dict[str, Any]

    @property
    def present(self) -> bool:
        return bool(self.control_plane.get("present"))

    @property
    def grace(self) -> bool | None:
        value = self.control_plane.get("grace")
        return value if self.present and isinstance(value, bool) else None

    @property
    def rejudges(self) -> bool | None:
        """Past the grace, does VALIDATE_GPU wait for data newer than the finding?

        ``True``: the chain's post-reset verdict follows the real post-reset
        batch (the variants may run). ``False``: the verdict is the state the
        minting batch left. ``None``: the probe could not decide.
        """

        value = self.control_plane.get("rejudges")
        return value if self.present and isinstance(value, bool) else None

    @property
    def error(self) -> str:
        return str(self.control_plane.get("error") or "")

    def describe(self) -> str:
        if self.rejudges is None:
            return (
                "the control plane's VALIDATE_GPU verdict for a WARNING "
                f"{self.flag} finding could not be read"
                + (f" ({self.error})" if self.error else "")
            )
        past = self.control_plane.get("past") or []
        if self.rejudges:
            return (
                f"the control plane's VALIDATE_GPU keeps WAITING past the warning "
                f"grace for a GPU-metrics batch newer than a WARNING {self.flag} "
                f"finding ({', '.join(str(key) for key in past) or 'pending'})"
            )
        failures = self.control_plane.get("past_failures") or []
        wait = (
            "the newer-metrics wait exists for other finding families"
            if self.control_plane.get("newer_metrics_wait")
            else "no newer-metrics wait exists at all"
        )
        grace = "a grace" if self.grace else "no grace"
        return (
            f"the control plane's VALIDATE_GPU judges a WARNING {self.flag} finding "
            f"at once ({', '.join(str(item) for item in failures) or 'failed'}; "
            f"{wait}; {grace} inside the warning window)"
        )

    def record(self) -> dict[str, Any]:
        return {
            "flag": self.flag,
            "present": self.present,
            "grace": self.grace,
            "rejudges": self.rejudges,
            "control_plane": dict(self.control_plane),
            "description": self.describe(),
        }


def pending_flag_generation(ctx: VariantContext, flag: str) -> PendingFlagGeneration:
    """Ask the deployed control plane; the answer goes to ``product-generation``."""

    answer = ctx.regional.cpu_python(PENDING_FLAG_VERDICT_PROBE, flag, timeout=120)
    value = PendingFlagGeneration(
        flag=flag, control_plane=dict(answer) if isinstance(answer, dict) else {}
    )
    ctx.evidence.write(GENERATION_EVIDENCE, value.record())
    return value


def judges_at_once_detail(generation: PendingFlagGeneration, *, chain: str) -> str:
    """Why the variant is INCONCLUSIVE on a product that judges the flag at once."""

    return (
        "expected on this product generation: the post-reset VALIDATE_GPU of "
        f"{chain} judges the {generation.flag} WARNING finding at once on the state "
        "the minting batch left -- adapters/gpu_validation._validate_active_findings "
        "grants the pending flag neither the warning grace nor the wait for a "
        "GPU-metrics batch newer than the finding (b202 / gap 26 covers the "
        "thermal, correctable-memory and power families only) -- so unless the "
        "first post-reset batch re-judges the flag the chain fails VALIDATE_GPU "
        "and the record ends QUARANTINED (RESTART_NODE where the ladder climbs; "
        "live 2026-10-26 14:54Z and 2026-10-27 14:36Z, HyperPod, R13g). Nothing "
        f"was deployed and the node was not touched; {generation.describe()}"
    )


def undecided_detail(generation: PendingFlagGeneration, *, chain: str) -> str:
    return (
        f"the product generation {chain} runs against could not be decided before "
        f"the reset ({generation.describe()}); nothing was deployed and the node "
        "was not touched -- a reset on a guess is not run"
    )


def require_rejudging_product(
    ctx: VariantContext, flag: str, *, chain: str
) -> PendingFlagGeneration:
    """INCONCLUSIVE before anything is deployed unless the product re-judges
    ``flag`` on data newer than the finding; returns the generation otherwise."""

    generation = pending_flag_generation(ctx, flag)
    if generation.rejudges is None:
        raise VariantInconclusive(undecided_detail(generation, chain=chain))
    if not generation.rejudges:
        raise VariantInconclusive(judges_at_once_detail(generation, chain=chain))
    return generation


def rejudging_answer(flag: str = PENDING_FLAGS[0]) -> dict[str, Any]:
    """The probe's answer from a product that waits for newer data (tests, docs)."""

    return {
        "flag": flag,
        "present": True,
        "grace": True,
        "rejudges": True,
        "inside": ["transient_gpu_warning_cooldown"],
        "inside_failures": [],
        "past": [
            "awaiting_newer_gpu_metrics",
            "awaiting_newer_gpu_metrics_findings",
            "grace_elapsed_at",
            "reason",
            "step_cap_at",
        ],
        "past_failures": [],
        "newer_metrics_wait": True,
    }


def judging_at_once_answer(
    flag: str = PENDING_FLAGS[0], *, newer_metrics_wait: bool = False
) -> dict[str, Any]:
    """The probe's answer from R13g (``newer_metrics_wait=False``) or from the
    R13h candidate (b202 present, the pending flag still outside it)."""

    return {
        "flag": flag,
        "present": True,
        "grace": False,
        "rejudges": False,
        "inside": None,
        "inside_failures": ["active_gpu_health_findings"],
        "past": None,
        "past_failures": ["active_gpu_health_findings"],
        "newer_metrics_wait": newer_metrics_wait,
    }


__all__ = [
    "GENERATION_EVIDENCE",
    "PENDING_FLAGS",
    "PENDING_FLAG_VERDICT_PROBE",
    "PendingFlagGeneration",
    "judges_at_once_detail",
    "judging_at_once_answer",
    "pending_flag_generation",
    "rejudging_answer",
    "require_rejudging_product",
    "undecided_detail",
]
