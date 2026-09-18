"""Guarded physical late-ownership subproof for PREEMPT-033 and DESTR-015.

This driver is composed into an approved live case, never invoked as a model
replay CLI. Its I/O owner supplies deployed product execution and independent
node witnesses. A missing capability must fail preflight before any mutation.
The ordinary DESTR-015 two-node success path is a separate, unchanged case.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Protocol

from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied, check_stop
from scripts.e2e.regional.late_ownership_contract import (
    AcceptanceEvidence,
    AcceptanceScope,
    CleanupReceipt,
    DecisionReceipt,
    MutationReceipt,
    QuiescenceReceipt,
    RecheckPermit,
    StopReceipt,
    WitnessEnd,
    WitnessStart,
)
from scripts.e2e.regional.late_ownership_verdicts import (
    acceptance_errors,
    cleanup_errors,
    decision_errors,
    mutation_errors,
    quiescence_errors,
    receipt_errors,
    witness_start_errors,
)


class PhysicalBoundaryIO(Protocol):
    """Only the approved case's owned probes may implement this live interface."""

    @property
    def evidence_mode(self) -> Literal["LIVE", "LOCAL_TEST"]: ...

    def preflight(self, scope: AcceptanceScope) -> None:
        """Verify release/UID pins, deployed guard, trace support and no residuals."""
        ...

    def arm_witnesses(
        self, scope: AcceptanceScope
    ) -> tuple[WitnessStart, WitnessStart]:
        """Attach and physically calibrate both observers before enabling STOP."""
        ...

    def stop_at_boundary(
        self, scope: AcceptanceScope, starts: tuple[WitnessStart, WitnessStart]
    ) -> StopReceipt:
        """Run real containment; return only when the pre-action callback is parked."""
        ...

    def mutate_owned_target(
        self, scope: AcceptanceScope, stop: StopReceipt
    ) -> MutationReceipt:
        """UID/resourceVersion-guarded injection plus fresh physical read-back."""
        ...

    def recheck(self, scope: AcceptanceScope, permit: RecheckPermit) -> DecisionReceipt:
        """Release into the installed product guard, with the original lease intact."""
        ...

    def quiesce(
        self, scope: AcceptanceScope, decision: DecisionReceipt
    ) -> QuiescenceReceipt:
        """Revoke the one-shot gate and prove commands/callbacks are terminal."""
        ...

    def finish_witnesses(
        self, scope: AcceptanceScope, quiet: QuiescenceReceipt
    ) -> tuple[WitnessEnd, WitnessEnd]:
        """Drain continuous physical traces, preserving any unexpected action."""
        ...

    def revoke(self, scope: AcceptanceScope) -> None:
        """Idempotently deny late callbacks, including after an unacknowledged start."""
        ...

    def cleanup(self, scope: AcceptanceScope) -> CleanupReceipt:
        """Clean only this run's UID-bound resources after physical quiescence."""
        ...


@dataclass
class RunResult:
    case_id: str
    scenario: str
    evidence_mode: Literal["LIVE", "LOCAL_TEST"]
    scope_sha256: str
    verdict: Literal["PASS", "FAIL"] = "FAIL"
    errors: list[str] = field(default_factory=list)
    evidence: AcceptanceEvidence | None = None
    cleanup: CleanupReceipt | None = None

    def summary(self) -> dict[str, object]:
        """No arbitrary adapter response, exception text, argv or credentials."""
        return {
            "case_id": self.case_id,
            "scenario": self.scenario,
            "subproof": "physical-late-ownership",
            "evidence_mode": self.evidence_mode,
            "scope_sha256": self.scope_sha256,
            "verdict": self.verdict,
            "errors": list(self.errors),
            "evidence_sha256": self.evidence.digest()
            if self.evidence is not None
            else None,
            "cleanup_sha256": self.cleanup.digest()
            if self.cleanup is not None
            else None,
            "promotes_ordinary_case": False,
        }


def require_no_errors(errors: list[str]) -> None:
    if errors:
        raise BoundaryDenied("; ".join(errors))


def run_case(
    scope: AcceptanceScope,
    io: PhysicalBoundaryIO,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> RunResult:
    """Execute the causal protocol; clean up before deriving the final verdict.

    Intent is recorded in memory before each I/O that can partly succeed. The
    outer approved case must also journal ownership before invoking this driver.
    In particular, an arm/STOP timeout cannot be interpreted as "nothing started".
    """
    result = RunResult(scope.case_id, scope.scenario, io.evidence_mode, scope.digest())
    started = False
    stages: dict[str, object] = {}
    stage = "preflight"

    def enter(name: str) -> None:
        nonlocal stage
        stage = name
        scope.check_window(now())

    try:
        enter("preflight")
        io.preflight(scope)
        enter("arm-witnesses")
        started = True
        starts = io.arm_witnesses(scope)
        require_no_errors(witness_start_errors(scope, starts))
        stages["witness_starts"] = starts
        enter("physical-stop")
        stop = io.stop_at_boundary(scope, starts)
        require_no_errors(receipt_errors(scope, stop))
        check_stop(scope, stop)
        if stop.witness_start_sha256 != tuple(item.digest() for item in starts):
            raise BoundaryDenied("STOP did not acknowledge both calibrated witnesses")
        stages["stop"] = stop
        enter("owned-mutation")
        mutation = io.mutate_owned_target(scope, stop)
        require_no_errors(mutation_errors(scope, stop, mutation))
        stages["mutation"] = mutation
        permit = RecheckPermit(
            scope_sha256=scope.digest(),
            boundary_id=stop.boundary_id,
            stop_sha256=stop.digest(),
            mutation_sha256=mutation.digest(),
        )
        stages["permit"] = permit
        enter("product-recheck")
        decision = io.recheck(scope, permit)
        require_no_errors(decision_errors(scope, stop, mutation, permit, decision))
        stages["decision"] = decision
        enter("quiescence")
        quiet = io.quiesce(scope, decision)
        require_no_errors(quiescence_errors(scope, stop, decision, quiet))
        stages["quiescence"] = quiet
        enter("physical-receipts")
        stages["witness_ends"] = io.finish_witnesses(scope, quiet)
    except Exception as exc:
        # External SDK/transport exceptions may contain authentication material.
        result.errors.append(f"{stage}: {type(exc).__name__}")
    finally:
        if started:
            try:
                io.revoke(scope)
            except Exception as exc:
                result.errors.append(f"revoke: {type(exc).__name__}")
            try:
                cleanup = io.cleanup(scope)
                result.cleanup = cleanup
                result.errors.extend(cleanup_errors(scope, cleanup))
            except Exception as exc:
                result.errors.append(f"cleanup: {type(exc).__name__}")
    if result.cleanup is not None and len(stages) == 7:
        try:
            proof = AcceptanceEvidence.model_validate(
                {"scope": scope, **stages, "cleanup": result.cleanup}
            )
            result.evidence = proof
            result.errors.extend(acceptance_errors(proof))
        except Exception as exc:
            result.errors.append(f"receipt-validation: {type(exc).__name__}")
    if not result.errors and result.evidence is not None:
        result.verdict = "PASS"
    return result
