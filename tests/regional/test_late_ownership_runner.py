from __future__ import annotations

from datetime import timedelta

import pytest

from scripts.e2e.regional import run_late_ownership_acceptance as runner
from tests.regional._late_ownership_support import evidence


class FakeBoundaryIO:
    evidence_mode = "LOCAL_TEST"

    def __init__(self, proof, *, fail_at=None, exception=None):
        self.proof = proof
        self.fail_at = fail_at
        self.exception = exception or RuntimeError(
            "private adapter material must not be echoed"
        )
        self.calls = []

    def enter(self, stage, scope):
        assert scope == self.proof.scope, "an I/O call changed the approved scope"
        self.calls.append(stage)
        if self.fail_at == stage:
            raise self.exception

    def preflight(self, scope):
        self.enter("preflight", scope)

    def arm_witnesses(self, scope):
        self.enter("arm", scope)
        return self.proof.witness_starts

    def stop_at_boundary(self, scope, starts):
        self.enter("stop", scope)
        assert starts == self.proof.witness_starts, (
            "STOP did not receive both physical witness roots"
        )
        return self.proof.stop

    def mutate_owned_target(self, scope, stop):
        self.enter("mutate", scope)
        assert stop == self.proof.stop, "mutation was not bound to the parked callback"
        return self.proof.mutation

    def recheck(self, scope, permit):
        self.enter("recheck", scope)
        assert permit == self.proof.permit, (
            "product received a different one-shot permit"
        )
        return self.proof.decision

    def quiesce(self, scope, decision):
        self.enter("quiesce", scope)
        assert decision == self.proof.decision, (
            "quiescence did not bind the product result"
        )
        return self.proof.quiescence

    def finish_witnesses(self, scope, quiet):
        self.enter("finish", scope)
        assert quiet == self.proof.quiescence, "witnesses ended before exact quiescence"
        return self.proof.witness_ends

    def revoke(self, scope):
        self.enter("revoke", scope)

    def cleanup(self, scope):
        self.enter("cleanup", scope)
        return self.proof.cleanup


STAGES = [
    "preflight",
    "arm",
    "stop",
    "mutate",
    "recheck",
    "quiesce",
    "finish",
    "revoke",
    "cleanup",
]


@pytest.mark.parametrize(
    "scenario", ["unchanged-owner", "ownership-drift", "late-sibling"]
)
def test_runner_executes_acknowledged_causal_stages_and_only_then_derives_pass(
    scenario,
):
    proof = evidence(scenario)
    io = FakeBoundaryIO(proof)
    result = runner.run_case(proof.scope, io)
    assert io.calls == STAGES
    assert result.verdict == "PASS"
    assert result.errors == []
    assert result.evidence == proof
    assert result.cleanup == proof.cleanup
    assert result.summary() == {
        "case_id": proof.scope.case_id,
        "scenario": scenario,
        "subproof": "physical-late-ownership",
        "evidence_mode": "LOCAL_TEST",
        "scope_sha256": proof.scope.digest(),
        "verdict": "PASS",
        "errors": [],
        "evidence_sha256": proof.digest(),
        "cleanup_sha256": proof.cleanup.digest(),
        "promotes_ordinary_case": False,
    }


@pytest.mark.parametrize("stage", STAGES)
def test_failure_at_each_io_boundary_revokes_and_cleans_partial_mutations(stage):
    proof = evidence()
    io = FakeBoundaryIO(proof, fail_at=stage)
    result = runner.run_case(proof.scope, io)
    assert result.verdict == "FAIL"
    assert "private adapter material" not in str(result.summary())
    expected = STAGES[: STAGES.index(stage) + 1]
    if stage not in {"preflight", "revoke", "cleanup"}:
        expected += ["revoke", "cleanup"]
    elif stage == "revoke":
        expected += ["cleanup"]
    assert io.calls == expected
    assert len(result.errors) == 1
    assert result.errors[0].endswith(": RuntimeError"), (
        "the report must retain the original I/O failure type"
    )
    if stage == "preflight":
        assert result.cleanup is None
        assert result.evidence is None
        assert result.summary()["evidence_sha256"] is None
        assert result.summary()["cleanup_sha256"] is None
    elif stage == "cleanup":
        assert result.cleanup is None
        assert result.evidence is None
    else:
        assert result.cleanup == proof.cleanup


@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
def test_interruption_while_parked_still_revokes_and_cleans(kind):
    proof = evidence()
    io = FakeBoundaryIO(proof, fail_at="stop", exception=kind())
    with pytest.raises(kind):
        runner.run_case(proof.scope, io)
    assert io.calls == ["preflight", "arm", "stop", "revoke", "cleanup"]


@pytest.mark.parametrize("boundary", range(7))
def test_expired_window_blocks_every_forward_stage_but_not_cleanup(boundary):
    proof = evidence()
    io = FakeBoundaryIO(proof)
    inside = proof.scope.maintenance_start + timedelta(seconds=1)
    stamps = iter([inside] * boundary + [proof.scope.maintenance_end])
    result = runner.run_case(proof.scope, io, now=lambda: next(stamps))
    assert result.verdict == "FAIL"
    expected = STAGES[:boundary]
    if boundary >= 2:
        expected += ["revoke", "cleanup"]
    assert io.calls == expected
    assert result.errors == [
        f"{['preflight', 'arm-witnesses', 'physical-stop', 'owned-mutation', 'product-recheck', 'quiescence', 'physical-receipts'][boundary]}: ValueError"
    ]


@pytest.mark.parametrize(
    "field", ["witness-start", "stop", "mutation", "decision", "quiescence"]
)
def test_wrong_uid_or_stale_receipt_stops_before_the_next_forward_stage(field):
    proof = evidence()
    if field == "witness-start":
        first = proof.witness_starts[0].model_copy(update={"executor_uid": "wrong"})
        proof = proof.model_copy(
            update={"witness_starts": (first, proof.witness_starts[1])}
        )
        last = "arm"
    else:
        part = getattr(proof, field)
        proof = proof.model_copy(
            update={field: part.model_copy(update={"executor_uid": "wrong"})}
        )
        last = {
            "stop": "stop",
            "mutation": "mutate",
            "decision": "recheck",
            "quiescence": "quiesce",
        }[field]
    io = FakeBoundaryIO(proof)
    result = runner.run_case(proof.scope, io)
    assert result.verdict == "FAIL"
    assert io.calls == STAGES[: STAGES.index(last) + 1] + ["revoke", "cleanup"]
    assert len(result.errors) == 1
    assert result.errors[0].endswith(": BoundaryDenied"), (
        "the report must retain the ownership-boundary refusal"
    )


def test_uncalibrated_stop_cannot_authorize_the_ownership_mutation():
    proof = evidence()
    proof = proof.model_copy(
        update={
            "stop": proof.stop.model_copy(
                update={"witness_start_sha256": ("0" * 64, "1" * 64)}
            )
        }
    )
    io = FakeBoundaryIO(proof)
    result = runner.run_case(proof.scope, io)
    assert result.errors == ["physical-stop: BoundaryDenied"]
    assert io.calls == ["preflight", "arm", "stop", "revoke", "cleanup"]


def test_physical_no_action_cannot_be_asserted_by_successful_model_cleanup():
    proof = evidence()
    action = evidence("unchanged-owner").witness_ends[0].actions
    first = proof.witness_ends[0].model_copy(update={"actions": action})
    proof = proof.model_copy(update={"witness_ends": (first, proof.witness_ends[1])})
    io = FakeBoundaryIO(proof)
    result = runner.run_case(proof.scope, io)
    assert io.calls == STAGES
    assert result.verdict == "FAIL"
    assert result.evidence == proof
    assert result.errors == [
        "hardware execution crossed a refused STOP ownership boundary"
    ]


def test_cleanup_refusal_does_not_discard_physical_evidence():
    proof = evidence()
    proof = proof.model_copy(
        update={
            "cleanup": proof.cleanup.model_copy(
                update={"owned_resources_absent": False}
            )
        }
    )
    result = runner.run_case(proof.scope, FakeBoundaryIO(proof))
    assert result.verdict == "FAIL"
    assert result.evidence == proof
    assert result.cleanup == proof.cleanup
    assert all(
        error
        == "cleanup did not prove revocation, drainage and owned-resource restoration"
        for error in result.errors
    ), "cleanup refusal must retain the specific restoration-proof error"


def test_unstructured_witness_output_cannot_be_promoted_to_evidence():
    proof = evidence()
    io = FakeBoundaryIO(proof)
    io.finish_witnesses = lambda scope, quiet: ({"counter": 0}, {"counter": 0})
    result = runner.run_case(proof.scope, io)
    assert result.verdict == "FAIL"
    assert result.evidence is None
    assert result.errors == ["receipt-validation: ValidationError"]
