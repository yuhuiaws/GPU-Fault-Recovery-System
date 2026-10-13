"""A failed HA-011 receipt names what failed, not only the exception type.

Live 2026-10-08 (round 7, eks-primary): ``--execute`` failed 13 s after a
clean plan with the summary ``{"error_type": "ProofError", ...}`` and nothing
else -- the runner raises ProofError at five sites and ``identity()`` at five
more, so the operator could not tell which fired or what differed. The receipt
now carries ``error`` (redacted, bounded) and, per refusal, the facts that
moved: the plan keys that differ with both values, the preflight errors, both
intent digests, the evidence errors, the identity leaves that drifted, and a
window shortfall with its numbers.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import ha011_contracts as contracts
from scripts.e2e.regional import run_ha011_busy_cpu_takeover as runner
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.regional_commands import RegionalCommandTimeout
from tests.regional._cov95_ha011_support import (
    blocked_external_transports as blocked_external_transports,
)
from tests.regional._cov95_ha011_support import (
    deadline,
    install_kubernetes,
    settings_at,
)
from tests.regional.test_cov95_ha011_runner import plan

SHA256 = re.compile(r"[a-f0-9]{64}")
DRIFT = "approved deployment, predecessor, or resource intent drifted"
RELEASE_STATE = "preflight.identity.release_state_sha256"


def _execute(
    settings: contracts.Settings,
    root: Path,
    capsys: pytest.CaptureFixture[str],
    window: datetime | None = None,
) -> dict[str, Any]:
    assert runner.execute_case(settings, root, 1, window or deadline()) == 1, (
        "every refusal here is a failed receipt"
    )
    printed = json.loads(capsys.readouterr().out)
    receipt = root / "cases" / contracts.CASE_ID / f"{contracts.CASE_ID}.json"
    assert json.loads(receipt.read_text(encoding="utf-8")) == printed, (
        "the printed summary is the receipt"
    )
    assert (printed["verdict"], printed["status"]) == ("FAIL", "FAILED")
    return printed


def test_plan_drift_names_the_moved_fact_with_both_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The live suspect: the release-state ConfigMap re-read seconds after the
    plan carries another document, so the planned identity digest no longer
    matches -- the receipt now says which leaf moved and how."""

    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    planned = plan(settings, tmp_path)
    state = fake.object("ConfigMap", "gpu-fault-regional-release-state", business=True)
    state["data"]["state.json"] = json.dumps(
        {
            "release_id": "release-fixture",
            "phase": "complete",
            "transaction_committed": True,
            "verified_at": "later",
        }
    )

    result = _execute(settings, tmp_path, capsys)

    assert (result["error_type"], result["error"]) == ("ProofError", DRIFT)
    assert result["drift"]["paths"] == [RELEASE_STATE], result["drift"]
    before = planned["details"]["preflight"]["identity"]["release_state_sha256"]
    assert result["drift"]["planned"][RELEASE_STATE] == before
    after = result["drift"]["current"][RELEASE_STATE]
    assert SHA256.fullmatch(after) and after != before, result["drift"]
    assert "preflight_errors" not in result, "a clean preflight lists no errors"
    assert not fake.created and not fake.armed, "drift still admits nothing"


def test_an_invalid_predecessor_lists_the_preflight_errors_and_its_leaves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    plan(settings, tmp_path)
    write_json_atomic(settings.predecessor_path, {"verdict": "FAIL"})

    result = _execute(settings, tmp_path, capsys)

    assert result["error"] == DRIFT
    assert result["preflight_errors"] == [
        "formal predecessor is not valid for this release and cluster"
    ]
    assert "preflight.errors" in result["drift"]["paths"], result["drift"]
    assert any(
        path.startswith("preflight.predecessor.") for path in result["drift"]["paths"]
    ), result["drift"]["paths"]
    assert not fake.created, "an invalid predecessor admits nothing"


@pytest.mark.parametrize(
    "window,reason",
    [
        (datetime(2000, 1, 1, tzinfo=timezone.utc), "480s required"),
        (datetime(2099, 1, 1), "carries no timezone"),
    ],
)
def test_a_window_that_cannot_cover_the_case_says_what_was_measured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    window: datetime,
    reason: str,
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    plan(settings, tmp_path)
    reads_before = len(fake.calls)

    result = _execute(settings, tmp_path, capsys, window)

    assert result["error_type"] == "RuntimeError"
    assert "approved maintenance window cannot cover" in result["error"]
    assert reason in result["error"] and window.isoformat() in result["error"]
    assert len(fake.calls) == reads_before, "a short window is refused before any read"


def test_probe_source_drift_records_both_intent_digests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    plan(settings, tmp_path)
    original = runner.source_bundle
    calls = 0

    def drifting(root: Path) -> str:
        nonlocal calls
        calls += 1
        return original(root) if calls == 1 else "different-bundle"

    monkeypatch.setattr(runner, "source_bundle", drifting)

    result = _execute(settings, tmp_path, capsys)

    assert result["error"] == "probe source changed after execution preflight"
    source = result["probe_source"]
    assert SHA256.fullmatch(source["preflight_intent_sha256"]), source
    assert SHA256.fullmatch(source["current_intent_sha256"]), source
    assert source["preflight_intent_sha256"] != source["current_intent_sha256"]
    assert not fake.created, "changed source is rejected before namespace creation"


def test_an_incomplete_proof_lists_the_evidence_errors_without_the_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    plan(settings, tmp_path)
    original = fake.dispatch

    def extra(args: list[str], namespace: str | None, text: str | None) -> str:
        value = original(args, namespace, text)
        if args[0] == "logs":
            proof = json.loads(value)
            proof["unexpected"] = "PROTECTED_TEST_PAYLOAD"
            return json.dumps(proof)
        return value

    fake.dispatch = extra

    result = _execute(settings, tmp_path, capsys)

    assert result["error"] == (
        "isolated runtime did not prove the complete busy takeover contract"
    )
    assert result["evidence_errors"] == [
        "probe contains missing or unrecognized evidence fields"
    ], "the errors name our invariants, never the probe's fields"
    assert "probe" not in result, "invalid raw evidence is never persisted"
    assert "PROTECTED_TEST_PAYLOAD" not in json.dumps(result), (
        "unapproved probe data must not leak through the error record"
    )
    assert fake.deleted, "an invalid probe still requires verified owned cleanup"


def test_identity_drift_after_the_proof_lists_the_moved_leaves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    plan(settings, tmp_path)

    def replace_business(args: list[str]) -> None:
        if fake.armed and args[:2] == ["get", "Deployment"]:
            fake.object("Deployment", contracts.DEPLOYMENT, business=True)["metadata"][
                "uid"
            ] = "replaced"

    fake.before = replace_business

    result = _execute(settings, tmp_path, capsys)

    assert result["error"] == (
        "business deployment identity drifted during isolated acceptance"
    )
    assert result["identity_drift"] == {
        "paths": ["deployment_uid"],
        "planned": {"deployment_uid": "business-deploy"},
        "current": {"deployment_uid": "replaced"},
    }
    assert fake.deleted, "post-proof drift still requires owned isolated cleanup"


def test_a_refusal_inside_the_identity_read_is_named_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Half the ProofError sites live in identity(); a business Pod that
    stopped being Ready between plan and execute never reaches the drift
    comparison, and the receipt used to look exactly like a drift."""

    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    plan(settings, tmp_path)

    def unready(args: list[str]) -> None:
        if args[:2] == ["get", "pods"]:
            fake.business_pods[0]["status"]["containerStatuses"][0]["ready"] = False

    fake.before = unready

    result = _execute(settings, tmp_path, capsys)

    assert (result["error_type"], result["error"]) == (
        "ProofError",
        "business worker Pod readiness or image drifted",
    )
    assert "drift" not in result, "the read failed before any comparison"
    assert not fake.created, "an unreadable identity admits nothing"


def test_a_cleanup_failure_records_its_message_beside_its_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    plan(settings, tmp_path)

    def refuse_delete(args: list[str]) -> None:
        if args[0] == "delete":
            raise RegionalCommandTimeout(args, 1)

    fake.before = refuse_delete

    result = _execute(settings, tmp_path, capsys)

    assert result["cleanup_error_type"] == "RegionalCommandTimeout"
    assert isinstance(result["cleanup_error"], str), result
    assert "error" not in result, (
        "a complete proof whose cleanup failed has no runtime error"
    )
    assert fake.armed and not fake.deleted, "a timeout is not a completed deletion"
