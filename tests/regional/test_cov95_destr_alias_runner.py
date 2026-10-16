from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr020_verdicts as verdicts
from scripts.e2e.regional import run_destr020_identity_mismatch_isolation as case
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_alias import (
    AliasHarness,
    accepted,
    eligible_item,
    nothing_names,
)
from tests.regional._cov95_destr_warm import NOW

NO_ALIAS_AGENT = {
    "found": False,
    "before": None,
    "lifecycle_state": None,
    "transition_id": None,
    "revoked_by_case": False,
}


def _alias_agent_evidence(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "cases" / case.CASE_ID / "alias-agent-cleanup.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def test_alias_case_proves_absence_and_bounded_escalation_without_node_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    preflight = h.plan(tmp_path)
    assert preflight["errors"] == [], preflight
    h.chain_pending = 1
    code, report = h.execute(tmp_path)
    assert code == 0 and report["errors"] == report["cleanup"]["errors"] == [], report
    assert h.posted["node_id"] == report["alias"] == h.alias, report
    assert h.posted["cluster_id"] == "cluster-a" and h.posted["product"] == "A100", (
        h.posted
    )
    assert h.posted["affected_workload_ids"] == [], h.posted
    assert h.clock.elapsed >= 35, h.clock.elapsed
    assert report["leftover_records"], report
    assert all(
        detail["args"][0] == "get" for name, detail in h.calls if name == "node.present"
    ), h.calls
    # The cleanup asked the CPU Pod to revoke the alias's Agent under the
    # case's own transition -- before it proved the fleet and the runtime --
    # and kept the read-back as evidence: nothing had registered one.
    run_id = case.run_identity(tmp_path, 1)
    [revoke] = [detail for name, detail in h.calls if name == "alias.agent.cleanup"]
    assert revoke == (
        "cluster-a",
        h.alias,
        verdicts.alias_cleanup_transition(run_id),
        verdicts.alias_cleanup_reason(run_id, h.alias),
    ), revoke
    names = [name for name, _ in h.calls]
    revoked_at = names.index("alias.agent.cleanup")
    assert "gpu.nodes" in names[revoked_at:], "the fleet is compared after it"
    assert "runtime.verify" in names[revoked_at:], "the runtime is verified after it"
    assert report["cleanup"]["alias_agent"] == NO_ALIAS_AGENT, report["cleanup"]
    assert _alias_agent_evidence(tmp_path) == NO_ALIAS_AGENT


@pytest.mark.parametrize("lifecycle", ["ACTIVE", "DRAINING", "REVOKED"])
def test_an_alias_agent_left_behind_is_revoked_by_the_cleanup(
    lifecycle: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    drained = None if lifecycle == "ACTIVE" else "earlier/drain"
    h.cleanup_agent = {"lifecycle_state": lifecycle, "transition_id": drained}

    code, report = h.execute(tmp_path)

    run_id = case.run_identity(tmp_path, 1)
    cleanup = report["cleanup"]["alias_agent"]
    assert code == 0 and report["cleanup"]["errors"] == [], report
    assert cleanup == {
        "found": True,
        "before": lifecycle,
        "lifecycle_state": "REVOKED",
        "transition_id": drained or verdicts.alias_cleanup_transition(run_id),
        "revoked_by_case": lifecycle != "REVOKED",
    }, cleanup
    assert _alias_agent_evidence(tmp_path) == cleanup
    if lifecycle != "REVOKED":
        assert h.cleanup_agent is not None, h.calls
        assert h.cleanup_agent["transition_reason"] == (
            verdicts.alias_cleanup_reason(run_id, h.alias)
        ), "audited as the case's cleanup"


def test_an_alias_agent_the_cleanup_cannot_revoke_fails_the_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.cleanup_agent = {"lifecycle_state": "ACTIVE", "transition_id": None}
    h.cleanup_agent_returns = True

    code, report = h.execute(tmp_path)

    assert code == 1 and report["verdict"] == "FAIL", report
    assert report["cleanup"]["errors"] == [
        "alias_agent: RegionalFixtureError: the alias Agent record is 'ACTIVE' "
        "after the cleanup, not REVOKED"
    ], report["cleanup"]
    assert _alias_agent_evidence(tmp_path)["lifecycle_state"] == "ACTIVE"
    assert {"fleet_after", "runtime_identity"} <= set(report["cleanup"]), (
        "the rest of the cleanup still runs"
    )


@pytest.mark.parametrize("defect", ["node", "agent", "event", "predecessor", "tests"])
def test_alias_preflight_rejects_prior_identity_or_failed_proofs(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    if defect == "node":
        h.alias_present = True
    elif defect == "agent":
        h.alias_agent = {"lifecycle_state": "ACTIVE"}
    elif defect == "event":
        h.alias_event = {"event_id": "earlier"}
    elif defect == "predecessor":
        h.predecessor_valid = False
    else:
        h.tests_pass = False
    assert h.plan(tmp_path)["errors"], h.calls
    with pytest.raises(RegionalFixtureError, match="preflight failed"):
        h.execute(tmp_path)
    assert not h.posted, h.calls


@pytest.mark.parametrize("busy", ["queue", "command"])
def test_alias_injection_rechecks_quiet_control_plane(
    busy: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if busy == "queue":
        h.quiet_queue = {"depth": 1}
    else:
        h.quiet_commands = {"open_by_cluster": {"cluster-a": 1}}
    code, report = h.execute(tmp_path)
    assert code == 1 and report["error"], report
    assert not h.posted, h.calls
    assert "runtime_identity" in report["cleanup"], report


@pytest.mark.parametrize(
    "phase",
    [
        "metadata",
        "event.post",
        "workflow.wait",
        "commands.read",
        "chain.read",
        "provider.events",
        "runtime.verify",
        "alias.agent.cleanup",
    ],
)
def test_alias_phase_failure_is_recorded_and_cleanup_still_checks_fleet(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures[phase] = RuntimeError(f"fake failure at {phase}")
    code, report = h.execute(tmp_path)
    assert code == 1 and phase in str(report), report
    assert "fleet_after" in report["cleanup"], report


@pytest.mark.parametrize(
    "defect", ["receipt", "decision", "command", "chain", "provider"]
)
def test_alias_bad_verdict_evidence_is_never_promoted_to_pass(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "receipt":
        h.receipt["receipt"]["status"] = 500
    elif defect == "decision":
        h.state["decision"]["action"] = "RESET_GPU"
    elif defect == "command":
        h.remote_commands[0]["status"] = "SUCCEEDED"
    elif defect == "chain":
        h.chain["second_order_workflow"] = {"request_id": "unexpected"}
    else:
        h.events = [{"event_name": "BatchRebootClusterNodes"}]
    code, report = h.execute(tmp_path)
    assert code == 1 and (report.get("errors") or report.get("error")), report
    if defect == "chain":
        assert h.clock.elapsed == 0, h.clock.elapsed
        assert any("second-order" in error for error in report["errors"]), report


def test_alias_cannot_pass_when_real_node_scheduling_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    original = h.regional.post_xid_event

    def post(payload: dict[str, Any]) -> dict[str, Any]:
        result = original(payload)
        h.node["unschedulable"] = True
        return result

    monkeypatch.setattr(h.regional, "post_xid_event", post)
    code, report = h.execute(tmp_path)
    assert code == 1 and report["cleanup"]["errors"], report


def test_alias_expired_window_and_abort_both_preserve_no_mutation_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    code, report = h.execute(tmp_path, seconds=-1)
    assert code == 1 and "window ended" in report["error"], report
    assert not h.posted, h.calls
    h.failures["event.post"] = RegionalFixtureAbort(2)
    with pytest.raises(RegionalFixtureAbort):
        h.execute(tmp_path)
    assert any(name == "runtime.verify" for name, _ in h.calls), h.calls


def test_alias_plan_reference_uid_drift_refuses_before_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.node["uid"] = "replacement"
    with pytest.raises(RegionalFixtureError, match="plan drifted"):
        h.execute(tmp_path)
    assert not h.posted, h.calls


# --------------------------------------------------------------------------- #
# The alias's departure: the administrator's exit, taken by the case
# --------------------------------------------------------------------------- #
GUARD_REASON = "agent heartbeat is younger than the 10-minute departure guard"


def _departed_evidence(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "cases" / case.CASE_ID / case.DEPARTED_EVIDENCE
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _admin_calls(h: AliasHarness) -> list[tuple[str, Any]]:
    return [(name, detail) for name, detail in h.calls if name.startswith("admin.")]


def _exit_shape(h: AliasHarness, argv: list[str]) -> bool:
    """Every command is the ledger's exit for the alias: nothing weaker."""

    head = [
        "workflow-reconcile",
        "--state-dir",
        str(h.settings.admin_state_dir),
        "--retire-departed-agents",
        "--node",
        h.alias,
        "--accept-unknown-provider-identity",
        "--reason",
    ]
    return argv[: len(head)] == head and len(argv) in {len(head) + 2, len(head) + 3}


@pytest.mark.parametrize("ends", ["CLEANED", "gone"])
def test_the_cleanup_takes_the_departed_node_exit_for_its_own_alias(
    ends: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.departed.cycle = 60
    h.departed.ends = ends

    code, report = h.execute(tmp_path)

    run_id = case.run_identity(tmp_path, 1)
    state_dir = h.settings.admin_state_dir
    reason = verdicts.departed_alias_reason(run_id, h.alias)
    reference = verdicts.departed_alias_reference(run_id)
    assert code == 0 and report["cleanup"]["errors"] == [], report["cleanup"]
    # The ledger's own exit through the state dir's deploy-host CLI: the dry
    # run first, then the apply once, under the case's reference.
    assert _admin_calls(h) == [
        (
            "admin.dry-run",
            case.departed_exit_argv(state_dir, h.alias, reason, reference=None),
        ),
        (
            "admin.apply",
            case.departed_exit_argv(state_dir, h.alias, reason, reference=reference),
        ),
    ], h.calls
    [(_, dry_run), (_, apply)] = _admin_calls(h)
    assert dry_run[-1] == "--dry-run" and "--reference" not in dry_run, dry_run
    assert apply[-2:] == ["--reference", reference] and "--dry-run" not in apply
    assert reference == f"{case.CASE_ID}/{run_id}/departed-alias", reference
    names = [name for name, _ in h.calls]
    assert names.index("admin.dry-run") > names.index("runtime.verify"), (
        "the exit is the cleanup's last step"
    )
    evidence = _departed_evidence(tmp_path)
    assert evidence["outcome"] == "accepted", evidence
    assert evidence["command"] == [
        str(state_dir / "deployer-venv" / "bin" / "gpu-fault-admin")
    ], "the site's own deploy-host CLI"
    assert [item["phase"] for item in evidence["attempts"]] == ["dry-run", "apply"]
    assert evidence["ledger_before"]["ledger"]["status"] == "REFUSED", evidence
    # The cleanup's next cycle ends the line: polled, never assumed.
    assert [item["status"] for item in evidence["readback"]] == [
        "REFUSED",
        "REFUSED",
        None if ends == "gone" else "CLEANED",
    ], evidence["readback"]
    assert report["cleanup"]["alias_departure"]["outcome"] == "accepted"


def test_a_case_that_posted_nothing_records_a_no_op_departure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.quiet_queue = {"depth": 1}

    code, report = h.execute(tmp_path)

    assert code == 1 and not h.posted, report
    assert report["cleanup"]["errors"] == [], report["cleanup"]
    assert _admin_calls(h) == [], "nothing can have listed the alias"
    evidence = _departed_evidence(tmp_path)
    assert evidence["outcome"] == "no-op" and "posted no XID" in evidence["note"]
    assert evidence["attempts"] == [], evidence


def test_an_alias_nothing_ever_listed_is_a_recorded_no_op_after_the_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.departed.line = None
    h.departed.plans = [nothing_names(h.alias)]

    code, report = h.execute(tmp_path)

    assert code == 0 and report["cleanup"]["errors"] == [], report["cleanup"]
    calls = _admin_calls(h)
    assert {name for name, _ in calls} == {"admin.dry-run"}, "no line: no apply"
    # Polled for the margin: a line still to be written had three cycles.
    margin = verdicts.DEPARTED_MARGIN_SECONDS // verdicts.DEPARTED_POLL_SECONDS
    assert len(calls) == margin + 1, calls
    evidence = _departed_evidence(tmp_path)
    assert evidence["outcome"] == "no-op", evidence
    assert evidence["ledger_at_deadline"] == {"acceptance": None, "ledger": None}


@pytest.mark.parametrize("stage", ["apply", "pod", "dry-run", "plan"])
def test_a_refused_exit_fails_the_case_and_is_never_retried_weaker(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    key = f"cluster-a/{h.alias}"
    if stage == "apply":
        fragment = f"refuses ineligible Agents: {key}: an unknown provider identity"
        h.departed.applies = [("refuse", f"retire-departed-agents {fragment} ...")]
    elif stage == "pod":
        fragment = "an Agent record for the node is ACTIVE; plan again"
        h.departed.applies = [
            (
                1,
                {
                    "accepted_identities": [],
                    "failed_agents": [key],
                    "failures": {key: f"ValueError: {fragment}"},
                },
            )
        ]
    elif stage == "dry-run":
        fragment = "workflow reconcile found no Running CPU ingress Pod"
        h.departed.plans = [("refuse", fragment)]
    else:
        fragment = (
            "the node's provider identity is known; retire it on the provider's "
            "proof, without --accept-unknown-provider-identity"
        )
        h.departed.plans = [eligible_item(h.alias, eligible=False, reasons=[fragment])]

    code, report = h.execute(tmp_path)

    assert code == 1 and report["verdict"] == "FAIL", report
    [error] = report["cleanup"]["errors"]
    assert error.startswith("alias_departure: RegionalFixtureError: "), error
    assert fragment in error, error
    calls = _admin_calls(h)
    applies = [name for name, _ in calls if name == "admin.apply"]
    assert len(applies) == (1 if stage in {"apply", "pod"} else 0), calls
    assert all(_exit_shape(h, argv) for _name, argv in calls), calls
    if stage == "plan":
        assert "did not open within 180s" in error, error
    evidence = _departed_evidence(tmp_path)
    assert evidence["outcome"] == "failed" and fragment in evidence["error"]


def test_the_administrator_lock_is_waited_out_not_failed_on_sight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.departed.plans = ["lock", "lock", eligible_item(h.alias)]
    h.departed.applies = ["lock", accepted(h.alias)]

    code, report = h.execute(tmp_path)

    assert code == 0 and report["cleanup"]["errors"] == [], report["cleanup"]
    assert [name for name, _ in _admin_calls(h)] == [
        *["admin.dry-run"] * 3,
        *["admin.apply"] * 2,
    ], h.calls
    evidence = _departed_evidence(tmp_path)
    assert [item["lock_busy"] for item in evidence["attempts"]] == [
        True,
        True,
        False,
        True,
        False,
    ], evidence["attempts"]
    assert "held by pid 4242" in evidence["attempts"][0]["refusal"], evidence
    assert evidence["outcome"] == "accepted", evidence


@pytest.mark.parametrize("stuck", ["lock", "chain", "heartbeat", "ledger"])
def test_a_departure_that_cannot_clear_within_its_budget_fails_the_case(
    stuck: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if stuck == "lock":
        h.departed.plans = ["lock"]
        fragment = "administrator lock stayed held through the case's 780s budget"
    elif stuck == "chain":
        # A report chain that restarted a minute ago: the exit opens once it
        # spans the guard -- and here it never does.
        h.departed.plans = [
            eligible_item(
                h.alias,
                eligible=False,
                kubernetes_reports={"open": False, "never_reported": True},
                kubernetes_reports_read={
                    "chain_started_at": (NOW - timedelta(minutes=1)).isoformat()
                },
                reasons=["covers 60s of the 600s guard"],
            )
        ]
        fragment = "did not open within"
    elif stuck == "heartbeat":
        # Something keeps the alias's Agent speaking: its own timestamps say
        # the exit opens past the budget, so the case stops waiting.
        h.departed.plans = [
            lambda harness: eligible_item(
                harness.alias,
                eligible=False,
                retire=True,
                last_seen_at=harness.clock.now().isoformat(),
                lease_expires_at=harness.clock.now().isoformat(),
                reasons=[GUARD_REASON],
            )
        ]
        fragment = "past the case's 780s budget: " + GUARD_REASON
    else:
        h.departed.ends = "stuck"
        fragment = f"still lists {h.alias} as REFUSED (provider_identity_unknown)"

    code, report = h.execute(tmp_path)

    assert code == 1 and report["verdict"] == "FAIL", report
    [error] = report["cleanup"]["errors"]
    assert fragment in error, error
    evidence = _departed_evidence(tmp_path)
    assert evidence["outcome"] == "failed", evidence
    polls = [item for item in evidence["attempts"] if item["phase"] == "dry-run"]
    if stuck == "lock":
        budget = verdicts.DEPARTED_EXIT_BUDGET_SECONDS // verdicts.DEPARTED_POLL_SECONDS
        assert len(polls) == budget + 1, len(polls)
    elif stuck == "chain":
        # Waited for the chain, not just the margin, and never past the cap.
        margin = verdicts.DEPARTED_MARGIN_SECONDS // verdicts.DEPARTED_POLL_SECONDS
        assert margin + 1 < len(polls) <= 27, len(polls)
    elif stuck == "ledger":
        assert len(evidence["readback"]) == 11, evidence["readback"]


def test_the_wait_follows_the_alias_records_own_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case's revoke failed and the alias's Agent is still ACTIVE: the
    exit retires it itself once the record's guard and lease allow, and the
    case waits exactly that long -- past the margin, inside the budget."""

    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["alias.agent.cleanup"] = RuntimeError("revoke did not land")
    seen = NOW - timedelta(minutes=4)
    record = {
        "retire": True,
        "lifecycle_state": "ACTIVE",
        "last_seen_at": seen.isoformat(),
        "lease_expires_at": (seen + timedelta(minutes=2)).isoformat(),
    }

    def plan(harness: AliasHarness) -> dict[str, Any]:
        if harness.clock.now() < seen + timedelta(minutes=10):
            return eligible_item(
                harness.alias, eligible=False, reasons=[GUARD_REASON], **record
            )
        return eligible_item(harness.alias, **record)

    h.departed.plans = [plan]

    code, report = h.execute(tmp_path)

    assert report["cleanup"]["errors"] == [
        "alias_agent: RuntimeError: revoke did not land"
    ], report["cleanup"]
    evidence = _departed_evidence(tmp_path)
    assert evidence["outcome"] == "accepted", evidence
    polls = [item for item in evidence["attempts"] if item["phase"] == "dry-run"]
    first, last = (verdicts.parse_time(polls[index]["at"]) for index in (0, -1))
    assert first is not None and last is not None, polls
    assert last >= seen + timedelta(minutes=10), "eligible only past the guard"
    assert (
        timedelta(seconds=verdicts.DEPARTED_MARGIN_SECONDS)
        < last - first
        < timedelta(seconds=verdicts.DEPARTED_EXIT_BUDGET_SECONDS)
    ), (first, last)
    assert code == 1, "the failed revoke stays the case's FAIL"


@pytest.mark.parametrize("defect", ["unnamed", "cluster", "ledger"])
def test_a_dry_run_that_does_not_settle_the_alias_fails_closed(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "unnamed":
        h.departed.plans = [{"items": ["junk", {"node_id": "acceptance-alias-other"}]}]
        fragment = f"dry run did not name {h.alias}"
    elif defect == "cluster":
        h.departed.plans = [eligible_item(h.alias, cluster_id="cluster-b")]
        fragment = "in cluster 'cluster-b', not the case's cluster-a"
    else:
        # The ledger lists the alias, yet the exit's plan finds nothing naming
        # it: never taken as a no-op.
        h.departed.plans = [nothing_names(h.alias)]
        fragment = verdicts.NOTHING_NAMES_THE_NODE

    code, report = h.execute(tmp_path)

    assert code == 1 and report["verdict"] == "FAIL", report
    [error] = report["cleanup"]["errors"]
    assert fragment in error, error
    assert "admin.apply" not in [name for name, _ in h.calls], "never applied"
    evidence = _departed_evidence(tmp_path)
    assert evidence["outcome"] == "failed", evidence
    if defect == "ledger":
        assert evidence["ledger_at_deadline"]["ledger"]["status"] == "REFUSED"
