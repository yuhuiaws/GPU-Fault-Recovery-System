"""GF-REGIONAL-BOOT-038 verdict and orchestration: every path, no cluster access.

The live run (``run_boot038_cross_cluster_membership``) feeds already-collected
evidence to ``boot038_membership``; these tests pin each verdict the brief
names by building that evidence directly and by driving ``run_membership`` with
a fake driver, so none of them touches a cluster.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.e2e.regional import boot038_membership as m
from scripts.e2e.regional import run_boot038_cross_cluster_membership as runner
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.site_profile import SITE_PROFILE_ENV


def _all_surfaces(value: bool) -> dict[str, bool]:
    return {name: value for name in m.FINDING_SURFACES}


def _pass_evidence(verb: str = m.REMOVE_VERB, **overrides: Any) -> m.VerbEvidence:
    base = m.VerbEvidence(
        verb=verb,
        window=m.WINDOW_KUBELET,
        exit_code=0,
        saw_fail_cluster=False,
        saw_rollback_cluster=False,
        window_opened=True,
        other_node_not_ready_before_verify=True,
        finding_surfaces=_all_surfaces(True),
        registry_convergence_pass=True,
        restore_ok=True,
        resume_cleanup_called=False,
    )
    return replace(base, **overrides)


# --- per-verb verdict --------------------------------------------------------


def test_a_clean_verb_with_the_finding_on_all_four_surfaces_passes() -> None:
    verdict, reasons = m.judge_verb(_pass_evidence())
    assert verdict == m.VERDICT_PASS, reasons


def test_a_missed_window_is_inconclusive_not_a_missing_finding_failure() -> None:
    # The node never went NotReady before the verify, so no finding could be
    # produced: that is INCONCLUSIVE, never FAIL on the absent finding.
    evidence = _pass_evidence(
        other_node_not_ready_before_verify=False, finding_surfaces=_all_surfaces(False)
    )
    verdict, reasons = m.judge_verb(evidence)
    assert verdict == m.VERDICT_INCONCLUSIVE
    assert any("window missed" in reason for reason in reasons), reasons


def test_a_window_that_never_opened_is_inconclusive() -> None:
    evidence = _pass_evidence(
        window_opened=False,
        other_node_not_ready_before_verify=False,
        finding_surfaces=_all_surfaces(False),
    )
    verdict, reasons = m.judge_verb(evidence)
    assert verdict == m.VERDICT_INCONCLUSIVE
    assert any("not opened" in reason for reason in reasons), reasons


def test_a_rollback_fails_the_verb() -> None:
    verdict, reasons = m.judge_verb(_pass_evidence(saw_rollback_cluster=True))
    assert verdict == m.VERDICT_FAIL
    assert any("rollback-cluster" in reason for reason in reasons), reasons


def test_a_non_zero_exit_fails_the_verb() -> None:
    verdict, reasons = m.judge_verb(_pass_evidence(exit_code=1))
    assert verdict == m.VERDICT_FAIL
    assert any("exited 1" in reason for reason in reasons), reasons


@pytest.mark.parametrize("surface", m.FINDING_SURFACES)
def test_a_missing_finding_surface_fails_when_the_window_held(surface: str) -> None:
    surfaces = _all_surfaces(True)
    surfaces[surface] = False
    verdict, reasons = m.judge_verb(_pass_evidence(finding_surfaces=surfaces))
    assert verdict == m.VERDICT_FAIL
    assert any(surface in reason for reason in reasons), (surface, reasons)


def test_a_lost_registry_convergence_fails_the_verb() -> None:
    verdict, reasons = m.judge_verb(_pass_evidence(registry_convergence_pass=False))
    assert verdict == m.VERDICT_FAIL
    assert any("registry convergence" in reason for reason in reasons), reasons


def test_a_failed_restore_fails_the_verb_and_names_resume_cleanup() -> None:
    evidence = _pass_evidence(restore_ok=False, resume_cleanup_called=True)
    verdict, reasons = m.judge_verb(evidence)
    assert verdict == m.VERDICT_FAIL
    assert any("resume_cleanup invoked" in reason for reason in reasons), reasons


def test_a_status_read_error_fails_the_verb_rather_than_reading_no_finding() -> None:
    # A status rc != 0 is an evidence error; the verb cannot pass even with the
    # window held and the other three surfaces recorded.
    evidence = _pass_evidence(status_error="status --full exited 2")
    verdict, reasons = m.judge_verb(evidence)
    assert verdict == m.VERDICT_FAIL
    assert any("status read did not succeed" in reason for reason in reasons), reasons


def test_notready_before_verify_begin_gives_a_judged_verb_the_finding_gates() -> None:
    # The poller saw NotReady before the verify began and the finding appeared
    # on all four surfaces: a judged verb (PASS), not INCONCLUSIVE.
    evidence = _pass_evidence(other_node_not_ready_before_verify=True)
    verdict, _ = m.judge_verb(evidence)
    assert verdict == m.VERDICT_PASS, "NotReady before verify -> judged"


def test_notready_only_after_verify_begin_gives_inconclusive() -> None:
    # The node was still Ready when the verify began (NotReady came later), so
    # no finding could appear: INCONCLUSIVE (window missed), never FAIL.
    evidence = _pass_evidence(
        other_node_not_ready_before_verify=False, finding_surfaces=_all_surfaces(False)
    )
    verdict, reasons = m.judge_verb(evidence)
    assert verdict == m.VERDICT_INCONCLUSIVE
    assert any("window missed" in reason for reason in reasons), reasons


def test_window_none_with_a_real_departure_passes_when_the_finding_is_recorded() -> (
    None
):
    evidence = _pass_evidence(
        window=m.WINDOW_NONE,
        window_opened=False,
        other_node_not_ready_before_verify=True,
        window_observation="ip-10-0-44-18 ready=Unknown",
    )
    verdict, reasons = m.judge_verb(evidence)
    assert verdict == m.VERDICT_PASS, reasons


def test_window_none_without_a_departure_is_inconclusive_and_records_what_it_saw() -> (
    None
):
    evidence = _pass_evidence(
        window=m.WINDOW_NONE,
        window_opened=False,
        other_node_not_ready_before_verify=None,
        finding_surfaces=_all_surfaces(False),
        window_observation="ip-10-0-44-18 ready=True",
    )
    verdict, reasons = m.judge_verb(evidence)
    assert verdict == m.VERDICT_INCONCLUSIVE
    assert any("ready=True" in reason for reason in reasons), reasons


# --- after-run ---------------------------------------------------------------


def _clean_after(**overrides: Any) -> m.AfterRun:
    base = m.AfterRun(
        other_node_ready=True,
        other_agent_retired=False,
        other_node_ledger_mail=(),
        target_departed_mail=(),
        target_agents_all_active=True,
        non_terminal_incident=False,
        window_residue=False,
    )
    return replace(base, **overrides)


def test_a_clean_after_run_passes() -> None:
    ok, reasons = m.judge_after_run(_clean_after(), window=m.WINDOW_KUBELET)
    assert ok, reasons


def test_departed_mail_for_a_target_node_fails_the_after_run() -> None:
    after = _clean_after(
        target_departed_mail=("[GPU node departed: operator action needed] hp: n1",)
    )
    ok, reasons = m.judge_after_run(after, window=m.WINDOW_KUBELET)
    assert not ok, reasons
    assert any("b249" in reason for reason in reasons), reasons


def test_a_retired_other_agent_fails_the_after_run() -> None:
    ok, reasons = m.judge_after_run(
        _clean_after(other_agent_retired=True), window=m.WINDOW_KUBELET
    )
    assert not ok, reasons
    assert any("retired" in reason for reason in reasons), reasons


def test_ledger_mail_for_the_other_node_fails_the_after_run() -> None:
    after = _clean_after(other_node_ledger_mail=("ip-10-0-44-18: CLEANED",))
    ok, reasons = m.judge_after_run(after, window=m.WINDOW_KUBELET)
    assert not ok, reasons
    assert any("ledger" in reason for reason in reasons), reasons


def test_a_window_residue_only_fails_plan_a() -> None:
    residue = _clean_after(window_residue=True)
    ok_a, _ = m.judge_after_run(residue, window=m.WINDOW_KUBELET)
    ok_b, _ = m.judge_after_run(residue, window=m.WINDOW_NONE)
    assert not ok_a and ok_b, "residue fails plan A only"


# --- case verdict ------------------------------------------------------------


def test_case_verdict_is_fail_when_the_after_run_fails() -> None:
    assert (
        m.case_verdict({m.REMOVE_VERB: m.VERDICT_PASS}, after_ok=False)
        == m.VERDICT_FAIL
    )


def test_case_verdict_is_inconclusive_when_a_verb_is_and_nothing_failed() -> None:
    verdicts = {m.REMOVE_VERB: m.VERDICT_PASS, m.JOIN_VERB: m.VERDICT_INCONCLUSIVE}
    assert m.case_verdict(verdicts, after_ok=True) == m.VERDICT_INCONCLUSIVE


def test_case_verdict_passes_only_when_both_verbs_pass() -> None:
    verdicts = {m.REMOVE_VERB: m.VERDICT_PASS, m.JOIN_VERB: m.VERDICT_PASS}
    assert m.case_verdict(verdicts, after_ok=True) == m.VERDICT_PASS


# --- the four surfaces -------------------------------------------------------


def test_finding_surfaces_reads_the_other_cluster_on_each_surface() -> None:
    other = "gpu-eks-cluster"
    finding = {"check": "control_api", "cluster_id": other, "finding": "coverage drift"}
    surfaces = m.finding_surfaces(
        other_cluster_id=other,
        result={"not_gating_findings": [finding]},
        verified_state={"evidence": {"VERIFIED": {"not_gating_findings": [finding]}}},
        record={"not_gating_findings": [finding]},
        status_lines=[
            "gpu-fault-admin status: join-cluster hp: not gating: "
            f"{other} control_api: coverage drift"
        ],
    )
    assert surfaces == _all_surfaces(True)


def test_finding_surfaces_ignores_a_finding_for_a_third_cluster() -> None:
    surfaces = m.finding_surfaces(
        other_cluster_id="gpu-eks-cluster",
        result={"not_gating_findings": [{"cluster_id": "some-other"}]},
        verified_state={},
        record=None,
        status_lines=["not gating: some-other control_api: x"],
    )
    assert surfaces == _all_surfaces(False)


# --- narration grammar -------------------------------------------------------


def test_removal_opens_on_remove_cluster_begin_then_verify_begin_and_end() -> None:
    narration = m.VerbNarration()
    stamp = "2026-11-04T18:47:47Z"
    assert (
        narration.observe(
            m.REMOVE_VERB, f"release-begin {stamp} mode=remove-cluster dry_run=false"
        )
        == "open"
    )
    # Its later release-end must not re-open (open already seen); it is ignored.
    assert (
        narration.observe(
            m.REMOVE_VERB,
            f"release-end {stamp} mode=remove-cluster exit_code=0 total=113.2s",
        )
        is None
    ), "the removal opens on begin, not end"
    assert (
        narration.observe(
            m.REMOVE_VERB, f"release-begin {stamp} mode=verify dry_run=false"
        )
        == "verify-begin"
    )
    assert (
        narration.observe(
            m.REMOVE_VERB, f"release-end {stamp} mode=verify exit_code=0 total=31.2s"
        )
        == "verify-end"
    )
    assert not narration.saw_fail_cluster and not narration.saw_rollback_cluster, (
        "a clean verb"
    )


def test_join_opens_on_the_last_wave_start_not_an_earlier_one() -> None:
    narration = m.VerbNarration()
    wave1 = (
        "fleet-wave 2026-11-04T19:01:43Z cluster=hp phase=join wave=1/2 "
        "ready=0/4 nodes=n total=246.1s"
    )
    wave2 = (
        "fleet-wave 2026-11-04T19:02:54Z cluster=hp phase=join wave=2/2 "
        "ready=1/4 nodes=n total=316.5s"
    )
    assert narration.observe(m.JOIN_VERB, wave1) is None, "wave 1/2 is not the last"
    assert narration.observe(m.JOIN_VERB, wave2) == "open", "wave 2/2 opens"
    assert narration.last_wave_start is not None, "the last wave start was recorded"


def test_join_opens_on_a_single_wave_start() -> None:
    narration = m.VerbNarration()
    wave = (
        "fleet-wave 2026-11-04T19:01:43Z cluster=hp phase=join wave=1/1 "
        "ready=0/8 nodes=n total=200.0s"
    )
    assert narration.observe(m.JOIN_VERB, wave) == "open", "a single wave=1/1 opens"


def test_narration_records_fail_and_rollback() -> None:
    narration = m.VerbNarration()
    narration.observe(
        m.JOIN_VERB,
        "release-begin 2026-11-04T12:20:50Z mode=fail-cluster dry_run=false",
    )
    narration.observe(
        m.JOIN_VERB,
        "release-begin 2026-11-04T12:32:04Z mode=rollback-cluster dry_run=false",
    )
    assert narration.saw_fail_cluster and narration.saw_rollback_cluster, (
        "both narrated"
    )


# --- real-log contract (line shapes copied from the S69 logs, no secrets) ----

# The gpu-fault-admin wrapper tees the verb's stdout (the echoed verify report,
# every `+ aws` JSON output, per-check details, the verb result) and the
# release narration (stderr) into one stream; these fixtures reproduce that
# shape, including the verify report whose closing `}` is glued to the stderr
# `release-end mode=verify` line with no newline between them.
_REMOVE_STREAM = [
    "+ aws sagemaker describe-cluster --cluster-name hp --region us-west-2\n",
    "release-begin 2026-11-04T18:37:33Z mode=drain-cluster dry_run=false\n",
    "release-end 2026-11-04T18:37:47Z mode=drain-cluster exit_code=0 total=14.9s\n",
    "release-begin 2026-11-04T18:47:47Z mode=remove-cluster dry_run=false\n",
    "release-end 2026-11-04T18:49:40Z mode=remove-cluster exit_code=0 total=113.2s\n",
    "release-begin 2026-11-04T18:50:37Z mode=sync-state dry_run=false\n",
    "release-end 2026-11-04T18:51:27Z mode=sync-state exit_code=0 total=49.9s\n",
    "release-begin 2026-11-04T18:51:27Z mode=verify dry_run=false\n",
    "{\n",
    '  "checks": [],\n',
    '  "healthy": true,\n',
    '  "mode": "verify"\n',
    "}release-end 2026-11-04T18:52:00Z mode=verify exit_code=0 total=32.9s\n",
    "+ aws sagemaker list-clusters --region us-west-2 --output json\n",
    "{\n",
    '  "cluster_id": "hp-cluster",\n',
    '  "cpu_control_plane": "PRESERVED",\n',
    '  "gpu_cluster": "PRESERVED",\n',
    '  "phase": "COMPLETED",\n',
    '  "remaining_cluster_ids": ["gpu-eks-cluster"]\n',
    "}\n",
]
# wave 1/2 start and done, then wave 2/2 START (the open), then its done, the
# join end and the glued verify report / release-end.
_JOIN_WAVES = [
    "fleet-wave 2026-11-04T19:01:43Z cluster=hp phase=join wave=1/2 ready=0/4 "
    "nodes=n total=246.1s\n",
    "fleet-wave-done 2026-11-04T19:02:54Z cluster=hp wave=1/2 ready=1/4 "
    "nodes=n total=314.0s\n",
    "fleet-wave 2026-11-04T19:02:54Z cluster=hp phase=join wave=2/2 ready=1/4 "
    "nodes=n total=316.5s\n",
    "fleet-wave-done 2026-11-04T19:04:07Z cluster=hp wave=2/2 ready=4/4 "
    "nodes=n total=374.6s\n",
    "release-end 2026-11-04T19:06:38Z mode=join-cluster exit_code=0 total=540.9s\n",
    "release-begin 2026-11-04T19:06:48Z mode=verify dry_run=false\n",
    "}release-end 2026-11-04T19:07:38Z mode=verify exit_code=0 total=49.1s\n",
]
_ROLLBACK_TAIL = [
    "release-begin 2026-11-04T12:20:50Z mode=fail-cluster dry_run=false\n",
    "release-end 2026-11-04T12:20:57Z mode=fail-cluster exit_code=0 total=7.6s\n",
    "release-begin 2026-11-04T12:32:04Z mode=rollback-cluster dry_run=false\n",
    "release-end 2026-11-04T12:32:16Z mode=rollback-cluster exit_code=0 total=11.6s\n",
]


def _follow(verb: str, lines: list[str]) -> tuple[m.VerbNarration, list[str]]:
    narration = m.VerbNarration()
    tokens = [token for line in lines if (token := narration.observe(verb, line))]
    return narration, tokens


def test_removal_stream_opens_once_then_verify_begin_and_end_in_order() -> None:
    narration, tokens = _follow(m.REMOVE_VERB, _REMOVE_STREAM)
    assert tokens == ["open", "verify-begin", "verify-end"], tokens
    assert not narration.saw_fail_cluster, "a clean removal"
    assert not narration.saw_rollback_cluster, "a clean removal"


def test_join_stream_opens_on_the_last_wave_start_and_detects_glued_verify_end() -> (
    None
):
    narration, tokens = _follow(m.JOIN_VERB, _JOIN_WAVES)
    # Only wave 2/2's START opens (wave 1/2 start and both done lines do not);
    # the glued "}release-end mode=verify" is a verify-end.
    assert tokens == ["open", "verify-begin", "verify-end"], tokens
    assert narration.last_wave_start is not None, "the last wave start was recorded"


def test_rolled_back_join_stream_sets_fail_and_rollback() -> None:
    narration, _ = _follow(m.JOIN_VERB, _JOIN_WAVES + _ROLLBACK_TAIL)
    assert narration.saw_fail_cluster, "fail-cluster narrated"
    assert narration.saw_rollback_cluster, "rollback-cluster narrated"


def test_verb_result_is_the_last_signature_object_not_the_verify_report() -> None:
    result = m.verb_result_from_output(_REMOVE_STREAM)
    assert result is not None, "a result is found"
    assert result.get("phase") == "COMPLETED", "the verb result, not the report"
    assert result.get("remaining_cluster_ids") == ["gpu-eks-cluster"], result


def test_verb_result_carries_not_gating_findings_when_the_window_held() -> None:
    stream = [
        "{\n",
        '  "checks": [], "healthy": true, "mode": "verify"\n',
        "}release-end 2026-11-04T12:20:40Z mode=verify exit_code=0 total=50.2s\n",
        "{\n",
        '  "cpu_control_plane": "PRESERVED",\n',
        '  "gpu_cluster": "PRESERVED",\n',
        '  "phase": "COMPLETED",\n',
        '  "not_gating_findings": [{"check": "control_api", '
        '"cluster_id": "gpu-eks-cluster", "finding": "coverage drift"}]\n',
        "}\n",
    ]
    result = m.verb_result_from_output(stream)
    assert m.result_findings(result)[0]["cluster_id"] == "gpu-eks-cluster", result


def test_verb_result_falls_back_when_the_verb_printed_no_result() -> None:
    # A rolled-back verb prints no result object; the fallback is the last
    # decodable object (the verify report), and its result surface is empty.
    stream = ["{\n", '  "checks": [], "healthy": false, "mode": "verify"\n', "}\n"]
    result = m.verb_result_from_output(stream)
    assert result is not None and result.get("mode") == "verify", result
    assert m.result_findings(result) == [], "a report carries no top-level findings"


# --- record readers ----------------------------------------------------------


def test_departed_mail_for_matches_both_subjects_and_the_node() -> None:
    notifications = [
        {
            "subject": "[GPU node departed: operator action needed] hp: n-target",
            "incident_id": "departed-node-n-target",
        },
        {
            "subject": "[NOTICE][GPU node departed: records ended] hp: n-other",
            "incident_id": "x",
        },
        {"subject": "unrelated", "incident_id": "y"},
    ]
    hits = m.departed_mail_for(notifications, ["n-target"])
    assert len(hits) == 1 and "n-target" in hits[0]


def test_ledger_mail_for_only_counts_cleaned_or_refused() -> None:
    ledger = [
        {"node_id": "n", "status": "PENDING"},
        {"node_id": "n", "status": "CLEANED"},
        {"node_id": "other", "status": "REFUSED"},
    ]
    assert m.ledger_mail_for(ledger, "n") == ["n: CLEANED"]


def test_other_agent_retired_detects_absent_or_revoked() -> None:
    agents = [{"node_id": "n", "lifecycle_state": "ACTIVE"}]
    assert m.other_agent_retired(agents, "n") is False
    assert m.other_agent_retired(agents, "gone") is True
    assert m.other_agent_retired(
        [{"node_id": "n", "lifecycle_state": "REVOKED"}], "n"
    ), "REVOKED counts as retired"


def test_agents_all_active_requires_every_agent_active() -> None:
    assert m.agents_all_active([{"lifecycle_state": "ACTIVE"}]), "one ACTIVE agent"
    assert not m.agents_all_active([{"lifecycle_state": "DRAINING"}]), (
        "DRAINING is not active"
    )
    assert not m.agents_all_active([]), "no agents is not active"


def test_generation_probe_reads_the_last_json_line() -> None:
    assert m.generation_is_b250(
        'noise\n{"b250": true, "record": "membership-verify/findings.json"}'
    ), "the last json line marks b250"
    assert not m.generation_is_b250(
        '{"b250": false, "error": "ModuleNotFoundError"}'
    ), "an import failure is pre-b250"
    assert not m.generation_is_b250("not json at all"), "non-json is pre-b250"


def test_verified_state_for_finds_the_matching_removal_and_join(tmp_path: Path) -> None:
    state = tmp_path
    remove = state / "remove-cluster" / "abc"
    remove.mkdir(parents=True)
    (remove / "state.json").write_text(
        '{"cluster_id": "hp", "phase": "COMPLETED", "evidence": {"VERIFIED": '
        '{"not_gating_findings": [{"cluster_id": "gpu-eks-cluster"}]}}}',
        encoding="utf-8",
    )
    found = m.verified_state_for(state, m.REMOVE_VERB, "hp")
    assert m.verified_evidence_findings(found)[0]["cluster_id"] == "gpu-eks-cluster"
    join = state / "join-cluster" / "3fa3"
    join.mkdir(parents=True)
    (join / "state.json").write_text(
        '{"requested_cluster_id": "hp", "evidence": {"VERIFIED": '
        '{"not_gating_findings": [{"cluster_id": "gpu-eks-cluster"}]}}}',
        encoding="utf-8",
    )
    assert m.verified_state_for(state, m.JOIN_VERB, "hp") is not None, (
        "join state found"
    )
    assert m.verified_state_for(state, m.JOIN_VERB, "absent") is None


# --- run_membership orchestration over a fake driver -------------------------


class _FakeDriver:
    def __init__(
        self,
        *,
        b250: bool = True,
        evidence: dict[str, m.VerbEvidence] | None = None,
        after: m.AfterRun | None = None,
        raise_on: str | None = None,
    ) -> None:
        self._b250 = b250
        self._evidence = evidence or {}
        self._after = after or _clean_after()
        self._raise_on = raise_on
        self.verbs_run: list[str] = []
        self.end_windows_calls = 0

    def b250_generation(self) -> bool:
        return self._b250

    def run_verb(self, verb: str) -> m.VerbEvidence:
        self.verbs_run.append(verb)
        if verb == self._raise_on:
            raise RegionalFixtureError("boom during the verb")
        return self._evidence[verb]

    def read_after(self) -> m.AfterRun:
        return self._after

    def end_windows(self) -> dict[str, Any]:
        self.end_windows_calls += 1
        return {"ended": [], "open_journals": []}


def test_an_old_product_is_not_applicable_and_mutates_nothing() -> None:
    driver = _FakeDriver(b250=False)
    record = m.run_membership(driver, window=m.WINDOW_KUBELET)
    assert record["verdict"] == m.VERDICT_NOT_APPLICABLE
    assert record["mutation_performed"] is False
    assert driver.verbs_run == []
    # No window was opened, so none is ended.
    assert driver.end_windows_calls == 0


def test_both_verbs_passing_passes_the_case_and_ends_the_window() -> None:
    driver = _FakeDriver(
        evidence={
            m.REMOVE_VERB: _pass_evidence(m.REMOVE_VERB),
            m.JOIN_VERB: _pass_evidence(m.JOIN_VERB),
        }
    )
    record = m.run_membership(driver, window=m.WINDOW_KUBELET)
    assert record["verdict"] == m.VERDICT_PASS
    assert record["mutation_performed"] is True
    assert driver.verbs_run == [m.REMOVE_VERB, m.JOIN_VERB]
    assert driver.end_windows_calls == 1


def test_a_failed_removal_skips_the_join_and_fails_the_case() -> None:
    driver = _FakeDriver(
        evidence={
            m.REMOVE_VERB: _pass_evidence(m.REMOVE_VERB, saw_rollback_cluster=True)
        }
    )
    record = m.run_membership(driver, window=m.WINDOW_KUBELET)
    assert driver.verbs_run == [m.REMOVE_VERB]
    assert record["verdict"] == m.VERDICT_FAIL
    join_entry = next(item for item in record["verbs"] if item["verb"] == m.JOIN_VERB)
    assert join_entry["verdict"] == m.VERDICT_NOT_RUN


def test_the_window_is_ended_even_when_a_verb_raises() -> None:
    driver = _FakeDriver(
        evidence={m.REMOVE_VERB: _pass_evidence(m.REMOVE_VERB)}, raise_on=m.REMOVE_VERB
    )
    with pytest.raises(RegionalFixtureError):
        m.run_membership(driver, window=m.WINDOW_KUBELET)
    assert driver.end_windows_calls == 1


# --- admin verb argv against the real CLI parser (point 2) -------------------


def _settings(tmp_path: Path, window: str = m.WINDOW_KUBELET) -> Any:
    return runner.Settings(
        state_dir=tmp_path,
        target_cluster_arn="arn:aws:eks:us-west-2:000000000000:cluster/tgt",
        target_cluster_id="hp",
        other_cluster_id="eks",
        other_node="ip-10-0-44-18",
        window=window,
        dry_run=False,
        regional=_fake_regional(tmp_path),
        host_probe_image="image@sha256:abc",
        cpu_cluster_arn="arn:aws:eks:us-west-2:000000000000:cluster/cpu",
        predecessor_id="GF-REGIONAL-BOOT-029",
        predecessor_path=None,
        run_token="deadbeef",
    )


def _verb_argv(settings: Any, verb: str) -> list[str]:
    """The runner's argv from the verb token on (dropping the binary prefix)."""

    command = runner.verb_command(settings, verb)
    return command[command.index(verb) :]


def test_remove_cluster_argv_is_accepted_by_the_real_cli_parser(tmp_path: Path) -> None:
    from gpu_fault.admin.cli import parser as cli_parser

    argv = _verb_argv(_settings(tmp_path), m.REMOVE_VERB)
    parsed = cli_parser().parse_args(argv)
    assert parsed.command == "remove-cluster", argv
    assert parsed.confirm == "REMOVE_GPU_CLUSTER", "the removal confirmation"
    assert parsed.gpu_cluster_arn.endswith("/tgt"), parsed.gpu_cluster_arn


def test_join_cluster_argv_and_profile_plan_rerun_parse(tmp_path: Path) -> None:
    from gpu_fault.admin.cli import parser as cli_parser

    argv = _verb_argv(_settings(tmp_path), m.JOIN_VERB)
    parsed = cli_parser().parse_args(argv)
    assert parsed.command == "join-cluster", argv
    assert parsed.gpu_cluster_arn == [
        "arn:aws:eks:us-west-2:000000000000:cluster/tgt"
    ], "join takes an appendable --gpu-cluster-arn"
    # The CLUSTER_PROFILE_NEW rerun flags S69 uses.
    rerun = cli_parser().parse_args(
        [*argv, "--approve-profile-plan", "a" * 64, "--reference", "BOOT038/x"]
    )
    assert rerun.approve_profile_plan == "a" * 64, "approve-profile-plan flag"
    assert rerun.reference == "BOOT038/x", "reference flag"


def test_the_verb_commands_target_the_deployer_venv_binary_when_present(
    tmp_path: Path,
) -> None:
    # admin_command prefers <state-dir>/deployer-venv/bin/gpu-fault-admin, the
    # site-bound CLI the S69 driver ran.
    from scripts.e2e.regional.admin_cli import admin_command

    binary = tmp_path / "deployer-venv" / "bin" / "gpu-fault-admin"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    assert admin_command(tmp_path) == [str(binary)], "the site-bound CLI"


# --- service window call signatures (point 3) --------------------------------


def test_service_window_fixture_accepts_the_driver_keyword_arguments() -> None:
    import inspect

    from scripts.e2e.regional.warm_spare_fixture import WarmSpareServiceFixture

    signature = inspect.signature(WarmSpareServiceFixture.__init__)
    expected = {
        "node",
        "image",
        "case_id",
        "run_id",
        "state_directory",
        "node_uid",
        "plan_sha256",
        "release_id",
        "maintenance_expires_at",
    }
    assert expected <= set(signature.parameters), (
        f"the runner passes {expected - set(signature.parameters)} the fixture lacks"
    )
    stop = inspect.signature(WarmSpareServiceFixture.stop)
    assert {"restore_seconds", "delay_seconds"} <= set(stop.parameters), stop
    for name in ("create", "restore", "cleanup", "resume_cleanup"):
        assert hasattr(WarmSpareServiceFixture, name), name
    # The restore-window bound stays inside the on-node probe's 60..600 s.
    assert 60 <= m.RESTORE_SECONDS <= 600, m.RESTORE_SECONDS
    assert m.STOP_DELAY_SECONDS + m.RESTORE_SECONDS <= m.RESTORE_CAP_SECONDS, "cap"
    assert m.SERVICE_WINDOW_FIXTURE_CASE_ID == "GF-REGIONAL-DESTR-008", (
        "the on-node probe trusts only the DESTR-008 window identity"
    )


def test_timing_constants_keep_the_stop_clear_of_evictions() -> None:
    assert m.RESTORE_SECONDS == 280, m.RESTORE_SECONDS
    assert m.STOP_DELAY_SECONDS == 15, m.STOP_DELAY_SECONDS
    assert m.RESTORE_CAP_SECONDS == 300, m.RESTORE_CAP_SECONDS
    # stop -> restore stays at/under the cap, and the cap is the 300 s eviction
    # grace after the taint, so the node restores before non-tolerating Pods go.
    assert m.STOP_DELAY_SECONDS + m.RESTORE_SECONDS <= m.RESTORE_CAP_SECONDS, "cap"


# --- status argv and rc handling (point 1) ----------------------------------


def test_status_command_includes_state_dir(tmp_path: Path) -> None:
    command = runner.status_command(tmp_path)
    assert "--state-dir" in command, command
    assert command[command.index("--state-dir") + 1] == str(tmp_path), command
    assert command[-4:] == ["status", "--full", "--state-dir", str(tmp_path)], command


def test_status_findings_nonzero_rc_is_an_evidence_error() -> None:
    lines, error = runner.status_findings(
        2, "gpu-fault-admin: ... bound to --state-dir"
    )
    assert lines == [], lines
    assert error is not None and "exited 2" in error, error


def test_status_findings_zero_rc_parses_the_not_gating_lines() -> None:
    stderr = (
        "gpu-fault-admin status: healthy: yes\n"
        "gpu-fault-admin status: finding: join-cluster hp: not gating: "
        "gpu-eks-cluster control_api: coverage drift\n"
        "gpu-fault-admin status: full JSON report: stdout\n"
    )
    lines, error = runner.status_findings(0, stderr)
    assert error is None, error
    assert len(lines) == 1 and "not gating: gpu-eks-cluster" in lines[0], lines


# --- the NotReady poller reads at verify-begin without waiting (point 2) -----


class _FakeWarm:
    def __init__(self, ready: str) -> None:
        self.ready = ready

    def node_snapshot(self, node: str) -> dict[str, str]:
        return {"ready": self.ready}


def test_poller_true_when_notready_was_seen_and_is_still_notready() -> None:
    warm = _FakeWarm("Unknown")
    poller = runner.NotReadyPoller(warm, "ip-10-0-44-18")
    # not_ready_now reads at once (no wait): the node is NotReady now, so it is
    # both seen and still NotReady -> a judged verb.
    assert poller.not_ready_now() is True, "seen and still NotReady -> judged"
    assert poller.not_ready_seen_at is not None, "the first NotReady is recorded"


def test_poller_false_when_node_was_ready_through_verify_begin() -> None:
    warm = _FakeWarm("True")
    poller = runner.NotReadyPoller(warm, "ip-10-0-44-18")
    assert poller.not_ready_now() is False, "Ready at verify -> window missed"
    assert poller.not_ready_seen_at is None, "never NotReady"


def test_poller_false_when_node_came_back_ready_by_verify_begin() -> None:
    warm = _FakeWarm("Unknown")
    poller = runner.NotReadyPoller(warm, "ip-10-0-44-18")
    assert poller.not_ready_now() is True, "seen NotReady earlier"
    warm.ready = "True"  # Ready again by the time the verify begins
    assert poller.not_ready_now() is False, "'still NotReady' is required"


# --- the runner's site lookup and configure ---------------------------------


def _site_document(tmp_path: Path, *, target: str, other: str) -> Path:
    document = {
        "apiVersion": "gpu-fault.aws/v1alpha1",
        "kind": "RegionalSite",
        "metadata": {"name": "test-site"},
        "spec": {
            "awsRegion": "us-west-2",
            "namespace": "gpu-fault-system",
            "gpuKubeconfig": str(tmp_path / "gpu.kubeconfig"),
            "cpu": {
                "kubeconfig": str(tmp_path / "cpu.kubeconfig"),
                "eksArn": "arn:aws:eks:us-west-2:000000000000:cluster/cpu",
            },
            "clusters": [
                {
                    "clusterId": target,
                    "context": "ctx-target",
                    "eksClusterArn": "arn:aws:eks:us-west-2:000000000000:cluster/tgt",
                },
                {
                    "clusterId": other,
                    "context": "ctx-other",
                    "provider": "eks-managed",
                    "eksClusterArn": "arn:aws:eks:us-west-2:000000000000:cluster/oth",
                },
            ],
        },
    }
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_configure_resolves_the_target_and_other_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _site_document(tmp_path, target="hp", other="eks")
    (tmp_path / "cpu.kubeconfig").write_text("x", encoding="utf-8")
    (tmp_path / "gpu.kubeconfig").write_text("x", encoding="utf-8")
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        yaml.safe_dump({"arguments": {"host-probe-image": "image@sha256:abc"}}),
        encoding="utf-8",
    )
    profile.chmod(0o600)
    monkeypatch.setenv(SITE_PROFILE_ENV, str(profile))
    arguments = runner.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--state-dir",
            str(tmp_path),
            "--target-cluster-arn",
            "arn:aws:eks:us-west-2:000000000000:cluster/tgt",
            "--other-cluster-id",
            "eks",
            "--other-node",
            "ip-10-0-44-18",
        ]
    )
    settings = runner.configure(arguments)
    assert settings.target_cluster_id == "hp", "the target ARN resolves to its id"
    assert settings.regional.gpu_context == "ctx-other", "the other cluster's context"
    assert settings.regional.cluster_id == "eks", "the window runs on the other cluster"
    assert settings.host_probe_image == "image@sha256:abc", "image from the profile"


def test_configure_refuses_the_target_as_its_own_other_cluster(tmp_path: Path) -> None:
    _site_document(tmp_path, target="hp", other="eks")
    arguments = runner.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--state-dir",
            str(tmp_path),
            "--target-cluster-arn",
            "arn:aws:eks:us-west-2:000000000000:cluster/tgt",
            "--other-cluster-id",
            "hp",
            "--other-node",
            "ip-10-0-44-18",
        ]
    )
    with pytest.raises(RegionalFixtureError, match="must differ"):
        runner.configure(arguments)


def test_configure_rejects_a_non_arn_target(tmp_path: Path) -> None:
    arguments = runner.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--state-dir",
            str(tmp_path),
            "--target-cluster-arn",
            "not-an-arn",
            "--other-cluster-id",
            "eks",
        ]
    )
    with pytest.raises(RegionalFixtureError, match="GPU cluster EKS ARN"):
        runner.configure(arguments)


def test_plan_details_names_the_window_the_premise_and_both_verb_commands(
    tmp_path: Path,
) -> None:
    settings = runner.Settings(
        state_dir=tmp_path,
        target_cluster_arn="arn:aws:eks:us-west-2:000000000000:cluster/tgt",
        target_cluster_id="hp",
        other_cluster_id="eks",
        other_node="ip-10-0-44-18",
        window=m.WINDOW_KUBELET,
        dry_run=True,
        regional=_fake_regional(tmp_path),
        host_probe_image="image@sha256:abc",
        cpu_cluster_arn="arn:aws:eks:us-west-2:000000000000:cluster/cpu",
        predecessor_id="GF-REGIONAL-BOOT-029",
        predecessor_path=None,
        run_token="deadbeef",
    )
    details = runner.plan_details(
        settings, {"identity": {}, "product_generation": "b250"}
    )
    assert details["risk"] == "destructive"
    assert "invisible to the product" in details["premise"]
    assert set(details["verb_commands"]) == {m.REMOVE_VERB, m.JOIN_VERB}
    assert "REMOVE_GPU_CLUSTER" in details["verb_commands"][m.REMOVE_VERB]


def _fake_regional(tmp_path: Path) -> Any:
    (tmp_path / "cpu.kubeconfig").write_text("x", encoding="utf-8")
    (tmp_path / "gpu.kubeconfig").write_text("x", encoding="utf-8")
    from scripts.e2e.regional.regional_live_fixture import RegionalLiveSettings

    return RegionalLiveSettings(
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
        gpu_context="ctx-other",
        namespace="gpu-fault-system",
        cluster_id="eks",
        region="us-west-2",
    )
