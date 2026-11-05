"""GF-REGIONAL-BOOT-038: the narration grammar and record readers at their edges.

``test_boot038_membership`` pins the verdicts on whole streams and evidence
records; these cases pin the small readers the run builds them from on the
inputs that make them say no: a ``{`` that starts no JSON object, a verb the
trigger grammar does not know, a fleet wave whose ready count is short, an
after-run with each whole-case check failing on its own, a generation probe
that printed nothing, and a state directory with no usable verb state.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import boot038_membership as m

STAMP = "2026-11-04T12:20:40Z"


def clean_after() -> m.AfterRun:
    return m.AfterRun(
        other_node_ready=True,
        other_agent_retired=False,
        other_node_ledger_mail=(),
        target_departed_mail=(),
        target_agents_all_active=True,
        non_terminal_incident=False,
        window_residue=False,
    )


def passing_evidence(**changes: Any) -> m.VerbEvidence:
    evidence = m.VerbEvidence(
        verb=m.REMOVE_VERB,
        window=m.WINDOW_KUBELET,
        exit_code=0,
        saw_fail_cluster=False,
        saw_rollback_cluster=False,
        window_opened=True,
        other_node_not_ready_before_verify=True,
        finding_surfaces={name: True for name in m.FINDING_SURFACES},
        registry_convergence_pass=True,
    )
    return replace(evidence, **changes)


# --- the merged verb output --------------------------------------------------


def test_a_brace_that_opens_no_json_object_is_skipped() -> None:
    stream = [
        "{\n",
        "  this is narration, not JSON\n",
        "{\n",
        '  "cpu_control_plane": "PRESERVED",\n',
        '  "gpu_cluster": "REMOVED",\n',
        '  "phase": "COMPLETED"\n',
        "}\n",
    ]

    result = m.verb_result_from_output(stream)

    assert result == {
        "cpu_control_plane": "PRESERVED",
        "gpu_cluster": "REMOVED",
        "phase": "COMPLETED",
    }


def test_a_stream_of_undecodable_braces_has_no_verb_result() -> None:
    assert m.verb_result_from_output(["{\n", "release-end truncated\n"]) is None


# --- the trigger grammar -----------------------------------------------------


def test_the_removal_opens_on_its_release_begin_and_the_join_on_a_wave_start() -> None:
    removal = m.verify_trigger(m.REMOVE_VERB)
    join = m.verify_trigger(m.JOIN_VERB)

    assert removal.search(f"release-begin {STAMP} mode=remove-cluster target=hp"), (
        "the removal's own release-begin opens the window"
    )
    assert not removal.search(f"release-begin {STAMP} mode=verify"), (
        "the verify's release-begin is not the removal trigger"
    )
    match = join.search(f"fleet-wave {STAMP} cluster=hp phase=join wave=2/2 nodes=1")
    assert match is not None and match.groups() == ("2", "2"), (
        "the join trigger reads the wave number and the wave count"
    )


def test_an_unknown_verb_has_no_trigger() -> None:
    with pytest.raises(ValueError, match="unknown membership verb 'verify'"):
        m.verify_trigger("verify")


@pytest.mark.parametrize(
    "line, full",
    [
        (f"fleet-wave-done {STAMP} cluster=hp wave=2/2 nodes=2 ready=2/2", True),
        (f"fleet-wave-done {STAMP} cluster=hp wave=2/2 nodes=2 ready=1/2", False),
        (f"fleet-wave-done {STAMP} cluster=hp wave=1/1 nodes=0 ready=0/0", False),
        (f"fleet-wave {STAMP} cluster=hp phase=join wave=2/2", False),
        ("release-end mode=verify", False),
    ],
    ids=["complete", "short", "empty-wave", "start-line", "other-line"],
)
def test_a_full_fleet_wave_is_a_done_line_with_every_node_ready(
    line: str, full: bool
) -> None:
    assert m.is_full_fleet_wave(line) is full


# --- the per-verb verdict ----------------------------------------------------


def test_a_fail_cluster_narration_fails_the_verb() -> None:
    verdict, reasons = m.judge_verb(passing_evidence(saw_fail_cluster=True))

    assert verdict == m.VERDICT_FAIL
    assert reasons == ["the verb printed mode=fail-cluster"]


# --- the after-run judgement -------------------------------------------------


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"other_node_ready": False}, "the other cluster's node is not Ready again"),
        ({"target_agents_all_active": False}, "not every target Agent is ACTIVE again"),
        ({"non_terminal_incident": True}, "a non-terminal incident is open"),
    ],
    ids=["other-node", "target-agents", "open-incident"],
)
def test_each_whole_case_check_fails_the_after_run_on_its_own(
    changes: dict[str, bool], reason: str
) -> None:
    ok, reasons = m.judge_after_run(
        replace(clean_after(), **changes), window=m.WINDOW_NONE
    )

    assert (ok, reasons) == (False, [reason])


def test_an_after_run_with_every_check_failing_names_them_all_in_order() -> None:
    after = m.AfterRun(
        other_node_ready=False,
        other_agent_retired=True,
        other_node_ledger_mail=("n-other: CLEANED",),
        target_departed_mail=("[GPU node departed: operator action needed] n1",),
        target_agents_all_active=False,
        non_terminal_incident=True,
        window_residue=True,
    )

    ok, reasons = m.judge_after_run(after, window=m.WINDOW_KUBELET)

    assert ok is False
    assert reasons == [
        "the other cluster's node is not Ready again",
        "the other node's Agent was retired (it must never be)",
        "the departed-node ledger shows mail for the other node: n-other: CLEANED",
        "a 'GPU node departed' notification went out for the target's nodes "
        "during the removal (b249): [GPU node departed: operator action needed] n1",
        "not every target Agent is ACTIVE again",
        "a non-terminal incident is open",
        "a service-window claim or probe residue is left on the node",
    ]


# --- the case verdict --------------------------------------------------------


def test_a_case_with_no_judged_verb_is_inconclusive() -> None:
    assert m.case_verdict({}, after_ok=True) == m.VERDICT_INCONCLUSIVE


def test_a_case_with_a_not_applicable_verb_is_not_a_pass() -> None:
    verdicts = {m.REMOVE_VERB: m.VERDICT_PASS, m.JOIN_VERB: m.VERDICT_NOT_RUN}

    assert m.case_verdict(verdicts, after_ok=True) == m.VERDICT_INCONCLUSIVE


# --- the product's own records -----------------------------------------------


def test_agents_all_active_requires_exactly_the_expected_agent_count() -> None:
    agents = [{"lifecycle_state": "ACTIVE"}, {"lifecycle_state": "ACTIVE"}]

    assert m.agents_all_active(agents, expected=2) is True
    assert m.agents_all_active(agents, expected=3) is False, (
        "an Agent missing from the expected fleet is not 'all active'"
    )
    assert m.agents_all_active(agents[:1], expected=2) is False


def test_the_generation_probe_skips_trailing_blank_lines() -> None:
    assert m.generation_is_b250('{"b250": true, "record": "findings.json"}\n\n  \n'), (
        "blank lines after the probe's JSON line are not its answer"
    )


@pytest.mark.parametrize("stdout", ["", "\n\n", "   "])
def test_a_generation_probe_that_printed_nothing_is_not_b250(stdout: str) -> None:
    assert m.generation_is_b250(stdout) is False


def test_a_state_dir_without_the_verbs_directory_has_no_verified_state(
    tmp_path: Path,
) -> None:
    assert m.verified_state_for(tmp_path, m.REMOVE_VERB, "hp") is None
    assert m.verified_state_for(tmp_path, m.JOIN_VERB, "hp") is None


def test_an_unreadable_verb_state_is_passed_over_for_the_next_one(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "remove-cluster" / "0-broken"
    broken.mkdir(parents=True)
    (broken / "state.json").write_text("{not json", encoding="utf-8")
    listed = tmp_path / "remove-cluster" / "1-list"
    listed.mkdir(parents=True)
    (listed / "state.json").write_text('["not", "an", "object"]', encoding="utf-8")
    good = tmp_path / "remove-cluster" / "2-good"
    good.mkdir(parents=True)
    (good / "state.json").write_text(
        '{"cluster_id": "x", "evidence": {"VERIFIED": {"cluster_id": "hp"}}}',
        encoding="utf-8",
    )

    found = m.verified_state_for(tmp_path, m.REMOVE_VERB, "hp")

    assert found == {"cluster_id": "x", "evidence": {"VERIFIED": {"cluster_id": "hp"}}}


def test_the_findings_record_lives_under_membership_verify(tmp_path: Path) -> None:
    path = m.findings_record_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"not_gating_findings": [{"check": "control_api", '
        '"cluster_id": "gpu-eks-cluster", "finding": "coverage drift"}]}',
        encoding="utf-8",
    )

    assert path == tmp_path / "membership-verify" / "findings.json"
    assert m.findings_record(m.load_json(path)) == [
        {
            "check": "control_api",
            "cluster_id": "gpu-eks-cluster",
            "finding": "coverage drift",
        }
    ]
