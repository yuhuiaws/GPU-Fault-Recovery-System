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


def test_removal_opens_on_remove_cluster_end_then_verify_begin_and_end() -> None:
    narration = m.VerbNarration()
    stamp = "2026-11-04T12:05:32Z"
    assert (
        narration.observe(
            m.REMOVE_VERB,
            f"release-end {stamp} mode=remove-cluster exit_code=0 total=99.4s",
        )
        == "open"
    )
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


def test_join_opens_only_on_a_complete_fleet_wave_done() -> None:
    narration = m.VerbNarration()
    partial = "fleet-wave-done 2026-11-04T12:16:08Z wave=1/2 ready=1/4 ... total=314.0s"
    full = "fleet-wave-done 2026-11-04T12:17:08Z wave=2/2 ready=4/4 ... total=374.6s"
    assert narration.observe(m.JOIN_VERB, partial) is None
    assert narration.observe(m.JOIN_VERB, full) == "open"
    assert narration.last_full_fleet_wave is not None, "the full wave was recorded"


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
