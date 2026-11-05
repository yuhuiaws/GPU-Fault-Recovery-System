"""join-cluster, alone and in a batch: the edges of its failure handling (cov95).

The join suites drive whole joins over ``JoinScenario``, the stateful fake of
every boundary a join crosses. These cases pin what they leave out: a recorded
identity older than ``node_selector_source``, a commit whose live registry
reads empty, a supervision loss whose fence cannot be written (alone and in a
batch next to an already-managed member), a batch whose every discovery
fails, an emptied site's anchor refusal when the discovery of the anchor's
provider failed, an operator interruption while a rollback resumes, and a
batch commit that cannot advance its other clusters' verification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_batch_join as batch
from gpu_fault.admin import cluster_join as join
from gpu_fault.admin import resource_registry as registry
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from tests.admin._cov95_join_support import JoinScenario, target
from tests.admin.test_multi_cluster_profile_review import MixedJoinScenario
from tests.admin.test_registration_anchor_departed import _emptied


@pytest.fixture
def scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JoinScenario:
    return JoinScenario(tmp_path, monkeypatch)


def fail_supervision_record(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``record_join_supervision_loss`` cannot write; returns who asked."""

    asked: list[str] = []

    def record(attempt: Any) -> None:
        asked.append(attempt.request.gpu_cluster_arn)
        raise OSError("state volume is read-only")

    monkeypatch.setattr(join, "record_join_supervision_loss", record)
    return asked


# --- the single join ---------------------------------------------------------------


def test_an_identity_recorded_before_node_selector_source_still_matches(
    scenario: JoinScenario,
) -> None:
    arn = target("b").eks_arn
    scenario.targets[arn] = replace(target("b"), node_selector_source="label")
    legacy = asdict(target("b"))
    del legacy["node_selector_source"]

    join.validate_saved_join_target(
        scenario.request("b"), {"target": legacy, "cluster_id": "hp-gpu-b"}, scenario
    )

    with pytest.raises(join.JoinTargetIdentityError, match="changed since discovery"):
        join.validate_saved_join_target(
            scenario.request("b"),
            {"target": asdict(target("b")), "cluster_id": "hp-gpu-b"},
            scenario,
        )
    assert scenario.events == ["discover:gpu-b", "discover:gpu-b"]


def test_a_commit_whose_live_registry_reads_empty_uses_the_discovery_snapshot(
    scenario: JoinScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = scenario.event
    exported = {item.resource_key for item in scenario.registry.live.resources}

    def event(name: str) -> None:
        record(name)
        if name == "sync-state":
            live = scenario.registry.live
            scenario.registry.live = registry.InstallationResourceSnapshot(
                site_id=live.site_id, resources=[]
            )

    monkeypatch.setattr(scenario, "event", event)

    assert scenario.join()["phase"] == "COMPLETED"

    state = scenario.state()
    delta = set(state["evidence"]["REGISTRY_UPDATED"]["delta_keys"])
    assert "cluster/hp-gpu-b/eks" in delta, delta
    after = json.loads(
        (
            scenario.state_path().parent / "installation-resources-after-001.json"
        ).read_text(encoding="utf-8")
    )
    kept = {item["resource_key"] for item in after["resources"]}
    assert exported <= kept, "the audit snapshot keeps every row the export saw"
    assert delta <= kept
    assert {item.resource_key for item in scenario.registry.live.resources} == delta, (
        "only the joined rows were upserted into the empty registry"
    )


def test_a_supervision_loss_whose_fence_cannot_be_written_says_so(
    scenario: JoinScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = fail_supervision_record(monkeypatch)
    scenario.failure = "verify"
    scenario.failure_error = ProcessSupervisionLost("verify driver ownership lost")

    with pytest.raises(ProcessSupervisionLost) as lost:
        scenario.join()

    assert "join could not persist the unproven command ownership" in (
        lost.value.__notes__
    )
    assert asked == [target("b").eks_arn]
    assert scenario.state()["phase"] != "SUPERVISION_LOST"
    assert ("activate-cluster", "hp-gpu-b") not in scenario.driver_calls
    assert not any(mode.startswith("rollback") for mode, _ in scenario.driver_calls), (
        "a supervision loss never starts compensation"
    )


# --- the batch join ------------------------------------------------------------------


def test_a_batch_supervision_loss_skips_a_completed_member_and_notes_the_fence(
    scenario: JoinScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario.targets[target().eks_arn] = replace(
        target("a"), input_arn=target().eks_arn
    )
    asked = fail_supervision_record(monkeypatch)
    scenario.failure = "join-cluster:hp-gpu-c"
    scenario.failure_error = ProcessSupervisionLost("deploy driver ownership lost")

    with pytest.raises(ProcessSupervisionLost) as lost:
        batch.join_clusters(
            (scenario.request(cluster_id="gpu-a"), scenario.request("c")),
            runner_factory=lambda: scenario,
        )

    assert "batch join could not persist unproven command ownership" in (
        lost.value.__notes__
    )
    assert asked == [target("c").eks_arn], "the already-managed member is not fenced"
    assert scenario.state()["phase"] == "COMPLETED"
    assert scenario.state()["evidence"]["DISCOVERED"]["already_managed"] is True


def test_a_batch_whose_every_discovery_fails_reports_both_and_changes_nothing(
    scenario: JoinScenario,
) -> None:
    scenario.discovery_error = BootstrapError("eks:DescribeCluster denied")

    with pytest.raises(BootstrapError) as failed:
        scenario.batch()

    message = str(failed.value)
    assert message.startswith("batch join completed with failed clusters: "), message
    for suffix in ("b", "c"):
        assert target(suffix).eks_arn in message
        assert scenario.state(suffix)["phase"] == "FAILED"
    summary = json.loads(message.split("; summary: ", 1)[1])
    assert (summary["phase"], summary["joined"]) == ("PARTIAL", [])
    assert set(summary["failed"].values()) == {"eks:DescribeCluster denied"}
    assert scenario.driver_calls == []


def test_an_emptied_sites_anchor_refusal_names_the_failed_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HyperPod cluster that could take over the anchor failed its
    discovery, so the batch brings only a plain EKS cluster: the refusal says
    why the HyperPod cluster is missing from it."""

    scenario = MixedJoinScenario(tmp_path / "site", monkeypatch)
    _emptied(scenario, tmp_path, monkeypatch)
    discover = scenario.discover
    hyperpod = target("b").eks_arn

    def flaky(runner: Any, *, cluster_arn: str, **options: Any) -> Any:
        if cluster_arn == hyperpod:
            raise BootstrapError("sagemaker:DescribeCluster throttled")
        return discover(runner, cluster_arn=cluster_arn, **options)

    monkeypatch.setattr(join, "discover_cluster", flaky)

    with pytest.raises(BootstrapError) as refused:
        scenario.join_batch("b", "e")

    message = str(refused.value)
    assert "the first cluster to join must be a hyperpod cluster" in message, message
    assert message.endswith(f"(discovery failed for {hyperpod})"), message
    assert scenario.journal("b")["phase"] == "FAILED"
    assert not any(event.startswith("executor-role") for event in scenario.events), (
        "no AWS mutation"
    )


def test_an_interruption_while_a_rollback_resumes_stops_the_batch(
    scenario: JoinScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume = join.resume_join_rollback
    resumed: list[str] = []

    def interrupted(attempt: Any, runner: Any) -> bool:
        resumed.append(attempt.request.gpu_cluster_arn)
        if attempt.request.gpu_cluster_arn == target("c").eks_arn:
            raise KeyboardInterrupt("operator interrupted the resumed rollback")
        return resume(attempt, runner)

    monkeypatch.setattr(join, "resume_join_rollback", interrupted)

    with pytest.raises(KeyboardInterrupt, match="interrupted the resumed rollback"):
        scenario.batch()

    assert resumed == [target("b").eks_arn, target("c").eks_arn]
    assert scenario.state("c")["phase"] == "FAILED"
    assert scenario.driver_calls == [] and scenario.events == [], (
        "nothing past the rollback resume ran"
    )


def test_a_commit_that_cannot_advance_the_other_verifications_fails_them(
    scenario: JoinScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    advanced: list[str] = []

    def advance(evidence: dict[str, Any], **options: Any) -> dict[str, Any]:
        advanced.append(str(options["cluster_id"]))
        raise BootstrapError("verification evidence cannot be advanced")

    monkeypatch.setattr(batch, "advance_batch_verification", advance)

    with pytest.raises(BootstrapError) as failed:
        scenario.batch()

    message = str(failed.value)
    assert f"failed clusters: {target('c').eks_arn}" in message, message
    assert advanced == ["hp-gpu-b"], "advanced once, after the first commit"
    assert scenario.state("b")["phase"] == "COMPLETED"
    assert scenario.state("c")["phase"] == "FAILED"
    assert ("activate-cluster", "hp-gpu-b") in scenario.driver_calls
    assert ("activate-cluster", "hp-gpu-c") not in scenario.driver_calls


def test_a_batch_rerun_keeps_the_entries_its_stopped_commits_wrote(
    scenario: JoinScenario,
) -> None:
    """Both commits stopped at the registry after writing the site: the rerun
    plans its candidate over a site that already holds both entries."""

    scenario.failure = "registry-sync"
    with pytest.raises(BootstrapError, match="failed clusters"):
        scenario.batch()
    committed = yaml.safe_load(scenario.path.read_text(encoding="utf-8"))
    entries = {item["clusterId"]: item for item in committed["spec"]["clusters"]}
    assert sorted(entries) == ["gpu-a", "hp-gpu-b", "hp-gpu-c"]
    assert scenario.state("b")["phase"] == scenario.state("c")["phase"] == "FAILED"
    scenario.failure = None

    result = scenario.batch()

    assert (result["phase"], result["joined"]) == (
        "COMPLETED",
        ["hp-gpu-b", "hp-gpu-c"],
    )
    final = yaml.safe_load(scenario.path.read_text(encoding="utf-8"))
    assert [item["clusterId"] for item in final["spec"]["clusters"]] == [
        "gpu-a",
        "hp-gpu-b",
        "hp-gpu-c",
    ], "no entry is added twice"
    assert {item["clusterId"]: item for item in final["spec"]["clusters"]} == entries
    candidates = sorted((scenario.path.parent / "join-cluster").glob("site.batch-*"))
    assert candidates, "the rerun wrote a batch candidate"
    for path in candidates:
        listed = yaml.safe_load(path.read_text(encoding="utf-8"))["spec"]["clusters"]
        ids = [item["clusterId"] for item in listed]
        assert len(ids) == len(set(ids)), f"{path.name} lists a cluster twice: {ids}"
