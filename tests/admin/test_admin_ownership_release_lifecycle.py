"""The OWNERSHIP RELEASE step inside the remove-cluster and uninstall journals.

Both commands end with the step; both journals resume at it when its write is
refused; ``--keep-ownership-label`` may arrive on that resume; the receipts
carry the evidence; ``site-id``, ``node-scope`` and ``status`` read the label's
lifecycle back.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin import node_scope_command as node_scope
from gpu_fault.admin import ownership_release as release
from gpu_fault.admin import site_identity_command as site_identity
from gpu_fault.admin import uninstall as lifecycle
from gpu_fault.admin.bootstrap_common import BootstrapError, ClusterIdentity
from gpu_fault.admin.cluster_removal import RemoveClusterRequest, remove_cluster
from gpu_fault.admin.cluster_removal_state import (
    LEGACY_COMPLETED_STEPS,
    OWNERSHIP_RELEASE_STEP,
    STEP_DEPENDENCIES,
    read_removal_state,
)
from gpu_fault.admin.site import load_site
from gpu_fault.admin.status_report import print_status_report
from gpu_fault.admin.uninstall import UninstallRequest
from gpu_fault.admin.uninstall_types import UNINSTALL_PHASES
from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from tests.admin._cluster_removal_support import EKS_ARN, RemovalScenario
from tests.admin.test_admin_node_scope_verb import (
    LEGACY_HP_SECRET,
    ScopeRunner,
    _hp_group,
    _hp_node,
)
from tests.admin.test_admin_site import site_file
from tests.admin.test_uninstall_lifecycle import Harness


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    return RemovalScenario(tmp_path, monkeypatch)


# --- remove-cluster ------------------------------------------------------------


def test_the_release_is_the_last_removal_step() -> None:
    assert STEP_DEPENDENCIES[OWNERSHIP_RELEASE_STEP] == ("VERIFIED",), (
        "the label leaves the node groups only after every step that selected "
        "nodes by it has been verified"
    )
    assert list(STEP_DEPENDENCIES)[-1] == OWNERSHIP_RELEASE_STEP
    assert LEGACY_COMPLETED_STEPS == frozenset(STEP_DEPENDENCIES) - {
        OWNERSHIP_RELEASE_STEP
    }
    assert UNINSTALL_PHASES[-2:] == ("OWNERSHIP_RELEASED", "COMPLETED"), (
        "the uninstall releases the label as its last phase before COMPLETED"
    )


def test_removal_releases_the_label_last_and_records_the_evidence(scenario) -> None:
    result = remove_cluster(scenario.request())

    assert result["phase"] == "COMPLETED"
    assert scenario.calls.index("verify") < scenario.calls.index("ownership-release")
    assert scenario.calls.count("ownership-release") == 1
    state = scenario.state()
    assert OWNERSHIP_RELEASE_STEP in state["completed_steps"]
    evidence = state["evidence"][OWNERSHIP_RELEASE_STEP]
    assert evidence["outcome"] == "released"
    assert evidence["groups"][0]["before"] == "test-site"
    assert evidence["groups"][0]["after"] is None
    assert result["ownership_label"] == {
        "outcome": "released",
        "label": OWNERSHIP_LABEL,
        "released_groups": ["spot-p5en"],
        "retained_groups": [],
        "manual": [],
    }, "the receipt names the outcome and the groups per outcome"


def test_a_refused_release_resumes_at_the_release_step_only(scenario) -> None:
    scenario.failure = "ownership-release"
    with pytest.raises(BootstrapError, match="ownership-release"):
        remove_cluster(scenario.request())

    state = scenario.state()
    assert state["phase"] == "VERIFIED", "the journal stops at the step before"
    assert OWNERSHIP_RELEASE_STEP not in state["completed_steps"]
    assert load_site(scenario.path).release_config["clusters"] == [], (
        "the site commit stays: the cluster is de-registered, only the label is left"
    )
    before = list(scenario.calls)

    scenario.failure = None
    site = load_site(scenario.path)
    resolved = removal.resolve_removal_cluster_id(
        site, EKS_ARN, discover=lambda _arn: (EKS_ARN, "hp-gpu-a")
    )
    result = remove_cluster(RemoveClusterRequest(site, resolved, "REMOVE_GPU_CLUSTER"))

    assert result["phase"] == "COMPLETED"
    new_calls = scenario.calls[len(before) :]
    assert new_calls == ["ownership-release"], (
        "the rerun reloads its saved discovery evidence and runs the release "
        f"alone: {new_calls}"
    )
    assert scenario.calls.count("cleanup") == 1
    assert scenario.calls.count("verify") == 1
    assert set(scenario.state()["completed_steps"]) == STEP_DEPENDENCIES.keys()


def test_keep_ownership_label_may_be_given_on_the_stuck_resume(scenario) -> None:
    scenario.failure = "ownership-release"
    with pytest.raises(BootstrapError, match="ownership-release"):
        remove_cluster(scenario.request())
    scenario.failure = None

    result = remove_cluster(scenario.request(keep_ownership_label=True))

    assert result["phase"] == "COMPLETED"
    assert result["ownership_label"]["outcome"] == "retained"
    assert result["ownership_label"]["retained_groups"] == ["spot-p5en"]
    evidence = scenario.state()["evidence"][OWNERSHIP_RELEASE_STEP]
    assert evidence["outcome"] == "retained"
    assert evidence["groups"][0]["after"] == "test-site"


def test_a_legacy_completed_journal_without_the_step_is_still_a_tombstone(
    scenario,
) -> None:
    remove_cluster(scenario.request())
    path = scenario.path.parent / "remove-cluster/gpu-a/state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["completed_steps"] = [
        step for step in state["completed_steps"] if step != OWNERSHIP_RELEASE_STEP
    ]
    del state["evidence"][OWNERSHIP_RELEASE_STEP]
    path.write_text(json.dumps(state), encoding="utf-8")

    legacy = read_removal_state(path)

    assert legacy["phase"] == "COMPLETED"
    assert removal.resolve_removal_cluster_id(load_site(scenario.path), EKS_ARN) == (
        "gpu-a"
    ), "a journal completed before the step existed still resolves the removal"
    from gpu_fault.admin.cluster_removal_state import ownership_release_summary

    assert ownership_release_summary(legacy)["outcome"] == "unreleased"


def test_remove_cluster_cli_passes_the_keep_flag(monkeypatch, tmp_path) -> None:
    path = site_file(tmp_path)
    requests: list[RemoveClusterRequest] = []

    def remove(request):
        requests.append(request)
        return {"phase": "COMPLETED"}

    monkeypatch.setattr(admin_cli, "remove_cluster", remove)
    base = [
        "remove-cluster",
        "--state-dir",
        str(path.parent),
        "--gpu-cluster-arn",
        EKS_ARN,
        "--confirm",
        "REMOVE_GPU_CLUSTER",
    ]

    assert admin_cli.run(admin_cli.parser().parse_args(base)) == 0
    assert (
        admin_cli.run(admin_cli.parser().parse_args([*base, "--keep-ownership-label"]))
        == 0
    )

    assert [item.keep_ownership_label for item in requests] == [False, True], (
        "release is the default; the flag opts out"
    )


# --- uninstall -----------------------------------------------------------------


def test_uninstall_releases_the_label_last_and_records_it(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)

    result = lifecycle.uninstall(harness.request(reset=True), runner=harness)

    assert harness.state()["phase"] == "COMPLETED"
    assert harness.events.index("delete:aurora") < harness.events.index(
        "ownership-release"
    ), "the label is released after the Aurora phase, last of all"
    assert harness.events.count("ownership-release") == 1
    recorded = harness.state()["ownership_release"]
    assert recorded["outcome"] == "released"
    assert recorded["clusters"][0]["cluster_id"] == "gpu-a"
    assert result["ownership_label"] == {
        "outcome": "released",
        "label": OWNERSHIP_LABEL,
        "clusters": {"gpu-a": "released"},
    }
    assert (tmp_path / "uninstall/installation-resources-final.json").is_file(), (
        "the final registry is written before the release, so a refused release "
        "never loses it"
    )


def test_a_refused_uninstall_release_resumes_without_repeating_the_cleanup(
    tmp_path, monkeypatch
):
    harness = Harness(tmp_path, monkeypatch)
    harness.fail_ownership_release = True

    with pytest.raises(BootstrapError, match="ownership release"):
        lifecycle.uninstall(harness.request(reset=True), runner=harness)

    assert harness.state()["phase"] == "AURORA_DELETE_IN_PROGRESS", (
        "the phase before the release; COMPLETED is never written past a refusal"
    )
    assert "ownership_release" not in harness.state()
    cleanups = harness.events.count("cleanup")

    result = lifecycle.uninstall(harness.request(reset=True), runner=harness)

    assert harness.state()["phase"] == "COMPLETED"
    assert harness.events.count("cleanup") == cleanups, "no second Kubernetes cleanup"
    assert harness.events.count("ownership-release") == 2
    assert result["ownership_label"]["outcome"] == "released"


def test_keep_ownership_label_is_accepted_on_the_stuck_uninstall_resume(
    tmp_path, monkeypatch
):
    harness = Harness(tmp_path, monkeypatch)
    harness.fail_ownership_release = True
    with pytest.raises(BootstrapError, match="ownership release"):
        lifecycle.uninstall(harness.request(reset=True), runner=harness)

    result = lifecycle.uninstall(
        harness.request(reset=True, keep_label=True), runner=harness
    )

    assert harness.state()["phase"] == "COMPLETED"
    assert harness.state()["ownership_release"]["outcome"] == "retained"
    assert result["ownership_label"]["clusters"] == {"gpu-a": "retained"}


def test_completed_uninstall_replay_does_not_release_again(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    lifecycle.uninstall(harness.request(), runner=harness)
    harness.no_mutations = True

    result = lifecycle.uninstall(harness.request(), runner=harness)

    assert harness.events.count("ownership-release") == 1
    assert result["ownership_label"]["outcome"] == "released"


def test_uninstall_request_validates_the_keep_flag(tmp_path):
    site = load_site(site_file(tmp_path))
    with pytest.raises(BootstrapError, match="keep_ownership_label"):
        UninstallRequest(
            site,
            "keep",
            "UNINSTALL_GPU_FAULT",
            keep_ownership_label="yes",  # type: ignore[arg-type]
        )


def test_uninstall_cli_passes_the_keep_flag(monkeypatch, tmp_path) -> None:
    path = site_file(tmp_path)
    requests: list[UninstallRequest] = []

    def fake_uninstall(request):
        requests.append(request)
        return {"cpu_cluster": "keep"}

    monkeypatch.setattr(admin_cli, "uninstall", fake_uninstall)
    base = [
        "uninstall",
        "--state-dir",
        str(path.parent),
        "--cpu-cluster",
        "keep",
        "--confirm",
        "UNINSTALL_GPU_FAULT",
    ]

    assert admin_cli.run(admin_cli.parser().parse_args(base)) == 0
    assert (
        admin_cli.run(admin_cli.parser().parse_args([*base, "--keep-ownership-label"]))
        == 0
    )

    assert [item.keep_ownership_label for item in requests] == [False, True]


# --- site-id, node-scope, status ----------------------------------------------


def test_site_id_states_the_release_semantics() -> None:
    cpu = ClusterIdentity(
        input_arn="arn:aws:eks:us-west-2:123456789012:cluster/control-plane-eks",
        role="cpu",
        region="us-west-2",
        account_id="123456789012",
        hyperpod_arn="arn:aws:sagemaker:us-west-2:123456789012:cluster/fixturecpuhp01",
        hyperpod_name="control-plane-fixture",
        eks_arn="arn:aws:eks:us-west-2:123456789012:cluster/control-plane-eks",
        eks_name="control-plane-eks",
        vpc_id="vpc-1",
        subnet_ids=("subnet-a",),
        node_recovery="None",
        context="cpu",
    )

    document = site_identity.site_id_document(cpu, site_id="site-x", state_dir=None)

    semantics = document["release"]
    assert "remove-cluster and uninstall" in semantics["semantics"]
    assert release.KEEP_FLAG in semantics["semantics"]
    assert semantics["eks_managed_node_group"] == release.release_command(
        "eks-managed", cluster="<EKS cluster>", group="<node group>"
    )
    assert semantics["hyperpod_instance_group"] == release.release_command(
        "hyperpod", cluster="<HyperPod cluster>", group="<instance group>"
    )
    assert "patch nodepool <NodePool>" in semantics["karpenter_nodepool"]
    assert semantics["operator_actions"] == [
        "eks:UpdateNodegroupConfig",
        "sagemaker:UpdateCluster",
    ]


def test_node_scope_reports_a_stale_claim_of_a_removed_cluster(
    scenario, monkeypatch, capsys
) -> None:
    remove_cluster(scenario.request())  # the tombstone; the fake release ran
    site = load_site(scenario.path)
    runner = ScopeRunner(
        hyperpod_groups=[
            _hp_group("spot-p5en", "ml.p5en.48xlarge", labelled=True),
            _hp_group("controller", "ml.m5.2xlarge", labelled=False),
        ]
    )
    arguments = admin_cli.parser().parse_args(
        ["node-scope", "--state-dir", str(scenario.path.parent)]
    )

    assert node_scope.run_node_scope_command(arguments, site=site, runner=runner) == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["clusters"] == [], "the site manages no cluster any more"
    (claim,) = result["stale_ownership_claims"]
    assert claim["cluster_id"] == "gpu-a"
    assert claim["deregistered_by"] == "remove-cluster"
    assert claim["recorded_outcome"] == "released", (
        "the journal said released; the provider still shows the label: stale"
    )
    assert claim["groups"] == ["spot-p5en"]
    assert claim["release_commands"] == [
        release.release_command("hyperpod", cluster="hp-gpu-a", group="spot-p5en")
    ]
    assert "finding: stale ownership claim: remove-cluster de-registered gpu-a" in (
        captured.err
    )


def test_node_scope_reports_no_stale_claim_for_a_cluster_that_joined_again(
    scenario, capsys
) -> None:
    listed = yaml.safe_load(scenario.path.read_text(encoding="utf-8"))["spec"][
        "clusters"
    ]
    remove_cluster(scenario.request())  # the COMPLETED tombstone of gpu-a
    document = yaml.safe_load(scenario.path.read_text(encoding="utf-8"))
    assert document["spec"]["clusters"] == [], "the removal de-registered gpu-a"
    # join-cluster lists the same cluster again (same id, EKS ARN, HyperPod name)
    document["spec"]["clusters"] = listed
    scenario.path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    site = load_site(scenario.path)
    runner = ScopeRunner(
        hyperpod_groups=[
            _hp_group("spot-p5en", "ml.p5en.48xlarge", labelled=True),
            _hp_group("controller", "ml.m5.2xlarge", labelled=False),
        ],
        nodes={"gpu-a": [_hp_node("hyperpod-i-1", "spot-p5en", labelled=True)]},
        secrets={"gpu-a": LEGACY_HP_SECRET},
    )
    arguments = admin_cli.parser().parse_args(
        ["node-scope", "--state-dir", str(scenario.path.parent)]
    )

    assert node_scope.run_node_scope_command(arguments, site=site, runner=runner) == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    (cluster,) = result["clusters"]
    assert cluster["cluster_id"] == "gpu-a"
    assert cluster["label"]["owned_groups"] == ["spot-p5en"], (
        "the label on spot-p5en is the member's own declaration"
    )
    assert result["stale_ownership_claims"] == [], (
        "a re-joined cluster's tombstone is superseded, never a stale claim"
    )
    assert "stale ownership claim" not in captured.err
    assert (scenario.path.parent / "remove-cluster/gpu-a/state.json").is_file(), (
        "the tombstone stays for the next remove-cluster and the re-join readers"
    )


def test_status_prints_stale_claim_findings_after_the_header(capsys) -> None:
    report = json.dumps({"mode": "status", "healthy": True})

    print_status_report(report, findings=["stale ownership claim: gpu-a spot-p5en"])

    captured = capsys.readouterr()
    assert captured.out == report, "the JSON document passes through unchanged"
    lines = captured.err.splitlines()
    assert lines[-1] == (
        "gpu-fault-admin status: finding: stale ownership claim: gpu-a spot-p5en"
    )
    assert lines[-2].startswith("gpu-fault-admin status: full JSON report:"), (
        "findings follow the five header lines"
    )


def test_status_asks_the_release_module_for_findings(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import subprocess

    site_file(tmp_path)
    document = json.dumps({"mode": "status", "healthy": True})
    asked: list[Any] = []

    def fake_run(arguments, **_kwargs):
        return subprocess.CompletedProcess(arguments, 0, stdout=document)

    def findings(site, runner=None):
        asked.append(site.release_config["site_name"])
        return ["stale ownership claim: fixture"]

    monkeypatch.setattr(admin_cli, "run_driver", fake_run)
    monkeypatch.setattr(admin_cli, "_live_release_state", lambda _site: {})
    monkeypatch.setattr("gpu_fault.admin.status_report.status_findings", findings)
    arguments = SimpleNamespace(
        command="status",
        file=None,
        state_dir=tmp_path,
        repo_root=None,
        show_effective_config=False,
        full=False,
    )

    assert admin_cli.run(arguments) == 0

    assert asked == ["test-site"], "status reads the de-registered clusters' labels"
    assert "gpu-fault-admin status: finding: stale ownership claim: fixture" in (
        capsys.readouterr().err
    )
