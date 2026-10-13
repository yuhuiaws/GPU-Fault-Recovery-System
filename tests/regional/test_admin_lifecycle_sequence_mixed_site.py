"""Offline contract of the GF-REGIONAL-BOOT-029 driver on a mixed-provider site.

A first ``deploy`` refuses GPU clusters of two providers (one site Runtime
Profile per first deploy, ``bootstrap_common.require_single_provider``), so the
driver deploys the clusters of one provider and joins the other provider's
clusters inside stage 1 and stage 5 with the site-bound ``join-cluster``,
approving the provider's own Runtime Profile plan when that join stops for
review. Stage 1 leads with the first ARN's provider; stage 5 reverses the order,
so one run proves both first-deploy providers (hyperpod-v1 and eks-v1 site
profiles) and both join directions. The fakes come from the single-provider
contract (``test_admin_lifecycle_sequence_script``): the fake ``aws`` answers
the HyperPod inventory the provider discovery reads, and the fake
``gpu-fault-admin join-cluster`` stops with a ``CLUSTER_PROFILE_NEW`` plan for
a cluster of the other provider until the plan is approved.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tests.regional.test_admin_lifecycle_sequence_script import (
    ADMIN_EMAIL,
    CPU_ARN,
    GPU_ARN,
    _admin_verbs,
    _install_fakes,
    _record,
    _run,
    _sha256,
    _tool_log,
)

EKS_GPU_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-eks-fixture"
EKS_NODEGROUP = "gpu-p5en-efa-spot-node-group"
OWNERSHIP_LABEL = "gpu-fault.io/control-plane"
SHA256 = re.compile(r"[0-9a-f]{64}")


def _joins(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("gpu-fault-admin join-cluster ")]


def _deploys(lines: list[str]) -> list[str]:
    return [line for line in lines if "admin deploy " in line]


def _approved_plan_sha(stopped_join: str) -> str:
    """The digest of the plan the fake join wrote when it stopped for review."""

    state = stopped_join.split("--state-dir ")[1].split()[0]
    approval = Path(state) / "release-deploy/profile-approval.json"
    return str(json.loads(approval.read_text(encoding="utf-8"))["plan_sha256"])


def _mixed(tmp_path: Path, env: dict[str, str], *extra: str):
    return _run(
        tmp_path,
        env,
        "--gpu-cluster-arn",
        GPU_ARN,
        "--gpu-cluster-arn",
        EKS_GPU_ARN,
        *extra,
    )


def _assert_no_selector_flags(line: str) -> None:
    assert "--gpu-nodegroup" not in line and "--node-selector" not in line, (
        line,
        "the node scope is declared on the node groups, never passed as a flag",
    )


def _assert_eks_join(join: str) -> None:
    assert f"--gpu-cluster-arn {EKS_GPU_ARN}" in join, (
        join,
        "the in-stage join names the eks cluster",
    )
    _assert_no_selector_flags(join)
    assert GPU_ARN not in join, (join, "the HyperPod cluster is not in the eks join")


def _assert_hyperpod_join(join: str) -> None:
    assert f"--gpu-cluster-arn {GPU_ARN}" in join, (
        join,
        "the in-stage join names the HyperPod cluster",
    )
    _assert_no_selector_flags(join)
    assert EKS_GPU_ARN not in join, (
        join,
        "the eks cluster is not in the HyperPod join",
    )


def _precondition_lines(record: dict, stage: str) -> dict:
    return record["stages"][stage]["node_scope_precondition"]


def _removed_arns(lines: list[str]) -> list[str]:
    """The ARN of every S2 ``remove-cluster``, in the order the driver ran them."""

    return [
        line.split("--gpu-cluster-arn ")[1].split()[0]
        for line in lines
        if line.startswith("gpu-fault-admin remove-cluster ")
    ]


def _assert_approved_rerun(stopped: str, approved: str, stage: int) -> None:
    assert "--approve-profile-plan" not in stopped, stopped
    assert (
        f"--approve-profile-plan {_approved_plan_sha(stopped)} "
        f"--reference GF-REGIONAL-BOOT-029/stage-{stage}" in approved
    ), (approved, "the rerun approves exactly the plan the stopped join wrote")


def test_a_mixed_site_leads_s1_with_the_first_arn_and_s5_with_the_other_provider(
    tmp_path: Path,
) -> None:
    """HyperPod ARN first: S1 deploys the HyperPod cluster and joins the
    eks-managed cluster (hyperpod-v1 site, eks-v1 override approved); S2
    removes one cluster at a time, the eks cluster first (remove-cluster refuses
    the HyperPod registration cluster while only the eks cluster would remain);
    S3 re-joins both in one command (the eks
    snapshot exists, no plan stop); S5 reverses the order -- the README deploy
    names the eks cluster (eks-v1 site) and the HyperPod cluster joins with
    its hyperpod-v1 override approved. No command carries a selector flag: the
    driver checked the ownership labels before every acting stage."""
    env = _install_fakes(tmp_path)

    result = _mixed(tmp_path, env)

    assert result.returncode == 0, result.stdout + result.stderr
    lines = _tool_log(env)
    assert _admin_verbs(lines) == [
        "deploy",
        "join-cluster",
        "join-cluster",
        "remove-cluster",
        "remove-cluster",
        "join-cluster",
        "uninstall",
        "deploy",
        "join-cluster",
        "join-cluster",
    ], "S1/S5 deploy one provider and join the other (plan stop, approved rerun)"
    deploys = _deploys(lines)
    assert f"--gpu-cluster-arn {GPU_ARN} --state-dir" in deploys[0], deploys[0]
    assert EKS_GPU_ARN not in deploys[0], (
        "S1's first deploy names the HyperPod cluster only"
    )
    _assert_no_selector_flags(deploys[0])
    assert deploys[1].split()[2:] == [
        "--cpu-cluster-arn",
        CPU_ARN,
        "--gpu-cluster-arn",
        EKS_GPU_ARN,
        "--state-dir",
        str(tmp_path / "state-second"),
        "--admin-email",
        ADMIN_EMAIL,
    ], "S5's README deploy names the eks cluster with the documented flag, nothing else"
    joins = _joins(lines)
    assert len(joins) == 5, joins
    _assert_eks_join(joins[0])
    _assert_eks_join(joins[1])
    _assert_approved_rerun(joins[0], joins[1], 1)
    assert f"--gpu-cluster-arn {GPU_ARN} --gpu-cluster-arn {EKS_GPU_ARN}" in joins[2], (
        joins[2],
        "S3 re-joins every cluster in one command",
    )
    _assert_no_selector_flags(joins[2])
    _assert_hyperpod_join(joins[3])
    _assert_hyperpod_join(joins[4])
    _assert_approved_rerun(joins[3], joins[4], 5)
    assert _removed_arns(lines) == [EKS_GPU_ARN, GPU_ARN], (
        "clusters are removed one at a time, the registration cluster's provider "
        "(the first ARN's) last"
    )
    inventory = next(line for line in lines if "sagemaker list-clusters" in line)
    assert lines.index(inventory) < lines.index(deploys[0]), (
        "the provider of every GPU ARN comes from the HyperPod inventory, before S1"
    )

    record = _record(tmp_path)
    assert all(stage["status"] == "PASS" for stage in record["stages"].values()), (
        record["stages"]
    )
    for number, order in (("2", (EKS_GPU_ARN, GPU_ARN)), ("3", (GPU_ARN, EKS_GPU_ARN))):
        journals = record["stages"][number]["journals"]
        assert [item["gpu_arn_sha256"] for item in journals] == [
            _sha256(arn) for arn in order
        ], (number, "one completion journal per GPU cluster, in the stage's order")
        assert len({item["journal_sha256"] for item in journals}) == 2, number
    expected = {
        "1": ("hyperpod", "eks-managed", EKS_GPU_ARN, "hyperpod-v1", "eks-v1"),
        "5": ("eks-managed", "hyperpod", GPU_ARN, "eks-v1", "hyperpod-v1"),
    }
    for number, (first, joined, joined_arn, base, override) in expected.items():
        stage = record["stages"][number]
        assert stage["gpu_cluster_count"] == 2, (number, stage)
        assert stage["first_deploy_gpu_cluster_count"] == 1, (number, stage)
        assert stage["first_deploy_provider"] == first, (number, stage)
        assert stage["joined_providers"] == [joined], (number, stage)
        assert [item["gpu_arn_sha256"] for item in stage["joins"]] == [
            _sha256(joined_arn)
        ], (number, "one COMPLETED join journal per cluster joined in-stage")
        assert stage["joins"][0]["phase"] == "COMPLETED", (number, stage)
        approved = stage["profile_plan_approved"]
        assert approved["provider"] == joined, (number, approved)
        assert (approved["current_version"], approved["desired_version"]) == (
            base,
            override,
        ), (number, "the site profile base and the joined cluster's own override")
        assert approved["change_kind"] == "CLUSTER_PROFILE_NEW", (number, approved)
        assert approved["reference"] == f"GF-REGIONAL-BOOT-029/stage-{number}", approved
        assert SHA256.fullmatch(approved["plan_sha256"]), (
            "the approved plan is recorded by its digest"
        )
    assert record["inputs"]["gpu_cluster_arns_sha256"] == _sha256(
        f"{GPU_ARN}\n{EKS_GPU_ARN}"
    ), "the input identity is the operator's ARN list, unchanged"
    assert "eks_flag_sha256" not in record["inputs"], (
        "no selector flag is part of the input identity any more"
    )
    for stage in ("1", "3", "5"):
        precondition = _precondition_lines(record, stage)
        assert precondition["ownership_label"] == OWNERSHIP_LABEL, (
            stage,
            "every acting stage records the ownership-label precondition it checked",
        )
        assert SHA256.fullmatch(
            precondition["site_id_sha256"].removeprefix("sha256:")
        ), "the site id enters the record as a digest only"
        assert all(
            item["owned_groups_sha256"] == [f"sha256:{_sha256(name)}"]
            for item, name in zip(
                precondition["clusters"], ("spot-p5en", EKS_NODEGROUP), strict=True
            )
        ), (stage, "node group names enter the record as digests only")
        assert [item["status"] for item in precondition["clusters"]] == [
            "labelled",
            "labelled",
        ], (stage, precondition)
        assert sorted(item["provider"] for item in precondition["clusters"]) == [
            "eks-managed",
            "hyperpod",
        ], (stage, precondition)
    receipt = record["stages"]["5"]["readme"]
    assert receipt["gpu_cluster_count"] == 1, receipt
    assert receipt["joined_gpu_cluster_count"] == 1, receipt
    assert receipt["first_provider"] == "eks-managed", receipt
    assert receipt["ownership_label_documented"] is True, (
        receipt,
        "the README deploy relies on the ownership-label precondition the README states",
    )
    assert "eks_flag" not in receipt and "joined_eks_flag" not in receipt, receipt
    assert receipt["mixed_site_join_documented"] is True, receipt
    assert receipt["plain_eks_first_deploy_documented"] is True, receipt
    serialized = json.dumps(record)
    cluster_ids = ("cluster-" + _sha256(arn)[:8] for arn in (GPU_ARN, EKS_GPU_ARN))
    for secret in (GPU_ARN, EKS_GPU_ARN, EKS_NODEGROUP, *cluster_ids):
        assert secret not in serialized, f"{secret} leaked into the record"
    assert "registration_cluster_id" not in serialized, (
        "the approved plan is recorded by digest, versions and provider only"
    )


def test_a_single_provider_list_runs_unchanged_and_joins_nothing_in_s1_or_s5(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)

    result = _run(tmp_path, env)

    assert result.returncode == 0, result.stdout + result.stderr
    lines = _tool_log(env)
    assert _admin_verbs(lines) == [
        "deploy",
        "remove-cluster",
        "join-cluster",
        "uninstall",
        "deploy",
    ], "one provider: no join inside the first-deploy stages, nothing to reverse"
    for deploy in _deploys(lines):
        assert f"--gpu-cluster-arn {GPU_ARN} " in deploy, deploy
        _assert_no_selector_flags(deploy)
    record = _record(tmp_path)
    for number in ("1", "5"):
        stage = record["stages"][number]
        assert stage["first_deploy_provider"] == "hyperpod", (number, stage)
        assert stage["joined_providers"] == [], (number, stage)
        assert stage["gpu_cluster_count"] == 1, (number, stage)
        assert stage["first_deploy_gpu_cluster_count"] == 1, (number, stage)
        assert "joins" not in stage and "profile_plan_approved" not in stage, (
            number,
            "a single-provider receipt records no in-stage join",
        )
    assert record["stages"]["5"]["readme"]["first_provider"] == "hyperpod", record[
        "stages"
    ]["5"]["readme"]


def test_an_eks_first_list_leads_s1_with_eks_and_s5_with_hyperpod(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)

    result = _run(
        tmp_path, env, "--gpu-cluster-arn", EKS_GPU_ARN, "--gpu-cluster-arn", GPU_ARN
    )

    assert result.returncode == 0, result.stdout + result.stderr
    lines = _tool_log(env)
    deploys = _deploys(lines)
    assert f"--gpu-cluster-arn {EKS_GPU_ARN} " in deploys[0], (
        deploys[0],
        "S1: an all-plain-EKS first deploy names the eks cluster",
    )
    _assert_no_selector_flags(deploys[0])
    assert GPU_ARN not in deploys[0], deploys[0]
    assert f"--gpu-cluster-arn {GPU_ARN} --state-dir" in deploys[1], (
        deploys[1],
        "S5 reverses: the README deploy names the HyperPod cluster",
    )
    assert EKS_GPU_ARN not in deploys[1], deploys[1]
    _assert_no_selector_flags(deploys[1])
    assert _removed_arns(lines) == [GPU_ARN, EKS_GPU_ARN], (
        "the eks registration cluster's provider goes last on an eks-first site"
    )
    joins = _joins(lines)
    _assert_hyperpod_join(joins[0])
    _assert_hyperpod_join(joins[1])
    _assert_approved_rerun(joins[0], joins[1], 1)
    _assert_eks_join(joins[3])
    _assert_eks_join(joins[4])
    _assert_approved_rerun(joins[3], joins[4], 5)
    record = _record(tmp_path)
    one = record["stages"]["1"]
    assert one["first_deploy_provider"] == "eks-managed", one
    assert one["joined_providers"] == ["hyperpod"], one
    assert one["profile_plan_approved"]["desired_version"] == "hyperpod-v1", one
    five = record["stages"]["5"]
    assert five["first_deploy_provider"] == "hyperpod", five
    assert five["joined_providers"] == ["eks-managed"], five
    assert five["profile_plan_approved"]["desired_version"] == "eks-v1", five
    receipt = five["readme"]
    assert receipt["first_provider"] == "hyperpod", receipt
    assert receipt["ownership_label_documented"] is True, receipt


def test_stage_one_resumes_after_a_finished_first_deploy_with_the_join_only(
    tmp_path: Path,
) -> None:
    """The deploy finished, the join did not. The resume must not deploy again
    (a second deploy is a release and no cold build) and must not prune the
    caches again; it continues with the join and keeps S1's proofs."""
    env = _install_fakes(tmp_path)

    failed = _mixed(tmp_path, {**env, "FAKE_ADMIN_FAIL_VERB": "join-cluster"})

    assert failed.returncode != 0, failed.stdout
    assert "FAILED at stage 1 (cold-first-deploy)" in failed.stderr, failed.stderr
    assert "no Runtime Profile plan awaiting approval" in failed.stderr, (
        failed.stderr,
        "a join failure that wrote no plan is not approved into anything",
    )
    assert _admin_verbs(_tool_log(env)) == ["deploy", "join-cluster"], (
        "the deploy finished; the first join failed; nothing else ran"
    )
    record = _record(tmp_path)
    assert record["stages"]["1"]["status"] == "FAIL", record["stages"]["1"]
    assert (tmp_path / "state/bootstrap-state.json").exists(), (
        "the finished first deploy stays in place"
    )

    resumed = _mixed(tmp_path, env, "--stage", "1")

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    lines = _tool_log(env)
    assert _admin_verbs(lines)[2:] == [
        "join-cluster",
        "join-cluster",
        "remove-cluster",
        "remove-cluster",
        "join-cluster",
        "uninstall",
        "deploy",
        "join-cluster",
        "join-cluster",
    ], "the resume joins (plan stop, approved rerun) without a second S1 deploy"
    prunes = [line for line in lines if line.startswith("docker buildx prune")]
    assert len(prunes) == 2, (prunes, "no prune on the resume; one per cold deploy")
    record = _record(tmp_path)
    stage = record["stages"]["1"]
    assert stage["status"] == "PASS" and stage["attempts"] == 2, stage
    assert stage["resumed_first_deploy"] is True, stage
    assert stage["first_deploy_already_complete"] is True, stage
    assert stage["cold_build"]["cached_layers"] == 0, (
        stage,
        "the cold-build proof is read from the attempt that built",
    )
    assert [item["gpu_arn_sha256"] for item in stage["joins"]] == [
        _sha256(EKS_GPU_ARN)
    ], stage
    assert stage["profile_plan_approved"]["desired_version"] == "eks-v1", stage


def test_stage_five_resumes_after_a_finished_readme_deploy_with_the_join_only(
    tmp_path: Path,
) -> None:
    """The README deploy of the eks cluster finished and the HyperPod join
    failed: the resume runs neither the README procedure nor a deploy again and
    keeps the first attempt's README receipt; only the join is added."""
    env = _install_fakes(tmp_path)
    second = tmp_path / "state-second"

    failed = _mixed(tmp_path, {**env, "FAKE_ADMIN_FAIL_JOIN_STATE": str(second)})

    assert failed.returncode != 0, failed.stdout
    assert "FAILED at stage 5 (cold-redeploy-readme)" in failed.stderr, failed.stderr
    assert _admin_verbs(_tool_log(env))[-2:] == ["deploy", "join-cluster"], (
        "the README deploy finished and the HyperPod join failed"
    )
    tools_before = len(_tool_log(env))
    readme_before = _record(tmp_path)["stages"]["5"]["readme"]

    resumed = _mixed(tmp_path, env, "--stage", "5")

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    new_tools = _tool_log(env)[tools_before:]
    assert [
        line.split()[1] for line in new_tools if line.startswith("gpu-fault-admin ")
    ] == ["join-cluster", "join-cluster"], (
        "the resume joins the HyperPod cluster (plan stop, approved rerun) and nothing else"
    )
    assert not any(
        line.startswith(("make ", "docker ", "aws ecr ")) for line in new_tools
    ), "no README procedure, no prune and no ECR clearing on the resume"
    for join in new_tools:
        if join.startswith("gpu-fault-admin join-cluster "):
            _assert_hyperpod_join(join)
            assert f"--state-dir {second} " in join, join
    record = _record(tmp_path)
    stage = record["stages"]["5"]
    assert stage["status"] == "PASS" and stage["attempts"] == 2, stage
    assert stage["first_deploy_already_complete"] is True, stage
    assert stage["readme"] == readme_before, (
        "the README receipt of the attempt that deployed is kept"
    )
    assert stage["first_deploy_provider"] == "eks-managed", stage
    assert stage["profile_plan_approved"]["desired_version"] == "hyperpod-v1", stage
    assert [item["gpu_arn_sha256"] for item in stage["joins"]] == [_sha256(GPU_ARN)], (
        stage
    )


def test_a_recorded_request_for_another_cluster_set_refuses_the_resume(
    tmp_path: Path,
) -> None:
    """The live shape: the product recorded the mixed two-ARN request before it
    refused the mixed first deploy; a resume that now deploys one provider would
    be refused by the product for conflicting with that request. The driver
    names the way out -- a new empty state directory -- and runs nothing."""
    env = _install_fakes(tmp_path)
    state = tmp_path / "state"

    failed = _mixed(tmp_path, {**env, "FAKE_ADMIN_FAIL_DEPLOY_STATE": str(state)})

    assert failed.returncode != 0, failed.stdout
    assert _admin_verbs(_tool_log(env)) == ["deploy"], "the deploy failed early"
    (state / "initial-deploy-request.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_dir": str(state),
                "cpu_cluster_arn": CPU_ARN,
                "gpu_cluster_arns": [GPU_ARN, EKS_GPU_ARN],
                "admin_email": ADMIN_EMAIL,
            }
        ),
        encoding="utf-8",
    )
    tools_before = _tool_log(env)

    resumed = _mixed(tmp_path, env, "--stage", "1")

    assert resumed.returncode != 0, resumed.stdout
    assert "FAILED at stage 1 (cold-first-deploy)" in resumed.stderr, resumed.stderr
    assert "initial-deploy-request.json" in resumed.stderr, resumed.stderr
    assert "new empty state directory" in resumed.stderr, resumed.stderr
    assert [
        line
        for line in _tool_log(env)[len(tools_before) :]
        if "gpu-fault-admin" in line
    ] == [], "no administrator command runs against the conflicting directory"
    record = _record(tmp_path)
    assert record["stages"]["1"]["status"] == "FAIL", record["stages"]["1"]
    assert record["stages"]["1"]["failure"], "the receipt records the refusal"


def test_provider_discovery_refusal_fails_stage_one_before_any_deploy(
    tmp_path: Path,
) -> None:
    """Two HyperPod clusters naming the same EKS orchestrator is the
    configuration the product refuses to guess about; so does the driver."""
    env = _install_fakes(tmp_path)

    result = _mixed(tmp_path, {**env, "FAKE_HYPERPOD_DUPLICATE": "1"})

    assert result.returncode != 0, result.stdout
    assert "FAILED at stage 1 (cold-first-deploy)" in result.stderr, result.stderr
    assert "provider discovery failed" in result.stderr, result.stderr
    assert _admin_verbs(_tool_log(env)) == [], "no deploy runs on an unknown provider"
    log = tmp_path / "state/acceptance/stage-1-cold-first-deploy-attempt-1.log"
    assert "at most one HyperPod cluster" in log.read_text(encoding="utf-8"), (
        "the product's own refusal is in the stage log"
    )


def test_stage_five_refuses_a_readme_that_no_longer_states_the_ownership_label(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    readme_path = tmp_path / "checkout/README.md"
    readme_path.write_text(
        readme_path.read_text(encoding="utf-8").replace(
            "`gpu-fault-admin site-id --cpu-cluster-arn <cpu-eks-or-hyperpod-arn>`",
            "the site-id verb",
        ),
        encoding="utf-8",
    )

    result = _mixed(tmp_path, env)

    assert result.returncode != 0, result.stdout
    assert "FAILED at stage 5 (cold-redeploy-readme)" in result.stderr, result.stderr
    assert "does not state that the GPU node groups carry" in result.stderr, (
        result.stderr
    )
    assert _admin_verbs(_tool_log(env)) == [
        "deploy",
        "join-cluster",
        "join-cluster",
        "remove-cluster",
        "remove-cluster",
        "join-cluster",
        "uninstall",
    ], "the driver's own stages still pass the flag through; only S5 refuses"
    assert not (tmp_path / "state-second/bootstrap-state.json").exists(), (
        "no deploy reaches the second state directory"
    )


def test_stage_five_refuses_a_readme_without_the_one_provider_rule(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    readme_path = tmp_path / "checkout/README.md"
    text = readme_path.read_text(encoding="utf-8")
    rule = next(
        line
        for line in text.splitlines()
        if "provider" in line and "join-cluster" in line
    )
    readme_path.write_text(text.replace(rule + "\n", ""), encoding="utf-8")

    result = _mixed(tmp_path, env)

    assert result.returncode != 0, result.stdout
    assert "FAILED at stage 5 (cold-redeploy-readme)" in result.stderr, result.stderr
    assert "deploys the clusters of one provider first" in result.stderr, result.stderr
    assert _admin_verbs(_tool_log(env))[-1] == "uninstall", (
        "S1-S4 pass; the README defect is found before the S5 deploy"
    )
    assert not (tmp_path / "state-second/bootstrap-state.json").exists(), (
        "no deploy reaches the second state directory"
    )


def test_stage_five_refuses_a_readme_without_the_plain_eks_first_deploy_prerequisites(
    tmp_path: Path,
) -> None:
    """S5 leads with the eks cluster: the README must state the plain-EKS first
    deploy's prerequisites (node auto repair off) or the stage refuses."""
    env = _install_fakes(tmp_path)
    readme_path = tmp_path / "checkout/README.md"
    text = readme_path.read_text(encoding="utf-8")
    assert "`nodeRepairConfig.enabled=false`" in text, "the real README states it"
    readme_path.write_text(
        text.replace("`nodeRepairConfig.enabled=false`", "node auto repair off"),
        encoding="utf-8",
    )

    result = _mixed(tmp_path, env)

    assert result.returncode != 0, result.stdout
    assert "FAILED at stage 5 (cold-redeploy-readme)" in result.stderr, result.stderr
    assert "plain-EKS first-deploy prerequisites" in result.stderr, result.stderr
    assert "nodeRepairConfig.enabled=false" in result.stderr, result.stderr
    assert _admin_verbs(_tool_log(env))[-1] == "uninstall", (
        "S1 (HyperPod first) is unaffected; the README defect is found before S5"
    )
    assert not (tmp_path / "state-second/bootstrap-state.json").exists(), (
        "no deploy reaches the second state directory"
    )


def test_an_unlabelled_gpu_cluster_stops_stage_one_before_any_build_or_deploy(
    tmp_path: Path,
) -> None:
    """The precondition the driver checks first: a GPU cluster whose node groups
    do not carry the site's ownership label would be refused by deploy, so the
    stage stops with the product's refusal and the describe evidence in the
    log, and no cache is pruned, no image built, no admin command run."""
    env = {**_install_fakes(tmp_path), "FAKE_UNLABELLED_GPU": "1"}

    result = _mixed(tmp_path, env)

    assert result.returncode != 0, result.stdout
    assert "FAILED at stage 1 (cold-first-deploy)" in result.stderr, result.stderr
    assert "would be refused by deploy/join-cluster" in result.stderr, result.stderr
    lines = _tool_log(env)
    assert _admin_verbs(lines) == [], "nothing was deployed or joined"
    assert not any(line.startswith("docker ") for line in lines), (
        "no cache was pruned and no image built"
    )
    log = next((tmp_path / "state/acceptance").glob("stage-1-*.log")).read_text(
        encoding="utf-8"
    )
    assert f"{OWNERSHIP_LABEL}=us-west-2-fixture-cpu-hyperpod-" in log, (
        "the refusal names the exact label and site id"
    )
    assert "spot-p5en instanceTypes=ml.p5en.48xlarge" in log, (
        "the describe-cluster evidence is in the stage log"
    )
    assert "gpu-fault-admin site-id" in log, "the verb that prints the id is named"
