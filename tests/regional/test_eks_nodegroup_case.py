"""What the BOOT-036/037 runners read beside the tool, and their plumbing.

The site record, the provider reads, the Nodes and the snapshot are read from
the verb's recorded world (``tests/regional/_eks_nodegroup_site``); the tool
document parser and the CLI invoker from what the deploy-host CLI prints.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import eks_nodegroup_case as case
from scripts.e2e.regional.acceptance_scope import (
    EXECUTION_SCOPE_ENV,
    SELECTION_REFERENCE_ENV,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.site_profile import SITE_PROFILE_ENV
from tests.regional._eks_nodegroup_site import (
    CPU_GROUP,
    GPU_GROUP,
    HP_EKS_ARN,
    Site,
    node_name,
    site_document,
)


def target_of(site: Site) -> case.SiteTarget:
    return case.site_target(site.state_dir, "gpu-eks-cluster")


# --- the site record --------------------------------------------------------------


def test_the_site_target_carries_the_member_and_the_hyperpod_members(
    tmp_path: Path,
) -> None:
    target = target_of(Site(tmp_path))

    assert target.eks_name == "gpu-eks-cluster" and target.dcgm_mode == "host"
    assert [item.eks_arn for item in target.hyperpod_members] == [HP_EKS_ARN]
    assert target.describe()["hyperpod_members"][0]["hyperpod_name"] == (
        "hp-member-fixture"
    )
    assert target.tool_arguments("--x", gpu_cluster_arn="arn:other")[3:] == [
        "arn:other",
        "--state-dir",
        str(target.state_dir),
        "--x",
    ]


def test_an_embedded_member_without_a_declaration_is_embedded(tmp_path: Path) -> None:
    assert target_of(Site(tmp_path, dcgm="")).dcgm_mode == "embedded"


def edited(tmp_path: Path, change: Any) -> Path:
    document = site_document()
    change(document)
    state = tmp_path / "state"
    state.mkdir()
    (state / "site.yaml").write_text(json.dumps(document), encoding="utf-8")
    return state


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda document: document["metadata"].pop("name"), "names no site id"),
        (lambda document: document["spec"]["clusters"].pop(), "no member cluster"),
        (
            lambda document: document["spec"]["clusters"][-1].pop("eksClusterArn"),
            "records no EKS cluster ARN",
        ),
        (lambda document: document.update(kind="Other"), "is not a RegionalSite"),
    ],
)
def test_a_site_record_that_cannot_name_the_target_is_refused(
    tmp_path: Path, change: Any, message: str
) -> None:
    with pytest.raises(RegionalFixtureError, match=message):
        case.site_target(edited(tmp_path, change), "gpu-eks-cluster")


def test_an_unreadable_site_record_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RegionalFixtureError, match="cannot read"):
        case.site_target(tmp_path, "gpu-eks-cluster")


# --- the tool ---------------------------------------------------------------------


def test_the_document_starts_at_the_first_brace_line() -> None:
    document = {"verb": case.TOOL_VERB, "status": "COMPLIANT"}
    stdout = "gpu-fault-admin: logging to /x\n" + json.dumps(document, indent=2) + "\n"

    assert case.tool_document(stdout) == document
    assert case.tool_document("no document\n") is None
    assert case.tool_document("{\nnot json\n") is None
    assert case.tool_document(json.dumps({"verb": "other"}, indent=2)) is None


def test_a_run_reports_its_reasons_and_provider_calls() -> None:
    run = case.ToolRun(
        ("--x",),
        2,
        "",
        "+ aws eks list-nodegroups --cluster-name c\n"
        "+ kubectl --kubeconfig k get namespaces\n"
        "gpu-fault-admin: logging to /state/logs/readonly/x.log\n"
        "gpu-fault-admin: EKS cluster c has no managed node group g\n"
        "gpu-fault-admin: full output in /state/logs/readonly/x.log\n",
    )

    assert run.error_lines == ["EKS cluster c has no managed node group g"]
    assert run.provider_calls == ["aws eks list-nodegroups", "kubectl --kubeconfig k"]
    assert run.describe()["document_status"] is None


def test_the_invoker_runs_the_sites_own_cli(tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    class Fixture:
        def run(self, command: list[str], **options: Any) -> SimpleNamespace:
            calls.append((command, options))
            return SimpleNamespace(returncode=0, stdout=None, stderr="err")

    own = tmp_path / "deployer-venv" / "bin"
    own.mkdir(parents=True)
    (own / "gpu-fault-admin").write_text("#!/bin/sh\n", encoding="utf-8")

    run = case.admin_tool(Fixture(), tmp_path)(["--help"], timeout=12.0)

    assert calls == [
        (
            [str(own / "gpu-fault-admin"), case.TOOL_VERB, "--help"],
            {"check": False, "timeout": 12.0},
        )
    ]
    assert (run.returncode, run.stdout, run.stderr) == (0, "", "err")


def test_cli_errors_needs_the_verb_in_the_help() -> None:
    def helped(text: str) -> case.ToolInvoker:
        def invoke(arguments: Any, *, timeout: float) -> case.ToolRun:
            return case.ToolRun(tuple(arguments), 0, text, "")

        return invoke

    assert case.cli_errors(helped("usage ... --apply --confirm"))[1] == []
    assert case.cli_errors(helped("usage: gpu-fault-admin {status,deploy}"))[1] == [
        "the site's CLI has no prepare-eks-nodegroup verb (exit 0)"
    ]


# --- the provider -----------------------------------------------------------------


class Answers:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout

    def run(self, _command: list[str], **_options: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=self.stdout, stderr="")


def test_aws_json_refuses_what_is_not_an_object() -> None:
    with pytest.raises(RegionalFixtureError, match="returned no JSON"):
        case.aws_json(Answers("not json"), "us-west-2", "eks", "describe-cluster")
    with pytest.raises(RegionalFixtureError, match="returned no object"):
        case.aws_json(Answers("[1]"), "us-west-2", "eks", "describe-cluster")
    assert case.aws_json(Answers(""), "us-west-2", "eks", "list-nodegroups") == {}


def test_the_provider_reads_answer_from_the_world(tmp_path: Path) -> None:
    site = Site(tmp_path)
    target = target_of(site)

    documents = case.cluster_nodegroups(site.fixture, target)

    assert sorted(documents) == [CPU_GROUP, GPU_GROUP]
    assert case.running_instances(site.fixture, target, GPU_GROUP) == [
        "i-old1",
        "i-old2",
    ]
    assert case.latest_template_version(site.fixture, target, "lt-0gpu") == 3
    assert case.hyperpod_cluster_arn(site.fixture, "us-west-2", "hp-member-fixture"), (
        'expected: case.hyperpod_cluster_arn(site.fixture, "us-west-2", "hp-member-fixtur'
    )
    empty = Answers(json.dumps({"LaunchTemplateVersions": []}))
    assert case.latest_template_version(empty, target, "lt-0gpu") is None


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ({"labels": {case.OWNERSHIP_LABEL: "site"}}, None),
        ({"status": "UPDATING", "amiType": "AL2_x86_64"}, None),
        ({"status": "ACTIVE", "amiType": "AL2_x86_64"}, "runs amiType AL2_x86_64"),
        ({"status": "ACTIVE", "amiType": ""}, "runs amiType unknown"),
        (
            {"status": "ACTIVE", "amiType": "AL2023_x86_64_NVIDIA"},
            "was created without a launch template",
        ),
        (
            {
                "status": "ACTIVE",
                "amiType": "AL2023_x86_64_NVIDIA",
                "launchTemplate": {"id": "lt-1"},
                "instanceTypes": ["m7i.xlarge"],
            },
            "the ownership label marks GPU node groups only",
        ),
        (
            {
                "status": "ACTIVE",
                "amiType": "AL2023_x86_64_NVIDIA",
                "launchTemplate": {"id": "lt-1"},
                "instanceTypes": ["p5en.48xlarge"],
            },
            None,
        ),
        (
            {
                "status": "ACTIVE",
                "amiType": "AL2023_x86_64_NVIDIA",
                "launchTemplate": {"id": "lt-1"},
            },
            None,
        ),
    ],
)
def test_an_unsupported_group_is_one_the_tool_refuses_for_certain(
    document: dict[str, Any], expected: str | None
) -> None:
    found = case.unsupported_nodegroup({"g": document})

    assert (found[1] if found else None) == expected


# --- Kubernetes and the store -------------------------------------------------------


def test_group_nodes_project_readiness_instance_and_isolation(tmp_path: Path) -> None:
    site = Site(tmp_path)
    held = node_name("i-old1")
    site.cordoned.add(held)
    site.taints[held] = [{"key": "gpu-fault.io/quiesce", "effect": "NoExecute"}]

    nodes = case.group_nodes(site.fixture, GPU_GROUP)

    assert [node["instance_id"] for node in nodes] == ["i-old1", "i-old2"]
    assert nodes[0]["unschedulable"] is True
    assert nodes[0]["blocking_taints"] == ["gpu-fault.io/quiesce:NoExecute"]
    assert nodes[1]["blocking_taints"] == [], "the node group's GPU taint is tolerated"

    class Bare:
        def kubectl(self, *_arguments: Any, **_options: Any) -> str:
            return json.dumps({"items": [{"metadata": {"name": "n"}}, "x"]})

    assert case.group_nodes(Bare(), GPU_GROUP)[0]["ready"] == "Unknown"


def test_the_store_probe_is_read_in_an_api_pod(tmp_path: Path) -> None:
    site = Site(tmp_path)

    state = case.store_state(site.fixture, "gpu-eks-cluster")

    assert site.store_reads == [("gpu-eks-cluster", "")]
    assert {item["lifecycle_state"] for item in state["agents"]} == {"ACTIVE"}


# --- the snapshot ---------------------------------------------------------------------


def test_the_snapshot_reads_configuration_only_for_probed_groups(
    tmp_path: Path,
) -> None:
    site = Site(tmp_path)
    target = target_of(site)

    before = case.snapshot(
        site.fixture, target, [GPU_GROUP], configuration_only=[CPU_GROUP]
    )

    assert before["journals"] == {}, "no journal before the first apply"
    assert "instances" in before["nodegroups"][GPU_GROUP]
    assert "instances" not in before["nodegroups"][CPU_GROUP]
    assert before["nodegroups"][CPU_GROUP]["latest_template_version"] is None
    site.prepare()
    (journal,) = case.snapshot(site.fixture, target, [GPU_GROUP])["journals"]
    assert journal.startswith("eks-nodegroup-prepare/gpu-eks-cluster-"), journal
    assert journal.endswith(f"/{GPU_GROUP}.json"), journal


def test_redaction_drops_credential_lines() -> None:
    text = "a\nGPU_FAULT_TOKEN=1\n--kubeconfig x\nb"

    assert case.redacted(text) == "a\nb"


def test_wait_until_stops_at_the_bound() -> None:
    class Clock(case.Clock):
        def __init__(self) -> None:
            self.ticks = 0.0

        def monotonic(self) -> float:
            return self.ticks

        def sleep(self, seconds: float) -> None:
            self.ticks += seconds

    reached, value, waited = case.wait_until(
        lambda: (False, "never"), timeout_seconds=25, interval_seconds=10, clock=Clock()
    )

    assert (reached, value, waited) == (False, "never", 25.0)


def test_the_real_clock_answers() -> None:
    clock = case.Clock()
    clock.sleep(0)

    assert clock.now().tzinfo is not None and clock.monotonic() > 0


# --- runner plumbing --------------------------------------------------------------------


def test_the_predecessor_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert case.predecessor_gate(None, None, {})["verdict"] == "NOT_REQUIRED"
    monkeypatch.setenv(EXECUTION_SCOPE_ENV, "selective")
    monkeypatch.setenv(SELECTION_REFERENCE_ENV, "CHG-147")

    gate = case.predecessor_gate(
        "GF-REGIONAL-BOOT-036",
        tmp_path / "absent.json",
        {"release_id": "r", "cluster_id": "c"},
    )

    assert gate["valid"] is True and gate["verdict"] == "SKIPPED_BY_OPERATOR"


def test_record_run_keeps_the_document_or_the_console(tmp_path: Path) -> None:
    document = json.dumps({"verb": case.TOOL_VERB, "status": "COMPLIANT"}, indent=2)

    case.record_run(tmp_path, "tool-a", case.ToolRun((), 0, document, "x"))
    case.record_run(tmp_path, "tool-b", case.ToolRun((), 2, "usage\n", "y"))
    case.record_run(tmp_path, "tool-c", case.ToolRun((), 2, "", "z"))

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "tool-a.json",
        "tool-a.stderr.log",
        "tool-b.stderr.log",
        "tool-b.stdout.log",
        "tool-c.stderr.log",
    ]
    with pytest.raises(ValueError, match="approved live-driver plan"):
        case.record_run(tmp_path, "plan", case.ToolRun((), 0, "", ""))


def test_verify_plan_identity_refuses_a_drift(tmp_path: Path) -> None:
    (tmp_path / "plan.json").write_text(
        json.dumps({"details": {"preflight_identity": {"a": 1}}})
    )

    case.verify_plan_identity(tmp_path, "X", {"a": 1})
    with pytest.raises(RegionalFixtureError, match="X plan drifted"):
        case.verify_plan_identity(tmp_path, "X", {"a": 2})


def test_profile_arguments_read_the_installed_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SITE_PROFILE_ENV, raising=False)
    assert case.profile_arguments() == {}
    path = tmp_path / "profile.yaml"
    path.write_text("arguments:\n  fault-node: n1\n", encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    monkeypatch.setenv(SITE_PROFILE_ENV, str(path))

    assert case.profile_arguments() == {"fault-node": "n1"}
