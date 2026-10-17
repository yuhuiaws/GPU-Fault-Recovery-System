"""``gpu-fault-admin prepare-eks-nodegroup``: plan, refusals, apply, resume.

The world (``_eks_nodegroup_world``) is the shape of gpu-eks-cluster: a Spot
``p5en.48xlarge`` managed node group on launch template version 3 (instance
type in the template, EFA interfaces), a CPU node group without a template.
The verb plans read-only (every call is checked for ``mutate``), applies the
planned steps in order, waits bounded for each update and verifies the nodes
over SSM; reruns converge from whatever a stopped run left behind.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import eks_nodegroup_prepare as verb
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.deploy_host_binding import is_readonly_command
from gpu_fault.admin.eks_nodegroup_user_data import (
    DCGM_HOSTENGINE_LABEL,
    DCGM_RESTART_DROPIN,
    MANAGED_PART_MARKER,
    dcgm_pin,
    managed_script,
    merge_managed_part,
    restart_dropin,
)
from gpu_fault.admin.operation_lock import site_operation_lock
from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from tests.admin._eks_nodegroup_world import (
    CPU_EKS_ARN,
    CPU_GROUP,
    GPU_ARN,
    GPU_GROUP,
    LT_ID,
    NODECONFIG_PART,
    PACKAGE_UNIT,
    RELEASE,
    SITE_ID,
    SITE_SCRIPT_PART,
    FakeClock,
    World,
    apply_arguments,
    booted_node,
    encoded,
    journal_file,
    old_node,
    run_verb,
    site_user_data,
)

MUTATING = (
    "ec2 create-launch-template-version",
    "eks update-nodegroup-config",
    "eks update-nodegroup-version",
)


def _run(
    world: World, capsys: pytest.CaptureFixture[str], *arguments: str, **options: Any
) -> tuple[int, dict[str, Any], str]:
    """Exit status, the JSON document and stderr; a raised refusal counts as 2."""

    try:
        code = run_verb(world, *arguments, **options)
        reason = ""
    except BootstrapError as exc:
        code, reason = 2, str(exc)
    captured = capsys.readouterr()
    document = json.loads(captured.out) if captured.out.strip() else {}
    return code, document, captured.err + reason


def _group(document: dict[str, Any]) -> dict[str, Any]:
    (group,) = document["nodegroups"]
    return group


def _prepared(world: World) -> None:
    """The node group as a completed run leaves it (managed part current)."""

    text = merge_managed_part(site_user_data(), managed_script()).text
    world.templates[LT_ID][3]["data"]["UserData"] = encoded(text)
    document = world.nodegroups[GPU_GROUP]
    document["labels"] = {**document["labels"], OWNERSHIP_LABEL: SITE_ID}
    document["nodeRepairConfig"] = {"enabled": False}
    document["updateConfig"] = {"maxUnavailable": 1, "updateStrategy": "MINIMAL"}


def _no_writes(world: World) -> bool:
    return not any(flag for _argv, flag in world.calls)


# --- CLI ---------------------------------------------------------------------


def test_the_verb_is_registered_and_read_only_until_apply(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = admin_cli.parser()
    base = [verb.VERB, "--cpu-cluster-arn", CPU_EKS_ARN, "--gpu-cluster-arn", GPU_ARN]

    plan = parser.parse_args(base)
    applying = parser.parse_args(
        [*base, "--work-dir", "/tmp/w", "--apply", "--confirm", verb.CONFIRMATION]
    )

    assert plan.command == verb.VERB
    assert plan.state_dir is None and plan.work_dir is None, (
        "the verb runs before any site exists"
    )
    assert is_readonly_command(plan, readonly_commands=admin_cli.READONLY_COMMANDS), (
        "the plan writes nothing: its log rotates under logs/readonly"
    )
    assert not is_readonly_command(
        applying, readonly_commands=admin_cli.READONLY_COMMANDS
    ), "--apply is a mutation"
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--state-dir", "/a", "--work-dir", "/b"])
    with pytest.raises(SystemExit):
        parser.parse_args([verb.VERB, "--gpu-cluster-arn", GPU_ARN])
    capsys.readouterr()
    with pytest.raises(SystemExit):
        admin_cli.parser().parse_args(["--help"])
    assert verb.VERB in capsys.readouterr().out, "listed among the public verbs"


def test_cli_run_dispatches_to_the_verb(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        verb, "run_command", lambda arguments: seen.append(arguments.command) or 7
    )
    arguments = admin_cli.parser().parse_args(
        [verb.VERB, "--cpu-cluster-arn", CPU_EKS_ARN, "--gpu-cluster-arn", GPU_ARN]
    )

    assert admin_cli.run(arguments) == 7
    assert seen == [verb.VERB]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--apply"], "--apply requires --confirm PREPARE_EKS_NODEGROUP"),
        (["--apply", "--confirm", "yes"], "--apply requires --confirm"),
        (["--confirm", "PREPARE_EKS_NODEGROUP"], "only used with --apply"),
        (
            ["--work-dir", "/tmp/w", "--apply", "--confirm", "PREPARE_EKS_NODEGROUP"],
            "--apply requires --reference",
        ),
        (
            ["--apply", "--confirm", "PREPARE_EKS_NODEGROUP", "--reference", "CHG-1"],
            "--state-dir \\(a site\\) or --work-dir",
        ),
        (["--reference", "x"], "--reference value is invalid"),
        (["--dcgm-version", "4.7.0-1"], "only with --dcgm-hostengine host"),
        (["--dcgm-hostengine", "host", "--dcgm-version", "4.7;reboot"], "version"),
        (["--nodegroup", "bad name"], "invalid --nodegroup"),
        (["--roll-timeout-minutes", "0"], "1..360"),
        (["--roll-timeout-minutes", "361"], "1..360"),
    ],
)
def test_flag_combinations_are_refused_before_any_aws_call(
    capsys: pytest.CaptureFixture[str], arguments: list[str], message: str
) -> None:
    world = World()

    with pytest.raises(BootstrapError, match=message):
        run_verb(world, *arguments)
    assert world.calls == [], "nothing is read for a refused command line"


# --- plan --------------------------------------------------------------------


def test_plan_of_an_unprepared_group_lists_the_commands_in_order_and_writes_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = World(user_data=site_user_data())

    code, document, err = _run(world, capsys)

    assert code == 0
    assert _no_writes(world), [argv for argv, flag in world.calls if flag]
    assert document["status"] == "CHANGES"
    assert document["site_id"] == SITE_ID
    group = _group(document)
    assert group["nodegroup"] == GPU_GROUP, (
        "the CPU group is no GPU inventory group; the GPU group is found through "
        "its launch template's instance type"
    )
    assert group["instance_types"] == ["p5en.48xlarge"]
    steps = group["steps"]
    assert [item["step"] for item in steps] == [
        "launch-template-version",
        "node-repair",
        "labels",
        "update-config",
        "roll",
    ]
    create = steps[0]["command"]
    assert f"--launch-template-id {LT_ID} --source-version 3" in create
    payload = json.loads(create.split("--launch-template-data ", 1)[1].split("'")[1])
    assert set(payload) == {"UserData"}, "only UserData changes in the new version"
    merged = base64.b64decode(payload["UserData"]).decode()
    assert merged == group["merged_user_data"]
    assert NODECONFIG_PART in merged and SITE_SCRIPT_PART in merged
    assert MANAGED_PART_MARKER in merged
    assert steps[1]["command"].endswith(
        "--node-repair-config '{\"enabled\":false}' --region us-west-2 --output json"
    ), "node auto repair is switched off explicitly"
    assert json.loads(steps[2]["command"].split("--labels ", 1)[1].split("'")[1]) == {
        "addOrUpdateLabels": {OWNERSHIP_LABEL: SITE_ID}
    }
    assert (
        f'--launch-template \'{{"id":"{LT_ID}","version":"<new version>"}}\' '
        f"--release-version {RELEASE}"
    ) in steps[4]["command"], "the roll keeps the AMI release version"
    assert "1 of 2 GPU nodes out at a time" in group["roll"]["caveat"], (
        "the roll after the update-config step terminates before it launches"
    )
    assert "Spot" in group["roll"]["caveat"]
    assert group["user_data"]["action"] == "add"
    assert document["apply_command"].endswith(
        "--work-dir <WORK_DIR> --apply --confirm PREPARE_EKS_NODEGROUP "
        "--reference <CHANGE_ID>"
    ), "the plan prints the apply command to review"
    assert f"{GPU_GROUP}: would create launch template" in err
    assert sorted(world.templates[LT_ID]) == [1, 3], "no launch template version"


def test_plan_of_a_prepared_group_is_compliant(
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = World()
    _prepared(world)

    code, document, err = _run(world, capsys)

    assert code == 0
    assert document["status"] == "COMPLIANT"
    group = _group(document)
    assert (group["verdict"], group["steps"]) == ("COMPLIANT", [])
    assert group["user_data"]["action"] == "keep"
    assert group["roll"] == {"needed": False}
    assert group["current_nodes"]["outcome"] == "passed"
    assert f"{GPU_GROUP}: compliant" in err
    assert _no_writes(world), "the plan of a compliant group writes nothing"


@pytest.mark.parametrize(
    ("labels", "flags", "expected"),
    [
        ({}, (), {"addOrUpdateLabels": {OWNERSHIP_LABEL: SITE_ID}}),
        (
            {OWNERSHIP_LABEL: SITE_ID},
            ("--dcgm-hostengine", "host"),
            {"addOrUpdateLabels": {DCGM_HOSTENGINE_LABEL: "host"}},
        ),
        ({OWNERSHIP_LABEL: SITE_ID, DCGM_HOSTENGINE_LABEL: "embedded"}, (), None),
        (
            {OWNERSHIP_LABEL: SITE_ID, DCGM_HOSTENGINE_LABEL: "bogus"},
            ("--dcgm-hostengine", "host"),
            {"addOrUpdateLabels": {DCGM_HOSTENGINE_LABEL: "host"}},
        ),
        (
            {OWNERSHIP_LABEL: SITE_ID, DCGM_HOSTENGINE_LABEL: "host"},
            ("--dcgm-hostengine", "embedded"),
            {"removeLabels": [DCGM_HOSTENGINE_LABEL]},
        ),
    ],
    ids=[
        "absent",
        "ownership-only-host-mode",
        "dcgm-embedded-kept",
        "dcgm-other-value-host-flag",
        "dcgm-host-switched-back",
    ],
)
def test_label_states_plan_exactly_the_missing_or_wrong_labels(
    capsys: pytest.CaptureFixture[str],
    labels: dict[str, str],
    flags: tuple[str, ...],
    expected: dict[str, Any] | None,
) -> None:
    world = World(user_data=site_user_data())
    world.nodegroups[GPU_GROUP]["labels"] = {"node-type": "gpu", **labels}

    code, document, _err = _run(world, capsys, *flags)

    assert code == 0
    steps = {item["step"]: item["command"] for item in _group(document)["steps"]}
    if expected is None:
        assert "labels" not in steps
    else:
        assert json.loads(steps["labels"].split("--labels ", 1)[1].split("'")[1]) == (
            expected
        ), "other labels are never touched: addOrUpdate/remove name only these"


def test_an_unknown_dcgm_label_value_is_refused_without_a_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = World()
    world.nodegroups[GPU_GROUP]["labels"] = {DCGM_HOSTENGINE_LABEL: "bogus"}

    code, document, err = _run(world, capsys)

    assert code == 2
    assert "declares gpu-fault.io/dcgm-hostengine=bogus" in _group(document)["refusal"]
    assert "cannot be prepared automatically" in err


@pytest.mark.parametrize(
    ("repair", "summary"),
    [
        ({"enabled": True}, "EKS node auto repair enabled -> disabled"),
        (None, "EKS node auto repair unset -> disabled"),
        ({"enabled": False}, None),
    ],
    ids=["enabled", "unset", "disabled"],
)
def test_node_repair_states(
    capsys: pytest.CaptureFixture[str],
    repair: dict[str, bool] | None,
    summary: str | None,
) -> None:
    world = World()
    _prepared(world)
    world.nodegroups[GPU_GROUP]["nodeRepairConfig"] = repair

    code, document, _err = _run(world, capsys)

    assert code == 0
    steps = _group(document)["steps"]
    if summary is None:
        assert steps == [], "disabled is compliant"
    else:
        assert [(item["step"], item["summary"]) for item in steps] == [
            ("node-repair", summary)
        ], "only the repair setting changes: no user data change, no roll"


@pytest.mark.parametrize(
    ("user_data", "action", "steps"),
    [
        (None, "create", ["launch-template-version", "roll"]),
        (
            'MIME-Version: 1.0\nContent-Type: multipart/mixed; boundary="//"\n\n--//\n'
            + NODECONFIG_PART
            + "--//--\n",
            "add",
            ["launch-template-version", "roll"],
        ),
    ],
    ids=["no-user-data", "nodeconfig-only"],
)
def test_user_data_states_without_a_managed_part(
    capsys: pytest.CaptureFixture[str],
    user_data: str | None,
    action: str,
    steps: list[str],
) -> None:
    world = World(user_data=user_data)
    _prepared(world)
    if user_data is None:
        world.templates[LT_ID][3]["data"].pop("UserData")
    else:
        world.templates[LT_ID][3]["data"]["UserData"] = encoded(user_data)

    code, document, _err = _run(world, capsys)

    assert code == 0
    group = _group(document)
    assert group["user_data"]["action"] == action
    assert [item["step"] for item in group["steps"]] == steps
    if user_data is not None:
        assert NODECONFIG_PART in group["merged_user_data"], "NodeConfig untouched"


def test_an_outdated_managed_part_is_replaced_and_only_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = World()
    _prepared(world)
    stale = merge_managed_part(
        site_user_data(), managed_script().replace("sleep 30", "sleep 10")
    ).text
    world.templates[LT_ID][3]["data"]["UserData"] = encoded(stale)

    code, document, _err = _run(world, capsys)

    assert code == 0
    group = _group(document)
    assert group["user_data"]["action"] == "replace"
    assert [item["step"] for item in group["steps"]] == [
        "launch-template-version",
        "roll",
    ]
    merged = group["merged_user_data"]
    assert merged.startswith(
        stale[
            : stale.index(
                'Content-Disposition: attachment; filename="gpu-fault-managed'
            )
        ]
    ), "the parts before the managed one are untouched"
    assert "sleep 10" not in merged and "sleep 30" in merged


def test_a_hand_written_part_is_kept_reported_and_adopted_only_on_request(
    capsys: pytest.CaptureFixture[str],
) -> None:
    hand_written = site_user_data().replace(
        "dnf install -y -q datacenter-gpu-manager-4-core",
        "dnf install -y -q python3.12 strace datacenter-gpu-manager-4-core",
    )
    world = World(user_data=hand_written)

    code, document, err = _run(world, capsys)

    assert code == 0
    parts = _group(document)["user_data"]["parts"]
    site_part = parts[1]
    assert site_part["filename"] == "site-tools.sh"
    assert site_part["disposition"] == "kept"
    assert site_part["overlaps"] == [
        "part 2 line 2: python3.12 (the managed part installs it too)",
        "part 2 line 2: strace (the managed part installs it too)",
    ]
    assert site_part["other_packages"] == [
        "part 2 line 2: datacenter-gpu-manager-4-core",
        "part 2 line 2: datacenter-gpu-manager-4-cuda13",
    ]
    assert "site-tools.sh" in _group(document)["merged_user_data"]
    assert (
        f"finding: {GPU_GROUP}: user data part 2 (site-tools.sh) is hand-written and "
        "kept beside the managed part"
    ) in err, "the default keep is said on the console, with the adopt flag"
    assert "--adopt-user-data-part 2 replaces it" in err

    code, adopted, err = _run(world, capsys, "--adopt-user-data-part", "site-tools.sh")

    assert code == 0
    group = _group(adopted)
    assert group["user_data"]["action"] == "adopt"
    assert group["user_data"]["parts"][1]["disposition"] == "adopted"
    assert group["user_data"]["parts"][1]["adoption_drops"] == [
        "datacenter-gpu-manager-4-core",
        "datacenter-gpu-manager-4-cuda13",
    ]
    assert "site-tools.sh" not in group["merged_user_data"]
    assert (
        f"finding: {GPU_GROUP}: adopting part 2 drops datacenter-gpu-manager-4-core, "
        "datacenter-gpu-manager-4-cuda13"
    ) in err


def test_the_printed_apply_command_repeats_every_plan_argument(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(user_data=site_user_data())
    work = tmp_path / "work dir"

    code, document, _err = _run(
        world,
        capsys,
        "--nodegroup",
        GPU_GROUP,
        "--work-dir",
        str(work),
        "--dcgm-hostengine",
        "host",
        "--dcgm-version",
        "4.7.0",
        "--allowed-namespace",
        "research",
        "--adopt-user-data-part",
        "site-tools.sh",
        "--roll-timeout-minutes",
        "45",
    )

    assert code == 0
    assert document["apply_command"] == (
        f"gpu-fault-admin prepare-eks-nodegroup --cpu-cluster-arn {CPU_EKS_ARN} "
        f"--gpu-cluster-arn {GPU_ARN} --nodegroup {GPU_GROUP} --work-dir "
        f"'{work}' --dcgm-hostengine host --dcgm-version 4.7.0 "
        "--allowed-namespace research --adopt-user-data-part site-tools.sh "
        "--roll-timeout-minutes 45 --apply --confirm PREPARE_EKS_NODEGROUP "
        "--reference <CHANGE_ID>"
    ), "an operator who copies it applies exactly what was planned"


def test_the_plan_shows_the_current_nodes_and_their_dcgm_drift(
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = World(user_data=site_user_data())
    world.node_facts = {
        "i-old1": old_node(
            packages={
                "datacenter-gpu-manager-4-core": "4.7.0-1",
                "datacenter-gpu-manager-4-cuda13": "4.7.0-1",
            }
        ),
        "i-old2": old_node(
            service="active",
            packages={
                "datacenter-gpu-manager-4-core": "4.7.0-2",
                "datacenter-gpu-manager-4-cuda13": "4.7.0-2",
            },
        ),
    }

    code, document, err = _run(world, capsys, "--dcgm-hostengine", "host")

    assert code == 0
    group = _group(document)
    nodes = group["current_nodes"]
    assert [item["facts"]["dcgm_packages"] for item in nodes["nodes"]] == [
        {
            "datacenter-gpu-manager-4-core": "4.7.0-1",
            "datacenter-gpu-manager-4-cuda13": "4.7.0-1",
        },
        {
            "datacenter-gpu-manager-4-core": "4.7.0-2",
            "datacenter-gpu-manager-4-cuda13": "4.7.0-2",
        },
    ]
    assert (
        f"finding: {GPU_GROUP}: nodes: datacenter-gpu-manager-4-core differs across "
        "the nodes: i-old1 4.7.0-1, i-old2 4.7.0-2"
    ) in err
    assert group["dcgm_hostengine"] == {
        "mode": "host",
        "source": "flag",
        "pin": "4.7.0-2",
        "installs": "exactly build 4.7.0-2 of both DCGM packages",
        "restart_dropin": restart_dropin(),
    }, "the plan names the restart drop-in host mode writes"
    site_part = group["user_data"]["parts"][1]
    assert (
        "part 2 line 2: datacenter-gpu-manager-4-core is unpinned (the newest build), "
        "the pin is 4.7.0-2"
    ) in site_part["contradictions"]
    assert _no_writes(world), "the inventory is a read-only SSM command"

    code, version_pinned, _err = _run(
        world, capsys, "--dcgm-hostengine", "host", "--dcgm-version", "4.7.0"
    )
    assert _group(version_pinned)["dcgm_hostengine"]["installs"].startswith(
        "version 4.7.0, any build"
    ), "a version pin states that any build is accepted"


def test_the_plan_reports_a_host_engine_running_in_embedded_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = World()
    _prepared(world)
    world.node_facts = {"i-old1": old_node(service="active")}

    code, document, err = _run(world, capsys)

    assert code == 0, "a finding never blocks"
    assert (
        f"{GPU_GROUP}: nodes: i-old1: a host DCGM engine is active (nvidia-dcgm "
        "active, nv-hostengine absent); it conflicts with the DCGM exporter's "
        "embedded engine"
    ) in err
    assert _group(document)["verdict"] == "COMPLIANT"


def test_kubernetes_findings_are_reported_with_the_plan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = World()
    _prepared(world)
    world.namespaces = ["default"]

    code, document, err = _run(world, capsys)

    assert code == 0
    assert document["findings"]["namespaces"]["missing"] == ["training"]
    assert "finding: allowed workload namespace(s) missing: training" in err


# --- refusals ----------------------------------------------------------------


def _ami(world: World) -> None:
    world.nodegroups[GPU_GROUP]["amiType"] = "AL2_x86_64_GPU"


def _custom(world: World) -> None:
    world.nodegroups[GPU_GROUP]["amiType"] = "CUSTOM"
    world.templates[LT_ID][3]["data"]["ImageId"] = "ami-0custom"


def _no_template(world: World) -> None:
    world.nodegroups[GPU_GROUP]["launchTemplate"] = None
    world.nodegroups[GPU_GROUP]["instanceTypes"] = ["p5en.48xlarge"]


def _image(world: World) -> None:
    world.templates[LT_ID][3]["data"]["ImageId"] = "ami-0custom"


def _degraded(world: World) -> None:
    world.nodegroups[GPU_GROUP]["status"] = "DEGRADED"


def _updating(world: World) -> None:
    world.nodegroups[GPU_GROUP]["status"] = "UPDATING"


def _not_gpu(world: World) -> None:
    world.templates[LT_ID][3]["data"]["InstanceType"] = "g6.12xlarge"


def _not_mime(world: World) -> None:
    world.templates[LT_ID][3]["data"]["UserData"] = encoded("#!/bin/bash\necho\n")


@pytest.mark.parametrize(
    ("break_it", "message"),
    [
        (_ami, "runs amiType AL2_x86_64_GPU"),
        (_custom, "runs amiType CUSTOM"),
        (_no_template, "created without a launch template and cannot start using one"),
        (_image, "names ImageId ami-0custom"),
        (_degraded, "is DEGRADED, not ACTIVE"),
        (_updating, "is UPDATING, not ACTIVE, and no update of this verb is pending"),
        (_not_gpu, "runs g6.12xlarge instance types"),
        (_not_mime, "not a MIME multipart document"),
    ],
    ids=[
        "ami-type",
        "custom-ami",
        "no-launch-template",
        "custom-image-id",
        "degraded",
        "updating-without-journal",
        "not-gpu",
        "user-data-not-mime",
    ],
)
def test_groups_the_verb_cannot_prepare_are_refused_with_the_manual_steps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], break_it: Any, message: str
) -> None:
    world = World(user_data=site_user_data())
    break_it(world)

    code, document, err = _run(
        world, capsys, "--nodegroup", GPU_GROUP, *apply_arguments(tmp_path)
    )

    assert code == 2
    group = _group(document)
    assert group["verdict"] == "REFUSED"
    assert message in group["refusal"]
    assert "administrator operations §5.0.1" in group["refusal"]
    assert document["manual_steps"].startswith("the manual steps of administrator"), (
        "the refusal points at the manual steps"
    )
    assert "cannot be prepared automatically; nothing was changed" in err
    assert _no_writes(world), "an apply over a refused group writes nothing"
    assert not (tmp_path / "eks-nodegroup-prepare").exists(), "no journal either"


def test_hyperpod_clusters_are_never_touched(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sagemaker = World()
    code, _document, err = _run(
        sagemaker,
        capsys,
        gpu_arn="arn:aws:sagemaker:us-west-2:123456789012:cluster/hp-gpu",
    )
    assert code == 2 and "never touches HyperPod instance groups" in err
    assert sagemaker.calls == []

    orchestrated = World()
    orchestrated.hyperpods = [
        {
            "ClusterName": "hp-gpu",
            "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/hp-gpu",
            "Orchestrator": {"Eks": {"ClusterArn": GPU_ARN}},
        }
    ]
    code, _document, err = _run(orchestrated, capsys)
    assert code == 2 and "orchestrated by HyperPod cluster hp-gpu" in err
    assert orchestrated.aws_operations() == [
        "sagemaker list-clusters",
        "sagemaker describe-cluster",
    ], "nothing of the EKS cluster is read once HyperPod owns it"


def test_cluster_level_refusals(capsys: pytest.CaptureFixture[str]) -> None:
    foreign = World()
    foreign.nodegroups[CPU_GROUP]["labels"] = {OWNERSHIP_LABEL: "another-site"}
    code, _document, err = _run(foreign, capsys)
    assert code == 2
    assert f"{CPU_GROUP} carries {OWNERSHIP_LABEL}=another-site" in err
    assert "One GPU cluster belongs to one site" in err

    inactive = World()
    inactive.cluster_status = "UPDATING"
    assert "is UPDATING, not ACTIVE" in _run(inactive, capsys)[2]

    missing = World()
    assert (
        "has no managed node group nope"
        in _run(missing, capsys, "--nodegroup", "nope")[2]
    )

    cpu = World()
    assert "is the CPU control plane" in _run(cpu, capsys, gpu_arn=CPU_EKS_ARN)[2]

    no_gpu = World()
    del no_gpu.nodegroups[GPU_GROUP]
    assert (
        "no managed node group running GPU inventory instance types"
        in _run(no_gpu, capsys)[2]
    )


# --- apply -------------------------------------------------------------------


def test_apply_runs_the_planned_steps_in_order_then_verifies_the_new_nodes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The release gate runs inside `gpu-fault-admin deploy`, whose own log
    # (GPU_FAULT_ADMIN_LOG) a nested admin command writes into instead of
    # opening one; the work-dir log asserted below is the stand-alone case.
    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    world = World(user_data=site_user_data())

    code, document, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 0, err
    assert world.aws_operations(mutate=True) == [
        "ec2 create-launch-template-version",
        "eks update-nodegroup-config",
        "eks update-nodegroup-config",
        "eks update-nodegroup-config",
        "eks update-nodegroup-version",
    ]
    assert not [
        operation
        for operation in world.aws_operations(mutate=False)
        if operation in MUTATING
    ], "every write is marked as one"
    operations = world.aws_operations()
    roll = operations.index("eks update-nodegroup-version")
    assert operations.index("ec2 create-launch-template-version") < operations.index(
        "eks update-nodegroup-config"
    )
    assert "eks describe-update" in operations[roll:], "the roll is waited for"
    assert operations[-3:] == [
        "ssm send-command",
        "ssm get-command-invocation",
        "ssm get-command-invocation",
    ], "the verification reads both new nodes last"
    version = world.templates[LT_ID][4]
    assert version["data"]["InstanceType"] == "p5en.48xlarge", "the source carried over"
    assert version["data"]["NetworkInterfaces"] == [{"InterfaceType": "efa"}]
    assert MANAGED_PART_MARKER in world.template_user_data(4)
    group = world.nodegroups[GPU_GROUP]
    assert group["launchTemplate"]["version"] == "4"
    assert group["labels"][OWNERSHIP_LABEL] == SITE_ID
    assert group["labels"]["node-type"] == "gpu", "other labels are untouched"
    assert group["nodeRepairConfig"] == {"enabled": False}
    (rolled,) = [
        item for item in world.updates.values() if item["kind"] == "VersionUpdate"
    ]
    assert (rolled["version"], rolled["release"]) == ("4", RELEASE)
    assert world.sent["instances"] == ["i-new1a", "i-new1b"], "the new nodes"
    result = document["apply"]
    assert result["status"] == "COMPLETED" and result["reference"] == "CHG-121"
    (outcome,) = result["results"]
    assert outcome["outcome"] == "COMPLETED"
    assert outcome["verification"]["outcome"] == "passed"
    journal = json.loads(journal_file(tmp_path).read_text(encoding="utf-8"))
    (attempt,) = journal["attempts"]
    assert attempt["operator"].startswith("arn:aws:sts::"), (
        "the journal names the STS identity of the operator"
    )
    assert attempt["reference"] == "CHG-121"
    assert [item["step"] for item in attempt["steps"]] == [
        "launch-template-version",
        "node-repair",
        "labels",
        "update-config",
        "roll",
        "verification",
    ]
    assert attempt["outcome"] == "COMPLETED" and journal["pending_update"] is None
    logs = list((tmp_path / "logs" / "mutating").glob("prepare-eks-nodegroup-*.log"))
    assert len(logs) == 1, "a work-dir apply keeps its own log"


def test_a_second_apply_changes_nothing_and_only_reverifies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(user_data=site_user_data())
    assert _run(world, capsys, *apply_arguments(tmp_path))[0] == 0
    world.calls.clear()

    code, document, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 0, err
    assert world.aws_operations(mutate=True) == [], "every step is already done"
    assert "ssm send-command" in world.aws_operations(), "the nodes are re-verified"
    assert sorted(world.templates[LT_ID]) == [1, 3, 4]
    assert _group(document)["verdict"] == "COMPLIANT"
    journal = json.loads(journal_file(tmp_path).read_text(encoding="utf-8"))
    assert [item["outcome"] for item in journal["attempts"]] == [
        "COMPLETED",
        "COMPLETED",
    ]


def test_node_creation_failure_is_a_retryable_stop_and_the_rerun_converges(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(user_data=site_user_data())
    world.roll_results = [(3, "Failed")]

    code, document, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 2
    assert "ROLL_FAILED" in err and "NodeCreationFailure" in err
    assert "rerun the same command later" in err
    assert "the prepared launch template version is reused" in err
    (outcome,) = document["apply"]["results"]
    assert outcome["outcome"] == "ROLL_FAILED"
    group = world.nodegroups[GPU_GROUP]
    assert (group["status"], group["launchTemplate"]["version"]) == ("ACTIVE", "3"), (
        "EKS kept the old nodes and launch template version"
    )
    journal = json.loads(journal_file(tmp_path).read_text(encoding="utf-8"))
    assert journal["pending_update"] is None
    world.calls.clear()

    code, document, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 0, err
    assert world.aws_operations(mutate=True) == ["eks update-nodegroup-version"], (
        "the launch template version, repair and labels are not done twice"
    )
    assert sorted(world.templates[LT_ID]) == [1, 3, 4]
    assert world.nodegroups[GPU_GROUP]["launchTemplate"]["version"] == "4"
    steps = document["apply"]["results"][0]["steps"]
    assert steps[0] == {
        "at": "2030-01-02T03:04:05+00:00",
        "outcome": "reused",
        "step": "launch-template-version",
        "version": 4,
    }


def test_a_roll_past_its_bound_stops_and_the_rerun_resumes_the_same_update(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(user_data=site_user_data())
    world.roll_results = [(10_000, "Successful")]
    clock = FakeClock()

    code, _document, err = _run(
        world,
        capsys,
        *apply_arguments(tmp_path, "--roll-timeout-minutes", "1"),
        clock=clock,
    )

    assert code == 2
    assert "ROLL_IN_PROGRESS" in err and "rerun the same command to keep waiting" in err
    assert max(clock.slept) <= 30.0, "polls every 30 seconds at most"
    journal = json.loads(journal_file(tmp_path).read_text(encoding="utf-8"))
    pending = journal["pending_update"]
    assert pending["step"] == "roll"
    assert world.nodegroups[GPU_GROUP]["status"] == "UPDATING"

    code, plan, _err = _run(world, capsys, "--work-dir", str(tmp_path))
    assert code == 0
    assert _group(plan)["verdict"] == "IN_PROGRESS", "the plan names the pending update"
    assert _group(plan)["pending_update"]["update_id"] == pending["update_id"]

    world.updates[pending["update_id"]]["polls"] = 1
    world.calls.clear()
    code, document, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 0, err
    assert "eks update-nodegroup-version" not in world.aws_operations(), (
        "the rerun waits for the journaled update instead of starting another"
    )
    steps = [item["step"] for item in document["apply"]["results"][0]["steps"]]
    assert steps[:2] == ["resume-roll", "replan"]
    assert steps[-1] == "verification"
    assert (
        json.loads(journal_file(tmp_path).read_text(encoding="utf-8"))["pending_update"]
        is None
    )


def test_a_group_changed_since_the_plan_stops_before_the_next_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class Racing(World):
        def ec2_create_launch_template_version(self, options):  # type: ignore[no-untyped-def]
            created = super().ec2_create_launch_template_version(options)
            self.nodegroups[GPU_GROUP]["launchTemplate"]["version"] = "1"
            return created

    world = Racing(user_data=site_user_data())

    code, document, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 2
    assert "CHANGED_SINCE_PLAN" in err
    assert (
        "now runs launch template lt-0gpu version 1, the plan saw lt-0gpu version 3"
        in err
    )
    assert world.aws_operations(mutate=True) == ["ec2 create-launch-template-version"]


def test_failed_or_impossible_verification_stops_with_the_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = World(user_data=site_user_data())
    missing.instances = {GPU_GROUP: ["i-old1", "i-old2"]}

    def boot_without_packages(world: World) -> None:
        original = world.eks_describe_update

        def describe(options):  # type: ignore[no-untyped-def]
            result = original(options)
            for instance in world.instances.get(GPU_GROUP, []):
                if instance.startswith("i-new"):
                    world.node_facts[instance] = old_node(strace="missing")
            return result

        world.eks_describe_update = describe  # type: ignore[method-assign]

    boot_without_packages(missing)
    code, _document, err = _run(missing, capsys, *apply_arguments(tmp_path / "a"))
    assert code == 2 and "VERIFICATION_FAILED" in err
    assert "i-new1a (Success: strace missing (expected ok))" in err

    denied = World(user_data=site_user_data())
    denied.ssm_send_errors = ["AccessDeniedException: ssm:SendCommand"] * 2
    code, _document, err = _run(denied, capsys, *apply_arguments(tmp_path / "b"))
    assert code == 2 and "UNVERIFIED" in err
    assert "ssm:SendCommand and ssm:GetCommandInvocation" in err


def test_a_refused_write_names_the_permission_and_is_journaled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class Denied(World):
        def eks_update_nodegroup_config(self, options):  # type: ignore[no-untyped-def]
            raise BootstrapError("AccessDeniedException: eks:UpdateNodegroupConfig")

    world = Denied(user_data=site_user_data())

    code, _document, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 2
    assert "The operator's AWS identity needs eks:UpdateNodegroupConfig" in err
    assert "aws eks update-nodegroup-config --cluster-name gpu-eks-cluster" in err, (
        "the step can be run by hand"
    )
    journal = json.loads(journal_file(tmp_path).read_text(encoding="utf-8"))
    (attempt,) = journal["attempts"]
    assert attempt["outcome"] == "FAILED"
    assert [item["step"] for item in attempt["steps"]] == ["launch-template-version"]


def test_a_journaled_update_eks_no_longer_knows_is_cleared_and_planned_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class Forgetful(World):
        def eks_describe_update(self, options):  # type: ignore[no-untyped-def]
            if options["--update-id"][0] not in self.updates:
                self.nodegroups[GPU_GROUP]["status"] = "ACTIVE"
                raise BootstrapError("ResourceNotFoundException: No update found")
            return super().eks_describe_update(options)

    world = Forgetful(user_data=site_user_data())
    _prepared(world)
    world.nodegroups[GPU_GROUP]["status"] = "UPDATING"
    assert _run(world, capsys, "--work-dir", str(tmp_path))[0] == 2, (
        "UPDATING without a journaled update is refused"
    )
    world.nodegroups[GPU_GROUP]["status"] = "ACTIVE"
    assert _run(world, capsys, *apply_arguments(tmp_path))[0] == 0
    path = journal_file(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["pending_update"] = {"update_id": "update-99", "step": "roll"}
    path.write_text(json.dumps(document), encoding="utf-8")
    world.nodegroups[GPU_GROUP]["status"] = "UPDATING"

    code, result, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 0, err
    steps = [item["step"] for item in result["apply"]["results"][0]["steps"]]
    assert steps[:2] == ["resume-roll", "replan"]
    assert json.loads(path.read_text(encoding="utf-8"))["pending_update"] is None


def test_a_stale_pending_entry_of_an_active_group_is_cleared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World()
    _prepared(world)
    assert _run(world, capsys, *apply_arguments(tmp_path))[0] == 0
    path = journal_file(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["pending_update"] = {"update_id": "update-old", "step": "labels"}
    path.write_text(json.dumps(document), encoding="utf-8")

    code, result, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 0, err
    assert result["apply"]["results"][0]["steps"][0]["step"] == "pending-cleared"
    assert json.loads(path.read_text(encoding="utf-8"))["pending_update"] is None


def test_host_mode_apply_pins_dcgm_labels_the_group_and_verifies_dcgm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(user_data=site_user_data())

    code, document, err = _run(
        world, capsys, *apply_arguments(tmp_path, "--dcgm-hostengine", "host")
    )

    assert code == 0, err
    script = world.template_user_data(4)
    assert "dcgm_version='4.7.0-2'" in script
    assert "systemctl enable --now nvidia-dcgm" in script
    labels = world.nodegroups[GPU_GROUP]["labels"]
    assert (labels[OWNERSHIP_LABEL], labels[DCGM_HOSTENGINE_LABEL]) == (SITE_ID, "host")
    nodes = document["apply"]["results"][0]["verification"]["nodes"]
    assert {
        name for item in nodes for name, check in item["checks"].items() if check["ok"]
    } >= {
        "python3.12-venv",
        "strace",
        "datacenter-gpu-manager-4-core",
        "dcgm-cuda-flavor",
        "dcgm-cuda-version",
        "nvidia-dcgm",
        "nvidia-dcgm-restart",
        "nvidia-dcgm-restart-dropin",
        "dcgmi-discovery",
    }
    assert {
        item["instance_id"]: (
            item["facts"]["nvidia-dcgm-unit"]["Restart"],
            item["facts"]["nvidia-dcgm-restart-dropin"],
        )
        for item in nodes
    } == {"i-new1a": ("on-failure", "present"), "i-new1b": ("on-failure", "present")}, (
        "every new node reports its restart policy and the drop-in"
    )
    assert script.index(DCGM_RESTART_DROPIN) < script.index(
        "systemctl enable --now nvidia-dcgm"
    )
    world.calls.clear()

    code, document, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 0, err
    assert world.aws_operations(mutate=True) == [], (
        "without the flag the group's label keeps host mode and its pin"
    )
    assert _group(document)["dcgm_hostengine"]["source"] == "label"

    code, plan, _err = _run(world, capsys, "--dcgm-hostengine", "embedded")
    group = _group(plan)
    assert [item["step"] for item in group["steps"]] == [
        "launch-template-version",
        "labels",
        "roll",
    ], "switching back removes the DCGM lines and the label, and rolls"
    assert "nvidia-dcgm" not in group["managed_part_script"]


def _host_prepared_before_the_dropin(world: World) -> str:
    """The group as a host-mode run of the verb left it before the drop-in."""

    full = managed_script(dcgm=dcgm_pin("4.7.0-2"))
    head, rest = full.split("# nvidia-dcgm restarts after a failure", 1)
    older = head + rest[rest.index("systemctl enable --now nvidia-dcgm") :]
    text = merge_managed_part(site_user_data(), older).text
    world.templates[LT_ID][3]["data"]["UserData"] = encoded(text)
    document = world.nodegroups[GPU_GROUP]
    document["labels"] = {
        **document["labels"],
        OWNERSHIP_LABEL: SITE_ID,
        DCGM_HOSTENGINE_LABEL: "host",
    }
    document["nodeRepairConfig"] = {"enabled": False}
    document["updateConfig"] = {"maxUnavailable": 1, "updateStrategy": "MINIMAL"}
    world.node_facts = {
        instance: booted_node(text) for instance in world.instances[GPU_GROUP]
    }
    return older


def test_the_host_mode_plan_shows_the_restart_dropin(
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = World(user_data=site_user_data())

    code, document, err = _run(world, capsys, "--dcgm-hostengine", "host")

    assert code == 0
    group = _group(document)
    assert group["dcgm_hostengine"]["restart_dropin"] == restart_dropin()
    assert DCGM_RESTART_DROPIN in group["managed_part_script"]
    assert DCGM_RESTART_DROPIN in group["merged_user_data"]
    assert (
        f"{GPU_GROUP}: host DCGM: the managed part writes the nvidia-dcgm drop-in "
        f"{DCGM_RESTART_DROPIN} (StartLimitIntervalSec=300 StartLimitBurst=10 "
        "Restart=on-failure RestartSec=5s) before enabling it"
    ) in err, "the console summary names the drop-in"
    assert any(
        DCGM_RESTART_DROPIN in line for line in group["verification"]["commands"]
    ), "the verification after the roll looks for it"

    code, embedded, err = _run(world, capsys)
    assert "restart_dropin" not in _group(embedded)["dcgm_hostengine"]
    assert "drop-in" not in err, "embedded mode writes no drop-in"


def test_a_host_group_prepared_before_the_dropin_is_rolled_once_then_compliant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World()
    older = _host_prepared_before_the_dropin(world)

    code, plan, _err = _run(world, capsys)

    assert code == 0
    group = _group(plan)
    assert (group["dcgm_hostengine"]["source"], group["dcgm_hostengine"]["pin"]) == (
        "label",
        "4.7.0-2",
    ), "the label keeps host mode and the part's pin is kept"
    assert group["user_data"]["action"] == "replace"
    assert [item["step"] for item in group["steps"]] == [
        "launch-template-version",
        "roll",
    ], "only the managed part changes: no label or repair step"
    current = group["current_nodes"]
    assert current["outcome"] == "failed"
    assert {
        name
        for item in current["nodes"]
        for name, check in item["checks"].items()
        if not check["ok"]
    } == {"nvidia-dcgm-restart", "nvidia-dcgm-restart-dropin"}, (
        "today's nodes run the package unit's Restart=on-abort"
    )

    code, document, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 0, err
    assert world.aws_operations(mutate=True) == [
        "ec2 create-launch-template-version",
        "eks update-nodegroup-version",
    ]
    text = world.template_user_data(4)
    assert DCGM_RESTART_DROPIN in text and older not in text
    verification = document["apply"]["results"][0]["verification"]
    assert verification["outcome"] == "passed"
    world.calls.clear()

    code, again, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 0, err
    assert world.aws_operations(mutate=True) == [], "a rerun changes nothing"
    assert _group(again)["verdict"] == "COMPLIANT"


def test_a_new_node_without_the_dropin_fails_the_host_verification(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(user_data=site_user_data())
    original = world.eks_describe_update

    def describe(options):  # type: ignore[no-untyped-def]
        result = original(options)
        for instance in world.instances.get(GPU_GROUP, []):
            facts = world.node_facts.get(instance)
            if instance.startswith("i-new") and facts is not None:
                facts.update(
                    {"restart-dropin": "absent", "unit-properties": dict(PACKAGE_UNIT)}
                )
        return result

    world.eks_describe_update = describe  # type: ignore[method-assign]

    code, document, err = _run(
        world, capsys, *apply_arguments(tmp_path, "--dcgm-hostengine", "host")
    )

    assert code == 2 and "VERIFICATION_FAILED" in err
    assert (
        "i-new1a (Success: nvidia-dcgm-restart on-abort (expected on-failure), "
        "nvidia-dcgm-restart-dropin absent (expected present))"
    ) in err, "the stop names the restart policy and the missing drop-in per node"


def test_adoption_in_apply_replaces_the_hand_written_part(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(user_data=site_user_data())

    code, _document, err = _run(
        world,
        capsys,
        *apply_arguments(tmp_path, "--adopt-user-data-part", "site-tools.sh"),
    )

    assert code == 0, err
    text = world.template_user_data(4)
    assert "site-tools.sh" not in text and NODECONFIG_PART in text
    journal = json.loads(journal_file(tmp_path).read_text(encoding="utf-8"))
    parts = journal["attempts"][0]["plan"]["user_data"]["parts"]
    assert parts[1]["adoption_drops"] == [
        "datacenter-gpu-manager-4-core",
        "datacenter-gpu-manager-4-cuda13",
    ], "the journal records what the adoption dropped"


# --- site --------------------------------------------------------------------


def _site(state_dir: Path, *, name: str = SITE_ID, managed: bool = False) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    clusters = [
        {
            "clusterId": "gpu-eks-cluster" if managed else "other",
            "eksClusterArn": GPU_ARN
            if managed
            else "arn:aws:eks:us-west-2:1:cluster/x",
            "allowedNamespaces": ["gpu-fault-system", "training", "research"],
        }
    ]
    (state_dir / "site.yaml").write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.aws/v1alpha1",
                "kind": "RegionalSite",
                "metadata": {"name": name},
                "spec": {"namespace": "gpu-fault-system", "clusters": clusters},
            }
        ),
        encoding="utf-8",
    )
    return state_dir


def test_a_state_dir_must_record_the_derived_site_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = _site(tmp_path / "other", name="test-site")
    world = World()

    code, _document, err = _run(world, capsys, "--state-dir", str(other))

    assert code == 2
    assert f"records site test-site, but {CPU_EKS_ARN} derives site id {SITE_ID}" in err
    assert world.calls == [], "nothing of the GPU cluster is read for another site"

    ours = _site(tmp_path / "ours")
    code, document, _err = _run(World(), capsys, "--state-dir", str(ours))
    assert code == 0
    assert document["recorded_site"] == {
        "state_dir": str(ours),
        "site_name": SITE_ID,
        "matches": True,
    }
    assert document["findings"]["namespaces"]["required"] == ["research", "training"], (
        "the site's own allowed namespaces are checked"
    )


def test_apply_with_a_state_dir_holds_the_site_lock(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_dir = _site(tmp_path / "site", managed=True)
    arguments = (
        "--state-dir",
        str(state_dir),
        "--apply",
        "--confirm",
        verb.CONFIRMATION,
        "--reference",
        "CHG-121",
    )
    busy = World(user_data=site_user_data())

    with site_operation_lock(state_dir, wait=False):
        code, _document, err = _run(busy, capsys, *arguments)

    assert code == 2
    assert "another administrator mutation is in progress" in err
    assert "never rolls a node group while gpu-fault-admin config/deploy may run" in err
    assert busy.calls == [], "refused before anything is read"

    world = World(user_data=site_user_data())
    code, document, err = _run(world, capsys, *arguments)

    assert code == 0, err
    assert journal_file(state_dir).is_file(), "the journal lives in the state dir"
    assert document["gpu_cluster"]["site_managed"] is True
    assert (
        "the node installer reconciler installs the new nodes"
        in (document["next_steps"][0])
    )
