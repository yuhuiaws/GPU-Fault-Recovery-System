"""Plain EKS GPU node groups roll with ``updateStrategy`` MINIMAL.

A DEFAULT roll launches a replacement before it terminates a node and needs
one more GPU instance of the type -- on the live Spot p5en group it failed with
``NodeCreationFailure`` after about 17 minutes. ``prepare-eks-nodegroup``
therefore sets MINIMAL (keeping the group's own ``maxUnavailable``) before any
roll, ``--keep-update-strategy`` keeps DEFAULT with a finding, and the
``regional_contexts`` facts behind ``status --full`` advise on a selected GPU
group that is not MINIMAL without changing any verdict. The world is
``_eks_nodegroup_world``: the live group's ``updateConfig`` is
``{"maxUnavailable": 1}`` (no strategy, i.e. DEFAULT), two nodes, Spot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.command_log import ADMIN_LOG_ENVIRONMENT
from gpu_fault.admin.eks_nodegroup_update_strategy import (
    advisory,
    minimal_update_config,
    nodes_out,
    update_strategy,
    update_strategy_findings,
)
from gpu_fault.admin.eks_nodegroup_user_data import managed_script, merge_managed_part
from gpu_fault.admin.status_report import print_status_report
from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from tests.admin._eks_nodegroup_world import (
    GPU_GROUP,
    LT_ID,
    SITE_ID,
    World,
    apply_arguments,
    encoded,
    journal_file,
    run_verb,
    site_user_data,
)

ADVISORY = (
    f"node group {GPU_GROUP} updates with DEFAULT (surge): a version update needs "
    "one more GPU instance; set updateStrategy=MINIMAL (administrator operations "
    "§5.0.1)"
)


def _run(
    world: World, capsys: pytest.CaptureFixture[str], *arguments: str
) -> tuple[int, dict[str, Any], str]:
    try:
        code, reason = run_verb(world, *arguments), ""
    except BootstrapError as exc:
        code, reason = 2, str(exc)
    captured = capsys.readouterr()
    document = json.loads(captured.out) if captured.out.strip() else {}
    return code, document, captured.err + reason


def _group(document: dict[str, Any]) -> dict[str, Any]:
    (group,) = document["nodegroups"]
    return group


def _prepared_but_default(world: World) -> None:
    """Everything a run sets except the strategy: the live group's DEFAULT."""

    text = merge_managed_part(site_user_data(), managed_script()).text
    world.templates[LT_ID][3]["data"]["UserData"] = encoded(text)
    document = world.nodegroups[GPU_GROUP]
    document["labels"] = {**document["labels"], OWNERSHIP_LABEL: SITE_ID}
    document["nodeRepairConfig"] = {"enabled": False}


def _payload(command: str) -> dict[str, Any]:
    return json.loads(command.split("--update-config ", 1)[1].split("'")[1])


def _call_index(world: World, *words: str, mutate: bool | None = None) -> int:
    return next(
        index
        for index, (argv, flag) in enumerate(world.calls)
        if all(word in argv for word in words) and (mutate is None or flag is mutate)
    )


# --- the strategy ------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "strategy"),
    [
        (None, "DEFAULT"),
        ({"maxUnavailable": 1}, "DEFAULT"),
        ({"maxUnavailable": 1, "updateStrategy": "DEFAULT"}, "DEFAULT"),
        ({"maxUnavailable": 1, "updateStrategy": "MINIMAL"}, "MINIMAL"),
    ],
    ids=["no-update-config", "live-group", "explicit-default", "minimal"],
)
def test_a_group_without_the_field_updates_with_default(
    config: dict[str, Any] | None, strategy: str
) -> None:
    assert update_strategy({"updateConfig": config}) == strategy
    assert (advisory("g", strategy) is None) is (strategy == "MINIMAL"), (
        "only a group that is not MINIMAL gets the advisory"
    )


@pytest.mark.parametrize(
    ("config", "desired", "payload", "out"),
    [
        (
            {"maxUnavailable": 1},
            2,
            {"maxUnavailable": 1, "updateStrategy": "MINIMAL"},
            "1 of 2 GPU nodes out at a time",
        ),
        (
            {"maxUnavailablePercentage": 50},
            4,
            {"maxUnavailablePercentage": 50, "updateStrategy": "MINIMAL"},
            "50% of 4 GPU nodes out at a time",
        ),
        (
            None,
            2,
            {"updateStrategy": "MINIMAL"},
            "the EKS default maxUnavailable of 2 GPU nodes out at a time",
        ),
        (
            {"maxUnavailable": 2, "updateStrategy": "DEFAULT"},
            2,
            {"maxUnavailable": 2, "updateStrategy": "MINIMAL"},
            "2 of 2 GPU nodes (the whole group) out at a time",
        ),
    ],
    ids=["max-unavailable", "percentage", "unset", "whole-group"],
)
def test_the_minimal_payload_keeps_the_groups_own_unavailability(
    capsys: pytest.CaptureFixture[str],
    config: dict[str, Any] | None,
    desired: int,
    payload: dict[str, Any],
    out: str,
) -> None:
    world = World()
    _prepared_but_default(world)
    group = world.nodegroups[GPU_GROUP]
    group["updateConfig"] = config
    group["scalingConfig"] = {"minSize": 1, "maxSize": desired, "desiredSize": desired}

    code, document, err = _run(world, capsys)

    assert code == 0, err
    assert minimal_update_config(group) == payload, "nothing invented or dropped"
    assert nodes_out(group) == out
    (step,) = _group(document)["steps"]
    assert step["step"] == "update-config"
    assert _payload(step["command"]) == payload
    whole = (
        f"finding: {GPU_GROUP}: updateStrategy MINIMAL with maxUnavailable 2: a roll "
        "takes 2 of 2 GPU nodes (the whole group) out at a time; lower "
        "maxUnavailable below the group size"
    )
    assert (whole in err) is ("whole group" in out), "the whole-group roll is named"


# --- plan --------------------------------------------------------------------


def test_the_live_group_plans_minimal_before_the_roll_and_shows_the_capacity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = World(user_data=site_user_data())

    code, document, err = _run(world, capsys)

    assert code == 0, err
    assert not any(flag for _argv, flag in world.calls), "the plan writes nothing"
    group = _group(document)
    names = [item["step"] for item in group["steps"]]
    assert names[-2:] == ["update-config", "roll"], "MINIMAL is set before any roll"
    update = group["steps"][-2]
    assert update["command"].startswith(
        "aws eks update-nodegroup-config --cluster-name gpu-eks-cluster "
        f"--nodegroup-name {GPU_GROUP} --update-config "
    ), update["command"]
    assert _payload(update["command"]) == {
        "maxUnavailable": 1,
        "updateStrategy": "MINIMAL",
    }, "the group's own maxUnavailable is carried over exactly"
    assert (
        update["summary"] == "updateStrategy DEFAULT -> MINIMAL (maxUnavailable 1 kept)"
    )
    assert group["update_strategy"] == {
        "current": "DEFAULT",
        "planned": "MINIMAL",
        "kept": False,
        "unavailable": {"maxUnavailable": 1},
        "capacity_effect": (
            "a roll terminates before it launches -- 1 of 2 GPU nodes out at a "
            "time, no extra GPU instance needed"
        ),
    }
    assert group["steps"][-1]["summary"].endswith(
        "(updateStrategy MINIMAL: a roll terminates before it launches -- 1 of 2 "
        "GPU nodes out at a time, no extra GPU instance needed)"
    ), "the roll states the capacity it costs"
    assert "1 of 2 GPU nodes out at a time" in group["roll"]["caveat"]
    assert "Spot capacity" in group["roll"]["caveat"]
    assert "updateStrategy DEFAULT -> MINIMAL (maxUnavailable 1 kept)" in err


def test_a_minimal_group_is_compliant(capsys: pytest.CaptureFixture[str]) -> None:
    world = World()
    _prepared_but_default(world)
    world.nodegroups[GPU_GROUP]["updateConfig"] = {
        "maxUnavailable": 1,
        "updateStrategy": "MINIMAL",
    }

    code, document, err = _run(world, capsys)

    assert code == 0, err
    group = _group(document)
    assert (group["verdict"], group["steps"]) == ("COMPLIANT", [])
    assert group["update_strategy"]["current"] == "MINIMAL"
    assert "finding" not in group["update_strategy"]


# --- apply -------------------------------------------------------------------


def test_apply_waits_for_minimal_before_it_rolls_and_journals_the_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(user_data=site_user_data())

    code, document, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 0, err
    written = _call_index(world, "--update-config", mutate=True)
    configs = [
        key for key, item in world.updates.items() if item["kind"] == "ConfigUpdate"
    ]
    waited = _call_index(world, "describe-update", configs[-1])
    rolled = _call_index(world, "update-nodegroup-version", mutate=True)
    assert written < waited < rolled, "the strategy update finishes before the roll"
    (roll,) = [
        item for item in world.updates.values() if item["kind"] == "VersionUpdate"
    ]
    assert roll["strategy"] == "MINIMAL", "EKS rolled the group with MINIMAL"
    assert world.nodegroups[GPU_GROUP]["updateConfig"] == {
        "maxUnavailable": 1,
        "updateStrategy": "MINIMAL",
    }, "maxUnavailable survived the update (the fake replaces updateConfig whole)"
    journal = json.loads(journal_file(tmp_path).read_text(encoding="utf-8"))
    (step,) = [
        item
        for item in journal["attempts"][0]["steps"]
        if item["step"] == "update-config"
    ]
    assert "--update-config" in step["command"]
    assert step["update"]["status"] == "Successful"
    assert journal["attempts"][0]["plan"]["update_strategy"]["planned"] == "MINIMAL"
    world.calls.clear()

    code, again, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 0, err
    assert world.aws_operations(mutate=True) == [], "a rerun changes nothing"
    assert _group(again)["verdict"] == "COMPLIANT"


def test_a_strategy_only_change_is_one_config_update_and_no_roll(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World()
    _prepared_but_default(world)

    code, document, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 0, err
    assert world.aws_operations(mutate=True) == ["eks update-nodegroup-config"]
    assert _group(document)["roll"] == {"needed": False}
    steps = [item["step"] for item in document["apply"]["results"][0]["steps"]]
    assert steps == ["update-config", "verification"]
    assert update_strategy(world.nodegroups[GPU_GROUP]) == "MINIMAL"


def test_keep_update_strategy_rolls_with_default_and_records_a_finding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(user_data=site_user_data())

    code, plan, err = _run(world, capsys, "--keep-update-strategy")

    assert code == 0, err
    group = _group(plan)
    assert "update-config" not in [item["step"] for item in group["steps"]]
    finding = (
        "updateStrategy DEFAULT kept (--keep-update-strategy): a roll launches a "
        "replacement before it terminates a node -- it needs one more GPU instance "
        "of the type; MINIMAL needs no spare instance"
    )
    assert group["update_strategy"]["kept"] is True
    assert group["update_strategy"]["finding"] == finding
    assert f"finding: {GPU_GROUP}: {finding}" in err
    assert "NodeCreationFailure" in group["roll"]["caveat"], "the surge caveat stays"
    assert "--keep-update-strategy --apply" in plan["apply_command"]

    code, document, err = _run(
        world, capsys, *apply_arguments(tmp_path, "--keep-update-strategy")
    )

    assert code == 0, err
    assert not [
        argv for argv, flag in world.calls if flag and "--update-config" in argv
    ], "the kept strategy is never written"
    (roll,) = [
        item for item in world.updates.values() if item["kind"] == "VersionUpdate"
    ]
    assert roll["strategy"] == "DEFAULT"
    journal = json.loads(journal_file(tmp_path).read_text(encoding="utf-8"))
    assert journal["attempts"][0]["plan"]["update_strategy"]["finding"] == finding


def test_the_roll_stops_when_the_group_does_not_update_with_minimal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class Ignoring(World):
        """EKS answers the update but the group still reports DEFAULT."""

        def eks_update_nodegroup_config(self, options):  # type: ignore[no-untyped-def]
            result = super().eks_update_nodegroup_config(options)
            if "--update-config" in options:
                self.nodegroups[GPU_GROUP]["updateConfig"] = {"maxUnavailable": 1}
            return result

    world = Ignoring(user_data=site_user_data())

    code, document, err = _run(world, capsys, *apply_arguments(tmp_path))

    assert code == 2
    assert "CHANGED_SINCE_PLAN" in err
    assert (
        f"node group {GPU_GROUP} updates with DEFAULT, the plan rolls it with "
        "updateStrategy MINIMAL"
    ) in err
    assert "eks update-nodegroup-version" not in world.aws_operations(), (
        "no surge roll of a GPU group"
    )


# --- status --full -----------------------------------------------------------


def _full_report(node_groups: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "status",
        "healthy": True,
        "health": {
            "checks": [
                {
                    "name": "regional_contexts",
                    "status": "PASS",
                    "details": {
                        "clusters": ["gpu-a", "gpu-eks-cluster"],
                        "providers": {
                            "gpu-a": {"provider": "hyperpod"},
                            "gpu-eks-cluster": {
                                "provider": "eks-managed",
                                "node_groups": node_groups,
                            },
                        },
                    },
                }
            ]
        },
    }


def test_status_full_advises_on_a_selected_gpu_group_not_on_minimal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(ADMIN_LOG_ENVIRONMENT, raising=False)
    facts = {"status": "ACTIVE", "node_repair": "disabled"}
    report = _full_report(
        {
            GPU_GROUP: {**facts, "update_strategy": "DEFAULT", "gpu": True},
            "gpu-minimal": {**facts, "update_strategy": "MINIMAL", "gpu": True},
            "cpu-node-group": {**facts, "update_strategy": "DEFAULT", "gpu": False},
            "older-engine": {**facts, "gpu": True},
        }
    )
    stdout = json.dumps(report)

    print_status_report(stdout)

    captured = capsys.readouterr()
    assert captured.out == stdout, "the JSON document passes through unchanged"
    findings = [line for line in captured.err.splitlines() if "finding:" in line]
    assert findings == [
        f"gpu-fault-admin status: finding: gpu-eks-cluster: {ADVISORY}"
    ], "only the selected GPU group not on MINIMAL; no strategy recorded, no line"
    assert "healthy: yes" in captured.err, "advice, not a failing check"
    assert update_strategy_findings({"mode": "status", "health": {"checks": []}}) == []
    assert update_strategy_findings(None) == []
