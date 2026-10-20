"""GF-REGIONAL-BOOT-037 runner against the verb's own recorded plain EKS world.

The roll is the real verb's apply, in-process (``tests/regional/
_eks_nodegroup_site``): the world publishes the launch template version, sets
the strategy, rolls the group onto new instances booted from the merged user
data and answers the tool's SSM verification from them. The runner's waits run
on a fake clock; the store, the Nodes and the Agents follow the world.
"""

from __future__ import annotations

import json
import os
import runpy
import stat
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_boot037_eks_nodegroup_roll as boot037
from scripts.e2e.regional.acceptance_scope import (
    EXECUTION_SCOPE_ENV,
    SELECTION_REFERENCE_ENV,
)
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from scripts.e2e.regional.site_profile import SITE_PROFILE_ENV
from tests.regional._eks_nodegroup_site import (
    CLUSTER_ID,
    GPU_GROUP,
    NOW,
    RELEASE_ID,
    Site,
    approve,
    install,
    node_name,
    runner_arguments,
)

#: The deliberate drift: the same paired DCGM release, pinned by version.
REPIN = (
    "--nodegroup",
    GPU_GROUP,
    "--dcgm-hostengine",
    "host",
    "--dcgm-version",
    "4.6.1",
)
DEADLINE = NOW + timedelta(hours=8)


@pytest.fixture(autouse=True)
def selective_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXECUTION_SCOPE_ENV, "selective")
    monkeypatch.setenv(SELECTION_REFERENCE_ENV, "CHG-147")
    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_WINDOW_END", raising=False)


def prepared_site(tmp_path: Path, **options: Any) -> Site:
    site = Site(tmp_path, **options)
    site.prepare()
    return site


def settings_for(
    tmp_path: Path, site: Site, monkeypatch: pytest.MonkeyPatch, *extra: str
) -> boot037.Settings:
    install(monkeypatch, boot037, site)
    arguments = boot037.parser().parse_args(
        runner_arguments(tmp_path, site, *(extra or REPIN))
    )
    return boot037.configure(arguments)


def planned(settings: boot037.Settings, run_dir: Path) -> Path:
    case_dir = run_dir / "cases" / boot037.CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = boot037.read_only_preflight(settings, case_dir)
    assert preflight["errors"] == [], preflight["errors"]
    approve(case_dir, boot037.plan_details(settings, preflight))
    return case_dir


def evidence(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "cases" / boot037.CASE_ID / f"{boot037.CASE_ID}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def new_nodes(site: Site) -> list[str]:
    """The names the world's next roll gives the replacements."""

    roll = site.world.rolls + 1
    return [node_name(f"i-new{roll}a"), node_name(f"i-new{roll}b")]


def site_profile(tmp_path: Path, site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    old = [node_name(item) for item in site.world.instances[GPU_GROUP]]
    path = tmp_path / "profile.yaml"
    path.write_text(
        "arguments:\n"
        f"  fault-node: {old[0]}\n"
        f"  spare-node: {old[1]}\n"
        f"  node-b: [{old[1]}]\n"
        "  cluster-id: gpu-eks-cluster\n",
        encoding="utf-8",
    )
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    monkeypatch.setenv(SITE_PROFILE_ENV, str(path))


# --- configure --------------------------------------------------------------------


def test_configure_derives_the_reference_and_the_target_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = Site(tmp_path)
    settings = settings_for(
        tmp_path, site, monkeypatch, *REPIN, "--adopt-user-data-part", "1"
    )

    assert settings.reference.startswith("GF-REGIONAL-BOOT-037:a1:"), (
        'expected: settings.reference.startswith("GF-REGIONAL-BOOT-037:a1:")'
    )
    assert settings.predecessor_id == "GF-REGIONAL-BOOT-036"
    assert settings.target_flags()[-2:] == ["--adopt-user-data-part", "1"]
    assert "--adopt-user-data-part" not in settings.target_flags(adopt=False), (
        "the re-plan after the roll no longer adopts: the part is the managed one"
    )
    assert settings.apply_flags()[-7:] == [
        "--roll-timeout-minutes",
        "150",
        "--apply",
        "--confirm",
        "PREPARE_EKS_NODEGROUP",
        "--reference",
        settings.reference,
    ]
    assert settings.required_window_seconds() > 150 * 60
    assert settings.environment()["GPU_FAULT_STATE_DIR"] == str(site.state_dir)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ((), "names exactly one node group"),
        (("--nodegroup", "a", "--nodegroup", "b"), "names exactly one node group"),
        (("--nodegroup", "bad name"), "names exactly one node group"),
        (("--nodegroup", GPU_GROUP, "--dcgm-version", "4.6.1"), "only with"),
        (("--nodegroup", GPU_GROUP, "--roll-timeout-minutes", "0"), "1..360"),
        (("--nodegroup", GPU_GROUP, "--reference", "x"), "not a change reference"),
    ],
)
def test_configure_refuses_an_unbounded_or_ambiguous_roll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: tuple[str, ...],
    message: str,
) -> None:
    site = Site(tmp_path)
    install(monkeypatch, boot037, site)
    arguments = boot037.parser().parse_args(runner_arguments(tmp_path, site, *extra))
    with pytest.raises(RegionalFixtureError, match=message):
        boot037.configure(arguments)


def test_configure_requires_the_state_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = Site(tmp_path)
    arguments = boot037.parser().parse_args(runner_arguments(tmp_path, site, *REPIN))
    arguments.state_dir = ""
    with pytest.raises(RegionalFixtureError, match="--state-dir"):
        boot037.configure(arguments)


# --- preflight --------------------------------------------------------------------


def test_preflight_approves_a_roll_of_the_repinned_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)

    preflight = boot037.read_only_preflight(settings, tmp_path)

    assert preflight["errors"] == [], preflight["errors"]
    assert [step["step"] for step in preflight["tool_plan"]["steps"]] == [
        "launch-template-version",
        "roll",
    ], "the re-pin changes the managed part only"
    assert preflight["tool_plan"]["dcgm"]["exporter_pairing"] == "paired"
    assert len(site.tool.runs) == 2, "--help and one read-only plan"
    assert site.world.aws_operations(mutate=True) == []
    assert (tmp_path / "tool-plan-before.json").is_file(), (
        'expected: (tmp_path / "tool-plan-before.json").is_file()'
    )


def test_preflight_refuses_a_group_that_already_runs_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch, "--nodegroup", GPU_GROUP)

    errors = boot037.read_only_preflight(settings, tmp_path)["errors"]

    assert any("already runs the target configuration" in item for item in errors), (
        errors
    )


def test_preflight_refuses_whatever_holds_the_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    first, second = (node_name(item) for item in site.world.instances[GPU_GROUP])
    site.workloads[first] = [{"namespace": "training", "name": "job-0"}]
    site.attempts[second] = ["attempt-7"]
    site.taints[second] = [{"key": "gpu-fault.io/quiesce", "effect": "NoExecute"}]
    site.pending_agents.add(first)
    site.workflows.append(
        {"request_id": "workflow-9", "status": "PENDING", "node_ids": ["ip-x"]}
    )
    settings = settings_for(tmp_path, site, monkeypatch)

    errors = boot037.read_only_preflight(settings, tmp_path)["errors"]

    assert any("runs business workloads" in item for item in errors), errors
    assert any("bound to attempts ['attempt-7']" in item for item in errors), errors
    assert any(f"{second} is held" in item for item in errors), errors
    assert any(f"{first}: Agent absent" in item for item in errors), errors
    assert any("executable workflow workflow-9" in item for item in errors), errors


def test_preflight_refuses_a_window_too_short_for_the_roll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    end = (NOW + timedelta(hours=1)).isoformat()
    settings = settings_for(
        tmp_path, site, monkeypatch, *REPIN, "--maintenance-window-end", end
    )

    errors = boot037.read_only_preflight(settings, tmp_path)["errors"]

    assert any("the maintenance window leaves 3600 s" in item for item in errors), (
        errors
    )


def test_a_window_long_enough_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(
        tmp_path,
        site,
        monkeypatch,
        *REPIN,
        "--maintenance-window-end",
        DEADLINE.isoformat().replace("+00:00", "Z"),
    )

    assert boot037.window_errors(settings) == []


def test_preflight_names_an_earlier_update_in_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    journal = next((site.state_dir / "eks-nodegroup-prepare").rglob("*.json"))
    document = json.loads(journal.read_text())
    document["pending_update"] = {"update_id": "update-77", "step": "roll"}
    journal.write_text(json.dumps(document))
    site.world.nodegroups[GPU_GROUP]["status"] = "UPDATING"
    settings = settings_for(tmp_path, site, monkeypatch)

    errors = boot037.read_only_preflight(settings, tmp_path)["errors"]

    assert any("update update-77 of an earlier apply" in item for item in errors), (
        errors
    )


def test_preflight_names_a_refused_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    site.world.nodegroups[GPU_GROUP]["amiType"] = "BOTTLEROCKET_x86_64_NVIDIA"
    settings = settings_for(tmp_path, site, monkeypatch)

    errors = boot037.read_only_preflight(settings, tmp_path)["errors"]

    assert any(f"{GPU_GROUP} is refused" in item for item in errors), errors


def test_preflight_fails_closed_on_an_unreadable_identity_and_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)

    def unreadable() -> dict[str, str]:
        raise RegionalFixtureError("release state is rolled-back")

    monkeypatch.setattr(site.fixture, "evidence_identity", unreadable)
    (site.state_dir / "site.yaml").write_text("kind: Other\n")

    errors = boot037.read_only_preflight(settings, tmp_path)["errors"]

    assert any("release identity unavailable" in item for item in errors), errors
    assert any("is not a RegionalSite" in item for item in errors), errors


def test_preflight_fails_a_missing_formal_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    monkeypatch.delenv(EXECUTION_SCOPE_ENV)
    monkeypatch.delenv(SELECTION_REFERENCE_ENV)

    errors = boot037.read_only_preflight(settings, tmp_path)["errors"]

    assert errors == ["formal predecessor GF-REGIONAL-BOOT-036 is not PASS"], errors


def test_the_plan_shows_the_apply_and_the_nodes_it_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    preflight = boot037.read_only_preflight(settings, tmp_path)

    details = boot037.plan_details(settings, preflight)

    assert details["risk"] == "destructive"
    assert "--apply --confirm PREPARE_EKS_NODEGROUP" in details["apply_command"]
    assert [node["name"] for node in details["nodes_to_replace"]] == sorted(
        node_name(item) for item in site.world.instances[GPU_GROUP]
    )
    assert details["preflight_identity"]["steps"] == ["launch-template-version", "roll"]


# --- execute ----------------------------------------------------------------------


def test_a_drifted_group_rolls_and_every_replacement_is_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    site_profile(tmp_path, site, monkeypatch)
    departed = sorted(node_name(item) for item in site.world.instances[GPU_GROUP])
    replacements = new_nodes(site)
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    case_dir = planned(settings, run_dir)

    assert boot037.execute_case(settings, run_dir, 1, DEADLINE) == 0

    result = evidence(run_dir)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["release_id"] == RELEASE_ID and result["cluster_id"] == CLUSTER_ID
    assert result["mutation_performed"] is True
    assert [item["name"] for item in result["replacement_nodes"]] == replacements
    assert [
        (item["departed"]["name"], item["replacement"]["name"])
        for item in result["node_renames"]
    ] == list(zip(departed, replacements, strict=True))
    assert result["profile_updates"] == [
        {
            "argument": "fault-node",
            "departed": departed[0],
            "replacement": replacements[0],
        },
        {"argument": "node-b", "departed": departed[1], "replacement": replacements[1]},
        {
            "argument": "spare-node",
            "departed": departed[1],
            "replacement": replacements[1],
        },
    ]
    assert result["exits"] == []
    assert site.world.aws_operations(mutate=True) == [
        "ec2 create-launch-template-version",
        "eks update-nodegroup-version",
    ], "the re-pin publishes one version and rolls; nothing else is written"
    assert (case_dir / "tool-apply.json").is_file() and (
        case_dir / "tool-plan-after.json"
    ).is_file(), "both tool documents are kept"
    assert result["agents"]["agents_active_after_seconds"] == 0.0


def test_a_roll_short_of_capacity_fails_with_the_rerun_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    site.world.roll_results = [(1, "Failed")]
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)

    assert boot037.execute_case(settings, run_dir, 1, DEADLINE) == 1

    result = evidence(run_dir)
    assert any("ROLL_FAILED" in item for item in result["stages"]["apply"]), result[
        "stages"
    ]["apply"]
    assert "replacement" not in result["stages"], "nothing waits after a failed roll"
    assert result["state_after_roll"]["nodegroup"]["status"] == "ACTIVE"
    assert len(result["exits"]) == 1 and "--apply" in result["exits"][0]


def test_a_departed_node_that_lingers_fails_the_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    site.linger_reads = 10_000
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)

    assert boot037.execute_case(settings, run_dir, 1, DEADLINE) == 1

    result = evidence(run_dir)
    assert any(
        "still Nodes of the group" in item for item in result["stages"]["replacement"]
    ), result["stages"]["replacement"]
    assert result["replacement_wait_seconds"] >= boot037.REPLACEMENT_WAIT_SECONDS
    assert "replan" not in result["stages"]


def test_a_departed_node_that_lingers_briefly_is_waited_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    site.linger_reads = 2
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)

    assert boot037.execute_case(settings, run_dir, 1, DEADLINE) == 0

    assert evidence(run_dir)["replacement_wait_seconds"] == 2 * boot037.POLL_SECONDS


def test_a_departed_agent_never_retired_names_the_reconcile_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    site.lingering_agents.update(
        node_name(item) for item in site.world.instances[GPU_GROUP]
    )
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)

    assert boot037.execute_case(settings, run_dir, 1, DEADLINE) == 1

    result = evidence(run_dir)
    assert any(
        "departed node's Agent is still ACTIVE" in item
        for item in result["stages"]["agents"]
    ), result["stages"]["agents"]
    assert any("--retire-departed-agents" in item for item in result["exits"]), result[
        "exits"
    ]
    assert (
        result["agents"]["departed_retired_after_seconds"]
        >= boot037.RETIRE_WAIT_SECONDS
    )


def test_a_replacement_the_installer_never_reaches_fails_the_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    site.pending_agents.update(new_nodes(site))
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)

    assert boot037.execute_case(settings, run_dir, 1, DEADLINE) == 1

    result = evidence(run_dir)
    assert any(
        "Agent absent, not ACTIVE" in item for item in result["stages"]["agents"]
    ), result["stages"]["agents"]
    assert "departed_retired_after_seconds" not in result["agents"]


def test_an_unstable_replacement_and_an_incident_fail_the_roll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    tainted = new_nodes(site)[0]
    site.taints[tainted] = [{"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}]
    site.incidents.append(
        {
            "incident_id": "incident-5",
            "state": "RECOVERED",
            "node_ids": [tainted],
            "created_at": (NOW + timedelta(minutes=5)).isoformat(),
            "active": False,
            "reason": "node NotReady",
        }
    )
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)

    assert boot037.execute_case(settings, run_dir, 1, DEADLINE) == 1

    stages = evidence(run_dir)["stages"]
    assert any(f"{tainted}: tainted" in item for item in stages["stability"]), stages
    assert any(
        "opened during the case: incident-5" in item for item in stages["incidents"]
    ), stages["incidents"]


def test_a_plan_that_drifted_since_approval_rolls_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)
    site.world.instances[GPU_GROUP] = list(
        reversed(site.world.instances[GPU_GROUP])
    ) + ["i-extra"]
    site.world.nodegroups[GPU_GROUP]["scalingConfig"]["desiredSize"] = 3

    assert boot037.execute_case(settings, run_dir, 1, DEADLINE) == 1

    result = evidence(run_dir)
    assert "plan drifted" in result["error"]
    assert result["mutation_performed"] is False
    assert site.world.aws_operations(mutate=True) == []


def test_a_preflight_that_fails_at_execute_rolls_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)
    site.attempts[node_name(site.world.instances[GPU_GROUP][0])] = ["attempt-1"]

    assert boot037.execute_case(settings, run_dir, 1, DEADLINE) == 1

    assert "preflight failed" in evidence(run_dir)["error"]
    assert site.world.aws_operations(mutate=True) == []


def test_a_window_too_short_at_execute_rolls_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)

    assert boot037.execute_case(settings, run_dir, 1, NOW + timedelta(hours=1)) == 1

    assert "the maintenance window leaves 3600 s" in evidence(run_dir)["error"]
    assert site.world.aws_operations(mutate=True) == []


def test_a_release_change_during_the_roll_fails_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)
    real = site.tool

    def deployed(arguments: list[str], *, timeout: float) -> Any:
        run = real(arguments, timeout=timeout)
        if "--apply" in arguments:
            site.release_id = "rel-148"
        return run

    monkeypatch.setattr(boot037, "admin_tool", lambda _fixture, _state_dir: deployed)

    assert boot037.execute_case(settings, run_dir, 1, DEADLINE) == 1

    assert evidence(run_dir)["stages"]["identity"][0].startswith(
        "the release changed during the case"
    ), (
        'expected: evidence(run_dir)["stages"]["identity"][0].startswith( "the release ch'
    )


def test_an_interrupt_leaves_a_fail_document_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)

    def interrupted(*_arguments: Any, **_options: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(boot037, "wait_replacement", interrupted)

    with pytest.raises(KeyboardInterrupt):
        boot037.execute_case(settings, run_dir, 1, DEADLINE)

    result = evidence(run_dir)
    assert result["verdict"] == "FAIL" and result["mutation_performed"] is True


def test_node_errors_name_a_short_group_and_foreign_instances() -> None:
    nodes = [
        {
            "name": "n1",
            "instance_id": "i-1",
            "ready": "True",
            "unschedulable": False,
            "blocking_taints": [],
        }
    ]

    errors = boot037.node_errors(
        GPU_GROUP, nodes, ["i-2"], desired=2, inventoried=["i-2"]
    )

    assert errors == [
        f"{GPU_GROUP} has 1 Nodes, desired 2",
        f"{GPU_GROUP}: Nodes ['i-1'], running instances ['i-2']",
    ]


def test_the_runner_answers_help_as_a_script(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(SITE_PROFILE_ENV, raising=False)
    monkeypatch.setattr(sys, "argv", [boot037.__file__, "--help"])

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(boot037.__file__, run_name="__main__")

    assert stopped.value.code == 0
    assert "--roll-timeout-minutes" in capsys.readouterr().out
