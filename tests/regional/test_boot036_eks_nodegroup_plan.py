"""GF-REGIONAL-BOOT-036 runner against the verb's own recorded plain EKS world.

Every run of ``prepare-eks-nodegroup`` here is the real verb, in-process, with
the CLI's parsing and failure reporting (``tests/regional/_eks_nodegroup_site``):
the plan documents, refusal lines and provider calls the runner judges are the
ones the deploy-host CLI prints. Nothing touches a cluster.
"""

from __future__ import annotations

import json
import runpy
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import eks_nodegroup_verdicts as verdicts
from scripts.e2e.regional import run_boot036_eks_nodegroup_plan as boot036
from scripts.e2e.regional.acceptance_scope import (
    EXECUTION_SCOPE_ENV,
    SELECTION_REFERENCE_ENV,
)
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._eks_nodegroup_site import (
    CLUSTER_ID,
    CPU_GROUP,
    GPU_GROUP,
    NOW,
    RELEASE_ID,
    SITE_ID,
    Site,
    approve,
    install,
    node_name,
    runner_arguments,
)


@pytest.fixture(autouse=True)
def selective_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXECUTION_SCOPE_ENV, "selective")
    monkeypatch.setenv(SELECTION_REFERENCE_ENV, "CHG-147")


def prepared_site(tmp_path: Path, **options: Any) -> Site:
    site = Site(tmp_path, **options)
    site.prepare()
    return site


def settings_for(
    tmp_path: Path, site: Site, monkeypatch: pytest.MonkeyPatch, *extra: str
) -> boot036.Settings:
    install(monkeypatch, boot036, site)
    arguments = boot036.parser().parse_args(runner_arguments(tmp_path, site, *extra))
    return boot036.configure(arguments)


def planned(settings: boot036.Settings, run_dir: Path) -> Path:
    case_dir = run_dir / "cases" / boot036.CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = boot036.read_only_preflight(settings, case_dir)
    assert preflight["errors"] == [], preflight["errors"]
    approve(case_dir, boot036.plan_details(settings, preflight))
    return case_dir


def evidence(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "cases" / boot036.CASE_ID / f"{boot036.CASE_ID}.json"
    return json.loads(path.read_text(encoding="utf-8"))


DEADLINE = NOW + timedelta(hours=6)


# --- configure --------------------------------------------------------------------


def test_configure_binds_the_profile_flags_and_the_run_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = Site(tmp_path)
    settings = settings_for(
        tmp_path, site, monkeypatch, "--nodegroup", GPU_GROUP, "--nodegroup", GPU_GROUP
    )

    assert settings.nodegroups == (GPU_GROUP,), "a repeated group is named once"
    assert settings.predecessor_id == "GF-REGIONAL-COLLECT-015"
    assert settings.environment()["GPU_FAULT_STATE_DIR"] == str(site.state_dir)
    assert settings.mode_flags() == [], "the groups' own declaration by default"
    assert len(settings.run_token) == 8


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--dcgm-version", "4.6.1-1"), "only with --dcgm-hostengine host"),
        (("--nodegroup", "bad name"), "invalid --nodegroup"),
    ],
)
def test_configure_refuses_inconsistent_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: tuple[str, ...],
    message: str,
) -> None:
    site = Site(tmp_path)
    with pytest.raises(RegionalFixtureError, match=message):
        settings_for(tmp_path, site, monkeypatch, *extra)


def test_configure_requires_the_state_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = Site(tmp_path)
    arguments = boot036.parser().parse_args(runner_arguments(tmp_path, site))
    arguments.state_dir = ""
    with pytest.raises(RegionalFixtureError, match="--state-dir"):
        boot036.configure(arguments)


def test_mode_flags_pass_the_declared_pin_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = Site(tmp_path)
    settings = settings_for(
        tmp_path,
        site,
        monkeypatch,
        "--dcgm-hostengine",
        "host",
        "--dcgm-version",
        "4.6.1-1",
    )

    assert settings.mode_flags() == [
        "--dcgm-hostengine",
        "host",
        "--dcgm-version",
        "4.6.1-1",
    ]


# --- preflight --------------------------------------------------------------------


def test_preflight_reads_the_labelled_groups_and_every_probe_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    preflight = boot036.read_only_preflight(settings, case_dir)

    assert preflight["errors"] == [], preflight["errors"]
    assert sorted(preflight["nodegroups"]) == [GPU_GROUP], (
        "the default is the groups labelled for the site; the CPU group is not"
    )
    targets = preflight["refusal_targets"]
    assert targets["unsupported_nodegroup"]["name"] == CPU_GROUP
    assert targets["hyperpod_member"]["hyperpod_name"] == "hp-member-fixture"
    assert targets["hyperpod_cluster_arn"].endswith(":cluster/hpmember01"), (
        'expected: targets["hyperpod_cluster_arn"].endswith(":cluster/hpmember01")'
    )
    assert preflight["identity"] == {"release_id": RELEASE_ID, "cluster_id": CLUSTER_ID}
    assert site.tool.runs == [["--help"]], "the preflight never plans"
    assert (case_dir / "preflight.json").is_file(), "the preflight is kept"


def test_preflight_without_hyperpod_member_probes_the_cpu_hyperpod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = Site(tmp_path, members=False)
    site.prepare()
    settings = settings_for(tmp_path, site, monkeypatch)

    preflight = boot036.read_only_preflight(settings, tmp_path)
    targets = preflight["refusal_targets"]

    assert targets["hyperpod_member"] is None
    assert targets["hyperpod_cluster_arn"].endswith(":cluster/fixturecpuhp01"), targets


def test_preflight_refuses_a_hyperpod_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = Site(tmp_path, provider="hyperpod")
    settings = settings_for(tmp_path, site, monkeypatch)

    preflight = boot036.read_only_preflight(settings, tmp_path)

    assert any(
        "prepares eks-managed node groups only" in item for item in preflight["errors"]
    ), preflight["errors"]


def test_preflight_refuses_when_no_group_is_labelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = Site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)

    preflight = boot036.read_only_preflight(settings, tmp_path)

    assert any(
        "carries gpu-fault.io/control-plane" in item for item in preflight["errors"]
    ), preflight["errors"]


def test_preflight_names_a_missing_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch, "--nodegroup", "absent-group")

    preflight = boot036.read_only_preflight(settings, tmp_path)

    assert any(
        "has no node group absent-group" in item for item in preflight["errors"]
    ), preflight["errors"]


def test_preflight_never_inventories_a_held_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    held = node_name(site.world.instances[GPU_GROUP][0])
    site.cordoned.add(held)
    site.incidents.append(
        {
            "incident_id": "incident-1",
            "state": "QUARANTINED",
            "node_ids": [held],
            "created_at": (NOW - timedelta(days=1)).isoformat(),
            "active": True,
            "reason": "XID 79",
        }
    )
    settings = settings_for(tmp_path, site, monkeypatch)

    errors = boot036.read_only_preflight(settings, tmp_path)["errors"]

    assert any(f"{held} is held" in item for item in errors), errors
    assert any("active incident: incident-1" in item for item in errors), errors


def test_preflight_ignores_an_incident_elsewhere_on_the_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    site.incidents.append(
        {
            "incident_id": "incident-2",
            "state": "DETECTED",
            "node_ids": ["ip-other"],
            "created_at": (NOW - timedelta(days=1)).isoformat(),
            "active": True,
            "reason": "",
        }
    )
    site.workflows.append(
        {"request_id": "workflow-2", "status": "RUNNING", "node_ids": ["ip-other"]}
    )
    settings = settings_for(tmp_path, site, monkeypatch)

    assert boot036.read_only_preflight(settings, tmp_path)["errors"] == []


def test_preflight_refuses_a_group_without_instances_or_not_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    site.world.instances[GPU_GROUP] = []
    site.world.nodegroups[GPU_GROUP]["status"] = "DEGRADED"
    settings = settings_for(tmp_path, site, monkeypatch)

    errors = boot036.read_only_preflight(settings, tmp_path)["errors"]

    assert any("is DEGRADED, not ACTIVE" in item for item in errors), errors
    assert any("runs no instance" in item for item in errors), errors


def test_preflight_fails_closed_on_an_unreadable_identity_and_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    monkeypatch.delenv(EXECUTION_SCOPE_ENV)
    monkeypatch.delenv(SELECTION_REFERENCE_ENV)

    errors = boot036.read_only_preflight(settings, tmp_path)["errors"]
    assert errors == ["formal predecessor GF-REGIONAL-COLLECT-015 is not PASS"], errors

    def unreadable() -> dict[str, str]:
        raise RegionalFixtureError("release state is rolled-back")

    monkeypatch.setattr(site.fixture, "evidence_identity", unreadable)
    errors = boot036.read_only_preflight(settings, tmp_path)["errors"]
    assert any("release identity unavailable" in item for item in errors), errors


def test_preflight_refuses_a_site_cli_without_the_verb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    real = site.tool

    def old_cli(arguments: list[str], *, timeout: float) -> Any:
        run = real(arguments, timeout=timeout)
        return type(run)(run.arguments, 2, "", "gpu-fault-admin: invalid choice\n")

    monkeypatch.setattr(boot036, "admin_tool", lambda _fixture, _state_dir: old_cli)

    errors = boot036.read_only_preflight(settings, tmp_path)["errors"]

    assert any("has no prepare-eks-nodegroup verb" in item for item in errors), errors


# --- execute ----------------------------------------------------------------------


def test_a_prepared_site_passes_every_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    case_dir = planned(settings, run_dir)

    assert boot036.execute_case(settings, run_dir, 1, DEADLINE) == 0

    result = evidence(run_dir)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["release_id"] == RELEASE_ID and result["cluster_id"] == CLUSTER_ID
    assert sorted(result["refusals"]) == [
        "adopt-absent-part",
        "cpu-control-plane",
        "hyperpod-cluster",
        "hyperpod-orchestrated-eks",
        "unknown-nodegroup",
        "unsupported-nodegroup",
    ]
    assert result["mutation_performed"] is False
    assert site.world.aws_operations(mutate=True) == [], "no run of the tool wrote"
    assert (case_dir / "tool-plan.json").is_file(), "the tool's plan is kept"
    assert (case_dir / "tool-plan.stderr.log").is_file(), "its console output too"
    assert json.loads((case_dir / "plan.json").read_text())["details"]["tool_runs"], (
        "the approved plan is not overwritten by the tool's"
    )


def test_the_plan_lists_every_tool_run_it_will_make(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    preflight = boot036.read_only_preflight(settings, tmp_path)

    details = boot036.plan_details(settings, preflight)

    names = [run["name"] for run in details["tool_runs"]]
    assert names[0] == "tool-plan" and len(names) == 7, names
    assert details["risk"] == "read-only-signal-replay"
    assert details["preflight_identity"]["nodegroups"][GPU_GROUP]["instances"]


def test_the_plan_without_groups_lists_no_tool_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = Site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)

    details = boot036.plan_details(settings, {"errors": ["x"]})

    assert details["tool_runs"] == []


def test_an_unprepared_group_fails_with_the_steps_it_would_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = Site(tmp_path)
    site.world.nodegroups[GPU_GROUP]["labels"]["gpu-fault.io/control-plane"] = SITE_ID
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)

    assert boot036.execute_case(settings, run_dir, 1, DEADLINE) == 1

    result = evidence(run_dir)
    assert result["verdict"] == "FAIL"
    assert any(
        "the plan would run" in item for item in result["stages"]["main_plan"]
    ), result["stages"]["main_plan"]
    adoption = result["refusals"]["adopt-absent-part"]
    assert adoption["expected"]["document"] == verdicts.REFUSED, (
        "without the managed part an absent adoption reference is refused"
    )
    assert adoption["errors"] == [], adoption
    assert result["stages"]["no_mutation"] == []


def test_a_plan_that_drifted_since_approval_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)
    site.world.instances[GPU_GROUP] = ["i-other1", "i-other2"]

    assert boot036.execute_case(settings, run_dir, 1, DEADLINE) == 1

    assert "plan drifted" in evidence(run_dir)["error"]


def test_a_failed_preflight_at_execute_is_a_fail_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)
    site.cordoned.add(node_name(site.world.instances[GPU_GROUP][0]))

    assert boot036.execute_case(settings, run_dir, 1, DEADLINE) == 1

    assert "preflight failed" in evidence(run_dir)["error"]


def test_an_expired_window_runs_no_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)
    site.tool.runs.clear()

    assert boot036.execute_case(settings, run_dir, 1, NOW) == 1

    assert "window has ended" in evidence(run_dir)["error"]
    assert site.tool.runs == [["--help"]], "nothing but the preflight's --help ran"


def test_an_incident_opened_and_a_release_change_during_the_case_fail_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)
    real = site.tool

    def disturbed(arguments: list[str], *, timeout: float) -> Any:
        run = real(arguments, timeout=timeout)
        if "--adopt-user-data-part" in arguments:
            site.release_id = "rel-148"
            site.incidents.append(
                {
                    "incident_id": "incident-3",
                    "state": "DETECTED",
                    "node_ids": ["ip-other"],
                    "created_at": (NOW + timedelta(seconds=1)).isoformat(),
                    "active": True,
                    "reason": "DCGM field missing",
                }
            )
        return run

    monkeypatch.setattr(boot036, "admin_tool", lambda _fixture, _state_dir: disturbed)

    assert boot036.execute_case(settings, run_dir, 1, DEADLINE) == 1

    stages = evidence(run_dir)["stages"]
    assert any("opened during the case" in item for item in stages["incidents"]), (
        stages["incidents"]
    )
    assert stages["identity"] == [
        "the release changed during the case: "
        + str({"release_id": "rel-148", "cluster_id": CLUSTER_ID})
    ]


def test_an_interrupt_leaves_a_fail_document_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = prepared_site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    run_dir = tmp_path / "run"
    planned(settings, run_dir)

    def interrupted(*_arguments: Any, **_options: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(boot036.case, "snapshot", interrupted)

    with pytest.raises(KeyboardInterrupt):
        boot036.execute_case(settings, run_dir, 1, DEADLINE)

    assert evidence(run_dir)["verdict"] == "FAIL"
    assert evidence(run_dir)["error"] == "KeyboardInterrupt: "


def test_a_site_without_optional_targets_runs_the_two_universal_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = Site(tmp_path)
    settings = settings_for(tmp_path, site, monkeypatch)
    target = boot036.case.site_target(site.state_dir, CLUSTER_ID)

    probes = boot036.refusal_probes(
        settings, target, {"absent_user_data_part": "absent.sh"}, [GPU_GROUP], None
    )

    assert [probe.refusal.name for probe in probes] == [
        "cpu-control-plane",
        "adopt-absent-part",
    ]


def test_the_runner_answers_help_as_a_script(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_SITE_PROFILE", raising=False)
    monkeypatch.setattr(sys, "argv", [boot036.__file__, "--help"])

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(boot036.__file__, run_name="__main__")

    assert stopped.value.code == 0
    assert "--execute" in capsys.readouterr().out
