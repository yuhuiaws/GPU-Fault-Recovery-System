"""GF-REGIONAL-BOOT-038 read-only preflight and ``--execute`` against a fake
site: every precondition the brief names refuses the case with its own error,
an old product is NOT_APPLICABLE, and the evidence is written on every exit
path -- an operator interrupt included."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.e2e.regional import boot038_membership as membership
from scripts.e2e.regional import run_boot038_cross_cluster_membership as runner
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_boot038_harness import (
    OTHER_ID,
    OTHER_NODE,
    PEER_NODE,
    TARGET_ID,
    World,
    threads_end_inside,
)


@pytest.fixture(autouse=True)
def threads_end_with_the_test() -> Iterator[None]:
    with threads_end_inside():
        yield


def world(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: str = membership.WINDOW_KUBELET,
) -> World:
    return World(tmp_path, monkeypatch, window=window)


# --------------------------------------------------------------------------- #
# Read-only preflight
# --------------------------------------------------------------------------- #
def test_a_healthy_site_preflights_clean_on_the_deployed_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    python = site.state_dir / "deployer-venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")

    preflight = site.plan()

    assert preflight["errors"] == [], preflight["errors"]
    assert preflight["product_generation"] == "b250", preflight
    assert "not_applicable" not in preflight, preflight
    assert preflight["predecessor"]["verdict"] == "NOT_REQUIRED", preflight
    assert preflight["other_gpu_nodes"] == [
        {"name": OTHER_NODE, "ready": "True"},
        {"name": PEER_NODE, "ready": "True"},
    ], preflight
    (probe,) = site.details("generation")
    assert probe["interpreter"] == str(python), "the site's own interpreter is asked"
    assert probe["env"] == {"PYTHONDONTWRITEBYTECODE": "1"}, probe
    assert site.details("store.precondition") == [[OTHER_ID, OTHER_NODE, TARGET_ID]]
    assert site.evidence("preflight.json")["errors"] == [], "the preflight is kept"


@pytest.mark.parametrize(
    "failure",
    [OSError("interpreter vanished"), subprocess.TimeoutExpired("python", 120)],
    ids=["os-error", "timeout"],
)
def test_an_unknown_generation_is_pre_b250_and_not_an_error(
    failure: BaseException, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.failures["generation"] = failure

    preflight = site.plan()

    assert preflight["errors"] == [], "NOT_APPLICABLE is a verdict, not a refusal"
    assert (preflight["product_generation"], preflight["not_applicable"]) == (
        "pre-b250",
        True,
    ), preflight


def test_an_unavailable_identity_and_an_old_product_are_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.failures["identity"] = RegionalFixtureError("release state unreadable")
    site.generation = '{"b250": false, "error": "ModuleNotFoundError"}'

    preflight = runner.read_only_preflight(site.settings, site.case_dir)

    assert preflight["errors"] == [
        "release identity unavailable: RegionalFixtureError: release state unreadable"
    ], preflight["errors"]
    assert preflight["identity"] == {} and preflight["not_applicable"] is True


def test_a_predecessor_that_did_not_pass_refuses_the_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.settings = replace(
        site.settings,
        predecessor_id="GF-REGIONAL-BOOT-029",
        predecessor_path=tmp_path / "absent-boot029.json",
    )

    preflight = runner.read_only_preflight(site.settings, site.case_dir)

    assert preflight["errors"] == [
        "formal predecessor GF-REGIONAL-BOOT-029 is not PASS"
    ], preflight["errors"]
    assert preflight["predecessor"]["valid"] is False, preflight["predecessor"]


def test_the_store_refuses_open_incidents_workflows_and_inactive_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.store["open_incidents"] = [
        {"cluster": OTHER_ID, "id": "incident-1", "state": "QUARANTINED"},
        {"cluster": TARGET_ID, "id": "incident-2", "state": "ESCALATED"},
    ]
    site.store["node_workflows"] = [{"request_id": "w-1", "node": OTHER_NODE}]
    site.store["target_agents"] = [
        {"node_id": "hp-node-1", "lifecycle_state": "DRAINING"}
    ]
    site.store["other_agents"] = [{"node_id": OTHER_NODE, "lifecycle_state": "REVOKED"}]

    preflight = runner.read_only_preflight(site.settings, site.case_dir)

    assert preflight["errors"] == [
        f"non-terminal incident(s) open: {OTHER_ID}/incident-1, {TARGET_ID}/incident-2",
        f"an active workflow names {OTHER_NODE}",
        "not every target Agent is ACTIVE",
        f"{OTHER_NODE} has no ACTIVE Agent",
    ], preflight["errors"]


def test_plan_b_needs_no_agent_or_ready_held_node_but_a_ready_peer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch, window=membership.WINDOW_NONE)
    site.store["other_agents"] = []
    site.gpu_nodes = [{"name": OTHER_NODE, "ready": "Unknown"}]

    preflight = runner.read_only_preflight(site.settings, site.case_dir)

    assert len(preflight["errors"]) == 1, preflight["errors"]
    assert preflight["errors"][0].startswith(
        f"{OTHER_ID} keeps no other Ready GPU node"
    ), preflight["errors"]


@pytest.mark.parametrize(
    ("nodes", "error"),
    [
        (
            [{"name": PEER_NODE, "ready": "True"}],
            f"{OTHER_NODE} is not a GPU node of {OTHER_ID}",
        ),
        (
            [
                {"name": OTHER_NODE, "ready": "False"},
                {"name": PEER_NODE, "ready": "True"},
            ],
            f"{OTHER_NODE} is not Ready before the window",
        ),
    ],
    ids=["absent", "not-ready"],
)
def test_the_held_node_must_be_a_ready_gpu_node_of_the_other_cluster(
    nodes: list[dict[str, str]],
    error: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = world(tmp_path, monkeypatch)
    site.gpu_nodes = nodes

    preflight = runner.read_only_preflight(site.settings, site.case_dir)

    assert preflight["errors"] == [error], preflight["errors"]


def test_unreadable_store_and_nodes_refuse_and_a_stale_journal_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.failures["store.precondition"] = RegionalFixtureError("API Pod gone")
    site.failures["gpu_nodes"] = RegionalFixtureError("kubectl timed out")
    journal = site.case_dir / "service-window" / "destr008-service-window-old.json"
    journal.parent.mkdir()
    journal.write_text(
        json.dumps(
            {
                "phase": "STOPPED",
                "scope": {
                    "node": OTHER_NODE,
                    "run_id": "boot038-old",
                    "release_id": "r",
                },
            }
        ),
        encoding="utf-8",
    )

    preflight = runner.read_only_preflight(site.settings, site.case_dir)

    errors = preflight["errors"]
    assert errors[:2] == [
        "store preflight failed: RegionalFixtureError: API Pod gone",
        "node preflight failed: RegionalFixtureError: kubectl timed out",
    ], errors
    assert errors[2].startswith(
        "a stale service-window journal is open: destr008-service-window-old.json"
    ), errors
    assert "boot038-old" in errors[2], errors


# --------------------------------------------------------------------------- #
# --execute
# --------------------------------------------------------------------------- #
def test_an_execute_whose_preflight_fails_mutates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.plan()
    site.store["node_workflows"] = [{"request_id": "w-2", "node": OTHER_NODE}]

    code, result = site.execute()

    assert code == 1 and result["verdict"] == membership.VERDICT_FAIL, result
    assert result["error"] == (
        f"RegionalFixtureError: preflight failed: an active workflow names {OTHER_NODE}"
    ), result
    assert result["mutation_performed"] is False, result
    assert not any(name.startswith("verb.") for name in site.names()), site.names()
    assert result["executed_at"] and result["release_id"] == "release-test", result


def test_an_execute_after_a_release_change_is_refused_as_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.plan()
    site.identity = {"release_id": "release-next", "cluster_id": OTHER_ID}

    code, result = site.execute()

    assert code == 1, result
    assert result["error"].startswith(
        "RegionalFixtureError: GF-REGIONAL-BOOT-038 plan drifted"
    ), result
    assert "window.create" not in site.names(), "a drifted plan opens no window"


def test_an_old_product_executes_as_not_applicable_without_a_verb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.generation = '{"b250": false}'
    site.plan()

    code, result = site.execute()

    assert code == 0, result
    assert result["verdict"] == membership.VERDICT_NOT_APPLICABLE, result
    assert result["mutation_performed"] is False, result
    assert "predates b250" in result["reason"], result
    assert not any(name.startswith(("verb.", "window.")) for name in site.names()), (
        "an old product is never asked to remove, join or stop anything"
    )


def test_remove_then_rejoin_under_a_kubelet_window_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.plan()

    code, result = site.execute()

    assert code == 0 and result["verdict"] == membership.VERDICT_PASS, result
    assert result["mutation_performed"] is True, result
    verbs = {item["verb"]: item for item in result["verbs"]}
    for verb in membership.VERBS:
        entry = verbs[verb]
        assert entry["verdict"] == membership.VERDICT_PASS, entry
        assert entry["finding_surfaces"] == dict.fromkeys(
            membership.FINDING_SURFACES, True
        ), entry
        assert set(entry["window_timeline"]) == {
            "stop",
            "verify_begin",
            "not_ready_seen",
            "restore",
        }, entry["window_timeline"]
    assert result["after_run"]["ok"] is True, result["after_run"]
    names = site.names()
    assert (
        names.index("verb.spawn:remove-cluster")
        < names.index("ownership.reapply")
        < names.index("verb.spawn:join-cluster")
    ), "the label is re-applied for the join"
    assert names.count("window.create") == 2 and names.count("window.close") == 2
    assert result["window_residue"]["open_journals"] == [], result["window_residue"]


def test_an_operator_interrupt_is_recorded_then_re_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = world(tmp_path, monkeypatch)
    site.plan()
    site.failures["verb.spawn:remove-cluster"] = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        site.execute()

    result = site.evidence()
    assert result["verdict"] == membership.VERDICT_FAIL, result
    assert result["error"] == "KeyboardInterrupt: ", result
    assert "executed_at" in result, "the evidence is final before the re-raise"
    names = site.names()
    assert "window.create" not in names, "the verb never reached its open trigger"
    assert names[-2:] == ["window.cleanup", "window.close"], (
        "the window prepared for the verb is still ended on the way out"
    )
