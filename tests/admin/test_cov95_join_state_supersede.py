"""A join attempt refused before it mutated anything must not bind the site.

Live sequence: ``join-cluster`` was refused on an orphaned token (journal
``BLOCKED_IDENTITY``, steps ``DISCOVERED``/``PRECHECKED`` only, nothing
changed); the product fix for that refusal was then deployed, which moved the
site's ``spec.repositoryRoot``; the retry -- the one the fix exists for -- was
refused with "join-cluster source site non-membership fields drifted" because
the never-mutated journal still pinned the old site. Such an attempt is now
archived and superseded by a fresh one bound to the current site; an attempt
that reached a mutating step keeps refusing on drift exactly as before.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict

import pytest
import yaml

from gpu_fault.admin import cluster_join as join
from gpu_fault.admin import cluster_join_inputs as inputs
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join_evidence import site_non_membership_sha256
from gpu_fault.admin.cluster_join_state import (
    DISCOVERED_STEP,
    PRE_MUTATION_STEPS,
    PRECHECKED_STEP,
    attempt_never_mutated,
    load_join_state,
    supersede_unmutated_attempt,
)
from tests.admin._cov95_join_support import JoinScenario, target

CLUSTER_ID = "hp-gpu-b"
DRIFTED = "0" * 64


def _drift_site(path) -> None:
    """What a deploy does to the site: a non-membership field moves."""

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["spec"]["release"]["agentConfigDigest"] = "f" * 64
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _journal(scenario, **changes) -> tuple:
    """A current journal with the given fields and a stale site binding."""

    request = scenario.request()
    directory, path, state = load_join_state(request)
    state.update(
        {
            "phase": "BLOCKED_IDENTITY",
            "completed_steps": [DISCOVERED_STEP, PRECHECKED_STEP],
            "evidence": {
                DISCOVERED_STEP: {"cluster_id": CLUSTER_ID, "target": asdict(target())}
            },
            "source_site_non_membership_sha256": DRIFTED,
            **changes,
        }
    )
    path.write_text(json.dumps(state), encoding="utf-8")
    return request, directory, path


def test_pre_mutation_steps_are_derived_from_the_recorder_constants() -> None:
    assert PRE_MUTATION_STEPS == frozenset({PRECHECKED_STEP, DISCOVERED_STEP})


@pytest.mark.parametrize(
    ("phase", "steps", "evidence_keys", "expected"),
    [
        ("BLOCKED_IDENTITY", ["DISCOVERED", "PRECHECKED"], ["DISCOVERED"], True),
        ("FAILED", ["PRECHECKED"], [], True),
        ("STARTED", [], [], True),
        ("FAILED", ["DISCOVERED", "LOCAL_INPUTS_STARTED", "PRECHECKED"], [], False),
        (
            "BLOCKED_IDENTITY",
            ["DISCOVERED"],
            ["DISCOVERED", "LOCAL_INPUTS_STARTED"],
            False,
        ),
        ("CANDIDATE_READY", ["DISCOVERED", "PRECHECKED"], [], False),
        ("SUPERVISION_LOST", ["DISCOVERED", "PRECHECKED"], [], False),
        ("ROLLED_BACK", ["PRECHECKED"], [], False),
        ("ROLLBACK_FAILED", [], [], False),
        ("COMPLETED", ["DISCOVERED"], [], False),
    ],
)
def test_attempt_never_mutated_truth_table(phase, steps, evidence_keys, expected):
    state = {
        "phase": phase,
        "completed_steps": steps,
        "evidence": {key: {} for key in evidence_keys},
    }
    assert attempt_never_mutated(state) is expected


def test_a_refused_unmutated_attempt_is_superseded_when_the_site_drifted(
    tmp_path, monkeypatch
):
    scenario = JoinScenario(tmp_path, monkeypatch)
    request, directory, path = _journal(scenario, attempt=3)
    current = site_non_membership_sha256(request.site.source)

    reloaded_dir, reloaded_path, state = load_join_state(request)

    assert (reloaded_dir, reloaded_path) == (directory, path)
    assert state["attempt"] == 4
    assert state["phase"] == "STARTED"
    assert state["completed_steps"] == []
    assert state["evidence"] == {}
    assert state["source_site_non_membership_sha256"] == current, (
        "the fresh attempt binds to the site as it stands now"
    )
    assert state["source_site_sha256"] == request.site.source_sha256
    archive = directory / "state.attempt-003.json"
    archived = json.loads(archive.read_text(encoding="utf-8"))
    assert archived["phase"] == "BLOCKED_IDENTITY", "the archive keeps its own phase"
    assert archived["attempt"] == 3
    assert archived["completed_steps"] == [DISCOVERED_STEP, PRECHECKED_STEP]
    assert archived["source_site_non_membership_sha256"] == DRIFTED
    assert archived["superseded_by_attempt"] == 4
    assert "T" in archived["superseded_at"]
    assert "refused before it mutated anything" in archived["superseded_reason"]
    assert state["superseded_attempts"] == [
        {
            "attempt": 3,
            "phase": "BLOCKED_IDENTITY",
            "completed_steps": [DISCOVERED_STEP, PRECHECKED_STEP],
            "archive": str(archive),
            "reason": archived["superseded_reason"],
            "superseded_at": archived["superseded_at"],
        }
    ]
    assert json.loads(path.read_text(encoding="utf-8")) == state, (
        "the fresh attempt is persisted before the join reads it"
    )


def test_supersession_carries_token_bindings_forward(tmp_path, monkeypatch):
    scenario = JoinScenario(tmp_path, monkeypatch)
    binding = {
        "path": "/site/secure/hp-gpu-b.token",
        "cluster_id": CLUSTER_ID,
        "eks_arn": target().eks_arn,
        "hyperpod_arn": target().hyperpod_arn,
    }
    proof = {"token_sha256": "a" * 64, "cluster_id": CLUSTER_ID}
    request, _directory, _path = _journal(
        scenario,
        retained_cluster_token=binding,
        reclaimed_cluster_token=proof,
        superseded_attempts=[{"attempt": 0, "phase": "FAILED"}],
    )

    _dir, _path, state = load_join_state(request)

    assert state["attempt"] == 2
    assert state["retained_cluster_token"] == binding, (
        "the token binding was proven against the cluster, not the site fields"
    )
    assert state["reclaimed_cluster_token"] == proof
    assert [item["attempt"] for item in state["superseded_attempts"]] == [0, 1]


def test_a_second_supersession_archives_under_the_next_attempt_number(
    tmp_path, monkeypatch
):
    scenario = JoinScenario(tmp_path, monkeypatch)
    request, directory, path = _journal(scenario)
    _dir, _path, state = load_join_state(request)
    assert state["attempt"] == 2
    state.update(
        {
            "phase": "BLOCKED_IDENTITY",
            "completed_steps": [PRECHECKED_STEP],
            "source_site_non_membership_sha256": DRIFTED,
        }
    )
    path.write_text(json.dumps(state), encoding="utf-8")

    _dir, _path, again = load_join_state(request)

    assert again["attempt"] == 3
    assert [item["attempt"] for item in again["superseded_attempts"]] == [1, 2]
    assert (directory / "state.attempt-001.json").is_file(), "first archive kept"
    assert (directory / "state.attempt-002.json").is_file(), "second archive written"


@pytest.mark.parametrize(
    ("phase", "steps"),
    [
        ("FAILED", ["DISCOVERED", "LOCAL_INPUTS_STARTED", "PRECHECKED"]),
        ("CANDIDATE_READY", ["DISCOVERED", "PRECHECKED"]),
        ("SUPERVISION_LOST", ["DISCOVERED", "PRECHECKED"]),
        ("BLOCKED_IDENTITY", ["DISCOVERED", "LOCAL_INPUTS_READY", "PRECHECKED"]),
    ],
)
def test_drift_still_refuses_an_attempt_that_mutated_or_is_unproven(
    tmp_path, monkeypatch, phase, steps
):
    scenario = JoinScenario(tmp_path, monkeypatch)
    request, directory, path = _journal(scenario, phase=phase, completed_steps=steps)
    before = path.read_text(encoding="utf-8")

    with pytest.raises(BootstrapError, match="non-membership fields drifted"):
        load_join_state(request)

    assert path.read_text(encoding="utf-8") == before, "a refusal archives nothing"
    assert list(directory.glob("state.attempt-*.json")) == []


def test_drift_on_a_recoverable_attempt_keeps_the_recovery_scope_path(
    tmp_path, monkeypatch
):
    scenario = JoinScenario(tmp_path, monkeypatch)
    request, directory, _path = _journal(
        scenario, phase="ROLLED_BACK", completed_steps=[PRECHECKED_STEP]
    )

    _dir, _path, state = load_join_state(request)

    assert state["attempt"] == 1, "a rolled-back attempt is resumed, not superseded"
    assert state["phase"] == "ROLLED_BACK"
    assert list(directory.glob("state.attempt-*.json")) == []


def test_an_unchanged_site_leaves_a_refused_attempt_in_place(tmp_path, monkeypatch):
    scenario = JoinScenario(tmp_path, monkeypatch)
    request = scenario.request()
    directory, path, state = load_join_state(request)
    state.update({"phase": "BLOCKED_IDENTITY", "completed_steps": [PRECHECKED_STEP]})
    path.write_text(json.dumps(state), encoding="utf-8")

    _dir, _path, reloaded = load_join_state(request)

    assert reloaded["attempt"] == 1
    assert reloaded["phase"] == "BLOCKED_IDENTITY"
    assert list(directory.glob("state.attempt-*.json")) == []


def test_supersede_refuses_an_attempt_that_mutated(tmp_path, monkeypatch):
    scenario = JoinScenario(tmp_path, monkeypatch)
    request = scenario.request()
    directory, path, state = load_join_state(request)
    state["completed_steps"] = ["LOCAL_INPUTS_STARTED"]

    with pytest.raises(BootstrapError, match="cannot be superseded"):
        supersede_unmutated_attempt(
            request, state_dir=directory, state_path=path, state=state
        )


# --- the live sequence, end to end ----------------------------------------------


def _secret(data: dict[str, str]) -> dict:
    return {
        "kind": "Secret",
        "data": {
            key: base64.b64encode(value.encode()).decode()
            for key, value in data.items()
        },
    }


def test_refused_join_then_deploy_then_reclaim_succeeds(tmp_path, monkeypatch):
    """Join, lose the membership, get refused on the orphaned token, deploy the
    fix (the site's release fields move), retry with the reclaim flag."""

    scenario = JoinScenario(tmp_path, monkeypatch, suffixes=("b",))
    assert scenario.join()["phase"] == "COMPLETED"
    token_file = scenario.path.parent / "secure" / f"{CLUSTER_ID}.token"
    document = yaml.safe_load(scenario.path.read_text())
    document["spec"]["clusters"] = [
        item for item in document["spec"]["clusters"] if item["clusterId"] != CLUSTER_ID
    ]
    scenario.path.write_text(yaml.safe_dump(document))
    with pytest.raises(BootstrapError, match="no matching join ownership"):
        scenario.join()
    refused = scenario.state()
    assert refused["phase"] == "BLOCKED_IDENTITY"
    assert refused["attempt"] == 2
    assert set(refused["completed_steps"]) <= PRE_MUTATION_STEPS
    _drift_site(scenario.path)
    scenario.connection_secrets[target().context] = _secret(
        {"cluster-token": token_file.read_text().strip(), "cluster-id": CLUSTER_ID}
    )
    monkeypatch.setattr(
        inputs,
        "reclaim_released_alias",
        lambda *_args, **_kwargs: {"released_by": {"registry": "absent"}},
    )

    result = join.join_cluster(
        scenario.request(reclaim_cluster_token=True), runner=scenario
    )

    assert result["phase"] == "COMPLETED"
    restored = yaml.safe_load(scenario.path.read_text())
    assert [item["clusterId"] for item in restored["spec"]["clusters"]] == [
        "gpu-a",
        CLUSTER_ID,
    ]
    state = scenario.state()
    assert state["attempt"] == 3
    assert [item["attempt"] for item in state["superseded_attempts"]] == [2]
    assert state["superseded_attempts"][0]["phase"] == "BLOCKED_IDENTITY"
    archive = json.loads(
        (scenario.state_path().parent / "state.attempt-002.json").read_text()
    )
    assert archive["phase"] == "BLOCKED_IDENTITY"
    assert archive["superseded_by_attempt"] == 3
    assert state["reclaimed_cluster_token"]["identity"]["attempt"] == 1, (
        "proof 1 came from the committed attempt, not the archived refusal"
    )
