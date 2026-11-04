"""A membership verb's verify judges remote commands per GPU cluster.

``control_api``'s shared half compared the fleet's oldest unclaimed remote
command and the executor's internal errors with their limits, so in the verify
of a ``join-cluster`` the command a spot-reclaimed node's Agent never claimed,
or an internal error of another cluster's executor, failed the joining
cluster's verify and the join was rolled back. Scoped to the verb's target,
the target's own value gates and another cluster's is its finding; an older
control plane without the per-cluster values, values that do not account for
the fleet's, and every unscoped verify (``deploy``, ``config``) still gate on
the fleet. Only modules that existed before the change are imported, so the
file shows the old behaviour on the old tree: the other cluster's value fails.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from gpu_fault_release import regional_verify_scope as SCOPE
from tests.regional.test_release_verify_departure_wait import (
    EKS,
    EKS_NODE,
    HP,
    HP_NODE,
    ControlPlane,
    by_name,
    health,
    node,
    two_cluster_release,
)

LIMIT = 300
BASELINE = 1_762_250_000.0
NODES = {EKS: [node(EKS_NODE)], HP: [node(HP_NODE)]}


class Commands(ControlPlane):
    """The control API probe with these ``remote_commands`` values."""

    def __init__(self, remote: dict[str, Any]) -> None:
        super().__init__({EKS: [EKS_NODE], HP: [HP_NODE]})
        self.remote = remote

    def __call__(self, release: Any, *, arguments: tuple[str, ...], **kw: Any) -> str:
        report = json.loads(super().__call__(release, arguments=arguments, **kw))
        report["remote_commands"].update(self.remote)
        return json.dumps(report)


def unclaimed(**ages: float) -> dict[str, Any]:
    return {
        "oldest_unclaimed_age_seconds": max(ages.values()),
        "oldest_unclaimed_age_seconds_by_cluster": dict(ages),
    }


def errors_since(**seen: float) -> dict[str, Any]:
    """Internal errors after the baseline, ``seen`` seconds past it per cluster."""

    return {
        "executor_internal_error_total": len(seen),
        "executor_internal_error_last_seen_timestamp_seconds": BASELINE
        + max(seen.values()),
        "executor_internal_error_total_by_cluster": {key: 1 for key in seen},
        "executor_internal_error_last_seen_timestamp_seconds_by_cluster": {
            key: BASELINE + value for key, value in seen.items()
        },
    }


#: The ``previous`` an upgrade recorded: the fleet's newest error at its start.
RECORDED = {
    "executor_internal_error_total": 0,
    "executor_internal_error_last_seen_timestamp_seconds": BASELINE,
}


def release_for(
    *targets: str, previous: dict[str, Any] | None = RECORDED, scoped: bool = True
) -> Any:
    release = two_cluster_release()
    release._load_state = lambda: {"previous": previous}
    if scoped:
        release.verify_scope = SCOPE.VerifyScope(SCOPE.JOIN_VERB, targets or (HP,))
    return release


def control_api(
    monkeypatch: pytest.MonkeyPatch, release: Any, remote: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = health(monkeypatch, release, nodes=NODES, plane=Commands(remote))
    return report, by_name(report)["control_api"]


# -- the oldest unclaimed remote command ---------------------------------------------


def test_another_clusters_unclaimed_command_is_a_finding_of_the_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before: ``oldest unclaimed remote command is 1200.0s; limit is 300s``
    FAILED the HyperPod join's verify -- the command was gpu-eks-cluster's."""

    report, record = control_api(
        monkeypatch, release_for(), unclaimed(**{EKS: 1200.0, HP: 4.0})
    )

    assert report["healthy"] is True, record
    assert record["status"] == "WARN", record
    assert record["details"]["not_gating"] == {
        EKS: [f"oldest unclaimed remote command is 1200.0s; limit is {LIMIT}s"]
    }
    assert [
        (item["check"], item["cluster_id"])
        for item in report["cluster_scope"]["not_gating_findings"]
    ] == [("control_api", EKS)]


def test_the_targets_own_unclaimed_command_still_fails_the_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, record = control_api(
        monkeypatch, release_for(), unclaimed(**{EKS: 1200.0, HP: 900.0})
    )

    assert report["healthy"] is False
    assert record["status"] == "FAIL"
    assert record["summary"] == (
        f"{HP}: oldest unclaimed remote command is 900.0s; limit is {LIMIT}s"
    )


@pytest.mark.parametrize(
    "remote",
    [
        {"oldest_unclaimed_age_seconds": 1200.0},
        unclaimed(**{EKS: 120.0}) | {"oldest_unclaimed_age_seconds": 1200.0},
        {"oldest_unclaimed_age_seconds": 1200.0}
        | {"oldest_unclaimed_age_seconds_by_cluster": {EKS: "old"}},
    ],
    ids=["older-control-plane", "breakdown-short-of-the-fleet", "unreadable"],
)
def test_an_unattributable_unclaimed_command_fails_on_the_fleets_value(
    monkeypatch: pytest.MonkeyPatch, remote: dict[str, Any]
) -> None:
    """An older control plane has no per-cluster ages; ages that do not account
    for the fleet's cannot say whose command it is. Both still gate."""

    report, record = control_api(monkeypatch, release_for(), remote)

    assert report["healthy"] is False
    assert record["summary"] == (
        f"oldest unclaimed remote command is 1200.0s; limit is {LIMIT}s"
    )


def test_a_deploy_verify_judges_every_clusters_unclaimed_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, record = control_api(
        monkeypatch, release_for(scoped=False), unclaimed(**{EKS: 1200.0, HP: 4.0})
    )

    assert record["status"] == "FAIL"
    assert record["summary"] == (
        f"oldest unclaimed remote command is 1200.0s; limit is {LIMIT}s"
    )
    assert "cluster_scope" not in report


def test_a_removal_reports_a_remaining_clusters_unclaimed_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = release_for()
    release.verify_scope = SCOPE.VerifyScope(SCOPE.REMOVE_VERB, ("gpu-removed",))

    report, record = control_api(
        monkeypatch, release, unclaimed(**{EKS: 1200.0, HP: 4.0})
    )
    assert report["healthy"] is True, record
    assert list(record["details"]["not_gating"]) == [EKS]

    report, record = control_api(
        monkeypatch, release, unclaimed(**{EKS: 4.0, "gpu-removed": 1200.0})
    )
    assert report["healthy"] is False, "the removed cluster's own command gates"
    assert record["summary"].startswith("gpu-removed: oldest unclaimed"), record


# -- the executor's internal errors --------------------------------------------------


def test_another_clusters_internal_error_is_a_finding_of_the_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before: ``executor internal remote-command error observed during release``
    FAILED the HyperPod join's verify -- gpu-eks-cluster's executor erred."""

    report, record = control_api(monkeypatch, release_for(), errors_since(**{EKS: 60}))

    assert report["healthy"] is True, record
    assert record["status"] == "WARN", record
    assert record["details"]["not_gating"] == {
        EKS: [
            "executor internal remote-command error observed during release: "
            f"{BASELINE:.6f}->{BASELINE + 60:.6f}"
        ]
    }


def test_the_targets_own_internal_error_still_fails_the_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, record = control_api(
        monkeypatch, release_for(), errors_since(**{EKS: 60, HP: 30})
    )

    assert report["healthy"] is False
    assert record["summary"] == (
        f"{HP}: executor internal remote-command error observed during release: "
        f"{BASELINE:.6f}->{BASELINE + 30:.6f}"
    )


def test_an_error_before_the_baseline_is_neither_gating_nor_a_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The baseline is the fleet's newest error at the last release: a cluster
    whose last error is older than it had none since, in any cluster."""

    remote = errors_since(**{EKS: 60})
    remote["executor_internal_error_last_seen_timestamp_seconds_by_cluster"][HP] = (
        BASELINE - 3600
    )
    remote["executor_internal_error_total_by_cluster"][HP] = 4
    remote["executor_internal_error_total"] = 5

    report, record = control_api(monkeypatch, release_for(), remote)

    assert report["healthy"] is True, record
    assert list(record["details"]["not_gating"]) == [EKS], "HP's error predates it"


@pytest.mark.parametrize(
    "breakdown",
    [None, {EKS: "yesterday"}, {EKS: -1.0}],
    ids=["older-control-plane", "unreadable", "negative"],
)
def test_an_unattributable_internal_error_fails_on_the_fleets_value(
    monkeypatch: pytest.MonkeyPatch, breakdown: dict[str, Any] | None
) -> None:
    remote = errors_since(**{EKS: 60})
    remote.pop("executor_internal_error_total_by_cluster")
    key = "executor_internal_error_last_seen_timestamp_seconds_by_cluster"
    if breakdown is None:
        remote.pop(key)
    else:
        remote[key] = breakdown

    report, record = control_api(monkeypatch, release_for(), remote)

    assert report["healthy"] is False
    assert record["summary"] == (
        "executor internal remote-command error observed during release: "
        f"{BASELINE:.6f}->{BASELINE + 60:.6f}"
    )


def test_a_breakdown_that_misses_the_fleets_newest_error_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = errors_since(**{EKS: 60})
    remote["executor_internal_error_last_seen_timestamp_seconds"] = BASELINE + 90

    report, record = control_api(monkeypatch, release_for(), remote)

    assert report["healthy"] is False, "an error no cluster accounts for gates"
    assert record["summary"] == (
        "executor internal remote-command error observed during release: "
        f"{BASELINE:.6f}->{BASELINE + 90:.6f}"
    )


def test_a_deploy_verify_judges_every_clusters_internal_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, record = control_api(
        monkeypatch, release_for(scoped=False), errors_since(**{EKS: 60})
    )

    assert record["status"] == "FAIL"
    assert record["summary"] == (
        "executor internal remote-command error observed during release: "
        f"{BASELINE:.6f}->{BASELINE + 60:.6f}"
    )


def test_without_a_recorded_baseline_every_error_is_attributed_to_its_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sync-state`` (every join/remove), a no-op deploy and a bootstrap leave
    no ``previous``: the baseline is a zero count, and every error is new."""

    remote = {
        "executor_internal_error_total": 3,
        "executor_internal_error_total_by_cluster": {EKS: 3},
    }

    report, record = control_api(monkeypatch, release_for(previous=None), remote)
    assert report["healthy"] is True, record
    assert record["details"]["not_gating"] == {
        EKS: ["executor internal remote-command errors increased during release: 0->3"]
    }

    remote["executor_internal_error_total"] = 4
    remote["executor_internal_error_total_by_cluster"][HP] = 1
    report, record = control_api(monkeypatch, release_for(previous=None), remote)
    assert record["summary"] == (
        f"{HP}: executor internal remote-command errors increased during release: 0->1"
    )


@pytest.mark.parametrize(
    "previous,remote",
    [
        (
            {"executor_internal_error_total": 2},
            {
                "executor_internal_error_total": 3,
                "executor_internal_error_total_by_cluster": {EKS: 3},
            },
        ),
        (
            None,
            {
                "executor_internal_error_total": 3,
                "executor_internal_error_total_by_cluster": {EKS: 2},
            },
        ),
        (None, {"executor_internal_error_total": 3}),
    ],
    ids=[
        "count-baseline-of-an-older-engine",
        "counts-short-of-the-fleet",
        "older-control-plane",
    ],
)
def test_a_count_that_cannot_be_attributed_fails_on_the_fleets_value(
    monkeypatch: pytest.MonkeyPatch,
    previous: dict[str, Any] | None,
    remote: dict[str, Any],
) -> None:
    report, record = control_api(monkeypatch, release_for(previous=previous), remote)

    baseline = (previous or {}).get("executor_internal_error_total", 0)
    assert report["healthy"] is False
    assert record["summary"] == (
        "executor internal remote-command errors increased during release: "
        f"{baseline}->3"
    )


@pytest.mark.parametrize(
    "previous,remote,problem",
    [
        (
            {"executor_internal_error_last_seen_timestamp_seconds": "soon"},
            {"executor_internal_error_last_seen_timestamp_seconds": 1.0},
            "error timestamp is invalid",
        ),
        (
            {"executor_internal_error_last_seen_timestamp_seconds": -1.0},
            {"executor_internal_error_last_seen_timestamp_seconds": 1.0},
            "error timestamp is invalid",
        ),
        (
            {"executor_internal_error_total": "two"},
            {
                "executor_internal_error_total": 3,
                "executor_internal_error_last_seen_timestamp_seconds": None,
            },
            "error baseline is invalid",
        ),
    ],
    ids=["unreadable-timestamp", "negative-timestamp", "unreadable-count"],
)
def test_an_invalid_baseline_fails_every_verify(
    monkeypatch: pytest.MonkeyPatch,
    previous: dict[str, Any],
    remote: dict[str, Any],
    problem: str,
) -> None:
    for scoped in (True, False):
        report, record = control_api(
            monkeypatch, release_for(previous=previous, scoped=scoped), remote
        )
        assert record["status"] == "FAIL", (scoped, record)
        assert problem in record["summary"], record


def test_errors_already_in_the_baseline_pass_without_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = {
        "executor_internal_error_total": 2,
        "executor_internal_error_last_seen_timestamp_seconds": None,
        "executor_internal_error_total_by_cluster": {EKS: 2},
    }

    report, record = control_api(
        monkeypatch, release_for(previous={"executor_internal_error_total": 2}), remote
    )

    assert record["status"] == "PASS", record
    assert report["cluster_scope"]["not_gating_findings"] == [], report
