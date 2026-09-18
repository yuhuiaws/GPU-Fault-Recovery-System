"""ISO-003/004 on a single-cluster site: cluster B may be an unregistered id.

Both cases judge a cluster-id binding -- the executor's Fleet proxy compares the
requested cluster with its own, the control plane binds the token's cluster to
every payload ``cluster_id`` -- so neither needs a second data plane. The AUTH
matrix already runs with an unregistered cluster B. What changes is only the
runner: ``site.target`` used to demand that B be listed in the site.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import identity_acceptance_common as common
from scripts.e2e.regional import identity_acceptance_iso as iso
from scripts.e2e.regional import run_identity_acceptance as entry


def _primary() -> common.ClusterTarget:
    return common.ClusterTarget(
        cluster_id="cluster-a",
        context="context-a",
        region="us-west-2",
        hyperpod_cluster_name="hp-a",
        eks_cluster_arn="arn:aws:eks:us-west-2:123456789012:cluster/a",
        executor_role_arn="arn:aws:iam::123456789012:role/executor-a",
        control_plane_url="https://control.example",
        ca_file=Path("/unused/ca.crt"),
    )


class _Site:
    """A one-cluster site: ``target`` knows cluster-a only."""

    def __init__(self) -> None:
        self.targets = {"cluster-a": _primary()}

    def target(self, cluster_id: str) -> common.ClusterTarget:
        if not cluster_id:
            return self.targets["cluster-a"]
        try:
            return self.targets[cluster_id]
        except KeyError as exc:
            raise common.IdentityAcceptanceError(
                f"cluster is not present in the site: {cluster_id}"
            ) from exc


def _arguments(case: str, secondary: str) -> argparse.Namespace:
    return argparse.Namespace(
        case=case,
        cluster_id="",
        secondary_cluster_id=secondary,
        node=[],
        fleet_master_file=None,
        host_probe_image="",
    )


@pytest.mark.parametrize("case", ["GF-REGIONAL-ISO-003", "GF-REGIONAL-ISO-004"])
def test_iso_cases_accept_an_unregistered_secondary_cluster_id(case: str) -> None:
    primary, secondary, _nodes = entry.validate_case_arguments(
        _arguments(case, "hp-cluster-iso-probe-b"), _Site()
    )
    assert secondary is not None, "the secondary target is synthesized, not dropped"
    assert secondary.cluster_id == "hp-cluster-iso-probe-b", (
        "the probe id passes through unchanged"
    )
    assert secondary.registered is False, (
        "the synthesized target says it is unregistered"
    )
    assert (secondary.context, secondary.control_plane_url, secondary.ca_file) == (
        primary.context,
        primary.control_plane_url,
        primary.ca_file,
    ), "the probes run through the primary's connection"
    assert primary.registered is True, "the primary stays a registered site cluster"


def test_registered_secondary_is_still_looked_up_in_the_site() -> None:
    site = _Site()
    site.targets["cluster-b"] = common.ClusterTarget(
        **{**_primary().__dict__, "cluster_id": "cluster-b", "context": "context-b"}
    )
    _primary_target, secondary, _nodes = entry.validate_case_arguments(
        _arguments("GF-REGIONAL-ISO-003", "cluster-b"), site
    )
    assert secondary is not None and secondary.registered is True, (
        "a site cluster is used as itself, never replaced by a synthesized target"
    )
    assert secondary.context == "context-b", (
        "the registered target keeps its own context"
    )


@pytest.mark.parametrize("case", ["GF-REGIONAL-AUTH-007", "GF-REGIONAL-AUTH-008"])
def test_auth_cases_still_require_a_registered_secondary(case: str) -> None:
    # AUTH-007 disables B in the registry and rolls the control plane: it needs
    # a real second registration, so the strict lookup stays.
    with pytest.raises(common.IdentityAcceptanceError, match="not present in the site"):
        entry.validate_case_arguments(
            _arguments(case, "hp-cluster-iso-probe-b"), _Site()
        )


def test_iso_cases_still_refuse_the_primary_as_secondary() -> None:
    with pytest.raises(common.IdentityAcceptanceError, match="must differ"):
        entry.validate_case_arguments(
            _arguments("GF-REGIONAL-ISO-003", "cluster-a"), _Site()
        )


def test_fleet_case_records_an_unregistered_secondary_as_a_degenerate_state_check():
    snapshots = iter([{"agents": []}, {"agents": []}])
    result = {
        "list_agents": {
            "rejected": True,
            "error": "cannot read agents for another cluster",
        },
        "revoke_agent": {
            "rejected": True,
            "error": "cannot transition an agent in another cluster",
        },
        "request_count": 0,
    }
    site = SimpleNamespace(
        regional=lambda target: SimpleNamespace(
            cpu_python=lambda *args: next(snapshots)
        ),
        any_executor_pod=lambda target: "executor",
        pod_json=lambda *args: result,
    )
    secondary = common.unregistered_secondary(_primary(), "hp-cluster-iso-probe-b")
    outcome = iso.run_iso003(site, _primary(), secondary)
    assert outcome["verdict"] == "PASS", "the denials alone carry the verdict"
    assert outcome["secondary_registered"] is False, (
        "the evidence says B was not a registered cluster"
    )
    assert any("not registered" in item for item in outcome["limitations"]), (
        "the empty-list agent comparison is declared as a limitation"
    )


def test_spare_health_case_records_secondary_registration() -> None:
    results = {
        "cluster-a": {
            "status": 200,
            "body": {"ready": False, "reasons": [iso.MISSING_AGENT_REASON]},
        },
        "hp-cluster-iso-probe-b": {
            "status": 403,
            "body": {
                "detail": "authenticated cluster does not match all payload cluster_id values"
            },
        },
    }
    site = SimpleNamespace(
        any_executor_pod=lambda target: "executor",
        pod_json=lambda *args: {"results": results},
    )
    secondary = common.unregistered_secondary(_primary(), "hp-cluster-iso-probe-b")
    outcome = iso.run_iso004(site, _primary(), secondary)
    assert outcome["verdict"] == "PASS", "the 403 is judged on the binding alone"
    assert outcome["secondary_registered"] is False, (
        "the evidence says B was not a registered cluster"
    )
    assert any("not registered" in item for item in outcome["limitations"]), (
        "the unregistered B is declared as a limitation"
    )
