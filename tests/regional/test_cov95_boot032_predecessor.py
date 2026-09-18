from __future__ import annotations

import json
from dataclasses import replace

import pytest

from scripts.e2e.regional import boot032_contract as contract
from scripts.e2e.regional import boot032_lifecycle as lifecycle
from scripts.e2e.regional import boot032_native as adapter
from scripts.e2e.regional import regional_case_contract as cases
from tests.regional._cov95_boot032_approval import approve, execute, restart
from tests.regional._cov95_boot032_membership import two_cluster_world
from tests.regional._cov95_boot032_native import NativeHarness


@pytest.mark.parametrize(
    ("receipt_cluster", "receipt_release", "expected"),
    [
        ("selected", "current", True),
        ("other-member", "current", False),
        ("foreign", "current", False),
        ("selected", "foreign", False),
    ],
)
def test_two_cluster_accepted_site_uses_explicit_approved_predecessor_cluster(
    tmp_path, monkeypatch, receipt_cluster, receipt_release, expected
):
    world = two_cluster_world(tmp_path, monkeypatch)
    try:
        binding = adapter.NativeBackend(world.settings).initial()
        selected = world.settings.protected_cluster_id
        other = sorted(binding["protected"]["runtime"])[0]
        assert selected != other, (
            "fixture must expose the lexicographic-primary regression"
        )
        release = binding["protected"]["runtime"][selected]["release_state"][
            "release_id"
        ]
        path = cases.case_evidence_path(tmp_path, "GF-REGIONAL-COLLECT-015")
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "case_id": "GF-REGIONAL-COLLECT-015",
                    "verdict": "PASS",
                    "status": "COMPLETED",
                    "cluster_id": {
                        "selected": selected,
                        "other-member": other,
                        "foreign": "foreign-cluster",
                    }[receipt_cluster],
                    "release_id": release
                    if receipt_release == "current"
                    else "foreign-release",
                }
            )
        )
        value = lifecycle.predecessor(world.settings, binding)
        assert value["valid"] is expected, (
            "receipt must match the selected protected member and its live release"
        )
        assert value["expected_cluster_id"] == selected, (
            "receipt cannot choose the target or fall back to sorting"
        )
        if expected:
            NativeHarness(world, monkeypatch)
            settings, plan, deadline = approve(world)
            assert settings.protected_cluster_id == selected, (
                "configuration must retain the selected protected member"
            )
            assert (
                plan["details"]["binding"]["inputs"]["protected_cluster_id"] == selected
            ), "shared approval must bind the explicit protected predecessor cluster"
            assert execute(world, deadline) == contract.RESTART_EXIT, (
                "the canonical selected-cluster predecessor must admit only the isolated first phase"
            )
            restart(world)
            assert execute(world, deadline) == 0, (
                "a two-GPU accepted site must remain protected through full native retirement"
            )
    finally:
        cases.expanded_order.cache_clear()


@pytest.mark.parametrize("release_id", [None, "", True])
def test_predecessor_cannot_disable_release_validation_with_unknown_identity(
    tmp_path, monkeypatch, release_id
):
    world = two_cluster_world(tmp_path, monkeypatch)
    try:
        binding = adapter.NativeBackend(world.settings).initial()
        binding["protected"]["runtime"][world.settings.protected_cluster_id][
            "release_state"
        ]["release_id"] = release_id
        with pytest.raises(contract.UninstallCaseError, match="release identity"):
            lifecycle.predecessor(world.settings, binding)
        assert not world.settings.native_dir.exists(), (
            "unknown accepted release cannot authorize teardown"
        )
    finally:
        cases.expanded_order.cache_clear()


def test_selected_predecessor_cluster_must_belong_to_protected_membership(
    tmp_path, monkeypatch
):
    world = two_cluster_world(tmp_path, monkeypatch)
    try:
        selected = replace(world.settings, protected_cluster_id="foreign-cluster")
        with pytest.raises(contract.UninstallCaseError, match="member"):
            selected.inputs()
    finally:
        cases.expanded_order.cache_clear()
