from __future__ import annotations

import pytest

from scripts.e2e.regional import boot_acceptance_lifecycle as lifecycle
from tests.regional._cov95_boot_site import BootSite
from tests.regional.test_cov95_boot_lifecycle import arguments


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"items": None},
        {"items": []},
        {"items": [None]},
        {"items": [{"metadata": None}]},
        {"items": [{"metadata": {"generation": 1}}]},
        {"items": [{"metadata": {"name": "", "generation": 1}}]},
        {"items": [{"metadata": {"name": 1, "generation": 1}}]},
        {"items": [{"metadata": {"name": "executor", "generation": -1}}]},
        {"items": [{"metadata": {"name": "executor", "generation": "1"}}]},
        {"items": [{"metadata": {"name": "executor", "generation": 1.0}}]},
        {"items": [{"metadata": {"name": "executor", "generation": 1}}] * 2},
    ],
)
def test_generation_snapshot_refuses_incomplete_or_ambiguous_inventory(value) -> None:
    with pytest.raises(lifecycle.BootAcceptanceError, match="generation"):
        lifecycle.deployment_generation_snapshot(value)


def test_generation_snapshot_preserves_all_valid_observations() -> None:
    value = {
        "items": [
            {"metadata": {"name": "worker", "generation": 7}},
            {"metadata": {"name": "api", "generation": 1}},
        ]
    }
    assert lifecycle.deployment_generation_snapshot(value) == {"worker": 7, "api": 1}, (
        "NOOP evidence must retain the observed generation of each Deployment"
    )


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_invalid_second_snapshot_cannot_retain_a_successful_bootstrap(
    plane, tmp_path, monkeypatch
) -> None:
    model = BootSite(tmp_path, monkeypatch)
    admin = model.admin

    def deploy_then_lose_generation(*args, **kwargs):
        result = admin(*args, **kwargs)
        if args[0] == "deploy" and model.deploys == 2:
            model.deployments[plane][0]["metadata"].pop("generation")
        return result

    monkeypatch.setattr(lifecycle, "admin_command", deploy_then_lose_generation)
    result = lifecycle.run_boot016(arguments(tmp_path), tmp_path / "case")
    assert model.deploys == 2, "the first snapshot must have admitted the NOOP rerun"
    assert result["verdict"] == "FAIL", (
        "the second unknown observation cannot prove NOOP"
    )
    assert result["cleanup"]["retained"] is False, (
        "invalid NOOP proof must not retain the site"
    )
    assert result["cleanup"]["uninstall_ran"] is True, (
        "failed proof still requires cleanup"
    )
