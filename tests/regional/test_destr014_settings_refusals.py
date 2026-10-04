"""GF-REGIONAL-DESTR-014 runner: the settings and preflight refusals that
happen before any fixture is built -- a managed-recovery timeout the control
plane would refuse at boot, and a site file or training manifest that does not
exist."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.e2e.regional import control_plane_env_window as control_window
from scripts.e2e.regional import run_destr014_branch_exhaustion as case
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr_warm import regional_settings


def _arguments(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "run_dir": tmp_path,
        "attempt": 1,
        "fault_node": "node-a",
        "sibling_node": "node-b",
        "job_id": "",
        "attempt_id": "",
        "managed_recovery_timeout_seconds": 1,
        "predecessor_evidence": "",
        "site_file": str(tmp_path / "site.yaml"),
        "manifest": str(tmp_path / "training.yaml"),
        "hyperpod_cluster": "fake-hyperpod",
        "host_probe_image": "example.test/probe",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_configure_refuses_a_managed_recovery_timeout_the_worker_would_reject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        case,
        "settings_from_arguments",
        lambda arguments: pytest.fail("no kubeconfig is read before the timing gate"),
    )
    expected = control_window.assignment_errors(
        {control_window.MANAGED_RECOVERY_VARIABLE: "1"}
    )
    assert expected, "the fixture value must be one the control plane refuses"
    with pytest.raises(RegionalFixtureError) as refused:
        case.configure(_arguments(tmp_path))
    message = str(refused.value)
    assert message.startswith("managed recovery timeout is not a value the control"), (
        message
    )
    assert all(problem in message for problem in expected), message


def _settings(tmp_path: Path, *, site: bool, manifest: bool) -> case.Settings:
    site_file = tmp_path / "site.yaml"
    manifest_file = tmp_path / "training.yaml"
    if site:
        site_file.write_text("clusters: []\n", encoding="utf-8")
    if manifest:
        manifest_file.write_text("kind: PyTorchJob\n", encoding="utf-8")
    return case.Settings(
        regional=regional_settings(tmp_path),
        site_file=site_file,
        manifest=manifest_file,
        hyperpod_cluster="fake-hyperpod",
        host_probe_image="example.test/probe",
        fault_node="node-a",
        fault_pci_bdf="0000:01:00.0",
        fault_device="/dev/nvidia0",
        sibling_node="node-b",
        sibling_pci_bdf="0000:02:00.0",
        job_id="job-1",
        attempt_id="job-1-a001",
        verify_max_attempts=6,
        managed_recovery_timeout_seconds=900,
        variant="single",
        predecessor_path=tmp_path / "predecessor.json",
    )


@pytest.mark.parametrize(
    ("site", "manifest"), [(False, True), (True, False), (False, False)]
)
def test_preflight_refuses_a_missing_site_file_or_manifest_before_any_fixture(
    site: bool, manifest: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        case,
        "RegionalLiveFixture",
        lambda _settings: pytest.fail("no fixture is built without the inputs"),
    )
    with pytest.raises(
        RegionalFixtureError, match="site file or training manifest does not exist"
    ):
        case.read_only_preflight(
            _settings(tmp_path, site=site, manifest=manifest), tmp_path / "case"
        )
    assert not (tmp_path / "case" / "preflight.json").exists(), (
        "no preflight record is written for missing inputs"
    )
