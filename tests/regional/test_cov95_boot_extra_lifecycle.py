from __future__ import annotations

import copy
import json
import subprocess

import pytest

from scripts.e2e.regional import boot_acceptance_common as common
from scripts.e2e.regional import boot_acceptance_lifecycle as lifecycle
from tests.regional._cov95_boot_extra_safety import (
    boot_extra_isolation as boot_extra_isolation,
)
from tests.regional._cov95_boot_site import BootRegion, BootSite, report
from tests.regional.test_cov95_boot_lifecycle import arguments


def test_admin_and_uninstall_wrappers_preserve_scope_timeout_and_environment(
    tmp_path, monkeypatch
):
    calls = []

    def transport(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 7, "unit stdout", "unit stderr")

    monkeypatch.setattr(lifecycle, "run", transport)
    result = lifecycle.admin_command(
        "status",
        "--state-dir",
        str(tmp_path),
        timeout=17,
        environment_overrides={"UNIT_OVERRIDE": "yes"},
    )
    assert result.returncode == 7
    argv, options = calls[0]
    assert argv[1:] == [
        "-m",
        "gpu_fault.admin.cli",
        "status",
        "--state-dir",
        str(tmp_path),
    ]
    assert options["check"] is False
    assert options["timeout"] == 17
    assert options["env"]["UNIT_OVERRIDE"] == "yes"
    assert options["env"]["PYTHONPATH"] == str(lifecycle.ROOT / "src")
    assert lifecycle.uninstall_site(tmp_path) is not None
    argv, options = calls[1]
    assert argv[-7:] == [
        "uninstall",
        "--state-dir",
        str(tmp_path),
        "--cpu-cluster",
        "keep",
        "--confirm",
        lifecycle.UNINSTALL_CONFIRMATION,
    ]
    assert options["timeout"] == 21600
    assert "UNIT_OVERRIDE" not in options["env"]


def test_site_identity_and_bootstrap_cannot_accept_an_empty_gpu_target_set(
    tmp_path, monkeypatch
):
    model = BootSite(tmp_path, monkeypatch)
    model.config["clusters"] = []
    with pytest.raises(lifecycle.BootAcceptanceError, match="identity is unavailable"):
        lifecycle.site_identity(tmp_path / "site.yaml")
    assert model.reads == []
    result = lifecycle.run_boot016(arguments(tmp_path), tmp_path / "case")
    assert result["verdict"] == "FAIL"
    assert "site contains no GPU clusters" in result["error"]
    assert result["cleanup"]["uninstall_ran"] is True
    assert [call[0][0] for call in model.admin_calls] == ["deploy", "uninstall"]


@pytest.mark.parametrize("stamp", ["2026-09-12T00:00:00", "not-a-timestamp", None])
def test_bootstrap_registry_timing_requires_parseable_timezone_bound_evidence(
    tmp_path, monkeypatch, stamp
):
    model = BootSite(tmp_path, monkeypatch)

    class Region(BootRegion):
        def kubectl(self, plane, *args, **kwargs):
            value = super().kubectl(plane, *args, **kwargs)
            if args[:2] == ("get", "secret"):
                document = json.loads(value)
                document["metadata"]["creationTimestamp"] = stamp
                return json.dumps(document)
            return value

    monkeypatch.setattr(
        common, "RegionalLiveFixture", lambda settings: Region(model, settings)
    )
    result = lifecycle.run_boot016(arguments(tmp_path), tmp_path / "case")
    assert result["verdict"] == "FAIL"
    assert result["checks"]["registry_created_before_workloads"] is False
    assert result["checks"]["failure_cleanup"] is True
    assert model.admin_calls[-1][0][0] == "uninstall"


def test_registry_cannot_prove_ordering_without_any_gpu_deployment_population():
    assert (
        lifecycle.registry_precedes_business_deployments(
            {"metadata": {"creationTimestamp": "2026-09-12T00:00:00Z"}}, [], {}
        )
        is False
    )


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
@pytest.mark.parametrize("generation", [None, 0, True])
def test_bootstrap_cannot_prove_noop_from_unknown_deployment_generations(
    tmp_path, monkeypatch, plane, generation
):
    model = BootSite(tmp_path, monkeypatch)
    model.deployments[plane][0]["metadata"]["generation"] = generation
    result = lifecycle.run_boot016(arguments(tmp_path), tmp_path / "case")
    assert result["verdict"] == "FAIL", (
        f"BOOT-016 reported PASS using unverified {plane} generation {generation!r}"
    )
    assert result["cleanup"]["retained"] is False, (
        "a bootstrap without verifiable generations was retained as a passed fixture"
    )
    assert result["cleanup"]["uninstall_ran"] is True


def test_abort_after_site_creation_still_attempts_isolated_site_cleanup(
    tmp_path, monkeypatch
):
    model = BootSite(tmp_path, monkeypatch)

    def admin(*args, **kwargs):
        completed = model.admin(*args, **kwargs)
        if args[0] == "deploy":
            raise KeyboardInterrupt("unit operator abort")
        return completed

    monkeypatch.setattr(lifecycle, "admin_command", admin)
    with pytest.raises(KeyboardInterrupt, match="unit operator abort"):
        lifecycle.run_boot016(arguments(tmp_path), tmp_path / "case")
    assert [call[0][0] for call in model.admin_calls] == ["deploy", "uninstall"]
    assert (tmp_path / "case/failed-deploy-cleanup.log").is_file(), (
        "interruption skipped the cleanup record"
    )


def test_cleanup_transport_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    model = BootSite(tmp_path, monkeypatch)
    model.failure = "deploy"

    def uninstall(state_dir):
        assert state_dir == tmp_path / "state"
        raise lifecycle.BootAcceptanceError("unit cleanup not confirmed")

    monkeypatch.setattr(lifecycle, "uninstall_site", uninstall)
    with pytest.raises(lifecycle.BootAcceptanceError, match="cleanup not confirmed"):
        lifecycle.run_boot016(arguments(tmp_path), tmp_path / "case")
    assert not (tmp_path / "case/failed-deploy-cleanup.log").exists(), (
        "a failed cleanup transport fabricated a completed uninstall log"
    )


def test_live_identity_detects_a_cpu_digest_mismatch_in_the_actual_replica_report(
    tmp_path, monkeypatch
):
    model = BootSite(tmp_path, monkeypatch)
    observed = copy.deepcopy(report())
    observed["checks"][0]["details"]["control_plane"]["deployments"]["api"][
        "cpu-pod"
    ] = "f" * 64
    result = lifecycle.runtime_identity_matches_release(
        observed, manifest=model.manifest, metadata=model.metadata, agents=model.agents
    )
    assert result["passed"] is False
    assert result["reasons"] == ["cpu/api/cpu-pod digest differs"]
    assert result["replica_count"] == 2


@pytest.mark.parametrize("output", ["[]", "null", '"not a report"'])
def test_status_report_requires_an_object_without_guessing_identity(output):
    with pytest.raises(lifecycle.BootAcceptanceError, match="not a JSON object"):
        lifecycle.parse_status_report(output)
