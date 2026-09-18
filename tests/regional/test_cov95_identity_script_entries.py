from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import audit_auth_boundary as audit
from scripts.e2e.regional import identity_acceptance_common as common
from scripts.e2e.regional import live_driver_guard, regional_live_fixture
from scripts.e2e.regional import run_blast_acceptance as blast
from scripts.e2e.regional import run_e2e002_multicluster_fault as e2e
from scripts.e2e.regional import run_identity_acceptance as identity
from scripts.e2e.regional import run_iso006_cluster_offline as offline
from scripts.e2e.regional import run_iso007_failed_recovery_isolation as asymmetric
from scripts.e2e.regional import run_workload_acceptance as workload
from scripts.e2e.regional.probes import auth013_certificate_probe as certificate
from scripts.e2e.regional.probes import auth015_node_probe as custody
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.mark.parametrize("module", [e2e, offline, asymmetric, workload])
def test_script_entry_delegates_to_the_supervised_main_wrapper_without_actions(
    monkeypatch: pytest.MonkeyPatch, module: Any
) -> None:
    calls = []
    monkeypatch.setattr(
        sys, "path", [path for path in sys.path if path != str(module.ROOT)]
    )
    monkeypatch.setattr(
        regional_live_fixture,
        "run_case_main",
        lambda main: calls.append(main.__name__) or 73,
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(module.__file__, run_name="__main__")
    assert stopped.value.code == 73
    assert calls == ["main"]


@pytest.mark.parametrize("module", [certificate, custody])
def test_host_probe_script_rejects_invalid_arguments_before_any_host_read(
    monkeypatch: pytest.MonkeyPatch, module: Any
) -> None:
    args = (
        ["--timer", "../unsafe"]
        if module is certificate
        else ["--master-sha256", "invalid"]
    )
    monkeypatch.setattr(sys, "argv", ["unit-probe", *args])
    monkeypatch.setattr(module.os, "umask", lambda mode: None)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(module.__file__, run_name="__main__")
    assert stopped.value.code == 1


def test_identity_script_refuses_missing_custody_inputs_before_plan_or_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "identity",
            "--run-dir",
            str(tmp_path),
            "--site",
            str(tmp_path / "unit-site"),
            "--case",
            "GF-REGIONAL-AUTH-015",
            "--cluster-id",
            "a",
        ],
    )
    monkeypatch.setattr(
        sys, "path", [path for path in sys.path if path != str(identity.ROOT)]
    )
    monkeypatch.setattr(live_driver_guard, "install_site_profile", lambda: None)
    monkeypatch.setattr(identity.os, "umask", lambda mode: None)
    monkeypatch.setattr(
        common,
        "IdentitySite",
        lambda path: SimpleNamespace(
            target=lambda name: SimpleNamespace(cluster_id=name)
        ),
    )
    calls = []
    monkeypatch.setattr(
        live_driver_guard, "build_plan", lambda **kwargs: calls.append("plan")
    )
    monkeypatch.setattr(
        live_driver_guard,
        "authorize_execution",
        lambda *args, **kwargs: calls.append("authorize"),
    )
    with pytest.raises(common.IdentityAcceptanceError, match="two distinct"):
        runpy.run_path(identity.__file__, run_name="__main__")
    assert calls == []


def test_blast_script_rejects_an_absent_site_before_constructing_the_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "blast",
            "--case",
            "GF-REGIONAL-BLAST-002",
            "--run-dir",
            str(tmp_path),
            "--site",
            str(tmp_path / "absent"),
        ],
    )
    monkeypatch.setattr(
        sys, "path", [path for path in sys.path if path != str(blast.ROOT)]
    )
    with pytest.raises(SystemExit, match="site file does not exist"):
        runpy.run_path(blast.__file__, run_name="__main__")


def test_auth_audit_script_requires_all_inputs_before_reading_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["audit", "matrix"])
    monkeypatch.setattr(
        sys, "path", [path for path in sys.path if path != str(audit.ROOT)]
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(audit.__file__, run_name="__main__")
    assert stopped.value.code == 2
