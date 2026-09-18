from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_identity_acceptance as identity
from scripts.e2e.regional import run_workload_acceptance as workload


@pytest.mark.parametrize("family", ["identity", "workload"])
@pytest.mark.parametrize("valid", [True, False, "false"])
def test_owned_plan_callers_pass_the_exact_arguments_and_a_real_preflight_boolean(
    family: str, valid: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = identity if family == "identity" else workload
    case_id = (
        "GF-REGIONAL-AUTH-013" if family == "identity" else "GF-REGIONAL-WORKLOAD-001"
    )
    arguments = module.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--site",
            str(tmp_path / "synthetic-site.yaml"),
            "--case",
            case_id,
            "--cluster-id",
            "cluster-test",
        ]
    )
    target = SimpleNamespace(cluster_id="cluster-test")
    regional = SimpleNamespace(
        evidence_identity=lambda: {
            "release_id": "release-test",
            "cluster_id": "cluster-test",
        }
    )
    site = SimpleNamespace(
        target=lambda cluster: target,
        regional=lambda target: regional,
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
    )
    for path in (arguments.site, site.cpu_kubeconfig, site.gpu_kubeconfig):
        path.write_text("unit connection fixture\n")
    monkeypatch.setattr(module, "install_site_profile", lambda: None)
    monkeypatch.setattr(
        module, "parser", lambda: SimpleNamespace(parse_args=lambda: arguments)
    )
    monkeypatch.setattr(
        module,
        "IdentitySite" if family == "identity" else "WorkloadSite",
        lambda path: site,
    )
    if family == "workload":
        monkeypatch.setattr(module, "install_abort_signals", lambda: None)
    monkeypatch.setattr(
        module,
        "predecessor_path",
        lambda *args: ("previous", tmp_path / "previous.json"),
    )
    monkeypatch.setattr(
        module, "predecessor_evidence", lambda *args, **kwargs: {"valid": valid}
    )
    monkeypatch.setattr(module, "case_plan", lambda *args, **kwargs: {})
    plans = []

    def build_plan(
        *, arguments: Namespace, preflight_passed: bool, **kwargs: Any
    ) -> dict[str, Any]:
        plans.append((arguments, preflight_passed))
        assert type(preflight_passed) is bool
        return {"schema_version": 3, "preflight_passed": preflight_passed}

    monkeypatch.setattr(module, "build_plan", build_plan)
    assert module.main() == (0 if valid is True else 1)
    assert plans == [(arguments, valid is True)]
