from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin.site import RenderedSite
from scripts.e2e.regional import run_capacity_acceptance as capacity


@pytest.mark.parametrize("valid", [True, False, "false", None])
def test_plan_binds_arguments_and_cannot_override_failed_preflight(
    valid: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = Namespace(
        case="GF-REGIONAL-CAP-001",
        site=tmp_path / "synthetic-site",
        run_dir=tmp_path,
        predecessor_evidence="",
        b_latency_factor=2.0,
        execute=False,
        attempt=1,
    )
    kubeconfig = tmp_path / "cpu-kubeconfig"
    kubeconfig.write_text("synthetic CPU kubeconfig\n")
    site = RenderedSite(
        source=arguments.site,
        repository_root=tmp_path,
        release_config={
            "cpu_kubeconfig": str(kubeconfig),
            "site_name": "unit-site",
            "aws_region": "us-east-1",
            "cpu_eks_arn": "unit-cpu-eks",
            "namespace": "unit-namespace",
        },
        environment={},
        source_sha256="a" * 64,
    )
    monkeypatch.setattr(capacity, "load_site", lambda _path: site)
    monkeypatch.setattr(
        capacity, "live_release_state", lambda _site: {"release_id": "unit-release"}
    )
    monkeypatch.setattr(capacity, "parse_args", lambda: arguments)
    monkeypatch.setattr(capacity, "install_site_profile", lambda: None)
    monkeypatch.setattr(capacity, "os", SimpleNamespace(umask=lambda _mask: 0o077))
    monkeypatch.setattr(
        capacity,
        "predecessor_path",
        lambda *_a: ("GF-REGIONAL-NOTIFY-007", tmp_path / "predecessor"),
    )
    monkeypatch.setattr(
        capacity, "predecessor_evidence", lambda *_a, **_k: {"valid": valid}
    )
    captured: dict[str, Any] = {}

    def plan(**kwargs: Any) -> dict[str, int]:
        captured.update(kwargs)
        return {"schema_version": 3}

    monkeypatch.setattr(capacity, "build_plan", plan)

    assert capacity.main() == (0 if valid is True else 1), (
        "failed or unknown proof must not be promoted to a passing plan"
    )
    assert captured["arguments"] is arguments, "the actual CLI targets must be bound"
    assert captured["preflight_passed"] is (valid is True), (
        "a generic truthy value is not a successful preflight"
    )
