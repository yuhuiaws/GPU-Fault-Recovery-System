from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import iso006_primary_recovery as primary
from scripts.e2e.regional import run_iso006_cluster_offline as runner
from tests.regional.test_identity_causal_review import lifecycle_harness


@pytest.mark.parametrize(
    "defect", ["none", "no-cut", "cut-lost", "binding", "ack", "cleanup"]
)
def test_iso006_primary_requires_causal_recovery_inside_a_verified_cut(
    defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, targets, events = lifecycle_harness(monkeypatch, tmp_path, defect=defect)
    monkeypatch.setattr(
        primary, "ImagePrewarmFixture", primary.workload.ImagePrewarmFixture
    )
    regional = site.regional(targets[0])
    fixture = primary.PrimaryRecovery(
        regional,
        site_file=site.site_file,
        case_dir=tmp_path,
        job_id="job-test",
        attempt_id="attempt-test",
        attempt=1,
        maintenance_window_end=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    fixture.prepare()
    samples = []

    def cut_is_active() -> bool:
        samples.append("sample")
        return defect != "no-cut" and not (defect == "cut-lost" and len(samples) > 1)

    if defect in {"no-cut", "cut-lost", "binding", "ack"}:
        with pytest.raises(primary.RegionalFixtureError):
            fixture.recover(time.monotonic() + 120, cut_is_active=cut_is_active)
        if defect == "no-cut":
            assert "inject-a" not in events
    else:
        outcome = fixture.recover(time.monotonic() + 120, cut_is_active=cut_is_active)
        assert outcome["state"]["workflow"]["status"] == "SUCCEEDED"
        assert outcome["state"]["event"]["evidence_ref"] == (
            "api-replay://GF-REGIONAL-ISO-006/iso006-attempt-test"
        )
        assert len(samples) == 2
    result: dict[str, Any] = {}
    errors = fixture.cleanup(result)
    assert bool(errors) is (defect == "cleanup")
    assert "delete-a" in events
    if defect not in {"no-cut"}:
        assert result["cluster_a_cleanup_quiescence"]["safe_to_delete"] is True


def test_iso006_target_site_and_workload_identity_are_explicit_plan_inputs(
    tmp_path: Path,
) -> None:
    for name in ("cpu", "a", "b", "site"):
        (tmp_path / name).write_text("synthetic test fixture\n", encoding="ascii")
    arguments = runner.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--site-file",
            str(tmp_path / "site"),
            "--cpu-kubeconfig",
            str(tmp_path / "cpu"),
            "--region",
            "us-west-2",
            "--namespace",
            "gpu-system",
            "--cluster-a",
            "a",
            "--gpu-a-kubeconfig",
            str(tmp_path / "a"),
            "--gpu-a-context",
            "context-a",
            "--cluster-b",
            "b",
            "--gpu-b-kubeconfig",
            str(tmp_path / "b"),
            "--gpu-b-context",
            "context-b",
            "--host-probe-image",
            "image@sha256:" + "a" * 64,
            "--control-plane-cidr",
            "10.0.0.0/24",
            "--job-id",
            "job-test",
            "--attempt-id",
            "attempt-test",
        ]
    )
    settings = runner.configure(arguments)
    assert settings.site_file == tmp_path / "site"
    assert settings.multi.namespace == "gpu-system"
    environment = settings.environment()
    assert environment["GPU_FAULT_SITE_FILE"] == str(tmp_path / "site")
    assert environment["GPU_FAULT_SHARED_JOB_ID"] == "job-test"
    assert environment["GPU_FAULT_SHARED_ATTEMPT_ID"] == "attempt-test"
    plan = runner.plan_details(
        settings,
        {
            "release_id": "release-test",
            "nodes_a": [{"uid": "a-uid", "name": "a"}],
            "nodes_b": [{"uid": "b-uid", "name": "b"}],
            "predecessor": {"valid": True},
            "focused_tests": {"passed": True},
        },
    )
    assert plan["shared_job_id"] == "job-test"
    assert plan["shared_attempt_id"] == "attempt-test"
    assert plan["risk"] == "live-workload-restart"
