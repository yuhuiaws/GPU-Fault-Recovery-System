"""GF-REGIONAL-DESTR-012 runner: the pure gates on inputs the happy fixtures
never produce -- CPU replicas that disagree on allowed namespaces, group D
settings and an unknown group, duplicate-window timestamps that are naive or
inverted -- and ``execute_case`` refusing a preflight that reports errors."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr012_managed_recovery_guard as destr012
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveSettings,
)

T0 = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)


def _regional_settings(tmp_path: Path) -> RegionalLiveSettings:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    return RegionalLiveSettings(
        cpu_kubeconfig=cpu,
        gpu_kubeconfig=gpu,
        gpu_context="gpu-context",
        namespace="gpu-fault-system",
        cluster_id="cluster-a",
        region="us-west-2",
    )


def _settings(tmp_path: Path) -> destr012.Settings:
    site = tmp_path / "site.yaml"
    site.write_text("clusters: []\n", encoding="utf-8")
    for group in ("a", "d"):
        (tmp_path / f"{group}.yaml").write_text("kind: Job\n", encoding="utf-8")
    return destr012.Settings(
        regional=_regional_settings(tmp_path),
        site_file=site,
        a_manifest=tmp_path / "a.yaml",
        d_manifest=tmp_path / "d.yaml",
        a_job_id="job-a",
        a_attempt_id="job-a-a001",
        d_job_id="job-d",
        d_attempt_id="job-d-a001",
        predecessor_path=tmp_path / "GF-REGIONAL-DESTR-009.json",
    )


class _Replicas:
    """A regional fixture with two Ready replicas per CPU app and no cluster."""

    def __init__(self, settings: RegionalLiveSettings) -> None:
        self.settings = settings
        self.asked: list[tuple[str, str]] = []

    def ready_pods(self, plane: str, app: str) -> list[dict[str, str]]:
        self.asked.append((plane, app))
        return [{"name": f"{app}-0"}, {"name": f"{app}-1"}]


def _owned_profile() -> dict[str, Any]:
    return {
        "capabilities": [
            {
                "capability": name,
                "mode": "OWN",
                "owner": "gpu-fault-kubernetes-adapter",
                "adapter": "regional-cluster-executor",
            }
            for name in ("workloadStop", "workloadRestart")
        ]
    }


def test_replica_audit_reports_namespace_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _Replicas(_regional_settings(tmp_path))

    def probe(
        _regional: Any, plane: str, pod: str, script: str, *arguments: str
    ) -> dict[str, Any]:
        namespaces = ["team-a"] if pod.endswith("-0") else ["team-a", "team-b"]
        return {
            "profile_sha256": "same" * 16,
            "allowed_namespaces": namespaces,
            "profile": _owned_profile(),
        }

    monkeypatch.setattr(destr012, "pod_python", probe)
    audit = destr012.profile_replica_audit(regional, profile_version="v7")
    assert audit["errors"] == ["CPU replicas disagree on allowed namespaces"]
    assert audit["allowed_namespaces"] == [], "no namespace list is trusted"
    assert audit["profile_sha256s"] == ["same" * 16]
    assert len(audit["replicas"]) == 4
    assert regional.asked == [
        ("cpu", "gpu-fault-api-ha"),
        ("cpu", "gpu-fault-control-worker"),
    ]


def test_replica_audit_agreement_yields_the_shared_namespace_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _Replicas(_regional_settings(tmp_path))
    monkeypatch.setattr(
        destr012,
        "pod_python",
        lambda *_a: {
            "profile_sha256": "a" * 64,
            "allowed_namespaces": ["team-b", "team-a"],
            "profile": _owned_profile(),
        },
    )
    audit = destr012.profile_replica_audit(regional, profile_version="v7")
    assert audit["errors"] == []
    assert audit["allowed_namespaces"] == ["team-a", "team-b"]


def test_workload_settings_select_group_d_and_refuse_unknown_groups(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    group_d = destr012.workload_settings(settings, group="D")
    assert group_d.manifest == settings.d_manifest
    assert (group_d.job_id, group_d.attempt_id) == ("job-d", "job-d-a001")
    assert group_d.predecessor_path == settings.predecessor_path
    group_a = destr012.workload_settings(settings, group="A")
    assert (group_a.manifest, group_a.job_id) == (settings.a_manifest, "job-a")
    with pytest.raises(ValueError, match="unknown DESTR-012 group: B"):
        destr012.workload_settings(settings, group="B")


@pytest.mark.parametrize(
    ("first_at", "now"),
    [
        (T0.replace(tzinfo=None), T0),
        (T0, T0.replace(tzinfo=None)),
        (T0 + timedelta(seconds=1), T0),
    ],
)
def test_duplicate_window_refuses_naive_or_inverted_timestamps(
    first_at: datetime, now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        destr012.time, "sleep", lambda seconds: pytest.fail("no wait on a refusal")
    )
    with pytest.raises(RegionalFixtureError, match="timestamps are invalid"):
        destr012.wait_out_duplicate_window(first_at, now=now)


def test_execute_case_refuses_a_failing_preflight_before_any_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    seen: list[dict[str, Any]] = []

    def failing_preflight(
        _settings: destr012.Settings, case_dir: Path, **kwargs: Any
    ) -> dict[str, Any]:
        seen.append({"case_dir": case_dir, **kwargs})
        return {"errors": ["target node is not Ready", "runtime profile has warnings"]}

    monkeypatch.setattr(destr012, "read_only_preflight", failing_preflight)
    monkeypatch.setattr(
        destr012,
        "run_group_a",
        lambda *a, **k: pytest.fail("no group runs after a failed preflight"),
    )
    with pytest.raises(
        RegionalFixtureError,
        match="preflight failed: target node is not Ready; runtime profile has",
    ):
        destr012.execute_case(
            settings, tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
        )
    assert seen == [
        {"case_dir": tmp_path / "cases" / destr012.CASE_ID, "reuse_focused_tests": True}
    ]
    assert (tmp_path / "cases" / destr012.CASE_ID).is_dir(), (
        "the case directory exists before the preflight is read"
    )
