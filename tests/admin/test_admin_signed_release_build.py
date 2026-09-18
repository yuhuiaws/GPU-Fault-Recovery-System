"""SignedReleaseBuild: the release build beside the bootstrap graph."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import bootstrap, notification_bootstrap, release_repositories
from gpu_fault.admin.bootstrap_common import BootstrapError, BootstrapRequest
from gpu_fault.admin.bootstrap_tasks import TaskGraph
from gpu_fault.admin.notifications import NotificationRouting
from gpu_fault.admin.release_repositories import SignedReleaseBuild
from tests.admin._bootstrap_support import _cluster


def _request(tmp_path: Path) -> BootstrapRequest:
    return BootstrapRequest(
        cpu_cluster_arn="arn:aws:eks:us-east-1:123456789012:cluster/cpu",
        gpu_cluster_arns=("arn:aws:eks:us-east-1:123456789012:cluster/gpu",),
        repository_root=tmp_path,
        state_dir=tmp_path / "state",
    )


def test_the_build_runs_off_the_calling_thread_and_is_checked_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads: list[str] = []
    refusals: list[Path] = []

    def prepare(**arguments: Any) -> dict[str, Any]:
        threads.append(threading.current_thread().name)
        assert arguments["request"].state_dir == tmp_path / "state"
        return {"manifest": str(tmp_path / "release.json"), "images": {}}

    monkeypatch.setattr(
        release_repositories,
        "refuse_unconsented_release",
        lambda *, state_dir, manifest_path, existing_site: refusals.append(
            manifest_path
        ),
    )
    build = SignedReleaseBuild(prepare, existing_site=None, request=_request(tmp_path))

    first = build.result()
    second = build.result()
    build.close()

    assert first is second, "one build, handed to every caller"
    assert threads and threads[0] != threading.current_thread().name, (
        "the build must not run on the bootstrap thread"
    )
    assert refusals == [tmp_path / "release.json"], (
        "the consent refusal runs exactly once, on first use"
    )


def test_a_failed_build_is_raised_to_every_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def prepare(**_arguments: Any) -> dict[str, Any]:
        raise BootstrapError("release gates failed")

    monkeypatch.setattr(
        release_repositories,
        "refuse_unconsented_release",
        lambda **_kwargs: pytest.fail(
            "a failed build must not reach the consent check"
        ),
    )
    build = SignedReleaseBuild(prepare, existing_site=None, request=_request(tmp_path))
    with pytest.raises(BootstrapError, match="release gates failed"):
        build.result()
    with pytest.raises(BootstrapError, match="release gates failed"):
        build.result()
    build.close()


def test_an_unconsented_release_is_refused_the_first_time_anything_needs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def prepare(**_arguments: Any) -> dict[str, Any]:
        return {"manifest": str(tmp_path / "release.json"), "images": {}}

    def refuse(**_kwargs: Any) -> None:
        raise BootstrapError("release not consented")

    monkeypatch.setattr(release_repositories, "refuse_unconsented_release", refuse)
    build = SignedReleaseBuild(
        prepare, existing_site={"site": "a"}, request=_request(tmp_path)
    )
    with pytest.raises(BootstrapError, match="not consented"):
        build.result()
    build.close()


@pytest.mark.parametrize("failure_at", ["notification", "access", "graph"])
@pytest.mark.parametrize("build_fails", [False, True])
def test_bootstrap_joins_background_writes_before_reporting_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: str, build_fails: bool
) -> None:
    started = threading.Event()
    failed = threading.Event()
    finished = threading.Event()
    request = _request(tmp_path)
    cpu = _cluster()
    gpu = replace(cpu, role="gpu", hyperpod_name="gpu", context="gpu")
    routing = NotificationRouting(
        sender="alerts@example.com",
        recipients=("ops@example.com",),
        subject_prefix="test",
    )
    monkeypatch.setattr(bootstrap, "validate_bootstrap_dependencies", lambda: None)
    monkeypatch.setattr(
        bootstrap, "discover_bootstrap_scope", lambda **_: (None, cpu, (gpu,))
    )
    monkeypatch.setattr(bootstrap, "bind_bootstrap_inputs", lambda *_, **__: None)
    monkeypatch.setattr(
        bootstrap, "_initial_secure_files", lambda **_: ({}, tmp_path / "secure")
    )
    monkeypatch.setattr(bootstrap, "grafana_settings", lambda *_, **__: None)

    def prepare(**arguments: Any) -> dict[str, Any]:
        started.set()
        assert failed.wait(5), "bootstrap did not reach its foreground error"
        arguments["state"].phase("release-ready")
        finished.set()
        if build_fails:
            raise BootstrapError("background build failed")
        return {"manifest": str(tmp_path / "release.json")}

    def foreground(stage: str, result: Any) -> Any:
        if stage == failure_at:
            assert started.wait(5), "background build was never started"
            failed.set()
            raise BootstrapError(f"{stage} failed")
        return result

    monkeypatch.setattr(bootstrap, "prepare_signed_release", prepare)
    monkeypatch.setattr(
        notification_bootstrap,
        "notification_routing",
        lambda *_, **__: foreground("notification", ("ops@example.com", routing)),
    )
    monkeypatch.setattr(
        bootstrap,
        "plan_cluster_access",
        lambda *_, **__: foreground(
            "access",
            bootstrap.ClusterAccessPlan(TaskGraph({}), tmp_path, tmp_path, tmp_path),
        ),
    )
    monkeypatch.setattr(
        bootstrap, "run_bootstrap_tasks", lambda **_: foreground("graph", {})
    )
    with pytest.raises(BootstrapError, match=f"{failure_at} failed"):
        bootstrap.bootstrap_from_arns(request)

    assert finished.is_set(), "the site lock could exit while the build still writes"
    state = json.loads((request.state_dir / "bootstrap-state.json").read_text())
    assert state["phase"] == "failed", "a late build write hid the foreground failure"
