"""SignedReleaseBuild: the release build beside the bootstrap graph."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import release_repositories
from gpu_fault.admin.bootstrap_common import BootstrapError, BootstrapRequest
from gpu_fault.admin.release_repositories import SignedReleaseBuild


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
