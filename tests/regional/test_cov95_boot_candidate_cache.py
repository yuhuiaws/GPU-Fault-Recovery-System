from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import boot020_release_candidates as candidates
from tests.regional._cov95_boot_candidates import CandidateBuild


@pytest.mark.parametrize(
    "manifest",
    [
        None,
        [],
        "not-an-object",
        {"release_id": ""},
        {"release_id": " "},
        {"release_id": 1},
    ],
)
def test_malformed_candidate_manifest_is_not_reusable(
    tmp_path: Path, manifest: Any
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    release_id = (
        manifest.get("release_id") if isinstance(manifest, dict) else "candidate"
    )
    assert (
        candidates.reusable_candidate(
            path,
            base_release_id="base",
            recorded={"release_id": release_id, "edits_sha256": "a" * 64},
            expected_edits_digest="a" * 64,
        )
        is False
    )


@pytest.mark.parametrize(
    "cache",
    [
        None,
        "invalid-cache",
        {"base_release_id": "base", "candidates": ["invalid-entry"]},
    ],
)
def test_malformed_candidate_summary_is_rebuilt_without_reusing_unverified_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache: Any
) -> None:
    fixture = CandidateBuild(tmp_path, monkeypatch)
    fixture.arguments.work_dir.mkdir()
    (fixture.arguments.work_dir / "candidates.json").write_text(json.dumps(cache))
    result = candidates.build_candidates(fixture.arguments)
    assert set(result["candidates"]) == {"B", "C", "D"}
    assert len(fixture.build_calls) == 3
    assert all(
        item["release_id"] != "base" for item in result["candidates"].values()
    ), "an invalid cache reused the base as a candidate"
