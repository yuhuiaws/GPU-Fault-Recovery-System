from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from scripts.e2e.regional import boot_acceptance_runtime as runtime
from scripts.e2e.regional import run_boot020_release_rolling as rolling
from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder
from scripts.e2e.regional.regional_live_fixture import predecessor_evidence


@pytest.mark.parametrize(
    "items",
    [
        [],
        [{"metadata": {"name": "api"}}],
        [{"metadata": {"name": "api", "generation": True}}],
        [{"metadata": {"name": "api", "generation": 0}}],
        [{"metadata": {"name": "api", "generation": 1}}] * 2,
    ],
)
def test_lifecycle_cannot_prove_noop_with_missing_generations(
    items: list[dict[str, Any]],
) -> None:
    release = SimpleNamespace(
        config=SimpleNamespace(namespace="test"),
        _get_json=lambda _args: {"items": items},
    )

    with pytest.raises(rolling.AcceptanceCheckError, match="inventory|generation"):
        rolling.deployment_generations(release, ["mock-kubectl"], ("api",))


def test_profile_probe_uses_the_selected_shared_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    versions: list[str] = []
    profile = SimpleNamespace(
        cluster_id="another-registration-anchor",
        profile_version="profile-a",
        capabilities=[SimpleNamespace(mode="OWN", owner="gpu-fault-node-agent")],
    )
    store = SimpleNamespace(
        get_profile=lambda version: versions.append(version) or profile
    )
    monkeypatch.setattr(
        ApplicationContext, "from_environment", lambda: SimpleNamespace(store=store)
    )
    monkeypatch.setattr(sys, "argv", ["probe", "profile-a", "gpu-fault-node-agent"])

    exec(runtime.PROFILE_OWNER_PROBE, {})

    result = json.loads(capsys.readouterr().out)
    assert versions == ["profile-a"], "cluster anchoring must not hide a shared Profile"
    assert result["profiles"] == [
        {
            "profile_version": "profile-a",
            "remote_owners": ["gpu-fault-node-agent"],
            "orphan_owners": [],
        }
    ], "the actual selected Profile owners must be checked"


def test_failed_reexecution_invalidates_canonical_predecessor_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "GF-REGIONAL-BOOT-020.json"
    recorder = EvidenceRecorder(path, case_id=rolling.CASE_ID, inputs={})
    recorder.document["verdict"] = "PASS"
    recorder.complete()
    assert predecessor_evidence(path, rolling.CASE_ID)["valid"] is True, (
        "the initial completed evidence should be accepted"
    )
    backend = SimpleNamespace(
        node_safety=lambda _scenario: {"safe": True, "clusters": {}},
        classify=lambda _scenario: (_ for _ in ()).throw(
            RuntimeError("classification failed")
        ),
    )

    with pytest.raises(RuntimeError, match="classification failed"):
        rolling.run_release_rolling(backend, recorder)

    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["verdict"] == "FAIL", (
        "a prior PASS must not survive a failed reexecution"
    )
    assert predecessor_evidence(path, rolling.CASE_ID)["valid"] is False, (
        "the shared reader must reject the newly failed canonical record"
    )


def test_completed_rolling_evidence_supplies_the_next_case_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "GF-REGIONAL-BOOT-020.json"
    recorder = EvidenceRecorder(path, case_id=rolling.CASE_ID, inputs={})
    recorder.document["stages"]["full_after"] = {
        "release_id": "release-full",
        "live": {"clusters": {"cluster-a": {}}},
    }

    rolling.complete_evidence(recorder)

    assert (
        predecessor_evidence(
            path, rolling.CASE_ID, release_id="release-full", cluster_id="cluster-a"
        )["valid"]
        is True
    ), "the canonical JSON must bridge actual final release identity"
