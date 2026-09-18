"""Canonical evidence and proof-layer binding for NET audit entrypoints."""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import audit_collector_outbox as outbox
from scripts.e2e.regional import audit_net004_dependency_boundary as dependency
from scripts.e2e.regional import run_net007_transient_api_outage as outage
from scripts.e2e.regional.regional_case_contract import (
    case_evidence_path,
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import predecessor_evidence

IDENTITY = {"release_id": "release-test", "cluster_id": "cluster-a"}


def predecessor(run_dir: Path, case_id: str, *, release: str = "release-test") -> None:
    previous, path = predecessor_path(run_dir, case_id, "")
    assert previous is not None and path is not None, (
        "test case must have a formal predecessor"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"case_id": previous, "verdict": "PASS", **IDENTITY, "release_id": release}
        )
    )


@pytest.mark.parametrize("binding", ["valid", "missing", "wrong-release"])
def test_outbox_evidence_is_explicitly_local_and_binds_only_the_enclosing_run(
    tmp_path: Path, binding: str
) -> None:
    if binding != "missing":
        predecessor(
            tmp_path,
            outbox.CASE_ID,
            release="old-release" if binding == "wrong-release" else "release-test",
        )
    arguments = ["--run-dir", str(tmp_path)]
    if binding != "missing":
        arguments += [
            "--release-id",
            IDENTITY["release_id"],
            "--cluster-id",
            IDENTITY["cluster_id"],
        ]
    result = outbox.main(arguments)
    path = case_evidence_path(tmp_path, outbox.CASE_ID)
    document = json.loads(path.read_text())
    assert document["validation_scope"] == "isolated-source"
    assert document["live_validation"] is False
    assert document["status"] == "COMPLETED"
    assert result == (1 if binding == "wrong-release" else 0)
    if binding == "valid":
        assert document["release_id"] == IDENTITY["release_id"]
        assert document["cluster_id"] == IDENTITY["cluster_id"]
        assert document["identity_source"] == "enclosing acceptance run"
        assert document["predecessor"]["valid"] is True
    elif binding == "missing":
        assert document["formal_sequence_satisfied"] is False
        assert predecessor_evidence(path, outbox.CASE_ID, **IDENTITY)["valid"] is False
    else:
        assert document["verdict"] == "FAIL"


def test_outbox_failure_still_writes_component_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail() -> dict[str, Any]:
        raise outbox.OutboxAuditFailure("replay order was not preserved")

    monkeypatch.setattr(outbox, "audit_outbox", fail)
    assert outbox.main(["--run-dir", str(tmp_path)]) == 1
    document = json.loads(case_evidence_path(tmp_path, outbox.CASE_ID).read_text())
    assert document["verdict"] == "FAIL"
    assert document["live_validation"] is False
    assert "replay order" in document["error"]


@pytest.mark.parametrize("failure", [None, "predecessor", "release-drift", "audit"])
def test_dependency_audit_writes_bound_evidence_on_success_or_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    chain = {
        "identity": IDENTITY,
        "predecessor": {
            "case_id": "GF-REGIONAL-NET-003",
            "valid": failure != "predecessor",
        },
        "errors": ["wrong predecessor"] if failure == "predecessor" else [],
    }
    calls: list[str] = []
    chains = [copy.deepcopy(chain), copy.deepcopy(chain)]
    if failure == "release-drift":
        chains[1]["identity"]["release_id"] = "different-release"

    def inspect() -> dict[str, Any]:
        calls.append("audit")
        if failure == "audit":
            raise dependency.CaseError("boundary read failed")
        return {
            "case_id": dependency.CASE_ID,
            "verdict": "PASS",
            "checks": {"observed": True},
        }

    monkeypatch.setattr(dependency, "configure", lambda args: None)
    monkeypatch.setattr(dependency, "chain_preflight", lambda *a: chains.pop(0))
    monkeypatch.setattr(dependency, "audit", inspect)
    monkeypatch.setattr(sys, "argv", ["dependency-audit", "--run-dir", str(tmp_path)])
    assert dependency.main() == (0 if failure is None else 1)
    document = json.loads(case_evidence_path(tmp_path, dependency.CASE_ID).read_text())
    assert document["release_id"] == "release-test"
    assert document["cluster_id"] == "cluster-a"
    assert document["predecessor"]["case_id"] == "GF-REGIONAL-NET-003"
    assert document["validation_scope"] == "deployed-readonly"
    assert calls == ([] if failure == "predecessor" else ["audit"])


def test_dependency_transport_uses_checked_supervision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def run(command: list[str], **kwargs: Any) -> Any:
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="observed")

    monkeypatch.setattr(dependency, "run_fixture_command", run)
    assert (
        dependency.run(["read-only-probe"], stdin="payload", timeout=12) == "observed"
    )
    assert calls == [
        (
            ["read-only-probe"],
            {"input_text": "payload", "timeout": 12, "cwd": dependency.ROOT},
        )
    ]


@pytest.mark.parametrize("release", [None, "old-release"])
def test_net007_refuses_unbound_predecessor_before_any_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release: str | None
) -> None:
    regional = SimpleNamespace(evidence_identity=lambda: dict(IDENTITY))
    if release is not None:
        predecessor(tmp_path, outage.CASE_ID, release=release)
    monkeypatch.setattr(outage, "RegionalLiveFixture", lambda settings: regional)
    monkeypatch.setattr(
        outage,
        "read_only_preflight",
        lambda settings, case_dir: outage.case_binding(regional, tmp_path),
    )
    monkeypatch.setattr(
        outage,
        "CollectorAcceptanceFixture",
        lambda *a, **k: pytest.fail(
            "invalid predecessor must stop before resource creation"
        ),
    )
    settings = outage.Settings(
        regional=SimpleNamespace(),
        node="node-a",
        site_file=tmp_path / "site.yaml",
        host_probe_image="image",
        outage_seconds=outage.verdicts.MIN_OUTAGE_SECONDS,
    )
    assert (
        outage.execute_case(
            settings, tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
        )
        == 1
    )
    document = json.loads(case_evidence_path(tmp_path, outage.CASE_ID).read_text())
    assert document["verdict"] == "FAIL"
    assert document["release_id"] == "release-test"
    assert document["predecessor"]["valid"] is False
