from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import audit_auth_boundary as audit
from scripts.e2e.regional import run_identity_acceptance as entry
from scripts.e2e.regional.identity_acceptance_common import IdentityAcceptanceError
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional.test_auth008_owned_backlog import backlog_site, cluster_target

CASE_ID = "GF-REGIONAL-AUTH-008"


def dispatch_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, defect: str = "none"
) -> tuple[list[str], Any, list[str], Path]:
    case_dir = tmp_path / "cases" / CASE_ID
    case_dir.mkdir(parents=True)
    site, context, _receipt, events = backlog_site(monkeypatch, case_dir, defect)
    targets = {cluster: cluster_target(cluster, tmp_path) for cluster in ("a", "b")}
    site.target = lambda cluster: targets[cluster]
    # The entry hashes the site inputs into the plan environment and pins
    # both kubeconfigs there, exactly as ``IdentitySite`` exposes them.
    site_file = tmp_path / "synthetic-site.yaml"
    site_file.write_text("synthetic: site\n", encoding="utf-8")
    site.cpu_kubeconfig = tmp_path / "cpu.kubeconfig"
    site.gpu_kubeconfig = tmp_path / "gpu.kubeconfig"
    site.cpu_kubeconfig.write_text("unit CPU connection fixture\n", encoding="utf-8")
    site.gpu_kubeconfig.write_text("unit GPU connection fixture\n", encoding="utf-8")
    monkeypatch.setattr(entry, "install_site_profile", lambda: None)
    monkeypatch.setattr(entry, "IdentitySite", lambda path: site)
    monkeypatch.setattr(os, "umask", lambda mask: None)
    monkeypatch.setattr(
        entry,
        "predecessor_path",
        lambda *args: ("GF-REGIONAL-AUTH-007", tmp_path / "previous.json"),
    )
    monkeypatch.setattr(
        entry,
        "predecessor_evidence",
        lambda *args, **kwargs: {
            "valid": True,
            "case_id": "GF-REGIONAL-AUTH-007",
            "verdict": "PASS",
        },
    )

    def authorize(arguments: Any, **kwargs: Any) -> None:
        events.append("authorize")
        assert arguments.case == kwargs["case_id"] == CASE_ID
        assert arguments.confirm == kwargs["confirmation"] == "AUTH008_EXECUTE"
        assert kwargs["environment"]["GPU_FAULT_PRIMARY_CLUSTER_ID"] == "a"
        assert kwargs["environment"]["GPU_FAULT_SECONDARY_CLUSTER_ID"] == "b"
        if defect == "denied":
            raise RuntimeError("authorization denied")

    handler: Callable[..., dict[str, Any]] = getattr(entry, "run_auth008")

    def invoke(
        actual_site: Any, primary: Any, secondary: Any, **kwargs: Any
    ) -> dict[str, Any]:
        events.append("handler")
        assert actual_site is site
        assert primary is targets["a"] and secondary is targets["b"]
        assert kwargs == {"case_dir": case_dir}
        return handler(actual_site, primary, secondary, **kwargs)

    monkeypatch.setattr(entry, "authorize_execution", authorize)
    monkeypatch.setattr(entry, "run_auth008", invoke)
    argv = [
        "run_identity_acceptance.py",
        "--run-dir",
        str(tmp_path),
        "--site",
        str(site_file),
        "--case",
        CASE_ID,
        "--cluster-id",
        "a",
        "--secondary-cluster-id",
        "b",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return argv, context, events, case_dir


@pytest.mark.parametrize(
    "defect", ["none", "positive-empty", "denied", "legacy-protocol"]
)
def test_auth008_cli_reaches_owned_handler_only_after_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    argv, context, events, case_dir = dispatch_fixture(
        monkeypatch, tmp_path, defect=defect
    )
    argv.extend(["--execute", "--confirm", "AUTH008_EXECUTE"])
    if defect == "denied":
        with pytest.raises(RuntimeError, match="authorization denied"):
            entry.main()
        assert events == ["authorize"]
        assert context.store.list_remote_commands() == []
        assert not (case_dir / f"{CASE_ID}.json").exists(), (
            "denied authorization must not publish case evidence"
        )
        return
    assert entry.main() == (0 if defect == "none" else 1)
    assert events[:3] == ["authorize", "handler", "seed"]
    document = json.loads((case_dir / f"{CASE_ID}.json").read_text())
    assert document["verdict"] == ("PASS" if defect == "none" else "FAIL")
    assert document["checks"]["owned_records_retired"] is True
    assert document["release_id"] == "release-test"
    assert document["cluster_id"] == "a"
    assert document["candidate_before"]["cluster_id"] == "b"
    assert len(context.store.list_remote_commands()) == 1
    assert context.store.list_remote_commands()[0].status.value == "FAILED"


def test_auth008_default_cli_builds_a_bound_plan_without_seeding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _argv, context, events, _case_dir = dispatch_fixture(monkeypatch, tmp_path)
    plans = []

    def build_plan(**kwargs: Any) -> dict[str, Any]:
        plans.append(kwargs)
        return {
            "case_id": kwargs["case_id"],
            "preflight_passed": kwargs["preflight_passed"],
        }

    monkeypatch.setattr(entry, "build_plan", build_plan)
    assert entry.main() == 0
    assert events == []
    assert context.store.list_remote_commands() == []
    assert len(plans) == 1
    assert plans[0]["preflight_passed"] is True
    assert plans[0]["details"]["primary"] == {"cluster_id": "a", "context": "context-a"}
    assert plans[0]["details"]["secondary"] == {
        "cluster_id": "b",
        "context": "context-b",
    }
    assert "FREEZE_EVIDENCE" in plans[0]["details"]["mutation"]


@pytest.mark.parametrize("secondary", ["", "a"])
def test_auth008_cli_requires_a_distinct_secondary_before_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, secondary: str
) -> None:
    argv, context, events, _case_dir = dispatch_fixture(monkeypatch, tmp_path)
    argv[-1] = secondary
    argv.extend(["--execute", "--confirm", "AUTH008_EXECUTE"])
    with pytest.raises(
        IdentityAcceptanceError, match="secondary-cluster-id|must differ"
    ):
        entry.main()
    assert events == []
    assert context.store.list_remote_commands() == []


@pytest.mark.parametrize(
    ("guarded_verdict", "matrix_verdict", "owned_verdict"),
    [
        ("PASS", "FAIL", "PASS"),
        ("FAIL", "PASS", "PASS"),
        (None, "PASS", "PASS"),
        ("PASS", "PASS", "FAIL"),
    ],
)
def test_legacy_matrix_never_publishes_or_overwrites_guarded_auth008(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    guarded_verdict: str | None,
    matrix_verdict: str,
    owned_verdict: str,
) -> None:
    guarded_path = tmp_path / "cases" / CASE_ID / f"{CASE_ID}.json"
    original = json.dumps(
        {"case_id": CASE_ID, "verdict": guarded_verdict, "owned_backlog": True}
    )
    if guarded_verdict is not None:
        guarded_path.parent.mkdir(parents=True)
        guarded_path.write_text(original)
    owned_case = "GF-REGIONAL-AUTH-001"
    documents = {
        owned_case: {"case_id": owned_case, "verdict": owned_verdict},
        CASE_ID: {"case_id": CASE_ID, "verdict": matrix_verdict},
    }
    monkeypatch.setattr(audit, "run_matrix", lambda arguments: {})
    monkeypatch.setattr(
        audit, "case_documents", lambda *args, **kwargs: dict(documents)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_auth_boundary.py",
            "matrix",
            "--url",
            "https://unused.invalid",
            "--ca-file",
            str(tmp_path / "unused-ca"),
            "--cluster-a",
            "a",
            "--cluster-b",
            "b",
            "--token-a-file",
            str(tmp_path / "unused-a"),
            "--token-b-file",
            str(tmp_path / "unused-b"),
            "--executor-artifact-sha256",
            "a" * 64,
            "--executor-compatibility-digest",
            "b" * 64,
            "--run-dir",
            str(tmp_path),
        ],
    )
    assert audit.main() == (0 if owned_verdict == "PASS" else 1)
    if guarded_verdict is None:
        assert not guarded_path.exists(), (
            "legacy diagnostics must not create formal AUTH-008 evidence"
        )
    else:
        assert guarded_path.read_text() == original
    output = json.loads(capsys.readouterr().out)
    assert output["verdicts"] == {owned_case: owned_verdict}
    assert CASE_ID in output["not_evaluated"]
    owned_path = tmp_path / "cases" / owned_case / f"{owned_case}.json"
    assert json.loads(owned_path.read_text()) == documents[owned_case]
