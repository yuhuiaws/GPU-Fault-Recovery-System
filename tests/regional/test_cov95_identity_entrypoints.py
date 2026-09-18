from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_identity_acceptance as entry
from tests.regional._cov95_identity_support import offline_guard as offline_guard


def cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case_id: str
) -> tuple[list[str], list[str], dict[str, Any]]:
    master = tmp_path / "unit-master"
    master.write_text("m" * 64, encoding="ascii")
    master.chmod(0o600)
    targets = {
        name: SimpleNamespace(cluster_id=name, context="context-" + name)
        for name in ("a", "b")
    }
    for name in ("site", "cpu.kubeconfig", "gpu.kubeconfig"):
        (tmp_path / name).write_text("unit connection fixture\n")
    site = SimpleNamespace(
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
        target=lambda name: targets[name],
        regional=lambda target: SimpleNamespace(
            evidence_identity=lambda: {
                "release_id": "unit-release",
                "cluster_id": target.cluster_id,
            }
        ),
    )
    events = []
    calls = {}
    monkeypatch.setattr(entry, "install_site_profile", lambda: None)
    monkeypatch.setattr(entry.os, "umask", lambda mask: None)
    monkeypatch.setattr(entry, "IdentitySite", lambda path: site)
    monkeypatch.setattr(
        entry, "predecessor_path", lambda *args: ("previous", tmp_path / "previous")
    )
    monkeypatch.setattr(
        entry, "predecessor_evidence", lambda *args, **kwargs: {"valid": True}
    )
    monkeypatch.setattr(entry, "reusable_focused_tests", lambda *args: {"passed": True})

    def authorize(*args: Any, **kwargs: Any) -> None:
        calls["authorization"] = kwargs
        events.append("authorize")

    monkeypatch.setattr(entry, "authorize_execution", authorize)
    argv = [
        "identity",
        "--run-dir",
        str(tmp_path),
        "--site",
        str(tmp_path / "site"),
        "--case",
        case_id,
        "--cluster-id",
        "a",
        "--secondary-cluster-id",
        "b",
    ]
    if case_id == "GF-REGIONAL-AUTH-015":
        argv.extend(
            [
                "--node",
                "node-a",
                "--node",
                "node-b",
                "--fleet-master-file",
                str(master),
                "--host-probe-image",
                "unit@sha256:" + "a" * 64,
            ]
        )
    monkeypatch.setattr(sys, "argv", argv)
    return argv, events, calls


@pytest.mark.parametrize("changed", ["cpu", "gpu", "site", "release"])
def test_identity_plan_rejects_connection_site_or_release_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: str
) -> None:
    from scripts.e2e.regional import live_driver_guard

    argv, _events, _calls = cli(monkeypatch, tmp_path, "GF-REGIONAL-ISO-005")
    site = entry.IdentitySite(tmp_path / "site")
    monkeypatch.setattr(
        entry, "authorize_execution", live_driver_guard.authorize_execution
    )
    monkeypatch.setattr(live_driver_guard, "source_digest", lambda: "a" * 64)
    assert entry.main() == 0
    plan = json.loads((tmp_path / "cases/GF-REGIONAL-ISO-005/plan.json").read_text())
    assert len(plan["connections"]) == 2
    assert plan["environment"]["DEPLOYED_RELEASE_ID"] == "unit-release"
    if changed == "release":
        site.regional = lambda target: SimpleNamespace(
            evidence_identity=lambda: {
                "release_id": "another-release",
                "cluster_id": target.cluster_id,
            }
        )
    else:
        path = tmp_path / ("site" if changed == "site" else f"{changed}.kubeconfig")
        path.write_text("changed unit fixture\n")
    called = []
    monkeypatch.setattr(entry, "run_iso005", lambda *args: called.append(args))
    argv.extend(
        [
            "--execute",
            "--confirm",
            "ISO005_EXECUTE",
            "--maintenance-window-end",
            (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        ]
    )
    with pytest.raises(
        RuntimeError, match="drifted at environment|drifted at connections"
    ):
        entry.main()
    assert called == [], "changed live inputs must be rejected before probe dispatch"


@pytest.mark.parametrize("case_id", entry.CASE_IDS)
def test_identity_main_dispatches_only_the_authorized_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case_id: str
) -> None:
    argv, events, calls = cli(monkeypatch, tmp_path, case_id)
    confirmation = case_id.removeprefix("GF-REGIONAL-").replace("-", "") + "_EXECUTE"
    argv.extend(["--execute", "--confirm", confirmation])
    handler_name = (
        "run_" + case_id.removeprefix("GF-REGIONAL-").replace("-", "").lower()
    )
    verdict = "FAIL" if case_id == "GF-REGIONAL-AUTH-015" else "PASS"

    def handler(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert events == ["authorize"]
        events.append("handler")
        calls["handler"] = (args, kwargs)
        assert args[1].cluster_id == "a"
        return {"verdict": verdict, "checks": {"offline_dispatch": True}}

    monkeypatch.setattr(entry, handler_name, handler)
    assert entry.main() == int(verdict == "FAIL")
    assert events == ["authorize", "handler"]
    assert calls["authorization"]["case_id"] == case_id
    assert calls["authorization"]["confirmation"] == confirmation
    if case_id == "GF-REGIONAL-AUTH-015":
        assert calls["handler"][1]["nodes"] == ("node-a", "node-b")
        assert calls["handler"][1]["focused_tests"] == {"passed": True}
    document = json.loads(
        (tmp_path / "cases" / case_id / f"{case_id}.json").read_text()
    )
    assert document["verdict"] == verdict
    assert document["cluster_id"] == "a" and document["release_id"] == "unit-release"


@pytest.mark.parametrize(
    "failure", ["confirmation", "predecessor", "handler", "partial"]
)
def test_identity_main_refuses_or_records_failure_without_fabricating_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    case_id = "GF-REGIONAL-AUTH-010"
    argv, events, _calls = cli(monkeypatch, tmp_path, case_id)
    argv.extend(
        [
            "--execute",
            "--confirm",
            "WRONG" if failure == "confirmation" else "AUTH010_EXECUTE",
        ]
    )
    if failure == "predecessor":
        monkeypatch.setattr(
            entry, "predecessor_evidence", lambda *args, **kwargs: {"valid": False}
        )

    def handler(*args: Any) -> Any:
        events.append("handler")
        if failure == "partial":
            raise entry.IdentityCaseFailure(
                "probe failed",
                details={
                    "cleanup_errors": ["unproven cleanup"],
                    "checks": {"first_probe": True},
                },
            )
        raise RuntimeError("synthetic read failed")

    monkeypatch.setattr(entry, "run_auth010", handler)
    path = tmp_path / "cases" / case_id / f"{case_id}.json"
    if failure in {"confirmation", "predecessor"}:
        with pytest.raises(
            entry.IdentityAcceptanceError, match="confirmation|predecessor"
        ):
            entry.main()
        assert "handler" not in events, "failed admission must not invoke the handler"
        if failure == "confirmation":
            assert not path.exists(), "unapproved execution must not create evidence"
        else:
            pending = json.loads(path.read_text())
            assert pending["case_id"] == case_id, pending
            assert pending["verdict"] == "NOT_RUN" and pending["status"] == "RUNNING", (
                "authorized predecessor failure must leave current non-PASS evidence",
                pending,
            )
    else:
        assert entry.main() == 1
        result = json.loads(path.read_text())
        assert result["verdict"] == "FAIL"
        if failure == "partial":
            assert result["partial"]["checks"] == {"first_probe": True}
            assert result["cleanup_errors"] == ["unproven cleanup"]
        else:
            assert result["error"] == "RuntimeError: synthetic read failed"


@pytest.mark.parametrize("focused_ok", [False, True])
def test_auth015_plan_requires_the_recorded_signature_test_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, focused_ok: bool
) -> None:
    _argv, events, _calls = cli(monkeypatch, tmp_path, "GF-REGIONAL-AUTH-015")
    monkeypatch.setattr(entry, "auth015_focused_tests", lambda: {"passed": focused_ok})
    monkeypatch.setattr(
        entry,
        "record_focused_tests",
        lambda details, tests: details.update({"focused_tests": tests}),
    )
    plans = []
    monkeypatch.setattr(
        entry,
        "build_plan",
        lambda **kwargs: plans.append(kwargs)
        or {"preflight_passed": kwargs["preflight_passed"]},
    )
    assert entry.main() == int(not focused_ok)
    assert events == []
    assert plans[0]["preflight_passed"] is focused_ok
    assert plans[0]["details"]["focused_tests"]["passed"] is focused_ok
    assert plans[0]["environment"]["GPU_FAULT_TARGET_NODES"] == "node-a,node-b"


@pytest.mark.parametrize(
    "defect",
    [
        "one-node",
        "same-node",
        "missing-master",
        "missing-image",
        "missing-file",
        "mutable-image",
    ],
)
def test_auth015_arguments_require_two_nodes_and_explicit_custody_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    argv, events, _calls = cli(monkeypatch, tmp_path, "GF-REGIONAL-AUTH-015")
    args = entry.parser().parse_args(argv[1:])
    if defect == "one-node":
        args.node = ["node-a"]
    elif defect == "same-node":
        args.node = ["node-a", "node-a"]
    elif defect == "missing-master":
        args.fleet_master_file = None
    elif defect == "missing-image":
        args.host_probe_image = ""
    elif defect == "missing-file":
        args.fleet_master_file = tmp_path / "absent"
    else:
        args.host_probe_image = "unit:latest"
    site = SimpleNamespace(target=lambda name: SimpleNamespace(cluster_id=name))
    with pytest.raises(
        entry.IdentityAcceptanceError, match="requires|does not exist|immutable"
    ):
        entry.validate_case_arguments(args, site)
    assert events == []


def test_auth013_argument_image_must_be_immutable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    argv, _events, _calls = cli(monkeypatch, tmp_path, "GF-REGIONAL-AUTH-013")
    args = entry.parser().parse_args(
        [*argv[1:], "--node", "node-a", "--host-probe-image", "unit:latest"]
    )
    with pytest.raises(entry.IdentityAcceptanceError, match="immutable"):
        entry.validate_case_arguments(
            args, SimpleNamespace(target=lambda name: SimpleNamespace(cluster_id=name))
        )
