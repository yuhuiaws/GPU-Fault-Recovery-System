"""NET-007 admission edges: unknown executor identity, a missing site file, a
webhook the server accepted without an owned UID, and the execute-time
refusals (closed window, changed identity binding) that write a FAIL record
instead of touching the cluster."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_net007_transient_api_outage as net007
from scripts.e2e.regional.regional_commands import RegionalFixtureError

NODE = "node-a"
PAST = datetime(2000, 1, 1, tzinfo=timezone.utc)
FAR = datetime(2099, 1, 1, tzinfo=timezone.utc)


def settings(site_file: Path) -> net007.Settings:
    return net007.Settings(
        regional=SimpleNamespace(namespace="gpu-fault-system"),  # type: ignore[arg-type]
        node=NODE,
        site_file=site_file,
        host_probe_image="img@sha256:" + "a" * 64,
        outage_seconds=60,
    )


def preflight(site_file: Path, **overrides: Any) -> list[str]:
    arguments: dict[str, Any] = {
        "node": {
            "ready": "True",
            "unschedulable": False,
            "ownership_annotations": {},
            "labels": {"kubernetes.io/hostname": NODE},
        },
        "workloads": [],
        "state": {
            "queue": {"depth": 0, "fault_backlog_depth": 0},
            "remote_commands": {},
        },
        "executor": {"service_account": "gpu-fault-cluster-executor", "image": "img"},
        "minor": 33,
        "classifier": True,
        "existing_webhook": "",
        "permissions": {"create validatingwebhookconfigurations": True},
        "tests_passed": True,
    }
    arguments.update(overrides)
    return net007.preflight_errors(settings(site_file), **arguments)


def test_preflight_requires_a_known_executor_service_account(tmp_path: Path) -> None:
    site_file = tmp_path / "site.yaml"
    site_file.write_text("site")
    errors = preflight(site_file, executor={"service_account": "", "image": "img"})
    assert errors == ["cluster executor ServiceAccount is unknown"]


def test_preflight_requires_the_site_file_to_exist(tmp_path: Path) -> None:
    errors = preflight(tmp_path / "missing-site.yaml")
    assert errors == ["regional site file does not exist"]


def test_open_webhook_refuses_a_server_object_without_an_owned_uid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: dict[str, Any] = {}
    calls: list[tuple[str, ...]] = []

    def kubectl(plane: str, *arguments: str, input_text: str | None = None) -> str:
        calls.append((plane, *arguments[:2]))
        if arguments[0] == "create":
            applied.update(json.loads(input_text or "{}"))
            return ""
        assert arguments[0] == "get", "only create and get reach the fake"
        return json.dumps(applied)

    regional = SimpleNamespace(
        kubectl=kubectl, settings=SimpleNamespace(namespace="gpu-fault-system")
    )
    names = net007.resource_names("net007-unit-a1")
    fixture = net007.OutageFixture(
        regional,  # type: ignore[arg-type]
        names=names,
        node=NODE,
        username="system:serviceaccount:gpu-fault-system:gpu-fault-cluster-executor",
        image="img@sha256:" + "a" * 64,
        run_id="net007-unit-a1",
        deadman_seconds=900,
    )
    fixture.deadman_armed = True
    fixture.restore_at = time.time() + 900
    removed: list[dict[str, Any]] = []
    monkeypatch.setattr(
        net007, "delete_owned_resource", lambda *a, **k: removed.append(k) or ""
    )

    with pytest.raises(RegionalFixtureError, match="ownership or UID is unknown"):
        fixture.open()
    assert fixture.webhook_uid is None
    assert fixture.webhook_created is False, "the unowned webhook is closed again"
    assert removed and removed[0]["expected_uid"] is None
    assert calls[0] == ("gpu", "create", "-f")


@pytest.mark.parametrize(
    ("window", "binding_errors", "fragment"),
    [
        (PAST, [], "approved maintenance window has ended"),
        (FAR, ["predecessor drifted"], "identity or predecessor changed"),
    ],
)
def test_execute_case_records_the_refusal_without_creating_fixtures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    window: datetime,
    binding_errors: list[str],
    fragment: str,
) -> None:
    identity = {"release_id": "release-unit", "cluster_id": "cluster-unit"}
    monkeypatch.setattr(
        net007,
        "read_only_preflight",
        lambda *a: {
            "errors": [],
            "identity": identity,
            "predecessor": {"case_id": "GF-REGIONAL-NET-006"},
        },
    )
    monkeypatch.setattr(net007, "RegionalLiveFixture", lambda *a: SimpleNamespace())
    monkeypatch.setattr(
        net007,
        "case_binding",
        lambda *a: {
            "errors": binding_errors,
            "identity": identity,
            "predecessor": {"case_id": "GF-REGIONAL-NET-006"},
        },
    )
    created: list[str] = []
    monkeypatch.setattr(
        net007, "CollectorAcceptanceFixture", lambda *a, **k: created.append("probe")
    )
    monkeypatch.setattr(
        net007, "OutageFixture", lambda *a, **k: created.append("outage")
    )

    assert (
        net007.execute_case(settings(tmp_path / "site.yaml"), tmp_path, 1, window) == 1
    )
    result = json.loads(
        (tmp_path / "cases" / net007.CASE_ID / f"{net007.CASE_ID}.json").read_text()
    )
    assert result["verdict"] == "FAIL"
    assert fragment in result["error"], result
    assert result["release_id"] == "release-unit"
    assert created == [], "a refused execution must not own any fixture"
