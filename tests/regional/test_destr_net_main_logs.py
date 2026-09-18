"""Silence needs retained logs and a stable, uniquely identified container."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest

from scripts.e2e.regional import run_destr009_workload_restart as restart
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError

SINCE = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)
APP = "gpu-fault-api-ha"


def pod_document() -> dict[str, Any]:
    return {
        "metadata": {"name": "pod-a", "uid": "uid-a"},
        "spec": {"containers": [{"name": "api"}]},
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True"}],
            "containerStatuses": [
                {
                    "name": "api",
                    "ready": True,
                    "restartCount": 0,
                    "containerID": "containerd://instance-a",
                    "state": {
                        "running": {
                            "startedAt": (SINCE - timedelta(hours=1)).isoformat()
                        }
                    },
                }
            ],
        },
    }


class LogRegional:
    def __init__(self) -> None:
        self.window = ""
        self.history = (
            f"{(SINCE - timedelta(minutes=5)).isoformat()} startup complete\n"
        )
        self.document = pod_document()
        self.after: dict[str, Any] | None = None
        self.history_after: str | None = None
        self.window_error = False
        self.history_error = False
        self.describe_error = False
        self.calls: list[tuple[str, ...]] = []
        self.described = self.histories = 0

    def ready_pods(self, _plane: str, app: str) -> list[dict[str, Any]]:
        return [{"name": "pod-a", "uid": "uid-a"}] if app == APP else []

    def kubectl(self, plane: str, *arguments: str, **kwargs: Any) -> str:
        assert plane == "cpu"
        assert kwargs.get("check", True) is True, "failed reads must not be swallowed"
        self.calls.append(arguments)
        if arguments[0] == "get":
            if self.describe_error:
                raise RegionalFixtureError("describe refused")
            self.described += 1
            return json.dumps(
                self.after
                if self.described > 1 and self.after is not None
                else self.document
            )
        if "--since-time" in arguments:
            if self.window_error:
                raise RegionalFixtureError("window refused")
            return self.window
        assert "--container=api" in arguments and "--timestamps" in arguments
        assert "--tail=-1" in arguments
        assert any(value.startswith("--limit-bytes=") for value in arguments), (
            "retention proof must use a byte-bounded log prefix"
        )
        if self.history_error:
            raise RegionalFixtureError("history refused")
        self.histories += 1
        return (
            self.history_after
            if self.histories > 1 and self.history_after is not None
            else self.history
        )


def snapshot(
    regional: LogRegional, *, apps: tuple[str, ...] = (APP,)
) -> dict[str, Any]:
    return restart.log_write_snapshot(
        cast(Any, regional),
        plane="cpu",
        apps=apps,
        since=SINCE,
        workload_name="training-a",
    )


def test_log_write_snapshot_trusts_silence_only_with_proof() -> None:
    regional = LogRegional()
    logs = snapshot(regional)
    assert logs["verdict"] == "CLEAN", logs
    assert logs["silent"] == [f"{APP}/pod-a"]
    assert logs["inconclusive"] == [] and logs["suspicious"] == []
    entry = logs["entries"][0]
    assert entry["classification"] == "silent"
    proof = entry["silence_evidence"]
    assert proof["pod_uid"] == "uid-a"
    assert proof["container_id"] == "containerd://instance-a"
    assert proof["history_tail_lines"] == 1
    assert proof["history_prefix_bytes"] == len(regional.history.encode())
    assert len(proof["history_sha256"]) == 64
    assert regional.described == regional.histories == 2
    assert restart.log_write_errors(logs, "control-plane") == []


@pytest.mark.parametrize(
    "drift",
    [
        "uid",
        "container-id",
        "restart-count",
        "started-during",
        "not-ready",
        "sidecar",
        "missing-uid",
        "invalid-time",
        "naive-time",
        "bool-restarts",
    ],
)
def test_missing_or_changed_container_identity_is_not_silence(drift: str) -> None:
    regional = LogRegional()
    status = regional.document["status"]["containerStatuses"][0]
    if drift in {"uid", "missing-uid"}:
        regional.document["metadata"]["uid"] = "foreign" if drift == "uid" else ""
    elif drift == "container-id":
        status["containerID"] = ""
    elif drift == "restart-count":
        del status["restartCount"]
    elif drift == "bool-restarts":
        status["restartCount"] = True
    elif drift == "not-ready":
        status["ready"] = False
    elif drift == "sidecar":
        regional.document["spec"]["containers"].append({"name": "sidecar"})
        regional.document["status"]["containerStatuses"].append(
            {**status, "name": "sidecar"}
        )
    else:
        status["state"]["running"]["startedAt"] = {
            "started-during": (SINCE + timedelta(seconds=30)).isoformat(),
            "invalid-time": "invalid",
            "naive-time": "2026-09-14T11:00:00",
        }[drift]
    logs = snapshot(regional)
    assert logs["verdict"] == "INCONCLUSIVE", logs
    assert logs["silent"] == []
    assert restart.log_write_errors(logs, "control-plane"), (
        "container identity drift must prevent a clean log verdict"
    )
    if drift == "started-during":
        assert "inside the window" in logs["entries"][0]["reason"]


@pytest.mark.parametrize(
    "history",
    [
        "",
        "\n",
        "unparsed historical output\n",
        "2026-09-14T11:55:00 no timezone\n",
        "2026-09-14T11:55:00+00:00",
        "2026-09-14T11:55:00+00:00 truncated",
        "2026-09-14T12:00:01+00:00 inside window\n",
        "2026-09-14T10:00:00+00:00 predates container\n",
        "2026-09-14T11:55:00+00:00 later\n2026-09-14T11:54:00+00:00 earlier\n",
        "x" * (restart.SILENCE_HISTORY_BYTES + 1),
    ],
)
def test_arbitrary_or_truncated_history_cannot_prove_retention(history: str) -> None:
    regional = LogRegional()
    regional.history = history
    logs = snapshot(regional)
    assert logs["verdict"] == "INCONCLUSIVE", logs
    assert logs["silent"] == []


@pytest.mark.parametrize("drift", ["uid", "container", "restarts", "history"])
def test_a_source_or_log_rotation_during_proof_is_inconclusive(drift: str) -> None:
    regional = LogRegional()
    regional.after = deepcopy(regional.document)
    status = regional.after["status"]["containerStatuses"][0]
    if drift == "uid":
        regional.after["metadata"]["uid"] = "recreated"
    elif drift == "container":
        status["containerID"] = "containerd://another"
    elif drift == "restarts":
        status["restartCount"] += 1
    else:
        regional.history_after = regional.history.replace("complete", "different")
    assert snapshot(regional)["verdict"] == "INCONCLUSIVE"


@pytest.mark.parametrize("failure", ["window_error", "describe_error", "history_error"])
def test_kubernetes_errors_do_not_become_clean_log_evidence(failure: str) -> None:
    regional = LogRegional()
    setattr(regional, failure, True)
    logs = snapshot(regional)
    assert logs["verdict"] == "INCONCLUSIVE", logs
    assert logs["silent"] == []


def test_one_checked_app_does_not_cover_an_absent_app() -> None:
    regional = LogRegional()
    regional.window = "ordinary log line\n"
    logs = snapshot(regional, apps=(APP, "gpu-fault-processor"))
    assert logs["verdict"] == "INCONCLUSIVE"
    assert "gpu-fault-processor/*" in logs["inconclusive"]


def test_suspicious_writes_override_silent_or_checked_entries() -> None:
    regional = LogRegional()
    regional.window = '{"verb":"patch","name":"training-a"}\n'
    logs = snapshot(regional)
    assert logs["verdict"] == "SUSPICIOUS"
    assert restart.log_write_errors(logs, "control-plane") == [
        "control-plane logs show a Kubernetes workload write"
    ]


@pytest.mark.parametrize(
    "logs", [{}, {"verdict": "UNKNOWN"}, {"verdict": "SUSPICIOUS"}]
)
def test_unusable_log_verdicts_are_not_accepted(logs: dict[str, Any]) -> None:
    assert restart.log_write_errors(logs, "control-plane"), (
        "missing, unknown or suspicious log verdicts must be rejected"
    )
