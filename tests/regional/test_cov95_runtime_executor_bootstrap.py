from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from gpu_fault.cluster_executor import bootstrap, regional_client
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize(
    "marker",
    [
        "missing",
        "invalid-json",
        "list",
        "empty",
        "invalid-time",
        "naive-time",
        "future",
        "valid",
    ],
)
def test_readiness_keeps_authentication_when_claim_marker_is_missing_or_malformed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, marker: str
) -> None:
    now = datetime.now(timezone.utc)
    path = tmp_path / "claim.json"
    document: Any = {
        "executor_id": "recorded-executor",
        "execution_owners": ["node-owner", 17, None],
        "last_successful_claim_at": (now - timedelta(seconds=5)).isoformat(),
    }
    if marker == "list":
        document = []
    elif marker == "empty":
        document = {}
    elif marker == "invalid-time":
        document["last_successful_claim_at"] = "not-a-timestamp"
    elif marker == "naive-time":
        document["last_successful_claim_at"] = now.replace(tzinfo=None).isoformat()
    elif marker == "future":
        document["last_successful_claim_at"] = (now + timedelta(seconds=10)).isoformat()
    if marker != "missing":
        path.write_text(
            "{" if marker == "invalid-json" else json.dumps(document), encoding="ascii"
        )
    monkeypatch.setenv("GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH", str(path))
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "cluster-a")
    monkeypatch.setenv("GPU_FAULT_CLUSTER_EXECUTOR_ID", "configured-executor")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://control.invalid")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_TOKEN", "synthetic-cluster-a-token")
    calls = []

    def send(request: Any, **kwargs: Any) -> io.BytesIO:
        calls.append((request, kwargs))
        return io.BytesIO(b'{"ready":true}')

    monkeypatch.setattr(regional_client, "urlopen", send)
    assert bootstrap.readiness_probe() == 0
    assert len(calls) == 1
    request, options = calls[0]
    assert request.get_header("Authorization") == "Bearer synthetic-cluster-a-token"
    assert request.get_header("X-gpu-fault-cluster-id") == "cluster-a"
    payload = json.loads(request.data)
    recorded = marker in {"invalid-time", "naive-time", "future", "valid"}
    assert payload["executor_id"] == (
        "recorded-executor" if recorded else "configured-executor"
    )
    assert payload["execution_owners"] == (["node-owner"] if recorded else [])
    age = payload["last_successful_claim_age_seconds"]
    if marker == "valid":
        assert 5 <= age < 30
    elif marker == "future":
        assert age == 0.0
    else:
        assert age is None
    assert options["timeout"] == 8.0


@pytest.mark.parametrize("reply", ["refused", "no-reasons", "unanswered", "forbidden"])
def test_readiness_refusal_or_transport_failure_does_not_become_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reply: str
) -> None:
    monkeypatch.setenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH", str(tmp_path / "missing.json")
    )
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "cluster-a")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://control.invalid")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_TOKEN", "synthetic-cluster-a-token")
    requests = []

    def send(request: Any, **kwargs: Any) -> io.BytesIO:
        requests.append(request)
        if reply == "unanswered":
            raise URLError("synthetic offline transport")
        if reply == "forbidden":
            raise HTTPError(request.full_url, 403, "forbidden", {}, io.BytesIO(b""))
        return io.BytesIO(
            json.dumps(
                {"ready": False, "reasons": ["registry revoked"]}
                if reply == "refused"
                else {}
            ).encode()
        )

    monkeypatch.setattr(regional_client, "urlopen", send)
    assert bootstrap.readiness_probe() == 1
    assert len(requests) == 1
