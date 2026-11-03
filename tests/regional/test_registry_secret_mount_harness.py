"""The harness reads ``secret_drift`` the way the runtime now means it (batch 245).

Two acceptance surfaces compare the registry Secret with the durable head:

* the BOOT-023 in-Pod probe computes the Secret's digest itself and must use
  the copy the runtime compares -- the mounted ``clusters.json`` where the Pod
  mounts it, the start-up env otherwise -- or a join since the Pod started would
  fail the case although ``/healthz`` correctly says no drift;
* the perf registry-alignment gate runs right after its own teardown restored
  the Secret. A replica that reads the mounted Secret sees the restore only
  after kubelet re-projects the volume, so it gets a bounded wait instead of a
  failure; a replica comparing its start-up copy keeps the old rules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.regional_registry import (
    REGIONAL_CLUSTERS_SECRET_FILE,
    configured_regional_registrations,
    regional_registry_config_sha256,
)
from scripts.e2e.regional import boot023_verdicts
from scripts.e2e.regional.probes import boot023_registry_probe as probe
from scripts.perf import regional_registry_alignment as alignment
from tests.regional._regional_support import TOKEN_A, TOKEN_B, registration

LEAKED_TOKEN = "plaintext-token-" + "y" * 24


def _entry(cluster_id: str, token: str) -> dict[str, Any]:
    item = registration(cluster_id, token).model_dump(mode="json")
    for field in ("token_sha256", "created_at", "updated_at", "lifecycle_state"):
        item.pop(field)
    return {**item, "token": token}


def _digest(entries: list[dict[str, Any]]) -> str:
    return regional_registry_config_sha256(configured_regional_registrations(entries))


# --------------------------------------------------------------------------- #
# BOOT-023 probe
# --------------------------------------------------------------------------- #
class _Response:
    status = 200

    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def _run_probe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    startup: list[dict[str, Any]],
    head: list[dict[str, Any]],
    healthz: dict[str, Any],
) -> dict[str, Any]:
    from gpu_fault.app import ApplicationContext

    registrations = configured_regional_registrations(head)
    context = SimpleNamespace(
        regional_registry_secret_sha256=_digest(startup),
        store=SimpleNamespace(
            get_regional_registry_head=lambda: SimpleNamespace(
                generation=13, content_sha256="c" * 64
            ),
            get_regional_registry_revision=lambda generation: SimpleNamespace(
                registrations=registrations
            ),
        ),
    )
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )
    monkeypatch.setattr(probe.socket, "gethostname", lambda: "api-ha-0")
    body = json.dumps(healthz).encode()
    monkeypatch.setattr(probe, "urlopen", lambda *args, **kwargs: _Response(body))
    monkeypatch.setattr(sys, "argv", ["registry-probe", "8090"])
    assert probe.main() == 0
    return json.loads(capsys.readouterr().out)


def test_the_probe_reads_the_same_file_the_runtime_compares() -> None:
    assert probe.MOUNTED_SECRET_FILE == REGIONAL_CLUSTERS_SECRET_FILE


def test_boot023_passes_a_pod_that_started_before_a_join(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    startup = [_entry("cluster-a", TOKEN_A)]
    joined = [*startup, _entry("cluster-b", TOKEN_B)]
    mounted = tmp_path / "clusters.json"
    mounted.write_text(json.dumps(joined), encoding="utf-8")
    monkeypatch.setattr(probe, "MOUNTED_SECRET_FILE", str(mounted))
    registry = {
        "ready": True,
        "secret_drift": False,
        "secret_source": "mounted-file",
        "secret_config_sha256": _digest(joined),
    }

    report = _run_probe(
        monkeypatch,
        capsys,
        startup=startup,
        head=joined,
        healthz={"regional_registry": registry},
    )

    assert report["secret_source"] == "mounted-file"
    assert report["secret_config_sha256"] == report["durable_config_sha256"]
    assert report["secret_startup_config_sha256"] == _digest(startup)
    assert boot023_verdicts.registry_probe_errors([{**report, "pod": "api-ha-0"}]) == []


def test_boot023_without_a_mount_keeps_the_start_up_comparison(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A release that does not mount the Secret yet: the old verdict holds, and
    a Pod that predates a join still fails it (restart it, then rerun)."""

    monkeypatch.setattr(probe, "MOUNTED_SECRET_FILE", str(tmp_path / "absent"))
    startup = [_entry("cluster-a", TOKEN_A)]
    joined = [*startup, _entry("cluster-b", TOKEN_B)]
    registry = {
        "ready": True,
        "secret_drift": True,
        "secret_config_sha256": _digest(startup),
    }

    report = _run_probe(
        monkeypatch,
        capsys,
        startup=startup,
        head=joined,
        healthz={"regional_registry": registry},
    )

    assert report["secret_source"] == "startup-environment"
    assert report["secret_config_sha256"] == _digest(startup)
    errors = boot023_verdicts.registry_probe_errors([{**report, "pod": "api-ha-0"}])
    assert any("did not publish durably" in item for item in errors), errors
    assert any("secret_drift=True" in item for item in errors), errors


def test_the_probe_reports_a_mounted_secret_that_does_not_load_without_it(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "clusters.json"
    broken.write_text('[{"token": "' + LEAKED_TOKEN + '"', encoding="utf-8")

    result = probe.mounted_secret_sha256(str(broken))

    assert result == {"sha256": None, "error": "mounted registry Secret does not load"}
    assert probe.mounted_secret_sha256(str(tmp_path / "absent")) is None


# --------------------------------------------------------------------------- #
# perf registry alignment
# --------------------------------------------------------------------------- #
class _Plane:
    """kubectl against api-ha: Pods and each replica's ``/healthz``; replicas
    reading the mounted Secret converge after ``converge_after`` reads."""

    def __init__(self, sources: dict[str, str | None], converge_after: int) -> None:
        self.sources = sources
        self.converge_after = converge_after
        self.reads: dict[str, int] = {pod: 0 for pod in sources}
        self.rollouts: list[tuple[str, ...]] = []

    def __call__(self, *args: str, **kwargs: Any) -> str:
        if args[:2] == ("get", "pod"):
            return json.dumps(
                {
                    "items": [
                        {
                            "metadata": {
                                "name": pod,
                                "creationTimestamp": "2026-11-03T10:00:00Z",
                            },
                            "status": {"phase": "Running"},
                            "spec": {
                                "containers": [
                                    {"ports": [{"name": "http", "containerPort": 8080}]}
                                ]
                            },
                        }
                        for pod in sorted(self.sources)
                    ]
                }
            )
        if args[0] == "exec":
            pod = args[2]
            self.reads[pod] += 1
            source = self.sources[pod]
            converged = (
                source == "mounted-file" and self.reads[pod] > self.converge_after
            )
            registry = {
                "ready": True,
                "secret_drift": not converged,
                "secret_config_sha256": "f" * 64,
            }
            if source is not None:
                registry["secret_source"] = source
            return json.dumps(
                {"http_status": 200, "payload": {"regional_registry": registry}}
            )
        assert args[0] == "rollout", args
        self.rollouts.append(args)
        return "rolled"


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


def _verify(plane: _Plane, clock: _Clock, artifacts: Path) -> dict[str, Any]:
    entries = [_entry("cluster-a", TOKEN_A)]
    head = [
        item.model_dump(mode="json")
        for item in configured_regional_registrations(entries)
    ]
    return alignment.verify_alignment(
        plane,
        secret_document=lambda: ({}, json.dumps(entries).encode(), entries),
        durable_revision=lambda: {"generation": 13, "registrations": head},
        artifacts=artifacts,
        phase="postflight",
        # The preflight ran after these replicas started: none is in-window.
        restart_window_start="2026-11-03T11:00:00+00:00",
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )


def test_mounted_replicas_converge_inside_the_kubelet_window(tmp_path: Path) -> None:
    plane = _Plane({"api-ha-0": "mounted-file", "api-ha-1": "mounted-file"}, 3)
    clock = _Clock()

    report = _verify(plane, clock, tmp_path)

    assert report["aligned"] is True, report["errors"]
    assert report["mounted_convergence_seconds"] == 30.0
    assert report["restarted_replicas"] == [] and plane.rollouts == [], (
        "a replica that converges by itself is never restarted"
    )
    saved = json.loads((tmp_path / "registry-alignment-postflight.json").read_text())
    assert [item["secret_source"] for item in saved["replicas"]] == [
        "mounted-file",
        "mounted-file",
    ]


def test_a_mounted_replica_that_never_converges_fails_after_the_bound(
    tmp_path: Path,
) -> None:
    plane = _Plane({"api-ha-0": "mounted-file"}, converge_after=10**6)
    clock = _Clock()

    with pytest.raises(RuntimeError, match="api-ha-0 reports secret_drift=True"):
        _verify(plane, clock, tmp_path)

    assert clock.now >= alignment.MOUNTED_CONVERGENCE_SECONDS
    assert clock.now < alignment.MOUNTED_CONVERGENCE_SECONDS + 2 * (
        alignment.MOUNTED_CONVERGENCE_POLL_SECONDS
    )
    assert plane.rollouts == [], "created before the window: never restarted away"


@pytest.mark.parametrize("source", [None, "startup-environment"])
def test_a_start_up_replica_is_not_waited_for(
    tmp_path: Path, source: str | None
) -> None:
    """A release without the mount (no ``secret_source``) or a Pod without it
    cannot converge by itself: no wait, the existing rules decide at once."""

    plane = _Plane({"api-ha-0": source}, converge_after=0)
    clock = _Clock()

    with pytest.raises(RuntimeError, match="secret_drift=True"):
        _verify(plane, clock, tmp_path)

    assert clock.sleeps == []
