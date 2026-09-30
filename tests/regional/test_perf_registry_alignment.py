"""The perf registry teardown leaves the Secret as it found it and proves the
control plane agrees: Secret bytes restored from the recorded serialization,
Secret config digest equal to the durable head, no api-ha replica reporting
``regional_registry.secret_drift``."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.regional import (
    RegionalClusterRegistration,
    regional_registry_content_sha256,
)
from scripts.perf import regional_capacity_registry as registry
from scripts.perf import regional_capacity_suite as suite
from scripts.perf import regional_registry_alignment as alignment

NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)


def production_entry() -> dict[str, Any]:
    return {
        "cluster_id": "production",
        "region": "us-west-2",
        "hyperpod_cluster_name": "production",
        "eks_cluster_arn": "arn:aws:eks:us-west-2:000000000000:cluster/production",
        "token_sha256": "a" * 64,
        "allowed_namespaces": ["default", "training"],
        "agent_endpoint_allowed_cidrs": ["10.0.0.0/8"],
    }


def synthetic_entry() -> dict[str, Any]:
    return registry.synthetic_cluster_entry(
        "perf-cap-000",
        run_id="run-a",
        expires_at=NOW + timedelta(hours=1),
        token="t" * 48,
    )


def revision_for(entries: list[dict[str, Any]]) -> dict[str, Any]:
    registrations = [
        RegionalClusterRegistration.model_validate(item)
        for item in registry.redacted_registry_entries(entries)
    ]
    return {
        "generation": 4,
        "registrations": [item.model_dump(mode="json") for item in registrations],
        "content_sha256": regional_registry_content_sha256(registrations),
    }


class ControlPlane:
    """kubectl against the control plane: the registry Secret, the api-ha
    Pods, the durable snapshot and each replica's ``/healthz``."""

    def __init__(
        self,
        raw: bytes,
        *,
        revision: dict[str, Any],
        drift: dict[str, bool],
        created: dict[str, str] | None = None,
    ) -> None:
        self.raw = raw
        self.revision = revision
        self.drift = drift
        self.created = created or {}
        self.uid = "registry-uid"
        self.version = "7"
        self.patches: list[list[dict[str, Any]]] = []
        self.healthz_pods: list[str] = []
        self.rollouts: list[tuple[str, ...]] = []

    def __call__(self, *args: str, **kwargs: Any) -> str:
        if args[:2] == ("get", "secret"):
            data = base64.b64encode(self.raw).decode()
            if args[-1] != "json":
                return data
            return json.dumps(
                {
                    "metadata": {"uid": self.uid, "resourceVersion": self.version},
                    "data": {"clusters.json": data},
                }
            )
        if args[:2] == ("get", "pod"):
            if "-o" in args and args[args.index("-o") + 1] == "json":
                return json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {
                                    "name": name,
                                    "creationTimestamp": self.created.get(
                                        name, "2026-09-25T00:00:00Z"
                                    ),
                                },
                                "status": {"phase": "Running"},
                                "spec": {
                                    "containers": [{"ports": [{"containerPort": 8080}]}]
                                },
                            }
                            for name in sorted(self.drift)
                        ]
                        + [
                            {
                                "metadata": {"name": "api-ha-pending"},
                                "status": {"phase": "Pending"},
                                "spec": {"containers": [{}]},
                            }
                        ]
                    }
                )
            return "api-ha-0"
        if args[0] == "exec":
            if "-c" in args:
                pod = args[2]
                self.healthz_pods.append(pod)
                return json.dumps(
                    {
                        "http_status": 200,
                        "payload": {
                            "regional_registry": {
                                "ready": True,
                                "secret_drift": self.drift[pod],
                                "secret_config_sha256": "f" * 64,
                            }
                        },
                    }
                )
            return json.dumps(self.revision)
        if args[0] == "rollout":
            self.rollouts.append(args)
            if args[1] == "restart":
                # Every replica restarts on the restored Secret: no drift left.
                self.drift = {name: False for name in self.drift}
            return "rolled"
        assert args[:2] == ("patch", "secret"), f"unexpected kubectl {args[:2]}"
        patch = json.loads(kwargs["stdin"])
        self.patches.append(patch)
        self.raw = base64.b64decode(patch[2]["value"])
        self.version = str(int(self.version) + 1)
        return "patched"


def install(monkeypatch: pytest.MonkeyPatch, plane: ControlPlane) -> None:
    def registry_api(method: str, path: str, payload: Any = None) -> dict[str, Any]:
        if method == "POST":
            pytest.fail("a restore of unchanged durable content publishes nothing")
        assert path == "/v1/regional/registry/status", path
        return {
            "generation": plane.revision["generation"],
            "content_sha256": plane.revision["content_sha256"],
            "converged": True,
        }

    monkeypatch.setattr(registry, "control", plane)
    monkeypatch.setattr(registry, "registry_api", registry_api)
    monkeypatch.setattr(registry.time, "sleep", lambda _seconds: None)


def test_secret_baseline_records_the_serialization_but_never_a_token(
    tmp_path: Path,
) -> None:
    entries = [{**production_entry(), "token": "plaintext-secret"}]
    raw = (json.dumps(entries, indent=2) + "\n").encode()

    record = registry.record_secret_baseline(tmp_path, raw, entries)

    saved = json.loads((tmp_path / alignment.SECRET_BASELINE_FILE).read_text())
    assert saved == record, "the returned record is the file's content"
    assert saved["clusters_json_sha256"] == hashlib.sha256(raw).hexdigest()
    assert saved["serialization"] == {
        "indent": 2,
        # json.dumps switches to "," between items once indent is set.
        "separators": [",", ": "],
        "sort_keys": False,
        "trailing_newline": True,
    }, "the exact json.dumps shape is what lets teardown rebuild the bytes"
    assert (
        "plaintext-secret"
        not in (tmp_path / alignment.SECRET_BASELINE_FILE).read_text()
    ), "the baseline file must hold digests and shape only"
    assert alignment.serialize_entries(entries, saved["serialization"]).encode() == raw


def test_unknown_serialization_is_recorded_as_null(tmp_path: Path) -> None:
    entries = [production_entry()]
    raw = b'  [ {"cluster_id": "production"} ]  '

    record = registry.record_secret_baseline(
        tmp_path, raw, [{"cluster_id": "production"}]
    )

    assert record["serialization"] is None, "an unrecognised shape is not guessed"
    assert registry.baseline_serialization(tmp_path, entries) is None


def test_cleanup_restores_the_secret_bytes_recorded_at_register(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = [production_entry()]
    original = (json.dumps(baseline, indent=1) + "\n").encode()
    registry.record_secret_baseline(tmp_path, original, baseline)
    during_run = json.dumps(baseline + [synthetic_entry()], separators=(",", ":"))
    plane = ControlPlane(
        during_run.encode(), revision=revision_for(baseline), drift={"api-ha-0": False}
    )
    install(monkeypatch, plane)

    removed = registry.cleanup_registry_residuals(
        scope="live", artifacts=tmp_path, phase="postflight", force=True, run_id="run-a"
    )

    assert removed == 1, "the run's own synthetic entry is what cleanup removes"
    assert plane.raw == original, "the Secret must return to its pre-run bytes"
    restore = json.loads((tmp_path / "registry-secret-restore.json").read_text())
    assert restore["byte_identical"] is True
    assert restore["sha256_after"] == hashlib.sha256(original).hexdigest()


def test_cleanup_without_a_matching_baseline_writes_compact_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = [production_entry()]
    changed = [{**production_entry(), "allowed_namespaces": ["default"]}]
    registry.record_secret_baseline(
        tmp_path, (json.dumps(baseline, indent=1) + "\n").encode(), baseline
    )
    plane = ControlPlane(
        json.dumps(changed + [synthetic_entry()]).encode(),
        revision=revision_for(changed),
        drift={"api-ha-0": False},
    )
    install(monkeypatch, plane)

    registry.cleanup_registry_residuals(
        scope="live", artifacts=tmp_path, phase="postflight", force=True, run_id="run-a"
    )

    assert plane.raw == json.dumps(changed, separators=(",", ":")).encode(), (
        "content another writer changed is restored as content, not as old bytes"
    )
    restore = json.loads((tmp_path / "registry-secret-restore.json").read_text())
    assert restore["byte_identical"] is False


def test_write_registry_refuses_a_serialization_of_other_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = [production_entry()]
    plane = ControlPlane(
        json.dumps(baseline).encode(),
        revision=revision_for(baseline),
        drift={"api-ha-0": False},
    )
    install(monkeypatch, plane)

    with pytest.raises(RuntimeError, match="serialization"):
        registry.write_registry(
            baseline,
            run_id="run-a",
            expected_entries=baseline,
            reason="test",
            serialization=json.dumps([{"cluster_id": "other"}]),
        )
    assert plane.patches == [], "nothing is written when the text lies"


def test_alignment_passes_when_secret_head_and_every_replica_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = [production_entry()]
    plane = ControlPlane(
        json.dumps(baseline).encode(),
        revision=revision_for(baseline),
        drift={"api-ha-0": False, "api-ha-1": False},
    )
    install(monkeypatch, plane)

    report = registry.verify_registry_alignment(artifacts=tmp_path, phase="postflight")

    assert report["aligned"] is True
    assert report["secret_config_sha256"] == report["durable_config_sha256"]
    assert [item["pod"] for item in report["replicas"]] == ["api-ha-0", "api-ha-1"], (
        "every Running api-ha replica is asked; the Pending one is skipped"
    )
    assert plane.healthz_pods == ["api-ha-0", "api-ha-1"]
    saved = json.loads((tmp_path / "registry-alignment-postflight.json").read_text())
    assert saved["aligned"] is True


def test_alignment_names_the_drifted_field_and_replica(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_side = [production_entry()]
    head_side = [{**production_entry(), "token_sha256": "b" * 64}]
    plane = ControlPlane(
        json.dumps(secret_side).encode(),
        revision=revision_for(head_side),
        drift={"api-ha-0": True, "api-ha-1": False},
    )
    install(monkeypatch, plane)

    with pytest.raises(RuntimeError) as failure:
        registry.verify_registry_alignment(artifacts=tmp_path, phase="postflight")

    message = str(failure.value)
    assert "postflight" in message
    assert "production: token_sha256" in message, "the differing field is named"
    assert "api-ha-0" in message and "secret_drift=True" in message
    assert "api-ha-1" not in message, "an aligned replica is not an error"
    saved = json.loads((tmp_path / "registry-alignment-postflight.json").read_text())
    assert saved["aligned"] is False
    assert saved["field_differences"] == {"production": ["token_sha256"]}


def test_alignment_reports_a_head_only_or_secret_only_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_side = [production_entry()]
    head_side = [
        production_entry(),
        {**production_entry(), "cluster_id": "extra", "hyperpod_cluster_name": "extra"},
    ]
    plane = ControlPlane(
        json.dumps(secret_side).encode(),
        revision=revision_for(head_side),
        drift={"api-ha-0": False},
    )
    install(monkeypatch, plane)

    with pytest.raises(RuntimeError, match="extra: only in the durable head"):
        registry.verify_registry_alignment(artifacts=None, phase="preflight")


def test_postflight_rolls_replicas_that_started_inside_the_run_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live 2026-09-25 HA-005: the case rolled api-ha mid-run, the new replicas
    started on the run's transient Secret and reported secret_drift after the
    byte-identical restore although Secret and head agreed. The postflight
    reconciles exactly those replicas with a rolling restart."""

    baseline = [production_entry()]
    (tmp_path / "registry-alignment-preflight.json").write_text(
        json.dumps({"phase": "preflight", "checked_at": "2026-09-25T11:29:00+00:00"})
    )
    plane = ControlPlane(
        json.dumps(baseline).encode(),
        revision=revision_for(baseline),
        drift={"api-ha-0": True, "api-ha-1": True},
        created={
            "api-ha-0": "2026-09-25T11:31:39Z",
            "api-ha-1": "2026-09-25T11:31:45Z",
        },
    )
    install(monkeypatch, plane)

    report = registry.verify_registry_alignment(artifacts=tmp_path, phase="postflight")

    assert report["aligned"] is True, report["errors"]
    assert report["restarted_replicas"] == ["api-ha-0", "api-ha-1"]
    assert plane.rollouts == [
        ("rollout", "restart", "deployment/gpu-fault-api-ha"),
        ("rollout", "status", "deployment/gpu-fault-api-ha", "--timeout=600s"),
    ]
    assert all(item["secret_drift"] is False for item in report["replicas"]), report
    assert "checked_at" in report, "the postflight records when it ran"


@pytest.mark.parametrize("reason", ["predates-window", "no-preflight", "head-differs"])
def test_postflight_never_restarts_away_a_drift_it_cannot_attribute_to_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    secret_side = [production_entry()]
    head_side = (
        [{**production_entry(), "token_sha256": "b" * 64}]
        if reason == "head-differs"
        else secret_side
    )
    if reason != "no-preflight":
        (tmp_path / "registry-alignment-preflight.json").write_text(
            json.dumps(
                {"phase": "preflight", "checked_at": "2026-09-25T11:29:00+00:00"}
            )
        )
    created = {
        "api-ha-0": (
            "2026-09-25T10:00:00Z"
            if reason == "predates-window"
            else "2026-09-25T11:31:39Z"
        )
    }
    plane = ControlPlane(
        json.dumps(secret_side).encode(),
        revision=revision_for(head_side),
        drift={"api-ha-0": True},
        created=created,
    )
    install(monkeypatch, plane)

    with pytest.raises(RuntimeError, match="secret_drift=True"):
        registry.verify_registry_alignment(artifacts=tmp_path, phase="postflight")

    assert plane.rollouts == [], "no rolling restart for a drift the run did not cause"


def test_register_and_deregister_run_the_alignment_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.perf import regional_capacity_data

    phases: list[str] = []
    baseline = [production_entry()]
    monkeypatch.setattr(
        registry, "validate_registry_target", lambda **_kwargs: "isolated"
    )
    monkeypatch.setattr(registry, "validate_notification_safety", lambda: None)
    monkeypatch.setattr(
        registry, "validate_alertmanager_drill_route", lambda: {"stub": True}
    )
    monkeypatch.setattr(registry, "cleanup_registry_residuals", lambda **_kwargs: 0)
    monkeypatch.setattr(registry, "load_registry", lambda: list(baseline))
    monkeypatch.setattr(
        registry,
        "capture_secret_baseline",
        lambda artifacts, entries: phases.append(f"baseline:{len(entries)}"),
    )
    monkeypatch.setattr(regional_capacity_data, "invoke", lambda *a, **k: {"total": 0})
    monkeypatch.setattr(registry, "write_registry", lambda *a, **k: None)
    monkeypatch.setattr(
        registry, "sync_dataplane_connection_secret", lambda: {"changed": False}
    )
    monkeypatch.setattr(
        registry,
        "upsert_secret",
        lambda name, files, *, run_id: {
            "run_id": run_id,
            "uid": "u",
            "resource_version": "1",
        },
    )

    def verify(*, artifacts: Path | None, phase: str) -> dict[str, Any]:
        phases.append(phase)
        if phase == "postflight":
            raise RuntimeError("registry alignment (postflight) failed: drift")
        return {"aligned": True}

    monkeypatch.setattr(registry, "verify_registry_alignment", verify)

    registry.register(
        1,
        tmp_path,
        run_id="run-a",
        expires_at=NOW + timedelta(hours=1),
        allow_live_registry=False,
        live_registry_confirmation=None,
    )
    assert phases == ["preflight", "baseline:1"], (
        "a drifted registry is refused before the run touches it, and the "
        "Secret baseline is captured before the run's own write"
    )

    with pytest.raises(RuntimeError, match="postflight"):
        registry.deregister(scope="isolated", artifacts=tmp_path, run_id="run-a")
    assert phases[-1] == "postflight", "teardown fails the run's cleanup on drift"


@pytest.mark.parametrize(
    ("state_document", "expected"),
    [
        (json.dumps({"release_id": "release-a", "phase": "complete"}), "release-a"),
        ("", ""),
        ("not-json", ""),
        (json.dumps({"phase": "complete"}), ""),
        (json.dumps(["release-a"]), ""),
    ],
)
def test_release_state_release_id_reads_the_release_state_configmap(
    monkeypatch: pytest.MonkeyPatch, state_document: str, expected: str
) -> None:
    calls: list[tuple[str, ...]] = []

    def control(*args: str, **_kwargs: Any) -> str:
        calls.append(args)
        return state_document + "\n"

    monkeypatch.setattr(registry, "control", control)

    assert registry.release_state_release_id() == expected
    assert len(calls) == 1, calls
    assert calls[0][:3] == ("get", "configmap", "gpu-fault-regional-release-state")
    assert "GPU_FAULT_RELEASE_ID" not in " ".join(calls[0]), (
        "the CPU Deployments never carry GPU_FAULT_RELEASE_ID; do not read it"
    )


@pytest.mark.parametrize("artifact_volume", [True, False])
def test_capacity_release_id_prefers_the_artifact_volume_then_the_release_state(
    monkeypatch: pytest.MonkeyPatch, artifact_volume: bool
) -> None:
    calls: list[tuple[str, ...]] = []

    def control(*args: str, **_kwargs: Any) -> str:
        calls.append(args)
        if args[1] == "deploy":
            return "gpu-fault-control-plane-wheel-0100\n" if artifact_volume else "\n"
        return json.dumps({"release_id": "release-a"}) + "\n"

    # The suite binds the registry's ``control`` at import; patch both names.
    monkeypatch.setattr(suite, "control", control)
    monkeypatch.setattr(registry, "control", control)

    assert suite.release_id() == ("0100" if artifact_volume else "release-a")
    assert len(calls) == (1 if artifact_volume else 2), calls
