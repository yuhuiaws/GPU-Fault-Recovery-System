from __future__ import annotations

import base64
import copy
import hashlib
import json
import socket
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.nvidia_logs import FabricManagerLogEvent
from gpu_fault.regional import (
    RegionalClusterLifecycle,
    RegionalClusterRegistration,
    regional_registry_content_sha256,
)
from gpu_fault.training_models import TrainingProgressHeartbeat
from gpu_fault.watcher import AttemptObservation
from scripts.perf import benchmark_synchronized_burst as burst
from scripts.perf import regional_capacity_registry as registry_module
from scripts.perf import regional_capacity_suite as suite

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)


def test_drain_targets_preserve_clean_environment_threshold() -> None:
    metrics = {
        "pod-a": (
            "gpu_fault_processor_queue_depth 0\ngpu_fault_telemetry_spool_depth 1\n"
        )
    }

    assert suite.drain_targets(metrics) == (1.0, 1.0)


def test_drain_targets_allow_online_steady_state() -> None:
    metrics = {
        "pod-a": (
            "gpu_fault_processor_queue_depth 8\ngpu_fault_telemetry_spool_depth 0\n"
        ),
        "pod-b": (
            "gpu_fault_processor_queue_depth 10\ngpu_fault_telemetry_spool_depth 2\n"
        ),
    }

    assert suite.drain_targets(metrics) == (14.0, 6.0)


def test_artifact_directory_uses_case_release_and_utc(tmp_path: Path) -> None:
    path = suite.artifact_dir(tmp_path, "burst 32c", "release/abc123")

    assert path.parent.parent.name == "burst-32c"
    assert path.parent.name == "release-abc123"
    assert path.name.endswith("Z"), (
        f"capacity artifact timestamp must be UTC: {path.name}"
    )


def test_status_and_aborted_directory_are_explicit(tmp_path: Path) -> None:
    path = suite.artifact_dir(tmp_path, "burst", "abc123")
    suite.write_status(path, status="aborted", reason="probe failed")

    status = json.loads((path / "status.json").read_text())
    assert status["status"] == "aborted"
    assert status["reason"] == "probe failed"

    moved = suite.move_to_aborted(tmp_path, path)
    assert moved.is_dir(), (
        f"aborted capacity artifact directory was not preserved: {moved}"
    )
    assert moved.relative_to(tmp_path).parts[0] == "_aborted"


def test_pod_log_collection_is_parallel_and_index_ordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_dataplane(*arguments, **_kwargs):
        nonlocal active, max_active
        if arguments[0] == "get":
            return "pod-b 1\npod-a 0\n"
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return json.dumps({"pod": arguments[1]})

    monkeypatch.setattr(suite, "dataplane", fake_dataplane)

    documents = suite.collect_logs("job-a", tmp_path)

    assert max_active == 2
    assert [item["pod"] for item in documents] == ["pod-a", "pod-b"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "0-pod-a.log",
        "1-pod-b.log",
    ]


def test_dataplane_context_must_be_explicit(monkeypatch) -> None:
    monkeypatch.setattr(suite, "DATAPLANE_CONTEXT", "")
    monkeypatch.setattr(sys, "argv", ["regional-capacity", "register"])

    with pytest.raises(SystemExit, match="2"):
        suite.main()


def test_capacity_run_refuses_drill_email_delivery(monkeypatch) -> None:
    monkeypatch.setattr(registry_module, "control_pods", lambda: ["pod-a", "pod-b"])

    def fake_control(*args, **_kwargs):
        pod = args[1]
        return "true\n" if pod == "pod-b" else "false\n"

    monkeypatch.setattr(registry_module, "control", fake_control)

    with pytest.raises(
        RuntimeError, match="capacity runs must not deliver drill notifications"
    ):
        registry_module.validate_notification_safety()


def test_capacity_run_allows_suppressed_drills(monkeypatch) -> None:
    monkeypatch.setattr(registry_module, "control_pods", lambda: ["pod-a", "pod-b"])
    monkeypatch.setattr(registry_module, "control", lambda *_args, **_kwargs: "false\n")

    registry_module.validate_notification_safety()


def test_synthetic_registry_uses_placeholder_aws_account() -> None:
    entry = registry_module.perf_cluster_entries(
        1, run_id="run-a", expires_at=NOW + timedelta(days=1)
    )[0]

    assert ":000000000000:cluster/" in entry["eks_cluster_arn"]
    assert entry["synthetic"] is True
    assert entry["synthetic_run_id"] == "run-a"
    assert entry["synthetic_expires_at"] == (NOW + timedelta(days=1)).isoformat()
    assert entry["agent_endpoint_allowed_cidrs"] == ["127.0.0.1/32"]


def test_synthetic_registry_metadata_passes_runtime_model() -> None:
    observed_at = datetime.now(timezone.utc)
    entry = registry_module.perf_cluster_entries(
        1, run_id="run-a", expires_at=observed_at + timedelta(days=1)
    )[0]
    token = entry.pop("token")
    entry["token_sha256"] = hashlib.sha256(token.encode()).hexdigest()

    registration = RegionalClusterRegistration(**entry)

    assert registration.is_active(observed_at), (
        "synthetic registration was inactive before TTL"
    )
    assert registration.authenticates(token), (
        "active synthetic registration rejected its token"
    )


def test_live_registry_requires_double_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(
        registry_module,
        "CONTROL_NAMESPACE",
        registry_module.PRODUCTION_CONTROL_NAMESPACE,
    )

    with pytest.raises(RuntimeError, match="production control plane"):
        registry_module.validate_registry_target(
            allow_live_registry=False, confirmation=None
        )

    assert (
        registry_module.validate_registry_target(
            allow_live_registry=True,
            confirmation=registry_module.LIVE_REGISTRY_CONFIRMATION,
        )
        == "live"
    )


def test_preflight_cleanup_refuses_legacy_synthetic_entries(
    tmp_path: Path, monkeypatch
) -> None:
    entries = [{"cluster_id": "production"}, {"cluster_id": "perf-cap-000"}]
    published = []
    monkeypatch.setattr(registry_module, "load_registry", lambda: list(entries))
    monkeypatch.setattr(
        registry_module,
        "write_registry",
        lambda values: entries.__setitem__(slice(None), values),
    )
    monkeypatch.setattr(
        registry_module,
        "publish_registry_revision",
        lambda values, **kwargs: published.append((list(values), kwargs)),
    )

    with pytest.raises(RuntimeError, match="preflight cannot purge"):
        registry_module.cleanup_registry_residuals(
            scope="isolated", artifacts=tmp_path, phase="preflight", force=False
        )

    assert entries == [{"cluster_id": "production"}, {"cluster_id": "perf-cap-000"}]
    assert published == []
    assert not (tmp_path / "registry-preflight.json").exists(), (
        "refused legacy cleanup must not write a successful preflight receipt"
    )


def test_register_publishes_one_online_revision(tmp_path: Path, monkeypatch) -> None:
    from scripts.perf import regional_capacity_data

    baseline = [{"cluster_id": "production", "token": "p" * 32}]
    written = []
    published = []
    token_secrets = []
    monkeypatch.setattr(
        registry_module, "validate_registry_target", lambda **_kwargs: "isolated"
    )
    monkeypatch.setattr(registry_module, "validate_notification_safety", lambda: None)
    monkeypatch.setattr(
        registry_module, "validate_alertmanager_drill_route", lambda: {"stub": True}
    )
    monkeypatch.setattr(
        registry_module, "cleanup_registry_residuals", lambda **_kwargs: 0
    )
    monkeypatch.setattr(registry_module, "load_registry", lambda: list(baseline))
    # The alignment gate and the Secret baseline are proven by their own tests.
    monkeypatch.setattr(
        registry_module, "verify_registry_alignment", lambda **_kwargs: {}
    )
    monkeypatch.setattr(registry_module, "capture_secret_baseline", lambda *_args: None)
    monkeypatch.setattr(regional_capacity_data, "invoke", lambda *a, **k: {"total": 0})

    def write_registry(values, **kwargs):
        assert token_secrets, "token creation must precede registry publication"
        assert kwargs["expected_entries"] == baseline
        assert kwargs["run_id"] == "run-a"
        written.append(list(values))
        published.append((list(values), kwargs["reason"]))

    monkeypatch.setattr(registry_module, "write_registry", write_registry)
    monkeypatch.setattr(
        registry_module,
        "sync_dataplane_connection_secret",
        lambda: {"changed": False},  # the mirror is exercised by its own tests
    )
    monkeypatch.setattr(
        registry_module,
        "publish_registry_revision",
        lambda *a, **k: pytest.fail("write_registry already owns the CAS publication"),
    )

    def upsert_secret(name, files, *, run_id):
        intent = json.loads(
            (tmp_path / "registry-registration-intent.json").read_text()
        )
        assert intent["run_id"] == run_id
        assert intent["data_empty_before_registration"] is True
        token_secrets.append((name, files))
        return {"run_id": run_id, "uid": "token-uid", "resource_version": "1"}

    monkeypatch.setattr(registry_module, "upsert_secret", upsert_secret)

    tokens = registry_module.register(
        2,
        tmp_path,
        run_id="run-a",
        expires_at=NOW + timedelta(hours=1),
        allow_live_registry=False,
        live_registry_confirmation=None,
    )

    assert len(tokens) == 2
    assert len(written) == 1
    assert len(published) == 1
    assert published[0][1] == "capacity register run-a"
    assert [item["cluster_id"] for item in published[0][0][-2:]] == [
        "perf-cap-000",
        "perf-cap-001",
    ]
    assert token_secrets[0][0] == registry_module.TOKEN_SECRET
    proof = json.loads((tmp_path / "registry-token-proof.json").read_text())
    assert proof == {"run_id": "run-a", "uid": "token-uid", "resource_version": "1"}


@pytest.mark.parametrize("failure", [None, "create-ack", "replaced"])
def test_token_secret_is_created_labelled_and_never_adopts_a_replacement(
    monkeypatch, failure: str | None
) -> None:
    resources = []
    creates = []

    def dataplane(*args, **kwargs):
        if args[0] == "get":
            return json.dumps(resources[0]) if resources else ""
        assert args[:3] == ("create", "-f", "-")
        item = json.loads(kwargs["stdin"])
        assert item["metadata"]["labels"] == {registry_module.RUN_LABEL: "run-a"}
        item["metadata"].update(uid="token-uid", resourceVersion="1")
        resources.append(item)
        creates.append(item)
        if failure == "create-ack":
            raise TimeoutError("create acknowledgement lost")
        acknowledgement = json.dumps(item["metadata"])
        if failure == "replaced":
            item["metadata"]["uid"] = "replacement"
        return acknowledgement

    monkeypatch.setattr(registry_module, "dataplane", dataplane)
    if failure == "replaced":
        with pytest.raises(RuntimeError, match="ownership"):
            registry_module.upsert_secret(
                registry_module.TOKEN_SECRET, {"clusters.json": b"[]"}, run_id="run-a"
            )
    else:
        assert registry_module.upsert_secret(
            registry_module.TOKEN_SECRET, {"clusters.json": b"[]"}, run_id="run-a"
        ) == {"uid": "token-uid", "resource_version": "1", "run_id": "run-a"}
    assert len(creates) == 1


def test_token_secret_from_another_run_is_not_overwritten(monkeypatch) -> None:
    calls = []

    def dataplane(*args, **kwargs):
        calls.append(args)
        return json.dumps(
            {
                "metadata": {
                    "uid": "foreign",
                    "resourceVersion": "1",
                    "labels": {registry_module.RUN_LABEL: "other-run"},
                },
                "data": {"clusters.json": base64.b64encode(b"[]").decode()},
            }
        )

    monkeypatch.setattr(registry_module, "dataplane", dataplane)
    with pytest.raises(RuntimeError, match="ownership"):
        registry_module.upsert_secret(
            registry_module.TOKEN_SECRET, {"clusters.json": b"[]"}, run_id="run-a"
        )
    assert [args[0] for args in calls] == ["get"]


def registry_wire(monkeypatch, *, failure=None) -> SimpleNamespace:
    registration = RegionalClusterRegistration(
        cluster_id="production",
        region="us-west-2",
        hyperpod_cluster_name="production",
        eks_cluster_arn="arn:aws:eks:us-west-2:000000000000:cluster/production",
        token_sha256="a" * 64,
        agent_endpoint_allowed_cidrs=["127.0.0.1/32"],
        created_at=NOW,
        updated_at=NOW,
    )
    durable = registration.model_copy(
        update={"lifecycle_state": RegionalClusterLifecycle.DRAINING}
    )
    wire = SimpleNamespace(
        baseline=[registration.model_dump(mode="json")],
        entries=[registration.model_dump(mode="json")],
        uid="registry-uid",
        version="1",
        patches=[],
        publications=[],
        revision={
            "generation": 1,
            "registrations": [durable.model_dump(mode="json")],
            "content_sha256": regional_registry_content_sha256([durable]),
        },
    )

    def control(*args, **kwargs):
        if args[:2] == ("get", "secret"):
            data = base64.b64encode(json.dumps(wire.entries).encode()).decode()
            if args[-1] != "json":
                return data
            return json.dumps(
                {
                    "metadata": {"uid": wire.uid, "resourceVersion": wire.version},
                    "data": {"clusters.json": data},
                }
            )
        if args[:2] == ("get", "pod"):
            return "cpu-api"
        if args[0] == "exec":
            assert "/opt/gpu-fault/control-plane/bin/python" in args
            return json.dumps(wire.revision)
        assert args[:2] == ("patch", "secret")
        assert "--patch-file=/dev/stdin" in args and "-p" not in args
        patch = json.loads(kwargs["stdin"])
        assert patch[:2] == [
            {"op": "test", "path": "/metadata/uid", "value": wire.uid},
            {"op": "test", "path": "/metadata/resourceVersion", "value": wire.version},
        ]
        wire.patches.append(patch)
        wire.entries = json.loads(base64.b64decode(patch[2]["value"]))
        wire.version = str(int(wire.version) + 1)
        if failure == "secret-replaced":
            wire.uid = "replacement"
        if failure == "secret-ack":
            raise TimeoutError("Secret acknowledgement lost")
        return "patched"

    def registry_api(method, path, payload=None):
        if method == "POST":
            assert payload["expected_generation"] == wire.revision["generation"]
            wire.publications.append(payload)
            if failure == "publication-rejected":
                raise RuntimeError("publication rejected")
            values = [
                RegionalClusterRegistration.model_validate(item)
                for item in payload["registrations"]
            ]
            wire.revision = {
                "generation": wire.revision["generation"] + 1,
                "registrations": payload["registrations"],
                "content_sha256": regional_registry_content_sha256(values),
            }
            if failure == "publication-ack":
                raise TimeoutError("publication acknowledgement lost")
        status = {
            "generation": wire.revision["generation"],
            "content_sha256": wire.revision["content_sha256"],
            "converged": True,
        }
        if failure == "head-drift" and wire.publications:
            status["generation"] += 1
        return status

    monkeypatch.setattr(registry_module, "control", control)
    monkeypatch.setattr(registry_module, "registry_api", registry_api)
    return wire


def synthetic_registration() -> dict:
    return RegionalClusterRegistration(
        cluster_id="perf-cap-000",
        region="us-west-2",
        hyperpod_cluster_name="perf-cap-000",
        eks_cluster_arn="arn:aws:eks:us-west-2:000000000000:cluster/perf-cap-000",
        token_sha256="b" * 64,
        agent_endpoint_allowed_cidrs=["127.0.0.1/32"],
        synthetic=True,
        synthetic_run_id="run-a",
        synthetic_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    ).model_dump(mode="json")


@pytest.mark.parametrize("failure", [None, "secret-ack", "publication-ack"])
def test_registry_write_reconciles_ack_without_replaying_or_resetting_lifecycle(
    monkeypatch, failure: str | None
) -> None:
    wire = registry_wire(monkeypatch, failure=failure)
    original = copy.deepcopy(wire.revision["registrations"][0])
    result = registry_module.write_registry(
        wire.baseline + [synthetic_registration()],
        run_id="run-a",
        expected_entries=wire.baseline,
        reason="test registration",
    )
    assert result["generation"] == 2
    assert result["secret_uid"] == "registry-uid"
    assert len(wire.patches) == len(wire.publications) == 1
    preserved = next(
        item for item in result["registrations"] if item["cluster_id"] == "production"
    )
    assert preserved == original


@pytest.mark.parametrize(
    "failure", ["secret-replaced", "head-drift", "publication-rejected"]
)
def test_registry_ack_reconciliation_refuses_identity_or_generation_drift(
    monkeypatch, failure: str
) -> None:
    wire = registry_wire(monkeypatch, failure=failure)
    with pytest.raises(RuntimeError, match="changed|unresolved"):
        registry_module.write_registry(
            wire.baseline + [synthetic_registration()],
            run_id="run-a",
            expected_entries=wire.baseline,
            reason="test registration",
        )
    assert len(wire.patches) == 1
    assert len(wire.publications) == (0 if failure == "secret-replaced" else 1)


def test_cleanup_recovers_a_durable_only_registration_after_partial_ack(
    monkeypatch,
) -> None:
    wire = registry_wire(monkeypatch)
    wire.revision["registrations"].append(synthetic_registration())
    wire.revision["content_sha256"] = regional_registry_content_sha256(
        [
            RegionalClusterRegistration.model_validate(item)
            for item in wire.revision["registrations"]
        ]
    )
    removed = registry_module.cleanup_registry_residuals(
        scope="isolated", artifacts=None, phase="postflight", force=True, run_id="run-a"
    )
    assert removed == 0, (
        "Secret already removed the row; durable cleanup must still run"
    )
    assert len(wire.publications) == 1
    assert [item["cluster_id"] for item in wire.revision["registrations"]] == [
        "production"
    ]
    assert wire.patches == []


def test_registry_cleanup_preserves_other_runs(monkeypatch) -> None:
    owned = synthetic_registration()
    foreign = {**owned, "cluster_id": "perf-cap-001", "synthetic_run_id": "other-run"}
    entries = [owned, foreign, {"cluster_id": "production"}]
    writes = []
    monkeypatch.setattr(registry_module, "load_registry", lambda: list(entries))

    def write(values, **kwargs):
        writes.append(kwargs)
        entries[:] = values

    monkeypatch.setattr(registry_module, "write_registry", write)
    monkeypatch.setattr(
        registry_module, "sync_dataplane_connection_secret", lambda: {"changed": False}
    )
    assert (
        registry_module.cleanup_registry_residuals(
            scope="isolated",
            artifacts=None,
            phase="postflight",
            force=True,
            run_id="run-a",
        )
        == 1
    )
    assert entries == [foreign, {"cluster_id": "production"}]
    assert writes[0]["expected_entries"][0] == owned


@pytest.mark.parametrize("run_id", [None, "", "run/other"])
def test_unscoped_capacity_cleanup_never_contacts_a_cluster(
    monkeypatch, run_id
) -> None:
    from scripts.e2e.regional import net_command_fixture

    monkeypatch.setattr(
        net_command_fixture,
        "database_residuals",
        lambda *a: pytest.fail("unscoped cleanup must not contact the database"),
    )
    monkeypatch.setattr(
        suite, "dataplane", lambda *a, **k: pytest.fail("unscoped teardown")
    )
    with pytest.raises(RuntimeError, match="explicit run identity"):
        suite.purge_audit_rows(run_id=run_id)
    with pytest.raises(RuntimeError, match="explicit run identity"):
        suite.teardown(purge=True, deregister_clusters=True, run_id=run_id, attempts=1)


@pytest.mark.parametrize("remaining", [0, 1, False, None])
@pytest.mark.parametrize("lost_ack", [False, True])
def test_capacity_teardown_stops_owned_jobs_and_requires_exact_cleanup_before_revocation(
    tmp_path: Path, monkeypatch, remaining, lost_ack: bool
) -> None:
    from scripts.perf import regional_capacity_data

    job = str(suite.CASES["burst"]["job"])
    resources = {
        f"{kind}/{name}": {
            "uid": f"uid-{kind}",
            "resourceVersion": "1",
            "name": name,
            "namespace": suite.NAMESPACE,
            "labels": {registry_module.RUN_LABEL: "run-a"},
        }
        for kind, name in (("job", job), ("secret", suite.TOKEN_SECRET))
    }
    events = []
    (tmp_path / "registry-registration-intent.json").write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "cluster_ids": ["perf-cap-000"],
                "data_empty_before_registration": True,
            }
        )
    )
    (tmp_path / "capacity-resources.json").write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "namespace": suite.NAMESPACE,
                "resources": {f"job/{job}": {"uid": "uid-job", "data_sha256": ""}},
            }
        )
    )
    (tmp_path / "registry-token-proof.json").write_text(
        '{"run_id":"run-a","uid":"uid-secret"}'
    )

    def dataplane(*args, **kwargs):
        if args[0] == "get":
            value = resources.get(f"{args[1]}/{args[2]}")
            if not value:
                return ""
            return json.dumps(
                value if args[-1] == "jsonpath={.metadata}" else {"metadata": value}
            )
        assert args[:2] == ("delete", "--raw")
        options = json.loads(kwargs["stdin"])
        uid = options["preconditions"]["uid"]
        key = next(key for key, item in resources.items() if item["uid"] == uid)
        assert options["preconditions"]["resourceVersion"] == "1"
        events.append(key)
        del resources[key]
        if lost_ack and key.startswith("job/"):
            raise TimeoutError("Job delete acknowledgement lost")
        return "deleted"

    def residuals(control, *, run_id, cluster_ids, cleanup, force_nonterminal=False):
        assert run_id == "run-a"
        assert cluster_ids == ["perf-cap-000"] and cleanup is True
        assert force_nonterminal is True, (
            "the teardown may remove the run's open synthetic commands"
        )
        assert f"job/{job}" not in resources
        events.append("residuals")
        return {"total": remaining}

    monkeypatch.setattr(suite, "dataplane", dataplane)
    monkeypatch.setattr(regional_capacity_data, "invoke", residuals)
    monkeypatch.setattr(suite, "deregister", lambda **k: events.append("deregister"))
    monkeypatch.setattr(
        suite.capacity_registry,
        "verify_registry_alignment",
        lambda **k: events.append(f"alignment:{k['phase']}"),
    )
    kwargs = {
        "purge": True,
        "deregister_clusters": True,
        "run_id": "run-a",
        "artifacts": tmp_path,
        "attempts": 1,
    }
    if type(remaining) is int and remaining == 0:
        suite.teardown(**kwargs)
        assert events == [
            f"job/{job}",
            "residuals",
            "deregister",
            f"secret/{suite.TOKEN_SECRET}",
            # the gate runs last: a failing gate never strands the token Secret
            "alignment:postflight",
        ]
        assert resources == {}
    else:
        with pytest.raises(RuntimeError, match="exact run-owned cleanup"):
            suite.teardown(**kwargs)
        assert events == [f"job/{job}", "residuals"]
        assert f"secret/{suite.TOKEN_SECRET}" in resources


def test_capacity_teardown_never_deletes_a_foreign_or_replaced_job(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "registry-registration-intent.json").write_text('{"run_id":"run-a"}')
    job = str(suite.CASES["burst"]["job"])
    metadata = {
        "uid": "replacement",
        "resourceVersion": "2",
        "name": job,
        "namespace": suite.NAMESPACE,
        "labels": {registry_module.RUN_LABEL: "other-run"},
    }

    def dataplane(*args, **kwargs):
        assert args[0] == "get", "an unowned Job must never be deleted"
        return json.dumps({"metadata": metadata}) if args[1:3] == ("job", job) else ""

    monkeypatch.setattr(suite, "dataplane", dataplane)
    (tmp_path / "capacity-resources.json").write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "namespace": suite.NAMESPACE,
                "resources": {f"job/{job}": {"uid": "original", "data_sha256": ""}},
            }
        )
    )
    with pytest.raises(RuntimeError, match="ownership or UID changed"):
        suite.teardown(
            purge=False,
            deregister_clusters=False,
            artifacts=tmp_path,
            run_id="run-a",
            attempts=1,
        )
    metadata["labels"][registry_module.RUN_LABEL] = "run-a"
    with pytest.raises(RuntimeError, match="ownership or UID changed"):
        suite.teardown(
            purge=False,
            deregister_clusters=False,
            artifacts=tmp_path,
            run_id="run-a",
            attempts=1,
        )


def test_registration_rejects_old_cluster_data_before_any_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.perf import regional_capacity_data
    from tests.regional._perf_caller_support import AmpWire

    amp = AmpWire(monkeypatch)
    monkeypatch.setattr(registry_module, "validate_notification_safety", lambda: None)
    monkeypatch.setattr(
        registry_module, "validate_registry_target", lambda **k: "isolated"
    )
    monkeypatch.setattr(registry_module, "cleanup_registry_residuals", lambda **k: 0)
    monkeypatch.setattr(registry_module, "load_registry", lambda: [])
    monkeypatch.setattr(registry_module, "verify_registry_alignment", lambda **k: {})
    monkeypatch.setattr(registry_module, "capture_secret_baseline", lambda *a: None)
    monkeypatch.setattr(regional_capacity_data, "invoke", lambda *a, **k: {"total": 1})
    monkeypatch.setattr(
        registry_module,
        "upsert_secret",
        lambda *a, **k: pytest.fail("old data must prevent creating a new token"),
    )
    with pytest.raises(RuntimeError, match="data already exists"):
        registry_module.register(
            1,
            tmp_path,
            run_id="run-a",
            expires_at=NOW + timedelta(hours=1),
            allow_live_registry=False,
            live_registry_confirmation=None,
        )
    assert amp.calls == ["list-workspaces", "describe-alert-manager-definition"], (
        "old-data rejection must be reached only after AMP notification preflight"
    )
    assert not (tmp_path / "registry-registration-intent.json").exists(), (
        "rejected preflight must not grant cleanup ownership of old data"
    )


def test_teardown_without_registration_receipt_is_refused_before_commands(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        suite, "dataplane", lambda *a, **k: pytest.fail("unowned cleanup")
    )
    with pytest.raises(RuntimeError, match="no registration ownership receipt"):
        suite.teardown(
            purge=True,
            deregister_clusters=True,
            run_id="run-a",
            artifacts=tmp_path,
            attempts=1,
        )


def test_preflight_refuses_another_active_synthetic_run(monkeypatch) -> None:
    monkeypatch.setattr(
        registry_module,
        "load_registry",
        lambda: [
            {
                "cluster_id": "perf-cap-000",
                "synthetic": True,
                "synthetic_run_id": "other-run",
                "synthetic_expires_at": (NOW + timedelta(hours=1)).isoformat(),
            }
        ],
    )
    with pytest.raises(RuntimeError, match="preflight cannot purge"):
        registry_module.cleanup_registry_residuals(
            scope="isolated", artifacts=None, phase="preflight", force=False, now=NOW
        )


def test_teardown_retries_idempotently(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(suite, "validate_registry_target", lambda **_kwargs: "isolated")
    monkeypatch.setattr(suite.time, "sleep", lambda _seconds: None)

    def teardown_once(**_kwargs):
        calls.append(True)
        if len(calls) < 3:
            raise RuntimeError("transient")

    monkeypatch.setattr(suite, "_teardown_once", teardown_once)

    suite.teardown(purge=True, deregister_clusters=True)

    assert len(calls) == 3


def test_capacity_exception_still_runs_teardown(tmp_path: Path, monkeypatch) -> None:
    from tests.regional._perf_caller_support import CapacityWire

    wire = CapacityWire(tmp_path, monkeypatch)
    args = SimpleNamespace(
        command="all",
        clusters=1,
        keep_registration=False,
        allow_live_registry=False,
        confirm_live_registry=None,
        no_purge=False,
        case="burst",
        nodes_per_cluster=1,
        gpu_evidence_total=0,
        host_evidence_total=0,
        training_heartbeat_total=0,
        workload_observation_total=0,
        correlate_attempt_faults=False,
        workers=1,
        duration_seconds=1,
        lead_seconds=1,
        cpu_request="1",
        cpu_limit="1",
        label="test",
        fault_only=False,
        prewarm_connections=False,
        artifact_root=tmp_path,
    )
    monkeypatch.setattr(
        suite,
        "execute_case",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )

    with pytest.raises(RuntimeError, match="failed"):
        suite.execute_capacity_command(
            args,
            artifacts=tmp_path,
            suite_id="run-a",
            expires_at=NOW + timedelta(hours=1),
            scope="isolated",
            xid_total=1,
            sxid_total=1,
        )

    assert wire.teardowns[0]["run_id"] == "run-a"
    assert wire.teardowns[0]["deregister_clusters"] is True


def test_registry_baseline_artifact_carries_digests_not_tokens() -> None:
    # baseline 行是从生产 registry Secret 里原样读出来的，带真 token。
    # 每次 register 都把它整行写进 artifacts/，77 份历史证据因此全都
    # 明文存着同一个区域集群 token。工件只需要回答「保留了哪几行、
    # 有没有放回同样的行」，摘要足够。
    token = "3f9c1a04be27d5610872ef4bc93d0a6f5e18720b4dcaf3961e05b8d2740cae63"
    entries = [{"cluster_id": "hp-a", "region": "us-east-2", "token": token}]

    redacted = registry_module.redacted_registry_entries(entries)

    assert "token" not in redacted[0]
    assert redacted[0]["token_sha256"] == hashlib.sha256(token.encode()).hexdigest()
    assert redacted[0]["cluster_id"] == "hp-a"
    # 原始入参不能被就地改写：同一批 entries 随后要写回 registry Secret。
    assert entries[0]["token"] == token


def test_capacity_cleanup_removes_action_workflows_without_cluster_payload() -> None:
    statements = {
        name: (sql, pattern) for name, sql, pattern in suite.AUDIT_PURGE_STATEMENTS
    }

    sql, pattern = statements["gpu_fault_action_workflows"]

    assert "kind='workflow'" in sql
    assert "key LIKE %s" in sql
    assert pattern == "action_workflow"


def test_capacity_cleanup_removes_synthetic_notification_chain() -> None:
    statements = {
        name: (sql, pattern) for name, sql, pattern in suite.AUDIT_PURGE_STATEMENTS
    }

    for name in (
        "gpu_fault_notification_results",
        "gpu_fault_notification_deliveries",
        "gpu_fault_notifications",
    ):
        sql, pattern = statements[name]
        assert "cluster_name" in sql
        assert pattern == "cluster"


def test_capacity_cleanup_removes_synthetic_links() -> None:
    statements = {
        name: (sql, pattern) for name, sql, pattern in suite.AUDIT_PURGE_STATEMENTS
    }

    sql, pattern = statements["gpu_fault_links"]

    assert "DELETE FROM gpu_fault_links" in sql
    assert "rtrim(%s, '%%')" in sql
    assert "strpos(link.key, pattern.prefix)" in sql
    assert "strpos(link.value, pattern.prefix)" in sql
    assert pattern == "cluster"


def test_capacity_cleanup_removes_markers_with_nested_cluster_scope() -> None:
    statements = {
        name: (sql, pattern) for name, sql, pattern in suite.AUDIT_PURGE_STATEMENTS
    }

    sql, pattern = statements["gpu_fault_markers"]

    assert "kind='marker'" in sql
    assert "payload::text LIKE %s" in sql
    assert pattern == "cluster_contains"


def test_repository_registry_baselines_are_redacted() -> None:
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "artifacts").rglob("registry-baseline.json")
        if any(
            "token" in entry for entry in json.loads(path.read_text(encoding="utf-8"))
        )
    ]

    assert offenders == []


def test_burst_job_keeps_indexed_start_gate_contract() -> None:
    manifest = suite.build_job(
        "burst",
        clusters=32,
        nodes_per_cluster=256,
        xid_total=500,
        sxid_total=500,
        gpu_evidence_total=0,
        host_evidence_total=0,
        training_heartbeat_total=0,
        workload_observation_total=0,
        correlate_attempt_faults=False,
        start_epoch=1234.0,
        workers=128,
        duration_seconds=60,
        cpu_request="2",
        cpu_limit="4",
        include_telemetry=True,
        prewarm_connections=False,
    )

    spec = manifest["spec"]
    container = spec["template"]["spec"]["containers"][0]
    environment = {item["name"]: item for item in container["env"]}

    assert spec["completionMode"] == "Indexed"
    assert spec["completions"] == 32
    assert container["readinessProbe"]["exec"]["command"][-1] == (
        "test -f /tmp/gpu-fault-load-ready"
    )
    assert environment["START_GATE_NAME"]["value"] == suite.START_GATE_CONFIGMAP
    assert environment["CLUSTER_OFFSET"]["valueFrom"]["fieldRef"]["fieldPath"] == (
        "metadata.annotations['batch.kubernetes.io/job-completion-index']"
    )
    assert environment["CACHE_CONTROL_PLANE_DNS"]["value"] == "true"


def test_complete_single_cluster_burst_covers_attempt_context() -> None:
    events = burst.build_events(
        cluster_count=1,
        cluster_offset=0,
        nodes_per_cluster=1000,
        xid_total=500,
        sxid_total=500,
        gpu_evidence_total=500,
        host_evidence_total=500,
        include_telemetry=True,
        training_heartbeat_total=1000,
        workload_observation_total=500,
        correlate_attempt_faults=True,
    )
    counts = {
        kind: sum(item[0] == kind for item in events)
        for kind in {item[0] for item in events}
    }

    assert counts == {
        "FABRIC_MANAGER_LOG": 500,
        "GPU_INVENTORY": 1000,
        "GPU_METRICS": 1000,
        "GPU_METRICS_EVIDENCE": 500,
        "HOST_TELEMETRY": 1000,
        "HOST_TELEMETRY_EVIDENCE": 500,
        "NVIDIA_KERNEL": 500,
        "TRAINING_PROGRESS": 1000,
        "WORKLOAD_OBSERVATION": 500,
    }
    xid_nodes = [node for kind, _template, node in events if kind == "NVIDIA_KERNEL"]
    sxid_nodes = [
        node for kind, _template, node in events if kind == "FABRIC_MANAGER_LOG"
    ]
    assert xid_nodes == list(range(0, 1000, 2))
    assert sxid_nodes == list(range(1, 1000, 2))


def test_complete_burst_payloads_share_two_node_attempt_identity() -> None:
    templates = json.loads(
        (ROOT / "scripts/perf/payload-templates.json").read_text(encoding="utf-8")
    )
    observation = burst.build_event_payload(
        templates=templates,
        kind="WORKLOAD_OBSERVATION",
        template_kind="WORKLOAD_OBSERVATION",
        cluster_id="cluster-a",
        node_index=24,
        sequence=1,
        correlate_attempt_faults=True,
    )
    heartbeat = burst.build_event_payload(
        templates=templates,
        kind="TRAINING_PROGRESS",
        template_kind="TRAINING_PROGRESS",
        cluster_id="cluster-a",
        node_index=25,
        sequence=2,
        correlate_attempt_faults=True,
    )
    fault = burst.build_event_payload(
        templates=templates,
        kind="FABRIC_MANAGER_LOG",
        template_kind="FABRIC_MANAGER_LOG",
        cluster_id="cluster-a",
        node_index=25,
        sequence=3,
        correlate_attempt_faults=True,
    )

    assert observation["attempt_id"] == "burst-attempt-0012"
    assert [item["node_id"] for item in observation["containers"]] == [
        "burst-node-0024",
        "burst-node-0025",
    ]
    assert heartbeat["attempt_id"] == observation["attempt_id"]
    assert heartbeat["rank"] == 1
    assert fault["affected_workload_ids"] == observation["workload_ids"]
    assert fault["workload_state"] == "ACTIVE"
    AttemptObservation(**observation)
    TrainingProgressHeartbeat(**heartbeat)
    FabricManagerLogEvent(**fault)


def test_result_aggregation_preserves_fault_latency_summary() -> None:
    summary = suite.aggregate(
        [
            {
                "events": 2,
                "wall_seconds": 1.0,
                "start_lag_seconds": 0.1,
                "client_cpu_cores": 0.5,
                "dns_mode": "process-cache",
                "dns_resolution_seconds": 0.02,
                "dns_resolution_attempts": 1,
                "dns_address_count": 2,
                "dns_cache_hits": 2,
                "paths": {
                    "NVIDIA_KERNEL": {
                        "raw_latencies_ms": [10.0, 20.0],
                        "status_counts": {"202": 2},
                        "transport_retries": {"URLError:ConnectionResetError": 1},
                    }
                },
            }
        ]
    )

    assert summary["requests"] == 2
    assert summary["throughput_req_s"] == 2.0
    assert summary["fault_p50_ms"] == 10.0
    assert summary["fault_p99_ms"] == 10.0
    assert summary["dns_modes"] == ["process-cache"]
    assert summary["dns_resolution_attempts_max"] == 1
    assert summary["dns_address_count_min"] == 2
    assert summary["dns_address_count_max"] == 2
    assert summary["dns_cache_hits"] == 2
    assert summary["paths"]["NVIDIA_KERNEL"]["transport_retries"] == {
        "URLError:ConnectionResetError": 1
    }


def test_control_plane_dns_resolution_retries_and_deduplicates() -> None:
    calls = 0
    sleeps: list[float] = []
    address = (
        socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        ("10.0.0.1", 443),
    )

    def resolver(*_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise socket.gaierror(socket.EAI_AGAIN, "temporary failure")
        return [address, address]

    resolution = burst.resolve_control_plane_dns(
        "control.example",
        443,
        attempts=3,
        base_delay_seconds=0.25,
        resolver=resolver,
        sleeper=sleeps.append,
    )

    assert resolution.addresses == (address,)
    assert resolution.attempts == 2
    assert sleeps == [0.25]


def test_process_dns_cache_rotates_addresses_and_delegates() -> None:
    first = (
        socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        ("10.0.0.1", 443),
    )
    second = (
        socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        ("10.0.0.2", 443),
    )
    delegated = (
        socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        ("10.0.0.3", 443),
    )
    fallback_calls = []

    def fallback(*args):
        fallback_calls.append(args)
        return [delegated]

    cache = burst.ProcessDnsCache(
        hostname="control.example",
        port=443,
        addresses=(first, second),
        fallback=fallback,
    )

    assert cache.getaddrinfo(
        "CONTROL.EXAMPLE.",
        443,
        socket.AF_UNSPEC,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
    ) == [first, second]
    assert cache.getaddrinfo(
        "control.example", 443, socket.AF_UNSPEC, socket.SOCK_STREAM, socket.IPPROTO_TCP
    ) == [second, first]
    assert cache.getaddrinfo(
        "other.example", 443, socket.AF_UNSPEC, socket.SOCK_STREAM, socket.IPPROTO_TCP
    ) == [delegated]
    assert cache.hits == 2
    assert len(fallback_calls) == 1


def test_process_dns_cache_restores_global_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    address = (
        socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        ("10.0.0.1", 443),
    )

    def original(*_args):
        return [address]

    monkeypatch.setattr(burst.socket, "getaddrinfo", original)
    cache = burst.ProcessDnsCache(
        hostname="control.example", port=443, addresses=(address,), fallback=original
    )

    with burst.process_dns_cache(cache):
        assert burst.socket.getaddrinfo(
            "control.example",
            443,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
        ) == [address]

    assert burst.socket.getaddrinfo is original


def test_synchronized_burst_records_recovered_transport_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    class Response:
        status = 202
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise burst.urllib_error.URLError(ConnectionResetError())
        return Response()

    monkeypatch.setattr(burst, "build_event_payload", lambda **_kwargs: {})
    monkeypatch.setattr(burst.urllib_request, "urlopen", fake_urlopen)

    result = burst.send_event(
        (0, ("NVIDIA_KERNEL", "NVIDIA_KERNEL", 0)),
        templates={},
        registration={"cluster_id": "cluster-a", "token": "token-a"},
        cluster_offset=0,
        correlate_attempt_faults=False,
        action_event_indexes={},
        action_run_id="",
        runtime_profile_version="profile-a",
        connections=[None],
        prewarm_connections=False,
        base_url="https://control.example",
        base_path="",
        ssl_context=SimpleNamespace(),
    )

    assert result[1] == 202
    assert result[3] is None
    assert result[5] == "URLError:ConnectionResetError"
    assert attempts == 2


def test_connection_secret_is_mirrored_from_the_identity_namespace(monkeypatch) -> None:
    """Live 2026-09-13: the perf namespace still held the connection Secret of a
    site uninstalled and rebuilt twelve days later; every load Pod failed TLS
    verification against the old CA and the round aborted without a log. The
    identity namespace's Secret is mirrored before anything is registered."""

    live = {
        "apiVersion": "v1",
        "kind": "Secret",
        "type": "Opaque",
        "metadata": {
            "name": "gpu-fault-regional-connection",
            "namespace": "gpu-fault-system",
            "uid": "abc",
            "resourceVersion": "7",
        },
        "data": {
            "ca.crt": "bmV3",
            "cluster-token": "dG9rZW4=",
            "control-plane-url": "aHR0cHM=",
        },
    }
    stale = {
        **live,
        "data": {
            "ca.crt": "b2xk",
            "cluster-token": "b2xk",
            "control-plane-url": "aHR0cHM=",
        },
    }
    applied: list[dict] = []

    monkeypatch.setattr(
        registry_module, "dataplane_identity", lambda *a, **k: json.dumps(live)
    )

    def dataplane(*args, stdin=None, **_kwargs):
        if args[:2] == ("get", "namespace"):
            return json.dumps({"kind": "Namespace"})
        if args[:2] == ("get", "secret"):
            return json.dumps(stale)
        if args[:1] == ("apply",):
            applied.append(json.loads(stdin))
            return ""
        raise AssertionError(f"unexpected kubectl: {args}")

    monkeypatch.setattr(registry_module, "dataplane", dataplane)

    report = registry_module.sync_dataplane_connection_secret()

    assert report["changed"] is True, "a stale mirror is replaced"
    assert applied and applied[0]["data"] == live["data"], "the live data is mirrored"
    assert applied[0]["metadata"]["namespace"] == registry_module.NAMESPACE, (
        "the mirror lands in the perf namespace"
    )
    assert "uid" not in applied[0]["metadata"], "source metadata is not copied"
    assert report["keys"] == ["ca.crt", "cluster-token", "control-plane-url"]
    assert report["namespace_created"] is False, "an existing namespace is untouched"


def test_connection_secret_mirror_creates_an_absent_perf_namespace(monkeypatch) -> None:
    """Live 2026-09-22: the perf namespace had been swept with acceptance residue;
    `kubectl apply` of the mirrored Secret failed with NotFound and round 1 of the
    matrix aborted at preflight. The namespace is created before the Secret."""

    live = {"data": {"ca.crt": "bmV3"}, "type": "Opaque"}
    applied: list[dict] = []
    monkeypatch.setattr(
        registry_module, "dataplane_identity", lambda *a, **k: json.dumps(live)
    )

    def dataplane(*args, stdin=None, **_kwargs):
        if args[:2] in {("get", "namespace"), ("get", "secret")}:
            return ""
        if args[:1] == ("apply",):
            applied.append(json.loads(stdin))
            return ""
        raise AssertionError(f"unexpected kubectl: {args}")

    monkeypatch.setattr(registry_module, "dataplane", dataplane)

    report = registry_module.sync_dataplane_connection_secret()

    assert [item["kind"] for item in applied] == ["Namespace", "Secret"], (
        "the namespace is created first, then the Secret is mirrored into it"
    )
    assert applied[0]["metadata"]["name"] == registry_module.NAMESPACE, (
        "the perf namespace is the one created"
    )
    assert report["namespace_created"] is True and report["changed"] is True, (
        "the report tells both facts"
    )


def test_connection_secret_mirror_fails_closed_without_a_joined_data_plane(
    monkeypatch,
) -> None:
    monkeypatch.setattr(registry_module, "dataplane_identity", lambda *a, **k: "")

    with pytest.raises(RuntimeError, match="is missing from the identity namespace"):
        registry_module.sync_dataplane_connection_secret()


def test_connection_secret_mirror_is_a_no_op_when_already_current(monkeypatch) -> None:
    live = {"data": {"ca.crt": "bmV3"}, "type": "Opaque"}
    monkeypatch.setattr(
        registry_module, "dataplane_identity", lambda *a, **k: json.dumps(live)
    )

    def dataplane(*args, **_kwargs):
        if args[:2] == ("get", "namespace"):
            return json.dumps({"kind": "Namespace"})
        if args[:2] == ("get", "secret"):
            return json.dumps(live)
        raise AssertionError(f"an unchanged mirror must not be re-applied: {args}")

    monkeypatch.setattr(registry_module, "dataplane", dataplane)

    assert registry_module.sync_dataplane_connection_secret()["changed"] is False


def _alertmanager_definition(alertmanager_config: str) -> str:
    import base64
    import subprocess as sp

    del sp
    outer = "alertmanager_config: |\n" + "".join(
        "  " + line + "\n" for line in alertmanager_config.splitlines()
    )
    return json.dumps(
        {
            "alertManagerDefinition": {
                "data": base64.b64encode(outer.encode()).decode(),
                "status": {"statusCode": "ACTIVE"},
            }
        }
    )


_SINKING = """\
route:
  receiver: gpu-fault-sns
  routes:
    - receiver: gpu-fault-drill-sink
      matchers:
        - cluster_id=~"perf-cap-.*"
receivers:
  - name: gpu-fault-drill-sink
  - name: gpu-fault-sns
    sns_configs:
      - topic_arn: arn:aws:sns:us-west-2:123456789012:t
"""


def test_capacity_run_requires_the_alertmanager_drill_sink(monkeypatch) -> None:
    """Live 2026-09-13: the dispatcher suppressed every drill notification, but
    Alertmanager mailed the synthetic clusters' AMP alerts straight to SNS. The
    live definition must sink perf-cap- alerts before anything is registered."""

    monkeypatch.setattr(registry_module, "AMP_WORKSPACE_ID", "ws-test")

    def run(command, **_kwargs):
        assert "describe-alert-manager-definition" in command, command
        return SimpleNamespace(stdout=_alertmanager_definition(_SINKING).encode())

    monkeypatch.setattr(registry_module, "run", run)

    report = registry_module.validate_alertmanager_drill_route()

    assert report["workspace_id"] == "ws-test"
    assert report["drill_sink_routes"] == [
        {"receiver": "gpu-fault-drill-sink", "matchers": ['cluster_id=~"perf-cap-.*"']}
    ], "the sink route is the evidence"

    mailing = _SINKING.replace("  - name: gpu-fault-drill-sink\n", "").replace(
        "gpu-fault-drill-sink", "gpu-fault-sns"
    )
    monkeypatch.setattr(
        registry_module,
        "run",
        lambda command, **_k: SimpleNamespace(
            stdout=_alertmanager_definition(mailing).encode()
        ),
    )
    with pytest.raises(RuntimeError, match="must not mail drill alerts"):
        registry_module.validate_alertmanager_drill_route()


class _FakeRunResources:
    def __init__(self, existing: set[tuple[str, str]]) -> None:
        self.existing = existing
        self.created: list[dict] = []

    def read(self, kind: str, name: str) -> dict | None:
        return {"kind": kind} if (kind, name) in self.existing else None

    def create(self, manifest: dict) -> dict:
        self.created.append(manifest)
        return manifest


def test_load_service_account_is_created_only_when_the_perf_namespace_lacks_it() -> (
    None
):
    """Live 2026-09-22: a recreated perf namespace had no load ServiceAccount, the
    Jobs could not create Pods and the round timed out. A run creates the account
    (owned, torn down with the run) when absent and never touches an existing one."""

    import scripts.perf.regional_capacity_suite as suite
    from scripts.perf.regional_capacity_resources import ensure_load_service_account

    ensure = lambda resources: ensure_load_service_account(  # noqa: E731
        resources, name=suite.LOAD_SERVICE_ACCOUNT, namespace=suite.NAMESPACE
    )
    absent = _FakeRunResources(set())
    assert ensure(absent) is True, "an absent account is created"
    assert [m["kind"] for m in absent.created] == ["ServiceAccount"], (
        "exactly one ServiceAccount manifest is created"
    )
    assert absent.created[0]["metadata"] == {
        "name": suite.LOAD_SERVICE_ACCOUNT,
        "namespace": suite.NAMESPACE,
    }, "the account the Jobs and the start-gate RoleBinding name"

    present = _FakeRunResources({("serviceaccount", suite.LOAD_SERVICE_ACCOUNT)})
    assert ensure(present) is False, "an existing account is left alone"
    assert present.created == [], "nothing is created over an existing account"
