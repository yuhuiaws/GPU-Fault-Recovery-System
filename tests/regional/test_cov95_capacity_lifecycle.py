from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from scripts.e2e.regional import capacity_acceptance_base as base
from tests.regional._cov95_capacity_support import capacity_fixture as capacity_fixture
from tests.regional._cov95_capacity_support import pod


class Forward:
    def __init__(self, *, exited=False, stubborn=False):
        self.exited = exited
        self.stubborn = stubborn
        self.events = []

    def poll(self):
        return 0 if self.exited else None

    def terminate(self):
        self.events.append("terminate")

    def kill(self):
        self.events.append("kill")
        self.stubborn = False

    def wait(self, *, timeout):
        self.events.append(("wait", timeout))
        if self.stubborn:
            raise subprocess.TimeoutExpired("unit-port-forward", timeout)
        self.exited = True
        return 0


def install_transport(
    harness, monkeypatch, *, exited=False, stubborn=False, health="ok"
):
    forward = Forward(exited=exited, stubborn=stubborn)
    calls = []
    clock = {"value": 0.0}

    def start(arguments, **options):
        calls.append(arguments)
        assert arguments[2] == harness.cpu_kubeconfig, (
            "forwarding must use only the CPU kubeconfig"
        )
        assert "port-forward" in arguments, (
            "the only allowed process is the owned forwarding transport"
        )
        return forward

    def get(url, **options):
        assert url == "http://127.0.0.1:12345/healthz", (
            "health probes use only the owned local forward"
        )
        if health == "error":
            raise httpx.ReadTimeout("simulated timeout")
        return SimpleNamespace(
            status_code=200 if health != "unavailable" else 503,
            json=lambda: {"status": health},
        )

    monkeypatch.setattr(harness, "reserve_port", lambda: 12345)
    monkeypatch.setattr(base.subprocess, "Popen", start)
    monkeypatch.setattr(base.httpx, "get", get)
    monkeypatch.setattr(base.time, "monotonic", lambda: clock["value"])
    monkeypatch.setattr(
        base.time, "sleep", lambda seconds: clock.update(value=clock["value"] + seconds)
    )
    return forward, calls


def test_probe_lifecycle_clones_only_required_public_configuration(
    capacity, monkeypatch
) -> None:
    harness, api = capacity
    live = harness.live_worker["spec"]["template"]["spec"]
    live["volumes"].append(
        {"name": "unrelated-credentials", "secret": {"secretName": "unrelated"}}
    )
    live["containers"][0]["volumeMounts"].append(
        {"name": "unrelated-credentials", "mountPath": "/unrelated"}
    )
    api.services = [
        {"spec": {"selector": {}}},
        {"spec": {"selector": {"app": "unrelated"}}},
    ]
    forward, calls = install_transport(harness, monkeypatch, stubborn=True)
    probe = harness.deploy_probe("CAP001", {})
    assert probe is harness.active_probe and len(calls) == 1, (
        "ready probe retains its owned forwarding handle"
    )
    service, deployment = api.applied
    assert (
        service["spec"]["selector"] == deployment["spec"]["selector"]["matchLabels"]
    ), "the private Service selects only this probe"
    spec = deployment["spec"]["template"]["spec"]
    assert {volume["name"] for volume in spec["volumes"]} == {
        "rds-ca-bundle",
        "scripts",
        "work",
    }, "production credential volumes are not cloned"
    assert spec["serviceAccountName"] == "cpu-service-account", (
        "the expected CPU identity is retained"
    )
    result = harness.cleanup_probe(probe)
    assert result == {"database_dropped": True, "residual_probe_pods": []}, (
        "cleanup proves both process and resource closure"
    )
    assert forward.events == ["terminate", ("wait", 5), "kill", ("wait", 3)], (
        "stubborn local forwarding is reaped"
    )
    assert harness.active_probe is None, (
        "the tracker is cleared only after cleanup proof"
    )


@pytest.mark.parametrize("problem", ["collision", "missing-ca", "writable-ca"])
def test_probe_refuses_unsafe_service_or_ca_inheritance(
    capacity, monkeypatch, problem
) -> None:
    harness, api = capacity
    install_transport(harness, monkeypatch)
    if problem == "collision":
        api.services = [{"spec": {"selector": {"app": "gpu-fault-capacity-probe"}}}]
    elif problem == "missing-ca":
        harness.live_worker["spec"]["template"]["spec"]["volumes"] = []
    else:
        harness.live_worker["spec"]["template"]["spec"]["containers"][0][
            "volumeMounts"
        ][0]["readOnly"] = False
    with pytest.raises(base.CapError, match="Service|RDS CA"):
        harness.deploy_probe("CAP001", {})
    assert not api.applied, "unsafe inheritance is refused before creating a probe"


@pytest.mark.parametrize("health", ["error", "starting", "unavailable"])
def test_unready_http_probe_times_out_and_cleans_owned_resources(
    capacity, monkeypatch, health
) -> None:
    harness, api = capacity
    forward, calls = install_transport(
        harness, monkeypatch, health=health, stubborn=True
    )
    with pytest.raises(base.CapError, match="did not become ready"):
        harness.deploy_probe("CAP001", {})
    assert len(calls) == 1 and forward.exited, (
        "timeout must reap the only forwarding process"
    )
    assert harness.active_probe is None and api.deleted, (
        "failed readiness still cleans its resources"
    )


def test_exited_forward_is_not_a_ready_probe(capacity, monkeypatch) -> None:
    harness, _ = capacity
    forward, _ = install_transport(harness, monkeypatch, exited=True)
    with pytest.raises(base.CapError, match="exited before readiness"):
        harness.deploy_probe("CAP001", {})
    assert forward.events == [], "an exited process is not signaled again"


def test_incomplete_replica_proof_retains_failed_cleanup_for_retry(
    capacity, monkeypatch
) -> None:
    harness, api = capacity
    _, calls = install_transport(harness, monkeypatch)
    api.probe_pods = [pod("one", "probe"), pod("two", "probe")]
    with pytest.raises(base.CapError, match="completely Ready") as error:
        harness.deploy_probe("CAP001", {})
    assert not calls, "unknown replica count cannot start forwarding"
    assert harness.active_probe is not None, (
        "remaining Pods must retain the cleanup target"
    )
    assert any("cleanup failed" in note for note in error.value.__notes__), (
        "cleanup failure does not conceal the original readiness error"
    )


def test_an_existing_probe_or_foreign_database_cannot_be_overwritten(capacity) -> None:
    harness, api = capacity
    harness.active_probe = object()
    with pytest.raises(base.CapError, match="previous capacity probe"):
        harness.deploy_probe("CAP001", {})
    harness.active_probe = None
    with pytest.raises(base.CapError, match="database target differs"):
        harness.deploy_probe("CAP001", {"CAP_DATABASE_NAME": "production"})
    assert not api.applied, "neither refusal may create resources"


@pytest.mark.parametrize(
    "database", ["production", "gpu_fault_another_cap001", "gpu_fault_unsafe';drop"]
)
def test_database_cleanup_refuses_targets_outside_the_run(capacity, database) -> None:
    harness, api = capacity
    before = len(api.calls)
    with pytest.raises(base.CapError, match="outside this capacity run"):
        harness.drop_database_fallback(database)
    assert len(api.calls) == before, (
        "unowned database names are rejected before any remote command"
    )


def test_database_cleanup_requires_a_production_worker(capacity) -> None:
    harness, api = capacity
    api.pods = []
    with pytest.raises(base.CapError, match="no production worker"):
        harness.drop_database_fallback(f"gpu_fault_{harness.run_id}_cap001")


def test_cleanup_retains_resources_until_executor_shutdown_is_confirmed(
    capacity,
) -> None:
    harness, api = capacity
    harness.cap004_executor_stopped = False
    calls = len(api.calls)
    assert harness.cleanup_all() == [
        "CAP004 Executor shutdown is unverified; resources retained"
    ], "in-flight executor work forbids teardown"
    with pytest.raises(base.CapError, match="shutdown is unverified"):
        harness.cleanup_probe(object())
    assert len(api.calls) == calls, "unconfirmed shutdown cannot delete any resource"


def test_cleanup_all_preserves_each_failure_and_keeps_trying(
    capacity, monkeypatch
) -> None:
    harness, _ = capacity
    harness.active_probe = SimpleNamespace(case="CAP001", deployment="probe")
    events = []

    def probe_cleanup(probe):
        events.append("probe")
        return {"residual_probe_pods": ["leftover"]}

    def common_cleanup():
        events.append("common")
        raise base.CapError("common resource still exists")

    monkeypatch.setattr(harness, "cleanup_probe", probe_cleanup)
    monkeypatch.setattr(harness, "cleanup_common", common_cleanup)
    errors = harness.cleanup_all()
    assert events == ["probe", "common"] and len(errors) == 2, (
        "one cleanup failure cannot hide another"
    )
    recorded = json.loads((harness.run_dir / "cap001-cleanup-on-exit.json").read_text())
    assert recorded["residual_probe_pods"] == ["leftover"], (
        "residual evidence remains auditable"
    )


def test_cleanup_all_records_probe_failure_before_common_cleanup(
    capacity, monkeypatch
) -> None:
    harness, api = capacity
    harness.active_probe = SimpleNamespace(case="CAP001", deployment="probe")

    def refuse(probe):
        raise base.CapError("probe teardown failed")

    monkeypatch.setattr(harness, "cleanup_probe", refuse)
    errors = harness.cleanup_all()
    assert len(errors) == 1 and "probe teardown failed" in errors[0], (
        "probe failure is retained"
    )
    assert (
        harness.secret_name in api.deleted and harness.configmap_name in api.deleted
    ), "common cleanup is still attempted"


def test_run_rejects_invalid_predecessor_before_creating_resources(capacity) -> None:
    harness, api = capacity
    harness.predecessor = {"valid": False}
    with pytest.raises(base.CapError, match="predecessor"):
        harness.run()
    assert not api.applied, "invalid formal ordering cannot start a capacity case"


@pytest.mark.parametrize("passed", [True, False])
def test_run_requires_case_success_after_verified_cleanup(
    capacity, monkeypatch, passed
) -> None:
    harness, api = capacity
    monkeypatch.setattr(
        harness, "case_001", lambda: {"status": "PASS" if passed else "FAIL"}
    )
    if passed:
        assert harness.run() == 0, "valid case and stable production baseline pass"
    else:
        with pytest.raises(base.CapError, match="passing result"):
            harness.run()
    result = json.loads((harness.run_dir / f"{harness.case_id}.json").read_text())
    assert result["verdict"] == ("PASS" if passed else "FAIL"), (
        "final evidence reflects the actual case outcome"
    )
    assert result["production_unchanged"] is True, (
        "both complete snapshots were compared"
    )
    assert (
        harness.secret_name in api.deleted and harness.configmap_name in api.deleted
    ), "common resources do not survive either outcome"
