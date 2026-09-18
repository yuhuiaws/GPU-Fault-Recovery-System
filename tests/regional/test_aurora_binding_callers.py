"""Behavioral proof that each Aurora caller reaches the binding guard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import aurora_binding as binding
from scripts.e2e.regional import ha009_refresh
from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional import run_ha009_aurora_credential_rotation as ha009
from scripts.e2e.regional import run_ha010_aurora_blackout_liveness as ha010
from tests.regional._aurora_binding_support import CLUSTER, PASSWORD, FakeAurora

DEADLINE = datetime(2099, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("succeeded", [False, True])
def test_refresh_failure_and_success_evidence_never_copy_job_logs(
    succeeded: bool, tmp_path: Path
) -> None:
    secret_text = FakeAurora().dsn
    resources = SimpleNamespace(
        create=lambda _: None, owned=lambda *a: {"metadata": {"uid": "unit"}}
    )

    def control(*args, **kwargs):
        if args[0] == "get":
            return json.dumps({"status": {"succeeded": int(succeeded)}})
        if args[0] == "logs":
            return secret_text + "\nrotated=True restarted=False"
        return secret_text

    def action():
        return ha009_refresh.run_refresh_job(
            tmp_path,
            {"metadata": {"name": "unit"}},
            resources,
            control=control,
            timeout_seconds=700,
        )

    if succeeded:
        result = action()
        assert result["logs"] == ["rotated=True restarted=False"], result
        assert PASSWORD not in json.dumps(result), "safe outcome is not raw Job output"
    else:
        with pytest.raises(ha009_refresh.CaseError) as failure:
            action()
        assert PASSWORD not in str(failure.value), "wait output may contain credentials"
    assert list(tmp_path.iterdir()) == [], "raw refresh logs must not become artifacts"


@pytest.mark.parametrize("module", [ha003, ha010])
def test_failover_entry_proves_binding_before_any_setup_mutation(
    module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment = FakeAurora()
    environment.cluster["Endpoint"] = "foreign.invalid"
    monkeypatch.setattr(module, "regional_binding", lambda *a: environment.guard)
    mutations = []
    monkeypatch.setattr(
        module, "RegionalLiveFixture", lambda settings: SimpleNamespace()
    )
    if module is ha003:
        monkeypatch.setattr(
            module, "HostProbeFixture", lambda *a: mutations.append("host")
        )
    else:
        monkeypatch.setattr(
            module, "start_sampler", lambda *a, **kw: mutations.append("sampler")
        )
    settings = SimpleNamespace(regional=SimpleNamespace(), rds_cluster_id=CLUSTER)
    with pytest.raises(binding.BindingError):
        module.execute_case(settings, tmp_path, 1, DEADLINE)
    assert mutations == [], "an unbound Store cannot authorize setup mutations"


def test_ha009_entry_refuses_before_registration_and_does_not_arm_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    environment = FakeAurora()
    environment.failure = RuntimeError(environment.dsn)
    monkeypatch.setattr(ha009, "aurora_guard", lambda: environment.guard)
    monkeypatch.setattr(ha009.BASE, "database_residuals", lambda: {"total": 0})
    monkeypatch.setattr(ha009.BASE, "registry_residuals", lambda: {"count": 0})
    monkeypatch.setattr(ha009.BASE, "kubernetes_residuals", lambda: {"count": 0})
    mutations = []
    monkeypatch.setattr(
        ha009.BASE, "register", lambda *a, **kw: mutations.append("register")
    )
    monkeypatch.setattr(
        ha009.BASE, "teardown", lambda *a, **kw: mutations.append("teardown")
    )
    assert ha009.run_case(tmp_path, 1, DEADLINE) == 1, (
        "unreadable proof must fail the case"
    )
    assert mutations == [], "preflight refusal must not mutate during cleanup"
    report = json.loads(
        (tmp_path / "cases" / ha009.CASE_ID / f"{ha009.CASE_ID}.json").read_text()
    )
    assert report["verdict"] == "FAIL", report
    assert PASSWORD not in json.dumps(report) + capsys.readouterr().out, (
        "canonical failure evidence must be sanitized"
    )


@pytest.mark.parametrize("drift", [False, True])
def test_ha009_revalidates_immediately_before_rotation_and_before_marking_it_started(
    drift: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = FakeAurora()
    state = {
        "aurora_binding": environment.guard.read(),
        "maintenance_window_end": DEADLINE,
        "rotation_started": False,
    }
    calls = []

    def read(expected):
        calls.append("binding")
        return environment.guard.read(expected)

    monkeypatch.setattr(ha009, "aurora_guard", lambda: SimpleNamespace(read=read))
    monkeypatch.setattr(ha009, "aws", lambda *a: calls.append(a[:2]) or {})
    if drift:
        environment.cluster["DbClusterResourceId"] = "replacement-cluster"
        with pytest.raises(binding.BindingError):
            ha009.request_rotation(state)
        assert state["rotation_started"] is False, (
            "refusal must not arm emergency rotation cleanup"
        )
        assert calls == ["binding"], calls
    else:
        ha009.request_rotation(state)
        assert calls == ["binding", ("rds", "modify-db-cluster")], calls
        assert state["rotation_started"] is True, (
            "ambiguous provider completion must retain recovery"
        )


def test_ha010_revalidates_its_own_boundary_before_failover_or_pod_deletion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment = FakeAurora()
    expected = environment.guard.read()
    environment.cluster["DbClusterResourceId"] = "replacement-cluster"
    monkeypatch.setattr(ha010, "regional_binding", lambda *a: environment.guard)
    calls = []
    monkeypatch.setattr(ha010, "_quiet_control_plane", lambda *a: calls.append("quiet"))
    monkeypatch.setattr(ha010.ha003, "aws_rds", lambda *a: calls.append("failover"))
    monkeypatch.setattr(ha010, "delete_pod", lambda *a: calls.append("delete"))
    run = SimpleNamespace(
        regional=SimpleNamespace(),
        settings=SimpleNamespace(rds_cluster_id=CLUSTER),
        preflight={"aurora_binding": expected},
        case_dir=tmp_path,
    )
    with pytest.raises(binding.BindingError):
        ha010.request_failover(run)
    assert calls == ["quiet"], "foreign Store/RDS proof must prevent both actions"


@pytest.mark.parametrize("refusal", ["reset", "failover", "terminal", "none"])
def test_ha003_checks_before_real_reset_and_again_before_failover(
    refusal: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment = FakeAurora()
    expected = environment.guard.read()
    events = []
    checks = 0
    claims = 0

    def read(value):
        nonlocal checks
        checks += 1
        events.append("binding")
        if (refusal == "reset" and checks == 1) or (
            refusal == "failover" and checks == 2
        ):
            environment.cluster["DbClusterResourceId"] += "-changed"
        return environment.guard.read(value)

    monkeypatch.setattr(
        ha003, "regional_binding", lambda *a: SimpleNamespace(read=read)
    )
    monkeypatch.setattr(
        ha003,
        "read_only_preflight",
        lambda *a: {
            "errors": [],
            "release_id": "unit-release",
            "store": {},
            "aurora_binding": expected,
        },
    )
    monkeypatch.setattr(ha003, "verify_plan_identity", lambda *a: None)

    class Host:
        host_script = "/unit/probe"

        def __init__(self, *args):
            pass

        def create(self):
            events.append("host")

        def execute(self, operation, *args, **kwargs):
            events.append(operation)
            if operation == "snapshot":
                return {
                    "compute_clients": [],
                    "gpu_inventory": [{"pci_bdf": "unit-bdf"}],
                }
            return {}

        def cleanup(self):
            events.append("cleanup")
            return {}

    monkeypatch.setattr(ha003, "HostProbeFixture", Host)
    monkeypatch.setattr(
        ha003, "HostProbeSettings", lambda **kwargs: SimpleNamespace(**kwargs)
    )

    def claimed(*args, observe_state, **kwargs):
        nonlocal claims
        claims += 1
        events.append("claim")
        state = {"incident": {"incident_id": "unit-incident"}}
        observe_state(state)
        return state, {
            "command_id": "unit-command",
            "status": "SUCCEEDED"
            if refusal == "terminal" and claims == 2
            else "LEASED",
        }

    def snapshot(**kwargs):
        state, command = claimed(observe_state=lambda _: None)
        return {**state, "commands": [{**command, "step": {"operation": "RESET_GPU"}}]}

    monkeypatch.setattr(
        ha003,
        "RegionalLiveFixture",
        lambda *a: SimpleNamespace(
            node_snapshot=lambda *a: {"ownership_annotations": {}},
            store_snapshot=snapshot,
        ),
    )

    def failover(*args):
        events.append("failover")
        raise RuntimeError("unit stop after observing dispatch")

    monkeypatch.setattr(ha003, "wait_reset_claim", claimed)
    monkeypatch.setattr(ha003, "aws_rds", failover)
    settings = SimpleNamespace(
        rds_cluster_id=CLUSTER,
        node="unit-node",
        host_probe_image="unit-image",
        regional=SimpleNamespace(
            cpu_kubeconfig=tmp_path / "cpu",
            gpu_kubeconfig=tmp_path / "gpu",
            gpu_context="unit",
            namespace="unit",
            cluster_id="unit-gpu",
        ),
    )
    assert ha003.execute_case(settings, tmp_path, 1, DEADLINE) == 1, (
        "mock boundary run must not claim live PASS"
    )
    assert ("write-xid46" in events) is (refusal != "reset"), events
    assert ("failover" in events) is (refusal == "none"), events
    if refusal == "none":
        assert events[events.index("failover") - 2 : events.index("failover")] == [
            "binding",
            "claim",
        ], "fresh identity and open reset are required at dispatch"
    if refusal != "reset":
        assert "restore-quiesce" in events, (
            "binding refusal must retain the already-created incident for cleanup"
        )
    assert "cleanup" in events, "owned host resources must still be cleaned"
