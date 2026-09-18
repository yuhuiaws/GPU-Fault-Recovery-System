from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import boot_acceptance_runtime as runtime
from scripts.e2e.regional.boot_acceptance_common import BootAcceptanceError, SiteFixture


class ReplicaTransport:
    def __init__(self) -> None:
        self.desired = 2
        self.items = [
            {
                "metadata": {"name": name, "uid": name},
                "spec": {"containers": [{"name": "executor"}]},
                "status": {
                    "phase": "Running",
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "containerStatuses": [{"name": "executor", "ready": True}],
                },
            }
            for name in ("executor-a", "executor-b")
        ]

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        if args[:2] == ("get", "deployment"):
            return json.dumps({"spec": {"replicas": self.desired}})
        assert args[:2] == ("get", "pod")
        return json.dumps({"items": self.items})


@pytest.mark.parametrize(
    "failure", ["missing", "not-ready", "terminating", "duplicate", "container", "spec"]
)
def test_site_fixture_cannot_hide_an_unhealthy_replica(failure: str) -> None:
    transport = ReplicaTransport()
    if failure == "missing":
        transport.items.pop()
    elif failure == "not-ready":
        transport.items[1]["status"]["conditions"][0]["status"] = "False"
    elif failure == "terminating":
        transport.items[1]["metadata"]["deletionTimestamp"] = "2026-09-12T00:00:00Z"
    elif failure == "container":
        transport.items[1]["status"]["containerStatuses"][0]["ready"] = False
    elif failure == "spec":
        del transport.items[1]["spec"]
    else:
        transport.items[1] = copy.deepcopy(transport.items[0])
    fixture = SiteFixture.__new__(SiteFixture)
    fixture.regional = transport  # type: ignore[assignment]

    with pytest.raises(BootAcceptanceError, match="replica"):
        fixture.pods("gpu", "gpu-fault-cluster-executor")


def test_site_fixture_returns_the_complete_ready_replica_set() -> None:
    fixture = SiteFixture.__new__(SiteFixture)
    fixture.regional = ReplicaTransport()  # type: ignore[assignment]

    assert fixture.pods("gpu", "gpu-fault-cluster-executor") == [
        "executor-a",
        "executor-b",
    ]


class RuntimeFixture:
    def __init__(self) -> None:
        self.regional = self
        self.region = "us-west-2"
        self.cluster_id = "cluster-a"
        self.config = {"runtime_profile": {"version": "profile-a"}}
        self.ca_error = "ssl.SSLError: [X509: NO_CERTIFICATE_OR_CRL_FOUND]"
        self.log_error = False
        self.calls: list[tuple[str, ...]] = []
        self.matrix = {
            "valid": {"status": 200},
            "wrong_token": {"status": 403},
            "wrong_pin": {"status": 503},
            "no_owner": {"status": 503},
            "stale_claim": {"status": 503},
        }
        self.ses: dict[str, Any] = {
            "region_present": True,
            "config_region": self.region,
            "client_region": self.region,
            "execution_enabled": True,
            "allow_email": True,
            "local_param_validation": True,
            "no_region_error": True,
        }
        self.notification: dict[str, Any] = {
            "allow_email": True,
            "async_delivery": True,
            "dispatcher_enabled": True,
            "deliver_backlog": False,
            "backlog_grace_seconds": 60,
            "notification_count": 3,
            "result_count": 3,
            "status_counts": {"SENT": 3},
        }
        self.notification["environment"] = {
            key: value
            for key, value in self.notification.items()
            if key
            in {
                "async_delivery",
                "dispatcher_enabled",
                "deliver_backlog",
                "backlog_grace_seconds",
            }
        }

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "release-a", "cluster_id": self.cluster_id}

    def pods(self, _plane: str, _app: str) -> list[str]:
        return ["replica-a", "replica-b"]

    def kubectl(self, _plane: str, *args: str, **kwargs: Any) -> str:
        self.calls.append(args)
        if args[:2] == ("get", "deployment"):
            return json.dumps(runtime.manifest_deployment())
        assert args[0] == "logs"
        if self.log_error:
            if kwargs.get("check", True):
                raise BootAcceptanceError("logs unavailable")
            return "forbidden"
        return ""

    def pod_json(
        self, _plane: str, _pod: str, script: str, *args: str
    ) -> dict[str, Any]:
        if script == runtime.BOOT012_ENV_PROBE:
            return {
                "ca_file": "/etc/gpu-fault/tls/ca.crt",
                "ca_exists": True,
                "ssl_cert_file_set": False,
                "requests_ca_bundle_set": False,
            }
        if script == runtime.BOOT012_STS_PROBE:
            return {"caller_identity_available": True}
        if script == runtime.BOOT013_PROBE:
            return copy.deepcopy(self.ses)
        if script == runtime.BOOT014_PROBE:
            return copy.deepcopy(self.notification)
        raise AssertionError("unexpected probe")

    def exec(
        self, _plane: str, pod: str, *args: str, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((pod, *args))
        if args[0] == "sh":
            if kwargs.get("check", True):
                raise BootAcceptanceError("empty CA command exited nonzero")
            return subprocess.CompletedProcess(args, 1, "", self.ca_error)
        if kwargs.get("input_text") == runtime.READINESS_PROBE:
            return subprocess.CompletedProcess(args, 0, json.dumps(self.matrix), "")
        return subprocess.CompletedProcess(args, 0, "", "")


@pytest.mark.parametrize(
    "error",
    [
        "ssl.SSLError: [X509: NO_CERTIFICATE_OR_CRL_FOUND]",
        "ssl.SSLCertVerificationError: CERTIFICATE_VERIFY_FAILED",
    ],
)
def test_boot012_accepts_only_a_specific_empty_ca_tls_refusal(
    error: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = RuntimeFixture()
    fixture.ca_error = error
    monkeypatch.setattr(runtime, "secret_key_contract", lambda: {"passed": True})
    result = runtime.run_boot012(fixture)  # type: ignore[arg-type]

    assert result["verdict"] == "PASS"
    assert "boot021_evidence" not in result, "BOOT-012 cannot claim a later formal case"
    assert len(result["replicas"]) == 2
    assert list(tmp_path.iterdir()) == [], (
        "BOOT-012 must not write future case evidence"
    )


def test_boot021_executes_its_own_matrix_with_the_executor_interpreter() -> None:
    fixture = RuntimeFixture()

    result = runtime.run_boot021(fixture)  # type: ignore[arg-type]

    assert result["verdict"] == "PASS", "the fresh complete matrix should pass"
    assert [call[0] for call in fixture.calls] == ["replica-a", "replica-b"], (
        "every desired replica needs a fresh readiness audit"
    )
    assert all(
        call[1:3] == ("/opt/gpu-fault/executor/bin/python", "-")
        for call in fixture.calls
    ), "system Python cannot load the executor business package"


def test_boot021_cannot_pass_a_failed_matrix_command() -> None:
    fixture = RuntimeFixture()
    fixture.exec = lambda *_a, **_k: subprocess.CompletedProcess(  # type: ignore[method-assign]
        [], 1, json.dumps(fixture.matrix), "failure"
    )

    assert runtime.run_boot021(fixture)["verdict"] == "FAIL", (  # type: ignore[arg-type]
        "printed statuses are not proof that the audit completed successfully"
    )


def test_boot012_does_not_accept_an_unrelated_empty_ca_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = RuntimeFixture()
    fixture.ca_error = "permission denied"
    monkeypatch.setattr(runtime, "secret_key_contract", lambda: {"passed": True})

    result = runtime.run_boot012(fixture)  # type: ignore[arg-type]

    assert result["verdict"] == "FAIL"
    assert result["checks"]["empty_ca_rejected"] is False


def test_boot012_unreadable_logs_are_not_zero_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = RuntimeFixture()
    fixture.log_error = True
    fixture.ca_error = "CERTIFICATE_VERIFY_FAILED"
    monkeypatch.setattr(runtime, "secret_key_contract", lambda: {"passed": True})

    with pytest.raises(BootAcceptanceError, match="logs unavailable"):
        runtime.run_boot012(fixture)  # type: ignore[arg-type]


def test_boot012_requires_the_whole_readiness_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = RuntimeFixture()
    fixture.ca_error = "CERTIFICATE_VERIFY_FAILED"
    fixture.matrix["wrong_token"]["status"] = 200
    monkeypatch.setattr(runtime, "secret_key_contract", lambda: {"passed": True})

    result = runtime.run_boot012(fixture)  # type: ignore[arg-type]

    assert result["verdict"] == "FAIL"
    assert result["checks"]["readiness_matrix"] is False


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("region_present", False),
        ("client_region", "us-east-1"),
        ("execution_enabled", False),
        ("local_param_validation", False),
        ("no_region_error", False),
    ],
)
def test_boot013_aggregates_actual_probe_failures(field: str, bad: Any) -> None:
    fixture = RuntimeFixture()
    assert runtime.run_boot013(fixture)["verdict"] == "PASS"  # type: ignore[arg-type]
    fixture.ses[field] = bad

    assert runtime.run_boot013(fixture)["verdict"] == "FAIL"  # type: ignore[arg-type]


def test_boot014_compares_service_configuration_to_the_environment() -> None:
    fixture = RuntimeFixture()
    assert runtime.run_boot014(fixture)["verdict"] == "PASS"  # type: ignore[arg-type]
    fixture.notification["environment"]["dispatcher_enabled"] = False

    result = runtime.run_boot014(fixture)  # type: ignore[arg-type]

    assert result["verdict"] == "FAIL"
    assert result["checks"]["service_matches_environment"] is False


def test_boot014_requires_no_missing_results_in_any_replica() -> None:
    fixture = RuntimeFixture()
    original = fixture.pod_json

    def probe(plane: str, pod: str, script: str, *args: str) -> dict[str, Any]:
        value = original(plane, pod, script, *args)
        if pod == "replica-b":
            value["result_count"] = 2
        return value

    fixture.pod_json = probe  # type: ignore[method-assign]
    result = runtime.run_boot014(fixture)  # type: ignore[arg-type]

    assert result["verdict"] == "FAIL"
    assert result["checks"]["backlog_without_result_zero"] is False


def test_boot015_insert_ack_loss_still_cleans_its_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = RuntimeFixture()
    calls: list[str] = []

    def cpu_python(script: str, *args: str) -> dict[str, Any]:
        if script == runtime.PROFILE_OWNER_PROBE:
            assert args[0] == "profile-a", (
                "Profiles are shared by version, not cluster anchor"
            )
            return {"profiles": [{"profile_version": "profile-a", "orphan_owners": []}]}
        if script == runtime.REMOTE_BASELINE_PROBE:
            return {"count": 0, "ids": []}
        if script == runtime.REMOTE_INJECT_PROBE:
            calls.append("insert")
            raise OSError("write acknowledged too late")
        if script == runtime.REMOTE_DELETE_PROBE:
            calls.append("delete")
            return {"remaining": 0}
        raise AssertionError("unexpected Store probe")

    fixture.cpu_python = cpu_python  # type: ignore[attr-defined]
    fixture.pod_json = lambda *_a: {"owners": ["owner-a"]}  # type: ignore[method-assign]
    monkeypatch.setattr(
        runtime, "run", lambda *_a, **_k: subprocess.CompletedProcess([], 0, "", "")
    )
    monkeypatch.setattr(runtime, "wait_amp_alert", lambda *_a, **_k: {"matched": True})

    with pytest.raises(OSError, match="write acknowledged"):
        runtime.run_boot015(fixture, case_dir=tmp_path, attempt=1)  # type: ignore[arg-type]

    assert calls == ["insert", "delete"]
    details = json.loads((tmp_path / "boot015-details.json").read_text())
    assert details["verdict"] == "FAIL"
    assert details["cleanup"]["synthetic_command_removed"] is True


def test_boot015_unresolved_alert_cannot_pass_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = RuntimeFixture()
    fixture.cpu_python = lambda *_a: {"count": 0, "ids": []}  # type: ignore[attr-defined]
    monkeypatch.setattr(runtime, "wait_amp_alert", lambda *_a, **_k: {"matched": False})

    result = runtime.boot015_cleanup(
        fixture,  # type: ignore[arg-type]
        command_id="owned-command",
        pods=["replica-a"],
        baseline_ids=set(),
    )

    assert result["synthetic_command_removed"] is True
    assert result["passed"] is False


def test_boot015_missing_profile_is_rejected_before_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = RuntimeFixture()
    fixture.pod_json = lambda *_a: {"owners": ["owner-a"]}  # type: ignore[method-assign]
    calls: list[str] = []

    def cpu_python(script: str, *args: str) -> dict[str, Any]:
        if script == runtime.PROFILE_OWNER_PROBE:
            return {"profiles": []}
        if script == runtime.REMOTE_BASELINE_PROBE:
            return {"count": 0, "ids": []}
        calls.append("unexpected")
        pytest.fail("missing Profile must not permit injection")

    fixture.cpu_python = cpu_python  # type: ignore[attr-defined]
    monkeypatch.setattr(
        runtime, "run", lambda *_a, **_k: subprocess.CompletedProcess([], 0, "", "")
    )

    with pytest.raises(BootAcceptanceError, match="Profile"):
        runtime.run_boot015(fixture, case_dir=tmp_path, attempt=1)  # type: ignore[arg-type]
    assert calls == []


@pytest.mark.parametrize("inventory", [{}, {"count": 1, "ids": []}, {"count": 0}])
def test_boot015_unknown_command_inventory_cannot_prove_cleanup(
    inventory: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = RuntimeFixture()
    fixture.cpu_python = lambda *_a: inventory  # type: ignore[attr-defined]
    monkeypatch.setattr(runtime, "wait_amp_alert", lambda *_a, **_k: {"matched": True})

    result = runtime.boot015_cleanup(
        fixture,  # type: ignore[arg-type]
        command_id="owned-command",
        pods=["replica-a"],
        baseline_ids=set(),
    )

    assert result["synthetic_command_removed"] is False, (
        "unknown absence is not cleanup"
    )
    assert result["passed"] is False, "readiness alone cannot prove command deletion"
