from __future__ import annotations

import copy
import hashlib
import json
import socket
import ssl
import tempfile
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import audit_auth_boundary as audit
from scripts.e2e.regional import identity_acceptance_auth as auth
from scripts.e2e.regional.probes import auth015_node_probe as node_probe


def matrix_results() -> dict[str, Any]:
    results = {
        name: {"status": status, "body": {}}
        for name, status in audit.expected_statuses({}).items()
    }
    for name, detail in audit.EXPECTED_DETAILS.items():
        results[name]["body"] = {"detail": detail}
    for name in ("AUTH-008-A-normal", "AUTH-008-A-fake-executor"):
        results[name]["body"] = {"commands": []}
    return results


def documents(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return audit.case_documents(
        results,
        cluster_a="a",
        store_errors={},
        identity={"cluster_id": "a", "release_id": "release-test"},
    )


@pytest.mark.parametrize(
    ("case_id", "entry"),
    [
        (case_id, entry)
        for case_id, entries in audit.CASE_ENTRIES.items()
        for entry in entries
    ],
)
def test_matrix_requires_every_case_entry(case_id: str, entry: str) -> None:
    complete = matrix_results()
    assert documents(complete)[case_id]["verdict"] == "PASS"
    del complete[entry]
    result = documents(complete)[case_id]
    assert result["verdict"] == "FAIL"
    assert entry in result["entry_errors"], "missing request has no diagnostic"


@pytest.mark.parametrize("entry", sorted(audit.EXPECTED_DETAILS))
def test_matrix_denial_reason_cannot_be_replaced_by_another_403(entry: str) -> None:
    results = matrix_results()
    results[entry]["body"] = {"detail": "a different denial"}
    errors = audit.matrix_errors(results, cluster_a="a")
    assert entry in errors, "the status hid the wrong authorization branch"


def test_near_token_response_cannot_leak_a_discriminatory_extra_field() -> None:
    results = matrix_results()
    results["AUTH-004-near"]["body"]["matched_prefix_length"] = 31
    assert documents(results)["GF-REGIONAL-AUTH-004"]["verdict"] == "FAIL"


@pytest.mark.parametrize("body", [None, [], "malformed", {}, {"commands": None}])
def test_matrix_requires_a_real_claim_commands_list(body: Any) -> None:
    results = matrix_results()
    results["AUTH-008-A-normal"]["body"] = body
    assert documents(results)["GF-REGIONAL-AUTH-008"]["verdict"] == "FAIL"
    assert audit.probe_claims_leased_nothing(results) is False


def test_non_json_metrics_reply_preserves_status_without_copying_its_body() -> None:
    raw = b'gpu_fault_metric{cluster_id="foreign"} 1\n'
    body = audit.response_body(raw)
    assert body == {
        "non_json_body_sha256": hashlib.sha256(raw).hexdigest(),
        "non_json_body_bytes": len(raw),
    }
    results = matrix_results()
    results["AUTH-011-metrics"] = {"status": 200, "body": body}
    assert documents(results)["GF-REGIONAL-AUTH-011"]["verdict"] == "FAIL"


def negative_snapshot() -> dict[str, Any]:
    return {
        "clusters": {
            "b": {
                "attempt_observations": 0,
                "agent_generations": [],
                "collector_samples": {"NVIDIA_KERNEL/node": 7},
                "evidence_count": 4,
                "agents_sha256": "a" * 64,
                "observations_sha256": "b" * 64,
                "evidence_sha256": "c" * 64,
            }
        },
        "commands": {
            "b-command": {
                "cluster_id": "b",
                "status": "PENDING",
                "lease_owner": None,
                "execution_owner": audit.ACCEPTANCE_PROBE_OWNER,
            }
        },
    }


@pytest.mark.parametrize("field", ["collector_samples", "evidence_count"])
def test_event_negative_checks_the_named_collector_and_evidence_invariants(
    field: str,
) -> None:
    before = negative_snapshot()
    after = copy.deepcopy(before)
    after["clusters"]["b"][field] = (
        {"NVIDIA_KERNEL/node": 8} if field == "collector_samples" else 5
    )
    errors = audit.store_negative_errors(before, after, cluster_a="a", cluster_b="b")
    assert "GF-REGIONAL-AUTH-009" in errors, "foreign ingestion went unmeasured"
    assert "GF-REGIONAL-AUTH-008" not in errors


def test_store_negative_rejects_missing_measurements_and_missing_b_backlog() -> None:
    errors = audit.store_negative_errors({}, {}, cluster_a="a", cluster_b="b")
    assert set(errors) == set(audit.STORE_NEGATIVE_CASES)
    before = negative_snapshot()
    assert (
        audit.store_negative_errors(before, before, cluster_a="a", cluster_b="b") == {}
    )
    before["commands"]["b-command"]["cluster_id"] = "a"
    assert "GF-REGIONAL-AUTH-008" in audit.store_negative_errors(
        before, before, cluster_a="a", cluster_b="b"
    ), "A's command cannot stand for an available B command"


@pytest.mark.parametrize(
    "gap", ["URLError", "probe-unavailable:RuntimeError", None, 403]
)
def test_rotation_does_not_discard_interruption_samples(gap: Any) -> None:
    samples = [
        {
            "phase": phase,
            "observed_at": "2026-09-12T00:00:00Z",
            "status": 200,
            "credential_slot": slot,
        }
        for phase, slot in (
            ("baseline", "old"),
            ("overlap", "old"),
            ("overlap", "new"),
            ("cutover", "new"),
            ("completed", "new"),
        )
    ]
    site = SimpleNamespace(registry_generation=lambda: 8)
    primary = SimpleNamespace(
        cpu_python=lambda *args: {
            "commands": [{"command_id": "c", "status": "SUCCEEDED"}]
        }
    )
    arguments = {
        "samples": samples,
        "before_commands": {"commands": [{"command_id": "c", "status": "PENDING"}]},
        "new_token_during_overlap": 200,
        "old_token_after_completion": 403,
        "restored": True,
        "registry_generation": 7,
        "control_rollout": 1.0,
        "executor_rollout": 1.0,
        "old_token": "o" * 32,
        "new_token": "n" * 32,
    }
    assert auth.auth016_result(site, primary, **arguments)["verdict"] == "PASS"
    samples.append(
        {"phase": "overlap", "observed_at": "2026-09-12T00:00:02Z", "status": gap}
    )
    outcome = auth.auth016_result(site, primary, **arguments)
    assert outcome["verdict"] == "FAIL"
    assert outcome["checks"]["overlap_only_200"] is False


@pytest.mark.parametrize(
    "wrong_name", ["accepted", "reset", "tls", "untrusted", "hostname"]
)
def test_tls_probe_distinguishes_hostname_verification_from_transport(
    wrong_name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
) -> None:
    ca_file = tmp_path / "public-ca.pem"
    ca_file.write_text("test public CA", encoding="ascii")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://control.example")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_CA_FILE", str(ca_file))
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    certificate = {
        "subjectAltName": (("DNS", "control.example"),),
        "notAfter": (datetime.now(timezone.utc) + timedelta(days=60)).strftime(
            "%b %d %H:%M:%S %Y GMT"
        ),
    }
    tls = SimpleNamespace(getpeercert=lambda: certificate)

    def wrap_socket(raw: object, *, server_hostname: str) -> Any:
        if server_hostname == "wrong.invalid":
            if wrong_name == "reset":
                raise OSError("connection reset")
            if wrong_name == "tls":
                raise ssl.SSLError("TLS alert")
            if wrong_name in {"untrusted", "hostname"}:
                error = ssl.SSLCertVerificationError(1, "certificate rejected")
                error.verify_code = 62 if wrong_name == "hostname" else 20
                raise error
        return nullcontext(tls)

    def context(*, cafile: str) -> Any:
        if Path(cafile).name == "empty-ca.pem":
            raise ssl.SSLError("no certificates")
        return SimpleNamespace(wrap_socket=wrap_socket)

    monkeypatch.setattr(ssl, "create_default_context", context)
    monkeypatch.setattr(
        socket, "create_connection", lambda *args, **kw: nullcontext(object())
    )
    exec(auth.TLS_BOUNDARY_PROBE, {})
    result = json.loads(capsys.readouterr().out)
    assert result["default_handshake_ok"] is True
    assert result["empty_ca_rejected"] is True
    assert result["wrong_hostname_rejected"] is (wrong_name == "hostname")
    assert list(tmp_path.glob("auth013-*")) == [], "empty CA temp files remain"


@pytest.mark.parametrize("cleanup", ["clean", "residual", "exception", "unknown"])
def test_tls_handler_never_passes_unproven_probe_cleanup(
    cleanup: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("test kubeconfig", encoding="ascii")

    class Probe:
        def create(self) -> None:
            pass

        def execute(self, *args: str) -> dict[str, Any]:
            return {
                "min_validity_seconds": 30 * 86400,
                "timer_enabled": True,
                "timer_active": True,
            }

        def cleanup(self) -> dict[str, Any]:
            if cleanup == "exception":
                raise RuntimeError("cleanup unavailable")
            if cleanup == "unknown":
                return {}
            return {"pod": cleanup == "residual", "configmap": False}

    site = SimpleNamespace(
        namespace="gpu-system",
        gpu_kubeconfig=kubeconfig,
        config={"health": {"certificate_min_validity_days": 30}},
        any_executor_pod=lambda target: "executor",
        pod_json=lambda *args: {
            "host": "control.example",
            "sans": ["control.example"],
            "ca_exists": True,
            "ssl_cert_file_set": False,
            "requests_ca_bundle_set": False,
            "hostname_in_san": True,
            "default_handshake_ok": True,
            "empty_ca_rejected": True,
            "wrong_hostname_rejected": True,
            "remaining_days": 60,
            "not_after": "2026-12-12T00:00:00Z",
        },
    )

    def probe_settings(**kwargs: Any) -> SimpleNamespace:
        assert kwargs["state_directory"] == tmp_path / "host-probes"
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(auth, "HostProbeSettings", probe_settings)
    monkeypatch.setattr(auth, "HostProbeFixture", lambda settings: Probe())
    result = auth.run_auth013(
        site,
        SimpleNamespace(context="gpu-context"),
        node="node-a",
        host_probe_image="image@sha256:" + "a" * 64,
        case_dir=tmp_path,
    )
    assert result["verdict"] == ("PASS" if cleanup == "clean" else "FAIL")
    assert result["checks"]["alert_probe_removed"] is (cleanup == "clean")


def host_scan_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, ...]:
    roots = tuple(tmp_path / name for name in ("tmp", "systemd", "kubelet"))
    proc = tmp_path / "proc"
    for directory in (*roots, proc):
        directory.mkdir()
    monkeypatch.setattr(node_probe, "SCAN_ROOTS", roots)
    monkeypatch.setattr(node_probe, "PROC_ROOT", proc)
    return (*roots, proc)


@pytest.mark.parametrize("location", ["tmp", "env", "projected", "process"])
def test_master_scan_uses_the_chrooted_host_paths_and_never_emits_the_value(
    location: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scratch, systemd, kubelet, proc = host_scan_layout(monkeypatch, tmp_path)
    master = b"test-master-" + b"m" * 40
    if location == "tmp":
        path = scratch / "install-key"
        data = master
    elif location == "env":
        path = systemd / "agent.env"
        data = b"KEY='" + master + b"'\n"
    elif location == "projected":
        path = kubelet / "key"
        data = master
    else:
        (proc / "101").mkdir()
        path = proc / "101" / "environ"
        data = b"KEY=" + master + b"\0OTHER=okay\0"
    path.write_bytes(data)
    result = node_probe.scan(hashlib.sha256(master).hexdigest())
    assert result["master_matches"], "the supplied master was not detected"
    assert result["values_scanned"] > 0
    assert master.decode() not in json.dumps(result), "scanner emitted a credential"


def test_master_scan_refuses_empty_or_missing_scan_surfaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    roots = host_scan_layout(monkeypatch, tmp_path)
    with pytest.raises(node_probe.ProbeError, match="no nonempty values"):
        node_probe.scan("a" * 64)
    roots[1].rmdir()
    with pytest.raises(node_probe.ProbeError, match="directory is missing"):
        node_probe.scan("a" * 64)


@pytest.mark.parametrize(
    "reference",
    [
        {"secret": {"name": auth.INSTALLER_SECRET}},
        {"secretRef": {"name": auth.INSTALLER_SECRET}},
        {"secret": {"secretName": auth.INSTALLER_SECRET}},
    ],
)
def test_master_reference_scan_covers_projected_and_envfrom_forms(
    reference: dict[str, Any],
) -> None:
    result = auth.master_reference_scan(
        {
            "items": [
                {
                    "kind": "Job",
                    "metadata": {"name": "installer-test"},
                    "spec": reference,
                }
            ]
        }
    )
    assert len(result["hits"]) == 1


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "missing-request",
        "metrics-write",
        "no-sg",
        "dual-success",
        "dual-wrong-denial",
        "dual-wrong-bucket",
    ],
)
def test_full_surface_audit_requires_complete_safe_buckets_and_a_security_group(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    routes = [
        {
            "path": path,
            "methods": ["GET" if path.endswith("agents") else "POST"],
            "bucket": bucket,
        }
        for path, bucket in auth.HIGH_RISK_ROUTE_BUCKETS.items()
    ]
    routes.extend(
        {"path": path, "methods": ["GET"], "bucket": "public-undocumented"}
        for path in ("/docs", "/redoc", "/openapi.json")
    )
    if defect == "metrics-write":
        routes.append({"path": "/v1/unsafe", "methods": ["POST"], "bucket": "metrics"})
    if defect == "dual-wrong-bucket":
        next(item for item in routes if item["path"] == "/v1/fleet/agents")[
            "bucket"
        ] = "execution-token"
    statuses = {
        "cluster-token": 401,
        "dual-credential": 403,
        "execution-token": 403,
        "metrics": 403,
        "public-undocumented": 200,
    }
    results = [
        {
            "path": item["path"],
            "method": item["methods"][0],
            "bucket": item["bucket"],
            "status": statuses[item["bucket"]],
        }
        for item in routes
    ]
    if defect == "missing-request":
        results.pop(0)
    if defect in {"dual-success", "dual-wrong-denial"}:
        next(item for item in results if item["path"] == "/v1/fleet/agents")[
            "status"
        ] = 200 if defect == "dual-success" else 401
    monkeypatch.setattr(auth, "route_inventory", lambda *args: routes)
    monkeypatch.setattr(auth, "anonymous_routes", lambda *args: results)
    monkeypatch.setattr(
        auth,
        "nlb_security_groups",
        lambda *args: {
            "hostname": "control.example",
            "security_group_count": 0 if defect == "no-sg" else 1,
            "broad_ipv4_rules": [],
            "broad_ipv6_rules": [],
        },
    )
    monkeypatch.setattr(auth, "outside_probe", lambda *args, **kw: {"valid": True})
    monkeypatch.setattr(auth, "data_plane_execution_token_hits", lambda *args: [])
    monkeypatch.setattr(
        auth, "authenticated_fleet_isolation", lambda *_args: {"passed": True}
    )
    result = auth.run_auth014(
        SimpleNamespace(), SimpleNamespace(), outside_probe_path=None
    )
    assert result["verdict"] == ("PASS" if defect == "none" else "FAIL"), (
        f"the surface audit must retain the failed boundary: {defect}"
    )
    failed_check = {
        "missing-request": "route_matrix_complete",
        "metrics-write": "write_routes_explicitly_protected",
        "no-sg": "nlb_has_security_groups",
        "dual-success": "route_matrix_matches_buckets",
        "dual-wrong-denial": "route_matrix_matches_buckets",
        "dual-wrong-bucket": "high_risk_routes_in_declared_buckets",
    }
    if defect in failed_check:
        assert result["checks"][failed_check[defect]] is False, (
            f"the verdict must identify the specific unsafe condition: {defect}"
        )
    else:
        assert all(result["checks"].values()), (
            "complete protected routes and network evidence must satisfy every check"
        )


def test_empty_checks_are_not_a_successful_acceptance() -> None:
    assert auth.verdict({}) == "FAIL"
