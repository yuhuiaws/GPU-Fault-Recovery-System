"""Contracts the identity / multi-cluster runners were found to break (review
2026-09-07): claim probes that leased production work, finally blocks that
stopped at the first restore failure, vacuous checks, and an ISO-006 block
that never cut the executor's egress."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from scripts.e2e.regional import audit_auth_boundary as audit
from scripts.e2e.regional import identity_acceptance_auth as auth
from scripts.e2e.regional import identity_acceptance_common as common
from scripts.e2e.regional import identity_acceptance_iso as iso
from scripts.e2e.regional import multi_cluster_fixture as multi
from scripts.e2e.regional import run_e2e002_multicluster_fault as e2e002
from scripts.e2e.regional import run_identity_acceptance as identity
from scripts.e2e.regional import run_iso006_cluster_offline as iso006
from scripts.e2e.regional.probes import auth013_certificate_probe as cert_probe
from scripts.e2e.regional.probes import cluster_network_probe
from tests._builders import asgi_client, build_context
from tests.regional._regional_support import (
    TOKEN_A,
    enqueue_remote_command,
    registration,
)

PROBE_OWNER = "gpu-fault-acceptance-probe"


# --------------------------------------------------------------------------- #
# 2. claim probes must never lease production work
# --------------------------------------------------------------------------- #
def test_every_claim_probe_advertises_the_acceptance_owner() -> None:
    assert common.ACCEPTANCE_PROBE_OWNER == PROBE_OWNER
    assert audit.ACCEPTANCE_PROBE_OWNER == PROBE_OWNER
    assert f'"{PROBE_OWNER}"' in common.CLAIM_PROBE
    assert "gpu-fault-kubernetes-adapter" not in common.CLAIM_PROBE
    payload = audit.claim_payload(
        executor_id="auth-probe",
        artifact_sha256="a" * 64,
        compatibility_digest="b" * 64,
    )
    assert payload["execution_owners"] == [PROBE_OWNER]


def test_direct_claim_uses_the_probe_owner_and_a_single_attempt() -> None:
    captured: dict[str, Any] = {}

    def executor_python(script: str, **kwargs: Any) -> dict[str, Any]:
        captured["script"] = script
        captured["kwargs"] = kwargs
        return {"status": 200}

    primary = SimpleNamespace(executor_python=executor_python)
    target: Any = SimpleNamespace(cluster_id="cluster-a")
    status = auth.direct_claim(
        primary,
        target,
        token="t" * 32,
        identity={
            "artifact": "a" * 64,
            "compatibility": "b" * 64,
            "owners": ["gpu-fault-kubernetes-adapter"],
        },
    )

    assert status == 200
    assert captured["kwargs"]["attempts"] == 1, "a retried claim is a second sample"
    assert f'"{PROBE_OWNER}"' in captured["script"]
    assert "gpu-fault-kubernetes-adapter" not in captured["script"]


def test_control_plane_answers_probe_owner_claim_with_200_and_no_commands() -> None:
    """The route only substitutes the adapter owner for an *empty* list; a
    non-empty unknown owner authenticates and claims nothing."""

    context = build_context()
    context.regional_mode = True
    context.execution_token = "e" * 32
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    enqueue_remote_command(context.store, "cmd-real", cluster_id="cluster-a")

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/regional/executors/claim",
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
                json={
                    "executor_id": "probe",
                    "execution_owners": [PROBE_OWNER],
                    "max_commands": 1,
                    "lease_seconds": 60,
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["commands"] == []

    asyncio.run(scenario())
    command = context.store.get_remote_command("cmd-real")
    assert command.status.value == "PENDING", "the probe leased real work"
    assert command.lease_owner is None


def test_commands_not_misterminated_needs_a_baseline_and_accepts_succeeded() -> None:
    assert auth.commands_not_misterminated({}, {}) is False, "empty baseline passed"
    assert auth.commands_not_misterminated({"c": "PENDING"}, {"c": "SUCCEEDED"}) is True
    assert auth.commands_not_misterminated({"c": "PENDING"}, {"c": "FAILED"}) is False
    assert auth.commands_not_misterminated({"c": "LEASED"}, {}) is False


def test_remote_status_probe_reports_tracked_terminal_commands() -> None:
    assert "tracked = set(sys.argv[1:])" in auth.REMOTE_STATUS_PROBE
    assert "or item.command_id in tracked" in auth.REMOTE_STATUS_PROBE


# --------------------------------------------------------------------------- #
# 3. finally blocks run every restore step and keep the original exception
# --------------------------------------------------------------------------- #
def test_run_cleanup_steps_runs_everything_and_collects_errors() -> None:
    calls: list[str] = []

    def failing() -> None:
        calls.append("first")
        raise RuntimeError("boom")

    def second() -> str:
        calls.append("second")
        return "ok"

    outcomes, errors = common.run_cleanup_steps(
        [("first", failing), ("second", second)]
    )

    assert calls == ["first", "second"], "the second step did not run"
    assert outcomes == {"second": "ok"}
    assert errors == ["first: RuntimeError: boom"]


def _target(cluster_id: str = "cluster-a") -> common.ClusterTarget:
    return common.ClusterTarget(
        cluster_id=cluster_id,
        context=f"ctx-{cluster_id}",
        region="us-west-2",
        hyperpod_cluster_name=f"hp-{cluster_id}",
        eks_cluster_arn=f"arn:aws:eks:us-west-2:000000000000:cluster/{cluster_id}",
        executor_role_arn="arn:aws:iam::000000000000:role/executor",
        control_plane_url="https://cp.example.internal",
        ca_file=Path("/dev/null"),
    )


def _fake_site(**overrides: Any) -> Any:
    site = SimpleNamespace(
        registry=lambda: [{"cluster_id": "cluster-b", "enabled": True}],
        write_registry=lambda entries, **kwargs: None,
        rollout_control=lambda: 0.5,
        last_registry_ready_seconds=1.25,
    )
    for key, value in overrides.items():
        setattr(site, key, value)
    return site


def test_auth007_failure_runs_every_restore_and_raises_the_original(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    writes: list[str] = []

    def write_registry(entries: list[dict[str, Any]], **kwargs: Any) -> None:
        writes.append("disabled" if not entries[0]["enabled"] else "restored")
        if writes[-1] == "restored":
            raise RuntimeError("registry restore failed")

    claims: list[str] = []

    def claim(site: Any, target: Any) -> dict[str, Any]:
        claims.append(target.cluster_id)
        if len(claims) == 1:
            raise ValueError("secondary claim exploded")
        return {"status": 200}

    monkeypatch.setattr(auth, "claim", claim)
    site = _fake_site(write_registry=write_registry)
    primary: Any = SimpleNamespace(cluster_id="cluster-a")
    secondary: Any = SimpleNamespace(cluster_id="cluster-b")

    with pytest.raises(common.IdentityCaseFailure) as caught:
        auth.run_auth007(site, primary, secondary, case_dir=tmp_path)

    assert isinstance(caught.value.__cause__, ValueError), "original exception lost"
    assert claims == ["cluster-b", "cluster-b"], "the restore claim did not run"
    assert caught.value.details["cleanup_errors"] == [
        "restore_registry: RuntimeError: registry restore failed"
    ]
    details = json.loads((tmp_path / "auth007-details.json").read_text())
    assert details["cleanup_errors"], "details were not written in finally"


def test_auth016_join_covers_a_whole_direct_claim() -> None:
    assert (
        auth.DIRECT_CLAIM_JOIN_SECONDS >= 300 + auth.DIRECT_CLAIM_EXEC_TIMEOUT_SECONDS
    )


def test_identity_case_failure_carries_partial_details_into_evidence() -> None:
    """The entry point's exception path writes the handler's partial checks."""

    failure = common.IdentityCaseFailure(
        "late failure", details={"checks": {"a": True}, "cleanup_errors": ["x"]}
    )
    assert isinstance(failure, common.IdentityAcceptanceError), (
        "the entry point's generic except must still catch it"
    )
    assert failure.details["cleanup_errors"] == ["x"]
    source = Path(str(identity.__file__)).read_text(encoding="utf-8")
    assert "except IdentityCaseFailure as exc:" in source
    assert '"partial": exc.details' in source
    assert "except Exception as exc:" in source, "generic failures still recorded"


# --------------------------------------------------------------------------- #
# 4. assertion fixes
# --------------------------------------------------------------------------- #
def test_verdict_rejects_truthy_non_boolean_checks() -> None:
    assert auth.verdict({"a": True}) == "PASS"
    assert auth.verdict({"a": "NOT_EVALUATED"}) == "FAIL"
    assert auth.verdict({"a": 1}) == "FAIL"


def test_write_registry_measures_post_to_last_ack_and_rollout_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = common.IdentitySite.__new__(common.IdentitySite)
    site.targets = {"cluster-a": _target()}
    site.last_registry_ready_seconds = None
    waited: list[int] = []
    monkeypatch.setattr(site, "registry_generation", lambda: 7)
    monkeypatch.setattr(
        site, "registry_api", lambda *args, **kwargs: {"status": 200, "body": {}}
    )
    monkeypatch.setattr(
        site, "wait_registry_ready", lambda generation, **kw: waited.append(generation)
    )

    site.write_registry([{"cluster_id": "cluster-a", "token_sha256": "a" * 64}])

    assert waited == [8]
    assert site.last_registry_ready_seconds is not None
    assert site.rollout_control() < 1.0
    assert waited == [8], "rollout_control waited again on an applied head"


def test_registry_api_reuses_the_api_pod_and_reselects_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = common.IdentitySite.__new__(common.IdentitySite)
    site.targets = {"cluster-a": _target()}
    site.last_registry_ready_seconds = None
    listings: list[int] = []
    pods = [["api-1"], ["api-2"]]

    def ready_pods(plane: str, app: str, target: Any) -> list[str]:
        listings.append(1)
        return pods[len(listings) - 1]

    monkeypatch.setattr(site, "ready_pods", ready_pods)
    seen: list[str] = []

    def pod_json(
        plane: str, target: Any, pod: str, script: str, *args: str
    ) -> dict[str, Any]:
        seen.append(pod)
        if pod == "api-1" and len(seen) == 3:
            raise RuntimeError("pod gone")
        return {"status": 200, "body": {}}

    monkeypatch.setattr(site, "pod_json", pod_json)

    site.registry_api("GET", "/x")
    site.registry_api("GET", "/x")
    assert listings == [1], "the API Pod was re-listed on every call"
    site.registry_api("GET", "/x")
    assert seen[-1] == "api-2", "a failed exec did not re-select the Pod"


def test_high_risk_routes_are_judged_by_bucket_not_counted() -> None:
    routes = [
        {"path": "/v1/runtime-profiles", "bucket": "execution-token"},
        {
            "path": "/v1/advisory-notifications/{notification_id}/send",
            "bucket": "public",
        },
        {"path": "/v1/fleet/agents", "bucket": "dual-credential"},
    ]
    errors = auth.high_risk_route_errors(routes)
    assert errors == [
        "/v1/advisory-notifications/{notification_id}/send is public, "
        "expected execution-token"
    ]
    assert auth.high_risk_route_errors(routes[:1])[0].endswith(
        "not in the route inventory"
    ), "a missing high-risk route must be named, not counted"


def test_route_inventory_probe_includes_the_openapi_surface() -> None:
    assert "UNDOCUMENTED_PUBLIC_PATHS" in auth.ROUTE_INVENTORY_PROBE
    assert "public-undocumented" in auth.ROUTE_INVENTORY_PROBE


def test_outside_probe_requires_host_age_and_records_digest(tmp_path: Path) -> None:
    now = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
    path = tmp_path / "outside.json"
    path.write_text(
        json.dumps(
            {
                "connection_blocked": True,
                "target_host": "nlb.example.internal",
                "observed_at": "2026-09-07T11:00:00Z",
            }
        )
    )
    good = auth.outside_probe(path, nlb_hostname="NLB.example.internal", now=now)
    assert good["valid"] is True, good
    assert len(good["sha256"]) == 64

    wrong_host = auth.outside_probe(path, nlb_hostname="other.internal", now=now)
    assert wrong_host["valid"] is False
    assert "target_host" in " ".join(wrong_host["errors"])

    stale = auth.outside_probe(
        path, nlb_hostname="nlb.example.internal", now=now + timedelta(days=2)
    )
    assert stale["valid"] is False
    assert any("24 hours" in item for item in stale["errors"]), stale


def test_world_open_rules_include_ipv6_and_missing_nlb_is_readable() -> None:
    groups = [
        {
            "GroupId": "sg-1",
            "IpPermissions": [
                {
                    "FromPort": 443,
                    "ToPort": 443,
                    "IpRanges": [{"CidrIp": "10.0.0.0/8"}],
                    "Ipv6Ranges": [{"CidrIpv6": "::/0"}],
                }
            ],
        }
    ]
    broad = auth.world_open_rules(groups)
    assert broad == [
        {"group_id": "sg-1", "from_port": 443, "to_port": 443, "sources": ["::/0"]}
    ]
    with pytest.raises(common.IdentityAcceptanceError, match="no ELBv2 load balancer"):
        auth.select_load_balancer([{"DNSName": "other"}], "nlb.example")


def test_describe_all_load_balancers_follows_next_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = iter(
        [
            {"LoadBalancers": [{"DNSName": "one"}], "NextMarker": "m1"},
            {"LoadBalancers": [{"DNSName": "two"}]},
        ]
    )
    markers: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        markers.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(next(pages)), "")

    monkeypatch.setattr(auth, "run", run)
    result = auth.describe_all_load_balancers("us-west-2")
    assert [item["DNSName"] for item in result] == ["one", "two"]
    assert "--marker" in markers[1] and "m1" in markers[1]


def test_execution_token_hits_compare_values_not_only_key_names() -> None:
    token = "super-secret-token"
    digest = hashlib.sha256(token.encode()).hexdigest()
    secrets_document = {
        "items": [
            {
                "metadata": {"name": "innocent"},
                "data": {"config": base64.b64encode(token.encode()).decode()},
            }
        ]
    }
    pods_document = {
        "items": [
            {
                "metadata": {"name": "pod-1"},
                "spec": {
                    "containers": [
                        {"env": [{"name": "GPU_FAULT_EXECUTION_TOKEN", "value": "x"}]}
                    ]
                },
            }
        ]
    }
    hits = auth.execution_token_hits(secrets_document, pods_document, digests={digest})
    assert {
        "kind": "Secret",
        "name": "innocent",
        "key": "config",
        "match": "value",
    } in hits
    assert any(item["kind"] == "Pod" and item["match"] == "name" for item in hits), hits
    for item in hits:
        assert token not in json.dumps(item)


def test_auth010_records_that_it_is_superseded() -> None:
    source = Path(auth.__file__).read_text(encoding="utf-8")
    assert '"superseded_by": "GF-REGIONAL-AUTH-014"' in source
    assert '"status": "superseded"' in source


def test_certificate_alert_checks_read_the_node_timer_not_the_site_file() -> None:
    assert auth.certificate_alert_checks(None, threshold_days=30) == {
        "expiry_threshold_configured": "NOT_EVALUATED"
    }
    armed = {
        "min_validity_seconds": 30 * 86400,
        "timer_enabled": True,
        "timer_active": True,
    }
    assert auth.certificate_alert_checks(armed, threshold_days=30) == {
        "expiry_threshold_configured": True
    }
    disabled = {**armed, "timer_enabled": False}
    assert auth.certificate_alert_checks(disabled, threshold_days=30) == {
        "expiry_threshold_configured": False
    }
    short = {**armed, "min_validity_seconds": 86400}
    assert auth.certificate_alert_checks(short, threshold_days=30) == {
        "expiry_threshold_configured": False
    }


def test_tls_probe_accepts_any_ssl_or_os_error_and_records_default_handshake() -> None:
    probe = auth.TLS_BOUNDARY_PROBE
    assert (
        "except (ssl.SSLError, OSError) as exc:\n    wrong_hostname_rejected = True"
        in probe
    )
    assert '"default_handshake_ok": default_handshake_ok' in probe
    assert "except ssl.SSLCertVerificationError" not in probe


def test_auth013_probe_reads_only_the_threshold_key() -> None:
    text = "GPU_FAULT_CONTROL_PLANE_TOKEN=secret\nGPU_FAULT_CERTIFICATE_MIN_VALIDITY_SECONDS='2592000'\n"
    assert cert_probe.threshold_seconds(text) == 2592000
    assert cert_probe.threshold_seconds("OTHER=1\n") is None
    source = Path(cert_probe.__file__).read_text(encoding="utf-8")
    assert "print(json.dumps(result" in source
    assert "read_text" in source and "CONTROL_PLANE_TOKEN" not in source


def test_master_reference_scan_says_when_no_installer_was_present() -> None:
    idle = auth.master_reference_scan(
        {"items": [{"kind": "Pod", "metadata": {"name": "exec"}}]}
    )
    assert idle == {"hits": [], "installer_resources": []}
    installer = auth.master_reference_scan(
        {
            "items": [
                {
                    "kind": "Job",
                    "metadata": {"name": "gpu-fault-installer-abc"},
                    "spec": {
                        "env": [
                            {
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "gpu-fault-control-plane-active",
                                        "key": "fleet-master",
                                    }
                                }
                            }
                        ]
                    },
                }
            ]
        }
    )
    assert installer["installer_resources"] == ["Job/gpu-fault-installer-abc"]
    assert installer["hits"][0]["key"] == "fleet-master"


def test_heartbeat_advanced_compares_timestamps() -> None:
    before = {"last_heartbeat_at": "2026-09-07T10:00:00+00:00"}
    after = {"last_heartbeat_at": "2026-09-07T10:00:30+00:00"}
    assert auth.heartbeat_advanced(before, after) is True
    assert auth.heartbeat_advanced(after, before) is False
    assert auth.heartbeat_advanced({}, after) is False


def test_iso004_asserts_the_missing_agent_reason_in_one_process() -> None:
    assert iso.MISSING_AGENT_REASON == "expected one matching agent, found 0"
    assert "for requested in sys.argv[1:]:" in iso.SPARE_HEALTH_PROBE
    source = Path(iso.__file__).read_text(encoding="utf-8")
    assert '"missing_node_reason_is_precise"' in source


def test_iso003_records_the_vacuous_check_as_not_evaluated() -> None:
    source = Path(iso.__file__).read_text(encoding="utf-8")
    start = source.index("def run_iso003")
    end = source.index("SPARE_HEALTH_PROBE")
    body = source[start:end]
    assert '"secondary_agent_state_unchanged": before == after' not in body
    assert '"not_evaluated"' in body and "n/a" in body


# --------------------------------------------------------------------------- #
# 5. ISO-005 command retirement
# --------------------------------------------------------------------------- #
def test_iso005_retires_commands_settle_cancel_then_delete() -> None:
    probe = iso.REMOTE_COMMAND_RETIRE_PROBE
    assert probe.index("get_remote_command") < probe.index("cancel_remote_command")
    assert probe.index("cancel_remote_command") < probe.index("_delete(")
    assert 'if value["status"] == "LEASED":' in probe, "a LEASED command is deleted"
    source = Path(iso.__file__).read_text(encoding="utf-8")
    finally_index = source.index("    finally:\n\n        def retire_commands")
    assert (
        source.index(
            'write_json_atomic(case_dir / "iso005-details.json"', finally_index
        )
        > 0
    )
    assert '"executor_completion"' in source
    assert (
        "ALLOWLIST_WORKFLOW_PROBE,\n            target.cluster_id,\n            suffix,"
        in source
    )
    assert source.count("attempts=1") >= 3


# --------------------------------------------------------------------------- #
# 1. ISO-006 actually cuts the executor and judges the catalog's expectations
# --------------------------------------------------------------------------- #
def _fake_probe_run(
    calls: list[list[str]],
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(
        command: list[str], *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return run


def test_network_probe_blocks_output_and_forward_and_restores_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(cluster_network_probe, "run", _fake_probe_run(calls))
    chain = cluster_network_probe.chain_name("iso006-run-a")

    cluster_network_probe.block(
        argparse.Namespace(
            run_id="iso006-run-a",
            control_plane_cidr=["10.0.0.0/24"],
            restore_seconds=180,
        )
    )
    inserts = [command for command in calls if command[:2] == ["iptables", "-I"]]
    assert [command[2] for command in inserts] == ["OUTPUT", "FORWARD"], inserts
    timer = next(command for command in calls if command[0] == "systemd-run")
    script = timer[-1]
    assert f"iptables -D OUTPUT -j {chain}" in script
    assert f"iptables -D FORWARD -j {chain}" in script

    calls.clear()
    cluster_network_probe.unblock(argparse.Namespace(run_id="iso006-run-a"))
    deletes = [command for command in calls if command[:2] == ["iptables", "-D"]]
    assert [command[2] for command in deletes] == ["OUTPUT", "FORWARD"], deletes


def test_iso006_cut_must_be_proven_at_the_transport() -> None:
    assert iso006.cut_proven({"status": None, "transport_error": "URLError"}) is True
    assert iso006.cut_proven({"status": 200, "latency_seconds": 0.1}) is False
    assert iso006.cut_proven({"status": 403}) is False


def test_iso006_latency_judgment_is_p95_within_twenty_percent_and_no_5xx() -> None:
    baseline = [{"status": 200, "latency_seconds": 0.10} for _ in range(10)]
    fine = [{"status": 200, "latency_seconds": 0.11} for _ in range(10)]
    slow = [{"status": 200, "latency_seconds": 0.13} for _ in range(10)]
    assert iso006.latency_errors(baseline, fine) == []
    assert any("p95 rose" in item for item in iso006.latency_errors(baseline, slow)), (
        "a 30% p95 increase passed the 20% bound"
    )
    assert any(
        "returned 503" in item
        for item in iso006.latency_errors(
            baseline, [*fine, {"status": 503, "latency_seconds": 0.1}]
        )
    ), "a 5xx during the block was not an error"
    assert iso006.latency_errors([], fine) == [
        "claim latency could not be measured in both phases"
    ]


def test_iso006_pressure_and_restart_judgments() -> None:
    before = {"cluster_queue_depth": 0.0, "rejections": {"x": 1.0}}
    assert iso006.pressure_errors(before, {"rejections": {"x": 1.0}}) == []
    assert iso006.pressure_errors(before, {"rejections": {"x": 2.0}}) == [
        "x increased during the block"
    ]
    earlier = {"api-1": {"uid": "u1", "containers": {"api": {"restart_count": 0}}}}
    restarted = {
        "api-1": {
            "uid": "u1",
            "containers": {
                "api": {"restart_count": 1, "last_terminated_reason": "OOMKilled"}
            },
        }
    }
    errors = multi.container_status_errors(earlier, restarted)
    assert errors == ["api-1/api: restartCount changed", "api-1/api: OOMKilled"]


def test_iso006_defaults_and_restore_timer_cover_the_window(tmp_path: Path) -> None:
    parsed = iso006.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--cluster-a",
            "a",
            "--gpu-a-kubeconfig",
            "/tmp/a",
            "--gpu-a-context",
            "ca",
            "--cluster-b",
            "b",
            "--gpu-b-kubeconfig",
            "/tmp/b",
            "--gpu-b-context",
            "cb",
        ]
    )
    assert parsed.duration_seconds == 900
    for name in ("cpu", "a", "b"):
        (tmp_path / f"{name}.kubeconfig").write_text("apiVersion: v1\n")
    settings = iso006.Settings(
        multi=multi.MultiClusterSettings(
            cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
            namespace="gpu-fault-system",
            region="us-west-2",
            cluster_a=multi.ClusterTarget("a", tmp_path / "a.kubeconfig", "ca"),
            cluster_b=multi.ClusterTarget("b", tmp_path / "b.kubeconfig", "cb"),
        ),
        host_probe_image="img@sha256:abc",
        control_plane_cidrs=("10.0.0.0/24",),
        duration_seconds=900,
        predecessor_path=tmp_path / "p.json",
    )
    assert settings.restore_seconds == 900 + iso006.RESTORE_MARGIN_SECONDS
    source = Path(str(iso006.__file__)).read_text(encoding="utf-8")
    assert "cpu_blast_snapshot() != preflight" not in source
    assert "pool.map(lambda probe: probe.create(), probes)" in source
    assert "if not cut_proven(cut):" in source
    assert '"validation_limitations": VALIDATION_LIMITATIONS' in source


def test_metrics_reading_extracts_cluster_depth_and_rejections() -> None:
    text = "\n".join(
        [
            "# HELP x",
            'gpu_fault_processor_cluster_queue_depth{cluster_id="b"} 4',
            'gpu_fault_processor_cluster_queue_depth{cluster_id="a"} 1',
            'gpu_fault_store_io_rejections_total{reason="capacity"} 2',
            'gpu_fault_store_io_rejections_total{reason="backend_unavailable"} 3',
            "gpu_fault_ingress_backpressure_rejections_total 0",
        ]
    )
    reading = multi.cluster_pressure_reading(text, "b")
    assert reading["cluster_queue_depth"] == 4.0
    assert reading["rejections"]["gpu_fault_store_io_rejections_total"] == 5.0


# --------------------------------------------------------------------------- #
# 6. E2E-002
# --------------------------------------------------------------------------- #
def test_e2e002_follows_iso001_and_runs_the_clusters_in_parallel() -> None:
    assert e2e002.PREDECESSOR_CASE_ID == "GF-REGIONAL-ISO-001"
    source = Path(e2e002.__file__).read_text(encoding="utf-8")
    assert "prepared = list(pool.map(prepare, range(len(targets))))" in source
    assert "lambda index: settle(index, injected_at, payloads[index])" in source


def test_e2e002_notification_and_command_scope_judgments() -> None:
    registrations = [
        {"cluster_id": "a", "hyperpod_cluster_name": "hp-a"},
        {"cluster_id": "b", "hyperpod_cluster_name": "hp-b"},
    ]
    states = [
        {
            "incident": {"incident_id": "inc-a", "cluster_id": "a"},
            "notifications": [
                {
                    "notification": {
                        "notification_id": "n1",
                        "incident_id": "inc-a",
                        "cluster_name": "hp-a",
                    }
                }
            ],
            "commands": [{"cluster_id": "a", "last_lease_owner": "exec-a"}],
        },
        {
            "incident": {"incident_id": "inc-b", "cluster_id": "b"},
            "notifications": [
                {
                    "notification": {
                        "notification_id": "n2",
                        "incident_id": "inc-b",
                        "cluster_name": "hp-a",
                    }
                }
            ],
            "commands": [{"cluster_id": "b", "last_lease_owner": "exec-a"}],
        },
    ]
    assert e2e002.notification_errors(states, ["a", "b"], registrations) == [
        "b: notification cluster_name is foreign"
    ]
    errors = e2e002.command_scope_errors(
        states, ["a", "b"], {"a": ["exec-a"], "b": ["exec-b"]}
    )
    assert "b: command leased by unknown executor 'exec-a'" in errors
    assert "one executor identity leased commands in both clusters" in errors
    assert e2e002.command_scope_errors(
        [{"incident": {}, "commands": []}], ["a"], {}
    ) == ["a: no remote command was recorded"]


# --------------------------------------------------------------------------- #
# 7. audit_auth_boundary evidence and negative store assertions
# --------------------------------------------------------------------------- #
def _matrix_results() -> dict[str, Any]:
    statuses = {
        "AUTH-001": 401,
        "AUTH-002-no-auth": 401,
        "AUTH-002-basic": 401,
        "AUTH-002-empty-bearer": 403,
        "AUTH-003": 403,
        "AUTH-004-zero": 403,
        "AUTH-004-near": 403,
        "AUTH-005": 403,
        "AUTH-006": 403,
        "AUTH-008-A-normal": 200,
        "AUTH-008-A-fake-executor": 200,
        "AUTH-008-B-header-A-token": 403,
        "AUTH-008-A-header-B-token": 403,
        "AUTH-009 /v1/gpu-events/xid": 403,
        "AUTH-011-health": 200,
        "AUTH-011-metrics": 403,
        "AUTH-011-clusters-anon": 403,
        "AUTH-011-clusters-cluster-token": 403,
    }
    results: dict[str, Any] = {
        name: {"status": status, "body": {}} for name, status in statuses.items()
    }
    for name in ("AUTH-004-zero", "AUTH-004-near"):
        results[name]["body"]["detail"] = "regional cluster authentication failed"
    return results


def test_audit_writes_one_verdict_per_case_and_needs_the_store_for_negatives() -> None:
    results = _matrix_results()
    documents = audit.case_documents(
        results,
        cluster_a="cluster-a",
        store_errors=None,
        identity={"cluster_id": "cluster-a"},
    )
    assert set(documents) == set(audit.CASE_ENTRIES)
    assert documents["GF-REGIONAL-AUTH-001"]["verdict"] == "PASS"
    assert documents["GF-REGIONAL-AUTH-005"]["verdict"] == "FAIL", (
        "a negative-store case passed without reading the store"
    )
    assert "store_unchanged" in documents["GF-REGIONAL-AUTH-005"]["not_evaluated"]
    with_store = audit.case_documents(
        results,
        cluster_a="cluster-a",
        store_errors={},
        identity={"cluster_id": "cluster-a", "release_id": "r1"},
    )
    assert with_store["GF-REGIONAL-AUTH-005"]["verdict"] == "PASS"
    assert with_store["GF-REGIONAL-AUTH-008"]["checks"]["probe_claims_leased_nothing"]
    assert with_store["GF-REGIONAL-AUTH-009"]["release_id"] == "r1"
    assert with_store["GF-REGIONAL-AUTH-009"]["schema_version"] == 2


def test_audit_store_negative_errors_catch_new_records_and_leased_commands() -> None:
    before = {
        "clusters": {"b": {"agent_generations": ["n1:1"], "attempt_observations": 2}},
        "commands": {
            "c1": {"cluster_id": "a", "status": "PENDING", "lease_owner": None}
        },
    }
    after = {
        "clusters": {
            "b": {"agent_generations": ["n1:1", "probe:1"], "attempt_observations": 3}
        },
        "commands": {
            "c1": {"cluster_id": "a", "status": "LEASED", "lease_owner": "probe"}
        },
    }
    errors = audit.store_negative_errors(before, after, cluster_a="a", cluster_b="b")
    assert set(errors) == {
        "GF-REGIONAL-AUTH-005",
        "GF-REGIONAL-AUTH-006",
        "GF-REGIONAL-AUTH-008",
        "GF-REGIONAL-AUTH-009",
    }
    assert (
        audit.store_negative_errors(before, before, cluster_a="a", cluster_b="b") == {}
    )


def test_audit_drops_probe_clusters_mode_and_redacts_lease_tokens() -> None:
    with pytest.raises(SystemExit):
        audit.parser().parse_args(
            [
                "probe-clusters",
                "--url",
                "https://x",
                "--ca-file",
                "/tmp/ca",
                "--cluster-a",
                "a",
                "--token-a-file",
                "/tmp/a",
                "--cluster-b",
                "b",
                "--token-b-file",
                "/tmp/b",
                "--executor-artifact-sha256",
                "a" * 64,
                "--executor-compatibility-digest",
                "b" * 64,
            ]
        )
    assert audit.redact_body(
        {"commands": [{"lease_token": "x", "command_id": "c"}]}
    ) == {"commands": [{"command_id": "c"}]}
    assert not hasattr(audit, "probe_clusters"), "probe-clusters mode still exists"


# --------------------------------------------------------------------------- #
# 8./9. efficiency and evidence identity
# --------------------------------------------------------------------------- #
def test_auth015_creates_probes_in_parallel_and_checks_the_node_set() -> None:
    source = Path(auth.__file__).read_text(encoding="utf-8")
    assert "scans_before = dict(pool.map(create_and_scan, probes))" in source
    assert "does not cover exactly the GPU node set" in source
    assert '"cpu_secret_restored"' in source and '"node_b_heartbeat_advanced"' in source


def test_runners_bind_evidence_to_release_and_cluster() -> None:
    for module in (identity, iso006, e2e002):
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        assert "evidence_identity()" in source, module.__name__
    identity_source = Path(str(identity.__file__)).read_text(encoding="utf-8")
    assert "predecessor_evidence(path, predecessor_id, **identity)" in identity_source
    for module in (iso006, e2e002):
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        assert 'release_id=identity["release_id"]' in source, module.__name__
        assert 'record_focused_tests(details, preflight["focused_tests"])' in source
        assert "reuse_focused_tests=True" in source


def test_auth013_accepts_an_optional_node_only_with_a_digest_image() -> None:
    site: Any = SimpleNamespace(target=lambda cluster_id: _target("a"))
    base = dict(
        case="GF-REGIONAL-AUTH-013",
        cluster_id="",
        secondary_cluster_id="",
        fleet_master_file=None,
    )
    ok = argparse.Namespace(**base, node=["n1"], host_probe_image="img@sha256:abc")
    assert identity.validate_case_arguments(ok, site)[2] == ("n1",)
    without_image = argparse.Namespace(**base, node=["n1"], host_probe_image="")
    with pytest.raises(common.IdentityAcceptanceError, match="at most one --node"):
        identity.validate_case_arguments(without_image, site)
    none = argparse.Namespace(**base, node=[], host_probe_image="")
    assert identity.validate_case_arguments(none, site)[2] == ()
