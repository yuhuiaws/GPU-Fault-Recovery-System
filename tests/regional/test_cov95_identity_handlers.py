from __future__ import annotations

import base64
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import identity_acceptance_auth as auth
from scripts.e2e.regional import identity_acceptance_common as common
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional._regional_support import TOKEN_A, registration
from tests.regional.test_cov95_identity_audit import bound_api, probe


@pytest.mark.parametrize(
    "defect", ["none", "same-cluster", "bootstrap", "disabled", "missing", "bad-denial"]
)
def test_registration_isolation_proves_a_availability_and_restores_b(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    rows = [
        registration(name, name * 32).model_dump(mode="json") for name in ("a", "b")
    ]
    if defect == "disabled":
        rows[1]["enabled"] = False
    elif defect == "missing":
        rows.pop()
    original = copy.deepcopy(rows)
    events = []

    def write(entries: list[dict[str, Any]], **kwargs: Any) -> None:
        assert kwargs["expected_entries"] == original
        events.append("disable")
        rows[:] = copy.deepcopy(entries)

    def restore(entries: list[dict[str, Any]]) -> None:
        events.append("restore")
        rows[:] = copy.deepcopy(entries)

    def claim(site: Any, target: Any) -> dict[str, Any]:
        events.append("claim-" + target.cluster_id)
        enabled = next(
            item["enabled"] for item in rows if item["cluster_id"] == target.cluster_id
        )
        return (
            {"status": 200, "command_count": 0}
            if enabled
            else {
                "status": 403,
                "detail": "not-json"
                if defect == "bad-denial"
                else json.dumps({"detail": "regional cluster authentication failed"}),
            }
        )

    site = SimpleNamespace(
        registry=lambda: copy.deepcopy(rows),
        registry_generation=lambda: None if defect == "bootstrap" else 1,
        write_registry=write,
        restore_registry=restore,
        rollout_control=lambda: events.append("control-ack") or 0.1,
        last_registry_ready_seconds=None,
    )
    monkeypatch.setattr(auth, "claim", claim)
    a, b = (
        SimpleNamespace(cluster_id="a"),
        SimpleNamespace(cluster_id="a" if defect == "same-cluster" else "b"),
    )
    if defect in {"same-cluster", "bootstrap", "disabled", "missing"}:
        with pytest.raises(
            common.IdentityAcceptanceError, match="distinct|durable|enabled|absent"
        ):
            auth.run_auth007(site, a, b, case_dir=tmp_path)
        assert events == [] and rows == original
    else:
        result = auth.run_auth007(site, a, b, case_dir=tmp_path)
        assert result["verdict"] == ("FAIL" if defect == "bad-denial" else "PASS")
        assert rows == original
        assert events == [
            "disable",
            "control-ack",
            "claim-b",
            "claim-a",
            "restore",
            "control-ack",
            "claim-b",
        ]
        assert result["checks"]["primary_remains_200"] is True
        assert result["checks"]["registry_restored"] is True


def test_anonymous_identity_handler_executes_the_real_route_registry_and_auth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context, bridge = bound_api(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", context.execution_token)
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://unit.invalid")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_CA_FILE", str(tmp_path / "public-ca"))
    calls = []

    def pod_json(
        plane: str, target: Any, pod: str, script: str, *args: str, **kwargs: Any
    ) -> dict[str, Any]:
        calls.append((plane, pod))
        return probe(script, *args)

    site = SimpleNamespace(
        ready_pods=lambda *args: ["api"],
        any_executor_pod=lambda *args: "executor",
        pod_json=pod_json,
        api_pod_json=lambda script: probe(script),
        gpu=lambda *args, **kwargs: '{"items":[]}',
    )
    result = auth.run_auth010(site, SimpleNamespace(cluster_id="a"))
    assert result["verdict"] == "PASS"
    assert result["route_count"] > 80
    assert result["status"] == "superseded"
    assert result["superseded_by"] == "GF-REGIONAL-AUTH-014"
    assert calls == [("cpu", "api"), ("gpu", "executor")]
    assert all(item["status"] == 401 for item in result["cluster_token_results"]), (
        "anonymous cluster-token routes must reject before payload validation"
    )
    assert len(bridge.requests) == len(result["cluster_token_results"])
    assert context.store.list_remote_commands() == []


def test_route_inventory_requires_a_ready_api_pod() -> None:
    site = SimpleNamespace(ready_pods=lambda *args: [])
    with pytest.raises(common.IdentityAcceptanceError, match="no Ready"):
        auth.route_inventory(site, None)


def test_executor_claim_identity_reads_deployment_pins_and_deployed_owner_set() -> None:
    env = [
        {"name": "GPU_FAULT_EXECUTOR_ARTIFACT_SHA256", "value": "a" * 64},
        {"name": "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST", "value": "b" * 64},
        {"name": "IGNORED", "valueFrom": {}},
    ]
    calls = []
    site = SimpleNamespace(
        gpu=lambda *args: json.dumps(
            {"spec": {"template": {"spec": {"containers": [{"env": env}]}}}}
        ),
        any_executor_pod=lambda *args: "executor",
        pod_json=lambda *args: calls.append(args) or {"owners": ["local-owner"]},
    )
    identity = auth.executor_claim_identity(site, SimpleNamespace(cluster_id="a"))
    assert identity == {
        "artifact": "a" * 64,
        "compatibility": "b" * 64,
        "owners": ["local-owner"],
    }
    assert calls[0][0] == "gpu" and calls[0][2] == "executor"


def test_direct_claim_uses_current_protocol_without_leasing_real_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context, bridge = bound_api(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://unit.invalid")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_CA_FILE", str(tmp_path / "public-ca"))
    regional = SimpleNamespace(executor_python=lambda script, **kwargs: probe(script))
    status = auth.direct_claim(
        regional,
        SimpleNamespace(cluster_id="a"),
        token=TOKEN_A,
        identity={"artifact": "a" * 64, "compatibility": "b" * 64},
    )
    assert status == 200
    assert context.store.list_remote_commands() == []
    payload = json.loads(bridge.requests[0].data)
    assert payload["execution_owners"] == [common.ACCEPTANCE_PROBE_OWNER]
    assert payload["max_commands"] == 1


@pytest.mark.parametrize("protocol", [None, "2"])
def test_common_claim_probe_default_uses_current_deployed_protocol(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, protocol: str | None
) -> None:
    context, bridge = bound_api(monkeypatch)
    for name, value in {
        "GPU_FAULT_CONTROL_PLANE_URL": "https://unit.invalid",
        "GPU_FAULT_CONTROL_PLANE_CA_FILE": str(tmp_path / "public-ca"),
        "GPU_FAULT_CONTROL_PLANE_TOKEN": TOKEN_A,
        "GPU_FAULT_CLUSTER_ID": "a",
        "GPU_FAULT_EXECUTOR_ARTIFACT_SHA256": "a" * 64,
        "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST": "b" * 64,
    }.items():
        monkeypatch.setenv(name, value)
    if protocol is not None:
        monkeypatch.setenv("GPU_FAULT_EXECUTOR_PROTOCOL_VERSION", protocol)
    response = common.claim_sample(
        SimpleNamespace(executor_python=lambda script, **kwargs: probe(script))
    )
    if protocol is None:
        assert response["status"] == 200 and response["command_count"] == 0
    else:
        assert response["status"] == 503
        assert "protocol version mismatch" in response["detail"]
    assert len(bridge.requests) == 1
    assert context.store.list_remote_commands() == []


def test_direct_claim_keeps_unavailable_probe_as_a_failed_sample() -> None:
    def failed(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("synthetic unavailable executor")

    assert (
        auth.direct_claim(
            SimpleNamespace(executor_python=failed),
            SimpleNamespace(cluster_id="a"),
            token=TOKEN_A,
            identity={"artifact": "a" * 64, "compatibility": "b" * 64},
        )
        == "probe-unavailable:RuntimeError"
    )
    assert (
        auth.direct_claim(
            SimpleNamespace(
                executor_python=lambda *args, **kwargs: {"status": "unclassified"}
            ),
            SimpleNamespace(cluster_id="a"),
            token=TOKEN_A,
            identity={"artifact": "a" * 64, "compatibility": "b" * 64},
        )
        == "unclassified"
    )


@pytest.mark.parametrize("defect", ["none", "already-old", "registry", "token"])
def test_rotation_restore_never_rolls_an_executor_before_credential_restore(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    original = [{"cluster_id": "a", "token": "example-old"}]
    current = [{"cluster_id": "a", "token": "example-new"}]
    token = ["example-old" if defect == "already-old" else "example-new"]
    events = []

    def restore(*args: Any, **kwargs: Any) -> None:
        events.append("registry")
        if defect != "registry":
            current[:] = copy.deepcopy(original)

    def write(*args: Any, **kwargs: Any) -> None:
        events.append("token")
        assert kwargs == {"expected_token": "example-new", "expected_uid": "uid"}
        if defect == "token":
            raise common.IdentityAcceptanceError("synthetic token restore refused")
        token[0] = "example-old"

    site = SimpleNamespace(
        registry=lambda: current,
        restore_registry=restore,
        rollout_control=lambda: events.append("control"),
    )
    monkeypatch.setattr(auth, "read_cluster_token", lambda *args: token[0])
    monkeypatch.setattr(auth, "write_cluster_token", write)
    monkeypatch.setattr(
        auth, "rollout_executor", lambda *args: events.append("executor") or 0.1
    )
    outcomes, errors = auth.restore_auth016(
        site,
        SimpleNamespace(cluster_id="a"),
        thread=SimpleNamespace(is_alive=lambda: False),
        original_registry=original,
        old_token="example-old",
        new_token="example-new",
        connection_uid="uid",
        registry_started=True,
        token_started=True,
    )
    assert bool(errors) is (defect in {"registry", "token"})
    assert outcomes["verify_restored"] is (defect in {"none", "already-old"})
    if defect in {"registry", "token"}:
        assert "executor" not in events
    else:
        assert events[-1] == "executor"
        assert events.index("registry") < events.index("executor")
        if defect != "already-old":
            assert events.index("token") < events.index("executor")


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "not-json",
        "list",
        "unblocked",
        "no-host",
        "wrong-host",
        "invalid-time",
        "naive",
        "future",
        "stale",
    ],
)
def test_outside_denial_evidence_requires_host_time_and_literal_denial(
    tmp_path: Path, defect: str
) -> None:
    now = datetime.now(timezone.utc)
    value: Any = {
        "connection_blocked": True,
        "target_host": "nlb.invalid",
        "observed_at": now.isoformat(),
    }
    if defect == "list":
        value = []
    elif defect == "unblocked":
        value["connection_blocked"] = False
    elif defect == "wrong-host":
        value["target_host"] = "foreign.invalid"
    elif defect == "invalid-time":
        value["observed_at"] = "invalid"
    elif defect == "naive":
        value["observed_at"] = now.replace(tzinfo=None).isoformat()
    elif defect in {"future", "stale"}:
        value["observed_at"] = (
            now + timedelta(hours=1 if defect == "future" else -25)
        ).isoformat()
    path = tmp_path / "outside.json"
    path.write_text(
        "[" if defect == "not-json" else json.dumps(value), encoding="ascii"
    )
    result = auth.outside_probe(
        path, nlb_hostname="" if defect == "no-host" else "NLB.INVALID", now=now
    )
    assert result["valid"] is (defect == "none")
    if defect == "none":
        assert len(result["sha256"]) == 64
    assert auth.outside_probe(None)["valid"] is False


@pytest.mark.parametrize("groups", [False, True])
def test_nlb_probe_resolves_hostname_and_all_attached_security_groups(
    monkeypatch: pytest.MonkeyPatch, groups: bool
) -> None:
    calls = []
    site = SimpleNamespace(
        region="us-west-2",
        cpu=lambda *args: json.dumps(
            {"status": {"loadBalancer": {"ingress": [{"hostname": "NLB.INVALID"}]}}}
        ),
    )
    monkeypatch.setattr(
        auth,
        "describe_all_load_balancers",
        lambda region: [
            {"DNSName": "other.invalid"},
            {
                "DNSName": "nlb.invalid",
                "Scheme": "internal",
                "SecurityGroups": ["sg-unit"] if groups else [],
            },
        ],
    )
    group = {
        "GroupId": "sg-unit",
        "IpPermissions": [
            {
                "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                "Ipv6Ranges": [{"CidrIpv6": "::/0"}],
            }
        ],
    }
    monkeypatch.setattr(
        auth,
        "run",
        lambda command, **kwargs: calls.append(command)
        or SimpleNamespace(stdout=json.dumps({"SecurityGroups": [group]})),
    )
    result = auth.nlb_security_groups(site)
    assert result["security_group_count"] == int(groups)
    assert bool(result["broad_ipv4_rules"]) is groups
    assert bool(result["broad_ipv6_rules"]) is groups
    assert len(calls) == int(groups)
    if groups:
        assert calls[0][:3] == ["aws", "ec2", "describe-security-groups"]
    with pytest.raises(common.IdentityAcceptanceError, match="no hostname"):
        auth.nlb_security_groups(SimpleNamespace(cpu=lambda *args: "{}"))


@pytest.mark.parametrize(
    "value",
    [None, {}, {"uid": "other"}, {"namespace": "other"}, {"resourceVersion": ""}],
)
def test_node_key_restore_refuses_unknown_or_drifted_secret_identity(
    monkeypatch: pytest.MonkeyPatch, value: Any
) -> None:
    original = {
        "metadata": {
            "name": "keys",
            "uid": "uid",
            "namespace": "training",
            "resourceVersion": "1",
        },
        "data": {"node": "old"},
    }
    current = copy.deepcopy(original)
    if value is None:
        original["metadata"]["uid"] = ""
    elif value == {}:
        current["metadata"] = {}
    else:
        current["metadata"].update(value)
    monkeypatch.setattr(auth, "secret_document", lambda *args: current)
    with pytest.raises(common.IdentityAcceptanceError, match="identity changed"):
        auth.restore_secret(None, "gpu", None, original)


def test_node_key_restore_is_idempotent_without_rewriting_a_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = {
        "metadata": {
            "name": "keys",
            "uid": "uid",
            "namespace": "training",
            "resourceVersion": "1",
        },
        "data": {"node": "old"},
    }
    monkeypatch.setattr(auth, "secret_document", lambda *args: copy.deepcopy(original))
    auth.restore_secret(None, "gpu", None, original)
    current = copy.deepcopy(original)
    current["data"]["node"] = "unowned"
    monkeypatch.setattr(auth, "secret_document", lambda *args: current)
    with pytest.raises(common.IdentityAcceptanceError, match="ownership is unproven"):
        auth.restore_secret(None, "gpu", None, original)


@pytest.mark.parametrize("defect", ["nodes", "mode", "length", "cpu-key", "target-key"])
def test_auth015_prepare_refuses_unsafe_custody_and_missing_node_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    master = tmp_path / "unit-master"
    master.write_text("short" if defect == "length" else "m" * 64, encoding="ascii")
    master.chmod(0o644 if defect == "mode" else 0o600)
    data = {
        name: base64.b64encode((name * 8).encode()).decode()
        for name in ("node-a", "node-b")
    }
    calls = []

    def secret(*args: Any) -> dict[str, Any]:
        calls.append(args[1])
        return {
            "data": {
                **data,
                **(
                    {"node-a": base64.b64encode(b"other").decode()}
                    if args[1] == "cpu" and defect == "cpu-key"
                    else {}
                ),
            }
        }

    monkeypatch.setattr(auth, "secret_document", secret)
    monkeypatch.setattr(
        auth,
        "gpu_master_reference_scan",
        lambda *args: {"installer_resources": ["Job/installer"], "hits": []},
    )
    region = SimpleNamespace(gpu_nodes=lambda: [{"name": name} for name in data])
    site = SimpleNamespace(regional=lambda *args: region)
    nodes = (
        "node-a",
        "node-a"
        if defect == "nodes"
        else "node-c"
        if defect == "target-key"
        else "node-b",
    )
    with pytest.raises(
        common.IdentityAcceptanceError,
        match="distinct|0600|too short|do not agree|no key",
    ):
        auth.prepare_auth015(
            site,
            None,
            nodes=nodes,
            fleet_master_file=master,
            host_probe_image="unit@sha256:" + "a" * 64,
            case_dir=tmp_path,
        )
    if defect in {"nodes", "mode", "length"}:
        assert calls == []


@pytest.mark.parametrize("code", [0, 1])
def test_auth015_focused_signature_test_failure_is_not_discarded(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    calls = []
    monkeypatch.setattr(
        auth,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or SimpleNamespace(returncode=code),
    )
    result = auth.auth015_focused_tests()
    assert result["passed"] is (code == 0)
    assert result["returncode"] == code
    assert calls[0][1] == {"check": False, "timeout": 600}
