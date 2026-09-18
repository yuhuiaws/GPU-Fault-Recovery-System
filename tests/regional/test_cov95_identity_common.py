from __future__ import annotations

import base64
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import identity_acceptance_common as common
from scripts.e2e.regional import identity_auth_checks as checks
from scripts.e2e.regional import identity_auth_sampling as sampling
from tests.regional._cov95_identity_support import Clock
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional._regional_support import TOKEN_A, registration
from tests.regional.test_multi_cluster_fixture_review import pod_document


def identity_site(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    count: int = 1,
    fallback: bool = False,
) -> common.IdentitySite:
    for name in ("cpu", "gpu"):
        (tmp_path / name).write_text("synthetic config", encoding="ascii")
    config = {
        "namespace": "gpu-system",
        "aws_region": "us-west-2",
        "cpu_kubeconfig": str(tmp_path / "cpu"),
        "clusters": [
            {
                "cluster_id": name,
                "context": "context-" + name,
                "region": "us-west-2",
                "hyperpod_cluster_name": "hp-" + name,
                "eks_cluster_arn": "arn:aws:eks:us-west-2:000000000000:cluster/" + name,
                "executor_irsa_role_arn": "arn:aws:iam::000000000000:role/" + name,
                "control_plane_url": "https://unit.invalid",
                "ca_file": str(tmp_path / "public-ca"),
            }
            for name in ("a", "b")[:count]
        ],
    }
    if not fallback:
        config["gpu_kubeconfig"] = str(tmp_path / "gpu")
    site = SimpleNamespace(
        release_config=config, environment={"KUBECONFIG": str(tmp_path / "gpu")}
    )
    monkeypatch.setattr(common, "load_site", lambda *args, **kwargs: site)
    return common.IdentitySite(tmp_path / "site")


@pytest.mark.parametrize("fallback", [False, True])
def test_identity_site_resolves_only_its_explicit_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fallback: bool
) -> None:
    site = identity_site(monkeypatch, tmp_path, count=2, fallback=fallback)
    assert site.target("a").context == "context-a"
    assert site.gpu_kubeconfig == tmp_path / "gpu"
    with pytest.raises(common.IdentityAcceptanceError, match="required"):
        site.target("")
    with pytest.raises(common.IdentityAcceptanceError, match="not present"):
        site.target("other")
    calls = []
    monkeypatch.setattr(
        common,
        "RegionalLiveFixture",
        lambda settings: calls.append(settings) or settings,
    )
    regional = site.regional(site.target("b"))
    assert regional.cluster_id == "b" and regional.gpu_context == "context-b"
    assert regional.cpu_kubeconfig == tmp_path / "cpu"
    assert len(calls) == 1


@pytest.mark.parametrize("defect", ["no-gpu", "no-targets"])
def test_identity_site_refuses_incomplete_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    site = identity_site(monkeypatch, tmp_path)
    config = copy.deepcopy(site.config)
    if defect == "no-gpu":
        config.pop("gpu_kubeconfig")
    else:
        config["clusters"] = []
    monkeypatch.setattr(
        common,
        "load_site",
        lambda *args, **kwargs: SimpleNamespace(release_config=config, environment={}),
    )
    with pytest.raises(
        common.IdentityAcceptanceError, match="GPU kubeconfig|no GPU clusters"
    ):
        common.IdentitySite(tmp_path / "site")


def test_identity_site_transport_preserves_the_selected_plane_and_probe_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site = identity_site(monkeypatch, tmp_path)
    target = site.target("")
    calls = []

    def kubectl(*args: str, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return json.dumps(pod_document()) if "get" in args else '{"status":200}'

    monkeypatch.setattr(
        site, "regional", lambda target: SimpleNamespace(kubectl=kubectl)
    )
    assert json.loads(site.cpu("get", "pods"))["items"]
    assert json.loads(site.gpu(target, "get", "pods"))["items"]
    assert site.ready_pods("gpu", common.EXECUTOR_APP, target) == ["api"]
    assert site.any_executor_pod(target) == "api"
    assert site.pod_json(
        "gpu", target, "api", "synthetic-probe", "argument", timeout=61
    ) == {"status": 200}
    assert calls[0][0][0] == "cpu" and calls[1][0][0] == "gpu"
    assert calls[-1][0][-1] == "argument"
    assert calls[-1][1] == {"input_text": "synthetic-probe", "timeout": 61}
    monkeypatch.setattr(site, "ready_pods", lambda *args: [])
    with pytest.raises(common.IdentityAcceptanceError, match="no Ready executor"):
        site.any_executor_pod(target)


def test_identity_pod_json_refuses_a_nonobject_protocol_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site = identity_site(monkeypatch, tmp_path)
    monkeypatch.setattr(
        site,
        "regional",
        lambda *args: SimpleNamespace(kubectl=lambda *args, **kwargs: "[]"),
    )
    with pytest.raises(common.IdentityAcceptanceError, match="JSON object"):
        site.pod_json("cpu", site.target("a"), "api", "probe")


@pytest.mark.parametrize(
    ("status", "expected"), [(200, 7), (404, None), (503, "error")]
)
def test_registry_generation_requires_an_exact_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int, expected: Any
) -> None:
    site = identity_site(monkeypatch, tmp_path)
    requests = []
    monkeypatch.setattr(
        site,
        "api_pod_json",
        lambda *args: requests.append(args)
        or {"status": status, "body": {"generation": 7}},
    )
    if expected == "error":
        with pytest.raises(common.IdentityAcceptanceError, match="status returned 503"):
            site.registry_generation()
    else:
        assert site.registry_generation() == expected
    assert requests[0][1:] == ("GET", "/v1/regional/registry/status")
    site.registry_api("POST", "/revisions", {"expected_generation": 7})
    assert json.loads(requests[-1][-1]) == {"expected_generation": 7}


@pytest.mark.parametrize("defect", ["none", "timeout"])
def test_registry_wait_requires_every_member_ack_and_a_fresh_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    site = identity_site(monkeypatch, tmp_path)
    clock = Clock()
    monkeypatch.setattr(common, "time", clock)
    samples = [
        {"status": 503, "body": {}},
        {"status": 200, "body": {"generation": 6, "missing_member_ids": []}},
        {"status": 200, "body": {"generation": 7, "missing_member_ids": ["member"]}},
        {"status": 200, "body": {"generation": 7}},
        {"status": 200, "body": {"generation": 7, "missing_member_ids": []}},
    ]
    calls = []

    def read(*args: str) -> dict[str, Any]:
        calls.append(args)
        return (
            samples.pop(0)
            if defect == "none"
            else {
                "status": 200,
                "body": {"generation": 7, "missing_member_ids": ["member"]},
            }
        )

    monkeypatch.setattr(site, "registry_api", read)
    if defect == "none":
        assert site.wait_registry_ready(7, timeout_seconds=30) == {
            "generation": 7,
            "missing_member_ids": [],
        }
        assert len(calls) == 5
    else:
        with pytest.raises(
            common.IdentityAcceptanceError, match="did not become ready"
        ):
            site.wait_registry_ready(7, timeout_seconds=5)
        assert calls, "registry timeout must observe the unacknowledged generation"


@pytest.mark.parametrize("durable", [False, True])
def test_registry_reads_and_writes_use_the_selected_storage_protocol(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, durable: bool
) -> None:
    site = identity_site(monkeypatch, tmp_path)
    rows = [registration("a", TOKEN_A).model_dump(mode="json")]
    calls = []
    monkeypatch.setattr(site, "registry_generation", lambda: 2 if durable else None)
    monkeypatch.setattr(site, "api_pod_json", lambda *args: {"registrations": rows})
    monkeypatch.setattr(
        site,
        "cpu",
        lambda *args, **kwargs: calls.append((args, kwargs))
        or json.dumps(
            {
                "data": {
                    "clusters.json": base64.b64encode(
                        json.dumps(rows).encode()
                    ).decode()
                }
            }
        ),
    )
    assert site.registry() == rows
    monkeypatch.setattr(
        site, "registry_api", lambda *args: calls.append(args) or {"status": 200}
    )
    monkeypatch.setattr(
        site,
        "wait_registry_ready",
        lambda generation: calls.append(("ready", generation)),
    )
    site.write_registry(rows)
    if durable:
        assert calls[-1] == ("ready", 3)
        assert calls[-2][0] == "POST"
        assert calls[-2][2]["expected_generation"] == 2
        assert calls[-2][2]["registrations"] == rows
    else:
        assert calls[-1][0][:2] == ("patch", "secret")
        assert json.loads(calls[-1][1]["input_text"]) == {
            "stringData": {"clusters.json": json.dumps(rows, separators=(",", ":"))}
        }
        assert site.last_registry_ready_seconds is None
        with pytest.raises(common.IdentityAcceptanceError, match="durable registry"):
            site.write_registry(rows, expected_entries=rows)


def test_registry_publish_failure_cannot_begin_the_readiness_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site = identity_site(monkeypatch, tmp_path)
    monkeypatch.setattr(site, "registry_generation", lambda: 2)
    monkeypatch.setattr(
        site,
        "registry_api",
        lambda *args: {"status": 409, "body": {"detail": "conflict"}},
    )
    waits = []
    monkeypatch.setattr(site, "wait_registry_ready", lambda *args: waits.append(args))
    with pytest.raises(common.IdentityAcceptanceError, match="publish returned 409"):
        site.write_registry([])
    assert waits == []


@pytest.mark.parametrize("changed", [False, True])
def test_registry_restore_is_idempotent_and_cas_bound_to_owned_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: bool
) -> None:
    site = identity_site(monkeypatch, tmp_path)
    original = [registration("a", TOKEN_A).model_dump(mode="json")]
    current = copy.deepcopy(original)
    if changed:
        current[0]["enabled"] = False
    site.last_registry_payload = copy.deepcopy(current)
    monkeypatch.setattr(site, "registry", lambda: current)
    writes = []
    monkeypatch.setattr(
        site, "write_registry", lambda *args, **kwargs: writes.append((args, kwargs))
    )
    site.restore_registry(original)
    assert len(writes) == int(changed)
    if changed:
        assert writes[0][1]["expected_entries"] == current
        assert writes[0][0] == (original,)


@pytest.mark.parametrize("durable", [False, True])
def test_control_rollout_is_noop_for_durable_registry_otherwise_waits_all_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, durable: bool
) -> None:
    site = identity_site(monkeypatch, tmp_path)
    monkeypatch.setattr(site, "registry_generation", lambda: 1 if durable else None)
    monkeypatch.setattr(common, "time", Clock())
    calls = []
    monkeypatch.setattr(
        site, "cpu", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    assert site.rollout_control() > 0
    if durable:
        assert calls == []
    else:
        assert [args[:2] for args, _ in calls] == [("rollout", "restart")] * 2 + [
            ("rollout", "status")
        ] * 2
        assert {args[2] for args, _ in calls} == {
            f"deployment/{app}" for app in common.CPU_APPS
        }
        assert calls[-1][1] == {"timeout": 660}


@pytest.mark.parametrize("missing", [False, True])
def test_api_pod_cache_requires_readiness_and_reuses_only_successful_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: bool
) -> None:
    site = identity_site(monkeypatch, tmp_path)
    selections = []
    monkeypatch.setattr(
        site,
        "ready_pods",
        lambda *args: selections.append(args) or ([] if missing else ["api"]),
    )
    requests = []
    monkeypatch.setattr(
        site, "pod_json", lambda *args: requests.append(args) or {"ok": True}
    )
    if missing:
        with pytest.raises(
            common.IdentityAcceptanceError, match="no Ready control-plane"
        ):
            site.api_pod_json("probe")
        assert requests == []
    else:
        assert site.api_pod_json("probe") == site.api_pod_json("probe") == {"ok": True}
        assert len(selections) == 1 and len(requests) == 2


def test_identity_helpers_delegate_without_changing_the_credential_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site = identity_site(monkeypatch, tmp_path)
    target = site.target("a")
    calls = []
    monkeypatch.setattr(site, "any_executor_pod", lambda *args: "executor")
    monkeypatch.setattr(
        site, "pod_json", lambda *args: calls.append(args) or {"status": 200}
    )
    assert common.claim(site, target) == {"status": 200}
    assert calls[0][:3] == ("gpu", target, "executor")
    value = base64.b64encode(b" example-only ").decode()
    monkeypatch.setattr(
        site,
        "gpu",
        lambda *args, **kwargs: calls.append((args, kwargs))
        or json.dumps({"data": {"cluster-token": value}}),
    )
    assert common.read_cluster_token(site, target) == "example-only"
    monkeypatch.setattr(common, "time", Clock())
    assert common.rollout_executor(site, target) > 0
    assert calls[-2][0][1:3] == ("rollout", "restart")
    assert calls[-1][0][1:3] == ("rollout", "status")
    forwarded = []
    monkeypatch.setattr(
        common,
        "run_fixture_command",
        lambda *args, **kwargs: forwarded.append((args, kwargs)) or "completed",
    )
    assert (
        common.run(
            ["unit"], input_text="input", check=False, timeout=9, cwd=tmp_path, env={}
        )
        == "completed"
    )
    assert forwarded == [
        (
            (["unit"],),
            {
                "input_text": "input",
                "check": False,
                "timeout": 9,
                "cwd": tmp_path,
                "env": {},
            },
        )
    ]


def test_token_digest_helpers_preserve_peer_rows_and_refuse_missing_targets() -> None:
    rows = [
        {"cluster_id": "peer", "token_sha256": "b" * 64},
        {"cluster_id": "a", "token": "example-only"},
    ]
    before = copy.deepcopy(rows)
    assert checks.registry_token_digests(rows, "a") == (
        common.secret_digest("example-only"),
        None,
        None,
    )
    updated = checks.update_registry_token(rows, "a", "example-new")
    assert updated[0] == rows[0]
    assert rows == before
    assert updated[1]["token"] == "example-new"
    with pytest.raises(common.IdentityAcceptanceError, match="absent"):
        checks.registry_token_digests(rows, "other")
    with pytest.raises(common.IdentityAcceptanceError, match="absent"):
        checks.update_registry_token(rows, "other", "example-new")


def test_execution_token_scan_handles_nonmatching_and_malformed_values() -> None:
    secrets = {
        "items": [
            {
                "metadata": {"name": "unit"},
                "data": {
                    "ordinary": base64.b64encode(b"ordinary").decode(),
                    "execution-token": base64.b64encode(b"named-placeholder").decode(),
                },
            }
        ]
    }
    pods = {
        "items": [
            {
                "metadata": {"name": "pod"},
                "spec": {
                    "containers": [
                        {
                            "env": [
                                {"name": "ordinary", "valueFrom": {}},
                                {"name": "safe", "value": "example-safe"},
                            ]
                        }
                    ]
                },
            }
        ]
    }
    assert checks.execution_token_hits(secrets, pods, digests={"a" * 64}) == [
        {"kind": "Secret", "name": "unit", "key": "execution-token", "match": "name"}
    ]
    secrets["items"][0]["data"] = {"ordinary": "a"}
    with pytest.raises(common.IdentityAcceptanceError, match="encoding is invalid"):
        checks.execution_token_hits(secrets, {"items": []}, digests=set())


def test_key_rotation_requires_the_exact_changed_key_set() -> None:
    document = {
        "data": {
            "node-a": base64.b64encode(b"old-a").decode(),
            "node-b": base64.b64encode(b"old-b").decode(),
        }
    }
    before = checks.node_key_digests(document)
    after = copy.deepcopy(document)
    after["data"]["node-a"] = base64.b64encode(b"new-a").decode()
    assert (
        checks.validated_node_key_rotation(before, before, after, after, "node-a")[
            "node-a"
        ]
        != before["node-a"]
    )
    with pytest.raises(common.IdentityAcceptanceError, match="unexpected key set"):
        checks.validated_node_key_rotation(before, before, document, after, "node-a")
    with pytest.raises(common.IdentityAcceptanceError, match="key data is missing"):
        checks.node_key_digests({})
    with pytest.raises(common.IdentityAcceptanceError, match="inventory is incomplete"):
        checks.master_reference_scan({})


def test_missing_route_inventory_is_not_proof_of_correct_auth_buckets() -> None:
    errors = checks.high_risk_route_errors([])
    assert len(errors) == 3
    assert all("not in the route inventory" in error for error in errors), errors
    assert (
        checks.world_open_rules(
            [
                {
                    "GroupId": "group",
                    "IpPermissions": [{"IpRanges": [{"CidrIp": "10.0.0.0/24"}]}],
                }
            ]
        )
        == []
    )


def test_sampler_stop_after_lock_snapshot_prevents_one_last_claim() -> None:
    samples = []
    sampler = sampling.TokenRotationSampler(
        "example-old", "example-new", lambda value: samples.append(value) or 200
    )
    checks_count = []

    def stopped() -> bool:
        checks_count.append(None)
        return len(checks_count) == 2

    sampler.stop = SimpleNamespace(is_set=stopped)
    sampler.run()
    assert len(checks_count) == 2
    assert samples == sampler.samples == []


def test_heartbeat_comparison_rejects_missing_or_invalid_time_fields() -> None:
    assert checks.heartbeat_advanced({}, {}) is False
    assert (
        checks.heartbeat_advanced(
            {"last_heartbeat_at": "invalid"},
            {"last_heartbeat_at": datetime.now(timezone.utc).isoformat()},
        )
        is False
    )
