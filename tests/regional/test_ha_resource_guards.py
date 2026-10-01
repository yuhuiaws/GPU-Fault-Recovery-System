from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import executor_env_window as window
from scripts.e2e.regional import run_ha002_pdb_topology as ha002
from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004
from scripts.e2e.regional.ha_kubernetes import (
    NODE_OWNER,
    cordon_node_with_retry,
    delete_pod,
    node_cordon_patch,
    node_restore_patch,
)
from scripts.e2e.regional.ha_probe_resources import OwnedProbeResources
from scripts.e2e.regional.regional_commands import RegionalFixtureError


def apply_test_patch(document, operations):
    """Minimal fake API for the four JSON Patch operations these callers issue."""
    result = copy.deepcopy(document)
    for operation in operations:
        parts = [
            part.replace("~1", "/").replace("~0", "~")
            for part in operation["path"].split("/")[1:]
        ]
        parent = result
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        key = int(parts[-1]) if isinstance(parent, list) else parts[-1]
        if operation["op"] == "test":
            assert parent[key] == operation["value"], "API CAS rejected"
        elif operation["op"] in {"add", "replace"}:
            parent[key] = copy.deepcopy(operation["value"])
        elif operation["op"] == "remove":
            del parent[key]
        else:
            raise AssertionError("unexpected fake API patch operation")
    return result


class ProbeApi:
    def __init__(self):
        self.items = {}
        self.calls = []
        self.lose_ack = False
        self.read_fails = False

    def command(self, args, body):
        self.calls.append((args, body))
        if args[0] == "get":
            if self.read_fails:
                raise RuntimeError("unit unavailable")
            value = self.items.get((args[1], args[2]))
            return json.dumps(value) if value else ""
        if args[0] == "create":
            value = json.loads(body)
            value["metadata"]["uid"] = f"uid-{len(self.items)}"
            self.items[(value["kind"], value["metadata"]["name"])] = value
            if self.lose_ack:
                raise RuntimeError("unit lost ACK")
        elif args[0] == "delete":
            uid = json.loads(body)["preconditions"]["uid"]
            key = next(
                key
                for key, value in self.items.items()
                if value["metadata"]["uid"] == uid
            )
            del self.items[key]
        return ""


def pod_manifest():
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "probe", "namespace": "unit"},
    }


@pytest.mark.parametrize("lose_ack", [False, True])
def test_probe_cleanup_uses_creation_nonce_and_atomic_uid_even_after_lost_ack(
    lose_ack, tmp_path: Path
) -> None:
    api = ProbeApi()
    resources = OwnedProbeResources(tmp_path / "receipt.json", api.command)
    api.lose_ack = lose_ack
    if lose_ack:
        with pytest.raises(RuntimeError, match="ACK"):
            resources.create(pod_manifest())
    else:
        resources.create(pod_manifest())
    resources.delete("Pod", "probe")
    assert api.items == {}
    receipt = json.loads(resources.path.read_text())
    assert receipt["resources"]["Pod/probe"]["deleted"] is True
    deletions = [
        (args, json.loads(body)) for args, body in api.calls if args[0] == "delete"
    ]
    assert deletions[0][0][1:3] == ["--raw", "/api/v1/namespaces/unit/pods/probe"]
    assert deletions[0][1]["preconditions"]["uid"] == "uid-0"


@pytest.mark.parametrize("drift", ["uid", "owner", "namespace", "unavailable"])
def test_probe_cleanup_never_deletes_replacements_or_unknown_resources(
    drift, tmp_path
) -> None:
    api = ProbeApi()
    resources = OwnedProbeResources(tmp_path / "receipt.json", api.command)
    resources.create(pod_manifest())
    metadata = api.items[("Pod", "probe")]["metadata"]
    if drift == "uid":
        metadata["uid"] = "replacement"
    elif drift == "owner":
        metadata["annotations"] = {}
    elif drift == "namespace":
        metadata["namespace"] = "other"
    else:
        api.read_fails = True
    with pytest.raises(RuntimeError):
        resources.delete("Pod", "probe")
    assert not any(args[0] == "delete" for args, _ in api.calls), api.calls


def test_probe_creation_does_not_predelete_an_existing_name(tmp_path) -> None:
    api = ProbeApi()
    api.items[("Pod", "probe")] = {
        **pod_manifest(),
        "metadata": {"name": "probe", "namespace": "unit", "uid": "foreign"},
    }
    resources = OwnedProbeResources(tmp_path / "receipt.json", api.command)
    with pytest.raises(RegionalFixtureError, match="unused name"):
        resources.create(pod_manifest())
    assert all(args[0] == "get" for args, _ in api.calls), api.calls
    assert not resources.path.exists(), (
        f"refused creation must not write an ownership receipt: {resources.path}"
    )


def test_node_restore_cas_rejects_replacement_or_lost_ownership() -> None:
    baseline = {
        "uid": "node-uid",
        "resource_version": "1",
        "unschedulable": False,
        "ha_owner": None,
        "annotations_present": False,
        "taints": [],
        "taints_present": True,
    }
    node = {
        "metadata": {"uid": "node-uid", "resourceVersion": "1"},
        "spec": {"taints": []},
    }
    cordoned = apply_test_patch(
        node, node_cordon_patch(baseline, uid="node-uid", owner="owned")
    )
    assert cordoned["spec"]["unschedulable"] is True
    patch = node_restore_patch(baseline, owner="owned")
    restored = apply_test_patch(cordoned, patch)
    assert restored["spec"]["unschedulable"] is False
    assert NODE_OWNER not in restored["metadata"]["annotations"]
    for changed in (
        {**cordoned, "metadata": {**cordoned["metadata"], "uid": "replacement"}},
        {
            **cordoned,
            "metadata": {**cordoned["metadata"], "annotations": {NODE_OWNER: "other"}},
        },
        {**cordoned, "spec": {**cordoned["spec"], "taints": [{"key": "other"}]}},
    ):
        with pytest.raises(AssertionError, match="CAS rejected"):
            apply_test_patch(changed, patch)


def test_eviction_rejection_probe_is_uid_bound_and_server_dry_run(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        ha002.COMMON, "run", lambda args, **kw: calls.append((args, kw))
    )
    ha002.eviction("pod-a", uid="uid-a", dry_run=True)
    body = json.loads(calls[0][1]["stdin"])
    assert body["deleteOptions"] == {
        "preconditions": {"uid": "uid-a"},
        "dryRun": ["All"],
    }
    assert calls[0][1]["check"] is False
    with pytest.raises(RegionalFixtureError, match="UID"):
        delete_pod(lambda *a: calls.append(a), "ns", {"name": "pod-a"})
    assert len(calls) == 1


class WindowApi:
    def __init__(self, tmp_path):
        self.settings = SimpleNamespace(
            cpu_kubeconfig=tmp_path / "cpu",
            gpu_kubeconfig=tmp_path / "gpu",
            gpu_context="unit-context",
            namespace="unit",
            cluster_id="unit-cluster",
            region="unit-region",
            environment=lambda: {"scope": "unit"},
        )
        self.document = {
            "metadata": {
                "uid": "deployment-uid",
                "resourceVersion": "1",
                "generation": 1,
            },
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "executor",
                                "env": [
                                    {"name": ha004.LEASE_ENV, "value": "30"},
                                    {"name": ha004.POLL_ENV, "value": "5"},
                                    {"name": "UNRELATED", "value": "private-fixture"},
                                ],
                            }
                        ]
                    }
                },
            },
            "status": {
                "replicas": 1,
                "readyReplicas": 1,
                "updatedReplicas": 1,
                "availableReplicas": 1,
                "observedGeneration": 1,
            },
        }
        self.calls = []

    def evidence_identity(self):
        return {"release_id": "unit-release", "cluster_id": "unit-cluster"}

    def ready_pods(self, *args):
        return [
            {"name": f"pod-{i}", "uid": f"uid-{i}"}
            for i in range(self.document["spec"]["replicas"])
        ]

    def kubectl(self, plane, *args, **kwargs):
        self.calls.append((plane, args, kwargs))
        if args[0] == "get":
            return json.dumps(self.document)
        if args[0] == "exec":
            env = {
                item["name"]: item["value"]
                for item in self.document["spec"]["template"]["spec"]["containers"][0][
                    "env"
                ]
            }
            return json.dumps(
                {name: env.get(name) for name in window.ALLOWED_VARIABLES}
            )
        if args[0] == "patch":
            assert "--patch-file=/dev/stdin" in args
            assert "private-fixture" not in " ".join(args)
            self.document = apply_test_patch(
                self.document, json.loads(kwargs["input_text"])
            )
            metadata = self.document["metadata"]
            metadata["generation"] += 1
            metadata["resourceVersion"] = str(int(metadata["resourceVersion"]) + 1)
            self.document["status"] = {
                **{
                    field: self.document["spec"]["replicas"]
                    for field in (
                        "replicas",
                        "readyReplicas",
                        "updatedReplicas",
                        "availableReplicas",
                    )
                },
                "observedGeneration": metadata["generation"],
            }
        return ""


def test_ha004_uses_real_schema2_window_and_restores_with_uid_cas(
    monkeypatch, tmp_path
) -> None:
    api = WindowApi(tmp_path)
    monkeypatch.setattr(
        ha004.ExecutorTimingFixture, "start_watchdog", lambda self: None
    )
    fixture = ha004.ExecutorTimingFixture(api, tmp_path)
    baseline = copy.deepcopy(api.document["spec"])
    applied = fixture.apply()
    assert applied["replicas"] == ha004.TEST_REPLICAS
    receipt = json.loads(fixture.window.baseline.read_text())
    assert receipt["schema_version"] == 2
    assert receipt["baseline"]["uid"] == "deployment-uid"
    assert receipt["scope"]["identity"] == api.evidence_identity()
    restored = fixture.restore()
    assert restored["restore_errors"] == []
    assert api.document["spec"] == baseline
    assert all(
        "--patch-file=/dev/stdin" in args
        for _, args, _ in api.calls
        if args[0] == "patch"
    ), api.calls


def test_ha004_never_restores_a_recreated_deployment(monkeypatch, tmp_path) -> None:
    api = WindowApi(tmp_path)
    monkeypatch.setattr(
        ha004.ExecutorTimingFixture, "start_watchdog", lambda self: None
    )
    fixture = ha004.ExecutorTimingFixture(api, tmp_path)
    fixture.apply()
    api.document["metadata"]["uid"] = "replacement"
    before = len(api.calls)
    with pytest.raises(RegionalFixtureError, match="replacement"):
        fixture.restore()
    assert all(args[0] == "get" for _, args, _ in api.calls[before:]), api.calls[
        before:
    ]


def _cordon_snapshot(resource_version: str) -> dict[str, object]:
    return {
        "uid": "node-uid",
        "unschedulable": False,
        "ha_owner": None,
        "resource_version": resource_version,
        "annotations_present": True,
    }


class _PatchRejected(RuntimeError):
    pass


def test_cordon_retry_rebuilds_the_patch_after_a_resource_version_race() -> None:
    reads = iter(["100", "101", "101"])
    applied: list[list[dict[str, object]]] = []
    logs: list[str] = []

    def apply(patch: list[dict[str, object]]) -> None:
        applied.append(patch)
        if len(applied) == 1:
            raise _PatchRejected("the server rejected our request")

    snapshot, attempts = cordon_node_with_retry(
        read_node=lambda: _cordon_snapshot(next(reads)),
        apply_patch=apply,
        uid="node-uid",
        owner="owned",
        retryable=lambda exc: isinstance(exc, _PatchRejected),
        log=logs.append,
    )

    assert attempts == 2, "the second attempt must be the one that succeeded"
    assert snapshot["resource_version"] == "101", (
        "the successful patch must be built from the re-read node"
    )
    assert [
        op["value"]
        for p in applied
        for op in p
        if op["path"] == "/metadata/resourceVersion"
    ] == ["100", "101"], "each attempt must test the resourceVersion it was built from"
    assert logs and "resourceVersion race" in logs[0], "the retry must be logged"


def test_cordon_retry_re_raises_when_the_resource_version_did_not_move() -> None:
    def apply(_patch: list[dict[str, object]]) -> None:
        raise _PatchRejected("forbidden")

    with pytest.raises(_PatchRejected, match="forbidden"):
        cordon_node_with_retry(
            read_node=lambda: _cordon_snapshot("100"),
            apply_patch=apply,
            uid="node-uid",
            owner="owned",
            retryable=lambda exc: isinstance(exc, _PatchRejected),
        )


def test_cordon_retry_fails_closed_when_invariants_break_on_re_read() -> None:
    reads = iter(
        [
            _cordon_snapshot("100"),
            _cordon_snapshot("101"),
            {**_cordon_snapshot("101"), "ha_owner": "someone"},
        ]
    )

    def apply(_patch: list[dict[str, object]]) -> None:
        raise _PatchRejected("rejected")

    with pytest.raises(RegionalFixtureError, match="changed before cordon"):
        cordon_node_with_retry(
            read_node=lambda: next(reads),
            apply_patch=apply,
            uid="node-uid",
            owner="owned",
            retryable=lambda exc: isinstance(exc, _PatchRejected),
        )


def test_cordon_retry_budget_is_bounded_and_non_retryable_errors_pass_through() -> None:
    versions = iter(str(n) for n in range(100, 120))

    def always_rejected(_patch: list[dict[str, object]]) -> None:
        raise _PatchRejected("rejected")

    with pytest.raises(_PatchRejected):
        cordon_node_with_retry(
            read_node=lambda: _cordon_snapshot(next(versions)),
            apply_patch=always_rejected,
            uid="node-uid",
            owner="owned",
            retryable=lambda exc: isinstance(exc, _PatchRejected),
            attempts=3,
        )
    assert next(versions) == "105", "three attempts read the node at most five times"

    def other_error(_patch: list[dict[str, object]]) -> None:
        raise ValueError("not a kubectl failure")

    with pytest.raises(ValueError, match="not a kubectl"):
        cordon_node_with_retry(
            read_node=lambda: _cordon_snapshot("1"),
            apply_patch=other_error,
            uid="node-uid",
            owner="owned",
            retryable=lambda exc: isinstance(exc, _PatchRejected),
        )
    with pytest.raises(RegionalFixtureError, match="retry budget"):
        cordon_node_with_retry(
            read_node=lambda: _cordon_snapshot("1"),
            apply_patch=other_error,
            uid="node-uid",
            owner="owned",
            retryable=lambda exc: True,
            attempts=0,
        )
