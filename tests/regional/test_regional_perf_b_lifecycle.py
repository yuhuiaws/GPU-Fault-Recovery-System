from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from scripts.perf import regional_capacity_registry as registry
from scripts.perf import regional_capacity_suite as base
from scripts.perf import regional_correlated_action_suite as correlated
from scripts.perf import regional_integrated_workflow_capacity_suite as integrated
from scripts.perf.regional_capacity_resources import RunResources
from tests.regional._perf_caller_support import AmpWire


class FakeDataplane:
    def __init__(self, artifacts: Path, events: list[str]) -> None:
        self.artifacts = artifacts
        self.events = events
        self.objects: dict[str, dict[str, Any]] = {}
        self.created: dict[str, dict[str, Any]] = {}
        self.lose_create_ack: str | None = None

    def __call__(self, *args: str, stdin: bytes | None = None, **_kwargs: Any) -> str:
        if args[0] == "get":
            if args[1] in {"pod", "pods"}:
                return ""
            value = self.objects.get(f"{args[1]}/{args[2]}")
            if value is None:
                return ""
            output = args[args.index("-o") + 1]
            if output == "name":
                return f"{args[1]}/{args[2]}"
            return json.dumps(
                value["metadata"] if output == "jsonpath={.metadata}" else value
            )
        if args[0] == "wait":
            return "condition met"
        assert stdin is not None, f"{args[0]} must carry a structured request"
        value = json.loads(stdin)
        if args[0] == "create":
            metadata = value["metadata"]
            key = f"{value['kind'].lower()}/{metadata['name']}"
            receipt = json.loads(
                (self.artifacts / "capacity-resources.json").read_text()
            )
            assert receipt["resources"][key]["uid"] is None, (
                "intent must precede create"
            )
            assert metadata["labels"][registry.RUN_LABEL] == receipt["run_id"]
            assert key not in self.objects, "create must not overwrite a resource"
            metadata.update(uid=f"uid-{len(self.created)}", resourceVersion="1")
            if value["kind"] == "Job":
                value["status"] = {
                    "succeeded": value["spec"]["completions"],
                    "conditions": [{"type": "Complete", "status": "True"}],
                }
            self.objects[key] = value
            self.created[key] = copy.deepcopy(value)
            self.events.append(f"create:{key}")
            if self.lose_create_ack == key:
                raise TimeoutError("resource create ACK lost")
            return json.dumps(value)
        if args[0] == "patch":
            key = f"{args[1]}/{args[2]}"
            current = self.objects[key]
            assert value[:2] == [
                {
                    "op": "test",
                    "path": "/metadata/uid",
                    "value": current["metadata"]["uid"],
                },
                {
                    "op": "test",
                    "path": "/metadata/resourceVersion",
                    "value": current["metadata"]["resourceVersion"],
                },
            ], "start gate update must compare UID and resourceVersion"
            current["data"] = value[2]["value"]
            current["metadata"]["resourceVersion"] = "2"
            self.events.append(f"patch:{key}")
            return json.dumps(current)
        assert args[:2] == ("delete", "--raw"), (
            "cleanup must not delete by name or selector"
        )
        plural, name = args[2].rsplit("/", 2)[-2:]
        kind = {
            "jobs": "job",
            "configmaps": "configmap",
            "roles": "role",
            "rolebindings": "rolebinding",
            "secrets": "secret",
            "serviceaccounts": "serviceaccount",
        }[plural]
        key = f"{kind}/{name}"
        metadata = self.objects[key]["metadata"]
        assert value["preconditions"] == {
            "uid": metadata["uid"],
            "resourceVersion": metadata["resourceVersion"],
        }
        assert value["propagationPolicy"] == "Foreground"
        del self.objects[key]
        self.events.append(f"delete:{key}")
        return "{}"


class CallerHarness:
    def __init__(self, module: ModuleType, root: Path) -> None:
        self.module, self.root = module, root
        self.artifacts = root / "run"
        self.artifacts.mkdir()
        self.events: list[str] = []
        self.amp: AmpWire | None = None
        self.connection_reads: list[str] = []
        self.connection_missing = False
        self.api = FakeDataplane(self.artifacts, self.events)
        self.requests: list[dict[str, Any]] = []
        self.teardowns: list[dict[str, Any]] = []
        self.registration_error: BaseException | None = None
        self.persist_intent = True
        self.data_error: Exception | None = None
        self.data_result: dict[str, Any] = {"total": 0}
        self.replace_job = False
        self.args = argparse.Namespace(
            clusters=32,
            workflows_per_cluster=4,
            allow_live_registry=True,
            confirm_live_registry="ALLOW_PERF_CAPACITY_LIVE_REGISTRY",
            suite_id="run-a",
            artifact_root=root,
            label=None,
            spool_mode="enabled",
            timeout_seconds=180,
            scenario_max_seconds=60,
            terminal_drain_seconds=0,
            synthetic_ttl_seconds=3600,
            executor_idle_seconds=1,
            ingress_workers=1,
            executor_workers=1,
            lead_seconds=1,
            aurora_cluster_id="test-aurora",
            aurora_preflight_timeout_seconds=1,
            configure_aurora_capacity=True,
        )
        self.purge: Callable[..., None] = (
            integrated.purge_integrated_rows
            if module is integrated
            else correlated.purge_scenario_rows
        )

    def read_connection(self, namespace: str, *args: str, **_kwargs: Any) -> str:
        if args == ("get", "namespace", registry.NAMESPACE, "-o", "json"):
            # The perf namespace exists in this harness; the runner creates it
            # only when this read comes back empty.
            return json.dumps({"kind": "Namespace", "metadata": {"name": namespace}})
        assert args == ("get", "secret", registry.CONNECTION_SECRET, "-o", "json"), (
            "the connection fake permits only the selected Secret read"
        )
        assert self.amp is not None and not self.amp.calls, (
            "connection preflight must precede the AMP check"
        )
        self.connection_reads.append(namespace)
        if self.connection_missing:
            return ""
        return json.dumps(
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "type": "Opaque",
                "metadata": {
                    "name": registry.CONNECTION_SECRET,
                    "namespace": namespace,
                },
                "data": {
                    "ca.crt": "dGVzdC1jYQ==",
                    "cluster-token": "dGVzdC10b2tlbg==",
                    "control-plane-url": "aHR0cHM6Ly9wZXJmLmludmFsaWQ=",
                },
            }
        )

    def register(
        self, count: int, artifacts: Path, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.events.append("register")
        if self.persist_intent:
            (artifacts / "registry-registration-intent.json").write_text(
                json.dumps(
                    {
                        "run_id": kwargs["run_id"],
                        "cluster_ids": [
                            f"perf-cap-{index:03d}" for index in range(count)
                        ],
                        "synthetic_expires_at": kwargs["expires_at"].isoformat(),
                        "data_empty_before_registration": True,
                    }
                )
            )
            (artifacts / "registry-token-proof.json").write_text(
                json.dumps(
                    {
                        "run_id": kwargs["run_id"],
                        "uid": "token-uid",
                        "resource_version": "1",
                    }
                )
            )
            self.api.objects[f"secret/{registry.TOKEN_SECRET}"] = {
                "kind": "Secret",
                "metadata": {
                    "name": registry.TOKEN_SECRET,
                    "namespace": registry.NAMESPACE,
                    "uid": "token-uid",
                    "resourceVersion": "1",
                    "labels": {registry.RUN_LABEL: kwargs["run_id"]},
                },
            }
        if self.registration_error is not None:
            raise self.registration_error
        return []

    def control(self, *args: str, **kwargs: Any) -> str:
        assert kwargs.get("check", True) is True, (
            "cleanup transport must remain checked"
        )
        if args[0] == "get":
            return "api-a"
        if args[0] == "exec" and "ACTION_SEED_MODE=run-heartbeats" in args:
            # The run's synthetic Agent heartbeat refresh (seed script mode).
            self.events.append("heartbeat")
            return json.dumps({"run_id": "run-a", "agents_refreshed": 1})
        self.events.append("purge")
        self.requests.append(json.loads(args[-1]))
        if self.data_error is not None:
            raise self.data_error
        return json.dumps(self.data_result)

    def teardown(self, **kwargs: Any) -> None:
        self.teardowns.append(kwargs)
        self.events.append("teardown")
        if self.replace_job:
            key = next(key for key in self.api.objects if key.startswith("job/"))
            self.api.objects[key]["metadata"]["uid"] = "replacement-uid"
        base.teardown(**kwargs, attempts=1)

    def run(self) -> int:
        return (
            integrated.run(self.args)
            if self.module is integrated
            else correlated.main()
        )

    def report(self, name: str) -> dict[str, Any]:
        directory = (
            self.artifacts if self.artifacts.exists() else self.root / "_aborted/run"
        )
        result: dict[str, Any] = json.loads((directory / name).read_text())
        return result


@pytest.fixture(params=[integrated, correlated], ids=["integrated", "correlated"])
def caller(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> CallerHarness:
    harness = CallerHarness(request.param, tmp_path)
    module = harness.module

    def forbidden(*_args: Any, **_kwargs: Any) -> str:
        pytest.fail("unmocked external command")

    monkeypatch.setattr(registry, "run", forbidden)
    monkeypatch.setattr(base, "dataplane", harness.api)
    monkeypatch.setattr(base, "control", harness.control)
    monkeypatch.setattr(base, "validate_registry_target", lambda **_kwargs: "live")
    monkeypatch.setattr(
        base, "deregister", lambda **_kwargs: harness.events.append("revoke")
    )
    # The teardown's alignment gate now runs after the token Secret deletion,
    # outside deregister(); it is the registry module's own contract.
    monkeypatch.setattr(
        registry, "verify_registry_alignment", lambda **_kwargs: {"aligned": True}
    )
    monkeypatch.setattr(module, "dataplane", harness.api)
    monkeypatch.setattr(module, "control", harness.control)
    monkeypatch.setattr(module, "register", harness.register)
    monkeypatch.setattr(module, "teardown", harness.teardown)
    monkeypatch.setattr(module, "artifact_dir", lambda *_args: harness.artifacts)
    monkeypatch.setattr(module, "validate_registry_target", lambda **_kwargs: "live")
    monkeypatch.setattr(
        module, "release_identity", lambda: {"release_id": "test-release"}
    )
    monkeypatch.setattr(
        module,
        "executor_identity",
        lambda **_kwargs: {
            "executor_protocol_version": 2,
            "executor_artifact_sha256": "a" * 64,
            "executor_compatibility_digest": "b" * 64,
        },
    )
    monkeypatch.setattr(module, "verdict", lambda *_args, **_kwargs: ("PASS", []))
    monkeypatch.setattr(module, "collect_logs", lambda *_args: [])
    if module is correlated:
        monkeypatch.setattr(
            module.argparse.ArgumentParser, "parse_args", lambda *_args: harness.args
        )
        monkeypatch.setattr(module, "validate_preemption_enabled", lambda: None)
        monkeypatch.setattr(
            module,
            "seed_agents",
            lambda *_args, **_kwargs: {"runtime_profile_version": "test-profile"},
        )
        monkeypatch.setattr(
            module, "wait_for_terminal_drain", lambda *_args, **_kwargs: ({}, [])
        )
    else:
        harness.amp = AmpWire(monkeypatch)
        monkeypatch.setattr(
            registry,
            "dataplane_identity",
            partial(harness.read_connection, registry.IDENTITY_NAMESPACE),
        )
        monkeypatch.setattr(
            registry, "dataplane", partial(harness.read_connection, registry.NAMESPACE)
        )
        monkeypatch.setattr(module, "shell_run", forbidden)
        monkeypatch.setattr(module, "control_spool_mode", lambda _mode: {})
        monkeypatch.setattr(
            module, "ingress_process_model_preflight", lambda: {"valid": True}
        )
        monkeypatch.setattr(
            module,
            "ensure_aurora_capacity",
            lambda *_args, **_kwargs: {"instances": {"test-writer": {"writer": True}}},
        )
        monkeypatch.setattr(module, "remediation_budget_preflight", lambda *_args: {})
        monkeypatch.setattr(
            module,
            "seed_integrated_agents",
            lambda **_kwargs: {"runtime_profile_version": "test-profile"},
        )
        monkeypatch.setattr(
            module,
            "integrated_agent_heartbeat_refresher",
            lambda **_kwargs: lambda: None,
        )
        monkeypatch.setattr(module, "control_pods", lambda: [])
        # The post-teardown late-residue sweep probes the CPU Pod through the data
        # helper and paces its attempts; the unit harness answers "clean" at once.
        monkeypatch.setattr(module, "invoke", lambda *_args, **_kwargs: {"total": 0})
        monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(module, "wait_for_executor_pods", lambda *_args: None)
        monkeypatch.setattr(
            module, "wait_for_load_pods", lambda *_args, **_kwargs: None
        )
        monkeypatch.setattr(module, "wait_for_job", lambda *_args: "Complete")
        monkeypatch.setattr(
            module, "wait_for_workflows", lambda *_args, **_kwargs: ({}, [])
        )
        monkeypatch.setattr(
            module, "collect_executor_logs", lambda *_args, **_kwargs: []
        )
        for name in (
            "control_pod_runtime_snapshot",
            "scrape_metrics",
            "scrape_cgroup",
            "postgres_counters",
            "drain_targets",
            "aggregate",
            "aggregate_executor_documents",
            "queue_drain",
            "aurora_window",
            "control_pod_lifecycle",
            "processor_priority_latency",
            "workflow_audit",
        ):
            monkeypatch.setattr(module, name, lambda *_args, **_kwargs: {})
        monkeypatch.setattr(module, "drain_targets", lambda *_args: (0, 0))
    return harness


def test_callers_create_labelled_receipts_and_pass_them_to_teardown(
    caller: CallerHarness,
) -> None:
    assert caller.run() == 0

    receipt = caller.report("capacity-resources.json")
    assert receipt["run_id"] == "run-a"
    assert receipt["namespace"] == registry.NAMESPACE
    assert set(receipt["resources"]) == set(caller.api.created)
    for key, resource in caller.api.created.items():
        assert receipt["resources"][key]["uid"] == resource["metadata"]["uid"]
        assert resource["metadata"]["labels"][registry.RUN_LABEL] == "run-a"
        if resource["kind"] == "Job":
            assert (
                resource["spec"]["template"]["metadata"]["labels"][registry.RUN_LABEL]
                == "run-a"
            )
    assert caller.api.objects == {}
    assert caller.teardowns == [
        {
            "purge": True,
            "deregister_clusters": True,
            "allow_live_registry": True,
            "live_registry_confirmation": caller.args.confirm_live_registry,
            "artifacts": caller.artifacts,
            "run_id": "run-a",
        }
    ]
    assert caller.requests == [
        {
            "run_id": "run-a",
            "cluster_ids": [f"perf-cap-{index:03d}" for index in range(32)],
            "cleanup": True,
            "force_nonterminal": True,
        }
    ]
    deleted = [
        event
        for event in caller.events
        if event.startswith("delete:") and not event.startswith("delete:secret/")
    ]
    assert deleted, "the actual caller resources were not stopped"
    assert all(
        caller.events.index(event) < caller.events.index("purge") for event in deleted
    ), "run data was purged before the resource cleanup completed"
    assert caller.events.index("purge") < caller.events.index("revoke")
    assert caller.events.index("revoke") < caller.events.index(
        f"delete:secret/{registry.TOKEN_SECRET}"
    )
    assert caller.report("status.json")["status"] == "ok"
    if caller.module is integrated:
        assert caller.connection_reads == [
            registry.IDENTITY_NAMESPACE,
            registry.NAMESPACE,
        ], "both connection Secret reads must precede the load"
        assert caller.report("connection-secret-preflight.json")["changed"] is False, (
            "the real connection preflight must record the already-current mirror"
        )
        assert caller.amp is not None, "integrated preflight must use mocked AMP reads"
        assert caller.amp.calls == [
            "list-workspaces",
            "describe-alert-manager-definition",
        ], "integrated callers must execute the real AMP drill-sink validator"
        assert caller.report("notification-safety-preflight.json") == {
            "workspace_id": "ws-perf-test",
            "definition_status": "ACTIVE",
            "drill_sink_routes": [
                {"receiver": "drill-sink", "matchers": ['cluster_id=~"perf-cap-.*"']}
            ],
        }, "the run must retain the checked AMP preflight receipt"
        assert f"patch:configmap/{base.START_GATE_CONFIGMAP}" in caller.events
        assert len(caller.api.created) == 9  # + the run-owned load ServiceAccount
    else:
        assert len(caller.api.created) == 3  # + the run-owned load ServiceAccount


@pytest.mark.parametrize("caller", [integrated], indirect=True, ids=["integrated"])
@pytest.mark.parametrize(
    "failure",
    ["missing-connection", "missing-sink", "amp-transport", "ingress-recycle"],
)
def test_integrated_preflight_refusal_cannot_register_or_authorize_cleanup(
    caller: CallerHarness, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    assert caller.amp is not None, "the integrated harness must mock AMP transport"
    if failure == "missing-connection":
        caller.connection_missing = True
        message = "is missing from the identity namespace"
    elif failure == "missing-sink":
        caller.amp.config["route"]["routes"] = []
        message = "must not mail drill alerts"
    elif failure == "amp-transport":
        caller.amp.error = TimeoutError("AMP preflight unavailable")
        message = "AMP preflight unavailable"
    else:
        monkeypatch.setattr(
            integrated,
            "ingress_process_model_preflight",
            lambda: {
                "valid": False,
                "request_count_recycling_flags": ["--limit-max-requests"],
            },
        )
        message = "live ingress uses request-count worker recycling"

    with pytest.raises((RuntimeError, TimeoutError), match=message) as caught:
        caller.run()

    if caller.amp.error is not None:
        assert caught.value is caller.amp.error, "the original AMP failure must survive"
    assert caller.events == [], (
        "preflight refusal must prevent registration and cleanup"
    )
    assert caller.api.objects == {}, "preflight refusal must prevent load creation"
    assert caller.requests == [], "preflight refusal must not purge unowned data"
    assert caller.teardowns == [], "preflight refusal cannot grant teardown authority"
    assert not (caller.artifacts / "registry-registration-intent.json").exists(), (
        "preflight refusal must not create registration ownership"
    )
    assert caller.report("status.json")["status"] == "aborted", (
        "preflight refusal must invalidate the run"
    )


def test_registration_failure_before_intent_does_not_cleanup(
    caller: CallerHarness,
) -> None:
    caller.persist_intent = False
    caller.registration_error = RuntimeError("registration preflight failed")

    with pytest.raises(RuntimeError, match="registration preflight failed"):
        caller.run()

    assert caller.teardowns == []
    assert caller.api.created == {}
    assert caller.requests == []
    assert caller.report("status.json")["status"] == "aborted"


@pytest.mark.parametrize("error_type", [TimeoutError, KeyboardInterrupt])
def test_registration_lost_ack_or_interrupt_uses_persisted_intent(
    caller: CallerHarness, error_type: type[BaseException]
) -> None:
    caller.registration_error = error_type("registration ACK lost")

    with pytest.raises(error_type, match="registration ACK lost") as caught:
        caller.run()

    assert caught.value is caller.registration_error
    assert len(caller.teardowns) == 1
    assert caller.events == [
        "register",
        "teardown",
        "purge",
        "revoke",
        f"delete:secret/{registry.TOKEN_SECRET}",
    ]
    assert caller.report("registry-token-proof.json")["uid"] == "token-uid"
    assert caller.report("status.json")["status"] == "aborted"


def test_create_lost_ack_keeps_uid_for_main_cleanup(caller: CallerHarness) -> None:
    job = integrated.LOAD_JOB if caller.module is integrated else correlated.JOB_NAME
    key = f"job/{job}"
    caller.api.lose_create_ack = key

    with pytest.raises(TimeoutError, match="resource create ACK lost"):
        caller.run()

    assert caller.events.count(f"create:{key}") == 1
    assert (
        caller.report("capacity-resources.json")["resources"][key]["uid"]
        == (caller.api.created[key]["metadata"]["uid"])
    )
    assert caller.api.objects == {}
    assert caller.events.index(f"delete:{key}") < caller.events.index("purge")


def test_replacement_uid_blocks_purge_and_revocation(caller: CallerHarness) -> None:
    caller.replace_job = True

    with pytest.raises(RuntimeError, match="UID changed"):
        caller.run()

    assert caller.requests == []
    assert "revoke" not in caller.events
    assert caller.report("status.json")["status"] == "aborted"


def test_main_preserves_primary_error_when_cleanup_transport_fails(
    caller: CallerHarness,
) -> None:
    caller.registration_error = TimeoutError("registration ACK lost")
    caller.data_error = OSError("cleanup transport failed")

    with pytest.raises(TimeoutError, match="registration ACK lost") as caught:
        caller.run()

    assert caught.value is caller.registration_error
    assert any("cleanup transport failed" in note for note in caught.value.__notes__), (
        "the cleanup failure must remain attached to the original exception"
    )
    assert "revoke" not in caller.events
    assert caller.report("status.json")["status"] == "aborted"


def test_successful_case_cannot_hide_cleanup_transport_failure(
    caller: CallerHarness,
) -> None:
    caller.data_error = OSError("cleanup transport failed")

    with pytest.raises(RuntimeError, match="cleanup transport failed") as caught:
        caller.run()

    assert caught.value.__cause__ is caller.data_error
    assert set(caller.api.objects) == {f"secret/{registry.TOKEN_SECRET}"}
    assert "revoke" not in caller.events
    assert caller.report("status.json")["status"] == "aborted"


@pytest.mark.parametrize("report", [{"total": 1}, {"total": "0"}, {"total": False}, {}])
def test_purge_rejects_residual_or_unknown_result(
    caller: CallerHarness, report: dict[str, Any]
) -> None:
    caller.data_result = report

    with pytest.raises(RuntimeError, match="left data|incomplete"):
        caller.run()

    assert "revoke" not in caller.events
    assert caller.report("status.json")["status"] == "aborted"


def test_purge_refuses_foreign_registration_intent(caller: CallerHarness) -> None:
    (caller.artifacts / "registry-registration-intent.json").write_text(
        json.dumps({"run_id": "other-run", "cluster_ids": ["perf-cap-007"]})
    )

    with pytest.raises(RuntimeError, match="another run"):
        caller.purge("run-a", artifacts=caller.artifacts)

    assert caller.requests == []


def test_retained_purge_stops_resources_before_exact_data_cleanup(
    caller: CallerHarness,
) -> None:
    caller.register(
        1, caller.artifacts, run_id="run-a", expires_at=datetime.now(timezone.utc)
    )
    resources = RunResources(caller.artifacts, "run-a", registry.NAMESPACE, caller.api)
    resources.create(
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": "test-producer", "namespace": registry.NAMESPACE},
            "spec": {"completions": 1, "template": {"spec": {}}},
        }
    )

    caller.purge("run-a", artifacts=caller.artifacts)

    assert caller.events.index("delete:job/test-producer") < caller.events.index(
        "purge"
    )
    assert caller.requests == [
        {
            "run_id": "run-a",
            "cluster_ids": ["perf-cap-000"],
            "cleanup": True,
            "force_nonterminal": True,
        }
    ]
    assert "revoke" not in caller.events
