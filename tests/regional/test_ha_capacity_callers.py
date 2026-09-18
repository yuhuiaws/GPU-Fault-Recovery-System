"""Exercise the HA entries through real perf registration and owned teardown."""

from __future__ import annotations

import copy
import importlib
import json
import runpy
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_ha005_rollout_continuity as ha005
from scripts.e2e.regional import run_ha006_executor_takeover as ha006
from scripts.e2e.regional import run_ha009_aurora_credential_rotation as ha009
from scripts.perf.regional_capacity_registry import RUN_LABEL
from tests.regional._aurora_binding_support import FakeAurora

registry = importlib.import_module("regional_capacity_registry")
capacity = importlib.import_module("regional_capacity_suite")
MODULES = (ha005, ha006, ha009)


class CallerEnvironment:
    def __init__(self, module: Any, fault: str) -> None:
        self.module, self.fault = module, fault
        self.items: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.created: list[dict[str, Any]] = []
        self.events: list[tuple[str, ...]] = []
        self.registrations = [
            {"cluster_id": "production", "token": "REPLACE_WITH_TEST_TOKEN"}
        ]
        self.rows = 0
        self.uid_counter = 0
        self.samples = 0
        self.run_id = ""

    def control(self, *args: str, **kwargs: Any) -> str:
        return self.command("cpu", args, **kwargs)

    def dataplane(self, *args: str, **kwargs: Any) -> str:
        return self.command("gpu", args, **kwargs)

    def command(
        self,
        plane: str,
        args: tuple[str, ...],
        *,
        stdin: bytes | None = None,
        **kwargs: Any,
    ) -> str:
        if plane == "cpu" and args[:2] == ("get", "pod"):
            return "cpu-api"
        if plane == "cpu" and args[0] == "exec":
            scope = json.loads(args[-1])
            assert scope["cluster_ids"] == ["perf-cap-000"], (
                "HA cleanup must use the exact registration scope"
            )
            self.run_id = scope["run_id"]
            if not scope["cleanup"]:
                self.events.append(("data-inspect", self.run_id))
                return json.dumps({"total": self.rows})
            assert not any(key[1] in {"pod", "job"} for key in self.items), (
                "all HA claimants and refresh Jobs must stop before SQL cleanup"
            )
            self.events.append(("data-cleanup", self.run_id))
            if self.fault == "sql":
                raise RuntimeError("unit SQL transport unavailable")
            if self.fault == "unknown-sql":
                return '{"total": false}'
            self.rows = 0
            return json.dumps({**scope, "total": 0})
        if args[:2] == ("get", "deployment"):
            return json.dumps(
                {
                    "spec": {
                        "template": {"spec": {"containers": [{"image": "unit-image"}]}}
                    }
                }
            )
        if args[:2] == ("create", "job"):
            assert "--dry-run=client" in args, (
                "refresh Jobs must use owned creation after rendering"
            )
            return json.dumps(
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "metadata": {"name": args[2], "namespace": "unit-ns"},
                    "spec": {"template": {"metadata": {}, "spec": {"containers": []}}},
                }
            )
        if args[:2] == ("create", "-f"):
            value = json.loads(stdin or b"")
            kind, name = value["kind"].lower(), value["metadata"]["name"]
            key = plane, kind, name
            assert key not in self.items, (
                "owned creation must not overwrite an existing name"
            )
            assert value["metadata"]["labels"][RUN_LABEL] == self.run_id, (
                "each HA resource must carry the registered run label"
            )
            if kind == "job":
                assert (
                    value["spec"]["template"]["metadata"]["labels"][RUN_LABEL]
                    == self.run_id
                ), "refresh Job Pods must inherit the run label"
            self.uid_counter += 1
            value["metadata"].update(uid=f"uid-{self.uid_counter}", resourceVersion="1")
            if kind == "job":
                value["status"] = {
                    "succeeded": int(not value["spec"].get("suspend", False))
                }
            self.items[key] = value
            self.created.append(copy.deepcopy(value))
            self.events.append(("create", plane, kind, name))
            if kind == "pod":
                self.rows = 1
                if self.fault == "create-ack":
                    raise RuntimeError("unit Pod create ACK lost")
            if (
                (kind == "secret" and self.fault == "token-create-ack")
                or (kind == "configmap" and self.fault == "configmap-ack")
                or (
                    kind == "job"
                    and name.endswith("-watchdog")
                    and self.fault == "watchdog-create-ack"
                )
                or (
                    kind == "job"
                    and name.endswith("-1")
                    and self.fault == "refresh-create-ack"
                )
            ):
                raise RuntimeError("unit create ACK lost")
            if kind == "job" and name.endswith("-1") and self.fault == "refresh-fail":
                value["status"]["succeeded"] = 0
            return (
                json.dumps(value["metadata"]) if "jsonpath={.metadata}" in args else ""
            )
        if args[0] == "get":
            value = self.items.get((plane, args[1].lower(), args[2]))
            if value is None:
                return ""
            return json.dumps(
                value["metadata"] if "jsonpath={.metadata}" in args else value
            )
        if args[:2] == ("delete", "--raw"):
            plural, name = args[2].rsplit("/", 2)[-2:]
            kind = {
                "pods": "pod",
                "jobs": "job",
                "secrets": "secret",
                "configmaps": "configmap",
            }[plural]
            value = self.items[(plane, kind, name)]
            options = json.loads(stdin or b"")
            assert options["preconditions"]["uid"] == value["metadata"]["uid"], (
                "deletion must be pinned to the creation UID"
            )
            assert options["propagationPolicy"] == "Foreground", (
                "cleanup must confirm dependent Pods are stopped"
            )
            self.events.append(("delete", plane, kind, name))
            del self.items[(plane, kind, name)]
            if kind == "secret":
                assert self.rows == 0, "token deletion must follow exact data cleanup"
                assert len(self.registrations) == 1, (
                    "token deletion must follow registry revocation"
                )
            elif self.fault == "token-uid":
                self.items[("gpu", "secret", registry.TOKEN_SECRET)]["metadata"][
                    "uid"
                ] = "replacement-token"
            return ""
        if args[0] == "wait":
            if self.fault == "probe-uid":
                for key, value in self.items.items():
                    if key[1] == "pod":
                        value["metadata"]["uid"] = "replacement-probe"
                        break
            return ""
        if args[:2] == ("rollout", "restart"):
            assert self.module is ha005, "HA009 must never roll a Deployment"
            self.events.append(("rollout", args[2].split("/", 1)[1]))
            return ""
        if args[0] == "logs":
            return (
                "rotated=False restarted=False"
                if args[1].endswith("-2")
                else "rotated=True restarted=False"
            )
        if args[0] == "exec":
            return ""
        raise AssertionError(f"unexpected mocked command: {plane} {args}")

    def write_registry(self, entries: list[dict], **kwargs: Any) -> None:
        assert kwargs["expected_entries"] == self.registrations, (
            "registry updates must retain their compare-and-swap baseline"
        )
        registering = len(entries) > 1
        if not registering:
            assert self.rows == 0, "registry revocation must follow exact data cleanup"
        self.events.append(
            ("register" if registering else "deregister", kwargs["run_id"])
        )
        self.registrations = copy.deepcopy(entries)
        if registering and self.fault == "register-ack":
            raise RuntimeError("unit registry publication ACK lost")

    def entries(self, count: int, *, run_id: str, expires_at: datetime) -> list[dict]:
        assert count == 1, "these HA cases authorize one synthetic cluster"
        return [
            {
                "cluster_id": "perf-cap-000",
                "token": "REPLACE_WITH_TEST_TOKEN",
                "synthetic": True,
                "synthetic_run_id": run_id,
                "synthetic_expires_at": expires_at.isoformat(),
            }
        ]

    def probe(self, **kwargs: Any) -> dict:
        self.samples += 20
        from tests.regional.test_acceptance_alignment_ha_telemetry import (
            telemetry_proof,
        )

        proof, _ = telemetry_proof(spooled=False)
        final = proof["attempted_events"][0]
        events = [
            {**final, "batch_id": f"batch-{index}"} for index in range(self.samples)
        ]
        result = {
            **proof,
            "attempted_events": events,
            "attempted_batch_ids": [item["batch_id"] for item in events],
            "admissions": [
                {
                    **proof["admissions"][0],
                    "batch_id": item["batch_id"],
                    "request_id": f"request-{index}",
                }
                for index, item in enumerate(events)
            ],
            "counters": {
                "event_attempts": self.samples,
                "event_accepted": self.samples,
                "event_failures": 0,
                "event_buffered": 0,
                "claim_success": 20,
            },
            "accepted_request_ids": [
                f"request-{index}" for index in range(self.samples)
            ],
            "outbox": {"records": 0, "replayable": 0},
            "error_types": {},
            "stopped": True,
        }
        self.last_probe = result
        return result

    def deployment(self, name: str = ha005.INGRESS_DEPLOYMENT) -> dict:
        generation = 2 if ("rollout", name) in self.events else 1
        return {
            "name": name,
            "uid": f"deployment-{name}",
            "generation": generation,
            "observed_generation": generation,
            "replicas": 1,
            "ready": 1,
            "updated": 1,
            "available": 1,
            "pods": [
                (
                    name,
                    {
                        "uid": f"{name}-{generation}",
                        "ready": True,
                        "restarts": 0,
                        "port": 8080,
                    },
                )
            ],
        }

    def seed(self, run_id: str) -> dict:
        return {
            "incident_id": f"incident-{run_id}",
            "event_id": f"event-{run_id}",
            "workflow_id": f"workflow-{run_id}",
            "command_id": f"command-{run_id}",
            "notification_id": f"notification-{run_id}",
            "deduplication_key": f"{run_id}/notice",
        }

    def cleanup_seed(self, *args: Any) -> dict:
        assert not any(key[1] in {"pod", "job"} for key in self.items), (
            "case-specific SQL cleanup must follow owned workload shutdown"
        )
        self.events.append(("seed-cleanup", self.run_id))
        if self.fault == "seed":
            raise RuntimeError("unit seed cleanup failed")
        return {"remaining_objects": 0, "remaining_links": 0}

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        base = ha005 if self.module is ha009 else self.module
        for module in (registry, capacity, base):
            monkeypatch.setattr(module, "control", self.control)
            monkeypatch.setattr(module, "dataplane", self.dataplane)
            monkeypatch.setattr(module, "NAMESPACE", "unit-ns")
        for module in (registry, capacity):
            monkeypatch.setattr(
                module, "validate_registry_target", lambda **kw: "isolated"
            )
        monkeypatch.setattr(registry, "validate_notification_safety", lambda: None)
        monkeypatch.setattr(registry, "validate_alertmanager_drill_route", lambda: None)
        monkeypatch.setattr(
            registry, "load_registry", lambda: copy.deepcopy(self.registrations)
        )
        monkeypatch.setattr(registry, "write_registry", self.write_registry)
        monkeypatch.setattr(registry, "perf_cluster_entries", self.entries)
        monkeypatch.setattr(base, "database_residuals", lambda: {"total": self.rows})
        monkeypatch.setattr(
            base, "registry_residuals", lambda: {"count": len(self.registrations) - 1}
        )
        monkeypatch.setattr(
            base, "kubernetes_residuals", lambda: {"count": len(self.items)}
        )
        monkeypatch.setattr(
            base,
            "executor_identity",
            lambda **kw: {
                "executor_artifact_sha256": "unit-sha",
                "executor_compatibility_digest": "unit-digest",
            },
        )
        monkeypatch.setattr(base, "wait_file", lambda *a: None)
        monkeypatch.setattr(base.time, "sleep", lambda _: None)
        if self.module is ha006:
            self.install_takeover(monkeypatch)
        else:

            def telemetry_receipt(final):
                from tests.regional.test_acceptance_alignment_ha_telemetry import (
                    telemetry_proof,
                )

                _, proof = telemetry_proof()
                return {
                    **proof["telemetry"],
                    "batch_id": final["attempted_events"][-1]["batch_id"],
                }

            monkeypatch.setattr(base, "telemetry_replay_receipt", telemetry_receipt)
            monkeypatch.setattr(base, "read_probe", self.probe)
            monkeypatch.setattr(base, "wait_probe_samples", self.probe)
            monkeypatch.setattr(base, "deployment_snapshot", self.deployment)
            monkeypatch.setattr(
                base, "rollout_targets", lambda _: list(ha005.ALL_DEPLOYMENTS)
            )
            monkeypatch.setattr(
                base,
                "wait_receipts",
                lambda ids, **kw: {
                    "missing": [],
                    "requests": [
                        {
                            "request_id": value,
                            "status": "COMPLETED",
                            "response_status": 200,
                        }
                        for value in ids
                    ],
                },
            )
            monkeypatch.setattr(
                base,
                "processor_receipts",
                lambda ids: {
                    "missing": [],
                    "requests": [
                        {
                            "request_id": value,
                            "status": "COMPLETED",
                            "response_status": 200,
                        }
                        for value in ids
                    ],
                },
            )
        if self.module is ha009:
            self.install_rotation(monkeypatch)

    def install_takeover(self, monkeypatch: pytest.MonkeyPatch) -> None:
        notice = {
            "notification_id": "unit-notification",
            "dedup_link_count": 1,
            "objects": {
                kind: {"count": 1, "status": "SKIPPED"}
                for kind in (
                    "notification",
                    "notification_delivery",
                    "notification_result",
                )
            },
        }
        first = {
            "status": "LEASED",
            "lease_owner": ha006.PODS[0],
            "lease_expires_at": (
                datetime.now(timezone.utc) + timedelta(seconds=30)
            ).isoformat(),
        }
        monkeypatch.setattr(ha006, "seed_command", self.seed)
        monkeypatch.setattr(ha006, "cleanup_seed", self.cleanup_seed)
        monkeypatch.setattr(ha006, "notification_snapshot", lambda _: notice)
        monkeypatch.setattr(ha006, "wait_first_owner", lambda _: (first, notice, []))
        monkeypatch.setattr(
            ha006, "waiting_branch", lambda _: {"reclaimed_by_other_replica": True}
        )
        monkeypatch.setattr(ha006, "command_snapshot", lambda _: first)
        monkeypatch.setattr(
            ha006,
            "wait_terminal",
            lambda _: (
                {
                    "status": "SUCCEEDED",
                    "last_lease_owner": ha006.PODS[1],
                    "updated_at": (
                        datetime.now(timezone.utc) + timedelta(seconds=1)
                    ).isoformat(),
                    "result_details": {
                        "cached": True,
                        "shared_notification_id": "unit-notification",
                    },
                },
                [],
            ),
        )
        monkeypatch.setattr(
            ha006,
            "read_state",
            lambda pod, path: {
                "physical_actions": int(pod == ha006.PODS[0]),
                "claimed_total": 1,
                "unexpected_failures": 0,
            },
        )
        monkeypatch.setattr(ha006, "production_lease_seconds", lambda: 30)

    def install_rotation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proof = {"identity": {"database": {"master_secret_arn": "unit-resource"}}}
        monkeypatch.setattr(
            ha009,
            "aurora_guard",
            lambda: SimpleNamespace(
                read=lambda expected=None: proof,
                refresh_job=lambda name, expected: json.loads(
                    self.control("create", "job", name, "--dry-run=client")
                ),
            ),
        )
        monkeypatch.setattr(ha009, "pool_max_idle_seconds", lambda: 1)
        monkeypatch.setattr(ha009, "master_secret_arn", lambda: "unit-resource")
        monkeypatch.setattr(
            ha009, "secret_versions", lambda _: {"stages": {"AWSCURRENT": "v1"}}
        )
        monkeypatch.setattr(
            ha009, "wait_rotated_secret", lambda *a: {"stages": {"AWSCURRENT": "v2"}}
        )
        monkeypatch.setattr(ha009, "kubernetes_secret_digest", lambda: "unit-digest")
        monkeypatch.setattr(
            ha009, "kubernetes_secret_dsn_digest", lambda: "unit-dsn-digest"
        )
        monkeypatch.setattr(
            ha009,
            "deployment_snapshot",
            lambda: {name: self.deployment(name) for name in ha009.DEPLOYMENTS},
        )
        monkeypatch.setattr(ha009, "seed_runtime_records", self.seed)
        monkeypatch.setattr(ha009, "cleanup_runtime_records", self.cleanup_seed)
        monkeypatch.setattr(ha009, "wait_runtime_records", lambda _: {})
        monkeypatch.setattr(
            ha009, "aws", lambda *a: {"DBCluster": {"DBClusterIdentifier": "unit"}}
        )
        monkeypatch.setattr(ha009, "wait_secret_propagated", lambda *a: {})
        monkeypatch.setattr(ha009, "observe_after_idle", lambda *a, **kw: {})
        monkeypatch.setattr(ha009, "rotation_errors", lambda **kw: [])

        def start_process(*args: Any, **kwargs: Any) -> SimpleNamespace:
            self.events.append(("watchdog-start", self.run_id))
            if self.fault == "watchdog-start":
                raise RuntimeError("unit watchdog start failed")
            return SimpleNamespace(pid=12345)

        monkeypatch.setattr(ha009.subprocess, "Popen", start_process)
        monkeypatch.setattr(
            ha009,
            "stop_refresh_watchdog",
            lambda _: (
                {"armed": True, "stop_error": "unit watchdog stop unverified"}
                if self.fault == "watchdog-stop"
                else {"armed": True, "disarmed": True}
            ),
        )


def run_entry(module: Any, run_dir: Path, attempt: int = 1) -> tuple[int, dict]:
    status = module.run_case(
        run_dir,
        attempt,
        datetime(2099, 1, 1, tzinfo=timezone.utc),
        **({"all_deployments": True} if module is ha005 else {}),
    )
    report = json.loads(
        (run_dir / "cases" / module.CASE_ID / f"{module.CASE_ID}.json").read_text()
    )
    return status, report


def test_rotation_binding_drift_after_watchdog_arm_disarms_without_rotating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment = CallerEnvironment(ha009, "")
    environment.install(monkeypatch)
    aurora = FakeAurora()
    monkeypatch.setattr(ha009, "aurora_guard", lambda: aurora.guard)
    original = ha009.start_refresh_watchdog

    def arm(*args, **kwargs):
        process = original(*args, **kwargs)
        aurora.cluster["DbClusterResourceId"] += "-replacement"
        return process

    monkeypatch.setattr(ha009, "start_refresh_watchdog", arm)
    mutations = []
    monkeypatch.setattr(ha009, "aws", lambda *a: mutations.append(a) or {})
    status, report = run_entry(ha009, tmp_path)
    assert status == 1 and report["verdict"] == "FAIL", report
    assert mutations == [], "binding refusal must precede the provider request"
    assert report["refresh_watchdog"]["disarmed"] is True, report
    assert not environment.items, (
        "all UID-owned Jobs and probe resources must be removed"
    )
    assert len(environment.registrations) == 1, (
        "existing registration cleanup must remain exact"
    )


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize(
    "fault",
    [
        "",
        "token-create-ack",
        "register-ack",
        "configmap-ack",
        "create-ack",
        "probe-uid",
        "token-uid",
        "sql",
        "unknown-sql",
    ],
)
def test_entry_to_cleanup_uses_registration_receipts_and_exact_data_scope(
    module: Any, fault: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment = CallerEnvironment(module, fault)
    environment.install(monkeypatch)
    run_dir = tmp_path / "acceptance-unit"
    status, report = run_entry(module, run_dir)
    case_dir = run_dir / "cases" / module.CASE_ID
    artifacts = case_dir / f"capacity-{environment.run_id}"
    intent = json.loads((artifacts / "registry-registration-intent.json").read_text())
    proof = json.loads((artifacts / "registry-token-proof.json").read_text())
    assert intent["run_id"] == proof["run_id"] == environment.run_id, (
        "entry and cleanup must share the registration identity"
    )
    assert intent["data_empty_before_registration"] is True, (
        "registration must retain the empty-data preflight proof"
    )
    assert intent["cluster_ids"] == ["perf-cap-000"], (
        "cleanup cannot infer a shared prefix"
    )
    receipts = list(case_dir.glob("*resources-*.json"))
    assert bool(receipts) == (fault != "register-ack"), (
        "HA receipts must match attempted resource creation"
    )
    assert not (artifacts / "capacity-resources.json").exists(), (
        "HA must reuse OwnedProbeResources without a parallel resource ledger"
    )
    if fault in {"", "token-create-ack", "register-ack", "configmap-ack", "create-ack"}:
        assert environment.items == {}, (
            "successful cleanup must remove every owned Pod, ConfigMap, Job and token"
        )
        assert environment.rows == 0, "successful cleanup must purge exact run data"
        assert len(environment.registrations) == 1, (
            "production registration must remain"
        )
        assert ("data-cleanup", environment.run_id) in environment.events, (
            "HA telemetry requires exact data cleanup even when seed creation never ran"
        )
        assert status == int(fault not in {"", "token-create-ack"}), report
    else:
        assert status == 1 and report["verdict"] == "FAIL", report
        assert ("gpu", "secret", registry.TOKEN_SECRET) in environment.items, (
            "unknown cleanup must preserve the run token"
        )
        assert len(environment.registrations) == 2, (
            "unknown cleanup must preserve the durable registration"
        )
        assert environment.rows == 1, "failed cleanup must not claim data was removed"
    if not fault:
        rollouts = [event[1] for event in environment.events if event[0] == "rollout"]
        assert rollouts == (list(ha005.ALL_DEPLOYMENTS) if module is ha005 else []), (
            "HA005 must cover all roles and HA009 must keep zero rollouts"
        )
        if module is ha009:
            jobs = [value for value in environment.created if value["kind"] == "Job"]
            assert len(jobs) == 3, (
                "watchdog and both refresh Jobs must use owned creation"
            )
            assert jobs[0]["spec"]["suspend"] is True, (
                "watchdog Job must be receipted before its delayed execution"
            )


@pytest.mark.parametrize("module", [ha006, ha009])
def test_seed_cleanup_failure_stops_before_shared_purge_and_revocation(
    module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment = CallerEnvironment(module, "seed")
    environment.install(monkeypatch)
    status, report = run_entry(module, tmp_path / "acceptance-unit")
    assert status == 1 and "cleanup_preserved" in report, report
    assert ("seed-cleanup", environment.run_id) in environment.events, (
        "case-specific cleanup must run"
    )
    assert ("data-cleanup", environment.run_id) not in environment.events, (
        "failed seed cleanup must not proceed to shared purge"
    )
    assert set(environment.items) == {("gpu", "secret", registry.TOKEN_SECRET)}, (
        "owned HA workloads must stop while registry and token remain recoverable"
    )
    assert len(environment.registrations) == 2, (
        "failed seed cleanup must preserve the registration"
    )


@pytest.mark.parametrize(
    "fault",
    ["watchdog-create-ack", "watchdog-start", "refresh-create-ack", "refresh-fail"],
)
def test_rotation_partial_creation_keeps_uid_receipts_and_emergency_cleanup(
    fault: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment = CallerEnvironment(ha009, fault)
    environment.install(monkeypatch)
    run_dir = tmp_path / "acceptance-unit"
    status, report = run_entry(ha009, run_dir)
    assert status == 1 and report["verdict"] == "FAIL", report
    assert environment.items == {}, (
        "failed rotation setup must remove all owned Jobs and probes"
    )
    assert environment.rows == 0 and len(environment.registrations) == 1, (
        "verified cleanup must purge exact run data and revoke its registration"
    )
    jobs = [value for value in environment.created if value["kind"] == "Job"]
    receipt_path = (
        run_dir
        / "cases"
        / ha009.CASE_ID
        / f"refresh-resources-{environment.run_id}.json"
    )
    receipt = json.loads(receipt_path.read_text())
    assert all(
        receipt["resources"][f"Job/{job['metadata']['name']}"]["uid"]
        == job["metadata"]["uid"]
        for job in jobs
    ), "even lost create ACKs must reconcile their original Job UID"
    if fault.startswith("refresh-"):
        assert report["emergency_refresh"]["succeeded"] is True, (
            "first refresh failure must retain the emergency refresh path"
        )
        assert jobs[-1]["metadata"]["name"].endswith("-emergency"), (
            "emergency refresh must use owned Job creation"
        )
    assert not any(event[0] == "rollout" for event in environment.events), (
        "credential recovery must never add a consumer rollout"
    )


def test_unverified_watchdog_shutdown_preserves_owned_job_and_registration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment = CallerEnvironment(ha009, "watchdog-stop")
    environment.install(monkeypatch)
    status, report = run_entry(ha009, tmp_path / "acceptance-unit")
    assert status == 1 and report["watchdog_cleanup_unverified"] is True, report
    assert report["cleanup_preserved"], "unknown watchdog completion must be explicit"
    assert set(environment.items) == {
        ("gpu", "secret", registry.TOKEN_SECRET),
        ("cpu", "job", f"gpu-fault-{environment.run_id}-watchdog"),
    }, "unverified watchdog must retain only its receipted Job and scoped token"
    assert not any(
        event[0] in {"seed-cleanup", "data-cleanup", "deregister"}
        for event in environment.events
    ), "an unverified watchdog must block destructive SQL and registry cleanup"


@pytest.mark.parametrize("module", MODULES)
def test_next_attempt_keeps_the_previous_registration_receipts(
    module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment = CallerEnvironment(module, "")
    environment.install(monkeypatch)
    run_dir = tmp_path / "acceptance-unit"
    assert run_entry(module, run_dir)[0] == 0, "first attempt must complete"
    first = run_dir / "cases" / module.CASE_ID / f"capacity-{environment.run_id}"
    previous = (first / "registry-token-proof.json").read_bytes()
    environment.events.clear()
    assert run_entry(module, run_dir, attempt=2)[0] == 0, (
        "next attempt must not adopt the previous run"
    )
    assert (first / "registry-token-proof.json").read_bytes() == previous, (
        "a new attempt must not overwrite earlier UID proof"
    )


def test_rotation_capacity_cleanup_uses_the_current_projected_dsn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import psycopg

    projected = "postgresql://fixture.invalid/current"
    path = tmp_path / "postgres-url"
    path.write_text(projected)
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://fixture.invalid/previous")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capacity-data-probe",
            json.dumps(
                {
                    "run_id": "ha009-unit",
                    "cluster_ids": ["perf-cap-000"],
                    "cleanup": True,
                }
            ),
        ],
    )
    connections = []

    class ConnectionIntercepted(Exception):
        pass

    def intercept(conninfo: str, **kwargs: Any) -> Any:
        connections.append(conninfo)
        raise ConnectionIntercepted

    monkeypatch.setattr(psycopg, "connect", intercept)
    # A standalone probe starts with its own module namespace.
    monkeypatch.delitem(
        sys.modules, "scripts.perf.regional_capacity_data", raising=False
    )
    with pytest.raises(ConnectionIntercepted):
        runpy.run_module("scripts.perf.regional_capacity_data", run_name="__main__")
    assert connections == [projected], (
        "HA009 cleanup must read the rotated DSN without restarting the CPU Pod"
    )
