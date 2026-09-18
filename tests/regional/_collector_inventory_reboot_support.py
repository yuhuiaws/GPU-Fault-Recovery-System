"""Owned, in-memory transports for inventory reboot runner regressions."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch
from scripts.e2e.regional import collector_inventory_reboot as runner
from scripts.e2e.regional.collector_acceptance_fixture import CollectorAcceptanceFixture
from scripts.e2e.regional.collector_case_cleanup import CaseCleanup
from scripts.e2e.regional.probes.collector_node_probe import inventory_receipt_digest
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture
from tests.regional._cov95_collect_net import Clock

ORIGIN = datetime(2026, 9, 17, tzinfo=timezone.utc)
BASELINE = "a" * 64
APPLIED = "b" * 64
INTENT = "c" * 64
ROLE = "arn:aws:iam::123456789012:role/Executor"


class InventoryHarness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, case_dir: Path) -> None:
        self.case_dir = case_dir
        self.calls: list[str] = []
        self.clock = Clock()
        self.problem: str | None = None
        self.failure: BaseException | None = None
        self.publication_failure: BaseException | None = None
        self.sample: dict[str, Any] | None = None
        self.published = False
        self.run_id = ""
        self.owner_nonce = ""
        self.restores = 0
        self.rebooted = False
        self.cleanup = CaseCleanup()
        self.scope = {
            "node_id": "node-a",
            "node_uid": "node-uid-a",
            "boot_id": "boot-a",
            "cluster_name": "cluster-a",
            "cluster_arn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/cluster-a",
            "instance_id": "i-00000000000000001",
            "executor_role_arn": ROLE,
        }
        self.settings = SimpleNamespace(
            regional=SimpleNamespace(cluster_id="cluster-a"),
            node="node-a",
            hyperpod_cluster="cluster-a",
            executor_role_arn=ROLE,
            debounce_tolerance=0.5,
        )
        self.regional = SimpleNamespace(
            cpu_python=self.cpu,
            wait_node_ready=self.ready,
            wait_provider_events=self.events,
            provider_events=self.events,
            provider_events_provisional=lambda *args: True,
            business_workloads=lambda *args: [],
            node_snapshot=self.node_snapshot,
        )
        self.collector = SimpleNamespace(
            node="node-a",
            snapshot=self.snapshot,
            execute=self.execute,
            recreate=self.recreate,
            restore_incidents=self.restore_incidents,
        )
        monkeypatch.setattr(runner, "time", self.clock)
        monkeypatch.setattr(
            runner,
            "datetime",
            SimpleNamespace(
                now=lambda tz=None: ORIGIN + timedelta(seconds=self.clock.now - 1000)
            ),
        )
        monkeypatch.setattr(runner, "run_collect003", self.healthy)
        monkeypatch.setattr(runner, "capture_reboot_scope", self.capture)
        monkeypatch.setattr(runner, "submitted_reboot_errors", lambda **kwargs: [])
        monkeypatch.setattr(
            runner,
            "prove_reboot_scope",
            lambda *args, **kwargs: {
                "errors": ["provider scope differs"]
                if self.problem == "provider"
                else [],
                "aws_request_id_joined": False,
            },
        )

    def healthy(self, collector: Any) -> dict[str, Any]:
        self.calls.append("healthy")
        return {
            "errors": ["unhealthy inventory"] if self.problem == "unhealthy" else []
        }

    def capture(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append("scope")
        return {
            **self.scope,
            "boot_id": "boot-b" if self.rebooted else "boot-a",
            **(
                {"instance_id": "i-00000000000000002"}
                if self.rebooted and self.problem == "provider-identity"
                else {}
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        self.calls.append("snapshot")
        return {
            "boot_id": "boot-b" if self.rebooted else "boot-a",
            "captured_at": ORIGIN.isoformat(),
            "collector_env_file": {
                "sha256": APPLIED
                if self.rebooted and self.problem == "digest"
                else BASELINE
            },
            "collector_env": {
                "GPU_FAULT_EXPECTED_GPU_COUNT": "8",
                "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
                "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES": (
                    "1" if self.problem == "debounce-config" else "2"
                ),
            },
        }

    def node_snapshot(self, node: str) -> dict[str, Any]:
        if self.published or (self.run_id and self.restores):
            if "node-ready" not in self.calls:
                self.calls.append("node-ready")
                self.clock.now = max(self.clock.now, 1060)
            self.rebooted = self.problem != "same-boot"
        return {
            "name": node,
            "uid": "other-node"
            if self.rebooted and self.problem == "node-identity"
            else self.scope["node_uid"],
            "boot_id": "boot-b" if self.rebooted else "boot-a",
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
        }

    def records(self) -> list[dict[str, Any]]:
        return [
            {
                "record_id": f"host-telemetry/host-node-a-batch-{counter}",
                "batch_id": f"host-node-a-batch-{counter}",
                "observed_at": (ORIGIN + timedelta(seconds=15 * counter)).isoformat(),
                "edge_filter_reasons": ["baseline"] if counter == 1 else [],
                "samples": [
                    {
                        "name": "gpu_inventory_mismatch",
                        "value": counter - 1,
                        "device": None,
                        "labels": {
                            "expected_count": "9",
                            "observed_count": "8",
                            "required_consecutive_samples": "2",
                            "consecutive_mismatch_samples": str(counter),
                            "node_instance_type": "ml.p5en.48xlarge",
                        },
                    }
                ],
            }
            for counter in (1, 2)
        ]

    def state(self) -> dict[str, Any]:
        workflow = {
            "request_id": "workflow-a",
            "incident_id": "incident-a",
            "status": "FAILED"
            if self.rebooted and self.problem == "failed-workflow"
            else "SUCCEEDED",
            "official_steps": []
            if self.problem == "missing-restart"
            else [{"operation": "RESTART_NODE", "node_ids": ["node-a"]}],
            "step_executions": [],
        }
        return {
            "event_id": (
                f"{self.sample['batches'][-1]['batch_id']}-gpu_inventory_mismatch-node"
                if self.sample is not None
                else "host-node-a-batch-2-gpu_inventory_mismatch-node"
            ),
            "incidents": [{"incident_id": "incident-a"}],
            "workflows": [workflow]
            + (
                [{**workflow, "request_id": "workflow-b"}]
                if self.problem == "extra-workflow"
                else []
            ),
            "commands": [],
            "node_workflow_ids": ["workflow-a"]
            + (["workflow-b"] if self.problem == "extra-workflow" else []),
            "submissions": [{"state": "SUBMITTED"}],
            "captured_at": (
                ORIGIN + timedelta(seconds=self.clock.now - 1000)
            ).isoformat(),
        }

    def cpu(self, script: str, *args: str) -> dict[str, Any]:
        if script == runner.HOST_INVENTORY_EVIDENCE:
            self.calls.append("evidence")
            self.clock.now = max(self.clock.now, 1035)
            if self.problem == "poll":
                raise RegionalFixtureError("evidence poll timed out")
            return {
                "records": [
                    {
                        "record_id": f"host-telemetry/{batch['batch_id']}",
                        "batch_id": batch["batch_id"],
                        "observed_at": batch["observed_at"],
                        "edge_filter_reasons": batch["edge_filter_reasons"],
                        # HOST_INVENTORY_EVIDENCE projects a persisted batch to
                        # its gpu_inventory_mismatch sample; mirror that shape.
                        "samples": [
                            deepcopy(item)
                            for item in batch["samples"]
                            if item["name"] == "gpu_inventory_mismatch"
                        ],
                    }
                    for batch in self.sample["batches"]
                ]
                if self.sample is not None
                else self.records()
            }
        assert script == runner.INVENTORY_RECOVERY_STATE
        self.calls.append("recovery")
        if self.problem == "recovery-read":
            raise RegionalFixtureError("recovery read failed")
        return self.state()

    def execute(self, verb: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(verb)
        run_id = args[args.index("--run-id") + 1]
        if verb == "sample-gpu-inventory":
            self.run_id = run_id
            if self.failure is not None:
                raise self.failure
            self.sample = self.isolated_sample(run_id)
            if self.problem == "sample":
                self.sample["kind"] = "not-isolated"
            return deepcopy(self.sample)
        if verb == "publish-gpu-inventory":
            assert self.sample is not None
            assert json.loads(args[args.index("--receipt-json") + 1]) == self.sample
            assert args[args.index("--expected-sha256") + 1] == self.sample["sha256"]
            assert args[args.index("--confirm") + 1] == "PUBLISH_GPU_INVENTORY"
            if self.problem == "publication":
                raise RegionalFixtureError("publication rejected")
            self.published = True
            if self.publication_failure is not None:
                raise self.publication_failure
            return {
                "run_id": run_id,
                "sample_sha256": self.sample["sha256"],
                "publication_performed": True,
                "batch_ids": [item["batch_id"] for item in self.sample["batches"]],
            }
        nonce = args[args.index("--owner-nonce") + 1]
        if verb == "override-expected-gpu-count":
            self.run_id, self.owner_nonce = run_id, nonce
            assert args[args.index("--expected-env-sha256") + 1] == BASELINE
            assert args[args.index("--expected-boot-id") + 1] == "boot-a"
            assert args[args.index("--cluster-id") + 1] == "cluster-a"
            assert args[args.index("--node-id") + 1] == "node-a"
            if self.failure is not None:
                raise self.failure
            return {
                "run_id": run_id,
                "cluster_id": "cluster-a",
                "node_id": "node-a",
                "boot_id": "boot-a",
                "mutation_started": True,
                "timer_armed": True,
                "boot_restore_armed": self.problem != "arming",
                "baseline_sha256": BASELINE,
                "applied_sha256": APPLIED,
                "intent_sha256": INTENT,
            }
        assert verb == "restore-collector-env"
        assert (run_id, nonce) == (self.run_id, self.owner_nonce)
        self.restores += 1
        if self.problem in {"restore", "poll-and-restore"}:
            raise RegionalFixtureError("restore rejected")
        return {
            "run_id": run_id,
            "restored": True,
            "cleanup_verified": True,
            "timer_disarmed": True,
            "state": "CLEANED",
            "baseline_sha256": BASELINE,
            "intent_sha256": INTENT,
        }

    def isolated_sample(self, run_id: str) -> dict[str, Any]:
        batches = []
        for index in range(2):
            stamp = ORIGIN + timedelta(seconds=15 * (index + 1))
            labels = {
                "expected_count": "9",
                "observed_count": "8",
                "required_consecutive_samples": "2",
                "consecutive_mismatch_samples": str(index + 1),
                "node_instance_type": "ml.p5en.48xlarge",
            }
            batch = HostTelemetryBatch(
                batch_id=f"host-node-a-{int(stamp.timestamp() * 1_000_000)}",
                cluster_id="cluster-a",
                node_id="node-a",
                observed_at=stamp,
                runtime_profile_version="profile-a",
                edge_filter_reasons=["baseline"] if index == 0 else ["threshold"],
                samples=[
                    HostMetricSample(
                        name=f"gpu_inventory_{name}", value=value, labels=labels
                    )
                    for name, value in (
                        ("expected_count", 9),
                        ("active_count", 8),
                        ("missing_count", 1),
                        ("excess_count", 0),
                        ("mismatch", index),
                    )
                ],
            )
            batches.append(batch.model_dump(mode="json"))
        values = {"GPU_FAULT_EXPECTED_GPU_COUNT": "8"}
        sample = {
            "schema_version": 1,
            "kind": "ISOLATED_GPU_INVENTORY",
            "run_id": run_id,
            "identity": {
                "cluster_id": "cluster-a",
                "node_id": "node-a",
                "boot_id": "boot-a",
                "runtime_profile_version": "profile-a",
                "node_instance_type": "ml.p5en.48xlarge",
                "configuration": {
                    "pid": 123,
                    "invocation_id": "host-before",
                    "file": {"sha256": BASELINE},
                    "file_env": values,
                    "running_env": values,
                },
            },
            "interval_seconds": 15,
            "expected_gpu_count": 9,
            "sampled_at": (ORIGIN + timedelta(seconds=31)).isoformat(),
            "sampler_pid": 456,
            "publication_performed": False,
            "batches": batches,
        }
        sample["sha256"] = inventory_receipt_digest(sample)
        self.clock.now = max(self.clock.now, 1031)
        return sample

    def ready(self, node: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append("node-ready")
        assert "restore-collector-env" in self.calls
        self.rebooted = True
        return {
            "uid": "other-node" if self.problem == "node-identity" else "node-uid-a",
            "boot_id": "boot-a" if self.problem == "same-boot" else "boot-b",
            "ready": "True",
        }

    def events(self, *args: Any, **kwargs: Any) -> list[dict[str, str]]:
        self.calls.append("provider-events")
        return [{"event_name": "BatchRebootClusterNodes"}]

    def recreate(self) -> None:
        self.calls.append("recreate")

    def restore_incidents(self, state: dict[str, Any], **kwargs: Any) -> list[Any]:
        self.calls.append("incident-cleanup")
        assert state["incidents"] == [{"incident_id": "incident-a"}]
        return []

    def run(self, *, attempt: int = 1) -> dict[str, Any]:
        return runner.run_collect004(
            self.settings,
            cast(RegionalLiveFixture, self.regional),
            cast(CollectorAcceptanceFixture, self.collector),
            self.case_dir,
            attempt,
            cleanup=self.cleanup,
            expected_scope=deepcopy(self.scope),
        )
