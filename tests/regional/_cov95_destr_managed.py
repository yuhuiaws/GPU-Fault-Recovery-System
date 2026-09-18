from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr012_managed_recovery_guard as case
from tests.regional._cov95_destr_actions import (
    ActionHarness,
    Boundary,
    WorkloadRegional,
    workload_case,
)
from tests.regional._cov95_destr_warm import NOW, profile
from tests.regional._destructive_acceptance_support import restart_state


class ManagedHarness(ActionHarness):
    def __init__(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, rerun: bool = False
    ) -> None:
        super().__init__(workload_case, tmp_path, monkeypatch)
        self.module = case
        self.settings = case.Settings(
            self.settings.regional,
            self.settings.site_file,
            case.DEFAULT_A_MANIFEST,
            case.DEFAULT_D_MANIFEST,
            "job-a",
            "job-a-a001",
            "job-d",
            "job-d-a001",
            tmp_path / "predecessor.json",
            rerun,
        )
        self.jobs: dict[str, ManagedJob] = {}
        self.ignore_annotation_clear = False
        self.profile = profile()
        self.profile["profile_version"] = "profile-a"
        self.profile_hash = "a" * 64
        self.namespace_objects: list[dict[str, Any]] = []
        self.invalid_profile_response = False
        self.profile_disagreement = False
        self.empty_pods = False
        self.regional = Boundary(self, "regional", ManagedRegional(self))
        self.state_a = restart_state(24)
        self.state_a["workflow"]["request_id"] = "workflow-a"
        self.state_d = restart_state(1)
        self.state_d["workflow"]["request_id"] = "workflow-d"
        self.blocked = {
            "event": {"xid": 11},
            "workflow": {
                "status": "FAILED",
                "request_id": "workflow-blocked",
                "step_executions": [
                    {
                        "operation": "STOP_WORKLOADS",
                        "status": "FAILED",
                        "error": "enable-job-auto-resume is enabled on job-d",
                        "details": {
                            "managed_job_recovery_workloads": ["wl-d"],
                            "required_annotation": case.AUTO_RESUME_ANNOTATION,
                            "required_annotation_value": "absent or false",
                            "remediation_commands": [
                                f"kubectl annotate job job-d {case.AUTO_RESUME_ANNOTATION}-"
                            ],
                        },
                    }
                ],
            },
        }
        self.preflight.update(
            group_b=case.group_b_audit(self.regional, profile_version="profile-a"),
            predecessor={
                "valid": True,
                "evidence_valid": True,
                "expected_release_id": "release-a",
                "expected_cluster_id": "cluster-a",
            },
        )
        self.settings.predecessor_path.write_text(
            json.dumps(
                {
                    "case_id": case.PREDECESSOR_CASE_ID,
                    "verdict": "PASS",
                    "status": "COMPLETED",
                    "workflow_official_steps": self.state_a["workflow"][
                        "official_steps"
                    ],
                    "workflow_request_id": "previous-workflow",
                    "release_id": "release-a",
                    "cluster_id": "cluster-a",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(case, "RegionalLiveFixture", lambda _s: self.regional)
        monkeypatch.setattr(case, "ManagedWorkloadFixture", self.job)
        monkeypatch.setattr(case, "ImagePrewarmFixture", lambda *_a, **_k: self.prewarm)
        monkeypatch.setattr(
            case, "read_only_preflight", lambda *_a, **_k: deepcopy(self.preflight)
        )
        monkeypatch.setattr(
            case,
            "record_focused_tests",
            lambda details, result: details.update(focused_tests=result),
        )
        monkeypatch.setattr(case, "time", self.clock)
        monkeypatch.setattr(case, "datetime", self.clock)

    def job(self, regional: Any, settings: Any) -> Boundary:
        if settings.job_id not in self.jobs:
            self.jobs[settings.job_id] = ManagedJob(self, settings)
        return Boundary(self, f"workload.{settings.job_id}", self.jobs[settings.job_id])

    def execute(self, run_dir: Path, seconds: int = 7200) -> tuple[int, dict[str, Any]]:
        code = case.execute_case(
            self.settings, run_dir, 1, NOW + timedelta(seconds=seconds)
        )
        path = run_dir / "cases" / case.CASE_ID / f"{case.CASE_ID}.json"
        return code, json.loads(path.read_text(encoding="utf-8"))


class ManagedRegional(WorkloadRegional):
    def __init__(self, harness: ManagedHarness) -> None:
        super().__init__(harness)
        self.settings = harness.settings.regional

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        is_d = kwargs.get("job_id") == self.h.settings.d_job_id
        if kwargs.get("marker"):
            state = (
                self.h.blocked
                if "-block-" in kwargs["marker"]
                else self.h.state_d
                if is_d
                else self.h.state_a
            )
            return {**deepcopy(state), "observations": [{"workload_phase": "STOPPED"}]}
        count = 1 if is_d else 24
        return {
            "observations": [
                {
                    "workload_phase": "RUNNING",
                    "runtime_profile_version": "profile-a",
                    "workload_ids": ["wl-d" if is_d else "wl-a"],
                    "containers": [
                        {
                            "gpu_count": count,
                            "gpu_uuids": [f"GPU-{i}" for i in range(count)],
                        }
                    ],
                }
            ]
        }

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        if "-block-" in kwargs["marker"]:
            return deepcopy(self.h.blocked)
        return deepcopy(
            self.h.state_d
            if kwargs["job_id"] == self.h.settings.d_job_id
            else self.h.state_a
        )

    def ready_pods(self, plane: str, app: str) -> list[dict[str, Any]]:
        return [] if self.h.empty_pods else [{"name": f"{app}-pod"}]

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        if args[0] == "exec":
            if self.h.invalid_profile_response:
                return "[]"
            digest = (
                "b" * 64
                if self.h.profile_disagreement and "worker" in args[2]
                else self.h.profile_hash
            )
            return json.dumps(
                {
                    "profile": self.h.profile,
                    "profile_sha256": digest,
                    "allowed_namespaces": ["training"],
                }
            )
        if args[:2] == ("get", "pytorchjobs.kubeflow.org,jobs.batch,pods"):
            return json.dumps({"items": self.h.namespace_objects})
        if args[0] == "logs":
            return "healthy executor"
        if args[:2] == ("get", "job"):
            job = self.h.jobs.get(args[2])
            return (
                f"job/{args[2]}"
                if job is not None and job.submitted and not job.deleted
                else ""
            )
        raise AssertionError(f"unexpected fake group request: {args}")


class ManagedJob:
    resource = "job"

    def __init__(self, harness: ManagedHarness, settings: Any) -> None:
        self.h = harness
        self.settings = settings
        self.name = settings.job_id
        self.submitted = False
        self.deleted = False
        self.annotation: str | None = None
        self.restart_state: dict[str, Any] | None = None

    def delete(self) -> None:
        self.deleted = True

    def submit(self) -> dict[str, Any]:
        self.submitted = True
        return {"submitted": True}

    def snapshot(self) -> dict[str, Any]:
        return {
            "pods": [
                {"uid": f"old-{self.name}-{i}", "node": f"gpu-{i}"}
                for i in range(self.settings.expected_pods)
            ],
            "workload": {
                "annotations": (
                    {case.AUTO_RESUME_ANNOTATION: self.annotation}
                    if self.annotation is not None
                    else {}
                ),
                "suspend": False,
            },
        }

    def wait_running(self, **kwargs: Any) -> dict[str, Any]:
        return self.snapshot()

    def authorize_restart(self, state: dict[str, Any]) -> None:
        self.restart_state = state

    def wait_restarted(self, source_uids: set[str], **kwargs: Any) -> dict[str, Any]:
        assert self.restart_state is not None
        return {
            "pods": [
                {"uid": f"new-{self.name}-{i}", "node": f"gpu-{i}"}
                for i in range(self.settings.expected_pods)
            ]
        }

    def annotate_auto_resume(self, value: str | None) -> None:
        if value is not None or not self.h.ignore_annotation_clear:
            self.annotation = value
