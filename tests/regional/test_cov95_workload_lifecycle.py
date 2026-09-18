from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_workload_acceptance as runner
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional.test_identity_causal_review import lifecycle_harness
from tests.regional.test_workload_acceptance_review import baseline_harness


@pytest.mark.parametrize(
    "defect",
    [
        "busy",
        "state-dir",
        "annotate",
        "server-dry-run",
        "prewarm-residual",
        "prewarm-error",
    ],
)
def test_finite_lifecycle_refuses_preconditions_and_retains_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    site, target, events = baseline_harness(monkeypatch, tmp_path)
    region = site.regional(target)
    case_id = "GF-REGIONAL-WORKLOAD-002"
    if defect == "busy":
        monkeypatch.setattr(region, "gpu_workloads", lambda: [{"name": "foreign"}])
    elif defect == "annotate":
        original = runner.run

        def run(command: list[str], **kwargs: Any) -> Any:
            result = original(command, **kwargs)
            if Path(command[0]).name == "gpu-fault-workload-annotate":
                result.returncode = 1
            return result

        monkeypatch.setattr(runner, "run", run)
    elif defect == "server-dry-run":
        original_run = region.run

        def rejected(command: list[str], **kwargs: Any) -> Any:
            result = original_run(command, **kwargs)
            result.returncode = 1
            return result

        monkeypatch.setattr(region, "run", rejected)
    elif defect.startswith("prewarm-"):
        original_factory = runner.ImagePrewarmFixture

        def prewarm(*args: Any, **kwargs: Any) -> Any:
            fixture = original_factory(*args, **kwargs)
            cleanup = fixture.cleanup

            def incomplete() -> dict[str, bool]:
                cleanup()
                if defect == "prewarm-error":
                    raise RuntimeError("synthetic cleanup unavailable")
                return {"pod": True}

            fixture.cleanup = incomplete
            return fixture

        monkeypatch.setattr(runner, "ImagePrewarmFixture", prewarm)
    kwargs = {
        "case_id": case_id,
        "site": site,
        "target": target,
        "case_dir": tmp_path,
        "job_id": "job-test",
        "attempt_id": "attempt-test",
        "state_dir": None if defect == "state-dir" else tmp_path,
        "attempt": 1,
    }
    if defect.startswith("prewarm-"):
        result = runner.run_workload_baseline(**kwargs)
        assert result["verdict"] == "FAIL"
        assert result["cleanup_errors"]
        assert events[-2:] == ["delete", "prewarm-cleanup"]
    else:
        with pytest.raises(
            (runner.WorkloadAcceptanceError, runner.RegionalFixtureError)
        ):
            runner.run_workload_baseline(**kwargs)
        if defect in {"busy", "state-dir"}:
            assert events == []
        else:
            assert "apply" not in events
            assert events[-1] == "prewarm-cleanup"


@pytest.mark.parametrize(
    "defect",
    [
        "physical",
        "busy",
        "observation",
        "refreshed",
        "expired",
        "restore",
        "prewarm-error",
        "prewarm-residual",
    ],
)
def test_iso001_cleanup_preserves_physical_a_b_scope_when_any_phase_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    site, targets, events = lifecycle_harness(monkeypatch, tmp_path)
    a, b = [site.regional(target) for target in targets]
    if defect == "physical":
        monkeypatch.setattr(runner, "registration_snapshot", lambda *args: [])
    elif defect == "busy":
        monkeypatch.setattr(b, "gpu_workloads", lambda: [{"name": "foreign"}])
    elif defect in {"observation", "refreshed"}:
        original = runner.workload_case.wait_observation
        counts = {"a": 0, "b": 0}

        def observation(region: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            value = original(region, *args, **kwargs)
            counts[region.cluster] += 1
            if region is a and counts["a"] == (1 if defect == "observation" else 2):
                value["containers"][0]["pod_uid"] = "foreign"
            return value

        monkeypatch.setattr(runner.workload_case, "wait_observation", observation)
    elif defect == "restore":
        original_post = runner.post_observation

        def post(region: Any, observation: dict[str, Any]) -> dict[str, Any]:
            if region is a and observation["node_id"].startswith("node-a"):
                raise RuntimeError("synthetic physical Observation restore failed")
            return original_post(region, observation)

        monkeypatch.setattr(runner, "post_observation", post)
    elif defect.startswith("prewarm-"):
        original_prewarm = runner.ImagePrewarmFixture

        class Prewarm(original_prewarm):
            def cleanup(self) -> dict[str, bool]:
                super().cleanup()
                if defect == "prewarm-error":
                    raise RuntimeError("synthetic prewarm cleanup failed")
                return {"pod": True}

        monkeypatch.setattr(runner, "ImagePrewarmFixture", Prewarm)
    kwargs = {
        "site": site,
        "primary": targets[0],
        "secondary": targets[1],
        "case_dir": tmp_path,
        "job_id": "job-test",
        "attempt_id": "attempt-test",
        "attempt": 1,
        "maintenance_window_end": datetime.now(timezone.utc)
        + timedelta(hours=-1 if defect == "expired" else 1),
    }
    if defect.startswith("prewarm-"):
        result = runner.run_iso001(**kwargs)
        assert result["verdict"] == "FAIL"
        assert (
            any(any(value.values()) for value in result["prewarm_residuals"])
            or result["cleanup_errors"]
        )
    else:
        with pytest.raises(
            (runner.RegionalFixtureError, runner.WorkloadCaseError)
        ) as failed:
            runner.run_iso001(**kwargs)
        assert not any(event.startswith("inject-") for event in events), (
            "failed isolation preparation must not inject either cluster"
        )
        if defect == "restore":
            assert any(
                "Observation restore" in error
                for error in failed.value.outcome["cleanup_errors"]
            ), "physical Observation restore failure was lost from the outcome"
    submitted = {name for name in ("a", "b") if "submit-" + name in events}
    deleted = {name for name in ("a", "b") if "delete-" + name in events}
    assert deleted == submitted, (
        "cleanup must delete exactly the scopes whose submission started"
    )
    if defect == "physical":
        assert events == []


@pytest.mark.parametrize(
    "defect",
    [
        "preflight",
        "busy",
        "observation",
        "node-uid",
        "host",
        "host-cleanup",
        "host-residual",
        "prewarm-cleanup",
        "prewarm-residual",
        "node-recheck",
    ],
)
def test_e2e001_refuses_unsafe_injection_and_never_publishes_pass_after_bad_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    site, targets, events = lifecycle_harness(monkeypatch, tmp_path)
    region = site.regional(targets[0])
    if defect == "preflight":
        monkeypatch.setattr(
            runner, "e2e_preflight", lambda *args: {"errors": ["unproven"], "nodes": []}
        )
    elif defect == "busy":
        monkeypatch.setattr(region, "gpu_workloads", lambda: [{"name": "foreign"}])
    elif defect == "observation":
        original = runner.workload_case.wait_observation

        def observation(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = original(*args, **kwargs)
            result["containers"][0]["pod_uid"] = "foreign"
            return result

        monkeypatch.setattr(runner.workload_case, "wait_observation", observation)
    elif defect in {"node-uid", "node-recheck"}:
        original_nodes = region.gpu_nodes
        calls = []

        def nodes() -> list[dict[str, Any]]:
            calls.append(None)
            if defect == "node-recheck" and "delete-a" in events:
                raise RuntimeError("synthetic node recheck unavailable")
            values = original_nodes()
            if defect == "node-uid" and len(calls) >= 3:
                values[0]["uid"] = "replacement"
            return values

        monkeypatch.setattr(region, "gpu_nodes", nodes)
    elif defect in {"host", "host-cleanup", "host-residual"}:
        original_host = runner.HostProbeFixture

        class Host(original_host):
            def execute(self, action: str, *args: str) -> dict[str, Any]:
                value = super().execute(action, *args)
                if action == "snapshot" and defect == "host":
                    value["kmsg_writable"] = False
                return value

            def cleanup(self) -> dict[str, bool]:
                super().cleanup()
                if defect == "host-cleanup":
                    raise RuntimeError("synthetic host cleanup failure")
                return {"pod": defect == "host-residual", "configmap": False}

        monkeypatch.setattr(runner, "HostProbeFixture", Host)
    else:
        original_prewarm = runner.ImagePrewarmFixture

        class Prewarm(original_prewarm):
            def cleanup(self) -> dict[str, bool]:
                super().cleanup()
                if defect == "prewarm-cleanup":
                    raise RuntimeError("synthetic prewarm cleanup failure")
                return {"pod": True}

        monkeypatch.setattr(runner, "ImagePrewarmFixture", Prewarm)
    kwargs = {
        "site": site,
        "target": targets[0],
        "case_dir": tmp_path,
        "job_id": "job-test",
        "attempt_id": "attempt-test",
        "host_probe_image": "unit@sha256:" + "a" * 64,
        "attempt": 1,
        "maintenance_window_end": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    if defect in {"preflight", "busy", "observation", "node-uid", "host"}:
        with pytest.raises((runner.WorkloadCaseError, runner.RegionalFixtureError)):
            runner.run_e2e001(**kwargs)
        assert "kmsg-inject" not in events
        assert not (tmp_path / "execution-card.json").exists(), (
            "a refused injection cannot publish a BLAST handoff"
        )
    else:
        result = runner.run_e2e001(**kwargs)
        assert result["verdict"] == "FAIL"
        card = json.loads((tmp_path / "execution-card.json").read_text())
        assert card["verdict"] == "FAIL" and card["cleanup_complete"] is False
    if "submit-a" in events:
        assert "delete-a" in events
    if defect in {"preflight", "busy"}:
        assert events == []


@pytest.mark.parametrize(
    "defect",
    ["reference", "boot-missing", "boot-mismatch", "clock", "gpu", "raw-records"],
)
def test_e2e001_requires_kernel_source_identity_not_just_successful_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    site, targets, events = lifecycle_harness(monkeypatch, tmp_path)
    region = site.regional(targets[0])
    original = region.wait_for_workflow

    def state(**kwargs: Any) -> dict[str, Any]:
        value = original(**kwargs)
        if defect == "reference":
            value["event"]["evidence_ref"] = "api-replay://not-kmsg"
        elif defect == "boot-missing":
            value["event"].pop("source_boot_id")
        elif defect == "boot-mismatch":
            value["event"]["source_boot_id"] = "other-boot"
        elif defect == "clock":
            value["event"]["source_monotonic_us"] = "unknown"
        elif defect == "gpu":
            value["event"]["gpu_uuid"] = "GPU-foreign"
        return value

    monkeypatch.setattr(region, "wait_for_workflow", state)
    if defect == "raw-records":
        monkeypatch.setattr(region, "cpu_python", lambda *args: {"evidence": []})
    result = runner.run_e2e001(
        site=site,
        target=targets[0],
        case_dir=tmp_path,
        job_id="job-test",
        attempt_id="attempt-test",
        host_probe_image="unit@sha256:" + "a" * 64,
        attempt=1,
        maintenance_window_end=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    assert result["verdict"] == "FAIL" and result["errors"]
    assert result["checks"]["workflow_contract"] is False
    assert events.index("delete-a") > events.index("settle-a")
    assert (
        json.loads((tmp_path / "execution-card.json").read_text())["verdict"] == "FAIL"
    )
