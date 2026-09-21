"""Exercise main orchestration with real safety services and controlled workloads."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gpu_fault.host_health import NodeHealthIngestionResult
from gpu_fault.models import WorkflowStatus
from scripts.e2e.regional import run_destr008_warm_spare_shortage as case
from scripts.e2e.regional.destr008_admission import FenceBinding
from scripts.e2e.regional.destr008_journal import ExecutionJournal
from scripts.e2e.regional.destr008_safety import ShortageSafety
from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture
from tests.regional._cov95_destr_warm import FAULT, INCIDENT, NOW, SPARE, WarmHarness
from tests.regional._destr008_causal import build_causal
from tests.regional.test_destr008_cancellation_probe import seed


class Resource:
    HOLD_SECONDS = 840

    def __init__(self, harness: MainHarness, label: str) -> None:
        self.h = harness
        self.label = label
        self.name = "controlled-holder"
        self.deadline_at: datetime | None = None
        self.failsafe_at: datetime | None = None

    def create(self) -> None:
        safety = self.h.causal.safety
        assert safety.fence.record["action_started"] is True, (
            "the complete fence must be admitted before fixture creation"
        )
        assert safety.control is not None and safety.control.assert_armed(), (
            "a real watchdog receipt must predate fixture creation"
        )
        self.h.warm.call(self.label + ".create")
        if self.label == "holder":
            self.deadline_at = self.h.at(self.HOLD_SECONDS)

    def stop(
        self, unit: str, *, restore_seconds: int, delay_seconds: int
    ) -> dict[str, Any]:
        self.h.warm.call("service.stop", unit)
        # The live arm->stop setup (watchdog arm plus the worker capability
        # probes, ~6 min) elapses before the failsafe timer is armed, so the
        # stop-time-anchored restore clears ``deadline + margin``.
        # ``cancellation_window_seconds`` spends exactly
        # ``SERVICE_SETUP_ALLOWANCE_SECONDS`` of that runway, so the fake must
        # model at least that much elapsed setup or ``require_bound`` rejects a
        # fake arm that never slept.
        self.h.causal.cpu.clock.sleep(
            case.SERVICE_SETUP_ALLOWANCE_SECONDS + case.BOUND_MARGIN_SECONDS
        )
        self.failsafe_at = self.h.at(restore_seconds + delay_seconds)
        self.h.recovery_at = self.failsafe_at
        if unit == "kubelet.service":
            self.h.warm.nodes[SPARE]["ready"] = "False"
        else:
            self.h.agent_ready = False
        return {"stopped": True}

    def restore(self) -> dict[str, Any]:
        assert (
            not self.h.causal.gpu.objects and self.h.causal.cpu.journal()["closed"]
        ), "command quiescence and safety retirement must precede restoration"
        self.h.warm.call(self.label + ".restore")
        self.h.agent_ready = True
        return {"restored": True}

    def cleanup(self) -> Any:
        assert (
            not self.h.causal.gpu.objects and self.h.causal.cpu.journal()["closed"]
        ), "a temporary shortage must not disappear before complete safety cleanup"
        self.h.warm.call(self.label + ".cleanup")
        return False if self.label == "holder" else {"pod": False}

    def close(self) -> None:
        self.h.warm.call(self.label + ".close")


class MainHarness:
    def __init__(
        self, path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
    ) -> None:
        self.path, self.scenario = path, scenario
        self.recovery_at: datetime | None = None
        self.agent_ready = True
        self.warm = WarmHarness(case, path, monkeypatch)
        self.causal = build_causal(path, monkeypatch)
        self.causal.cpu.clock.epoch = int(NOW.timestamp())
        self.warm.settings = replace(
            self.warm.settings,
            regional=self.causal.cpu.regional.settings,
            scenarios=(scenario,),
        )
        self.settings = self.warm.settings
        monkeypatch.setattr(self.warm.warm, "regional", self.causal.cpu.regional)
        clock = SimpleNamespace(
            now=lambda _tz=None: self.at(),
            time=self.causal.cpu.clock.now,
            monotonic=self.causal.cpu.clock.monotonic,
            sleep=self.causal.cpu.clock.sleep,
            fromisoformat=datetime.fromisoformat,
        )
        monkeypatch.setattr(self.warm, "clock", clock)
        monkeypatch.setattr(case, "datetime", clock)
        monkeypatch.setattr(case, "time", clock)
        directory = path / "scenarios" / scenario / "safety"
        self.causal.cpu.directory = self.causal.gpu.directory = directory
        job_id, attempt_id = case.scenario_identity(path, 1, scenario)
        self.causal.gpu.binding = FenceBinding(
            job_id,
            self.settings.regional.cluster_id,
            SPARE,
            f"uid-{SPARE}",
            "release-a",
        )
        self.causal.gpu.node["metadata"].update(name=SPARE, uid=f"uid-{SPARE}")
        real_safety = ShortageSafety

        def safety(*args: Any, **kwargs: Any) -> ShortageSafety:
            value = real_safety(*args, **kwargs)
            self.causal.safety = value
            return value

        monkeypatch.setattr(case, "ShortageSafety", safety)
        harness = self

        class Holder(Resource):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(harness, "holder")

        monkeypatch.setattr(case, "GpuHolderFixture", Holder)
        monkeypatch.setattr(
            case, "WarmSpareServiceFixture", lambda *_a, **_k: Resource(self, "service")
        )
        original_post = self.warm.warm.post_synthetic_replacement

        def post(payload: dict[str, Any]) -> dict[str, Any]:
            assert payload["activation_forbidden"] is True
            assert self.causal.safety.control is not None
            assert (
                self.causal.safety.control.read().control.producer.state == "SUBMITTING"
            )
            reply = original_post(payload)
            bound = self.causal.safety.plan
            assert bound is not None and bound.job_id == job_id
            seed(
                self.causal.store,
                bound,
                status=WorkflowStatus.FAILED,
                incident_id=INCIDENT,
                workflow_id="workflow-shortage",
            )
            return {
                **reply,
                "body": NodeHealthIngestionResult(
                    batch_id=bound.event_id,
                    duplicate=False,
                    incident_ids=[INCIDENT],
                    workflow_request_ids=["workflow-shortage"],
                ).model_dump(mode="json"),
            }

        original_wait = self.warm.warm.wait_for_workflow

        def wait(**kwargs: Any) -> dict[str, Any]:
            result = original_wait(**kwargs)
            for execution in result["workflow"]["step_executions"]:
                execution["updated_at"] = self.at().isoformat()
            return result

        monkeypatch.setattr(self.warm.warm, "post_synthetic_replacement", post)
        monkeypatch.setattr(self.warm.warm, "wait_for_workflow", wait)
        monkeypatch.setattr(
            self.warm.warm, "wait_node_ready", self.wait_node_ready, raising=False
        )
        monkeypatch.setattr(
            self.warm.warm, "wait_fleet_readiness", self.wait_fleet_ready, raising=False
        )
        monkeypatch.setattr(
            self.causal.cpu.regional,
            "provider_events",
            self.warm.regional.provider_events,
        )
        monkeypatch.setattr(
            self.causal.cpu.regional,
            "provider_events_provisional",
            self.warm.regional.provider_events_provisional,
        )
        assert (FAULT, SPARE) == (self.settings.fault_node, self.settings.spare_node)
        assert attempt_id, "the source attempt must remain deterministic"

    def at(self, seconds: int = 0) -> datetime:
        return datetime.fromtimestamp(
            self.causal.cpu.clock.now() + seconds, timezone.utc
        )

    def wait_node_ready(self, node: str, *, ready: bool, timeout_seconds: int) -> None:
        assert node == SPARE
        if ready and self.warm.nodes[SPARE]["ready"] != "True":
            assert self.recovery_at is not None
            wait = max(0.0, (self.recovery_at - self.at()).total_seconds())
            assert wait <= timeout_seconds, (
                "the independent restoration must be in budget"
            )
            self.causal.cpu.clock.sleep(wait)
            self.warm.nodes[SPARE]["ready"] = "True"
        assert (self.warm.nodes[SPARE]["ready"] == "True") is ready
        self.warm.call("fixture.node-ready", ready)

    def wait_fleet_ready(self, node: str, *, ready: bool, timeout_seconds: int) -> None:
        assert node == SPARE and timeout_seconds > 0
        assert self.agent_ready is ready
        self.warm.call("fixture.agent-ready", ready)

    def run(self, *, journal: ExecutionJournal | None = None) -> dict[str, Any]:
        profile: Any = self.warm.fault_state["profile"]
        return case.run_scenario(
            self.settings,
            warm=cast(WarmSpareLiveFixture, self.warm.warm),
            regional=self.causal.cpu.regional,
            case_dir=self.path,
            run_dir=self.path,
            attempt=1,
            scenario=self.scenario,
            maintenance_window_end=self.at(3600),
            provider_baseline=self.warm.provider,
            profile_version=profile["profile_version"],
            capabilities=self.warm.capabilities(self.causal.cpu.regional),
            release_id="release-a",
            spare_uid=f"uid-{SPARE}",
            journal=journal,
        )
