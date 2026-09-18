"""Compose admission inhibition and independent cancellation for one shortage run."""

from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.e2e.regional.destr008_admission import ActivationFence, FenceBinding
from scripts.e2e.regional.destr008_watchdog_control import CancellationControl
from scripts.e2e.regional.destr008_watchdog_resources import read_runtime
from scripts.e2e.regional.probes.destr008_cancellation_protocol import (
    Plan,
    Receipt,
    source_sha256,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
)

if TYPE_CHECKING:
    from scripts.e2e.regional.destr008_cancellation import CancellationWatchdog

MINIMUM_HEADROOM_SECONDS = 120


class ShortageSafety:
    def __init__(
        self,
        regional: RegionalLiveFixture,
        *,
        run_id: str,
        attempt_id: str,
        event_id: str,
        fault_node: str,
        spare_node: str,
        spare_uid: str,
        release_id: str,
        directory: Path,
    ) -> None:
        self.regional = regional
        self.directory = directory
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.event_id = event_id
        self.fault_node = fault_node
        self.spare_node = spare_node
        self.release_id = release_id
        self.fence = ActivationFence(
            regional,
            FenceBinding(
                run_id, regional.settings.cluster_id, spare_node, spare_uid, release_id
            ),
            directory,
        )
        self.watchdog: CancellationWatchdog | None = None
        self.control: CancellationControl | None = None
        self.plan: Plan | None = None
        self.quiescent = False

    def arm(
        self,
        observation: dict[str, Any],
        *,
        window_seconds: int,
        maintenance_window_end: datetime,
    ) -> None:
        from scripts.e2e.regional.destr008_cancellation import CancellationWatchdog

        now = int(time.time())
        deadline = min(
            now + window_seconds, int(maintenance_window_end.timestamp()) - 60
        )
        if deadline - now < MINIMUM_HEADROOM_SECONDS:
            raise RegionalFixtureError(
                "insufficient maintenance window for independent cancellation"
            )
        runtime = read_runtime(self.regional)
        fence = self.fence.arm()
        self.plan = Plan.model_validate(
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "cluster_id": self.regional.settings.cluster_id,
                "job_id": self.run_id,
                "attempt_id": self.attempt_id,
                "event_id": self.event_id,
                "release_id": self.release_id,
                "fault_node": self.fault_node,
                "spare_node": self.spare_node,
                "runtime_profile_version": observation["runtime_profile_version"],
                "workload_ids": observation["workload_ids"],
                "probe_sha256": source_sha256(),
                "created_at": now,
                "deadline_at": deadline,
                "fence": fence,
            }
        )
        self.watchdog = CancellationWatchdog(
            self.regional, self.plan, runtime, self.directory
        )
        self.control = self.watchdog.arm()
        if self.plan.deadline_at - int(time.time()) < MINIMUM_HEADROOM_SECONDS:
            raise RegionalFixtureError(
                "watchdog startup consumed the required action headroom"
            )
        self.fence.bind_watchdog(self.control)

    def admit_fixture(self) -> None:
        if self.watchdog is None or self.control is None or self.plan is None:
            raise RegionalFixtureError("shortage safety was not armed")
        self.watchdog.validate_running()
        self.control.assert_armed()
        self.fence.protect_action(self.control)

    def require_bound(self, bound_at: datetime | None, *, margin: int) -> None:
        if self.plan is None or bound_at is None:
            raise RegionalFixtureError("bounded shortage has no immutable expiry")
        if (
            type(margin) is not int
            or margin < 60
            or bound_at.tzinfo is None
            or bound_at.utcoffset() is None
            or not math.isfinite(bound_at.timestamp())
        ):
            raise RegionalFixtureError(
                "bounded shortage requires an aware expiry and at least 60 seconds margin"
            )
        if self.plan.deadline_at + margin > bound_at.timestamp():
            raise RegionalFixtureError(
                "fixture expiry precedes independent cancellation"
            )

    def before_post(self) -> str:
        if self.watchdog is None or self.control is None:
            raise RegionalFixtureError("replacement producer has no safety owner")
        self.watchdog.validate_running()
        self.fence.probe()
        return self.control.claim()

    def acknowledge(self, claim_id: str, response: dict[str, Any]) -> None:
        if self.control is None:
            raise RegionalFixtureError("replacement producer has no control identity")
        self.control.acknowledge(claim_id, response)

    def remaining_seconds(self) -> int:
        if self.plan is None:
            raise RegionalFixtureError("shortage cancellation deadline is unknown")
        return max(1, self.plan.deadline_at - int(time.time()))

    def resume_cleanup(self) -> dict[str, Any]:
        self.quiescent = False
        from scripts.e2e.regional.destr008_cancellation import (
            CancellationWatchdog,
            has_saved_plan,
            load_saved_plan,
        )

        if has_saved_plan(self.directory, self.run_id):
            plan, runtime = load_saved_plan(self.directory, self.run_id)
            if (
                plan.run_id != self.run_id
                or plan.attempt_id != self.attempt_id
                or plan.event_id != self.event_id
                or plan.fault_node != self.fault_node
                or plan.spare_node != self.spare_node
                or plan.release_id != self.release_id
                or plan.fence.node_uid != self.fence.binding.node_uid
            ):
                raise RegionalFixtureError("saved shortage cancellation plan differs")
            self.plan = plan
            self.watchdog = CancellationWatchdog(
                self.regional, plan, runtime, self.directory
            )
        return self.finish(cleanup_only=True)

    def finish(self, *, cleanup_only: bool = False) -> dict[str, Any]:
        self.quiescent = False
        result: dict[str, Any] = {"quiescent": False, "retired": False, "errors": []}
        receipt: Receipt | None = None
        try:
            fence_closed = False
            if self.watchdog is not None:
                with self.fence.operation():
                    fence_closed = self.fence.record["phase"] == "CLOSED"
            if self.watchdog is not None and fence_closed:
                if not self.watchdog.record.closed:
                    result["errors"].append(
                        "continuing previously incomplete watchdog retirement"
                    )
                self.fence.close()
                self.watchdog.cleanup()
                receipt = self.watchdog.record.quiescence
            elif self.watchdog is not None and cleanup_only:
                if not self.watchdog.record.ever_armed:
                    self.watchdog.cleanup()
                    self.fence.close()
                else:
                    receipt = self.watchdog.resume_cleanup(seconds=180)
                    self.control = self.watchdog.control_client()
                    self.fence.close(self.control)
            elif self.control is not None and self.watchdog is not None:
                try:
                    if self.control.read().control.producer.state != "SUBMITTING":
                        self.control.request_close()
                    receipt = self.watchdog.wait_quiescence()
                except Exception as exc:
                    result["errors"].append(
                        f"original watchdog completion: {type(exc).__name__}"
                    )
                    receipt = self.watchdog.resume_cleanup(seconds=180)
                self.fence.close(self.control)
            else:
                if self.control is not None:
                    raise RegionalFixtureError(
                        "watchdog resource ownership is unavailable"
                    )
                if self.watchdog is not None:
                    self.watchdog.cleanup()
                self.fence.close()
            self.quiescent = True
            result["quiescent"] = True
            if receipt is not None:
                result["receipt"] = receipt.model_dump(mode="json")
                if receipt.case_failed:
                    result["errors"].append("independent watchdog rejected this case")
                if (
                    receipt.revocation is not None
                    and receipt.revocation.reason == "DEADLINE"
                ):
                    result["errors"].append(
                        "independent cancellation deadline elapsed before normal closure"
                    )
        except Exception as exc:
            result["errors"].append(f"quiescence: {type(exc).__name__}")
        if self.quiescent and self.watchdog is not None:
            try:
                self.watchdog.cleanup()
                result["retired"] = True
            except Exception as exc:
                result["errors"].append(f"watchdog cleanup: {type(exc).__name__}")
        elif self.quiescent:
            result["retired"] = True
        return result
