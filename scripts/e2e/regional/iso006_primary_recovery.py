from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable

from scripts.e2e.regional import run_destr009_workload_restart as recovery
from scripts.e2e.regional import run_workload_acceptance as workload
from scripts.e2e.regional.managed_workload_fixture import ImagePrewarmFixture
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture

CASE_ID = "GF-REGIONAL-ISO-006"


class PrimaryRecovery:
    """One owned A recovery observed while the ISO-006 B network cut is held."""

    def __init__(
        self,
        regional: RegionalLiveFixture,
        *,
        site_file: Path,
        case_dir: Path,
        job_id: str,
        attempt_id: str,
        attempt: int,
        maintenance_window_end: datetime,
    ) -> None:
        self.regional = regional
        self.case_dir = case_dir
        self.job_id = job_id
        self.attempt_id = attempt_id
        self.maintenance_window_end = maintenance_window_end
        self.settings = workload.workload_case_settings(
            regional=regional,
            site_file=site_file,
            manifest=workload.LONG_RUNNING_MANIFEST,
            job_id=job_id,
            attempt_id=attempt_id,
        )
        self.workload = workload.managed_fixture(
            regional,
            manifest=workload.LONG_RUNNING_MANIFEST,
            site_file=site_file,
            job_id=job_id,
            attempt_id=attempt_id,
        )
        self.prewarm = ImagePrewarmFixture(
            regional, case_id=CASE_ID, run_id=f"iso006-a-{attempt}"
        )
        self.submitted = False
        self.source: dict[str, Any] = {}
        self.injection: dict[str, Any] | None = None
        self.nodes: list[dict[str, Any]] = []

    def prepare(self) -> None:
        if datetime.now(timezone.utc) >= self.maintenance_window_end:
            raise RegionalFixtureError("maintenance window ended before A preparation")
        if self.regional.gpu_workloads():
            raise RegionalFixtureError("A already has a GPU workload")
        baseline = workload.workload_store(
            self.regional, job_id=self.job_id, attempt_id=self.attempt_id
        )
        workload.assert_clean_identity_baseline(
            [baseline], job_id=self.job_id, attempt_id=self.attempt_id
        )
        self.nodes = self.regional.gpu_nodes()
        if not workload.gpu_nodes_clean(self.nodes):
            raise RegionalFixtureError("A node state is not clean")
        self.prewarm.create(workload.prewarm_nodes(self.regional))
        self.submitted = True
        self.workload.submit()
        self.source = self.workload.wait_running(timeout_seconds=900)
        self.observation()

    def observation(self) -> dict[str, Any]:
        value = recovery.wait_observation(
            self.regional,
            self.settings,
            node=str(self.source["pods"][0]["node"]),
            expected_gpu_count=24,
        )
        if (
            workload.source_observation_errors(
                value,
                self.source,
                cluster_id=self.regional.settings.cluster_id,
                job_id=self.job_id,
                attempt_id=self.attempt_id,
            )
            or value.get("restart_budget") != 1
        ):
            raise RegionalFixtureError(
                "A source Observation is not fresh and workload-bound"
            )
        return value

    def recover(
        self, observation_deadline: float, *, cut_is_active: Callable[[], bool]
    ) -> dict[str, Any]:
        observation = self.observation()
        if (
            datetime.now(timezone.utc) >= self.maintenance_window_end
            or time.monotonic() >= observation_deadline
            or self.regional.gpu_nodes() != self.nodes
            or not cut_is_active()
        ):
            raise RegionalFixtureError("A recovery window or node identity changed")
        node = str(self.source["pods"][0]["node"])
        injected_at = datetime.now(timezone.utc)
        marker = f"iso006-{self.attempt_id}"
        self.injection = {"node": node, "marker": marker, "observed_after": injected_at}
        payload = recovery.xid11_payload(
            self.settings,
            case_id=CASE_ID,
            marker=marker,
            node=node,
            product=recovery.normalize_product(
                self.regional.node_metadata(node).get("product")
            ),
            observation=observation,
            observed_at=injected_at,
        )
        receipt = self.regional.post_xid_event(payload)
        state = self.regional.wait_for_workflow(
            node=node,
            marker=marker,
            observed_after=injected_at,
            case_dir=self.case_dir,
            timeout_seconds=max(1, int(observation_deadline - time.monotonic())),
            job_id=self.job_id,
            attempt_id=self.attempt_id,
        )
        self.injection["workflow_request_ids"] = [
            str((state.get("workflow") or {}).get("request_id") or "")
        ]
        errors = recovery.workflow_errors(state, expected_gpu_count=24)
        errors.extend(
            workload.recovery_identity_errors(
                state,
                cluster_id=self.regional.settings.cluster_id,
                job_id=self.job_id,
                attempt_id=self.attempt_id,
                node=node,
                marker=marker,
                observed_after=injected_at,
            )
        )
        errors.extend(workload.notification_errors(state))
        if (state.get("event") or {}).get("evidence_ref") != payload["evidence_ref"]:
            errors.append("A event does not bind the submitted software replay")
        if errors:
            raise RegionalFixtureError(
                "A recovery evidence failed: " + "; ".join(errors)
            )
        self.workload.authorize_restart(state)
        target = self.workload.wait_restarted(
            {str(pod["uid"]) for pod in self.source["pods"]},
            timeout_seconds=max(1, int(observation_deadline - time.monotonic())),
        )
        finished = time.monotonic()
        target_attempts = {pod.get("attempt_id") for pod in target.get("pods") or []}
        if (
            finished > observation_deadline
            or datetime.now(timezone.utc) >= self.maintenance_window_end
            or not cut_is_active()
        ):
            raise RegionalFixtureError(
                "A recovery did not complete inside the B-cut window"
            )
        if (
            len(target_attempts) != 1
            or not next(iter(target_attempts))
            or self.attempt_id in target_attempts
        ):
            raise RegionalFixtureError("A target workload has no new unique attempt")
        return {
            "source": self.source,
            "target": target,
            "state": state,
            "injection": receipt,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    def cleanup(self, result: dict[str, Any]) -> list[str]:
        errors = []
        if self.submitted:
            errors.extend(
                workload.cleanup_workload(
                    regional=self.regional,
                    workload=self.workload,
                    case_dir=self.case_dir,
                    job_id=self.job_id,
                    attempt_id=self.attempt_id,
                    injection=self.injection,
                    result=result,
                    label="cluster_a",
                )
            )
        try:
            residuals = self.prewarm.cleanup()
            result["a_prewarm_residuals"] = residuals
            if any(residuals.values()):
                errors.append("A prewarm resources remain")
        except Exception as exc:
            errors.append(f"A prewarm cleanup: {type(exc).__name__}")
        if self.nodes and self.regional.gpu_nodes() != self.nodes:
            errors.append("A node state changed across recovery/cleanup")
        return errors
