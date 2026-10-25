"""GF-ENV-LIVE-010: scope and concurrency, with an independent control task.

The spec (``docs/components/gpu-environment-live-acceptance.md``, 010): a test
node with at least two GPUs and a control node; only one GPU's reading is
changed; the control GPUs' raw readings stay unchanged and the finding /
diagnostic scope names exactly that GPU; the isolation may be node-level; the
STOP must cover the affected distributed job's owned scope, never one rank;
concurrent reports of the same fault merge into one workflow, one notification
and no duplicate STOP; an independent control task on the control node proves
"no change" -- without a second node and a control task the case is NOT_RUN.

This module owns the two parts the 004 chain does not have:

* :class:`ControlTask` -- the independent control task: the single-node
  training fixture pinned to the control node, idle GPU duty, submitted and
  removed through ``managed_workload_fixture`` (``gpu-training-submit``, the
  product's own enrolment; the fixture's ownership journal decides what is
  deleted) and image-prewarmed like the GF-GENERAL-LIVE managed jobs;
* :class:`ScopeJudge` -- the 004 review judge (``env_live_judges.ReviewJudge``)
  plus the scope, control and concurrency stages, fed by the poll's ``extra``
  read (the control node's Node document, its activity since the arm and the
  control task's Pods).

Nothing here decides a verdict for the control plane: the records are the
product's own; a control task that moved, a control node that was touched or
a second workflow for the same fault is FAIL on positive evidence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.e2e.regional import env_live_verdicts as verdicts
from scripts.e2e.regional.env_live_expectations import Expectation, same_value
from scripts.e2e.regional.env_live_judges import (
    SENT_NOTIFICATIONS,
    ReviewFacts,
    ReviewJudge,
)
from scripts.e2e.regional.env_live_watch import ReplayProgress, Snapshot
from scripts.e2e.regional.general_live_verdicts import (
    final_execution,
    operations,
    parse_time,
)
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture

#: The GF-GENERAL-LIVE single-node training fixture (``torchrun`` on the node's
#: eight GPUs, heartbeat lines); rendered node-pinned with idle GPU duty.
CONTROL_TASK_MANIFEST = "single-node-warm-spare-pytorchjob.yaml"
CONTROL_TASK_GPU_COUNT = 8
CONTROL_TASK_START_SECONDS = 900
#: The ``gpu-fault.io/`` taints and ownership marks a touched node carries.
ISOLATION_TAINT_PREFIX = "gpu-fault.io/"


def _time(value: Any) -> datetime | None:
    try:
        return parse_time(value)
    except ValueError:
        return None


def control_task_name(case_id: str, run_id: str) -> str:
    """A DNS-1035 job id for one run's control task (``envctl-<10 hex>``)."""

    digest = hashlib.sha256(f"{case_id}\0{run_id}".encode()).hexdigest()[:10]
    return f"envctl-{digest}"


class ControlTask:
    """The independent control task on the control node (start / snapshot / remove)."""

    def __init__(
        self,
        regional: RegionalLiveFixture,
        *,
        site_file: Path,
        case_dir: Path,
        case_id: str,
        run_id: str,
        node: str,
        gpu_count: int = CONTROL_TASK_GPU_COUNT,
    ) -> None:
        from scripts.e2e.regional.managed_workload_fixture import (
            ImagePrewarmFixture,
            ManagedWorkloadFixture,
            ManagedWorkloadSettings,
            name_managed_workload,
            render_node_pinned_manifest,
        )
        from scripts.e2e.regional.acceptance_gpu_load import set_gpu_duty_percent

        yaml = __import__("yaml")
        self.node = node
        self.name = control_task_name(case_id, run_id)
        self.attempt_id = f"{self.name}-a001"
        source = (
            Path(__file__).with_name("manifests") / "training" / CONTROL_TASK_MANIFEST
        )
        rendered = render_node_pinned_manifest(
            source, case_dir / f"{self.name}.source.yaml", node=node
        )
        document = yaml.safe_load(rendered.read_text(encoding="utf-8"))
        name_managed_workload(document, self.name)
        set_gpu_duty_percent(document, 0)
        document.setdefault("metadata", {}).setdefault("labels", {})
        document["metadata"]["labels"].update(
            {
                "gpu-fault.io/job-id": self.name,
                "gpu-fault.io/attempt-id": self.attempt_id,
            }
        )
        rendered.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        self.prewarm = ImagePrewarmFixture(
            regional,
            case_id=case_id,
            run_id=f"{run_id}-{self.name}",
            state_path=case_dir / f"{self.name}.prewarm.json",
        )
        self.fixture = ManagedWorkloadFixture(
            regional,
            ManagedWorkloadSettings(
                manifest=rendered,
                site_file=site_file,
                job_id=self.name,
                attempt_id=self.attempt_id,
                restart_budget=0,
                expected_pods=1,
                expected_gpu_count=gpu_count,
            ),
            state_path=case_dir / f"{self.name}.ownership.json",
        )

    def start(
        self, *, timeout_seconds: int = CONTROL_TASK_START_SECONDS
    ) -> dict[str, Any]:
        self.prewarm.create([self.node])
        self.fixture.submit()
        return self.fixture.wait_running(timeout_seconds=timeout_seconds)

    def snapshot(self) -> dict[str, Any]:
        """The control task's Pods now (name, uid, node, phase, ready)."""

        return {
            "job_id": self.name,
            "attempt_id": self.attempt_id,
            "node": self.node,
            "pods": self.fixture.pods(),
        }

    def cleanup(self) -> dict[str, Any]:
        from scripts.e2e.regional.managed_workload_fixture import remove_managed_job

        return remove_managed_job(self.fixture, self.prewarm)


def task_pods(snapshot: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("uid")): dict(item)
        for item in ((snapshot or {}).get("pods") or [])
        if item.get("uid")
    }


class ScopeJudge(ReviewJudge):
    """010: the 004 review on the target GPU, plus scope, control and concurrency."""

    reads_extra = True

    def __init__(
        self,
        expectation: Expectation,
        *,
        task_before: Mapping[str, Any] | None,
        control_before: Mapping[str, Any] | None,
        node_before: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(expectation, node_before=node_before)
        self.control_node = expectation.control_node or ""
        self.task_before = task_pods(task_before)
        before = control_before or {}
        self.control_cordoned_before = bool(before.get("unschedulable"))
        self.control_taints_before = self._taints(before)
        self.control_annotations_before = set(before.get("ownership_annotations") or {})
        self.other_samples: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.control_timeline: list[dict[str, Any]] = []
        self.control_activity: dict[str, dict[str, Any]] = {}
        self.task_timeline: list[dict[str, Any]] = []

    @staticmethod
    def _taints(node: Mapping[str, Any]) -> set[str]:
        return {
            str(item.get("key"))
            for item in node.get("taints") or []
            if str(item.get("key") or "").startswith(ISOLATION_TAINT_PREFIX)
        }

    # -- intake ----------------------------------------------------------- #
    def _absorb_state(self, state: Mapping[str, Any], at: datetime) -> None:
        super()._absorb_state(state, at)
        for batch in state.get("batches") or []:
            batch_id = str(batch.get("batch_id"))
            for sample in batch.get("samples") or []:
                if (
                    sample.get("gpu") == self.x.gpu
                    or sample.get("name") != self.x.field
                ):
                    continue
                key = (str(sample.get("gpu")), str(batch.get("observed_at")), batch_id)
                self.other_samples[key] = {
                    "gpu": sample.get("gpu"),
                    "value": sample.get("value"),
                    "observed_at": batch.get("observed_at"),
                    "batch_id": batch_id,
                }

    def observe(self, snapshot: Snapshot, progress: ReplayProgress) -> None:
        super().observe(snapshot, progress)
        extra = snapshot.extra
        if extra is None:
            return
        node = extra.get("node")
        if isinstance(node, Mapping):
            taints = self._taints(node)
            self.control_timeline.append(
                {
                    "at": snapshot.at,
                    "cordoned": bool(node.get("unschedulable")),
                    "touched": (
                        bool(node.get("unschedulable"))
                        and not self.control_cordoned_before
                    )
                    or bool(taints - self.control_taints_before)
                    or bool(
                        set(node.get("ownership_annotations") or {})
                        - self.control_annotations_before
                    ),
                }
            )
        activity = extra.get("activity")
        if isinstance(activity, Mapping):
            for kind, key in (
                ("incidents", "incident_id"),
                ("workflows", "request_id"),
                ("notifications", "notification_id"),
            ):
                for item in activity.get(kind) or []:
                    if self.since_arm(item.get("created_at")) or kind == "workflows":
                        self.control_activity[f"{kind}/{item.get(key)}"] = {
                            "kind": kind,
                            **{k: item.get(k) for k in (key, "state", "status")},
                        }
        task = extra.get("task")
        if isinstance(task, Mapping):
            pods = task_pods(task)
            self.task_timeline.append(
                {
                    "at": snapshot.at,
                    "uids": sorted(pods),
                    "unhealthy": sorted(
                        uid
                        for uid, pod in pods.items()
                        if pod.get("phase") != "Running" or pod.get("ready") is False
                    ),
                    "changed": set(pods) != set(self.task_before),
                }
            )

    # -- facts ------------------------------------------------------------ #
    def control_gpu_findings(self) -> list[dict[str, Any]]:
        """Findings on the replayed field for any GPU but the target, since the arm."""

        return [
            item
            for item in self.findings.values()
            if item.get("canonical_name") == self.x.field
            and item.get("gpu_uuid") not in {None, self.x.gpu}
            and self.since_arm(item.get("observed_at"))
        ]

    def control_gpu_trigger_samples(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.other_samples.values()
            if self.since_arm(item.get("observed_at"))
            and any(same_value(item.get("value"), v) for v in self.x.trigger_values)
        ]

    def review_workflows(self) -> list[dict[str, Any]]:
        """Workflows with an isolating step among the target field's incidents."""

        found = []
        for incident in self.incidents_of(self.gpu_findings({str(self.x.field)})):
            for workflow in self.workflows_of(incident):
                if set(operations(workflow)) & verdicts.ISOLATING_OPERATIONS:
                    found.append(workflow)
        return found

    def stop_executions(self) -> list[dict[str, Any]]:
        result = []
        for workflow in self.workflows.values():
            executions = [
                item
                for item in workflow.get("step_executions") or []
                if item.get("operation") == "STOP_WORKLOADS"
            ]
            result.extend(executions)
            if not executions and final_execution(dict(workflow), "STOP_WORKLOADS"):
                result.append(final_execution(dict(workflow), "STOP_WORKLOADS") or {})
        return result

    # -- decision --------------------------------------------------------- #
    def errors(self, facts: ReviewFacts, now: datetime) -> dict[str, list[str]]:
        stages = super().errors(facts, now)
        label = self.label
        stages.setdefault("scope", [])
        stages["control"] = []
        stages["concurrency"] = []
        for item in self.control_gpu_findings():
            stages["scope"].append(
                f"{label}: {item.get('canonical_name')}={item.get('value')} was found "
                f"on control GPU {item.get('gpu_uuid')}, only {self.x.gpu} was replayed"
            )
        changed = self.control_gpu_trigger_samples()
        if changed:
            stages["scope"].append(
                f"{label}: {len(changed)} accepted {self.x.field} sample(s) of control "
                f"GPUs carry the replayed value ({sorted({i['gpu'] for i in changed})})"
            )
        if any(item["touched"] for item in self.control_timeline):
            stages["control"].append(
                f"{label}: control node {self.control_node} was cordoned, tainted or "
                "taken into ownership during the run"
            )
        if self.control_activity:
            stages["control"].append(
                f"{label}: control node {self.control_node} received "
                f"{sorted(self.control_activity)[:6]} during the run"
            )
        if any(item["changed"] or item["unhealthy"] for item in self.task_timeline):
            last = self.task_timeline[-1]
            stages["control"].append(
                f"{label}: the independent control task changed (pods {last['uids']}, "
                f"unhealthy {last['unhealthy']}, before {sorted(self.task_before)})"
            )
        reviews = self.review_workflows()
        if len({str(w.get("request_id")) for w in reviews}) > 1:
            stages["concurrency"].append(
                f"{label}: concurrent reports of one fault opened "
                f"{sorted(str(w.get('request_id')) for w in reviews)} isolating workflows"
            )
        if facts.incident is not None:
            sent = [
                str(item.get("notification_id"))
                for item in self.notifications.values()
                if item.get("incident_id") == facts.incident.get("incident_id")
                and str(item.get("status")) == "SENT"
            ]
            if len(sent) > 1:
                stages["concurrency"].append(
                    f"{label}: the review incident sent {len(sent)} notifications "
                    f"({sent[:4]}); duplicates must be deduplicated"
                )
        stops = [
            item for item in self.stop_executions() if item.get("status") == "SUCCEEDED"
        ]
        if len(stops) > 1:
            stages["concurrency"].append(
                f"{label}: STOP_WORKLOADS executed {len(stops)} times for one fault"
            )
        return stages

    def evidence(self, facts: ReviewFacts) -> dict[str, Any]:
        value = super().evidence(facts)
        last_task = self.task_timeline[-1] if self.task_timeline else {}
        value.update(
            {
                "control_node": self.control_node,
                "control_node_touched": any(
                    i["touched"] for i in self.control_timeline
                ),
                "control_node_activity": sorted(self.control_activity)[:20],
                "control_task": {
                    "pods_before": sorted(self.task_before),
                    "pods_last": last_task.get("uids"),
                    "unhealthy_last": last_task.get("unhealthy"),
                    "observations": len(self.task_timeline),
                },
                "control_gpu_findings": len(self.control_gpu_findings()),
                "control_gpu_samples": len(
                    [
                        s
                        for s in self.other_samples.values()
                        if self.since_arm(s["observed_at"])
                    ]
                ),
                "review_workflows": sorted(
                    str(w.get("request_id")) for w in self.review_workflows()
                ),
                "notification_statuses": facts.notification_statuses,
                "duplicate_notifications": sorted(
                    str(item.get("notification_id"))
                    for item in self.notifications.values()
                    if str(item.get("status")) in SENT_NOTIFICATIONS - {"SENT"}
                ),
                "stop_executions": len(self.stop_executions()),
            }
        )
        return value

    def complete(self, progress: ReplayProgress, facts: ReviewFacts) -> bool:
        return bool(
            super().complete(progress, facts)
            and self.task_timeline
            and self.control_timeline
            and any(
                item["at"] >= (self.seen.get("review_succeeded") or item["at"])
                for item in self.task_timeline
            )
        )


__all__ = [
    "CONTROL_TASK_GPU_COUNT",
    "CONTROL_TASK_MANIFEST",
    "ControlTask",
    "ScopeJudge",
    "control_task_name",
    "task_pods",
]
