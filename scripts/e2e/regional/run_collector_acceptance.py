#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.kmsg_evidence_records import (  # noqa: E402
    kmsg_record_errors,
)
from scripts.e2e.regional.collector_acceptance_fixture import (  # noqa: E402
    STORE_POLL_SECONDS,
    CollectorAcceptanceFixture,
    collector_setting,
    select_workflow,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseSurface,
    add_live_arguments,
    record_focused_tests,
    reusable_focused_tests,
    run_selected_case,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    required,
    run_case_main,
    settings_from_arguments,
)

CASE_IDS = (
    "GF-REGIONAL-COLLECT-001",
    "GF-REGIONAL-COLLECT-002",
    "GF-REGIONAL-COLLECT-003",
    "GF-REGIONAL-COLLECT-005",
    "GF-REGIONAL-COLLECT-009",
    "GF-REGIONAL-COLLECT-010",
    "GF-REGIONAL-COLLECT-011",
    "GF-REGIONAL-COLLECT-012",
)
PREDECESSORS = {
    "GF-REGIONAL-COLLECT-001": "GF-REGIONAL-E2E-002",
    "GF-REGIONAL-COLLECT-002": "GF-REGIONAL-COLLECT-001",
    "GF-REGIONAL-COLLECT-003": "GF-REGIONAL-COLLECT-002",
    "GF-REGIONAL-COLLECT-005": "GF-REGIONAL-COLLECT-004",
    "GF-REGIONAL-COLLECT-009": "GF-REGIONAL-COLLECT-007",
    "GF-REGIONAL-COLLECT-010": "GF-REGIONAL-COLLECT-009",
    "GF-REGIONAL-COLLECT-011": "GF-REGIONAL-COLLECT-010",
    "GF-REGIONAL-COLLECT-012": "GF-REGIONAL-COLLECT-011",
}
CONFIRMATIONS = {
    case_id: case_id.replace("GF-REGIONAL-", "").replace("-", "") + "_EXECUTE"
    for case_id in CASE_IDS
}
NODE_COUNTS = {
    **{case_id: 1 for case_id in CASE_IDS},
    "GF-REGIONAL-COLLECT-011": 2,
}
# COLLECT-011 injects a fatal SXID whose reset scope the control plane cannot
# trust; the BLOCKED workflow is the one that decided one of these, not "the
# latest for the node" (a restore or an unrelated finding may be newer).
SCOPE_DEPENDENT_ACTIONS = frozenset(
    {"RESET_PARTICIPATING_GPUS", "RESET_ALL_GPUS_AND_NVSWITCHES"}
)
# COLLECT-009: the wrong acknowledgement must be *observed* to be refused. The
# dispatcher re-reads the annotation every ~5s, so three cycles is the least
# that proves the step looked and did not complete.
WRONG_ACKNOWLEDGEMENT_OBSERVATION_SECONDS = 15
# COLLECT-005: how the replay claim is sampled after the collector restart.
# One read right after the restart only proves the collector had not yet
# re-read the file; the count has to stay put over more than one cycle.
REPLAY_OBSERVATION_SAMPLES = 3
REPLAY_OBSERVATION_SECONDS = 10
FM_CURSOR_TIMEOUT_SECONDS = 60


RECENT_EVIDENCE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext

cluster_id, node_id, observed_after_text = sys.argv[1:]
observed_after = datetime.fromisoformat(observed_after_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store
records = [
    item.model_dump(mode="json")
    for item in store.list_raw_evidence(cluster_id, node_id=node_id, limit=5000)
    if item.observed_at >= observed_after
]
print(json.dumps({"records": records}, sort_keys=True, default=str))
"""

COLLECTOR_STATUS_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

# Every accepted batch stamps a collector status, whether or not it also became
# raw evidence. That makes it the only complete record of the delivery cadence.
cluster_id, node_id = sys.argv[1:]
store = ApplicationContext.from_environment().store
print(json.dumps({"records": [
    {
        "collector": str(item.collector),
        "observed_at": item.observed_at.isoformat(),
        "batch_id": item.batch_id or "",
        "sample_count": item.sample_count,
        "errors": list(item.errors),
    }
    for item in store.list_collector_statuses(cluster_id, node_id)
]}, sort_keys=True, default=str))
"""

# The two on-node delivery chains COLLECT-001 judges, keyed by the ``batch_id``
# prefix the producing collector stamps. The prefix is load-bearing: a collector
# status row is keyed by (cluster, node, collector), and the control plane's own
# Kubernetes allocatable collector reports on the same HOST_TELEMETRY channel as
# the on-node host collector, on its own summary phase. Counting the merged
# channel would measure two collectors and attribute the result to one.
COLLECT001_PRODUCERS = {
    "GPU_METRICS": "dcgm-",
    "HOST_TELEMETRY": "host-",
}
# Each chain runs on its own interval and summary period; judging both against
# the max() of the two let the faster chain deliver a whole extra summary and
# still pass, and made the slower chain's "gap" threshold too lax.
COLLECT001_SETTINGS = {
    "GPU_METRICS": (
        "GPU_FAULT_METRICS_INTERVAL_SECONDS",
        "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS",
    ),
    "HOST_TELEMETRY": (
        "GPU_FAULT_HOST_INTERVAL_SECONDS",
        "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS",
    ),
}


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    case_id: str
    nodes: tuple[str, ...]
    host_probe_image: str
    predecessor_path: Path

    @property
    def confirmation(self) -> str:
        return CONFIRMATIONS[self.case_id]

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_COLLECT_CASE": self.case_id,
            "GPU_FAULT_COLLECT_NODES": ",".join(self.nodes),
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def configure(arguments: argparse.Namespace) -> Settings:
    case_id = required(arguments.case, "case ID")
    nodes = tuple(arguments.node)
    expected = NODE_COUNTS[case_id]
    if len(nodes) != expected or len(set(nodes)) != expected:
        raise RegionalFixtureError(
            f"{case_id} requires exactly {expected} distinct --node values"
        )
    predecessor_id = PREDECESSORS[case_id]
    predecessor = (
        Path(arguments.predecessor_evidence).expanduser().resolve()
        if arguments.predecessor_evidence
        else (
            arguments.run_dir / "cases" / predecessor_id / f"{predecessor_id}.json"
        ).resolve()
    )
    return Settings(
        regional=settings_from_arguments(arguments),
        case_id=case_id,
        nodes=nodes,
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        predecessor_path=predecessor,
    )


FOCUSED_TESTS = {
    "GF-REGIONAL-COLLECT-001": [
        "tests/collectors/test_gpu.py::"
        "test_dcgm_edge_filter_suppresses_unchanged_healthy_samples",
        "tests/collectors/test_host.py::"
        "test_host_edge_filter_suppresses_health_and_delivers_edges",
    ],
    "GF-REGIONAL-COLLECT-002": [
        "tests/collectors/test_gpu.py::"
        "test_dcgm_edge_filter_emits_counter_delta_and_candidate_recovery",
    ],
    "GF-REGIONAL-COLLECT-003": [
        "tests/collectors/test_host.py::"
        "test_efa_inventory_ignores_non_efa_rdma_devices",
        "tests/collectors/test_gpu.py::"
        "test_host_collector_reports_persistent_gpu_and_efa_card_loss",
    ],
    "GF-REGIONAL-COLLECT-005": [
        "tests/collectors/test_logs.py::"
        "test_fabric_manager_file_collector_persists_offsets",
        "tests/collectors/test_logs.py::"
        "test_fabric_manager_file_cursor_rolls_back_on_delivery_failure",
    ],
    "GF-REGIONAL-COLLECT-009": [
        "tests/execution/test_misc.py::"
        "test_mechanical_inspection_waits_for_incident_fenced_annotation",
    ],
    "GF-REGIONAL-COLLECT-010": [
        "tests/orchestration/test_misc.py::"
        "test_failed_firmware_remediation_escalates_directly_to_support",
    ],
    "GF-REGIONAL-COLLECT-011": [
        "tests/orchestration/test_sxid_topology.py::"
        "test_conflicting_or_untrusted_topology_fails_closed",
    ],
    "GF-REGIONAL-COLLECT-012": [
        "tests/orchestration/test_misc.py::"
        "test_restart_app_on_an_idle_node_does_not_open_a_workflow",
    ],
}


def focused_tests(
    case_id: str,
    case_dir: Path,
    *,
    reuse_plan: Path | None = None,
) -> dict[str, Any]:
    """Run the case's focused pytest, or reuse the plan's run on the same tree.

    ``--plan`` and ``--execute`` are minutes apart on the same checkout; the
    plan recorded its result with a source digest, and when that digest still
    matches, the ``--execute`` preflight takes the recorded result instead of
    paying for pytest a second time.
    """

    if reuse_plan is not None:
        recorded = reusable_focused_tests(reuse_plan)
        if recorded is not None:
            return {**recorded, "reused_from_plan": str(reuse_plan)}
    command = [sys.executable, "-m", "pytest", "-q", *FOCUSED_TESTS[case_id]]
    completed = RegionalLiveFixture.run(
        command,
        cwd=ROOT,
        check=False,
        timeout=300,
    )
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
    *,
    reuse_plan: Path | None = None,
) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    nodes = [regional.node_snapshot(node) for node in settings.nodes]
    states = [regional.store_snapshot(node=node) for node in settings.nodes]
    predecessor_id = PREDECESSORS[settings.case_id]
    identity = regional.evidence_identity()
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        predecessor_id,
        **identity,
    )
    tests = focused_tests(settings.case_id, case_dir, reuse_plan=reuse_plan)
    errors = []
    if not predecessor["valid"]:
        errors.append(f"{predecessor_id} predecessor evidence is not PASS")
    for node, state in zip(nodes, states, strict=True):
        if node["ready"] != "True":
            errors.append(f"{node['name']} is not Ready")
        if node["ownership_annotations"]:
            errors.append(f"{node['name']} has pre-existing workflow ownership")
        if (state.get("agent") or {}).get("lifecycle_state") != "ACTIVE":
            errors.append(f"{node['name']} Node Agent is not ACTIVE")
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    result = {
        **identity,
        "nodes": nodes,
        "stores": states,
        "predecessor": predecessor,
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def recent_evidence(
    regional: RegionalLiveFixture,
    *,
    node: str,
    observed_after: datetime,
) -> list[dict[str, Any]]:
    value = regional.cpu_python(
        RECENT_EVIDENCE,
        regional.settings.cluster_id,
        node,
        observed_after.isoformat(),
    )
    return cast(list[dict[str, Any]], value.get("records") or [])


def collector_statuses(
    regional: RegionalLiveFixture,
    *,
    node: str,
) -> list[dict[str, Any]]:
    value = regional.cpu_python(
        COLLECTOR_STATUS_PROBE,
        regional.settings.cluster_id,
        node,
    )
    return cast(list[dict[str, Any]], value.get("records") or [])


def evidence_kinds(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        result.setdefault(str(item.get("kind")), []).append(item)
    return result


def parse_stamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def service_state_errors(
    baseline: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    """Every unit that was active before the injection is active after it.

    The fail-closed cases promise "zero side effects beyond the quarantine";
    a collector or agent unit that died along the way is a side effect the
    boot-ID check cannot see.
    """

    errors = []
    for unit, before in sorted((baseline.get("services") or {}).items()):
        if before.get("ActiveState") != "active":
            continue
        current = (after.get("services") or {}).get(unit) or {}
        if current.get("ActiveState") != "active":
            errors.append(
                f"{unit} is {current.get('ActiveState') or 'absent'} after the "
                "case, active before it"
            )
    return errors


@dataclass
class CaseCleanup:
    """What a case took hold of, so ``execute_case`` can let go of it on every path.

    Handlers register a state the moment the store shows a workflow for their
    injection, and an annotation the moment they write it; the success path
    restores through the same registry (``restore``/``remove_annotation``) and
    marks the item done. Whatever is still registered when the case ends --
    because an assertion raised, a wait timed out, or a handler forgot -- is
    restored best-effort by ``finish`` and its failures are recorded as
    cleanup errors, which fail the case.
    """

    incident_states: list[tuple[CollectorAcceptanceFixture, dict[str, Any]]] = field(
        default_factory=list
    )
    annotations: list[tuple[CollectorAcceptanceFixture, str]] = field(
        default_factory=list
    )
    restore_workflows: list[list[dict[str, Any]]] = field(default_factory=list)

    def register_state(
        self,
        fixture: CollectorAcceptanceFixture,
        state: dict[str, Any],
    ) -> None:
        self.incident_states.append((fixture, state))

    def register_annotation(
        self,
        fixture: CollectorAcceptanceFixture,
        annotation: str,
    ) -> None:
        self.annotations.append((fixture, annotation))

    def restore(
        self,
        fixture: CollectorAcceptanceFixture,
        state: dict[str, Any],
        *,
        profile_version: str,
        reason: str,
    ) -> list[dict[str, Any]]:
        result = fixture.restore_incidents(
            state,
            profile_version=profile_version,
            reason=reason,
        )
        # A validated restore that returned proved the node holds no ownership
        # or quarantine taint at all (restore_incidents raises otherwise), so
        # every state registered for this node is released, not only ``state``.
        self.incident_states = [
            item for item in self.incident_states if item[0] is not fixture
        ]
        self.restore_workflows.append(result)
        return result

    def remove_annotation(
        self,
        fixture: CollectorAcceptanceFixture,
        annotation: str,
    ) -> None:
        fixture.regional.kubectl(
            "gpu",
            "annotate",
            "node",
            fixture.node,
            f"{annotation}-",
            check=False,
        )
        self.annotations = [
            item for item in self.annotations if item != (fixture, annotation)
        ]

    def finish(self, *, profile_version: str, reason: str) -> dict[str, Any]:
        """Release everything still registered; never raises, always reports."""

        errors: list[str] = []
        restored: list[list[dict[str, Any]]] = []
        for fixture, annotation in list(self.annotations):
            try:
                self.remove_annotation(fixture, annotation)
            except Exception as exc:
                errors.append(
                    f"{fixture.node}: annotation {annotation} removal failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        for fixture, state in list(reversed(self.incident_states)):
            try:
                restored.append(
                    self.restore(
                        fixture,
                        state,
                        profile_version=profile_version,
                        reason=reason,
                    )
                )
            except Exception as exc:
                errors.append(
                    f"{fixture.node}: validated restore failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        return {"restore_workflows": restored, "errors": errors}


def run_collect001(
    fixture: CollectorAcceptanceFixture,
    case_dir: Path,
) -> dict[str, Any]:
    baseline = fixture.snapshot()
    env = baseline["collector_env"]
    chains = {
        kind: {
            "interval": collector_setting(env, interval_key),
            "summary": collector_setting(env, summary_key),
        }
        for kind, (interval_key, summary_key) in COLLECT001_SETTINGS.items()
    }
    poll_interval = min(item["interval"] for item in chains.values())
    started_at = datetime.now(timezone.utc)
    # Two full summary cycles of the *slowest* chain have to fit inside the
    # window with room for the phase offset at either end, or "at least two
    # deliveries" is not a property of the window for that chain.
    duration = max(
        item["summary"] * 2 + item["interval"] * 4 for item in chains.values()
    )
    # Poll the collector statuses rather than reading the evidence history once
    # at the end. Steady-state deliveries are deliberately *not* persisted as
    # evidence -- the control plane skips capture when a batch has no findings,
    # no collection errors and no reason but `health-summary` -- so the evidence
    # table cannot see the suppressed cadence this case exists to measure. It
    # counted zero GPU_METRICS records on a healthy node and called the chain
    # dead. The status row is upserted on every accepted batch, and polling once
    # per collection interval cannot miss a delivery that is a summary apart.
    deliveries: dict[str, list[str]] = {key: [] for key in COLLECT001_PRODUCERS}
    timeline: list[dict[str, Any]] = []
    deadline = time.monotonic() + duration
    while True:
        for record in collector_statuses(fixture.regional, node=fixture.node):
            prefix = COLLECT001_PRODUCERS.get(str(record.get("collector")))
            if prefix is None or not str(record.get("batch_id")).startswith(prefix):
                continue
            stamp = str(record.get("observed_at"))
            if parse_stamp(stamp) < started_at:
                # A delivery from before the window. It is a real one, but its
                # distance to the first in-window delivery is not a gap this
                # window measured.
                continue
            series = deliveries[str(record["collector"])]
            if stamp not in series:
                series.append(stamp)
        timeline.append(
            {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "delivery_counts": {
                    key: len(value) for key, value in sorted(deliveries.items())
                },
            }
        )
        write_json_atomic(case_dir / "timeline.json", {"entries": timeline})
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_interval)
    records = recent_evidence(
        fixture.regional,
        node=fixture.node,
        observed_after=started_at,
    )
    by_kind = evidence_kinds(records)
    errors = []
    observed: dict[str, Any] = {}
    for kind in sorted(COLLECT001_PRODUCERS):
        interval = chains[kind]["interval"]
        summary = chains[kind]["summary"]
        # What this chain's sampling loop would have delivered with the filter
        # switched off, and what a suppressed steady state should deliver.
        sampled = duration // interval
        suppressed = duration // summary
        stamps = sorted(parse_stamp(item) for item in deliveries[kind])
        gaps = [
            (right - left).total_seconds()
            for left, right in zip(stamps, stamps[1:], strict=False)
        ]
        observed[kind] = {
            "interval_seconds": interval,
            "summary_seconds": summary,
            "collection_samples_in_window": sampled,
            "summary_deliveries_in_window": suppressed,
            "delivered": len(stamps),
            "observed_at": [item.isoformat() for item in stamps],
            "gaps_seconds": gaps,
        }
        if len(stamps) < 2:
            errors.append(f"{kind} delivered fewer than two batches in the window")
            continue
        # One extra delivery beyond the summary count is an edge the node was
        # entitled to report; the sampling cadence is an order of magnitude away
        # from that, which is what makes the criterion decidable.
        if len(stamps) > suppressed + 1:
            errors.append(
                f"{kind} delivered {len(stamps)} batches, closer to the "
                f"{sampled}-sample collection cadence than to the {suppressed} "
                "the summary cadence allows"
            )
        if max(gaps) < summary - interval * 2:
            errors.append(
                f"{kind} never went a full summary window without delivering, "
                f"so no suppression was observed: gaps {gaps}"
            )
    # And the other half of the same mechanism: a batch whose only reason is the
    # periodic summary must not be persisted as evidence at all. This is what
    # the control plane's capture condition promises, and a collector that
    # mislabels its own periodic delivery breaks it silently -- the Kubernetes
    # allocatable collector called every healthy summary a `recovered:` edge and
    # so wrote one raw evidence record per node per summary, forever.
    persisted_summaries = [
        item
        for item in records
        if set(item.get("payload", {}).get("edge_filter_reasons") or [])
        == {"health-summary"}
    ]
    if persisted_summaries:
        errors.append(
            f"{len(persisted_summaries)} steady-state health summaries were "
            "persisted as raw evidence"
        )
    # A recovery is a transition, and preflight established that nothing was
    # broken when the window opened, so nothing inside it can have recovered.
    # `recovered:<resource>` is the mislabel that made every healthy Kubernetes
    # summary look like an edge; the host collector's own recovery reason is the
    # bare `recovered`, which stays legitimate here.
    false_recoveries = [
        item
        for item in records
        if any(
            str(reason).startswith("recovered:")
            for reason in item.get("payload", {}).get("edge_filter_reasons") or []
        )
    ]
    if false_recoveries:
        errors.append(
            f"{len(false_recoveries)} evidence records claim a resource recovery "
            "in a window that began with every resource healthy"
        )
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "collector_env": env,
        "observation_seconds": duration,
        "chains": chains,
        "deliveries": observed,
        "persisted_health_summaries": persisted_summaries,
        "false_recovery_records": false_recoveries,
        "evidence_counts": {key: len(value) for key, value in sorted(by_kind.items())},
    }


def gpu_metrics_stamp(fixture: CollectorAcceptanceFixture) -> str | None:
    """The GPU chain's latest delivery stamp, or ``None`` if it has never spoken.

    Split by ``batch_id`` prefix for the same reason COLLECT-001 does it: the
    status row is keyed by (cluster, node, collector) and more than one producer
    can land on a channel.
    """

    prefix = COLLECT001_PRODUCERS["GPU_METRICS"]
    stamps = [
        str(record.get("observed_at"))
        for record in collector_statuses(fixture.regional, node=fixture.node)
        if str(record.get("collector")) == "GPU_METRICS"
        and str(record.get("batch_id")).startswith(prefix)
    ]
    return max(stamps) if stamps else None


def seconds_until_next_summary(
    last_stamp: str | None,
    *,
    summary: int,
    now: datetime | None = None,
) -> float:
    """How long until the GPU chain's next periodic delivery is due.

    The last status stamp is the phase of the summary clock; the next delivery
    is one summary period after it. Sleeping until then replaces polling the
    store every interval for up to a summary period (330s of exec calls on a
    300s summary) with one computed pause.
    """

    if last_stamp is None:
        return 0.0
    current = now or datetime.now(timezone.utc)
    due = parse_stamp(last_stamp) + timedelta(seconds=summary)
    return max(0.0, min(float(summary), (due - current).total_seconds()))


def run_collect002(
    fixture: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    baseline = fixture.snapshot()
    env = baseline["collector_env"]
    interval = collector_setting(env, "GPU_FAULT_METRICS_INTERVAL_SECONDS")
    summary = collector_setting(env, "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS")
    confirmation = collector_setting(env, "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES")
    if not baseline.get("gpu_power"):
        raise RegionalFixtureError("host probe could not read GPU power limits")
    run_id = f"collect002-a{attempt}"
    # Step one of the procedure: begin from a delivery that just happened, so the
    # next health summary is a whole summary period away. Injecting just before a
    # summary would let the periodic delivery masquerade as the anomaly edge.
    opening = gpu_metrics_stamp(fixture)
    phase_sleep = seconds_until_next_summary(opening, summary=summary)
    if phase_sleep > 0:
        time.sleep(phase_sleep)
    settle_deadline = time.monotonic() + interval * 3
    while True:
        current = gpu_metrics_stamp(fixture)
        if current is not None and current != opening:
            break
        if time.monotonic() >= settle_deadline:
            raise RegionalFixtureError(
                "GPU metrics chain did not deliver within a summary period, so "
                "the case cannot start from a known-quiet point"
            )
        time.sleep(interval)
    quiet_stamp: str = current
    # The edge filter deliberately confirms a candidate over
    # `confirmation` consecutive samples before it speaks, so the earliest
    # honest bound is that many intervals plus the one the anomaly appeared in.
    # Anything past that is the filter waiting for its summary, which is the P0.
    allowed_latency = interval * (confirmation + 1)
    load_seconds = interval * (confirmation + 6)
    started_at = datetime.now(timezone.utc)
    injection = fixture.execute(
        "throttle-gpu",
        "--run-id",
        run_id,
        "--gpu-index",
        str(int(baseline["gpu_power"][0]["index"])),
        "--load-seconds",
        str(load_seconds),
        "--restore-seconds",
        str(load_seconds + interval * 8),
        timeout=300,
    )
    timeline: list[dict[str, Any]] = []
    delivered_at: str | None = None
    restore: dict[str, Any] = {}
    cleanup_error: str | None = None
    try:
        deadline = time.monotonic() + load_seconds
        while True:
            current = gpu_metrics_stamp(fixture)
            timeline.append(
                {
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "gpu_metrics_stamp": current,
                }
            )
            write_json_atomic(case_dir / "timeline.json", {"entries": timeline})
            if current is not None and current != quiet_stamp:
                delivered_at = current
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(interval)
    except BaseException:
        # The restore must still run, but its own failure must not replace
        # the exception that got us here: the deadman timer still covers the
        # power limit, and the case's error is the one the operator needs.
        try:
            fixture.execute("restore-gpu-power-limit", "--run-id", run_id, timeout=300)
        except Exception as exc:  # noqa: BLE001 - recorded, original re-raised
            cleanup_error = f"{type(exc).__name__}: {exc}"
            write_json_atomic(
                case_dir / "cleanup-error.json", {"cleanup_error": cleanup_error}
            )
        raise
    else:
        try:
            restore = fixture.execute(
                "restore-gpu-power-limit", "--run-id", run_id, timeout=300
            )
        except Exception as exc:  # noqa: BLE001 - recorded as a case failure
            cleanup_error = f"{type(exc).__name__}: {exc}"
    records = recent_evidence(
        fixture.regional,
        node=fixture.node,
        observed_after=started_at,
    )
    gpu = evidence_kinds(records).get("GPU_METRICS", [])
    candidate = [
        item
        for item in gpu
        if any(
            "candidate" in str(reason).lower() or "threshold" in str(reason).lower()
            for reason in item.get("payload", {}).get("edge_filter_reasons") or []
        )
    ]
    errors = []
    if cleanup_error:
        errors.append(f"GPU power limit restore failed: {cleanup_error}")
    latency: float | None = None
    if delivered_at is None:
        errors.append(
            "GPU metrics chain delivered nothing while the GPU was pinned at a "
            f"lowered power limit for {load_seconds}s"
        )
    else:
        latency = (parse_stamp(delivered_at) - started_at).total_seconds()
        if latency > allowed_latency:
            errors.append(
                f"the anomaly took {latency:.1f}s to deliver, more than the "
                f"{allowed_latency}s that {confirmation} confirmation samples at "
                f"a {interval}s interval allow"
            )
    if not candidate:
        errors.append("no delivered evidence names a candidate or threshold reason")
    after = fixture.snapshot()
    defaults = {
        item["power_default_limit_w"] for item in after.get("gpu_power") or [{}]
    }
    if (
        len(defaults) != 1
        or {item["power_limit_w"] for item in after.get("gpu_power") or [{}]}
        != defaults
    ):
        errors.append("GPU power limits did not return to the driver default")
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "collector_env": env,
        "opening_stamp": opening,
        "phase_sleep_seconds": phase_sleep,
        "quiet_stamp": quiet_stamp,
        "injection": injection,
        "restore": restore,
        "cleanup_error": cleanup_error,
        "allowed_latency_seconds": allowed_latency,
        "summary_seconds": summary,
        "delivered_at": delivered_at,
        "delivery_latency_seconds": latency,
        "candidate_records": candidate,
        "gpu_power_after": after.get("gpu_power"),
    }


def run_collect003(
    fixture: CollectorAcceptanceFixture,
) -> dict[str, Any]:
    snapshot = fixture.snapshot()
    env = snapshot["collector_env"]
    gpu_actual = len(snapshot["gpu_inventory"])
    efa = snapshot["efa_inventory"]
    expected_gpu = collector_setting(env, "GPU_FAULT_EXPECTED_GPU_COUNT")
    expected_efa = collector_setting(env, "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT")
    errors = []
    if gpu_actual != expected_gpu:
        errors.append("actual GPU count differs from collector configuration")
    if int(efa["active_count"]) != expected_efa:
        errors.append("active EFA count differs from collector configuration")
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "gpu_actual": gpu_actual,
        "gpu_expected": expected_gpu,
        "efa_inventory": efa,
        "efa_expected": expected_efa,
    }


def fm_cursor_entry(reading: dict[str, Any]) -> dict[str, Any] | None:
    """The cursor the FM collector keeps for the live log, out of an `fm-cursor` read."""

    log = reading.get("fabric_manager_log") or {}
    cursor = reading.get("fabric_manager_cursor") or {}
    files = cursor.get("files") or {}
    entry = files.get(str(log.get("path") or ""))
    return cast(dict[str, Any], entry) if isinstance(entry, dict) else None


def fm_cursor_caught_up(reading: dict[str, Any]) -> bool:
    """The collector is active and its cursor sits at the log's current EOF/inode."""

    service = reading.get("service") or {}
    log = reading.get("fabric_manager_log") or {}
    entry = fm_cursor_entry(reading)
    return (
        service.get("ActiveState") == "active"
        and entry is not None
        and int(entry.get("offset", -1)) == int(log.get("size", -2))
        and int(entry.get("inode", -1)) == int(log.get("inode", -2))
    )


def wait_fm_cursor(
    fixture: CollectorAcceptanceFixture,
    *,
    timeout_seconds: int = FM_CURSOR_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Poll `fm-cursor` until the restarted collector is active and caught up."""

    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = fixture.execute("fm-cursor", timeout=60)
        if fm_cursor_caught_up(last):
            return last
        time.sleep(3)
    raise RegionalFixtureError(
        "Fabric Manager collector did not come back active with its cursor at "
        f"the log's EOF: {last}"
    )


def cursor_errors(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    """COLLECT-005's cursor claim: it moved to EOF and survived the restart.

    ``before``/``after`` are the probe's full snapshots around the injection.
    The old check compared the log's size to its own earlier size, which any
    unrelated FM line satisfies; the claim is about the collector's *cursor*.
    """

    errors = []
    log_after = after.get("fabric_manager_log") or {}
    entry_before = fm_cursor_entry(before)
    entry_after = fm_cursor_entry(after)
    if entry_after is None:
        errors.append("Fabric Manager collector has no persisted cursor for the log")
        return errors
    if int(entry_after.get("offset", -1)) != int(log_after.get("size", -2)):
        errors.append(
            f"persisted cursor offset {entry_after.get('offset')} is not the log's "
            f"EOF {log_after.get('size')}"
        )
    if int(entry_after.get("inode", -1)) != int(log_after.get("inode", -2)):
        errors.append("persisted cursor inode is not the live log's inode")
    if entry_before is not None and int(entry_after.get("offset", 0)) <= int(
        entry_before.get("offset", 0)
    ):
        errors.append("persisted cursor did not advance past the appended line")
    return errors


def run_collect005(
    fixture: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    before = fixture.snapshot()
    gpu = before["gpu_inventory"][0]
    marker = f"c005-{int(time.time())}-a{attempt}"
    fixture.execute(
        "append-sxid",
        "--sxid",
        "99999",
        "--marker",
        marker,
        "--pci-bdf",
        str(gpu["pci_bdf"]),
        "--classification",
        "Non-fatal",
        "--message",
        "unknown collection-only event",
    )
    first = fixture.wait_marker(marker, case_dir=case_dir, timeout_seconds=120)
    count = len(first.get("evidence") or [])
    fixture.execute(
        "restart-service",
        "--service",
        "gpu-fault-fabric-manager-collector.service",
    )
    cursor_after_restart = wait_fm_cursor(fixture)
    # The replay claim is sampled over more than one collector cycle; a single
    # read right after the restart could precede the re-read that would replay.
    replay_counts = []
    for index in range(REPLAY_OBSERVATION_SAMPLES):
        if index:
            time.sleep(REPLAY_OBSERVATION_SECONDS)
        replay_counts.append(len(fixture.store_snapshot(marker).get("evidence") or []))
    second = fixture.store_snapshot(marker)
    errors = []
    if count != 1 or any(item != 1 for item in replay_counts):
        errors.append(
            f"Fabric Manager event was lost or replayed: first {count}, "
            f"after restart {replay_counts}"
        )
    if first.get("workflows") or second.get("workflows"):
        errors.append("unknown non-fatal SXID created a workflow")
    after = fixture.snapshot()
    errors.extend(cursor_errors(before, after))
    errors.extend(service_state_errors(before, after))
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "marker": marker,
        "replay_counts": replay_counts,
        "cursor_after_restart": cursor_after_restart,
        "before": before,
        "after": after,
    }


def waiting_workflow(
    fixture: CollectorAcceptanceFixture,
    marker: str,
    *,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = fixture.store_snapshot(marker, scan_evidence=False)
        if any(
            execution.get("status") == "WAITING"
            for workflow in last.get("workflows") or []
            for execution in workflow.get("step_executions", [])
        ):
            return last
        time.sleep(2)
    raise RegionalFixtureError(f"workflow did not reach WAITING: {last}")


def observe_marker(
    fixture: CollectorAcceptanceFixture,
    marker: str,
    *,
    seconds: int,
    poll_seconds: int = 5,
) -> list[dict[str, Any]]:
    """Light store reads over ``seconds``; every one is returned, not just the last."""

    deadline = time.monotonic() + seconds
    samples = [fixture.store_snapshot(marker, scan_evidence=False)]
    while time.monotonic() < deadline:
        time.sleep(max(1, min(poll_seconds, int(deadline - time.monotonic()) + 1)))
        samples.append(fixture.store_snapshot(marker, scan_evidence=False))
    return samples


def mechanical_acknowledgement_details(
    execution: dict[str, Any],
    commands: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Where CHECK_MECHANICALS tells the operator what to write.

    In the single-cluster adapter the step's own ``details`` carry
    ``required_annotation`` and ``required_annotation_value``. On a regional
    site the step is a remote node action: its ``details`` hold only
    ``remote_command_id``/``remote_status`` pointers and the Node Agent's answer
    -- annotation key, ``<incident_id>:<fencing_token>`` value, notification id
    -- travels on the remote command's ``result_details`` (observed live on
    COLLECT-009 attempt 2, which failed with "omitted annotation details" while
    the command had them). Follow the pointer; prefer the step when it answers.
    """

    details = dict(execution.get("details") or {})
    if details.get("required_annotation") and details.get("required_annotation_value"):
        return details
    command_id = str(details.get("remote_command_id") or "")
    for command in commands or []:
        if command_id and str(command.get("command_id")) != command_id:
            continue
        step = command.get("step") or {}
        if not command_id and step.get("operation") != "CHECK_MECHANICALS":
            continue
        result = command.get("result_details") or {}
        if result.get("required_annotation") and result.get(
            "required_annotation_value"
        ):
            return {**details, **result}
    return details


def node_isolation_errors(node: dict[str, Any], *, label: str) -> list[str]:
    """COLLECT-009's "Ready, schedulable, untainted" reading of one node snapshot."""

    errors = []
    if node.get("ready") != "True":
        errors.append(f"node is not Ready {label}")
    if node.get("unschedulable"):
        errors.append(f"node is unschedulable {label}")
    taints = [
        str(item.get("key"))
        for item in node.get("taints") or []
        if str(item.get("key") or "").startswith("gpu-fault.io/")
    ]
    if taints:
        errors.append(f"node carries gpu-fault taints {taints} {label}")
    return errors


def run_collect009(
    fixture: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
    *,
    cleanup: CaseCleanup | None = None,
) -> dict[str, Any]:
    cleanup = cleanup or CaseCleanup()
    baseline = fixture.snapshot()
    gpu = baseline["gpu_inventory"][0]
    marker = f"c009-{int(time.time())}-a{attempt}"
    started_at = datetime.now(timezone.utc)
    fixture.execute(
        "write-xid",
        "--xid",
        "54",
        "--marker",
        marker,
        "--pci-bdf",
        str(gpu["pci_bdf"]),
        "--message",
        "Auxiliary power is not connected to the GPU board",
    )
    state = waiting_workflow(fixture, marker)
    cleanup.register_state(fixture, state)
    errors = []
    # The contract: the node stays Ready, schedulable and untainted for the
    # whole wait. Read it while the step is WAITING, not only at the end.
    errors.extend(
        node_isolation_errors(
            fixture.regional.node_snapshot(fixture.node), label="during WAITING"
        )
    )
    workflow = (state.get("workflows") or [None])[0] or {}
    execution: dict[str, Any] = next(
        (
            item
            for item in workflow.get("step_executions", [])
            if item.get("operation") == "CHECK_MECHANICALS"
        ),
        {},
    )
    details = mechanical_acknowledgement_details(execution, state.get("commands"))
    annotation = str(details.get("required_annotation") or "")
    expected = str(details.get("required_annotation_value") or "")
    if not annotation or not expected:
        raise RegionalFixtureError("CHECK_MECHANICALS omitted annotation details")
    cleanup.register_annotation(fixture, annotation)
    fixture.regional.kubectl(
        "gpu",
        "annotate",
        "node",
        fixture.node,
        f"{annotation}=wrong:0",
        "--overwrite",
    )
    # Three dispatcher cycles with the wrong value in place, each one read.
    wrong_samples = observe_marker(
        fixture,
        marker,
        seconds=WRONG_ACKNOWLEDGEMENT_OBSERVATION_SECONDS,
    )
    fixture.regional.kubectl(
        "gpu",
        "annotate",
        "node",
        fixture.node,
        f"{annotation}={expected}",
        "--overwrite",
    )
    final = fixture.wait_marker(
        marker,
        case_dir=case_dir,
        timeout_seconds=300,
        terminal_workflow=True,
    )
    cleanup.remove_annotation(fixture, annotation)
    if any(
        workflow.get("status") == "SUCCEEDED"
        for sample in wrong_samples
        for workflow in sample.get("workflows") or []
    ):
        errors.append("wrong mechanical acknowledgement was accepted")
    if not any(
        workflow.get("status") == "SUCCEEDED"
        for workflow in final.get("workflows") or []
    ):
        errors.append("correct mechanical acknowledgement did not complete")
    after = fixture.snapshot()
    if after["boot_id"] != baseline["boot_id"]:
        errors.append("node rebooted during the mechanical inspection wait")
    errors.extend(
        node_isolation_errors(
            fixture.regional.node_snapshot(fixture.node), label="after completion"
        )
    )
    errors.extend(service_state_errors(baseline, after))
    ended_at = datetime.now(timezone.utc)
    provider = fixture.regional.provider_events(started_at, ended_at)
    if provider:
        errors.append(f"provider mutations during the case: {provider}")
    # No workflow held the node, so the final restore has nothing to do; the
    # registry still checks that nothing owns it (a fenced step that left an
    # annotation behind would show here).
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "marker": marker,
        "required_annotation": annotation,
        "required_value": expected,
        "wrong_acknowledgement_samples": len(wrong_samples),
        "provider_events": provider,
        # CloudTrail delivers within 15 min; an empty read inside that window
        # is a provisional negative, and is labelled so for the auditor.
        "provider_events_provisional": fixture.regional.provider_events_provisional(
            ended_at
        ),
    }


def inject_blocked_xid(
    fixture: CollectorAcceptanceFixture,
    case_dir: Path,
    *,
    xid: int,
    marker: str,
    message: str,
    minimum_evidence: int = 1,
    observed_after: datetime | None = None,
) -> dict[str, Any]:
    gpu = fixture.snapshot()["gpu_inventory"][0]
    fixture.execute(
        "write-xid",
        "--xid",
        str(xid),
        "--marker",
        marker,
        "--pci-bdf",
        str(gpu["pci_bdf"]),
        "--message",
        message,
    )
    return fixture.wait_marker(
        marker,
        case_dir=case_dir,
        minimum_evidence=minimum_evidence,
        timeout_seconds=300,
        terminal_workflow=True,
        observed_after=observed_after,
    )


def run_collect010(
    fixture: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
    profile_version: str,
    *,
    cleanup: CaseCleanup | None = None,
) -> dict[str, Any]:
    cleanup = cleanup or CaseCleanup()
    baseline = fixture.snapshot()
    marker = f"c010-{int(time.time())}-a{attempt}"
    state = inject_blocked_xid(
        fixture,
        case_dir,
        xid=78,
        marker=marker,
        message="GPU firmware update required",
    )
    cleanup.register_state(fixture, state)
    errors = []
    workflows = state.get("workflows") or []
    if not workflows or workflows[0].get("status") != "BLOCKED":
        errors.append("UPDATE_SWFW workflow is not BLOCKED")
    operations = {
        item.get("operation")
        for workflow in workflows
        for item in workflow.get("official_steps", [])
    }
    if "UPDATE_SOFTWARE_FIRMWARE" in operations:
        errors.append("blocked workflow contains firmware mutation")
    after = fixture.snapshot()
    if baseline["boot_id"] != after["boot_id"]:
        errors.append("blocked firmware case rebooted the node")
    errors.extend(service_state_errors(baseline, after))
    restore = cleanup.restore(
        fixture,
        state,
        profile_version=profile_version,
        reason="COLLECT-010 validated cleanup",
    )
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "marker": marker,
        "restore_workflows": restore,
    }


def nvswitch_pci_bdf(gpu_inventory: list[dict[str, Any]]) -> str:
    """A PCI address for the injected SXid line that is not one of the node's GPUs.

    The acceptance text uses `0000:ab:00.0`, the NVSwitch address NVIDIA's own
    examples carry. It is only usable while no GPU sits in that slot, so the
    live inventory is checked and the next free bus taken otherwise.
    """

    taken = {
        str(item.get("pci_bdf") or "").lower().split(".")[0] for item in gpu_inventory
    }
    for bus in ("ab", "ac", "ad", "ae", "af", "ba", "bb", "bc"):
        candidate = f"0000:{bus}:00"
        if candidate not in taken:
            return candidate + ".0"
    raise RegionalFixtureError("no PCI slot free of GPUs for the SXid line")


def run_collect011(
    fixtures: list[CollectorAcceptanceFixture],
    case_dir: Path,
    attempt: int,
    profile_version: str,
    *,
    cleanup: CaseCleanup | None = None,
) -> dict[str, Any]:
    cleanup = cleanup or CaseCleanup()
    markers = [
        f"c011-access-{int(time.time())}-a{attempt}",
        f"c011-unknown-{int(time.time())}-a{attempt}",
    ]
    baselines = []
    states = []
    for index, (fixture, marker) in enumerate(zip(fixtures, markers, strict=True)):
        baseline = fixture.snapshot()
        baselines.append(baseline)
        inventory = baseline["gpu_inventory"]
        # The SXid line names the NVSwitch, never a GPU. With a GPU's own BDF
        # here the control plane resolves the ACCESS scope to that GPU
        # (`FaultIngestionService._enrich_sxid_scope`), the reset becomes
        # executable, and the case that exists to prove the fail-closed gate
        # instead resets a production GPU (observed 2026-09-06 02:13Z).
        switch_bdf = nvswitch_pci_bdf(inventory)
        injected_at = datetime.now(timezone.utc)
        fixture.execute(
            "append-sxid",
            "--sxid",
            "11001",
            "--marker",
            marker,
            "--pci-bdf",
            switch_bdf,
            "--classification",
            "Fatal",
            "--message",
            "NVLINK_FATAL_ERROR",
            "--include-switch" if index == 0 else "--no-include-switch",
        )
        state = fixture.wait_marker(
            marker,
            case_dir=case_dir / f"direction-{index + 1}",
            timeout_seconds=300,
            terminal_workflow=True,
            observed_after=injected_at,
        )
        cleanup.register_state(fixture, state)
        states.append(state)
    errors = []
    selected = []
    for fixture, baseline, state in zip(fixtures, baselines, states, strict=True):
        workflow = select_workflow(
            state.get("workflows") or [],
            official_actions=SCOPE_DEPENDENT_ACTIONS,
        )
        selected.append(workflow)
        if workflow is None:
            errors.append(
                f"{fixture.node}: no workflow decided a scope-dependent reset"
            )
        elif workflow.get("status") != "BLOCKED":
            errors.append(f"{fixture.node}: scope-dependent SXID did not fail closed")
        after = fixture.snapshot()
        if after["boot_id"] != baseline["boot_id"]:
            errors.append(f"{fixture.node}: node rebooted during the fail-closed SXID")
        errors.extend(service_state_errors(baseline, after))
    restores = [
        cleanup.restore(
            fixture,
            state,
            profile_version=profile_version,
            reason="COLLECT-011 validated cleanup",
        )
        for fixture, state in zip(fixtures, states, strict=True)
    ]
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "markers": markers,
        "selected_workflow_ids": [(item or {}).get("request_id") for item in selected],
        "restore_workflows": restores,
    }


def restart_app_monitor_only_errors(
    decisions: list[dict[str, Any]],
    workflows: list[dict[str, Any]],
    incidents: list[dict[str, Any]],
) -> list[str]:
    """COLLECT-012: RESTART_APP on an idle node is MONITOR_ONLY, not a workflow.

    Before 55272a0 the compiler wrapped this in STOP -> RESTART with no
    workload id, hit the generic fail-closed path and quarantined the node
    (SAFETY_PENDING -> BLOCKED). Now ``_direct_resolution`` returns
    MONITOR_ONLY / NO_ACTION with the official action preserved: an INFO
    marker and the investigatory notification, the incident closes RECOVERED,
    and no workflow (hence no MARK_UNSCHEDULABLE) is ever planned.
    """

    errors: list[str] = []
    decision = next(
        (item for item in decisions if item.get("official_action") == "RESTART_APP"),
        None,
    )
    if decision is None:
        errors.append("no RESTART_APP decision for the injected XID")
    else:
        if decision.get("disposition") != "MONITOR_ONLY":
            errors.append(
                f"RESTART_APP disposition={decision.get('disposition')!r}, "
                "expected MONITOR_ONLY"
            )
        if decision.get("action") != "NO_ACTION":
            errors.append(
                f"RESTART_APP action={decision.get('action')!r}, expected NO_ACTION"
            )
        if not any(
            "no managed application to restart" in str(item)
            for item in decision.get("reasons") or []
        ):
            errors.append(
                f"RESTART_APP reasons do not name the idle-node rule: "
                f"{decision.get('reasons')}"
            )
        if decision.get("workflow_request_id") is not None:
            errors.append("RESTART_APP decision points at a workflow on an idle node")
    if workflows:
        errors.append(
            "RESTART_APP on an idle node opened a workflow: "
            f"{[item.get('status') for item in workflows]}"
        )
    if not any(item.get("state") == "RECOVERED" for item in incidents):
        errors.append(
            "no RECOVERED incident for the RESTART_APP XID: "
            f"{[item.get('state') for item in incidents]}"
        )
    if any(item.get("workflow_request_id") for item in incidents):
        errors.append("RESTART_APP incident opened a workflow on an idle node")
    return errors


def wait_monitor_only(
    fixture: CollectorAcceptanceFixture,
    marker: str,
    *,
    case_dir: Path,
    minimum_evidence: int = 1,
    timeout_seconds: int = 300,
    observed_after: datetime | None = None,
) -> dict[str, Any]:
    """Poll until the RESTART_APP decision is recorded and the evidence has landed.

    ``wait_marker`` keys on a *workflow* reaching a terminal status; a
    MONITOR_ONLY decision opens none, so it would spin until timeout. The
    decision (and the RECOVERED incident the store probe reads from it) is the
    terminal signal here. Each poll is the light read; the evidence scan runs
    only once the decision exists.
    """

    deadline = time.monotonic() + timeout_seconds
    timeline: list[dict[str, Any]] = []
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = fixture.store_snapshot(
            marker, observed_after=observed_after, scan_evidence=False
        )
        decided = any(
            item.get("official_action") == "RESTART_APP"
            for item in last.get("decisions") or []
        )
        if decided:
            last = fixture.store_snapshot(
                marker, observed_after=observed_after, scan_evidence=True
            )
        timeline.append(
            {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "decision_count": len(last.get("decisions") or []),
                "evidence_count": len(last.get("evidence") or []),
                "workflow_count": len(last.get("workflows") or []),
                "incident_states": [
                    item.get("state") for item in last.get("incidents") or []
                ],
            }
        )
        write_json_atomic(case_dir / "timeline.json", {"entries": timeline})
        if decided and len(last.get("evidence") or []) >= minimum_evidence:
            return last
        time.sleep(STORE_POLL_SECONDS)
    raise RegionalFixtureError(
        f"RESTART_APP decision/evidence did not converge: {last}"
    )


def run_collect012(
    fixture: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
    profile_version: str,
    *,
    cleanup: CaseCleanup | None = None,
) -> dict[str, Any]:
    baseline = fixture.snapshot()
    boot_id = str(baseline.get("boot_id") or "")
    markers = []
    states: list[dict[str, Any]] = []
    errors: list[str] = []
    # Samples 1 and 2 are the same XID 13 text written twice under one marker,
    # so the kernel lines are byte-identical: the second write must earn its
    # own kmsg sequence (distinct evidence). Sample 3 is the second XID (31),
    # the most common MMU fault. Since a RESTART_APP on an idle node no longer
    # quarantines it (55272a0), the node is never isolated between the two
    # XIDs -- there is no ownership fence to restore around, so both inject
    # straight through without a restore step.
    stamp = int(time.time())
    for offset, xid in enumerate((13, 13, 31), start=1):
        marker = (
            f"c012-13-{stamp}-a{attempt}"
            if xid == 13
            else f"c012-31-{stamp}-a{attempt}"
        )
        markers.append(marker)
        gpu = fixture.snapshot()["gpu_inventory"][0]
        injected_at = datetime.now(timezone.utc)
        fixture.execute(
            "write-xid",
            "--xid",
            str(xid),
            "--marker",
            marker,
            "--pci-bdf",
            str(gpu["pci_bdf"]),
            "--message",
            "RESTART_APP collector path",
        )
        state = wait_monitor_only(
            fixture,
            marker,
            case_dir=case_dir / f"sample-{offset}",
            minimum_evidence=2 if offset == 2 else 1,
            observed_after=injected_at,
        )
        states.append(state)
    record_ids = {
        item.get("record_id")
        for state in states
        for item in state.get("evidence") or []
    }
    if len(record_ids) < 3:
        errors.append("distinct kmsg sequences did not create distinct evidence")
    errors.extend(
        kmsg_record_errors(
            states[0].get("evidence") or [],
            states[1].get("evidence") or [],
            boot_id=boot_id,
        )
    )
    for state in states:
        errors.extend(
            restart_app_monitor_only_errors(
                state.get("decisions") or [],
                state.get("workflows") or [],
                state.get("incidents") or [],
            )
        )
    after = fixture.snapshot()
    if after.get("boot_id") != baseline.get("boot_id"):
        errors.append("node rebooted during the RESTART_APP MONITOR_ONLY case")
    errors.extend(service_state_errors(baseline, after))
    # No workflow ever held the node, so there is nothing to restore; the node
    # must have stayed schedulable and untainted the whole time.
    errors.extend(
        node_isolation_errors(
            fixture.regional.node_snapshot(fixture.node),
            label="after the RESTART_APP XIDs",
        )
    )
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "markers": markers,
        "boot_id": boot_id,
        "record_ids": sorted(str(item) for item in record_ids if item),
    }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    details = {
        "risk": "case-defined",
        "case_id": settings.case_id,
        "predecessor": preflight["predecessor"],
        "nodes": list(settings.nodes),
        "mutation": {
            "GF-REGIONAL-COLLECT-001": "read evidence history for two summary cycles",
            "GF-REGIONAL-COLLECT-002": (
                "lower every GPU's enforced power limit to the driver minimum "
                "and load one GPU against it, with a deadman restore"
            ),
            "GF-REGIONAL-COLLECT-003": "read live GPU/EFA inventory and env",
            "GF-REGIONAL-COLLECT-005": "append one unknown non-fatal SXID and restart collector",
            "GF-REGIONAL-COLLECT-009": "inject XID54 and write exact acknowledgement annotation",
            "GF-REGIONAL-COLLECT-010": "inject XID78 fail-closed quarantine only",
            "GF-REGIONAL-COLLECT-011": "append scope-dependent SXID on two nodes",
            "GF-REGIONAL-COLLECT-012": (
                "inject XID13/31 via real kmsg; idle node reads MONITOR_ONLY, "
                "no workflow, node untouched"
            ),
        }[settings.case_id],
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "node_uids": [item["uid"] for item in preflight["nodes"]],
        },
        "stop_conditions": [
            "predecessor evidence is not PASS",
            "node, Agent or release baseline drifts",
            "host probe cannot read current collector configuration",
            "case-specific evidence or rollback assertion fails",
        ],
        "rollback": {
            "host_probe_has_active_deadline": True,
            "temporary_persistence_or_annotations_are_restored": True,
            "quarantine_is_restored_only_by_validation_workflow": True,
        },
        "preflight": preflight,
    }
    record_focused_tests(details, preflight["focused_tests"])
    return details


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / settings.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(
        settings,
        case_dir,
        reuse_plan=case_dir / "plan.json",
    )
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    if datetime.now(timezone.utc) >= maintenance_window_end:
        raise RegionalFixtureError("approved maintenance window has ended")
    regional = RegionalLiveFixture(settings.regional)
    fixtures = [
        CollectorAcceptanceFixture(
            regional,
            node=node,
            image=settings.host_probe_image,
            case_id=settings.case_id,
            run_id=f"{settings.case_id.lower()}-{attempt}-{index}",
        )
        for index, node in enumerate(settings.nodes)
    ]
    result: dict[str, Any] = {
        "case_id": settings.case_id,
        "attempt": attempt,
        "verdict": "FAIL",
        **regional.evidence_identity(),
    }
    profile_version = str(
        (preflight["stores"][0].get("profile") or {}).get("profile_version") or ""
    )
    cleanup = CaseCleanup()
    try:
        for fixture in fixtures:
            fixture.create()
        handlers = {
            "GF-REGIONAL-COLLECT-001": lambda: run_collect001(fixtures[0], case_dir),
            "GF-REGIONAL-COLLECT-002": lambda: run_collect002(
                fixtures[0], case_dir, attempt
            ),
            "GF-REGIONAL-COLLECT-003": lambda: run_collect003(fixtures[0]),
            "GF-REGIONAL-COLLECT-005": lambda: run_collect005(
                fixtures[0], case_dir, attempt
            ),
            "GF-REGIONAL-COLLECT-009": lambda: run_collect009(
                fixtures[0], case_dir, attempt, cleanup=cleanup
            ),
            "GF-REGIONAL-COLLECT-010": lambda: run_collect010(
                fixtures[0], case_dir, attempt, profile_version, cleanup=cleanup
            ),
            "GF-REGIONAL-COLLECT-011": lambda: run_collect011(
                fixtures, case_dir, attempt, profile_version, cleanup=cleanup
            ),
            "GF-REGIONAL-COLLECT-012": lambda: run_collect012(
                fixtures[0], case_dir, attempt, profile_version, cleanup=cleanup
            ),
        }
        outcome = handlers[settings.case_id]()
        result.update(outcome)
        if regional.cpu_blast_snapshot() != preflight["cpu_blast"]:
            result.setdefault("errors", []).append(
                "control-plane EKS state differs from baseline"
            )
            result["verdict"] = "FAIL"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Whatever the case still holds -- a quarantine its assertions never
        # got to release, an acknowledgement annotation -- goes back through
        # the validated path here, and a failure to do so fails the case.
        released = cleanup.finish(
            profile_version=profile_version,
            reason=f"{settings.case_id} cleanup after case end",
        )
        if released["restore_workflows"]:
            result["cleanup_restore_workflows"] = released["restore_workflows"]
        if released["errors"]:
            result.setdefault("cleanup_errors", []).extend(released["errors"])
            result["verdict"] = "FAIL"
        residuals = {}
        for fixture in fixtures:
            try:
                residuals[fixture.node] = fixture.cleanup()
            except Exception as exc:
                residuals[fixture.node] = {"cleanup_error": True}
                result.setdefault("cleanup_errors", []).append(
                    f"{fixture.node}: {type(exc).__name__}: {exc}"
                )
                result["verdict"] = "FAIL"
        result["probe_residuals"] = residuals
        if any(any(value.values()) for value in residuals.values()):
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{settings.case_id}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run one guarded regional Collector acceptance case."
    )
    add_live_arguments(value, confirmation="CASE_SPECIFIC_CONFIRMATION")
    value.add_argument("--case", choices=CASE_IDS, required=True)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--node", action="append", default=[])
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


CASE = CaseSurface(
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_selected_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
