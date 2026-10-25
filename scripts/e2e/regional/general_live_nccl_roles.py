"""GF-GENERAL-LIVE-011 role, thread, process and attempt-change variants.

Native. ``role_mismatch``, ``tid_namespace``, ``wchan_zero``, ``cpu_interval``
and ``pyspy_nonzero`` are the ``active_to_zero`` stall of ``general_live_nccl``
(an active EFA baseline, one rank in an extra collective, the pair at zero for
the hung threshold, the read-only diagnostic DAG to ESCALATED, traffic back on
release) on the managed two-node job in one of the role modes of
``general_live_nccl_role_trainer``; the verdict adds what the product's
``COLLECT_HUNG_TRIAGE`` made of the ranks' declarations and behaviour. The
records are read as the executor stored them: each node's triage result is the
summary codec's envelope, decoded here; the classification
(``hung_triage_decision``, ``execution/hung_classification.py``) is read from
the bundle step the product bound it into (``hung_bundle.bind_hung_bundle``),
else from the last triage execution. Nothing is posted.

The oracle for every expectation is the trainer's own account: the ``ROLES``,
``SHARER`` and ``HOLDER`` lines each rank printed say what it declared, by
container-native thread id, and which child processes it started. A sampled
rank whose declaration never reached the log, a child the container could not
start, a holder it could not build: INCONCLUSIVE, never FAIL. A node the
product sampled is owed every one of its ranks (eight per node, under the
product's sixteen-process cap) and every GPU process the mode added: a missing
one is a coverage FAIL. Which node the product samples first is the product's
choice; both are sampled in a wave of their own, so every cohort is node-local
and the expectations are stated per sampled node.

``attempt_change`` is a negative test of the EFA rule's attempt identity:
attempt A is stalled only until the product warned on both nodes, held at zero
for a while under the hung threshold, then deleted; the same job comes back as
attempt B (``gpu-training-submit --attempt-id <job>-a002``, the trainer in its
quiet mode) and goes active. B's startup silence must start at WARMUP: no
ZERO_WARNING or HUNG_SUSPECTED may ever name B, no HUNG_SUSPECTED may name A,
and each node's EFA traffic state must name B, first observed after the
switch, with no inherited zero streak (``efa_traffic_state.
next_efa_traffic_state``: a newer sample of another attempt starts at WARMUP).

Lanes: both -- the plain-EKS cluster has a second EFA node for ``--node-b``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from gpu_fault.hung_summary_codec import decode_summary
from scripts.e2e.regional import general_live_nccl as nccl
from scripts.e2e.regional import general_live_nccl_role_trainer as plan
from scripts.e2e.regional.general_live_common import run_survey
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    run_identity,
)
from scripts.e2e.regional.general_live_nccl_job import ManagedNcclJob
from scripts.e2e.regional.general_live_registry import NATIVE, VariantSpec
from scripts.e2e.regional.general_live_verdicts import parse_time, verdict

CASE = nccl.CASE
LANES = ("eks", "hp")
WORLD_SIZE = 2 * plan.PROCS_PER_NODE
#: The whole trainer log fits: 3600 heartbeats per Pod plus the markers.
LOG_TAIL = 6000
#: ``collect_process_series`` samples three times, about a second apart.
STACK_SAMPLES = 3
PROCESS_WINDOWS = 2
#: Peer comparison verdicts (``execution/process_cohorts.py``).
ROLE_COHORT = "ROLE_COHORT"
NO_COHORT = "GROUP_ROLE_OR_HARDWARE_UNKNOWN"
FEW_PEERS = "INSUFFICIENT_OR_NONOVERLAPPING_PEERS"
#: A cohort needs three overlapping peers before anything is compared.
MIN_COHORT_PEERS = 3
DEVICE_NOT_PID = "DEVICE_NOT_PID"
#: Observation codes (``execution/process_observations.py``).
THREAD_WAIT = "REPEATED_THREAD_WAIT"
STACK_CODES = frozenset({"REPEATED_CALL_LOCATION", "REPEATED_DISTRIBUTED_CALL_CONTEXT"})
#: The spinner's share of a core while stalled, under any reasonable load.
SPINNER_MIN_CORES = 0.5
#: Slack between a window's own elapsed time and its samples' timestamps.
INTERVAL_TOLERANCE_SECONDS = 0.5
#: attempt_change: how long A stays at zero (warning at 60 s, delivered within
#: the 300 s host summary) and the latest A may be deleted, both under the
#: 600 s hung threshold with a host summary of margin.
A_ZERO_SECONDS = 420
A_DELETE_BY_SECONDS = 540
SWITCH_TOLERANCE_SECONDS = 30
HOLDER_NAME = plan.HOLDER_EXE

#: Read-only: this job's SITE_EFA_TRAFFIC incidents of the given attempts on
#: the attempt nodes, their workflows, and each node's EFA traffic state row
#: (the product's per-node attempt state, ``models.EfaTrafficState``).
ATTEMPT_PROBE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState
from gpu_fault.store import NotFoundError

cluster_id, node_list, job_id, attempt_list, since_text = sys.argv[1:6]
node_ids = {item for item in node_list.split(",") if item}
attempt_ids = {item for item in attempt_list.split(",") if item}
since = datetime.fromisoformat(since_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store

incidents = []
workflows = {}
for incident in store.list_incidents_by_state(
    cluster_id, list(IncidentState), node_ids=node_ids
):
    if incident.policy_source != "SITE_EFA_TRAFFIC":
        continue
    if incident.job_id != job_id and incident.attempt_id not in attempt_ids:
        continue
    if incident.created_at < since and incident.updated_at < since:
        continue
    incidents.append({
        "incident_id": incident.incident_id,
        "event_id": incident.event_id,
        "event_type": incident.event_type,
        "state": incident.state.value,
        "attempt_id": incident.attempt_id,
        "job_id": incident.job_id,
        "node_ids": list(incident.node_ids),
        "reasons": list(incident.reasons or []),
        "workflow_request_id": incident.workflow_request_id,
        "created_at": incident.created_at.isoformat(),
        "updated_at": incident.updated_at.isoformat(),
    })
    if incident.workflow_request_id and incident.workflow_request_id not in workflows:
        try:
            workflows[incident.workflow_request_id] = store.get_workflow(
                incident.workflow_request_id
            ).model_dump(mode="json")
        except NotFoundError:
            pass

states = {}
for node_id in sorted(node_ids):
    try:
        states[node_id] = store.get_efa_traffic_state(
            store.efa_traffic_state_key(cluster_id, node_id)
        ).model_dump(mode="json")
    except NotFoundError:
        states[node_id] = None

print(json.dumps({
    "incidents": incidents,
    "workflows": workflows,
    "efa_states": states,
}, sort_keys=True, default=str))
"""


# --------------------------------------------------------------------------- #
# the trainer's account (what each rank declared and started)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Declaration:
    """One rank's ``ROLES`` line."""

    rank: int
    mode: str
    pid: int
    group: str | None
    stage: str | None
    expert: str | None
    members: tuple[int, ...]
    #: ``{native tid: (role, thread name)}``; empty for an undeclared rank.
    roles: Mapping[str, tuple[str, str]]
    clk_tck: int
    burn: bool
    share_gpu: bool
    holder: bool

    @property
    def declared(self) -> bool:
        return self.group is not None

    def tids(self, role: str, name_prefix: str = "") -> list[str]:
        return sorted(
            tid
            for tid, (kind, name) in self.roles.items()
            if kind == role and name.startswith(name_prefix)
        )

    @property
    def training_tid(self) -> str | None:
        tids = self.tids("TRAINING")
        return tids[0] if len(tids) == 1 else None


def _optional(value: str) -> str | None:
    return None if value in {"", "-"} else value


def read_declaration(tokens: Mapping[str, str]) -> Declaration:
    """A ``ROLES`` line's tokens as a :class:`Declaration` (ValueError if not)."""

    try:
        roles = plan.declared_roles(tokens)
        members = tuple(
            int(item)
            for item in (_optional(tokens.get("members", "-")) or "").split(",")
            if item
        )
        return Declaration(
            rank=int(tokens["rank"]),
            mode=tokens["mode"],
            pid=int(tokens["pid"]),
            group=_optional(tokens.get("group", "-")),
            stage=_optional(tokens.get("stage", "-")),
            expert=_optional(tokens.get("expert", "-")),
            members=members,
            roles=roles,
            clk_tck=int(tokens["clk_tck"]),
            burn=tokens.get("burn") == "1",
            share_gpu=tokens.get("share_gpu") == "1",
            holder=tokens.get("holder") == "1",
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"malformed ROLES line {dict(tokens)}: {exc}") from exc


def declarations(account: Mapping[str, Sequence[str]]) -> dict[int, Declaration]:
    """Each rank's last ``ROLES`` line across the Pods' logs."""

    found: dict[int, Declaration] = {}
    for lines in account.values():
        for tokens in plan.marker_lines(lines, plan.ROLES_MARKER):
            declaration = read_declaration(tokens)
            found[declaration.rank] = declaration
    return found


def children(
    account: Mapping[str, Sequence[str]], marker: str
) -> dict[int, dict[str, str]]:
    """``{rank: tokens}`` of the ``SHARER``/``HOLDER`` lines."""

    found: dict[int, dict[str, str]] = {}
    for lines in account.values():
        for tokens in plan.marker_lines(lines, marker):
            if tokens.get("rank", "").isdecimal() and tokens.get("pid", "").isdecimal():
                found[int(tokens["rank"])] = dict(tokens)
    return found


def fixture_errors(account: Mapping[str, Sequence[str]]) -> list[str]:
    return [
        f"rank {tokens.get('rank')}: {tokens.get('what')} {tokens.get('error')}"
        for lines in account.values()
        for tokens in plan.marker_lines(lines, plan.FIXTURE_ERROR_MARKER)
    ]


def read_account(job: ManagedNcclJob) -> dict[str, list[str]]:
    """The marker lines of each Pod's trainer log (its own account)."""

    account: dict[str, list[str]] = {}
    for item in job.fixture.pods():
        text = job.ctx.regional.kubectl(
            "gpu", "logs", str(item["name"]), "-c", "pytorch", f"--tail={LOG_TAIL}"
        )
        account[str(item["name"])] = [
            line
            for line in text.splitlines()
            if any(marker in line for marker in plan.MARKERS)
        ]
    return account


# --------------------------------------------------------------------------- #
# the product's records (pure)
# --------------------------------------------------------------------------- #
def hung_workflow(records: Mapping[str, Any]) -> dict[str, Any]:
    incident = nccl.hung_incident(dict(records)) or {}
    return (records.get("workflows") or {}).get(
        str(incident.get("workflow_request_id"))
    ) or {}


def triage_rows(records: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Every rank row ``COLLECT_HUNG_TRIAGE`` returned, with its node, and the
    results that did not decode (each node's result is the summary codec's
    envelope; an undecodable one is named, never read as an empty node)."""

    rows: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for execution in hung_workflow(records).get("step_executions") or []:
        if execution.get("operation") != "COLLECT_HUNG_TRIAGE":
            continue
        results = (execution.get("details") or {}).get("node_results") or {}
        for node, value in sorted(results.items()):
            try:
                document = decode_summary(value)
            except ValueError as exc:
                unreadable.append(
                    f"COLLECT_HUNG_TRIAGE result of {node} is unreadable: {exc}"
                )
                continue
            rows.extend(
                {"node_id": node, **item}
                for item in document.get("ranks") or []
                if isinstance(item, dict)
            )
    return rows, unreadable


def triage_decision(records: Mapping[str, Any]) -> dict[str, Any] | None:
    """The classification the product bound the bundle with, else the last
    triage execution's."""

    workflow = hung_workflow(records)
    for step in workflow.get("official_steps") or []:
        if step.get("operation") == "COLLECT_DIAGNOSTIC_BUNDLE":
            decision = (step.get("parameters") or {}).get("hung_triage_decision")
            if isinstance(decision, dict):
                return decision
    for item in reversed(workflow.get("step_executions") or []):
        if item.get("operation") == "COLLECT_HUNG_TRIAGE":
            decision = (item.get("details") or {}).get("hung_triage_decision")
            if isinstance(decision, dict):
                return decision
    return None


def process_reports(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    branch = (decision.get("evidence_branches") or {}).get("process_observations") or {}
    return [item for item in branch.get("processes") or [] if isinstance(item, dict)]


def cohorts(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    branch = (decision.get("evidence_branches") or {}).get("process_observations") or {}
    return [item for item in branch.get("cohorts") or [] if isinstance(item, dict)]


def process_key(row: Mapping[str, Any]) -> tuple[str, int, int] | None:
    identity = row.get("process_identity") or {}
    node, pid, start = (
        row.get("node_id"),
        row.get("pid"),
        identity.get("start_time_ticks"),
    )
    if isinstance(node, str) and type(pid) is int and type(start) is int:
        return node, pid, start
    return None


def report_for(
    reports: Sequence[Mapping[str, Any]], row: Mapping[str, Any]
) -> dict[str, Any] | None:
    key = process_key(row)
    for report in reports:
        if key == (
            report.get("node_id"),
            report.get("pid"),
            report.get("start_time_ticks"),
        ):
            return dict(report)
    return None


def kernel_threads(row: Mapping[str, Any], index: int = 0) -> dict[str, dict[str, Any]]:
    samples = (row.get("process_series") or {}).get("samples") or []
    if index >= len(samples):
        return {}
    return {
        str(tid): dict(value)
        for tid, value in (samples[index].get("threads") or {}).items()
        if isinstance(value, dict)
    }


def host_tids(row: Mapping[str, Any], native_tid: str) -> list[str]:
    """The host thread ids whose innermost namespace id is ``native_tid``."""

    return sorted(
        tid
        for tid, value in kernel_threads(row).items()
        if str((value.get("namespace_ids") or [tid])[-1]) == native_tid
    )


def namespaced(row: Mapping[str, Any]) -> bool:
    """Whether any thread was seen through a pid namespace boundary."""

    return any(
        len(value.get("namespace_ids") or []) >= 2
        for value in kernel_threads(row).values()
    )


def windows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in (row.get("process_series") or {}).get("windows") or []
        if isinstance(item, dict)
    ]


def stack_samples(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in (row.get("stack_series") or {}).get("samples") or []
        if isinstance(item, dict)
    ]


def collected_stacks(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in stack_samples(row)
        if item.get("status") == "COLLECTED" and item.get("returncode") == 0
    ]


def node_ranks(nodes: Sequence[str]) -> dict[str, list[int]]:
    """Each attempt node's ranks: Master (``--fault-node``) first."""

    return {
        node: list(
            range(index * plan.PROCS_PER_NODE, (index + 1) * plan.PROCS_PER_NODE)
        )
        for index, node in enumerate(nodes)
    }


def ranked_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """``{(node, rank): row}`` of the rows that carry a rank."""

    return {
        (str(row["node_id"]), int(row["rank"])): dict(row)
        for row in rows
        if type(row.get("rank")) is int
    }


def _label(row: Mapping[str, Any]) -> str:
    return f"{row.get('node_id')} rank {row.get('rank')} pid {row.get('pid')}"


# --------------------------------------------------------------------------- #
# graders (pure)
# --------------------------------------------------------------------------- #
def coverage_errors(
    rows: Sequence[Mapping[str, Any]], nodes: Sequence[str]
) -> list[str]:
    """Every rank of a sampled node was sampled (eight, under the 16 cap)."""

    sampled = {str(row.get("node_id")) for row in rows}
    if not sampled:
        return ["COLLECT_HUNG_TRIAGE sampled no process on either node"]
    by_node = ranked_rows(rows)
    errors = []
    for node, ranks in node_ranks(nodes).items():
        if node not in sampled:
            continue
        missing = [rank for rank in ranks if (node, rank) not in by_node]
        if missing:
            errors.append(f"{node}: ranks {missing} were not sampled")
    return errors


def premise_errors(
    rows: Sequence[Mapping[str, Any]],
    decls: Mapping[int, Declaration],
    account: Mapping[str, Sequence[str]],
) -> list[str]:
    """Why the fixture, not the product, leaves nothing to judge."""

    errors = fixture_errors(account)
    missing = sorted(rank for (_node, rank) in ranked_rows(rows) if rank not in decls)
    if missing:
        errors.append(f"no ROLES line for sampled ranks {missing}")
    return errors


def _membership(members: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(members), separators=(",", ":")).encode()
    ).hexdigest()


def context_errors(
    rows: Sequence[Mapping[str, Any]],
    decls: Mapping[int, Declaration],
    *,
    mode: str,
    job_id: str,
    attempt_id: str,
) -> list[str]:
    """The triage read each rank's declaration back, verified, and nothing
    for an undeclared rank."""

    errors = []
    for (node, rank), row in sorted(ranked_rows(rows).items()):
        decl = decls.get(rank)
        if decl is None:
            continue
        context = row.get("comparison_context") or {}
        expected = plan.expected_context(mode, rank, world_size=WORLD_SIZE)
        if not decl.declared or expected is None:
            if context.get("pipeline_stage") or context.get("verified"):
                errors.append(
                    f"{_label(row)}: undeclared rank read back a context {context}"
                )
            continue
        if context.get("verified") is not True:
            errors.append(f"{_label(row)}: comparison_context is not verified")
            continue
        wanted = {
            **{key: value for key, value in expected.items() if key != "members"},
            "member_count": len(expected["members"]),
            "membership_sha256": _membership(expected["members"]),
            "membership_verified": True,
            "declared_rank": rank,
            "job_id": job_id,
            "attempt_id": attempt_id,
            "thread_roles": {tid: role for tid, (role, _name) in decl.roles.items()},
        }
        wrong = {
            key: context.get(key)
            for key, value in wanted.items()
            if context.get(key) != value
        }
        if wrong:
            errors.append(
                f"{_label(row)}: context differs from the declaration: {wrong}"
            )
        if decl.group != expected["group_name"] or tuple(decl.members) != tuple(
            expected["members"]
        ):
            errors.append(
                f"{_label(row)}: the trainer declared {decl.group} {decl.members}, "
                f"the plan says {expected['group_name']} {expected['members']}"
            )
    return errors


def host_thread_errors(
    rows: Sequence[Mapping[str, Any]], decls: Mapping[int, Declaration]
) -> list[str]:
    """Container-native thread ids map to one host thread each through NSpid,
    the training thread to the process's own host pid, and the product's
    ``host_thread_roles`` says the same."""

    errors = []
    for (_node, rank), row in sorted(ranked_rows(rows).items()):
        decl = decls.get(rank)
        if decl is None or not decl.declared:
            continue
        context = row.get("comparison_context") or {}
        host_roles = {
            str(k): v for k, v in (context.get("host_thread_roles") or {}).items()
        }
        mapped: dict[str, str] = {}
        for native, (role, name) in sorted(decl.roles.items()):
            hosts = host_tids(row, native)
            if len(hosts) != 1:
                errors.append(
                    f"{_label(row)}: native tid {native} ({name}) maps to host tids {hosts}"
                )
                continue
            mapped[hosts[0]] = role
            if host_roles.get(hosts[0]) != role:
                errors.append(
                    f"{_label(row)}: host tid {hosts[0]} ({name}) reads "
                    f"{host_roles.get(hosts[0])}, declared {role}"
                )
            if role == "TRAINING" and hosts[0] != str(row.get("pid")):
                errors.append(
                    f"{_label(row)}: the training thread is host tid {hosts[0]}, not the pid"
                )
        extra = sorted(set(host_roles) - set(mapped))
        if extra:
            errors.append(
                f"{_label(row)}: host_thread_roles names undeclared tids {extra}"
            )
    return errors


def stack_identity_errors(
    rows: Sequence[Mapping[str, Any]], decls: Mapping[int, Declaration]
) -> list[str]:
    """In every collected stack the training thread is the one declared, bound
    to the host pid, and only declared threads carry application roles."""

    errors = []
    for (_node, rank), row in sorted(ranked_rows(rows).items()):
        decl = decls.get(rank)
        if decl is None or not decl.declared or decl.training_tid is None:
            continue
        collected = collected_stacks(row)
        if not collected:
            errors.append(f"{_label(row)}: no collected stack sample")
            continue
        for index, sample in enumerate(collected):
            threads = [
                item for item in sample.get("threads") or [] if isinstance(item, dict)
            ]
            training = [item for item in threads if item.get("role") == "TRAINING"]
            if len(training) != 1:
                errors.append(
                    f"{_label(row)}: stack {index} has {len(training)} training threads"
                )
            else:
                thread = training[0]
                if (
                    str(thread.get("os_thread_id")) != decl.training_tid
                    or thread.get("host_thread_id") != row.get("pid")
                    or thread.get("role_source") != "APPLICATION"
                ):
                    errors.append(
                        f"{_label(row)}: stack {index} training thread is "
                        f"native {thread.get('os_thread_id')} host {thread.get('host_thread_id')} "
                        f"from {thread.get('role_source')}"
                    )
            for thread in threads:
                native = str(thread.get("os_thread_id"))
                declared = decl.roles.get(native)
                if thread.get("role_source") == "APPLICATION" and (
                    declared is None or declared[0] != thread.get("role")
                ):
                    errors.append(
                        f"{_label(row)}: stack {index} thread {native} reads "
                        f"{thread.get('role')} from the application, declared {declared}"
                    )
                if (
                    thread.get("role_source") != "APPLICATION"
                    and thread.get("role") in plan.THREAD_ROLES
                ):
                    errors.append(
                        f"{_label(row)}: stack {index} thread {native} got role "
                        f"{thread.get('role')} without a declaration"
                    )
    return errors


def stability_errors(
    rows: Sequence[Mapping[str, Any]],
    decls: Mapping[int, Declaration],
    reports: Sequence[Mapping[str, Any]],
    *,
    blocked_rank: int = 0,
) -> list[str]:
    """Stable threads keep one host id and start time across the samples; the
    blocked rank's training thread is among them."""

    errors = []
    for (_node, rank), row in sorted(ranked_rows(rows).items()):
        decl = decls.get(rank)
        report = report_for(reports, row)
        if decl is None or not decl.declared or report is None:
            continue
        stable = [
            item
            for item in (report.get("stack") or {}).get("stable_threads") or []
            if isinstance(item, dict)
        ]
        for thread in stable:
            host = str(thread.get("host_thread_id"))
            starts = {
                kernel_threads(row, index).get(host, {}).get("start_ticks")
                for index in range(STACK_SAMPLES)
            }
            if starts != {thread.get("thread_start_ticks")}:
                errors.append(
                    f"{_label(row)}: stable thread {host} start {thread.get('thread_start_ticks')}"
                    f" against sampled starts {sorted(map(str, starts))}"
                )
            native = str(
                (kernel_threads(row).get(host, {}).get("namespace_ids") or [host])[-1]
            )
            declared = decl.roles.get(native)
            if declared is not None and thread.get("role") != declared[0]:
                errors.append(
                    f"{_label(row)}: stable thread {host} reads {thread.get('role')}, "
                    f"declared {declared[0]}"
                )
        if rank == blocked_rank and not any(
            item.get("role") == "TRAINING"
            and str(item.get("host_thread_id")) == str(row.get("pid"))
            for item in stable
        ):
            errors.append(
                f"{_label(row)}: the blocked rank's training thread is not a stable thread"
            )
    return errors


def _cores_errors(row: Mapping[str, Any], decl: Declaration) -> list[str]:
    """CPU cores are tick deltas over the measured interval at SC_CLK_TCK."""

    errors = []
    items = windows(row)
    if len(items) != PROCESS_WINDOWS:
        return [
            f"{_label(row)}: {len(items)} process windows, expected {PROCESS_WINDOWS}"
        ]
    for index, window in enumerate(items):
        if window.get("status") != "COLLECTED":
            errors.append(f"{_label(row)}: window {index} is {window.get('status')}")
            continue
        elapsed, ticks = window.get("elapsed_seconds"), window.get("cpu_ticks_delta")
        clock = window.get("clock_ticks_per_second")
        cores = window.get("cpu_cores")
        if clock != decl.clk_tck:
            errors.append(
                f"{_label(row)}: window {index} uses {clock} ticks/s, the node has {decl.clk_tck}"
            )
        if not isinstance(elapsed, (int, float)) or not 0.5 <= elapsed <= 30:
            errors.append(
                f"{_label(row)}: window {index} elapsed {elapsed} s is not a measured interval"
            )
            continue
        if type(ticks) is not int or not isinstance(cores, (int, float)) or not clock:
            errors.append(
                f"{_label(row)}: window {index} has no CPU delta ({ticks}, {cores})"
            )
            continue
        if abs(cores - ticks / clock / elapsed) > 1e-6:
            errors.append(
                f"{_label(row)}: window {index} cpu_cores {cores} != {ticks}/{clock}/{elapsed}"
            )
    return errors


def wchan_errors(
    rows: Sequence[Mapping[str, Any]],
    decls: Mapping[int, Declaration],
    reports: Sequence[Mapping[str, Any]],
) -> list[str]:
    """A spinning thread's wait channel 0 is no wait location; the sleeping
    thread's real channel is; CPU is normalized by SC_CLK_TCK and elapsed."""

    errors = []
    for (_node, rank), row in sorted(ranked_rows(rows).items()):
        decl = decls.get(rank)
        if decl is None or not decl.declared:
            continue
        spinner, sleeper = (
            decl.tids("COMMUNICATION", "gfrs-spinner"),
            decl.tids("BACKGROUND", "gfrs-sleeper"),
        )
        if len(spinner) != 1 or len(sleeper) != 1:
            errors.append(
                f"{_label(row)}: declared {len(spinner)} spinner(s), {len(sleeper)} sleeper(s)"
            )
            continue
        spin_hosts, sleep_hosts = host_tids(row, spinner[0]), host_tids(row, sleeper[0])
        if len(spin_hosts) != 1 or len(sleep_hosts) != 1:
            errors.append(
                f"{_label(row)}: spinner maps to {spin_hosts}, sleeper to {sleep_hosts}"
            )
            continue
        spin, sleep = spin_hosts[0], sleep_hosts[0]
        states, channels = set(), set()
        for index in range(STACK_SAMPLES):
            threads = kernel_threads(row, index)
            if spin in threads and threads[spin].get("wchan") is not None:
                errors.append(
                    f"{_label(row)}: sample {index} reads wchan {threads[spin]['wchan']!r} for the spinner"
                )
            if spin in threads:
                states.add(threads[spin].get("state"))
            if sleep in threads:
                channels.add(threads[sleep].get("wchan"))
        if "R" not in states:
            errors.append(
                f"{_label(row)}: the spinner was never sampled running (states {sorted(map(str, states))})"
            )
        if len(channels) != 1 or not next(iter(channels)):
            errors.append(
                f"{_label(row)}: the sleeper's wait channel reads {sorted(map(str, channels))}"
            )
        for index, window in enumerate(windows(row)):
            waits = {
                str(item.get("tid")): item for item in window.get("wait_threads") or []
            }
            if spin in waits:
                errors.append(
                    f"{_label(row)}: window {index} lists the spinner as a wait thread"
                )
            if sleep not in waits or not waits[sleep].get("same_wait"):
                errors.append(
                    f"{_label(row)}: window {index} misses the sleeper's repeated wait"
                )
            cores = window.get("cpu_cores")
            if isinstance(cores, (int, float)) and cores < SPINNER_MIN_CORES:
                errors.append(
                    f"{_label(row)}: window {index} reads {cores:.2f} cores with a spinning thread"
                )
        errors.extend(_cores_errors(row, decl))
        report = report_for(reports, row)
        if report is None:
            errors.append(f"{_label(row)}: no process report in the decision")
            continue
        waits_seen = [
            item
            for item in (report.get("behavior") or {}).get("observations") or []
            if item.get("code") == THREAD_WAIT
        ]
        if any(str(item.get("thread_id")) == spin for item in waits_seen):
            errors.append(
                f"{_label(row)}: the decision reports a repeated wait for the spinner"
            )
        sleeper_waits = [
            item for item in waits_seen if str(item.get("thread_id")) == sleep
        ]
        if not sleeper_waits:
            errors.append(
                f"{_label(row)}: the decision has no repeated wait for the sleeper"
            )
        elif any(
            item.get("role") != "BACKGROUND" or item.get("candidate")
            for item in sleeper_waits
        ):
            errors.append(
                f"{_label(row)}: the sleeper's wait is {sleeper_waits[0]}, not a BACKGROUND non-candidate"
            )
    return errors


def cpu_errors(
    rows: Sequence[Mapping[str, Any]],
    decls: Mapping[int, Declaration],
    reports: Sequence[Mapping[str, Any]],
    sharers: Mapping[int, Mapping[str, str]],
) -> list[str]:
    """Cores over the actual sample interval; a shared GPU is never claimed."""

    errors = []
    by_node_rank = ranked_rows(rows)
    sampled_nodes = {str(row.get("node_id")) for row in rows}
    for (node, rank), row in sorted(by_node_rank.items()):
        decl = decls.get(rank)
        if decl is None or not decl.declared:
            continue
        errors.extend(_cores_errors(row, decl))
        samples = (row.get("process_series") or {}).get("samples") or []
        stamps = [parse_time(item.get("observed_at")) for item in samples]
        for index, window in enumerate(windows(row)):
            if index + 1 < len(stamps) and None not in stamps[index : index + 2]:
                gap = (stamps[index + 1] - stamps[index]).total_seconds()  # type: ignore[operator]
                elapsed = window.get("elapsed_seconds")
                if (
                    isinstance(elapsed, (int, float))
                    and abs(elapsed - gap) > INTERVAL_TOLERANCE_SECONDS
                ):
                    errors.append(
                        f"{_label(row)}: window {index} elapsed {elapsed} s against samples {gap:.2f} s apart"
                    )
        gpu = row.get("gpu_series") or {}
        report = report_for(reports, row) or {}
        behavior = report.get("behavior") or {}
        if (
            gpu.get("scope") != DEVICE_NOT_PID
            or behavior.get("gpu_scope") != DEVICE_NOT_PID
        ):
            errors.append(
                f"{_label(row)}: GPU utilization is not scoped {DEVICE_NOT_PID}"
            )
        if gpu.get("mapping_verified") is not True:
            errors.append(f"{_label(row)}: the PID's GPU mapping is not verified")
        shared = gpu.get("shared_with_other_processes")
        if decl.share_gpu and shared is not True:
            errors.append(f"{_label(row)}: a shared GPU reads shared={shared}")
        if not decl.share_gpu and shared is not False:
            errors.append(f"{_label(row)}: an exclusive GPU reads shared={shared}")
        comparison = report.get("peer_comparison")
        if decl.share_gpu and comparison != NO_COHORT:
            errors.append(
                f"{_label(row)}: a rank on a shared GPU joined a cohort ({comparison})"
            )
        if not decl.share_gpu and comparison != ROLE_COHORT:
            errors.append(
                f"{_label(row)}: an exclusive declared rank is {comparison}, not {ROLE_COHORT}"
            )
        cores = behavior.get("cpu_cores")
        valid = [
            w
            for w in windows(row)
            if w.get("status") == "COLLECTED"
            and isinstance(w.get("cpu_cores"), (int, float))
        ]
        if valid and isinstance(cores, (int, float)):
            seconds = sum(w["elapsed_seconds"] for w in valid)
            mean = sum(w["cpu_cores"] * w["elapsed_seconds"] for w in valid) / seconds
            if abs(cores - mean) > 1e-6 or behavior.get("cpu_valid_windows") != len(
                valid
            ):
                errors.append(
                    f"{_label(row)}: the decision's {cores} cores is not the windows' weighted mean {mean:.4f}"
                )
        elif report:
            errors.append(f"{_label(row)}: the decision has no CPU cores")
    for node in sorted(sampled_nodes):
        ranks_here = {rank for (other, rank) in by_node_rank if other == node}
        started = sorted(rank for rank in sharers if rank in ranks_here)
        pids = {str(sharers[rank].get("pid")) for rank in started}
        sharer_rows = [
            row
            for row in rows
            if str(row.get("node_id")) == node
            and row.get("rank") is None
            and str(
                (
                    kernel_threads(row)
                    .get(str(row.get("pid")), {})
                    .get("namespace_ids")
                    or [row.get("pid")]
                )[-1]
            )
            in pids
        ]
        if len(sharer_rows) != len(started):
            errors.append(
                f"{node}: {len(sharer_rows)} sharer process(es) sampled, {len(started)} started"
            )
        for row in sharer_rows:
            if (row.get("gpu_series") or {}).get(
                "shared_with_other_processes"
            ) is not True:
                errors.append(f"{_label(row)}: the sharer's GPU does not read shared")
    return errors


def pyspy_errors(
    rows: Sequence[Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
    holders: Mapping[int, Mapping[str, str]],
) -> list[str]:
    """py-spy fails on the non-Python holder: a failed sample, no signature,
    its /proc evidence kept; the Python ranks' stacks parse as thread JSON."""

    errors = []
    by_node_rank = ranked_rows(rows)
    sampled_nodes = sorted({str(row.get("node_id")) for row in rows})
    for node in sampled_nodes:
        started = [
            rank
            for rank in holders
            if any(r == rank for (n, r) in by_node_rank if n == node)
        ]
        found = [
            row
            for row in rows
            if str(row.get("node_id")) == node
            and row.get("rank") is None
            and HOLDER_NAME in str(row.get("process_name") or "")
        ]
        if len(found) != len(started):
            errors.append(
                f"{node}: {len(found)} holder process(es) sampled, {len(started)} started"
            )
        for row in found:
            if (row.get("process_identity") or {}).get("verified") is not True:
                errors.append(f"{_label(row)}: the holder's identity is not verified")
            samples = stack_samples(row)
            if len(samples) != STACK_SAMPLES:
                errors.append(
                    f"{_label(row)}: {len(samples)} stack samples of the holder"
                )
            for index, sample in enumerate(samples):
                if (
                    sample.get("status") != "FAILED"
                    or sample.get("returncode") in (0, None)
                    or not sample.get("error")
                    or sample.get("signature")
                ):
                    errors.append(f"{_label(row)}: holder stack {index} is {sample}")
            if (row.get("python_stack") or {}).get("status") != "FAILED":
                errors.append(f"{_label(row)}: the holder's python_stack is not FAILED")
            if (
                any(w.get("status") != "COLLECTED" for w in windows(row))
                or len(windows(row)) != PROCESS_WINDOWS
            ):
                errors.append(
                    f"{_label(row)}: the holder's /proc windows were not kept"
                )
            report = report_for(reports, row)
            if report is None:
                errors.append(
                    f"{_label(row)}: the holder has no process report in the decision"
                )
                continue
            stack = report.get("stack") or {}
            if (
                stack.get("valid_samples") != 0
                or stack.get("missing_samples") != STACK_SAMPLES
            ):
                errors.append(
                    f"{_label(row)}: the decision counts {stack.get('valid_samples')} valid holder stacks"
                )
            if not stack.get("errors") or stack.get("stable_threads"):
                errors.append(
                    f"{_label(row)}: the decision kept no py-spy error or invented stable threads"
                )
            if (report.get("behavior") or {}).get("valid_windows") != PROCESS_WINDOWS:
                errors.append(
                    f"{_label(row)}: the decision dropped the holder's process windows"
                )
            if any(
                item.get("code") in STACK_CODES
                for item in report.get("observations") or []
            ):
                errors.append(
                    f"{_label(row)}: a stack observation on a process py-spy could not sample"
                )
    for (_node, _rank), row in sorted(by_node_rank.items()):
        collected = collected_stacks(row)
        if not collected:
            errors.append(f"{_label(row)}: no real py-spy capture of a Python rank")
            continue
        for thread in collected[-1].get("threads") or []:
            key = str(thread.get("thread_key") or "")
            frames = thread.get("frames")
            if (
                key.count("/") != 1
                or not all(part.isdecimal() for part in key.split("/"))
                or type(thread.get("os_thread_id")) is not int
                or not isinstance(frames, list)
                or any(
                    not isinstance(frame, dict)
                    or not isinstance(frame.get("function"), str)
                    or not isinstance(frame.get("file"), str)
                    or type(frame.get("line")) is not int
                    for frame in frames
                )
                or (frames and not isinstance(thread.get("signature"), str))
            ):
                errors.append(f"{_label(row)}: py-spy thread JSON structure {thread}")
    return errors


def cohort_errors(
    rows: Sequence[Mapping[str, Any]],
    decls: Mapping[int, Declaration],
    reports: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Cohorts hold one declared role each, undeclared ranks none, every group
    of three or more sampled peers compared, and no cross-role lead."""

    errors = []
    by_key: dict[tuple[str, int, int], tuple[dict[str, Any], Declaration | None]] = {}
    for row in rows:
        key = process_key(row)
        if key is not None:
            by_key[key] = (
                dict(row),
                decls.get(row["rank"]) if type(row.get("rank")) is int else None,
            )
    for cohort in groups:
        definition = cohort.get("definition") or {}
        members = [
            tuple(item)
            for item in cohort.get("processes") or []
            if isinstance(item, list) and len(item) == 3
        ]
        declared = []
        for key in members:
            row, decl = by_key.get(key, ({}, None))
            if decl is None or not decl.declared:
                errors.append(
                    f"cohort {str(cohort.get('cohort'))[:12]} holds an undeclared process {key}"
                )
                continue
            declared.append((decl.group, decl.stage, decl.expert))
            if (decl.stage, decl.expert, decl.group) != (
                definition.get("pipeline_stage"),
                definition.get("expert_role"),
                definition.get("group_name"),
            ):
                errors.append(
                    f"cohort {str(cohort.get('cohort'))[:12]} ({definition.get('pipeline_stage')}/"
                    f"{definition.get('expert_role')}) holds {_label(row)} declaring "
                    f"{decl.stage}/{decl.expert}"
                )
        if len(set(declared)) > 1:
            errors.append(
                f"cohort {str(cohort.get('cohort'))[:12]} mixes roles {sorted(set(declared))}"
            )
    for (node, rank), row in sorted(ranked_rows(rows).items()):
        decl = decls.get(rank)
        report = report_for(reports, row)
        if decl is None or report is None:
            continue
        comparison = report.get("peer_comparison")
        peers = sum(
            1
            for (other_node, other_rank), _ in ranked_rows(rows).items()
            if other_node == node
            and decls.get(other_rank) is not None
            and decls[other_rank].group == decl.group
        )
        if not decl.declared:
            wanted = NO_COHORT
        elif peers >= MIN_COHORT_PEERS:
            wanted = ROLE_COHORT
        else:
            wanted = FEW_PEERS
        if comparison != wanted:
            errors.append(
                f"{_label(row)}: peer comparison {comparison}, expected {wanted}"
            )
        for observation in report.get("observations") or []:
            code = str(observation.get("code") or "")
            if not code.startswith("ROLE_COHORT_"):
                continue
            cohort = next(
                (c for c in groups if c.get("cohort") == observation.get("cohort")),
                None,
            )
            if cohort is None or list(process_key(row) or ()) not in [
                list(k) for k in cohort.get("processes") or []
            ]:
                errors.append(
                    f"{_label(row)}: {code} cites a cohort the process is not in"
                )
            if (
                not decl.declared
                or (cohort or {}).get("definition", {}).get("pipeline_stage")
                != decl.stage
            ):
                errors.append(f"{_label(row)}: {code} attributed across roles")
            if code == "ROLE_COHORT_CPU_DIFFERENCE":
                errors.append(f"{_label(row)}: a uniform cohort produced {code}")
    return errors


def cohort_notes(
    reports: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]
) -> list[str]:
    compared = [c for c in groups if (c.get("process_count") or 0) >= MIN_COHORT_PEERS]
    leads = sorted(
        {
            str(item.get("code"))
            for report in reports
            for item in report.get("observations") or []
            if str(item.get("code") or "").startswith("ROLE_COHORT_")
        }
    )
    return [
        f"{len(compared)} cohort(s) of >= {MIN_COHORT_PEERS} peers compared on the sampled nodes",
        f"role-cohort leads: {leads or 'none'}",
    ]


# --------------------------------------------------------------------------- #
# attempt_change (pure)
# --------------------------------------------------------------------------- #
def _attempt_incidents(
    records: Mapping[str, Any], attempt_id: str, suffix: str
) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in records.get("incidents") or []
        if item.get("attempt_id") == attempt_id
        and str(item.get("event_id", "")).endswith(suffix)
    ]


def warning_nodes(records: Mapping[str, Any], attempt_id: str) -> set[str]:
    return {
        str(node)
        for item in _attempt_incidents(records, attempt_id, nccl.WARNING_SUFFIX)
        for node in item.get("node_ids") or []
    }


def warned_errors(
    records: Mapping[str, Any], attempt_id: str, started: datetime, nodes: Sequence[str]
) -> list[str]:
    """Attempt A got ZERO_WARNING on every node, not before the threshold."""

    errors = []
    warnings = _attempt_incidents(records, attempt_id, nccl.WARNING_SUFFIX)
    for node in nodes:
        times = [
            moment
            for item in warnings
            if item.get("node_ids") == [node]
            and (moment := parse_time(item.get("created_at"))) is not None
        ]
        if not times:
            errors.append(f"{node}: no ZERO_WARNING for attempt {attempt_id}")
            continue
        offset = (min(times) - started).total_seconds()
        if offset < nccl.ZERO_WARNING_SECONDS - nccl.CLOCK_TOLERANCE_SECONDS:
            errors.append(
                f"{node}: ZERO_WARNING {offset:.0f} s after the stall, before the threshold"
            )
    return errors


def no_hung_errors(records: Mapping[str, Any], attempts: Sequence[str]) -> list[str]:
    return [
        f"attempt {attempt}: HUNG_SUSPECTED {item.get('event_id')} on a zero run under the threshold"
        for attempt in attempts
        for item in _attempt_incidents(records, attempt, nccl.HUNG_SUFFIX)
    ]


def fresh_attempt_errors(
    records: Mapping[str, Any],
    attempt_b: str,
    switched_at: datetime,
    nodes: Sequence[str],
) -> list[str]:
    """Attempt B never warned or suspected, and each node's EFA state is B's
    own: first observed after the switch, active, with no zero streak."""

    errors = [
        f"attempt B raised {item.get('event_id')}"
        for item in records.get("incidents") or []
        if item.get("attempt_id") == attempt_b
        and str(item.get("event_id", "")).endswith(
            (nccl.WARNING_SUFFIX, nccl.HUNG_SUFFIX)
        )
    ]
    states = records.get("efa_states") or {}
    slack = timedelta(seconds=SWITCH_TOLERANCE_SECONDS)
    for node in nodes:
        state = states.get(node)
        if not isinstance(state, dict):
            errors.append(f"{node}: no EFA traffic state row")
            continue
        if state.get("attempt_id") != attempt_b:
            errors.append(
                f"{node}: the EFA state names attempt {state.get('attempt_id')}, not B"
            )
            continue
        first = parse_time(state.get("first_observed_at"))
        if first is None or first < switched_at - slack:
            errors.append(
                f"{node}: B's state was first observed {first}, before the switch {switched_at}"
            )
        if state.get("had_active_traffic") is not True:
            errors.append(f"{node}: B's state never saw active traffic")
        zero_since = parse_time(state.get("zero_since"))
        if state.get("consecutive_zero_samples") or (
            zero_since is not None and zero_since < switched_at - slack
        ):
            errors.append(
                f"{node}: B carries a zero streak ({state.get('consecutive_zero_samples')} samples since {zero_since})"
            )
        if state.get("signal") in {"ZERO_WARNING", "HUNG_SUSPECTED"}:
            errors.append(f"{node}: B's state reads {state.get('signal')}")
    return errors


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
class SecondAttemptNcclJob(ManagedNcclJob):
    """The same job's attempt 2, submitted by the runner after attempt 1 was
    deleted (``gpu-training-submit --attempt-id <job>-a002``)."""

    attempt_number = 2


def _job(
    ctx: VariantContext,
    suffix: str,
    *,
    stall_mode: str,
    job_class: Callable[..., ManagedNcclJob] = ManagedNcclJob,
) -> ManagedNcclJob:
    name = run_identity(CASE, suffix, ctx.attempt, ctx.case_dir).replace(
        "gl011", "gl11"
    )
    job = job_class(ctx, name=name, stall_mode=stall_mode)
    ctx.cleanup.push(f"job-{job.attempt_id}", job.delete)
    job.start()
    ctx.evidence.write(f"{suffix}-{job.attempt_id}-job", job.identity())
    return job


def _role_chain(
    ctx: VariantContext, suffix: str, mode: str
) -> tuple[ManagedNcclJob, dict[str, Any], dict[str, list[str]], dict[str, list[str]]]:
    """Start, prove active, stall, read the settled records and the trainer's
    account while the stall still holds, release."""

    run_survey(ctx)
    job = _job(ctx, suffix, stall_mode=mode)
    nccl.prove_baseline(ctx, name=f"{suffix}-active")
    started, _events = nccl.stall_job(ctx, job, name=suffix)
    nodes = nccl.attempt_nodes(ctx)
    records = nccl.hung_records(ctx, job, started)
    account = read_account(job)
    ctx.evidence.write(f"{suffix}-account", account)
    stages = {
        **nccl.episode_stages(records, started, nodes),
        "resumed": nccl.release_and_resume(ctx, job, name=suffix),
    }
    return job, records, stages, account


def _triage(
    ctx: VariantContext,
    job: ManagedNcclJob,
    records: dict[str, Any],
    account: Mapping[str, Sequence[str]],
    stages: dict[str, list[str]],
) -> tuple[
    list[dict[str, Any]], dict[int, Declaration], list[dict[str, Any]], dict[str, Any]
]:
    """The triage rows, declarations, reports and decision, or INCONCLUSIVE
    on a fixture premise; an unreadable or absent triage is a chain FAIL."""

    rows, unreadable = triage_rows(records)
    decls = declarations(account)
    premises = premise_errors(rows, decls, account)
    if premises:
        raise VariantInconclusive(
            "the role fixture did not hold: " + "; ".join(premises)
        )
    decision = triage_decision(records) or {}
    stages["triage"] = [
        *unreadable,
        *coverage_errors(rows, nccl.attempt_nodes(ctx)),
        *(
            []
            if decision
            else ["no hung_triage_decision in the bundle step or a triage execution"]
        ),
    ]
    ctx.evidence.write(
        "triage-view",
        {
            "rows": rows,
            "decision": decision,
            "declarations": {k: v.__dict__ for k, v in decls.items()},
        },
    )
    return rows, decls, process_reports(decision), decision


def _result(
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes or [],
        **(extra or {}),
    }


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def role_mismatch(ctx: VariantContext) -> dict[str, Any]:
    job, records, stages, account = _role_chain(ctx, "role", plan.ROLES)
    rows, decls, reports, decision = _triage(ctx, job, records, account, stages)
    groups = cohorts(decision)
    stages["context"] = context_errors(
        rows, decls, mode=plan.ROLES, job_id=job.name, attempt_id=job.attempt_id
    )
    stages["cohorts"] = cohort_errors(rows, decls, reports, groups)
    return _result(
        stages,
        notes=[
            "pp0/expert ranks burn CPU while stalled, pp1/dense and the undeclared"
            " rank stay quiet: a cross-role comparison would flag the quiet ranks",
            *cohort_notes(reports, groups),
        ],
        extra={"attempt": job.identity()},
    )


def tid_namespace(ctx: VariantContext) -> dict[str, Any]:
    job, records, stages, account = _role_chain(ctx, "tid", plan.THREADS)
    rows, decls, reports, _decision = _triage(ctx, job, records, account, stages)
    if rows and not any(namespaced(row) for row in rows):
        raise VariantInconclusive(
            "the job's container shares the host pid namespace: no NSpid boundary to reconcile"
        )
    stages["context"] = context_errors(
        rows, decls, mode=plan.THREADS, job_id=job.name, attempt_id=job.attempt_id
    )
    stages["host_threads"] = host_thread_errors(rows, decls)
    stages["stacks"] = stack_identity_errors(rows, decls)
    stages["stability"] = stability_errors(rows, decls, reports)
    return _result(
        stages,
        notes=[
            "odd ranks start their helpers in reverse order and a churn thread keeps"
            " the thread list changing; the training thread keeps its role, host id"
            " and start time"
        ],
        extra={"attempt": job.identity()},
    )


def wchan_zero(ctx: VariantContext) -> dict[str, Any]:
    job, records, stages, account = _role_chain(ctx, "wchan", plan.WCHAN)
    rows, decls, reports, _decision = _triage(ctx, job, records, account, stages)
    stages["context"] = context_errors(
        rows, decls, mode=plan.WCHAN, job_id=job.name, attempt_id=job.attempt_id
    )
    stages["wchan"] = wchan_errors(rows, decls, reports)
    return _result(
        stages,
        notes=["a spinning helper (wchan 0) is never a wait; the sleeping helper is"],
        extra={"attempt": job.identity()},
    )


def cpu_interval(ctx: VariantContext) -> dict[str, Any]:
    job, records, stages, account = _role_chain(ctx, "cpu", plan.CPU)
    rows, decls, reports, _decision = _triage(ctx, job, records, account, stages)
    sharers = children(account, plan.SHARER_MARKER)
    expected = [
        rank
        for rank in range(WORLD_SIZE)
        if rank % plan.PROCS_PER_NODE in plan.sharing_locals(plan.CPU)
    ]
    if sorted(sharers) != expected:
        raise VariantInconclusive(
            f"GPU sharer processes started on ranks {sorted(sharers)}, planned {expected}"
        )
    stages["context"] = context_errors(
        rows, decls, mode=plan.CPU, job_id=job.name, attempt_id=job.attempt_id
    )
    stages["cpu"] = cpu_errors(rows, decls, reports, sharers)
    return _result(
        stages,
        notes=["local ranks 6 and 7 share their GPU with an unranked CUDA process"],
        extra={"attempt": job.identity(), "sharers": sharers},
    )


def pyspy_nonzero(ctx: VariantContext) -> dict[str, Any]:
    job, records, stages, account = _role_chain(ctx, "pyspy", plan.PYSPY)
    rows, _decls, reports, _decision = _triage(ctx, job, records, account, stages)
    holders = children(account, plan.HOLDER_MARKER)
    expected = [
        rank
        for rank in range(WORLD_SIZE)
        if rank % plan.PROCS_PER_NODE in plan.holder_locals(plan.PYSPY)
    ]
    if sorted(holders) != expected:
        raise VariantInconclusive(
            f"non-Python CUDA holders started on ranks {sorted(holders)}, planned {expected}"
        )
    stages["pyspy"] = pyspy_errors(rows, reports, holders)
    return _result(
        stages,
        notes=[
            "py-spy legitimately fails on the non-Python CUDA holder (native, no"
            " shadowed tool): a failed stack sample, no signature, /proc evidence kept"
        ],
        extra={"attempt": job.identity(), "holders": holders},
    )


def attempt_change(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    nodes = nccl.attempt_nodes(ctx)
    job_a = _job(ctx, "ac", stall_mode="quiet")
    nccl.prove_baseline(ctx, name="ac-active")
    started = ctx.clock()
    ctx.cleanup.push("release", job_a.release)
    ctx.evidence.write("ac-stall", {"at": started.isoformat(), **job_a.stall()})
    _minimum, zero = nccl.thresholds(ctx)
    ctx.hold(nccl.STALL_SETTLE_SECONDS, name="ac-settle")
    window = nccl.efa_window(
        ctx, seconds=A_ZERO_SECONDS - nccl.STALL_SETTLE_SECONDS, name="ac-zero"
    )
    errors = nccl.drop_errors(
        window,
        zero_bps=zero,
        minimum_seconds=A_ZERO_SECONDS - 2 * nccl.STALL_SETTLE_SECONDS,
    )
    liveness, events = job_a.alive(), job_a.trainer_events()
    ctx.evidence.write("ac-job", {"liveness": liveness, "trainer": events})
    if not liveness.get("alive"):
        errors.append("attempt A did not stay alive through its stall")
    errors.extend(nccl.stall_event_errors(events, job_a.stall_mode))
    if errors:
        raise VariantInconclusive(
            "attempt A did not hold at zero EFA traffic: " + "; ".join(errors)
        )

    def warned() -> dict[str, Any] | None:
        records = _attempt_records(ctx, job_a.name, [job_a.attempt_id], started)
        return (
            records if set(nodes) <= warning_nodes(records, job_a.attempt_id) else None
        )

    budget = max(0.0, A_DELETE_BY_SECONDS - (ctx.clock() - started).total_seconds())
    records_a = ctx.wait_until(
        warned, timeout_seconds=budget, poll_seconds=20, name="ac-warned"
    )
    if records_a is None:
        records_a = _attempt_records(ctx, job_a.name, [job_a.attempt_id], started)
    ctx.evidence.write("ac-a-records", records_a)
    # The attempt change: A is deleted before its zero run can reach the hung
    # threshold; releasing it is no longer a cleanup.
    ctx.evidence.write("ac-delete-a", job_a.delete())
    ctx.cleanup.discard("release")
    ctx.cleanup.discard(f"job-{job_a.attempt_id}")
    switched_at = ctx.clock()
    job_b = _job(ctx, "ac", stall_mode="quiet", job_class=SecondAttemptNcclJob)
    nccl.prove_baseline(ctx, name="ac-b-active")
    records = _attempt_records(
        ctx, job_a.name, [job_a.attempt_id, job_b.attempt_id], started
    )
    ctx.evidence.write("ac-records", records)
    stages = {
        "a_warned": warned_errors(records_a, job_a.attempt_id, started, nodes),
        "no_hung": no_hung_errors(records, [job_a.attempt_id, job_b.attempt_id]),
        "fresh_attempt": fresh_attempt_errors(
            records, job_b.attempt_id, switched_at, nodes
        ),
    }
    return _result(
        stages,
        notes=[
            "attempt A warned and was deleted under the hung threshold; B starts at WARMUP"
        ],
        extra={
            "attempt_a": job_a.identity(),
            "attempt_b": job_b.identity(),
            "switched_at": switched_at.isoformat(),
        },
    )


def _attempt_records(
    ctx: VariantContext, job_id: str, attempts: Sequence[str], since: datetime
) -> Any:
    return ctx.regional.cpu_python(
        ATTEMPT_PROBE,
        ctx.settings.regional.cluster_id,
        ",".join(nccl.attempt_nodes(ctx)),
        job_id,
        ",".join(attempts),
        since.isoformat(),
    )


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def _spec(
    variant: str,
    injection: str,
    expected: str,
    execute: Callable[[VariantContext], dict[str, Any]],
    *,
    max_minutes: int = 65,
    cleanup: str = (
        "release the stall and prove the EFA pair resumes; delete the test Job "
        "(its helper threads and child processes end with it); confirm no node "
        "was isolated/stopped/reset"
    ),
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE,
        variant=variant,
        proof_mode=NATIVE,
        lanes=LANES,
        risk="M",
        flags="S",
        injection=injection,
        expected=expected,
        cleanup=cleanup,
        max_minutes=max_minutes,
        execute=execute,
    )


VARIANTS = [
    _spec(
        "role_mismatch",
        "the two-node stall on a mixed-parallelism job: per node, local ranks 0-3"
        " declare PP stage pp0 / EP expert (and burn CPU while stalled), 4-6 declare"
        " pp1 / dense (quiet), 7 declares no role",
        "triage reads every declaration back verified; cohorts hold one declared"
        " role each, the undeclared rank none; no ROLE_COHORT lead crosses roles;"
        " chain -> ESCALATED",
        role_mismatch,
    ),
    _spec(
        "tid_namespace",
        "every rank declares its training, communication and background threads"
        " by container-native tid, odd ranks in reverse start order, with a"
        " churning undeclared thread while stalled",
        "native tids reconcile to host tids through NSpid (training thread = host"
        " pid) in host_thread_roles, the stack samples and the stable threads'"
        " start ticks; chain -> ESCALATED",
        tid_namespace,
    ),
    _spec(
        "wchan_zero",
        "a declared COMMUNICATION helper spins (wchan 0, state R) and a declared"
        " BACKGROUND helper sleeps on a futex while the job is stalled",
        "wchan 0 is no wait location (no repeated wait for the spinner), the"
        " sleeper's channel is; cpu_cores = tick delta / SC_CLK_TCK / elapsed;"
        " chain -> ESCALATED",
        wchan_zero,
    ),
    _spec(
        "cpu_interval",
        "a declared helper burns CPU at a 50 % duty while stalled; local ranks 6"
        " and 7 share their GPU with an unranked CUDA process",
        "windows use the measured interval and tick deltas, the decision's cores"
        " are their weighted mean; a shared GPU reads shared, DEVICE_NOT_PID, and"
        " joins no cohort; chain -> ESCALATED",
        cpu_interval,
    ),
    _spec(
        "pyspy_nonzero",
        "local rank 7 of each node runs a compiled non-Python CUDA context holder"
        " in the training container; the Node Agent's own py-spy cannot sample it",
        "the holder's stack samples are FAILED with the exit status and error text,"
        " no signature, its /proc windows kept; the Python ranks' stacks parse as"
        " thread JSON; chain -> ESCALATED",
        pyspy_nonzero,
    ),
    _spec(
        "attempt_change",
        "attempt A of the two-node job is stalled until both nodes warn, held at"
        " zero under the hung threshold and deleted; the same job returns as"
        " attempt B and goes active",
        "A: ZERO_WARNING per node, never HUNG_SUSPECTED; B: no warning or hung"
        " suspicion, each node's EFA state names B, first observed after the"
        " switch, active, with no inherited zero streak",
        attempt_change,
        max_minutes=100,
        cleanup=(
            "delete attempt A (the attempt change itself) and attempt B; confirm "
            "no node was isolated/stopped/reset"
        ),
    ),
]
