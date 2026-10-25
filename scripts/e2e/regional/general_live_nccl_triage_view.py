"""What the 011 role variants read: the trainer's account and the triage view.

The trainer of ``general_live_nccl_role_trainer`` prints one ``ROLES`` line per
rank (its declared group, PP/EP roles and ``<tid>:<role>:<thread name>`` thread
roles by container-native id), a ``SHARER``/``HOLDER`` line per child process
and a ``GFRS-FIXTURE-ERROR`` line when it could not start one;
:func:`read_account` collects them from the Pods' logs and
:func:`declarations` turns them into :class:`Declaration` objects, the oracle
``general_live_nccl_roles`` grades against.

The product side is read as the executor stored it: :func:`triage_rows` decodes
each node's ``COLLECT_HUNG_TRIAGE`` result (the summary codec's zlib/JSON
envelope) into rank rows carrying their node, :func:`triage_decision` finds the
``hung_triage_decision`` the product bound the bundle step with (else the last
triage execution's), and the accessors below read a row's kernel thread
samples (``process-series/v1``), windows and py-spy samples
(``stack-series/v1``) without guessing at absent fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gpu_fault.hung_summary_codec import decode_summary
from scripts.e2e.regional import general_live_nccl as nccl
from scripts.e2e.regional import general_live_nccl_role_trainer as plan
from scripts.e2e.regional.general_live_nccl_job import ManagedNcclJob

#: The whole trainer log fits: 3600 heartbeats per Pod plus the markers.
LOG_TAIL = 6000


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


def label(row: Mapping[str, Any]) -> str:
    return f"{row.get('node_id')} rank {row.get('rank')} pid {row.get('pid')}"
