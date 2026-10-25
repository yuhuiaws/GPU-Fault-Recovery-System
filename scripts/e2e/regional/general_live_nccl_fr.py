"""GF-GENERAL-LIVE-011 flight-recorder forensics variants (live-replay).

The same two-node managed NCCL stall as ``general_live_nccl`` (an active EFA
baseline, one rank in an extra collective, the pair at zero for the hung
threshold, the read-only diagnostic DAG to ESCALATED, traffic back on release),
with the job's PyTorch flight recorder off and the reviewed test dump responder
of ``general_live_fr_replay`` answering ``COLLECT_HUNG_TRIAGE``'s dump requests
in each Pod with dumps of a known shape. The product's FIFO protocol, mount
namespace mapping, parsing, two-dump series, classification and bundle binding
run unchanged on the replayed reading, so the verdict can read what they made of
it: the ``hung_triage_decision`` the COLLECT_DIAGNOSTIC_BUNDLE step was bound
with (``execution/hung_survey.after_triage`` -> ``hung_bundle.bind_hung_bundle``;
also on the last triage execution), the bundle step's scope parameters, and
each sampled rank's ``flight_recorder_series``. Nothing is posted; the hung DAG
stays read-only.

Which rank the product samples is the product's choice: wave 1 samples the
reporting node, wave 2 the peer unless wave 1 already had sufficient local
evidence (``targeted_capture_eligible``). Every expectation here is therefore
stated over the ranks the triage did sample -- a lead the shape put on an
unsampled node is not owed -- and the deciding process groups are node-local,
because without a trusted ``clock_uncertainty_seconds`` (which this site's DAG
does not carry) the product refuses to compare progress across nodes by design.

Fixture faults (no responder ledger, a responder error, a FIFO under the wrong
name) make a variant INCONCLUSIVE. A FIFO that was there but the product never
asked, a dump it did not pair, a lead it did not report or a bundle scope it
got wrong is FAIL.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from scripts.e2e.regional import general_live_nccl as nccl
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
)
from scripts.e2e.regional.general_live_fr_replay import (
    ADVANCE,
    DELTAS,
    ENQUEUE_ONLY,
    PENDING,
    PROOF_SCOPE,
    FlightRecorderReplay,
    ambiguous_candidates_shape,
    different_process_groups_shape,
    few_candidates_shape,
    node_ranks,
    pg_name,
    plus_process_shape,
    request_errors,
    responder_errors,
    second_missing_shape,
    temporal_completion_shape,
    whole_group_shape,
)
from scripts.e2e.regional.general_live_nccl_job import ManagedNcclJob
from scripts.e2e.regional.general_live_registry import LIVE_REPLAY, VariantSpec
from scripts.e2e.regional.general_live_verdicts import parse_time

CASE = nccl.CASE
#: The product's second-dump interval (``orchestration/hung_dag.py``).
FR_INTERVAL_SECONDS = 2.0
#: The bundle scopes ``hung_bundle.bind_hung_bundle`` binds.
TARGETED_SCOPE = "identity_bound_candidates"
CONSERVATIVE_SCOPE = "bounded_triage_nodes"
#: More leads than this keep the conservative scope (``hung_classification``).
MAX_TARGETED_CANDIDATES = 3
#: The process observation the write-mode stall's held ranks produce.
HOST_WRITES = "HOST_WRITES_OBSERVED"


# --------------------------------------------------------------------------- #
# product records (pure)
# --------------------------------------------------------------------------- #
def bundle_step(records: Mapping[str, Any]) -> dict[str, Any]:
    """The COLLECT_DIAGNOSTIC_BUNDLE step as the product bound it (or ``{}``)."""

    return next(
        (
            dict(step)
            for step in nccl.hung_workflow(dict(records)).get("official_steps") or []
            if step.get("operation") == "COLLECT_DIAGNOSTIC_BUNDLE"
        ),
        {},
    )


def decision_of(records: Mapping[str, Any]) -> dict[str, Any] | None:
    """The hung triage decision the bundle was bound with.

    ``bind_hung_bundle`` copies it into the bundle step's parameters; the same
    decision is recorded on the triage execution that produced it, which is the
    fallback when the bundle step was never rebound.
    """

    bound = (bundle_step(records).get("parameters") or {}).get("hung_triage_decision")
    if isinstance(bound, dict):
        return bound
    recorded = [
        item
        for item in nccl.hung_workflow(dict(records)).get("step_executions") or []
        if item.get("operation") == "COLLECT_HUNG_TRIAGE"
        and isinstance((item.get("details") or {}).get("hung_triage_decision"), dict)
    ]
    if not recorded:
        return None
    last = max(recorded, key=lambda item: int(item.get("step_index") or 0))
    return dict(last["details"]["hung_triage_decision"])


def sampled_rows(records: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """The ranked rows the triage sampled, by rank, and the unreadable results."""

    triage = nccl.triage_ranks(dict(records))
    if triage is None:
        return [], ["no COLLECT_HUNG_TRIAGE execution in the hung workflow"]
    rows, unreadable = triage
    ranked = sorted(
        (row for row in rows if type(row.get("rank")) is int),
        key=lambda row: int(row["rank"]),
    )
    return ranked, unreadable


def temporal_branch(decision: Mapping[str, Any]) -> dict[str, Any] | None:
    value = (decision.get("evidence_branches") or {}).get("fr_temporal")
    return dict(value) if isinstance(value, dict) else None


def finding_for(
    decision: Mapping[str, Any], name: str, members: Sequence[int]
) -> dict[str, Any] | None:
    """The FR temporal finding of process group ``name`` with ``members``."""

    branch = temporal_branch(decision) or {}
    for finding in branch.get("process_groups") or []:
        if finding.get("pg_name") == name and set(
            finding.get("expected_members") or []
        ) == set(members):
            return dict(finding)
    return None


@dataclass
class Reading:
    """Everything a flight-recorder verdict reads, from the product's records."""

    shape: FlightRecorderReplay
    nodes: list[str]
    rows: list[dict[str, Any]]
    unreadable: list[str]
    decision: dict[str, Any] | None
    bundle: dict[str, Any]
    ledgers: dict[str, Any] = field(default_factory=dict)

    @property
    def ranks(self) -> list[int]:
        return [int(row["rank"]) for row in self.rows]

    @property
    def sampled_nodes(self) -> list[str]:
        return sorted({str(row.get("node_id")) for row in self.rows})

    def pid_of(self, rank: int) -> tuple[str, int] | None:
        for row in self.rows:
            if row.get("rank") == rank and type(row.get("pid")) is int:
                return str(row.get("node_id")), int(row["pid"])
        return None

    def groups(self) -> list[dict[str, Any]]:
        """The shape's expanded groups a sampled rank is a member of, once each."""

        seen: set[tuple[str, tuple[int, ...]]] = set()
        result = []
        for rank in self.ranks:
            for group in self.shape.rank_groups(rank):
                key = (pg_name(group["uid"], group["desc"]), tuple(group["members"]))
                if key not in seen:
                    seen.add(key)
                    result.append({**group, "name": key[0]})
        return result

    def sampled_members(self, group: Mapping[str, Any]) -> list[int]:
        return sorted(set(group["members"]) & set(self.ranks))


def reading(
    ctx: VariantContext,
    records: Mapping[str, Any],
    shape: FlightRecorderReplay,
    ledgers: Mapping[str, Any] | None = None,
) -> Reading:
    rows, unreadable = sampled_rows(records)
    return Reading(
        shape=shape,
        nodes=nccl.attempt_nodes(ctx),
        rows=rows,
        unreadable=unreadable,
        decision=decision_of(records),
        bundle=bundle_step(records),
        ledgers=dict(ledgers or {}),
    )


# --------------------------------------------------------------------------- #
# verdict stages (pure)
# --------------------------------------------------------------------------- #
def _series(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("flight_recorder_series")
    return dict(value) if isinstance(value, dict) else {}


def paired_errors(view: Reading) -> list[str]:
    """Two real dumps per sampled PID, at least the product's interval apart.

    Every rank of a sampled node was sampled; its series is ``PAIRED`` with both
    snapshots ``dumped``, ``interval_seconds`` at or over the interval and the
    second request stamped after the first observation by at least that much.
    """

    errors = list(view.unreadable)
    if not view.rows:
        return [*errors, "COLLECT_HUNG_TRIAGE sampled no ranked process"]
    owed = {
        rank
        for node, ranks in node_ranks(view.nodes).items()
        if node in view.sampled_nodes
        for rank in ranks
    }
    missing = sorted(owed - set(view.ranks))
    if missing:
        errors.append(f"ranks {missing} of a sampled node were not sampled")
    for row in view.rows:
        rank, series = row.get("rank"), _series(row)
        status = series.get("status")
        if status != "PAIRED":
            errors.append(f"rank {rank}: FR series is {status}, not PAIRED")
            continue
        for point in ("first", "second"):
            if (series.get(point) or {}).get("status") != "dumped":
                errors.append(f"rank {rank}: the {point} dump is not 'dumped'")
        interval = series.get("interval_seconds")
        if not isinstance(interval, (int, float)) or interval < FR_INTERVAL_SECONDS:
            errors.append(
                f"rank {rank}: the two dumps are {interval} s apart, under "
                f"{FR_INTERVAL_SECONDS:g} s"
            )
        first = parse_time(series.get("first_observed_at"))
        requested = parse_time(series.get("second_requested_at"))
        observed = parse_time(series.get("second_observed_at"))
        if first is None or requested is None or observed is None:
            errors.append(f"rank {rank}: the series lacks a timestamp")
        elif not first < requested <= observed:
            errors.append(f"rank {rank}: series timestamps are out of order")
        elif (requested - first).total_seconds() < FR_INTERVAL_SECONDS - 0.5:
            errors.append(
                f"rank {rank}: the second dump was requested "
                f"{(requested - first).total_seconds():.1f} s after the first"
            )
    return errors


def withheld_errors(view: Reading) -> list[str]:
    """The first dump read, the withheld second one recorded as unavailable."""

    errors = list(view.unreadable)
    if not view.rows:
        return [*errors, "COLLECT_HUNG_TRIAGE sampled no ranked process"]
    for row in view.rows:
        rank, series = row.get("rank"), _series(row)
        if (row.get("flight_recorder") or {}).get("status") != "dumped":
            errors.append(f"rank {rank}: the first dump was not read as dumped")
        if series.get("status") != "SECOND_UNAVAILABLE":
            errors.append(
                f"rank {rank}: FR series is {series.get('status')}, not SECOND_UNAVAILABLE"
            )
        if (series.get("second") or {}).get("status") != "dump_missing":
            errors.append(f"rank {rank}: the withheld second dump is not dump_missing")
    return errors


def _decision(view: Reading) -> tuple[dict[str, Any], list[str]]:
    if view.decision is None:
        return {}, ["no hung_triage_decision on the bundle step or a triage execution"]
    return view.decision, []


def _bundle_scope_errors(
    view: Reading, *, targeted: bool, pids: dict[str, set[int]]
) -> list[str]:
    parameters = view.bundle.get("parameters") or {}
    if not view.bundle:
        return ["no COLLECT_DIAGNOSTIC_BUNDLE step in the hung workflow"]
    errors = []
    scope = parameters.get("bundle_scope")
    expected = TARGETED_SCOPE if targeted else CONSERVATIVE_SCOPE
    if scope != expected:
        errors.append(f"bundle scope is {scope!r}, expected {expected!r}")
    if parameters.get("capture_process_state") is not targeted:
        errors.append(
            f"bundle capture_process_state is {parameters.get('capture_process_state')}"
        )
    if parameters.get("operator_escalation_required") is targeted:
        errors.append(
            "bundle operator_escalation_required is "
            f"{parameters.get('operator_escalation_required')}"
        )
    bound = {
        str(node): {int(pid) for pid in values}
        for node, values in (parameters.get("target_pids_by_node") or {}).items()
    }
    if targeted:
        if bound != pids:
            errors.append(f"bundle targets {bound}, expected the candidate PIDs {pids}")
        if parameters.get("max_processes") != sum(len(v) for v in pids.values()):
            errors.append(f"bundle max_processes is {parameters.get('max_processes')}")
        if sorted(view.bundle.get("node_ids") or []) != sorted(pids):
            errors.append(
                f"bundle nodes {view.bundle.get('node_ids')} are not the candidates' "
                f"{sorted(pids)}"
            )
    else:
        if bound:
            errors.append(f"a conservative bundle still targets PIDs {bound}")
        if parameters.get("max_processes") != 0:
            errors.append(
                f"bundle max_processes is {parameters.get('max_processes')}, not 0"
            )
    return errors


def _fr_candidate_codes(decision: Mapping[str, Any]) -> list[str]:
    return sorted(
        {
            str(code)
            for item in decision.get("candidate_processes") or []
            for code in item.get("reasons") or []
            if str(code).startswith("FR_")
        }
    )


def candidate_errors(view: Reading, *, targeted: bool) -> list[str]:
    """The leads the shape planted on the sampled ranks, and the bundle scope.

    ``targeted``: 1-3 leads -> ``targeted_capture_eligible`` and a bundle bound
    to exactly the culprit and control PIDs; otherwise the leads are reported
    (``culprit_ranks``) but the bundle keeps the conservative scope.
    """

    decision, errors = _decision(view)
    if errors:
        return errors
    expected = sorted(view.shape.lagging_ranks() & set(view.ranks))
    culprits = sorted(int(rank) for rank in decision.get("culprit_ranks") or [])
    if culprits != expected:
        errors.append(f"culprit ranks {culprits} != the planted leads {expected}")
    if decision.get("candidate_count") != len(expected):
        errors.append(
            f"candidate_count {decision.get('candidate_count')} != {len(expected)}"
        )
    controls = [int(rank) for rank in decision.get("control_ranks") or []]
    if len(controls) > 1 or set(controls) & set(culprits):
        errors.append(f"control ranks {controls} must be at most one non-culprit rank")
    if controls and controls[0] not in view.ranks:
        errors.append(f"control rank {controls[0]} was never sampled")
    if decision.get("classification") != "PLAUSIBLE":
        errors.append(
            f"classification {decision.get('classification')!r}, expected PLAUSIBLE"
        )
    for flag in ("temporal_verified", "membership_verified"):
        if decision.get(flag) is not True:
            errors.append(f"{flag} is {decision.get(flag)}")
    eligible = decision.get("targeted_capture_eligible")
    if eligible is not targeted:
        errors.append(f"targeted_capture_eligible is {eligible}, expected {targeted}")
    if not targeted and len(expected) <= MAX_TARGETED_CANDIDATES:
        errors.append(f"only {len(expected)} leads were planted; that is not ambiguous")
    pids: dict[str, set[int]] = {}
    for rank in [*culprits, *controls] if targeted else []:
        located = view.pid_of(rank)
        if located is None:
            errors.append(f"rank {rank} has no PID in the triage rows")
        else:
            pids.setdefault(located[0], set()).add(located[1])
    errors.extend(_bundle_scope_errors(view, targeted=targeted, pids=pids))
    return errors


def watermark_errors(view: Reading) -> list[str]:
    """Each group's enqueued/started/completed deltas are the shape's, per rank."""

    decision, errors = _decision(view)
    if errors:
        return errors
    if temporal_branch(decision) is None:
        return ["the decision has no fr_temporal branch (no PAIRED series)"]
    for group in view.groups():
        kind, members = group["progress"], view.sampled_members(group)
        finding = finding_for(decision, group["name"], group["members"])
        if finding is None:
            errors.append(
                f"no FR temporal finding for group {group['name']} {group['members']}"
            )
            continue
        deltas = finding.get("watermark_deltas_by_rank") or {}
        for rank in members:
            if deltas.get(str(rank)) != DELTAS[kind]:
                errors.append(
                    f"{group['name']} rank {rank}: watermark deltas {deltas.get(str(rank))} "
                    f"!= {DELTAS[kind]} ({kind})"
                )
        expectations = {
            "completion_advanced_ranks": members if kind == ADVANCE else [],
            "enqueue_advanced_ranks": members
            if kind in (ADVANCE, ENQUEUE_ONLY)
            else [],
            "persistent_pending_ranks": members
            if kind in (PENDING, ENQUEUE_ONLY)
            else [],
        }
        for key, wanted in expectations.items():
            if sorted(int(r) for r in finding.get(key) or []) != wanted:
                errors.append(f"{group['name']}: {key} {finding.get(key)} != {wanted}")
    if decision.get("culprit_ranks"):
        errors.append(
            f"uniform progress yielded culprits {decision.get('culprit_ranks')}"
        )
    return errors


def whole_group_errors(view: Reading) -> list[str]:
    """A group pending as a whole is a stall, never a single rank's fault."""

    decision, errors = _decision(view)
    if errors:
        return errors
    stalled = list(decision.get("stalled_process_groups") or [])
    local = [
        g
        for g in view.groups()
        if g["progress"] == PENDING
        and len(g["members"]) < len(view.shape.document()["ranks"])
    ]
    if not local:
        errors.append("the shape has no node-local pending group to observe")
    for group in local:
        finding = finding_for(decision, group["name"], group["members"])
        if finding is None:
            errors.append(f"no FR temporal finding for group {group['name']}")
            continue
        if finding.get("group_stall_observed") is not True:
            errors.append(
                f"{group['name']}: group_stall_observed is {finding.get('group_stall_observed')} "
                f"({finding.get('time_reason')}, members {finding.get('observed_members')})"
            )
        if group["name"] not in stalled:
            errors.append(f"{group['name']} is not in stalled_process_groups {stalled}")
        pending = sorted(int(r) for r in finding.get("persistent_pending_ranks") or [])
        if pending != view.sampled_members(group):
            errors.append(f"{group['name']}: persistent_pending_ranks {pending}")
    for group in view.groups():
        if group["progress"] != PENDING or group in local:
            continue
        finding = finding_for(decision, group["name"], group["members"])
        pending = sorted(
            int(r) for r in (finding or {}).get("persistent_pending_ranks") or []
        )
        if pending != view.sampled_members(group):
            errors.append(
                f"{group['name']}: persistent_pending_ranks {pending} != every sampled rank"
            )
    if decision.get("culprit_ranks"):
        errors.append(
            f"a whole-group stall was blamed on ranks {decision['culprit_ranks']}"
        )
    if decision.get("candidate_count"):
        errors.append(
            f"candidate_count {decision.get('candidate_count')} for a whole-group stall"
        )
    if "FR_PROGRESS_OUTLIER" in _fr_candidate_codes(decision):
        errors.append("a rank was marked FR_PROGRESS_OUTLIER in a whole-group stall")
    if decision.get("targeted_capture_eligible") is not False:
        errors.append("targeted_capture_eligible is not False")
    errors.extend(_bundle_scope_errors(view, targeted=False, pids={}))
    return errors


def separate_groups_errors(view: Reading) -> list[str]:
    """Progress is compared inside a process group, never across groups."""

    decision, errors = _decision(view)
    if errors:
        return errors
    branch = temporal_branch(decision)
    if branch is None:
        return ["the decision has no fr_temporal branch (no PAIRED series)"]
    by_node = node_ranks(view.nodes)
    for finding in branch.get("process_groups") or []:
        observed = set(int(r) for r in finding.get("observed_members") or [])
        expected = set(int(r) for r in finding.get("expected_members") or [])
        if not observed <= expected:
            errors.append(
                f"{finding.get('pg_name')}: observed {sorted(observed)} outside its "
                f"members {sorted(expected)}"
            )
        if expected and len(expected) < len(view.shape.document()["ranks"]):
            homes = {node for node, ranks in by_node.items() if expected & set(ranks)}
            if len(homes) != 1:
                errors.append(
                    f"{finding.get('pg_name')}: a node group spans {sorted(homes)}"
                )
    local = [g for g in view.groups() if g["progress"] == PENDING]
    seen = []
    for group in local:
        finding = finding_for(decision, group["name"], group["members"])
        if finding is None:
            errors.append(f"no FR temporal finding for group {group['name']}")
            continue
        seen.append(finding.get("membership_sha256"))
        if finding.get("group_stall_observed") is not True:
            errors.append(f"{group['name']}: its own stall was not observed")
    if len(seen) != len(set(seen)):
        errors.append("two different node groups share one membership digest")
    if decision.get("culprit_ranks"):
        errors.append(
            f"ranks {decision['culprit_ranks']} were blamed although groups only differ"
        )
    if decision.get("candidate_count"):
        errors.append(f"candidate_count {decision.get('candidate_count')}")
    return errors


def second_missing_errors(view: Reading) -> list[str]:
    """A withheld second dump is a forensics gap, never a stall or a lead."""

    decision, errors = _decision(view)
    if errors:
        return errors
    branches = decision.get("evidence_branches") or {}
    if branches.get("fr_temporal") is not None:
        errors.append("an FR temporal verdict was formed without a paired series")
    if branches.get("snapshot_authoritative") is not True:
        errors.append("the single snapshot is not marked authoritative")
    if decision.get("stalled_process_groups"):
        errors.append(
            f"stalled_process_groups {decision['stalled_process_groups']} from one dump"
        )
    if decision.get("culprit_ranks"):
        errors.append(f"culprit ranks {decision['culprit_ranks']} from one dump")
    if decision.get("classification") == "PLAUSIBLE":
        errors.append("classification PLAUSIBLE without a comparable second dump")
    if decision.get("targeted_capture_eligible") is not False:
        errors.append("targeted_capture_eligible is not False")
    codes = _fr_candidate_codes(decision)
    if codes:
        errors.append(f"FR candidate codes {codes} without a second dump")
    errors.extend(_bundle_scope_errors(view, targeted=False, pids={}))
    return errors


def process_evidence_errors(view: Reading) -> list[str]:
    """FR completion progress leaves the independent process I/O evidence in place."""

    decision, errors = _decision(view)
    if errors:
        return errors
    processes = (
        (decision.get("evidence_branches") or {}).get("process_observations") or {}
    ).get("processes") or []
    if not processes:
        return ["the decision carries no process observations"]
    writers = 0
    for item in processes:
        label = f"{item.get('node_id')}/{item.get('pid')}"
        if not item.get("fr_completion_advanced_groups"):
            errors.append(f"{label}: FR completion did not read as advanced")
        if item.get("fr_persistent_pending_groups"):
            errors.append(
                f"{label}: a pending group {item['fr_persistent_pending_groups']}"
            )
        behavior = item.get("behavior") or {}
        if not isinstance(behavior.get("io_windows"), list) or not behavior.get(
            "io_windows"
        ):
            errors.append(f"{label}: no process I/O windows kept beside the FR branch")
        codes = {obs.get("code") for obs in behavior.get("observations") or []}
        writers += HOST_WRITES in codes
        kept = {obs.get("code") for obs in item.get("observations") or []}
        if not codes <= kept:
            errors.append(
                f"{label}: behavior observations {sorted(codes - kept)} dropped"
            )
    if not writers:
        errors.append(
            f"no sampled process kept {HOST_WRITES} (held ranks rewrite 4 MiB/s)"
        )
    if decision.get("culprit_ranks"):
        errors.append(
            f"advancing FR progress yielded culprits {decision['culprit_ranks']}"
        )
    if _fr_candidate_codes(decision):
        errors.append(
            f"FR candidate codes {_fr_candidate_codes(decision)} on advancing progress"
        )
    return errors


def decision_excerpt(view: Reading) -> dict[str, Any]:
    decision = view.decision or {}
    parameters = view.bundle.get("parameters") or {}
    return {
        "classification": decision.get("classification"),
        "culprit_ranks": decision.get("culprit_ranks"),
        "control_ranks": decision.get("control_ranks"),
        "candidate_count": decision.get("candidate_count"),
        "targeted_capture_eligible": decision.get("targeted_capture_eligible"),
        "stalled_process_groups": decision.get("stalled_process_groups"),
        "sampled_ranks": view.ranks,
        "sampled_nodes": view.sampled_nodes,
        "bundle_scope": parameters.get("bundle_scope"),
        "bundle_nodes": view.bundle.get("node_ids"),
        "target_pids_by_node": parameters.get("target_pids_by_node"),
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
Check = Callable[[Reading], list[str]]


def _replayed(
    ctx: VariantContext,
    suffix: str,
    shape: FlightRecorderReplay,
    checks: Mapping[str, Check],
    *,
    notes: Sequence[str],
    stall_mode: str = "quiet",
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def after_hung(job: ManagedNcclJob, _records: dict[str, Any]) -> None:
        captured["ledgers"] = job.flight_ledgers()

    identity, records, stages = nccl.stalled_chain(
        ctx, suffix, stall_mode=stall_mode, flight_recorder=shape, after_hung=after_hung
    )
    ledgers = captured.get("ledgers") or {}
    ctx.evidence.write(f"{suffix}-fr-ledgers", ledgers)
    ctx.evidence.write(f"{suffix}-fr-replay", shape.evidence())
    view = reading(ctx, records, shape, ledgers)
    fixture = responder_errors(ledgers, view.nodes)
    if fixture:
        raise VariantInconclusive(
            "the test flight-recorder responder did not hold: " + "; ".join(fixture)
        )
    stages["replay_requests"] = request_errors(
        ledgers, view.nodes, sampled_nodes=view.sampled_nodes
    )
    for name, check in checks.items():
        stages[name] = check(view)
    return nccl.variant_result(
        stages,
        notes=[*notes, "proof scope live-replay: only the FR dumps were replaced"],
        extra={
            "attempt": identity,
            "proof_scope": PROOF_SCOPE,
            "replay": shape.evidence(),
            "decision": decision_excerpt(view),
        },
    )


def few_candidates(ctx: VariantContext) -> dict[str, Any]:
    return _replayed(
        ctx,
        "fewc",
        few_candidates_shape(),
        {
            "fr_paired": paired_errors,
            "candidates": lambda v: candidate_errors(v, targeted=True),
        },
        notes=[
            "one rank per node a collective behind its node-local group: 1-2 trusted"
            " leads -> targeted bundle on exactly those PIDs plus at most one control rank"
        ],
    )


def ambiguous_candidates(ctx: VariantContext) -> dict[str, Any]:
    return _replayed(
        ctx,
        "ambc",
        ambiguous_candidates_shape(),
        {
            "fr_paired": paired_errors,
            "candidates": lambda v: candidate_errors(v, targeted=False),
        },
        notes=[
            "two node-local groups with two laggards each: four leads on one node are"
            " reported but keep the conservative bundle scope"
        ],
    )


def different_process_groups(ctx: VariantContext) -> dict[str, Any]:
    return _replayed(
        ctx,
        "dpg",
        different_process_groups_shape(),
        {"fr_paired": paired_errors, "separate_groups": separate_groups_errors},
        notes=[
            "node groups pending at sequence 1200 and 700: compared only inside each"
            " group, never across; no rank behind, no culprit"
        ],
    )


def fr_temporal_completion(ctx: VariantContext) -> dict[str, Any]:
    return _replayed(
        ctx,
        "frtc",
        temporal_completion_shape(),
        {"fr_paired": paired_errors, "watermarks": watermark_errors},
        notes=[
            "two dumps per PID >= 2 s apart; completion advancing in the default group"
            " (+2/+2/+2), enqueue without completion in the second (+1/0/0, pending persists)"
        ],
    )


def fr_second_missing(ctx: VariantContext) -> dict[str, Any]:
    return _replayed(
        ctx,
        "frsm",
        second_missing_shape(),
        {"fr_withheld": withheld_errors, "forensics_gap": second_missing_errors},
        notes=[
            "the responder withholds every second dump: SECOND_UNAVAILABLE per PID, no"
            " temporal verdict, no stall or lead invented, conservative bundle"
        ],
    )


def fr_whole_group(ctx: VariantContext) -> dict[str, Any]:
    return _replayed(
        ctx,
        "frwg",
        whole_group_shape(),
        {"fr_paired": paired_errors, "whole_group": whole_group_errors},
        notes=[
            "every rank pending on the same collective: the node group's stall is"
            " observed, no single culprit rank, conservative bundle"
        ],
    )


def fr_plus_process(ctx: VariantContext) -> dict[str, Any]:
    return _replayed(
        ctx,
        "frpp",
        plus_process_shape(),
        {"fr_paired": paired_errors, "process_evidence": process_evidence_errors},
        notes=[
            "write-mode stall: held ranks rewrite 4 MiB/s; FR completion advancing does"
            " not silence HOST_WRITES_OBSERVED / the process I/O windows (a D-state"
            " wait cannot be forced natively)"
        ],
        stall_mode="write",
    )


def _spec(variant: str, injection: str, expected: str, execute: Any) -> VariantSpec:
    return nccl.variant_spec(
        variant,
        injection,
        expected,
        execute,
        proof=LIVE_REPLAY,
        max_minutes=65,
    )


REPLAY = (
    "the stalled two-node job runs PyTorch's recorder off; the test dump responder in"
    " each Pod answers the product's FIFO dump requests with "
)

VARIANTS = [
    _spec(
        "few_candidates",
        REPLAY + "one rank per node a collective behind its node-local group",
        "1-3 trusted candidates: PLAUSIBLE, targeted_capture_eligible, the bundle bound to"
        " exactly the culprit and control PIDs (identity_bound_candidates); chain -> ESCALATED",
        few_candidates,
    ),
    _spec(
        "ambiguous_candidates",
        REPLAY + "two node-local groups with two laggards each (four leads per node)",
        "too many candidates: leads reported, targeted_capture_eligible False, the bundle"
        " keeps bounded_triage_nodes with no target PIDs; chain -> ESCALATED",
        ambiguous_candidates,
    ),
    _spec(
        "different_process_groups",
        REPLAY + "node groups pending at sequence 1200 (node A) and 700 (node B)",
        "collective progress is compared inside each process group only: no culprit, each"
        " group's own stall observed, memberships kept apart; chain -> ESCALATED",
        different_process_groups,
    ),
    _spec(
        "fr_temporal_completion",
        REPLAY
        + "a default group whose completion advances and a group that only enqueues",
        "two new dumps per PID/PG >= 2 s apart; watermark deltas +2/+2/+2 and +1/0/0 with"
        " the pending operation persisting; no culprit; chain -> ESCALATED",
        fr_temporal_completion,
    ),
    _spec(
        "fr_second_missing",
        REPLAY + "a first dump only; the second request is withheld",
        "SECOND_UNAVAILABLE per PID, no temporal verdict, snapshot authoritative, no stall"
        " or lead invented, conservative bundle; chain -> ESCALATED",
        fr_second_missing,
    ),
    _spec(
        "fr_whole_group",
        REPLAY
        + "every rank pending on the same collective of its node group and the job",
        "group_stall_observed for the node group, stalled_process_groups named, no culprit"
        " rank, conservative bundle; chain -> ESCALATED",
        fr_whole_group,
    ),
    _spec(
        "fr_plus_process",
        REPLAY
        + "completion advancing everywhere while the held ranks rewrite checkpoints",
        "FR completion advanced per process and the independent process I/O evidence"
        " (HOST_WRITES_OBSERVED, io windows) both kept; no FR lead; chain -> ESCALATED",
        fr_plus_process,
    ),
]
