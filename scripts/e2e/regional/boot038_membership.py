"""GF-REGIONAL-BOOT-038: cross-cluster membership while another cluster is NotReady.

This module holds everything BOOT-038 can decide without a live cluster: the
trigger grammar of the admin verb's narration, the four surfaces a scoped
verify records another cluster's finding on, the per-verb verdict and the
after-run judgement. ``run_boot038_cross_cluster_membership`` keeps the live
I/O (spawning the verb, the kubelet service window, the store reads) and hands
these functions already-collected evidence, so every verdict path is a unit
test with no cluster access.

The premise (step 1 of the brief), settled by reading the product:

* The DESTR-008 service window the harness opens is **invisible to the
  product**. Its claim is a node-local file under
  ``/var/lib/gpu-fault-acceptance/service-window/`` plus a controller journal
  under the harness's own state directory
  (``destr008_service_window.ServiceWindowController``); nothing in ``src/`` or
  ``deploy/`` reads either (``grep`` finds no ``gpu-fault-acceptance`` /
  ``service-window`` consumer). The coverage check
  (``regional_admin_checks_coverage.check_agent_coverage``) and the
  departed-node cleanup (``departed_node_cleanup._candidates``) read only the
  Kubernetes Node list the executor reports, the fleet Agent records and the
  provider -- never the claim.
* Stopping ``kubelet.service`` (the node Agent runs under its own systemd unit,
  not kubelet, so it keeps heart-beating over its outbound channel and its
  lease stays live) makes the node's Kubernetes Node go NotReady while its
  Agent stays ACTIVE. That is exactly the live 2026-11-04 12:20Z state: the
  verify sees the other cluster's coverage drift (an Agent whose Node is listed
  but not Ready), and the departed-node cleanup never makes the node a
  candidate (its lease is live), so no "departed" mail goes out for it.

So the ordinary **claimed** stop reproduces the unplanned loss and keeps the
node-side recovery safety intact; no unclaimed variant is needed. The claim
changes nothing the verify or the cleanup sees.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

CASE_ID = "GF-REGIONAL-BOOT-038"
CONFIRMATION = "BOOT038_CROSS_CLUSTER_MEMBERSHIP_WINDOW"

REMOVE_VERB = "remove-cluster"
JOIN_VERB = "join-cluster"
VERBS = (REMOVE_VERB, JOIN_VERB)

WINDOW_KUBELET = "kubelet"
WINDOW_NONE = "none"
WINDOWS = (WINDOW_KUBELET, WINDOW_NONE)

#: The on-node service the window stops (the node Agent is never touched).
KUBELET_SERVICE = "kubelet.service"
#: The fixture identity the on-node probe trusts (``warm_spare_node_probe``
#: honours a window only for this case id); BOOT-038 reuses the DESTR-008
#: window exactly as GF-GENERAL-LIVE-030 does, under its own run id.
SERVICE_WINDOW_FIXTURE_CASE_ID = "GF-REGIONAL-DESTR-008"

#: kubelet down time, stop to restore. On this site a node shows NotReady
#: later than 60-80 s after the stop (live 2026-11-04: an Agent coverage read
#: about 80 s after the stop still saw it Ready), and the node lifecycle
#: controller taints it at about stop + 50-90 s with eviction of non-tolerating
#: Pods 300 s after the taint -- so a stop -> restore of 280 s stays clear of
#: evictions while a slower verb still reads the drift before the node-side
#: restore unit fires. The explicit restore at the verify's ``release-end``
#: ends it sooner when the verify finishes first.
RESTORE_SECONDS = 280
STOP_DELAY_SECONDS = 15
#: Hard cap on stop -> restore; the explicit restore is issued at the verify's
#: ``release-end`` and never later than this. The on-node restore unit is the
#: backstop past it, and eviction is 300 s after the taint.
RESTORE_CAP_SECONDS = 300
#: The poller runs from the window open; at verify-begin its state is read at
#: once with no wait -- "NotReady before the verify" must mean already NotReady
#: when the verify started, not NotReady within a grace after it (live
#: 2026-11-04 the removal's verify read the fleet while the node was still
#: Ready, so no finding appeared; a 90 s grace then saw NotReady and mis-graded
#: a missed window as a judged verb).
NOT_READY_POLL_SECONDS = 5.0

# --- the admin verb's narration grammar (regional_release_narration) ---------
# release-begin/-end <stamp> mode=<mode> ...; a fleet wave's START line is
# ``fleet-wave <ts> cluster=<id> phase=join wave=N/M ...`` and its done line is
# ``fleet-wave-done ... wave=N/M ... ready=R/T``.
_RELEASE_BEGIN = r"release-begin\s+\S+\s+mode={mode}\b"
_RELEASE_END = r"release-end\s+\S+\s+mode={mode}\b"
REMOVE_CLUSTER_BEGIN = re.compile(_RELEASE_BEGIN.format(mode="remove-cluster"))
VERIFY_BEGIN = re.compile(_RELEASE_BEGIN.format(mode="verify"))
VERIFY_END = re.compile(_RELEASE_END.format(mode="verify"))
FAIL_CLUSTER = re.compile(r"release-(?:begin|end)\s+\S+\s+mode=fail-cluster\b")
ROLLBACK_CLUSTER = re.compile(r"release-(?:begin|end)\s+\S+\s+mode=rollback-cluster\b")
#: A fleet wave's START line (``wave=N/M``). The join opens the window on the
#: last wave's start (``wave=M/M``) -- earlier than its ``fleet-wave-done``,
#: because this site's NotReady latency outran the later trigger.
FLEET_WAVE_START = re.compile(
    r"fleet-wave\s+\S+\s+cluster=\S+\s+phase=join\s+wave=(\d+)/(\d+)\b"
)
#: ``fleet-wave-done ... wave=N/M ... ready=R/T`` -- read to tell the last wave
#: from an earlier one when checking the window-open stream after the fact.
FLEET_WAVE_DONE = re.compile(
    r"fleet-wave-done\b.*\bwave=(\d+)/(\d+)\b.*\bready=(\d+)/(\d+)\b"
)

#: The subject prefixes the departed-node cleanup mails (departed_node_cleanup).
DEPARTED_SUBJECTS = (
    "[GPU node departed: operator action needed]",
    "[NOTICE][GPU node departed: records ended]",
)

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICT_NOT_APPLICABLE = "NOT_APPLICABLE"
VERDICT_NOT_RUN = "NOT_RUN"
VERDICT_BLOCKED = "BLOCKED"

#: The four surfaces b250 records a not-gating finding on.
FINDING_SURFACES = ("result", "verified_evidence", "findings_record", "status_lines")


#: Keys a membership verb's own result carries (``removal_result`` /
#: ``_complete_join``), telling it apart from the many other top-level JSON
#: objects the same stream also carries: the gpu-fault-admin wrapper tees the
#: verb's stdout -- every ``+ aws`` command's JSON output, each check's details,
#: the echoed verify report -- and the release narration (stderr) into one log,
#: so a real removal/join stream holds dozens of ``{`` objects and the verb
#: result is only the last one with this signature.
VERB_RESULT_SIGNATURE = ("cpu_control_plane", "gpu_cluster")


def _top_level_objects(lines: Sequence[str]) -> list[dict[str, Any]]:
    """Every top-level JSON object in the merged verb output, in order.

    An object printed by ``json.dumps(..., indent=2)`` (the verb result, the
    verify report, an AWS CLI output) starts with ``{`` on its own line; a line
    that is exactly ``{`` begins a candidate and the decoder reads to its end.
    """

    found: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if line.strip() != "{":
            continue
        try:
            value, _end = json.JSONDecoder().raw_decode("".join(lines[index:]))
        except ValueError:
            continue
        if isinstance(value, dict):
            found.append(value)
    return found


def verb_result_from_output(lines: Sequence[str]) -> dict[str, Any] | None:
    """The membership verb's own result object from its merged output.

    The last top-level object carrying the verb-result signature
    (``cpu_control_plane`` + ``gpu_cluster``); the last decodable object as a
    fallback; ``None`` when the stream carries no JSON object. Reading the first
    ``{`` is wrong -- that is an AWS output or the echoed verify report, not the
    verb's ``not_gating_findings``-bearing result.
    """

    objects = _top_level_objects(lines)
    for value in reversed(objects):
        if all(key in value for key in VERB_RESULT_SIGNATURE):
            return value
    return objects[-1] if objects else None


def verify_trigger(verb: str) -> re.Pattern[str]:
    """The line after which the window opens for ``verb``.

    Removal: ``release-begin mode=remove-cluster`` -- the window opens before
    the ~113 s remove, the ~57 s gap and the ~50 s sync-state so the node is
    NotReady by the time the verify reads the fleet (its later ``release-end``
    left too little time on this site). Join: the last wave's START line
    (``is_last_wave_start``), earlier than its ``fleet-wave-done``.
    """

    if verb == REMOVE_VERB:
        return REMOVE_CLUSTER_BEGIN
    if verb == JOIN_VERB:
        return FLEET_WAVE_START
    raise ValueError(f"unknown membership verb {verb!r}")


def is_last_wave_start(line: str) -> bool:
    """A fleet-wave START line for the last wave (``wave=M/M``).

    A join of N nodes runs one or more waves; the last is ``wave=M/M`` (N == M),
    and a single-wave join is ``wave=1/1`` -- both open the window on their own
    start line. An earlier ``wave=k/M`` (k < M) does not.
    """

    match = FLEET_WAVE_START.search(line)
    if match is None:
        return False
    return match.group(1) == match.group(2) and match.group(2) != "0"


def is_full_fleet_wave(line: str) -> bool:
    """A ``fleet-wave-done`` line whose ``ready=T/T`` is complete (R == T)."""

    match = FLEET_WAVE_DONE.search(line)
    if match is None:
        return False
    return match.group(3) == match.group(4) and match.group(4) != "0"


# --- parsing the followed output ---------------------------------------------


@dataclass
class VerbNarration:
    """What following one admin verb's output showed, trigger by trigger."""

    open_trigger_seen: bool = False
    verify_began: bool = False
    verify_ended: bool = False
    saw_fail_cluster: bool = False
    saw_rollback_cluster: bool = False
    last_wave_start: str | None = None

    def observe(self, verb: str, line: str) -> str | None:
        """Fold one output line in; return a transition token the caller acts on.

        Tokens: ``open`` (open the window), ``verify-begin`` (read the poller's
        state), ``verify-end`` (restore). Removal opens on ``release-begin
        mode=remove-cluster``; join opens on the last wave's START line
        (``wave=M/M``), the caller opening the window only on the first ``open``.
        """

        if FAIL_CLUSTER.search(line):
            self.saw_fail_cluster = True
        if ROLLBACK_CLUSTER.search(line):
            self.saw_rollback_cluster = True
        if VERIFY_END.search(line):
            self.verify_ended = True
            return "verify-end"
        if VERIFY_BEGIN.search(line):
            self.verify_began = True
            return "verify-begin"
        if verb == REMOVE_VERB and REMOVE_CLUSTER_BEGIN.search(line):
            self.open_trigger_seen = True
            return "open"
        if verb == JOIN_VERB and is_last_wave_start(line):
            self.last_wave_start = line.strip()
            self.open_trigger_seen = True
            return "open"
        return None


# --- the four finding surfaces -----------------------------------------------


def _findings_list(value: Any) -> list[dict[str, str]]:
    return [
        {key: str(item.get(key) or "") for key in ("check", "cluster_id", "finding")}
        for item in (value if isinstance(value, list) else [])
        if isinstance(item, dict)
    ]


def finding_names_cluster(
    findings: Sequence[Mapping[str, Any]], cluster_id: str
) -> bool:
    """True when some not-gating finding is the other cluster's own."""

    return any(str(item.get("cluster_id")) == cluster_id for item in findings)


def result_findings(result: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """``not_gating_findings`` the verb printed in its result JSON."""

    value = (result or {}).get("not_gating_findings")
    return _findings_list(value)


def verified_evidence_findings(state: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """``not_gating_findings`` from the verb state's ``evidence.VERIFIED``."""

    verified = ((state or {}).get("evidence") or {}).get("VERIFIED")
    value = verified.get("not_gating_findings") if isinstance(verified, dict) else None
    return _findings_list(value)


def findings_record(record: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """``not_gating_findings`` from ``<state-dir>/membership-verify/findings.json``."""

    return _findings_list((record or {}).get("not_gating_findings"))


def status_line_names_cluster(status_lines: Sequence[str], cluster_id: str) -> bool:
    """A ``status`` ``not gating`` finding line that names the other cluster.

    ``status`` prints ``<verb> <target>: not gating: <cluster_id> <check>: ...``
    (``membership_verify.status_view``); the line names the other cluster id.
    """

    return any(
        "not gating:" in line and cluster_id in line.split("not gating:", 1)[1]
        for line in status_lines
    )


def finding_surfaces(
    *,
    other_cluster_id: str,
    result: Mapping[str, Any] | None,
    verified_state: Mapping[str, Any] | None,
    record: Mapping[str, Any] | None,
    status_lines: Sequence[str],
) -> dict[str, bool]:
    """Whether the other cluster's finding is recorded on each of the four
    surfaces b250 writes it to."""

    return {
        "result": finding_names_cluster(result_findings(result), other_cluster_id),
        "verified_evidence": finding_names_cluster(
            verified_evidence_findings(verified_state), other_cluster_id
        ),
        "findings_record": finding_names_cluster(
            findings_record(record), other_cluster_id
        ),
        "status_lines": status_line_names_cluster(status_lines, other_cluster_id),
    }


# --- per-verb evidence and verdict -------------------------------------------


@dataclass
class VerbEvidence:
    """Everything the per-verb verdict reads, collected by the live run."""

    verb: str
    window: str
    exit_code: int
    saw_fail_cluster: bool
    saw_rollback_cluster: bool
    #: kubelet (plan A): the window was opened on the trigger. none (plan B):
    #: always False.
    window_opened: bool
    #: True when the other node was NotReady before the verify began (plan A),
    #: or observed NotReady/departed at the verify (plan B); False when it was
    #: not; None when there was nothing to observe (plan B, no real departure).
    other_node_not_ready_before_verify: bool | None
    #: The four finding surfaces (``finding_surfaces``).
    finding_surfaces: dict[str, bool]
    #: The target's own checks: verb exit 0 and no fail/rollback is the proxy,
    #: plus S69 registry convergence.
    registry_convergence_pass: bool
    #: The explicit restore (plan A) finished -- kubelet back, node Ready.
    restore_ok: bool = True
    #: ``resume_cleanup`` was invoked after a failed restore (set by the run).
    resume_cleanup_called: bool = False
    #: Why the ``status`` read did not succeed, else ``None``; a status rc != 0
    #: is an evidence error, never read silently as "no finding".
    status_error: str | None = None
    #: Recorded for plan B: what each verify saw of the other node.
    window_observation: str = ""
    #: ISO timestamps of the window: ``stop`` (open), ``not_ready_seen`` (first
    #: NotReady the poller saw), ``verify_begin``, ``restore`` -- so a missed
    #: window is diagnosable (live 2026-11-04 the removal's verify began before
    #: the node went NotReady).
    window_timeline: dict[str, str] = field(default_factory=dict)


def proof_expected(evidence: VerbEvidence) -> bool:
    """Whether the scenario actually put the other cluster NotReady at verify.

    Only then must the not-gating finding appear; a missed window (plan A) or a
    departure that never came (plan B) cannot have produced one.
    """

    return evidence.other_node_not_ready_before_verify is True


def judge_verb(evidence: VerbEvidence) -> tuple[str, list[str]]:
    """The verb's verdict and its reasons (PASS / FAIL / INCONCLUSIVE).

    FAIL dominates: a rollback, a non-zero exit, a lost registry convergence, a
    failed restore or a status read that did not succeed fails the verb whatever
    the window did. Only when nothing failed does a missed window downgrade PASS
    to INCONCLUSIVE.
    """

    fail: list[str] = []
    note: list[str] = []
    if not evidence.restore_ok:
        fail.append(
            "the kubelet restore did not finish (resume_cleanup "
            + ("invoked" if evidence.resume_cleanup_called else "NOT invoked")
            + ")"
        )
    if evidence.exit_code != 0:
        fail.append(f"{evidence.verb} exited {evidence.exit_code}, not 0")
    if evidence.saw_fail_cluster:
        fail.append("the verb printed mode=fail-cluster")
    if evidence.saw_rollback_cluster:
        fail.append("the verb printed mode=rollback-cluster")
    if not evidence.registry_convergence_pass:
        fail.append("S69 registry convergence did not hold after the verb")
    if evidence.status_error:
        # A status read that did not succeed is an error, not an empty finding
        # set: the status surface is then indeterminate, so the verb cannot pass.
        fail.append(f"the status read did not succeed: {evidence.status_error}")
    expected = proof_expected(evidence)
    if expected:
        missing = [
            name for name in FINDING_SURFACES if not evidence.finding_surfaces.get(name)
        ]
        if missing:
            fail.append(
                "the other cluster's finding is not recorded as not gating on: "
                + ", ".join(missing)
            )
        else:
            note.append(
                "the other cluster's finding was recorded as not gating on all "
                "four surfaces (result, VERIFIED evidence, findings.json, status)"
            )
    if fail:
        return VERDICT_FAIL, fail
    if not expected:
        if evidence.window == WINDOW_KUBELET:
            reason = (
                "the window was not opened"
                if not evidence.window_opened
                else "the node did not show NotReady before the verify began "
                "(window missed)"
            )
        else:
            reason = (
                "no real departure reached the other cluster during the verify "
                f"({evidence.window_observation or 'none observed'})"
            )
        return VERDICT_INCONCLUSIVE, [
            f"{evidence.verb} completed cleanly but the scenario was not "
            f"reproduced: {reason}; the scoped-verify proof was not exercised"
        ]
    return VERDICT_PASS, note


# --- after-run judgement -----------------------------------------------------


@dataclass
class AfterRun:
    """The whole-case checks after both verbs and the restore."""

    other_node_ready: bool
    other_agent_retired: bool
    #: departed ledger lines for the other node that reached CLEANED or REFUSED.
    other_node_ledger_mail: Sequence[str] = field(default_factory=tuple)
    #: "[GPU node departed ...]" notifications for any of the target's nodes
    #: raised during the removal (b249): must be empty.
    target_departed_mail: Sequence[str] = field(default_factory=tuple)
    target_agents_all_active: bool = True
    non_terminal_incident: bool = False
    window_residue: bool = False


def judge_after_run(after: AfterRun, *, window: str) -> tuple[bool, list[str]]:
    """``(ok, reasons)`` for the whole-case after-run checks."""

    reasons: list[str] = []
    if not after.other_node_ready:
        reasons.append("the other cluster's node is not Ready again")
    if after.other_agent_retired:
        reasons.append("the other node's Agent was retired (it must never be)")
    if after.other_node_ledger_mail:
        reasons.append(
            "the departed-node ledger shows mail for the other node: "
            + ", ".join(after.other_node_ledger_mail)
        )
    if after.target_departed_mail:
        reasons.append(
            "a 'GPU node departed' notification went out for the target's "
            "nodes during the removal (b249): " + ", ".join(after.target_departed_mail)
        )
    if not after.target_agents_all_active:
        reasons.append("not every target Agent is ACTIVE again")
    if after.non_terminal_incident:
        reasons.append("a non-terminal incident is open")
    if after.window_residue and window == WINDOW_KUBELET:
        reasons.append("a service-window claim or probe residue is left on the node")
    return (not reasons), reasons


def case_verdict(verb_verdicts: Mapping[str, str], after_ok: bool) -> str:
    """The run's verdict from the per-verb verdicts and the after-run outcome.

    FAIL if any verb failed or the after-run failed; INCONCLUSIVE if a verb was
    inconclusive and nothing failed; PASS only when every verb passed and the
    after-run is clean.
    """

    values = list(verb_verdicts.values())
    if not after_ok or VERDICT_FAIL in values:
        return VERDICT_FAIL
    if VERDICT_INCONCLUSIVE in values:
        return VERDICT_INCONCLUSIVE
    if values and all(value == VERDICT_PASS for value in values):
        return VERDICT_PASS
    return VERDICT_INCONCLUSIVE


# --- reading the product's own records (pure, fed already-parsed documents) --


def departed_mail_for(
    notifications: Sequence[Mapping[str, Any]], node_ids: Sequence[str]
) -> list[str]:
    """Subjects of departed-node notifications naming any of ``node_ids``.

    A notification is one of the departed-node cleanup's two mails
    (``DEPARTED_SUBJECTS``) and its ``incident_id`` is ``departed-node-<node>``
    or its subject ends with the node name.
    """

    names = {str(node) for node in node_ids}
    hits: list[str] = []
    for item in notifications:
        subject = str(item.get("subject") or "")
        if not any(subject.startswith(prefix) for prefix in DEPARTED_SUBJECTS):
            continue
        incident = str(item.get("incident_id") or "")
        named = {
            node
            for node in names
            if incident == f"departed-node-{node}" or subject.rstrip().endswith(node)
        }
        if named:
            hits.append(f"{subject} ({', '.join(sorted(named))})")
    return hits


def ledger_mail_for(ledger: Sequence[Mapping[str, Any]], node_id: str) -> list[str]:
    """Departed-ledger lines for ``node_id`` that reached CLEANED or REFUSED.

    A node whose Agent lease stayed live is never a cleanup candidate, so the
    ledger must hold no mailing line for it. PENDING without a mail is not one.
    """

    hits: list[str] = []
    for item in ledger:
        if str(item.get("node_id")) != node_id:
            continue
        status = str(item.get("status") or "")
        if status in {"CLEANED", "REFUSED"}:
            hits.append(f"{node_id}: {status}")
    return hits


def agents_all_active(
    agents: Sequence[Mapping[str, Any]], *, expected: int | None = None
) -> bool:
    """Every Agent ACTIVE (and, when given, exactly ``expected`` of them)."""

    states = [str(item.get("lifecycle_state")) for item in agents]
    if expected is not None and len(states) != expected:
        return False
    return bool(states) and all(state == "ACTIVE" for state in states)


def agent_for_node(
    agents: Sequence[Mapping[str, Any]], node_id: str
) -> Mapping[str, Any] | None:
    return next((item for item in agents if str(item.get("node_id")) == node_id), None)


def other_agent_retired(agents: Sequence[Mapping[str, Any]], node_id: str) -> bool:
    """The other node's Agent is gone or REVOKED -- it must never be retired."""

    agent = agent_for_node(agents, node_id)
    if agent is None:
        return True
    return str(agent.get("lifecycle_state")) == "REVOKED"


# --- the product-generation probe (step 2 guard) -----------------------------

#: The import the deployed interpreter runs to tell whether b250 shipped: the
#: scoped verify and its findings record live in these modules, so a product
#: predating b250 cannot import them and the case is NOT_APPLICABLE.
GENERATION_PROBE = (
    "import json\n"
    "try:\n"
    "    from gpu_fault.admin import membership_verify as m\n"
    "    from gpu_fault_release import regional_verify_scope as s\n"
    "    ok = hasattr(m, 'verify_arguments') and hasattr(m, 'RECORD_FILE') "
    "and hasattr(s, 'VerifyScope')\n"
    "    print(json.dumps({'b250': bool(ok), 'record': str(getattr(m, 'RECORD_FILE', ''))}))\n"
    "except Exception as exc:\n"
    "    print(json.dumps({'b250': False, 'error': type(exc).__name__}))\n"
)


def generation_is_b250(probe_stdout: str) -> bool:
    """Read the generation probe's last JSON line; True only on a clean b250."""

    for line in reversed(probe_stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except ValueError:
            return False
        return isinstance(value, dict) and value.get("b250") is True
    return False


# --- small record readers the runner uses (pure) -----------------------------


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def verified_state_for(
    state_dir: Path, verb: str, cluster_id: str
) -> dict[str, Any] | None:
    """The verb's state file carrying ``evidence.VERIFIED`` for ``cluster_id``.

    Removal: ``remove-cluster/*/state.json`` whose ``cluster_id`` matches and
    phase is COMPLETED. Join: ``join-cluster/*/state.json`` whose
    ``requested_cluster_id`` (or ``VERIFIED.cluster_id``) matches.
    """

    if verb == REMOVE_VERB:
        root = state_dir / "remove-cluster"
        key = "cluster_id"
    else:
        root = state_dir / "join-cluster"
        key = "requested_cluster_id"
    if not root.is_dir():
        return None
    for path in sorted(root.glob("*/state.json")):
        value = load_json(path)
        if value is None:
            continue
        verified = (value.get("evidence") or {}).get("VERIFIED") or {}
        names = {str(value.get(key)), str(verified.get("cluster_id"))}
        if cluster_id in names:
            return value
    return None


def findings_record_path(state_dir: Path) -> Path:
    return state_dir / "membership-verify" / "findings.json"


# --- the run's control flow (pure over an injected driver) -------------------


class MembershipDriver(Protocol):
    """What the run needs from the live side; a fake supplies canned evidence.

    ``run_verb`` spawns the admin verb, follows its output, drives the window
    (open on the trigger, confirm NotReady before the verify, restore at the
    verify's end -- or ``resume_cleanup`` on a failed restore) and reads the
    four finding surfaces, returning the verb's :class:`VerbEvidence`.
    ``end_windows`` is called on every exit path and must leave no window open.
    """

    def b250_generation(self) -> bool: ...

    def run_verb(self, verb: str) -> VerbEvidence: ...

    def read_after(self) -> AfterRun: ...

    def end_windows(self) -> dict[str, Any]: ...


def _cluster_left(evidence: VerbEvidence) -> bool:
    """Whether the removal actually took the cluster out (so the join may run)."""

    return (
        evidence.verb == REMOVE_VERB
        and evidence.exit_code == 0
        and not evidence.saw_rollback_cluster
    )


def run_membership(driver: MembershipDriver, *, window: str) -> dict[str, Any]:
    """Drive remove then re-join, judging each verb and the after-run.

    Returns the run record: the per-verb verdicts and reasons, the after-run
    outcome, the overall verdict and whether anything was mutated. The window
    is ended on every exit path before the record is returned. On a product
    without b250 nothing is spawned and the verdict is ``NOT_APPLICABLE``.
    """

    record: dict[str, Any] = {
        "verbs": [],
        "after_run": None,
        "verdict": VERDICT_NOT_APPLICABLE,
        "mutation_performed": False,
        "window_residue": None,
    }
    if not driver.b250_generation():
        record["reason"] = (
            "the deployed product predates b250 (no cluster-scoped membership "
            "verify / findings record); nothing was removed, joined or stopped"
        )
        return record

    verb_verdicts: dict[str, str] = {}
    removed = False
    try:
        for verb in VERBS:
            if verb == JOIN_VERB and not removed:
                record["verbs"].append(
                    {
                        "verb": verb,
                        "verdict": VERDICT_NOT_RUN,
                        "reasons": [
                            "the removal did not take the cluster out, so the "
                            "re-join was not attempted"
                        ],
                    }
                )
                continue
            evidence = driver.run_verb(verb)
            record["mutation_performed"] = True
            verdict, reasons = judge_verb(evidence)
            verb_verdicts[verb] = verdict
            record["verbs"].append(
                {
                    "verb": verb,
                    "window": evidence.window,
                    "verdict": verdict,
                    "reasons": reasons,
                    "exit_code": evidence.exit_code,
                    "window_opened": evidence.window_opened,
                    "other_node_not_ready_before_verify": (
                        evidence.other_node_not_ready_before_verify
                    ),
                    "finding_surfaces": dict(evidence.finding_surfaces),
                    "status_error": evidence.status_error,
                    "registry_convergence_pass": evidence.registry_convergence_pass,
                    "restore_ok": evidence.restore_ok,
                    "resume_cleanup_called": evidence.resume_cleanup_called,
                    "window_observation": evidence.window_observation,
                    "window_timeline": dict(evidence.window_timeline),
                }
            )
            if verb == REMOVE_VERB:
                removed = _cluster_left(evidence)
        after = driver.read_after()
        after_ok, after_reasons = judge_after_run(after, window=window)
        record["after_run"] = {
            "ok": after_ok,
            "reasons": after_reasons,
            "other_node_ready": after.other_node_ready,
            "other_agent_retired": after.other_agent_retired,
            "other_node_ledger_mail": list(after.other_node_ledger_mail),
            "target_departed_mail": list(after.target_departed_mail),
            "target_agents_all_active": after.target_agents_all_active,
            "non_terminal_incident": after.non_terminal_incident,
        }
        record["verdict"] = case_verdict(verb_verdicts, after_ok)
    finally:
        record["window_residue"] = driver.end_windows()
    return record
