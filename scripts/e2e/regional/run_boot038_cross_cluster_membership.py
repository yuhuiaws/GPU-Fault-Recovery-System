#!/usr/bin/env python3
"""GF-REGIONAL-BOOT-038: a membership verb on one GPU cluster while another is NotReady.

``remove-cluster`` then ``join-cluster`` of one GPU cluster (the target), each
while ONE GPU node of a different cluster is held NotReady with its Agent still
ACTIVE -- the live 2026-11-04 12:20Z state that rolled a join back. The case
proves the R19 fixes:

* **b250** -- each verb's final ``verify`` is scoped to its target, so the other
  cluster's coverage drift is a WARN finding recorded (not gating) on four
  surfaces (the verb result, the ``VERIFIED`` evidence, ``<state-dir>/
  membership-verify/findings.json`` and the ``status`` finding lines), never a
  rollback;
* **b250b** -- the join's preflight, ``join-cluster`` mode, ``workflow_safety``
  and the shared ``control_api`` halves are scoped the same way (exercised on
  the join);
* **b249** -- no "GPU node departed" mail goes out for the target's own nodes
  during the removal.

The other node is held NotReady by a bounded, self-restoring ``kubelet.service``
stop through the existing DESTR-008 service window
(``warm_spare_fixture.WarmSpareServiceFixture`` over
``probes/warm_spare_node_probe.py``) -- plan A (``--window kubelet``). Plan B
(``--window none``) runs the same remove/re-join opportunistically against a
real departure (an EKS node-group roll) and records what each verify saw.

The premise is settled in ``boot038_membership`` (step 1 of the brief): the
service window is invisible to the product, so an ordinary claimed stop
reproduces an unplanned loss and keeps the node-side recovery safety; the
verify and the departed-node cleanup read only Kubernetes, the fleet and the
provider, never the window's claim. No live access here: the main session runs
the case; this driver spawns the verb and never kills it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import boot029_node_scope as node_scope  # noqa: E402
from scripts.e2e.regional import boot029_ownership as ownership  # noqa: E402
from scripts.e2e.regional import boot029_registry_drift as registry  # noqa: E402
from scripts.e2e.regional import boot038_membership as membership  # noqa: E402
from scripts.e2e.regional import eks_nodegroup_case as case  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.admin_cli import admin_command  # noqa: E402
from scripts.e2e.regional.boot029_receipts import (  # noqa: E402
    pending_profile_plan,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    run_standard_case,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    run_case_main,
)
from scripts.e2e.regional.service_window_node_restore import (  # noqa: E402
    NodeRestoreError,
    open_windows,
    restore_through_node,
)

CASE_ID = membership.CASE_ID
CONFIRMATION = membership.CONFIRMATION


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


#: Wall clock; the restore/evidence timestamps read it (tests replace it).
CLOCK_NOW = _utc_now
#: The on-node restore unit's authority is restore_at + RECOVERY_SECONDS; a
#: node that does not return by then plus a margin fails the restore.
RESTORE_READY_TIMEOUT_SECONDS = (
    membership.RESTORE_SECONDS + membership.STOP_DELAY_SECONDS + 300
)
VERB_TIMEOUT_SECONDS = 1800.0


# --- settings and site lookup ------------------------------------------------


@dataclass(frozen=True)
class Settings:
    state_dir: Path
    target_cluster_arn: str
    target_cluster_id: str
    other_cluster_id: str
    other_node: str
    window: str
    dry_run: bool
    regional: RegionalLiveSettings
    host_probe_image: str
    cpu_cluster_arn: str
    predecessor_id: str | None
    predecessor_path: Path | None
    run_token: str

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_STATE_DIR": str(self.state_dir),
        }


def _site_document(state_dir: Path) -> dict[str, Any]:
    document = membership.load_json(state_dir / "site.yaml")
    if document is None:
        # site.yaml is YAML, not JSON; read it with the yaml the rest uses.
        import yaml  # type: ignore[import-untyped,unused-ignore]

        try:
            document = yaml.safe_load((state_dir / "site.yaml").read_text("utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise RegionalFixtureError(f"cannot read {state_dir}/site.yaml: {exc}")
    if not isinstance(document, dict) or document.get("kind") != "RegionalSite":
        raise RegionalFixtureError(f"{state_dir}/site.yaml is not a RegionalSite")
    return document


def _clusters(document: dict[str, Any]) -> list[dict[str, Any]]:
    spec = document.get("spec")
    spec = spec if isinstance(spec, dict) else {}
    return [item for item in spec.get("clusters") or () if isinstance(item, dict)]


def _cluster_entry(document: dict[str, Any], cluster_id: str) -> dict[str, Any]:
    for item in _clusters(document):
        if str(item.get("clusterId")) == cluster_id:
            return item
    raise RegionalFixtureError(f"site names no member cluster {cluster_id}")


def _cluster_for_arn(document: dict[str, Any], arn: str) -> str:
    for item in _clusters(document):
        if str(item.get("eksClusterArn")) == arn:
            return str(item.get("clusterId"))
    raise RegionalFixtureError(f"site names no member cluster with EKS ARN {arn}")


def configure(arguments: argparse.Namespace) -> Settings:
    state_dir = Path(arguments.state_dir or "").expanduser()
    if not str(arguments.state_dir or "").strip():
        raise RegionalFixtureError(
            "--state-dir (the site's state directory) is required"
        )
    state_dir = state_dir.resolve()
    if arguments.window not in membership.WINDOWS:
        raise RegionalFixtureError(
            f"--window must be one of {', '.join(membership.WINDOWS)}"
        )
    target_arn = str(arguments.target_cluster_arn or "").strip()
    other_id = str(arguments.other_cluster_id or "").strip()
    other_node = str(arguments.other_node or "").strip()
    if not target_arn.startswith("arn:") or "/" not in target_arn:
        raise RegionalFixtureError("--target-cluster-arn must be a GPU cluster EKS ARN")
    if not other_id:
        raise RegionalFixtureError("--other-cluster-id is required")
    if arguments.window == membership.WINDOW_KUBELET and not other_node:
        raise RegionalFixtureError("--other-node is required for --window kubelet")
    document = _site_document(state_dir)
    spec = document.get("spec") or {}
    cpu = spec.get("cpu") or {}
    target_cluster_id = _cluster_for_arn(document, target_arn)
    if target_cluster_id == other_id:
        raise RegionalFixtureError(
            "--other-cluster-id must differ from the target cluster"
        )
    other = _cluster_entry(document, other_id)
    # The texts are checked, not Paths: Path("") is Path("."), which is truthy.
    gpu_kubeconfig = case._text(spec.get("gpuKubeconfig"))  # noqa: SLF001 - shared text reader
    cpu_kubeconfig = case._text(cpu.get("kubeconfig"))  # noqa: SLF001
    region = case._text(spec.get("awsRegion"))  # noqa: SLF001
    namespace = case._text(spec.get("namespace")) or "gpu-fault-system"  # noqa: SLF001
    other_context = case._text(other.get("context"))  # noqa: SLF001
    if not (gpu_kubeconfig and cpu_kubeconfig and region and other_context):
        raise RegionalFixtureError(
            "site.yaml is missing the GPU/CPU kubeconfig, Region or the other "
            "cluster's context"
        )
    regional = RegionalLiveSettings(
        cpu_kubeconfig=Path(cpu_kubeconfig).expanduser().resolve(),
        gpu_kubeconfig=Path(gpu_kubeconfig).expanduser().resolve(),
        gpu_context=other_context,
        namespace=namespace,
        cluster_id=other_id,
        region=region,
    )
    run_dir = Path(arguments.run_dir).expanduser().resolve()
    previous, path = predecessor_path(run_dir, CASE_ID, arguments.predecessor_evidence)
    return Settings(
        state_dir=state_dir,
        target_cluster_arn=target_arn,
        target_cluster_id=target_cluster_id,
        other_cluster_id=other_id,
        other_node=other_node,
        window=arguments.window,
        dry_run=bool(getattr(arguments, "dry_run", False)),
        regional=regional,
        host_probe_image=_host_probe_image(),
        cpu_cluster_arn=case._text(cpu.get("eksArn")),  # noqa: SLF001
        predecessor_id=previous,
        predecessor_path=path,
        run_token=hashlib.sha256(str(run_dir).encode()).hexdigest()[:8],
    )


def _host_probe_image() -> str:
    """The acceptance host-probe OCI image from the installed site profile."""

    image = str(case.profile_arguments().get("host-probe-image") or "").strip()
    if not image:
        raise RegionalFixtureError(
            "the site profile names no host-probe-image for the service window"
        )
    return image


# --- timings and command lines (the plan) ------------------------------------

TIMINGS = {
    "node_not_ready_after_kubelet_stop_seconds": ">60-80 on this site; taint ~50-90",
    "removal": {
        "remove_cluster_mode_seconds": "~113",
        "gap_before_sync_state_seconds": "~57",
        "sync_state_seconds": "~50",
        "verify_starts_after_remove_begin_seconds": "~220",
        "verify_seconds": "~33",
        "window_opens_on": "release-begin mode=remove-cluster",
    },
    "join": {
        "preflight_seconds": 20,
        "join_cluster_mode_minutes": "7.7-9",
        "verify_starts_after_last_wave_start_seconds": "160-234",
        "verify_seconds": "~50",
        "window_opens_on": "the last fleet wave's start line (fleet-wave ... wave=M/M)",
    },
    "window": {
        "stop_delay_seconds": membership.STOP_DELAY_SECONDS,
        "restore_seconds": membership.RESTORE_SECONDS,
        "stop_to_restore_cap_seconds": membership.RESTORE_CAP_SECONDS,
        "eviction_after_taint_seconds": 300,
        "node_side_restore_is_the_backstop": True,
    },
}


def verb_command(settings: Settings, verb: str) -> list[str]:
    command = [
        *admin_command(settings.state_dir),
        verb,
        "--state-dir",
        str(settings.state_dir),
        "--gpu-cluster-arn",
        settings.target_cluster_arn,
    ]
    if verb == membership.REMOVE_VERB:
        command += ["--confirm", "REMOVE_GPU_CLUSTER"]
    return command


def status_command(state_dir: Path) -> list[str]:
    """``gpu-fault-admin status --full --state-dir <S>``.

    ``--state-dir`` is required: the installed deploy-host CLI refuses (rc 2,
    "status requires that managed state") without it, so a status without it
    read as "no finding" (live 2026-11-04).
    """

    return [
        *admin_command(state_dir),
        "status",
        "--full",
        "--state-dir",
        str(state_dir),
    ]


def status_findings(returncode: int, stderr: str) -> tuple[list[str], str | None]:
    """The ``not gating`` finding lines ``status`` printed, and an error.

    A non-zero rc is returned as an evidence error, never swallowed as an empty
    finding set -- the status surface is then indeterminate, not "no finding".
    """

    if returncode != 0:
        return [], f"status --full exited {returncode}"
    lines = [
        line
        for line in stderr.splitlines()
        if "gpu-fault-admin status:" in line and "not gating:" in line
    ]
    return lines, None


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    preflight.setdefault("preflight_identity", _plan_identity(preflight))
    return {
        "risk": "destructive",
        "premise": (
            "the DESTR-008 service window is invisible to the product (claim is "
            "a node-local file and a harness-side controller journal; the verify "
            "and departed-node cleanup read only Kubernetes, the fleet and the "
            "provider); a kubelet stop with the Agent left running reproduces the "
            "unplanned NotReady-with-ACTIVE-Agent loss, so no unclaimed variant "
            "is needed and the node-side recovery safety is kept"
        ),
        "predecessor": preflight.get("predecessor"),
        "target_cluster": {
            "arn": settings.target_cluster_arn,
            "cluster_id": settings.target_cluster_id,
        },
        "other_cluster": {
            "cluster_id": settings.other_cluster_id,
            "node": settings.other_node,
            "context": settings.regional.gpu_context,
        },
        "window": settings.window,
        "product_generation": preflight.get("product_generation"),
        "mutation": (
            "gpu-fault-admin remove-cluster then join-cluster of the target GPU "
            "cluster (the join re-applies the ownership label the removal "
            "released, from the remove tombstone, as S69 does); each ends in a "
            "scoped verify. "
            + (
                "One GPU node of the other cluster is held NotReady by a bounded, "
                "self-restoring kubelet.service stop (DESTR-008 service window) "
                "around each verb's verify; the node Agent is never touched and "
                "the on-node restore unit is the backstop."
                if settings.window == membership.WINDOW_KUBELET
                else "No window is opened; a real external departure (a node-group "
                "roll) supplies the other cluster's NotReady node and the case "
                "records what each verify saw."
            )
        ),
        "verb_commands": {
            verb: " ".join(verb_command(settings, verb)) for verb in membership.VERBS
        },
        "timings": TIMINGS,
        "stop_conditions": [
            "formal predecessor evidence (BOOT-029) is not PASS",
            "the deployed product predates b250 (NOT_APPLICABLE, no mutation)",
            "the site is not healthy, a non-terminal incident is open, or the "
            "other node is not Ready with an ACTIVE Agent and no workflow/claim",
            "the other cluster does not keep at least one other Ready GPU node",
            "the target's remove/re-join prerequisites are absent (ownership "
            "label, removal tombstone, CLUSTER_PROFILE_NEW approval as in S69)",
            "the window missed (the node did not show NotReady before the verify "
            "began): the verb is judged INCONCLUSIVE, the rest still judged",
        ],
        "verdicts": {
            "per_verb_PASS": [
                "the verb exits 0",
                "no fail-cluster or rollback-cluster",
                "the other cluster's finding is recorded as not gating on all "
                "four surfaces (result, VERIFIED evidence, findings.json, status)",
                "the target's own checks pass and S69 registry convergence holds",
            ],
            "per_verb_INCONCLUSIVE": "the window missed; the rest is still judged",
            "per_verb_FAIL": "a rollback, a non-zero exit, a missing finding, a "
            "lost registry convergence, or a failed restore",
            "after_run": [
                "the other node is Ready again and its Agent was never retired",
                "the departed-node ledger shows no CLEANED/REFUSED mail for it",
                "no 'GPU node departed' mail for the target's nodes during the "
                "removal (b249)",
                "the target's Agents are all ACTIVE, no non-terminal incident, "
                "no service-window residue",
            ],
        },
        "rollback": {
            "a_failed_removal_leaves_the_target_removed": True,
            "the_join_is_not_attempted_unless_the_removal_completed": True,
            "every_exit_path_ends_the_window_first": True,
        },
        "preflight_identity": _plan_identity(preflight),
        "preflight": preflight,
    }


# --- read-only preflight -----------------------------------------------------

_PRECONDITION_PROBE = r"""
import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState, WorkflowStatus

other_cluster, other_node, target_cluster = sys.argv[1:4]
store = ApplicationContext.from_environment().store
OPEN = [
    IncidentState.DETECTED,
    IncidentState.ACTION_PENDING,
    IncidentState.SAFETY_PENDING,
    IncidentState.QUARANTINED,
    IncidentState.ESCALATED,
]


def agents(cluster):
    return [
        {
            "node_id": a.node_id,
            "lifecycle_state": str(getattr(a.lifecycle_state, "value", a.lifecycle_state)),
        }
        for a in store.list_agents(cluster)
    ]


incidents = []
for cluster in (other_cluster, target_cluster):
    for i in store.list_incidents_by_state(cluster, OPEN, limit=200):
        incidents.append({"cluster": cluster, "id": i.incident_id, "state": str(i.state)})
workflows = []
for w in store.list_workflows(
    [WorkflowStatus.PENDING, WorkflowStatus.SAFETY_PENDING, WorkflowStatus.RUNNING],
    limit=500,
):
    try:
        owner = store.get_incident(w.incident_id)
    except Exception:
        owner = None
    if owner is not None and other_node in list(owner.node_ids):
        workflows.append({"request_id": w.request_id, "node": other_node})
print(
    json.dumps(
        {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "open_incidents": incidents,
            "other_agents": agents(other_cluster),
            "target_agents": agents(target_cluster),
            "node_workflows": workflows,
        },
        sort_keys=True,
    )
)
"""


def _product_generation(settings: Settings) -> str:
    """``b250`` or ``pre-b250`` from the deployed interpreter (read-only)."""

    python = Path(settings.state_dir) / "deployer-venv" / "bin" / "python"
    interpreter = str(python) if python.is_file() else sys.executable
    try:
        completed = subprocess.run(
            [interpreter, "-I", "-c", membership.GENERATION_PROBE],
            capture_output=True,
            text=True,
            timeout=120,
            env={"PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "pre-b250"
    return "b250" if membership.generation_is_b250(completed.stdout) else "pre-b250"


def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    result: dict[str, Any] = {"errors": errors, "identity": {}, "predecessor": {}}
    fixture = RegionalLiveFixture(settings.regional)
    try:
        result["identity"] = dict(fixture.evidence_identity())
        result["predecessor"] = case.predecessor_gate(
            settings.predecessor_id, settings.predecessor_path, result["identity"]
        )
        if result["predecessor"].get("valid") is not True:
            errors.append(f"formal predecessor {settings.predecessor_id} is not PASS")
    except Exception as exc:  # noqa: BLE001 - recorded, fails the preflight
        errors.append(f"release identity unavailable: {type(exc).__name__}: {exc}")
    result["product_generation"] = _product_generation(settings)
    if result["product_generation"] != "b250":
        # Not an error: NOT_APPLICABLE is the verdict, and the preflight still
        # passes so a plan prints. Execute reports NOT_APPLICABLE and mutates
        # nothing.
        result["not_applicable"] = True
    try:
        snapshot = fixture.cpu_python(
            _PRECONDITION_PROBE,
            settings.other_cluster_id,
            settings.other_node,
            settings.target_cluster_id,
        )
        result["store"] = snapshot
        if snapshot.get("open_incidents"):
            errors.append(
                "non-terminal incident(s) open: "
                + ", ".join(
                    f"{item['cluster']}/{item['id']}"
                    for item in snapshot["open_incidents"]
                )
            )
        if snapshot.get("node_workflows"):
            errors.append(f"an active workflow names {settings.other_node}")
        if not membership.agents_all_active(snapshot.get("target_agents") or []):
            errors.append("not every target Agent is ACTIVE")
        other_agent = membership.agent_for_node(
            snapshot.get("other_agents") or [], settings.other_node
        )
        if settings.window == membership.WINDOW_KUBELET and (
            other_agent is None or str(other_agent.get("lifecycle_state")) != "ACTIVE"
        ):
            errors.append(f"{settings.other_node} has no ACTIVE Agent")
    except Exception as exc:  # noqa: BLE001 - recorded, fails the preflight
        errors.append(f"store preflight failed: {type(exc).__name__}: {exc}")
    try:
        nodes = fixture.gpu_nodes()
        result["other_gpu_nodes"] = [
            {"name": item["name"], "ready": item["ready"]} for item in nodes
        ]
        ready_other = [
            item
            for item in nodes
            if item["ready"] == "True" and item["name"] != settings.other_node
        ]
        if settings.window == membership.WINDOW_KUBELET:
            held = next(
                (item for item in nodes if item["name"] == settings.other_node), None
            )
            if held is None:
                errors.append(
                    f"{settings.other_node} is not a GPU node of {settings.other_cluster_id}"
                )
            elif held["ready"] != "True":
                errors.append(f"{settings.other_node} is not Ready before the window")
        if not ready_other:
            errors.append(
                f"{settings.other_cluster_id} keeps no other Ready GPU node; one "
                "node is not enough: the held node must leave a Ready peer so the "
                "cluster is not fully down and the drift is the single held node"
            )
    except Exception as exc:  # noqa: BLE001 - recorded, fails the preflight
        errors.append(f"node preflight failed: {type(exc).__name__}: {exc}")
    # A stale service-window claim left on the node blocks a new window.
    residue = open_windows(case_dir / "service-window", settings.other_node)
    if residue:
        errors.append("a stale service-window journal is open: " + "; ".join(residue))
    write_json_atomic(case_dir / "preflight.json", result)
    return result


# --- the live driver ---------------------------------------------------------


@dataclass
class _Follower:
    """A thread that drains the verb's merged output into a queue of lines."""

    process: subprocess.Popen[str]
    lines: list[str] = field(default_factory=list)
    _queue: queue.Queue[str | None] = field(default_factory=queue.Queue)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        # stderr was merged into stdout (narration and the result JSON share one
        # stream), so the follower drains stdout.
        stream = self.process.stdout
        assert stream is not None
        for line in stream:
            self._queue.put(line)
        self._queue.put(None)

    def __iter__(self) -> Iterator[str]:
        while True:
            line = self._queue.get()
            if line is None:
                return
            self.lines.append(line)
            yield line


class LiveMembershipDriver:
    """Spawns the admin verb, follows its output and drives the kubelet window.

    Not unit-tested here (no live access); the verdict logic it feeds lives in
    ``boot038_membership`` and is tested with a fake driver.
    """

    def __init__(self, settings: Settings, case_dir: Path, deadline: datetime) -> None:
        self.settings = settings
        self.case_dir = case_dir
        self.deadline = deadline
        self.fixture = RegionalLiveFixture(settings.regional)
        from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture

        self.warm = WarmSpareLiveFixture(
            self.fixture,
            hyperpod_cluster=settings.other_cluster_id,
            provider="eks-managed",
        )
        self._windows: list[Any] = []
        self._baseline_notifications: list[dict[str, Any]] = []

    # -- generation --------------------------------------------------------
    def b250_generation(self) -> bool:
        return _product_generation(self.settings) == "b250"

    # -- one verb ----------------------------------------------------------
    def run_verb(self, verb: str) -> membership.VerbEvidence:
        self._snapshot_notifications(verb)
        if verb == membership.JOIN_VERB:
            self._reapply_ownership_label()
        baseline = self.case_dir / f"{verb}.registry-baseline.json"
        self._registry_baseline(verb, baseline)
        window = (
            self._new_window(verb)
            if self.settings.window == membership.WINDOW_KUBELET
            else None
        )
        not_ready_before = None if window is None else False
        timeline: dict[str, str] = {}
        process = subprocess.Popen(  # noqa: S603 - the site's own deploy-host CLI
            self._verb_command(verb),
            cwd=self.settings.state_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        follower = _Follower(process)
        follower.start()
        narration = membership.VerbNarration()
        poller: NotReadyPoller | None = None
        restore_ok = True
        resume_called = False
        try:
            for line in follower:
                token = narration.observe(verb, line)
                if token == "open" and window is not None and poller is None:
                    self._open_window(window)
                    timeline["stop"] = CLOCK_NOW().isoformat()
                    # Poll from the window open so the verify-begin read is of a
                    # state already reached, not a grace that waits for it.
                    poller = NotReadyPoller(self.warm, self.settings.other_node)
                    poller.start()
                elif token == "verify-begin":
                    timeline["verify_begin"] = CLOCK_NOW().isoformat()
                    if poller is not None:
                        # Read at once, no wait: already NotReady and still so.
                        not_ready_before = poller.not_ready_now()
                        if poller.not_ready_seen_at is not None:
                            timeline["not_ready_seen"] = poller.not_ready_seen_at
                elif token == "verify-end" and window is not None:
                    try:
                        self._restore_window(window)
                    except NodeRestoreError:
                        restore_ok = False
                        resume_called = True
                        self._resume(window)
                    timeline["restore"] = CLOCK_NOW().isoformat()
            process.wait(timeout=VERB_TIMEOUT_SECONDS)
        finally:
            if poller is not None:
                poller.stop()
                if poller.not_ready_seen_at is not None:
                    timeline.setdefault("not_ready_seen", poller.not_ready_seen_at)
        result_json = membership.verb_result_from_output(follower.lines)
        if verb == membership.JOIN_VERB and process.returncode != 0:
            result_json, process_rc = self._answer_profile_plan(verb, result_json)
        else:
            process_rc = process.returncode
        records = self._read_surfaces(verb)
        convergence = self._registry_check(verb, baseline)
        return membership.VerbEvidence(
            verb=verb,
            window=self.settings.window,
            exit_code=int(process_rc),
            saw_fail_cluster=narration.saw_fail_cluster,
            saw_rollback_cluster=narration.saw_rollback_cluster,
            window_opened=window is not None and poller is not None,
            other_node_not_ready_before_verify=not_ready_before,
            finding_surfaces=membership.finding_surfaces(
                other_cluster_id=self.settings.other_cluster_id,
                result=result_json,
                verified_state=records["verified_state"],
                record=records["record"],
                status_lines=records["status_lines"],
            ),
            status_error=records["status_error"],
            registry_convergence_pass=convergence,
            restore_ok=restore_ok,
            resume_cleanup_called=resume_called,
            window_observation=("" if window is not None else self._observe_node()),
            window_timeline=timeline,
        )

    # -- after the run -----------------------------------------------------
    def read_after(self) -> membership.AfterRun:
        snapshot = self.fixture.cpu_python(
            _AFTER_PROBE,
            self.settings.other_cluster_id,
            self.settings.other_node,
            self.settings.target_cluster_id,
        )
        node = self.warm.node_snapshot(self.settings.other_node)
        target_nodes = [item["node_id"] for item in snapshot.get("target_agents") or []]
        target_mail = membership.departed_mail_for(
            snapshot.get("notifications") or [], target_nodes
        )
        baseline_subjects = {
            item.get("subject") for item in self._baseline_notifications
        }
        target_mail = [
            line for line in target_mail if line.split(" (")[0] not in baseline_subjects
        ]
        return membership.AfterRun(
            other_node_ready=node.get("ready") == "True",
            other_agent_retired=membership.other_agent_retired(
                snapshot.get("other_agents") or [], self.settings.other_node
            ),
            other_node_ledger_mail=membership.ledger_mail_for(
                snapshot.get("departed_ledger") or [], self.settings.other_node
            ),
            target_departed_mail=target_mail,
            target_agents_all_active=membership.agents_all_active(
                snapshot.get("target_agents") or []
            ),
            non_terminal_incident=bool(snapshot.get("open_incidents")),
            window_residue=bool(
                open_windows(self.case_dir / "service-window", self.settings.other_node)
            ),
        )

    def end_windows(self) -> dict[str, Any]:
        residue: dict[str, Any] = {"ended": [], "resume_cleanup": []}
        for window in self._windows:
            try:
                residue["ended"].append(window.cleanup())
            except Exception:  # noqa: BLE001 - try resume on any failure
                try:
                    residue["resume_cleanup"].append(window.resume_cleanup())
                except Exception as exc:  # noqa: BLE001 - recorded, not raised
                    residue["resume_cleanup"].append(
                        {"error": f"{type(exc).__name__}: {exc}"}
                    )
            finally:
                window.close()
        residue["open_journals"] = open_windows(
            self.case_dir / "service-window", self.settings.other_node
        )
        return residue

    # -- window helpers ----------------------------------------------------
    def _new_window(self, verb: str) -> Any:
        from scripts.e2e.regional.warm_spare_fixture import WarmSpareServiceFixture

        run_id = f"boot038-{self.settings.run_token}-{verb}"
        plan_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "case": membership.SERVICE_WINDOW_FIXTURE_CASE_ID,
                    "run_id": run_id,
                    "node": self.settings.other_node,
                    "service": membership.KUBELET_SERVICE,
                    "intent": "cross-cluster-membership-window",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        window = WarmSpareServiceFixture(
            self.warm,
            node=self.settings.other_node,
            image=self.settings.host_probe_image,
            case_id=membership.SERVICE_WINDOW_FIXTURE_CASE_ID,
            run_id=run_id,
            state_directory=self.case_dir / "service-window",
            node_uid=self.warm.node_snapshot(self.settings.other_node).get("uid", ""),
            plan_sha256=plan_sha256,
            release_id=self.fixture.release_id(),
            maintenance_expires_at=self.deadline,
        )
        self._windows.append(window)
        return window

    def _open_window(self, window: Any) -> None:
        window.create()
        window.stop(
            membership.KUBELET_SERVICE,
            restore_seconds=membership.RESTORE_SECONDS,
            delay_seconds=membership.STOP_DELAY_SECONDS,
        )

    def _restore_window(self, window: Any) -> dict[str, Any]:
        return restore_through_node(
            service=membership.KUBELET_SERVICE,
            node=self.settings.other_node,
            undo=window.cleanup,
            node_snapshot=lambda: self.warm.node_snapshot(self.settings.other_node),
            binding=window.window.data.get("binding"),
            clock=CLOCK_NOW,
            sleep=time.sleep,
            ready_timeout_seconds=RESTORE_READY_TIMEOUT_SECONDS,
        )

    def _resume(self, window: Any) -> None:
        window.resume_cleanup()

    # -- membership prerequisites ------------------------------------------
    def _reapply_ownership_label(self) -> None:
        """Put the ownership label back on the target's node groups before the
        join, from its COMPLETED remove-cluster tombstone (stage 3, as the S69
        driver did). remove-cluster released the label; join-cluster's own
        node-scope precondition refuses without it. Fail closed."""

        try:
            record, _lines = ownership.reapply_report(
                self.settings.cpu_cluster_arn,
                [self.settings.target_cluster_arn],
                state=self.settings.state_dir,
                stage=3,
            )
            scope, _scope_lines = node_scope.precondition_report(
                self.settings.cpu_cluster_arn, [self.settings.target_cluster_arn]
            )
        except Exception as exc:  # noqa: BLE001 - recorded, fails closed
            raise RegionalFixtureError(
                "re-applying the ownership label before the join failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        write_json_atomic(
            self.case_dir / "join-ownership-reapply.json",
            {"reapply": record, "node_scope_precondition": scope},
        )
        refused = [
            item
            for item in scope.get("clusters", [])
            if item.get("status") == "refused"
        ]
        if refused:
            raise RegionalFixtureError(
                "the target's node groups do not carry the ownership label after "
                "re-apply; join-cluster would be refused"
            )

    def _observe_node(self) -> str:
        try:
            node = self.warm.node_snapshot(self.settings.other_node)
        except Exception as exc:  # noqa: BLE001 - recorded only
            return f"unreadable ({type(exc).__name__})"
        return f"{self.settings.other_node} ready={node.get('ready')}"

    # -- verb plumbing -----------------------------------------------------
    def _verb_command(self, verb: str) -> list[str]:
        return verb_command(self.settings, verb)

    def _answer_profile_plan(
        self, verb: str, result_json: dict[str, Any] | None
    ) -> tuple[dict[str, Any] | None, int]:
        """Approve a CLUSTER_PROFILE_NEW plan and rerun the join (S69)."""

        plan = pending_profile_plan(self.settings.state_dir)
        if not plan:
            return result_json, 1
        command = [
            *self._verb_command(verb),
            "--approve-profile-plan",
            str(plan["plan_sha256"]),
            "--reference",
            f"{CASE_ID}/{self.settings.run_token}",
        ]
        completed = subprocess.run(  # noqa: S603 - the site's own CLI
            command,
            cwd=self.settings.state_dir,
            capture_output=True,
            text=True,
            timeout=VERB_TIMEOUT_SECONDS,
            check=False,
        )
        return membership.verb_result_from_output(
            completed.stdout.splitlines(keepends=True)
        ), int(completed.returncode)

    def _read_surfaces(self, verb: str) -> dict[str, Any]:
        record = membership.load_json(
            membership.findings_record_path(self.settings.state_dir)
        )
        verified_state = membership.verified_state_for(
            self.settings.state_dir, verb, self.settings.target_cluster_id
        )
        status_lines, status_error = self._status_finding_lines()
        return {
            "record": record,
            "verified_state": verified_state,
            "status_lines": status_lines,
            "status_error": status_error,
        }

    def _status_finding_lines(self) -> tuple[list[str], str | None]:
        """The ``not gating`` finding lines ``status --full`` prints, and an
        error when the call did not succeed."""

        command = status_command(self.settings.state_dir)
        try:
            completed = subprocess.run(  # noqa: S603 - the site's own CLI
                command,
                cwd=self.settings.state_dir,
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return [], f"status --full could not run: {type(exc).__name__}"
        return status_findings(completed.returncode, completed.stderr)

    def _registry_baseline(self, verb: str, out: Path) -> None:
        try:
            control = registry.kubectl_control(
                *registry.site_cpu_target(self.settings.state_dir)
            )
            roles = registry.cpu_inventory(control)
            write_json_atomic(
                out, {"schema_version": registry.BASELINE_SCHEMA, "roles": roles}
            )
        except Exception:  # noqa: BLE001 - the check fails closed if unread
            pass

    def _registry_check(self, verb: str, baseline: Path) -> bool:
        document = membership.load_json(baseline)
        if document is None:
            return False
        try:
            control = registry.kubectl_control(
                *registry.site_cpu_target(self.settings.state_dir)
            )
            verdict = registry.check_operation(
                control, baseline=document["roles"], operation=verb
            )
        except Exception:  # noqa: BLE001 - fail closed
            return False
        write_json_atomic(self.case_dir / f"{verb}.registry-check.json", verdict)
        return verdict.get("verdict") in {registry.PASS, registry.NOT_APPLICABLE}

    def _snapshot_notifications(self, verb: str) -> None:
        if verb != membership.REMOVE_VERB:
            return
        try:
            snapshot = self.fixture.cpu_python(
                _NOTIFICATIONS_PROBE, self.settings.target_cluster_id
            )
            self._baseline_notifications = list(snapshot.get("notifications") or [])
        except Exception:  # noqa: BLE001 - an unread baseline just widens the diff
            self._baseline_notifications = []


class NotReadyPoller:
    """Polls the other node's Ready condition from the window open.

    It runs from the window open (not from verify-begin), recording the first
    time the node left Ready (``not_ready_seen_at``) and the most recent
    reading. ``not_ready_now`` is read at verify-begin with no wait: the window
    was hit only when the node was already NotReady and still is -- a 90 s grace
    after verify-begin would call a missed window a hit (live 2026-11-04).
    """

    def __init__(self, warm: Any, node: str) -> None:
        self.warm = warm
        self.node = node
        self.not_ready_seen_at: str | None = None
        self._last_ready: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _read(self) -> str | None:
        try:
            return str(self.warm.node_snapshot(self.node).get("ready"))
        except Exception:  # noqa: BLE001 - a failed read is retried
            return None

    def _record(self, ready: str | None) -> None:
        with self._lock:
            self._last_ready = ready
            if ready is not None and ready != "True" and self.not_ready_seen_at is None:
                self.not_ready_seen_at = CLOCK_NOW().isoformat()

    def _poll(self) -> None:
        while not self._stop.is_set():
            self._record(self._read())
            self._stop.wait(membership.NOT_READY_POLL_SECONDS)

    def not_ready_now(self) -> bool:
        """At verify-begin, with no wait: NotReady was seen and the node is
        still not Ready. One fresh read decides "still", falling back to the
        last poll when the fresh read fails."""

        fresh = self._read()
        self._record(fresh)
        with self._lock:
            current = fresh if fresh is not None else self._last_ready
            return self.not_ready_seen_at is not None and current not in (None, "True")

    def stop(self) -> None:
        self._stop.set()


_AFTER_PROBE = r"""
import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState

other_cluster, other_node, target_cluster = sys.argv[1:4]
store = ApplicationContext.from_environment().store
OPEN = [
    IncidentState.DETECTED,
    IncidentState.ACTION_PENDING,
    IncidentState.SAFETY_PENDING,
    IncidentState.QUARANTINED,
    IncidentState.ESCALATED,
]


def agents(cluster):
    return [
        {"node_id": a.node_id, "lifecycle_state": str(getattr(a.lifecycle_state, "value", a.lifecycle_state))}
        for a in store.list_agents(cluster)
    ]


incidents = []
for cluster in (other_cluster, target_cluster):
    for i in store.list_incidents_by_state(cluster, OPEN, limit=200):
        incidents.append({"cluster": cluster, "id": i.incident_id})
ledger = [
    {"node_id": r.node_id, "status": str(getattr(r.status, "value", r.status))}
    for r in store.list_departed_node_cleanups(other_cluster)
]
notifications = [
    {"subject": n.subject, "incident_id": n.incident_id, "cluster_name": n.cluster_name}
    for n in store.list_notifications(limit=500, newest_first=True)
]
print(
    json.dumps(
        {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "open_incidents": incidents,
            "other_agents": agents(other_cluster),
            "target_agents": agents(target_cluster),
            "departed_ledger": ledger,
            "notifications": notifications,
        },
        sort_keys=True,
    )
)
"""

_NOTIFICATIONS_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

store = ApplicationContext.from_environment().store
notifications = [
    {"subject": n.subject, "incident_id": n.incident_id}
    for n in store.list_notifications(limit=500, newest_first=True)
]
print(json.dumps({"notifications": notifications}, sort_keys=True))
"""


# --- execute -----------------------------------------------------------------


def execute_case(
    settings: Settings, run_dir: Path, attempt: int, deadline: datetime
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    evidence = case_dir / f"{CASE_ID}.json"
    started = CLOCK_NOW()
    result: dict[str, Any] = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": membership.VERDICT_FAIL,
        "started_at": started.isoformat(),
        "maintenance_window_end": deadline.isoformat(),
        "window": settings.window,
        "target_cluster_id": settings.target_cluster_id,
        "other_cluster_id": settings.other_cluster_id,
        "mutation_performed": False,
    }
    write_json_atomic(evidence, {**result, "stage": "preflight"})
    preflight = read_only_preflight(settings, case_dir)
    result.update(preflight.get("identity") or {})
    result["predecessor"] = preflight.get("predecessor")
    result["product_generation"] = preflight.get("product_generation")
    interrupted: BaseException | None = None
    try:
        if preflight["errors"]:
            raise RegionalFixtureError(
                "preflight failed: " + "; ".join(preflight["errors"])
            )
        case.verify_plan_identity(case_dir, CASE_ID, _plan_identity(preflight))
        driver = LiveMembershipDriver(settings, case_dir, deadline)
        run = membership.run_membership(driver, window=settings.window)
        result.update(run)
    except BaseException as exc:
        result["verdict"] = membership.VERDICT_FAIL
        result["error"] = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, Exception):
            interrupted = exc
    result["executed_at"] = CLOCK_NOW().isoformat()
    write_json_atomic(evidence, result)
    print(json.dumps(result, sort_keys=True))
    if interrupted is not None:
        raise interrupted
    return (
        0
        if result["verdict"]
        in {membership.VERDICT_PASS, membership.VERDICT_NOT_APPLICABLE}
        else 1
    )


def _plan_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    identity = preflight.get("identity") or {}
    return {
        "release_id": identity.get("release_id"),
        "cluster_id": identity.get("cluster_id"),
        "product_generation": preflight.get("product_generation"),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded BOOT-038 acceptance: remove and re-join one GPU "
            "cluster while one GPU node of another cluster is held NotReady with "
            "its Agent ACTIVE; prove the scoped verify (b250/b250b) and that no "
            "departed mail goes out for the target's nodes (b249)."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    for name in (
        "state-dir",
        "target-cluster-arn",
        "other-cluster-id",
        "other-node",
        "predecessor-evidence",
    ):
        value.add_argument(f"--{name}", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument(
        "--window",
        choices=membership.WINDOWS,
        default=membership.WINDOW_KUBELET,
        help="kubelet (plan A: a bounded self-restoring kubelet stop) or none "
        "(plan B: an external real departure such as a node-group roll)",
    )
    value.add_argument(
        "--dry-run",
        action="store_true",
        help="alias of --plan: print the preconditions, the plan and the "
        "expected timings; mutate nothing",
    )
    return value


CASE = CaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_standard_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
