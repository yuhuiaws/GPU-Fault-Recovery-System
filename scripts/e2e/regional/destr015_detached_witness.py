"""Runner-side control of the detached DESTR-015 reset-interval witnesses.

``QUIESCE_GPU_SERVICES`` stops kubelet on the node it quiesces, so from the
quiesce until ``RESTORE_GPU_SERVICES`` no ``kubectl exec`` reaches that node;
DESTR-015 quiesces both nodes in parallel. The witness therefore runs detached
on the host (``probes/destr015_witness_probe.py`` as a ``systemd-run`` unit)
and this controller talks to it only while kubelet answers: ``arm`` before the
injection, ``collect`` after the node is Ready again, ``disarm``/``status`` for
cleanup. Each request is one bounded ``kubectl exec -i`` that delivers the
pinned bundle and a single JSON line; the runner's monotonic clock is sampled
around the arm and collect requests and paired with the host clock the probe
returns, which is what lets the verdict align the host's interval records to
the runner's clock with a bounded uncertainty.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.destr015_physical_evidence import ResetIntervalScope
from scripts.e2e.regional.destr015_verdicts import (
    clock_binding,
    physical_witness_errors,
)
from scripts.e2e.regional.destr_barrier_authorization import host_reachability
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.late_ownership_probe_bundle import (
    probe_program,
    stdin_loader,
)
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError

ROLE = "reset-interval-detached"
HOST_PYTHON = "/opt/gpu-fault/current/venv/bin/python"
STATE_ROOT = "/var/lib/gpu-fault-acceptance/destr015"
UNIT_PREFIX = "gpu-fault-destr015-witness-"
# The witness must outlive the injection-to-terminal observation (2400 s), the
# restart wait (900 s) and the wait for both nodes to answer again (900 s); the
# case's estimated duration plus slack is the floor below which the remaining
# maintenance window cannot complete the case at all.
MAX_LIFETIME_SECONDS = 4200
MIN_LIFETIME_SECONDS = 1800
# How long the runner waits for kubelet to answer again before it may exec into
# a node; only the restore brings kubelet back.
HOST_RETURN_BUDGET_SECONDS = 900
# One request: kubectl overhead plus the probe's own bounded waits (45 s).
REQUEST_TIMEOUT_SECONDS = 150
RESPONSES = {
    "arm": "armed",
    "collect": "collected",
    "disarm": "disarmed",
    "status": "status",
}
EXCHANGE_KINDS = frozenset({"arm", "collect"})


def witness_lifetime_seconds(*, now: datetime, maintenance_end: datetime) -> int:
    """The bounded lifetime one witness unit is armed with.

    Bounded by the maintenance window and by the case's own budgets; a window
    that cannot hold the minimum is refused before the injection.
    """

    if now.tzinfo is None or maintenance_end.tzinfo is None:
        raise RegionalFixtureError("witness lifetime needs aware timestamps")
    remaining = int((maintenance_end - now).total_seconds())
    lifetime = min(remaining, MAX_LIFETIME_SECONDS)
    if lifetime < MIN_LIFETIME_SECONDS:
        raise RegionalFixtureError(
            f"the maintenance window has {remaining}s left; the detached witness "
            f"needs at least {MIN_LIFETIME_SECONDS}s to outlive the parallel reset "
            "and the nodes' return"
        )
    return lifetime


def _clock(value: Any) -> dict[str, int] | None:
    if (
        not isinstance(value, dict)
        or type(value.get("monotonic_ns")) is not int
        or type(value.get("realtime_ns")) is not int
    ):
        return None
    return {"monotonic_ns": value["monotonic_ns"], "realtime_ns": value["realtime_ns"]}


def arm_receipt_errors(
    receipt: dict[str, Any],
    *,
    scope: ResetIntervalScope,
    lifetime_seconds: int,
    program_sha256: str | None = None,
) -> list[str]:
    """Everything the arm receipt must bind before the injection is allowed."""

    errors: list[str] = []
    if receipt.get("run_id") != scope.run_id or receipt.get("node") != scope.node:
        errors.append("arm receipt names another run or node")
    if receipt.get("boot_id") != scope.boot_id:
        errors.append("arm receipt boot id differs from the scope")
    if receipt.get("lifetime_seconds") != lifetime_seconds:
        errors.append("arm receipt lifetime differs from the requested lifetime")
    unit = str(receipt.get("unit") or "")
    if not unit.startswith(UNIT_PREFIX) or not unit.endswith(".service"):
        errors.append("arm receipt unit is not a DESTR-015 witness unit")
    if not str(receipt.get("state_dir") or "").startswith(STATE_ROOT + "/"):
        errors.append("arm receipt state directory is outside the acceptance state")
    if program_sha256 is not None and receipt.get("program_sha256") != program_sha256:
        errors.append("the durable program copy does not match the delivered bundle")
    armed = receipt.get("armed")
    start_clock = _clock(receipt.get("host_clock_start"))
    end_clock = _clock(receipt.get("host_clock"))
    if start_clock is None or end_clock is None:
        errors.append("arm receipt carries no host clock exchange")
    if not isinstance(armed, dict) or armed.get("record") != "armed":
        errors.append("arm receipt carries no armed record")
        return errors
    start = armed.get("start") or {}
    if (
        armed.get("scope_sha256") != scope.digest()
        or start.get("scope_sha256") != scope.digest()
        or armed.get("run_id") != scope.run_id
    ):
        errors.append("armed record is not bound to this scope")
    if (
        armed.get("boot_id") != scope.boot_id
        or (start.get("tracee") or {}).get("boot_id") != scope.boot_id
        or (start.get("producer") or {}).get("boot_id") != scope.boot_id
    ):
        errors.append("armed record was not written on the scope's boot")
    if armed.get("unit") != unit:
        errors.append("armed record names another unit")
    if type(armed.get("monotonic_ns")) is not int or (
        start_clock is not None
        and end_clock is not None
        and not (
            start_clock["monotonic_ns"]
            <= armed["monotonic_ns"]
            <= end_clock["monotonic_ns"]
        )
    ):
        errors.append(
            "armed record was not written between the arm request and its response"
        )
    return errors


def collection_binding_errors(
    collection: dict[str, Any],
    *,
    scope: ResetIntervalScope,
    receipt: dict[str, Any],
) -> list[str]:
    """The transport-level binding a collection must show before its records
    are read; the verdict module re-derives everything from the records."""

    errors: list[str] = []
    if collection.get("run_id") != scope.run_id:
        errors.append("collection names another run")
    if collection.get("boot_id") != scope.boot_id:
        errors.append("host boot id at collection differs from the scope")
    state = collection.get("state") or {}
    if (
        state.get("scope_sha256") != scope.digest()
        or state.get("run_id") != scope.run_id
        or state.get("boot_id") != scope.boot_id
    ):
        errors.append("collected witness state is not this scope's")
    if collection.get("unit") != receipt.get("unit"):
        errors.append("collection names another unit than the arm receipt")
    if collection.get("armed") != receipt.get("armed"):
        errors.append("the armed record changed between arm and collect")
    if _clock(collection.get("host_clock")) is None:
        errors.append("collection carries no host clock exchange")
    return errors


class DetachedResetWitness:
    """One node's detached witness, driven through bounded one-shot execs."""

    def __init__(
        self,
        regional: Any,
        probe: HostProbeFixture,
        scope: ResetIntervalScope,
        *,
        lifetime_seconds: int,
    ) -> None:
        self.regional = regional
        self.probe = probe
        self.scope = scope
        self.lifetime_seconds = int(lifetime_seconds)
        self.program, self.bundle_sha256 = probe_program(ROLE)
        self.receipt: dict[str, Any] | None = None
        self.collection: dict[str, Any] | None = None
        self.last_status: dict[str, Any] | None = None
        self.exchanges: list[dict[str, Any]] = []
        self.armed_at: datetime | None = None

    def _request(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        program, digest = probe_program(ROLE)
        if digest != self.bundle_sha256:
            raise BoundaryDenied("physical witness source changed during the case")
        self.probe._check_target()
        request = json.dumps(
            {"kind": kind, "scope_sha256": self.scope.digest(), "payload": payload},
            separators=(",", ":"),
            allow_nan=False,
        )
        sent, sent_wall = time.monotonic_ns(), time.time_ns()
        output = self.regional.kubectl(
            "gpu",
            "exec",
            "-i",
            self.probe.pod,
            "-c",
            "probe",
            "--",
            "chroot",
            "/host",
            HOST_PYTHON,
            "-I",
            "-u",
            "-c",
            stdin_loader(program),
            input_text=program + request + "\n",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        received, received_wall = time.monotonic_ns(), time.time_ns()
        self.probe._check_target()
        lines = [line for line in str(output).splitlines() if line.strip()]
        if not lines:
            raise BoundaryDenied(f"physical witness {kind} returned no response")
        try:
            response = json.loads(lines[-1])
        except ValueError:
            raise BoundaryDenied(
                f"physical witness {kind} response is not a protocol message"
            ) from None
        if (
            not isinstance(response, dict)
            or set(response) != {"kind", "scope_sha256", "payload"}
            or not isinstance(response["payload"], dict)
        ):
            raise BoundaryDenied(f"physical witness {kind} response is unbound")
        if response["kind"] == "refused":
            raise BoundaryDenied(
                f"physical witness {kind} refused: {response['payload'].get('reason')}"
            )
        if (
            response["scope_sha256"] != self.scope.digest()
            or response["kind"] != RESPONSES[kind]
        ):
            raise BoundaryDenied(f"physical witness {kind} response is out of scope")
        result: dict[str, Any] = response["payload"]
        host_clock = _clock(result.get("host_clock"))
        if host_clock is None:
            raise BoundaryDenied(f"physical witness {kind} returned no clock binding")
        if kind in EXCHANGE_KINDS:
            self.exchanges.append(
                {
                    "kind": kind,
                    "sent_ns": sent,
                    "received_ns": received,
                    "monotonic_ns": host_clock["monotonic_ns"],
                    "realtime_ns": host_clock["realtime_ns"],
                    "runner_sent_realtime_ns": sent_wall,
                    "runner_received_realtime_ns": received_wall,
                }
            )
        return result

    def arm(self) -> dict[str, Any]:
        if self.receipt is not None:
            raise BoundaryDenied("physical witness cannot be armed twice")
        receipt = self._request(
            "arm",
            {
                "scope": self.scope.model_dump(mode="json"),
                "lifetime_seconds": self.lifetime_seconds,
                "runner_clock": {
                    "monotonic_ns": time.monotonic_ns(),
                    "realtime_ns": time.time_ns(),
                },
            },
        )
        errors = arm_receipt_errors(
            receipt,
            scope=self.scope,
            lifetime_seconds=self.lifetime_seconds,
            program_sha256=self.bundle_sha256,
        )
        if errors:
            raise BoundaryDenied(
                "physical witness arm receipt is unbound: " + "; ".join(errors)
            )
        self.receipt = receipt
        self.armed_at = datetime.now(tz=self.scope.maintenance_end.tzinfo)
        return receipt

    def collect(self) -> dict[str, Any]:
        if self.receipt is None:
            raise BoundaryDenied("physical witness was never armed")
        collection = self._request("collect", {"run_id": self.scope.run_id})
        errors = collection_binding_errors(
            collection, scope=self.scope, receipt=self.receipt
        )
        if errors:
            raise BoundaryDenied(
                "physical witness collection is unbound: " + "; ".join(errors)
            )
        self.collection = collection
        return collection

    def disarm(self) -> dict[str, Any]:
        return self._request("disarm", {"run_id": self.scope.run_id})

    def status(self) -> dict[str, Any]:
        self.last_status = self._request("status", {"run_id": self.scope.run_id})
        return self.last_status

    def evidence(self) -> dict[str, Any]:
        return {
            "scope": self.scope.model_dump(mode="json"),
            "lifetime_seconds": self.lifetime_seconds,
            "bundle_sha256": self.bundle_sha256,
            "receipt": self.receipt,
            "collection": self.collection,
            "exchanges": list(self.exchanges),
        }


# --------------------------------------------------------------------------- #
# Runner flow: the node must answer again before any exec
# --------------------------------------------------------------------------- #
# ``run`` is the runner's live-run record: ``regional``, ``settings.nodes``,
# ``probes``, ``baselines``, ``preflight``, ``case_dir``, ``physical_witnesses``,
# ``witness_armed_at``, ``witness_lifetime_seconds`` and ``hosts_returned``.
def host_return(run: Any, node: str) -> dict[str, Any]:
    """Wait for kubelet to answer again, then decide whether an exec may follow.

    The same rule as DESTR-016's cleanup: a node that never answers is only
    known to have lost its witness unit when its boot id changed or the unit's
    bounded lifetime has passed; otherwise nothing is assumed and no exec is
    attempted.
    """

    baseline = run.baselines.get(node) or run.preflight["nodes"].get(node) or {}
    return host_reachability(
        run.regional,
        node=node,
        baseline_boot_id=baseline.get("boot_id"),
        holder_armed_at=run.witness_armed_at.get(node),
        max_hold_seconds=run.witness_lifetime_seconds,
        budget_seconds=HOST_RETURN_BUDGET_SECONDS,
    )


def await_hosts_return(run: Any) -> dict[str, dict[str, Any]]:
    """No exec reaches a quiesced node: wait until both are Ready again.

    A node that never answers within the budget leaves its witness records on
    the host unread; the case fails with that reason rather than exec'ing a
    NotReady node, and cleanup applies the same rule. Both probe Pods are
    confirmed (recreated if the outage took them) before the first host read.
    """

    returned = {node: host_return(run, node) for node in run.settings.nodes}
    run.hosts_returned = returned
    write_json_atomic(run.case_dir / "hosts-return.json", returned)
    lost = [node for node, reach in returned.items() if not reach["exec_allowed"]]
    if lost:
        reasons = "; ".join(
            f"{node}: {returned[node]['reason']} ({returned[node].get('wait_error')})"
            for node in lost
        )
        raise RegionalFixtureError(
            f"{', '.join(lost)} did not return Ready within "
            f"{HOST_RETURN_BUDGET_SECONDS}s after the parallel quiesce ({reasons}); "
            "the detached witness records there are treated as lost and no exec "
            "is attempted on a NotReady node"
        )
    for probe in run.probes.values():
        probe.create()
    return returned


def collect_witness_intervals(
    run: Any,
    state: dict[str, Any],
    hosts: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[str], dict[str, Any]]:
    """One exec per node, after both are Ready: read the durable records and
    grade them together with the post-hoc host snapshots."""

    errors: list[str] = []
    witnesses: dict[str, dict[str, Any]] = {}
    for node, witness in run.physical_witnesses.items():
        try:
            witness.collect()
        except Exception as exc:  # noqa: BLE001 - a refused collection fails the case
            errors.append(
                f"{node} physical witness collection refused: "
                f"{type(exc).__name__}: {exc}"
            )
        witnesses[node] = {
            "scope": witness.scope,
            "receipt": witness.receipt,
            "collection": witness.collection,
            "exchanges": witness.exchanges,
        }
    intervals = {
        node: {**witness.evidence(), "clock_binding": clock_binding(witness.exchanges)}
        for node, witness in run.physical_witnesses.items()
    }
    write_json_atomic(run.case_dir / "physical-reset-intervals.json", intervals)
    errors.extend(
        physical_witness_errors(
            witnesses, workflow=state.get("workflow") or {}, hosts=hosts
        )
    )
    return errors, intervals


def node_answers_for_cleanup(run: Any, node: str, result: dict[str, Any]) -> bool:
    """Cleanup's gate: exec into ``node`` only once kubelet answers again.

    Records the reachability decision; when the node never answers, records
    what may be assumed about the witness (gone with a new boot or an expired
    lifetime, otherwise unknown), leaves the host resources for manual
    reconciliation and marks the cleanup failed. No exec is attempted.
    """

    try:
        reach = host_return(run, node)
    except Exception as exc:  # noqa: BLE001 - an unreadable node is not reachable
        reach = {
            "exec_allowed": False,
            "assume_disarmed": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    result[f"host_reachability:{node}"] = reach
    if reach["exec_allowed"]:
        return True
    result[f"witness_disarm:{node}"] = {
        "disarmed": "assumed" if reach["assume_disarmed"] else "unknown",
        "reason": reach["reason"],
    }
    result[f"probe_cleanup_{node}"] = {"deferred": True, "reason": reach["reason"]}
    result["errors"].append(
        f"{node} never answered again ({reach['reason']}); the witness unit and "
        "the host probe resources are left for manual reconciliation, no exec "
        "is attempted on a NotReady node"
    )
    return False
