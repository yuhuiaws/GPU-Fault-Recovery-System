"""A release transaction refuses to start while a node install is in flight.

R4 (2026-09-09) changed the node-action command id of REMEDIATE_DRIVER /
UPDATE_SOFTWARE_FIRMWARE / REMEDIATE_EFA_DRIVER from ``<key>/<node>/agent-N``
to ``<key>/<node>``. The new control plane reads the old rows back through a
one-release shim, but a rollback to the previous release runs old code that
derives ``agent-N`` again, finds no ledger row under that id, and submits a
SECOND install on top of the one the node agent is still running (install
timeout 1800 s). The same holds for an upgrade that lands while the old control
plane has such a step WAITING.

The ruling is a release-tooling precondition, pinned here: before ``deploy``
opens a transaction (upgrade, resume, supersede or rollback) and before the
engine's automatic rollback, one control-plane store read lists the workflows
whose install steps are PENDING or WAITING; any hit refuses with the workflow
ids and nodes named, unless the operator passed ``--allow-inflight-installs``.
A store that cannot answer refuses too (fail closed), naming the error, unless
no Running control-plane Pod could run the probe at all and the caller is the
automatic rollback or carries the consent. A refused automatic rollback logs why
and leaves the transaction in ``failed``, the phase the resume / supersede
levers already handle.
"""

from __future__ import annotations

import ast
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cli as admin_cli
from gpu_fault.execution.config import NODE_INSTALL_OPERATIONS
from gpu_fault.operation_registry import GENERATION_STABLE_COMMAND_OPERATIONS
from gpu_fault_release import regional_deployment_inventory as INVENTORY
from gpu_fault_release import regional_release_diff as DIFF
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_release_probes as PROBES
from gpu_fault_release import regional_release_store_preflight as GATE
from gpu_fault_release import rollout as MODULE

ROOT = Path(__file__).resolve().parents[2]
ENV = "GPU_FAULT_RELEASE_ALLOW_INFLIGHT_INSTALLS"
FLAG = "--allow-inflight-installs"
NAMESPACE = "gpu-fault-system"
# The line the shell wrapper prints after the probe, whatever the probe did.
MARKER = "__GPU_FAULT_PROBE_EXIT"


# --- the operation set and the spellings ---------------------------------------


def test_the_gate_covers_exactly_the_generation_stable_operations() -> None:
    """The three operations R4 moved off the agent-generation suffix are the
    three whose in-flight rows an old control plane would re-submit -- and the
    engine reads them from the executor's install set, not a private copy."""

    assert set(GATE.INSTALL_OPERATIONS) == {
        operation.value for operation in GENERATION_STABLE_COMMAND_OPERATIONS
    }
    assert GATE.INSTALL_OPERATIONS == tuple(
        operation.value for operation in NODE_INSTALL_OPERATIONS
    )
    assert set(GATE.INSTALL_OPERATIONS) == {
        "REMEDIATE_DRIVER",
        "UPDATE_SOFTWARE_FIRMWARE",
        "REMEDIATE_EFA_DRIVER",
    }


def test_the_probe_inlines_the_operation_set_instead_of_importing_it() -> None:
    """The probe runs against whatever ``gpu_fault`` is already deployed. For the
    release that first carries R4 -- and for every rollback target -- that is a
    module without ``GENERATION_STABLE_COMMAND_OPERATIONS``."""

    source = PROBES.probe_source("inflight_installs")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }
    assert "gpu_fault.operation_registry" not in imported, imported
    for name in GATE.INSTALL_OPERATIONS:
        assert f"WorkflowOperation.{name}" in source


def test_the_cli_and_the_engine_spell_the_variable_and_the_flag_the_same() -> None:
    assert admin_cli.ALLOW_INFLIGHT_INSTALLS_ENV == ENV
    assert GATE.ALLOW_INFLIGHT_INSTALLS_ENV == ENV
    assert GATE.ALLOW_INFLIGHT_INSTALLS_FLAG == FLAG


def test_the_release_object_binds_the_gate_like_the_other_preflights() -> None:
    assert (
        MODULE.RegionalRelease._require_no_inflight_installs
        is GATE.require_no_inflight_installs
    )


# --- the gate ---------------------------------------------------------------------


def _pods(*pods: tuple[str, bool]) -> str:
    """What the gate's ``get pod`` prints: one Running Pod per line, its name,
    a tab, then the ``ready`` flag of each container (``true false`` for a Pod
    with one ready and one crashing container; nothing at all for a Pod whose
    containers have no status yet)."""

    return "\n".join(
        f"{name}\t{'true' if ready else 'false'}" if ready is not None else name
        for name, ready in pods
    )


class Runner:
    """Every control-plane role has one Running Pod (``cpu-pod``, ``ready``
    unless said otherwise); every exec answers the same way: a string is what
    ``kubectl exec`` printed, an exception is what ``Runner.run`` raised
    (kubectl exited non-zero, or timed out). ``list_error`` is what ``get pod``
    itself raises (API server unreachable, expired token, no list RBAC)."""

    dry_run = False

    def __init__(
        self,
        result: str | Exception,
        *,
        ready: bool = True,
        list_error: Exception | None = None,
    ) -> None:
        self.result = result
        self.ready = ready
        self.list_error = list_error
        self.execs: list[list[str]] = []
        self.exec_kwargs: list[dict[str, Any]] = []
        self.lists = 0

    def run(self, args, **kwargs):
        if "get" in args and "pod" in args:
            self.lists += 1
            if self.list_error is not None:
                raise self.list_error
            return _pods(("cpu-pod", self.ready))
        self.execs.append(list(args))
        self.exec_kwargs.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def probe(self, _args, **_kwargs) -> bool:
        return True


def _release(
    result: str | Exception, *, ready: bool = True, list_error: Exception | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        runner=Runner(result, ready=ready, list_error=list_error),
        config=SimpleNamespace(namespace=NAMESPACE),
        _cpu=lambda *args: ["kubectl", *args],
    )


def _probe_output(*lines: str, code: int = 0) -> str:
    """What the shell wrapper hands back: whatever the probe printed, then the
    exit marker. The marker is the proof that the probe process ran at all."""

    return "\n".join([*lines, f"{MARKER}={code}"])


def _snapshot(*items: dict[str, Any], bounded: bool = True) -> str:
    return _probe_output(
        json.dumps(
            {
                "inflight": list(items),
                "inflight_count": len(items),
                "bounded": bounded,
                "scanned": len(items),
            }
        )
    )


def _timed_out() -> MODULE.ReleaseError:
    """What ``Runner.run`` raises when ``timeout_seconds`` expires."""

    error = MODULE.ReleaseError("command timed out after 120s: kubectl")
    error.__cause__ = subprocess.TimeoutExpired(cmd="kubectl", timeout=120)
    return error


class RoleRunner:
    """Resolves the Running Pods of each control-plane Deployment (``pods`` maps
    the role to what ``get pod`` prints, see ``_pods``) and answers an exec per
    Pod: a string is stdout, an exception is what ``kubectl exec`` raised."""

    dry_run = False

    def __init__(self, pods: dict[str, str], answers: dict[str, str | Exception]):
        self.pods = pods
        self.answers = answers
        self.execs: list[str] = []

    def run(self, args, **_kwargs):
        if "get" in args and "pod" in args:
            label = args[args.index("-l") + 1].removeprefix("app=")
            return self.pods.get(label, "")
        pod = next(
            argument
            for argument in args[args.index("exec") + 1 :]
            if not argument.startswith("-")
        )
        self.execs.append(pod)
        answer = self.answers[pod]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def probe(self, _args, **_kwargs) -> bool:
        return True


def _role_release(runner: RoleRunner) -> SimpleNamespace:
    return SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(namespace=NAMESPACE),
        _cpu=lambda *args: ["kubectl", *args],
    )


def _no_pod_release() -> SimpleNamespace:
    """No control-plane role has a Running Pod: the probe provably never ran."""

    return _role_release(RoleRunner(pods={}, answers={}))


DRIVER_STEP = {
    "workflow_id": "workflow-7f3a",
    "incident_id": "incident-1",
    "workflow_status": "RUNNING",
    "operation": "REMEDIATE_DRIVER",
    "step_index": 3,
    "node_ids": ["ip-10-0-1-17"],
    "step_status": "WAITING",
}


def test_a_pending_driver_install_refuses_the_transaction_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV, raising=False)
    release = _release(_snapshot(DRIVER_STEP))

    with pytest.raises(GATE.InflightInstallsRefused) as failure:
        GATE.require_no_inflight_installs(release, action="upgrade")

    message = str(failure.value)
    assert isinstance(failure.value, MODULE.ReleaseError), (
        "the engine's main() only reports ReleaseError cleanly"
    )
    assert "upgrade" in message
    assert "workflow-7f3a" in message
    assert "ip-10-0-1-17" in message
    assert "REMEDIATE_DRIVER" in message
    assert FLAG in message
    assert len(release.runner.execs) == 1, "one store read"


def test_the_override_lets_the_transaction_proceed_and_says_what_it_skipped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV, "1")
    release = _release(_snapshot(DRIVER_STEP))

    result = GATE.require_no_inflight_installs(release, action="rollback")

    assert result["inflight_count"] == 1
    assert result["verdict"] == "overridden"
    assert result["checked"] is True
    assert result["reason"] == FLAG
    assert len(result["steps"]) == 1 and "workflow-7f3a" in result["steps"][0], (
        "the verdict carries the listed set the consent covered"
    )
    logged = capsys.readouterr().err
    assert "workflow-7f3a" in logged
    assert "ip-10-0-1-17" in logged
    assert FLAG in logged


def test_no_in_flight_install_lets_the_transaction_proceed_quietly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(ENV, raising=False)

    result = GATE.require_no_inflight_installs(_release(_snapshot()), action="upgrade")

    checked_at = result.pop("checked_at")
    assert datetime.fromisoformat(checked_at).utcoffset() == timedelta(0), checked_at
    assert result == {
        "checked": True,
        "verdict": "clear",
        "reason": None,
        "steps": [],
        "inflight_count": 0,
        "scanned": 0,
    }
    assert capsys.readouterr().err == ""


def test_every_verdict_says_when_it_was_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rollback re-entered after ``rollback-cpu-restored`` skips the gate and
    leaves the FIRST attempt's verdict in the state; a reader has to be able to
    tell how old it is (fix round 4, LOW-4). UTC, ISO-8601, with the offset."""

    before = datetime.now(timezone.utc)
    monkeypatch.delenv(ENV, raising=False)
    verdicts = [
        GATE.require_no_inflight_installs(_release(_snapshot()), action="upgrade"),
        GATE.require_no_inflight_installs(
            _no_pod_release(), action="rollback", unreadable="proceed"
        ),
    ]
    monkeypatch.setenv(ENV, "1")
    verdicts.append(
        GATE.require_no_inflight_installs(
            _release(_snapshot(DRIVER_STEP)), action="upgrade"
        )
    )

    assert [verdict["verdict"] for verdict in verdicts] == [
        "clear",
        "unchecked",
        "overridden",
    ]
    for verdict in verdicts:
        checked_at = datetime.fromisoformat(verdict["checked_at"])
        assert checked_at.tzinfo is not None, "the timestamp carries its offset"
        assert checked_at.utcoffset() == timedelta(0), "UTC, like the driver's record"
        assert before <= checked_at <= datetime.now(timezone.utc), verdict


def test_the_probe_runs_through_a_shell_wrapper_that_marks_its_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Runner.run`` raises the same ``command failed (1): kubectl`` whether
    kubectl never reached the Pod or the probe launched and died on line one.
    So the probe is not the exec'd command: a ``sh`` wrapper is, it feeds the
    probe to ``python -`` on stdin, merges stderr into the captured output so
    the cause survives, and prints the exit marker last whatever happened, so
    the wrapper itself always exits 0. The marker's presence is the proof that
    the process ran; kubectl failing before it = transport. The read is not
    ``sensitive`` (its output is workflow ids the refusal prints anyway) and it
    has a timeout: a hung store read must refuse, not hang the engine."""

    monkeypatch.delenv(ENV, raising=False)
    release = _release(_snapshot())

    GATE.require_no_inflight_installs(release, action="upgrade")

    (args,) = release.runner.execs
    (kwargs,) = release.runner.exec_kwargs
    assert "-i" in args, "the probe travels on stdin"
    assert args[-3:-1] == ["sh", "-c"], args
    wrapper = args[-1]
    assert "python -" in wrapper and f'echo "{MARKER}=$?"' in wrapper, wrapper
    assert "2>&1" in wrapper, "the probe's own traceback must reach the log"
    assert kwargs["input_text"] == PROBES.probe_source("inflight_installs")
    assert kwargs.get("sensitive", False) is False, "the cause must survive"
    assert kwargs["timeout_seconds"] == GATE.PROBE_TIMEOUT_SECONDS == 120
    assert kwargs["capture"] is True, "the answer is read, not streamed"


def test_an_unreachable_store_refuses_and_names_the_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV, raising=False)
    release = _release(
        MODULE.ReleaseError("command failed (1): kubectl exec"), ready=False
    )

    with pytest.raises(GATE.InflightInstallsRefused) as failure:
        GATE.require_no_inflight_installs(release, action="upgrade")

    message = str(failure.value)
    assert "command failed (1): kubectl exec" in message
    assert FLAG in message


def test_unreadable_evidence_refuses_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV, raising=False)

    for garbage in (
        "not json",  # no marker: kubectl exited 0 without the wrapper's line
        _probe_output("not json"),
        _probe_output("[]"),
        _probe_output('{"inflight": "x"}'),
    ):
        with pytest.raises(GATE.InflightInstallsRefused, match="evidence"):
            GATE.require_no_inflight_installs(_release(garbage), action="upgrade")


def test_the_override_also_covers_a_store_that_cannot_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rollback of a control plane that is itself down must stay possible."""

    monkeypatch.setenv(ENV, "1")

    result = GATE.require_no_inflight_installs(_no_pod_release(), action="rollback")

    assert result["verdict"] == "unchecked"
    assert result["checked"] is False
    assert result["reason"] == FLAG
    assert "could not reach any Running control-plane Pod" in capsys.readouterr().err


def test_consent_does_not_cover_an_evidence_defect(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--allow-inflight-installs`` is consent to proceed over a LISTED
    in-flight set, or over a store no Pod could answer for. It is not consent
    to proceed over a check that ran and could not vouch for anything: an
    unbounded scan, no JSON, the wrong shape, a probe that reported its own
    error, or a probe that hung. Those refuse in every mode and the message
    points at the cause, not at the flag (fix round 3, MEDIUM-4)."""

    monkeypatch.setenv(ENV, "1")
    cases: list[tuple[str | Exception, str]] = [
        (_snapshot(bounded=False), "could not bound"),
        (_probe_output("Traceback (most recent call last):", code=1), "exited 1"),
        (_probe_output("", code=0), "evidence"),
        (_probe_output('{"inflight": 1}'), "evidence"),
        (
            _probe_output(
                json.dumps({"probe_error": "OperationalError: connection refused"}),
                code=1,
            ),
            "OperationalError: connection refused",
        ),
        (_timed_out(), "did not answer within"),
    ]
    for evidence, cause in cases:
        for unreadable in ("refuse", "proceed"):
            with pytest.raises(GATE.InflightInstallsRefused) as failure:
                GATE.require_no_inflight_installs(
                    _release(evidence), action="rollback", unreadable=unreadable
                )
            message = str(failure.value)
            assert cause in message, (evidence, message)
            assert f"{FLAG} does not cover" in message, message
            assert "rerun" not in message.split("does not cover")[0], (
                "the lever is not offered as the way out"
            )
    assert capsys.readouterr().err == "", "an evidence defect is refused, not logged"


def test_the_lever_named_matches_the_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``deploy`` has the flag; ``rollout-regional-release.sh rollback`` has no
    flag at all, only the variable (F3)."""

    monkeypatch.delenv(ENV, raising=False)

    with pytest.raises(GATE.InflightInstallsRefused) as upgrade:
        GATE.require_no_inflight_installs(
            _release(_snapshot(DRIVER_STEP)), action="upgrade"
        )
    assert "gpu-fault-admin deploy" in str(upgrade.value)
    assert FLAG in str(upgrade.value)
    assert f"{ENV}=1" not in str(upgrade.value)

    with pytest.raises(GATE.InflightInstallsRefused) as rollback:
        GATE.require_no_inflight_installs(
            _release(_snapshot(DRIVER_STEP)), action="rollback"
        )
    assert f"{ENV}=1" in str(rollback.value)
    assert f"rerun with {FLAG}" not in str(rollback.value)

    with pytest.raises(GATE.InflightInstallsRefused) as unreachable:
        GATE.require_no_inflight_installs(
            _release(MODULE.ReleaseError("exec failed")), action="rollback"
        )
    assert f"{ENV}=1" in str(unreachable.value)


def test_an_unbounded_scan_refuses_instead_of_reading_zero_as_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV, raising=False)
    release = _release(_snapshot(bounded=False))

    with pytest.raises(GATE.InflightInstallsRefused) as failure:
        GATE.require_no_inflight_installs(release, action="upgrade")

    message = str(failure.value)
    assert "could not bound" in message
    assert "1000" in message
    assert f"{FLAG} does not cover" in message, "consent is not the way out"
    assert "drain" in message, "the message points at the cause"


def test_the_automatic_rollback_proceeds_only_when_no_pod_could_run_the_probe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The primary automatic-rollback scenario is a control plane that is down
    (CrashLoop, Aurora window, broken wheel); with no Running Pod in any role
    nothing dispatches, and wedging production in ``failed`` is the worse
    failure (F2). But a Running Pod whose probe ran and exited without evidence
    is the opposite case: the old dispatcher's warm pool may still be
    dispatching while the probe's fresh connection failed, so the automatic
    rollback REFUSES (fix round 3, HIGH-1). An actual in-flight install still
    refuses."""

    monkeypatch.delenv(ENV, raising=False)

    result = GATE.require_no_inflight_installs(
        _no_pod_release(), action="rollback", unreadable="proceed"
    )

    assert result["verdict"] == "unchecked"
    assert result["checked"] is False
    assert "no Running control-plane Pod" in result["reason"]
    logged = capsys.readouterr().err
    assert "inflight-installs-unchecked" in logged
    assert "could not reach any Running control-plane Pod" in logged
    assert "automatic" in logged

    died = _release(_probe_output("ImportError: cannot import name 'x'", code=1))
    with pytest.raises(GATE.InflightInstallsRefused) as failure:
        GATE.require_no_inflight_installs(died, action="rollback", unreadable="proceed")
    assert "exited 1" in str(failure.value)
    assert "ImportError: cannot import name 'x'" in str(failure.value), (
        "the cause the wrapper captured reaches the refusal"
    )

    with pytest.raises(GATE.InflightInstallsRefused, match="workflow-7f3a"):
        GATE.require_no_inflight_installs(
            _release(_snapshot(DRIVER_STEP)), action="rollback", unreadable="proceed"
        )


def test_only_a_probe_that_could_not_run_counts_as_unreadable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Classified by WHERE the failure happened. Once the wrapper's marker is
    on stdout the probe process ran, and everything after that is evidence,
    binding in every mode: an overflowed scan, no JSON at all, the wrong shape,
    the probe's own ``probe_error``, or a hang past the timeout (the exec did
    start; the store did not answer in time). Only "no Running Pod in any role"
    is unreachable."""

    monkeypatch.delenv(ENV, raising=False)
    evidence_defects: list[str | Exception] = [
        _snapshot(bounded=False),
        "not json",
        _probe_output("not json"),
        _probe_output("[]"),
        _probe_output('{"inflight": 1}'),
        _probe_output("", code=137),
        _probe_output(json.dumps({"probe_error": "OperationalError: timeout"})),
        _timed_out(),
    ]
    for evidence in evidence_defects:
        with pytest.raises(GATE.InflightInstallsRefused) as failure:
            GATE.require_no_inflight_installs(
                _release(evidence), action="rollback", unreadable="proceed"
            )
        assert not isinstance(failure.value.__cause__, GATE.StoreUnreachable), (
            evidence,
            str(failure.value),
        )
    assert capsys.readouterr().err == "", "an evidence defect is refused, not logged"

    result = GATE.require_no_inflight_installs(
        _no_pod_release(), action="rollback", unreadable="proceed"
    )
    assert result["verdict"] == "unchecked"
    assert "inflight-installs-unchecked" in capsys.readouterr().err


def test_a_probe_that_reports_its_own_error_refuses_even_the_automatic_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact HIGH-1 shape: Running Pod, probe ran, fresh store connection
    failed. The refusal names the probe's error so the operator sees the
    rotation window, not a bare exit code."""

    monkeypatch.delenv(ENV, raising=False)
    release = _release(
        _probe_output(
            json.dumps(
                {"probe_error": "OperationalError: password authentication failed"}
            ),
            code=1,
        )
    )

    with pytest.raises(GATE.InflightInstallsRefused) as failure:
        GATE.require_no_inflight_installs(
            release, action="rollback", unreadable="proceed"
        )

    message = str(failure.value)
    assert "OperationalError: password authentication failed" in message
    assert "probe reported" in message
    assert len(release.runner.execs) == 1, "an answer from one Pod is final"


def test_a_hung_probe_refuses_instead_of_hanging_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store read that does not return inside the timeout is an answer of
    sorts -- the store is not healthy -- and not "no Pod could run it": the exec
    started. Refused in every mode; the other roles are not tried (they read
    the same store, and each would cost the whole window again)."""

    monkeypatch.delenv(ENV, raising=False)
    release = _release(_timed_out())

    for unreadable in ("refuse", "proceed"):
        with pytest.raises(GATE.InflightInstallsRefused) as failure:
            GATE.require_no_inflight_installs(
                release, action="rollback", unreadable=unreadable
            )
        assert "did not answer within 120" in str(failure.value)
        assert not isinstance(failure.value.__cause__, GATE.StoreUnreachable), (
            "a timeout is an evidence defect, not an unreachable store"
        )
    assert len(release.runner.execs) == 2, "one exec per call, no role fallback"


def test_a_kubectl_failure_on_a_pod_with_no_ready_container_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The wrapper always exits 0, so a non-zero ``kubectl exec`` with no marker
    on stdout means kubectl never got the wrapper to run. When the Pod is phase
    Running but NO container of it is ready -- CrashLoopBackOff (a broken
    control-plane wheel), ContainerCreating, terminating with its containers
    stopped -- nothing in it dispatches, so there is no evidence the probe
    started and no dispatcher to have asked: ``StoreUnreachable`` (the
    automatic rollback proceeds, logging), tried on every Running Pod of every
    role first."""

    monkeypatch.delenv(ENV, raising=False)
    release = _release(MODULE.ReleaseError("command failed (1): kubectl"), ready=False)

    result = GATE.require_no_inflight_installs(
        release, action="rollback", unreadable="proceed"
    )

    assert result["verdict"] == "unchecked"
    assert len(release.runner.execs) == len(INVENTORY.CPU_RUNTIME_DEPLOYMENTS), (
        "every role's Running Pod was tried before giving up"
    )
    logged = capsys.readouterr().err
    assert "command failed (1): kubectl" in logged
    assert "no ready container" in logged, "the log says why the Pod did not count"

    with pytest.raises(GATE.InflightInstallsRefused, match="could not read"):
        GATE.require_no_inflight_installs(release, action="rollback")


def test_a_kubectl_failure_on_a_ready_pod_refuses_in_every_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Pod with a READY container is a live dispatcher: its warm pool keeps
    handing installs to node agents while kubectl cannot get into it -- kubelet
    down or the node partitioned (the API server reports Running for up to the
    eviction timeout, ``error dialing backend``), a deploy-host identity without
    ``pods/exec``, an image without ``sh``. Reading that as "unreachable" let
    every automatic rollback proceed unchecked over a dispatching control plane
    (fix round 4, HIGH-1). It is a refusal in every mode, and the message names
    kubectl / exec / RBAC -- not the store, which was never asked."""

    monkeypatch.delenv(ENV, raising=False)
    release = _release(MODULE.ReleaseError("command failed (1): kubectl"), ready=True)

    for unreadable in ("refuse", "proceed"):
        with pytest.raises(GATE.InflightInstallsRefused) as failure:
            GATE.require_no_inflight_installs(
                release, action="rollback", unreadable=unreadable
            )
        message = str(failure.value)
        assert "command failed (1): kubectl" in message
        assert "kubectl" in message and "exec" in message and "RBAC" in message, message
        assert "ready" in message, "the message says why the Pod counted"
        assert "could not read the control-plane store" not in message, (
            "the store was never asked; do not blame it"
        )
        assert f"{ENV}=1" in message, (
            "consent is offered: a transport fault is checkable"
        )
        assert isinstance(failure.value.__cause__, GATE.KubectlFailure), failure.value
        assert not isinstance(failure.value.__cause__, GATE.StoreUnreachable), (
            "a ready Pod kubectl cannot enter is not an unreachable store"
        )
    assert len(release.runner.execs) == 2 * len(INVENTORY.CPU_RUNTIME_DEPLOYMENTS), (
        "every role's ready Pod was tried before refusing, in both modes"
    )
    assert capsys.readouterr().err == "", "a refusal is not logged as unchecked"


def test_consent_overrides_a_kubectl_failure_on_a_ready_pod_and_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unlike a probe defect, a kubectl / RBAC fault is something the operator
    can check by hand before consenting, so the consent env may open this one.
    The narration is honest about what was skipped and why."""

    monkeypatch.setenv(ENV, "1")
    release = _release(MODULE.ReleaseError("command failed (1): kubectl"), ready=True)

    result = GATE.require_no_inflight_installs(release, action="rollback")

    assert result["verdict"] == "unchecked"
    assert result["checked"] is False
    assert "exec failed on a ready Pod" in result["reason"], result
    assert "consent given" in result["reason"], result
    assert "command failed (1): kubectl" in result["error"]
    logged = capsys.readouterr().err
    assert "inflight-installs-unchecked" in logged
    assert "exec failed on a ready Pod, consent given" in logged, logged


def test_one_ready_pod_that_kubectl_cannot_enter_outweighs_the_unready_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck rollout: the new Pod CrashLoops (not ready), the old one is
    ready and dispatching, and kubectl fails against both. The old Pod is the
    one that matters, so the gate refuses rather than proceeding unchecked --
    but evidence from any Pod that did answer still wins over both."""

    monkeypatch.delenv(ENV, raising=False)
    ingress = INVENTORY.CPU_INGRESS_DEPLOYMENT
    other = next(name for name in INVENTORY.CPU_RUNTIME_DEPLOYMENTS if name != ingress)
    failure = MODULE.ReleaseError("command failed (1): kubectl")
    runner = RoleRunner(
        pods={ingress: _pods(("ingress-new", False), ("ingress-old", True))},
        answers={"ingress-new": failure, "ingress-old": failure},
    )

    with pytest.raises(GATE.InflightInstallsRefused) as refused:
        GATE.require_no_inflight_installs(
            _role_release(runner), action="rollback", unreadable="proceed"
        )
    assert isinstance(refused.value.__cause__, GATE.KubectlFailure), (
        "the ready Pod's failure decides the class"
    )
    assert "ingress-old" in str(refused.value), "the ready Pod is named"
    assert runner.execs == ["ingress-new", "ingress-old"]

    answering = RoleRunner(
        pods={
            ingress: _pods(("ingress-new", False), ("ingress-old", True)),
            other: _pods(("worker-pod", True)),
        },
        answers={
            "ingress-new": failure,
            "ingress-old": failure,
            "worker-pod": _snapshot(),
        },
    )
    result = GATE.require_no_inflight_installs(
        _role_release(answering), action="rollback", unreadable="proceed"
    )
    assert result["verdict"] == "clear", "evidence from a Pod that answered wins"


def test_a_failed_pod_list_refuses_and_names_kubectl_not_the_store(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``get pod`` itself failing -- API server unreachable, an expired ``aws
    eks get-token``, no list RBAC -- used to escape as a generic evidence
    defect with the "drain the backlog / fix the probe" text and no consent
    (fix round 4, MEDIUM-2). It is a kubectl-level failure: refused in every
    mode, honestly worded, and the consent env may override it."""

    monkeypatch.delenv(ENV, raising=False)
    release = _release(
        _snapshot(), list_error=MODULE.ReleaseError("command failed (1): kubectl")
    )

    for unreadable in ("refuse", "proceed"):
        with pytest.raises(GATE.InflightInstallsRefused) as failure:
            GATE.require_no_inflight_installs(
                release, action="rollback", unreadable=unreadable
            )
        message = str(failure.value)
        assert "kubectl could not list control-plane Pods" in message, message
        assert "API server" in message, message
        assert "command failed (1): kubectl" in message
        assert "could not read the control-plane store" not in message
        assert "drain the backlog" not in message, "not an evidence defect"
        assert isinstance(failure.value.__cause__, GATE.KubectlFailure), failure.value
    assert release.runner.execs == [], "no Pod was ever exec'd"
    assert release.runner.lists == 2 * len(INVENTORY.CPU_RUNTIME_DEPLOYMENTS), (
        "every role's list was attempted before refusing, in both modes"
    )
    assert capsys.readouterr().err == ""

    monkeypatch.setenv(ENV, "1")
    result = GATE.require_no_inflight_installs(release, action="rollback")
    assert result["verdict"] == "unchecked"
    assert "kubectl could not list" in result["reason"], result
    assert "consent given" in result["reason"], result
    assert "inflight-installs-unchecked" in capsys.readouterr().err


def test_a_non_zero_exit_distrusts_even_well_formed_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe that printed a valid snapshot and then exited non-zero did not
    finish the way it meant to (an exception after the print, a killed
    interpreter); its evidence is not trusted (fix round 4, LOW-6)."""

    monkeypatch.delenv(ENV, raising=False)
    clear = {"inflight": [], "inflight_count": 0, "bounded": True, "scanned": 3}
    release = _release(_probe_output(json.dumps(clear), code=1))

    for unreadable in ("refuse", "proceed"):
        with pytest.raises(GATE.InflightInstallsRefused) as failure:
            GATE.require_no_inflight_installs(
                release, action="rollback", unreadable=unreadable
            )
        message = str(failure.value)
        assert "exited 1" in message, message
        assert "not trusted" in message, message


def test_every_running_pod_of_a_role_is_tried_not_only_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck rollout leaves the old ReplicaSet's Pod Running beside the new
    CrashLooping one; ``items[0]`` may be the crashing Pod. The old Pod runs the
    same store read and is the one still dispatching."""

    monkeypatch.delenv(ENV, raising=False)
    ingress = INVENTORY.CPU_INGRESS_DEPLOYMENT
    runner = RoleRunner(
        pods={ingress: _pods(("ingress-new", False), ("ingress-old", True))},
        answers={
            "ingress-new": MODULE.ReleaseError("command failed (1): kubectl"),
            "ingress-old": _snapshot(DRIVER_STEP),
        },
    )

    with pytest.raises(GATE.InflightInstallsRefused, match="workflow-7f3a"):
        GATE.require_no_inflight_installs(_role_release(runner), action="rollback")

    assert runner.execs == ["ingress-new", "ingress-old"]


def test_the_three_failure_kinds_are_distinct_exception_types() -> None:
    for kind in (GATE.StoreUnreachable, GATE.KubectlFailure):
        assert issubclass(kind, GATE.ReleaseError), (
            "main() reports every refusal as a ReleaseError"
        )
        assert not issubclass(kind, GATE.InflightInstallsRefused), (
            "a cause the gate decides on, not a refusal by itself"
        )
    assert not issubclass(GATE.KubectlFailure, GATE.StoreUnreachable), (
        "a live control plane kubectl could not ask is not an unreachable store"
    )
    assert not issubclass(GATE.StoreUnreachable, GATE.KubectlFailure), (
        "nor the other way round: the automatic rollback keys on the exact type"
    )
    with pytest.raises(GATE.StoreUnreachable):
        GATE.inflight_install_snapshot(_no_pod_release())
    with pytest.raises(GATE.ReleaseError) as defect:
        GATE.inflight_install_snapshot(_release(_snapshot(bounded=False)))
    assert not isinstance(defect.value, GATE.StoreUnreachable), (
        "an overflowed scan is an answer, not an unreachable store"
    )


def test_the_read_falls_back_to_another_running_control_plane_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ingress Pod is the one the failed upgrade just rolled; any Running
    control-plane role can answer the same store read (F2)."""

    monkeypatch.delenv(ENV, raising=False)
    ingress = INVENTORY.CPU_INGRESS_DEPLOYMENT
    other = next(name for name in INVENTORY.CPU_RUNTIME_DEPLOYMENTS if name != ingress)
    runner = RoleRunner(
        pods={
            ingress: _pods(("ingress-pod", True)),
            other: _pods(("worker-pod", True)),
        },
        answers={
            "ingress-pod": MODULE.ReleaseError("command failed (137): kubectl"),
            "worker-pod": _snapshot(DRIVER_STEP),
        },
    )

    with pytest.raises(GATE.InflightInstallsRefused, match="workflow-7f3a"):
        GATE.require_no_inflight_installs(_role_release(runner), action="rollback")

    assert runner.execs[0] == "ingress-pod"
    assert "worker-pod" in runner.execs
    assert runner.execs.count("ingress-pod") <= 2, (
        "one retry at most, then the next role"
    )


def test_when_no_role_answers_the_error_names_every_role_tried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV, raising=False)

    with pytest.raises(GATE.InflightInstallsRefused) as failure:
        GATE.require_no_inflight_installs(_no_pod_release(), action="upgrade")

    message = str(failure.value)
    for name in INVENTORY.CPU_RUNTIME_DEPLOYMENTS:
        assert name in message


# --- where the orchestrator calls it -------------------------------------------


class _Reached(RuntimeError):
    pass


def _upgrade_double(calls: list[str], **stubs: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "config": SimpleNamespace(
            auto_rollback=True, schema_rollback_compatible=True, clusters=()
        ),
        "state": {},
        "_ensure_contexts": lambda: None,
        "_require_cpu_secrets": lambda: None,
        "_apply_rds_ca_bundle": lambda: None,
        "_refresh_aurora_credentials": lambda: None,
        "_remote_commands_are_idle": lambda: (calls.append("idle"), True)[1],
        "_require_no_inflight_installs": lambda **kwargs: calls.append(
            f"inflight-installs:{kwargs['action']}"
        ),
    }
    fields.update(stubs)
    return SimpleNamespace(**fields)


def test_upgrade_checks_for_in_flight_installs_before_capturing_previous() -> None:
    calls: list[str] = []
    release = _upgrade_double(
        calls, _capture_previous=lambda **_kwargs: (_ for _ in ()).throw(_Reached())
    )
    diff = DIFF.ReleaseDiff(
        kind=DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"control_plane_wheel"}),
    )

    with pytest.raises(_Reached):
        ORCHESTRATION.upgrade_release(release, diff=diff)

    assert calls == ["idle", "inflight-installs:upgrade"]


def test_a_refused_upgrade_writes_no_state(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    release = _upgrade_double(
        calls,
        _require_no_inflight_installs=lambda **_kwargs: (_ for _ in ()).throw(
            GATE.InflightInstallsRefused("upgrade refused: workflow-7f3a")
        ),
        _capture_previous=lambda **_kwargs: pytest.fail("previous was captured"),
    )
    diff = DIFF.ReleaseDiff(
        kind=DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"control_plane_wheel"}),
    )

    with pytest.raises(GATE.InflightInstallsRefused, match="workflow-7f3a"):
        ORCHESTRATION.upgrade_release(release, diff=diff)
    assert release.state == {}


def test_the_upgrade_keeps_the_gates_verdict_for_its_first_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rollback has persisted its verdict since fix round 3; the upgrade
    discarded it (``_upgrade_context`` rebinds ``release.state`` right after
    the gate). An upgrade that proceeded under consent now writes the
    ``overridden`` verdict -- the listed set it was consented over -- into the
    state before the ``preflight`` checkpoint (fix round 4, MEDIUM-3)."""

    calls: list[str] = []
    verdict = {
        "checked": True,
        "verdict": "overridden",
        "reason": FLAG,
        "steps": ["workflow-7f3a REMEDIATE_DRIVER step 3 on ip-10-0-1-17 (WAITING)"],
        "inflight_count": 1,
        "scanned": 4,
    }
    checkpoints: list[tuple[str, dict[str, Any]]] = []

    def save_state(phase: str, **_updates: Any) -> None:
        checkpoints.append((phase, dict(release.state)))
        raise _Reached()

    release = _upgrade_double(
        calls,
        _require_no_inflight_installs=lambda **_kwargs: dict(verdict),
        _save_state=save_state,
    )
    monkeypatch.setattr(
        ORCHESTRATION, "_validate_upgrade_transaction", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        ORCHESTRATION,
        "_upgrade_context",
        lambda *_a, **_k: ({"metadata": {}}, set(), set(), False),
    )
    diff = DIFF.ReleaseDiff(
        kind=DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"control_plane_wheel"}),
    )

    with pytest.raises(_Reached):
        ORCHESTRATION.upgrade_release(release, diff=diff)

    assert [phase for phase, _state in checkpoints] == ["preflight"]
    assert checkpoints[0][1].get("inflight_installs") == verdict, (
        "the verdict is in the state the first checkpoint persists"
    )
    assert release.state["inflight_installs"] == verdict


def _rollback_double(calls: list[str], **stubs: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "config": SimpleNamespace(auto_rollback=True, clusters=()),
        "state": {},
        "_refresh_aurora_credentials": lambda: calls.append("aurora-refresh"),
        "_require_no_inflight_installs": lambda **kwargs: calls.append(
            f"inflight-installs:{kwargs['action']}:{kwargs['unreadable']}"
        ),
        "_save_state": lambda phase, **_updates: calls.append(f"state:{phase}"),
    }
    fields.update(stubs)
    return SimpleNamespace(**fields)


def _plan_reached(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    def plan(*_args, **_kwargs):
        calls.append("compensation-plan")
        raise _Reached()

    monkeypatch.setattr(ORCHESTRATION, "build_rollback_compensation_plan", plan)


def test_rollback_checks_for_in_flight_installs_after_the_credential_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the Aurora refresh -- a rotation-window failure gets its chance to
    be named first -- and before any restore is planned. A manual rollback
    fails closed on a store that cannot answer."""

    calls: list[str] = []
    _plan_reached(monkeypatch, calls)

    with pytest.raises(_Reached):
        ORCHESTRATION.rollback_release(
            _rollback_double(calls), state={"metadata": {}, "cpu_wheel": "w"}
        )

    assert calls == [
        "aurora-refresh",
        "inflight-installs:rollback:refuse",
        "compensation-plan",
    ]


def test_the_automatic_rollback_tells_the_gate_to_proceed_on_an_unreadable_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _plan_reached(monkeypatch, calls)

    with pytest.raises(_Reached):
        ORCHESTRATION.rollback_release(
            _rollback_double(calls),
            state={"metadata": {}, "cpu_wheel": "w"},
            automatic=True,
        )

    assert calls == [
        "aurora-refresh",
        "inflight-installs:rollback:proceed",
        "compensation-plan",
    ]


def test_the_rollback_keeps_the_gates_verdict_for_its_first_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``inflight-installs-unchecked`` used to be one stderr line and the
    snapshot the gate returned was dropped at the call site. The verdict now
    lands in the transaction state before ``rollback-started`` is written, so
    the durable record says whether the rollback was checked, and on what
    (fix round 3, MEDIUM-3). ``render_persisted_state`` writes ``release.state``
    whole, so a top-level key is all that is needed."""

    calls: list[str] = []
    _plan_reached(monkeypatch, calls)
    verdict = {
        "checked": False,
        "verdict": "unchecked",
        "reason": "automatic rollback: no Running control-plane Pod could run the probe",
        "steps": [],
    }
    release = _rollback_double(
        calls, _require_no_inflight_installs=lambda **_kwargs: dict(verdict)
    )

    with pytest.raises(_Reached):
        ORCHESTRATION.rollback_release(
            release, state={"metadata": {}, "cpu_wheel": "w"}, automatic=True
        )

    assert release.state["inflight_installs"] == verdict


def test_a_rollback_re_entered_after_the_control_plane_restore_skips_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once ``rollback-cpu-restored`` is checkpointed the previous control plane
    is already the one dispatching; refusing the cleanup re-entry would only
    strand the transaction."""

    calls: list[str] = []
    _plan_reached(monkeypatch, calls)
    release = _rollback_double(
        calls,
        state={
            "rollback_completed_phases": ["rollback-started", "rollback-cpu-restored"]
        },
    )

    with pytest.raises(_Reached):
        ORCHESTRATION.rollback_release(
            release, state={"metadata": {}, "cpu_wheel": "w"}
        )

    assert calls == ["aurora-refresh", "compensation-plan"]


def test_a_refused_rollback_touches_nothing_after_the_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    release = _rollback_double(
        calls,
        _require_no_inflight_installs=lambda **_kwargs: (_ for _ in ()).throw(
            GATE.InflightInstallsRefused("rollback refused: workflow-7f3a")
        ),
    )
    monkeypatch.setattr(
        ORCHESTRATION,
        "build_rollback_compensation_plan",
        lambda *_a, **_k: pytest.fail("the rollback was planned"),
    )

    with pytest.raises(GATE.InflightInstallsRefused, match="workflow-7f3a"):
        ORCHESTRATION.rollback_release(release, state={"metadata": {}})
    assert calls == ["aurora-refresh"], "the credential refresh is not a restore"


def _failing_upgrade(
    monkeypatch: pytest.MonkeyPatch, rollback: Any
) -> tuple[SimpleNamespace, DIFF.ReleaseDiff]:
    calls: list[str] = []
    # Late-bound on purpose: ``upgrade_release`` rebinds ``release.state`` to a
    # fresh dict for a new transaction, so the stubs must read the attribute
    # at call time rather than capture the initial dict.
    release = _upgrade_double(
        calls,
        rollback=rollback,
        _save_state=lambda phase, **updates: release.state.update(
            {"phase": phase, **updates}
        ),
        _load_state=lambda: dict(release.state),
    )
    monkeypatch.setattr(
        ORCHESTRATION, "_validate_upgrade_transaction", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        ORCHESTRATION,
        "_upgrade_context",
        lambda *_a, **_k: ({"metadata": {}}, set(), set(), False),
    )
    monkeypatch.setattr(
        ORCHESTRATION,
        "run_upgrade_phases",
        lambda *_a, **_k: (_ for _ in ()).throw(
            MODULE.ReleaseError("cpu finalize barrier failed")
        ),
    )
    diff = DIFF.ReleaseDiff(
        kind=DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"control_plane_wheel"}),
    )
    return release, diff


def test_a_refused_automatic_rollback_is_logged_and_leaves_the_failed_phase(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Not ``rollback-failed``: nothing was restored, so the transaction stays
    where ``record_upgrade_failure`` put it -- ``failed``, which resume and
    ``--supersede-failed-transaction`` already know how to handle."""

    monkeypatch.delenv(ENV, raising=False)
    attempts: list[dict[str, Any]] = []

    def rollback(**kwargs):
        attempts.append(kwargs)
        raise GATE.InflightInstallsRefused(
            "rollback refused: workflow-7f3a REMEDIATE_DRIVER on ip-10-0-1-17"
        )

    release, diff = _failing_upgrade(monkeypatch, rollback)

    with pytest.raises(MODULE.ReleaseError) as failure:
        ORCHESTRATION.upgrade_release(release, diff=diff)

    message = str(failure.value)
    assert "automatic rollback" in message and "refused" in message
    assert "workflow-7f3a" in message
    assert "cpu finalize barrier failed" in message
    assert FLAG in message
    assert failure.value.__cause__ is not None
    assert len(attempts) == 1
    assert attempts[0]["automatic"] is True, (
        "the engine's rollback must know it is automatic (unreadable store → proceed)"
    )
    assert release.state["phase"] == "failed"
    assert release.state["release_lifecycle"] == "FAILED"
    assert "rollback_failure" not in release.state
    logged = capsys.readouterr().err
    assert "automatic-rollback-refused" in logged
    assert "workflow-7f3a" in logged


def test_a_rollback_that_fails_for_another_reason_still_records_rollback_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rollback(**_kwargs):
        raise MODULE.ReleaseError("previous Agent identities are unavailable")

    release, diff = _failing_upgrade(monkeypatch, rollback)

    with pytest.raises(MODULE.ReleaseError, match="rollback also failed"):
        ORCHESTRATION.upgrade_release(release, diff=diff)

    assert release.state["phase"] == "rollback-failed"


# --- the CLI ---------------------------------------------------------------------


def test_the_flag_travels_to_the_source_preparer_as_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        admin_cli, "run_source_deploy", lambda **kwargs: calls.append(kwargs) or 0
    )
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(admin_cli.ACCEPT_SCHEMA_CHANGE_ENV, raising=False)
    monkeypatch.delenv(admin_cli.SUPERSEDE_FAILED_TRANSACTION_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    base = [
        "deploy",
        "--cpu-cluster-arn",
        "arn:aws:eks:us-east-1:123456789012:cluster/cpu",
        "--gpu-cluster-arn",
        "arn:aws:eks:us-east-1:123456789012:cluster/gpu",
        "--state-dir",
        str(tmp_path / "state"),
        "--admin-email",
        "operations@example.com",
    ]

    assert admin_cli.run(admin_cli.parser().parse_args(base)) == 0
    assert ENV not in calls[-1]["extra_environment"], "no flag, no variable"
    assert admin_cli.run(admin_cli.parser().parse_args([*base, FLAG])) == 0
    assert calls[-1]["extra_environment"][ENV] == "1"
    assert GATE.inflight_installs_allowed(calls[-1]["extra_environment"]) is True
    assert GATE.inflight_installs_allowed({}) is False


def test_an_explicit_variable_wins_over_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV, "1")
    arguments = admin_cli.parser().parse_args(["deploy", FLAG, "--state-dir", "/tmp/x"])

    assert admin_cli.inflight_installs_environment(arguments) == {}


def test_the_flag_is_a_deploy_option_not_a_verb() -> None:
    """A flag on ``deploy``, not an eleventh-plus verb and not a global option."""

    parser = admin_cli.parser()
    arguments = parser.parse_args(["deploy", FLAG, "--state-dir", "/tmp/x"])
    assert arguments.allow_inflight_installs is True, "deploy accepts the flag"
    for argv in ([FLAG], [FLAG[2:]], ["status", FLAG]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


# --- the engine entrypoint --------------------------------------------------------


def test_the_engine_exits_with_the_refusal_code_the_driver_classifies_on(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``scripts/release_deploy.py`` sees only the exit code of
    ``rollout-regional-release.sh rollback``; a refusal must not look like the
    generic 2 that means "rollback failed" (F4)."""

    monkeypatch.setattr(
        MODULE,
        "parser",
        lambda: SimpleNamespace(
            parse_args=lambda argv=None: SimpleNamespace(
                mode="rollback", dry_run=False, config="/tmp/site.json"
            )
        ),
    )
    # The engine's first act is loading the config; raising there stands in for
    # the gate refusing a few calls later.
    monkeypatch.setattr(
        MODULE.ReleaseConfig,
        "load",
        classmethod(
            lambda cls, _path: (_ for _ in ()).throw(
                GATE.InflightInstallsRefused("rollback refused: workflow-7f3a")
            )
        ),
    )

    assert MODULE.main() == GATE.INFLIGHT_INSTALLS_REFUSED_EXIT_CODE
    assert GATE.INFLIGHT_INSTALLS_REFUSED_EXIT_CODE not in (0, 1, 2)
    assert "workflow-7f3a" in capsys.readouterr().err

    monkeypatch.setattr(
        MODULE.ReleaseConfig,
        "load",
        classmethod(
            lambda cls, _path: (_ for _ in ()).throw(MODULE.ReleaseError("broke"))
        ),
    )
    assert MODULE.main() == 2, "every other engine error keeps the generic code"


def test_the_rollback_mode_takes_an_automatic_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = MODULE.parser().parse_args(
        ["rollback", "--config", "/tmp/site.json", "--automatic"]
    )
    assert parsed.automatic is True, "the driver marks its rollback automatic"
    assert (
        MODULE.parser().parse_args(["rollback", "--config", "/tmp/x"]).automatic
        is False
    )

    seen: list[dict[str, Any]] = []
    release = SimpleNamespace(rollback=lambda **kwargs: seen.append(kwargs))
    monkeypatch.setattr(
        MODULE, "parser", lambda: SimpleNamespace(parse_args=lambda argv=None: parsed)
    )
    monkeypatch.setattr(MODULE.ReleaseConfig, "load", classmethod(lambda cls, _p: None))
    monkeypatch.setattr(MODULE, "RegionalRelease", lambda _config, _runner: release)

    assert MODULE.main() == 0
    assert seen == [{"automatic": True}]


def test_the_automatic_marker_is_engine_internal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Hidden from ``--help`` (it is set by ``recover_failed_upgrade`` and the
    release driver, never typed by an operator) and refused on every mode but
    ``rollback``, where it would otherwise be accepted silently."""

    assert "--automatic" not in MODULE.parser().format_help()

    parsed = MODULE.parse_arguments(["rollback", "--config", "/tmp/x", "--automatic"])
    assert parsed.automatic is True, "rollback keeps the marker"
    assert MODULE.parse_arguments(["upgrade", "--config", "/tmp/x"]).automatic is False

    for mode in ("upgrade", "deploy", "resume", "status"):
        with pytest.raises(SystemExit) as failure:
            MODULE.parse_arguments([mode, "--config", "/tmp/x", "--automatic"])
        assert failure.value.code == 2, "argparse usage error, like any bad argv"
        assert "--automatic" in capsys.readouterr().err
