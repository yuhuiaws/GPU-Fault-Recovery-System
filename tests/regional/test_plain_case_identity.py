"""Exercise real plain entrypoints with explicit, process-free collaborators."""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import acceptance_supervision, regional_commands
from scripts.e2e.regional import live_driver_guard as guard
from scripts.e2e.regional import plain_case_identity as plain
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.acceptance_scope import (
    EXECUTION_SCOPE_ENV,
    SELECTION_REFERENCE_ENV,
)
from scripts.e2e.regional.regional_case_contract import (
    case_evidence_path,
    formal_predecessor,
)
from scripts.e2e.regional.regional_live_fixture import (
    RUNTIME_IDENTITY_DEPLOYMENTS,
    RegionalFixtureAbort,
    RegionalLiveSettings,
    abort_on_signal,
    predecessor_evidence,
)
from scripts.e2e.regional.site_profile import SITE_PROFILE_ENV

PLAIN_RUNNERS = (
    "run_cmd017_barrier_hold",
    "run_cmd018_open_sibling_hold",
    "run_net006_lease_loss_withheld_result",
)


@pytest.fixture(autouse=True)
def isolated_process(monkeypatch: pytest.MonkeyPatch):
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("plain unit test attempted an external process")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(regional_commands, "run_command", refuse)
    monkeypatch.setattr(guard, "source_digest", lambda: "unit-source")
    for key in (SITE_PROFILE_ENV, EXECUTION_SCOPE_ENV, SELECTION_REFERENCE_ENV):
        monkeypatch.delenv(key, raising=False)
    original_umask = os.umask(0o077)
    handlers = {
        number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)
    }
    yield
    os.umask(original_umask)
    for number, handler in handlers.items():
        signal.signal(number, handler)


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RegionalLiveSettings:
    cpu, gpu = tmp_path / "cpu.kubeconfig", tmp_path / "gpu.kubeconfig"
    for path in (cpu, gpu):
        path.write_text("apiVersion: v1\nkind: Config\n")
    settings = RegionalLiveSettings(
        cpu_kubeconfig=cpu,
        gpu_kubeconfig=gpu,
        gpu_context="unit-gpu-context",
        namespace="gpu-fault-system",
        cluster_id="physical-unit",
        region="us-west-2",
    )
    for key, value in {
        **settings.environment(),
        "GPU_FAULT_CONTROL_KUBECONFIG": str(cpu),
        "KUBECONFIG": str(gpu),
        "GPU_FAULT_DATAPLANE_CONTEXT": settings.gpu_context,
        "GPU_FAULT_PERF_AWS_REGION": settings.region,
        "GPU_FAULT_PERF_CONTROL_NAMESPACE": settings.namespace,
        "GPU_FAULT_PERF_DATAPLANE_NAMESPACE": settings.namespace,
    }.items():
        monkeypatch.setenv(key, value)
    for name, fields in {
        "scripts.e2e.regional.seeded_command_fixture": (
            "AWS_REGION",
            "CONTROL_NAMESPACE",
            "NAMESPACE",
            "DATAPLANE_CONTEXT",
        ),
        "regional_capacity_registry": (
            "AWS_REGION",
            "CONTROL_NAMESPACE",
            "NAMESPACE",
            "DATAPLANE_CONTEXT",
            "CONTROL_KUBECONFIG",
            "IDENTITY_NAMESPACE",
        ),
        "regional_capacity_suite": (
            "AWS_REGION",
            "CONTROL_NAMESPACE",
            "NAMESPACE",
            "DATAPLANE_CONTEXT",
        ),
        "regional_action_capacity_suite": ("NAMESPACE",),
    }.items():
        module = importlib.import_module(name)
        values = {
            "AWS_REGION": settings.region,
            "CONTROL_NAMESPACE": settings.namespace,
            "NAMESPACE": settings.namespace,
            "IDENTITY_NAMESPACE": settings.namespace,
            "DATAPLANE_CONTEXT": settings.gpu_context,
            "CONTROL_KUBECONFIG": str(cpu),
        }
        for field in fields:
            monkeypatch.setattr(module, field, values[field])
    return settings


class IdentityProbe:
    def __init__(self, settings: RegionalLiveSettings) -> None:
        self.identity = {
            "release_id": "release-unit",
            "cluster_id": settings.cluster_id,
        }
        self.executor = {"cluster_id": settings.cluster_id}
        self.runtime: dict[str, Any] = {
            "release_state": {"release_id": "release-unit", "phase": "COMMITTED"},
            "deployments": {
                plane: {
                    name: {
                        "generation": 1,
                        "desired_replicas": 1,
                        "observed_generation": 1,
                        "updated_replicas": 1,
                        "ready_replicas": 1,
                        "available_replicas": 1,
                        "template_sha256": f"{plane}-{name}",
                    }
                    for name in names
                }
                for plane, names in RUNTIME_IDENTITY_DEPLOYMENTS.items()
            },
        }
        self.calls: list[str] = []

    def evidence_identity(self) -> dict[str, str]:
        self.calls.append("identity")
        return dict(self.identity)

    def executor_python(self, _script: str) -> dict[str, str]:
        self.calls.append("executor")
        return dict(self.executor)

    def runtime_identity(self) -> dict[str, Any]:
        self.calls.append("runtime")
        return json.loads(json.dumps(self.runtime))


class PlainHarness:
    def __init__(
        self,
        module_name: str,
        root: Path,
        monkeypatch: pytest.MonkeyPatch,
        probe: IdentityProbe,
    ) -> None:
        self.module = importlib.import_module(f"scripts.e2e.regional.{module_name}")
        self.run_dir = root / "run"
        self.monkeypatch = monkeypatch
        self.probe = probe
        self.body_calls: list[tuple[Path, int, datetime]] = []
        self.body_result: dict[str, Any] | None = {
            "case_id": self.module.CASE_ID,
            "attempt": 1,
            "verdict": "PASS",
            "errors": [],
            "preflight": {"cluster_id": "perf-cap-000"},
            "seed": {"cluster_id": "perf-cap-000"},
        }
        self.body_code = 0
        self.body_error: BaseException | None = None
        monkeypatch.setattr(
            self.module, "CASE", replace(self.module.CASE, run_case=self.body)
        )
        self.predecessor = formal_predecessor(self.module.CASE_ID)
        assert self.predecessor is not None, (
            "every plain runner needs a formal predecessor"
        )
        self.predecessor_path = case_evidence_path(self.run_dir, self.predecessor)
        write_json_atomic(
            self.predecessor_path,
            {
                "case_id": self.predecessor,
                "verdict": "PASS",
                "status": "COMPLETED",
                **probe.identity,
            },
        )

    @property
    def case_dir(self) -> Path:
        return self.run_dir / "cases" / self.module.CASE_ID

    def result(self) -> dict[str, Any]:
        return json.loads(
            case_evidence_path(self.run_dir, self.module.CASE_ID).read_text()
        )

    def plan(self) -> dict[str, Any]:
        return json.loads((self.case_dir / "plan.json").read_text())

    def invoke(self, *, execute: bool = False, extra: tuple[str, ...] = ()) -> int:
        argv = [self.module.__file__, "--run-dir", str(self.run_dir), "--attempt", "1"]
        if execute:
            argv += [
                "--execute",
                "--confirm",
                self.module.CONFIRMATION,
                "--maintenance-window-end",
                (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
            ]
        else:
            argv.append("--plan")
        self.monkeypatch.setattr(sys, "argv", [*argv, *extra])
        return self.module.main()

    def body(self, run_dir: Path, attempt: int, deadline: datetime) -> int:
        self.body_calls.append((run_dir, attempt, deadline))
        if self.body_result is not None:
            write_json_atomic(
                case_evidence_path(run_dir, self.module.CASE_ID), self.body_result
            )
        if self.body_error is not None:
            raise self.body_error
        return self.body_code


@pytest.fixture
def plain_harness(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    site: RegionalLiveSettings,
) -> PlainHarness:
    probe = IdentityProbe(site)
    monkeypatch.setattr(plain, "RegionalLiveFixture", lambda _settings: probe)
    return PlainHarness(
        getattr(request, "param", PLAIN_RUNNERS[0]), tmp_path, monkeypatch, probe
    )


@pytest.mark.parametrize("plain_harness", PLAIN_RUNNERS, indirect=True)
def test_real_plain_plan_execute_roundtrip(plain_harness: PlainHarness) -> None:
    case = plain_harness
    assert case.invoke() == 0, (
        "a real read-only identity and predecessor gate should plan"
    )
    plan = case.plan()
    assert plan["preflight_passed"] is True, (
        "only an empty error list authorizes execution"
    )
    assert plan["details"]["preflight"]["errors"] == [], (
        "the preflight must be recorded"
    )
    assert case.body_calls == [], "planning must not reach the case body"
    assert case.probe.calls == ["identity", "executor", "runtime"], (
        "plan must read live identity"
    )
    assert case.result()["verdict"] == "NOT_RUN", "planning is never execution evidence"
    assert signal.getsignal(signal.SIGTERM) is abort_on_signal, "termination must abort"
    assert signal.getsignal(signal.SIGINT) is abort_on_signal, "interrupt must abort"
    assert case.invoke(execute=True) == 0, "matching current identity should execute"
    assert len(case.body_calls) == 1, "exactly one case execution is authorized"
    assert case.body_calls[0][:2] == (case.run_dir, 1), (
        "the legacy signature must be retained"
    )
    assert case.probe.calls == ["identity", "executor", "runtime"] * 2, (
        "execute must recheck"
    )
    result = case.result()
    assert result["release_id"] == "release-unit", (
        "result must identify the current release"
    )
    assert result["cluster_id"] == "physical-unit", "canonical cluster must be physical"
    assert result["synthetic_cluster_id"] == "perf-cap-000", (
        "synthetic identity stays separate"
    )
    assert result["seed"]["cluster_id"] == "perf-cap-000", (
        "body evidence must not be relabelled"
    )
    assert result["preflight"]["cluster_id"] == "perf-cap-000", (
        "body preflight remains intact"
    )
    assert (
        predecessor_evidence(
            case_evidence_path(case.run_dir, case.module.CASE_ID),
            case.module.CASE_ID,
            release_id="release-unit",
            cluster_id="physical-unit",
        )["valid"]
        is True
    ), "a completed bound PASS may satisfy its successor"
    assert (case.case_dir / "plan.json").stat().st_mode & 0o777 == 0o600, (
        "plan stays private"
    )


@pytest.mark.parametrize(
    "problem",
    ["missing", "wrong-release", "wrong-cluster", "partial", "running", "selective"],
)
def test_plain_plan_refuses_invalid_predecessor(
    plain_harness: PlainHarness, problem: str
) -> None:
    case = plain_harness
    if problem == "missing":
        case.predecessor_path.unlink()
    else:
        previous = json.loads(case.predecessor_path.read_text())
        previous.update(
            {
                "wrong-release": {"release_id": "other-release"},
                "wrong-cluster": {"cluster_id": "other-physical"},
                "partial": {"verdict": "PARTIAL"},
                "running": {"status": "RUNNING"},
                "selective": {
                    "execution_scope": "selective",
                    "formal_sequence_satisfied": False,
                },
            }[problem]
        )
        write_json_atomic(case.predecessor_path, previous)
    assert case.invoke() == 1, "unusable predecessor evidence must fail the plan"
    assert case.plan()["preflight_passed"] is False, "failed gates cannot be approved"
    assert case.plan()["details"]["preflight"]["errors"], (
        "the actual gate failure must be recorded"
    )
    assert case.body_calls == [], "no body may run when the predecessor is invalid"
    with pytest.raises(RuntimeError, match="did not pass"):
        case.invoke(execute=True)


@pytest.mark.parametrize("problem", ["release", "cluster", "executor", "runtime"])
def test_plain_plan_refuses_missing_or_inconsistent_identity(
    plain_harness: PlainHarness, problem: str
) -> None:
    case = plain_harness
    if problem in {"release", "cluster"}:
        case.probe.identity[f"{problem}_id"] = ""
    elif problem == "executor":
        case.probe.executor["cluster_id"] = "some-other-cluster"
    else:
        case.probe.runtime["deployments"] = {}
    assert case.invoke() == 1, "incomplete deployed identity must fail closed"
    assert case.plan()["preflight_passed"] is False, (
        "identity failure must reach the plan"
    )
    assert case.body_calls == [], "unverified identity cannot reach the body"


@pytest.mark.parametrize(
    "drift", ["release", "executor", "runtime", "predecessor", "target"]
)
def test_plain_execute_rechecks_current_identity_and_predecessor(
    plain_harness: PlainHarness, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    case = plain_harness
    assert case.invoke() == 0, "initial identity must be approvable"
    if drift == "release":
        case.probe.identity["release_id"] = "new-release"
        case.probe.runtime["release_state"]["release_id"] = "new-release"
        previous = json.loads(case.predecessor_path.read_text())
        write_json_atomic(
            case.predecessor_path, {**previous, "release_id": "new-release"}
        )
    elif drift == "executor":
        case.probe.executor["cluster_id"] = "other-physical"
    elif drift == "runtime":
        case.probe.runtime["deployments"]["gpu"]["gpu-fault-cluster-executor"][
            "template_sha256"
        ] = "new"
    elif drift == "predecessor":
        case.predecessor_path.unlink()
    else:
        registry = importlib.import_module("regional_capacity_registry")
        monkeypatch.setattr(registry, "NAMESPACE", "other-namespace")
    with pytest.raises(RuntimeError, match="preflight failed|drifted at details"):
        case.invoke(execute=True)
    assert case.body_calls == [], "current drift must be refused before execution"
    assert case.result()["verdict"] == "FAIL", (
        "drift must invalidate any canonical PASS"
    )


@pytest.mark.parametrize(
    "drift", ["arguments", "source", "kubeconfig", "environment", "profile"]
)
def test_plain_preserves_plan3_guards(
    plain_harness: PlainHarness,
    monkeypatch: pytest.MonkeyPatch,
    site: RegionalLiveSettings,
    drift: str,
) -> None:
    case = plain_harness
    assert case.invoke() == 0, "initial plan should pass"
    extra: tuple[str, ...] = ()
    if drift == "arguments":
        extra = ("--attempt", "2")
    elif drift == "source":
        monkeypatch.setattr(guard, "source_digest", lambda: "changed-source")
    elif drift == "kubeconfig":
        site.gpu_kubeconfig.write_text("apiVersion: v1\nkind: Config\nclusters: []\n")
    elif drift == "environment":
        monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "other-physical")
    else:
        monkeypatch.setattr(
            guard, "applied_site_profile", lambda: {"sha256": "other-profile"}
        )
    with pytest.raises(RuntimeError, match="drifted at"):
        case.invoke(execute=True, extra=extra)
    assert case.probe.calls == ["identity", "executor", "runtime"], (
        "approval drift blocks probes"
    )
    assert case.body_calls == [], "Plan 3 refusal must precede the body"


def test_plain_supervision_loss_refuses_replan_and_execute(
    plain_harness: PlainHarness,
) -> None:
    case = plain_harness
    assert case.invoke() == 0, "initial plan should pass"
    acceptance_supervision.bind_command_supervision(case.run_dir)
    acceptance_supervision.record_supervision_loss()
    for execute in (False, True):
        with pytest.raises(RuntimeError, match="lost command supervision"):
            case.invoke(execute=execute)
    assert case.probe.calls == ["identity", "executor", "runtime"], (
        "durable refusal precedes probes"
    )
    assert case.body_calls == [], "no execution is allowed after lost supervision"


def test_plain_execute_without_plan_never_probes(plain_harness: PlainHarness) -> None:
    with pytest.raises(RuntimeError, match="run --plan"):
        plain_harness.invoke(execute=True)
    assert plain_harness.probe.calls == [], (
        "missing approval must be refused before probing"
    )
    assert plain_harness.body_calls == [], "missing approval must not invoke the body"


@pytest.mark.parametrize(
    "verdict,code", [("FAIL", 5), ("PARTIAL", 7), ("PASS", 9), ("FAIL", 0)]
)
def test_plain_preserves_failed_and_partial_body_results(
    plain_harness: PlainHarness, verdict: str, code: int
) -> None:
    case = plain_harness
    assert case.invoke() == 0, "initial plan should pass"
    case.body_result = {
        "case_id": case.module.CASE_ID,
        "verdict": verdict,
        "error": "body error",
        "cleanup_error": "cleanup observation",
        "partial_observations": [{"claimed": False}],
    }
    case.body_code = code
    assert case.invoke(execute=True) == (code or 1), (
        "a body failure cannot return success"
    )
    result = case.result()
    assert result["verdict"] == ("FAIL" if verdict == "PASS" else verdict), (
        "failed verdict must survive"
    )
    assert result["error"] == "body error", (
        "the body error must survive canonical binding"
    )
    assert result["cleanup_error"] == "cleanup observation", (
        "cleanup evidence must be retained"
    )
    assert result["partial_observations"] == [{"claimed": False}], (
        "partial observations must survive"
    )
    assert result["release_id"] == "release-unit", (
        "failure evidence still needs release identity"
    )
    assert result["cluster_id"] == "physical-unit", (
        "failure evidence still needs physical identity"
    )


@pytest.mark.parametrize("signal_number", [None, signal.SIGTERM, signal.SIGINT])
def test_plain_preserves_partial_result_on_exception_or_abort(
    plain_harness: PlainHarness, signal_number: int | None
) -> None:
    case = plain_harness
    assert case.invoke() == 0, "initial plan should pass"
    case.body_result = {
        "case_id": case.module.CASE_ID,
        "verdict": "PARTIAL",
        "cleanup": {"completed": False},
        "error": "case observation",
    }
    error = (
        RuntimeError("body failed")
        if signal_number is None
        else RegionalFixtureAbort(signal_number)
    )
    case.body_error = error
    with pytest.raises(type(error)):
        case.invoke(execute=True)
    result = case.result()
    assert result["verdict"] == "PARTIAL", (
        "an abort cannot discard a recorded partial result"
    )
    assert result["cleanup"] == {"completed": False}, (
        "partial cleanup evidence must survive"
    )
    assert result["error"] == "case observation", (
        "guard errors must not replace body errors"
    )
    assert result["status"] == "FAILED", (
        "an interrupted result cannot be completed evidence"
    )
    assert result["cluster_id"] == "physical-unit", (
        "abort evidence must bind the physical cluster"
    )


@pytest.mark.parametrize(
    "result_kind", ["missing", "wrong-case", "wrong-release", "synthetic-cluster"]
)
def test_plain_refuses_missing_or_misbound_body_result(
    plain_harness: PlainHarness, result_kind: str
) -> None:
    case = plain_harness
    assert case.invoke() == 0, "initial plan should pass"
    if result_kind == "missing":
        case.body_result = None
    else:
        assert case.body_result is not None, "fixture begins with a result"
        case.body_result.update(
            {
                "wrong-case": {"case_id": "other-case"},
                "wrong-release": {"release_id": "other-release"},
                "synthetic-cluster": {"cluster_id": "perf-cap-000"},
            }[result_kind]
        )
    with pytest.raises(RuntimeError, match="completed result|identity differs"):
        case.invoke(execute=True)
    assert case.result()["verdict"] == "FAIL", (
        "unbound or absent body evidence cannot pass"
    )


def test_plain_failed_status_cannot_be_promoted_to_pass(
    plain_harness: PlainHarness,
) -> None:
    case = plain_harness
    assert case.invoke() == 0, "initial plan should pass"
    assert case.body_result is not None, "fixture begins with a result"
    case.body_result["status"] = "FAILED"
    assert case.invoke(execute=True) == 1, "failed status must veto a nominal PASS"
    assert case.result()["verdict"] == "FAIL", (
        "canonical binding cannot promote failed evidence"
    )


@pytest.mark.parametrize(
    "field,value",
    [("formal_sequence_satisfied", False), ("execution_scope", "selective")],
)
def test_plain_body_cannot_upgrade_nonformal_evidence(
    plain_harness: PlainHarness, field: str, value: object
) -> None:
    case = plain_harness
    assert case.invoke() == 0, "initial plan should pass"
    assert case.body_result is not None, "fixture begins with a result"
    case.body_result[field] = value
    if field == "execution_scope":
        with pytest.raises(RuntimeError, match="result scope differs"):
            case.invoke(execute=True)
    else:
        assert case.invoke(execute=True) == 1, (
            "explicitly nonformal evidence cannot pass formally"
        )
    result = case.result()
    assert result["verdict"] == "FAIL", (
        "body evidence cannot be upgraded to a formal PASS"
    )
    assert result["formal_sequence_satisfied"] is False, (
        "body scope refusal must be retained"
    )


def test_plain_retry_cannot_reuse_an_old_pass(plain_harness: PlainHarness) -> None:
    case = plain_harness
    assert case.invoke() == 0, "initial plan should pass"
    assert case.invoke(execute=True) == 0, "the initial body writes a current PASS"
    case.body_result = None
    with pytest.raises(RuntimeError, match="completed result"):
        case.invoke(execute=True)
    assert case.result()["verdict"] == "FAIL", (
        "a missing retry result cannot reuse an older PASS"
    )


def test_plain_identity_read_failure_is_a_failed_plan(
    plain_harness: PlainHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = plain_harness

    def fail_identity() -> dict[str, str]:
        raise regional_commands.RegionalFixtureError("identity read unavailable")

    monkeypatch.setattr(case.probe, "evidence_identity", fail_identity)
    assert case.invoke() == 1, "identity read failure must be a failed preflight"
    assert case.plan()["preflight_passed"] is False, (
        "read errors cannot produce approval"
    )
    assert case.body_calls == [], "read failures must not reach the body"


def test_plain_old_pass_is_invalidated_before_configuration_failure(
    plain_harness: PlainHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = plain_harness
    old = {
        "case_id": case.module.CASE_ID,
        "verdict": "PASS",
        "release_id": "old-release",
        "cluster_id": "old-physical",
        "partial_observation": {"retained": True},
    }
    write_json_atomic(case_evidence_path(case.run_dir, case.module.CASE_ID), old)
    monkeypatch.delenv("GPU_FAULT_CLUSTER_ID")
    with pytest.raises(RuntimeError, match="cluster ID is required"):
        case.invoke()
    result = case.result()
    assert result["verdict"] == "FAIL", (
        "configuration failure must invalidate the old PASS"
    )
    assert "release_id" not in result, (
        "old release identity cannot describe a failed new attempt"
    )
    history = list((case.case_dir / "plain-history").glob("*.json"))
    assert len(history) == 1, "the earlier evidence must be retained privately"
    assert json.loads(history[0].read_text())["previous_result"] == old, (
        "prior observations are preserved"
    )
    assert case.probe.calls == [], "configuration failure must not make live calls"


@pytest.mark.parametrize(
    "preflight", [{}, {"errors": None}, {"errors": "none"}, {"errors": [False]}]
)
def test_plain_never_defaults_missing_errors_to_success(
    plain_harness: PlainHarness,
    monkeypatch: pytest.MonkeyPatch,
    preflight: dict[str, Any],
) -> None:
    case = plain_harness
    monkeypatch.setattr(
        case.module,
        "CASE",
        replace(case.module.CASE, read_only_preflight=lambda *_args: preflight),
    )
    with pytest.raises(RuntimeError, match="errors are missing or malformed"):
        case.invoke()
    assert case.result()["verdict"] == "FAIL", "malformed preflight cannot leave a PASS"
    assert case.body_calls == [], "malformed preflight cannot reach execution"


def test_plain_empty_errors_without_identity_proof_is_not_success(
    plain_harness: PlainHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = plain_harness
    monkeypatch.setattr(
        case.module,
        "CASE",
        replace(case.module.CASE, read_only_preflight=lambda *_args: {"errors": []}),
    )
    with pytest.raises(RuntimeError, match="lacks release/cluster/predecessor proof"):
        case.invoke()
    assert case.result()["verdict"] == "FAIL", (
        "empty errors alone cannot invent identity proof"
    )


def test_plain_requires_explicit_preflight_and_configuration_callbacks(
    plain_harness: PlainHarness,
) -> None:
    case = plain_harness.module.CASE
    with pytest.raises(TypeError, match="configure.*read_only_preflight"):
        guard.PlainCaseRunner(  # type: ignore[call-arg]
            case_id=case.case_id,
            confirmation=case.confirmation,
            parser=case.parser,
            plan_details=case.plan_details,
            run_case=case.run_case,
        )


def test_plain_preflight_requires_an_explicit_environment_reader(
    site: RegionalLiveSettings, tmp_path: Path
) -> None:
    with pytest.raises(TypeError, match="read_environment"):
        plain.plain_case_preflight(site, tmp_path)  # type: ignore[call-arg]


def test_plain_empty_environment_callback_never_reaches_live_probes(
    plain_harness: PlainHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = plain_harness
    monkeypatch.setattr(
        case.module,
        "CASE",
        replace(
            case.module.CASE,
            read_only_preflight=lambda settings, case_dir: plain.plain_case_preflight(
                settings, case_dir, read_environment=lambda: {}
            ),
        ),
    )
    assert case.invoke() == 1, "an incomplete environment callback must fail planning"
    assert case.probe.calls == [], (
        "empty environment cannot trigger live identity reads"
    )
    assert case.plan()["preflight_passed"] is False, (
        "missing environment is not approval"
    )


@pytest.mark.parametrize("target", ["environment", "imported-target"])
def test_plain_target_mismatch_is_refused_before_identity_read(
    plain_harness: PlainHarness, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    case = plain_harness
    if target == "environment":
        monkeypatch.setenv("GPU_FAULT_PERF_CONTROL_NAMESPACE", "other-namespace")
    else:
        registry = importlib.import_module("regional_capacity_registry")
        monkeypatch.setattr(registry, "CONTROL_KUBECONFIG", "/unselected/config")
    assert case.invoke() == 1, "target disagreement must fail the plan"
    assert case.probe.calls == [], (
        "identity must not be read against unbound execution targets"
    )
    assert case.plan()["preflight_passed"] is False, (
        "target disagreement is a failed preflight"
    )


def test_plain_explicit_identity_flags_are_not_silently_retargeted(
    plain_harness: PlainHarness,
) -> None:
    case = plain_harness
    assert case.invoke(extra=("--gpu-context", "other-context")) == 1, (
        "CLI and body target must agree"
    )
    assert case.probe.calls == [], (
        "unmatched CLI identity cannot probe the original cluster"
    )


def test_plain_selective_execution_never_satisfies_formal_sequence(
    plain_harness: PlainHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = plain_harness
    case.predecessor_path.unlink()
    monkeypatch.setenv(EXECUTION_SCOPE_ENV, "selective")
    monkeypatch.setenv(SELECTION_REFERENCE_ENV, "UNIT-PLAIN")
    assert case.invoke() == 0, (
        "explicit selective scope uses the shared predecessor policy"
    )
    assert case.invoke(execute=True) == 0, (
        "selective execution can retain its local PASS"
    )
    result = case.result()
    assert result["verdict"] == "PASS", "the selective body result should be retained"
    assert result["formal_sequence_satisfied"] is False, (
        "selective PASS is not formal evidence"
    )
    monkeypatch.delenv(EXECUTION_SCOPE_ENV)
    monkeypatch.delenv(SELECTION_REFERENCE_ENV)
    assert (
        predecessor_evidence(
            case_evidence_path(case.run_dir, case.module.CASE_ID),
            case.module.CASE_ID,
            release_id="release-unit",
            cluster_id="physical-unit",
        )["valid"]
        is False
    ), "a later formal case must reject selective evidence"
