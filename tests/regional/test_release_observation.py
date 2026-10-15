"""An observation the engine could not make is retried, never judged.

Live on 2026-10-15 a `gpu-fault-admin config` rolled its change cleanly, then
the release driver's `verify` and `stability` both died two seconds in on
`exec credential plugin exited 253` and the automatic rollback took the healthy
change back. These cases pin the line between "the engine could not
authenticate to or reach an API" (take the observation again, within the
stability window's budget) and everything else (a finding, judged at once),
and that a spent budget still fails closed, saying health was not observed.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import boto3
import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]
from botocore.credentials import ReadOnlyCredentials
from botocore.exceptions import CredentialRetrievalError

from gpu_fault.admin.command_log import FAILURE_EXIT_CODE_ATTRIBUTE
from gpu_fault.admin.native_http import NativeHttpError
from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_adot_self_metrics as ADOT
from gpu_fault_release import regional_aurora_credentials as AURORA
from gpu_fault_release import regional_kubeconfig_cache as CACHE
from gpu_fault_release import regional_release_observation as OBS
from gpu_fault_release import regional_release_probes as PROBES
from gpu_fault_release import regional_release_validation as VALIDATION
from gpu_fault_release import rollout as ROLLOUT
from gpu_fault_release.regional_release_config import CheckFailure, ReleaseError
from gpu_fault_release.regional_release_store_preflight import (
    INFLIGHT_INSTALLS_REFUSED_EXIT_CODE,
)
from gpu_fault_release.regional_resource_probe import ResourceRef, probe_resource
from scripts import release_deploy
from tests.regional._release_orchestrator_support import config_file

UNREACHABLE = "Unable to connect to the server: dial tcp 10.0.0.1:443: i/o timeout"
NO_CREDENTIALS = (
    "Unable to locate credentials. You can configure credentials by running "
    '"aws configure".'
)


def _unavailable(reason: str = "network") -> OBS.ApiUnavailable:
    return OBS.ApiUnavailable(f"command failed (1): kubectl: {reason}", reason=reason)


# --- classification -------------------------------------------------------


@pytest.mark.parametrize(
    "text,reason",
    [
        (NO_CREDENTIALS, "no-credentials"),
        (
            "Partial credentials found in env, missing: AWS_SECRET_ACCESS_KEY",
            "no-credentials",
        ),
        ("AWS credentials are unavailable for AMP alert query", "no-credentials"),
        ("The SSO session associated with this profile has expired", "expired"),
        ("An error occurred (ExpiredTokenException) when calling", "expired"),
        ("An error occurred (AccessDeniedException) when calling", "access-denied"),
        ("is not authorized to perform: eks:DescribeCluster", "access-denied"),
        ("An error occurred (ThrottlingException): Rate exceeded", "throttled"),
        ("the server has received too many requests and has asked us", "throttled"),
        ("An error occurred (ServiceUnavailable) when calling", "throttled"),
        ('Could not connect to the endpoint URL: "https://eks.example/"', "network"),
        ("Connect timeout on endpoint URL", "network"),
        ("Connection was closed before we received a valid response", "network"),
        ("dial tcp: lookup api.example on 10.0.0.2:53: no such host", "network"),
        ("net/http: TLS handshake timeout", "network"),
        ("<urlopen error [Errno -3] Temporary failure in name resolution>", "network"),
        ("x509: certificate signed by unknown authority", "unknown"),
        ("", "unknown"),
        (None, "unknown"),
    ],
)
def test_a_failure_is_named_with_one_fixed_word(text: str | None, reason: str) -> None:
    assert OBS.failure_reason(text) == reason
    assert reason in OBS.REASONS


def test_a_definitive_cause_wins_over_the_transport_words_around_it() -> None:
    combined = f"{UNREACHABLE}\nAn error occurred (ExpiredToken): expired"
    assert OBS.failure_reason(combined) == "expired"


def _failure(code: int = 1) -> ReleaseError:
    error = ReleaseError(f"command failed ({code}): kubectl get pods")
    setattr(error, FAILURE_EXIT_CODE_ATTRIBUTE, code)
    return error


@pytest.mark.parametrize(
    "arguments,stderr,reason",
    [
        (["kubectl", "get", "pods"], UNREACHABLE, "network"),
        (
            ["kubectl", "get", "pods"],
            NO_CREDENTIALS
            + "\nUnable to connect to the server: getting credentials: exec: "
            "executable aws failed with exit code 253",
            "no-credentials",
        ),
        (
            ["kubectl", "get", "pods"],
            "Unable to connect to the server: getting credentials: exec: executable "
            "aws failed with exit code 254\nAn error occurred (AccessDenied)",
            "access-denied",
        ),
        (
            ["/usr/local/bin/kubectl", "get", "pods"],
            "Error from server (TooManyRequests): the server has received too many "
            "requests",
            "throttled",
        ),
        (["aws", "eks", "describe-cluster"], NO_CREDENTIALS, "no-credentials"),
        (["aws", "amp", "list-workspaces"], "Rate exceeded (Throttling)", "throttled"),
        (
            ["aws", "sts", "get-caller-identity"],
            "Read timeout on endpoint URL",
            "network",
        ),
    ],
)
def test_a_client_that_never_got_an_answer_is_unavailable(
    arguments: list[str], stderr: str, reason: str
) -> None:
    original = _failure()
    promoted = OBS.unavailable_failure(original, arguments, stderr)

    assert isinstance(promoted, OBS.ApiUnavailable), "the failure stayed a verdict"
    assert promoted.reason == reason
    assert str(promoted) == str(original), "the operator's line must not change"
    assert getattr(promoted, FAILURE_EXIT_CODE_ATTRIBUTE) == 1, (
        "callers read the child's exit status from the error"
    )
    assert CheckFailure.details_of(promoted) == {"unobservable": reason}, (
        "a check failing with it must record why it was not observed"
    )


@pytest.mark.parametrize(
    "arguments,stderr",
    [
        # RBAC and absence are answers from the API, not a missing one.
        (
            ["kubectl", "get", "pods"],
            "Error from server (Forbidden): pods is forbidden",
        ),
        (
            ["kubectl", "get", "pods"],
            'Error from server (NotFound): pods "x" not found',
        ),
        (
            ["kubectl", "get", "pods"],
            "error: You must be logged in to the server (Unauthorized)",
        ),
        # A trust failure needs an operator, not another attempt.
        (
            ["kubectl", "get", "pods"],
            "Unable to connect to the server: x509: certificate signed by unknown "
            "authority",
        ),
        # The remote command ran: what it printed is the Pod's finding.
        (
            ["kubectl", "exec", "pod", "--", "python"],
            "psycopg.OperationalError: connection refused\n"
            "command terminated with exit code 1",
        ),
        (["aws", "eks", "describe-cluster"], "An error occurred (AccessDenied)"),
        (["aws", "eks", "describe-cluster"], "An error occurred (ExpiredToken)"),
        (["aws", "eks", "describe-cluster"], "An error occurred (ResourceNotFound)"),
        (["python3", "-c", "probe"], NO_CREDENTIALS),
        (["kubectl", "get", "pods"], None),
    ],
)
def test_a_finding_stays_a_plain_failure(
    arguments: list[str], stderr: str | None
) -> None:
    original = _failure()
    assert OBS.unavailable_failure(original, arguments, stderr) is original


def test_the_runner_reports_an_unreachable_api_server_as_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        ROLLOUT,
        "run_command",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            list(arguments), 1, "", UNREACHABLE
        ),
    )
    runner = ROLLOUT.Runner()

    with pytest.raises(OBS.ApiUnavailable) as caught:
        runner.run(["kubectl", "get", "pods"], capture=True)
    assert caught.value.reason == "network"
    assert str(caught.value).startswith("command failed (1): kubectl get pods"), (
        "the operator's line must stay the runner's own"
    )

    monkeypatch.setattr(
        ROLLOUT,
        "run_command",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            list(arguments), 1, "", "Error from server (Forbidden): forbidden"
        ),
    )
    with pytest.raises(ReleaseError) as refused:
        runner.run(["kubectl", "get", "pods"], capture=True)
    assert type(refused.value) is ReleaseError, "an RBAC refusal is a finding"
    capsys.readouterr()


def _probe_runner(result: Any) -> Any:
    def probe_output(arguments: list[str], **_kwargs: Any) -> Any:
        if isinstance(result, Exception):
            raise result
        return result

    return SimpleNamespace(probe_output=probe_output)


POD = ResourceRef("pod", "Pod", "api-0", "gpu-fault-system")


def test_a_three_state_read_keeps_why_it_got_no_answer() -> None:
    unreachable = probe_resource(_probe_runner((1, "", UNREACHABLE)), ["kubectl"], POD)
    assert unreachable.unavailable == "network"
    with pytest.raises(OBS.ApiUnavailable, match="cannot read pod/api-0"):
        unreachable.require_readable()

    refused = probe_resource(
        _probe_runner((1, "", "Error from server (Forbidden): forbidden")),
        ["kubectl"],
        POD,
    )
    assert refused.unavailable is None
    with pytest.raises(ReleaseError) as caught:
        refused.require_readable()
    assert type(caught.value) is ReleaseError, "a refusal is a finding"

    fetch = probe_resource(
        _probe_runner(_unavailable("no-credentials")), ["kubectl"], POD
    )
    assert fetch.unavailable == "no-credentials", (
        "a credential fetch that failed before the read is still unavailable"
    )


def test_an_unreadable_refresh_cronjob_is_unobservable_not_failed() -> None:
    release = SimpleNamespace(
        runner=_probe_runner((1, "", UNREACHABLE)),
        _cpu=lambda *arguments: ["kubectl", *arguments],
        config=SimpleNamespace(namespace="gpu-fault-system"),
    )
    with pytest.raises(OBS.ApiUnavailable, match="state is unknown"):
        AURORA.read_aurora_refresh_cronjob(release)


AMP_RELEASE = SimpleNamespace(
    config=SimpleNamespace(
        aws_region="us-west-2",
        health=SimpleNamespace(amp_workspace_id="workspace-example"),
    )
)
EXPORTED = {
    "AccessKeyId": "AKIDEXAMPLE",
    "SecretAccessKey": "example",
    "SessionToken": "",
}


def test_an_amp_query_that_got_no_answer_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def export(**_kwargs: Any) -> dict[str, str]:
        raise NativeHttpError("credential resolution: Unable to locate credentials")

    monkeypatch.setattr(ADOT, "export_aws_credentials", export)
    with pytest.raises(OBS.ApiUnavailable) as caught:
        ADOT.amp_instant_query(AMP_RELEASE, "up")
    assert caught.value.reason == "no-credentials"
    assert str(caught.value) == "AMP query transport or credential resolution failed"

    monkeypatch.setattr(ADOT, "export_aws_credentials", lambda **_kwargs: EXPORTED)
    for status, expected in ((503, OBS.ApiUnavailable), (429, OBS.ApiUnavailable)):
        monkeypatch.setattr(
            ADOT, "http_request", lambda *_a, status=status, **_k: (status, "")
        )
        with pytest.raises(expected, match=f"HTTP {status}"):
            ADOT.amp_instant_query(AMP_RELEASE, "up")
    monkeypatch.setattr(ADOT, "http_request", lambda *_a, **_k: (403, ""))
    with pytest.raises(ReleaseError) as refused:
        ADOT.amp_instant_query(AMP_RELEASE, "up")
    assert type(refused.value) is ReleaseError, "a 403 is a finding about the role"


# --- the budget -------------------------------------------------------------


def _flaky(failures: int, value: Any = "observed") -> tuple[list[int], Any]:
    calls: list[int] = []

    def observe() -> Any:
        calls.append(1)
        if len(calls) <= failures:
            raise _unavailable()
        return value

    return calls, observe


def test_the_budget_takes_an_unobserved_observation_again(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pauses: list[float] = []
    budget = OBS.ObservationBudget(120, mode="verify", sleep=pauses.append)
    calls, observe = _flaky(2)

    assert budget.attempt(observe, what="report") == "observed"
    assert len(calls) == 3, "each failure costs one more attempt, no more"
    assert pauses == list(OBS.OBSERVATION_RETRY_DELAYS[:2]), "pauses back off"
    assert budget.retries == 2
    assert budget.unobserved >= sum(pauses), "every pause is charged"
    summary = budget.summary()
    assert summary["observed"] is True and summary["reasons"] == {"network": 2}
    lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("observation-retry ")
    ]
    assert len(lines) == 2, lines
    for field in ("mode=verify", "what=report", "reason=network", "/120s"):
        assert field in lines[1], f"the retry line must say {field}: {lines[1]}"


def test_a_spent_budget_fails_closed_saying_health_was_not_observed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pauses: list[float] = []
    budget = OBS.ObservationBudget(120, mode="stability", sleep=pauses.append)
    _calls, observe = _flaky(10_000)

    with pytest.raises(OBS.HealthUnobservable) as caught:
        budget.attempt(observe, what="sample")

    message = str(caught.value)
    assert message.startswith("release health could not be observed: "), message
    assert "the stability sample could not authenticate to or reach an API" in message
    assert "(network)" in message and "120s observation budget" in message
    assert sum(pauses) <= 120, "the budget was overrun"
    assert pauses == [5.0, 10.0, 20.0, 30.0, 30.0], "the last pause repeats"
    assert isinstance(caught.value, ReleaseError), "it must still fail the run"
    assert not isinstance(caught.value, OBS.ApiUnavailable), (
        "a spent budget must not be retried again by an outer layer"
    )
    capsys.readouterr()


def test_the_budget_never_takes_a_finding_again() -> None:
    pauses: list[float] = []
    budget = OBS.ObservationBudget(120, mode="verify", sleep=pauses.append)

    def observe() -> None:
        raise ReleaseError("release stability window observed a restart: cpu/x")

    with pytest.raises(ReleaseError, match="observed a restart"):
        budget.attempt(observe, what="sample")
    assert pauses == [] and budget.retries == 0, "a finding was retried"


def test_only_verify_and_stability_get_a_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OBS.STABILITY_SECONDS_ENV, raising=False)
    for mode in ("deploy", "rollback", "status", "preflight", "commit"):
        assert OBS.ObservationBudget.for_mode(mode, dry_run=False) is None, mode
    assert OBS.ObservationBudget.for_mode("verify", dry_run=True) is None
    budget = OBS.ObservationBudget.for_mode("stability", dry_run=False)
    assert budget is not None and budget.seconds == 120


@pytest.mark.parametrize(
    "value,seconds", [("180", 180), ("300", 300), ("119", 120), ("x", 120), ("", 120)]
)
def test_the_budget_is_the_configured_stability_window(
    value: str, seconds: int
) -> None:
    assert OBS.observation_seconds({OBS.STABILITY_SECONDS_ENV: value}) == seconds


def test_exit_codes_stay_distinct_and_the_driver_mirrors_them() -> None:
    assert (
        OBS.HEALTH_UNOBSERVABLE_EXIT_CODE
        == release_deploy.HEALTH_UNOBSERVABLE_EXIT_CODE
    )
    assert OBS.HEALTH_UNOBSERVABLE_EXIT_CODE not in (
        0,
        1,
        2,
        INFLIGHT_INSTALLS_REFUSED_EXIT_CODE,
    ), "the code must not collide with a report verdict, an error or a refusal"


# --- verify ------------------------------------------------------------------


def _check(name: str, status: str = "PASS", details: Any = None) -> dict[str, Any]:
    return {"name": name, "status": status, "summary": name, "details": details}


def _report(*checks: dict[str, Any]) -> dict[str, Any]:
    failed = sum(item["status"] == "FAIL" for item in checks)
    return {
        "mode": "verify",
        "healthy": failed == 0,
        "summary": {"PASS": len(checks) - failed, "WARN": 0, "FAIL": failed, "SKIP": 0},
        "checks": list(checks),
    }


UNOBSERVED = _report(
    _check("cpu_workloads"),
    _check("aurora", "FAIL", {"unobservable": "no-credentials"}),
    _check("monitoring", "FAIL", {"unobservable": "no-credentials"}),
)
HEALTHY = _report(_check("cpu_workloads"), _check("aurora"), _check("monitoring"))


CHECK_FUNCTIONS = (
    "_check_contexts",
    "check_cpu_secrets",
    "_check_cpu_workloads",
    "check_email_notifications",
    "check_aurora_refresh",
    "check_adot_self_metrics",
    "_verify_profile",
    "run_read_only_verifiers",
    "_check_runtime_component_identity",
    "_check_control_api",
    "_check_gpu_cluster",
    "_check_nlb_runtime",
    "_check_aurora",
    "_check_monitoring",
)


def test_a_check_that_could_not_reach_its_api_is_recorded_as_unobserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bridge from a check's exception to the verify retry is its FAIL record."""

    unreachable = {"_check_aurora", "_check_monitoring"}
    for name in CHECK_FUNCTIONS:

        def check(*_args: Any, name: str = name, **_kwargs: Any) -> Any:
            if name in unreachable:
                raise _unavailable("no-credentials")
            return CHECKS.CheckValue(name)

        monkeypatch.setattr(CHECKS, name, check)
    monkeypatch.delenv(CHECKS.QUICK_VALIDATION_EVIDENCE_ENV, raising=False)
    release = SimpleNamespace(
        config=SimpleNamespace(site_name="test-site", clusters=[])
    )

    report = CHECKS.build_health_report(release, mode="verify")

    failed = [item for item in report["checks"] if item["status"] == "FAIL"]
    assert [item["name"] for item in failed] == ["aurora", "monitoring"]
    assert all(
        item["details"] == {"unobservable": "no-credentials"} for item in failed
    ), failed
    assert OBS.unobserved_checks(report) == [
        ("aurora", "no-credentials"),
        ("monitoring", "no-credentials"),
    ]


class VerifyRelease:
    def __init__(self, *, state_failures: int = 0) -> None:
        self.calls: list[str] = []
        self.state_failures = state_failures

    def _load_state(self) -> dict[str, Any]:
        self.calls.append("load-state")
        if self.state_failures:
            self.state_failures -= 1
            raise _unavailable()
        return {"phase": "complete"}

    def _apply_health_baseline(self, state: dict[str, Any]) -> None:
        self.calls.append("health-baseline")


def _builder(*reports: dict[str, Any]) -> tuple[list[int], Any]:
    pending: Iterator[dict[str, Any]] = iter(reports)
    calls: list[int] = []

    def build(release: Any, *, mode: str) -> dict[str, Any]:
        assert mode == "verify"
        calls.append(1)
        return next(pending)

    return calls, build


def test_verify_takes_a_report_it_could_not_observe_again() -> None:
    release = VerifyRelease()
    pauses: list[float] = []
    budget = OBS.ObservationBudget(120, mode="verify", sleep=pauses.append)
    calls, build = _builder(UNOBSERVED, HEALTHY)

    report, code = OBS.observed_health_report(
        release, budget, build, ROLLOUT.report_exit_code
    )

    assert code == 0, "a report that passed after a retry is a pass"
    assert len(calls) == 2 and pauses == [5.0]
    assert report["healthy"] is True
    assert report["observation"]["retries"] == 1
    assert report["observation"]["reasons"] == {"no-credentials": 1}
    assert release.calls == ["load-state", "health-baseline"] * 2, (
        "every attempt reads the state and applies the baseline again"
    )


def test_one_real_failure_is_the_verdict_at_once() -> None:
    budget = OBS.ObservationBudget(120, mode="verify", sleep=lambda _seconds: None)
    mixed = _report(
        _check("aurora", "FAIL", {"unobservable": "network"}),
        _check("cpu_workloads", "FAIL", None),
    )
    calls, build = _builder(mixed)

    report, code = OBS.observed_health_report(
        VerifyRelease(), budget, build, ROLLOUT.report_exit_code
    )

    assert code == 1 and len(calls) == 1, "a finding beside the gap was retried"
    assert "observation" not in report and budget.retries == 0


def test_verify_without_its_budget_spent_reports_health_unobserved(
    capsys: pytest.CaptureFixture[str],
) -> None:
    budget = OBS.ObservationBudget(120, mode="verify", sleep=lambda _seconds: None)
    _calls, build = _builder(*([UNOBSERVED] * 20))

    report, code = OBS.observed_health_report(
        VerifyRelease(), budget, build, ROLLOUT.report_exit_code
    )

    assert code == OBS.HEALTH_UNOBSERVABLE_EXIT_CODE
    assert report["healthy"] is False, "an unobserved report is never healthy"
    assert report["observation"]["observed"] is False
    assert report["checks"] == UNOBSERVED["checks"], "the evidence is the last report"
    error = capsys.readouterr().err
    assert "ERROR: release health could not be observed: " in error
    assert "verify could not observe aurora (no-credentials)" in error


def test_a_state_read_that_could_not_reach_the_api_is_taken_again() -> None:
    release = VerifyRelease(state_failures=1)
    budget = OBS.ObservationBudget(120, mode="verify", sleep=lambda _seconds: None)
    _calls, build = _builder(HEALTHY)

    _report_value, code = OBS.observed_health_report(
        release, budget, build, ROLLOUT.report_exit_code
    )

    assert code == 0 and budget.retries == 1
    never = OBS.ObservationBudget(120, mode="verify", sleep=lambda _seconds: None)
    with pytest.raises(OBS.HealthUnobservable, match="verify report"):
        OBS.observed_health_report(
            VerifyRelease(state_failures=10_000),
            never,
            _builder(HEALTHY)[1],
            ROLLOUT.report_exit_code,
        )


# --- stability ---------------------------------------------------------------


def _sample(*, not_ready: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "restarts": {"cpu/pod/container": 0},
        "not_ready": list(not_ready),
        "queue": {"depth": 0, "oldest_age_seconds": 0.0},
        "remote_commands": {"by_status": {}},
        "critical_alerts": {"count": 0, "alerts": []},
    }


def _window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]
) -> tuple[Any, dict[str, float]]:
    """A release whose window reads ``outcomes`` on a clock that moves on sleep."""

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    monkeypatch.delenv("GPU_FAULT_RELEASE_QUEUE_GROWTH_MIN_DEPTH", raising=False)
    monkeypatch.delenv("GPU_FAULT_RELEASE_STABILITY_GRACE_ALERTS", raising=False)
    config = ROLLOUT.ReleaseConfig.load(config_file(tmp_path))
    release = ROLLOUT.RegionalRelease(config, ROLLOUT.Runner(dry_run=False))
    pending = iter(outcomes)

    def snapshot() -> dict[str, Any]:
        outcome = next(pending, outcomes[-1])
        if isinstance(outcome, Exception):
            raise outcome
        return dict(outcome)

    monkeypatch.setattr(release, "_stability_snapshot", snapshot)
    clock = {"now": 0.0}
    monkeypatch.setattr(VALIDATION.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        VALIDATION.time,
        "sleep",
        lambda seconds: clock.update(now=clock["now"] + seconds),
    )
    return release, clock


def test_a_sample_it_could_not_take_extends_the_window_instead_of_failing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    release, clock = _window(
        tmp_path, monkeypatch, [_sample(), _unavailable(), _sample(), _sample()]
    )

    report = release.validate_stability_window(window_seconds=120, sample_seconds=60)

    assert report["healthy"] is True
    assert report["sample_count"] == 3, "the verdict rests on a full window"
    assert clock["now"] == 125.0, "the window ran on by the five seconds it missed"
    assert report["observation"]["retries"] == 1
    assert report["observation"]["unobserved_seconds"] == pytest.approx(5.0, abs=1)
    assert "what=sample" in capsys.readouterr().err


def test_a_control_plane_only_window_with_a_gap_never_closes_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, _clock = _window(
        tmp_path, monkeypatch, [_sample(), _unavailable(), _sample()]
    )
    monkeypatch.setattr(
        VALIDATION,
        "control_plane_deployments_converged",
        lambda _release: pytest.fail("an unclean window asked to close early"),
    )

    report = release.validate_stability_window(
        window_seconds=120, sample_seconds=30, release_kind="CONTROL_PLANE_ONLY"
    )

    assert report["early_exit"] is False
    assert report["early_exit_reason"] == "full window kept: a sample had to be retried"


def test_a_baseline_it_could_not_take_is_taken_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, _clock = _window(
        tmp_path, monkeypatch, [_unavailable("no-credentials"), _sample()]
    )

    report = release.validate_stability_window(window_seconds=120, sample_seconds=60)

    assert report["healthy"] is True
    assert report["observation"]["reasons"] == {"no-credentials": 1}


def test_a_finding_after_a_gap_is_still_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, _clock = _window(
        tmp_path,
        monkeypatch,
        [_sample(), _unavailable(), _sample(not_ready=("cpu/api-0",))],
    )

    with pytest.raises(ReleaseError, match="non-Ready Pods") as caught:
        release.validate_stability_window(window_seconds=120, sample_seconds=60)
    assert not isinstance(caught.value, OBS.HealthUnobservable), (
        "the retried sample's finding must be reported as itself"
    )


def test_a_window_that_cannot_observe_for_its_whole_budget_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, _clock = _window(tmp_path, monkeypatch, [_sample(), _unavailable()])

    with pytest.raises(OBS.HealthUnobservable, match="the stability sample"):
        release.validate_stability_window(window_seconds=120, sample_seconds=60)


def test_an_unreachable_amp_is_unavailable_not_a_count() -> None:
    release = SimpleNamespace(
        config=SimpleNamespace(
            health=SimpleNamespace(amp_workspace_id="workspace-example"),
            aws_region="us-west-2",
        ),
        runner=SimpleNamespace(
            run=lambda *_args, **_kwargs: '{"unavailable": "network"}'
        ),
    )

    with pytest.raises(OBS.ApiUnavailable) as caught:
        VALIDATION.critical_amp_alerts(release)
    assert caught.value.reason == "network"


# --- the AMP alert probe ------------------------------------------------------


def _amp_probe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    credentials: Any,
    urlopen: Any,
) -> dict[str, Any]:
    def session(**_kwargs: Any) -> Any:
        def get_credentials() -> Any:
            if isinstance(credentials, Exception):
                raise credentials
            return credentials

        return SimpleNamespace(get_credentials=get_credentials)

    monkeypatch.setattr(boto3, "Session", session)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    request = {"region": "us-west-2", "workspace_id": "workspace-example"}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    source = PROBES.probe_source("critical_amp_alerts")
    exec(compile(source, "critical_amp_alerts.py", "exec"), {"__name__": "__main__"})
    return dict(json.loads(capsys.readouterr().out))


FROZEN = SimpleNamespace(
    get_frozen_credentials=lambda: ReadOnlyCredentials("AKIDEXAMPLE", "example", None)
)


def _raise(error: Exception) -> Any:
    def urlopen(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    return urlopen


@pytest.mark.parametrize(
    "credentials,urlopen,expected",
    [
        (None, _raise(AssertionError("never sent")), {"unavailable": "no-credentials"}),
        (
            CredentialRetrievalError(provider="iam-role", error_msg="metadata timeout"),
            _raise(AssertionError("never sent")),
            {"unavailable": "no-credentials"},
        ),
        (
            FROZEN,
            _raise(urllib.error.URLError("timed out")),
            {"unavailable": "network"},
        ),
        (FROZEN, _raise(TimeoutError("read timed out")), {"unavailable": "network"}),
        (
            FROZEN,
            _raise(urllib.error.HTTPError("https://amp", 503, "busy", {}, None)),
            {"unavailable": "throttled"},
        ),
    ],
)
def test_the_amp_probe_reports_a_query_that_got_no_answer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    credentials: Any,
    urlopen: Any,
    expected: dict[str, str],
) -> None:
    assert (
        _amp_probe(monkeypatch, capsys, credentials=credentials, urlopen=urlopen)
        == expected
    )


def test_the_amp_probe_still_fails_on_a_refusal_and_counts_an_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    forbidden = urllib.error.HTTPError("https://amp", 403, "denied", {}, None)
    with pytest.raises(urllib.error.HTTPError):
        _amp_probe(monkeypatch, capsys, credentials=FROZEN, urlopen=_raise(forbidden))

    alerts = [
        {"labels": {"alertname": "GpuFaultStoreIoRejected", "severity": "critical"}},
        {"labels": {"alertname": "Info", "severity": "warning"}},
    ]
    response = SimpleNamespace(read=lambda: json.dumps(alerts).encode())
    answered = _amp_probe(
        monkeypatch,
        capsys,
        credentials=FROZEN,
        urlopen=lambda *_args, **_kwargs: nullcontext(response),
    )
    assert answered == {
        "count": 1,
        "alerts": [{"alertname": "GpuFaultStoreIoRejected", "severity": "critical"}],
    }


# --- the entry point -----------------------------------------------------------

PLUGIN = """\
import json, os, sys, time
from pathlib import Path
state = Path(os.environ["PLUGIN_STATE"])
attempt = int(state.read_text()) if state.is_file() else 0
state.write_text(str(attempt + 1))
if attempt < int(os.environ["PLUGIN_FAILURES"]):
    print(os.environ["PLUGIN_STDERR"], file=sys.stderr)
    sys.exit(253)
expiry = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 900))
print(json.dumps({
    "apiVersion": "client.authentication.k8s.io/v1beta1",
    "kind": "ExecCredential",
    "status": {"token": "k8s-aws-v1.entry-point-token", "expirationTimestamp": expiry},
}))
"""


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The rollout entry point over a real token cache and a scripted `aws`."""

    script = tmp_path / "bin" / "aws"
    script.parent.mkdir()
    script.write_text(f"#!{sys.executable}\n{PLUGIN}", encoding="utf-8")
    script.chmod(0o700)
    kubeconfig = tmp_path / "cpu.kubeconfig"
    kubeconfig.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "v1",
                "kind": "Config",
                "clusters": [{"name": "cpu", "cluster": {"server": "https://cpu"}}],
                "contexts": [
                    {"name": "cpu", "context": {"cluster": "cpu", "user": "cpu"}}
                ],
                "current-context": "cpu",
                "users": [
                    {
                        "name": "cpu",
                        "user": {
                            "exec": {
                                "apiVersion": "client.authentication.k8s.io/v1beta1",
                                "command": "aws",
                                "args": ["eks", "get-token", "--cluster-name", "cpu"],
                                "env": [
                                    {
                                        "name": "PATH",
                                        "value": f"{script.parent}:/usr/bin:/bin",
                                    }
                                ],
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PLUGIN_STATE", str(tmp_path / "attempts"))
    monkeypatch.setenv("PLUGIN_STDERR", NO_CREDENTIALS)
    monkeypatch.delenv(CACHE.TOKEN_CACHE_ENV, raising=False)
    monkeypatch.delenv(OBS.STABILITY_SECONDS_ENV, raising=False)
    monkeypatch.setattr(CACHE, "EXEC_RETRY_DELAYS", (0.0, 0.0))
    run = CACHE.run_command

    def local_plugin(arguments: list[str], **kwargs: Any) -> Any:
        source = Path(shutil.which("aws", path=kwargs["environment"]["PATH"]) or "")
        assert source.is_relative_to(tmp_path), "only the local fixture may run"
        executable = source.with_name("exec-credential-fixture.py")
        shutil.copyfile(source, executable)
        return run([sys.executable, str(executable), *arguments[1:]], **kwargs)

    monkeypatch.setattr(CACHE, "run_command", local_plugin)
    pauses: list[float] = []
    monkeypatch.setattr(
        OBS.ObservationBudget,
        "for_mode",
        classmethod(
            lambda cls, mode, *, dry_run: None
            if mode not in OBS.OBSERVATION_MODES or dry_run
            else cls(120, mode=mode, sleep=pauses.append)
        ),
    )
    seen: dict[str, Any] = {"pauses": pauses, "attempts": tmp_path / "attempts"}

    class Release:
        def __init__(self, config: Any, runner: Any) -> None:
            seen["kubeconfig"] = config.cpu_kubeconfig

        def _load_state(self) -> dict[str, Any]:
            return {"phase": "complete"}

        def _apply_health_baseline(self, state: dict[str, Any]) -> None:
            return None

    monkeypatch.setattr(
        ROLLOUT.ReleaseConfig,
        "load",
        staticmethod(lambda _path: SimpleNamespace(cpu_kubeconfig=str(kubeconfig))),
    )
    monkeypatch.setattr(ROLLOUT, "RegionalRelease", Release)
    monkeypatch.setattr(
        ROLLOUT, "build_health_report", lambda _release, **_kwargs: HEALTHY
    )
    monkeypatch.setattr(ROLLOUT, "deployment_api_budget", nullcontext)
    monkeypatch.setattr(
        ROLLOUT, "deployment_deadline", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(ROLLOUT.sys, "argv", ["rollout", "verify", "--config", "x"])
    return seen


@pytest.mark.parametrize("failures", [1, CACHE.EXEC_ATTEMPTS])
def test_a_credential_plugin_that_fails_then_recovers_lets_verify_pass(
    engine: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failures: int,
) -> None:
    """One 253 is absorbed by the fetch; a whole failed fetch by the budget."""

    monkeypatch.setenv("PLUGIN_FAILURES", str(failures))

    assert ROLLOUT.main() == 0, "verify failed: the driver would have rolled back"

    output = capsys.readouterr()
    assert json.loads(output.out)["healthy"] is True
    assert int(engine["attempts"].read_text()) == failures + 1
    assert engine["pauses"] == ([] if failures == 1 else [5.0]), (
        "only a fetch that failed outright costs an observation retry"
    )
    assert "entry-point-token" not in output.out + output.err


def test_a_credential_plugin_that_never_recovers_fails_verify_closed(
    engine: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PLUGIN_FAILURES", "1000000")

    started = time.monotonic()
    code = ROLLOUT.main()
    elapsed = time.monotonic() - started

    assert code == OBS.HEALTH_UNOBSERVABLE_EXIT_CODE
    error = capsys.readouterr().err
    assert "ERROR: release health could not be observed: " in error
    assert "verify credentials could not authenticate" in error
    assert "exec credential plugin exited 253 (no-credentials)" in error
    assert "Unable to locate credentials" not in error, "plugin output leaked"
    assert "release-end" in error and "exit_code=4" in error
    assert sum(engine["pauses"]) <= 120, "the observation budget was overrun"
    assert elapsed < 60, f"the fake budget must not cost real time: {elapsed}"


def test_a_window_that_could_not_observe_exits_with_its_own_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Release:
        def __init__(self, config: Any, runner: Any) -> None:
            pass

        def _load_state(self) -> dict[str, Any]:
            return {}

        def validate_stability_window(self, **_kwargs: Any) -> dict[str, Any]:
            raise OBS.HealthUnobservable(
                "release health could not be observed: fixture", last=_unavailable()
            )

    monkeypatch.setattr(
        ROLLOUT.ReleaseConfig,
        "load",
        staticmethod(lambda _path: SimpleNamespace(cpu_kubeconfig="missing.config")),
    )
    monkeypatch.setattr(ROLLOUT, "RegionalRelease", Release)
    monkeypatch.setattr(ROLLOUT, "deployment_api_budget", nullcontext)
    monkeypatch.setattr(
        ROLLOUT, "deployment_deadline", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(ROLLOUT.sys, "argv", ["rollout", "stability", "--config", "x"])

    assert ROLLOUT.main() == OBS.HEALTH_UNOBSERVABLE_EXIT_CODE
    output = capsys.readouterr()
    assert output.out == "", "a window that saw nothing prints no report"
    assert "ERROR: release health could not be observed: fixture" in output.err
