from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault import notification_service
from scripts.e2e.regional import run_notify007_delivery_states as runner
from scripts.e2e.regional.probes import notify007_delivery_drill
from tests.regional._cov95_ha001_harness import DEADLINE, Clock
from tests.regional._cov95_notify_harness import Notifier
from tests.regional.test_notification_runner_fail_closed import metric_text


class Regional:
    def __init__(self, patch: pytest.MonkeyPatch) -> None:
        self.patch = patch
        self.settings = SimpleNamespace(
            cluster_id="unit-cluster", environment=lambda: {}
        )
        self.replicas = {
            runner.CONTROL_WORKER_APP: 2,
            runner.API_APP: 3,
            "gpu-fault-telemetry-spool-worker": 0,
        }
        self.completed = False
        self.changed_metrics = False
        self.bad_identity = False
        self.bad_drill = ""
        self.calls: list[str] = []
        self.notifier = Notifier()
        self.clock = Clock()
        self.epoch = datetime.now(timezone.utc)
        patch.setattr(
            notify007_delivery_drill,
            "notification_notifier_from_environment",
            lambda: self.notifier,
        )
        patch.setattr(
            notify007_delivery_drill, "time", SimpleNamespace(sleep=self.clock.sleep)
        )
        patch.setattr(
            notification_service,
            "datetime",
            SimpleNamespace(
                now=lambda tz=None: datetime.now(timezone.utc)
                + timedelta(seconds=self.clock.now)
            ),
        )

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "unit-release", "cluster_id": "unit-cluster"}

    def ready_pods(self, plane: str, app: str) -> list[dict[str, str]]:
        return [
            {"name": f"{app}-{i}", "uid": "" if self.bad_identity else f"uid-{app}-{i}"}
            for i in range(self.replicas[app])
        ]

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        assert plane == "cpu"
        if args[0] == "get":
            return json.dumps({"spec": {"replicas": self.replicas[args[2]]}})
        self.calls.append(args[2])
        failed = int(self.completed and self.changed_metrics and args[2].endswith("-1"))
        return json.dumps({"metrics": metric_text(failed, 0)})

    def pod_python(
        self, plane: str, app: str, script: str, *args: str, **kwargs: Any
    ) -> dict:
        assert (plane, app) == ("cpu", runner.CONTROL_WORKER_APP)
        output = io.StringIO()
        with self.patch.context() as local, redirect_stdout(output):
            local.setattr(sys, "argv", ["unit-drill", *args])
            assert notify007_delivery_drill.main() == 0
        self.completed = True
        result = json.loads(output.getvalue())
        if self.bad_drill == "retry":
            result["retry"]["after_first_cycle"]["stats"]["by_status"]["LEASED"] = 1
        elif self.bad_drill == "dead":
            result["dead"]["stats"]["by_status"]["SENT"] = 1
        return result


@pytest.fixture
def regional(monkeypatch: pytest.MonkeyPatch) -> Regional:
    return Regional(monkeypatch)


def test_full_delivery_drill_uses_real_outbox_and_every_cpu_replica(
    regional: Regional, tmp_path: Path
) -> None:
    result = runner.execute(regional, tmp_path, 1)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["stages"] == {
        "retry": [],
        "dead": [],
        "exported_families": [],
        "production_untouched": [],
        "alert_rule": [],
    }
    assert len(regional.calls) == 10
    assert len(set(regional.calls)) == 5
    assert len(regional.notifier.sent) == 1
    assert result["drill"]["retry"]["after_first_cycle"]["result"] is None
    assert result["drill"]["dead"]["result"]["status"] == "FAILED"


@pytest.mark.parametrize("defect", ["metric", "window", "severity", "hold", "anchor"])
def test_full_delivery_drill_keeps_alert_contract_drift_as_a_failure(
    regional: Regional, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    document = yaml.safe_load(runner.RULES.read_text(encoding="utf-8"))
    failing = next(
        rule
        for group in document["groups"]
        for rule in group["rules"]
        if rule.get("alert") == runner.verdicts.FAILING_ALERT
    )
    if defect == "metric":
        metric = "gpu_fault_notification_terminal_failure_last_seen_timestamp_seconds"
        assert failing["expr"].count(metric) == 2
        failing["expr"] = failing["expr"].replace(
            metric, 'gpu_fault_notification_total{status="FAILED"}'
        )
    elif defect == "window":
        assert failing["expr"].count("[15m]") == 2
        failing["expr"] = failing["expr"].replace("[15m]", "[30m]")
    elif defect == "severity":
        failing["labels"]["severity"] = "warning"
    elif defect == "hold":
        failing["for"] = "1m"
    else:
        failing["annotations"]["runbook_url"] += "-missing"
    rules = tmp_path / "rules.yaml"
    rules.write_text(yaml.safe_dump(document), encoding="utf-8")
    monkeypatch.setattr(runner, "RULES", rules)

    result = runner.execute(regional, tmp_path, 1)

    assert result["verdict"] == "FAIL"
    assert result["stages"]["alert_rule"], defect
    assert {key: errors for key, errors in result["stages"].items() if errors} == {
        "alert_rule": result["stages"]["alert_rule"]
    }, "only the alert contract changed; delivery and production proofs remain valid"
    assert len(regional.notifier.sent) == 1


def test_per_replica_failed_growth_cannot_hide_in_a_complete_metric_set(
    regional: Regional, tmp_path: Path
) -> None:
    regional.changed_metrics = True
    result = runner.execute(regional, tmp_path, 1)
    assert result["verdict"] == "FAIL"
    assert result["stages"]["production_untouched"], (
        "growth must be attributed to its replica"
    )


@pytest.mark.parametrize("stage", ["retry", "dead"])
def test_contradictory_delivery_state_totals_cannot_pass(
    regional: Regional, tmp_path: Path, stage: str
) -> None:
    regional.bad_drill = stage
    result = runner.execute(regional, tmp_path, 1)
    assert result["verdict"] == "FAIL"
    assert result["stages"][stage], "a delivery cannot occupy two states simultaneously"


@pytest.mark.parametrize("bad", [True, -1, None])
def test_metric_collection_requires_typed_nonnegative_replicas(
    regional: Regional, bad: Any
) -> None:
    regional.replicas[runner.CONTROL_WORKER_APP] = bad
    with pytest.raises(RuntimeError, match="replica count"):
        runner.control_plane_metrics(regional)
    assert regional.calls == []


def test_metric_collection_requires_complete_pod_identity(regional: Regional) -> None:
    regional.bad_identity = True
    with pytest.raises(RuntimeError, match="without identity"):
        runner.control_plane_metrics(regional)
    assert regional.calls == []


@pytest.mark.parametrize("execute", [False, True])
def test_main_preserves_formal_identity_in_fake_plan_and_execution(
    regional: Regional, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, execute: bool
) -> None:
    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(runner, "install_abort_signals", lambda: None)
    monkeypatch.setattr(runner, "os", SimpleNamespace(umask=lambda _: None))
    monkeypatch.setattr(runner, "settings_from_arguments", lambda _: regional.settings)
    monkeypatch.setattr(runner, "RegionalLiveFixture", lambda _: regional)
    monkeypatch.setattr(runner, "predecessor_path", lambda *a: (None, None))
    monkeypatch.setattr(runner, "authorize_execution", lambda *a, **kw: DEADLINE)
    monkeypatch.setattr(
        runner, "build_plan", lambda **kw: {"preflight_passed": kw["preflight_passed"]}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unit",
            "--run-dir",
            str(tmp_path),
            *(["--execute", "--confirm", runner.CONFIRMATION] if execute else []),
        ],
    )
    assert runner.main() == 0
    assert len(regional.notifier.sent) == int(execute)
    if execute:
        path = tmp_path / "cases" / runner.CASE_ID / f"{runner.CASE_ID}.json"
        result = json.loads(path.read_text())
        assert result["verdict"] == "PASS"
        assert result["stages"]["alert_rule"] == []
        assert result["release_id"] == "unit-release"
        assert result["cluster_id"] == "unit-cluster"


@pytest.mark.parametrize(
    "defect",
    [
        "first-dead",
        "first-report",
        "second-state",
        "second-dead",
        "cycle",
        "dead-attempts",
    ],
)
def test_delivery_verdict_rejects_each_invalid_cycle_fact(
    regional: Regional, defect: str
) -> None:
    document = runner.run_drill(regional, drill_id="unit-verdict")
    retry, dead = document["retry"], document["dead"]
    if defect == "first-dead":
        retry["after_first_cycle"]["dead_lettered_total"] = 1
    elif defect == "first-report":
        retry["after_first_cycle"]["report"]["failed"] = 0
    elif defect == "second-state":
        retry["after_second_cycle"]["stats"]["by_status"]["SENT"] = 0
    elif defect == "second-dead":
        retry["after_second_cycle"]["dead_lettered_total"] = 1
    elif defect == "cycle":
        retry["last_cycle_timestamp_seconds"] = 0
    else:
        dead["notifier_attempts"] = 2
    errors = runner.verdicts.retry_phase_errors(
        retry
    ) + runner.verdicts.dead_phase_errors(dead)
    assert errors, f"{defect} must fail the delivery-state proof"


def test_missing_rules_and_nonfinite_metrics_cannot_prove_safe_delivery() -> None:
    verdicts = runner.verdicts
    text = metric_text(0, 0) + f"\n{verdicts.DISPATCH_CYCLE_METRIC} NaN"
    with pytest.raises(RuntimeError, match="not finite"):
        verdicts.exported_family_errors([text])
    text = metric_text(float("nan"), 0)
    with pytest.raises(RuntimeError, match="not finite"):
        verdicts.production_untouched_errors([text], [text])
    assert any(
        "not defined" in error for error in verdicts.alert_rule_errors("{}", "")
    ), "missing rule documents must not satisfy alert acceptance"


@pytest.mark.parametrize(
    "mode", ["plan-predecessor", "confirmation", "predecessor", "transport"]
)
def test_main_records_or_refuses_each_failure_before_a_false_pass(
    regional: Regional, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(runner, "install_abort_signals", lambda: None)
    monkeypatch.setattr(runner, "os", SimpleNamespace(umask=lambda _: None))
    monkeypatch.setattr(runner, "settings_from_arguments", lambda _: regional.settings)
    monkeypatch.setattr(runner, "RegionalLiveFixture", lambda _: regional)
    monkeypatch.setattr(
        runner, "predecessor_path", lambda *a: ("previous", tmp_path / "previous")
    )
    monkeypatch.setattr(
        runner,
        "predecessor_evidence",
        lambda *a, **kw: {"valid": mode not in {"plan-predecessor", "predecessor"}},
    )
    monkeypatch.setattr(runner, "authorize_execution", lambda *a, **kw: DEADLINE)
    monkeypatch.setattr(
        runner, "build_plan", lambda **kw: {"preflight_passed": kw["preflight_passed"]}
    )
    argv = ["unit", "--run-dir", str(tmp_path)]
    if mode != "plan-predecessor":
        argv.extend(
            [
                "--execute",
                "--confirm",
                "wrong" if mode == "confirmation" else runner.CONFIRMATION,
            ]
        )
    monkeypatch.setattr(sys, "argv", argv)
    if mode == "transport":

        def fail(*args: Any, **kwargs: Any) -> Any:
            raise OSError("unit drill transport failed")

        monkeypatch.setattr(regional, "pod_python", fail)
    if mode in {"confirmation", "predecessor"}:
        with pytest.raises(RuntimeError, match="confirmation|predecessor"):
            runner.main()
    else:
        assert runner.main() == 1
    assert regional.notifier.sent == []
