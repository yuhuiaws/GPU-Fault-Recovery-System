from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.lifecycle import ShutdownCoordinator
from gpu_fault.processor import coordinator
from scripts.e2e.regional import run_ha007_control_worker_shutdown as ha007
from scripts.e2e.regional import run_ha008_processor_exit_acceptance as ha008
from scripts.e2e.regional import run_ha008_processor_exit_probe as probe
from tests.regional._cov95_ha001_harness import Clock
from tests.regional._cov95_ha_exit_harness import ExitHarness, child_arguments


def test_real_processor_fencing_through_simulated_child_transports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = ExitHarness(monkeypatch)
    result = ha008.run_acceptance(tmp_path)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["sensitive_temp_files_removed"] is True
    assert len(harness.commands) == 4
    assert [item["status_after_exit"] for item in result["branches"]] == [
        "PENDING",
        "LEASED",
    ]
    assert all(item["stale_result_rejected"] for item in result["branches"]), (
        "old owners must be fenced"
    )
    assert all(item["final_status"] == "COMPLETED" for item in result["branches"]), (
        "takeover must complete"
    )
    assert not any(Path(command[3]).exists() for command in harness.commands), (
        "temporary stores must be removed"
    )


@pytest.mark.parametrize("entrypoint", ["acceptance", "main"])
@pytest.mark.parametrize(
    "suppression",
    [
        "global-disable",
        "logger-disable",
        "logger-level",
        "logger-filter",
        "parent-handler",
    ],
)
def test_simulated_children_keep_logging_and_fencing_isolated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    entrypoint: str,
    suppression: str,
) -> None:
    parent_logger = coordinator.LOGGER
    original_level = parent_logger.level
    original_disable = logging.root.manager.disable
    parent_output = io.StringIO()
    handler = logging.StreamHandler(parent_output)
    with monkeypatch.context() as parent:
        parent.setattr(parent_logger, "handlers", [handler])
        parent.setattr(parent_logger, "propagate", False)
        parent.setattr(parent_logger, "disabled", suppression == "logger-disable")
        parent.setattr(
            parent_logger,
            "filters",
            [logging.Filter("unrelated.parent")]
            if suppression == "logger-filter"
            else [],
        )
        try:
            parent_logger.setLevel(
                logging.CRITICAL + 1 if suppression == "logger-level" else logging.INFO
            )
            logging.disable(
                logging.CRITICAL if suppression == "global-disable" else logging.NOTSET
            )
            parent_state = (
                parent_logger.level,
                parent_logger.disabled,
                parent_logger.propagate,
                tuple(parent_logger.handlers),
                tuple(parent_logger.filters),
                logging.root.manager.disable,
            )
            harness = ExitHarness(monkeypatch)
            if entrypoint == "main":
                monkeypatch.setattr(
                    ha008,
                    "os",
                    SimpleNamespace(**{**vars(os), "umask": lambda _: None}),
                )
                monkeypatch.setattr(sys, "argv", ["unit"])
                code = ha008.main()
                result = json.loads(capsys.readouterr().out)
                assert code == 0, result["errors"]
                assert result["validation_scope"] == "isolated-source"
                assert result["formal_sequence_satisfied"] is False
            else:
                result = ha008.run_acceptance(tmp_path)
            assert result["verdict"] == "PASS", result["errors"]
            assert len(harness.commands) == 4
            assert [item["status_after_exit"] for item in result["branches"]] == [
                "PENDING",
                "LEASED",
            ]
            for branch in result["branches"]:
                assert branch["exit_code"] == 70
                assert branch["deadline_exceeded_logged"] is True
                assert branch["release_failure_logged"] is branch["fail_release"]
                assert branch["stale_result_rejected"] is True
                assert branch["second_lane_epoch"] > branch["first_lane_epoch"]
                assert branch["first_owner"] != branch["second_owner"]
                assert branch["final_status"] == "COMPLETED"
                assert branch["final_response_status"] == 200
            assert coordinator.LOGGER is parent_logger
            assert (
                parent_logger.level,
                parent_logger.disabled,
                parent_logger.propagate,
                tuple(parent_logger.handlers),
                tuple(parent_logger.filters),
                logging.root.manager.disable,
            ) == parent_state, "simulated children must preserve parent logging state"
            assert not parent_output.getvalue(), (
                "simulated child records must not reach parent handlers"
            )
        finally:
            parent_logger.setLevel(original_level)
            logging.disable(original_disable)
            handler.close()


@pytest.mark.parametrize(
    ("message", "error"),
    [
        (ha008.DEADLINE_EXCEEDED_MESSAGE, "deadline-exceeded"),
        (ha008.RELEASE_FAILURE_MESSAGE, "could not release request"),
    ],
)
def test_real_fencing_cannot_replace_missing_processor_error_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, message: str, error: str
) -> None:
    harness = ExitHarness(monkeypatch)

    def drop_record(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        completed = harness.run(argv, **kwargs)
        stderr = "\n".join(
            line
            for line in completed.stderr.splitlines()
            if not (
                line.startswith(f"ERROR {ha008.PROCESSOR_LOGGER} ") and message in line
            )
        )
        return subprocess.CompletedProcess(
            argv, completed.returncode, completed.stdout, stderr
        )

    monkeypatch.setattr(
        ha008, "subprocess", SimpleNamespace(**{**vars(subprocess), "run": drop_record})
    )
    result = ha008.run_acceptance(tmp_path)
    assert result["verdict"] == "FAIL"
    assert any(error in item for item in result["errors"]), result["errors"]
    assert len(result["branches"]) == 2
    assert all(item["stale_result_rejected"] for item in result["branches"]), (
        "missing log evidence must not bypass the real fencing exercise"
    )
    assert all(item["final_status"] == "COMPLETED" for item in result["branches"]), (
        "successful takeover alone must not satisfy the required error-log proof"
    )


def test_simulated_transport_preserves_the_actual_processor_logger_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        coordinator, "LOGGER", logging.Logger("gpu_fault.unexpected.processor")
    )
    ExitHarness(monkeypatch)
    result = ha008.run_acceptance(tmp_path)
    assert result["verdict"] == "FAIL"
    assert len(result["branches"]) == 2
    assert all(not item["deadline_exceeded_logged"] for item in result["branches"]), (
        "records from another logger cannot satisfy the processor error proof"
    )
    assert result["branches"][1]["release_failure_logged"] is False
    assert all(item["stale_result_rejected"] for item in result["branches"]), (
        "logger identity failure must retain the real fencing result"
    )
    assert (
        "ERROR gpu_fault.unexpected.processor "
        in (tmp_path / "release-fail.stderr").read_text()
    )


@pytest.mark.parametrize("defect", ["missing-claim", "failed-takeover", "exit-code"])
def test_child_failure_or_wrong_exit_cannot_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    harness = ExitHarness(monkeypatch)
    harness.omit_claim = defect == "missing-claim"
    harness.takeover_failure = defect == "failed-takeover"
    harness.returncode_override = 0 if defect == "exit-code" else None
    report = ha008.run_acceptance(tmp_path)
    assert report["verdict"] == "FAIL"
    assert report["errors"], "transport and exit failures must remain explicit"


@pytest.mark.parametrize(
    "override,fragment",
    [
        ({"stale_result_rejected": False}, "accepted the old result"),
        ({"final_status": "LEASED"}, "not completed"),
        ({"final_response_status": 500}, "not 200"),
        ({"second_process_id": 11}, "another process"),
        ({"second_process_id": None}, "another process"),
    ],
)
def test_acceptance_rejects_incomplete_takeover_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    override: dict[str, Any],
    fragment: str,
) -> None:
    harness = ExitHarness(monkeypatch)
    harness.takeover_override = override
    report = ha008.run_acceptance(tmp_path)
    assert report["verdict"] == "FAIL"
    assert any(fragment in error for error in report["errors"]), report["errors"]


def test_takeover_never_claims_before_the_original_lease_expires(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = ExitHarness(monkeypatch)
    command = child_arguments(tmp_path, fail_release=True)
    assert harness.run(command).returncode == 70
    with pytest.raises(RuntimeError, match="could not claim the same request"):
        probe.take_over(command[3], command[5], "replacement")
    harness.clock.sleep(2)
    result = probe.take_over(command[3], command[5], "replacement")
    assert result["stale_result_rejected"] is True
    assert result["final_status"] == "COMPLETED"


@pytest.mark.parametrize("mode", ["completed", "deadline", "signal-timeout"])
def test_shutdown_child_uses_real_coordinator_with_fake_threads_and_signals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    clock = Clock()
    duration = 2 if mode == "deadline" else 0.1
    callbacks = []
    threads = []

    class Thread:
        def __init__(self, *, target: Any, name: str, daemon: bool) -> None:
            self.target, self.name = target, name
            self.running = False
            threads.append(self)

        def start(self) -> None:
            self.running = True

        def join(self, *, timeout: float) -> None:
            if duration <= timeout:
                self.target()
                self.running = False
            else:
                clock.sleep(timeout)

        def is_alive(self) -> bool:
            return self.running

    def register(sig: int, callback: Any) -> None:
        callbacks.append(callback)
        if mode != "signal-timeout":
            callback(sig, None)

    monkeypatch.setattr(ha007, "signal", SimpleNamespace(SIGTERM=15, signal=register))
    monkeypatch.setattr(ha007, "Thread", Thread)
    monkeypatch.setattr(ha007, "time", clock)
    monkeypatch.setattr(
        ha007,
        "ShutdownCoordinator",
        lambda budget: ShutdownCoordinator(budget, now=clock.monotonic),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unit",
            "--child",
            "--duration",
            str(duration),
            "--started-file",
            str(tmp_path / "started"),
            "--result-file",
            str(tmp_path / "result.json"),
            "--lifespan-budget-seconds",
            "1",
            "--signal-timeout-seconds",
            "0.001",
        ],
    )
    assert ha007.main() == {"completed": 0, "deadline": 1, "signal-timeout": 3}[mode]
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["completed"] == int(mode == "completed")
    assert result["signal_timed_out"] is (mode == "signal-timeout")
    assert result["shutdown_failures"] == (
        [ha007.WORKER_THREAD_NAME] if mode == "deadline" else []
    )
    assert len(callbacks) == len(threads) == 1


@pytest.mark.parametrize(
    "mode", ["ok", "missing-result", "timeout", "early-exit", "never-ready"]
)
def test_shutdown_parent_reaps_children_on_every_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    clock = Clock()
    calls = []
    instances = []

    class Process:
        def __init__(self, argv: list[str]) -> None:
            self.returncode: int | None = 7 if mode == "early-exit" else None
            self.result = Path(argv[argv.index("--result-file") + 1])
            if mode not in {"early-exit", "never-ready"}:
                Path(argv[argv.index("--started-file") + 1]).write_text("ready")
            instances.append(self)

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            calls.append("kill")
            self.returncode = -9

        def send_signal(self, signum: int) -> None:
            calls.append("signal")
            if mode not in {"missing-result", "timeout"}:
                self.result.write_text(
                    json.dumps(
                        {
                            "completed": 1,
                            "duration_seconds": 1,
                            "shutdown_seconds": 1,
                            "shutdown_failures": [],
                        }
                    )
                )

        def wait(self, *, timeout: float) -> int | None:
            calls.append("wait")
            if mode == "timeout" and self.returncode is None:
                raise subprocess.TimeoutExpired("unit", timeout)
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    monkeypatch.setattr(
        ha007,
        "subprocess",
        SimpleNamespace(
            **{**vars(subprocess), "Popen": lambda argv, **kw: Process(argv)}
        ),
    )
    monkeypatch.setattr(ha007, "time", clock)
    budgets = {"lifespan_budget_seconds": 2, "kubernetes_grace_seconds": 5}
    if mode in {"early-exit", "never-ready"}:
        with pytest.raises(RuntimeError, match="before readiness|did not become ready"):
            ha007.run_probe(tmp_path, [1], budgets=budgets)
    else:
        result = ha007.run_probe(tmp_path, [1], budgets=budgets)
        assert result["verdict"] == ("PASS" if mode == "ok" else "FAIL")
    assert "wait" in calls
    assert all(process.poll() is not None for process in instances), (
        "parent must reap each child"
    )


@pytest.mark.parametrize("arguments", [[], ["--child"], ["--child", "--duration", "1"]])
def test_shutdown_main_requires_complete_mode_arguments(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["unit", *arguments])
    with pytest.raises(SystemExit, match="required|requires"):
        ha007.main()


@pytest.mark.parametrize("module", [ha007, ha008])
def test_isolated_main_records_failed_probe_without_live_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: Any
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("unit probe failed")

    monkeypatch.setattr(
        module, "run_probe" if module is ha007 else "run_acceptance", fail
    )
    monkeypatch.setattr(module, "os", SimpleNamespace(umask=lambda _: None))
    monkeypatch.setattr(sys, "argv", ["unit", "--run-dir", str(tmp_path)])
    assert module.main() == 1
    path = tmp_path / "cases" / module.CASE_ID / f"{module.CASE_ID}.json"
    result = json.loads(path.read_text())
    assert result["verdict"] == "FAIL"
    assert "release_id" not in result
