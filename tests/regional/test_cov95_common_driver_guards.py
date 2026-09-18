from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import live_driver_guard as guard

CASE = "GF-REGIONAL-CAP-001"
CONFIRMATION = "UNIT_ONLY_CONFIRMATION"


def approval(tmp_path, **overrides):
    return Namespace(
        **{
            "execute": True,
            "confirm": CONFIRMATION,
            "maintenance_window_end": (
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat(),
            "run_dir": tmp_path,
            "attempt": 1,
            **overrides,
        }
    )


@pytest.mark.parametrize("value", ["", " \n "])
def test_missing_environment_is_an_explicit_configuration_error(
    value, monkeypatch
) -> None:
    monkeypatch.setenv("FIXTURE_REQUIRED", value)
    with pytest.raises(RuntimeError, match="FIXTURE_REQUIRED"):
        guard.environment_snapshot(("FIXTURE_REQUIRED",))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"execute": False}, "outside --execute"),
        ({"maintenance_window_end": ""}, "is required"),
        ({"maintenance_window_end": "2030-01-01T00:00:00"}, "timezone"),
        ({}, "run --plan"),
    ],
)
def test_authorization_rejects_missing_mode_deadline_or_plan(
    override, message, tmp_path
) -> None:
    with pytest.raises(RuntimeError, match=message):
        guard.authorize_execution(
            approval(tmp_path, **override),
            case_id=CASE,
            confirmation=CONFIRMATION,
            environment={},
        )


@pytest.mark.parametrize("document", [None, [], "plan"])
def test_nonobject_plan_cannot_authorize_execution(document, tmp_path) -> None:
    path = tmp_path / "cases" / CASE / "plan.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="not an object"):
        guard.authorize_execution(
            approval(tmp_path), case_id=CASE, confirmation=CONFIRMATION, environment={}
        )


@pytest.mark.parametrize("passed", [None, "true", 1])
def test_plan_requires_boolean_preflight_before_recording(passed, tmp_path) -> None:
    with pytest.raises(RuntimeError, match="must be a boolean"):
        guard.build_plan(
            run_dir=tmp_path,
            case_id=CASE,
            attempt=1,
            confirmation=CONFIRMATION,
            details={},
            arguments=approval(tmp_path),
            preflight_passed=passed,
            environment={},
        )
    assert not (tmp_path / "cases").exists(), "invalid preflight must not mint a plan"


@pytest.mark.parametrize("method", ["build", "authorize"])
def test_do_not_run_case_cannot_be_planned_or_authorized(method, tmp_path) -> None:
    case_id = "GF-REGIONAL-DESTR-004"
    with pytest.raises(RuntimeError, match="DO_NOT_RUN"):
        if method == "build":
            guard.build_plan(
                run_dir=tmp_path,
                case_id=case_id,
                attempt=1,
                confirmation=CONFIRMATION,
                details={},
                arguments=approval(tmp_path),
                preflight_passed=True,
                environment={},
            )
        else:
            guard.authorize_execution(
                approval(tmp_path),
                case_id=case_id,
                confirmation=CONFIRMATION,
                environment={},
            )


def test_missing_details_object_rejects_an_otherwise_matching_plan(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(guard, "source_digest", lambda: "fixture-source")
    args = approval(tmp_path)
    plan = guard.build_plan(
        run_dir=tmp_path,
        case_id=CASE,
        attempt=1,
        confirmation=CONFIRMATION,
        details={},
        arguments=args,
        preflight_passed=True,
        environment={},
    )
    plan["details"] = []
    (tmp_path / "cases" / CASE / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(RuntimeError, match="drifted at details"):
        guard.authorize_execution(
            args, case_id=CASE, confirmation=CONFIRMATION, environment={}
        )


@pytest.mark.parametrize(
    "document", [[], {"details": []}, {"details": {"focused_tests": []}}]
)
def test_cached_focus_result_requires_complete_objects(document, tmp_path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document))
    assert guard.reusable_focused_tests(path) is None, (
        "malformed cache must cause a rerun"
    )


@pytest.mark.parametrize(
    "kind", ["absolute", "traversal", "symlink", "missing", "git-error"]
)
def test_source_digest_refuses_unavailable_or_escaping_inputs(
    kind, tmp_path, monkeypatch
) -> None:
    source = tmp_path / "input.py"
    source.write_text("fixture data")
    relative = {
        "absolute": str(source),
        "traversal": "../outside.py",
        "symlink": "link.py",
        "missing": "missing.py",
        "git-error": "input.py",
    }[kind]
    if kind == "symlink":
        (tmp_path / relative).symlink_to(source)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == tmp_path, "digest reads must stay in the isolated tree"
        output = "fixture-head" if command[1] == "rev-parse" else relative + "\0"
        if "--deleted" in command:
            output = ""
        return subprocess.CompletedProcess(
            command, 1 if kind == "git-error" else 0, output, "fixture git failure"
        )

    monkeypatch.setattr(guard, "ROOT", tmp_path)
    monkeypatch.setattr(
        guard, "subprocess", SimpleNamespace(run=run, PIPE=subprocess.PIPE)
    )
    with pytest.raises(
        RuntimeError, match="repository|symbolic|unavailable|git .* failed"
    ):
        guard.source_digest()
    assert calls, "source binding must consult the fake tracked/untracked inventory"


def test_connection_read_failure_is_not_a_missing_identity(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "empty.kubeconfig"
    path.touch()
    original = Path.open

    def denied(self, *args, **kwargs):
        if self == path:
            raise OSError("fixture read denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(RuntimeError, match="cannot bind kubeconfig identity"):
        guard.connection_identity(Namespace(cpu_kubeconfig=path), {})
