from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpu_fault_release import rollout
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_resource_probe import (
    ProbeState,
    ResourceRef,
    probe_resource,
)


@pytest.mark.parametrize(
    "arguments",
    [
        ["kubectl", "wait", "pod/example", "--for=condition=Ready"],
        ["kubectl", "rollout", "status", "deployment/example"],
    ],
)
def test_real_runner_accepts_both_advisory_wait_forms(arguments, monkeypatch):
    calls = []

    def execute(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(rollout, "run_command", execute)
    assert rollout.Runner().condition(arguments, timeout_seconds=3) is True
    assert len(calls) == 1, "an advisory wait must execute exactly one bounded command"
    actual_arguments, options = calls[0]
    assert actual_arguments == arguments
    assert set(options) == {"capture", "input_text", "environment", "timeout_seconds"}
    assert options["capture"] is True
    assert options["input_text"] is None
    # The engine pins its own interpreter directory for every child (deploy
    # tools call bare ``python3``); an advisory wait is no exception.
    assert options["environment"]["PATH"].split(os.pathsep)[0] == str(
        Path(sys.executable).absolute().parent
    )
    assert 0 < options["timeout_seconds"] <= 3, (
        "environment preparation must consume the same advisory wait deadline"
    )


def test_real_runner_never_turns_resource_reads_into_boolean_proofs(monkeypatch):
    monkeypatch.setattr(
        rollout,
        "run_command",
        lambda *_args, **_kwargs: pytest.fail(
            "a resource read entered the advisory waiter"
        ),
    )
    with pytest.raises(rollout.ReleaseError, match="advisory wait"):
        rollout.Runner().condition(["kubectl", "get", "configmap", "example"])


REFERENCE = ResourceRef("secret", "Secret", "test-secret", "test-namespace")


class Runner:
    def __init__(self, code=0, output="", error=""):
        self.result = (code, output, error)
        self.calls = []

    def probe_output(self, arguments, **kwargs):
        self.calls.append((arguments, kwargs))
        return self.result


def document():
    return {
        "kind": "Secret",
        "metadata": {
            "name": REFERENCE.name,
            "namespace": REFERENCE.namespace,
            "uid": "uid-test",
        },
        "data": {"value": "private-placeholder"},
    }


def test_named_read_preserves_identity_and_never_echoes_secret_data():
    runner = Runner(output=json.dumps(document()))
    result = probe_resource(runner, ["kubectl", "--context", "test"], REFERENCE)
    assert result.state is ProbeState.PRESENT
    assert result.require_readable() == document()
    assert result.exists() is True
    assert "private-placeholder" not in repr(result)
    arguments, options = runner.calls[0]
    assert arguments[-4:] == [
        "--ignore-not-found",
        "-o",
        "json",
        "--request-timeout=15s",
    ]
    assert options["timeout_seconds"] == 20
    with pytest.raises(TypeError, match="explicit state"):
        bool(result)


def test_only_successful_empty_output_is_absence():
    result = probe_resource(Runner(output=" \n"), ["kubectl"], REFERENCE)
    assert result.state is ProbeState.ABSENT
    assert result.require_readable() is None
    assert result.exists() is False


@pytest.mark.parametrize(
    "error", ["Forbidden", "NotFound", "connection reset", "Unauthorized"]
)
def test_nonzero_read_is_not_absence(error):
    result = probe_resource(Runner(code=1, error=error), ["kubectl"], REFERENCE)
    assert result.state is ProbeState.ERROR
    with pytest.raises(ReleaseError, match="read failed"):
        result.exists()


@pytest.mark.parametrize("field", ["name", "namespace", "uid"])
def test_identity_mismatch_is_not_an_existing_resource(field):
    value = document()
    value["metadata"][field] = "" if field == "uid" else "other"
    result = probe_resource(Runner(output=json.dumps(value)), ["kubectl"], REFERENCE)
    assert result.state is ProbeState.ERROR
    with pytest.raises(ReleaseError, match="identity"):
        result.require_readable()


@pytest.mark.parametrize("output", ["[]", "not-json", '{"kind":"Deployment"}'])
def test_invalid_output_is_not_absence(output):
    result = probe_resource(Runner(output=output), ["kubectl"], REFERENCE)
    assert result.state is ProbeState.ERROR
