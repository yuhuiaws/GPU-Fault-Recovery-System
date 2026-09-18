from __future__ import annotations

import json
import subprocess

import pytest

from gpu_fault.admin import bootstrap_common as common
from gpu_fault.admin.bootstrap_task_inputs import TaskInputSpec


@pytest.mark.parametrize("value", ["", "not-an-arn", "arn:aws:eks:invalid"])
def test_bootstrap_rejects_malformed_arn_before_any_lookup(value):
    with pytest.raises(common.BootstrapError, match="invalid AWS ARN"):
        common.Arn.parse(value)


def test_bootstrap_resource_name_cannot_normalize_to_empty():
    with pytest.raises(common.BootstrapError, match="cannot derive a resource name"):
        common.safe_name("   $$$  ")


@pytest.mark.parametrize("version", [1, common.BOOTSTRAP_STATE_VERSION + 1])
def test_bootstrap_journal_version_requires_revalidation_or_refusal(tmp_path, version):
    path = tmp_path / "state.json"
    document = {
        "schema_version": version,
        "site_id": "example",
        "phase": "old",
        "completed_tasks": ["example"],
        "resources": {"example": {"owned": True}},
    }
    path.write_text(json.dumps(document))
    if version > common.BOOTSTRAP_STATE_VERSION:
        with pytest.raises(common.BootstrapError, match="newer version"):
            common.BootstrapState(path, site_id="example")
        assert json.loads(path.read_text()) == document
    else:
        state = common.BootstrapState(path, site_id="example")
        assert state.value["completed_tasks"] == []
        assert state.result("example") == {"owned": True}
        assert state.value["schema_version"] == common.BOOTSTRAP_STATE_VERSION


def test_bootstrap_journal_rejects_other_site_without_rewriting(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"site_id":"different"}')
    before = path.read_bytes()
    with pytest.raises(common.BootstrapError, match="another site"):
        common.BootstrapState(path, site_id="example")
    assert path.read_bytes() == before


@pytest.mark.parametrize("same", [False, True])
def test_legacy_input_binding_invalidates_only_changed_whole_run(tmp_path, same):
    state = common.BootstrapState(tmp_path / "state.json", site_id="example")
    state.bind_inputs("a" * 64)
    state.record("task", "recorded")
    state.complete("task")
    state.bind_inputs(("a" if same else "b") * 64)
    assert state.is_complete("task") is same
    assert state.result("task") == "recorded"


def test_empty_cache_hit_record_does_not_write_a_new_journal(tmp_path):
    state = common.BootstrapState(tmp_path / "state.json", site_id="example")
    state.record_cache_hits(())
    assert not state.path.exists(), "empty cache-hit update created a journal"
    with pytest.raises(common.BootstrapError, match="has not recorded a result"):
        state.result("missing")


def test_task_policy_mismatch_stops_before_scheduling(tmp_path):
    calls = []
    state = common.BootstrapState(tmp_path / "state.json", site_id="example")
    with pytest.raises(common.BootstrapError, match="policies do not cover the graph"):
        common.run_parallel(
            {"task": lambda: calls.append("task")},
            state=state,
            input_policies={
                "other": TaskInputSpec("other", always_revalidate_reason="example")
            },
        )
    assert calls == []


def test_bootstrap_task_counts_real_runner_boundary_calls_without_subprocesses(
    tmp_path, monkeypatch
):
    commands = []

    def run(arguments, **_options):
        commands.append(list(arguments))
        return subprocess.CompletedProcess(arguments, 0, "example", "")

    monkeypatch.setattr(common, "run_command", run)
    runner = common.CommandRunner()
    state = common.BootstrapState(tmp_path / "state.json", site_id="example")
    result = common.run_parallel(
        {
            "task": lambda: [
                runner.run(arguments)
                for arguments in (
                    ["aws", "example-read"],
                    ["kubectl", "example-read"],
                    ["example-helper"],
                )
            ]
        },
        state=state,
    )
    assert result == {"task": ["example", "example", "example"]}
    assert state.value["task_reports"]["task"]["runner_commands"] == {
        "total": 3,
        "aws": 1,
        "kubectl": 1,
    }
    assert len(commands) == 3


@pytest.mark.parametrize("value", [None, "", "first\nsecond", 3])
def test_agent_digest_rejects_invalid_installer_environment_values(
    tmp_path, monkeypatch, value
):
    calls = []
    environment = {
        key: "example" for key in common.NODE_INSTALLER_CONFIG_DIGEST_ENVIRONMENT_KEYS
    }
    environment[next(iter(environment))] = value

    def run(arguments, **_options):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, json.dumps(environment), "")

    monkeypatch.setattr(common, "run_command", run)
    with pytest.raises(common.BootstrapError, match="invalid values"):
        common.compute_agent_config_digest(
            common.CommandRunner(),
            repository_root=tmp_path,
            runtime_profile_version="example",
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "outputs",
    [
        ["invalid-json"],
        [
            json.dumps(
                {
                    key: "example"
                    for key in common.NODE_INSTALLER_CONFIG_DIGEST_ENVIRONMENT_KEYS
                }
            ),
            "invalid-digest",
        ],
    ],
)
def test_agent_digest_requires_valid_json_and_exact_digest(
    tmp_path, monkeypatch, outputs
):
    pending = iter(outputs)
    monkeypatch.setattr(
        common,
        "run_command",
        lambda arguments, **_options: subprocess.CompletedProcess(
            arguments, 0, next(pending), ""
        ),
    )
    with pytest.raises(common.BootstrapError, match="invalid JSON|digest is invalid"):
        common.compute_agent_config_digest(
            common.CommandRunner(),
            repository_root=tmp_path,
            runtime_profile_version="example",
        )
