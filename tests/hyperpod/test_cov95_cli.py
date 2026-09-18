from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault import hyperpod, hyperpod_cli
from gpu_fault.hyperpod import HyperPodAction, HyperPodRecoveryState


class Adapter:
    def __init__(self) -> None:
        self.calls = []

    def result(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        return SimpleNamespace(
            model_dump_json=lambda **options: json.dumps({"stage": name})
        )

    def discover(self):
        return self.result("discover")

    def resolve_recovery_ownership(self, evidence):
        return self.result("ownership", evidence)

    def preflight(self, *args, **kwargs):
        return self.result("preflight", *args, **kwargs)

    def submit(self, *args, **kwargs):
        return self.result("submit", *args, **kwargs)


@pytest.fixture
def harness(monkeypatch):
    adapter = Adapter()
    configurations = []
    config = object()

    def from_environment(**kwargs):
        configurations.append(kwargs)
        return config

    def construct(value):
        assert value is config, "the exact selected configuration reaches the adapter"
        return adapter

    monkeypatch.setattr(
        hyperpod.HyperPodAdapterConfig, "from_environment", from_environment
    )
    monkeypatch.setattr(hyperpod_cli, "HyperPodLifecycleAdapter", construct)
    monkeypatch.setattr(hyperpod, "HyperPodLifecycleAdapter", construct)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu-fault-hyperpod", "--cluster", "test-cluster", "--region", "us-east-1"],
    )
    return adapter, configurations


def test_discovery_uses_explicit_cluster_configuration(harness, capsys) -> None:
    adapter, configurations = harness
    sys.argv.append("discover")
    hyperpod_cli.main()
    assert configurations == [
        {"cluster_name": "test-cluster", "region_name": "us-east-1"}
    ], "discovery cannot silently select another cluster"
    assert json.loads(capsys.readouterr().out) == {"stage": "discover"}, (
        "structured output is retained"
    )
    assert adapter.calls == [("discover", (), {})], "discovery never calls a mutation"


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("enabled", HyperPodRecoveryState.ENABLED),
        ("disabled", HyperPodRecoveryState.DISABLED),
        ("unknown", HyperPodRecoveryState.UNKNOWN),
    ],
)
def test_ownership_keeps_the_explicit_evidence_state(
    harness, capsys, state, expected
) -> None:
    adapter, _ = harness
    sys.argv.extend(
        ["ownership", "--workload-recovery", state, "--workload-id", "job-a"]
    )
    hyperpod_cli.main()
    name, args, kwargs = adapter.calls[0]
    assert name == "ownership" and not kwargs, (
        "ownership resolution is the only operation"
    )
    assert args[0].state is expected and args[0].workload_id == "job-a", (
        "UNKNOWN is not coerced into disabled"
    )
    assert json.loads(capsys.readouterr().out)["stage"] == "ownership", (
        "the adapter verdict is returned"
    )


@pytest.mark.parametrize(
    ("arguments", "action"),
    [
        (["preflight"], HyperPodAction.REBOOT),
        (["preflight", "--action", "replace"], HyperPodAction.REPLACE),
        (["reboot"], HyperPodAction.REBOOT),
        (["replace"], HyperPodAction.REPLACE),
    ],
)
def test_action_commands_remain_preflight_only_without_execute(
    harness, capsys, arguments, action
) -> None:
    adapter, _ = harness
    sys.argv.extend([*arguments, "--node", "node-a", "--isolated-node", "node-a"])
    hyperpod_cli.main()
    assert adapter.calls == [
        ("preflight", (action, ["node-a"]), {"isolation_verified_nodes": ["node-a"]})
    ], "an action name alone does not authorize provider mutation"
    assert json.loads(capsys.readouterr().out)["stage"] == "preflight", (
        "preflight output is preserved"
    )


@pytest.mark.parametrize(
    "missing",
    [
        "--confirm-cluster",
        "--workflow-fencing-token",
        "--expected-fencing-token",
        "--idempotency-key",
    ],
)
def test_execute_requires_every_confirmation_before_adapter_submit(
    harness, missing
) -> None:
    adapter, _ = harness
    proof = {
        "--confirm-cluster": "test-cluster",
        "--workflow-fencing-token": "7",
        "--expected-fencing-token": "7",
        "--idempotency-key": "test-command",
    }
    sys.argv.extend(["reboot", "--node", "node-a", "--execute"])
    for flag, value in proof.items():
        if flag != missing:
            sys.argv.extend([flag, value])
    with pytest.raises(SystemExit, match=missing):
        hyperpod_cli.main()
    assert [call[0] for call in adapter.calls] == ["preflight"], (
        "missing authorization cannot reach submit"
    )


def test_complete_request_passes_fencing_and_scope_to_the_mocked_adapter(
    harness, capsys
) -> None:
    adapter, _ = harness
    sys.argv.extend(
        [
            "reboot",
            "--node",
            "node-a",
            "--isolated-node",
            "node-a",
            "--execute",
            "--confirm-cluster",
            "test-cluster",
            "--workflow-fencing-token",
            "7",
            "--expected-fencing-token",
            "7",
            "--idempotency-key",
            "test-command",
        ]
    )
    hyperpod_cli.main()
    assert adapter.calls[-1] == (
        "submit",
        (HyperPodAction.REBOOT, ["node-a"]),
        {
            "isolation_verified_nodes": ["node-a"],
            "confirm_cluster_name": "test-cluster",
            "workflow_fencing_token": 7,
            "expected_fencing_token": 7,
            "idempotency_key": "test-command",
        },
    ), "CLI must not discard the adapter's authorization inputs"
    assert json.loads(capsys.readouterr().out)["stage"] == "submit", (
        "the mocked result is serialized"
    )


def test_module_entrypoint_uses_the_same_mocked_adapter(harness, capsys) -> None:
    adapter, _ = harness
    sys.argv.append("discover")
    runpy.run_path(str(Path(hyperpod_cli.__file__)), run_name="__main__")
    assert [call[0] for call in adapter.calls] == ["discover"], (
        "entrypoint does not bypass the selected adapter"
    )
    assert json.loads(capsys.readouterr().out)["stage"] == "discover", (
        "entrypoint produces structured output"
    )
