from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import seeded_command_fixture as seeded
from tests.regional._cov95_cap_cases import Clock
from tests.regional._cov95_common_seeded import RUN_ID, ResourceAPI, metadata, probe


def test_cpu_probe_and_seed_wrappers_keep_all_identity_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []
    response = {"status": "PENDING"}

    def control(*args: str, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return (
            " cpu-pod\n" if args[0] == "get" else "diagnostic\n" + json.dumps(response)
        )

    monkeypatch.setattr(seeded, "control", control)
    assert seeded.cpu_python("unit probe", "unit-argument") == response
    assert calls[-1][0][2:6] == (
        "cpu-pod",
        "--",
        "/opt/gpu-fault/control-plane/bin/python",
        "-",
    )
    assert calls[-1][1]["stdin"] == b"unit probe"
    assert calls[-1][1]["timeout"] == 120
    assert (
        seeded.seed_command(
            RUN_ID,
            owner="unit-owner",
            operation="COLLECT_DIAGNOSTIC_BUNDLE",
            node_ids=["synthetic-a", "synthetic-b"],
        )
        == response
    )
    assert calls[-1][0][-6:-1] == (
        RUN_ID,
        seeded.SYNTHETIC_CLUSTER_ID,
        "unit-owner",
        "COLLECT_DIAGNOSTIC_BUNDLE",
        "synthetic-a,synthetic-b",
    )
    assert int(calls[-1][0][-1]) >= 60
    assert seeded.command_snapshot("remote-unit") == response
    assert calls[-1][0][-1] == "remote-unit"
    response = {"remaining": [], "remaining_links": 0}
    selected = seeded.seed_identity(RUN_ID)
    assert seeded.purge_seed(selected) == response
    assert calls[-1][0][-4:] == (
        selected["command_id"],
        selected["workflow_id"],
        selected["incident_id"],
        selected["event_id"],
    )
    assert seeded.run_identity(tmp_path / "run-ABC", 2, "prefix") == "prefix-abc-a2"


@pytest.mark.parametrize(
    "nodes,lease", [([], 60), ([""], 60), (["has,comma"], 60), (["unit"], 59)]
)
def test_invalid_seed_scope_never_reaches_the_transport(
    monkeypatch: pytest.MonkeyPatch, nodes: list[str], lease: int
) -> None:
    calls = []
    monkeypatch.setattr(
        seeded, "control", lambda *args, **kwargs: calls.append(args) or ""
    )
    with pytest.raises(seeded.SeededCommandError):
        seeded.seed_command(
            RUN_ID,
            owner="unit",
            operation="COLLECT_DIAGNOSTIC_BUNDLE",
            node_ids=nodes,
            lease_seconds=lease,
        )
    assert calls == []


def test_registry_residuals_report_scope_without_copying_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        seeded,
        "load_registry",
        lambda: [
            {"cluster_id": "normal", "synthetic": False},
            {
                "cluster_id": "unit",
                "synthetic": True,
                "synthetic_run_id": RUN_ID,
                "token": "fixture-only",
            },
            {"cluster_id": "perf-cap-000", "synthetic_run_id": "legacy"},
        ],
    )
    assert seeded.registry_residuals() == {
        "count": 2,
        "entries": [
            {"cluster_id": "unit", "synthetic_run_id": RUN_ID},
            {"cluster_id": "perf-cap-000", "synthetic_run_id": "legacy"},
        ],
    }


@pytest.mark.parametrize("stage", ["database", "registry", "kubernetes"])
@pytest.mark.parametrize("value", [None, False, 1])
def test_preflight_requires_explicit_zero_in_every_residual_domain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str, value: Any
) -> None:
    results = {
        "database": {"total": 0},
        "registry": {"count": 0},
        "kubernetes": {"count": 0},
    }
    results[stage]["total" if stage == "database" else "count"] = value
    monkeypatch.setattr(
        seeded, "database_residuals", lambda _prefix: results["database"]
    )
    monkeypatch.setattr(seeded, "registry_residuals", lambda: results["registry"])
    monkeypatch.setattr(
        seeded, "kubernetes_residuals", lambda _probe: results["kubernetes"]
    )
    with pytest.raises(seeded.SeededCommandError, match="preflight found residuals"):
        seeded.preflight_residuals(probe(tmp_path), tmp_path)


def test_kubernetes_residual_read_error_is_not_resource_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = ResourceAPI()
    monkeypatch.setattr(seeded, "dataplane", api)
    selected = probe(tmp_path)
    api.resources[("pod", selected.pod)] = metadata("pod")
    result = seeded.kubernetes_residuals(selected)
    assert result["count"] == 1
    assert result["resources"][f"pod/{selected.pod}"] is True
    api.read_error = TimeoutError("synthetic read failure")
    with pytest.raises(TimeoutError):
        seeded.kubernetes_residuals(selected)


def test_file_and_state_probes_keep_paths_and_stop_at_their_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = ResourceAPI()
    selected = probe(tmp_path)
    clock = Clock()
    monkeypatch.setattr(seeded, "dataplane", api)
    monkeypatch.setattr(seeded, "time", clock)
    api.file_responses = ["", "present"]
    seeded.wait_file(selected, "/state/unit file", 3)
    assert clock.elapsed == 1
    assert "'/state/unit file'" in api.calls[0][0][-1]
    assert seeded.file_present(selected, "/state/ready.json") is True
    api.file_responses = [""]
    assert seeded.file_present(selected, "/state/missing") is False
    with pytest.raises(seeded.SeededCommandError, match="timed out waiting"):
        seeded.wait_file(selected, "/state/missing", 2)
    assert clock.elapsed == 3
    assert seeded.read_state(selected, "/state/ready.json") == {"ready": True}
    seeded.touch(selected, "/state/stop")
    seeded.remove(selected, "/state/stop")
    assert api.calls[-2][0] == ("exec", selected.pod, "--", "touch", "/state/stop")
    assert api.calls[-1][0] == ("exec", selected.pod, "--", "rm", "-f", "/state/stop")
    assert seeded.pod_logs(selected) == "unit log"
    assert seeded.pod_phase(selected) == ""


def test_command_and_executor_waits_preserve_final_observation_on_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = Clock()
    monkeypatch.setattr(seeded, "time", clock)
    states = iter([{"status": "PENDING"}, {"status": "SUCCEEDED"}])
    monkeypatch.setattr(seeded, "command_snapshot", lambda _command: next(states))
    assert seeded.wait_command(
        "unit", lambda value: value["status"] == "SUCCEEDED", 3
    ) == {"status": "SUCCEEDED"}
    assert clock.elapsed == 2
    monkeypatch.setattr(
        seeded, "command_snapshot", lambda _command: {"status": "PENDING"}
    )
    with pytest.raises(seeded.SeededCommandError, match="expected state"):
        seeded.wait_command("unit", lambda value: value["status"] == "SUCCEEDED", 1)
    selected = probe(tmp_path)
    monkeypatch.setattr(seeded, "read_state", lambda *_args: {"status": "pending"})
    assert seeded.wait_executor_state(
        selected, lambda value: value["status"] == "ready", 2
    ) == {"status": "pending"}
    monkeypatch.setattr(seeded, "read_state", lambda *_args: {"status": "ready"})
    assert seeded.wait_executor_state(
        selected, lambda value: value["status"] == "ready", 2
    ) == {"status": "ready"}


@pytest.mark.parametrize("value", ["{}", "[]", "null"])
def test_resource_metadata_rejects_untyped_success_responses(value: str) -> None:
    with pytest.raises(seeded.SeededCommandError, match="metadata is not an object"):
        seeded.resource_metadata("pod", "unit", client=lambda *_args, **_kwargs: value)


@pytest.mark.parametrize("fault", ["run", "empty-uid", "numeric-uid"])
def test_uid_journal_must_identify_the_same_run_and_a_real_uid(
    tmp_path: Path, fault: str
) -> None:
    value = {"run_id": RUN_ID, "resources": {"pod/unit": "uid-pod"}}
    if fault == "run":
        value["run_id"] = "other"
    else:
        value["resources"]["pod/unit"] = "" if fault == "empty-uid" else 1
    seeded.write_json(tmp_path / "probe-resource-identities.json", value)
    with pytest.raises(seeded.SeededCommandError, match="another run|UID proof"):
        seeded.probe_resource_uid(tmp_path, "pod", "unit", RUN_ID)
