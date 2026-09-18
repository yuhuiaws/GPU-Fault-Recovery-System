"""Negative Collector cases reject unsafe premises and inconsistent decisions."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import collector_negative_evidence as negative
from scripts.e2e.regional import run_collector_acceptance as runner
from tests.regional._cov95_collect_net import Clock, no_external_effects  # noqa: F401
from tests.regional.test_collector_remaining_guards import FirmwareApi, firmware_proof


@pytest.mark.parametrize("problem", ["projection", "profile-drift", "configured-node"])
def test_firmware_premise_rejects_unproved_cpu_or_node_configuration(
    problem: str,
) -> None:
    class Api(FirmwareApi):
        profile_reads = 0

        def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
            value = super().store_snapshot(**kwargs)
            self.profile_reads += 1
            if problem == "profile-drift" and self.profile_reads > 1:
                value["profile"]["profile_version"] = "changed"
            return value

        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            if problem == "projection" and args[0] == "exec":
                return "[]"
            return super().kubectl(plane, *args, **kwargs)

    node = firmware_proof()["node_agent"]
    if problem == "configured-node":
        node["target_present"] = True
    with pytest.raises(runner.RegionalFixtureError):
        negative.firmware_premise(Api(), node, node="node-a")
    assert "CPU firmware configuration is unknown" in negative.firmware_premise_errors(
        {"cpu_replicas": [], "node_agent": node}
    )


@pytest.mark.parametrize(
    "problem",
    [None, "wrong-action", "firmware-step", "executed-step", "command", "boot"],
)
def test_firmware_negative_case_checks_exact_blocked_direction_and_restores(
    tmp_path: Path, problem: str | None
) -> None:
    api = FirmwareApi()
    state = {
        "decisions": [
            {"official_action": "UPDATE_SWFW", "workflow_request_id": "workflow-a"}
        ],
        "workflows": [
            {
                "request_id": "workflow-a",
                "status": "BLOCKED",
                "blocked_kind": "SAFETY_SETTLED",
                "blocked_reasons": sorted(negative.FIRMWARE_REASONS),
                "official_steps": [],
                "step_executions": [],
            }
        ],
        "commands": [],
    }
    if problem == "wrong-action":
        state["decisions"][0]["official_action"] = "RESTART_NODE"
    elif problem == "firmware-step":
        state["workflows"][0]["official_steps"] = [
            {"operation": "UPDATE_SOFTWARE_FIRMWARE"}
        ]
    elif problem == "executed-step":
        state["workflows"][0]["step_executions"] = [
            {"operation": "RESET_GPU", "status": "SUCCEEDED", "details": {}}
        ]
    elif problem == "command":
        state["commands"] = [
            {
                "status": "SUCCEEDED",
                "step": {"operation": "RESET_GPU"},
                "result_details": {},
            }
        ]
    calls = []
    reads = 0

    def snapshot() -> dict[str, Any]:
        nonlocal reads
        reads += 1
        return {
            "boot_id": "boot-b" if problem == "boot" and reads > 1 else "boot-a",
            "gpu_inventory": [{"pci_bdf": "0000:01:00.0"}],
        }

    def execute(verb: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((verb, args))
        return firmware_proof()["node_agent"] if verb == "firmware-premise" else {}

    fixture = SimpleNamespace(
        node="node-a",
        regional=api,
        snapshot=snapshot,
        execute=execute,
        wait_marker=lambda *a, **k: deepcopy(state),
        restore_incidents=lambda value, **kw: calls.append(("restore", value)) or [],
    )
    result = runner.run_collect010(fixture, tmp_path, 1, "profile-a")
    assert result["verdict"] == ("PASS" if problem is None else "FAIL")
    assert calls[-1][0] == "restore"
    assert len([call for call in calls if call[0] == "write-xid"]) == 1


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "busy",
        "scope",
        "participating-premise",
        "event",
        "decision",
        "workflow-missing",
        "workflow-open",
        "participants",
        "command",
        "boot",
    ],
)
def test_sxid_negative_cases_bind_scope_and_stop_after_first_failed_direction(
    monkeypatch: Any, tmp_path: Path, problem: str | None
) -> None:
    monkeypatch.setattr(runner, "time", Clock())
    calls = []
    fixtures = []
    for index, scope in enumerate(("ACCESS", "UNKNOWN")):
        node = f"node-{index}"
        current_problem = problem if index == 0 else None
        event_id = f"event-{index}"
        proof = {
            "cluster_id": "cluster-a",
            "node_id": node,
            "workload_state": "IDLE",
            "link_scope": scope,
            "link_scope_source": "NVIDIA_PRODUCT_INVARIANT",
            "participating_gpu_uuids": [],
        }
        state = {
            "fabric_events": [
                {
                    "event_id": event_id,
                    "sxid": 11001,
                    "classification": "FATAL",
                    "port": "12",
                    "switch_id": "switch-a" if index == 0 else None,
                }
            ],
            "decisions": [
                {
                    "event_id": event_id,
                    "event_type": "SXID",
                    "disposition": "BLOCKED_MISSING_EVIDENCE",
                    "official_action": negative.SCOPE_ACTIONS[scope],
                    "action": None,
                    "reasons": [negative.SCOPE_REASONS[scope]],
                    "workflow_request_id": f"workflow-{index}",
                    "incident_id": f"incident-{index}",
                }
            ],
            "workflows": [
                {
                    "request_id": f"workflow-{index}",
                    "incident_id": f"incident-{index}",
                    "status": "BLOCKED",
                    "blocked_kind": "SAFETY_SETTLED",
                    "official_action": negative.SCOPE_ACTIONS[scope],
                }
            ],
            "incidents": [{"incident_id": f"incident-{index}", "gpu_uuids": []}],
            "commands": [],
        }
        if current_problem == "scope":
            proof["node_id"] = "foreign"
        elif current_problem == "participating-premise":
            proof["participating_gpu_uuids"] = ["GPU-a"]
        elif current_problem == "event":
            state["fabric_events"] = []
        elif current_problem == "decision":
            state["decisions"][0]["action"] = "RESET_GPU"
        elif current_problem == "workflow-missing":
            state["workflows"][0]["official_action"] = "OTHER"
        elif current_problem == "workflow-open":
            state["workflows"][0]["status"] = "RUNNING"
        elif current_problem == "participants":
            state["incidents"][0]["gpu_uuids"] = ["GPU-a"]
        elif current_problem == "command":
            state["commands"] = [
                {
                    "status": "SUCCEEDED",
                    "step": {"operation": "RESET_GPU"},
                    "result_details": {},
                }
            ]
        regional = SimpleNamespace(
            settings=SimpleNamespace(cluster_id="cluster-a"),
            business_workloads=lambda node, problem=current_problem: ["workload"]
            if problem == "busy"
            else [],
            cpu_python=lambda *args, proof=proof: deepcopy(proof),
        )
        observations = {"after": False}

        def snapshot(
            observations=observations, problem=current_problem
        ) -> dict[str, Any]:
            return {
                "boot_id": "boot-b"
                if problem == "boot" and observations["after"]
                else "boot-a",
                "gpu_inventory": [{"pci_bdf": "0000:01:00.0"}],
            }

        def execute(
            verb: str,
            *args: str,
            node=node,
            observations=observations,
            state=state,
            **kwargs: Any,
        ) -> dict[str, Any]:
            calls.append((node, verb))
            if verb == "append-sxid":
                observations["after"] = True
                sxid = int(args[args.index("--sxid") + 1])
                if state["fabric_events"]:
                    state["fabric_events"][0]["sxid"] = sxid
                    state["fabric_events"][0]["event_id"] = f"{node}-{sxid}"
                    state["decisions"][0]["event_id"] = f"{node}-{sxid}"
            return {"product": "H100"}

        fixture = SimpleNamespace(
            node=node,
            regional=regional,
            snapshot=snapshot,
            execute=execute,
            wait_marker=lambda *a, state=state, **kw: deepcopy(state),
            restore_incidents=lambda *a, node=node, **kw: calls.append(
                (node, "restore")
            )
            or [],
        )
        fixtures.append(fixture)
    if problem in {"busy", "scope", "participating-premise"}:
        with pytest.raises(runner.RegionalFixtureError):
            runner.run_collect011(fixtures, tmp_path, 1, "profile-a")
        assert not any(verb == "append-sxid" for _, verb in calls), (
            "unproven premise must stop before fatal SXID injection"
        )
    elif problem == "workflow-open":
        with pytest.raises(runner.RegionalFixtureError, match="operator hold"):
            runner.run_collect011(fixtures, tmp_path, 1, "profile-a")
        assert not any(verb == "restore" for _, verb in calls), (
            "an open workflow must preserve its operator hold without restoration"
        )
    else:
        result = runner.run_collect011(fixtures, tmp_path, 1, "profile-a")
        assert result["verdict"] == ("PASS" if problem is None else "FAIL")
        injections = [node for node, verb in calls if verb == "append-sxid"]
        assert injections == (
            ["node-0", "node-1"] * 3 if problem is None else ["node-0"]
        )
        assert [node for node, verb in calls if verb == "restore"] == injections


def test_sxid_slot_refuses_when_every_allowed_address_is_a_gpu() -> None:
    with pytest.raises(runner.RegionalFixtureError, match="no PCI slot"):
        runner.nvswitch_pci_bdf(
            [
                {"pci_bdf": f"0000:{bus}:00.0"}
                for bus in ("ab", "ac", "ad", "ae", "af", "ba", "bb", "bc")
            ]
        )


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "details",
        "wrong-accepted",
        "right-refused",
        "boot",
        "provider",
        "not-ready",
    ],
)
def test_mechanical_acknowledgement_is_observed_before_accepting_final_result(
    monkeypatch: Any, tmp_path: Path, problem: str | None
) -> None:
    monkeypatch.setattr(runner, "time", Clock())
    calls = []
    acknowledged = False
    annotation = "gpu-fault.io/mechanical"
    details = (
        {}
        if problem == "details"
        else {
            "required_annotation": annotation,
            "required_annotation_value": "incident-a:1",
        }
    )

    def state() -> dict[str, Any]:
        return {
            "workflows": [
                {
                    "status": "SUCCEEDED" if problem == "wrong-accepted" else "RUNNING",
                    "step_executions": [
                        {
                            "operation": "CHECK_MECHANICALS",
                            "status": "WAITING",
                            "details": details,
                        }
                    ],
                }
            ]
        }

    def kubectl(plane: str, *args: str, **kwargs: Any) -> str:
        nonlocal acknowledged
        calls.append(args)
        if any(value.endswith("=incident-a:1") for value in args):
            acknowledged = True
        return ""

    regional = SimpleNamespace(
        kubectl=kubectl,
        node_snapshot=lambda node: {
            "ready": "False" if problem == "not-ready" else "True",
            "unschedulable": False,
            "taints": [],
        },
        provider_events=lambda *a: [{"event_name": "RebootClusterNodes"}]
        if problem == "provider"
        else [],
        provider_events_provisional=lambda ended: True,
    )
    fixture = SimpleNamespace(
        node="node-a",
        regional=regional,
        snapshot=lambda: {
            "boot_id": "boot-b" if acknowledged and problem == "boot" else "boot-a",
            "gpu_inventory": [{"pci_bdf": "0000:01:00.0"}],
        },
        execute=lambda *a, **k: {},
        store_snapshot=lambda *a, **k: state(),
        wait_marker=lambda *a, **k: {
            "workflows": [
                {"status": "FAILED" if problem == "right-refused" else "SUCCEEDED"}
            ]
        },
    )
    if problem == "details":
        with pytest.raises(runner.RegionalFixtureError, match="annotation details"):
            runner.run_collect009(fixture, tmp_path, 1)
        assert calls == [], "missing annotation identity cannot authorize a write"
    else:
        result = runner.run_collect009(fixture, tmp_path, 1)
        assert result["verdict"] == ("PASS" if problem is None else "FAIL")
        assert calls[-1][-1] == annotation + "-"
        assert result["wrong_acknowledgement_samples"] >= 3
