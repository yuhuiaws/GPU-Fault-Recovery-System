from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from scripts.e2e.regional import run_destr002_hyperpod_reboot as reboot
from scripts.e2e.regional import run_destr009_workload_restart as workload
from scripts.e2e.regional import run_destr012_managed_recovery_guard as managed
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr_branches import ready_runtime
from tests.regional._cov95_destr_edge_data import change
from tests.regional._cov95_destr_warm import profile, regional_settings


class Preflight:
    def __init__(
        self, module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        regional = regional_settings(tmp_path)
        self.settings = regional
        site = tmp_path / "site.yaml"
        site.write_text("fake site\n", encoding="utf-8")
        if module is reboot:
            self.config = module.Settings(
                regional,
                "node-a",
                "fake",
                "fake-hp",
                "arn:aws:iam::123456789012:role/fake",
                tmp_path / "previous.json",
            )
        elif module is workload:
            self.config = module.Settings(
                regional,
                site,
                module.DEFAULT_MANIFEST,
                "job-a",
                "job-a-a001",
                tmp_path / "previous.json",
            )
        else:
            self.config = module.Settings(
                regional,
                site,
                module.DEFAULT_A_MANIFEST,
                module.DEFAULT_D_MANIFEST,
                "job-a",
                "job-a-a001",
                "job-d",
                "job-d-a001",
                tmp_path / "previous.json",
            )
        self.frame: dict[str, Any] = {
            "state": {
                "release_id": "release-a",
                "profile": profile(),
                "agent": {"lifecycle_state": "ACTIVE"},
                "queue": {"depth": 0, "fault_backlog_depth": 0},
                "remote_commands": {"open_by_cluster": {}},
            },
            "node": {
                "name": "node-a",
                "uid": "uid-a",
                "ready": "True",
                "unschedulable": False,
                "taints": [],
            },
            "predecessor": {"valid": True},
            "tests": {"passed": True},
            "workloads": [],
            "gpu_workloads": [],
            "runtime": ready_runtime(),
        }
        self.frame["state"]["profile"]["capabilities"].append(
            {
                "capability": "nodeReboot",
                "mode": "OWN",
                "owner": "gpu-fault-hyperpod-adapter",
                "adapter": "regional-cluster-executor",
            }
        )
        self.nodes = [
            {**deepcopy(self.frame["node"]), "name": f"node-{i}", "uid": f"uid-{i}"}
            for i in range(3)
        ]
        self.proof = {
            "configured_cluster_name": "fake-hp",
            "positive": {"safe_to_submit": True, "node_recovery": "None"},
            "missing_isolation": {
                "safe_to_submit": False,
                "gate_failures": ["trusted scheduler isolation evidence is missing"],
            },
            "wrong_isolation": {
                "safe_to_submit": False,
                "gate_failures": ["trusted scheduler isolation evidence is missing"],
            },
            "disabled": {
                "safe_to_submit": False,
                "gate_failures": [
                    "HyperPod REBOOT mutation is disabled by configuration"
                ],
            },
        }
        self.calls: list[Any] = []
        monkeypatch.setattr(module, "RegionalLiveFixture", lambda _s: self)
        monkeypatch.setattr(
            module, "focused_tests", lambda *_a, **_k: deepcopy(self.frame["tests"])
        )
        monkeypatch.setattr(
            module,
            "predecessor_evidence",
            lambda *_a, **_k: deepcopy(self.frame["predecessor"]),
        )

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "release-a", "cluster_id": "cluster-a"}

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("store", kwargs))
        return deepcopy(self.frame["state"])

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return deepcopy(self.frame["node"])

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return deepcopy(self.frame["workloads"])

    def gpu_nodes(self) -> list[dict[str, Any]]:
        return deepcopy(self.nodes)

    def gpu_workloads(self) -> list[dict[str, Any]]:
        return deepcopy(self.frame["gpu_workloads"])

    def executor_python(self, *args: Any) -> dict[str, Any]:
        return deepcopy(self.proof)

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return {}

    def runtime_identity(self) -> dict[str, Any]:
        return deepcopy(self.frame["runtime"])

    def ready_pods(self, plane: str, app: str) -> list[dict[str, str]]:
        return [{"name": f"{app}-unit"}]

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        self.calls.append(("kubectl", args, kwargs))
        if args[0] == "exec":
            return json.dumps(
                {
                    "profile": self.frame["state"]["profile"],
                    "profile_sha256": "a" * 64,
                    "allowed_namespaces": ["training"],
                }
            )
        return json.dumps({"items": []})


@pytest.mark.parametrize("module", [reboot, workload, managed])
@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("predecessor", "valid"), False, "predecessor evidence"),
        (("state", "profile", "warnings"), ["drift"], "profile has warnings"),
        (("state", "queue", "fault_backlog_depth"), 1, "processor queue"),
        (
            ("state", "remote_commands", "open_by_cluster"),
            {"cluster-a": 1},
            "remote command queue",
        ),
        (("tests", "passed"), False, "regression tests failed"),
    ],
)
def test_preflight_checks_real_guard_against_each_fake_unproven_input(
    module: Any,
    path: tuple[str | int, ...],
    value: Any,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Preflight(module, tmp_path, monkeypatch)
    baseline = module.read_only_preflight(h.config, tmp_path)
    assert baseline["errors"] == [], baseline
    change(h.frame, path, value)
    result = module.read_only_preflight(h.config, tmp_path)
    assert any(expected in error for error in result["errors"]), result


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("node", "ready"), "False", "not Ready"),
        (("node", "unschedulable"), True, "already unschedulable"),
        (("node", "taints"), [{"key": "foreign"}], "pre-existing taints"),
        (("workloads",), [{"name": "foreign"}], "non-system running Pods"),
        (("state", "agent", "lifecycle_state"), "REVOKED", "not ACTIVE"),
        (("state", "profile", "capabilities"), [], "no nodeReboot"),
        (("state", "profile", "capabilities", 3, "mode"), "OBSERVE", "not OWN"),
        (("state", "event"), {"xid": 79}, "recent XID"),
    ],
)
def test_reboot_preflight_refuses_node_and_capability_drift(
    path: tuple[str | int, ...],
    value: Any,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Preflight(reboot, tmp_path, monkeypatch)
    change(h.frame, path, value)
    result = reboot.read_only_preflight(h.config, tmp_path)
    assert any(expected in error for error in result["errors"]), result


@pytest.mark.parametrize("module", [workload, managed])
@pytest.mark.parametrize(
    "defect", ["nodes", "workloads", "image", "profile", "owner", "runtime"]
)
def test_workload_preflight_refuses_missing_placement_and_release_proof(
    module: Any, defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = Preflight(module, tmp_path, monkeypatch)
    expected = ""
    if defect == "nodes":
        h.nodes = h.nodes[:2]
        expected = "fewer than three"
    elif defect == "workloads":
        h.frame["gpu_workloads"] = [{"name": "busy"}]
        expected = "GPU workloads"
    elif defect == "image":
        other = tmp_path / "other.yaml"
        other.write_text("image: fake-unpinned\n", encoding="utf-8")
        field = "manifest" if module is workload else "a_manifest"
        h.config = replace(h.config, **{field: other})
        expected = "image digest differs"
    elif defect == "profile":
        h.frame["state"]["profile"]["capabilities"] = []
        expected = "no workloadStop"
    elif defect == "owner":
        h.frame["state"]["profile"]["capabilities"][1]["mode"] = "OBSERVE"
        expected = "not OWN"
    else:
        h.frame["runtime"]["release_state"]["phase"] = "rolling-back"
        expected = "rollback is active"
    result = module.read_only_preflight(h.config, tmp_path)
    assert any(expected in error for error in result["errors"]), result


@pytest.mark.parametrize("module", [workload, managed])
def test_empty_candidate_inventory_never_fabricates_a_quiet_store_snapshot(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = Preflight(module, tmp_path, monkeypatch)
    h.nodes = []
    with pytest.raises(ValueError, match="processor queue"):
        module.read_only_preflight(h.config, tmp_path)
    assert not any(call[0] == "store" for call in h.calls), h.calls


@pytest.mark.parametrize("module", [workload, managed])
@pytest.mark.parametrize("field", ["site_file", "manifest"])
def test_preflight_missing_local_inputs_never_attempts_a_transport(
    module: Any, field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = Preflight(module, tmp_path, monkeypatch)
    field = "a_manifest" if module is managed and field == "manifest" else field
    h.config = replace(h.config, **{field: tmp_path / "missing"})
    with pytest.raises(RegionalFixtureError, match="does not exist"):
        module.read_only_preflight(h.config, tmp_path)
    assert h.calls == [], h.calls
