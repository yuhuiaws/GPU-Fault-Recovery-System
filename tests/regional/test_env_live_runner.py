"""run_env_live plan/execute branches, driven with fakes (no cluster).

The execute tests run the real bounded observation against a scripted world
(``_env_live_world``): a fake clock the poll loop's sleep advances, a proxy
whose reads advance one per collector interval, and product records that
appear at scripted offsets from the arm.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import env_live_fixture as fixture
from scripts.e2e.regional import run_env_live as runner
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._env_live_world import (
    H200_LIMITS,
    World,
    baseline_values,
    h200_preflight,
    review_chain,
)

H200_TARGET = {
    "gpu_uuid": "GPU-aaaa",
    "gpu_index": "0",
    "product": "NVIDIA H200",
    "pci_bdf": "59:00",
    "limits": dict(H200_LIMITS),
}
COLLECTOR_ENV = {
    "GPU_FAULT_METRICS_INTERVAL_SECONDS": "15",
    "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES": "3",
    "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "300",
}


class FakeRegionalSettings:
    cluster_id = "gpu-eks-cluster"

    def environment(self) -> dict[str, str]:
        return {"GPU_FAULT_CLUSTER_ID": self.cluster_id}


def _settings(case_id: str, **overrides: Any) -> fixture.EnvLiveSettings:
    values = {
        "regional": FakeRegionalSettings(),
        "case_id": case_id,
        "confirmation": runner.confirmation_for(case_id),
        "node": "ip-10-0-43-5",
        "host_probe_image": "public.ecr.aws/x@sha256:" + "0" * 64,
        "gpu_uuid": "GPU-aaaa",
        "predecessor_path": None,
        "predecessor_id": None,
        "site_file": None,
        "isolate_confirm": runner.isolate_token(case_id),
        "stop_confirm": runner.stop_token(case_id),
    }
    values.update(overrides)
    return fixture.EnvLiveSettings(**values)


def _preflight(
    target: dict[str, Any] | None = H200_TARGET, errors: list[str] | None = None
) -> dict[str, Any]:
    return h200_preflight(
        devices=[target] if target else [], target_gpu=target or {}, errors=errors or []
    )


class FakeRegional:
    world: World

    def __init__(self, settings: Any) -> None:
        self.settings = settings

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "rel-1", "cluster_id": "gpu-eks-cluster"}

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return self.world.node_snapshot()


class RecordingController:
    """Stands in for EnvReplayController; records the proxy lifecycle order."""

    instances: list["RecordingController"] = []
    restored_endpoint: str | None = None
    after_state = "active"
    world: World

    def __init__(self, regional: Any, **options: Any) -> None:
        self.options = options
        self.calls: list[str] = []
        RecordingController.instances.append(self)

    def create(self) -> None:
        self.calls.append("create")

    def preflight(self) -> dict[str, Any]:
        self.calls.append("preflight")
        return {
            "metrics_unit": {"ActiveState": "active"},
            "collector_env": COLLECTOR_ENV,
        }

    def deploy_proxy(
        self, plan_path: Path, gpu_uuid: str, **options: Any
    ) -> dict[str, Any]:
        self.calls.append("deploy_proxy")
        self.plan = Path(plan_path).read_text()
        return {"endpoint": "http://127.0.0.1:9411/metrics", "status": {"armed": False}}

    def repoint(self, **options: Any) -> dict[str, Any]:
        self.calls.append("repoint")
        return self.world.repoint()

    def status(self, **options: Any) -> dict[str, Any]:
        self.calls.append("status")
        return self.world.status()

    def arm(self, **options: Any) -> dict[str, Any]:
        self.calls.append("arm")
        return self.world.arm()

    def cleanup(self) -> dict[str, Any]:
        self.calls.append("cleanup")
        # The live probe's shape: no endpoint override is left on the unit
        # (None), so the collector runs with collector.env's endpoint again.
        return {
            "restore": {
                "restored_endpoint": self.restored_endpoint,
                "original_endpoint": "http://127.0.0.1:9400/metrics",
                "after": {"ActiveState": self.after_state},
                "dropin_removed": True,
            },
            "stop_proxy": {"ok": True},
            "probe_residuals": {"pod": False, "configmap": False},
        }


@pytest.fixture(autouse=True)
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    """A fresh scripted world per test, its clock starting at the real now."""

    value = World(start=datetime.now(timezone.utc))
    RecordingController.instances = []
    monkeypatch.setattr(RecordingController, "world", value, raising=False)
    monkeypatch.setattr(FakeRegional, "world", value, raising=False)
    monkeypatch.setattr(runner, "RegionalLiveFixture", FakeRegional)
    monkeypatch.setattr(fixture, "EnvReplayController", RecordingController)
    monkeypatch.setattr(fixture, "gpu_records", value.gpu_state)
    monkeypatch.setattr(fixture, "node_records", value.activity)
    monkeypatch.setattr(
        fixture, "node_state", lambda regional, node: value.node_snapshot()
    )
    monkeypatch.setattr(fixture, "utc_now", value.clock)
    monkeypatch.setattr(runner, "SLEEP", value.sleep)
    monkeypatch.setattr(
        runner, "read_only_preflight", lambda settings, case_dir: _preflight()
    )
    return value


def _window() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=1)


# --------------------------------------------------------------------------- #
def test_confirmation_and_branch_tokens_follow_the_case_number() -> None:
    assert runner.confirmation_for("GF-ENV-LIVE-004") == "ENVLIVE004_EXECUTE"
    assert runner.isolate_token("GF-ENV-LIVE-011") == "ENVLIVE011_ISOLATE"
    assert runner.stop_token("GF-ENV-LIVE-007") == "ENVLIVE007_STOP"
    with pytest.raises(RegionalFixtureError):
        runner.case_number("GF-REGIONAL-DESTR-001")


def test_plan_details_names_runnable_and_not_run_variants_for_h200() -> None:
    details = runner.plan_details(_settings("GF-ENV-LIVE-004"), _preflight())
    assert "core_tlimit/severe" in details["replay_plans"]
    assert "core_absolute" in details["not_run_variants"]
    assert details["branch_confirmations"] == {
        "isolate": "ENVLIVE004_ISOLATE",
        "stop": "ENVLIVE004_STOP",
    }
    assert details["proxy_mechanism"]["tool"] == "tools/gpu_environment_replay.py"


def test_plan_details_011_is_not_run_without_an_absolute_shutdown() -> None:
    details = runner.plan_details(_settings("GF-ENV-LIVE-011"), _preflight())
    assert "shutdown_absolute" in details["not_run_variants"]
    assert details["replay_plans"] == {}


def test_plan_details_012_is_not_applicable() -> None:
    details = runner.plan_details(_settings("GF-ENV-LIVE-012"), _preflight())
    assert details["verdict_if_not_run"] == "NOT_RUN"
    assert "physical" in details["not_run_reason"]


def test_execute_012_records_not_run_without_touching_a_node(tmp_path: Path) -> None:
    code = runner.execute_case(_settings("GF-ENV-LIVE-012"), tmp_path, 1, _window())
    assert code == 0
    assert RecordingController.instances == []
    result = _read_case(tmp_path, "GF-ENV-LIVE-012")
    assert result["verdict"] == "NOT_RUN"


def test_execute_011_is_not_run_without_an_absolute_shutdown(tmp_path: Path) -> None:
    # The live H200 reports only the T.Limit scale, so 011 has no runnable plan
    # and is NOT_RUN rather than armed.
    code = runner.execute_case(_settings("GF-ENV-LIVE-011"), tmp_path, 1, _window())
    assert code == 0
    result = _read_case(tmp_path, "GF-ENV-LIVE-011")
    assert result["verdict"] == "NOT_RUN" and result["not_run_variants"]
    assert RecordingController.instances == []


def test_execute_002_blocks_and_delegates_to_collect_002(tmp_path: Path) -> None:
    code = runner.execute_case(_settings("GF-ENV-LIVE-002"), tmp_path, 1, _window())
    assert code == 0
    result = _read_case(tmp_path, "GF-ENV-LIVE-002")
    assert result["verdict"] == "BLOCKED" and "COLLECT-002" in result["blocked_reason"]
    assert RecordingController.instances == []


def test_execute_010_is_not_run_without_a_control_node(tmp_path: Path) -> None:
    code = runner.execute_case(_settings("GF-ENV-LIVE-010"), tmp_path, 1, _window())
    assert code == 0
    result = _read_case(tmp_path, "GF-ENV-LIVE-010")
    assert result["verdict"] == "NOT_RUN" and "control task" in result["not_run_reason"]


def test_execute_004_without_branch_tokens_refuses(tmp_path: Path) -> None:
    settings = _settings("GF-ENV-LIVE-004", isolate_confirm="", stop_confirm="")
    with pytest.raises(RegionalFixtureError, match="--confirm-isolate"):
        runner.execute_case(settings, tmp_path, 1, _window())


def test_execute_preflight_errors_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "read_only_preflight",
        lambda s, d: _preflight(errors=["node not clean"]),
    )
    with pytest.raises(RegionalFixtureError, match="preflight failed"):
        runner.execute_case(_settings("GF-ENV-LIVE-004"), tmp_path, 1, _window())


def test_execute_001_runs_passthrough_without_arming(tmp_path: Path) -> None:
    runner.execute_case(_settings("GF-ENV-LIVE-001"), tmp_path, 1, _window())
    controller = RecordingController.instances[0]
    assert "arm" not in controller.calls, "001 observes pass-through only"
    assert controller.calls[:4] == ["create", "preflight", "deploy_proxy", "repoint"]
    assert controller.calls[-1] == "cleanup"
    result = _read_case(tmp_path, "GF-ENV-LIVE-001")
    assert result["proof_scope"] == "live-telemetry-replay"
    # No batch of the repointed collector reached the (scripted) store: the
    # preflight's collector statuses are no evidence of the pass-through.
    assert result["verdict"] == "INCONCLUSIVE", result["stages"]
    assert any("repointed collector" in item for item in result["inconclusive"]), (
        result["inconclusive"]
    )
    # Live 2026-10-24: a restore that left no override was judged "collector
    # endpoint was not restored" although the unit was back on collector.env.
    assert result["stages"]["cleanup"] == [], result["stages"]["cleanup"]


def test_execute_001_passes_on_the_repointed_collectors_batches(
    tmp_path: Path, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {
        "latest": [
            {"gpu": "GPU-aaaa", "name": name, "value": value}
            for name, value in baseline_values().items()
        ]
    }
    calls: list[str] = []

    def records(regional: Any, node: str, since: datetime, names: Any) -> Any:
        calls.append(since.isoformat())
        return before if world.repoint_at is None else world.gpu_state()

    monkeypatch.setattr(fixture, "gpu_records", records)
    world.accept(15, baseline_values(), reasons=("initial-baseline",))
    code = runner.execute_case(_settings("GF-ENV-LIVE-001"), tmp_path, 1, _window())
    result = _read_case(tmp_path, "GF-ENV-LIVE-001")
    assert code == 0 and result["verdict"] == "PASS", result["stages"]
    observation = result["receipts"]["observation"]
    assert observation["passthrough_reads"] >= 3
    assert observation["fresh_batch"]["edge_filter_reasons"] == ["initial-baseline"]
    assert observation["task"]["measured"] is False


@pytest.mark.parametrize(
    ("endpoint", "state"),
    [("http://127.0.0.1:9411/metrics", "active"), (None, "failed")],
)
def test_execute_001_flags_a_collector_that_did_not_come_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str | None, state: str
) -> None:
    monkeypatch.setattr(RecordingController, "restored_endpoint", endpoint)
    monkeypatch.setattr(RecordingController, "after_state", state)
    runner.execute_case(_settings("GF-ENV-LIVE-001"), tmp_path, 1, _window())
    result = _read_case(tmp_path, "GF-ENV-LIVE-001")
    assert result["verdict"] == "FAIL", result["verdict"]
    assert any(
        "collector endpoint was not restored" in item
        for item in result["stages"]["cleanup"]
    ), result["stages"]["cleanup"]


def test_execute_004_arms_and_restores_in_order(tmp_path: Path) -> None:
    runner.execute_case(_settings("GF-ENV-LIVE-004"), tmp_path, 1, _window())
    controller = RecordingController.instances[0]
    assert "arm" in controller.calls
    assert controller.calls.index("repoint") < controller.calls.index("arm")
    assert controller.calls.index("arm") < controller.calls.index("cleanup")
    # The armed plan is one of the reviewed plans and validates against the tool.
    import json as _json

    from tools.gpu_environment_replay import validate_plan

    validate_plan(_json.loads(controller.plan))


@pytest.mark.parametrize("number", range(1, 13))
def test_plan_details_builds_for_every_case(number: int) -> None:
    case_id = f"GF-ENV-LIVE-{number:03d}"
    details = runner.plan_details(_settings(case_id), _preflight())
    assert details["case_id"] == case_id
    assert details["proxy_mechanism"]["max_window_seconds"] == 600
    assert isinstance(details["replay_plans"], dict), details
    # Temperature / brake / duration / unknown / interruption cases arm a plan.
    if number in {3, 4, 5, 6, 7, 8, 9}:
        assert details["replay_plans"], case_id
    if number == 12:
        assert details["verdict_if_not_run"] == "NOT_RUN"


def test_replay_plans_cover_the_specific_families() -> None:
    pre = _preflight()
    assert set(runner.replay_plans_for_case("GF-ENV-LIVE-003", pre)["plans"]) >= {
        "core_tlimit/warning"
    }
    transient = runner.replay_plans_for_case("GF-ENV-LIVE-005", pre)["plans"]
    assert any(name.startswith("transient/") for name in transient), transient
    brake = runner.replay_plans_for_case("GF-ENV-LIVE-007", pre)["plans"]
    assert "transient" in brake and "sustained" in brake
    duration = runner.replay_plans_for_case("GF-ENV-LIVE-006", pre)["plans"]
    assert "sw_thermal" in duration and "hw_thermal" in duration
    unknown = runner.replay_plans_for_case("GF-ENV-LIVE-008", pre)["plans"]
    assert "omit_reason" in unknown
    nine = runner.replay_plans_for_case("GF-ENV-LIVE-009", pre)["plans"]
    assert "core_tlimit/severe" in nine


def test_replay_plans_report_no_target_uuid() -> None:
    pre = _preflight(target={"gpu_index": "0", "limits": {}})
    out = runner.replay_plans_for_case("GF-ENV-LIVE-004", pre)
    assert out["plans"] == {} and "no target GPU UUID" in out["error"]


def test_execute_004_without_a_product_reaction_is_not_a_verdict(
    tmp_path: Path, world: World
) -> None:
    # The live defect: one snapshot right after /arm. Now the runner waits the
    # derived window, the proxy serves the plan, and with no accepted record
    # the case is INCONCLUSIVE -- neither the old "no workflow" FAIL nor a PASS.
    code = runner.execute_case(_settings("GF-ENV-LIVE-004"), tmp_path, 1, _window())
    result = _read_case(tmp_path, "GF-ENV-LIVE-004")
    assert code == 1 and result["verdict"] == "INCONCLUSIVE", result["stages"]
    observation = result["receipts"]["observation"]
    assert observation["replay"]["consumed_at"] and observation["polls"] > 10
    assert observation["window"]["terms"]["cooldown_hold"] == 315
    calls = RecordingController.instances[0].calls
    assert calls.index("arm") < calls.index("cleanup") == len(calls) - 1
    assert calls.count("status") > 3, "the proxy was polled until consumed"
    assert world.sleeps and all(item <= 15 for item in world.sleeps)


def test_execute_004_waits_for_the_review_chain_and_passes(
    tmp_path: Path, world: World
) -> None:
    review_chain(world)
    code = runner.execute_case(_settings("GF-ENV-LIVE-004"), tmp_path, 1, _window())
    result = _read_case(tmp_path, "GF-ENV-LIVE-004")
    assert code == 0 and result["verdict"] == "PASS", (
        result["stages"],
        result["inconclusive"],
    )
    observation = result["receipts"]["observation"]
    assert observation["review_workflow"] == "wf-review"
    assert observation["hold"]["held"] is True
    assert (
        tmp_path / "cases" / "GF-ENV-LIVE-004" / "observation-final.json"
    ).is_file(), "the final poll's records are kept as evidence"
    assert RecordingController.instances[0].calls[-1] == "cleanup"


def test_execute_004_cleanup_failure_fails_even_a_good_chain(
    tmp_path: Path, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_chain(world)
    monkeypatch.setattr(RecordingController, "after_state", "failed")
    runner.execute_case(_settings("GF-ENV-LIVE-004"), tmp_path, 1, _window())
    result = _read_case(tmp_path, "GF-ENV-LIVE-004")
    assert result["verdict"] == "FAIL" and result["stages"]["cleanup"]


def test_execute_009_is_blocked_and_never_arms(tmp_path: Path) -> None:
    code = runner.execute_case(_settings("GF-ENV-LIVE-009"), tmp_path, 1, _window())
    result = _read_case(tmp_path, "GF-ENV-LIVE-009")
    assert code == 0 and result["verdict"] == "BLOCKED"
    assert ">60 s pause" in result["blocked_reason"]
    assert RecordingController.instances == []
    details = runner.plan_details(_settings("GF-ENV-LIVE-009"), _preflight())
    assert details["verdict_if_blocked"] == "BLOCKED"


def test_plan_details_names_the_observation_rule() -> None:
    details = runner.plan_details(_settings("GF-ENV-LIVE-004"), _preflight())
    observation = details["observation"]
    assert (
        observation["kind"] == "review" and observation["plan"] == "core_tlimit/severe"
    )
    assert observation["derived_seconds"] == sum(observation["terms"].values())
    assert (
        runner.plan_details(_settings("GF-ENV-LIVE-012"), _preflight())["observation"]
        is None
    )


def _read_case(run_dir: Path, case_id: str) -> dict[str, Any]:
    import json

    return json.loads((run_dir / "cases" / case_id / f"{case_id}.json").read_text())
