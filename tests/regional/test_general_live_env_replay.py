"""The GF-ENV-LIVE telemetry replay inside a GF-GENERAL-LIVE variant, over fakes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_env_replay as er
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_gpu_composite import VARIANTS
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
G1 = "GPU-11111111-1111-4111-8111-111111111111"
REAL = "http://127.0.0.1:9400/metrics"
PROXY = "http://127.0.0.1:9411/metrics"
PLAN = {
    "steps": [
        {"reads": 3, "values": {}},
        {"reads": 2, "values": {"gpu_tlimit_c": 0.5, "clock_throttle_reasons": 0x20}},
    ]
}


def _script(
    *, armed_at_deploy: bool = False, reads_after: int = 9, aborted: str | None = None
) -> dict[str, Any]:
    return {
        "preflight": {"boot_id": "b"},
        "install-source": {"ok": True},
        "deploy-proxy": {"status": {"armed": False}, "endpoint": PROXY},
        "repoint-collector": {
            "original_endpoint": REAL,
            "proxy_endpoint": PROXY,
            "after": {"ActiveState": "active"},
        },
        "arm": {"armed": True, "status": {"armed": True, "reads": 3}},
        "status": [
            {"status": {"armed": armed_at_deploy, "reads": 2, "aborted": None}},
            {"status": {"armed": True, "reads": reads_after, "aborted": aborted}},
        ],
        "restore-collector": {
            "restored_endpoint": REAL,
            "original_endpoint": REAL,
            "after": {"ActiveState": "active"},
            "dropin_removed": True,
        },
        "stop-proxy": {"audit_was_present": True},
    }


def _ctx(tmp_path: Path, fake: FakeProbe) -> Any:
    spec = VARIANTS[0]
    regional = FakeRegional(cpu_python=lambda *_: {})
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"{er.PROBE_KEY}@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
    )


def test_deploy_arm_wait_close_keep_one_producer_and_record_receipts(
    tmp_path: Path,
) -> None:
    fake = FakeProbe(_script())
    ctx = _ctx(tmp_path, fake)
    replay = er.env_replay(ctx, PLAN, G1, 600)
    replay.deploy()
    verbs = [call[0] for call in fake.calls]
    assert verbs.count("install-source") == 2, "the plan and the audited tool"
    assert (
        verbs.index("deploy-proxy")
        < verbs.index("repoint-collector")
        < verbs.index("status")
    )
    assert er.CLEANUP_NAME in ctx.cleanup.pending()
    assert (ctx.case_dir / "plan.json").is_file(), "the reviewed plan is shipped"
    deploy_call = next(call for call in fake.calls if call[0] == "deploy-proxy")
    assert deploy_call[deploy_call.index("--max-seconds") + 1] == "600", (
        "the proxy window is capped at the probe's limit, whatever the hold"
    )
    repoint_call = next(call for call in fake.calls if call[0] == "repoint-collector")
    assert repoint_call[repoint_call.index("--restore-seconds") + 1] == "1200"
    assert replay.remaining_seconds() == 600 - er.WINDOW_CLOSE_MARGIN_SECONDS
    replay.arm()
    assert replay.reads_at_arm == 3
    assert replay.armed_at is not None
    assert replay.remaining_seconds() <= 600 - er.WINDOW_CLOSE_MARGIN_SECONDS
    status = replay.wait_served(5, timeout_seconds=60, name="served")
    assert replay.served_reads(status) == 6
    receipts = replay.close()
    verbs = [call[0] for call in fake.calls]
    assert verbs.index("restore-collector") < verbs.index("stop-proxy"), (
        "the endpoint goes back to the real exporter before the proxy stops"
    )
    assert receipts["restore"]["restored_endpoint"] == REAL
    assert replay.close() == receipts, "closing twice is a no-op"
    cleaned = replay.cleanup()
    assert cleaned["probe_residuals"] == {"pod": False, "configmap": False}
    evidence = replay.evidence()
    assert evidence["proof_scope"] == "live-replay"
    assert evidence["plan_reads"] == 5
    assert evidence["plan_fields"] == ["clock_throttle_reasons", "gpu_tlimit_c"]
    assert len(evidence["tool_sha256"]) == 64
    assert (ctx.case_dir / "env-replay-deploy.json").is_file(), "deploy receipts"
    assert (ctx.case_dir / "env-replay-close.json").is_file(), "close receipts"
    assert (ctx.case_dir / "oracle.jsonl").is_file(), "status readings are the oracle"


def test_deploy_is_inconclusive_when_the_proxy_came_up_armed(tmp_path: Path) -> None:
    replay = er.env_replay(
        _ctx(tmp_path, FakeProbe(_script(armed_at_deploy=True))), PLAN, G1, 600
    )
    with pytest.raises(VariantInconclusive, match="armed before the baseline"):
        replay.deploy()


def test_wait_served_is_inconclusive_on_abort_or_stalled_reads(tmp_path: Path) -> None:
    replay = er.env_replay(
        _ctx(
            tmp_path,
            FakeProbe(_script(aborted="live safety stop: DCGM_FI_DEV_GPU_TEMP")),
        ),
        PLAN,
        G1,
        600,
    )
    replay.deploy()
    replay.arm()
    with pytest.raises(VariantInconclusive, match="aborted"):
        replay.wait_served(1, timeout_seconds=30, name="served")

    stalled = er.env_replay(
        _ctx(tmp_path / "b", FakeProbe(_script(reads_after=3))), PLAN, G1, 600
    )
    stalled.deploy()
    stalled.arm()
    with pytest.raises(VariantInconclusive, match="consumed 0 of the 5"):
        stalled.wait_served(5, timeout_seconds=5, name="served")


def test_close_is_inconclusive_when_the_endpoint_is_not_restored(
    tmp_path: Path,
) -> None:
    script = _script()
    script["restore-collector"] = {
        "restored_endpoint": PROXY,
        "original_endpoint": REAL,
        "after": {"ActiveState": "active"},
    }
    replay = er.env_replay(_ctx(tmp_path, FakeProbe(script)), PLAN, G1, 600)
    replay.deploy()
    with pytest.raises(VariantInconclusive, match="not restored"):
        replay.close()


def test_cleanup_raises_on_probe_residue(tmp_path: Path) -> None:
    fake = FakeProbe(_script())
    fake.cleanup = lambda: {"pod": True, "configmap": False}  # type: ignore[method-assign]
    replay = er.env_replay(_ctx(tmp_path, fake), PLAN, G1, 600)
    replay.deploy()
    with pytest.raises(RuntimeError, match="residue"):
        replay.cleanup()


def test_checked_plan_refuses_fields_outside_the_environment_allow_list(
    tmp_path: Path,
) -> None:
    bad = {"steps": [{"reads": 1, "values": {"ecc_sbe_volatile_total": 1}}]}
    with pytest.raises(RegionalFixtureError, match="rejected"):
        er.checked_plan(bad)
    with pytest.raises(RegionalFixtureError):
        er.env_replay(_ctx(tmp_path, FakeProbe(_script())), bad, G1, 600)
    with pytest.raises(RegionalFixtureError, match="GPU UUID"):
        er.env_replay(
            _ctx(tmp_path / "b", FakeProbe(_script())), PLAN, "not-a-gpu", 600
        )
    assert er.plan_reads(PLAN) == 5
    assert er.plan_fields(PLAN) == {"gpu_tlimit_c", "clock_throttle_reasons"}
    assert er.plan_sha256(PLAN) == er.plan_sha256(dict(PLAN))
