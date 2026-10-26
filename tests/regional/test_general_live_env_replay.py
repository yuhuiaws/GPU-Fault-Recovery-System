"""The GF-ENV-LIVE telemetry replay inside a GF-GENERAL-LIVE variant, over fakes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_env_replay as er
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_gpu_composite import VARIANTS
from scripts.e2e.regional.host_probe_fixture import HostProbeError
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
ENDPOINT = "GPU_FAULT_DCGM_METRICS_URL"
PLAN = {
    "steps": [
        {"reads": 3, "values": {}},
        {"reads": 2, "values": {"gpu_tlimit_c": 0.5, "clock_throttle_reasons": 0x20}},
    ]
}
#: The probe's proof that the restarted collector reads the proxy: its process
#: environment names the proxy and the proxy counted its first pass-through read.
FIRST_READ = {
    "observed": True,
    "reads_before": 0,
    "reads_after": 1,
    "waited_seconds": 18.4,
    "budget_seconds": 65,
}


def _repointed(**overrides: Any) -> dict[str, Any]:
    receipt = {
        "repointed": True,
        "original_endpoint": REAL,
        "proxy_endpoint": PROXY,
        "process_environment": {ENDPOINT: PROXY},
        "first_read": dict(FIRST_READ),
        "after": {"ActiveState": "active"},
    }
    receipt.update(overrides)
    return receipt


def _rolled_back(reason: str, **overrides: Any) -> dict[str, Any]:
    """The probe's answer to a repoint it judged ineffective and undid."""

    receipt = {
        "repointed": False,
        "reason": reason,
        "original_endpoint": REAL,
        "proxy_endpoint": PROXY,
        "process_environment": {ENDPOINT: REAL},
        "rollback": {"restarted": True, "restored_endpoint": REAL},
        "after": {"ActiveState": "active"},
    }
    receipt.update(overrides)
    return receipt


def _script(
    *, armed_at_deploy: bool = False, reads_after: int = 9, aborted: str | None = None
) -> dict[str, Any]:
    return {
        "preflight": {"boot_id": "b"},
        "install-source": {"ok": True},
        "deploy-proxy": {"status": {"armed": False}, "endpoint": PROXY},
        "repoint-collector": _repointed(),
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
            "restarted": True,
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
    assert evidence["collector_process_endpoint"] == PROXY
    assert evidence["collector_first_read"] == FIRST_READ
    assert (ctx.case_dir / "env-replay-deploy.json").is_file(), "deploy receipts"
    assert (ctx.case_dir / "env-replay-close.json").is_file(), "close receipts"
    assert (ctx.case_dir / "oracle.jsonl").is_file(), "status readings are the oracle"


def test_env_replay_keeps_its_call_sequence(tmp_path: Path) -> None:
    # Pinned deliberately: the proof that the restarted collector reads the
    # proxy rides in the repoint-collector receipt (the probe waits for the
    # first pass-through read on the node), so proving the repoint costs no
    # extra probe call before arming -- one pass-through status, arm, one
    # served status, then the restore before the proxy stops.
    fake = FakeProbe(_script())
    replay = er.env_replay(_ctx(tmp_path, fake), PLAN, G1, 600)
    replay.deploy()
    replay.arm()
    replay.wait_served(5, timeout_seconds=60, name="served")
    replay.close()
    assert [call[0] for call in fake.calls] == [
        "preflight",
        "install-source",
        "install-source",
        "deploy-proxy",
        "repoint-collector",
        "status",
        "arm",
        "status",
        "restore-collector",
        "stop-proxy",
    ]
    repoint = next(call for call in fake.calls if call[0] == "repoint-collector")
    assert fake.timeouts[fake.calls.index(repoint)] == ("repoint-collector", 180), (
        "the exec timeout covers the probe's settle and first-read waits"
    )


def test_deploy_is_a_fixture_error_when_the_repoint_did_not_take_effect(
    tmp_path: Path,
) -> None:
    # The live 2026-10-26 shape, now answered by the probe itself: the unit
    # restarted, the process kept collector.env's endpoint, the probe rolled
    # the drop-in back. Nothing is armed and no replay budget is waited out.
    script = _script()
    script["repoint-collector"] = _rolled_back(
        "the restarted collector process does not read the proxy endpoint; "
        "the override never reached it"
    )
    fake = FakeProbe(script)
    replay = er.env_replay(_ctx(tmp_path, fake), PLAN, G1, 600)
    with pytest.raises(
        RegionalFixtureError, match="did not take effect.*never reached it"
    ):
        replay.deploy()
    verbs = [call[0] for call in fake.calls]
    assert "arm" not in verbs and "status" not in verbs
    assert replay.repoint_attempted and not replay.deployed and not replay.armed
    receipts = replay.close()
    verbs = [call[0] for call in fake.calls]
    assert verbs[-2:] == ["restore-collector", "stop-proxy"], (
        "symmetric: the unit is restored (idempotently) before the proxy stops"
    )
    assert receipts["restore"]["restored_endpoint"] == REAL
    assert replay.cleanup()["probe_residuals"] == {"pod": False, "configmap": False}


@pytest.mark.parametrize(
    ("receipt", "fragment"),
    [
        ({"original_endpoint": REAL, "proxy_endpoint": PROXY}, "no first-read proof"),
        (_repointed(first_read=None), "no first-read proof"),
        (
            _repointed(process_environment={ENDPOINT: REAL}),
            "reads 'http://127.0.0.1:9400/metrics', not the proxy",
        ),
        (
            _repointed(first_read={**FIRST_READ, "observed": False, "reads_after": 0}),
            "no pass-through read of the proxy",
        ),
    ],
)
def test_deploy_refuses_to_arm_without_the_full_repoint_proof(
    tmp_path: Path, receipt: dict[str, Any], fragment: str
) -> None:
    script = _script()
    script["repoint-collector"] = receipt
    fake = FakeProbe(script)
    replay = er.env_replay(_ctx(tmp_path, fake), PLAN, G1, 600)
    with pytest.raises(RegionalFixtureError, match=fragment):
        replay.deploy()
    assert "arm" not in [call[0] for call in fake.calls]
    with pytest.raises(RegionalFixtureError, match=fragment):
        er.repoint_proof(receipt, PROXY)
    assert er.repoint_proof(_repointed(), PROXY) == {
        "process_endpoint": PROXY,
        "first_read": FIRST_READ,
    }


def test_close_restores_after_a_repoint_call_that_failed_midway(tmp_path: Path) -> None:
    script = _script()

    def failing(arguments: tuple[str, ...]) -> dict[str, Any]:
        raise HostProbeError(
            "host probe failed (exit 1; error ProbeError 0123456789abcdef at "
            "repoint_collector:700); output withheld"
        )

    script["repoint-collector"] = failing
    fake = FakeProbe(script)
    replay = er.env_replay(_ctx(tmp_path, fake), PLAN, G1, 600)
    with pytest.raises(HostProbeError):
        replay.deploy()
    assert replay.repoint_attempted, "the call may have changed the node"
    replay.close()
    verbs = [call[0] for call in fake.calls]
    assert verbs.index("restore-collector") < verbs.index("stop-proxy")


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
