"""GF-GENERAL-LIVE-010 ``link_inactive``: the bounded exposure follows the
collector's observed ticks (``observed_exposure``), the fallback deadman never
decides; split from ``test_general_live_efa`` (file size), on its fakes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_efa_fixture as fx
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from tests.regional._general_live_fakes import incident, workflow
from tests.regional.test_general_live_efa import (
    FALLBACK,
    NODE,
    PORT_DEVICE,
    RESTORE_AT,
    SPECS,
    STATE_DEST,
    batch,
    calls,
    diagnostic,
    doc,
    future,
    installed,
    inventory,
    run,
    sample,
    source_probe,
    ticks_receipt,
    window_flags,
)


def _link_doc(
    *,
    validation: str = "FAILED",
    failure_mode: str = "LINK_INACTIVE",
    judged_after_restore: bool = False,
    late_inactive: bool = False,
    stale_gate: bool = False,
) -> dict[str, Any]:
    """The held tick (before the restore), the link chain judged on it, the
    LINK_INACTIVE co-finding concluded on the restored port, fresh zeros."""

    inc, wf = diagnostic(
        fx.LINK_METRIC,
        PORT_DEVICE,
        validation=validation,
        details={"node_failures": {NODE: ["rdma_link_down"]}},
        reasons=(
            "RDMA/EFA/InfiniBand port is not active",
            "diagnostic inconclusive: validation failed",
        ),
        workflow_status="FAILED" if validation == "FAILED" else "SUCCEEDED",
    )
    if judged_after_restore:
        wf["step_executions"][0]["updated_at"] = future(RESTORE_AT + 20)
    inv_inc = incident(
        "inc-inv", fx.INVENTORY_METRIC, state="RECOVERED", workflow="wf-inv"
    )
    inv_wf = workflow(
        "wf-inv",
        "inc-inv",
        ["FREEZE_EVIDENCE", "VALIDATE_FABRIC"],
        status="SUCCEEDED",
        executions=[
            {
                "operation": "VALIDATE_FABRIC",
                "status": "SUCCEEDED",
                "details": {
                    "node_pending": {
                        NODE: {"missing_inventory_action_completion": ["x"]}
                    }
                }
                if stale_gate
                else {"inventory_evidence": {NODE: {}}},
            }
        ],
    )
    batches = [
        batch(
            [
                sample(fx.LINK_METRIC, 1, PORT_DEVICE),
                sample(fx.INVENTORY_METRIC, 1, None, {"failure_mode": failure_mode}),
                sample(fx.MISSING_METRIC, 1, None, {"failure_mode": failure_mode}),
            ],
            observed_at=future(10),
        ),
        batch([sample(fx.LINK_METRIC, 0, PORT_DEVICE)], observed_at=future(120)),
        batch([sample(fx.LINK_METRIC, 0, PORT_DEVICE)], observed_at=future(150)),
    ]
    if late_inactive:
        batches.append(
            batch(
                [sample(fx.MISSING_METRIC, 1, None)],
                observed_at=future(RESTORE_AT + 30),
            )
        )
    return doc(incidents=[inc, inv_inc], workflows=[wf, inv_wf], batches=batches)


def test_link_inactive_binds_the_state_file_for_a_bounded_exposure(
    tmp_path: Path,
) -> None:
    source = source_probe(inventory(), read_back="4: ACTIVE")
    result, source = run("link_inactive", tmp_path, records=_link_doc(), source=source)
    assert result["verdict"] == "PASS", result["errors"]
    assert installed(source, "efa-state") == [b"1: DOWN\n"]
    flags = window_flags(source)
    assert flags["--bind"] == [f"efa-state:{STATE_DEST}"]
    assert flags["--restore-seconds"] == [str(FALLBACK)], (
        "the deadman is the fallback: startup phase + jitter + 2 ticks + margins"
    )
    waited = calls(source, "await-ticks")[0]
    assert waited[:8] == (
        "await-ticks",
        "--count",
        "2",
        "--interval",
        "15",
        "--timeout",
        str(FALLBACK),
        "--close",
    ), "the probe closes right after the collector's second tick, seen on the node"
    assert not calls(source, "close-window"), "await-ticks closed; nothing closes twice"
    assert result["closed_by"] == "probe" and result["ticks_observed"] == 2
    assert result["ticks_required"] == 2 and isinstance(
        result["exposure_seconds"], float
    )
    assert result["stages"]["recovery"] == [] and result["stages"]["settled"] == []
    assert result["stages"]["co_finding_diagnosed"] == [], result["stages"]
    spec = SPECS["link_inactive"]
    assert (
        spec.approvals == ("HOST_SOURCE_REPLAY",)
        and "bounded exposure" in spec.injection
    )


def test_link_inactive_fails_on_the_held_port_judged_clean_or_a_wrong_mode(
    tmp_path: Path,
) -> None:
    result, source = run(
        "link_inactive",
        tmp_path,
        records=_link_doc(validation="SUCCEEDED"),
        source=source_probe(inventory(), read_back="4: ACTIVE"),
    )
    assert result["verdict"] == "FAIL"
    assert result["closed_by"] == "probe"
    assert any(
        "succeeded while the fault was still held" in e for e in result["errors"]
    ), result["errors"]
    result, _source = run(
        "link_inactive",
        tmp_path,
        records=_link_doc(failure_mode="PCI_DEVICE_MISSING"),
        source=source_probe(inventory(), read_back="4: ACTIVE"),
    )
    assert result["verdict"] == "FAIL"
    assert any("inventory_grading" in e for e in result["errors"]), result["errors"]
    result, _source = run(
        "link_inactive",
        tmp_path,
        records=_link_doc(stale_gate=True),
        source=source_probe(inventory(), read_back="4: ACTIVE"),
    )
    assert result["verdict"] == "FAIL"
    assert any("missing_inventory_action_completion" in e for e in result["errors"]), (
        result["errors"]
    )


def test_link_inactive_is_inconclusive_when_the_bound_decided_nothing(
    tmp_path: Path,
) -> None:
    with pytest.raises(VariantInconclusive, match="after the bounded exposure ended"):
        run(
            "link_inactive",
            tmp_path,
            records=_link_doc(validation="SUCCEEDED", judged_after_restore=True),
            source=source_probe(inventory(), read_back="4: ACTIVE"),
        )
    with pytest.raises(VariantInconclusive, match="exposure bound did not hold"):
        run(
            "link_inactive",
            tmp_path,
            records=_link_doc(late_inactive=True),
            source=source_probe(inventory(), read_back="4: ACTIVE"),
        )
    with pytest.raises(VariantInconclusive, match="no sample") as caught:
        run(
            "link_inactive",
            tmp_path,
            records=doc(),
            source=source_probe(inventory(), read_back="4: ACTIVE"),
        )
    assert "closed by probe" in str(caught.value), str(caught.value)
    assert "tick 1 ended " in str(caught.value) and (
        "no batch of it reached the product -- not a collector read, or its batch "
        "was never delivered"
    ) in str(caught.value), "the missing reading is named (live eks 2026-10-26)"
    assert "2 of the 2 confirming rdma_link_down readings never happened" in str(
        caught.value
    ), str(caught.value)
    # The fallback deadman got there first: the record says so, the verdict stands.
    result, _source = run(
        "link_inactive",
        tmp_path,
        records=_link_doc(),
        source=source_probe(
            inventory(),
            read_back="4: ACTIVE",
            **{
                "await-ticks": ticks_receipt(already_closed=True, observed=0),
                "open-window": {
                    "after": {"ActiveState": "active"},
                    "proof_scope": "live-replay",
                    "verification": {"ok": True},
                    "opened_at": future(RESTORE_AT - FALLBACK),
                    "restore_seconds": FALLBACK,
                },
            },
        ),
    )
    assert result["closed_by"] == "deadman" and result["ticks_observed"] == 0
    assert (
        result["restore_at"] == future(RESTORE_AT)[:16] + result["restore_at"][16:]
    ), "the deadman's schedule, read off the open receipt, times the restore"
