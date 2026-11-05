"""GF-GENERAL-LIVE unknown / missing-evidence XIDs: verdict and preflight edges."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_unknown as mod
from scripts.e2e.regional.general_live_fixture import VariantBlocked
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)
from tests.regional._general_live_gpu_fakes import survey as gpu_survey

NODE = "ip-10-0-43-5.us-west-2.compute.internal"


def topology_decision(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "disposition": mod.BLOCKED_MISSING_EVIDENCE,
        "reasons": [f"blocked: {mod.TOPOLOGY_REASON}"],
        "requires_operator": True,
        "action": None,
    }
    value.update(overrides)
    return value


def test_a_fatal_sxid_without_topology_must_not_earn_an_action() -> None:
    assert mod.missing_topology_errors([topology_decision()]) == []
    assert mod.missing_topology_errors(
        [topology_decision(action="RESET_ALL_GPUS_NVSWITCHES")]
    ) == [
        "a fatal SXID without topology evidence earned action RESET_ALL_GPUS_NVSWITCHES"
    ]


def test_an_xid_absent_from_the_pinned_catalog_has_no_version_gate() -> None:
    assert mod.version_gate_errors("NVIDIA B200", xid=9999) == [
        "XID 9999 is absent from the pinned catalog"
    ]


def test_the_reviewed_line_needs_the_gpus_pci_bdf() -> None:
    line = mod.missing_version_line("0000:1b:00", "run-7")
    assert f"Xid (PCI:0000:1b:00): {mod.MISSING_VERSION_XID}" in line

    with pytest.raises(ValueError, match="a kernel XID line needs the GPU's PCI BDF"):
        mod.missing_version_line("unknown", "run-7")


def version_decision(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "decision": True,
        "duplicate": False,
        "intr_info": "0x1",
        "error_status": "0x2",
        "disposition": mod.BLOCKED_MISSING_EVIDENCE,
        "reasons": [f"blocked: {mod.VERSION_REASON}"],
        "official_action": mod.NVLINK5_WORKFLOW,
        "requires_operator": True,
        "action": None,
    }
    value.update(overrides)
    return value


def test_a_withheld_decision_must_keep_the_nvlink5_official_action() -> None:
    assert mod.missing_version_errors([version_decision()]) == []
    assert mod.missing_version_errors(
        [version_decision(official_action="RESET_GPU")]
    ) == [f"the withheld official action is RESET_GPU, expected {mod.NVLINK5_WORKFLOW}"]


def version_context(tmp_path: Path, survey: dict[str, Any]) -> tuple[Any, FakeProbe]:
    spec = next(item for item in mod.VARIANTS if item.variant == "missing_version")
    window = FakeProbe({})
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(),
        probes={
            f"node@{NODE}": FakeProbe({"survey": {"residue": {}}}),
            f"dcgm@{NODE}": FakeProbe({"survey": survey}),
            f"kernel-collector-replay@{NODE}": window,
        },
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    return ctx, window


def test_missing_version_is_blocked_when_the_survey_lists_no_gpu(
    tmp_path: Path,
) -> None:
    survey = gpu_survey(gpus=(), model="NVIDIA B200")
    ctx, window = version_context(tmp_path, survey)

    with pytest.raises(VariantBlocked, match="the DCGM survey lists no GPU"):
        mod.missing_version(ctx)

    assert window.calls == [], "nothing was installed or opened"


def test_missing_version_is_blocked_on_a_gpu_without_a_pci_bdf(tmp_path: Path) -> None:
    survey = gpu_survey(
        gpus=("GPU-00000000-0000-4000-8000-000000000000",), model="NVIDIA B200"
    )
    for entry in survey["exporter"]["gpus"].values():
        entry["pci_bdf"] = ""
    ctx, window = version_context(tmp_path, survey)

    with pytest.raises(VariantBlocked, match="has no PCI BDF") as caught:
        mod.missing_version(ctx)

    assert caught.value.reason == "FIXTURE_NOT_READY"
    assert window.calls == [], "nothing was installed or opened"
