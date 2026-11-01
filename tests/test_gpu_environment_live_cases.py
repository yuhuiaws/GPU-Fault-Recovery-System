"""Catalog/procedure traceability is not evidence of live execution."""

from pathlib import Path

from tools.run_fault_test_cases import load_catalog, select_cases
from tools.scenario_requirements import load_requirements

ROOT = Path(__file__).resolve().parents[1]


def test_live_environment_cases_are_registered_but_not_claimed_as_pass():
    catalog = load_catalog(ROOT / "testcases/fault-scenarios.yaml")
    cases = [case for case in catalog if case["category"] == "gpu-environment-live"]
    assert [case["id"] for case in cases] == [
        f"GF-ENV-LIVE-{number:03}" for number in range(1, 13)
    ]
    procedure = (
        ROOT / "docs/components/gpu-environment-live-acceptance.md"
    ).read_text()
    for case in cases:
        assert case["automation"] == "manual" and case["evidence"] == {
            "verdict": "NOT_RUN"
        }, case
        assert len(case["expected"]) >= 3, case
        assert f"### {case['id']}\n" in procedure, case
        assert case["procedure"].endswith("#" + case["id"].lower()), case
    assert (
        select_cases(
            catalog,
            case_ids=[],
            categories={"gpu-environment-live"},
            levels=set(),
            include_manual=False,
            include_live=False,
        )
        == []
    ), "ordinary runner invocation must not execute live acceptance"
    by_id = {case["id"]: case for case in catalog}
    requirements = {
        requirement.id: requirement
        for requirement in load_requirements(
            ROOT / "testcases/scenario-requirements.yaml", root=ROOT, catalog=by_id
        )
    }
    assert requirements["GF-REQ-ENV-LIVE"].gap, "unexecuted live work keeps its gap"
    assert requirements["GF-REQ-ENV-LIVE"].cases == [case["id"] for case in cases]
    assert [check.expected for check in requirements["GF-REQ-ENV-SOFTWARE"].checks] == [
        4,
        1,
    ]


def test_live_mutation_cases_are_not_labeled_as_read_only():
    cases = {
        case["id"]: case
        for case in load_catalog(ROOT / "testcases/fault-scenarios.yaml")
    }
    for number in range(2, 13):
        assert cases[f"GF-ENV-LIVE-{number:03}"]["risk"] in {
            "live-node-mutation",
            "live-workload-restart",
            "live-service-action",
            "destructive",
        }, number


def test_case_011_runs_on_any_device_source_of_the_shutdown_limit():
    """Batch 232: the precondition is the product's published absolute limit.

    A T.Limit GPU (H200, driver 580) carries it from DCGM field 159, so the case
    reads it from the product and cites NVIDIA for that source; the offset of
    the T.Limit scale still never substitutes and nothing is physically heated.
    """

    procedure = " ".join(
        (ROOT / "docs/components/gpu-environment-live-acceptance.md")
        .read_text(encoding="utf-8")
        .split()
    )
    section = procedure.split("### GF-ENV-LIVE-011", 1)[1].split("###", 1)[0]
    for required in (
        "任一设备来源",
        "DCGM_FI_DEV_SHUTDOWN_TEMP",
        "DCGM_FI_DEV_GPU_TEMP",
        "dcgm-api-field-ids.html#c.DCGM_FI_DEV_GPU_TEMP_SHUTDOWN_CELSIUS",
        "Shutdown temperature for the device.",
        "group__nvmlDeviceQueries.html",
        "is no longer the preferred interface",
        "不能替代本例",
        "绝对禁止物理加热",
    ):
        assert required in section, f"GF-ENV-LIVE-011 lacks {required!r}"
