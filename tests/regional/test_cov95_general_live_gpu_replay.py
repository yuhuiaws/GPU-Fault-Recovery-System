"""DCGM replay fixture: survey questions, replay preconditions and the oracle gate."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_replay as replay_module
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.general_live_gpu_replay import (
    DcgmReplay,
    GpuSurvey,
    add,
    arm_before_repoint,
    leading_passthrough_reads,
    plan,
    replay_boot_id,
    require_replayable,
    residue_lines,
    step,
)

GPU = "GPU-11111111-1111-4111-8111-111111111111"
FIELD = "ecc_sbe_volatile_total"
KEY = f"{GPU}/{FIELD}"
OTHER_KEY = f"{GPU}/ecc_dbe_volatile_total"


def survey(**overrides: Any) -> GpuSurvey:
    raw: dict[str, Any] = {
        "venv_python": "/opt/gpu-fault/venv/bin/python",
        "tools": {"systemd-run": "/usr/bin/systemd-run", "systemctl": "/bin/systemctl"},
        "metrics_unit": {"ActiveState": "active"},
        "collector_env": {"GPU_FAULT_METRICS_MODE": "dcgm"},
        "exporter": {"gpus": {GPU: {"gpu_index": "0", "values": {FIELD: 0.0}}}},
        "boot_id": "boot-current",
    }
    raw.update(overrides)
    return GpuSurvey(raw)


def test_survey_settings_fall_back_to_the_default_for_malformed_values() -> None:
    assert survey().sustain == 180
    configured = survey(
        collector_env={"GPU_FAULT_DCGM_COLLECTION_ERROR_SUSTAIN_SECONDS": "240"}
    )
    assert configured.sustain == 240
    malformed = survey(
        collector_env={"GPU_FAULT_DCGM_COLLECTION_ERROR_SUSTAIN_SECONDS": "3 min"}
    )
    assert malformed.sustain == 180
    assert (
        survey(collector_env={"GPU_FAULT_METRICS_INTERVAL_SECONDS": [30]}).interval
        == 15
    )


def test_target_is_not_run_when_no_gpu_reports_the_field() -> None:
    assert survey().target(FIELD) == GPU
    with pytest.raises(
        VariantNotRun, match="no GPU on the node reports pcie_replay"
    ) as caught:
        survey().target("pcie_replay")
    assert caught.value.reason == "CAPABILITY_ABSENT"
    with pytest.raises(VariantNotRun, match=f"reports {FIELD}"):
        survey().target(FIELD, avoid=[GPU])


@pytest.mark.parametrize(
    ("overrides", "reason", "fragment"),
    [
        ({"venv_python": ""}, "CAPABILITY_ABSENT", "the node has no gpu_fault venv"),
        (
            {"tools": {"systemctl": "/bin/systemctl"}},
            "CAPABILITY_ABSENT",
            "the node lacks systemd-run",
        ),
        ({"tools": None}, "CAPABILITY_ABSENT", "the node lacks systemd-run, systemctl"),
        (
            {"metrics_unit": {"ActiveState": "failed"}},
            "FIXTURE_NOT_READY",
            "gpu-fault-metrics-collector.service is not active",
        ),
        (
            {"collector_env": {"GPU_FAULT_METRICS_MODE": "nvml"}},
            "CAPABILITY_ABSENT",
            "the metrics collector runs in nvml mode, not dcgm",
        ),
        (
            {"exporter": {"error": "connection refused", "gpus": {GPU: {}}}},
            "FIXTURE_NOT_READY",
            "the real DCGM exporter scrape is not usable: connection refused",
        ),
        (
            {"exporter": {"gpus": {}}},
            "FIXTURE_NOT_READY",
            "the real DCGM exporter scrape is not usable: no GPU",
        ),
    ],
)
def test_require_replayable_names_the_missing_precondition(
    overrides: dict[str, Any], reason: str, fragment: str
) -> None:
    require_replayable(survey())

    with pytest.raises(VariantNotRun) as caught:
        require_replayable(survey(**overrides))

    assert caught.value.reason == reason
    assert caught.value.detail == fragment


def test_residue_lines_skip_a_run_without_objects() -> None:
    lines = residue_lines(
        {
            "runs": {
                "digest-a": {"run_id": "run-a", "objects": []},
                "digest-b": None,
                "digest-c": {"objects": ["unit gfrs-replay-c.service"]},
            }
        }
    )

    assert lines == ["earlier run digest-c: unit gfrs-replay-c.service"]


def test_replay_boot_id_without_an_earlier_boot_is_a_deterministic_replay_id() -> None:
    digest = hashlib.sha256(b"boot/run-7").hexdigest()
    expected = (
        str(uuid.UUID(digest[:32])),
        "replay id (no earlier boot in the journal)",
    )

    assert replay_boot_id(survey(), "run-7") == expected
    # The current boot and empty journal entries are not an earlier boot.
    assert (
        replay_boot_id(survey(prior_boot_ids=["", "boot-current"]), "run-7") == expected
    )
    assert replay_boot_id(survey(prior_boot_ids=["boot-current", "boot-old"]), "x") == (
        "boot-old",
        "journal --list-boots",
    )


def test_leading_passthrough_reads_of_a_plan_without_operations_counts_every_step() -> (
    None
):
    passthrough = plan("memory", step(3), step(2))
    assert leading_passthrough_reads(passthrough) == 5
    assert leading_passthrough_reads({}) == 0
    assert (
        leading_passthrough_reads(plan("memory", step(3), step(4, add(FIELD, [GPU]))))
        == 3
    )


class FakeEvidence:
    def __init__(self) -> None:
        self.documents: dict[str, Any] = {}
        self.oracle: list[tuple[list[dict[str, Any]], str]] = []

    def write(self, name: str, document: Any) -> None:
        self.documents[name] = document

    def append_oracle(self, records: list[dict[str, Any]], *, source: str) -> None:
        self.oracle.append((list(records), source))


class FakeCleanup:
    def __init__(self) -> None:
        self.pushed: list[str] = []

    def push(self, name: str, callback: Any) -> None:
        self.pushed.append(name)


class FakeDcgmProbe:
    """``ctx.dcgm``: records each probe command; serves journal pages in order."""

    def __init__(self, pages: list[dict[str, Any]] | None = None) -> None:
        self.pages = list(pages or [])
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *arguments: str, timeout: int) -> dict[str, Any]:
        self.calls.append(arguments)
        if arguments[0] == "audit-read":
            if self.pages:
                return self.pages.pop(0)
            return {"records": [], "next_offset": None, "total": 0}
        if arguments[0] == "repoint-collector":
            return {"process_environment": {replay_module.INTERVAL_KEY: "10"}}
        return {"command": arguments[0]}


class FakeContext:
    def __init__(self, tmp_path: Path, probe: FakeDcgmProbe) -> None:
        self.dcgm = probe
        self.evidence = FakeEvidence()
        self.cleanup = FakeCleanup()
        self.case_dir = tmp_path


def make_replay(tmp_path: Path, pages: list[dict[str, Any]] | None = None) -> Any:
    ctx = FakeContext(tmp_path, FakeDcgmProbe(pages))
    replay = DcgmReplay(
        ctx=ctx,  # type: ignore[arg-type]
        plan=plan("memory", step(3), step(40, add(FIELD, [GPU]), hold=True)),
        seconds=600,
        node_boot_id="boot-current",
    )
    return replay, ctx


def read(number: int, served: dict[str, Any], aborted: str | None = None) -> dict:
    return {"read": number, "armed": True, "served": served, "aborted": aborted}


def test_arm_before_repoint_without_keys_serves_arms_and_repoints_without_a_gate(
    tmp_path: Path,
) -> None:
    replay, ctx = make_replay(tmp_path)

    arm_before_repoint(replay)

    commands = [call[0] for call in ctx.dcgm.calls]
    assert commands == [
        "install-source",
        "install-source",
        "deploy-proxy",
        "arm",
        "repoint-collector",
    ]
    assert ctx.dcgm.calls[3] == (
        "arm",
        "--port",
        str(replay_module.PORT),
        "--before-repoint",
    )
    assert ctx.cleanup.pushed == ["dcgm-replay"]
    assert (replay.armed, replay.repointed, replay.interval) == (True, True, 10)
    assert (tmp_path / "plan.json").is_file(), (
        "the reviewed plan is written for install"
    )


def test_journal_reads_follow_the_offset_until_the_total_is_reached(
    tmp_path: Path,
) -> None:
    first, second = read(1, {}), read(2, {KEY: 1.0})
    replay, ctx = make_replay(
        tmp_path,
        [
            {"records": [first], "next_offset": 1, "total": 2},
            {"records": [second], "next_offset": 2, "total": 2},
        ],
    )

    assert replay.audit() == [first, second]

    offsets = [call[4] for call in ctx.dcgm.calls]
    assert offsets == ["0", "1"]
    assert replay.audit_offset == 2
    assert replay.records == [first, second]
    assert ctx.evidence.oracle == [([first, second], "replay")]
    # An empty page ends the read and appends nothing.
    assert replay.inventory_audit() == []
    assert ctx.evidence.oracle == [([first, second], "replay")]


def test_served_any_and_aborted_read_the_journal_first(tmp_path: Path) -> None:
    replay, ctx = make_replay(
        tmp_path,
        [
            {
                "records": [read(1, {KEY: None}), read(2, {KEY: 4.0})],
                "next_offset": 2,
                "total": 2,
            },
            {
                "records": [read(3, {}, aborted="deadman fired")],
                "next_offset": 3,
                "total": 3,
            },
        ],
    )

    assert replay.served_any(KEY) is True
    assert replay.aborted() == "deadman fired"
    assert replay.served_any(OTHER_KEY) is False
    assert [call[0] for call in ctx.dcgm.calls] == ["audit-read"] * 3


def test_require_effective_names_every_key_the_replay_never_served(
    tmp_path: Path,
) -> None:
    replay, _ = make_replay(
        tmp_path, [{"records": [read(1, {KEY: 2.0})], "next_offset": 1, "total": 1}]
    )

    replay.require_effective(KEY)
    with pytest.raises(
        VariantInconclusive, match=f"the replay never served {OTHER_KEY}$"
    ):
        replay.require_effective(KEY, OTHER_KEY)
