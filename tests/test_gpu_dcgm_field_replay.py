"""The reviewed DCGM field replay tool: local fixtures only, never a real node."""

import io
import json
import math
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from gpu_fault.collectors.gpu.dcgm import parse_dcgm_samples
from gpu_fault.gpu_observation import normalize_gpu_observation
from tools import gpu_dcgm_field_replay as tool
from tools import gpu_environment_replay as environment_tool

A = "GPU-aaaaaaaa-0000-0000-0000-000000000001"
B = "GPU-bbbbbbbb-0000-0000-0000-000000000002"


def exporter(**overrides: float) -> bytes:
    """A two-GPU scrape in the exporter's own text format."""

    values = {
        "sbe": 0,
        "dbe": 0,
        "pending": 0,
        "failure": 0,
        "crc": 0,
        "replay": 0,
        "recovery": 0,
        "pcie": 0,
        "fabric": 3,
        **overrides,
    }
    lines = []
    for gpu, uuid in (("0", A), ("1", B)):
        labels = f'gpu="{gpu}",UUID="{uuid}",pci_bus_id="0000:{gpu}9:00.0",modelName="NVIDIA H200"'
        for name, key in (
            ("DCGM_FI_DEV_ECC_SBE_VOL_TOTAL", "sbe"),
            ("DCGM_FI_DEV_ECC_DBE_VOL_TOTAL", "dbe"),
            ("DCGM_FI_DEV_ROW_REMAP_PENDING", "pending"),
            ("DCGM_FI_DEV_ROW_REMAP_FAILURE", "failure"),
            ("DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL", "crc"),
            ("DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL", "replay"),
            ("DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL", "recovery"),
            ("DCGM_FI_DEV_PCIE_REPLAY_COUNTER", "pcie"),
            ("DCGM_FI_DEV_FABRIC_MANAGER_STATUS", "fabric"),
        ):
            lines.append(f"{name}{{{labels}}} {values[key]}")
        lines.append(f"DCGM_FI_DEV_GPU_TEMP{{{labels}}} 40")
        lines.append(f"DCGM_FI_DEV_CLOCKS_EVENT_REASONS{{{labels}}} 0")
        lines.append(f"DCGM_FI_DEV_POWER_MGMT_LIMIT{{{labels}}} 700")
    return ("\n".join(lines) + "\n").encode()


#: The DL aggregate NVLink counters and their canonical names: what a Hopper
#: (NVLink4) exporter publishes instead of the per-class counters above
#: (live 2026-10-26, p5en H200, DCGM 4.6/4.7).
DL_AGGREGATES = (
    ("DCGM_FI_DEV_NVLINK_ERROR_DL_CRC", "nvlink_crc_aggregate_error_total"),
    ("DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY", "nvlink_replay_aggregate_error_total"),
    ("DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY", "nvlink_recovery_aggregate_error_total"),
)


def hopper_exporter(**overrides: float) -> bytes:
    """A two-GPU Hopper scrape: typed DL aggregate counters, no per-class ones.

    The exporter declares the DL fields ``counter``, so the product's parser
    (and this tool's) sees them as ``<name>_total`` samples -- the shape the
    live survey recorded (``DCGM_FI_DEV_NVLINK_ERROR_DL_CRC_total``).
    """

    values = {"crc": 0, "replay": 0, "recovery": 0, "pcie": 0, **overrides}
    lines = []
    for name, _canonical in DL_AGGREGATES:
        lines.append(f"# TYPE {name} counter")
        key = name.rsplit("_", 1)[1].lower()
        for gpu, uuid in (("0", A), ("1", B)):
            labels = f'gpu="{gpu}",UUID="{uuid}",pci_bus_id="0000:{gpu}9:00.0",modelName="NVIDIA H200"'
            lines.append(f"{name}{{{labels}}} {values[key]}")
    for gpu, uuid in (("0", A), ("1", B)):
        labels = f'gpu="{gpu}",UUID="{uuid}",pci_bus_id="0000:{gpu}9:00.0",modelName="NVIDIA H200"'
        lines.append(f"DCGM_FI_DEV_PCIE_REPLAY_COUNTER{{{labels}}} {values['pcie']}")
        lines.append(f"DCGM_FI_DEV_GPU_TEMP{{{labels}}} 40")
    return ("\n".join(lines) + "\n").encode()


def product_view(body: bytes) -> dict[tuple[str, str], list[float]]:
    """What the deployed collector's own parser makes of a served scrape."""

    result: dict[tuple[str, str], list[float]] = {}
    for sample in parse_dcgm_samples(body.decode()):
        result.setdefault((sample.gpu_uuid, sample.canonical_name), []).append(
            sample.value
        )
    return result


def plan(profile: str, *steps: tuple[int, list[dict]]) -> dict:
    return tool.validate_plan(
        {
            "profile": profile,
            "steps": [{"reads": reads, "ops": ops} for reads, ops in steps],
        }
    )


def armed(profile: str, *steps: tuple[int, list[dict]], seconds: int = 600):
    replay = tool.Replay(plan(profile, *steps), seconds)
    replay.arm()
    return replay


def serve_once(replay, body: bytes) -> tuple[bytes, dict]:
    output, receipt = replay.render(body)
    replay.reads += 1
    return output, receipt


def test_value_profiles_never_reach_the_environment_tool_fields():
    environment = set(environment_tool.ALLOWED_FIELDS)
    for name in ("memory", "nvlink", "fabric", "pcie", "composite"):
        overlap = tool.PROFILES[name].fields & environment
        assert not overlap, (
            f"{name} profile widens the environment allow-list: {overlap}"
        )
    structural = tool.PROFILES["completeness"]
    assert not structural.fields & tool.SAFETY_FIELDS, "safety readings stay untouched"
    assert structural.operations.isdisjoint({"add", "set", "label"}), (
        "the completeness profile changes structure, never a value"
    )


@pytest.mark.parametrize(
    ("profile", "operation"),
    [
        (
            "memory",
            {"op": "add", "field": "gpu_temperature_c", "gpus": [A], "value": 1},
        ),
        (
            "memory",
            {
                "op": "add",
                "field": "nvlink_replay_error_total",
                "gpus": [A],
                "value": 1,
            },
        ),
        (
            "nvlink",
            {"op": "add", "field": "ecc_dbe_volatile_total", "gpus": [A], "value": 1},
        ),
        ("pcie", {"op": "set", "field": "pcie_replay_total", "gpus": [A], "value": 1}),
        (
            "memory",
            {"op": "set", "field": "ecc_dbe_volatile_total", "gpus": [A], "value": 1},
        ),
        (
            "memory",
            {"op": "set", "field": "row_remap_pending", "gpus": [A], "value": 2},
        ),
        (
            "memory",
            {"op": "add", "field": "ecc_sbe_volatile_total", "gpus": [A], "value": 0},
        ),
        (
            "memory",
            {
                "op": "add",
                "field": "ecc_sbe_volatile_total",
                "gpus": [A],
                "value": True,
            },
        ),
        (
            "memory",
            {
                "op": "add",
                "field": "ecc_sbe_volatile_total",
                "gpus": ["GPU-x"],
                "value": 1,
            },
        ),
        (
            "memory",
            {
                "op": "add",
                "field": "ecc_sbe_volatile_total",
                "gpus": [A, A],
                "value": 1,
            },
        ),
        ("memory", {"op": "omit", "field": "ecc_sbe_volatile_total", "gpus": [A]}),
        (
            "fabric",
            {
                "op": "label",
                "field": "fabric_manager_status",
                "gpus": [A],
                "label": "UUID",
                "value": "x",
            },
        ),
        (
            "fabric",
            {
                "op": "label",
                "field": "fabric_manager_status",
                "gpus": [A],
                "label": "modelName",
                "value": "rm -rf",
            },
        ),
        (
            "pcie",
            {
                "op": "label",
                "field": "pcie_replay_total",
                "gpus": [A],
                "label": "modelName",
                "value": "NVIDIA L40S",
            },
        ),
        ("completeness", {"op": "nan", "field": "gpu_temperature_c", "gpus": [A]}),
        (
            "completeness",
            {
                "op": "alias",
                "field": "clock_throttle_reasons",
                "gpus": [A],
                "name": "DCGM_FI_DEV_GPU_TEMP",
                "value": 1,
            },
        ),
        (
            "completeness",
            {"op": "duplicate", "field": "power_limit_w", "gpus": [A], "value": -5},
        ),
        (
            "completeness",
            {"op": "add", "field": "pcie_replay_total", "gpus": [A], "value": 1},
        ),
        ("nvlink", {"op": "stale", "age": 0}),
        (
            "nvlink",
            {
                "op": "add",
                "field": "nvlink_crc_flit_error_total",
                "gpus": [A],
                "value": 1,
                "extra": 1,
            },
        ),
    ],
)
def test_plan_refuses_what_a_reviewer_would_refuse(profile, operation):
    with pytest.raises(ValueError):
        plan(profile, (1, [operation]))


def test_plan_bounds_profiles_steps_and_reads():
    with pytest.raises(ValueError, match="unknown profile"):
        tool.validate_plan(
            {"profile": "environment", "steps": [{"reads": 1, "ops": []}]}
        )
    with pytest.raises(ValueError, match="exactly profile and steps"):
        tool.validate_plan({"steps": [{"reads": 1, "ops": []}]})
    with pytest.raises(ValueError, match="collector reads"):
        plan("nvlink", *[(200, [])] * 3)
    with pytest.raises(ValueError, match="nothing else"):
        plan(
            "nvlink",
            (
                1,
                [
                    {"op": "stale", "age": 1},
                    {
                        "op": "add",
                        "field": "nvlink_crc_flit_error_total",
                        "gpus": [A],
                        "value": 1,
                    },
                ],
            ),
        )


def test_pass_through_is_byte_identical_until_armed_and_after_the_plan():
    replay = tool.Replay(
        plan(
            "memory",
            (
                1,
                [
                    {
                        "op": "add",
                        "field": "ecc_sbe_volatile_total",
                        "gpus": [A],
                        "value": 1,
                    }
                ],
            ),
        ),
        600,
    )
    body = exporter()
    assert serve_once(replay, body)[0] == body, "an unarmed proxy changes nothing"
    replay.arm()
    output, receipt = serve_once(replay, body)
    assert product_view(output)[(A, "ecc_sbe_volatile_total")] == [1.0], output
    assert product_view(output)[(B, "ecc_sbe_volatile_total")] == [0.0], (
        "other GPUs untouched"
    )
    assert (
        receipt["proof_scope"] == "live-replay"
        and not receipt["physical_fault_injected"]
    )
    assert serve_once(replay, body)[0] == body, "the finished plan passes through"


def test_add_accumulates_per_read_persists_across_steps_and_clear_resets():
    replay = armed(
        "pcie",
        (3, [{"op": "add", "field": "pcie_replay_total", "gpus": [A], "value": 3}]),
        (2, []),
        (1, [{"op": "clear", "field": "pcie_replay_total", "gpus": [A]}]),
        (1, [{"op": "add", "field": "pcie_replay_total", "gpus": [A], "value": 2}]),
    )
    served = [
        product_view(serve_once(replay, exporter(pcie=5))[0])[(A, "pcie_replay_total")][
            0
        ]
        for _ in range(7)
    ]
    assert served == [8, 11, 14, 14, 14, 5, 7], served


def test_set_holds_only_for_its_step_and_label_rewrites_one_sample():
    replay = armed(
        "fabric",
        (
            2,
            [
                {
                    "op": "set",
                    "field": "fabric_manager_status",
                    "gpus": [A, B],
                    "value": 2,
                },
                {
                    "op": "label",
                    "field": "fabric_manager_status",
                    "gpus": [B],
                    "label": "modelName",
                    "value": "NVIDIA L40S",
                },
            ],
        ),
        (1, []),
    )
    first = serve_once(replay, exporter())[0]
    by_gpu = {
        sample.gpu_uuid: sample
        for sample in parse_dcgm_samples(first.decode())
        if sample.canonical_name == "fabric_manager_status"
    }
    assert by_gpu[A].value == 2 and by_gpu[B].value == 2, by_gpu
    assert by_gpu[A].labels["modelName"] == "NVIDIA H200"
    assert by_gpu[B].labels["modelName"] == "NVIDIA L40S"
    others = [
        sample.labels["modelName"]
        for sample in parse_dcgm_samples(first.decode())
        if sample.gpu_uuid == B and sample.canonical_name != "fabric_manager_status"
    ]
    assert set(others) == {"NVIDIA H200"}, "only the targeted sample is relabelled"
    serve_once(replay, exporter())
    third = serve_once(replay, exporter())[0]
    assert product_view(third)[(A, "fabric_manager_status")] == [3.0], (
        "set does not persist"
    )


def test_structural_operations_reach_the_product_parser_and_normalizer():
    replay = armed(
        "completeness",
        (
            1,
            [
                {"op": "omit", "field": "row_remap_failure", "gpus": [A]},
                {"op": "nan", "field": "ecc_dbe_volatile_total", "gpus": [A]},
                {"op": "negative", "field": "pcie_replay_total", "gpus": [A]},
                {
                    "op": "duplicate",
                    "field": "power_limit_w",
                    "gpus": [A],
                    "value": 650,
                },
                {
                    "op": "alias",
                    "field": "clock_throttle_reasons",
                    "gpus": [A],
                    "name": "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS",
                    "value": 1,
                },
            ],
        ),
    )
    output, receipt = serve_once(replay, exporter())
    view = product_view(output)
    assert (A, "row_remap_failure") not in view, "the omitted slot is gone"
    assert (B, "row_remap_failure") in view, "the other GPU keeps its field"
    assert (A, "ecc_dbe_volatile_total") not in view, "the parser drops a NaN slot"
    assert view[(A, "pcie_replay_total")] == [-1.0]
    assert sorted(view[(A, "power_limit_w")]) == [650.0, 700.0]
    assert "nan" in output.decode().lower()
    assert receipt["served"][f"{A}/row_remap_failure"] == "omitted"
    batch = type(
        "Batch",
        (),
        {
            "collection_errors": [],
            "samples": parse_dcgm_samples(output.decode()),
            "model_copy": lambda self, update: update,
        },
    )()
    normalized = normalize_gpu_observation(batch)
    joined = " ".join(normalized["collection_errors"])
    assert f"{A}/pcie_replay_total" in joined, joined
    assert f"{A}/power_limit_w" in joined, joined


def test_omit_gpu_removes_every_sample_of_that_gpu_only():
    replay = armed("completeness", (1, [{"op": "omit_gpu", "gpus": [B]}]))
    view = product_view(serve_once(replay, exporter())[0])
    assert not [key for key in view if key[0] == B], view
    assert (A, "gpu_temperature_c") in view


def test_stale_serves_an_earlier_armed_scrape_verbatim():
    replay = armed(
        "nvlink",
        (2, []),
        (
            1,
            [
                {
                    "op": "add",
                    "field": "nvlink_crc_flit_error_total",
                    "gpus": [A],
                    "value": 1,
                }
            ],
        ),
        (1, [{"op": "stale", "age": 2}]),
    )
    first = serve_once(replay, exporter())[0]
    serve_once(replay, exporter())
    increased = serve_once(replay, exporter())[0]
    assert product_view(increased)[(A, "nvlink_crc_flit_error_total")] == [1.0]
    stale, receipt = serve_once(replay, exporter())
    assert receipt["served"] == {"stale_age_reads": 2}
    assert product_view(stale) == product_view(first), "the older scrape comes back"


def test_an_absent_target_or_a_real_change_aborts_and_is_never_masked():
    replay = armed(
        "memory",
        (
            5,
            [
                {
                    "op": "set",
                    "field": "row_remap_pending",
                    "gpus": ["GPU-cccccccc-0000"],
                    "value": 1,
                }
            ],
        ),
    )
    body = exporter()
    output, receipt = serve_once(replay, body)
    assert output == body and "invented" in receipt["aborted"], receipt
    replay = armed(
        "memory",
        (
            5,
            [{"op": "add", "field": "ecc_sbe_volatile_total", "gpus": [A], "value": 1}],
        ),
    )
    serve_once(replay, exporter())
    real_fault = exporter(sbe=7)
    output, receipt = serve_once(replay, real_fault)
    assert output == real_fault, "a real error passes through unchanged"
    assert "real upstream change" in receipt["aborted"], receipt
    assert serve_once(replay, exporter())[0] == exporter(), "the abort is latched"


def test_disarm_and_deadline_end_the_replay_for_good(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(tool.time, "monotonic", lambda: clock[0])
    operation = {
        "op": "add",
        "field": "ecc_sbe_volatile_total",
        "gpus": [A],
        "value": 1,
    }
    replay = armed("memory", (50, [operation]), seconds=60)
    assert serve_once(replay, exporter())[0] != exporter()
    clock[0] = 161.0
    assert serve_once(replay, exporter())[0] == exporter(), "past the deadline"
    replay = armed("memory", (50, [operation]))
    replay.disarm()
    assert serve_once(replay, exporter())[0] == exporter()
    with pytest.raises(ValueError, match="new audited run"):
        replay.arm()


def test_loopback_http_arm_disarm_status_and_upstream_failure(monkeypatch):
    failed = [False]

    def upstream(*_args, **_kwargs):
        if failed[0]:
            raise OSError("upstream unavailable")
        return io.BytesIO(exporter())

    monkeypatch.setattr(tool, "open_upstream", upstream)
    replay = tool.Replay(
        plan(
            "memory",
            (3, [{"op": "set", "field": "row_remap_pending", "gpus": [A], "value": 1}]),
        ),
        600,
    )
    audit = io.StringIO()
    server = tool.make_server("http://127.0.0.1:9400/metrics", 0, replay, audit)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    acknowledged = {"X-GFRS-Replay-Ack": tool.ACKNOWLEDGEMENT}
    try:
        with urlopen(base + "/status", timeout=5) as response:
            status = json.load(response)
        assert status["armed"] is False and status["reads"] == 0, status
        with pytest.raises(HTTPError) as refused:
            urlopen(Request(base + "/arm", method="POST"), timeout=5)
        assert refused.value.code == 403
        with urlopen(
            Request(base + "/arm", method="POST", headers=acknowledged), timeout=5
        ):
            pass
        with urlopen(base + "/metrics", timeout=5) as response:
            assert product_view(response.read())[(A, "row_remap_pending")] == [1.0]
        with urlopen(
            Request(base + "/disarm", method="POST", headers=acknowledged), timeout=5
        ):
            pass
        with urlopen(base + "/metrics", timeout=5) as response:
            assert response.read() == exporter(), "disarmed means pass-through"
        failed[0] = True
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/metrics", timeout=5)
        assert error.value.code == 503
        assert replay.reads == 2, "a failed upstream read consumes no step"
        receipts = [json.loads(line) for line in audit.getvalue().splitlines()]
        events = [item.get("event") for item in receipts if "event" in item]
        assert events == ["ARM", "DISARM"], receipts
        assert receipts[-1]["error"] == "upstream unavailable"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_cli_validates_by_default_and_serving_needs_ack_audit_and_profile(
    tmp_path, monkeypatch
):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "profile": "nvlink",
                "steps": [
                    {
                        "reads": 2,
                        "ops": [
                            {
                                "op": "add",
                                "field": "nvlink_crc_flit_error_total",
                                "gpus": [A],
                                "value": 1,
                            }
                        ],
                    }
                ],
            }
        )
    )
    calls = []
    monkeypatch.setattr(tool, "serve", lambda *args: calls.append(args))
    assert tool.main(["--plan", str(path), "--profile", "nvlink"]) == 0
    for arguments in (
        ["--plan", str(path), "--profile", "memory"],
        ["--plan", str(path), "--profile", "nvlink", "--serve"],
        [
            "--plan",
            str(path),
            "--profile",
            "nvlink",
            "--port",
            "9400",
            "--upstream",
            "http://127.0.0.1:9400/metrics",
        ],
        ["--plan", str(path), "--profile", "nvlink", "--max-seconds", "7200"],
    ):
        with pytest.raises(SystemExit) as error:
            tool.main(arguments)
        assert error.value.code == 2, arguments
    assert calls == [], "no path above starts a network endpoint"
    audit = tmp_path / "audit.jsonl"
    assert (
        tool.main(
            [
                "--plan",
                str(path),
                "--profile",
                "nvlink",
                "--serve",
                "--ack",
                tool.ACKNOWLEDGEMENT,
                "--audit",
                str(audit),
            ]
        )
        == 0
    )
    assert len(calls) == 1 and audit.exists(), calls
    with pytest.raises(FileExistsError):
        tool.main(
            [
                "--plan",
                str(path),
                "--profile",
                "nvlink",
                "--serve",
                "--ack",
                tool.ACKNOWLEDGEMENT,
                "--audit",
                str(audit),
            ]
        )


def test_nvlink_profile_spans_both_counter_generations():
    from gpu_fault.dcgm_fields import DCGM_FABRIC_CANONICAL_GROUPS

    members = {
        name for group in DCGM_FABRIC_CANONICAL_GROUPS.values() for name in group
    }
    assert tool.PROFILES["nvlink"].fields == members
    assert tool.PROFILES["composite"].fields >= members
    assert members <= tool.MONITORED_FIELDS, "a real DL error aborts the replay too"
    for name, canonical in DL_AGGREGATES:
        assert tool.canonical_name(name) == canonical
        assert tool.canonical_name(name + "_total") == canonical, "typed counter"
        assert canonical in tool.PROFILES["nvlink"].fields
    with pytest.raises(ValueError, match="allow-list"):
        plan(
            "nvlink",
            (
                1,
                [
                    {
                        "op": "add",
                        "field": "nvlink_dl_crc_error_total",
                        "gpus": [A],
                        "value": 1,
                    }
                ],
            ),
        )
    with pytest.raises(ValueError, match="allow-list"):
        plan(
            "nvlink",
            (
                1,
                [
                    {
                        "op": "add",
                        "field": "DCGM_FI_DEV_NVLINK_ERROR_DL_CRC",
                        "gpus": [A],
                        "value": 1,
                    }
                ],
            ),
        )


@pytest.mark.parametrize(("metric", "field"), DL_AGGREGATES)
def test_nvlink_profile_alters_a_dl_aggregate_like_a_legacy_counter(metric, field):
    key = metric.rsplit("_", 1)[1].lower()
    replay = armed(
        "nvlink",
        (2, [{"op": "add", "field": field, "gpus": [A], "value": 1}]),
        (1, [{"op": "clear", "field": field, "gpus": [A]}]),
        (1, [{"op": "stale", "age": 1}]),
    )
    body = hopper_exporter(**{key: 4})
    first, receipt = serve_once(replay, body)
    view = product_view(first)
    assert view[(A, field)] == [5.0], view
    assert view[(B, field)] == [4.0], "the other GPU is untouched"
    assert receipt["served"] == {f"{A}/{field}": 5.0}
    assert receipt["real"] == {f"{A}/{field}": 4.0}, (
        "the DL sample is the real baseline"
    )
    assert f"{metric}_total{{" in first.decode(), "served under its typed exporter name"
    assert product_view(serve_once(replay, body)[0])[(A, field)] == [6.0], "accumulates"
    cleared = product_view(serve_once(replay, body)[0])
    assert cleared[(A, field)] == [4.0], "clear drops the offset: a counter reset"
    stale, receipt = serve_once(replay, body)
    assert receipt["served"] == {"stale_age_reads": 1}
    assert product_view(stale) == cleared, "the stale step re-serves the last scrape"
    assert serve_once(replay, body)[0] == body, "plan done: byte-identical pass-through"
    others = {k: v for k, v in product_view(first).items() if k[1] != field}
    assert others == {k: v for k, v in product_view(body).items() if k[1] != field}


def test_a_legacy_counter_targeted_on_a_hopper_scrape_aborts_as_absent():
    # The failure mode behind the 2026-10-26 live NOT_RUNs, had the survey not
    # refused first: the flit counter does not exist upstream, so the replay
    # aborts rather than invent it, and the scrape passes through unchanged.
    replay = armed(
        "nvlink",
        (
            2,
            [
                {
                    "op": "add",
                    "field": "nvlink_crc_flit_error_total",
                    "gpus": [A],
                    "value": 1,
                }
            ],
        ),
    )
    body = hopper_exporter()
    output, receipt = serve_once(replay, body)
    assert output == body
    assert "refusing invented samples" in receipt["aborted"], receipt
    assert f"{A}/nvlink_crc_flit_error_total" in receipt["aborted"]


def test_a_real_dl_aggregate_change_aborts_and_is_never_masked():
    field = "nvlink_replay_aggregate_error_total"
    replay = armed(
        "nvlink", (5, [{"op": "add", "field": field, "gpus": [A], "value": 1}])
    )
    serve_once(replay, hopper_exporter())
    real_fault = hopper_exporter(replay=3)
    output, receipt = serve_once(replay, real_fault)
    assert output == real_fault, "a real DL error passes through unchanged"
    assert "real upstream change" in receipt["aborted"], receipt


def test_canonical_names_follow_the_product_parser():
    assert (
        tool.canonical_name("DCGM_FI_DEV_PCIE_REPLAY_COUNTER_total")
        == "pcie_replay_total"
    )
    assert (
        tool.canonical_name("DCGM_FI_DEV_CLOCK_THROTTLE_REASONS")
        == "clock_throttle_reasons"
    )
    assert tool.canonical_name("DCGM_FI_DEV_UNKNOWN") is None
    assert math.isnan(float("nan")), "sanity"


def test_a_held_step_lasts_until_advance_and_its_bound_ends_in_pass_through():
    hold = {"op": "set", "field": "fabric_manager_status", "gpus": [A, B], "value": 2}
    straggler = {"op": "set", "field": "fabric_manager_status", "gpus": [B], "value": 2}
    replay = tool.Replay(
        tool.validate_plan(
            {
                "profile": "fabric",
                "steps": [
                    {"reads": 5, "ops": [hold], "until_advance": True},
                    {"reads": 2, "ops": [straggler]},
                ],
            }
        ),
        600,
    )
    with pytest.raises(ValueError, match="not on a step"):
        replay.advance()
    replay.arm()
    for _ in range(3):
        view = product_view(serve_once(replay, exporter())[0])
        assert view[(A, "fabric_manager_status")] == [2.0], view
    assert replay.advance() == 1
    view = product_view(serve_once(replay, exporter())[0])
    assert view[(A, "fabric_manager_status")] == [3.0], "A left the held state"
    assert view[(B, "fabric_manager_status")] == [2.0], "the straggler stays"
    with pytest.raises(ValueError, match="not held"):
        replay.advance()
    serve_once(replay, exporter())
    assert serve_once(replay, exporter())[0] == exporter(), "plan done: pass-through"
    bounded = armed("fabric", (2, [hold]))
    bounded.plan["steps"][0]["until_advance"] = True
    serve_once(bounded, exporter())
    serve_once(bounded, exporter())
    assert serve_once(bounded, exporter())[0] == exporter(), (
        "a held step's bound ends it"
    )


def test_the_xid_gauge_may_only_be_replayed_as_xid_32_beside_the_pcie_counter():
    with pytest.raises(ValueError, match="only be set"):
        plan(
            "pcie_xid",
            (1, [{"op": "set", "field": "xid_last_error", "gpus": [A], "value": 79}]),
        )
    with pytest.raises(ValueError, match="allow-list"):
        plan(
            "pcie",
            (1, [{"op": "add", "field": "xid_last_error", "gpus": [A], "value": 32}]),
        )
    body = exporter().decode() + f'DCGM_FI_DEV_XID_ERRORS{{gpu="0",UUID="{A}"}} 0\n'
    replay = armed(
        "pcie_xid",
        (
            2,
            [
                {"op": "set", "field": "xid_last_error", "gpus": [A], "value": 32},
                {"op": "add", "field": "pcie_replay_total", "gpus": [A], "value": 3},
            ],
        ),
    )
    view = product_view(serve_once(replay, body.encode())[0])
    assert view[(A, "xid_last_error")] == [32.0] and view[(A, "pcie_replay_total")] == [
        3.0
    ]


def test_a_held_plan_step_needs_a_boolean_hold():
    with pytest.raises(ValueError, match="boolean"):
        tool.validate_plan(
            {
                "profile": "nvlink",
                "steps": [{"reads": 1, "ops": [], "until_advance": 1}],
            }
        )
    with pytest.raises(ValueError, match="may hold"):
        tool.validate_plan(
            {"profile": "nvlink", "steps": [{"reads": 1, "ops": [], "hold": True}]}
        )


def test_composite_profile_serves_the_second_memory_component_the_exporter_has():
    # The 2026-10-26 live INCONCLUSIVE of 024 fresh_components, had the survey
    # not refused first: a Hopper scrape has no retired-page counter, so the
    # composite replay aborts rather than invent it, while the correctable
    # row-remap counter the same part publishes is served like any counter.
    # The pending-repair flags the memory_failure / wrong_scope_source plans may
    # choose are in the profile too.
    for field in (
        "ecc_sbe_volatile_total",
        "retired_pages_sbe_total",
        "row_remap_correctable_total",
        "ecc_dbe_volatile_total",
        "row_remap_pending",
        "retired_pages_pending",
    ):
        assert field in tool.PROFILES["composite"].fields, field
    assert {"add", "set"} <= tool.PROFILES["composite"].operations
    assert tool.canonical_name("DCGM_FI_DEV_RETIRED_SBE") == "retired_pages_sbe_total"
    assert (
        tool.canonical_name("DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS")
        == "row_remap_correctable_total"
    )
    pair = [
        {"op": "add", "field": "ecc_sbe_volatile_total", "gpus": [A], "value": 1},
        {"op": "add", "field": "retired_pages_sbe_total", "gpus": [A], "value": 1},
    ]
    body = exporter()
    output, receipt = serve_once(armed("composite", (2, pair)), body)
    assert output == body
    assert "refusing invented samples" in receipt["aborted"], receipt
    assert f"{A}/retired_pages_sbe_total" in receipt["aborted"]
    assert f"{A}/ecc_sbe_volatile_total" not in receipt["aborted"], (
        "only the absent reading is named"
    )
    hopper = body + b"".join(
        f'DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS{{gpu="{gpu}",UUID="{uuid}",'
        f'pci_bus_id="0000:{gpu}9:00.0",modelName="NVIDIA H200"}} 0\n'.encode()
        for gpu, uuid in (("0", A), ("1", B))
    )
    pair[1]["field"] = "row_remap_correctable_total"
    first, receipt = serve_once(armed("composite", (2, pair)), hopper)
    view = product_view(first)
    assert view[(A, "ecc_sbe_volatile_total")] == [1.0], view
    assert view[(A, "row_remap_correctable_total")] == [1.0], view
    assert view[(B, "row_remap_correctable_total")] == [0.0], "the other GPU untouched"
    assert receipt["served"] == {
        f"{A}/ecc_sbe_volatile_total": 1.0,
        f"{A}/row_remap_correctable_total": 1.0,
    }
    assert receipt["aborted"] is None
