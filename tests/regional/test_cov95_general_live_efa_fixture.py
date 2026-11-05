"""GF-GENERAL-LIVE EFA source fixture: target selection, window close, read-back."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_efa as efa
from scripts.e2e.regional import general_live_efa_fixture as fixture
from scripts.e2e.regional.general_live_fixture import VariantNotRun
from tests.regional._general_live_fakes import FakeRegional, make_context

SPEC = efa.VARIANTS[0]


def test_state_target_needs_a_port_with_both_state_files() -> None:
    inventory = {
        "devices": {
            "rdmap16s27": {"efa": True, "ports": {"1": {"state": "4: ACTIVE"}}},
            "rdmap32s27": {"efa": True, "ports": {"1": {"phys_state": "5: LinkUp"}}},
            "mlx5_0": {"efa": False, "ports": {"1": {"state": "x", "phys_state": "y"}}},
        }
    }

    with pytest.raises(
        VariantNotRun, match="no EFA port has readable state files"
    ) as caught:
        fixture.state_target(inventory)

    assert caught.value.reason == "CAPABILITY_ABSENT"


def test_pair_target_on_a_node_without_efa_is_capability_absent() -> None:
    with pytest.raises(VariantNotRun, match=re.escape(fixture.NO_EFA)):
        fixture.pair_target({"devices": {"mlx5_0": {"efa": False}}})


def test_episodes_skip_readings_that_are_not_numbers() -> None:
    metric, device = "efa_rdma_read_resp_err_delta", "rdmap16s27"

    def batch(value: Any) -> dict[str, Any]:
        return {
            "observed_at": "2026-11-05T04:00:00+00:00",
            "samples": [{"name": metric, "device": device, "value": value}],
        }

    document = {"batches": [batch(5), batch("n/a"), batch(0), batch(6)]}

    assert fixture.episodes(document, metric, device, threshold=1) == 2


def test_a_validation_that_did_not_finish_is_named() -> None:
    document = {
        "workflows": [
            {
                "request_id": "wf",
                "incident_id": "inc",
                "official_steps": [{"operation": "VALIDATE_FABRIC"}],
                "step_executions": [
                    {
                        "step_index": 0,
                        "operation": "VALIDATE_FABRIC",
                        "status": "RUNNING",
                    }
                ],
            }
        ]
    }

    assert fixture.validation_errors(document, {"incident_id": "inc"}, "m") == [
        "VALIDATE_FABRIC ended RUNNING"
    ]


class Replay:
    """The host source replay: records the calls, answers the read-backs."""

    def __init__(self, views: dict[str, dict[str, Any]], closed: dict[str, Any]):
        self.views = views
        self.closed = closed
        self.installed: list[str] = []

    def install(self, name: str, payload: bytes) -> None:
        self.installed.append(name)

    def close(self) -> dict[str, Any]:
        return dict(self.closed)

    def read_back(self, dest: str) -> dict[str, Any]:
        return dict(self.views[dest])


def source_fixture(tmp_path: Path, replay: Replay) -> fixture.EfaSourceFixture:
    ctx = make_context(SPEC, tmp_path=tmp_path, regional=FakeRegional(), probes={})
    return fixture.EfaSourceFixture(ctx, replay)  # type: ignore[arg-type]


def test_attaching_a_netdev_needs_one_named_on_the_target(tmp_path: Path) -> None:
    source = source_fixture(tmp_path, Replay({}, {}))
    target = fixture.EfaTarget("rdmap16s27", "1", None, None, "4: ACTIVE", None)

    with pytest.raises(ValueError, match="the target names no netdev to attach"):
        source.attach_netdev(target)


def test_a_window_that_did_not_close_clean_is_an_error(tmp_path: Path) -> None:
    source = source_fixture(tmp_path, Replay({}, {"clean": False, "left": ["bind"]}))

    with pytest.raises(RuntimeError, match="the source window did not close clean"):
        source.close()

    assert source.closed_at is not None, "the close time is kept for the verdict"


def test_restored_errors_name_each_source_still_replaced(tmp_path: Path) -> None:
    readable = {"readable": True, "value": "17"}
    replay = Replay(
        {
            "/sys/a": {"unit": {"readable": False}, "host": readable, "same": False},
            "/sys/b": {"unit": readable, "host": readable, "same": False},
            "/sys/c": {
                "unit": {"readable": True, "value": "99"},
                "host": readable,
                "same": True,
            },
            "/sys/d": {"unit": readable, "host": readable, "same": True},
        },
        {"clean": True},
    )
    source = source_fixture(tmp_path, replay)
    for name, dest, value in (
        ("a", "/sys/a", "5"),
        ("b", "/sys/b", "5"),
        ("c", "/sys/c", "99"),
        ("d", "/sys/d", "5"),
    ):
        source.bind(name, dest, value)

    assert source.restored_errors() == [
        "/sys/a: not readable after the window closed",
        "/sys/b: the unit still reads another value than the host",
        "/sys/c: the unit still reads the fixture value",
    ]
    assert replay.installed == ["a", "b", "c", "d"]
