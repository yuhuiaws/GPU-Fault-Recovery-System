"""The collector window probe: the lists a window may extend, and its CLI contract.

GF-GENERAL-LIVE adds one test mount or one test interface to the host
collector's own lists for the length of a window (``--append-env``); the
node's entries stay in front. These cases pin that rule on the probe's own
``appended_settings`` (which names and units may be extended, with which test
entries, how many, deduplicated and appended after the node's list or its
default) and through ``open_window`` on the window fixture host, a drop-in
that appears while a window opens, and the probe's answer contract: one JSON
object on stdout and exit 1 for a refused command, also when run as a script.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import collector_window_probe as probe
from tests.regional import _cov95_collect_window as window_support
from tests.regional._cov95_collect_net import no_external_effects  # noqa: F401
from tests.regional._cov95_collect_window import window_arguments

window_host = window_support.window_host

HOST = probe.APPENDABLE_UNIT
MOUNT = "/mnt/gfrs-live/0123456789ab/xfs"
OTHER_MOUNT = "/mnt/gfrs-live/0123456789ab/ext"
INTERFACE = "gfrs00aa11b"
SCRIPT = Path(probe.__file__).resolve()


@pytest.fixture
def collector_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "collector.env"
    path.write_text(
        "GPU_FAULT_CLUSTER_ID=cluster-a\nGPU_FAULT_FILESYSTEMS='/,/var'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(probe, "COLLECTOR_ENV", path)
    return path


# --------------------------------------------------------------------------- #
# appended_settings
# --------------------------------------------------------------------------- #
def test_test_entries_follow_the_nodes_own_list_once_each(collector_env: Path) -> None:
    settings = probe.appended_settings(
        HOST,
        [
            f"GPU_FAULT_FILESYSTEMS={MOUNT}",
            f"GPU_FAULT_FILESYSTEMS={MOUNT}",
            f"GPU_FAULT_REQUIRED_INTERFACES={INTERFACE}",
        ],
    )

    assert settings == {
        "GPU_FAULT_FILESYSTEMS": {"items": [MOUNT], "value": f"/,/var,{MOUNT}"},
        "GPU_FAULT_REQUIRED_INTERFACES": {"items": [INTERFACE], "value": INTERFACE},
    }, "the node's list stays in front; an unset list starts from its default"


def test_a_list_the_node_does_not_set_starts_from_the_collector_default(
    collector_env: Path,
) -> None:
    collector_env.write_text("GPU_FAULT_CLUSTER_ID=cluster-a\n", encoding="utf-8")

    settings = probe.appended_settings(
        HOST, [f"GPU_FAULT_FILESYSTEMS={MOUNT}", f"GPU_FAULT_FILESYSTEMS={OTHER_MOUNT}"]
    )

    assert settings["GPU_FAULT_FILESYSTEMS"] == {
        "items": [MOUNT, OTHER_MOUNT],
        "value": f"/,/var,/tmp,{MOUNT},{OTHER_MOUNT}",
    }


def test_nothing_to_append_reads_no_collector_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe, "COLLECTOR_ENV", tmp_path / "absent.env")

    assert probe.appended_settings(HOST, []) == {}


@pytest.mark.parametrize(
    "unit, pair, message",
    [
        (HOST, "GPU_FAULT_FILESYSTEMS", "'GPU_FAULT_FILESYSTEMS' is not an appendable"),
        (HOST, "GPU_FAULT_EXPECTED_GPU_COUNT=9", "is not an appendable collector"),
        (
            "gpu-fault-kernel-collector.service",
            f"GPU_FAULT_FILESYSTEMS={MOUNT}",
            "only the host collector's lists may be extended",
        ),
        (
            HOST,
            "GPU_FAULT_FILESYSTEMS=/data",
            "GPU_FAULT_FILESYSTEMS accepts only a test entry",
        ),
        (
            HOST,
            "GPU_FAULT_REQUIRED_INTERFACES=eth0",
            "GPU_FAULT_REQUIRED_INTERFACES accepts only a test entry",
        ),
    ],
    ids=["no-value", "not-a-list", "other-unit", "real-mount", "real-interface"],
)
def test_only_test_entries_of_the_host_collectors_lists_are_accepted(
    collector_env: Path, unit: str, pair: str, message: str
) -> None:
    with pytest.raises(probe.ProbeError, match=message):
        probe.appended_settings(unit, [pair])


def test_a_list_takes_at_most_four_test_entries(collector_env: Path) -> None:
    pairs = [
        f"GPU_FAULT_FILESYSTEMS=/mnt/gfrs-live/0123456789ab/{name}"
        for name in ("a", "b", "c", "d", "e")
    ]

    with pytest.raises(probe.ProbeError, match="accepts at most 4 test entries"):
        probe.appended_settings(HOST, pairs)


# --------------------------------------------------------------------------- #
# open_window on the fixture host
# --------------------------------------------------------------------------- #
def test_a_window_installs_the_extended_list_beside_the_nodes_own(
    window_host: Any,
) -> None:
    arguments = window_arguments(
        unit=HOST, unset=[], append_env=[f"GPU_FAULT_FILESYSTEMS={MOUNT}"]
    )

    probe.open_window(arguments)

    opened = window_host.emitted[-1]
    assert opened["appended"] == {"GPU_FAULT_FILESYSTEMS": [MOUNT]}
    assert opened["overrides"] == ["GPU_FAULT_FILESYSTEMS"]
    paths = probe.window_paths(arguments.run_id)
    assert paths["override_env"].read_text(encoding="utf-8") == (
        f"GPU_FAULT_FILESYSTEMS=/,/var,/tmp,{MOUNT}\n"
    ), "the collector's default list is kept in front of the test mount"
    dropin = probe.dropin_path(HOST, arguments.run_id).read_text(encoding="utf-8")
    assert f"EnvironmentFile=-{paths['override_env']}" in dropin, dropin


def test_a_drop_in_that_appears_while_the_window_opens_is_left_alone(
    window_host: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = window_arguments(unit=HOST)
    foreign = probe.dropin_path(HOST, arguments.run_id)
    fixture_run = probe.subprocess.run

    def run(command: list[str], **kwargs: Any) -> Any:
        if command[:3] == ["systemctl", "show", HOST]:
            foreign.parent.mkdir(parents=True, exist_ok=True)
            foreign.write_text(
                "[Service]\n# another tool's drop-in\n", encoding="utf-8"
            )
        return fixture_run(command, **kwargs)

    monkeypatch.setattr(probe.subprocess, "run", run)

    with pytest.raises(probe.ProbeError, match="an untracked collector drop-in"):
        probe.open_window(arguments)

    assert foreign.read_text(encoding="utf-8") == (
        "[Service]\n# another tool's drop-in\n"
    ), "the probe never removes a drop-in it did not write"
    assert not probe.window_paths(arguments.run_id)["root"].exists(), (
        "the half-opened window's own directory is removed"
    )
    assert not any(call[0][0] == "systemd-run" for call in window_host.calls), (
        "no deadman is armed for a window that never opened"
    )
    assert window_host.emitted == []


# --------------------------------------------------------------------------- #
# the probe's answer contract
# --------------------------------------------------------------------------- #
def test_a_refused_command_answers_one_json_error_and_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["collector_window_probe.py", "close-window", "--run-id", "../x"]
    )

    assert probe.main() == 1

    lines = capsys.readouterr().out.splitlines()
    assert lines == ['{"command":"close-window","error":"ProbeError: unsafe run ID"}']


def test_run_as_a_script_the_probe_exits_with_its_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "close-window", "--run-id", "../x"])

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(SCRIPT), run_name="__main__")

    assert stopped.value.code == 1
    answer = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert answer == {"command": "close-window", "error": "ProbeError: unsafe run ID"}
