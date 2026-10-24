"""The reviewed nvidia-smi inventory wrapper: local fake binaries only."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from gpu_fault.collectors.gpu.discovery import query_gpu_inventory
from tools import nvidia_smi_inventory_replay as wrapper

REAL = (
    "0, GPU-11111111-1111-1111-1111-111111111111, 00000000:18:00.0, NVIDIA H200\n"
    "1, GPU-22222222-2222-2222-2222-222222222222, 00000000:2A:00.0, NVIDIA H200\n"
    "2, GPU-33333333-3333-3333-3333-333333333333, 00000000:3A:00.0, NVIDIA H200\n"
)
REPLAY_UUID = "GPU-abcdefab-0000-4000-8000-000000000026"


def product_inventory(text: str):
    """The deployed parser's reading of a served inventory."""

    def runner(_command, **_options):
        return subprocess.CompletedProcess(_command, 0, stdout=text, stderr="")

    return query_gpu_inventory(runner)


def plan(*operations, not_after=4_000_000_000.0):
    return wrapper.validate_plan(
        {"operations": list(operations), "not_after": not_after}
    )


def test_replace_uuid_changes_one_row_and_the_product_parses_it():
    served, unchanged = wrapper.rewrite(
        REAL, plan({"op": "replace_uuid", "index": 1, "uuid": REPLAY_UUID}), now=1.0
    )
    assert unchanged == [], unchanged
    devices = product_inventory(served)
    assert [item.gpu_uuid for item in devices] == [
        "GPU-11111111-1111-1111-1111-111111111111",
        REPLAY_UUID,
        "GPU-33333333-3333-3333-3333-333333333333",
    ]
    assert [item.pci_bdf for item in devices] == [
        item.pci_bdf for item in product_inventory(REAL)
    ], "the bus ids and the count are untouched"


def test_swap_bdf_keeps_uuids_and_swaps_the_two_bus_ids():
    served, unchanged = wrapper.rewrite(
        REAL, plan({"op": "swap_bdf", "indexes": [0, 2]}), now=1.0
    )
    assert unchanged == []
    before = {item.gpu_uuid: item.pci_bdf for item in product_inventory(REAL)}
    after = {item.gpu_uuid: item.pci_bdf for item in product_inventory(served)}
    assert set(before) == set(after), "the UUID set is the same"
    first, _, third = sorted(before)
    assert after[first] == before[third] and after[third] == before[first]


@pytest.mark.parametrize(
    ("operation", "reason"),
    [
        ({"op": "replace_uuid", "index": 7, "uuid": REPLAY_UUID}, "not in the real"),
        (
            {
                "op": "replace_uuid",
                "index": 0,
                "uuid": "GPU-22222222-2222-2222-2222-222222222222",
            },
            "names a real GPU",
        ),
        ({"op": "swap_bdf", "indexes": [0, 9]}, "not both present"),
    ],
)
def test_anything_not_exactly_applicable_passes_the_real_output(operation, reason):
    served, unchanged = wrapper.rewrite(REAL, plan(operation), now=1.0)
    assert served == REAL and reason in unchanged[0], unchanged


def test_a_past_deadline_and_a_malformed_row_pass_through():
    late = plan({"op": "swap_bdf", "indexes": [0, 1]}, not_after=10.0)
    assert wrapper.rewrite(REAL, late, now=11.0) == (REAL, ["past the plan deadline"])
    broken = "0, GPU-x, 00000000:18:00.0\n"
    assert (
        wrapper.rewrite(broken, plan({"op": "swap_bdf", "indexes": [0, 1]}), now=1.0)[0]
        == broken
    )


@pytest.mark.parametrize(
    "value",
    [
        {"operations": [], "not_after": 5},
        {"operations": [{"op": "drop", "index": 1}], "not_after": 5},
        {
            "operations": [{"op": "replace_uuid", "index": 1, "uuid": "GPU-1"}],
            "not_after": 5,
        },
        {
            "operations": [{"op": "replace_uuid", "index": True, "uuid": REPLAY_UUID}],
            "not_after": 5,
        },
        {"operations": [{"op": "swap_bdf", "indexes": [1, 1]}], "not_after": 5},
        {
            "operations": [
                {"op": "swap_bdf", "indexes": [0, 1]},
                {"op": "replace_uuid", "index": 1, "uuid": REPLAY_UUID},
            ],
            "not_after": 5,
        },
        {"operations": [{"op": "swap_bdf", "indexes": [0, 1]}], "not_after": 0},
        {"operations": [{"op": "swap_bdf", "indexes": [0, 1]}]},
    ],
)
def test_the_plan_refuses_unreviewed_rewrites(value):
    with pytest.raises(ValueError):
        wrapper.validate_plan(value)


def _fake_binary(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "nvidia-smi"
    path.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _scripted_real(monkeypatch, *, stdout: str, returncode: int = 0) -> list[list[str]]:
    """The real binary, scripted: the wrapper's subprocess call is recorded."""

    calls: list[list[str]] = []

    def run(command, **_options):
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout=stdout,
            stderr="NVML failed" if returncode else "",
        )

    monkeypatch.setattr(wrapper.subprocess, "run", run)
    return calls


def test_main_rewrites_only_the_inventory_query_and_journals_it(
    tmp_path, monkeypatch, capsys
):
    real = _fake_binary(tmp_path / "usr-bin", "pass")
    calls = _scripted_real(monkeypatch, stdout=REAL)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "operations": [{"op": "replace_uuid", "index": 2, "uuid": REPLAY_UUID}],
                "not_after": 4_000_000_000,
            }
        )
    )
    audit = tmp_path / "inventory.jsonl"
    monkeypatch.setenv("PATH", f"{real.parent}{os.pathsep}/nonexistent")
    monkeypatch.setenv(wrapper.PLAN_ENV, str(plan_path))
    monkeypatch.setenv(wrapper.AUDIT_ENV, str(audit))
    assert wrapper.main(list(wrapper.INVENTORY_ARGV)) == 0
    assert calls == [[str(real), *wrapper.INVENTORY_ARGV]], calls
    served = capsys.readouterr().out
    assert REPLAY_UUID in served and "33333333-3333" not in served, served
    record = json.loads(audit.read_text().splitlines()[-1])
    assert record["rewritten"] is True and record["proof_scope"] == "live-replay"
    executed = []
    monkeypatch.setattr(wrapper.os, "execv", lambda path, args: executed.append(args))
    wrapper.main(["--query-gpu=index,uuid,name", "--format=csv,noheader"])
    assert executed and executed[0][1:] == [
        "--query-gpu=index,uuid,name",
        "--format=csv,noheader",
    ], "every other query goes to the real binary unchanged"


def test_main_without_a_plan_or_with_a_failing_query_passes_through(
    tmp_path, monkeypatch, capsys
):
    real = _fake_binary(tmp_path / "bin", "pass")
    monkeypatch.setenv("PATH", str(real.parent))
    monkeypatch.delenv(wrapper.PLAN_ENV, raising=False)
    _scripted_real(monkeypatch, stdout=REAL)
    assert wrapper.main(list(wrapper.INVENTORY_ARGV)) == 0
    assert capsys.readouterr().out == REAL
    _scripted_real(monkeypatch, stdout="", returncode=9)
    assert wrapper.main(list(wrapper.INVENTORY_ARGV)) == 9, "the real failure is kept"
    assert wrapper.main.__module__ == wrapper.__name__


def test_main_without_a_real_binary_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert wrapper.main(list(wrapper.INVENTORY_ARGV)) == 127
    assert "no real nvidia-smi" in capsys.readouterr().err


def test_the_replay_directory_is_never_taken_for_the_real_binary(tmp_path, monkeypatch):
    shim = tmp_path / "shim"
    _fake_binary(shim, "pass")
    monkeypatch.setattr(wrapper, "REPLAY_ROOT", str(tmp_path / "shim"))
    real_dir = tmp_path / "real"
    real = _fake_binary(real_dir, "pass")
    assert wrapper.real_binary(f"{shim}{os.pathsep}{real_dir}") == str(real)
    assert wrapper.real_binary(str(shim)) is None
