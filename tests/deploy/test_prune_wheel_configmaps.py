"""The wheel ConfigMap pruner judges a chunked set's parts with their named ConfigMap."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy/control-plane/tools/prune-wheel-configmaps.sh"

REFERENCED = "gpu-fault-control-plane-wheel-0100-aaaaaaaaaaaa"
STALE = "gpu-fault-control-plane-wheel-0100-bbbbbbbbbbbb"
ORPHAN_PART = "gpu-fault-control-plane-wheel-0100-cccccccccccc-part-0"
NEWEST = "gpu-fault-executor-wheel-0100-dddddddddddd"
HAND_PINNED = "gpu-fault-control-plane-wheel-reboot-email-reason-0723"

FAKE_KUBECTL = """#!{python}
import json, os, sys
from pathlib import Path
state = Path(os.environ["FAKE_STATE"])
args = sys.argv[1:]
with (state / "calls.jsonl").open("a") as handle:
    handle.write(json.dumps(args) + "\\n")
if "delete" in args:
    raise SystemExit(0)
if args[args.index("get") + 1] == "configmap":
    print((state / "configmaps.json").read_text())
else:
    print((state / "workloads.json").read_text())
"""


def _config_map(name: str, created: str) -> dict:
    return {"metadata": {"name": name, "creationTimestamp": created}}


def _run(
    tmp_path: Path, *, apply: bool
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    state = tmp_path / "api"
    state.mkdir()
    (state / "configmaps.json").write_text(
        json.dumps(
            {
                "items": [
                    _config_map(STALE, "2026-01-01T00:00:00Z"),
                    _config_map(STALE + "-part-0", "2026-01-01T00:00:01Z"),
                    _config_map(ORPHAN_PART, "2026-01-02T00:00:00Z"),
                    _config_map(REFERENCED, "2026-01-03T00:00:00Z"),
                    _config_map(REFERENCED + "-part-0", "2026-01-03T00:00:01Z"),
                    _config_map(REFERENCED + "-part-1", "2026-01-03T00:00:02Z"),
                    _config_map(NEWEST, "2026-01-04T00:00:00Z"),
                    _config_map(HAND_PINNED, "2025-07-23T00:00:00Z"),
                    _config_map("gpu-fault-release-metadata", "2026-01-01T00:00:00Z"),
                ]
            }
        )
    )
    (state / "workloads.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "kind": "Deployment",
                        "metadata": {"name": "gpu-fault-api-ha"},
                        "spec": {
                            "template": {
                                "spec": {
                                    "volumes": [
                                        {
                                            "name": "artifact",
                                            "configMap": {"name": REFERENCED},
                                        }
                                    ],
                                    "containers": [],
                                }
                            }
                        },
                    }
                ]
            }
        )
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    kubectl = binary / "kubectl"
    kubectl.write_text(FAKE_KUBECTL.format(python=sys.executable), encoding="utf-8")
    kubectl.chmod(kubectl.stat().st_mode | stat.S_IXUSR)
    environment = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "FAKE_STATE": str(state),
        "GPU_FAULT_PRUNE_KEEP_RECENT": "1",
        "GPU_FAULT_PRUNE_APPLY": "true" if apply else "false",
    }
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    calls = [
        json.loads(line) for line in (state / "calls.jsonl").read_text().splitlines()
    ]
    return completed, calls


def test_parts_follow_their_named_config_map_and_orphans_are_reclaimed(
    tmp_path: Path,
) -> None:
    completed, calls = _run(tmp_path, apply=True)
    assert completed.returncode == 0, completed.stderr
    deleted = [call[call.index("configmap") + 1] for call in calls if "delete" in call]
    assert sorted(deleted) == sorted([STALE, STALE + "-part-0", ORPHAN_PART]), (
        completed.stdout
    )
    plan = completed.stdout
    assert (
        f"KEEP   {REFERENCED} " in plan
        and "referenced by Deployment/gpu-fault-api-ha" in plan
    )
    assert f"KEEP   {REFERENCED}-part-0 " in plan and f"part of {REFERENCED}" in plan
    assert f"KEEP   {REFERENCED}-part-1 " in plan
    assert f"KEEP   {NEWEST} " in plan and "within newest 1" in plan
    assert HAND_PINNED not in plan.split("prune:")[1], (
        "hand-pinned wheels are never pruned"
    )
    assert "1 hand-pinned (never pruned)" in plan
    assert "4 parts" in plan, "every part of every set is counted"


def test_dry_run_plans_the_same_set_without_deleting(tmp_path: Path) -> None:
    completed, calls = _run(tmp_path, apply=False)
    assert completed.returncode == 0, completed.stderr
    assert not any("delete" in call for call in calls), "a dry run must not delete"
    assert "Dry run" in completed.stdout
    pruned = completed.stdout.split("prune: 3")[1]
    assert STALE + "-part-0" in pruned and ORPHAN_PART in pruned
