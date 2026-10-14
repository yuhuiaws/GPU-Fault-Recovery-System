"""``spec.workflow.departedNodeCleanup``: the departed-node cleanup's switch.

On by default and omitted from records while at its default, so every live
``desired.json`` keeps its digest; rendered to
``GPU_FAULT_DEPARTED_NODE_CLEANUP_ENABLED`` in every role's core ConfigMap
(the worker runs the cleanup, the ingress answers ``status`` from the same
value) and read back from the worker's core ConfigMap when a release
captures the live config.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.config import (
    AdminConfig,
    WorkflowConfig,
    admin_config_spec,
    default_admin_config,
)
from gpu_fault.admin.config_patch import PRESET_NAMES, apply_patch, preset_admin_config
from gpu_fault.departed_node_cleanup import cleanup_enabled
from gpu_fault_release import regional_release_state as state
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_rendering import (
    admin_config_renderer_environment,
)
from tests.regional._cov95_release_support import ResourceRelease, deployment

ROOT = Path(__file__).resolve().parents[2]
VARIABLE = "GPU_FAULT_DEPARTED_NODE_CLEANUP_ENABLED"
# The one field no ConfigMap holds; the live release state records it.
RECORDED = {"aurora": {"min_acu": 8.0, "max_acu": 32.0}}


def test_the_default_is_on_and_stays_out_of_records_yaml_and_digests() -> None:
    assert WorkflowConfig().departed_node_cleanup is True
    default = default_admin_config()
    assert "departed_node_cleanup" not in default.as_dict()["workflow"]
    assert "departedNodeCleanup" not in admin_config_spec(default)["workflow"]
    for name in PRESET_NAMES:
        preset = preset_admin_config(name)
        assert preset.workflow.departed_node_cleanup is True, name
        assert "departed_node_cleanup" not in preset.as_dict()["workflow"], name
    assert AdminConfig.from_mapping(default.as_dict()) == default


def test_switching_it_off_round_trips_and_moves_every_role_digest() -> None:
    config = apply_patch(
        default_admin_config(), {"workflow": {"departedNodeCleanup": False}}
    )

    assert config.workflow.departed_node_cleanup is False
    assert config.as_dict()["workflow"]["departed_node_cleanup"] is False
    assert admin_config_spec(config)["workflow"]["departedNodeCleanup"] is False
    assert AdminConfig.from_mapping(config.as_dict()) == config
    before = default_admin_config().role_sha256()
    after = config.role_sha256()
    assert all(before[role] != after[role] for role in before), (
        "the workflow section is in every role digest"
    )


def test_a_non_boolean_value_is_refused() -> None:
    from gpu_fault.admin.config import AdminConfigError

    with pytest.raises(AdminConfigError):
        apply_patch(default_admin_config(), {"workflow": {"departedNodeCleanup": "no"}})


def test_the_release_renderer_environment_carries_the_switch() -> None:
    assert admin_config_renderer_environment(default_admin_config())[VARIABLE] == (
        "true"
    )
    off = apply_patch(
        default_admin_config(), {"workflow": {"departedNodeCleanup": False}}
    )
    assert admin_config_renderer_environment(off)[VARIABLE] == "false"


def _render(environment: dict[str, str]) -> dict[str, Any]:
    from tests.regional.test_control_plane_role_split import _deployment

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "deploy/control-plane/tools/render_control_plane_role_split.py"),
            "--json",
        ],
        input=json.dumps(_deployment()),
        text=True,
        capture_output=True,
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            **environment,
        },
    )
    return {
        item["metadata"]["name"]: item for item in json.loads(result.stdout)["items"]
    }


@pytest.mark.parametrize(("value", "expected"), [(None, "true"), ("false", "false")])
def test_the_role_split_renders_it_into_every_roles_core_configmap(
    value: str | None, expected: str
) -> None:
    items = _render({} if value is None else {VARIABLE: value})

    for role in (
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
    ):
        assert items[f"{role}-config-core"]["data"][VARIABLE] == expected, role


def _release_with(worker_core: dict[str, str]) -> tuple[ResourceRelease, dict]:
    release = ResourceRelease()
    release.documents[("cpu", "deployment", "gpu-fault-control-worker")] = deployment(
        "worker", replicas=6
    )
    release.documents[("cpu", "deployment", "gpu-fault-telemetry-spool-worker")] = (
        deployment("spool", replicas=0)
    )
    return release, {"gpu-fault-control-worker-config-core": worker_core}


@pytest.mark.parametrize(
    ("live", "expected"),
    [({}, True), ({VARIABLE: "true"}, True), ({VARIABLE: "false"}, False)],
)
def test_a_release_captures_the_live_switch_from_the_worker(
    live: dict[str, str], expected: bool
) -> None:
    release, snapshots = _release_with(live)

    captured = state.captured_admin_config(release, snapshots, recorded=RECORDED)

    assert captured.workflow.departed_node_cleanup is expected


def test_an_unreadable_live_switch_is_refused() -> None:
    release, snapshots = _release_with({VARIABLE: "maybe"})

    with pytest.raises(ReleaseError, match="cannot capture a valid live"):
        state.captured_admin_config(release, snapshots, recorded=RECORDED)


def test_the_runtime_reads_the_rendered_value() -> None:
    assert cleanup_enabled({VARIABLE: "true"}) is True
    assert cleanup_enabled({VARIABLE: "false"}) is False
    assert cleanup_enabled({}) is True
