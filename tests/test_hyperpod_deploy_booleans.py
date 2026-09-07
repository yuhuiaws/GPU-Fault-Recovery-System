"""The legacy deploy script reads its boolean switches the way the Python side
does: ``1/true/yes/on`` and ``0/false/no/off``, any case, nothing else.

``tr '[:upper:]' '[:lower:]'`` accepted any spelling and a later ``== "true"``
quietly read every other one -- ``TRUE ``, ``enabled``, a typo -- as *false*, so
a misspelled ``GPU_FAULT_ENABLE_*`` disabled the feature without a word. The
helper is run under bash here rather than grepped for: a substring check would
stay green after the helper stopped rejecting anything.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy/hyperpod/deploy.sh"


def _helper() -> str:
    script = DEPLOY.read_text(encoding="utf-8")
    start = script.index("normalize_bool() {")
    end = script.index("\n}\n", start) + 3
    return script[start:end]


def _normalize(value: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            _helper() + 'normalize_bool GPU_FAULT_ENABLE_X "$1"',
            "_",
            value,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", "ON"])
def test_every_truthy_token_normalizes_to_true(value: str) -> None:
    completed = _normalize(value)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "true"


@pytest.mark.parametrize("value", ["0", "false", "False", "NO", "off", "Off"])
def test_every_falsy_token_normalizes_to_false(value: str) -> None:
    completed = _normalize(value)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "false"


@pytest.mark.parametrize("value", ["", "enabled", "t", "y", "2", "true ", "ture"])
def test_anything_else_is_refused_by_name_instead_of_read_as_false(value: str) -> None:
    completed = _normalize(value)

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "GPU_FAULT_ENABLE_X" in completed.stderr
    assert "1/true/yes/on" in completed.stderr and "0/false/no/off" in completed.stderr


def test_the_switches_the_manual_documents_all_go_through_the_helper() -> None:
    """Every ``GPU_FAULT_*`` switch the script compares against ``"true"`` is
    normalized once, at assignment, so no raw spelling reaches a comparison."""

    script = DEPLOY.read_text(encoding="utf-8")

    for name in (
        "GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR",
        "GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR",
        "GPU_FAULT_ENABLE_KUBERNETES_HMA_COLLECTOR",
        "GPU_FAULT_ENABLE_NVIDIA_SMI_METRICS_COLLECTOR",
        "GPU_FAULT_ENABLE_FIRMWARE_UPDATE",
        "GPU_FAULT_ENABLE_FIELD_DIAGNOSTIC",
        "GPU_FAULT_ALLOW_EMAIL",
        "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL",
        "GPU_FAULT_PROCESSOR_EXIT_ON_DEADLINE",
        "GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT",
        "GPU_FAULT_NOTIFICATION_DELIVER_BACKLOG",
        "GPU_FAULT_ALLOW_HYPERPOD_MUTATION",
        "GPU_FAULT_ALLOW_HYPERPOD_REBOOT",
        "GPU_FAULT_ALLOW_HYPERPOD_REPLACE",
    ):
        assert f"normalize_bool {name} " in script, name
    # The one remaining case-fold is the DCGM exporter mode, an enum with its
    # own membership check; no boolean is folded any more.
    assert script.count("tr '[:upper:]' '[:lower:]'") == 1
