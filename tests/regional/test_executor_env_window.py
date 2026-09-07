"""executor_env_window: the lease/poll variables HA-004 needs are on the allow-list."""

from __future__ import annotations

import pytest

from scripts.e2e.regional import executor_env_window as env_window


def test_lease_and_poll_variables_are_allowed_and_validated() -> None:
    assignments = env_window.parse_assignments(
        [
            "GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS=10",
            "GPU_FAULT_CLUSTER_EXECUTOR_POLL_SECONDS=2",
        ]
    )
    assert assignments == {
        "GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS": "10",
        "GPU_FAULT_CLUSTER_EXECUTOR_POLL_SECONDS": "2",
    }
    with pytest.raises(env_window.RegionalFixtureError):
        env_window.parse_assignments(["GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS=0"])
    with pytest.raises(env_window.RegionalFixtureError):
        env_window.parse_assignments(["GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS=ten"])


def test_allow_list_still_refuses_anything_else() -> None:
    with pytest.raises(env_window.RegionalFixtureError, match="allow-list"):
        env_window.parse_assignments(["GPU_FAULT_ALLOW_HYPERPOD_REBOOT=1"])


def test_restore_arguments_cover_the_lease_and_poll_baseline() -> None:
    baseline = {
        "variables": {
            "GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS": {
                "present": True,
                "value": "120",
            },
            "GPU_FAULT_CLUSTER_EXECUTOR_POLL_SECONDS": {
                "present": False,
                "value": None,
            },
        }
    }
    assert env_window.restore_arguments(baseline) == [
        "GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS=120",
        "GPU_FAULT_CLUSTER_EXECUTOR_POLL_SECONDS-",
    ]
