from __future__ import annotations

import json
from typing import Any, cast

import pytest

from scripts.e2e.regional import run_destr002_hyperpod_reboot as reboot
from scripts.e2e.regional import run_destr012_managed_recovery_guard as recovery
from scripts.e2e.regional import run_destr014_branch_exhaustion as branches
from scripts.e2e.regional import run_destr018_lifetime_deadline as lifetime
from scripts.e2e.regional import run_destr022_spare_reservation_reclaim as reclaim
from scripts.e2e.regional.regional_live_fixture import component_python
from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture


class PodTransport:
    def __init__(self) -> None:
        self.commands: list[tuple[str, tuple[str, ...]]] = []

    def ready_pods(self, _plane: str, _app: str) -> list[dict[str, str]]:
        return [{"name": "pod-a", "uid": "uid-a"}]

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        self.commands.append((plane, args))
        if args[:2] == ("get", "pod"):
            return json.dumps({"metadata": {"name": "pod-a", "uid": "uid-a"}})
        assert args[0] == "exec", args
        assert args[args.index("--") + 1] == component_python(plane), args
        return json.dumps({"metrics": "sample 1", "duplicate": True})


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_managed_recovery_probe_uses_the_selected_component(plane: str) -> None:
    transport = PodTransport()

    recovery.pod_python(cast(Any, transport), plane, "pod-a", "import gpu_fault")

    assert len(transport.commands) == 1
    assert transport.commands[0][0] == plane


@pytest.mark.parametrize(
    "reader",
    [
        "executor_environment",
        "synthetic_replacement_gates",
        "executor_snapshot",
        "worker_metrics",
        "reclaim_probe",
        "replacement_replay",
    ],
)
def test_owned_pod_probes_never_fall_back_to_system_or_node_python(reader: str) -> None:
    transport = PodTransport()
    regional = cast(Any, transport)
    warm = WarmSpareLiveFixture(regional, "")
    if reader == "executor_environment":
        warm.executor_environment()
    elif reader == "synthetic_replacement_gates":
        warm.synthetic_replacement_gates()
    elif reader == "executor_snapshot":
        branches.executor_env_snapshot(regional)
    elif reader == "worker_metrics":
        lifetime.worker_metrics(regional)
    elif reader == "reclaim_probe":
        reclaim.executor_probes(regional)
    else:
        reboot.replay_from_replacement(
            regional,
            {},
            {
                "before": [{"name": "old", "uid": "old-uid"}],
                "after": [{"name": "pod-a", "uid": "uid-a"}],
            },
        )
    executed = [
        (plane, args) for plane, args in transport.commands if args[0] == "exec"
    ]
    assert len(executed) == 1
    expected_plane = (
        "cpu" if reader in {"synthetic_replacement_gates", "worker_metrics"} else "gpu"
    )
    assert executed[0][0] == expected_plane
