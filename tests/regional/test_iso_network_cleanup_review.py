from __future__ import annotations

import subprocess
from argparse import Namespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import cluster_network_probe as probe


@pytest.mark.parametrize("defect", ["residual", "read", "foreign"])
def test_network_cleanup_keeps_the_restore_timer_when_removal_is_unproven(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "iso006-test"
    chain = probe.chain_name(run_id)
    calls = []
    rules = (
        f"-N {chain}\n"
        f"-A {chain} -m comment --comment {'other' if defect == 'foreign' else run_id} -j REJECT\n"
        f"-A OUTPUT -j {chain}\n-A FORWARD -j {chain}\n"
    )

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command == ["iptables", "-S"]:
            if defect == "read":
                raise probe.ProbeError("rules unavailable")
            return subprocess.CompletedProcess(command, 0, rules, "")
        return subprocess.CompletedProcess(command, 1, "", "mock removal failure")

    monkeypatch.setattr(probe, "run", run)
    with pytest.raises(probe.ProbeError):
        probe.unblock(Namespace(run_id=run_id))
    assert not any(command[:2] == ["systemctl", "stop"] for command in calls), (
        "the restore timer must stay armed while rule removal is unproven"
    )
    if defect in {"read", "foreign"}:
        assert not any(command[:2] == ["iptables", "-D"] for command in calls), (
            "unknown or foreign rules must not be deleted during cleanup"
        )


def test_block_refuses_an_existing_chain_without_touching_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    chain = probe.chain_name("iso006-test")
    monkeypatch.setattr(
        probe,
        "run",
        lambda command, **kwargs: calls.append(command)
        or subprocess.CompletedProcess(command, 0, f"-N {chain}\n", ""),
    )
    with pytest.raises(probe.ProbeError, match="already exists"):
        probe.block(
            Namespace(
                run_id="iso006-test",
                control_plane_cidr=["10.0.0.0/24"],
                restore_seconds=60,
            )
        )
    assert calls == [["iptables", "-S"]]
