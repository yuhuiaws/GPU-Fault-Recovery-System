"""The final ownership boundary's in-window path is bounded under holder churn.

Live (gpu-eks-cluster, p5en, DESTR-001/015): the pre-permit reads were
outside the window as designed, yet the window still expired -- a host sampler
spawned a fresh short-lived ``nvidia-smi`` every few seconds (each holding the
device for seconds with persistence mode off), the first in-window holder sample
saw it, the second sample and the classification followed, and the window
re-resolved the device-node map with ``nvidia-smi -q -x``: 5-8 s inside a 3-s
permit, OWNERSHIP_PERMIT_EXPIRED on every retry. Inside the window the boundary
now spawns nothing (one device-map window spans the permit), samples once and
ignores a query-only holder without a second sample, samples once more only for
a holder that is not a query, and renews the permit a single time when the
window still ran out -- re-validating only the fast local authority before the
physical call.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from gpu_fault.node_agent.late_ownership import (
    OwnershipRefused,
    command_identity,
    execute_with_final_ownership,
    final_ownership_boundary,
    require_physical_ownership,
    sign_permit,
)
from gpu_fault.node_agent.protocol import NodeActionCommand, SignedNodeAction
from tests.regional.test_late_ownership_agent_gate import KEY, NOW
from tests.regional.test_late_ownership_agent_gate import challenge as challenge_fixture
from tests.regional.test_late_ownership_agent_gate import command as command_fixture

QUERY = {
    "gpu_uuid": "GPU-a",
    "pid": "4242",
    "device": "/dev/nvidia0",
    "process_name": "nvidia-smi",
}
WORKER = {
    "gpu_uuid": "GPU-a",
    "pid": "5151",
    "device": "/dev/nvidia0",
    "process_name": "python",
}


def _process(proc: Any, pid: str, comm: str, *targets: str) -> None:
    process = proc / pid
    (process / "fd").mkdir(parents=True)
    (process / "comm").write_text(comm + "\n", encoding="ascii")
    for index, target in enumerate(targets):
        (process / "fd" / str(index)).symlink_to(target)


class WindowAgent:
    """A fake Agent whose in-window holder samples and clock are scripted.

    ``samples`` is consumed one holder list per ``device_client_finder`` call
    made after the permit; the pre-permit classification (two samples) always
    sees nothing. ``advance`` moves the clock by that many seconds on every
    in-window sample, so a slow sample can outlive the 3-s permit.
    """

    def __init__(
        self, proc: Any, *, samples: list[list[dict[str, str]]], advance: float = 0.0
    ) -> None:
        self.proc_root = proc
        self.calls: list[str] = []
        self.now_value = NOW
        self.now = lambda: self.now_value
        self.sleep = lambda _seconds: None
        self.ownership_gate = SimpleNamespace(require=self.grant)
        self.ledger = SimpleNamespace(accept_fencing=lambda *args: True)
        self.service_quiesce_enabled = False
        self.quiesce_manager: Any = None
        self.permitted = False
        self.grants = 0
        self.samples = list(samples)
        self.advance = advance
        self.in_window_samples = 0

    def grant(self, value: NodeActionCommand) -> Any:
        self.calls.append("permit")
        self.grants += 1
        self.permitted = True
        return sign_permit(
            challenge_fixture(
                command_sha256=command_identity(value), nonce=f"{self.grants:064x}"
            ),
            KEY,
            allowed=True,
            reason="OK",
            now=self.now_value,
        )

    def validate_submission(self, envelope: SignedNodeAction) -> None:
        self.calls.append("validate")

    def _verify_no_clients(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append("compute")

    @contextmanager
    def _device_path_cache_window(self) -> Iterator[None]:
        self.calls.append("cache")
        yield

    def _require_resolvable_targets(self, targets: list[str]) -> None:
        self.calls.append("targets")

    def device_client_finder(self, targets: set[str]) -> list[dict[str, str]]:
        self.calls.append("devices")
        if not self.permitted:
            return []
        self.in_window_samples += 1
        self.now_value = self.now_value + timedelta(seconds=self.advance)
        return list(self.samples.pop(0)) if self.samples else []

    @final_ownership_boundary
    def effect(self, value: NodeActionCommand) -> dict[str, Any]:
        require_physical_ownership()
        self.calls.append("effect")
        return {"effect": True}


def _signed() -> SignedNodeAction:
    return SignedNodeAction(command=command_fixture(), signature="unused")


def test_a_query_only_holder_in_the_window_costs_one_sample_and_nothing_spawned(
    tmp_path: Any,
) -> None:
    proc = tmp_path / "proc"
    _process(proc, "4242", "nvidia-smi", "/dev/nvidiactl", "/dev/nvidia0")
    agent = WindowAgent(proc, samples=[[QUERY]])

    details = execute_with_final_ownership(agent, _signed(), agent.effect)

    assert details["effect"] is True
    assert agent.in_window_samples == 1, "a query is ignored without a second sample"
    assert agent.calls.count("cache") == 1 and agent.calls.index(
        "cache"
    ) < agent.calls.index("permit"), (
        "one device-map window, opened before the permit: nothing resolves it in the window"
    )
    (receipt,) = details["physical_ownership_checks"]
    assert receipt["in_window_holder_samples"] == 1
    assert receipt["permit_renewals"] == 0
    assert receipt["query_only_device_clients"] == [QUERY]
    assert receipt["query_only_device_client_count"] == 1


def test_a_transient_worker_in_the_window_is_sampled_once_more_then_let_through(
    tmp_path: Any,
) -> None:
    proc = tmp_path / "proc"
    _process(
        proc, "5151", "python", "/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm"
    )
    agent = WindowAgent(proc, samples=[[WORKER], []])

    details = execute_with_final_ownership(agent, _signed(), agent.effect)

    assert details["effect"] is True
    assert agent.in_window_samples == 2
    assert details["physical_ownership_checks"][0]["in_window_holder_samples"] == 2


def test_a_persistent_worker_in_the_window_still_refuses(tmp_path: Any) -> None:
    proc = tmp_path / "proc"
    _process(
        proc, "5151", "python", "/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm"
    )
    agent = WindowAgent(proc, samples=[[WORKER, QUERY], [WORKER]])

    with pytest.raises(OwnershipRefused) as refused:
        execute_with_final_ownership(agent, _signed(), agent.effect)

    assert "effect" not in agent.calls
    details = refused.value.action_details
    assert details["reason"] == "OWNERSHIP_FINAL_CLIENTS_CHANGED"
    assert details["persistent_device_clients"][0]["pid"] == "5151"
    assert details["node_action_outcome_known"] is True
    assert agent.in_window_samples == 2


def test_a_window_the_sample_outlived_is_renewed_once_and_the_call_follows(
    tmp_path: Any,
) -> None:
    """The in-window sample took longer than the permit (a slow /proc scan on a
    busy node): the boundary asks for one fresh permit, re-validates the fast
    local authority and proceeds; the receipt is the renewed permit's."""

    proc = tmp_path / "proc"
    _process(proc, "4242", "nvidia-smi", "/dev/nvidiactl", "/dev/nvidia0")
    agent = WindowAgent(proc, samples=[[QUERY]], advance=3.5)

    details = execute_with_final_ownership(agent, _signed(), agent.effect)

    assert details["effect"] is True
    assert agent.grants == 2
    (receipt,) = details["physical_ownership_checks"]
    assert receipt["permit_renewals"] == 1
    assert receipt["challenge_nonce"] == f"{2:064x}", "the renewed permit's receipt"
    assert {"renewal_wait_seconds", "post_renewal_seconds"} <= set(receipt)
    assert agent.calls[-3:] == ["permit", "validate", "effect"], (
        "after the renewal only the fast local authority is re-validated"
    )


def test_the_renewal_is_single_shot(tmp_path: Any) -> None:
    proc = tmp_path / "proc"
    agent = WindowAgent(proc, samples=[[]], advance=3.5)
    # The renewed permit is signed at the advanced clock; a validation that
    # advances the clock once more spends it as well.
    original_validate = agent.validate_submission

    def slow_validate(envelope: SignedNodeAction) -> None:
        original_validate(envelope)
        if agent.grants == 2:
            agent.now_value = agent.now_value + timedelta(seconds=3.5)

    agent.validate_submission = slow_validate  # type: ignore[method-assign]

    with pytest.raises(OwnershipRefused) as refused:
        execute_with_final_ownership(agent, _signed(), agent.effect)

    assert agent.grants == 2, "no third permit is asked for"
    assert "effect" not in agent.calls
    timing = refused.value.action_details["ownership_permit_timing"]
    assert timing["permit_renewals"] == 1
    assert refused.value.action_details["reason"] == "OWNERSHIP_PERMIT_EXPIRED"


def test_a_clock_that_went_backwards_is_drift_not_a_spent_window(tmp_path: Any) -> None:
    proc = tmp_path / "proc"
    agent = WindowAgent(proc, samples=[[]], advance=-1.0)

    with pytest.raises(OwnershipRefused) as refused:
        execute_with_final_ownership(agent, _signed(), agent.effect)

    assert agent.grants == 1, "a backwards clock is not renewed over"
    assert refused.value.action_details["reason"] == "OWNERSHIP_PERMIT_EXPIRED"
    assert (
        refused.value.action_details["ownership_permit_timing"]["permit_renewals"] == 0
    )
