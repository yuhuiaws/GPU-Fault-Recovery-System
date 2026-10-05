"""The Aurora binding counts the live replica set, not Pods on their way out."""

from __future__ import annotations

from scripts.e2e.regional.aurora_binding import live_pod_items


def _pod(name: str, *, terminating: bool = False, phase: str = "Running") -> dict:
    metadata = {"name": name, "uid": f"uid-{name}"}
    if terminating:
        metadata["deletionTimestamp"] = "2026-10-05T00:38:30Z"
    return {"metadata": metadata, "status": {"phase": phase}}


def test_terminating_and_terminal_pods_are_not_part_of_the_live_inventory() -> None:
    """Live 2026-10-05 (HA-003 on the mixed site): an admin config apply rolled
    gpu-fault-api-ha while the binding read it; the old Pod was Terminating next
    to its replacement and the count said "inventory is incomplete"."""

    items = [
        _pod("api-a"),
        _pod("api-b"),
        _pod("api-c"),
        _pod("api-old", terminating=True),
        _pod("api-evicted", phase="Failed"),
        "not-a-pod",
    ]
    assert [item["metadata"]["name"] for item in live_pod_items(items)] == [
        "api-a",
        "api-b",
        "api-c",
    ]
    assert live_pod_items([_pod("api-a")]) == [_pod("api-a")], (
        "a steady set is returned unchanged, so fewer live Pods still fail closed"
    )
