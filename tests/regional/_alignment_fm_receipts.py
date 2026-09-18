from __future__ import annotations

from typing import Any

from scripts.e2e.regional.collector_delivery_evidence import identity_digest

OLD = "1" * 32
OLD_UNIT = "2" * 32
NEW = "3" * 32
NEW_UNIT = "4" * 32
RECORD_ID = "fm-file-" + "a" * 64
SCOPE = {"cluster_id": "cluster-a", "node": "node-a", "boot_id": "boot-a"}


def receipt(
    seq: int,
    round_seq: int,
    stage: str,
    outcome: str,
    *,
    attempt: int = 0,
    delivered: int = 0,
    rounds: int = 1,
    new: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "producer_invocation_id": NEW if new else OLD,
        "systemd_invocation_id": NEW_UNIT if new else OLD_UNIT,
        "pid": 234 if new else 123,
        "receipt_seq": seq,
        "round_seq": round_seq,
        "attempt_seq": attempt,
        "stage": stage,
        "outcome": outcome,
        "source": None if stage == "ROUND" else "file",
        "record_id_sha256": None if stage == "ROUND" else identity_digest(RECORD_ID),
        "cluster_id_sha256": identity_digest(SCOPE["cluster_id"]),
        "node_id_sha256": identity_digest(SCOPE["node"]),
        "boot_id_sha256": identity_digest(SCOPE["boot_id"]),
        "source_config_sha256": identity_digest("fixture-source"),
        "counter_exhausted": False,
        "attempts_total": attempt,
        "delivered_total": delivered,
        "buffered_total": 0,
        "failed_total": 0,
        "rounds_completed_total": rounds,
        "rounds_failed_total": 0,
        "omitted_receipts_total": 0,
    }


def window(*, restarted: bool = False) -> dict[str, Any]:
    bodies = [
        receipt(1, 1, "ROUND", "COMPLETE"),
        receipt(2, 2, "ATTEMPT", "STARTED", attempt=1),
        receipt(3, 2, "COMPLETION", "DELIVERED", attempt=1, delivered=1),
        receipt(4, 2, "ROUND", "COMPLETE", attempt=1, delivered=1, rounds=2),
    ]
    if restarted:
        bodies.extend(
            [
                receipt(1, 1, "ROUND", "COMPLETE", new=True),
                receipt(2, 2, "ROUND", "COMPLETE", rounds=2, new=True),
            ]
        )
    return {
        "service": {
            "ActiveState": "active",
            "MainPID": "234" if restarted else "123",
            "InvocationID": NEW_UNIT if restarted else OLD_UNIT,
        },
        "anchor_cursor": "s=fixture;i=1",
        "complete": True,
        "captured_monotonic_us": 100_000,
        "records": [
            {
                "cursor": f"s=fixture;i={index}",
                "monotonic_us": index * 1000,
                "receipt": body,
            }
            for index, body in enumerate(bodies, start=1)
        ],
    }


def baseline() -> dict[str, Any]:
    result = window()
    result["anchor_cursor"] = None
    result["records"] = result["records"][:1]
    return result


def replay_window() -> dict[str, Any]:
    result = window(restarted=True)
    bodies = [
        receipt(3, 3, "ATTEMPT", "STARTED", attempt=1, rounds=2, new=True),
        receipt(
            4, 3, "COMPLETION", "DELIVERED", attempt=1, delivered=1, rounds=2, new=True
        ),
        receipt(5, 3, "ROUND", "COMPLETE", attempt=1, delivered=1, rounds=3, new=True),
    ]
    result["records"].extend(
        {
            "cursor": f"s=fixture;i={index}",
            "monotonic_us": index * 1000,
            "receipt": body,
        }
        for index, body in enumerate(bodies, start=7)
    )
    return result


class FmProducerFixture:
    def __init__(self) -> None:
        from types import SimpleNamespace

        self.node = "node-a"
        self.regional = SimpleNamespace(
            settings=SimpleNamespace(cluster_id="cluster-a")
        )
        self.calls: list[str] = []
        self.appended = False
        self.restarted = False
        self.problem: str | None = None
        self.delivery_reads = 0

    def snapshot(self) -> dict[str, Any]:
        size = 20 if self.appended else 10
        return {
            "boot_id": "boot-a",
            "gpu_inventory": [{"pci_bdf": "0000:01:00.0"}],
            "service": {"ActiveState": "active"},
            "fabric_manager_log": {"path": "private.log", "size": size, "inode": 1},
            "fabric_manager_cursor": {
                "files": {"private.log": {"offset": size, "inode": 1}}
            },
        }

    def execute(self, command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(command)
        if command == "fm-delivery-evidence":
            self.delivery_reads += 1
            if not self.appended:
                return baseline()
            if self.problem == "replay" and self.restarted:
                return replay_window()
            return window(restarted=self.restarted)
        if command == "append-sxid":
            self.appended = True
            return {}
        if command == "restart-service":
            self.restarted = True
            return {
                "before": {"InvocationID": OLD_UNIT},
                "after": {"InvocationID": NEW_UNIT},
            }
        assert command == "fm-cursor"
        value = self.snapshot()
        if self.problem == "cursor-timeout":
            value["service"]["ActiveState"] = "inactive"
        return value

    def wait_marker(self, marker: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "evidence": [{"payload": {"record_id": RECORD_ID}}]
            * (2 if self.problem == "initial-count" else 1)
        }

    def store_snapshot(self, marker: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "evidence": [{"payload": {"record_id": RECORD_ID}}],
            "workflows": [{}] if self.problem == "workflow" else [],
        }
