"""Compressed lease time for the CAP004 proofs against the in-process API.

The proofs drive the production ``ClusterActionExecutor`` against a local
``create_app`` API. The lease a *claim* grants (10 s in the proof) is
compressed so that 25 actions outlive their original lease within seconds,
and the renewer's tick (``lease_seconds / 3``) is compressed by the same
factor so renewals keep their production shape against the shortened claim.
Two things in the proofs must expire: the queued second command's un-renewed
claim, which the competitor reclaims, and the injected lost lease, which the
executor recovers once its last renewal lapses.

A successful *renewal* grants the production lease. Compressing it too left
the whole interpreter -- renewer threads, the API event loop and its store
I/O thread share one GIL -- two thirds of the compressed lease (1.3 s at
scale 0.2) to land every renewal of a running batch. One full garbage
collection of a long-lived xdist worker's heap takes about a second under
gate load (1.07 s measured at 620 MB), the renewals queued behind it miss the
rest, every lease of the batch expires at once, the next claim returns
recovery claims and the proof's exact-claim ledger fails. The only wait on a
renewed lease is the injected loss's recovery, which now takes the production
lease rather than the compressed one.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import pytest

RENEWER_TICK_EVENT = "gpu_fault.cluster_executor.lease.Event"


class LeaseStore(Protocol):
    def renew_remote_command_lease(
        self,
        cluster_id: str,
        command_id: str,
        executor_id: str,
        lease_token: str,
        *,
        lease_seconds: int,
    ) -> Any: ...


def compress_claim_leases(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scale: float,
    store_class: type[LeaseStore],
    deadline_modules: tuple[str, ...],
) -> None:
    """Compress claim deadlines and renewer ticks by ``scale``, renewals by 1."""

    class ScaledTick(threading.Event):
        def wait(self, timeout: float | None = None) -> bool:
            return super().wait(None if timeout is None else timeout * scale)

    def lease_deadline(seconds: int) -> datetime:
        return datetime.now(timezone.utc) + timedelta(seconds=seconds * scale)

    renew = store_class.renew_remote_command_lease

    def renew_at_production_length(
        store: LeaseStore,
        cluster_id: str,
        command_id: str,
        executor_id: str,
        lease_token: str,
        *,
        lease_seconds: int,
    ) -> Any:
        # ``lease_deadline`` compresses whatever it is handed, so the renewal
        # asks for the production length pre-expanded.
        return renew(
            store,
            cluster_id,
            command_id,
            executor_id,
            lease_token,
            lease_seconds=round(lease_seconds / scale),
        )

    monkeypatch.setattr(RENEWER_TICK_EVENT, ScaledTick)
    for module in deadline_modules:
        monkeypatch.setattr(f"{module}.lease_deadline", lease_deadline)
    monkeypatch.setattr(
        store_class, "renew_remote_command_lease", renew_at_production_length
    )
