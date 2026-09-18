"""Bounded producer evidence; receipts never include source or transport payloads."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Callable, Literal
from uuid import uuid4

FM_RECEIPT_PREFIX = "GPU_FAULT_FM_RECEIPT_V1 "
FM_RECEIPT_MAX_BYTES = 2048
FM_RECEIPT_PAIR_BUDGET = 256
FM_RECEIPT_ROUND_BUDGET = 256
FM_RECEIPT_BUDGET_SECONDS = 60
FM_RECEIPT_COUNTER_MAX = (1 << 63) - 1
FM_RECEIPT_COUNTERS = (
    "receipt_seq",
    "round_seq",
    "attempts_total",
    "delivered_total",
    "buffered_total",
    "failed_total",
    "rounds_completed_total",
    "rounds_failed_total",
    "omitted_receipts_total",
)
FM_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "producer_invocation_id",
        "systemd_invocation_id",
        "pid",
        "attempt_seq",
        "stage",
        "outcome",
        "source",
        "record_id_sha256",
        "cluster_id_sha256",
        "node_id_sha256",
        "boot_id_sha256",
        "source_config_sha256",
        "counter_exhausted",
        *FM_RECEIPT_COUNTERS,
    }
)

DeliveryOutcome = Literal["DELIVERED", "BUFFERED", "FAILED"]


def identity_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


@dataclass(frozen=True)
class RoundReceipt:
    invocation_id: str
    round_seq: int
    attempts_at_start: int


@dataclass(frozen=True)
class AttemptReceipt:
    invocation_id: str
    round_seq: int
    attempt_seq: int
    record_id_sha256: str
    source: str | None
    emit_pair: bool


class FabricManagerReceiptLog:
    """One serial producer invocation, bounded event logs and aggregate progress."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        cluster_id: str,
        node_id: str,
        boot_id: str,
        source_configuration: str,
        summary_seconds: float,
        monotonic: Callable[[], float],
    ) -> None:
        if not math.isfinite(summary_seconds) or summary_seconds <= 0:
            raise ValueError("FM receipt summary interval must be positive and finite")
        self.logger = logger
        self.monotonic = monotonic
        self.summary_seconds = max(30.0, summary_seconds)
        self.scope = {
            "cluster_id_sha256": identity_sha256(cluster_id),
            "node_id_sha256": identity_sha256(node_id),
            "boot_id_sha256": identity_sha256(boot_id),
            "source_config_sha256": identity_sha256(source_configuration),
        }
        self._reset_process()

    def _reset_process(self) -> None:
        self.pid = os.getpid()
        self.invocation_id = uuid4().hex
        configured = os.getenv("INVOCATION_ID", "")
        self.systemd_invocation_id = (
            configured if re.fullmatch(r"[0-9a-f]{32}", configured) else None
        )
        self.counts = dict.fromkeys(FM_RECEIPT_COUNTERS, 0)
        self.counter_exhausted = False
        self._budget_bucket: int | None = None
        self._pairs_in_bucket = 0
        self._rounds_in_bucket = 0
        self._next_checkpoint_at: float | None = None
        self._next_buffer_warning_at: float | None = None

    def _check_process(self) -> None:
        if self.pid != os.getpid():
            self._reset_process()

    def _advance(self, name: str) -> int:
        value = self.counts[name]
        if value >= FM_RECEIPT_COUNTER_MAX:
            self.counter_exhausted = True
        else:
            self.counts[name] = value + 1
        return self.counts[name]

    def _refresh_budget(self, now: float) -> None:
        bucket = int(now // FM_RECEIPT_BUDGET_SECONDS)
        if self._budget_bucket != bucket:
            self._budget_bucket = bucket
            self._pairs_in_bucket = self._rounds_in_bucket = 0

    def begin_round(self) -> RoundReceipt:
        self._check_process()
        return RoundReceipt(
            self.invocation_id,
            self._advance("round_seq"),
            self.counts["attempts_total"],
        )

    def begin_attempt(
        self, round_receipt: RoundReceipt, *, record_id: str, source: str | None
    ) -> AttemptReceipt:
        self._check_process()
        if round_receipt.invocation_id != self.invocation_id:
            self.counter_exhausted = True
        self._refresh_budget(self.monotonic())
        emit_pair = self._pairs_in_bucket < FM_RECEIPT_PAIR_BUDGET
        if emit_pair:
            self._pairs_in_bucket += 1
        attempt = AttemptReceipt(
            self.invocation_id,
            round_receipt.round_seq,
            self._advance("attempts_total"),
            identity_sha256(record_id),
            source if source in {"file", "journal"} else None,
            emit_pair,
        )
        self._emit(
            stage="ATTEMPT",
            outcome="STARTED",
            round_seq=attempt.round_seq,
            attempt_seq=attempt.attempt_seq,
            record_id_sha256=attempt.record_id_sha256,
            source=attempt.source,
            permitted=emit_pair,
        )
        return attempt

    def complete_attempt(
        self, attempt: AttemptReceipt, outcome: DeliveryOutcome
    ) -> None:
        self._check_process()
        if attempt.invocation_id != self.invocation_id:
            self.counter_exhausted = True
        counter = {
            "DELIVERED": "delivered_total",
            "BUFFERED": "buffered_total",
            "FAILED": "failed_total",
        }[outcome]
        self._advance(counter)
        now = self.monotonic()
        if outcome == "BUFFERED" and (
            self._next_buffer_warning_at is None or now >= self._next_buffer_warning_at
        ):
            self._next_buffer_warning_at = now + self.summary_seconds
            self.logger.warning(
                "Fabric Manager record persisted to the collector outbox; "
                "awaiting replay; consult producer receipt counters"
            )
        # Reserve completion with its attempt, even across a budget boundary.
        self._emit(
            stage="COMPLETION",
            outcome=outcome,
            round_seq=attempt.round_seq,
            attempt_seq=attempt.attempt_seq,
            record_id_sha256=attempt.record_id_sha256,
            source=attempt.source,
            permitted=attempt.emit_pair,
        )

    def complete_round(self, receipt: RoundReceipt, *, complete: bool) -> None:
        self._check_process()
        if receipt.invocation_id != self.invocation_id:
            self.counter_exhausted = True
            complete = False
        self._advance("rounds_completed_total" if complete else "rounds_failed_total")
        now = self.monotonic()
        periodic = (
            receipt.round_seq <= 2
            or self._next_checkpoint_at is None
            or now >= self._next_checkpoint_at
        )
        active = self.counts["attempts_total"] != receipt.attempts_at_start
        if not (periodic or active or not complete):
            return
        self._refresh_budget(now)
        permitted = periodic or self._rounds_in_bucket < FM_RECEIPT_ROUND_BUDGET
        if permitted:
            self._rounds_in_bucket += 1
            self._next_checkpoint_at = now + self.summary_seconds
        self._emit(
            stage="ROUND",
            outcome="COMPLETE" if complete else "FAILED",
            round_seq=receipt.round_seq,
            attempt_seq=self.counts["attempts_total"],
            record_id_sha256=None,
            source=None,
            permitted=permitted,
        )

    def _emit(
        self,
        *,
        stage: str,
        outcome: str,
        round_seq: int,
        attempt_seq: int,
        record_id_sha256: str | None,
        source: str | None,
        permitted: bool,
    ) -> None:
        self._advance("receipt_seq")
        if not permitted or not self.logger.isEnabledFor(logging.INFO):
            self._advance("omitted_receipts_total")
            return
        payload = {
            "schema_version": 1,
            "producer_invocation_id": self.invocation_id,
            "systemd_invocation_id": self.systemd_invocation_id,
            "pid": self.pid,
            **self.scope,
            **self.counts,
            "round_seq": round_seq,
            "attempt_seq": attempt_seq,
            "stage": stage,
            "outcome": outcome,
            "source": source,
            "record_id_sha256": record_id_sha256,
            "counter_exhausted": self.counter_exhausted,
        }
        message = FM_RECEIPT_PREFIX + json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        if len(message) > FM_RECEIPT_MAX_BYTES:
            self._advance("omitted_receipts_total")
            self.counter_exhausted = True
            return
        self.logger.info("%s", message)
