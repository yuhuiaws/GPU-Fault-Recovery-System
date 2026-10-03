"""Durable idempotency for provider node-lifecycle submissions.

A reboot submitted to the provider restarts the very Pod that submitted it,
and a control-plane failover moves the retry to another replica, so "did this
step already submit?" cannot be answered from process memory. Both lifecycle
providers (HyperPod ``BatchRebootClusterNodes`` and EC2 ``RebootInstances``)
share this ledger: the record is written INTENDED before the provider call
and updated to SUBMITTED, or to UNKNOWN when the call raised after possibly
taking effect, so a crash in between is recoverable as "unknown outcome" and
never silently retried. Without a store it degrades to the process-local
dictionaries, which is only safe for tests and one-shot CLI use.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpu_fault.hyperpod import (
        HyperPodAction,
        HyperPodSubmissionRecord,
        HyperPodSubmissionResult,
    )

LOGGER = logging.getLogger(__name__)


class ProviderSubmissionLedger:
    """One provider's submission records, durable when a store is supplied."""

    def __init__(
        self,
        *,
        cluster_name: str,
        store: Any | None,
        error: type[ValueError],
        provider_label: str,
        record_type: type[HyperPodSubmissionRecord],
    ) -> None:
        # The record class is injected so this module imports nothing from the
        # provider modules that import it.
        self.cluster_name = cluster_name
        self.store = store
        self.error = error
        self.provider_label = provider_label
        self.record_type = record_type
        self.submissions: dict[str, HyperPodSubmissionResult] = {}
        self.submission_requests: dict[str, tuple[HyperPodAction, tuple[str, ...]]] = {}

    @staticmethod
    def request_identity(
        action: HyperPodAction, node_identifiers: list[str]
    ) -> tuple[HyperPodAction, tuple[str, ...]]:
        return action, tuple(sorted(node_identifiers))

    def replay(
        self,
        action: HyperPodAction,
        node_identifiers: list[str],
        idempotency_key: str,
    ) -> HyperPodSubmissionResult | None:
        """The recorded outcome for an already-used key, or None if unused."""

        if self.store is not None:
            try:
                record = self.store.get_hyperpod_submission(
                    self.cluster_name, idempotency_key
                )
            except (KeyError, LookupError):
                return None
            return self.durable_outcome(record, action, node_identifiers)
        existing = self.submissions.get(idempotency_key)
        if existing is None:
            return None
        if self.submission_requests[idempotency_key] != self.request_identity(
            action, node_identifiers
        ):
            raise self.error(
                "idempotency key was already used for a different "
                f"{self.provider_label} request"
            )
        return existing.model_copy(update={"duplicate": True})

    def durable_outcome(
        self,
        record: HyperPodSubmissionRecord,
        action: HyperPodAction,
        node_identifiers: list[str],
    ) -> HyperPodSubmissionResult | None:
        if record.request_identity != self.request_identity(action, node_identifiers):
            raise self.error(
                "idempotency key was already used for a different "
                f"{self.provider_label} request"
            )
        if record.state == "SUBMITTED" and record.result is not None:
            return record.result.model_copy(update={"duplicate": True})
        if record.state == "UNKNOWN":
            raise self.error(
                f"a previous {self.provider_label} submission for this idempotency "
                "key has an unknown outcome; an operator must confirm "
                "the node state before retrying: "
                + (record.error or "no error recorded")
            )
        return None

    def reserve(
        self,
        action: HyperPodAction,
        node_identifiers: list[str],
        idempotency_key: str,
    ) -> tuple[HyperPodSubmissionRecord | None, HyperPodSubmissionResult | None]:
        """Write the INTENDED record; ``(record, None)`` means this caller submits.

        A reservation another replica already holds is replayed instead:
        ``(None, outcome)`` when it recorded one, otherwise an error, because
        retrying here could submit a second batch for the same step.
        """

        if self.store is None:
            return None, None
        record, reserved = self.store.reserve_hyperpod_submission(
            self.record_type(
                cluster_name=self.cluster_name,
                idempotency_key=idempotency_key,
                action=action,
                requested_node_identifiers=list(node_identifiers),
            )
        )
        if reserved:
            return record, None
        duplicate = self.durable_outcome(record, action, node_identifiers)
        if duplicate is not None:
            return None, duplicate
        raise self.error(
            f"a {self.provider_label} submission for this idempotency "
            "key is already reserved and has no recorded "
            "outcome; it is either in flight on another "
            "replica or was interrupted, so retrying here "
            "could submit a second batch"
        )

    def remember(
        self,
        idempotency_key: str,
        result: HyperPodSubmissionResult,
        action: HyperPodAction,
        node_identifiers: list[str],
        record: HyperPodSubmissionRecord | None,
    ) -> None:
        self.submissions[idempotency_key] = result
        self.submission_requests[idempotency_key] = self.request_identity(
            action, node_identifiers
        )
        self.record_outcome(record, state="SUBMITTED", result=result)

    def record_outcome(
        self,
        record: HyperPodSubmissionRecord | None,
        *,
        state: str,
        result: HyperPodSubmissionResult | None = None,
        error: str | None = None,
    ) -> None:
        if self.store is None or record is None:
            return
        try:
            self.store.save_hyperpod_submission(
                record.model_copy(
                    update={
                        "state": state,
                        "result": result,
                        "error": error,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
            )
        except Exception:
            # The provider call already happened. Losing the outcome record
            # leaves the key reserved in the INTENDED state, which fails
            # closed on the next attempt rather than re-submitting, so log
            # and do not mask the real result.
            LOGGER.exception(
                "could not persist %s submission outcome: cluster=%s key=%s state=%s",
                self.provider_label,
                record.cluster_name,
                record.idempotency_key,
                state,
            )
