from __future__ import annotations

import base64
import gzip
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable


from gpu_fault.hma import (
    HMA_FAULT_DETAILS,
    HMA_FAULT_REASONS,
    HMA_FAULT_TYPES,
    HMA_HEALTH_STATUS,
)


from gpu_fault.collectors.context import (
    context_from_environment,
    sink_from_environment,
)
from gpu_fault.collectors.models import CollectorContext, CollectorStats
from gpu_fault.collectors.sinks import (
    CollectorError,
    EventSink,
    SqsEventSink,
    deliver_event,
    is_retryable_collector_status,
)

LOGGER = logging.getLogger(__name__)

HMA_KEYS = {
    HMA_HEALTH_STATUS,
    HMA_FAULT_TYPES,
    HMA_FAULT_REASONS,
    HMA_FAULT_DETAILS,
}

#: How many failed receives make a queued message poison rather than unlucky.
#: There is no redrive policy on this queue, so a message nothing can parse was
#: retried every visibility timeout forever, logging a traceback each time.
POISON_RECEIVE_COUNT = 5


class InvalidQueuedHmaEvent(CollectorError):
    """A queued body that is not an HMA event: no redelivery can fix it.

    Its own class because the receive count alone must never authorise a drop:
    this failure is about the message, a transport failure is not, and both used
    to arrive as a bare :class:`CollectorError` with no status.
    """


def _is_poison_failure(exc: BaseException) -> bool:
    """Whether a sixth attempt at this message would fail the same way.

    Only the message can be poison. A transport failure carries no status code
    (or a retryable/auth one): the event behind it is real and the queue, whose
    retention is 14 days, is the only place it still exists.
    """

    if isinstance(exc, InvalidQueuedHmaEvent):
        return True
    if isinstance(exc, CollectorError):
        return exc.status_code is not None and not is_retryable_collector_status(
            exc.status_code
        )
    # ``json.JSONDecodeError`` is a ``ValueError``; a body without ``path`` or
    # ``payload`` is a ``KeyError``. Everything else -- ``OSError`` and friends
    # -- is the transport, and stays on the queue however often it is received.
    return isinstance(exc, (KeyError, ValueError, TypeError))


#: The receive backoff ladder: never a tight loop against a throttled queue,
#: never long enough to look like the consumer has stopped.
RECEIVE_BACKOFF_SECONDS = 1.0
MAX_RECEIVE_BACKOFF_SECONDS = 60.0


def cloudwatch_lambda_handler(
    event: dict[str, Any], _lambda_context: Any
) -> dict[str, int]:
    queue_url = os.getenv("GPU_FAULT_HMA_QUEUE_URL")
    collector = CloudWatchHmaCollector(
        (SqsEventSink(queue_url) if queue_url else sink_from_environment()),
        context_from_environment(),
        node_pattern=os.getenv("GPU_FAULT_HMA_NODE_REGEX"),
        node_prefix=os.getenv("GPU_FAULT_HMA_NODE_PREFIX", ""),
    )
    return collector.collect_subscription(event).model_dump()


class CloudWatchHmaCollector:
    def __init__(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        node_pattern: str | None = None,
        node_prefix: str = "",
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.sink = sink
        self.context = context
        self.node_pattern = re.compile(node_pattern) if node_pattern else None
        self.node_prefix = node_prefix
        self.now = now or (lambda: datetime.now(timezone.utc))

    def collect_subscription(self, event: dict[str, Any]) -> CollectorStats:
        payload = self.decode_subscription(event)
        if payload.get("messageType") == "CONTROL_MESSAGE":
            return CollectorStats(skipped=1)
        log_stream = str(payload.get("logStream") or "")
        node_id = self._node_id(log_stream)
        collected_at = self.now()
        stats = CollectorStats()
        for item in payload.get("logEvents") or []:
            stats = stats.model_copy(update={"observed": stats.observed + 1})
            message = str(item.get("message") or "")
            if "HealthMonitoringAgentDetectionEvent" not in message:
                stats = stats.model_copy(update={"skipped": stats.skipped + 1})
                continue
            event_id = str(item.get("id") or "")
            if not event_id:
                raise CollectorError("CloudWatch log event id is required")
            timestamp = datetime.fromtimestamp(
                int(item["timestamp"]) / 1000, tz=timezone.utc
            )
            result = deliver_event(
                self.sink,
                "/v1/provider-events/hyperpod-hma/cloudwatch",
                {
                    **self.context.model_dump(mode="json"),
                    "node_id": node_id,
                    "log_event_id": event_id,
                    "observed_at": timestamp.isoformat(),
                    "collected_at": collected_at.isoformat(),
                    "message": message,
                    "log_stream": log_stream,
                    "evidence_ref": (
                        f"cloudwatch://{payload.get('logGroup', '')}/"
                        f"{log_stream}/{event_id}"
                    ),
                },
            )
            # An event the outbox took is durable, so the rest of the
            # subscription batch is still forwarded; only an event that went
            # nowhere aborts the batch, which is what makes CloudWatch Logs
            # retry the whole delivery (ARCH-G3).
            result.raise_for_failure()
            if result.buffered:
                LOGGER.warning(
                    "HMA log event %s persisted to the collector outbox: %s",
                    event_id,
                    result.error,
                )
            stats = stats.model_copy(update={"delivered": stats.delivered + 1})
        return stats

    @staticmethod
    def decode_subscription(event: dict[str, Any]) -> dict[str, Any]:
        try:
            compressed = base64.b64decode(event["awslogs"]["data"])
            return json.loads(gzip.decompress(compressed))
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            raise CollectorError(
                "invalid CloudWatch Logs subscription envelope"
            ) from exc

    def _node_id(self, log_stream: str) -> str:
        if self.node_pattern:
            match = self.node_pattern.search(log_stream)
            if not match:
                raise CollectorError(
                    "CloudWatch log stream does not match GPU_FAULT_HMA_NODE_REGEX"
                )
            if "node_id" in match.groupdict():
                return self.node_prefix + match.group("node_id")
            return self.node_prefix + match.group(1)
        marker = "/SagemakerHealthMonitoringAgent"
        if marker in log_stream:
            prefix = log_stream.split(marker, 1)[0].rstrip("/")
            if prefix:
                return self.node_prefix + prefix.rsplit("/", 1)[-1]
        raise CollectorError(
            "cannot derive node ID from HMA log stream; configure "
            "GPU_FAULT_HMA_NODE_REGEX with a node_id capture group"
        )


class SqsHmaConsumer:
    """Long-polls normalized HMA events and forwards them to ClusterIP."""

    def __init__(
        self,
        sink: EventSink,
        queue_url: str,
        client: Any | None = None,
    ) -> None:
        self.sink = sink
        self.queue_url = queue_url
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise CollectorError(
                    "install gpu-fault-control-plane[hyperpod] for SQS support"
                ) from exc
            client = boto3.client("sqs")
        self.client = client

    def run_once(self, wait_time_seconds: int = 20) -> int:
        response = self.client.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=wait_time_seconds,
            VisibilityTimeout=60,
            MessageSystemAttributeNames=["ApproximateReceiveCount"],
        )
        delivered = 0
        for message in response.get("Messages", []):
            try:
                body = json.loads(message["Body"])
                path = str(body["path"])
                payload = body["payload"]
                if not path.startswith(
                    "/v1/provider-events/hyperpod-hma/"
                ) or not isinstance(payload, dict):
                    raise InvalidQueuedHmaEvent("invalid queued HMA event")
                result = deliver_event(self.sink, path, payload)
                result.raise_for_failure()
                if result.buffered:
                    # The queue is the stronger durability layer: it retains the
                    # message for 14 days, while the collector outbox is an
                    # emptyDir the optional HMA manifests do not even configure.
                    # So a buffered forward keeps the message and lets SQS
                    # redeliver it after the visibility timeout; the control
                    # plane dedupes by CloudWatch log event id, so the outbox
                    # replay and the redelivery collapse into one record.
                    LOGGER.warning(
                        "queued HMA event %s persisted to the collector outbox; "
                        "leaving it on the queue for redelivery: %s",
                        message.get("MessageId", "<unknown>"),
                        result.error,
                    )
                else:
                    self.client.delete_message(
                        QueueUrl=self.queue_url,
                        ReceiptHandle=message["ReceiptHandle"],
                    )
                delivered += 1
            except Exception as exc:
                receives = self._receive_count(message)
                if receives >= POISON_RECEIVE_COUNT and _is_poison_failure(exc):
                    # Two conditions, and the second is the important one. The
                    # visibility timeout is 60 s, so a five-minute control-plane
                    # outage takes every message past five receives: dropping on
                    # the count alone deleted the whole backlog mid-outage, and
                    # the optional consumer manifest configures no outbox, so a
                    # transport failure there is FAILED, not BUFFERED -- the
                    # queue's 14-day retention was the only copy. So a message
                    # leaves only when the failure is about the message itself
                    # and a sixth attempt would fail identically.
                    #
                    # Neither the body nor the rejection text is logged -- a 4xx
                    # detail quotes the payload back, and HMA fault text names
                    # nodes -- only the failure class, the status and the body
                    # digest, which is enough to recognise the same message
                    # twice. The earlier receives already logged the full reason.
                    LOGGER.warning(
                        "dropping queued HMA event %s after %d receives "
                        "(body sha256 %s): %s status=%s",
                        message.get("MessageId", "<unknown>"),
                        receives,
                        hashlib.sha256(
                            str(message.get("Body", "")).encode()
                        ).hexdigest(),
                        type(exc).__name__,
                        getattr(exc, "status_code", None),
                    )
                    self.client.delete_message(
                        QueueUrl=self.queue_url,
                        ReceiptHandle=message["ReceiptHandle"],
                    )
                else:
                    LOGGER.warning(
                        "queued HMA event %s delivery failed on receive %d; "
                        "leaving it on the queue for redelivery",
                        message.get("MessageId", "<unknown>"),
                        receives,
                        exc_info=True,
                    )
        return delivered

    @staticmethod
    def _receive_count(message: dict[str, Any]) -> int:
        """How many times SQS has handed this message out, 0 if it did not say."""

        raw = (message.get("Attributes") or {}).get("ApproximateReceiveCount")
        try:
            return int(str(raw))
        except (TypeError, ValueError):
            return 0

    def run(self) -> None:
        backoff = RECEIVE_BACKOFF_SECONDS
        while True:
            try:
                self.run_once()
            except Exception:
                # A throttle or an expired IRSA token used to end the process:
                # Kubernetes restarted the Pod straight into the same error, so
                # the queue went unattended for the whole outage and the HMA
                # events aged out of its 14-day retention.
                LOGGER.exception(
                    "HMA queue receive failed; retrying in %.0f s", backoff
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_RECEIVE_BACKOFF_SECONDS)
            else:
                backoff = RECEIVE_BACKOFF_SECONDS
