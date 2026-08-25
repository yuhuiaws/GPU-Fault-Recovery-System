from __future__ import annotations

import base64
import gzip
import json
import logging
import os
import re
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
)

LOGGER = logging.getLogger(__name__)

HMA_KEYS = {
    HMA_HEALTH_STATUS,
    HMA_FAULT_TYPES,
    HMA_FAULT_REASONS,
    HMA_FAULT_DETAILS,
}


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
            self.sink.post(
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
                    raise CollectorError("invalid queued HMA event")
                self.sink.post(path, payload)
                self.client.delete_message(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=message["ReceiptHandle"],
                )
                delivered += 1
            except Exception:
                LOGGER.exception("queued HMA event delivery failed")
        return delivered

    def run(self) -> None:
        while True:
            self.run_once()
