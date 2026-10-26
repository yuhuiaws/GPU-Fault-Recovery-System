"""Opt-in loopback telemetry replay for an isolated GPU acceptance node.

No GPU, Kubernetes or collector configuration is changed by this tool.
Do not point a production collector at this test endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

from prometheus_client.exposition import generate_latest
from prometheus_client.parser import text_string_to_metric_families

from gpu_fault.collectors.gpu.dcgm import DCGM_METRICS, DCGM_NANOSECOND_DURATION_FIELDS
from gpu_fault.gpu_environment_policy import valid_clock_mask

MAX_BYTES = 2 * 1024 * 1024
ACKNOWLEDGEMENT = "ISOLATED_GPU_TEST_NODE"
ALLOWED_FIELDS = frozenset(
    {
        "gpu_temperature_c",
        "memory_temperature_c",
        "gpu_tlimit_c",
        "clock_throttle_reasons",
        "power_violation_total_us",
        "thermal_violation_total_us",
        "power_usage_w",
        "power_limit_w",
        "gpu_utilization_percent",
    }
)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("upstream redirects are forbidden")


def open_upstream(url: str, *, timeout: int):
    return build_opener(ProxyHandler({}), NoRedirect()).open(url, timeout=timeout)


def validate_plan(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"steps"}:
        raise ValueError("plan must contain only steps")
    steps = value["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 100:
        raise ValueError("plan requires 1..100 steps")
    total = 0
    for step in steps:
        if not isinstance(step, dict) or set(step) != {"reads", "values"}:
            raise ValueError("each step requires reads and values")
        reads, values = step["reads"], step["values"]
        if type(reads) is not int or not 1 <= reads <= 100:
            raise ValueError("reads must be an integer in 1..100")
        total += reads
        if not isinstance(values, dict) or not values.keys() <= ALLOWED_FIELDS:
            raise ValueError(
                "unsupported field; hardware limit metadata cannot be changed"
            )
        for number in values.values():
            if number is not None and (
                type(number) not in (int, float) or not math.isfinite(number)
            ):
                raise ValueError("values must be finite numbers or null (omit field)")
    if total > 300:
        raise ValueError("plan exceeds 300 collector reads")
    return steps


def validate_upstream(url: str, listen_port: int) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/metrics"
        or parsed.query
        or parsed.fragment
        or parsed.port is None
        or parsed.port == listen_port
    ):
        raise ValueError("upstream must be http://127.0.0.1:<different-port>/metrics")
    return url


def canonical_mapping(sample_name: str) -> tuple[str, str | None] | None:
    """The product's ``(canonical, unit)`` for one parsed exporter sample.

    ``prometheus_client``'s text parser renames the samples of a family the
    exporter declares ``counter`` to ``<name>_total`` (``DCGM_FI_DEV_THERMAL_
    VIOLATION`` and ``DCGM_FI_DEV_POWER_VIOLATION`` are counters in
    ``deploy/dataplane/dcgm-counters.csv``), so the bare name misses them. The
    collector's own parser (``collectors/gpu/dcgm.parse_dcgm_samples``) falls
    back to the bare name; this tool must see the same fields it does, or it
    refuses a plan the collector would have accepted (live 2026-10-26,
    GF-ENV-LIVE-006 ``hw_thermal``: every armed read answered 503 "requested
    fields are absent" and the collector consumed 0 of 7 reads).
    """

    mapping = DCGM_METRICS.get(sample_name)
    if mapping is None and sample_name.endswith("_total"):
        mapping = DCGM_METRICS.get(sample_name[:-6])
    return mapping


def transform(text: str, gpu_uuid: str, values: dict[str, float | None]) -> bytes:
    families = list(text_string_to_metric_families(text))
    found = set()
    targets = 0
    for family in families:
        replaced = []
        for sample in family.samples:
            mapping = canonical_mapping(sample.name)
            target = (
                sample.labels.get("UUID") or sample.labels.get("uuid")
            ) == gpu_uuid
            targets += int(target)
            if not target or mapping is None or mapping[0] not in values:
                replaced.append(sample)
                continue
            canonical = mapping[0]
            found.add(canonical)
            value = values[canonical]
            if value is None:
                continue
            # Plans use GFRS canonical units; the exporter wire values use ns.
            if canonical in DCGM_NANOSECOND_DURATION_FIELDS:
                value *= 1000
            replaced.append(sample._replace(value=value))
        family.samples = replaced
    if not targets or found != set(values):
        raise ValueError(
            "target GPU or requested fields are absent; refusing invented samples"
        )

    class Registry:
        def collect(self):
            return families

    return generate_latest(Registry())


def unsafe_upstream(body: bytes, gpu_stop: float, memory_stop: float) -> str | None:
    for family in text_string_to_metric_families(body.decode("utf-8")):
        for sample in family.samples:
            mapping = canonical_mapping(sample.name)
            if mapping is None:
                continue
            name, _ = mapping
            if name not in {
                "gpu_temperature_c",
                "memory_temperature_c",
                "gpu_tlimit_c",
                "clock_throttle_reasons",
            }:
                continue
            if (
                not math.isfinite(sample.value)
                or abs(sample.value) >= 2**52
                or name != "gpu_tlimit_c"
                and sample.value < 0
                or name == "clock_throttle_reasons"
                and not valid_clock_mask(float(sample.value))
            ):
                return f"unusable live safety reading: {sample.name}"
            if (
                name == "gpu_temperature_c"
                and sample.value >= gpu_stop
                or name == "memory_temperature_c"
                and sample.value >= memory_stop
                or name == "gpu_tlimit_c"
                and sample.value <= 0
                or name == "clock_throttle_reasons"
                and int(sample.value) & 0xC0
            ):
                return f"live safety stop: {sample.name}"
    return None


class Replay:
    def __init__(
        self,
        gpu_uuid: str,
        steps: list[dict[str, Any]],
        seconds: int,
        gpu_stop: float = 80,
        memory_stop: float = 85,
    ):
        self.gpu_uuid, self.steps = gpu_uuid, steps
        self.seconds = seconds
        self.deadline = 0.0
        self.start_read: int | None = None
        self.reads = 0
        self.gpu_stop, self.memory_stop = gpu_stop, memory_stop
        self.aborted: str | None = None
        #: ``GET /metrics`` requests answered 503 (upstream or replay failure);
        #: they consume no read, so a collector that keeps scraping a proxy that
        #: keeps refusing shows up here, not as a frozen ``reads`` count only.
        self.failed_reads = 0
        self.last_error: str | None = None

    def values(self) -> dict[str, float | None]:
        if self.start_read is None or self.aborted or time.monotonic() >= self.deadline:
            return {}
        remaining = self.reads - self.start_read
        for step in self.steps:
            if remaining < step["reads"]:
                return dict(step["values"])
            remaining -= step["reads"]
        return {}

    def arm(self) -> None:
        if self.start_read is not None or self.aborted:
            raise ValueError("replay already armed or aborted; start a new audited run")
        self.start_read = self.reads
        self.deadline = time.monotonic() + self.seconds

    def render(self, body: bytes) -> tuple[bytes, dict[str, Any]]:
        if len(body) > MAX_BYTES:
            raise ValueError("upstream response exceeds byte bound")
        self.aborted = self.aborted or unsafe_upstream(
            body, self.gpu_stop, self.memory_stop
        )
        values = self.values()
        output = (
            transform(body.decode("utf-8"), self.gpu_uuid, values) if values else body
        )
        receipt = {
            "read": self.reads + 1,
            "gpu_uuid": self.gpu_uuid,
            "proof_scope": "live-telemetry-replay",
            "values": values,
            "physical_fault_injected": False,
            "aborted": self.aborted,
            "upstream_sha256": hashlib.sha256(body).hexdigest(),
            "output_sha256": hashlib.sha256(output).hexdigest(),
        }
        return output, receipt


def make_server(upstream: str, port: int, replay: Replay, audit: Any) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def do_GET(self):
            if self.path == "/status":
                body = json.dumps(
                    {
                        "reads": replay.reads,
                        "next_values": replay.values(),
                        "proof_scope": "live-telemetry-replay",
                        "aborted": replay.aborted,
                        "armed": replay.start_read is not None,
                        "failed_reads": replay.failed_reads,
                        "last_error": replay.last_error,
                    }
                ).encode()
                self.respond(200, body, "application/json")
                return
            if self.path != "/metrics":
                self.respond(404, b"not found\n")
                return
            try:
                with open_upstream(upstream, timeout=5) as response:
                    original = response.read(MAX_BYTES + 1)
                output, receipt = replay.render(original)
            except (OSError, ValueError, UnicodeError) as error:
                replay.failed_reads += 1
                replay.last_error = str(error)
                audit.write(
                    json.dumps({"error": str(error), "read": replay.reads + 1}) + "\n"
                )
                audit.flush()
                self.respond(
                    503, b"upstream/replay unavailable; no healthy data fabricated\n"
                )
                return
            # Journal before publishing synthetic data. The real collector/API
            # receipts, not this prepared record, prove successful delivery.
            audit.write(
                json.dumps({**receipt, "delivery": "prepared"}, allow_nan=False) + "\n"
            )
            audit.flush()
            self.respond(200, output)
            replay.reads += 1

        def do_POST(self):
            if (
                self.path != "/arm"
                or self.headers.get("X-GFRS-Replay-Ack") != ACKNOWLEDGEMENT
            ):
                self.respond(403, b"explicit local replay acknowledgement required\n")
                return
            try:
                replay.arm()
            except ValueError as error:
                self.respond(409, str(error).encode())
                return
            audit.write(
                json.dumps({"event": "ARM", "start_read": replay.start_read}) + "\n"
            )
            audit.flush()
            self.respond(200, b"armed\n")

        def respond(
            self, status: int, body: bytes, content_type="text/plain; version=0.0.4"
        ):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return HTTPServer(("127.0.0.1", port), Handler)


def serve(upstream: str, port: int, replay: Replay, audit: Any) -> None:
    with make_server(upstream, port, replay, audit) as server:
        print(
            f"Replay endpoint http://127.0.0.1:{port}/metrics; /status does not consume steps",
            flush=True,
        )
        print(
            "Initially pass-through. POST /arm with X-GFRS-Replay-Ack to start the plan.",
            flush=True,
        )
        print(
            "After the plan or deadline, fresh upstream data passes through unchanged.",
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--upstream", default="http://127.0.0.1:9400/metrics")
    parser.add_argument("--port", type=int, default=9411)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--gpu-stop-c", type=float, default=80)
    parser.add_argument("--memory-stop-c", type=float, default=85)
    parser.add_argument(
        "--serve", action="store_true", help="default validates only; no network"
    )
    parser.add_argument("--ack", default="")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not 1 <= args.max_seconds <= 1800:
        parser.error("invalid port or max-seconds (1..1800)")
    if not args.gpu_uuid.startswith("GPU-"):
        parser.error("a physical GPU UUID is required")
    if not 0 < args.gpu_stop_c <= 80 or not 0 < args.memory_stop_c <= 85:
        parser.error("safety limits may only be lowered from 80 C GPU / 85 C memory")
    try:
        steps = validate_plan(json.loads(args.plan.read_text()))
        upstream = validate_upstream(args.upstream, args.port)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not args.serve:
        print(json.dumps({"validated": True, "network_started": False, "steps": steps}))
        return 0
    if args.ack != ACKNOWLEDGEMENT or args.audit is None:
        parser.error(
            "--serve requires --ack ISOLATED_GPU_TEST_NODE and a new --audit path"
        )
    # Exclusive creation prevents mixing evidence from separate acceptance runs.
    with args.audit.open("x", encoding="utf-8") as audit:
        serve(
            upstream,
            args.port,
            Replay(
                args.gpu_uuid,
                steps,
                args.max_seconds,
                args.gpu_stop_c,
                args.memory_stop_c,
            ),
            audit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
