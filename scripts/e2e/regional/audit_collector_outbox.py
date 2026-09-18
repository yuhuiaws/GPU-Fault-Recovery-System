from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import gpu_fault.collectors.sinks as collector_sinks
from gpu_fault.collectors import CollectorError, HttpEventSink

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic  # noqa: E402
from scripts.e2e.regional.acceptance_scope import current_acceptance_scope  # noqa: E402
from scripts.e2e.regional.ha_evidence import isolated_chain  # noqa: E402
from scripts.e2e.regional.regional_case_contract import case_evidence_path  # noqa: E402

CASE_ID = "GF-REGIONAL-NET-005"


class OutboxAuditFailure(RuntimeError):
    """One delivery-contract expectation the Collector sink did not meet."""


def expect(condition: bool, message: str, observed: Any = None) -> None:
    """A bare ``assert`` vanishes under ``python -O``; this does not."""

    if condition:
        return
    detail = message if observed is None else f"{message}: observed {observed!r}"
    raise OutboxAuditFailure(detail)


def audit_outbox() -> dict[str, Any]:
    original = collector_sinks.urlopen
    try:
        with tempfile.TemporaryDirectory() as directory:
            outbox = f"{directory}/events.ndjson"
            calls = []
            available = [False]

            class Response:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

                def read(self):
                    return b'{"accepted":true}'

            def probe(request, **_kwargs):
                calls.append(json.loads(request.data))
                if not available[0]:
                    raise OSError("network unavailable")
                return Response()

            collector_sinks.urlopen = probe
            sink = HttpEventSink(
                "https://control",
                max_attempts=1,
                outbox_path=outbox,
            )
            try:
                sink.post("/events", {"sequence": 1})
            except CollectorError:
                pass
            available[0] = True
            sink.post("/events", {"sequence": 2})
            replay_order = [item["sequence"] for item in calls]
            expect(
                replay_order == [1, 2, 1],
                "live post did not replay the buffered event after itself",
                replay_order,
            )
            expect(open(outbox).read() == "", "outbox was not drained after replay")

            available[0] = False
            background_path = f"{directory}/background.ndjson"
            background = HttpEventSink(
                "https://control",
                max_attempts=1,
                outbox_path=background_path,
                outbox_replay_batch_size=2,
                outbox_replay_background_interval_seconds=0,
            )
            for sequence in range(30, 37):
                try:
                    background.post("/events", {"sequence": sequence})
                except CollectorError:
                    pass
            calls.clear()
            available[0] = True
            background.post("/events", {"sequence": 99})
            expect(
                background.wait_for_outbox_replay(2),
                "background replay did not finish within two seconds",
            )
            background_replay_order = [item["sequence"] for item in calls]
            expect(
                background_replay_order == [99, *range(30, 37)],
                "background replay order is wrong",
                background_replay_order,
            )
            expect(
                open(background_path).read() == "",
                "background outbox was not drained",
            )

            def permanent(*_args, **_kwargs):
                raise HTTPError(
                    "https://control/events",
                    400,
                    "bad",
                    Message(),
                    io.BytesIO(b"invalid schema"),
                )

            collector_sinks.urlopen = permanent
            try:
                sink.post("/events", {"sequence": 3})
            except CollectorError:
                pass
            dead = json.loads(open(outbox).read())
            expect(
                dead["replayable"] is False,
                "a 400 was buffered as replayable",
                dead,
            )

            def unavailable(*_args, **_kwargs):
                raise OSError("network unavailable")

            collector_sinks.urlopen = unavailable
            bounded_path = f"{directory}/bounded.ndjson"
            bounded = HttpEventSink(
                "https://control",
                max_attempts=1,
                outbox_path=bounded_path,
                outbox_max_records=3,
            )
            for sequence in range(10, 15):
                try:
                    bounded.post("/events", {"sequence": sequence})
                except CollectorError:
                    pass
            bounded_records = [
                json.loads(line)
                for line in open(bounded_path).read().splitlines()
                if line
            ]
            bounded_sequences = [
                item["payload"]["sequence"] for item in bounded_records
            ]
            expect(
                bounded_sequences == [12, 13, 14],
                "bounded outbox did not keep the newest three records",
                bounded_sequences,
            )

            blocked_parent = f"{directory}/not-a-directory"
            with open(blocked_parent, "w") as handle:
                handle.write("file")
            unwritable = HttpEventSink(
                "https://control",
                max_attempts=1,
                outbox_path=f"{blocked_parent}/events.ndjson",
            )
            try:
                unwritable.post("/events", {"sequence": 20})
            except CollectorError as exc:
                expect(
                    exc.buffered is False,
                    "unwritable outbox claimed to have buffered the event",
                    exc.buffered,
                )
            else:
                raise OutboxAuditFailure(
                    "unwritable outbox unexpectedly buffered an event"
                )

            return {
                "replay_order": replay_order,
                "background_replay_order": background_replay_order,
                "background_drained_without_new_live_event": True,
                "bounded_sequences": bounded_sequences,
                "unwritable_buffered": False,
                "dead_letter_replayable": dead["replayable"],
                "dead_letter_error": dead["error"],
            }
    finally:
        collector_sinks.urlopen = original


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit local Collector outbox delivery contracts."
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--release-id", default="")
    parser.add_argument("--cluster-id", default="")
    arguments = parser.parse_args(argv or [])
    result: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "fault-acceptance",
        "case_id": CASE_ID,
        "verdict": "FAIL",
        "validation_scope": "isolated-source",
        "live_validation": False,
        "identity_source": "not supplied",
        **current_acceptance_scope().result_fields(),
        "formal_sequence_satisfied": False,
    }
    try:
        if arguments.run_dir is not None:
            result.update(current_acceptance_scope().result_fields())
            result.update(isolated_chain(arguments, CASE_ID))
        elif arguments.release_id or arguments.cluster_id:
            raise OutboxAuditFailure("enclosing acceptance identity requires --run-dir")
        result.update(audit_outbox())
        result["verdict"] = "PASS"
    except Exception as exc:
        result["verdict"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["status"] = "COMPLETED"
    result["executed_at"] = datetime.now(timezone.utc).isoformat()
    if arguments.run_dir is not None:
        write_json_atomic(case_evidence_path(arguments.run_dir, CASE_ID), result)
    print(result["verdict"], result)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
