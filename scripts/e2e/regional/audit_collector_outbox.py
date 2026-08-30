from __future__ import annotations

import io
import json
import tempfile
from email.message import Message
from urllib.error import HTTPError

import gpu_fault.collectors.sinks as collector_sinks
from gpu_fault.collectors import CollectorError, HttpEventSink


def main() -> None:
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
            assert [item["sequence"] for item in calls] == [1, 2, 1]
            assert open(outbox).read() == ""

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
            assert background.wait_for_outbox_replay(2)
            background_replay_order = [item["sequence"] for item in calls]
            assert background_replay_order == [99, *range(30, 37)]
            assert open(background_path).read() == ""

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
            assert dead["replayable"] is False

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
            assert [item["payload"]["sequence"] for item in bounded_records] == [
                12,
                13,
                14,
            ]

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
                assert exc.buffered is False
            else:
                raise AssertionError("unwritable outbox unexpectedly buffered an event")

            print(
                "PASS",
                {
                    "replay_order": [1, 2, 1],
                    "background_replay_order": background_replay_order,
                    "background_drained_without_new_live_event": True,
                    "bounded_sequences": [12, 13, 14],
                    "unwritable_buffered": False,
                    "dead_letter_replayable": dead["replayable"],
                    "dead_letter_error": dead["error"],
                },
            )
    finally:
        collector_sinks.urlopen = original


if __name__ == "__main__":
    main()
