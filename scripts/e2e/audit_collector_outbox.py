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
        # The live event must not wait behind a potentially large replay
        # backlog. The first "1" is the failed live attempt, "2" is the
        # next live event, and the final "1" is the bounded replay.
        assert [item["sequence"] for item in calls] == [1, 2, 1]
        assert open(outbox).read() == ""

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
        print(
            "PASS",
            {
                "replay_order": [1, 2, 1],
                "dead_letter_replayable": dead["replayable"],
                "dead_letter_error": dead["error"],
            },
        )
    collector_sinks.urlopen = original


if __name__ == "__main__":
    main()
