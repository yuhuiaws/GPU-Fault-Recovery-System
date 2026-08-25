from __future__ import annotations

import io
import json
import tempfile
from email.message import Message
from urllib.error import HTTPError

import gpu_fault.collectors as collectors
from gpu_fault.collectors import CollectorError, HttpEventSink


def main() -> None:
    original = collectors.urlopen
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

        collectors.urlopen = probe
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
        assert [item["sequence"] for item in calls] == [1, 1, 2]
        assert open(outbox).read() == ""

        def permanent(*_args, **_kwargs):
            raise HTTPError(
                "https://control/events",
                400,
                "bad",
                Message(),
                io.BytesIO(b"invalid schema"),
            )

        collectors.urlopen = permanent
        try:
            sink.post("/events", {"sequence": 3})
        except CollectorError:
            pass
        dead = json.loads(open(outbox).read())
        assert dead["replayable"] is False
        print(
            "PASS",
            {
                "replay_order": [1, 1, 2],
                "dead_letter_replayable": dead["replayable"],
                "dead_letter_error": dead["error"],
            },
        )
    collectors.urlopen = original


if __name__ == "__main__":
    main()
