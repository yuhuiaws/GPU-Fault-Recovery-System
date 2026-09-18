"""Prove a fresh, complete zero view of a Pod's Store-I/O rejection counters.

Parameters arrive as **one JSON object in argv**, not on stdin::

    python -c "<source>" '{"metric": "...", "port": 9105}'

That is the exception to the stdin convention in ``README.md``, and it is here
because the ``kubectl exec`` that runs this probe has no ``-i``. Switching to
stdin would mean adding ``-i`` to that exec, which changes how the exec handles
stdin for a check that only needs two scalars; ``python -c cmd arg`` puts ``arg``
in ``sys.argv[1]`` with nothing else to arrange.

Response fields: ``series_count``, ``process_count``, ``complete``,
``all_labeled`` and ``all_zero``. Only all three booleans together prove zero.
Completeness requires a non-degraded merger, 1..16 fresh publications and all
three reasons for each distinct process slot. Slots need not be contiguous.
Older merged counters cannot prove this view, even when their values are zero.
This probe only authorizes transient critical-alert settling, not rollback.
Read/parse failures return an incomplete report without echoing metric data.
"""

import json
import math
import re
import sys
from decimal import Decimal
from urllib.error import HTTPError
from urllib.request import urlopen


METRIC = "gpu_fault_store_io_rejections_total"
PROCESSES = "gpu_fault_metrics_aggregation_processes"
DEGRADED = "gpu_fault_metrics_aggregation_degraded"
REASONS = frozenset({"capacity", "deadline", "backend_unavailable"})
SLOTS = frozenset(str(slot) for slot in range(16))
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
NAME = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
SAMPLE = re.compile(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?[ \t]+([^ \t]+)[ \t]*")
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)[ \t]*=[ \t]*(?=")')
QUOTED_VALUE = re.compile(r'"(?:[^"\\\x00-\x1f]|\\[\\n"])*"')
DECODER = json.JSONDecoder()


def labels(raw: str | None) -> dict[str, str]:
    remaining = (raw or "").strip(" \t")
    result: dict[str, str] = {}
    while remaining:
        match = LABEL.match(remaining)
        if match is None:
            raise ValueError
        key = match.group(1)
        value, end = DECODER.raw_decode(remaining, match.end())
        if (
            not isinstance(value, str)
            or not QUOTED_VALUE.fullmatch(remaining[match.end() : end])
            or key in result
        ):
            raise ValueError
        result[key] = value
        remaining = remaining[end:].lstrip(" \t")
        if remaining:
            if not remaining.startswith(","):
                raise ValueError
            remaining = remaining[1:].lstrip(" \t")
            if not remaining:
                raise ValueError
    return result


def zero_view(text: str) -> dict[str, int | bool]:
    series: dict[tuple[str, str], Decimal] = {}
    metadata: dict[str, Decimal] = {}
    for raw_line in text.split("\n"):
        line = raw_line.strip(" \t\r")
        name = NAME.match(line)
        if name is None or name.group() not in {METRIC, PROCESSES, DEGRADED}:
            continue
        sample = SAMPLE.fullmatch(line)
        if sample is None or not NUMBER.fullmatch(sample.group(3)):
            raise ValueError
        sample_labels = labels(sample.group(2))
        # Preserve exact zero/integer tests even when a float would underflow
        # or round a fractional publication count to an integer.
        value = Decimal(sample.group(3))
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError
        if name.group() != METRIC:
            if sample_labels or name.group() in metadata:
                raise ValueError
            metadata[name.group()] = value
            continue
        if (
            sample_labels.keys() - {"process", "reason", "process_id"}
            or sample_labels.get("process") not in SLOTS
            or sample_labels.get("reason") not in REASONS
        ):
            raise ValueError
        key = sample_labels["process"], sample_labels["reason"]
        if key in series:
            raise ValueError
        series[key] = value

    count = metadata.get(PROCESSES, Decimal(0))
    if PROCESSES in metadata and (not 1 <= count <= len(SLOTS) or count != int(count)):
        raise ValueError
    slots = {slot for slot, _reason in series}
    complete = (
        metadata.get(DEGRADED) == 0
        and count > 0
        and len(slots) == count
        and len(series) == len(REASONS) * count
    )
    return {
        "series_count": len(series),
        "process_count": int(count),
        "complete": complete,
        "all_labeled": bool(series),
        "all_zero": bool(series) and all(value == 0 for value in series.values()),
    }


def metric_text(port: int) -> str:
    try:
        response = urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10)
    except HTTPError as error:
        error.close()
        raise
    with response:
        body: bytes = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError
    return body.decode("utf-8")


def main() -> None:
    report: dict[str, int | bool] = {
        "series_count": 0,
        "process_count": 0,
        "complete": False,
        "all_labeled": False,
        "all_zero": False,
    }
    try:
        if len(sys.argv) != 2:
            raise ValueError
        request = json.loads(sys.argv[1])
        port = request["port"]
        if (
            request["metric"] != METRIC
            or type(port) is not int
            or not 1 <= port <= 65535
        ):
            raise ValueError
        report = zero_view(metric_text(port))
    except Exception:
        # Transport and parser exceptions can contain response data.
        pass
    print(json.dumps(report, sort_keys=True))


main()
