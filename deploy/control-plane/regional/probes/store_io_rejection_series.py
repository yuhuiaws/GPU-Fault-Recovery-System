"""Report whether a Pod's store-I/O rejection counter is split by reason and idle.

Parameters arrive as **one JSON object in argv**, not on stdin::

    python -c "<source>" '{"metric": "...", "port": 9105}'

That is the exception to the stdin convention in ``README.md``, and it is here
because the ``kubectl exec`` that runs this probe has no ``-i``. Switching to
stdin would mean adding ``-i`` to that exec, which changes how the exec handles
stdin for a check that only needs two scalars; ``python -c cmd arg`` puts ``arg``
in ``sys.argv[1]`` with nothing else to arrange.

Response: ``{"series_count": N, "all_labeled": bool, "all_zero": bool}``.

``all_labeled`` is the real subject. The counter is split by ``reason``
(capacity / deadline / backend_unavailable) and the Pod's /metrics sums the
four uvicorn workers' samples per reason, so every series must carry
``reason``; an unlabeled series is the pre-split shape, whose value the alert
rules can no longer classify. ``all_zero`` is the health part. Both are
``bool(series) and ...`` so that "no series at all" is false rather than
vacuously true -- a Pod not publishing the family must not read as ready.
"""

import json
import sys
from urllib.request import urlopen


def main() -> None:
    request = json.loads(sys.argv[1])
    metric = request["metric"]
    text = (
        urlopen(
            f"http://127.0.0.1:{int(request['port'])}/metrics",
            timeout=10,
        )
        .read()
        .decode()
    )
    series = []
    for line in text.splitlines():
        if not (line.startswith(metric + "{") or line.startswith(metric + " ")):
            continue
        name, raw = line.rsplit(None, 1)
        series.append(
            {
                "labeled": 'reason="' in name,
                "value": float(raw),
            }
        )
    print(
        json.dumps(
            {
                "series_count": len(series),
                "all_labeled": bool(series) and all(item["labeled"] for item in series),
                "all_zero": bool(series) and all(item["value"] == 0 for item in series),
            },
            sort_keys=True,
        )
    )


main()
