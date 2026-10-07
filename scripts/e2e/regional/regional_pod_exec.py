"""One exec of a probe script into a named Pod, and its JSON answer.

Split out of ``RegionalLiveFixture.pod_python`` so the fixture can try the
component's Pods in turn: an exec kubectl refuses before the script runs (the
Pod is restarting on a node the case just handed over, its container is gone)
moves to the next Pod; a timeout is never retried by anyone, because the first
attempt may have run to completion with only the receipt lost.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

from scripts.e2e.regional.regional_commands import RegionalFixtureError


def exec_pod_script(
    fixture: Any,
    plane: str,
    pod: str,
    script: str,
    arguments: Sequence[str],
    *,
    interpreter: str,
    timeout: int,
) -> dict[str, Any]:
    """Run ``script`` under ``interpreter`` in ``pod``; the JSON object it printed.

    Command failures and timeouts propagate as raised (``RegionalCommandFailed``,
    ``RegionalCommandTimeout``); the caller decides what each means.
    """

    output = fixture.kubectl(
        plane,
        "exec",
        "-i",
        pod,
        "--",
        interpreter,
        "-",
        *arguments,
        input_text=script,
        timeout=timeout,
    )
    try:
        value = json.loads(output.splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise RegionalFixtureError(
            f"Pod probe did not return JSON: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise RegionalFixtureError("Pod probe did not return a JSON object")
    return cast(dict[str, Any], value)
