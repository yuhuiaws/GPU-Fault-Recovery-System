"""Compile a runtime profile and report what the Store already holds.

Request: the runtime profile document on stdin.
Response: one JSON object on stdout, ``{"desired": ..., "existing": ... | null}``.

``existing`` is null when the Store has never seen this ``profile_version``,
which is how the engine tells "register it" from "compare it".
"""

import json
import sys
from typing import Any

from gpu_fault.app import ApplicationContext
from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.models import RuntimeProfile
from gpu_fault.store.shared.errors import NotFoundError


def main() -> None:
    payload = json.load(sys.stdin)
    desired = compile_runtime_profile(RuntimeProfile.model_validate(payload))
    existing: Any = None
    try:
        existing = ApplicationContext.from_environment().store.get_profile(
            desired.profile_version
        )
    except NotFoundError:
        existing = None
    print(
        json.dumps(
            {
                "desired": desired.model_dump(mode="json"),
                "existing": (
                    existing.model_dump(mode="json") if existing is not None else None
                ),
            },
            separators=(",", ":"),
        )
    )


main()
