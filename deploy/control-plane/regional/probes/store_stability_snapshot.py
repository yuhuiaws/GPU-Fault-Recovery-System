"""One sample of the Store-side stability signals.

Request: nothing on stdin.
Response: ``{"queue": {...}, "remote_commands": {...}}`` on stdout.

``default=float`` is load-bearing: ``processor_queue_stats`` returns Decimal ages
from PostgreSQL, which ``json`` cannot encode, and a raise here reads as an
unstable release rather than as a serialization bug.
"""

import json

from gpu_fault.app import ApplicationContext


def main() -> None:
    store = ApplicationContext.from_environment().store
    print(
        json.dumps(
            {
                "queue": store.processor_queue_stats(),
                "remote_commands": store.remote_command_stats(),
            },
            default=float,
            sort_keys=True,
        )
    )


main()
