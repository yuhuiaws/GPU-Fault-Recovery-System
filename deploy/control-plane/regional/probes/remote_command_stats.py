"""Report the Store's remote command counters.

Request: nothing on stdin.
Response: the ``remote_command_stats()`` mapping, compactly on stdout.

Trivial by design. It exists as a probe rather than an HTTP call because the
release engine needs the counters from the Store the *candidate* Pod is bound
to, which is the one thing an endpoint read cannot guarantee mid-rollout.
"""

import json

from gpu_fault.app import ApplicationContext


def main() -> None:
    print(
        json.dumps(
            ApplicationContext.from_environment().store.remote_command_stats(),
            separators=(",", ":"),
        )
    )


main()
