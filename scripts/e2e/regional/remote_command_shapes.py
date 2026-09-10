"""What a remote command covers, whether it carries one step or a batch.

Since the round-trip batch (protocol 3) a node command may carry the head
step in ``step`` and the steps that followed it on the same node in
``batched_steps`` -- DESTR-023's reset ran QUIESCE, VERIFY_NO_GPU_CLIENTS,
RESET_GPU and RESTORE_GPU_SERVICES as one command. Verdicts that read only
``command["step"]["operation"]`` saw three of six remote operations and failed
a run whose workflow had SUCCEEDED; read the batch too.
"""

from __future__ import annotations

from typing import Any


def command_operations(command: dict[str, Any]) -> list[str]:
    """The operations one remote command executed, head step first."""

    operations: list[str] = []
    head = (command.get("step") or {}).get("operation")
    if head:
        operations.append(str(head))
    for entry in command.get("batched_steps") or []:
        operation = ((entry or {}).get("step") or {}).get("operation")
        if operation:
            operations.append(str(operation))
    return operations
