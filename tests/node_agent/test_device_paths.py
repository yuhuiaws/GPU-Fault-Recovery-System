from __future__ import annotations

from ._support import (
    CompletedProcess,
    NodeActionStatus,
    WorkflowOperation,
    command,
    envelope,
    node_action_executor,
)


TWO_GPU_XML = """<?xml version="1.0" ?>
<nvidia_smi_log>
  <gpu id="00000000:53:00.0">
    <uuid>GPU-a</uuid>
    <minor_number>1</minor_number>
  </gpu>
  <gpu id="00000000:64:00.0">
    <uuid>GPU-b</uuid>
    <minor_number>0</minor_number>
  </gpu>
</nvidia_smi_log>
"""


class CountingRunner:
    """nvidia-smi on a two-GPU node, counting each inventory query."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, argv, **_):
        self.commands.append(list(argv))
        if argv[:3] == ["nvidia-smi", "-q", "-x"]:
            return CompletedProcess(argv, 0, stdout=TWO_GPU_XML, stderr="")
        if "--query-gpu=uuid" in argv:
            return CompletedProcess(argv, 0, stdout="GPU-a\nGPU-b\n", stderr="")
        return CompletedProcess(argv, 0, stdout="", stderr="")

    def inventory_queries(self) -> int:
        return len([item for item in self.commands if "--query-gpu=uuid" in item])


def test_naming_several_wrong_uuids_probes_the_inventory_once(tmp_path) -> None:
    """One refusal is one inventory query, however many UUIDs it names.

    The cause "no GPU on this node reports this UUID" is established against
    the local inventory, and it was established per UUID: a step carrying a
    stale plan's eight GPUs spent eight ``nvidia-smi`` calls -- 15 s timeout
    each, on a node whose driver is already misbehaving -- to build one error
    message, and the destructive step it gates sat there for the duration.
    """

    runner = CountingRunner()
    proc = tmp_path / "empty-proc"
    proc.mkdir()
    agent = node_action_executor(
        tmp_path,
        "inventory-probe.db",
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        runner=runner,
        proc_root=str(proc),
        sleep=lambda _seconds: None,
    )

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                gpu_uuids=["GPU-x", "GPU-y", "GPU-z"],
            )
        )
    )

    assert result.status is NodeActionStatus.FAILED, result.details
    for gpu_uuid in ("GPU-x", "GPU-y", "GPU-z"):
        assert f"{gpu_uuid} (no GPU on this node reports this UUID)" in (
            result.error or ""
        ), result.error
    assert runner.inventory_queries() == 1, (
        "the inventory is one probe for the whole refusal, not one per UUID: "
        f"{runner.commands}"
    )
