"""The chain rig's stand-in for a rank's PyTorch flight-recorder pipe.

The agent asks for a dump as ``_trigger_flight_recorder`` does: open the
rank's TORCH_NCCL_DEBUG_INFO_PIPE_FILE FIFO ``O_WRONLY | O_NONBLOCK``, write
``b"1\\n"``, close. ``ChainNode`` answers from its own thread, as PyTorch's
monitor thread does. The e2e chain depends on two properties pinned here: no
request is lost, and every request is answered before the agent looks again.
"""

import os
import time
from types import SimpleNamespace

from tests.execution import _nccl_chain_node as chain_node
from tests.execution._nccl_chain_support import chain_fixture as chain_fixture


def request(pipe, descriptor=None):
    if descriptor is None:
        descriptor = os.open(pipe, os.O_WRONLY | os.O_NONBLOCK)
    try:
        os.write(descriptor, b"1\n")
    finally:
        os.close(descriptor)


def test_a_request_opened_between_the_responders_select_and_read_is_answered(
    chain, monkeypatch
):
    rig = chain(nodes=1, mode="fr-whole-group")
    [actor] = rig.actors
    pipe = actor.namespace / "trace0.pipe"
    request(pipe)
    rig.node_clock.sleep(0.25)
    assert actor.requests == 1, actor.errors
    # The agent's next request opens right after the responder's select()
    # returned and before its read(). With only the agent ever writing, the
    # answered FIFO read as end-of-file, select() returned at once, the read
    # met the open writer with nothing written yet and raised EAGAIN; the
    # thread died and the node's second dump was never written.
    opened = []
    real = chain_node.select.select

    def select(readers, writers, errors, timeout):
        ready = real(readers, writers, errors, timeout)
        if not opened:
            opened.append(os.open(pipe, os.O_WRONLY | os.O_NONBLOCK))
        return ready

    monkeypatch.setattr(chain_node, "select", SimpleNamespace(select=select))
    deadline = time.monotonic() + 10
    while not opened and time.monotonic() < deadline:
        time.sleep(0.001)
    assert opened, "the responder never returned from select()"
    request(pipe, opened[0])
    rig.node_clock.sleep(0.25)
    assert (actor.requests, actor.errors) == (2, []), actor.errors


def test_the_agents_sleep_returns_only_after_a_late_responder_answered(
    chain, monkeypatch
):
    rig = chain(nodes=1, mode="fr-whole-group")
    [actor] = rig.actors
    entry = chain_node.entry

    def late(*args, **kwargs):
        # The monitor thread is not scheduled for a while: one node-clock
        # sleep is 0.25 s of the agent's budget but only 3 ms of real time.
        time.sleep(0.1)
        return entry(*args, **kwargs)

    monkeypatch.setattr(chain_node, "entry", late)
    request(actor.namespace / "trace0.pipe")
    rig.node_clock.sleep(0.25)
    assert (actor.namespace / "dump0").is_file(), "the agent looked before the dump"
    assert (actor.requests, actor.errors) == (1, []), actor.errors
