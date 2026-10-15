"""The real executor client against the real control plane over a real socket.

uvicorn closes a kept-alive connection once it has idled for its keep-alive
timeout (here 1 s; 100 s in production, where a control-plane roll closes them
too). The executor's next post on that connection used to meet a dead socket:
a claim then cost a warning and the client's 1 s repeat, and a result, renewal
or progress post cost an error and a lease-sized delay. The pool now sees the
close before it writes and opens a fresh connection, for every post.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from collections.abc import Iterator
from threading import Thread
from types import SimpleNamespace

import pytest
import uvicorn

from gpu_fault.app import create_app
from gpu_fault.cluster_executor.regional_client import RegionalExecutorClient
from gpu_fault.regional import RemoteCommandResult, RemoteCommandStatus
from gpu_fault.store import InMemoryStore
from gpu_fault.transport.http_client import CONNECTION_POOL
from tests._builders import build_context
from tests.regional._regional_support import (
    TOKEN_A,
    enqueue_remote_command,
    registration,
)

CLUSTER = "cluster-a"
EXECUTOR = "cluster-a/executor-0"
OWNER = "gpu-fault-kubernetes-adapter"
KEEP_ALIVE_SECONDS = 1


@pytest.fixture
def control_plane(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    store = InMemoryStore()
    context = build_context(store=store, execution_token="e" * 32)
    context.regional_mode = True
    store.save_regional_cluster(registration(CLUSTER, TOKEN_A))
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "false")
    monkeypatch.setenv("GPU_FAULT_STORE_IO_WORKERS", "4")
    monkeypatch.setenv(
        "GPU_FAULT_REGIONAL_CLUSTERS_JSON",
        json.dumps(
            [
                {
                    "cluster_id": CLUSTER,
                    "token": TOKEN_A,
                    "hyperpod_cluster_name": f"hp-{CLUSTER}",
                }
            ]
        ),
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(context),
            log_level="error",
            timeout_keep_alive=KEEP_ALIVE_SECONDS,
        )
    )
    serving = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    serving.start()
    deadline = time.monotonic() + 15
    while not server.started and serving.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "the local control plane must start"
    CONNECTION_POOL.close()
    CONNECTION_POOL.reset_stats()
    try:
        yield SimpleNamespace(
            url=f"http://127.0.0.1:{listener.getsockname()[1]}",
            store=store,
            server=server,
        )
    finally:
        CONNECTION_POOL.close()
        server.should_exit = True
        serving.join(15)
        listener.close()
        assert not serving.is_alive(), "the local control plane must stop"


def idle_past_the_keep_alive(control_plane: SimpleNamespace) -> None:
    """Idle until uvicorn has closed the kept-alive connection, not for a guess."""

    deadline = time.monotonic() + 15
    while control_plane.server.server_state.connections and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not control_plane.server.server_state.connections, "closed while idle"
    # Loopback delivers the FIN with the close; the margin covers a loaded host.
    time.sleep(0.05)


def test_every_post_after_the_control_plane_closed_the_idle_connection_lands(
    control_plane: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    enqueue_remote_command(control_plane.store, "remote-" + "f" * 24, owner=OWNER)
    client = RegionalExecutorClient(control_plane.url, CLUSTER, TOKEN_A)

    with caplog.at_level(logging.WARNING, logger="gpu_fault.cluster_executor"):
        [command] = client.claim(
            EXECUTOR, execution_owners=[OWNER], max_commands=1, lease_seconds=60
        )
        idle_past_the_keep_alive(control_plane)
        renewed = client.renew(command, EXECUTOR, 60)
        idle_past_the_keep_alive(control_plane)
        done = client.complete(
            command,
            RemoteCommandResult(
                lease_token=command.lease_token, status=RemoteCommandStatus.SUCCEEDED
            ),
        )
        idle_past_the_keep_alive(control_plane)
        assert (
            client.claim(
                EXECUTOR, execution_owners=[OWNER], max_commands=1, lease_seconds=60
            )
            == []
        )

    assert renewed.lease_token == command.lease_token
    assert done.status is RemoteCommandStatus.SUCCEEDED
    stored = control_plane.store.get_remote_command(command.command_id)
    assert stored.status is RemoteCommandStatus.SUCCEEDED
    stats = CONNECTION_POOL.stats()
    assert stats["peer_closed_total"] == 3, "renewal, result and the second claim"
    assert stats["stale_connection_failures_total"] == 0
    assert stats["stale_connection_retries_total"] == 0, "no request had to repeat"
    assert not [
        record.getMessage()
        for record in caplog.records
        if "without a response" in record.getMessage()
    ], "no post met the closed connection"
