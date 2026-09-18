from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from kubernetes import client, config, stream, watch
from kubernetes.config.config_exception import ConfigException

from gpu_fault import completion_controller as module
from gpu_fault import completion_outbox as outbox_module
from gpu_fault import completion_reconcile_loop as loop
from gpu_fault.completion_outbox import KubernetesCompletionOutbox
from tests.completion._cov95_runtime_support import StopLoop, make_controller
from tests.completion._support import FakeCoreApi, FakeSink, FakeWatch, pod


@pytest.mark.parametrize("in_cluster", [True, False])
@pytest.mark.parametrize("discovery", [True, False])
def test_environment_factory_uses_real_outbox_and_fake_transports(
    monkeypatch, in_cluster, discovery
) -> None:
    core = FakeCoreApi([pod(0, include_gpu_uuids=False, gpu_count=1)])
    core.connect_get_namespaced_pod_exec = lambda *a, **kw: None
    http_calls, config_calls, streams = [], [], []

    class HttpSink(FakeSink):
        def __init__(self, url, **kwargs):
            super().__init__()
            http_calls.append((url, kwargs, self))

    def load_incluster():
        config_calls.append("incluster")
        if not in_cluster:
            raise ConfigException("local unit simulator")

    def exec_stream(*args, **kwargs):
        streams.append((args, kwargs))
        return "GPU-local\n\nMIG-local\n"

    monkeypatch.setattr(config, "load_incluster_config", load_incluster)
    monkeypatch.setattr(
        config, "load_kube_config", lambda: config_calls.append("fallback")
    )
    monkeypatch.setattr(client, "CoreV1Api", lambda: core)
    monkeypatch.setattr(client, "BatchV1Api", lambda: object())
    monkeypatch.setattr(client, "CustomObjectsApi", lambda: object())
    monkeypatch.setattr(watch, "Watch", lambda: FakeWatch([]))
    monkeypatch.setattr(stream, "stream", exec_stream)
    monkeypatch.setattr(outbox_module, "HttpEventSink", HttpSink)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "hp-cluster")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://control.invalid")
    monkeypatch.setenv("GPU_FAULT_DISCOVER_POD_GPU_UUIDS", str(discovery).lower())
    monkeypatch.setenv(
        "GPU_FAULT_PASSIVE_STOP_FALLBACK_SECONDS", "" if discovery else "7"
    )
    controller = module.controller_from_environment()
    assert isinstance(controller.sink.wrapped_sink, KubernetesCompletionOutbox), (
        "startup must retain the real write-ahead outbox"
    )
    controller.run_once()
    assert config_calls == (["incluster"] if in_cluster else ["incluster", "fallback"])
    assert len(http_calls) == 1
    url, kwargs, transport = http_calls[0]
    assert url == "https://control.invalid"
    assert kwargs == {
        "bearer_token": None,
        "timeout_seconds": 10,
        "processor_receipt_timeout_seconds": 120,
        "processor_receipt_poll_seconds": 0.25,
    }
    assert controller.emergency_fallback_seconds == (None if discovery else 7)
    observations = [
        payload
        for path, payload in transport.posts
        if path.endswith("workload-observations")
    ]
    assert observations[-1]["containers"][0]["gpu_uuids"] == (
        ["GPU-local", "MIG-local"] if discovery else []
    )
    if discovery:
        assert len(streams) == 1
        args, kwargs = streams[0]
        assert args == (core.connect_get_namespaced_pod_exec, "worker-0", "default")
        assert kwargs["command"] == [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ]
        assert kwargs["_request_timeout"] == 10
        assert kwargs["stdin"] is False
        assert kwargs["tty"] is False
    else:
        assert streams == []


def test_missing_kubernetes_dependency_is_actionable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "kubernetes", None)
    with pytest.raises(
        RuntimeError, match=r"install gpu-fault-control-plane\[collectors\]"
    ):
        module.controller_from_environment()


def test_main_runs_validation_then_the_controller_with_metrics_cleanup(monkeypatch):
    events = []
    controller = make_controller()
    monkeypatch.setattr(module, "configure_logging", lambda: events.append("logging"))
    monkeypatch.setattr(
        module,
        "validate_gpu_fault_environment",
        lambda **kwargs: events.append(("validation", kwargs)),
    )
    monkeypatch.setattr(module, "controller_from_environment", lambda: controller)
    monkeypatch.setattr(
        loop,
        "start_completion_metrics_server",
        lambda *a, **kw: SimpleNamespace(stop=lambda: events.append("metrics-stop")),
    )

    def sleep(seconds):
        raise StopLoop

    monkeypatch.setattr(loop, "time", SimpleNamespace(sleep=sleep))
    with pytest.raises(StopLoop):
        module.main([])
    assert events == [
        "logging",
        ("validation", {"process_name": "gpu-fault-completion-watcher"}),
        "metrics-stop",
    ]
    assert controller.reconcile_runs_total == 1


def test_operator_replay_mode_never_starts_the_watch_loop(monkeypatch) -> None:
    core, sink = FakeCoreApi(), FakeSink()
    controller = make_controller(core, KubernetesCompletionOutbox(core, sink))
    monkeypatch.setattr(module, "configure_logging", lambda: None)
    monkeypatch.setattr(module, "validate_gpu_fault_environment", lambda **kwargs: None)
    monkeypatch.setattr(module, "controller_from_environment", lambda: controller)
    with pytest.raises(SystemExit) as exited:
        module.main(["--replay-quarantined"])
    assert exited.value.code == 0
    assert core.list_calls == 0
    assert controller.reconcile_runs_total == 0
