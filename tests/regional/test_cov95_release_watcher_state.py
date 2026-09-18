from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault_release import regional_release_gpu_rollout as gpu
from gpu_fault_release import rollout
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._release_orchestrator_support import config_file
from tests.regional._resource_probe_fakes import resource_probe_result


class WatcherRunner:
    dry_run = False

    def __init__(
        self, existing: tuple[str, ...] = gpu.COMPLETION_WATCHER_STATE_CONFIG_MAPS
    ) -> None:
        self.existing = existing
        self.responses: dict[str, tuple[int, str, str] | Exception] = {}
        self.probes: list[tuple[list[str], float | None]] = []
        self.applied: list[dict[str, Any]] = []

    def probe_output(
        self, arguments: list[str], *, timeout_seconds: float | None = None
    ) -> tuple[int, str, str]:
        self.probes.append((list(arguments), timeout_seconds))
        name = arguments[arguments.index("get") + 2]
        response = self.responses.get(name)
        if isinstance(response, Exception):
            raise response
        if response is not None:
            return response
        return resource_probe_result(arguments, present=name in self.existing)

    def run(self, arguments: list[str], **kwargs: Any) -> str:
        assert arguments[-3:] == ["apply", "-f", "-"]
        self.applied.extend(yaml.safe_load_all(kwargs["input_text"]))
        return ""


def watcher_release(root: Path, runner: WatcherRunner) -> rollout.RegionalRelease:
    return rollout.RegionalRelease(
        rollout.ReleaseConfig.load(config_file(root)), runner
    )


@pytest.mark.parametrize("name", gpu.COMPLETION_WATCHER_STATE_CONFIG_MAPS)
@pytest.mark.parametrize(
    "code,error",
    [
        (1, "credential plugin NotFound: authentication failed"),
        (127, "kubectl: command not found"),
        (1, 'Error from server (NotFound): configmaps "other" not found'),
    ],
)
def test_read_errors_never_authorize_empty_state_reapply(
    tmp_path: Path, name: str, code: int, error: str
) -> None:
    runner = WatcherRunner()
    runner.responses[name] = (code, "", error)
    release = watcher_release(tmp_path, runner)
    with pytest.raises(ReleaseError, match=f"cannot read configmap/{name}"):
        gpu.reassert_completion_watcher_state(release, release.config.clusters[0])
    assert runner.applied == []
    assert runner.probes[-1][0][runner.probes[-1][0].index("get") + 2] == name


@pytest.mark.parametrize(
    "fault", ["invalid-json", "null", "list", "kind", "name", "namespace", "uid"]
)
def test_untrusted_resource_response_is_not_presence_or_absence(
    tmp_path: Path, fault: str
) -> None:
    runner = WatcherRunner()
    name = gpu.COMPLETION_WATCHER_STATE_CONFIG_MAPS[0]
    document: dict[str, Any] = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "namespace": "gpu-fault-system",
            "uid": "example-uid",
        },
    }
    if fault in {"invalid-json", "null", "list"}:
        output = {"invalid-json": "{", "null": "null", "list": "[]"}[fault]
    else:
        if fault == "kind":
            document["kind"] = "Secret"
        else:
            document["metadata"][fault] = "" if fault == "uid" else "other"
        output = json.dumps(document)
    runner.responses[name] = (0, output, "")
    release = watcher_release(tmp_path, runner)
    with pytest.raises(ReleaseError, match="invalid response"):
        gpu.reassert_completion_watcher_state(release, release.config.clusters[0])
    assert runner.applied == []


@pytest.mark.parametrize(
    "error", [TimeoutError("read timeout"), OSError("read failed")]
)
def test_probe_exception_stops_watcher_repair(tmp_path: Path, error: Exception) -> None:
    runner = WatcherRunner()
    runner.responses[gpu.COMPLETION_WATCHER_STATE_CONFIG_MAPS[0]] = error
    release = watcher_release(tmp_path, runner)
    with pytest.raises(ReleaseError, match="cannot read configmap"):
        gpu.reassert_completion_watcher_state(release, release.config.clusters[0])
    assert runner.applied == []


@pytest.mark.parametrize(
    "existing",
    [
        (),
        (gpu.COMPLETION_WATCHER_STATE_CONFIG_MAPS[0],),
        gpu.COMPLETION_WATCHER_STATE_CONFIG_MAPS,
    ],
)
def test_only_confirmed_absent_state_is_created_after_both_bounded_reads(
    tmp_path: Path, existing: tuple[str, ...]
) -> None:
    runner = WatcherRunner(existing)
    release = watcher_release(tmp_path, runner)
    gpu.reassert_completion_watcher_state(release, release.config.clusters[0])
    assert {
        item["metadata"]["name"]
        for item in runner.applied
        if item["kind"] == "ConfigMap"
    } == set(gpu.COMPLETION_WATCHER_STATE_CONFIG_MAPS) - set(existing)
    assert [item["kind"] for item in runner.applied if item["kind"] != "ConfigMap"] == [
        "ClusterRole",
        "ClusterRoleBinding",
    ]
    assert len(runner.probes) == 2
    for arguments, timeout in runner.probes:
        assert "--ignore-not-found" in arguments
        assert "--request-timeout=15s" in arguments
        assert arguments[arguments.index("-o") + 1] == "json"
        assert arguments[arguments.index("-n") + 1] == release.config.namespace
        assert timeout == 20
