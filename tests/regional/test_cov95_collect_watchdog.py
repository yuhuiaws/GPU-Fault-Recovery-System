"""Plugin watchdog restoration, journal ownership and admission failures."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kubernetes import client, config

from scripts.e2e.regional import collector_plugin_watchdog as watchdog
from scripts.e2e.regional.probes import collector_plugin_watchdog as probe
from tests.regional._cov95_collect_net import Clock, no_external_effects  # noqa: F401
from tests.regional.test_collector_device_plugin_fixture import PluginApi


def new_watchdog(api: PluginApi, tmp_path: Path) -> watchdog.PluginWatchdog:
    return watchdog.PluginWatchdog(
        api,
        baseline=deepcopy(api.document),
        excluded_affinity={"nodeAffinity": {}},
        image="executor@sha256:" + "b" * 64,
        case_dir=tmp_path,
    )


@pytest.mark.parametrize(
    "prior", ["unclosed", "closed", "archive-match", "archive-conflict"]
)
def test_journal_requires_closed_previous_window_and_immutable_archive(
    tmp_path: Path, prior: str
) -> None:
    api = PluginApi({})
    first = new_watchdog(api, tmp_path)
    record = {**first.record, "state": "PREPARING" if prior == "unclosed" else "CLOSED"}
    first.path.parent.mkdir(parents=True)
    first.path.write_text(json.dumps(record))
    archive = first.path.with_name(f"{first.path.stem}-{record['window_id']}.json")
    if prior.startswith("archive"):
        archive.write_text(
            json.dumps(record if prior == "archive-match" else {"different": True})
        )
    if prior in {"unclosed", "archive-conflict"}:
        with pytest.raises(watchdog.RegionalFixtureError):
            new_watchdog(api, tmp_path)
    else:
        second = new_watchdog(api, tmp_path)
        assert json.loads(archive.read_text()) == record
        assert second.record["window_id"] != first.record["window_id"]


@pytest.mark.parametrize(
    "problem", ["uid", "namespace", "image", "owned", "changed", "version", "strategy"]
)
def test_watchdog_arm_rejects_unproven_resource_identity(
    tmp_path: Path, problem: str
) -> None:
    api = PluginApi({})
    current = new_watchdog(api, tmp_path)
    if problem == "uid":
        current.record["daemonset_uid"] = ""
    elif problem == "namespace":
        current.namespace_uid = ""
    elif problem == "image":
        current.image = "mutable:latest"
    elif problem == "owned":
        api.document["metadata"]["annotations"] = {probe.WINDOW_KEY: "foreign"}
    elif problem == "changed":
        api.document["metadata"]["uid"] = "foreign"
    elif problem == "version":
        del api.document["metadata"]["resourceVersion"]
    else:
        api.document["spec"]["updateStrategy"]["type"] = "RollingUpdate"
    with pytest.raises(watchdog.RegionalFixtureError):
        current.arm()
    assert api.patches == []
    assert api.resources == {}


def test_arm_preserves_existing_annotations_and_ignores_nonjson_logs(
    tmp_path: Path,
) -> None:
    class Api(PluginApi):
        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            result = super().kubectl(plane, *args, **kwargs)
            return "not-json\n" + result if args[0] == "logs" else result

    api = Api({})
    api.document["metadata"]["annotations"] = {"preserve": "value"}
    current = new_watchdog(api, tmp_path)
    current.arm()
    assert current.record["state"] == "ARMED"
    current.restore()
    assert api.document["metadata"]["annotations"] == {"preserve": "value"}


def test_arm_does_not_adopt_unreceipted_resource(tmp_path: Path) -> None:
    class Api(PluginApi):
        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            if args[:2] == ("get", "serviceaccount"):
                return '{"metadata":{"uid":"foreign"}}'
            return super().kubectl(plane, *args, **kwargs)

    api = Api({})
    with pytest.raises(watchdog.RegionalFixtureError, match="without creation proof"):
        new_watchdog(api, tmp_path).arm()
    assert api.resources == {}


@pytest.mark.parametrize(
    "failure",
    [
        "near-deadline",
        "scope",
        "namespace",
        "readback",
        "resource",
        "retirement",
        "resource-absent",
    ],
)
def test_restore_never_discards_foreign_or_unconfirmed_resources(
    tmp_path: Path, failure: str, monkeypatch: Any
) -> None:
    class Api(PluginApi):
        suppress_patch = False
        retirement = False
        namespace_uid = "namespace-a"

        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            if args[:2] == ("get", "namespace"):
                return json.dumps({"metadata": {"uid": self.namespace_uid}})
            if args[:2] == ("patch", "daemonset"):
                patch = json.loads(args[args.index("-p") + 1])
                if self.suppress_patch or (
                    self.retirement
                    and patch[-1] == {"op": "remove", "path": probe.WINDOW_PATH}
                ):
                    return ""
            return super().kubectl(plane, *args, **kwargs)

    api = Api({})
    current = new_watchdog(api, tmp_path)
    current.restore()
    assert api.patches == [], "unclaimed watchdog cleanup is a no-op"
    current.arm()
    if failure == "near-deadline":
        monkeypatch.setattr(watchdog, "time", Clock(current.record["restore_at"] - 179))
        with pytest.raises(watchdog.RegionalFixtureError, match="too close"):
            current.require_armed()
        return
    if failure == "scope":
        monkeypatch.setattr(api, "evidence_identity", lambda: {"cluster_id": "foreign"})
    elif failure == "namespace":
        api.namespace_uid = "foreign"
    elif failure == "readback":
        api.suppress_patch = True
    elif failure == "resource":
        api.resources["job"]["metadata"]["uid"] = "foreign"
    elif failure == "retirement":
        api.retirement = True
    else:
        api.resources.clear()
    if failure == "resource-absent":
        current.restore()
        assert current.record["state"] == "CLOSED"
    else:
        with pytest.raises(watchdog.RegionalFixtureError):
            current.restore()
        assert current.claimed, "unconfirmed cleanup must retain owned recovery journal"
        assert current.record["state"] != "CLOSED"


def probe_state() -> tuple[dict[str, Any], dict[str, Any]]:
    record = {
        "daemonset_uid": "uid-a",
        "namespace": "namespace-a",
        "daemonset": "plugin",
        "window_id": "window-a",
        "restore_at": 1010,
        "baseline": {"present": False, "value": None},
        "excluded_affinity": {"nodeAffinity": {}},
    }
    document = {
        "metadata": {
            "uid": "uid-a",
            "resourceVersion": "1",
            "annotations": {probe.WINDOW_KEY: "window-a"},
        },
        "spec": {
            "updateStrategy": {"type": "OnDelete"},
            "template": {"spec": {"affinity": {"nodeAffinity": {}}}},
        },
    }
    return record, document


@pytest.mark.parametrize("problem", ["uid", "version", "owner", "affinity"])
def test_restoration_patch_refuses_foreign_state(problem: str) -> None:
    record, document = probe_state()
    if problem in {"uid", "version"}:
        document["metadata"]["uid" if problem == "uid" else "resourceVersion"] = ""
    elif problem == "owner":
        document["metadata"]["annotations"][probe.WINDOW_KEY] = "another"
    else:
        document["spec"]["template"]["spec"]["affinity"] = {"foreign": True}
    with pytest.raises(RuntimeError):
        probe.restoration_patch(document, record)


def test_restore_requires_post_patch_confirmation_and_is_idempotent() -> None:
    record, document = probe_state()
    patches = []
    api = SimpleNamespace(
        api_client=SimpleNamespace(sanitize_for_serialization=deepcopy),
        read_namespaced_daemon_set=lambda *a, **k: document,
        patch_namespaced_daemon_set=lambda *a, **k: patches.append(a[2]),
    )
    with pytest.raises(RuntimeError, match="readback"):
        probe.restore(api, record)
    assert len(patches) == 1
    document["metadata"]["annotations"][probe.WINDOW_KEY] = "CLOSED:window-a"
    del document["spec"]["template"]["spec"]["affinity"]
    probe.restore(api, record)
    assert len(patches) == 1, "already restored window must not patch twice"


@pytest.mark.parametrize(
    "mode", ["success", "retry", "permanent", "timeout", "invalid"]
)
def test_probe_main_retries_only_transient_api_failures(
    monkeypatch: Any, capsys: Any, mode: str
) -> None:
    record, document = probe_state()
    clock = Clock()
    monkeypatch.setattr(probe, "time", clock)
    if mode == "invalid":
        record["restore_at"] = True
    monkeypatch.setenv("COLLECTOR_PLUGIN_WINDOW", json.dumps(record))
    api = SimpleNamespace(
        api_client=SimpleNamespace(sanitize_for_serialization=deepcopy),
        read_namespaced_daemon_set=lambda *a, **k: document,
    )
    monkeypatch.setattr(config, "load_incluster_config", lambda: None)
    monkeypatch.setattr(client, "AppsV1Api", lambda: api)
    attempts = []

    def restore(instance: Any, current: Any) -> None:
        assert instance is api
        assert current == record
        attempts.append(clock.now)
        if mode == "permanent":
            raise client.exceptions.ApiException(status=403)
        if mode == "timeout":
            clock.now += 181
            raise client.exceptions.ApiException(status=503)
        if mode == "retry" and len(attempts) == 1:
            raise client.exceptions.ApiException(status=409)

    monkeypatch.setattr(probe, "restore", restore)
    if mode in {"invalid", "permanent", "timeout"}:
        with pytest.raises(RuntimeError):
            probe.main()
        assert len(attempts) == (0 if mode == "invalid" else 1)
    else:
        probe.main()
        assert attempts == ([1010, 1012] if mode == "retry" else [1010])
        assert [
            json.loads(line)["state"] for line in capsys.readouterr().out.splitlines()
        ] == ["ARMED", "RESTORED"]
