"""Device-plugin public lifecycle races and allocatable convergence."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import collector_device_plugin_fixture as module
from scripts.e2e.regional.probes.collector_plugin_watchdog import WINDOW_KEY
from tests.regional._cov95_collect_net import Clock, no_external_effects  # noqa: F401
from tests.regional.test_collector_device_plugin_fixture import PluginApi, fixture


@pytest.mark.parametrize(
    "terms",
    [
        [{"matchFields": "invalid"}],
        [{"matchExpressions": [{"key": "gpu"}], "matchFields": {}}],
        "not-list",
        [None],
    ],
)
def test_exclusion_refuses_malformed_selector_terms(terms: Any) -> None:
    original = {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": terms
            }
        }
    }
    before = deepcopy(original)
    with pytest.raises(module.RegionalFixtureError):
        module.excluding_node_affinity(original, "node-a")
    assert original == before


@pytest.mark.parametrize("mode", ["empty", "duplicate", "wrong-token", "inactive"])
def test_discovery_requires_exactly_one_active_matching_daemonset(
    tmp_path: Path, mode: str
) -> None:
    class Api(PluginApi):
        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            if args[:2] == ("get", "daemonset"):
                document = deepcopy(self.document)
                if mode == "inactive":
                    document["status"]["desiredNumberScheduled"] = 0
                if mode == "wrong-token":
                    document["metadata"]["name"] = "another"
                return json.dumps(
                    {
                        "items": []
                        if mode == "empty"
                        else [document] * (2 if mode == "duplicate" else 1)
                    }
                )
            return super().kubectl(plane, *args, **kwargs)

    with pytest.raises(module.RegionalFixtureError, match="expected one"):
        fixture(Api({}), tmp_path).discover()


@pytest.mark.parametrize("problem", ["read-shape", "executor", "pod-uid", "restore"])
def test_plugin_refuses_unproven_read_image_and_pod_ownership(
    tmp_path: Path, problem: str
) -> None:
    class Api(PluginApi):
        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            if (
                args[:2] == ("get", "daemonset")
                and not kwargs.get("all_namespaces")
                and problem == "read-shape"
            ):
                return "[]"
            if args[:2] == ("get", "deployment") and problem == "executor":
                return '{"spec":{"template":{"spec":{"containers":[]}}}}'
            return super().kubectl(plane, *args, **kwargs)

    api = Api({})
    plugin = fixture(api, tmp_path)
    plugin.restore()
    plugin.discover()
    if problem == "pod-uid":
        api.pods.append(
            {
                "metadata": {
                    "name": "target",
                    "ownerReferences": [
                        {"uid": plugin.uid, "name": plugin.name, "kind": "DaemonSet"}
                    ],
                },
                "spec": {"nodeName": plugin.node},
            }
        )
    if problem == "restore":
        api.document["spec"]["template"]["spec"]["affinity"] = {"foreign": True}
        with pytest.raises(module.RegionalFixtureError, match="restoration"):
            plugin.restore()
    else:
        with pytest.raises(module.RegionalFixtureError):
            plugin.exclude_node()
    assert api.deleted == [], "unproven Pod identity cannot authorize deletion"


@pytest.mark.parametrize(
    "race", ["uid", "version", "affinity", "already-excluded", "strategy", "readback"]
)
def test_exclusion_rechecks_state_after_watchdog_admission(
    tmp_path: Path, monkeypatch: Any, race: str
) -> None:
    class Api(PluginApi):
        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            result = super().kubectl(plane, *args, **kwargs)
            if args[:2] == ("patch", "daemonset") and race == "readback":
                self.document["spec"]["template"]["spec"]["affinity"] = {
                    "foreign": True
                }
            return result

    api = Api({})

    class Watchdog:
        def __init__(self, regional: Any, **kwargs: Any) -> None:
            self.record = {"window_id": "window-a"}
            self.affinity = kwargs["excluded_affinity"]
            self.calls = 0

        def arm(self) -> None:
            api.document["metadata"]["annotations"] = {WINDOW_KEY: "window-a"}

        def require_armed(self) -> None:
            self.calls += 1
            if self.calls != 1:
                return
            if race == "uid":
                api.document["metadata"]["uid"] = "foreign"
            elif race == "version":
                api.document["metadata"]["resourceVersion"] = ""
            elif race == "affinity":
                api.document["spec"]["template"]["spec"]["affinity"] = {"foreign": True}
            elif race == "already-excluded":
                api.document["spec"]["template"]["spec"]["affinity"] = deepcopy(
                    self.affinity
                )
            elif race == "strategy":
                api.document["spec"]["updateStrategy"]["type"] = "RollingUpdate"

    monkeypatch.setattr(module, "PluginWatchdog", Watchdog)
    plugin = fixture(api, tmp_path)
    if race == "already-excluded":
        plugin.exclude_node()
        assert api.patches == [], "already matching exclusion must not repatch"
    else:
        with pytest.raises(module.RegionalFixtureError):
            plugin.exclude_node()
        assert api.deleted == [], (
            "failed exclusion verification cannot delete plugin Pods"
        )


@pytest.mark.parametrize("converges", [False, True])
def test_allocatable_polling_has_bounded_timeout_and_returns_observed_metadata(
    tmp_path: Path, monkeypatch: Any, converges: bool
) -> None:
    clock = Clock()
    monkeypatch.setattr(module, "time", clock)

    class Api(PluginApi):
        def node_metadata(self, node: str) -> dict[str, Any]:
            return {"node": node, "uid": "uid-a"}

        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            if args[:2] == ("get", "node"):
                count = 2 if converges and clock.now >= 1005 else 0
                return json.dumps(
                    {"status": {"allocatable": {"nvidia.com/gpu": str(count)}}}
                )
            return super().kubectl(plane, *args, **kwargs)

    plugin = fixture(Api({}), tmp_path)
    if converges:
        assert plugin.wait_allocatable(2, timeout_seconds=10) == {
            "node": "selected-node",
            "uid": "uid-a",
            "allocatable": 2,
        }
    else:
        with pytest.raises(module.RegionalFixtureError, match="did not become 2"):
            plugin.wait_allocatable(2, timeout_seconds=10)
    assert clock.now <= 1010
