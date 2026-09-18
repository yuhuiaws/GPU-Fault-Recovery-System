from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import control_plane_env_window, executor_env_window
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError


class WindowAPI:
    def __init__(self, module) -> None:
        self.module = module
        self.settings = SimpleNamespace(
            cluster_id="test-cluster",
            environment=lambda: {"context": "test-context", "namespace": "test-ns"},
        )
        self.release_id = "release-one"
        self.patches: list[dict] = []
        self.patch_arguments: list[tuple[str, ...]] = []
        self.fail_after_patch = False
        self.fail_rollout = False
        self.visible_replicas = 2
        self.deployment = {
            "metadata": {
                "uid": "deployment-one",
                "resourceVersion": "1",
                "generation": 1,
            },
            "spec": {
                "replicas": 2,
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": module.CONTAINER,
                                "env": [
                                    {"name": "UNRELATED", "value": "preserved"},
                                    {
                                        "name": "PASSWORD",
                                        "value": "private-fixture-literal",
                                    },
                                ],
                            }
                        ]
                    }
                },
            },
            "status": {
                "observedGeneration": 1,
                "replicas": 2,
                "readyReplicas": 2,
                "updatedReplicas": 2,
                "availableReplicas": 2,
            },
        }

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": self.release_id, "cluster_id": "test-cluster"}

    def values(self) -> dict:
        values = {
            name: getattr(self.module, "SHIPPED_DEFAULTS", {}).get(name)
            for name in self.module.ALLOWED_VARIABLES
        }
        values.update(
            {
                item["name"]: item.get("value", "")
                for item in self.deployment["spec"]["template"]["spec"]["containers"][
                    0
                ]["env"]
            }
        )
        return values

    def ready_pods(self, *_args) -> list[dict]:
        return [{"name": f"pod-{index}"} for index in range(self.visible_replicas)]

    def kubectl(self, plane: str, *arguments: str, **kwargs) -> str:
        if arguments[:2] == ("get", "deployment"):
            return json.dumps(self.deployment)
        if arguments[:2] == ("patch", "deployment"):
            self.patch_arguments.append(arguments)
            patch = json.loads(kwargs["input_text"])
            assert patch[0] == {
                "op": "test",
                "path": "/metadata/uid",
                "value": self.deployment["metadata"]["uid"],
            }
            assert patch[1] == {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": self.deployment["metadata"]["resourceVersion"],
            }
            assert patch[2]["path"] == "/spec/template/spec/containers/0/env"
            self.patches.append(patch)
            self.deployment["spec"]["template"]["spec"]["containers"][0]["env"] = patch[
                2
            ]["value"]
            self.deployment["metadata"]["generation"] += 1
            self.deployment["metadata"]["resourceVersion"] = str(
                int(self.deployment["metadata"]["resourceVersion"]) + 1
            )
            self.deployment["status"]["observedGeneration"] = self.deployment[
                "metadata"
            ]["generation"]
            if self.fail_after_patch:
                self.fail_after_patch = False
                raise RegionalFixtureError("patch ACK was lost")
            return "{}"
        if arguments[:2] == ("rollout", "status"):
            if self.fail_rollout:
                raise RegionalFixtureError("rollout did not finish")
            return "ready"
        if arguments[0] == "exec":
            return json.dumps(self.values())
        raise AssertionError(f"unexpected fake API operation: {arguments[:2]}")


@pytest.fixture(params=[control_plane_env_window, executor_env_window])
def window(request, tmp_path: Path):
    module = request.param
    api = WindowAPI(module)
    settings = module.Settings(
        baseline=tmp_path / "window.json", rollout_timeout_seconds=1
    )
    if module is control_plane_env_window:
        assignments = {
            module.LIFETIME_VARIABLE: "180",
            module.EXECUTION_TIMEOUT_VARIABLE: "180",
        }
    else:
        assignments = {module.ALLOWED_VARIABLES[0]: "10"}
    return module, api, settings, assignments


def open_window(window) -> dict:
    module, api, settings, assignments = window
    return module.open_window(settings, api, module.survey(api), assignments)


def test_window_round_trip_preserves_uncontrolled_environment(window) -> None:
    module, api, settings, _assignments = window
    original = copy.deepcopy(
        api.deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    )
    opened = open_window(window)
    assert opened["state"] == "OPEN"
    assert opened["baseline"]["uid"] == "deployment-one"
    closed = module.close_window(settings, api, module.survey(api))
    assert closed["state"] == "CLOSED"
    assert closed["closed_at"], "closed status needs completed restoration proof"
    assert (
        api.deployment["spec"]["template"]["spec"]["containers"][0]["env"] == original
    )
    assert all("--patch-file=/dev/stdin" in args for args in api.patch_arguments), (
        "the environment must travel through private stdin"
    )
    assert all(
        "private-fixture-literal" not in " ".join(args) for args in api.patch_arguments
    ), "an unrelated credential literal entered the process argv"


def test_window_close_failure_does_not_claim_closed_and_can_resume(window) -> None:
    module, api, settings, _assignments = window
    open_window(window)
    api.fail_rollout = True
    with pytest.raises(RegionalFixtureError, match="rollout"):
        module.close_window(settings, api, module.survey(api))
    saved = json.loads(settings.baseline.read_text())
    assert "closed_at" not in saved, "failed restoration was recorded as complete"
    assert saved["state"] == "CLOSING"
    api.fail_rollout = False
    assert module.close_window(settings, api, module.survey(api))["state"] == "CLOSED"
    assert len(api.patches) == 2, "an acknowledged restore should not be repeated"


@pytest.mark.parametrize("drift", ["uid", "release", "variable"])
def test_window_refuses_to_restore_a_changed_target(window, drift: str) -> None:
    module, api, settings, assignments = window
    open_window(window)
    if drift == "uid":
        api.deployment["metadata"]["uid"] = "replacement-deployment"
    elif drift == "release":
        api.release_id = "replacement-release"
    else:
        name = next(iter(assignments))
        for entry in api.deployment["spec"]["template"]["spec"]["containers"][0]["env"]:
            if entry["name"] == name:
                entry["value"] = "foreign-value"
    with pytest.raises(RegionalFixtureError):
        module.close_window(settings, api, module.survey(api))
    assert len(api.patches) == 1, "a different owner or value must not be overwritten"


def test_window_resumes_an_ack_lost_open_without_reapplying(window) -> None:
    module, api, settings, assignments = window
    api.fail_after_patch = True
    with pytest.raises(RegionalFixtureError, match="ACK"):
        open_window(window)
    saved = json.loads(settings.baseline.read_text())
    assert saved["state"] == "OPENING"
    result = module.open_window(settings, api, module.survey(api), assignments)
    assert result["state"] == "OPEN"
    assert len(api.patches) == 1, "a lost ACK must not repeat a committed change"


def test_window_requires_the_complete_replica_population(window) -> None:
    module, api, settings, _assignments = window
    api.visible_replicas = 1
    with pytest.raises(RegionalFixtureError, match="replicas"):
        open_window(window)
    assert not settings.baseline.exists(), (
        "an incomplete population must fail before mutation"
    )
    assert api.patches == [], "one surviving replica cannot authorize an env change"
