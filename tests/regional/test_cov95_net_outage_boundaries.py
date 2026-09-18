"""NET-007 public preflight and cleanup paths with fake Kubernetes transport."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_net007_transient_api_outage as runner
from tests.regional._cov95_collect_net import Clock, no_external_effects  # noqa: F401
from tests.regional.test_net007_outage_lifecycle import Regional, fixture


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "ownership",
        "queue",
        "commands",
        "image",
        "classifier",
        "version",
        "permission",
        "label",
        "existing",
        "focused",
        "bound",
    ],
)
def test_read_only_preflight_rejects_each_unsafe_observation(
    tmp_path: Path, monkeypatch: Any, problem: str | None
) -> None:
    site = tmp_path / "fixture-site"
    site.touch()
    calls = []
    identity = {"release_id": "release-a", "cluster_id": "cluster-a"}

    class Api:
        def evidence_identity(self) -> dict[str, str]:
            return dict(identity)

        def node_snapshot(self, node: str) -> dict[str, Any]:
            return {
                "name": node,
                "uid": "uid-a",
                "ready": "True",
                "unschedulable": False,
                "ownership_annotations": {"owner": "other"}
                if problem == "ownership"
                else {},
            }

        def business_workloads(self, node: str) -> list[Any]:
            return []

        def node_workload_view(self, node: str) -> dict[str, Any]:
            # The control plane's own view: inside the watcher's missing-Pod
            # grace a node with no Pods is still bound to the attempt that
            # left it, and the preflight must refuse it (NET-007 attempt 2).
            if problem == "bound":
                return {
                    "workload_state": "ACTIVE",
                    "attempt_ids": ["train-a1-r-c6e14c8e"],
                    "workload_ids": ["ns/pytorchjob/trainer"],
                }
            return {"workload_state": "IDLE", "attempt_ids": [], "workload_ids": []}

        def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "queue": {"depth": int(problem == "queue")},
                "remote_commands": {
                    "open_by_cluster": {"cluster-a": 1} if problem == "commands" else {}
                },
            }

        def cpu_blast_snapshot(self) -> dict[str, Any]:
            return {}

        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            calls.append(args)
            assert plane == "gpu"
            if args[0] == "auth":
                return "no" if problem == "permission" else "yes"
            if args[0] == "version":
                return json.dumps(
                    {
                        "serverVersion": {
                            "minor": "29" if problem == "version" else "30+"
                        }
                    }
                )
            if args[:2] == ("get", "node"):
                return (
                    "[]"
                    if problem == "label"
                    else '{"kubernetes.io/hostname":"node-a"}'
                )
            if args[:2] == ("get", "deployment"):
                return json.dumps(
                    {
                        "spec": {
                            "template": {
                                "spec": {
                                    "serviceAccountName": "executor",
                                    "containers": []
                                    if problem == "image"
                                    else [{"image": "image@sha256:" + "a" * 64}],
                                }
                            }
                        }
                    }
                )
            if args[:2] == ("get", "validatingwebhookconfiguration"):
                return "preexisting" if problem == "existing" else ""
            assert args[0] == "exec"
            assert args[args.index("--") + 1] == "/opt/gpu-fault/executor/bin/python"
            return json.dumps({"classifier": problem != "classifier"})

    api = Api()

    class Factory:
        def __new__(cls, settings: Any) -> Api:
            return api

        @staticmethod
        def run(command: list[str], **kwargs: Any) -> Any:
            return subprocess.CompletedProcess(
                command, int(problem == "focused"), "fixture", ""
            )

    monkeypatch.setattr(runner, "RegionalLiveFixture", Factory)
    monkeypatch.setattr(
        runner, "predecessor_path", lambda *a: ("previous", tmp_path / "previous")
    )
    monkeypatch.setattr(
        runner,
        "predecessor_evidence",
        lambda *a, **kwargs: {"valid": True, "case_id": "previous", **kwargs},
    )
    settings = runner.Settings(
        SimpleNamespace(namespace="fixture"), "node-a", site, "image", 60
    )
    result = runner.read_only_preflight(settings, tmp_path)
    assert len(result["errors"]) == (0 if problem is None else 1)
    assert result["identity"] == identity
    assert result["predecessor"]["release_id"] == "release-a"
    assert not any(
        args[0] in {"create", "delete", "patch", "apply"} for args in calls
    ), "preflight must remain read-only"
    assert (
        runner.plan_details(settings, result)["preflight_identity"]["node_uid"]
        == "uid-a"
    )


def test_configuration_and_unknown_binding_refuse_before_resource_actions(
    tmp_path: Path, monkeypatch: Any
) -> None:
    regional = SimpleNamespace(environment=lambda: {"CLUSTER": "cluster-a"})
    monkeypatch.setattr(runner, "settings_from_arguments", lambda args: regional)
    args = runner.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--node",
            "node-a",
            "--site-file",
            str(tmp_path / "site"),
            "--host-probe-image",
            "image",
            "--predecessor-evidence",
            str(tmp_path / "previous"),
        ]
    )
    settings = runner.configure(args)
    assert settings.predecessor_file == tmp_path / "previous"
    assert settings.environment()["GPU_FAULT_NET007_OUTAGE_SECONDS"] == "60"
    args.outage_seconds = 1
    with pytest.raises(runner.RegionalFixtureError, match="outside"):
        runner.configure(args)
    with pytest.raises(runner.RegionalFixtureError, match="required"):
        runner.case_binding(SimpleNamespace(evidence_identity=lambda: {}), tmp_path)
    monkeypatch.setattr(runner, "predecessor_path", lambda *a: (None, None))
    with pytest.raises(runner.RegionalFixtureError, match="unknown"):
        runner.case_binding(
            SimpleNamespace(
                evidence_identity=lambda: {"release_id": "r", "cluster_id": "c"}
            ),
            tmp_path,
        )


def test_deadman_ignores_nonjson_logs_but_refuses_widened_webhook(
    monkeypatch: Any,
) -> None:
    class Api(Regional):
        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            result = super().kubectl(plane, *args, **kwargs)
            if args[0] == "logs":
                return "not-json\n" + result
            if (
                args[:2] == ("get", "validatingwebhookconfiguration")
                and args[-1] == "json"
            ):
                value = json.loads(result)
                value["webhooks"][0]["timeoutSeconds"] = 5
                return json.dumps(value)
            return result

    api = Api()
    outage = fixture(api)
    outage.arm_deadman()
    with pytest.raises(runner.RegionalFixtureError, match="scoping"):
        outage.open()
    assert not outage.webhook_created, "unsafe webhook must be removed"
    assert any(kind == "job" for kind, name in api.resources), (
        "deadman remains until resource cleanup is explicitly confirmed"
    )
    assert not any(outage.cleanup().values()), "all owned resources must be removed"


def test_outage_cleanup_collects_independent_errors_and_retries(
    monkeypatch: Any,
) -> None:
    api = Regional()
    outage = fixture(api)
    outage.arm_deadman()
    original = runner.delete_owned_resource
    calls = []

    def remove(kind: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(kind)
        if kind == "job":
            raise RuntimeError("job deletion unavailable")
        return original(kind, *args, **kwargs)

    monkeypatch.setattr(runner, "delete_owned_resource", remove)
    with pytest.raises(runner.RegionalFixtureError, match="job cleanup"):
        outage.cleanup()
    assert set(calls) == {"job", "clusterrolebinding", "clusterrole", "serviceaccount"}
    monkeypatch.setattr(runner, "delete_owned_resource", original)
    assert not any(outage.cleanup().values()), "retry must clean the retained owned Job"
    assert api.resources == {}


def test_outage_cleanup_waits_for_final_absence_observation(monkeypatch: Any) -> None:
    clock = Clock()
    monkeypatch.setattr(runner, "time", clock)
    outage = fixture(Regional())
    readings = iter([{"job": True}, {"job": False}])
    monkeypatch.setattr(outage, "residuals", lambda: next(readings))
    assert outage.cleanup() == {"job": False}
    assert clock.sleeps == [2]


@pytest.mark.parametrize(
    "problem", ["logs", "probe", "probe-residual", "node", "provider"]
)
def test_case_cleanup_failures_are_recorded_without_suppressing_other_cleanup(
    tmp_path: Path, monkeypatch: Any, problem: str
) -> None:
    calls = []

    class Api(Regional):
        log_reads = 0

        def node_snapshot(self, node: str) -> dict[str, Any]:
            calls.append("node")
            if problem == "node":
                raise RuntimeError("node unavailable")
            return {
                "ready": "True",
                "unschedulable": False,
                "taints": [],
                "ownership_annotations": {},
            }

        def provider_events(self, *args: Any) -> list[Any]:
            calls.append("provider")
            if problem == "provider":
                raise RuntimeError("provider unavailable")
            return []

        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            if args[0] == "logs":
                self.log_reads += 1
                if problem == "logs" and self.log_reads > 1:
                    raise RuntimeError("logs unavailable")
            return super().kubectl(plane, *args, **kwargs)

    api = Api()
    previous, path = runner.predecessor_path(tmp_path, runner.CASE_ID, "")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"case_id": previous, "verdict": "PASS", **api.evidence_identity()})
    )

    class Collector:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def create(self) -> None:
            calls.append("create")

        def snapshot(self) -> dict[str, Any]:
            return {"efa_inventory": {"devices": []}}

        def cleanup(self) -> dict[str, bool]:
            calls.append("probe")
            if problem == "probe":
                raise RuntimeError("probe cleanup unavailable")
            return {"pod": problem == "probe-residual"}

    monkeypatch.setattr(runner, "RegionalLiveFixture", lambda settings: api)
    monkeypatch.setattr(runner, "CollectorAcceptanceFixture", Collector)
    monkeypatch.setattr(
        runner,
        "read_only_preflight",
        lambda *args: {
            **runner.case_binding(api, tmp_path),
            "executor": {"service_account": "executor", "image": "image"},
        },
    )
    settings = runner.Settings(
        SimpleNamespace(namespace="gpu-ns", cluster_id="cluster-a"),
        "node-a",
        tmp_path / "site",
        "image",
        60,
    )
    assert (
        runner.execute_case(
            settings, tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
        )
        == 1
    )
    result = json.loads(
        (tmp_path / "cases" / runner.CASE_ID / f"{runner.CASE_ID}.json").read_text()
    )
    assert "no bound EFA" in result["error"]
    assert result["cleanup"]["errors"], "cleanup uncertainty must be recorded"
    assert set(calls) >= {"probe", "node", "provider"}
    assert api.resources == {}, (
        "probe/final-read failure must not suppress outage cleanup"
    )
