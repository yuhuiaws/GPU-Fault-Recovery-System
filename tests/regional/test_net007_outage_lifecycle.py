"""NET-007 admission and rollback ownership through fake public transports."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kubernetes import client

from scripts.e2e.regional import run_net007_transient_api_outage as runner
from scripts.e2e.regional.probes import net007_deadman as probe


class Regional:
    def __init__(self, *, acknowledge: bool = True) -> None:
        self.settings = SimpleNamespace(namespace="gpu-ns")
        self.resources: dict[tuple[str, str], dict[str, Any]] = {}
        self.acknowledge = acknowledge
        self.fail_delete = False
        self.lose_create_ack = False
        self.deletions: list[str] = []

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "release-test", "cluster_id": "cluster-a"}

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        assert plane == "gpu"
        assert kwargs.get("check", True), "outage transport errors must not be ignored"
        verb = args[0]
        if verb == "create":
            resource = json.loads(kwargs["input_text"])
            metadata = resource["metadata"]
            metadata.update(uid=f"uid-{resource['kind']}", resourceVersion="1")
            key = (resource["kind"].lower(), metadata["name"])
            assert key not in self.resources, "creation cannot adopt existing resources"
            self.resources[key] = resource
            if key[0] == "validatingwebhookconfiguration" and self.lose_create_ack:
                raise RuntimeError("webhook creation acknowledgement lost")
            return "created"
        if verb == "wait":
            return "ready"
        if verb == "logs":
            if not self.acknowledge:
                return '{"state":"STARTING"}'
            job = next(
                item for (kind, _), item in self.resources.items() if kind == "job"
            )
            env = {
                item["name"]: item["value"]
                for item in job["spec"]["template"]["spec"]["containers"][0]["env"]
            }
            return json.dumps(
                {
                    "state": "ARMED",
                    "run_id": env["NET007_RUN_ID"],
                    "restore_at": float(env["NET007_RESTORE_AT"]),
                }
            )
        if verb == "get":
            resource = self.resources.get((args[1], args[2]))
            if resource is None:
                return ""
            if args[-1] == "jsonpath={.metadata}":
                return json.dumps(resource["metadata"])
            if args[-1] == "json":
                return json.dumps(resource)
            return args[2]
        if verb == "delete":
            assert args[1] == "--raw"
            options = json.loads(kwargs["input_text"])
            uid = options["preconditions"]["uid"]
            key = next(
                key
                for key, resource in self.resources.items()
                if resource["metadata"]["uid"] == uid
            )
            if key[0] == "validatingwebhookconfiguration" and self.fail_delete:
                raise RuntimeError("webhook removal uncertain")
            assert options["preconditions"]["resourceVersion"] == "1"
            self.deletions.append(key[0])
            del self.resources[key]
            return "deleted"
        raise AssertionError(f"unexpected verb {verb}")


def fixture(regional: Regional) -> runner.OutageFixture:
    return runner.OutageFixture(
        regional,
        names=runner.resource_names("review-007"),
        node="node-a",
        username="system:serviceaccount:gpu-ns:executor",
        image="image@sha256:" + "a" * 64,
        run_id="review-007",
        deadman_seconds=840,
    )


def test_webhook_cannot_open_without_deadman_acknowledgement() -> None:
    regional = Regional(acknowledge=False)
    outage = fixture(regional)
    with pytest.raises(runner.RegionalFixtureError, match="acknowledge"):
        outage.arm_deadman()
    with pytest.raises(runner.RegionalFixtureError, match="armed"):
        outage.open()
    assert not any(
        kind == "validatingwebhookconfiguration" for kind, _ in regional.resources
    ), "an unacknowledged deadman must not authorize the webhook"
    assert not any(outage.cleanup().values()), (
        "cleanup reported unarmed outage resources"
    )
    assert not regional.resources, (
        f"unarmed resources remain: {sorted(regional.resources)}"
    )


@pytest.mark.parametrize("lose_ack", [False, True])
def test_creation_ack_loss_still_cleans_only_owned_resources(lose_ack: bool) -> None:
    regional = Regional()
    outage = fixture(regional)
    outage.arm_deadman()
    regional.lose_create_ack = lose_ack
    if lose_ack:
        with pytest.raises(RuntimeError, match="acknowledgement lost"):
            outage.open()
    else:
        outage.open()
    assert outage.webhook_created, "create intent must survive a lost webhook ACK"
    assert not any(outage.cleanup().values()), "owned outage cleanup reported residuals"
    assert regional.deletions[0] == "validatingwebhookconfiguration"
    assert not regional.resources, (
        f"owned resources remain: {sorted(regional.resources)}"
    )


def test_uncertain_or_foreign_webhook_keeps_its_deadman() -> None:
    regional = Regional()
    outage = fixture(regional)
    outage.arm_deadman()
    outage.open()
    regional.fail_delete = True
    with pytest.raises(RuntimeError, match="uncertain"):
        outage.cleanup()
    assert outage.webhook_created and outage.deadman_armed
    assert not regional.deletions, (
        f"uncertain webhook removal deleted {regional.deletions}"
    )
    regional.fail_delete = False
    regional.resources[("validatingwebhookconfiguration", outage.names["webhook"])][
        "metadata"
    ]["labels"][probe.RUN_LABEL] = "foreign"
    with pytest.raises(RuntimeError, match="ownership"):
        outage.cleanup()
    assert not regional.deletions, (
        f"foreign webhook ownership deleted {regional.deletions}"
    )


def webhook(*, run_id: str = "review-007", uid: str = "uid-1") -> Any:
    return SimpleNamespace(
        metadata=SimpleNamespace(
            uid=uid, resource_version="version-1", labels={probe.RUN_LABEL: run_id}
        )
    )


@pytest.mark.parametrize("foreign", [False, True])
def test_deadman_binds_delete_preconditions_and_checks_absence(foreign: bool) -> None:
    current = webhook(run_id="foreign" if foreign else "review-007")
    deletions: list[Any] = []

    def read(*args: Any, **kwargs: Any) -> Any:
        if deletions:
            raise client.exceptions.ApiException(status=404)
        return current

    api = SimpleNamespace(
        read_validating_webhook_configuration=read,
        delete_validating_webhook_configuration=lambda name, **kwargs: deletions.append(
            kwargs["body"]
        ),
    )
    if foreign:
        with pytest.raises(RuntimeError, match="ownership"):
            probe.remove_webhook(api, name="webhook", run_id="review-007")
        assert not deletions, "the deadman must preserve another run's webhook"
    else:
        assert probe.remove_webhook(api, name="webhook", run_id="review-007"), (
            "owned webhook removal did not confirm absence"
        )
        assert deletions[0].preconditions.uid == "uid-1"
        assert deletions[0].preconditions.resource_version == "version-1"


def test_deadman_retries_transient_failure_after_its_acknowledged_deadline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clock = [1000.0]
    reads = [0]
    deletes: list[float] = []

    def read(*args: Any, **kwargs: Any) -> Any:
        reads[0] += 1
        if reads[0] == 1 or deletes:
            raise client.exceptions.ApiException(status=404)
        if reads[0] == 2:
            raise client.exceptions.ApiException(status=503)
        return webhook()

    monkeypatch.setattr(probe.time, "time", lambda: clock[0])
    monkeypatch.setattr(probe.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        probe.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    api = SimpleNamespace(
        read_validating_webhook_configuration=read,
        delete_validating_webhook_configuration=lambda *a, **k: deletes.append(
            clock[0]
        ),
    )
    probe.watch(api, name="webhook", run_id="review-007", restore_at=1010.0)
    assert deletes == [1012.0]
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert messages == [
        {"state": "ARMED", "run_id": "review-007", "restore_at": 1010.0},
        {"state": "REMOVED", "run_id": "review-007"},
    ]


@pytest.mark.parametrize(
    "failure", [None, "early-success", "replaced-webhook", "release-drift"]
)
def test_outage_hold_requires_its_full_duration_and_continuous_owned_webhook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    clock = [1000.0]
    closed: list[float] = []
    paths: list[Path] = []
    restored: list[str] = []
    duration = runner.verdicts.MIN_OUTAGE_SECONDS
    operations = list(runner.verdicts.EFA_REMEDIATION_STEPS)
    step_index = operations.index("MARK_UNSCHEDULABLE")
    finished = {
        "workflow": {
            "request_id": "workflow-1",
            "status": "SUCCEEDED",
            "official_steps": [{"operation": operation} for operation in operations],
            "step_executions": [
                {
                    "operation": operation,
                    "step_index": index,
                    "status": "SUCCEEDED",
                    "adapter_operation_id": "remote/cmd-1",
                }
                for index, operation in enumerate(operations)
            ],
        },
        "incident": {"state": "RECOVERED"},
    }

    class CaseRegional(Regional):
        def evidence_identity(self) -> dict[str, str]:
            identity = super().evidence_identity()
            if failure == "release-drift" and closed:
                identity["release_id"] = "different-release"
            return identity

        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            if failure == "replaced-webhook" and clock[0] > 1000:
                for (kind, _), item in self.resources.items():
                    if kind == "validatingwebhookconfiguration":
                        item["metadata"]["uid"] = "replacement-uid"
            if (
                args[:2] == ("delete", "--raw")
                and "validatingwebhookconfigurations" in args[2]
            ):
                closed.append(clock[0])
            return super().kubectl(plane, *args, **kwargs)

        def cpu_python(self, source: str, *args: str, **kwargs: Any) -> dict[str, Any]:
            if len(args) > 1:
                workflow = {
                    "request_id": "workflow-1",
                    "status": "SUCCEEDED"
                    if failure == "early-success" and clock[0] > 1000
                    else "RUNNING",
                    "step_executions": [
                        {
                            "step_index": step_index,
                            "operation": "MARK_UNSCHEDULABLE",
                            "status": "WAITING",
                            "details": {"remote_command_id": "cmd-1"},
                        }
                    ],
                }
                return {"matches": [{"workflow": workflow}]}
            return {
                "remote_commands": [
                    {
                        "command_id": "cmd-1",
                        "step_index": 1,
                        "operation": "MARK_UNSCHEDULABLE",
                        "status": "SUCCEEDED" if closed else "WAITING",
                        "result_details": {"retryable_adapter_error": True},
                    }
                ]
            }

        def node_snapshot(self, node: str) -> dict[str, Any]:
            return {
                "ready": "True",
                "unschedulable": False,
                "ownership_annotations": {},
                "taints": [],
            }

        def provider_events(self, *args: Any) -> list[Any]:
            return []

    regional = CaseRegional()
    previous, previous_path = runner.predecessor_path(tmp_path, runner.CASE_ID, "")
    assert previous is not None and previous_path is not None, (
        "NET-007 predecessor must resolve"
    )
    previous_path.parent.mkdir(parents=True, exist_ok=True)
    previous_path.write_text(
        json.dumps(
            {"case_id": previous, "verdict": "PASS", **regional.evidence_identity()}
        )
    )

    class Collector:
        def __init__(self, fixture: Any, *, case_dir: Path, **kwargs: Any) -> None:
            paths.append(case_dir)
            self.unbound = False

        def create(self) -> None:
            return None

        def snapshot(self) -> dict[str, Any]:
            count = 1 if self.unbound and not closed else 2
            return {
                "efa_inventory": {
                    "discovered_count": count,
                    "devices": [
                        {"pci_bdf": f"0000:00:0{index}.0", "driver": "efa"}
                        for index in range(1, count + 1)
                    ],
                }
            }

        def execute(self, verb: str, *args: str, **kwargs: Any) -> dict[str, Any]:
            if verb == "unbind-efa":
                self.unbound = True
            elif verb == "restore-efa":
                self.unbound = False
                restored.append(verb)
            else:
                raise AssertionError(verb)
            return {}

        def cleanup(self) -> dict[str, bool]:
            return {}

    settings = runner.Settings(
        regional=SimpleNamespace(namespace="gpu-ns", cluster_id="cluster-a"),
        node="node-a",
        site_file=tmp_path / "site.yaml",
        host_probe_image="image@sha256:" + "a" * 64,
        outage_seconds=duration,
    )
    monkeypatch.setattr(runner, "RegionalLiveFixture", lambda settings: regional)
    monkeypatch.setattr(runner, "CollectorAcceptanceFixture", Collector)
    monkeypatch.setattr(runner, "run_identity", lambda *a: "review-007")
    monkeypatch.setattr(
        runner,
        "read_only_preflight",
        lambda *a: {
            **runner.case_binding(regional, tmp_path),
            "executor": {
                "service_account": "executor",
                "image": settings.host_probe_image,
            },
        },
    )
    monkeypatch.setattr(runner.base, "latest_node_workflow", lambda *a, **k: finished)
    monkeypatch.setattr(runner.c017, "efa_unbind_errors", lambda *a, **k: [])
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        runner.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    result = runner.execute_case(
        settings, tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )
    assert result == (0 if failure is None else 1)
    assert paths == [tmp_path / "cases" / runner.CASE_ID]
    assert restored == ["restore-efa"]
    result_document = json.loads(
        (tmp_path / "cases" / runner.CASE_ID / f"{runner.CASE_ID}.json").read_text()
    )
    assert result_document["release_id"] == "release-test"
    assert result_document["cluster_id"] == "cluster-a"
    assert result_document["predecessor"]["valid"] is True
    if failure == "release-drift":
        assert any(
            "final identity" in error for error in result_document["cleanup"]["errors"]
        ), "release drift must fail the final evidence contract"
    if failure == "replaced-webhook":
        assert closed == []
        assert any(kind == "job" for kind, _ in regional.resources), (
            "a replaced webhook must retain its independent deadman"
        )
    elif failure == "early-success":
        assert closed == [1005.0]
    else:
        assert closed == [1000.0 + duration]
        assert not regional.resources, (
            f"successful outage left {sorted(regional.resources)}"
        )
