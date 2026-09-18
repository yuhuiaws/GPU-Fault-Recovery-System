from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_observability_drift as DRIFT
from gpu_fault_release import regional_release_validation as VALIDATION
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import (
    build_execution_plan,
    diff_from_changed,
)


class Monitoring:
    def __init__(self) -> None:
        self.diff_code = 0
        self.ready = 1
        self.calls: list[list[str]] = []
        self.definitions = {
            "rules": {"groups": []},
            "manager": {"alertmanager_config": {}},
        }
        self.status = "ACTIVE"
        self.error = ""
        self.drift = False
        self.config = SimpleNamespace(
            namespace="gpu-fault-system",
            aws_region="us-east-1",
            health=SimpleNamespace(
                amp_workspace_id="workspace",
                amp_rule_namespace="rules",
                sns_topic_arn="arn:sns:topic",
            ),
        )
        self.adot_image = "collector@sha256:" + "a" * 64
        self.runner = SimpleNamespace(dry_run=False, probe_output=self.probe)

    def _observability_drift(self) -> bool:
        return self.drift

    def _cpu(self, *args: str) -> list[str]:
        return ["fake-kubernetes-api", *args]

    def _get_json(self, args: list[str]) -> dict:
        self.calls.append(args)
        assert "get" in args and "--ignore-not-found" in args
        return {"status": {"availableReplicas": self.ready}} if self.ready else {}

    def probe(self, args: list[str], **kwargs) -> tuple[int, str, str]:
        self.calls.append(args)
        if "diff" in args:
            rendered = Path(args[args.index("-f") + 1])
            assert rendered.stat().st_mode & 0o777 == 0o600
            assert self.adot_image in rendered.read_text()
            assert kwargs["timeout_seconds"] == 60
            return self.diff_code, "private-diff", "private-error"
        assert args[:2] == ["aws", "amp"]
        if self.error:
            return 1, "", self.error
        name = args[args.index("--name") + 1] if "--name" in args else "manager"
        if name not in self.definitions:
            return 1, "", "ResourceNotFoundException"
        document = {
            "status": {"statusCode": self.status},
            "data": base64.b64encode(
                json.dumps(self.definitions[name]).encode()
            ).decode(),
        }
        root = "alertManagerDefinition" if name == "manager" else "ruleGroupsNamespace"
        return 0, json.dumps({root: document}), ""


@pytest.fixture
def monitoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Monitoring:
    directory = tmp_path / "deploy/observability"
    directory.mkdir(parents=True)
    (directory / "adot-control-plane.yaml").write_text("image: REPLACE_WITH_ADOT_IMAGE")
    (directory / "amp-rules.yaml").write_text("groups: []")
    (directory / "amp-alertmanager.yaml").write_text("alertmanager_config: {}")
    monkeypatch.setattr(DRIFT, "ROOT", tmp_path)
    monkeypatch.setattr(DRIFT, "render_dataplane_expected_rules", lambda _: None)
    return Monitoring()


def test_unchanged_monitoring_is_proved_read_only(monitoring: Monitoring) -> None:
    assert not DRIFT.observability_drift(monitoring), (
        "current monitoring was reported as drifted"
    )
    assert len(monitoring.calls) == 5
    assert all(
        not {"apply", "create", "delete"} & set(call) for call in monitoring.calls
    ), "a drift probe attempted a persistent mutation"


@pytest.mark.parametrize(
    "drift", ["manifest", "deleted", "rules", "manager", "pending", "unexpected-rules"]
)
def test_every_monitoring_drift_schedules_repair(
    monitoring: Monitoring, drift: str
) -> None:
    if drift == "manifest":
        monitoring.diff_code = 1
    elif drift == "deleted":
        monitoring.ready = 0
    elif drift in {"rules", "manager"}:
        monitoring.definitions.pop(drift)
    elif drift == "pending":
        monitoring.status = "UPDATING"
    else:
        monitoring.definitions[DRIFT.DATAPLANE_EXPECTED_RULE_NAMESPACE] = {"groups": []}
    assert DRIFT.observability_drift(monitoring), f"{drift} did not select repair"


@pytest.mark.parametrize("failure", ["diff", "AccessDenied", "unknown-status"])
def test_unknown_state_does_not_authorize_repair(
    monitoring: Monitoring, failure: str
) -> None:
    if failure == "diff":
        monitoring.diff_code = 2
    elif failure == "unknown-status":
        monitoring.status = "UNKNOWN"
    else:
        monitoring.error = failure
    with pytest.raises(ReleaseError) as caught:
        DRIFT.observability_drift(monitoring)
    assert "private-diff" not in str(caught.value)
    assert "private-error" not in str(caught.value)


def test_post_rollout_drift_is_not_reported_as_success(
    monitoring: Monitoring, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        VALIDATION,
        "validate_release_components",
        lambda *_, **__: calls.append("verify"),
    )
    monitoring.drift = True
    plan = build_execution_plan(diff_from_changed({"observability_drift"}))
    with pytest.raises(ReleaseError, match="after its rollout"):
        VALIDATION.validate_release_quick(monitoring, plan)
    assert calls == []
    monitoring.drift = False
    VALIDATION.validate_release_quick(monitoring, plan)
    assert calls == ["verify"]
