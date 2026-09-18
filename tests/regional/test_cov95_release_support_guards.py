from __future__ import annotations

import base64
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import gpu_fault_release
from gpu_fault_release import regional_adot_self_metrics as adot
from gpu_fault_release import regional_aurora_credentials as aurora
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_monitoring_safety as monitoring
from gpu_fault_release import regional_notifications as notifications
from gpu_fault_release import regional_resource_probe as resource_probe
from gpu_fault_release import regional_secret_checks as secret_shapes
from gpu_fault_release import regional_validation_evidence as evidence
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._cov95_release_support import ResourceRelease


def encoded(value: object) -> str:
    return base64.b64encode(json.dumps(value).encode()).decode()


@pytest.mark.parametrize(
    "response,problem",
    [
        ((503, ""), "HTTP 503"),
        ((200, "{"), "invalid response"),
        ((200, json.dumps({"status": "failure"})), "invalid response"),
        ((200, json.dumps({"status": "success", "data": []})), "invalid result vector"),
        (
            (200, json.dumps({"status": "success", "data": {"result": [None]}})),
            "invalid result vector",
        ),
    ],
)
def test_amp_query_rejects_failed_or_malformed_public_transport_responses(
    monkeypatch: pytest.MonkeyPatch, response: tuple[int, str], problem: str
) -> None:
    release = ResourceRelease()
    monkeypatch.setattr(
        adot,
        "export_aws_credentials",
        lambda **_kwargs: {
            "AccessKeyId": "example-access-key",
            "SecretAccessKey": "example-secret-key",
            "SessionToken": "",
        },
    )
    calls = []

    def http(*args: Any, **kwargs: Any) -> tuple[int, str]:
        calls.append((args[0], kwargs["backend"]))
        return response

    monkeypatch.setattr(adot, "http_request", http)
    with pytest.raises(ReleaseError, match=problem):
        adot.amp_instant_query(release, "up")
    assert calls == [("POST", "aws")]
    assert release.runner.calls == []


@pytest.mark.parametrize(
    "payload,problem",
    [([], "not an object"), ({"finished_at": "not-a-timestamp"}, "not a timestamp")],
)
def test_aurora_status_shape_and_timestamp_cannot_be_guessed(
    payload: Any, problem: str
) -> None:
    release = ResourceRelease()
    release.documents[("cpu", "secret", "gpu-fault-aurora")] = {
        "data": {aurora.REFRESH_STATUS_KEY: encoded(payload)}
    }
    with pytest.raises(ReleaseError, match=problem):
        aurora.aurora_refresh_status(release)


def test_aurora_status_rejects_corrupt_bytes_and_marks_missing_time_stale() -> None:
    release = ResourceRelease()
    document = {
        "data": {aurora.REFRESH_STATUS_KEY: base64.b64encode(b"invalid-json").decode()}
    }
    release.documents[("cpu", "secret", "gpu-fault-aurora")] = document
    with pytest.raises(ReleaseError, match="not JSON"):
        aurora.aurora_refresh_status(release)
    document["data"][aurora.REFRESH_STATUS_KEY] = encoded({"status": "ok"})
    status = aurora.aurora_refresh_status(release)
    assert status["age_seconds"] is None
    assert status["stale"] is True
    document["data"][aurora.REFRESH_STATUS_KEY] = encoded(
        {"status": "ok", "finished_at": "2000-01-01T00:00:00"}
    )
    assert aurora.aurora_refresh_status(release)["stale"] is True


@pytest.mark.parametrize(
    "condition",
    [{}, {"type": "Failed", "status": "True", "lastTransitionTime": "invalid"}],
)
def test_refresh_success_cannot_supersede_a_failure_without_comparable_timestamp(
    condition: dict[str, Any],
) -> None:
    status = {"status": "ok", "stale": False, "finished_at": "2026-01-02T00:00:00Z"}
    assert aurora.refresh_status_supersedes_failure(status, [condition]) is False


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("document-count", "document set is incomplete"),
        ("data", "data is malformed"),
        ("aurora", "gpu-fault-aurora is missing"),
        ("active", "control-plane-active is missing"),
        ("node-keys", "node-action-keys is empty"),
    ],
)
def test_cpu_credential_shape_checks_never_accept_incomplete_input(
    fault: str, problem: str
) -> None:
    documents = [
        {
            "data": {
                "postgres-url": "example-reference",
                "master-secret-arn": "example-reference",
            }
        },
        {
            "data": {
                name: base64.b64encode(
                    (f"example-{name}-" + "x" * 32).encode()
                ).decode()
                for name in (
                    "execution-token",
                    "processor-replay-secret",
                    "node-action-secret",
                )
            }
        },
        {"data": {"node-a": "example-reference"}},
    ]
    if fault == "document-count":
        documents.pop()
    elif fault == "data":
        documents[0]["data"] = ["invalid"]
    elif fault == "aurora":
        documents[0]["data"].pop("postgres-url")
    elif fault == "active":
        documents[1]["data"].pop("execution-token")
    else:
        documents[2]["data"].clear()
    with pytest.raises(ReleaseError, match=problem):
        secret_shapes.cpu_secret_shapes(
            documents, context=["fixture"], namespace="example", requires_node_keys=True
        )


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("missing", "has no data"),
        ("encoding", "not valid base64"),
        ("topic", "configured SNS topic"),
    ],
)
def test_monitoring_configuration_requires_decodable_rules_and_bound_topic(
    fault: str, problem: str
) -> None:
    rules = {"ruleGroupsNamespace": {"data": base64.b64encode(b"groups: []").decode()}}
    manager = {
        "alertManagerDefinition": {
            "data": base64.b64encode(b"topic: expected-topic").decode()
        }
    }
    if fault == "missing":
        rules["ruleGroupsNamespace"].clear()
    elif fault == "encoding":
        rules["ruleGroupsNamespace"]["data"] = base64.b64encode(b"\xff").decode()
    else:
        manager["alertManagerDefinition"]["data"] = base64.b64encode(
            b"topic: other"
        ).decode()
    with pytest.raises(ReleaseError, match=problem):
        monitoring.decode_monitoring_configuration(rules, manager, "expected-topic")


def test_multiple_pending_subscription_records_remain_ambiguous() -> None:
    pending = {
        "Protocol": "email",
        "Endpoint": "ops@example.com",
        "SubscriptionArn": "PendingConfirmation",
    }
    with pytest.raises(ReleaseError, match="duplicate pending"):
        monitoring.email_subscription_summary(
            [pending, dict(pending)], "ops@example.com"
        )


def notification_release() -> ResourceRelease:
    release = ResourceRelease()
    release.config.notifications = SimpleNamespace(
        channel="ses",
        allow_email=True,
        acknowledge_external_alert_channel=False,
        admin_email="ops@example.com",
        email_sender="sender@example.com",
        email_recipients=("ops@example.com",),
        email_subject_prefix="[EXAMPLE]",
    )
    return release


@pytest.mark.parametrize("fault", ["disabled", "no-admin", "no-sender"])
def test_notification_configuration_refuses_missing_addresses_before_transport(
    fault: str,
) -> None:
    release = notification_release()
    if fault == "disabled":
        release.config.notifications.allow_email = False
        notifications.ensure_notification_secret(release)
    else:
        if fault == "no-admin":
            release.config.notifications.admin_email = None
        else:
            release.config.notifications.email_sender = None
        with pytest.raises(ReleaseError, match="addresses are missing"):
            notifications.ensure_notification_secret(release)
    assert release.runner.calls == []


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("addresses", "addresses are missing"),
        ("unverified", "identity is not verified"),
        ("disabled", "sending is disabled"),
        ("mismatch", "declared site addresses"),
    ],
)
def test_ses_channel_validation_rejects_unproved_delivery_configuration(
    fault: str, problem: str
) -> None:
    release = notification_release()
    if fault == "addresses":
        release.config.notifications.email_recipients = ()
    values = {
        "email-sender": "different@example.com"
        if fault == "mismatch"
        else "sender@example.com",
        "email-recipients": "ops@example.com",
        "email-subject-prefix": "[EXAMPLE]",
        "site-id": "example-site",
        "aws-account-id": "123456789012",
    }
    calls = []

    def aws(arguments: list[str]) -> dict[str, Any]:
        calls.append(arguments)
        if "get-email-identity" in arguments:
            return {"VerifiedForSendingStatus": fault != "unverified"}
        return {"SendingEnabled": fault != "disabled"}

    with pytest.raises(ReleaseError, match=problem):
        notifications.check_notification_channel(
            release,
            aws_json=aws,
            read_secret=lambda _name: {
                "data": {
                    key: base64.b64encode(value.encode()).decode()
                    for key, value in values.items()
                }
            },
            decode_secret=base64.b64decode,
        )
    assert len(calls) == (
        0 if fault == "addresses" else 1 if fault == "unverified" else 2
    )


@pytest.mark.parametrize("fault", ["missing", "public", "schema", "identity", "future"])
def test_quick_evidence_is_rejected_when_private_binding_or_freshness_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    release = ResourceRelease()
    path = tmp_path / "evidence.json"
    value = {
        "schema_version": 1,
        "release_id": release.release_id,
        "release_delivery_sha256": release.config.release_delivery_sha256,
        "site_identity": {
            "site_name": release.config.site_name,
            "aws_region": release.config.aws_region,
            "cpu_eks_arn": release.config.cpu_eks_arn,
            "cluster_ids": ["gpu-a"],
        },
        "verified_at_epoch": int(time.time()),
        "checks": ["control_plane_role_split"],
        "release_state_sha256": evidence.canonical_sha256(release.live_state),
    }
    if fault == "schema":
        value["schema_version"] = 2
    elif fault == "identity":
        value["release_id"] = "foreign"
    elif fault == "future":
        value["verified_at_epoch"] += 1000
    if fault != "missing":
        path.write_text(json.dumps(value))
        path.chmod(0o644 if fault == "public" else 0o600)
    monkeypatch.setenv(evidence.QUICK_VALIDATION_EVIDENCE_ENV, str(path))
    reused, reason = evidence.quick_validation_evidence(release)
    assert reused is None
    assert reason, "rejected quick-validation evidence must explain its refusal"
    assert release.runner.calls == []


def test_inventory_import_refuses_incomplete_generated_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "deploy/control-plane/regional/cleanup-inventory.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"cpu": {"resources": []}, "gpu": {"resources": []}}))
    monkeypatch.setattr(gpu_fault_release, "repository_root", lambda: tmp_path)
    spec = importlib.util.spec_from_file_location("cov95_inventory", inventory.__file__)
    assert spec is not None and spec.loader is not None
    with pytest.raises(RuntimeError, match="deployment roles are incomplete"):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_resource_probe_refuses_invalid_scope_and_translates_transport_timeout() -> (
    None
):
    with pytest.raises(ValueError, match="scope is invalid"):
        resource_probe.ResourceRef("pod", "Pod", "name", "bad/namespace")

    class Runner:
        def probe_output(self, *_args: Any, **_kwargs: Any) -> tuple[int, str, str]:
            raise TimeoutError("read timed out")

    observation = resource_probe.probe_resource(
        Runner(),
        ["kubectl"],
        resource_probe.ResourceRef("pod", "Pod", "owned", "example"),
    )
    assert observation.state is resource_probe.ProbeState.ERROR
    with pytest.raises(ReleaseError, match="cannot read pod"):
        observation.exists()
