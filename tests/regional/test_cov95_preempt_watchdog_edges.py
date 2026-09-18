from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import HTTPError

from scripts.e2e.regional import preempt037_verdicts as verdicts
from scripts.e2e.regional.probes import preempt037_dispatcher_watchdog as probe
from tests.regional.test_preempt037_watchdog import IMAGE, deployment


@pytest.mark.parametrize(
    "failure", ["version", "baseline", "container", "reference", "duplicate"]
)
def test_watchdog_refuses_ambiguous_or_unbound_restore_targets(failure) -> None:
    current = deployment("false")
    baseline = {"present": True, "value": "true", "image": IMAGE}
    container = current.spec.template.spec.containers[0]
    if failure == "version":
        current.metadata.resource_version = ""
    elif failure == "baseline":
        baseline["present"] = 1
    elif failure == "container":
        container.name = "different"
    elif failure == "reference":
        container.env[-1].value_from = {"secretKeyRef": {}}
    else:
        container.env.append(container.env[-1])
    with pytest.raises(RuntimeError, match="version|baseline|container|ambiguous"):
        probe.deployment_patch(current, uid="uid-1", baseline=baseline)


def test_restore_does_not_trust_an_acknowledged_but_ineffective_patch() -> None:
    patches = []
    api = SimpleNamespace(
        read_namespaced_deployment=lambda *_a, **_k: deployment("false"),
        patch_namespaced_deployment=lambda *args, **_k: patches.append(args[2]),
    )
    with pytest.raises(RuntimeError, match="restoration was not confirmed"):
        probe.restore(
            api, "fixture", "uid-1", {"present": True, "value": "true", "image": IMAGE}
        )
    assert len(patches) == 1, (
        "a failed readback must not trigger an unbounded write loop"
    )


def install_watchdog_environment(monkeypatch, *, restore_at="1000"):
    monkeypatch.setenv("PREEMPT037_RESTORE_AT", restore_at)
    monkeypatch.setenv("PREEMPT037_NAMESPACE", "fixture")
    monkeypatch.setenv("PREEMPT037_DEPLOYMENT_UID", "uid-1")
    monkeypatch.setenv(
        "PREEMPT037_BASELINE",
        json.dumps({"present": True, "value": "true", "image": IMAGE}),
    )
    monkeypatch.setattr(probe.config, "load_incluster_config", lambda: None)


@pytest.mark.parametrize("deadline", ["nan", "inf", "0", "-1"])
def test_invalid_watchdog_deadline_cannot_open_an_api_client(
    deadline, monkeypatch
) -> None:
    install_watchdog_environment(monkeypatch, restore_at=deadline)
    calls = []
    monkeypatch.setattr(probe.client, "AppsV1Api", lambda: calls.append(True))
    with pytest.raises(RuntimeError, match="deadline is invalid"):
        probe.main()
    assert calls == [], "unknown deadlines must stop before any API access"


def test_watchdog_must_be_armed_before_the_window_opens(monkeypatch, capsys) -> None:
    install_watchdog_environment(monkeypatch)
    monkeypatch.setattr(
        probe.client,
        "AppsV1Api",
        lambda: SimpleNamespace(
            read_namespaced_deployment=lambda *_a, **_k: deployment("false")
        ),
    )
    with pytest.raises(RuntimeError, match="before the watchdog was armed"):
        probe.main()
    assert capsys.readouterr().out == "", "unsafe startup must not publish ARMED"


@pytest.mark.parametrize("failure", ["service", "timeout", "http"])
def test_watchdog_retry_budget_ends_without_claiming_restoration(
    failure, monkeypatch, capsys
) -> None:
    install_watchdog_environment(monkeypatch)
    state = SimpleNamespace(now=1000.0, reads=0)

    def read(*_args, **kwargs):
        assert kwargs["_request_timeout"] == (5, 10), (
            "each read must retain its own transport bound"
        )
        state.reads += 1
        if state.reads == 1:
            return deployment("true")
        if failure == "service":
            raise ApiException(status=503)
        if failure == "http":
            raise HTTPError("fixture transport unavailable")
        raise TimeoutError("fixture read timeout")

    monkeypatch.setattr(
        probe.client,
        "AppsV1Api",
        lambda: SimpleNamespace(read_namespaced_deployment=read),
    )
    monkeypatch.setattr(
        probe,
        "time",
        SimpleNamespace(
            time=lambda: state.now,
            monotonic=lambda: state.now,
            sleep=lambda seconds: setattr(state, "now", state.now + seconds),
        ),
    )
    with pytest.raises(RuntimeError, match="could not confirm restoration"):
        probe.main()
    assert state.now == 1120 and state.reads == 62, (
        "retry duration must stop at the 120-second budget"
    )
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [message["state"] for message in messages] == ["ARMED"], (
        "unconfirmed restoration must never emit RESTORED"
    )


@pytest.mark.parametrize("duration", ["0m", "-1", "invalid", "1d"])
def test_alert_rule_rejects_unknown_or_nonpositive_hold_duration(duration) -> None:
    rules = {
        "groups": [
            {
                "rules": [
                    {
                        "alert": verdicts.STALLED_ALERT,
                        "expr": f"time() - max by (control_plane_cluster, region) ({verdicts.DISPATCH_METRIC}) > 300",
                        "for": duration,
                    }
                ]
            }
        ]
    }
    with pytest.raises(ValueError, match="positive supported duration"):
        verdicts.stall_rule_parameters(json.dumps(rules))


def test_rule_and_window_need_explicit_evidence_fields() -> None:
    errors = verdicts.rule_errors({}, "")
    assert len(errors) == 3, "metric, runbook and severity are independent requirements"
    assert len(verdicts.window_errors({})) == 2, (
        "a missing setting and replica set cannot pass"
    )
    with pytest.raises(ValueError, match="no replica metric series"):
        verdicts.stalled([], now=1000, threshold_seconds=300)


@pytest.mark.parametrize(
    "sample",
    [
        {},
        {"observed_epoch": True, "stalled": True, "periodic_alive": True},
        {"observed_epoch": float("inf"), "stalled": True, "periodic_alive": True},
        {"observed_epoch": 1000, "stalled": "true", "periodic_alive": True},
        {"observed_epoch": 1000, "stalled": True, "periodic_alive": None},
    ],
)
def test_unknown_stall_observations_cannot_prove_a_continuous_alert(sample) -> None:
    errors = verdicts.stall_timeline_errors([sample], for_seconds=300)
    assert len(errors) == 1 and ("invalid" in errors[0] or "unknown" in errors[0]), (
        "typed timestamp and liveness observations are required before duration arithmetic"
    )
