"""The admission check must distinguish a real fence from generic API denial."""

from __future__ import annotations

import copy
import json
import runpy
import sys
from typing import Any

import pytest
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from scripts.e2e.regional.probes import destr008_admission_probe as probe

IDENTITY = {
    "node": "spare-a",
    "uid": "node-uid-a",
    "policy": "gpu-fault-d008-policy",
    "binding": "gpu-fault-d008-binding",
    "marker": "fence-marker-a",
}


def denial(**changes: Any) -> ApiException:
    status = {
        "kind": "Status",
        "status": "Failure",
        "code": 403,
        "reason": "Forbidden",
        "details": {"name": IDENTITY["node"], "kind": "nodes"},
        "message": (
            f"ValidatingAdmissionPolicy {IDENTITY['policy']} with binding "
            f"{IDENTITY['binding']} denied: {probe.DENIAL_PREFIX}{IDENTITY['marker']}"
        ),
        **changes,
    }
    error = ApiException(status=403, reason="Forbidden")
    error.body = json.dumps(status)
    return error


class Api:
    def __init__(self) -> None:
        self.node = client.V1Node(
            metadata=client.V1ObjectMeta(
                name=IDENTITY["node"],
                uid=IDENTITY["uid"],
                resource_version="17",
                annotations={"unrelated": "retained"},
            ),
            spec=client.V1NodeSpec(unschedulable=True),
        )
        self.calls: list[tuple[str, Any, Any]] = []
        self.refusal: ApiException | None = denial()
        self.safe_acknowledged = True
        self.persist_probe = False
        self.replace_after_probe = False

    def read_node(self, name: str, **kwargs: Any) -> client.V1Node:
        self.calls.append(("read", name, kwargs))
        return copy.deepcopy(self.node)

    def patch_node(
        self, name: str, body: list[dict[str, Any]], **kwargs: Any
    ) -> client.V1Node:
        self.calls.append(("patch", copy.deepcopy(body), kwargs))
        result = copy.deepcopy(self.node)
        if body[-1]["path"] == "/spec/unschedulable":
            if self.replace_after_probe:
                self.node.metadata.uid = "replacement-uid"
            if self.refusal is not None:
                raise self.refusal
            result.spec.unschedulable = False
        elif self.safe_acknowledged:
            result.metadata.annotations = copy.deepcopy(body[-1]["value"])
            if self.persist_probe:
                self.node.metadata.annotations = copy.deepcopy(
                    result.metadata.annotations
                )
        return result


def test_probe_requires_both_controls_and_never_persists_a_patch() -> None:
    api = Api()
    result = probe.probe_fence(api, **IDENTITY)
    assert result == {
        "state": "DENYING_ACTIVATION",
        "node": IDENTITY["node"],
        "node_uid": IDENTITY["uid"],
        "policy": IDENTITY["policy"],
        "binding": IDENTITY["binding"],
        "marker": IDENTITY["marker"],
        "safe_dry_run_acknowledged": True,
        "activation_dry_run_denied": True,
        "probe_not_persisted": True,
        "source_sha256": probe.source_identity(),
    }, "both actual dry-run outcomes must enter the bound receipt"
    assert [row[0] for row in api.calls] == ["read", "patch", "patch", "read"], (
        "the successful control must precede the denial and postflight"
    )
    for operation, body, options in api.calls:
        assert options["_request_timeout"] == (5, 10), "every API read/write is bounded"
        if operation == "patch":
            assert options["dry_run"] == "All", "a probe must never persist its update"
            assert body[:2] == [
                {"op": "test", "path": "/metadata/uid", "value": IDENTITY["uid"]},
                {"op": "test", "path": "/metadata/resourceVersion", "value": "17"},
            ], "both controls must target the exact observed node version"
    assert api.node.spec.unschedulable is True, "the spare must remain cordoned"
    assert api.node.metadata.annotations == {"unrelated": "retained"}, (
        "no dry-run annotations may remain on the node"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "Other"},
        {"status": "Success"},
        {"code": True},
        {"code": 422},
        {"reason": "TooManyRequests"},
        {"details": None},
        {"details": {"name": "foreign-node", "kind": "nodes"}},
        {"details": {"name": IDENTITY["node"], "kind": "secrets"}},
        {"message": "RBAC denies patching nodes"},
        {"message": []},
        {"message": f"{IDENTITY['policy']} {IDENTITY['binding']} unrelated"},
        {"message": f"{IDENTITY['policy']} {probe.DENIAL_PREFIX}{IDENTITY['marker']}"},
        {"message": f"{IDENTITY['binding']} {probe.DENIAL_PREFIX}{IDENTITY['marker']}"},
    ],
)
def test_unbound_or_generic_denial_cannot_establish_activation_fence(
    changes: dict[str, Any],
) -> None:
    api = Api()
    api.refusal = denial(**changes)
    with pytest.raises(probe.AdmissionProbeError, match="not issued by"):
        probe.probe_fence(api, **IDENTITY)
    assert [row[0] for row in api.calls] == ["read", "patch", "patch"], (
        "an unproven denial must stop before emitting a successful receipt"
    )


@pytest.mark.parametrize("body", ["not-json", "[]", "x" * 65537, None])
def test_unreadable_denial_is_not_a_fence(body: str | None) -> None:
    error = denial()
    error.body = body
    assert (
        probe.is_fence_denial(
            error, **{k: v for k, v in IDENTITY.items() if k != "uid"}
        )
        is False
    ), "missing, malformed and oversized denial bodies must be refused"


@pytest.mark.parametrize("status", [401, 404, 409, 429, 500])
def test_unrelated_http_status_does_not_prove_activation_refusal(status: int) -> None:
    error = denial()
    error.status = status
    assert (
        probe.is_fence_denial(
            error, **{k: v for k, v in IDENTITY.items() if k != "uid"}
        )
        is False
    ), "conflicts, authentication failures and unavailable APIs are not admission proof"


def test_invalid_admission_validation_status_is_also_recognized() -> None:
    error = denial(code=422, reason="Invalid")
    error.status = 422
    assert probe.is_fence_denial(
        error, **{k: v for k, v in IDENTITY.items() if k != "uid"}
    ), "Kubernetes may report validation rejection as an Invalid Status"


@pytest.mark.parametrize(
    "change", ["uid", "name", "version", "uncordoned", "metadata", "spec"]
)
def test_unknown_or_replaced_node_is_refused_before_any_patch(change: str) -> None:
    api = Api()
    if change == "uid":
        api.node.metadata.uid = "other-uid"
    elif change == "name":
        api.node.metadata.name = "other-node"
    elif change == "version":
        api.node.metadata.resource_version = None
    elif change == "uncordoned":
        api.node.spec.unschedulable = False
    elif change == "metadata":
        api.node.metadata = None
    else:
        api.node.spec = None
    with pytest.raises(probe.AdmissionProbeError, match="bound cordoned"):
        probe.probe_fence(api, **IDENTITY)
    assert len(api.calls) == 1, "an unknown baseline cannot authorize a dry-run update"


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ("safe_acknowledged", "not acknowledged"),
        ("refusal", "accepted a forbidden"),
        ("persist_probe", "persistent node metadata"),
        ("replace_after_probe", "bound cordoned"),
    ],
)
def test_failed_control_or_postflight_never_emits_success(
    setting: str, message: str
) -> None:
    api = Api()
    setattr(
        api, setting, None if setting == "refusal" else setting != "safe_acknowledged"
    )
    with pytest.raises(probe.AdmissionProbeError, match=message):
        probe.probe_fence(api, **IDENTITY)


@pytest.mark.parametrize("value", ["", "space name", "' || true", "a" * 254, None])
def test_invalid_scope_is_rejected_without_any_api_call(value: Any) -> None:
    api = Api()
    with pytest.raises(probe.AdmissionProbeError, match="identity is invalid"):
        probe.probe_fence(api, **{**IDENTITY, "node": value})
    assert api.calls == [], "scope validation must precede all cluster access"


def configure_cli(
    monkeypatch: pytest.MonkeyPatch, api: Api, *, host: str, verify: bool
) -> list[str]:
    def load(**kwargs: Any) -> None:
        configuration = kwargs["client_configuration"]
        configuration.host = host
        configuration.verify_ssl = verify

    monkeypatch.setattr(config, "load_kube_config", load)
    monkeypatch.setattr(client, "CoreV1Api", lambda api_client: api)
    arguments = ["--kubeconfig", "/fixture/kubeconfig", "--context", "fixture-only"]
    for name, value in IDENTITY.items():
        arguments.extend([f"--{name}", value])
    return arguments


def test_cli_prints_only_the_bound_complete_receipt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = Api()
    arguments = configure_cli(
        monkeypatch, api, host="https://fixture.invalid", verify=True
    )
    assert probe.main(arguments) == 0, "verified dry-run controls must succeed"
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "DENYING_ACTIVATION", (
        "the public entry must emit its actual proof"
    )
    assert result["node_uid"] == IDENTITY["uid"], (
        "the result must bind the observed node"
    )


@pytest.mark.parametrize(
    ("host", "verify"),
    [("http://fixture.invalid", True), ("https://fixture.invalid", False)],
)
def test_cli_requires_https_and_certificate_verification_before_cluster_io(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    host: str,
    verify: bool,
) -> None:
    api = Api()
    arguments = configure_cli(monkeypatch, api, host=host, verify=verify)
    assert probe.main(arguments) == 1, "an insecure API cannot provide admission proof"
    assert api.calls == [], "TLS refusal must precede the first API request"
    assert json.loads(capsys.readouterr().out) == {
        "state": "FAILED",
        "error_type": "AdmissionProbeError",
    }, "failure output must not contain connection or credential material"


def test_cli_never_prints_a_credential_provider_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = configure_cli(
        monkeypatch, Api(), host="https://fixture.invalid", verify=True
    )

    def fail(**kwargs: Any) -> None:
        raise OSError("must-not-print-private-provider-output")

    monkeypatch.setattr(config, "load_kube_config", fail)
    assert probe.main(arguments) == 1, "provider errors must refuse the proof"
    assert json.loads(capsys.readouterr().out) == {
        "state": "FAILED",
        "error_type": "OSError",
    }, "raw provider diagnostics must never enter ordinary proof output"


def test_script_entry_uses_the_same_real_probe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = Api()
    arguments = configure_cli(
        monkeypatch, api, host="https://fixture.invalid", verify=True
    )
    monkeypatch.setattr(sys, "argv", [str(probe.__file__), *arguments])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(probe.__file__), run_name="__main__")
    assert stopped.value.code == 0, (
        "the shipped script must use the verified control path"
    )
    assert json.loads(capsys.readouterr().out)["state"] == "DENYING_ACTIVATION", (
        "script output must not substitute an unexecuted static result"
    )
    assert len(api.calls) == 4, (
        "both controls and postflight must run through the entrypoint"
    )
