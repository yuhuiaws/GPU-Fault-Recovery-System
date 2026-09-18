from __future__ import annotations

import json
import runpy
import subprocess
import sys

import pytest

from gpu_fault.admin import execution
from gpu_fault.admin.node_key_custody import (
    ProvisionCustody,
    provisioning_input_identity,
)
from gpu_fault.admin.node_key_custody_models import CustodyError, canonical
from tests.deploy._node_action_key_api import NAMESPACE, encoded, secret
from tests.deploy._node_key_custody_support import (
    MASTER,
    PROVISION,
    PROVISION_PATH,
    ProvisionFixture,
)
from tests.regional._cov95_identity_support import offline_guard as offline_guard


def arguments(fixture):
    return [
        str(PROVISION_PATH),
        "--gpu-context",
        "gpu-context",
        "--namespace",
        NAMESPACE,
        "--cluster-id",
        "cluster-a",
        "--hyperpod-cluster",
        "hyperpod-a",
        "--master-file",
        str(fixture.master_file),
        "--secret-name",
        "gpu-fault-node-action-keys",
        "--cpu-context",
        "cpu-context",
        "--cpu-namespace",
        NAMESPACE,
    ]


@pytest.fixture
def fixture(tmp_path):
    return ProvisionFixture(tmp_path)


def test_provisioning_main_consumes_explicit_custody_paths_and_emits_only_safe_counts(
    fixture, monkeypatch, capsys
):
    session = fixture.session()
    argv = arguments(fixture) + [
        "--custody-request",
        str(fixture.root / "request"),
        "--custody-trust-sha256",
        fixture.authorities.trust_pin,
    ]
    original = PROVISION.provision
    loads = []

    def load(path, pin):
        loads.append((path, pin))
        return session

    def provision(*args, **kwargs):
        return original(*args, **kwargs, runner=fixture.run)

    monkeypatch.setattr(PROVISION, "load_node_key_custody_request", load)
    monkeypatch.setattr(PROVISION, "provision", provision)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv(
        "GPU_FAULT_NODE_KEY_EXPECTED_NODES_JSON", json.dumps(fixture.binding.nodes)
    )
    assert PROVISION.main() == 0
    assert loads == [(fixture.root / "request", fixture.authorities.trust_pin)]
    output = capsys.readouterr()
    assert "provisioned 2 node action key(s)" in output.out
    assert "synchronized" in output.out
    assert not output.err, "successful provisioning must not emit stderr diagnostics"
    assert MASTER.decode() not in output.out


@pytest.mark.parametrize(
    "expected", ['["node-a","node-b"]', '{"node-a":"uid-node-a","node-b":"uid-node-b"}']
)
def test_cli_accepts_valid_name_or_uid_inventory_without_invoking_real_io(
    fixture, monkeypatch, expected
):
    observed = []
    monkeypatch.setenv("GPU_FAULT_NODE_KEY_EXPECTED_NODES_JSON", expected)
    monkeypatch.setattr(sys, "argv", arguments(fixture))
    monkeypatch.setattr(
        PROVISION, "provision", lambda *args, **kwargs: observed.append(kwargs) or 2
    )
    assert PROVISION.main() == 0
    assert set(observed[0]["expected_nodes"]) == {"node-a", "node-b"}


@pytest.mark.parametrize(
    "expected",
    [
        "",
        "not-json",
        "null",
        "[]",
        "{}",
        '["../invalid"]',
        '["node-a","node-a"]',
        '{"node-a":""}',
        '{"node-a":"uid","node-b":"uid"}',
        '{"node-a":"one","node-a":"two"}',
    ],
)
def test_invalid_expected_inventory_is_rejected_before_dispatch(
    fixture, monkeypatch, expected
):
    monkeypatch.setenv("GPU_FAULT_NODE_KEY_EXPECTED_NODES_JSON", expected)
    monkeypatch.setattr(sys, "argv", arguments(fixture))
    monkeypatch.setattr(
        PROVISION,
        "provision",
        lambda *a, **k: pytest.fail("invalid inventory reached provisioning"),
    )
    assert PROVISION.main() == 1


@pytest.mark.parametrize("flag", ["request", "pin", "input-pin"])
def test_custody_cli_never_implicitly_trusts_a_request_file(fixture, monkeypatch, flag):
    argv = arguments(fixture)
    argv += (
        ["--custody-request", str(fixture.root / "request")]
        if flag == "request"
        else ["--custody-trust-sha256", "a" * 64]
        if flag == "pin"
        else ["--custody-input-sha256", "a" * 64]
    )
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        PROVISION,
        "provision",
        lambda *a, **k: pytest.fail("unbound custody request was dispatched"),
    )
    assert PROVISION.main() == 1


@pytest.mark.parametrize("pin", ["valid", "wrong-content", "wrong-trust"])
def test_pinned_cli_handoff_loads_the_request_before_fake_provisioning(
    fixture, monkeypatch, pin
):
    session = fixture.session()
    request_path = fixture.root / "pinned-request.json"
    request_path.write_bytes(canonical(session.inputs))
    trust = fixture.authorities.trust_pin
    identity = provisioning_input_identity(request_path, trust)
    original_load = ProvisionCustody.load
    original_provision = PROVISION.provision

    def load(path, trust_sha256, **kwargs):
        return original_load(
            path,
            trust_sha256,
            crypto_factory=lambda *args: fixture.authorities.crypto(),
            **kwargs,
        )

    monkeypatch.setattr(ProvisionCustody, "load", load)
    monkeypatch.setattr(
        PROVISION,
        "provision",
        lambda *args, **kwargs: original_provision(*args, **kwargs, runner=fixture.run),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        arguments(fixture)
        + [
            "--custody-request",
            str(request_path),
            "--custody-trust-sha256",
            "0" * 64 if pin == "wrong-trust" else trust,
            "--custody-input-sha256",
            "0" * 64 if pin == "wrong-content" else identity,
        ],
    )
    assert PROVISION.main() == int(pin != "valid")
    assert bool(fixture.api.state["write_attempts"]) == (pin == "valid")
    assert bool(list(session.directory.glob("*.chain.json"))) == (pin == "valid")


def test_entrypoint_uses_only_injected_kubernetes_io_and_no_custody_claim(
    fixture, monkeypatch, capsys
):
    fixture.api.state["secrets"] = {
        plane: secret(plane, {"node-a": encoded("a"), "node-b": encoded("b")})
        for plane in ("cpu", "gpu")
    }
    monkeypatch.setattr(execution, "run_command", fixture.api.run)
    monkeypatch.setattr(sys, "argv", arguments(fixture))
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(PROVISION_PATH), run_name="__main__")
    assert caught.value.code == 0
    assert fixture.api.state["writes"] == []
    assert "custody" not in capsys.readouterr().out


@pytest.mark.parametrize("cpu", ["kubeconfig", "none"])
def test_cpu_scope_construction_preserves_explicit_or_absent_cpu_binding(
    fixture, monkeypatch, cpu
):
    argv = arguments(fixture)
    position = argv.index("--cpu-context")
    del argv[position : position + 2]
    position = argv.index("--gpu-context")
    del argv[position : position + 2]
    if cpu == "kubeconfig":
        argv += ["--cpu-kubeconfig", str(fixture.root / "cpu-config")]
    observed = []
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        PROVISION, "provision", lambda *args, **kwargs: observed.append(args) or 2
    )
    assert PROVISION.main() == 0
    assert (observed[0][1] is None) is (cpu == "none")


def test_unexpected_private_error_is_not_echoed_by_main(fixture, monkeypatch, capsys):
    sentinel = "synthetic-private-message-" * 5

    def fail(*args, **kwargs):
        raise RuntimeError(sentinel)

    monkeypatch.setattr(sys, "argv", arguments(fixture))
    monkeypatch.setattr(PROVISION, "provision", fail)
    assert PROVISION.main() == 1
    assert sentinel not in capsys.readouterr().err


@pytest.mark.parametrize("scope", ["no-cpu", "gpu-name", "cpu-name"])
def test_custody_rejects_unbound_key_source_scopes_before_any_io(fixture, scope):
    session = fixture.session()
    gpu = PROVISION.Scope(
        ("kubectl",),
        NAMESPACE,
        "wrong" if scope == "gpu-name" else "gpu-fault-node-action-keys",
    )
    cpu = (
        None
        if scope == "no-cpu"
        else PROVISION.Scope(
            ("kubectl",),
            NAMESPACE,
            "wrong" if scope == "cpu-name" else "gpu-fault-node-action-keys",
        )
    )
    with pytest.raises(CustodyError, match="sources"):
        PROVISION.provision(
            gpu,
            cpu,
            cluster_id="cluster-a",
            hyperpod_cluster="hyperpod-a",
            master_file=fixture.master_file,
            runner=lambda *a, **k: pytest.fail("scope check followed I/O"),
            custody=session,
        )


@pytest.mark.parametrize("target", ["namespace", "master"])
def test_custody_read_errors_never_mean_absence_or_issue_a_start(fixture, target):
    session = fixture.session()
    original = fixture.run

    def runner(arguments, **kwargs):
        if (
            ("namespace" in arguments)
            if target == "namespace"
            else (fixture.binding.master.secret_name in arguments)
        ):
            return subprocess.CompletedProcess(
                arguments, 1, "synthetic-sensitive-output", ""
            )
        return original(arguments, **kwargs)

    with pytest.raises(CustodyError, match="could not be read"):
        fixture.provision(session, runner=runner)
    assert fixture.events == []
    assert not list(session.directory.iterdir()), (
        "failed custody identity reads must not create receipts"
    )
