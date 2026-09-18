"""The selected Aurora must be the deployed CPU Store, before any action."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import gpu_fault
from gpu_fault.admin.bootstrap_services import irsa_trust_document
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import aurora_binding as binding
from tests.regional._aurora_binding_support import (
    CPU_ARN,
    CRONJOB,
    MASTER_ARN,
    MODULE_DIGEST,
    NAMESPACE,
    PASSWORD,
    ROLE_ARN,
    SECRET,
    FakeAurora,
    encoded,
)


def test_proof_binds_actual_release_store_and_refresher_without_exporting_secrets() -> (
    None
):
    environment = FakeAurora()
    proof = environment.guard.read()
    assert (
        proof["identity"]["database"]["cluster_resource_id"] == "cluster-unit-resource"
    ), proof
    assert proof["identity"]["cpu"]["cluster_arn"] == CPU_ARN, proof
    assert proof["identity"]["refresher"]["iam"]["role_arn"] == ROLE_ARN, proof
    assert len(environment.probes) == 2, "both enabled CPU roles must prove their Store"
    assert all(p["authenticate"] for p in environment.probes), (
        "pre-action proof requires fresh SQL"
    )
    assert PASSWORD not in json.dumps(proof), "proof must never contain a password"
    assert environment.dsn not in json.dumps(proof), "proof must never contain a DSN"
    assert encoded(environment.dsn) not in json.dumps(proof), (
        "encoding is not redaction"
    )
    assert {call[1] for call in environment.calls if call[0] == "kubectl"} <= {
        "get",
        "config",
        "exec",
    }, environment.calls
    assert not any(call[-1] == "get-secret-value" for call in environment.calls), (
        "AWS secret values are unnecessary"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "cluster-id",
        "cluster-arn",
        "resource-id",
        "endpoint",
        "database",
        "user",
        "port",
        "master-reference",
        "master-account",
        "pending",
        "engine",
        "source-secret",
        "literal-dsn",
        "subpath",
        "optional-source",
        "projection",
        "release-image",
        "release-phase",
        "namespace",
        "context",
        "refresher-secret",
        "refresher-master",
        "refresher-image",
        "refresher-rollout",
        "refresher-command",
        "refresher-override",
        "refresher-sa",
        "refresher-role",
        "refresher-subject",
        "refresher-active",
        "iam-target",
        "iam-trust",
        "iam-association",
    ],
)
def test_mismatch_or_unknown_identity_refuses_read_only_proof(fault: str) -> None:
    environment = FakeAurora()
    cluster_fields = {
        "cluster-id": ("DBClusterIdentifier", "other"),
        "cluster-arn": (
            "DBClusterArn",
            "arn:aws:rds:us-east-1:123456789012:cluster:other",
        ),
        "resource-id": ("DbClusterResourceId", ""),
        "endpoint": ("Endpoint", "foreign.unit.invalid"),
        "database": ("DatabaseName", "other"),
        "user": ("MasterUsername", "other"),
        "port": ("Port", 5433),
        "pending": ("PendingModifiedValues", {"Port": 5433}),
        "engine": ("Engine", "postgres"),
    }
    container = environment.pods["gpu-fault-api-ha"][0]["spec"]["containers"][0]
    cronjob = environment.documents["cronjob", CRONJOB]
    if fault in cluster_fields:
        field, value = cluster_fields[fault]
        environment.cluster[field] = value
    elif fault == "master-reference":
        environment.documents["secret", SECRET]["data"]["master-secret-arn"] = encoded(
            MASTER_ARN + "-other"
        )
    elif fault == "master-account":
        environment.cluster["MasterUserSecret"]["SecretArn"] = MASTER_ARN.replace(
            "111122223333", "123456789012"
        )
    elif fault == "source-secret":
        container["env"][0]["valueFrom"]["secretKeyRef"]["name"] = "foreign"
    elif fault == "literal-dsn":
        container["env"][0] = {"name": "GPU_FAULT_STORE_URL", "value": environment.dsn}
    elif fault == "subpath":
        container["volumeMounts"][0]["subPath"] = "postgres-url"
    elif fault == "optional-source":
        environment.pods["gpu-fault-api-ha"][0]["spec"]["volumes"][0]["secret"][
            "optional"
        ] = True
    elif fault == "projection":
        environment.verified = False
    elif fault.startswith("release-"):
        environment.state["runtime_image" if fault == "release-image" else "phase"] = (
            "foreign"
        )
        environment.documents["configmap", "gpu-fault-regional-release-state"]["data"][
            "state.json"
        ] = json.dumps(environment.state)
    elif fault == "namespace":
        environment.documents["secret", SECRET]["metadata"]["namespace"] = "other"
    elif fault == "context":
        environment.config["clusters"][0]["cluster"]["server"] = (
            "https://foreign.invalid"
        )
    elif fault in {
        "refresher-secret",
        "refresher-master",
        "refresher-rollout",
        "refresher-override",
    }:
        key = {
            "refresher-secret": "GPU_FAULT_AURORA_SECRET",
            "refresher-master": "GPU_FAULT_AURORA_MASTER_SECRET_ARN",
            "refresher-rollout": "GPU_FAULT_AURORA_REFRESH_RESTART_DEPLOYMENTS",
            "refresher-override": "GPU_FAULT_AURORA_SECRET_KEY",
        }[fault]
        environment.change_refresh_env(
            key, "true" if fault == "refresher-rollout" else "foreign"
        )
    elif fault == "refresher-image":
        environment.refresh_container()["image"] = "foreign"
    elif fault == "refresher-command":
        environment.refresh_container()["args"] = ["--restart-deployments"]
    elif fault == "refresher-sa":
        cronjob["spec"]["jobTemplate"]["spec"]["template"]["spec"][
            "serviceAccountName"
        ] = "foreign"
    elif fault == "refresher-role":
        environment.documents["role", CRONJOB]["rules"][0]["resourceNames"] = [
            "foreign"
        ]
    elif fault == "refresher-subject":
        environment.documents["rolebinding", CRONJOB]["subjects"][0]["namespace"] = (
            "foreign"
        )
    elif fault == "refresher-active":
        cronjob["status"] = {"active": [{"uid": "foreign-job"}]}
    elif fault == "iam-target":
        environment.policy["Statement"][0]["Resource"] = MASTER_ARN + "-other"
    elif fault == "iam-trust":
        environment.iam_role["AssumeRolePolicyDocument"]["Statement"][0].pop(
            "Condition"
        )
    elif fault == "iam-association":
        environment.association["namespace"] = "foreign"
    with pytest.raises(binding.BindingError, match="Aurora binding") as failure:
        environment.guard.read()
    assert PASSWORD not in str(failure.value), (
        "failure must never render input credentials"
    )


@pytest.mark.parametrize(
    "suffix",
    [
        "&hostaddr=127.0.0.1",
        "&host=other.invalid",
        "&service=other",
        "&passfile=/tmp/other",
        "&sslmode=require",
        "&options=-csearch_path=other",
    ],
)
def test_libpq_overrides_do_not_bypass_store_target_binding(suffix: str) -> None:
    environment = FakeAurora()
    environment.documents["secret", SECRET]["data"]["postgres-url"] = encoded(
        environment.dsn + suffix
    )
    with pytest.raises(binding.BindingError):
        environment.guard.read()
    assert environment.probes == [], "invalid DSN must fail before a Pod SQL probe"


@pytest.mark.parametrize(
    "field",
    [
        "namespace-uid",
        "cluster-resource",
        "secret-uid",
        "pod-uid",
        "cronjob-uid",
        "role-id",
        "credential",
        "writer",
    ],
)
def test_boundary_revalidation_refuses_identity_drift(field: str) -> None:
    environment = FakeAurora()
    expected = environment.guard.read()
    if field == "namespace-uid":
        environment.documents["namespace", NAMESPACE]["metadata"]["uid"] += "-new"
    elif field == "cluster-resource":
        environment.cluster["DbClusterResourceId"] += "-new"
    elif field == "secret-uid":
        environment.documents["secret", SECRET]["metadata"]["uid"] += "-new"
    elif field == "pod-uid":
        environment.pods["gpu-fault-api-ha"][0]["metadata"]["uid"] += "-new"
    elif field == "cronjob-uid":
        environment.documents["cronjob", CRONJOB]["metadata"]["uid"] += "-new"
    elif field == "role-id":
        environment.iam_role["RoleId"] += "-new"
    elif field == "credential":
        environment.dsn = environment.dsn.replace(PASSWORD, PASSWORD + "_NEW")
        environment.documents["secret", SECRET]["data"]["postgres-url"] = encoded(
            environment.dsn
        )
    else:
        environment.cluster["DBClusterMembers"].reverse()
        for member in environment.cluster["DBClusterMembers"]:
            member["IsClusterWriter"] = not member["IsClusterWriter"]
    with pytest.raises(binding.BindingError, match="changed since approval"):
        environment.guard.read(expected)


def test_bound_refresh_uses_verified_template_and_allows_only_password_progression() -> (
    None
):
    environment = FakeAurora()
    expected = environment.guard.read()
    environment.probes.clear()
    environment.dsn = environment.dsn.replace(PASSWORD, PASSWORD + "_NEW")
    environment.documents["secret", SECRET]["data"]["postgres-url"] = encoded(
        environment.dsn
    )
    job = environment.guard.refresh_job("unit-job", expected)
    assert job["metadata"]["ownerReferences"][0]["uid"] == "CronJob-uid", job
    assert (
        job["spec"]
        == environment.documents["cronjob", CRONJOB]["spec"]["jobTemplate"]["spec"]
    ), "Job must use the just-verified immutable target"
    assert all(p["authenticate"] is False for p in environment.probes), (
        "refresh must work when the old password no longer authenticates"
    )
    environment.change_refresh_env("GPU_FAULT_AURORA_SECRET", "foreign")
    with pytest.raises(binding.BindingError):
        environment.guard.refresh_job("unit-emergency", expected)


def test_verified_irsa_is_supported_and_dual_identity_is_refused() -> None:
    environment = FakeAurora()
    environment.associations = []
    environment.documents["serviceaccount", CRONJOB]["metadata"]["annotations"] = {
        "eks.amazonaws.com/role-arn": ROLE_ARN
    }
    issuer = "oidc.unit.invalid/id/UNIT"
    environment.iam_role["AssumeRolePolicyDocument"] = irsa_trust_document(
        provider_arn=f"arn:aws:iam::111122223333:oidc-provider/{issuer}",
        issuer=issuer,
        namespace=NAMESPACE,
        service_account=CRONJOB,
    )
    proof = environment.guard.read()
    assert proof["identity"]["refresher"]["iam"]["association_id"] is None, proof
    environment.associations = [{"associationId": "ambiguous"}]
    with pytest.raises(binding.BindingError, match="ambiguous"):
        environment.guard.read()


def test_refresh_can_recover_known_notready_consumers_without_accepting_unknown_state() -> (
    None
):
    environment = FakeAurora()
    expected = environment.guard.read()
    for name, pods in environment.pods.items():
        environment.documents["deployment", name]["status"].update(
            readyReplicas=0, availableReplicas=0
        )
        for pod in pods:
            pod["status"]["conditions"][0]["status"] = "False"
    job = environment.guard.refresh_job("unit-recovery", expected)
    assert job["kind"] == "Job", "expired credentials must not block a bound refresh"
    environment.pods["gpu-fault-api-ha"][0]["status"]["conditions"] = []
    with pytest.raises(binding.BindingError, match="readiness"):
        environment.guard.refresh_job("unit-unknown", expected)


def test_failures_are_sanitized_without_swallowing_supervision_loss() -> None:
    environment = FakeAurora()
    environment.failure = RuntimeError(environment.dsn)
    for action in (
        environment.guard.read,
        lambda: environment.guard.refresh_job("unit", {}),
    ):
        with pytest.raises(binding.BindingError) as failure:
            action()
        assert PASSWORD not in str(failure.value), (
            "transport diagnostics must not cross the evidence boundary"
        )
        assert failure.value.__suppress_context__, (
            "tracebacks must not expose the transport exception"
        )
    environment.failure = ProcessSupervisionLost("unit supervisor loss")
    with pytest.raises(ProcessSupervisionLost):
        environment.guard.read()


@pytest.mark.parametrize(
    "fault",
    [
        "",
        "missing-file",
        "wrong-host",
        "foreign-master",
        "old-module",
        "wrong-sql",
        "driver-error",
        "ambient-pg",
    ],
)
def test_actual_cpu_probe_uses_projected_dsn_and_sanitizes_all_failures(
    fault: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import psycopg

    environment = FakeAurora()
    proof = environment.guard.read()
    expected = {
        "database": proof["identity"]["database"],
        "module_digest": MODULE_DIGEST,
        "credential_sha256": hashlib.sha256(environment.dsn.encode()).hexdigest(),
        "authenticate": True,
    }
    monkeypatch.setenv("GPU_FAULT_STORE_URL", environment.dsn)
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", binding.DSN_FILE)
    if fault == "ambient-pg":
        monkeypatch.setenv("PGHOSTADDR", "127.0.0.1")
    monkeypatch.setattr(sys, "argv", ["-", json.dumps(expected)])
    monkeypatch.setattr(
        gpu_fault,
        "module_digest",
        lambda: "foreign" if fault == "old-module" else MODULE_DIGEST,
    )
    monkeypatch.setattr(logging, "disable", lambda _: None)

    def read_text(path: Path, **kwargs: object) -> str:
        if str(path) == binding.MASTER_FILE:
            return MASTER_ARN + ("-foreign" if fault == "foreign-master" else "")
        if fault == "missing-file":
            raise OSError(environment.dsn)
        return (
            environment.dsn.replace("writer.unit.invalid", "foreign.invalid")
            if fault == "wrong-host"
            else environment.dsn
        )

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", lambda path: read_text(path).encode())
    connections = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, query):
            return SimpleNamespace(
                fetchone=lambda: ("foreign", "unit_admin", False)
                if fault == "wrong-sql"
                else ("gpu_fault", "unit_admin", False)
            )

    def connect(**kwargs):
        connections.append(kwargs)
        if fault == "driver-error":
            raise RuntimeError(environment.dsn)
        return Connection()

    monkeypatch.setattr(psycopg, "connect", connect)
    source = binding.pod_probe_source()
    exec(compile(source, "<unit-binding-probe>", "exec"), {})
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["verified"] is (not fault), result
    assert PASSWORD not in output.out + output.err, (
        "CPU probe cannot export credentials on any branch"
    )
    if not fault:
        assert connections[0]["host"] == "writer.unit.invalid", (
            "SQL must target the bound endpoint"
        )
        assert "default_transaction_read_only=on" in connections[0]["options"], (
            "probe may never write or initialize Store schema"
        )
