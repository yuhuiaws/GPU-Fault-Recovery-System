from __future__ import annotations

import json

import pytest

from scripts.e2e.regional import boot_acceptance_runtime as runtime
from tests.regional._cov95_boot_extra_runtime import GreenfieldFixture
from tests.regional._cov95_boot_extra_safety import (
    boot_extra_isolation as boot_extra_isolation,
)


@pytest.mark.parametrize(
    ("fault", "failed_check"),
    [
        ("", None),
        ("database-url", "no_database_credentials"),
        ("database-file", "no_database_credentials"),
        ("pool-short", "no_database_credentials"),
        ("pool-prefixed", "no_database_credentials"),
        ("secret-key", "required_secret_keys_present"),
        ("endpoint-required", "optional_endpoint_key"),
        ("missing-transition", "executor_ready_within_180s"),
        ("late-transition", "executor_ready_within_180s"),
        ("foreign-lease", "isolated_executor_never_owned_production_lease"),
        ("config-log", "no_config_or_tls_errors"),
        ("production-log", "production_identity_absent"),
        ("overlapping-id", "isolated_registry_only"),
        ("wrong-role", "dedicated_irsa_effective"),
        ("wildcard-role", "dedicated_irsa_effective"),
    ],
)
def test_greenfield_case_uses_actual_replica_lease_trust_and_secret_key_observations(
    tmp_path, monkeypatch, fault, failed_check
):
    fixture = GreenfieldFixture(tmp_path, monkeypatch)
    env = fixture.deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    database_names = {
        "database-url": "GPU_FAULT_STORE_URL",
        "database-file": "GPU_FAULT_STORE_URL_FILE",
        "pool-short": "POSTGRES_POOL_SIZE",
        "pool-prefixed": "GPU_FAULT_POSTGRES_POOL_SIZE",
    }
    if fault in database_names:
        env.append({"name": database_names[fault], "value": "unit-forbidden-setting"})
    elif fault == "secret-key":
        fixture.secret["data"].pop(next(iter(fixture.secret["data"])))
    elif fault == "endpoint-required":
        endpoint = next(
            item for item in env if item["name"] == "GPU_FAULT_NODE_AGENT_ENDPOINTS"
        )
        endpoint["valueFrom"]["secretKeyRef"]["optional"] = False
    elif fault == "missing-transition":
        fixture.pod_state["items"][0]["metadata"].pop("creationTimestamp")
    elif fault == "late-transition":
        fixture.pod_state["items"][0]["status"]["conditions"][-1][
            "lastTransitionTime"
        ] = "2026-09-12T00:03:01Z"
    elif fault == "foreign-lease":
        fixture.production_owners.append(f"{fixture.cluster_id}/replica-a")
    elif fault == "config-log":
        fixture.logs = "CreateContainerConfigError"
    elif fault == "production-log":
        fixture.logs = "production-a"
    elif fault == "overlapping-id":
        fixture.production_config["clusters"].append({"cluster_id": fixture.cluster_id})
    elif fault == "wrong-role":
        fixture.service_account["metadata"]["annotations"][
            "eks.amazonaws.com/role-arn"
        ] = "different-unit-role"
    elif fault == "wildcard-role":
        fixture.trust["Statement"][0]["Principal"] = {"AWS": "*"}

    result = runtime.run_boot011(fixture, production_site=fixture.production_site)
    assert result["verdict"] == ("FAIL" if fault else "PASS")
    if failed_check is not None:
        assert result["checks"][failed_check] is False
    else:
        assert all(result["checks"].values()), (
            "valid independent observations did not pass"
        )
        assert result["details"]["executor_readiness"]["ready_seconds"] == {
            "replica-a": 15,
            "replica-b": 15,
        }
    assert result["cluster_id"] == fixture.cluster_id
    assert result["details"]["secret_key_names"] == sorted(fixture.secret["data"])
    assert "REPLACE_WITH_UNIT_VALUE" not in json.dumps(result)
    assert fixture.production_reads[-1] == ("owners",)
    assert fixture.commands == [
        (
            [
                "aws",
                "iam",
                "get-role",
                "--role-name",
                "unit-isolated-executor",
                "--output",
                "json",
            ],
            {"timeout": 120},
        )
    ]


@pytest.mark.parametrize("fault", ["no-production-cluster", "log-read"])
def test_greenfield_failed_observations_stop_before_claiming_isolation(
    tmp_path, monkeypatch, fault
):
    fixture = GreenfieldFixture(tmp_path, monkeypatch)
    if fault == "no-production-cluster":
        fixture.production_config["clusters"] = []
    else:
        fixture.log_error = True
    with pytest.raises(
        runtime.BootAcceptanceError, match="no GPU clusters|log transport"
    ):
        runtime.run_boot011(fixture, production_site=fixture.production_site)
    assert fixture.commands == [], "a failed precondition reached the IAM read"
    assert ("owners",) not in fixture.production_reads
