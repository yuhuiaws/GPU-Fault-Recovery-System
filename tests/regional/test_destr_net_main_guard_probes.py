"""Run the exact emitted Automatic-recovery probe against recorded local facts."""

from __future__ import annotations

import hashlib
import io
import json
from contextlib import redirect_stdout
from typing import Any

import pytest

from scripts.e2e.regional import audit_warm_spare_guardrails as audit
from tests.regional._cov95_warm_guard import Kubernetes
from tests.regional._cov95_warm_guard import configured as configured
from tests.regional.test_guardrail_audit_evidence import recording

pytestmark = pytest.mark.usefixtures("configured")


def test_both_emitted_automatic_guard_arms_cross_only_the_synthetic_isolation_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = Kubernetes()
    namespaces: list[dict[str, Any]] = []

    def kubectl(kubeconfig: Any, context: str, *args: str, **kwargs: Any) -> str:
        if "exec" not in args:
            return transport(kubeconfig, context, *args, **kwargs)
        namespace: dict[str, Any] = {}
        namespaces.append(namespace)
        output = io.StringIO()
        with redirect_stdout(output):
            exec(kwargs["stdin"], namespace)
        return output.getvalue()

    monkeypatch.setattr(audit, "kubectl", kubectl)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "cluster-a")
    snapshot = recording()
    payloads = snapshot["payloads"]
    payloads["describe_cluster"]["ClusterStatus"] = "InService"
    payloads["describe_cluster_node"]["node-1"]["InstanceStatus"] = {
        "Status": "Running"
    }
    snapshot["payload_digest"] = hashlib.sha256(
        json.dumps(payloads, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    snapshot["recorded_by"] = "local synthetic contract"

    result = audit.deployed_automatic_recovery_probe(snapshot)

    assert audit.probe_errors("GF-REGIONAL-DESTR-006", None, None, result) == [], result
    assert result["observed_node_recovery"] == "Automatic"
    assert result["isolation_source"] == "synthetic-read-only"
    assert set(result["isolation_reads"]) == {
        "warm_spare_guard",
        "control_without_warm_spare_strategy",
    }
    assert all(reads == ["node-1"] for reads in result["isolation_reads"].values()), (
        "each isolation probe must read only the recorded target node"
    )
    assert result["payload_provenance"]["payload_digest"] == snapshot["payload_digest"]
    provider = namespaces[0]["adapter"].client
    assert not hasattr(provider, "batch_replace_cluster_nodes"), (
        "the synthetic provider must not expose node replacement"
    )
    assert not hasattr(provider, "batch_reboot_cluster_nodes"), (
        "the read-only synthetic provider must not expose reboot"
    )
    assert all("exec" not in call[2] for call in transport.calls), (
        "no external probe ran"
    )
