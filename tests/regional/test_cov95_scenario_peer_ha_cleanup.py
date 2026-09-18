from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.e2e.regional import ha011_contracts as contracts
from scripts.e2e.regional import ha011_resources as resources
from tests.regional._cov95_scenario_peer_safety import (
    scenario_peer_transport_guard_fixture as scenario_peer_transport_guard_fixture,
)


class ReplacedNamespace:
    def __init__(self, settings, failure):
        self.settings = settings
        self.failure = failure
        self.current = None
        self.acknowledged = []
        self.deleted = []
        self.priority_class_checks = []
        self.readback_pending = False

    def call(self, arguments, *, namespace=None, input_text=None, **kwargs):
        if arguments[0] == "create":
            self.current = json.loads(input_text)
            self.current["metadata"]["uid"] = "acknowledged-namespace"
            self.acknowledged.append("acknowledged-namespace")
            self.readback_pending = True
            return "acknowledged-namespace"
        if arguments[:2] == ["delete", "--raw"]:
            options = json.loads(input_text)
            self.deleted.append(options["preconditions"]["uid"])
            self.current = None
            return ""
        raise AssertionError("namespace regression reached an unexpected command")

    def read(self, kind, name, *, namespace=None, optional=False):
        if kind == "PriorityClass":
            # The pre-adoption check is the only PriorityClass read this fake
            # may see: the Namespace readback fails first, so no PriorityClass
            # is ever created and cleanup refuses before reaching it.
            assert (name, namespace, optional) == (
                self.settings.priority_class_name,
                None,
                True,
            ), "the PriorityClass read was not the optional pre-adoption check"
            self.priority_class_checks.append(name)
            return None
        assert (kind, name, namespace) == (
            "Namespace",
            self.settings.isolated_namespace,
            None,
        ), "the ownership regression addressed an unrelated Kubernetes resource"
        if self.readback_pending:
            self.readback_pending = False
            self.current["metadata"]["uid"] = "replacement-namespace"
            if self.failure == "read-error":
                raise contracts.ProofError(
                    "unit readback failed after an acknowledged create"
                )
        return copy.deepcopy(self.current)


@pytest.mark.parametrize("failure", ["replaced-readback", "read-error"])
def test_acknowledged_namespace_identity_still_fences_cleanup_after_failed_creation_readback(
    monkeypatch, tmp_path, failure
):
    config = tmp_path / "cpu-config"
    config.write_text("unit CPU configuration; transport is fully faked")
    settings = contracts.Settings(
        cpu_kubeconfig=config,
        cpu_context="unit-cpu",
        namespace="unit-business",
        cluster_id="unit-cluster",
        isolation_id="1" * 32,
        postgres_image="postgres:16-bookworm@sha256:" + "a" * 64,
        predecessor_case="unit-previous-case",
        predecessor_path=tmp_path / "previous.json",
        region="us-east-1",
    )
    api = ReplacedNamespace(settings, failure)
    placement = {"name": "unit-cpu-node"}
    monkeypatch.setattr(resources, "cpu_node", lambda *args: placement)
    lifetime = resources.IsolatedResources(
        api,
        intent_sha256="b" * 64,
        deadline=datetime.now(timezone.utc) + timedelta(minutes=10),
        cpu_node_identity=placement,
    )

    with pytest.raises(contracts.ProofError):
        lifetime.start([])
    refused = False
    try:
        lifetime.cleanup()
    except contracts.ProofError:
        refused = True

    assert api.priority_class_checks == [settings.priority_class_name], (
        "start did not refuse-to-adopt the PriorityClass exactly once before "
        f"creating the Namespace: {api.priority_class_checks}"
    )
    assert api.acknowledged == ["acknowledged-namespace"], (
        "the test did not receive a real fake-transport UID before the readback failure"
    )
    assert api.deleted == [], (
        f"cleanup discarded the acknowledged UID and deleted replacements: {api.deleted}"
    )
    assert refused is True, "cleanup did not refuse the known-UID replacement"
    assert api.current["metadata"]["uid"] == "replacement-namespace", (
        "the foreign Namespace did not survive refused cleanup"
    )
