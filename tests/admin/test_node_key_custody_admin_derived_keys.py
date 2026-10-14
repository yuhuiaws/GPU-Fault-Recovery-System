"""A completed custody receipt and a key the node-key sync added after it.

Every deploy re-checks a completed custody transaction against the live key
Secrets (``verify_completed_custody``), and any GPU key outside the receipt used
to refuse it ("does not match deployed key sources") -- so the first node the
executor's node-key sync keyed between deploys blocked every later deploy of a
custody site. A key of a node the authorization does not bind is now accepted
when it is exactly that node's derivation under the bound master, recomputed
from the master read privately from its bound source, and the task evidence
names it. Anything else still refuses, as one set.
"""

from __future__ import annotations

import base64

import pytest

from gpu_fault.admin.node_key_custody_admin import (
    CustodyPreparationRequired,
    CustodyReconciliationRequired,
)
from gpu_fault.node_action_keys import derive_node_action_secret
from tests.admin._node_key_custody_admin_support import AdminWorld
from tests.admin.test_node_key_custody_admin import provision
from tests.deploy._node_action_key_api import encoded
from tests.deploy._node_key_custody_support import MASTER
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def installed(tmp_path, monkeypatch):
    world = AdminWorld(tmp_path, monkeypatch)
    world.configure()
    world.access()
    world.release_ready = True
    with pytest.raises(CustodyPreparationRequired):
        provision(world)
    world.authorize()
    provision(world)
    return world


def _derived(world: AdminWorld, node: str) -> str:
    return base64.b64encode(
        derive_node_action_secret(MASTER.decode(), world.cluster_id, node).encode()
    ).decode()


def test_a_completed_custody_accepts_a_derived_key_added_after_it(installed) -> None:
    world = installed
    world.api.state["secrets"]["gpu"]["data"]["node-c"] = _derived(world, "node-c")

    result = provision(world, probe=True)

    assert result["custody_derived_extra_keys"] == "node-c", result
    assert world.helper_calls == 1, "nothing is re-provisioned for it"


def test_without_extra_keys_the_evidence_is_unchanged(installed) -> None:
    result = provision(installed, probe=True)

    assert "custody_derived_extra_keys" not in result, result


@pytest.mark.parametrize(
    "extras",
    [
        {"node-c": encoded("foreign-value")},
        {"node-c": "derived", "node-d": encoded("foreign-value")},
    ],
)
def test_a_completed_custody_still_refuses_any_other_extra_key(
    installed, extras
) -> None:
    world = installed
    gpu = world.api.state["secrets"]["gpu"]["data"]
    gpu.update(
        {
            node: _derived(world, node) if value == "derived" else value
            for node, value in extras.items()
        }
    )

    with pytest.raises(CustodyReconciliationRequired, match="deployed key sources"):
        provision(world, probe=True)
