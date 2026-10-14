"""Custody provisioning and a live key the node-key sync added.

Between deploys the executor's node-key sync keys a node that joined, with the
key the control plane derives from the fleet master
(``derive_node_action_secret``). Such a key is not in the signed predecessor,
and custody used to refuse every rotation after it ("live keys differ from the
signed predecessor"). Now a live extra key is accepted only when it is provably
that derivation -- recomputed from the verified master and compared -- and the
start receipt records it (node -> digest). What must hold:

* an extra key that is exactly its node's derivation lets the rotation go on,
  is recorded in the signed start, and is written out of the GPU Secret with
  every other key of a node the authorization does not bind;
* any other value, a mixed set, or a CPU mirror value that differs refuses the
  whole rotation before anything is written;
* the receipt chain refuses derived keys on an installation or on an
  authorized node, and a start without them canonicalizes as before.
"""

from __future__ import annotations

import base64
import hashlib

import pytest

from gpu_fault.admin.node_key_custody_chain import validate_transition, verify_chain
from gpu_fault.admin.node_key_custody_models import (
    Chain,
    CustodyError,
    canonical,
    statement_sha256,
)
from gpu_fault.node_action_keys import derive_node_action_secret
from tests.deploy._node_action_key_api import encoded
from tests.deploy._node_key_custody_support import MASTER, ProvisionFixture


@pytest.fixture
def fixture(tmp_path):
    return ProvisionFixture(tmp_path)


def _derived(node: str) -> str:
    return base64.b64encode(
        derive_node_action_secret(MASTER.decode(), "cluster-a", node).encode()
    ).decode()


def _digest(encoded_key: str) -> str:
    return hashlib.sha256(base64.b64decode(encoded_key)).hexdigest()


def _activated_install(fixture: ProvisionFixture) -> Chain:
    return fixture.activate(fixture.provision(fixture.session()))


def test_a_derived_key_the_sync_added_does_not_block_a_rotation(fixture) -> None:
    active = _activated_install(fixture)
    gpu = fixture.api.state["secrets"]["gpu"]
    gpu["data"]["node-c"] = _derived("node-c")
    cpu_before = dict(fixture.api.state["secrets"]["cpu"]["data"])

    rotated = fixture.provision(fixture.session(active))

    head = verify_chain(rotated, fixture.crypto)
    started = head.started.statement
    assert started.derived_extra_keys == {"node-c": _digest(_derived("node-c"))}
    assert set(started.planned_keys) == {"node-a", "node-b"}
    assert "node-c" not in fixture.api.state["secrets"]["gpu"]["data"], (
        "a key of a node the authorization does not bind is written out"
    )
    assert "node-c" not in fixture.api.state["secrets"]["cpu"]["data"], (
        "the CPU mirror gains nothing for it"
    )
    assert set(cpu_before) <= set(fixture.api.state["secrets"]["cpu"]["data"])
    text = canonical(rotated).decode()
    assert _derived("node-c") not in text, "the receipt carries digests only"
    assert base64.b64decode(_derived("node-c")).decode() not in text


@pytest.mark.parametrize(
    "extras",
    [
        {"node-c": encoded("foreign-value")},
        {"node-c": _derived("node-c"), "node-d": encoded("foreign-value")},
        # A derivation for another name is not this node's key.
        {"node-c": _derived("node-d")},
    ],
)
def test_any_extra_key_that_is_not_its_derivation_refuses_the_whole_set(
    fixture, extras
) -> None:
    active = _activated_install(fixture)
    gpu = fixture.api.state["secrets"]["gpu"]
    gpu["data"].update(extras)
    before = dict(gpu["data"])
    events = list(fixture.events)

    with pytest.raises(CustodyError, match="live keys differ from the signed"):
        fixture.provision(fixture.session(active))

    assert fixture.events == events, "nothing was written"
    assert fixture.api.state["secrets"]["gpu"]["data"] == before


def test_a_derived_key_the_cpu_mirror_disagrees_with_refuses(fixture) -> None:
    active = _activated_install(fixture)
    fixture.api.state["secrets"]["gpu"]["data"]["node-c"] = _derived("node-c")
    fixture.api.state["secrets"]["cpu"]["data"]["node-c"] = encoded("other-value")

    with pytest.raises(CustodyError, match="live keys differ from the signed"):
        fixture.provision(fixture.session(active))


def test_the_chain_refuses_derived_keys_on_an_installation(fixture) -> None:
    installed = fixture.provision(fixture.session())
    transaction = installed.transactions[-1]
    started = transaction.started.statement.model_copy(
        update={"derived_extra_keys": {"node-c": "c" * 64}}
    )
    completed = transaction.completed.statement.model_copy(
        update={"started_sha256": statement_sha256(started)}
    )

    with pytest.raises(CustodyError, match="prospective absent keys"):
        validate_transition(
            transaction.authorization.statement, started, completed, None
        )


def test_the_chain_refuses_a_derived_key_for_an_authorized_node(fixture) -> None:
    active = _activated_install(fixture)
    fixture.api.state["secrets"]["gpu"]["data"]["node-c"] = _derived("node-c")
    rotated = fixture.provision(fixture.session(active))
    previous, transaction = rotated.transactions
    started = transaction.started.statement.model_copy(
        update={"derived_extra_keys": {"node-a": "c" * 64}}
    )
    completed = transaction.completed.statement.model_copy(
        update={"started_sha256": statement_sha256(started)}
    )

    with pytest.raises(CustodyError, match="invalid provisioning edge"):
        validate_transition(
            transaction.authorization.statement, started, completed, previous
        )


def test_a_start_without_derived_keys_canonicalizes_as_before(fixture) -> None:
    installed = fixture.provision(fixture.session())
    started = installed.transactions[-1].started.statement

    assert started.derived_extra_keys == {}
    assert b"derived_extra_keys" not in canonical(started), (
        "receipts signed before the field existed must verify unchanged"
    )
    assert verify_chain(installed, fixture.crypto).started.statement == started
