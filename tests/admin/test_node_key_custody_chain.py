from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.admin.node_key_custody_chain import (
    authorize_now,
    planned_keys,
    verify_chain,
)
from gpu_fault.admin.node_key_custody_crypto import parse
from gpu_fault.admin.node_key_custody_models import (
    Chain,
    CustodyError,
    KeyMap,
    Signed,
    Started,
    canonical,
    statement_sha256,
)
from tests.deploy._node_key_custody_support import ProvisionFixture
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def fixture(tmp_path):
    value = ProvisionFixture(tmp_path)
    value.initial = value.activate(value.provision(value.session()))
    value.rotated = value.provision(value.session(value.initial))
    return value


def resign(fixture, transaction, field, statement):
    role = {
        "authorization": "approval",
        "started": "provisioner",
        "completed": "provisioner",
        "activated": "witness",
    }[field]
    return transaction.model_copy(
        update={field: fixture.authorities.envelope(statement, role)}
    )


@pytest.mark.parametrize(
    "field", ["authorization", "started", "completed", "activated"]
)
def test_changing_any_signed_receipt_cannot_be_repaired_by_an_unsigned_digest(
    fixture, field
):
    chain = fixture.initial
    head = chain.transactions[0]
    statement = getattr(head, field).statement
    changed = (
        statement.model_copy(update={"producer_sha256": "f" * 64})
        if field == "authorization"
        else statement.model_copy(
            update={"observed_at": statement.observed_at + timedelta(microseconds=1)}
        )
    )
    envelope = getattr(head, field).model_copy(update={"statement": changed})
    forged = Chain(transactions=[head.model_copy(update={field: envelope})])
    assert statement_sha256(forged), "a self-computed digest is not authentication"
    with pytest.raises(CustodyError, match="signature"):
        verify_chain(forged, fixture.crypto)


@pytest.mark.parametrize(
    "field", ["authorization", "started", "completed", "activated"]
)
def test_valid_signature_from_wrong_role_does_not_authorize_a_receipt(fixture, field):
    head = fixture.initial.transactions[0]
    wrong = fixture.authorities.envelope(
        getattr(head, field).statement,
        "witness" if field != "activated" else "provisioner",
    )
    with pytest.raises(CustodyError, match="wrong authority"):
        verify_chain(
            Chain(transactions=[head.model_copy(update={field: wrong})]), fixture.crypto
        )


@pytest.mark.parametrize(
    "failure",
    [
        "authorization-link",
        "start-link",
        "before-window",
        "after-window",
        "naive-start",
        "gpu-map",
        "cpu-map",
        "shared-uid",
        "cpu-incarnation",
        "other-keys",
        "node-uid",
        "master-as-key",
        "duplicate-keys",
        "missing-node",
        "generation",
        "existing-install",
    ],
)
def test_signed_but_inconsistent_provisioning_edges_are_refused(fixture, failure):
    head = fixture.initial.transactions[0].model_copy(update={"activated": None})
    start, complete = head.started.statement, head.completed.statement
    authorization = head.authorization.statement
    if failure == "authorization-link":
        start = start.model_copy(update={"authorization_sha256": "f" * 64})
    elif failure == "start-link":
        complete = complete.model_copy(update={"started_sha256": "f" * 64})
    elif failure == "before-window":
        start = start.model_copy(
            update={"observed_at": authorization.not_before - timedelta(seconds=1)}
        )
    elif failure == "after-window":
        complete = complete.model_copy(
            update={"observed_at": authorization.expires_at + timedelta(seconds=1)}
        )
    elif failure == "naive-start":
        start = start.model_copy(
            update={"observed_at": start.observed_at.replace(tzinfo=None)}
        )
    elif failure == "shared-uid":
        complete = complete.model_copy(
            update={"cpu": complete.cpu.model_copy(update={"uid": complete.gpu.uid})}
        )
    elif failure == "cpu-incarnation":
        start = start.model_copy(update={"before_cpu_uid": "other-cpu"})
    elif failure == "other-keys":
        complete = complete.model_copy(update={"cpu_other_keys_sha256": "f" * 64})
    elif failure == "existing-install":
        start = start.model_copy(update={"before_gpu": complete.gpu})
    else:
        keys = dict(complete.gpu.keys)
        if failure == "missing-node":
            del keys["node-b"]
        else:
            update = {
                "gpu-map": {"sha256": "f" * 64},
                "cpu-map": {"sha256": "f" * 64},
                "node-uid": {"node_uid": "foreign"},
                "master-as-key": {"sha256": authorization.binding.master.sha256},
                "duplicate-keys": {"sha256": keys["node-b"].sha256},
                "generation": {"generation": 2},
            }[failure]
            keys["node-a"] = keys["node-a"].model_copy(update=update)
        if failure == "cpu-map":
            complete = complete.model_copy(
                update={"cpu": complete.cpu.model_copy(update={"keys": keys})}
            )
        else:
            complete = complete.model_copy(
                update={"gpu": complete.gpu.model_copy(update={"keys": keys})}
            )
            if failure != "gpu-map":
                complete = complete.model_copy(
                    update={"cpu": complete.cpu.model_copy(update={"keys": keys})}
                )
                start = start.model_copy(update={"planned_keys": keys})
    if failure != "start-link":
        complete = complete.model_copy(
            update={"started_sha256": statement_sha256(start)}
        )
    head = resign(fixture, head, "started", start)
    head = resign(fixture, head, "completed", complete)
    with pytest.raises(CustodyError):
        verify_chain(
            Chain(transactions=[head]),
            fixture.crypto,
            now=authorization.expires_at + timedelta(seconds=2),
        )


@pytest.mark.parametrize(
    "failure",
    [
        "previous-link",
        "binding",
        "missing-before",
        "before-uid",
        "before-key",
        "gpu-uid",
        "cpu-uid",
        "unchanged-key",
        "generation-skip",
        "peer-key",
        "inactive-peer",
    ],
)
def test_rotation_requires_exact_activated_predecessor_and_single_generation_advance(
    fixture, failure
):
    first, head = fixture.rotated.transactions
    authorization = head.authorization.statement
    start, complete = head.started.statement, head.completed.statement
    if failure == "previous-link":
        authorization = authorization.model_copy(
            update={"previous_receipt_sha256": "f" * 64}
        )
    elif failure == "binding":
        authorization = authorization.model_copy(
            update={
                "binding": authorization.binding.model_copy(
                    update={
                        "site": authorization.binding.site.model_copy(
                            update={"site_name": "different"}
                        )
                    }
                )
            }
        )
    elif failure == "missing-before":
        start = start.model_copy(update={"before_gpu": None})
    elif failure.startswith("before-"):
        before = start.before_gpu
        if failure == "before-uid":
            before = before.model_copy(update={"uid": "foreign"})
        else:
            keys = dict(before.keys)
            keys["node-a"] = keys["node-a"].model_copy(update={"sha256": "f" * 64})
            before = before.model_copy(update={"keys": keys})
        start = start.model_copy(update={"before_gpu": before})
    elif failure in {"gpu-uid", "cpu-uid"}:
        plane = failure.split("-")[0]
        complete = complete.model_copy(
            update={
                plane: getattr(complete, plane).model_copy(update={"uid": "foreign"})
            }
        )
        if plane == "cpu":
            start = start.model_copy(update={"before_cpu_uid": "foreign"})
    elif failure == "inactive-peer":
        activation = first.activated.statement.model_copy(
            update={
                "nodes": {
                    "node-b": first.activated.statement.nodes["node-b"],
                    "node-c": first.activated.statement.nodes["node-a"].model_copy(
                        update={"node_uid": "foreign"}
                    ),
                }
            }
        )
        first = resign(fixture, first, "activated", activation)
        authorization = authorization.model_copy(
            update={"previous_receipt_sha256": statement_sha256(first)}
        )
    else:
        keys = dict(complete.gpu.keys)
        name = "node-b" if failure == "peer-key" else "node-a"
        updates = {
            "unchanged-key": {
                "sha256": first.completed.statement.gpu.keys[name].sha256
            },
            "generation-skip": {"generation": 3},
            "peer-key": {"sha256": "f" * 64},
        }[failure]
        keys[name] = keys[name].model_copy(update=updates)
        start = start.model_copy(update={"planned_keys": keys})
        complete = complete.model_copy(
            update={
                plane: getattr(complete, plane).model_copy(update={"keys": keys})
                for plane in ("gpu", "cpu")
            }
        )
    start = start.model_copy(
        update={"authorization_sha256": statement_sha256(authorization)}
    )
    complete = complete.model_copy(update={"started_sha256": statement_sha256(start)})
    head = resign(fixture, head, "authorization", authorization)
    head = resign(fixture, head, "started", start)
    head = resign(fixture, head, "completed", complete)
    with pytest.raises(CustodyError):
        verify_chain(Chain(transactions=[first, head]), fixture.crypto)


@pytest.mark.parametrize(
    "failure",
    [
        "completion",
        "binding",
        "future",
        "before",
        "node",
        "endpoint",
        "retired",
        "duplicate-endpoint",
    ],
)
def test_independent_activation_must_bind_completed_keys_and_runtime_nodes(
    fixture, failure
):
    head = fixture.initial.transactions[0]
    activation = head.activated.statement
    update = {
        "completion": {"completed_sha256": "f" * 64},
        "binding": {"binding_sha256": "f" * 64},
        "future": {"observed_at": datetime.now(timezone.utc) + timedelta(days=1)},
        "before": {
            "observed_at": head.completed.statement.observed_at - timedelta(seconds=1)
        },
        "retired": {"retired_key_denied": True},
    }.get(failure)
    if update is None:
        nodes = dict(activation.nodes)
        nodes["node-a"] = nodes["node-a"].model_copy(
            update={
                "node": {"node_uid": "foreign"},
                "endpoint": {"endpoint": "http://10.0.1.1"},
                "duplicate-endpoint": {"endpoint": nodes["node-b"].endpoint},
            }[failure]
        )
        update = {"nodes": nodes}
    head = resign(fixture, head, "activated", activation.model_copy(update=update))
    with pytest.raises(CustodyError):
        verify_chain(Chain(transactions=[head]), fixture.crypto)


@pytest.mark.parametrize(
    "mutation",
    [
        {"rotate_node": "node-a"},
        {"previous_receipt_sha256": "f" * 64},
        {"purpose": "rotate"},
        {"expires_at": datetime(2000, 1, 1, tzinfo=timezone.utc)},
    ],
)
def test_invalid_authorization_purpose_or_window_never_admits_provisioning(
    fixture, mutation
):
    value = fixture.authorization().model_copy(update=mutation)
    with pytest.raises(CustodyError):
        authorize_now(value, datetime.now(timezone.utc))


def test_duplicate_transactions_foreign_bindings_and_empty_chains_fail_closed(fixture):
    head = fixture.initial.transactions[0]
    with pytest.raises(CustodyError, match="replayed"):
        verify_chain(Chain(transactions=[head, head]), fixture.crypto)
    other = fixture.binding.model_copy(update={"nodes": {"node-x": "foreign"}})
    with pytest.raises(CustodyError, match="foreign"):
        verify_chain(fixture.initial, fixture.crypto, binding=other)
    with pytest.raises(CustodyError, match="schema"):
        parse(Chain, b'{"schema_version":1,"transactions":[]}')
    with pytest.raises(CustodyError, match="inventory"):
        planned_keys(fixture.binding, {}, None, None)


def test_unsigned_start_is_not_a_receipt_even_with_a_correct_payload_digest(fixture):
    start = fixture.initial.transactions[0].started.statement
    with pytest.raises(CustodyError, match="schema"):
        parse(Signed[Started], canonical(start))
    assert isinstance(
        fixture.initial.transactions[0].completed.statement.gpu, KeyMap
    ), "completed custody receipt must contain a validated GPU key map"
