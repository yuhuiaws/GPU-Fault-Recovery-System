"""Store-side verdicts of the AUTH boundary matrix.

A denied request must also have written nothing. ``store_negative_errors``
compares the two store snapshots taken around the matrix; these tests pin every
way it can refuse -- a probe identity that is missing, changed or already
present, cluster B measurements that are absent, malformed or drifted, and the
probe-owned cluster B backlog that must still be PENDING afterwards -- plus the
probe-identity guards on the snapshot and matrix entry points and the pinned
transport note the main entry records in its evidence identity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import audit_auth_boundary as audit

CLUSTER_A = "cluster-a"
CLUSTER_B = "cluster-b"
PROBE = "auth-probe-" + "0" * 32
DIGEST = "a" * 64


def cluster_b(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "attempt_observations": 0,
        "observations_sha256": DIGEST,
        "agent_generations": [],
        "agents_sha256": DIGEST,
        "collector_samples": {},
        "evidence_count": 0,
        "evidence_sha256": DIGEST,
    }
    value.update(overrides)
    return value


def pending_command(**overrides: Any) -> dict[str, Any]:
    value = {
        "cluster_id": CLUSTER_B,
        "status": "PENDING",
        "execution_owner": audit.ACCEPTANCE_PROBE_OWNER,
        "lease_owner": None,
    }
    value.update(overrides)
    return value


def snapshot(
    *,
    probe_id: str | None = PROBE,
    b: dict[str, Any] | None = None,
    commands: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "clusters": {CLUSTER_A: {}, CLUSTER_B: cluster_b() if b is None else b},
        "commands": {"cmd-b": pending_command()} if commands is None else commands,
    }
    if probe_id is not None:
        value["probe_id"] = probe_id
    return value


def errors(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[str]]:
    return audit.store_negative_errors(
        before, after, cluster_a=CLUSTER_A, cluster_b=CLUSTER_B
    )


def test_identical_snapshots_with_a_pending_probe_command_are_clean() -> None:
    assert errors(snapshot(), snapshot()) == {}


@pytest.mark.parametrize(
    "before_probe, after_probe",
    [(None, PROBE), ("", PROBE), (PROBE, "auth-probe-" + "1" * 32)],
    ids=["missing-before", "empty-before", "changed"],
)
def test_probe_identity_must_be_present_and_stable(
    before_probe: str | None, after_probe: str
) -> None:
    result = errors(snapshot(probe_id=before_probe), snapshot(probe_id=after_probe))

    for case_id in audit.STORE_NEGATIVE_CASES:
        assert "probe identity is missing or changed" in result[case_id], case_id


def test_without_probe_ids_only_the_measurements_are_compared() -> None:
    before = snapshot(probe_id=None, b=cluster_b(attempt_observations=4))
    after = snapshot(probe_id=None, b=cluster_b(attempt_observations=4))

    assert errors(before, after) == {}


def test_probe_identity_that_already_left_traces_is_refused() -> None:
    before = snapshot(b=cluster_b(evidence_count=2))

    result = errors(before, snapshot(b=cluster_b(evidence_count=2)))

    for case_id in (
        "GF-REGIONAL-AUTH-005",
        "GF-REGIONAL-AUTH-006",
        "GF-REGIONAL-AUTH-009",
    ):
        assert result[case_id] == ["probe identity already exists"], case_id
    assert "GF-REGIONAL-AUTH-008" not in result


def test_unmeasured_and_malformed_cluster_b_fields_are_reported_once_each() -> None:
    # A float where an int is expected (falsy, so not an "already exists"
    # trace) and a digest that is not 64 hex characters.
    before = snapshot(
        b=cluster_b(attempt_observations=0.0, observations_sha256="not-hex")
    )
    after = snapshot(
        b=cluster_b(attempt_observations=0.0, observations_sha256="not-hex")
    )

    result = errors(before, after)

    assert result["GF-REGIONAL-AUTH-005"] == [
        "cluster B attempt_observations not measured",
        "cluster B observations_sha256 is invalid",
    ]
    assert "GF-REGIONAL-AUTH-006" not in result


def test_uppercase_hex_digest_is_invalid() -> None:
    bad = "A" * 64
    before = snapshot(b=cluster_b(agents_sha256=bad))
    after = snapshot(b=cluster_b(agents_sha256=bad))

    assert errors(before, after)["GF-REGIONAL-AUTH-006"] == [
        "cluster B agents_sha256 is invalid"
    ]


def test_changed_cluster_b_measurements_name_the_field_and_the_cases() -> None:
    before = snapshot()
    after = snapshot(
        b=cluster_b(
            attempt_observations=1,
            agent_generations=[{"node_id": "n1", "generation": 2}],
            evidence_sha256="b" * 64,
        )
    )

    result = errors(before, after)

    assert result["GF-REGIONAL-AUTH-005"] == [
        "cluster B attempt_observations changed",
        "cluster B attempt observations changed",
    ]
    assert result["GF-REGIONAL-AUTH-006"] == [
        "cluster B agent_generations changed",
        "cluster B agents changed",
    ]
    assert result["GF-REGIONAL-AUTH-009"] == [
        "cluster B evidence_sha256 changed",
        "cluster B attempt observations changed",
        "cluster B agents changed",
    ]


def test_missing_cluster_b_pending_probe_command_before_the_matrix_fails_auth008() -> (
    None
):
    before = snapshot(commands={})

    result = errors(before, snapshot(commands={}))

    assert result == {
        "GF-REGIONAL-AUTH-008": [
            "no cluster B PENDING probe-owner command was measured before the matrix"
        ]
    }


def test_probe_commands_that_vanish_or_get_leased_fail_auth008_only_when_pending() -> (
    None
):
    before = snapshot(
        commands={
            "gone-pending": pending_command(),
            "gone-done": pending_command(status="SUCCEEDED"),
            "leased": pending_command(),
            "foreign": pending_command(cluster_id=CLUSTER_A),
            "other-owner": pending_command(execution_owner="gpu-fault-node-agent"),
            "untouched": pending_command(),
        }
    )
    after = snapshot(
        commands={
            "leased": pending_command(status="LEASED", lease_owner="exec-1"),
            "untouched": pending_command(),
            # Foreign and other-owner commands may do anything.
            "foreign": pending_command(cluster_id=CLUSTER_A, status="SUCCEEDED"),
        }
    )

    result = errors(before, after)

    assert result == {
        "GF-REGIONAL-AUTH-008": [
            "command gone-pending left PENDING during the matrix",
            "command leased was leased or changed by the matrix",
        ]
    }


def test_a_rebound_lease_owner_on_a_still_pending_command_counts_as_a_change() -> None:
    before = snapshot(commands={"cmd": pending_command(lease_owner=None)})
    after = snapshot(commands={"cmd": pending_command(lease_owner="exec-2")})

    assert errors(before, after)["GF-REGIONAL-AUTH-008"] == [
        "command cmd was leased or changed by the matrix"
    ]


# --------------------------------------------------------------------------- #
# probe identity guards
# --------------------------------------------------------------------------- #
def test_store_snapshot_refuses_a_malformed_probe_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        audit, "run_fixture_command", lambda command, **_k: calls.append(command)
    )

    with pytest.raises(ValueError, match="invalid authentication probe identity"):
        audit.store_snapshot(
            tmp_path / "cpu", "gpu-fault-system", "api", [CLUSTER_A], probe_id="x"
        )
    assert calls == [], "nothing may be executed in the Pod with a bad identity"


def test_store_snapshot_refuses_a_reply_for_another_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Completed:
        stdout = json.dumps({"probe_id": "auth-probe-node", "clusters": {}}) + "\n"

    monkeypatch.setattr(audit, "run_fixture_command", lambda *a, **k: Completed())

    with pytest.raises(RuntimeError, match="different probe identity"):
        audit.store_snapshot(
            tmp_path / "cpu", "gpu-fault-system", "api", [CLUSTER_A], probe_id=PROBE
        )


def test_run_matrix_requires_a_fresh_probe_identity(tmp_path: Path) -> None:
    token_a = tmp_path / "a"
    token_b = tmp_path / "b"
    token_a.write_text("token-a-placeholder\n", encoding="utf-8")
    token_b.write_text("token-b-placeholder\n", encoding="utf-8")
    arguments = audit.parser().parse_args(
        [
            "matrix",
            "--url",
            "https://api.example.internal:8443",
            "--ca-file",
            str(tmp_path / "ca.crt"),
            "--cluster-a",
            CLUSTER_A,
            "--token-a-file",
            str(token_a),
            "--cluster-b",
            CLUSTER_B,
            "--token-b-file",
            str(token_b),
            "--executor-artifact-sha256",
            DIGEST,
            "--executor-compatibility-digest",
            DIGEST,
        ]
    )
    arguments.probe_id = "auth-probe-node"

    with pytest.raises(ValueError, match="fresh probe identity"):
        audit.run_matrix(arguments)


# --------------------------------------------------------------------------- #
# main: the pinned forward is named in the evidence identity
# --------------------------------------------------------------------------- #
@pytest.fixture
def reset_pins() -> Any:
    yield None
    audit.pin_resolution([])


def test_main_records_the_pinned_forward_in_the_evidence_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    reset_pins: None,
) -> None:
    for name in ("a", "b"):
        (tmp_path / name).write_text(f"token-{name}-placeholder\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "matrix",
            "--url",
            "http://api.example.internal:18080",
            "--ca-file",
            str(tmp_path / "ca.crt"),
            "--cluster-a",
            CLUSTER_A,
            "--token-a-file",
            str(tmp_path / "a"),
            "--cluster-b",
            CLUSTER_B,
            "--token-b-file",
            str(tmp_path / "b"),
            "--executor-artifact-sha256",
            DIGEST,
            "--executor-compatibility-digest",
            DIGEST,
            "--run-dir",
            str(tmp_path / "run"),
            "--resolve",
            "api.example.internal:18080:127.0.0.1",
        ],
    )
    seen: dict[str, Any] = {}

    def run_matrix(arguments: Any) -> dict[str, Any]:
        seen["probe_id"] = arguments.probe_id
        return {"AUTH-001-anon": {"status": 401, "body": {}}}

    def case_documents(
        results: dict[str, Any], *, cluster_a: str, store_errors: Any, identity: Any
    ) -> dict[str, dict[str, Any]]:
        seen["identity"] = dict(identity)
        seen["store_errors"] = store_errors
        return {
            "GF-REGIONAL-AUTH-001": {"verdict": "PASS", "identity": identity},
            audit.GUARDED_AUTH008_CASE: {"verdict": "PASS"},
        }

    monkeypatch.setattr(audit, "run_matrix", run_matrix)
    monkeypatch.setattr(audit, "case_documents", case_documents)

    assert audit.main() == 0

    assert seen["identity"] == {
        "cluster_id": CLUSTER_A,
        "transport": "pinned-forward api.example.internal:18080->127.0.0.1",
    }
    assert seen["store_errors"] is None, "no kubeconfig means no store snapshots"
    assert seen["probe_id"].startswith("auth-probe-") and len(seen["probe_id"]) == 43
    written = (
        tmp_path
        / "run"
        / "cases"
        / "GF-REGIONAL-AUTH-001"
        / "GF-REGIONAL-AUTH-001.json"
    )
    assert json.loads(written.read_text(encoding="utf-8"))["verdict"] == "PASS"
    assert not (tmp_path / "run" / "cases" / audit.GUARDED_AUTH008_CASE).exists(), (
        "the diagnostic matrix must not publish guarded AUTH008 evidence"
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["verdicts"] == {"GF-REGIONAL-AUTH-001": "PASS"}
    assert printed["store_errors"] is None
