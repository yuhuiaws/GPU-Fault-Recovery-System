"""AUTH-007/008 on a one-cluster site: cluster B as a synthetic logical cluster.

``--secondary-registration synthetic`` makes the runner register B itself (a
``synthetic`` durable registration with its own token), run B's executor Pod in
B's own namespace on the primary's EKS, run the case body against that Pod and
remove everything afterwards. Site mode (the default) is what it always was.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.regional import RegionalClusterRegistration, cluster_token_sha256
from scripts.e2e.regional import audit_auth_boundary as audit
from scripts.e2e.regional import identity_acceptance_common as common
from scripts.e2e.regional import identity_auth_backlog as backlog
from scripts.e2e.regional import identity_synthetic_secondary as synthetic
from scripts.e2e.regional import live_driver_guard
from scripts.e2e.regional import run_identity_acceptance as entry
from scripts.e2e.regional import seeded_command_fixture as seeded
from scripts.e2e.regional.identity_auth_probes import (
    AUTH008_BACKLOG_PROBE,
    AUTH008_EXECUTOR_IDENTITY_PROBE,
)
from scripts.e2e.regional.probes import auth_logical_secondary_hold as hold
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture
from scripts.perf import regional_capacity_registry as perf_registry
from tests._builders import asgi_client, build_context
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional._identity_synthetic_support import (
    ARTIFACT,
    COMPATIBILITY,
    IMAGE,
    POD_ENVIRONMENT,
    PRIMARY,
    SECONDARY,
    SITE_NAMESPACE,
    FakeSite,
)
from tests.regional._regional_support import TOKEN_A
from tests.regional.test_auth008_owned_backlog import execute_probe

AUTH007 = "GF-REGIONAL-AUTH-007"
AUTH008 = "GF-REGIONAL-AUTH-008"


def _primary(cluster_id: str = PRIMARY) -> common.ClusterTarget:
    return common.ClusterTarget(
        cluster_id=cluster_id,
        context="context-a",
        region="us-west-2",
        hyperpod_cluster_name="hp-a",
        eks_cluster_arn="arn:aws:eks:us-west-2:123456789012:cluster/a",
        executor_role_arn="arn:aws:iam::123456789012:role/executor-a",
        control_plane_url="https://control.example",
        ca_file=Path("/unused/ca.crt"),
    )


class _Site:
    """A one-cluster site whose only other id happens to look synthetic."""

    def __init__(self) -> None:
        self.targets = {
            PRIMARY: _primary(),
            "auth-logical-real": _primary("auth-logical-real"),
        }

    def target(self, cluster_id: str) -> common.ClusterTarget:
        try:
            return self.targets[cluster_id or PRIMARY]
        except KeyError as exc:
            raise common.IdentityAcceptanceError(
                f"cluster is not present in the site: {cluster_id}"
            ) from exc


def _arguments(
    case: str, secondary: str, *, mode: str = "site", allow: bool = False
) -> argparse.Namespace:
    return argparse.Namespace(
        case=case,
        cluster_id=PRIMARY,
        secondary_cluster_id=secondary,
        secondary_registration=mode,
        allow_synthetic_secondary=allow,
        node=[],
        fleet_master_file=None,
        host_probe_image="",
    )


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("case", "secondary", "mode", "allow", "match"),
    [
        (
            AUTH007,
            SECONDARY,
            "synthetic",
            False,
            "requires --allow-synthetic-secondary",
        ),
        (AUTH008, "perf-cap-000", "synthetic", True, "must match auth-logical"),
        (AUTH007, "Auth-Logical-B", "synthetic", True, "must match auth-logical"),
        (AUTH007, "auth-logical-", "synthetic", True, "must match auth-logical"),
        (AUTH007, "auth-logical-" + "x" * 60, "synthetic", True, "must match"),
        (AUTH007, "auth-logical-b/../c", "synthetic", True, "must match"),
        (AUTH007, PRIMARY, "synthetic", True, "must match auth-logical"),
        (AUTH007, "auth-logical-real", "synthetic", True, "collides with a site"),
        ("GF-REGIONAL-ISO-003", SECONDARY, "synthetic", True, "only for"),
        ("GF-REGIONAL-AUTH-010", "", "synthetic", True, "only for"),
        (
            AUTH007,
            "auth-logical-real",
            "site",
            True,
            "requires --secondary-registration",
        ),
    ],
)
def test_synthetic_mode_is_refused_without_its_switch_or_with_an_unsafe_id(
    case: str, secondary: str, mode: str, allow: bool, match: str
) -> None:
    with pytest.raises(common.IdentityAcceptanceError, match=match):
        entry.validate_case_arguments(
            _arguments(case, secondary, mode=mode, allow=allow), _Site()
        )


@pytest.mark.parametrize("case", [AUTH007, AUTH008])
def test_synthetic_mode_builds_b_on_the_primary_plane_in_its_own_namespace(
    case: str,
) -> None:
    primary, secondary, _nodes = entry.validate_case_arguments(
        _arguments(case, SECONDARY, mode="synthetic", allow=True), _Site()
    )
    assert secondary is not None, "the synthetic target is synthesized, not dropped"
    assert secondary.kind == synthetic.SYNTHETIC_SECONDARY_KIND
    assert secondary.cluster_id == SECONDARY
    assert secondary.executor_namespace == SECONDARY, (
        "B lives in a namespace named after it"
    )
    assert secondary.registered is False, "site.yaml does not list B"
    assert (secondary.context, secondary.control_plane_url, secondary.ca_file) == (
        primary.context,
        primary.control_plane_url,
        primary.ca_file,
    ), "B shares the primary's EKS, control plane and CA"
    assert (secondary.eks_cluster_arn, secondary.executor_role_arn) == ("", ""), (
        "B has no physical identity of its own"
    )
    assert primary.kind == "site" and primary.executor_namespace == ""


def test_site_mode_is_the_default_and_keeps_its_plan_shape() -> None:
    arguments = entry.parser().parse_args(
        [
            "--run-dir",
            "/tmp/unit",
            "--site",
            "/tmp/unit/site",
            "--case",
            AUTH007,
            "--secondary-cluster-id",
            "auth-logical-real",
        ]
    )
    assert arguments.secondary_registration == "site"
    assert arguments.allow_synthetic_secondary is False
    primary, secondary, _nodes = entry.validate_case_arguments(arguments, _Site())
    assert secondary is not None and secondary.kind == "site"
    plan = entry.case_plan(
        AUTH007,
        primary=primary,
        secondary=secondary,
        nodes=(),
        predecessor={"valid": True},
        evidence_identity={"release_id": "r", "cluster_id": PRIMARY},
    )
    assert plan["secondary"] == {
        "cluster_id": "auth-logical-real",
        "context": "context-a",
        "registered": True,
    }, "a site B's plan carries exactly the keys it always had"
    synthetic_plan = entry.case_plan(
        AUTH007,
        primary=primary,
        secondary=synthetic.synthetic_secondary(primary, SECONDARY),
        nodes=(),
        predecessor={"valid": True},
        evidence_identity={"release_id": "r", "cluster_id": PRIMARY},
    )
    assert synthetic_plan["secondary"] == {
        "cluster_id": SECONDARY,
        "context": "context-a",
        "registered": False,
        "kind": synthetic.SYNTHETIC_SECONDARY_KIND,
        "executor_namespace": SECONDARY,
    }


# --------------------------------------------------------------------------- #
# transport, builders, manifests, probe
# --------------------------------------------------------------------------- #
def test_secondary_namespace_fixture_routes_only_the_gpu_plane(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeSite(tmp_path)
    site = fake.install(monkeypatch)
    primary = site.target(PRIMARY)
    secondary = synthetic.synthetic_secondary(primary, SECONDARY)
    assert type(site.regional(primary)) is RegionalLiveFixture, (
        "a site cluster keeps the plain fixture"
    )
    fixture = site.regional(secondary)
    assert isinstance(fixture, common.SecondaryNamespaceFixture), (
        "a synthetic B gets the namespace-routing fixture"
    )
    argv: list[list[str]] = []
    monkeypatch.setattr(
        RegionalLiveFixture,
        "run",
        staticmethod(
            lambda command, **kwargs: argv.append(command)
            or subprocess.CompletedProcess(command, 0, "{}", "")
        ),
    )
    fixture.kubectl("gpu", "get", "pod")
    fixture.kubectl("cpu", "get", "pod")
    fixture.kubectl("gpu", "get", "pod", namespace="other")
    fixture.kubectl("gpu", "get", "pod", all_namespaces=True)
    namespaces = [
        command[command.index("-n") + 1] if "-n" in command else None
        for command in argv
    ]
    assert namespaces == [SECONDARY, SITE_NAMESPACE, "other", None]
    assert argv[3][-1] == "--all-namespaces"
    with pytest.raises(ValueError, match="must not be empty"):
        common.SecondaryNamespaceFixture(fixture.settings, executor_namespace="")


def test_synthetic_registration_is_the_perf_suites_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expires = datetime.now(timezone.utc) + timedelta(minutes=30)
    perf = perf_registry.perf_cluster_entries(2, run_id="run-a", expires_at=expires)
    direct = [
        perf_registry.synthetic_cluster_entry(
            f"perf-cap-{index:03d}", run_id="run-a", expires_at=expires
        )
        for index in range(2)
    ]
    for left, right in zip(perf, direct, strict=True):
        assert len(left.pop("token")) >= 32 and len(right.pop("token")) >= 32
        assert left == right, "perf_cluster_entries is the per-id builder applied"
    calls: list[dict[str, Any]] = []
    original = perf_registry.synthetic_cluster_entry

    def spy(cluster_id: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"cluster_id": cluster_id, **kwargs})
        return original(cluster_id, **kwargs)

    monkeypatch.setattr(perf_registry, "synthetic_cluster_entry", spy)
    entry_b = synthetic.synthetic_registration(
        SECONDARY,
        run_id="auth-logical-b-unit-a1",
        expires_at=expires,
        region="eu-west-1",
        namespace=SECONDARY,
    )
    assert calls == [
        {
            "cluster_id": SECONDARY,
            "run_id": "auth-logical-b-unit-a1",
            "expires_at": expires,
            "region": "eu-west-1",
            "allowed_namespaces": [SECONDARY],
            "token": None,
        }
    ]
    assert entry_b["synthetic"] is True
    assert entry_b["allowed_namespaces"] == [SECONDARY]
    assert entry_b["region"] == "eu-west-1"
    assert ":000000000000:cluster/" + SECONDARY in entry_b["eks_cluster_arn"]
    assert entry_b["agent_endpoint_allowed_cidrs"] == ["127.0.0.1/32"]
    redacted = synthetic.redacted_registration(entry_b)
    assert "token" not in redacted
    assert redacted["token_sha256"] == cluster_token_sha256(entry_b["token"])
    model = RegionalClusterRegistration.model_validate(redacted)
    assert model.is_active(), "B is an active registration until its expiry"
    assert model.authenticates(entry_b["token"]), "B's own token authenticates"


def test_secondary_pod_is_an_executor_pod_wired_to_b_without_a_claim_loop() -> None:
    manifest = synthetic.secondary_pod_manifest(
        cluster_id=SECONDARY,
        namespace=SECONDARY,
        run_id="run-1",
        case_id=AUTH007,
        image=IMAGE,
        pins={"artifact": ARTIFACT, "compatibility": COMPATIBILITY},
        lifetime_seconds=1800,
    )
    assert manifest["metadata"]["namespace"] == SECONDARY
    assert manifest["metadata"]["labels"]["app"] == common.EXECUTOR_APP
    assert manifest["metadata"]["labels"][seeded.RUN_LABEL] == "run-1"
    spec = manifest["spec"]
    assert spec["restartPolicy"] == "Never" and spec["activeDeadlineSeconds"] == 1800
    assert spec["automountServiceAccountToken"] is False
    container = spec["containers"][0]
    assert container["image"] == IMAGE
    assert container["command"] == [
        "/opt/gpu-fault/executor/bin/python",
        "/scripts/auth_logical_secondary_hold.py",
    ], "the hold script replaces the claim loop"
    environment = {item["name"]: item for item in container["env"]}
    for name, key in (
        ("GPU_FAULT_CONTROL_PLANE_URL", "control-plane-url"),
        ("GPU_FAULT_CONTROL_PLANE_TOKEN", "cluster-token"),
        ("GPU_FAULT_CLUSTER_ID", "cluster-id"),
    ):
        assert environment[name]["valueFrom"]["secretKeyRef"] == {
            "name": common.CONNECTION_SECRET,
            "key": key,
        }
    assert environment["GPU_FAULT_EXECUTOR_ARTIFACT_SHA256"]["value"] == ARTIFACT
    assert environment["GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST"]["value"] == (
        COMPATIBILITY
    )
    assert environment["GPU_FAULT_CONTROL_PLANE_CA_FILE"]["value"] == (
        "/etc/gpu-fault/tls/ca.crt"
    )
    volumes = {item["name"]: item for item in spec["volumes"]}
    assert volumes["tls"]["secret"] == {
        "secretName": common.CONNECTION_SECRET,
        "items": [{"key": "ca.crt", "path": "ca.crt"}],
    }
    assert volumes["script"]["configMap"] == {
        "name": synthetic.SECONDARY_HOLD_CONFIGMAP
    }
    assert json.dumps(manifest).count(ARTIFACT) == 1, "no token or foreign value leaks"


def test_connection_secret_copies_only_the_control_plane_location_and_trust() -> None:
    copied = {"control-plane-url": "dXJs", "ca.crt": "Y2E="}
    manifest = synthetic.connection_secret_manifest(
        cluster_id=SECONDARY,
        namespace=SECONDARY,
        run_id="run-1",
        case_id=AUTH007,
        token="t" * 48,
        copied={**copied, "cluster-token": "QUJD", "cluster-id": "YQ=="},
    )
    assert manifest["metadata"] == {
        "name": common.CONNECTION_SECRET,
        "namespace": SECONDARY,
        "labels": {
            seeded.RUN_LABEL: "run-1",
            synthetic.CASE_LABEL: AUTH007,
            synthetic.SECONDARY_LABEL: SECONDARY,
        },
    }
    assert manifest["data"] == {
        **copied,
        "cluster-id": "YXV0aC1sb2dpY2FsLWI=",
        "cluster-token": base64.b64encode(("t" * 48).encode()).decode(),
        "allowed-namespaces": "YXV0aC1sb2dpY2FsLWI=",
    }, "A's token and id never travel; B's own do"
    with pytest.raises(common.IdentityAcceptanceError, match="lacks keys to copy"):
        synthetic.connection_secret_manifest(
            cluster_id=SECONDARY,
            namespace=SECONDARY,
            run_id="run-1",
            case_id=AUTH007,
            token="t" * 48,
            copied={"control-plane-url": "dXJs"},
        )


def test_hold_probe_publishes_readiness_and_stops_on_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", SECONDARY)
    payload = hold.write_ready(tmp_path / "state")
    written = json.loads((tmp_path / "state" / "ready.json").read_text())
    assert written == json.loads(json.dumps(payload, sort_keys=True))
    assert written["cluster_id"] == SECONDARY
    assert written["claims_issued"] == 0, "the hold Pod never claims"
    assert hold.hold_until_stopped([True], poll_seconds=0) == 0


def test_owned_namespace_deletion_uses_the_cluster_scoped_path() -> None:
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    state = {"present": True}

    def client(*args: str, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        if args[0] == "delete":
            state["present"] = False
            return ""
        return (
            json.dumps(
                {
                    "uid": "uid-ns",
                    "resourceVersion": "7",
                    "labels": {seeded.RUN_LABEL: "run-1"},
                }
            )
            if state["present"]
            else ""
        )

    seeded.delete_owned_resource(
        "namespace",
        SECONDARY,
        "run-1",
        client=client,
        namespace=SECONDARY,
        expected_uid="uid-ns",
        require_uid=True,
    )
    deletion = next(args for args, _kwargs in calls if args[0] == "delete")
    assert deletion == ("delete", "--raw", f"/api/v1/namespaces/{SECONDARY}", "-f", "-")
    options = json.loads(
        next(kwargs["stdin"] for args, kwargs in calls if args[0] == "delete")
    )
    assert options["preconditions"] == {"uid": "uid-ns", "resourceVersion": "7"}


# --------------------------------------------------------------------------- #
# plan and execute through the entry point
# --------------------------------------------------------------------------- #
def cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, case: str = AUTH007
) -> tuple[FakeSite, common.IdentitySite, list[str], Path]:
    fake = FakeSite(tmp_path / "fake")
    site = fake.install(monkeypatch)
    run_dir = tmp_path / "run-20260101T000000Z"
    run_dir.mkdir()
    monkeypatch.setattr(entry, "install_site_profile", lambda: None)
    monkeypatch.setattr(entry.os, "umask", lambda mask: None)
    monkeypatch.setattr(entry, "IdentitySite", lambda path: site)
    monkeypatch.setattr(entry, "predecessor_path", lambda *args: (None, None))
    monkeypatch.setattr(live_driver_guard, "source_digest", lambda: "a" * 64)
    argv = [
        "identity",
        "--run-dir",
        str(run_dir),
        "--site",
        str(tmp_path / "fake" / "site"),
        "--case",
        case,
        "--cluster-id",
        PRIMARY,
        "--secondary-cluster-id",
        SECONDARY,
        "--secondary-registration",
        "synthetic",
        "--allow-synthetic-secondary",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return fake, site, argv, run_dir


def execute_arguments(case: str) -> list[str]:
    return [
        "--execute",
        "--confirm",
        case.removeprefix("GF-REGIONAL-").replace("-", "") + "_EXECUTE",
        "--maintenance-window-end",
        (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
    ]


def test_plan_records_the_intended_registration_and_pod_without_mutating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake, _site, _argv, run_dir = cli(monkeypatch, tmp_path)
    before = json.loads(json.dumps(fake.registry))
    assert entry.main() == 0
    plan = json.loads((run_dir / "cases" / AUTH007 / "plan.json").read_text())
    assert plan["preflight_passed"] is True
    assert plan["environment"]["GPU_FAULT_SECONDARY_REGISTRATION"] == "synthetic"
    assert plan["environment"]["GPU_FAULT_SYNTHETIC_SECONDARY_NAMESPACE"] == SECONDARY
    assert plan["details"]["secondary"] == {
        "cluster_id": SECONDARY,
        "context": "context-a",
        "registered": False,
        "kind": synthetic.SYNTHETIC_SECONDARY_KIND,
        "executor_namespace": SECONDARY,
    }
    card = plan["details"]["synthetic_secondary"]
    assert card["in_site"] is False
    assert card["registration"]["cluster_id"] == SECONDARY
    assert card["registration"]["synthetic"] is True
    assert card["registration"]["allowed_namespaces"] == [SECONDARY]
    assert card["registration"]["region"] == "us-west-2"
    assert card["registration"]["token"].startswith("minted at execute"), (
        "the plan never carries a token"
    )
    assert "token_sha256" not in card["registration"]
    assert card["pod"] == {
        "name": synthetic.SECONDARY_POD,
        "namespace": SECONDARY,
        "app": common.EXECUTOR_APP,
        "image": IMAGE,
        "probe": "auth_logical_secondary_hold.py",
        "connection_secret": common.CONNECTION_SECRET,
        "copied_from_primary": ["control-plane-url", "ca.crt"],
    }
    assert card["preflight"]["errors"] == []
    assert card["preflight"]["registration_absent"] is True
    assert card["preflight"]["namespace_absent"] is True
    assert fake.registry == before and fake.revisions == [] and fake.created == [], (
        "the plan is read-only"
    )
    assert set(fake.namespaces) == {SITE_NAMESPACE}


@pytest.mark.parametrize("residue", ["namespace", "registration", "bootstrap"])
def test_plan_fails_its_preflight_when_b_already_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, residue: str
) -> None:
    fake, _site, _argv, run_dir = cli(monkeypatch, tmp_path)
    if residue == "namespace":
        fake.namespaces[SECONDARY] = {
            "name": SECONDARY,
            "uid": "x",
            "resourceVersion": "1",
            "labels": {},
        }
    elif residue == "registration":
        fake.registry.append(
            RegionalClusterRegistration.model_validate(
                synthetic.redacted_registration(
                    synthetic.synthetic_registration(
                        SECONDARY,
                        run_id="other-run",
                        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                        region="us-west-2",
                        namespace=SECONDARY,
                    )
                )
            ).model_dump(mode="json")
        )
    else:
        monkeypatch.setattr(
            fake, "_registry_api", lambda arguments: {"status": 404, "body": {}}
        )
    assert entry.main() == 1
    plan = json.loads((run_dir / "cases" / AUTH007 / "plan.json").read_text())
    assert plan["preflight_passed"] is False
    errors = plan["details"]["synthetic_secondary"]["preflight"]["errors"]
    expected = {
        "namespace": "already exists",
        "registration": "already exists",
        "bootstrap": "durable registry",
    }[residue]
    assert any(expected in error for error in errors), errors
    assert fake.created == [] and fake.revisions == []


def test_execute_registers_b_runs_the_body_against_its_pod_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake, site, argv, run_dir = cli(monkeypatch, tmp_path)
    assert entry.main() == 0, "plan first"
    original = json.loads(json.dumps(fake.registry))
    observed: dict[str, Any] = {}

    def body(
        actual_site: Any, primary: Any, secondary: Any, *, case_dir: Path
    ) -> dict[str, Any]:
        assert actual_site is site
        row = fake.registration_of(SECONDARY)
        assert row is not None and row["enabled"] is True and row["synthetic"] is True
        observed["run_id"] = row["synthetic_run_id"]
        observed["token_sha256"] = row["token_sha256"]
        observed["pod"] = actual_site.any_executor_pod(secondary)
        observed["identity"] = actual_site.pod_json(
            "gpu", secondary, observed["pod"], AUTH008_EXECUTOR_IDENTITY_PROBE
        )
        observed["claim_b"] = common.claim(actual_site, secondary)
        observed["claim_a"] = common.claim(actual_site, primary)
        observed["secret_token_sha256"] = cluster_token_sha256(fake.secondary_token())
        observed["namespaces"] = sorted(fake.namespaces)
        assert (case_dir / synthetic.JOURNAL_FILE).is_file(), (
            "journal precedes the body"
        )
        return {"verdict": "PASS", "checks": {"body": True}, "cleanup_errors": []}

    monkeypatch.setattr(entry, "run_auth007", body)
    monkeypatch.setattr(sys, "argv", [*argv, *execute_arguments(AUTH007)])
    assert entry.main() == 0
    assert observed["pod"] == synthetic.SECONDARY_POD, "B's data plane is its own Pod"
    assert observed["identity"]["cluster_id"] == SECONDARY
    assert observed["identity"]["artifact"] == ARTIFACT
    assert observed["claim_b"]["status"] == 200 and observed["claim_a"]["status"] == 200
    assert observed["token_sha256"] == observed["secret_token_sha256"], (
        "the Pod's token is the one the registry digested"
    )
    assert observed["run_id"] == f"{SECONDARY}-20260101t000000z-a1"
    assert observed["namespaces"] == sorted([SITE_NAMESPACE, SECONDARY])
    document = json.loads((run_dir / "cases" / AUTH007 / f"{AUTH007}.json").read_text())
    assert document["verdict"] == "PASS"
    assert document["secondary_registered"] is True
    assert document["secondary_kind"] == "synthetic-logical"
    assert document["checks"] == {
        "body": True,
        "synthetic_secondary_cleanup_completed": True,
        "synthetic_secondary_residue_free": True,
    }
    card = document["synthetic_secondary"]
    assert (
        card["registry_generation_before"],
        card["registry_generation_with_secondary"],
        card["registry_generation_after"],
    ) == (3, 4, 5)
    assert card["registration"]["token_sha256"] == observed["token_sha256"]
    assert card["registration"]["synthetic_run_id"] == observed["run_id"]
    assert card["pod"]["name"] == synthetic.SECONDARY_POD
    assert card["pod"]["namespace"] == SECONDARY
    assert card["pod"]["image"] == IMAGE
    assert card["pod"]["uid"].startswith("uid-") and card["pod"]["node"] == "node-1"
    assert card["proof"] == {
        "registry_generation_after": 5,
        "registration_absent": True,
        "primary_registration_untouched": True,
        "namespace_absent": True,
        "pod_absent": True,
        "residue_free": True,
    }
    assert card["cleanup_errors"] == []
    assert set(card["cleanup"]) == {
        "delete_pod",
        "remove_registration",
        "delete_connection_secret",
        "delete_hold_configmap",
        "delete_namespace",
        "proof",
    }
    assert any(
        item.startswith(synthetic.SYNTHETIC_LIMITATION[:40])
        for item in document["limitations"]
    ), "the synthetic B limitation is declared"
    assert fake.registry == original, "A's registration is what it was; B is gone"
    assert [item["reason"].split(": ")[-1] for item in fake.revisions] == [
        f"register {SECONDARY}",
        f"remove {SECONDARY}",
    ]
    assert fake.owned_resources() == [] and set(fake.namespaces) == {SITE_NAMESPACE}
    assert fake.deleted == [
        f"pod/{SECONDARY}/{synthetic.SECONDARY_POD}",
        f"secret/{SECONDARY}/{common.CONNECTION_SECRET}",
        f"configmap/{SECONDARY}/{synthetic.SECONDARY_HOLD_CONFIGMAP}",
        f"namespace/{SECONDARY}",
    ]
    journal = json.loads(
        (run_dir / "cases" / AUTH007 / synthetic.JOURNAL_FILE).read_text()
    )
    assert (
        journal["registry_started"] is True
        and journal["pod"]["name"] == synthetic.SECONDARY_POD
    )


def test_execute_cleans_up_even_when_the_body_fails_mid_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake, _site, argv, run_dir = cli(monkeypatch, tmp_path)
    assert entry.main() == 0
    original = json.loads(json.dumps(fake.registry))

    def body(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert fake.registration_of(SECONDARY) is not None, "B was registered"
        raise RuntimeError("probe exploded mid-case")

    monkeypatch.setattr(entry, "run_auth007", body)
    monkeypatch.setattr(sys, "argv", [*argv, *execute_arguments(AUTH007)])
    assert entry.main() == 1
    document = json.loads((run_dir / "cases" / AUTH007 / f"{AUTH007}.json").read_text())
    assert document["verdict"] == "FAIL"
    assert document["error"] == "RuntimeError: probe exploded mid-case"
    assert document["cleanup_errors"] == []
    partial = document["partial"]
    assert partial["secondary_registered"] is True
    assert partial["synthetic_secondary"]["proof"]["residue_free"] is True
    assert partial["synthetic_secondary"]["registry_generation_after"] == 5
    assert fake.registry == original and fake.owned_resources() == []
    assert set(fake.namespaces) == {SITE_NAMESPACE}


def test_execute_refuses_before_any_mutation_when_b_appeared_after_the_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake, _site, argv, run_dir = cli(monkeypatch, tmp_path)
    assert entry.main() == 0
    fake.namespaces[SECONDARY] = {
        "name": SECONDARY,
        "uid": "foreign",
        "resourceVersion": "1",
        "labels": {},
    }
    monkeypatch.setattr(
        entry, "run_auth007", lambda *args, **kwargs: pytest.fail("body must not run")
    )
    monkeypatch.setattr(sys, "argv", [*argv, *execute_arguments(AUTH007)])
    assert entry.main() == 1
    document = json.loads((run_dir / "cases" / AUTH007 / f"{AUTH007}.json").read_text())
    assert document["verdict"] == "FAIL"
    assert "already exists" in document["error"]
    assert fake.created == [] and fake.revisions == [] and fake.deleted == [], (
        "a refused preflight mutates nothing and deletes nothing foreign"
    )
    assert document["partial"]["synthetic_secondary"]["cleanup"] == {
        "skipped": "no owned mutation attempted",
        "proof": document["partial"]["synthetic_secondary"]["proof"],
    }
    assert (
        document["partial"]["synthetic_secondary"]["proof"]["namespace_absent"] is False
    )


def test_real_auth007_body_disables_and_restores_the_synthetic_b(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake, _site, argv, run_dir = cli(monkeypatch, tmp_path)
    assert entry.main() == 0
    original = json.loads(json.dumps(fake.registry))
    monkeypatch.setattr(sys, "argv", [*argv, *execute_arguments(AUTH007)])
    assert entry.main() == 0
    document = json.loads((run_dir / "cases" / AUTH007 / f"{AUTH007}.json").read_text())
    assert document["verdict"] == "PASS", document
    assert document["checks"] == {
        "secondary_disabled_403": True,
        "primary_remains_200": True,
        "secondary_recovers_200": True,
        "registry_restored": True,
        "cleanup_completed": True,
        "synthetic_secondary_cleanup_completed": True,
        "synthetic_secondary_residue_free": True,
    }
    assert [claim["cluster_id"] for claim in fake.claims] == [
        SECONDARY,
        PRIMARY,
        SECONDARY,
    ], "B disabled, A healthy, B recovered -- all through Pods of their own"
    assert [item["reason"].split(": ")[-1] for item in fake.revisions][0] == (
        f"register {SECONDARY}"
    )
    assert [item["cluster_ids"] for item in fake.revisions] == [
        [PRIMARY, SECONDARY],
        [PRIMARY, SECONDARY],
        [PRIMARY, SECONDARY],
        [PRIMARY],
    ], "register, disable, restore, remove"
    generations = document["synthetic_secondary"]
    assert (
        generations["registry_generation_before"],
        generations["registry_generation_with_secondary"],
        generations["registry_generation_after"],
    ) == (3, 4, 7)
    assert fake.registry == original and fake.owned_resources() == []


def test_real_auth008_body_leases_only_through_the_synthetic_b(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeSite(tmp_path / "fake")
    site = fake.install(monkeypatch)
    context = build_context()
    context.regional_mode = True

    def mirror(rows: list[dict[str, Any]]) -> None:
        wanted = {row["cluster_id"] for row in rows}
        for cluster_id in list(context.store.list_regional_cluster_ids()):
            if cluster_id not in wanted:
                context.store.delete_regional_cluster(cluster_id)
        for row in rows:
            context.store.save_regional_cluster(
                RegionalClusterRegistration.model_validate(row)
            )

    mirror(fake.registry)
    fake.on_publish = mirror
    monkeypatch.setattr(ApplicationContext, "from_environment", lambda: context)
    # A CPU Pod exports the release pins, never GPU_FAULT_RELEASE_ID.
    monkeypatch.delenv("GPU_FAULT_RELEASE_ID", raising=False)
    for name, value in POD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    fake.exec_hooks[AUTH008_BACKLOG_PROBE] = lambda _env, args: execute_probe(
        monkeypatch, AUTH008_BACKLOG_PROBE, args[0]
    )
    posted: list[tuple[str, str, int]] = []

    def claim(regional: Any, **kwargs: Any) -> dict[str, Any]:
        origin = regional.settings.cluster_id
        token = TOKEN_A if origin == PRIMARY else fake.secondary_token()
        payload = {
            **audit.claim_payload(
                executor_id=kwargs["executor_id"],
                artifact_sha256=kwargs["identity"]["artifact"],
                compatibility_digest=kwargs["identity"]["compatibility"],
            ),
            "executor_protocol_version": kwargs["identity"]["protocol"],
        }

        async def post() -> Any:
            async with asgi_client(context) as client:
                return await client.post(
                    "/v1/regional/executors/claim",
                    headers={
                        "Authorization": "Bearer " + token,
                        "X-GPU-Fault-Cluster-ID": kwargs["header_cluster"],
                    },
                    json=payload,
                )

        response = asyncio.run(post())
        body = response.json()
        if "commands" in body:
            body["commands"] = [
                {
                    key: command[key]
                    for key in (
                        "command_id",
                        "cluster_id",
                        "workflow_request_id",
                        "incident_id",
                        "status",
                    )
                }
                for command in body["commands"]
            ]
        posted.append((origin, kwargs["header_cluster"], response.status_code))
        return {"status": response.status_code, "body": body}

    monkeypatch.setattr(backlog, "auth008_claim", claim)
    primary = site.target(PRIMARY)
    secondary = synthetic.synthetic_secondary(primary, SECONDARY)
    case_dir = tmp_path / "cases" / AUTH008
    case_dir.mkdir(parents=True)
    lifecycle = synthetic.SyntheticSecondary(
        site,
        primary,
        secondary,
        case_id=AUTH008,
        case_dir=case_dir,
        run_id=f"{SECONDARY}-unit-a1",
    )
    outcome = synthetic.run_with_synthetic_secondary(
        lifecycle,
        lambda: backlog.run_auth008(site, primary, secondary, case_dir=case_dir),
    )
    assert outcome["verdict"] == "PASS", outcome
    assert outcome["checks"]["b_can_claim_exact_candidate"] is True
    assert outcome["checks"]["precise_claim_responses"] is True
    assert outcome["entries"]["AUTH-008-B-header-A-token"]["status"] == 403
    assert outcome["entries"]["AUTH-008-A-header-B-token"]["status"] == 403
    assert outcome["entries"]["AUTH-008-A-normal"]["status"] == 200
    assert outcome["positive_b_claim"]["status"] == 200
    assert outcome["positive_b_claim"]["body"]["commands"][0]["cluster_id"] == SECONDARY
    assert posted == [
        (PRIMARY, PRIMARY, 200),
        (PRIMARY, PRIMARY, 200),
        (PRIMARY, SECONDARY, 403),
        (SECONDARY, PRIMARY, 403),
        (SECONDARY, SECONDARY, 200),
    ], "negatives cross header and token; the positive is B's own Pod, header and token"
    assert outcome["candidate_before"]["cluster_id"] == SECONDARY
    assert outcome["secondary_registered"] is True
    assert outcome["synthetic_secondary"]["proof"]["residue_free"] is True
    assert context.store.list_regional_cluster_ids() == [PRIMARY], (
        "B left the store with the removing revision"
    )
    assert context.store.list_remote_commands()[0].status.value == "FAILED", (
        "the owned B command was retired by the body's own cleanup"
    )
    assert fake.owned_resources() == [] and set(fake.namespaces) == {SITE_NAMESPACE}
