"""BOOT-019 resume-identity guards and the synthetic secondary's refusals.

``epoch_targets`` with a resume document must accept only the membership the
recorded phase permits, and a revocation probe must refuse when the trust root
it would present has changed since the token was captured. The AUTH-007/008
synthetic cluster B must refuse an identity that collides with A, an unknown
registration mode, an executor Deployment without its pins, and must report a
disabled primary registration or a connection Secret missing copy keys as
preflight errors rather than crash.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import identity_acceptance_common as common
from scripts.e2e.regional import identity_synthetic_secondary as synthetic
from scripts.e2e.regional import run_boot019_admin_lifecycle as boot019
from tests.regional._regional_support import TOKEN_A, registration

# --- BOOT-019 -----------------------------------------------------------------


def _arn(name: str) -> str:
    return f"arn:aws:eks:us-west-2:000000000000:cluster/{name}"


DISPOSABLE_CPU = _arn("disposable-cpu")
DISPOSABLE_GPU = _arn("disposable-gpu")
PROTECTED_CPU = _arn("persistent-cpu")
PROTECTED_GPU = _arn("persistent-gpu")
JOIN = _arn("new-disposable-gpu")


def _sites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *gpu: str
) -> tuple[Path, Path]:
    disposable = {
        "cpu_eks_arn": DISPOSABLE_CPU,
        "clusters": [{"eks_cluster_arn": arn} for arn in (gpu or (DISPOSABLE_GPU,))],
    }
    protected = {
        "cpu_eks_arn": PROTECTED_CPU,
        "clusters": [{"eks_cluster_arn": PROTECTED_GPU}],
    }
    left, right = tmp_path / "disposable", tmp_path / "protected"
    monkeypatch.setattr(
        boot019,
        "load_site",
        lambda path, **_kw: SimpleNamespace(
            release_config=disposable if path == left else protected
        ),
    )
    return left, right


def _resume(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "case_id": boot019.CASE_ID,
        "inputs": {
            "epoch_targets": {
                "lifecycle_cpu_eks_arn": DISPOSABLE_CPU,
                "lifecycle_gpu_eks_arns": [DISPOSABLE_GPU],
                "join_gpu_eks_arn": JOIN,
                "protected_cpu_eks_arn": PROTECTED_CPU,
                "protected_gpu_eks_arns": [PROTECTED_GPU],
            }
        },
        "stages": {},
    }
    document.update(overrides)
    return document


def test_resume_accepts_the_recorded_baseline_membership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    left, right = _sites(monkeypatch, tmp_path)

    result = boot019.epoch_targets(left, right, JOIN, resume=_resume())

    assert result["lifecycle_gpu_eks_arns"] == [DISPOSABLE_GPU]
    assert result["join_gpu_eks_arn"] == JOIN


@pytest.mark.parametrize(
    "drift",
    [
        {"case_id": "GF-REGIONAL-BOOT-018"},
        {"inputs": {}},
        {"inputs": {"epoch_targets": "not-a-mapping"}},
    ],
)
def test_resume_refuses_another_case_or_a_missing_target_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, drift: dict[str, Any]
) -> None:
    left, right = _sites(monkeypatch, tmp_path)

    with pytest.raises(boot019.AcceptanceCheckError, match="target identity changed"):
        boot019.epoch_targets(left, right, JOIN, resume=_resume(**drift))


def test_resume_refuses_a_changed_physical_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    left, right = _sites(monkeypatch, tmp_path)
    resume = _resume()
    resume["inputs"]["epoch_targets"]["protected_gpu_eks_arns"] = [_arn("other")]

    with pytest.raises(boot019.AcceptanceCheckError, match="target identity changed"):
        boot019.epoch_targets(left, right, JOIN, resume=resume)


@pytest.mark.parametrize("baseline", [None, [], [DISPOSABLE_GPU, _arn("second")]])
def test_resume_requires_exactly_one_recorded_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, baseline: Any
) -> None:
    left, right = _sites(monkeypatch, tmp_path)
    resume = _resume()
    resume["inputs"]["epoch_targets"]["lifecycle_gpu_eks_arns"] = baseline

    with pytest.raises(boot019.AcceptanceCheckError, match="lacks its baseline"):
        boot019.epoch_targets(left, right, JOIN, resume=resume)


def test_resume_after_a_failed_join_permits_the_joined_cluster_in_the_site(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    left, right = _sites(monkeypatch, tmp_path, DISPOSABLE_GPU, JOIN)
    resume = _resume(stages={"join_failure_before_activation": {"ok": True}})

    result = boot019.epoch_targets(left, right, JOIN, resume=resume)

    # The baseline, not the live two-cluster membership, is what the epoch runs on.
    assert result["lifecycle_gpu_eks_arns"] == [DISPOSABLE_GPU]


def test_resume_after_the_remove_snapshot_permits_an_empty_site(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    disposable = {"cpu_eks_arn": DISPOSABLE_CPU, "clusters": []}
    protected = {
        "cpu_eks_arn": PROTECTED_CPU,
        "clusters": [{"eks_cluster_arn": PROTECTED_GPU}],
    }
    left, right = tmp_path / "disposable", tmp_path / "protected"
    monkeypatch.setattr(
        boot019,
        "load_site",
        lambda path, **_kw: SimpleNamespace(
            release_config=disposable if path == left else protected
        ),
    )
    resume = _resume(stages={"post_remove_snapshot": {"ok": True}})

    result = boot019.epoch_targets(left, right, JOIN, resume=resume)

    assert result["lifecycle_gpu_eks_arns"] == [DISPOSABLE_GPU]


@pytest.mark.parametrize(
    ("gpu", "stages"),
    [
        ((JOIN,), {}),
        ((DISPOSABLE_GPU, JOIN), {}),
        ((), {}),
        ((DISPOSABLE_GPU, DISPOSABLE_GPU), {"join_failure_before_activation": {}}),
    ],
)
def test_resume_refuses_membership_outside_the_recorded_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gpu: tuple[str, ...],
    stages: dict[str, Any],
) -> None:
    disposable = {
        "cpu_eks_arn": DISPOSABLE_CPU,
        "clusters": [{"eks_cluster_arn": arn} for arn in gpu],
    }
    protected = {
        "cpu_eks_arn": PROTECTED_CPU,
        "clusters": [{"eks_cluster_arn": PROTECTED_GPU}],
    }
    left, right = tmp_path / "disposable", tmp_path / "protected"
    monkeypatch.setattr(
        boot019,
        "load_site",
        lambda path, **_kw: SimpleNamespace(
            release_config=disposable if path == left else protected
        ),
    )

    with pytest.raises(boot019.AcceptanceCheckError, match="recorded phase"):
        boot019.epoch_targets(left, right, JOIN, resume=_resume(stages=stages))


def test_revocation_probe_refuses_a_changed_trust_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_file = tmp_path / "cluster-b.token"
    token_file.write_bytes(b"t" * 40 + b"\n")
    ca_file = tmp_path / "ca.pem"
    ca_file.write_bytes(b"fixture-trust-root")
    site = SimpleNamespace(
        release_config={
            "clusters": [
                {
                    "cluster_id": "cluster-b",
                    "token_file": str(token_file),
                    "control_plane_url": "https://control.example.invalid",
                    "ca_file": str(ca_file),
                }
            ]
        }
    )
    monkeypatch.setattr(boot019, "load_site", lambda *_a, **_k: site)
    backend = boot019.LiveAdminLifecycleBackend(
        site_path=tmp_path / "site.yaml",
        gpu_cluster_arn="arn:aws:eks:us-west-2:000000000000:cluster/gpu",
        cluster_id=None,
        allowed_namespaces=("default",),
        join_state_dir=tmp_path / "join",
        run_dir=tmp_path / "run",
    )
    capture = backend.capture_joined_token("cluster-b")
    requests: list[Any] = []
    monkeypatch.setattr(boot019.urllib.request, "urlopen", requests.append)

    ca_file.write_bytes(b"a-different-trust-root")

    with pytest.raises(boot019.AcceptanceCheckError, match="trust identity changed"):
        backend.probe_revoked_token(capture)
    assert requests == []


# --- synthetic secondary --------------------------------------------------------

PRIMARY = "cluster-a"
SECONDARY = "auth-logical-b"


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


def test_a_synthetic_secondary_must_not_be_the_primary() -> None:
    with pytest.raises(common.IdentityAcceptanceError, match="must differ"):
        synthetic.synthetic_secondary(_primary(), PRIMARY)


def test_an_unknown_registration_mode_is_refused() -> None:
    arguments = argparse.Namespace(
        case="GF-REGIONAL-AUTH-007",
        cluster_id=PRIMARY,
        secondary_cluster_id=SECONDARY,
        secondary_registration="imaginary",
        allow_synthetic_secondary=True,
    )

    with pytest.raises(common.IdentityAcceptanceError, match="unknown secondary"):
        synthetic.synthetic_secondary_from_arguments(
            arguments, SimpleNamespace(), _primary()
        )


def _deployment(*, image: str, artifact: str, compatibility: str) -> str:
    return json.dumps(
        {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "image": image,
                                "env": [
                                    {
                                        "name": "GPU_FAULT_EXECUTOR_ARTIFACT_SHA256",
                                        "value": artifact,
                                    },
                                    {
                                        "name": "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST",
                                        "value": compatibility,
                                    },
                                    {"name": "FROM_SECRET", "valueFrom": {}},
                                ],
                            }
                        ]
                    }
                }
            }
        }
    )


@pytest.mark.parametrize(
    ("image", "artifact", "compatibility"),
    [
        ("", "a" * 64, "b" * 64),
        ("registry/executor@sha256:" + "0" * 64, "short", "b" * 64),
        ("registry/executor@sha256:" + "0" * 64, "a" * 64, "B" * 64),
    ],
)
def test_executor_template_requires_an_image_and_hex_pins(
    image: str, artifact: str, compatibility: str
) -> None:
    site = SimpleNamespace(
        gpu=lambda *_a: _deployment(
            image=image, artifact=artifact, compatibility=compatibility
        )
    )

    with pytest.raises(common.IdentityAcceptanceError, match="64-hex"):
        synthetic.executor_template(site, _primary())


def _secondary() -> common.ClusterTarget:
    return synthetic.synthetic_secondary(_primary(), SECONDARY)


def _build(
    site: Any, secondary: common.ClusterTarget, tmp_path: Path, **overrides: Any
) -> synthetic.SyntheticSecondary:
    return synthetic.SyntheticSecondary(
        site,
        _primary(),
        secondary,
        case_id="GF-REGIONAL-AUTH-007",
        case_dir=tmp_path,
        run_id="run-1",
        **overrides,
    )


def test_the_secondary_must_be_a_complete_synthetic_target(tmp_path: Path) -> None:
    with pytest.raises(common.IdentityAcceptanceError, match="not a synthetic target"):
        _build(SimpleNamespace(), _primary("auth-logical-site"), tmp_path)
    with pytest.raises(common.IdentityAcceptanceError, match="identity is incomplete"):
        _build(
            SimpleNamespace(), replace(_secondary(), executor_namespace=""), tmp_path
        )
    with pytest.raises(common.IdentityAcceptanceError, match="identity is incomplete"):
        _build(SimpleNamespace(), replace(_secondary(), cluster_id=PRIMARY), tmp_path)


@pytest.mark.parametrize(
    "lifetime", [timedelta(minutes=4), timedelta(hours=2, seconds=1)]
)
def test_the_secondary_lifetime_is_bounded(tmp_path: Path, lifetime: timedelta) -> None:
    with pytest.raises(common.IdentityAcceptanceError, match="outside 5m..2h"):
        _build(SimpleNamespace(), _secondary(), tmp_path, lifetime=lifetime)


class _PlaneSite:
    """The site surface ``read_only_preflight`` reads, scripted per test."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        generation: int | None = 3,
        secret_data: dict[str, str] | None = None,
    ) -> None:
        self.rows = rows
        self.generation = generation
        self.secret_data = (
            {"control-plane-url": "aHR0cHM6Ly9jb250cm9s", "ca.crt": "Y2E="}
            if secret_data is None
            else secret_data
        )
        self.gpu_calls: list[tuple[str, ...]] = []

    def registry_generation(self) -> int | None:
        return self.generation

    def registry(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.rows]

    def gpu(self, _target: common.ClusterTarget, *arguments: str) -> str:
        self.gpu_calls.append(arguments)
        if arguments[1] == "deployment":
            return _deployment(
                image="registry/executor@sha256:" + "0" * 64,
                artifact="a" * 64,
                compatibility="b" * 64,
            )
        if arguments[1] == "secret":
            return json.dumps({"data": self.secret_data})
        raise AssertionError(arguments)

    def regional(self, _target: common.ClusterTarget) -> Any:
        return SimpleNamespace(kubectl=lambda *_a, **_k: "")


def test_preflight_reports_a_disabled_primary_and_missing_copy_keys(
    tmp_path: Path,
) -> None:
    primary_row = registration(PRIMARY, TOKEN_A).model_dump(mode="json")
    primary_row["enabled"] = False
    site = _PlaneSite([primary_row], secret_data={"ca.crt": "Y2E="})
    fixture = _build(site, _secondary(), tmp_path)

    result = fixture.read_only_preflight()

    assert result["primary_registration_enabled"] is False
    assert result["registration_absent"] is True
    assert result["namespace_absent"] is True
    assert result["executor"]["artifact"] == "a" * 64
    assert "connection_keys_copied" not in result
    assert result["errors"] == [
        "the primary registration is absent or disabled",
        "connection Secret: IdentityAcceptanceError: primary connection Secret "
        "lacks keys to copy: control-plane-url",
    ]
    assert fixture.journal.primary_registration is not None
    assert fixture.journal.primary_registration["cluster_id"] == PRIMARY


def test_preflight_keeps_the_primary_row_it_already_journaled(tmp_path: Path) -> None:
    other = registration("cluster-z", TOKEN_A).model_dump(mode="json")
    primary_row = registration(PRIMARY, TOKEN_A).model_dump(mode="json")
    site = _PlaneSite([other, primary_row])
    fixture = _build(site, _secondary(), tmp_path)
    fixture.journal.primary_registration = {"cluster_id": PRIMARY, "frozen": True}

    result = fixture.read_only_preflight()

    assert result["errors"] == []
    assert result["connection_keys_copied"] == ["control-plane-url", "ca.crt"]
    assert fixture.journal.primary_registration == {
        "cluster_id": PRIMARY,
        "frozen": True,
    }


def test_preflight_journals_the_primary_row_found_after_other_clusters(
    tmp_path: Path,
) -> None:
    other = registration("cluster-z", TOKEN_A).model_dump(mode="json")
    primary_row = registration(PRIMARY, TOKEN_A).model_dump(mode="json")
    site = _PlaneSite([other, primary_row])
    fixture = _build(site, _secondary(), tmp_path)

    fixture.read_only_preflight()

    journaled = fixture.journal.primary_registration
    assert journaled is not None
    assert journaled["cluster_id"] == PRIMARY
    assert journaled["token_sha256"] == primary_row["token_sha256"]


def test_preflight_journals_nothing_when_the_primary_row_is_absent(
    tmp_path: Path,
) -> None:
    other = registration("cluster-z", TOKEN_A).model_dump(mode="json")
    fixture = _build(_PlaneSite([other], generation=None), _secondary(), tmp_path)

    result = fixture.read_only_preflight()

    assert fixture.journal.primary_registration is None
    assert result["errors"][:2] == [
        "a synthetic secondary requires a durable registry",
        "the primary registration is absent or disabled",
    ]
