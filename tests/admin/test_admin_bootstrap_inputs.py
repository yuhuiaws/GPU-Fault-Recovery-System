"""The two inputs bootstrap derives from a cluster it did not create.

The fleet master secret is the key every node's action tokens are derived from, so
a bootstrap that regenerates it on a second run silently invalidates the whole
fleet; and it must never reach a log. The ADOT image is the only container image
this solution runs that is not part of its own signed release, so where it comes
from decides whether a bootstrap can pull it at all in a private-registry account.
"""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import pytest

from gpu_fault.admin import bootstrap as admin_bootstrap
from gpu_fault.admin.bootstrap import (
    DEFAULT_ADOT_IMAGE_AMD64,
    _discover_adot_image,
    _ensure_base_secrets,
)
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner

NAMESPACE = "gpu-fault-system"
SECRET = "gpu-fault-control-plane-active"
EXISTING_MASTER = "9" * 64


class Kubectl:
    """The control-plane cluster's answer to the reads bootstrap makes.

    ``present`` decides whether the base Secret already exists, which is the one
    branch that separates a first bootstrap from a re-run.
    """

    def __init__(
        self,
        *,
        present: bool = False,
        master: str = EXISTING_MASTER,
        documents: Sequence[dict[str, Any]] = (),
        architectures: Sequence[str] = ("amd64",),
    ) -> None:
        self.present = present
        self.master = master
        self.documents = list(documents)
        self.architectures = list(architectures)
        self.calls: list[list[str]] = []
        self.inputs: list[str] = []

    def __call__(
        self, arguments: Sequence[Any], **keywords: Any
    ) -> subprocess.CompletedProcess:
        argv = [str(item) for item in arguments]
        self.calls.append(argv)
        stdin = keywords.get("input")
        if isinstance(stdin, str):
            self.inputs.append(stdin)
        line = " ".join(argv)
        if "jsonpath={.data.node-action-secret}" in line:
            encoded = base64.b64encode(self.master.encode()).decode()
            return subprocess.CompletedProcess(argv, 0, encoded, "")
        if "deployment,daemonset" in line:
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"items": self.documents}), ""
            )
        if "nodes" in argv:
            items = [
                {"status": {"nodeInfo": {"architecture": value}}}
                for value in self.architectures
            ]
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"items": items}), ""
            )
        if "get" in argv and SECRET in argv:
            return subprocess.CompletedProcess(
                argv, 0 if self.present else 1, "", "NotFound"
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    def matching(self, fragment: str) -> list[list[str]]:
        return [argv for argv in self.calls if fragment in " ".join(argv)]

    def applied_secrets(self) -> list[dict[str, str]]:
        """Every Secret manifest that reached kubectl over stdin (never argv)."""
        secrets_seen: list[dict[str, str]] = []
        for text in self.inputs:
            try:
                document = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(document, dict) and document.get("kind") == "Secret":
                secrets_seen.append(
                    {
                        str(key): str(value)
                        for key, value in (document.get("stringData") or {}).items()
                    }
                )
        return secrets_seen

    def literals(self) -> dict[str, str]:
        secrets_seen = self.applied_secrets()
        assert secrets_seen, "the base Secret was never applied"
        return secrets_seen[0]


def _secrets(kubectl: Kubectl, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(admin_bootstrap.subprocess, "run", kubectl)
    return _ensure_base_secrets(
        CommandRunner(),
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace=NAMESPACE,
        secure_dir=tmp_path / "secure",
    )


def test_a_first_bootstrap_mints_three_independent_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The execution token, replay secret and node action key are separate keys.

    They protect three different paths -- API execution, processor replay, and the
    per-node action keys -- so a single shared value would let one leaked secret
    stand in for the others.
    """

    kubectl = Kubectl(present=False)

    master_file = _secrets(kubectl, monkeypatch, tmp_path)

    literals = kubectl.literals()
    assert set(literals) == {
        "execution-token",
        "processor-replay-secret",
        "node-action-secret",
    }
    assert len(set(literals.values())) == 3, "two base secrets were given one value"
    for value in literals.values():
        assert len(value) == 64, "a base secret is shorter than 32 bytes of entropy"
    assert master_file.read_text(encoding="utf-8") == literals["node-action-secret"]


def test_the_fleet_master_file_is_readable_only_by_its_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every node action key is derived from this file.

    It sits on the operator's workstation for the length of the bootstrap; a
    group-readable copy is a fleet-wide compromise.
    """

    master_file = _secrets(Kubectl(present=False), monkeypatch, tmp_path)

    assert master_file.stat().st_mode & 0o777 == 0o600
    assert master_file.parent.stat().st_mode & 0o777 == 0o700


def test_a_re_run_reads_the_existing_master_instead_of_replacing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regenerating the master key would invalidate every node already joined.

    Their action keys are derived from it, so the nodes would keep reporting faults
    that the control plane then refuses to act on.
    """

    kubectl = Kubectl(present=True)

    master_file = _secrets(kubectl, monkeypatch, tmp_path)

    assert master_file.read_text(encoding="utf-8") == EXISTING_MASTER
    assert kubectl.applied_secrets() == [], "an existing base Secret was overwritten"


def test_the_secret_values_never_reach_the_command_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bootstrap echoes every command it runs, and this one carries three keys.

    The operator's terminal and CI log would otherwise hold the fleet master secret
    in clear text for as long as the log is kept. The keys travel in the Secret
    manifest over kubectl's stdin, so neither the echoed command nor the file
    should ever contain them.
    """

    kubectl = Kubectl(present=False)

    master_file = _secrets(kubectl, monkeypatch, tmp_path)
    logged = capsys.readouterr().err

    assert master_file.read_text(encoding="utf-8") not in logged
    for value in kubectl.literals().values():
        assert value not in logged, "a base secret was echoed to the log"


def test_the_secret_values_never_reach_the_kubectl_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H-7: a secret on the argv is world-readable via /proc and shell history.

    The base Secret is created by piping a manifest to ``kubectl apply -f -`` over
    stdin, so no minted key may appear on any command line bootstrap constructs.
    """

    kubectl = Kubectl(present=False)

    _secrets(kubectl, monkeypatch, tmp_path)

    minted = kubectl.literals()
    assert set(minted) == {
        "execution-token",
        "processor-replay-secret",
        "node-action-secret",
    }
    assert not any(
        argv for argv in kubectl.calls if any("--from-literal" in item for item in argv)
    ), "a secret was passed with --from-literal"
    for value in minted.values():
        for argv in kubectl.calls:
            assert value not in argv, "a secret value reached the kubectl argv"
            assert not any(value in item for item in argv), (
                "a secret value was embedded in a kubectl argument"
            )


def test_an_existing_master_is_read_back_from_its_stored_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kubernetes stores Secret data base64-encoded.

    Writing the encoded form to the fleet master file would derive node action keys
    that no node can reproduce, and the mismatch only shows up as rejected actions
    at recovery time.
    """

    kubectl = Kubectl(present=True, master="a1b2c3")

    master_file = _secrets(kubectl, monkeypatch, tmp_path)

    assert master_file.read_text(encoding="utf-8") == "a1b2c3"


def _deployment(*images: str) -> dict[str, Any]:
    return {
        "spec": {
            "template": {"spec": {"containers": [{"image": image} for image in images]}}
        }
    }


def _adot(
    kubectl: Kubectl,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    configured: str | None = None,
) -> str:
    # The variable is read from the real environment, so every test states whether
    # it is set rather than inheriting whatever the shell had.
    if configured is None:
        monkeypatch.delenv("GPU_FAULT_ADOT_IMAGE", raising=False)
    else:
        monkeypatch.setenv("GPU_FAULT_ADOT_IMAGE", configured)
    monkeypatch.setattr(admin_bootstrap.subprocess, "run", kubectl)
    return _discover_adot_image(
        CommandRunner(), cpu_kubeconfig=tmp_path / "cpu.kubeconfig"
    )


def test_an_operator_supplied_adot_image_wins_over_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An account with a private registry has to be able to name its own mirror.

    Discovery would otherwise pick a public image the cluster cannot pull, and the
    control plane would come up with no metrics pipeline at all.
    """

    kubectl = Kubectl()

    image = _adot(
        kubectl, monkeypatch, tmp_path, configured=" registry.example/adot@sha256:abc "
    )

    assert image == "registry.example/adot@sha256:abc"
    assert kubectl.calls == [], "the cluster was queried despite an explicit image"


def test_an_image_already_running_in_the_cluster_is_preferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cluster has already proved it can pull that image.

    Reusing it keeps the collector on whatever registry and version the account has
    approved instead of introducing a second source.
    """

    kubectl = Kubectl(
        documents=[
            _deployment("public.ecr.aws/foo/unrelated:1"),
            _deployment(
                "registry.example/aws-observability/aws-otel-collector:v0.40.0"
            ),
        ]
    )

    assert _adot(kubectl, monkeypatch, tmp_path) == (
        "registry.example/aws-observability/aws-otel-collector:v0.40.0"
    )
    assert kubectl.matching("get nodes") == [], (
        "the fallback ran even though the cluster already had an image"
    )


def test_the_choice_between_several_images_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two bootstraps of the same cluster must select the same image.

    The image is part of the rendered control-plane manifests, so an unstable choice
    would show up as a spurious diff -- and a redeploy -- on every re-run.
    """

    images = ["registry.example/adot:v2", "registry.example/aws-otel-collector:v1"]
    forward = Kubectl(documents=[_deployment(*images)])
    reverse = Kubectl(documents=[_deployment(*reversed(images))])

    assert _adot(forward, monkeypatch, tmp_path) == _adot(
        reverse, monkeypatch, tmp_path
    )


def test_a_uniformly_amd64_cluster_falls_back_to_the_pinned_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The built-in default is pinned by digest, and it is amd64 only.

    Pinning by digest is what makes the fallback auditable; using it on a cluster
    whose nodes are amd64 is the one case where it is known to run.
    """

    kubectl = Kubectl(documents=[], architectures=["amd64", "amd64"])

    assert _adot(kubectl, monkeypatch, tmp_path) == DEFAULT_ADOT_IMAGE_AMD64
    assert "@sha256:" in DEFAULT_ADOT_IMAGE_AMD64, (
        "the fallback image is not pinned by digest"
    )


def test_a_mixed_architecture_cluster_is_refused_with_the_way_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An amd64-only image would CrashLoop on the arm64 nodes.

    The collector runs as a Deployment that can land on any node, so the failure
    would be intermittent; refusing at bootstrap names the environment variable that
    fixes it.
    """

    kubectl = Kubectl(documents=[], architectures=["amd64", "arm64"])

    with pytest.raises(BootstrapError, match="GPU_FAULT_ADOT_IMAGE"):
        _adot(kubectl, monkeypatch, tmp_path)
