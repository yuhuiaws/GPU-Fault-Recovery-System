"""The DCGM exporter's counters CSV and the roll that delivers it (batch 127).

Sibling of ``test_dataplane_manifest_contracts.py`` (which is at the
architecture ratchet's file limit), in its style: every check names the outage
it exists to stop. Both outages hit every GPU node at once:

* A counters-only change could not reach the exporter. dcgm-exporter reads
  its collectors CSV only at process start, and the DaemonSet mounts the
  ``gpu-fault-dcgm-counters`` ConfigMap with ``subPath``, which kubelet never
  refreshes in a running Pod. The release applied the new ConfigMap and an
  unchanged pod template -- an unchanged template restarts nothing -- so new
  fields (the Fabric Manager state the passive fabric detection reads) would
  never have been exported while every rollout passed; latent only because
  the CSV had not changed since its import. The pod template now carries
  ``gpu-fault.io/dcgm-counters-sha256``, the digest of the CSV the ConfigMap is
  built from: a counters change is a template change and rolls the exporter
  one node at a time.
* The pinned exporter (``dcgm-exporter:4.4.1-4.5.2``, go-dcgm ``a9c939c020ec``)
  parses the CSV with Go's ``encoding/csv`` (``Comment='#'``) and calls
  ``os.Exit(1)`` at startup on a record that is not exactly three fields (a
  comma in the help text), on a field name its go-dcgm does not know, or on an
  unknown Prometheus type: one bad line crash-loops the exporter everywhere.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault import dcgm_fields as DCGM_FIELDS
from gpu_fault.collectors.gpu.dcgm import DCGM_METRICS
from gpu_fault_release import regional_gpu_bootstrap as BOOTSTRAP
from gpu_fault_release import regional_release_rendering as RENDERING
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._release_orchestrator_support import RELEASE_MODULE as MODULE
from tests.regional._release_orchestrator_support import config_file
from tests.regional.test_regional_release_eks_managed import (
    _eks_cluster,
    _hyperpod_cluster,
)

ROOT = Path(__file__).resolve().parents[2]
COUNTERS = ROOT / "deploy" / "dataplane" / "dcgm-counters.csv"
EXPORTER_MANIFEST = "deploy/dataplane/hyperpod-dcgm-exporter.yaml"
LEGACY_DEPLOY = ROOT / "deploy" / "hyperpod" / "deploy.sh"
ANNOTATION = "gpu-fault.io/dcgm-counters-sha256"
IMAGE = "registry.example/dcgm@sha256:" + "b" * 64
#: An eks-managed cluster whose node groups declare a host DCGM.
HOST_ENTRY = {"dcgm_hostengine": "host", "dcgm_hostengine_source": "node-group-label"}
#: Every text the renderer produces: HyperPod (no provider, or named) and the
#: eks variant in both DCGM host engine modes.
VARIANTS = (
    pytest.param({}, id="hyperpod-default"),
    pytest.param({"provider": "hyperpod"}, id="hyperpod"),
    pytest.param({"provider": "eks-managed"}, id="eks-embedded"),
    pytest.param(
        {"provider": "eks-managed", "dcgm_hostengine": "embedded"},
        id="eks-embedded-explicit",
    ),
    pytest.param({"provider": "eks-managed", "dcgm_hostengine": "host"}, id="eks-host"),
)
#: The Prometheus types the collector reads. The exporter also accepts
#: histogram, summary and label; no DCGM field of this file is one of them.
COUNTER_TYPES = frozenset({"gauge", "counter"})
#: Names newer than the pinned exporter's go-dcgm: the fabric health fields
#: (174, 175) and DCGM 4.6's rename of field 171. Each makes
#: ``counters.GetCounterSet`` fail, and the exporter exits at startup.
UNKNOWN_TO_THE_PINNED_EXPORTER = frozenset(
    {
        "DCGM_FI_DEV_FABRIC_HEALTH_MASK",
        "DCGM_FI_DEV_FABRIC_HEALTH_SUMMARY",
        "DCGM_FI_DEV_FABRIC_MANAGER_ERROR",
    }
)
#: Fields 170 and 171, both in the pinned go-dcgm's map (DCGM_FT_INT64).
FABRIC_MANAGER_FIELDS = (
    "DCGM_FI_DEV_FABRIC_MANAGER_STATUS",
    "DCGM_FI_DEV_FABRIC_MANAGER_ERROR_CODE",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _render(**options: Any) -> str:
    return RENDERING.render_dcgm_exporter_manifest(
        namespace="gpu-fault-system", image=IMAGE, **options
    )


def _daemonset(text: str) -> dict[str, Any]:
    (daemonset,) = [
        document
        for document in yaml.safe_load_all(text)
        if isinstance(document, dict) and document.get("kind") == "DaemonSet"
    ]
    return daemonset


def _template_digest(daemonset: dict[str, Any]) -> str | None:
    metadata = daemonset["spec"]["template"].get("metadata") or {}
    return (metadata.get("annotations") or {}).get(ANNOTATION)


def _exporter_records(text: str) -> list[tuple[int, list[str]]]:
    """The CSV's records as the pinned exporter reads them.

    Go's ``encoding/csv`` with ``Comment='#'``: an empty line is skipped, only
    a line whose first character is ``#`` is a comment (an indented ``#`` is a
    record, so is a line of spaces), and the exporter then trims spaces -- and
    only spaces -- off each field (``strings.Trim(r, " ")``). With no ``"`` in
    the file (checked below) Go splits every other line on every comma.
    """

    records = []
    for number, line in enumerate(text.split("\n"), start=1):
        line = line.removesuffix("\r")
        if not line or line.startswith("#"):
            continue
        records.append((number, [field.strip(" ") for field in line.split(",")]))
    return records


@pytest.mark.parametrize("options", VARIANTS)
def test_every_exporter_variant_carries_the_digest_of_the_counters_csv(
    options: dict[str, str],
) -> None:
    text = _render(**options)

    assert "REPLACE_WITH" not in text, (
        f"a placeholder survived rendering ({options}): the release would apply "
        "the checked-in text verbatim"
    )
    digest = _template_digest(_daemonset(text))
    assert digest == _sha256(COUNTERS.read_bytes()), (
        f"the exporter pod template ({options}) must carry the sha256 of "
        "deploy/dataplane/dcgm-counters.csv: the exporter reads it only at start "
        "and its subPath mount is never refreshed, so without the digest a "
        f"counters-only release restarts no Pod (got {digest!r})"
    )


class _KubectlRunner:
    """Records every call; answers the ConfigMap dry run the way kubectl does,
    with the content of the file the ``--from-file`` argument names."""

    dry_run = False

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(self, arguments: Any, **kwargs: Any) -> str:
        arguments = [str(item) for item in arguments]
        self.calls.append((arguments, dict(kwargs)))
        if "create" not in arguments or "configmap" not in arguments:
            return ""
        data = {}
        for item in arguments:
            if item.startswith("--from-file="):
                key, _, path = item.removeprefix("--from-file=").partition("=")
                data[key] = Path(path).read_text(encoding="utf-8")
        name = arguments[arguments.index("configmap") + 1]
        return str(
            yaml.safe_dump(
                {
                    "apiVersion": "v1",
                    "kind": "ConfigMap",
                    "metadata": {"name": name},
                    "data": data,
                }
            )
        )


def test_the_applied_exporter_carries_the_digest_of_the_configmap_applied_with_it(
    tmp_path: Path,
) -> None:
    """The digest has to be the one of the CSV the cluster actually receives.

    Read back from the two documents ``apply_gpu_dcgm_exporter`` hands to
    ``kubectl apply`` on a HyperPod and a plain EKS cluster, not from the
    renderer, so a ConfigMap built from another file than the digested one
    cannot pass.
    """

    clusters = [_hyperpod_cluster(tmp_path), {**_eks_cluster(tmp_path), **HOST_ENTRY}]
    release = MODULE.RegionalRelease(
        MODULE.ReleaseConfig.load(config_file(tmp_path, clusters=clusters)),
        _KubectlRunner(),
    )
    release.dcgm_exporter_image = IMAGE
    runner = release.runner
    assert isinstance(runner, _KubectlRunner), "the release must keep this runner"

    for target in release.config.clusters:
        before = len(runner.calls)
        BOOTSTRAP.apply_gpu_dcgm_exporter(release, target)
        applied = {
            document["kind"]: document
            for arguments, options in runner.calls[before:]
            if arguments[-3:] == ["apply", "-f", "-"]
            for document in yaml.safe_load_all(str(options.get("input_text") or ""))
            if isinstance(document, dict)
        }
        assert set(applied) == {"ConfigMap", "DaemonSet"}, (
            f"{target.cluster_id}: the exporter apply must hand kubectl the counters "
            f"ConfigMap and the DaemonSet, got {sorted(applied)}"
        )
        counters = applied["ConfigMap"]["data"]["gpu-fault-counters.csv"]
        assert _template_digest(applied["DaemonSet"]) == _sha256(
            counters.encode("utf-8")
        ), (
            f"{target.cluster_id}: the pod template digests another CSV than the "
            "ConfigMap applied with it, so a counters change can land without "
            "rolling the exporter (or roll it for nothing)"
        )


def _fake_checkout(tmp_path: Path) -> Path:
    """The two files the renderer reads, under a repository root of their own."""

    for relative in (EXPORTER_MANIFEST, RENDERING.DCGM_COUNTERS_CSV):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    return tmp_path / RENDERING.DCGM_COUNTERS_CSV


@pytest.mark.parametrize("options", VARIANTS)
def test_a_counters_only_change_changes_the_exporter_pod_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: dict[str, str]
) -> None:
    counters = _fake_checkout(tmp_path)
    monkeypatch.setattr(RENDERING, "ROOT", tmp_path)
    before = _render(**options)

    assert _render(**options) == before, (
        "rendering one CSV twice must be byte-identical, or every plan shows a "
        "dcgm change and every release rolls the exporter on every GPU node"
    )
    original = counters.read_bytes()
    # The subtlest counters change there is: help text only, no field added.
    edited = original.replace(b"Last XID error code", b"Last XID error code seen", 1)
    assert edited != original, "the CSV no longer has the XID line this test edits"
    counters.write_bytes(edited)
    after = _render(**options)
    old = _daemonset(before)["spec"]["template"]
    new = _daemonset(after)["spec"]["template"]

    assert new != old, (
        f"a counters-only change left the pod template ({options}) unchanged: the "
        "release re-applies an identical DaemonSet, no Pod restarts, and the "
        "exporter keeps serving the CSV it read at start"
    )
    assert _template_digest(_daemonset(after)) == _sha256(edited), (
        "the rolled template must carry the digest of the CSV it rolls for"
    )
    for template in (old, new):
        template["metadata"]["annotations"].pop(ANNOTATION)
    assert new == old, (
        "a counters change must reach the pod template through its digest alone; "
        "any other difference is the renderer reading the CSV for something else"
    )


def test_a_render_that_keeps_the_counters_placeholder_fails_the_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed like the other two placeholders: a template that keeps the
    placeholder carries one constant digest forever and never rolls again."""

    placeholder = RENDERING.DCGM_COUNTERS_SHA256_PLACEHOLDER
    monkeypatch.setattr(RENDERING, "dcgm_counters_sha256", lambda: placeholder)

    for options in ({}, {"provider": "eks-managed"}):
        with pytest.raises(ReleaseError, match=placeholder):
            _render(**options)


def test_the_legacy_deploy_path_substitutes_the_counters_digest(tmp_path: Path) -> None:
    """``deploy/hyperpod/deploy.sh`` applies this DaemonSet through sed too.

    Left alone it would apply the placeholder as the digest: a valid string,
    so nothing fails, and a constant one, so it never rolls a Pod by itself.
    The digest line and the sed that reads the manifest are run here as the
    script spells them, with ``kubectl`` printing what it would apply.
    """

    script = LEGACY_DEPLOY.read_text(encoding="utf-8")
    placeholder = RENDERING.DCGM_COUNTERS_SHA256_PLACEHOLDER
    manifest = '"${REPO_DIR}/' + EXPORTER_MANIFEST + '"'
    csv = '"${REPO_DIR}/' + RENDERING.DCGM_COUNTERS_CSV + '"'

    assert script.count(manifest) == 1, (
        "the legacy deploy must read the exporter manifest in exactly one place"
    )
    end = script.index(manifest)
    pipeline = script[script.rindex("sed \\\n", 0, end) : end]
    assert f"s#{placeholder}#" in pipeline, (
        "the substitution must belong to the sed that reads the manifest, so the "
        "text handed to `kubectl apply` never carries the placeholder"
    )
    assert f"--from-file=gpu-fault-counters.csv={csv}" in script, (
        "the legacy ConfigMap is no longer built from the CSV the digest is of"
    )
    start = script.rindex("\n", 0, script.rindex("counters_sha256=", 0, end)) + 1
    stop = script.index("kubectl apply -f -", end) + len("kubectl apply -f -")
    excerpt = script[start:stop]
    assert f"sha256sum {csv}" in excerpt, (
        "the legacy digest must be the sha256 of the CSV its ConfigMap is built from"
    )
    harness = tmp_path / "legacy-exporter-apply.sh"
    harness.write_text(
        "set -euo pipefail\n"
        'REPO_DIR="$1"\n'
        f'DEFAULT_DCGM_EXPORTER_IMAGE="{RENDERING.DEFAULT_DCGM_EXPORTER_IMAGE}"\n'
        f'DCGM_EXPORTER_IMAGE="{IMAGE}"\n'
        'instance_type="ml.p5en.48xlarge"\n'
        "kubectl() { cat; }\n" + excerpt + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(harness), str(ROOT)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )

    assert result.returncode == 0, (
        f"the legacy exporter apply fails as the script spells it: {result.stderr}"
    )
    assert "REPLACE_WITH" not in result.stdout, (
        "the legacy deploy hands kubectl a DaemonSet that still carries a placeholder"
    )
    assert _template_digest(_daemonset(result.stdout)) == _sha256(
        COUNTERS.read_bytes()
    ), (
        "the legacy deploy applies a pod template whose digest is not the "
        "counters CSV's, so a counters change does not roll it"
    )


def test_the_counters_csv_parses_the_way_the_pinned_exporter_reads_it() -> None:
    text = COUNTERS.read_text(encoding="utf-8")

    assert '"' not in text, (
        "a double quote after the ', ' separator is a bare-quote parse error in "
        "Go's encoding/csv (the exporter exits at startup); this file quotes "
        "nothing, keep help texts free of quotes and commas"
    )
    records = _exporter_records(text)
    assert records, "the counters CSV has no records: the exporter exports nothing"
    names: list[str] = []
    for number, fields in records:
        where = f"dcgm-counters.csv:{number}"
        assert len(fields) == 3, (
            f"{where}: the pinned exporter requires exactly three fields (DCGM "
            "field, Prometheus type, help) and calls os.Exit(1) otherwise -- a "
            f"comma in the help text crash-loops it on every GPU node: {fields}"
        )
        name, kind, _help = fields
        assert re.fullmatch(r"DCGM_FI_[A-Z0-9_]+", name), (
            f"{where}: {name!r} is no DCGM field name; an unknown name makes the "
            "exporter exit at startup"
        )
        assert kind in COUNTER_TYPES, (
            f"{where}: {kind!r} is not gauge or counter; an unknown type makes the "
            "exporter exit at startup, and the collector reads only these two"
        )
        names.append(name)
    repeated = sorted({name for name in names if names.count(name) > 1})
    assert not repeated, (
        f"fields listed twice: {repeated}; a DCGM field is one metric family and "
        "must be listed once"
    )
    unknown = sorted(UNKNOWN_TO_THE_PINNED_EXPORTER.intersection(names))
    assert not unknown, (
        f"{unknown} are unknown to the pinned exporter's go-dcgm (a9c939c020ec): "
        "counters.GetCounterSet fails and the exporter calls os.Exit(1) on every "
        "GPU node; DCGM_FI_DEV_FABRIC_MANAGER_ERROR_CODE is the name it knows"
    )
    kinds = {fields[0]: fields[1] for _number, fields in records}
    for field in FABRIC_MANAGER_FIELDS:
        assert kinds.get(field) == "gauge", (
            f"{field} must be exported as a gauge: it is a state, not a count, and "
            "the passive fabric detection reads it"
        )


def test_the_fabric_manager_fields_are_never_required_of_an_exporter() -> None:
    """Exported, never required.

    Field 171 has no sample until fabric registration completes (the exporter
    drops blank values), and a customer ``existing`` exporter may serve
    neither field. Neither is a reason for a collection error, nor for the
    legacy deploy's metrics probe to call an exporter incompatible.
    """

    required = {
        name for group in DCGM_FIELDS.DCGM_EXPORTER_FIELD_GROUPS for name in group
    }
    expression = DCGM_FIELDS.dcgm_exporter_required_expression()
    for field in FABRIC_MANAGER_FIELDS:
        assert field not in required and field not in expression, (
            f"{field} became a required exporter field: an exporter without it "
            "fails the metrics probe and every collection reports it missing"
        )
    served = {
        DCGM_METRICS[name][0]
        for _number, (name, *_rest) in _exporter_records(
            COUNTERS.read_text(encoding="utf-8")
        )
        if name in DCGM_METRICS and name not in FABRIC_MANAGER_FIELDS
    }
    assert DCGM_FIELDS.missing_dcgm_metric_groups(served) == [], (
        "an exporter that serves every counters field but the Fabric Manager "
        "ones must not raise 'DCGM exporter is missing required fields': "
        f"{DCGM_FIELDS.missing_dcgm_metric_groups(served)}"
    )
