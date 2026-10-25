"""The artifact ConfigMap layouts: one object while it fits, a chunked set past it."""

from __future__ import annotations

import base64
import hashlib
import json
import lzma
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import artifact_configmaps as module
from gpu_fault.admin.artifact_configmaps import (
    ARTIFACT_PART_BYTES,
    CONFIG_MAP_PAYLOAD_LIMIT,
    MANIFEST_KEY,
    ArtifactLayoutError,
    ArtifactTooLarge,
    artifact_manifest,
    artifact_part_names,
    artifact_set_name,
    part_config_map_name,
    plan_artifact_config_maps,
    read_artifact_digest,
    stored_artifact_key,
)


def _artifact(tmp_path: Path, name: str, payload: bytes) -> tuple[Path, str]:
    path = tmp_path / name
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _documents(plan: module.ArtifactPlan) -> dict[str, dict[str, Any]]:
    """The ConfigMaps kubectl would hold for ``plan``: text in data, bytes in binaryData."""

    documents: dict[str, dict[str, Any]] = {}
    for item in plan.objects:
        if item.binary:
            documents[item.name] = {
                "binaryData": {item.key: base64.b64encode(item.payload).decode()}
            }
        else:
            documents[item.name] = {"data": {item.key: item.payload.decode()}}
    return documents


@pytest.fixture
def small_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 4 KiB ceiling and 1 KiB parts: the layouts without megabyte fixtures."""

    monkeypatch.setattr(module, "CONFIG_MAP_PAYLOAD_LIMIT", 4096)
    monkeypatch.setattr(module, "ARTIFACT_PART_BYTES", 1024)


def test_a_fitting_artifact_is_one_object_and_a_raw_one_is_read_in_place(
    tmp_path: Path,
) -> None:
    bundle, sha = _artifact(tmp_path, "bundle.tar.gz", b"\x1f\x8b" + b"bundle" * 100)
    plan = plan_artifact_config_maps(
        bundle, name="cm", key=bundle.name, compress=False, splittable=False
    )
    assert not plan.chunked and plan.layout == "none"
    assert plan.primary.key == "bundle.tar.gz" and plan.primary.source == bundle
    assert module.write_artifact_object(plan.primary, tmp_path / "scratch") == bundle, (
        "a raw single object is handed to kubectl as the artifact file itself"
    )
    assert plan.sha256 == sha and plan.payload_size == plan.size

    wheel, wheel_sha = _artifact(tmp_path, "a.whl", b"PK\x03\x04" + b"wheel" * 1000)
    plan = plan_artifact_config_maps(
        wheel, name="cm", key=wheel.name, compress=True, splittable=True
    )
    assert plan.layout == "xz" and plan.primary.key == "a.whl.xz"
    assert (
        plan.primary.source is None
        and lzma.decompress(plan.primary.payload) == wheel.read_bytes()
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    written = module.write_artifact_object(plan.primary, scratch)
    assert (
        written == scratch / "a.whl.xz" and written.read_bytes() == plan.primary.payload
    )
    assert plan.sha256 == wheel_sha


def test_an_artifact_over_the_ceiling_becomes_parts_and_a_manifest(
    tmp_path: Path, small_limits: None
) -> None:
    wheel, sha = _artifact(tmp_path, "a.whl", os.urandom(4096 + 1500))
    plan = plan_artifact_config_maps(
        wheel,
        name="gpu-fault-control-plane-wheel-0100-abcdefabcdef",
        key=wheel.name,
        compress=True,
        splittable=True,
    )
    assert plan.chunked, "a stream over the ceiling must be split"
    assert len(plan.parts) == -(-plan.payload_size // 1024), "parts cover the stream"
    assert [part.name for part in plan.parts] == [
        part_config_map_name(plan.name, index) for index in range(len(plan.parts))
    ]
    assert all(part.size <= 1024 for part in plan.parts), (
        "a part never exceeds its slice"
    )
    assert [part.key for part in plan.parts] == [
        f"a.whl.xz.{index:02d}" for index in range(len(plan.parts))
    ]
    assert plan.primary.key == MANIFEST_KEY and plan.primary.binary is False
    manifest = json.loads(plan.primary.payload)
    assert manifest["schema_version"] == 1
    assert manifest["artifact"] == "a.whl" and manifest["sha256"] == sha
    assert manifest["encoding"] == "xz"
    assert manifest["payload_size"] == sum(part.size for part in plan.parts)
    assert [part["sha256"] for part in manifest["parts"]] == [
        part.sha256 for part in plan.parts
    ]
    assert plan.layout == f"chunked:{len(plan.parts)}"
    assert artifact_part_names(
        {"data": {MANIFEST_KEY: plan.primary.payload.decode()}}
    ) == tuple(part.name for part in plan.parts)


def test_an_unsplittable_or_absurd_artifact_is_refused_with_its_numbers(
    tmp_path: Path, small_limits: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _sha = _artifact(tmp_path, "bundle.tar.gz", os.urandom(5000))
    with pytest.raises(ArtifactTooLarge) as refused:
        plan_artifact_config_maps(
            bundle,
            name="cm",
            key=bundle.name,
            compress=False,
            splittable=False,
            label="node installer bundle",
        )
    message = str(refused.value)
    assert "node installer bundle bundle.tar.gz is 5,000 bytes" in message
    assert "at most 4,096 bytes" in message and "cannot be split" in message

    monkeypatch.setattr(module, "MAX_PARTS", 2)
    with pytest.raises(ArtifactTooLarge, match="at most 2"):
        plan_artifact_config_maps(
            bundle, name="cm", key=bundle.name, compress=False, splittable=True
        )
    with pytest.raises(ArtifactLayoutError, match="name is invalid"):
        plan_artifact_config_maps(
            bundle, name="Not_A_Name", key=bundle.name, compress=False, splittable=True
        )


def test_set_names_and_stored_keys_cover_every_layout() -> None:
    assert artifact_set_name(
        "gpu-fault-control-plane-wheel-0100-abcdefabcdef-part-3"
    ) == ("gpu-fault-control-plane-wheel-0100-abcdefabcdef")
    assert artifact_set_name("gpu-fault-control-plane-wheel-0100-abcdefabcdef") == (
        "gpu-fault-control-plane-wheel-0100-abcdefabcdef"
    )
    assert stored_artifact_key({"binaryData": {"a.whl.xz": "AA=="}}) == "a.whl.xz"
    assert stored_artifact_key({"binaryData": {"a.whl": "AA=="}}) == "a.whl"
    assert stored_artifact_key({"binaryData": {"a": "AA==", "b": "AA=="}}) is None
    assert stored_artifact_key({}) is None
    manifest = {
        "schema_version": 1,
        "artifact": "a.whl",
        "sha256": "0" * 64,
        "size": 1,
        "encoding": "xz",
        "payload_sha256": "1" * 64,
        "payload_size": 1,
        "parts": [
            {
                "config_map": "cm-part-0",
                "key": "a.whl.xz.00",
                "size": 1,
                "sha256": "2" * 64,
            }
        ],
    }
    assert (
        stored_artifact_key({"data": {MANIFEST_KEY: json.dumps(manifest)}}) == "a.whl"
    )
    assert artifact_manifest({"data": {"other": "x"}}) is None
    with pytest.raises(ArtifactLayoutError, match="binaryData is invalid"):
        stored_artifact_key({"binaryData": ["list"]})


@pytest.mark.parametrize(
    "mutation,problem",
    [
        (lambda m: m.update(schema_version=2), "schema_version"),
        (lambda m: m.update(artifact=""), "names no artifact"),
        (lambda m: m.update(sha256="xyz"), "sha256 is not a sha256"),
        (lambda m: m.update(size=-1), "size is not a size"),
        (lambda m: m.update(encoding="gzip"), "encoding is unknown"),
        (lambda m: m.update(parts=[]), "no usable parts"),
        (lambda m: m["parts"][0].pop("sha256"), "malformed part"),
        (lambda m: m["parts"][0].update(config_map="Bad Name"), "malformed part"),
    ],
)
def test_a_malformed_manifest_is_refused_not_guessed(
    mutation: Any, problem: str
) -> None:
    manifest = {
        "schema_version": 1,
        "artifact": "a.whl",
        "sha256": "0" * 64,
        "size": 1,
        "encoding": "xz",
        "payload_sha256": "1" * 64,
        "payload_size": 1,
        "parts": [
            {
                "config_map": "cm-part-0",
                "key": "a.whl.xz.00",
                "size": 1,
                "sha256": "2" * 64,
            }
        ],
    }
    mutation(manifest)
    with pytest.raises(ArtifactLayoutError, match=problem):
        artifact_manifest({"data": {MANIFEST_KEY: json.dumps(manifest)}})
    with pytest.raises(ArtifactLayoutError, match="not JSON"):
        artifact_manifest({"data": {MANIFEST_KEY: "{"}})


def test_reading_a_chunked_set_re_hashes_every_part_the_stream_and_the_artifact(
    tmp_path: Path, small_limits: None
) -> None:
    wheel, sha = _artifact(tmp_path, "a.whl", os.urandom(4096 + 700))
    plan = plan_artifact_config_maps(
        wheel, name="cm", key=wheel.name, compress=True, splittable=True
    )
    documents = _documents(plan)
    fetched: list[str] = []

    def fetch(name: str) -> dict[str, Any]:
        fetched.append(name)
        return documents[name]

    assert read_artifact_digest(documents["cm"], "a.whl", fetch, name="cm") == (
        "a.whl",
        sha,
    )
    assert fetched == [part.name for part in plan.parts], (
        "parts are read in manifest order"
    )

    # A part a kubelet-free kubectl filed under data (a UTF-8 slice) reads the same.
    text_plan = plan_artifact_config_maps(
        _artifact(tmp_path, "t.whl", ("text " * 2000).encode())[0],
        name="cm",
        key="t.whl",
        compress=False,
        splittable=True,
    )
    text_documents = {
        item.name: {"data": {item.key: item.payload.decode()}}
        for item in text_plan.objects
    }
    assert read_artifact_digest(
        text_documents["cm"], "t.whl", lambda name: text_documents[name], name="cm"
    ) == ("t.whl", text_plan.sha256)

    corrupted = json.loads(json.dumps(documents))
    first = plan.parts[0]
    corrupted[first.name]["binaryData"][first.key] = base64.b64encode(
        b"\x00" * first.size
    ).decode()
    with pytest.raises(ArtifactLayoutError, match="does not match its manifest"):
        read_artifact_digest(
            corrupted["cm"], "a.whl", lambda name: corrupted[name], name="cm"
        )

    claimed = json.loads(json.dumps(documents))
    manifest = json.loads(claimed["cm"]["data"][MANIFEST_KEY])
    manifest["sha256"] = "f" * 64
    claimed["cm"]["data"][MANIFEST_KEY] = json.dumps(manifest)
    with pytest.raises(ArtifactLayoutError, match="disagrees with its parts"):
        read_artifact_digest(
            claimed["cm"], "a.whl", lambda name: claimed[name], name="cm"
        )

    missing = json.loads(json.dumps(documents))
    missing[first.name] = {"binaryData": {}}
    with pytest.raises(ArtifactLayoutError, match="is missing"):
        read_artifact_digest(
            missing["cm"], "a.whl", lambda name: missing[name], name="cm"
        )

    # The legacy layouts never call fetch.
    legacy = {
        "binaryData": {
            "a.whl.xz": base64.b64encode(lzma.compress(wheel.read_bytes())).decode()
        }
    }
    assert read_artifact_digest(legacy, "a.whl", fetch) == ("a.whl.xz", sha)
    assert read_artifact_digest({"binaryData": {}}, "a.whl", fetch) is None


FAKE_KUBECTL = """#!PYTHON_EXECUTABLE
import base64, json, os, sys
from pathlib import Path
LIMIT = 1048576
state = Path(os.environ["FAKE_STATE"])
args = sys.argv[1:]
with (state / "calls.jsonl").open("a") as handle:
    handle.write(json.dumps(args) + "\\n")
name = args[args.index("configmap") + 1]
target = state / (name + ".json")
if "get" in args:
    if not target.exists():
        print('Error from server (NotFound): configmaps "%s" not found' % name, file=sys.stderr)
        raise SystemExit(1)
    print(target.read_text())
elif "create" in args:
    if target.exists():
        print('Error from server (AlreadyExists): configmaps "%s" already exists' % name, file=sys.stderr)
        raise SystemExit(1)
    field = next(item for item in args if item.startswith("--from-file="))
    key, path = field[len("--from-file="):].split("=", 1)
    payload = Path(path).read_bytes()
    if len(payload) > LIMIT:
        print('error: failed to create configmap: ConfigMap "%s" is invalid: []: Too long: may not be more than 1048576 bytes' % name, file=sys.stderr)
        raise SystemExit(1)
    document = {"kind": "ConfigMap", "metadata": {"name": name, "namespace": args[args.index("-n") + 1]}}
    try:
        document["data"] = {key: payload.decode("utf-8")}
    except UnicodeDecodeError:
        document["binaryData"] = {key: base64.b64encode(payload).decode()}
    target.write_text(json.dumps(document))
else:
    raise SystemExit("unexpected kubectl use: " + " ".join(args))
"""


@pytest.fixture
def fake_kubectl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state = tmp_path / "api"
    state.mkdir()
    binary = tmp_path / "bin"
    binary.mkdir()
    kubectl = binary / "kubectl"
    kubectl.write_text(
        FAKE_KUBECTL.replace("PYTHON_EXECUTABLE", sys.executable), encoding="utf-8"
    )
    kubectl.chmod(kubectl.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_STATE", str(state))
    return state


def _calls(state: Path) -> list[list[str]]:
    return [
        json.loads(line) for line in (state / "calls.jsonl").read_text().splitlines()
    ]


@pytest.mark.allows_cluster_binaries("kubectl")
def test_cli_uploads_a_wheel_over_the_ceiling_and_reads_its_digest_back(
    tmp_path: Path, fake_kubectl: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel, sha = _artifact(
        tmp_path,
        "gpu_fault_control_plane-0.10.0-py3-none-any.whl",
        os.urandom(CONFIG_MAP_PAYLOAD_LIMIT + 2048),
    )
    name = "gpu-fault-control-plane-wheel-0100-" + sha[:12]
    common = ["--kubeconfig", "/dev/null", "--namespace", "gpu-fault-system"]

    assert (
        module.main(
            [
                *common,
                "upload",
                "--name",
                name,
                "--artifact",
                str(wheel),
                "--compress",
                "--splittable",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["sha256"] == sha and out["layout"] == "chunked:2"
    assert out["objects"] == [
        part_config_map_name(name, 0),
        part_config_map_name(name, 1),
        name,
    ]
    created = [
        call[call.index("configmap") + 1]
        for call in _calls(fake_kubectl)
        if "create" in call
    ]
    assert created == out["objects"], "parts are created before the manifest"
    assert all(
        call[:2] == ["--kubeconfig", "/dev/null"] for call in _calls(fake_kubectl)
    ), "every kubectl call carries the operator's kubeconfig"

    assert module.main([*common, "digest", "--name", name, "--key", wheel.name]) == 0
    assert capsys.readouterr().out.strip() == sha

    # The same upload again creates nothing and still proves the digest.
    before = len(_calls(fake_kubectl))
    assert (
        module.main(
            [
                *common,
                "upload",
                "--name",
                name,
                "--artifact",
                str(wheel),
                "--compress",
                "--splittable",
            ]
        )
        == 0
    )
    assert not any("create" in call for call in _calls(fake_kubectl)[before:]), (
        "an existing set is verified, never recreated"
    )
    assert json.loads(capsys.readouterr().out)["sha256"] == sha


@pytest.mark.allows_cluster_binaries("kubectl")
def test_cli_refuses_an_unsplittable_artifact_and_reports_missing_sets(
    tmp_path: Path, fake_kubectl: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle, sha = _artifact(
        tmp_path,
        "gpu-fault-node-installer-0.10.0.tar.gz",
        os.urandom(CONFIG_MAP_PAYLOAD_LIMIT + 1),
    )
    common = ["--namespace", "gpu-fault-system"]
    assert (
        module.main(
            [
                *common,
                "upload",
                "--name",
                "gpu-fault-node-installer-0100-" + sha[:12],
                "--artifact",
                str(bundle),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "cannot be split" in captured.err and captured.out == ""
    assert not (fake_kubectl / "calls.jsonl").exists(), (
        "the refusal precedes every kubectl call"
    )

    assert (
        module.main(
            [
                *common,
                "digest",
                "--name",
                "gpu-fault-control-plane-wheel-0100-000000000000",
            ]
        )
        == 1
    )
    assert "does not exist" in capsys.readouterr().err

    # A small bundle takes the raw single-object path kubectl reads in place.
    small, small_sha = _artifact(
        tmp_path, "small-bundle.tar.gz", b"\x1f\x8b" + b"bundle" * 64
    )
    assert (
        module.main(
            [
                *common,
                "upload",
                "--name",
                "gpu-fault-node-installer-0100-" + small_sha[:12],
                "--artifact",
                str(small),
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["layout"] == "none" and out["sha256"] == small_sha
    create = next(call for call in _calls(fake_kubectl) if "create" in call)
    assert create[-1] == f"--from-file={small.name}={small}"


def test_module_runs_as_a_script_for_the_deploy_tools(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "gpu_fault.admin.artifact_configmaps", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "upload" in completed.stdout and "digest" in completed.stdout
    assert textwrap.dedent(completed.stdout).startswith("usage:"), completed.stdout
    assert ARTIFACT_PART_BYTES < CONFIG_MAP_PAYLOAD_LIMIT, (
        "a part must always fit one object"
    )
