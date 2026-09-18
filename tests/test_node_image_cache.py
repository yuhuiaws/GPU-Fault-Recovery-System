from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts import node_wheelhouse, release_image, release_image_set
from scripts.node_wheelhouse import (
    NodeImageCache,
    build_wheelhouse,
    sha256,
    validate_wheelhouse_inventory,
)
from tests.test_release_image import inspected_image
from tests.test_release_image_set import write_node_dependency_locks

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = "a" * 64
IMAGE = {"digest": "sha256:" + "b" * 64, "image_inputs": {}}


def test_missing_cache_is_a_cold_miss_without_creating_files(tmp_path):
    directory = tmp_path / "missing/cache"
    assert NodeImageCache(directory).load(IDENTITY) is None
    assert not directory.parent.exists(), "read-only cache miss created directories"


@pytest.fixture
def cache(tmp_path):
    result = NodeImageCache(tmp_path / "cache")
    result.store(IDENTITY, IMAGE)
    return result


def test_receipt_round_trip_is_private_and_contains_no_wheel_payload(cache):
    assert cache.load(IDENTITY) == IMAGE
    assert cache.directory.stat().st_mode & 0o777 == 0o700
    assert {path.name for path in cache.directory.iterdir()} == {
        ".receipt-key",
        ".lock",
        f"{IDENTITY}.json",
    }
    for path in cache.directory.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
    assert (cache.directory / ".receipt-key").stat().st_size == 32


@pytest.mark.parametrize(
    "damage",
    [
        "payload",
        "unkeyed-hash",
        "unsigned-inventory",
        "malformed",
        "oversized",
        "missing-key",
        "short-key",
    ],
)
def test_forged_or_damaged_receipts_fail_closed(cache, damage):
    receipt = cache.directory / f"{IDENTITY}.json"
    envelope = json.loads(receipt.read_bytes())
    if damage in {"payload", "unkeyed-hash"}:
        envelope["receipt"]["image"]["digest"] = "sha256:" + "c" * 64
        if damage == "unkeyed-hash":
            envelope["hmac_sha256"] = hashlib.sha256(
                json.dumps(
                    envelope["receipt"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        receipt.write_text(json.dumps(envelope), encoding="utf-8")
    elif damage == "unsigned-inventory":
        receipt.write_text(json.dumps({"files": {}, "sha256": IDENTITY}))
    elif damage == "malformed":
        receipt.write_bytes(b"{")
    elif damage == "oversized":
        receipt.write_bytes(b" " * 131073)
    elif damage == "missing-key":
        (cache.directory / ".receipt-key").unlink()
    else:
        (cache.directory / ".receipt-key").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="cache"):
        cache.load(IDENTITY)
    with pytest.raises(ValueError, match="cache"):
        cache.store(IDENTITY, IMAGE)


@pytest.mark.parametrize("damage", ["missing", "short"])
def test_a_new_input_cache_miss_does_not_hide_a_broken_authentication_key(
    cache, damage
):
    key = cache.directory / ".receipt-key"
    if damage == "missing":
        key.unlink()
    else:
        key.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="authentication key"):
        cache.load("e" * 64)


@pytest.mark.parametrize("name", [".receipt-key", ".lock", f"{IDENTITY}.json", "."])
def test_nonprivate_cache_permissions_are_rejected(cache, name):
    path = cache.directory / name
    path.chmod(0o755 if name == "." else 0o644)
    with pytest.raises(ValueError, match="private"):
        cache.load(IDENTITY)


@pytest.mark.parametrize("name", [".receipt-key", ".lock", f"{IDENTITY}.json"])
def test_cache_symlinks_are_rejected(cache, tmp_path, name):
    path = cache.directory / name
    target = tmp_path / "target"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        cache.load(IDENTITY)


@pytest.mark.parametrize("kind", ["hardlink", "fifo", "directory"])
def test_cache_nonregular_or_shared_receipts_are_rejected(cache, tmp_path, kind):
    path = cache.directory / f"{IDENTITY}.json"
    if kind == "hardlink":
        os.link(path, tmp_path / "linked-receipt")
    else:
        path.unlink()
        if kind == "fifo":
            os.mkfifo(path, mode=0o600)
        else:
            path.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="regular"):
        cache.load(IDENTITY)


def test_cache_directory_symlink_is_rejected(cache, tmp_path):
    link = tmp_path / "linked-cache"
    link.symlink_to(cache.directory, target_is_directory=True)
    with pytest.raises(ValueError, match="trusted"):
        NodeImageCache(link).load(IDENTITY)


def test_cache_writable_ancestor_is_rejected(tmp_path):
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o777)
    parent.chmod(0o777)
    with pytest.raises(ValueError, match="trusted"):
        NodeImageCache(parent / "cache").store(IDENTITY, IMAGE)


@pytest.mark.parametrize("identity", ["", "../escape", "a" * 63, "z" * 64])
def test_cache_identity_cannot_escape_its_directory(cache, identity):
    with pytest.raises(ValueError, match="identity"):
        cache.load(identity)
    with pytest.raises(ValueError, match="identity"):
        cache.store(identity, IMAGE)


def test_receipts_cannot_be_replayed_under_another_identity_or_key(cache, tmp_path):
    other_identity = "d" * 64
    source = cache.directory / f"{IDENTITY}.json"
    shutil.copy2(source, cache.directory / f"{other_identity}.json")
    with pytest.raises(ValueError, match="authentication"):
        cache.load(other_identity)
    other = NodeImageCache(tmp_path / "other-cache")
    other.store(other_identity, IMAGE)
    shutil.copy2(source, other.directory / source.name)
    with pytest.raises(ValueError, match="authentication"):
        other.load(IDENTITY)


def test_parallel_cache_writers_share_one_key_and_publish_complete_receipts(tmp_path):
    cache = NodeImageCache(tmp_path / "cache")
    identities = [hashlib.sha256(str(index).encode()).hexdigest() for index in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda identity: cache.store(identity, IMAGE), identities))
    assert all(cache.load(identity) == IMAGE for identity in identities), (
        "parallel receipt writers lost or mixed identities"
    )


class NodeRegistry:
    def __init__(self, wheels):
        self.commands = []
        self.images = {}
        self.wheels = wheels
        self.on_download = lambda _command: None
        self.on_inspect = lambda _command: None

    def run(self, command, **_kwargs):
        self.commands.append(command)
        if "download" in command:
            destination = Path(command[command.index("--dest") + 1])
            lock = Path(command[-1])
            filename, payload = self.wheels[lock.name]
            (destination / filename).write_bytes(payload)
            self.on_download(command)
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:3] == ["docker", "buildx", "imagetools"]:
            self.on_inspect(command)
            content = self.images.get(command[-1])
            return subprocess.CompletedProcess(
                command,
                0 if content else 1,
                content or "",
                "" if content else "manifest unknown",
            )
        assert command[:3] == ["docker", "buildx", "build"]
        context = Path(command[-1])
        assert (context / "wheelhouse/inventory.json").is_file(), (
            "image build has no wheelhouse inventory"
        )
        labels = dict(
            item.split("=", 1)
            for index, item in enumerate(command)
            if index and command[index - 1] == "--label"
        )
        digest = (
            "sha256:"
            + hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest()
        )
        tag = command[command.index("--tag") + 1]
        content = inspected_image(labels, digest=digest)
        self.images[tag] = content
        self.images[tag.rsplit(":", 1)[0] + "@" + digest] = content
        Path(command[command.index("--metadata-file") + 1]).write_text(
            json.dumps({"containerimage.digest": digest}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, "", "")


@pytest.fixture
def node_build(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    for relative in (
        "deploy/image/Dockerfile.node-dependencies",
        "config/release-images.lock.json",
        "requirements/node-runtime.lock",
        "requirements/node-tools.lock",
        "scripts/node_wheelhouse.py",
        "scripts/release_image.py",
        "scripts/release_image_set.py",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (root / "source-marker").write_text("source-before", encoding="utf-8")

    def identity(path):
        return {"sha256": sha256(path / "source-marker")}

    monkeypatch.setattr(release_image_set, "build_release_identity", identity)
    monkeypatch.setattr(release_image, "build_release_identity", identity)
    monkeypatch.setenv("HOME", str(tmp_path))
    registry = NodeRegistry(write_node_dependency_locks(root))

    def build(**kwargs):
        return release_image_set.build_node_dependency_image(
            root,
            repository=kwargs.pop("repository", "registry.example/runtime"),
            push=kwargs.pop("push", True),
            runner=registry.run,
            **kwargs,
        )

    return root, registry, build


def test_early_reuse_skips_download_and_build_but_reproves_the_pinned_image(node_build):
    root, registry, build = node_build
    first = build()
    assert sum("download" in command for command in registry.commands) == 2
    assert (
        sum(
            command[:3] == ["docker", "buildx", "build"]
            for command in registry.commands
        )
        == 1
    )
    registry.images[first["tag"]] = "{}"
    registry.commands.clear()
    (root / "source-marker").write_text("source-after", encoding="utf-8")
    second = build(cache_from=("type=local,src=local-cache",))
    assert len(registry.commands) == 1
    assert registry.commands[0][:3] == ["docker", "buildx", "imagetools"]
    assert registry.commands[0][-1] == first["reference"]
    assert second["reference"] == first["reference"]
    assert second["image_inputs"] == first["image_inputs"]
    assert second["source_identity_sha256"] == sha256(root / "source-marker")
    assert second["source_identity_sha256"] != first["source_identity_sha256"]
    assert second["build_cache"] == {"from_configured": True, "to_configured": False}
    assert second["registry_reused"] is True


@pytest.mark.parametrize("relative", [".", "artifacts", "dist"])
def test_cache_inside_checkout_is_rejected_before_creating_any_files(
    node_build, monkeypatch, relative
):
    root, registry, build = node_build
    home = root / relative
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(release_image.ReleaseImageError, match="outside the source"):
        build()
    assert not registry.commands, "source-local cache reached registry or build"
    assert not (home / ".cache").exists(), "refused cache location created files"
    assert not list(root.rglob(".receipt-key")), (
        "cache authentication key leaked into source"
    )


def test_cache_symlink_into_checkout_is_rejected_before_publication(
    node_build, tmp_path, monkeypatch
):
    root, registry, build = node_build
    home = tmp_path / "outside-home"
    home.mkdir()
    (home / ".cache").symlink_to(root / "artifacts", target_is_directory=True)
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(release_image.ReleaseImageError, match="outside the source"):
        build()
    assert not registry.commands, "invalid cache location reached registry or build"
    assert not (root / "artifacts").exists(), (
        "refused cache location wrote release artifacts"
    )


@pytest.mark.parametrize("warm", [False, True])
def test_published_descriptors_do_not_contain_private_cache_records(node_build, warm):
    _root, _registry, build = node_build
    if warm:
        build()
    descriptor = build()
    encoded = json.dumps(descriptor)
    assert str(NodeImageCache().directory) not in encoded
    assert ".receipt-key" not in encoded
    assert "hmac_sha256" not in encoded
    assert "receipt" not in descriptor
    assert "inventory" not in descriptor


@pytest.mark.parametrize(
    "relative",
    [
        "requirements/node-runtime.lock",
        "requirements/node-tools.lock",
        "deploy/image/Dockerfile.node-dependencies",
        "scripts/node_wheelhouse.py",
        "scripts/release_image.py",
        "scripts/release_image_set.py",
        "config/release-images.lock.json",
    ],
)
def test_changed_build_inputs_never_use_the_early_receipt(node_build, relative):
    root, registry, build = node_build
    first = build()
    path = root / relative
    if relative.endswith(".json"):
        lock = json.loads(path.read_bytes())
        lock["images"]["node_installer"]["reference"] = (
            "registry.example/base@sha256:" + "c" * 64
        )
        path.write_text(json.dumps(lock), encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes() + b"\n# changed input\n")
    registry.commands.clear()
    second = build()
    assert "download" in registry.commands[0]
    assert sum("download" in command for command in registry.commands) == 2
    if relative not in {"scripts/release_image.py", "scripts/release_image_set.py"}:
        assert second["image_input_sha256"] != first["image_input_sha256"]


def test_repository_change_cannot_reuse_another_repository_receipt(node_build):
    _root, registry, build = node_build
    build()
    registry.commands.clear()
    descriptor = build(repository="registry.example/other")
    assert "download" in registry.commands[0]
    assert descriptor["reference"].startswith("registry.example/other@sha256:"), (
        "new repository reused another repository's image"
    )


def test_missing_pinned_image_falls_back_to_the_unchanged_cold_path(node_build):
    _root, registry, build = node_build
    first = build()
    registry.images.clear()
    registry.commands.clear()
    second = build()
    assert registry.commands[0][-1] == first["reference"]
    assert sum("download" in command for command in registry.commands) == 2
    assert second["registry_reused"] is False
    assert second["reference"] == first["reference"]


@pytest.mark.parametrize("damage", ["labels", "platform", "digest", "json"])
def test_early_registry_proof_rejects_drift_without_download(node_build, damage):
    _root, registry, build = node_build
    first = build()
    value = json.loads(registry.images[first["reference"]])
    if damage == "labels":
        value["image"]["linux/amd64"]["config"]["Labels"][
            "gpu-fault.node-wheelhouse.sha256"
        ] = "0" * 64
    elif damage == "platform":
        value["image"]["linux/amd64"]["architecture"] = "arm64"
    elif damage == "digest":
        value["manifest"]["digest"] = "sha256:" + "0" * 64
    registry.images[first["reference"]] = "{" if damage == "json" else json.dumps(value)
    registry.commands.clear()
    with pytest.raises(release_image.ReleaseImageError):
        build()
    assert len(registry.commands) == 1


def test_early_registry_errors_do_not_fall_back_to_download(node_build):
    _root, registry, build = node_build
    build()
    registry.commands.clear()

    def denied(_command):
        raise release_image.ReleaseImageError("registry access denied")

    registry.on_inspect = denied
    with pytest.raises(release_image.ReleaseImageError, match="denied"):
        build()
    assert len(registry.commands) == 1


@pytest.mark.parametrize(
    "error",
    [
        'ERROR: error getting credentials - exec: "docker-credential-ecr-login": '
        "executable file not found in $PATH",
        "ERROR: unauthorized: manifest unknown",
        "ERROR: failed to parse config: file not found",
    ],
)
def test_warm_receipt_registry_tool_errors_do_not_trigger_cold_work(
    node_build, monkeypatch, error
):
    _root, registry, build = node_build
    first = build()
    registry.commands.clear()

    def failed(command, **_kwargs):
        registry.commands.append(command)
        assert command[-1] == first["reference"], (
            "a warm registry error triggered downloads or publication"
        )
        return subprocess.CompletedProcess(command, 1, "", error)

    monkeypatch.setattr(registry, "run", failed)
    with pytest.raises(release_image.ReleaseImageError, match="inspection failed"):
        build()
    assert len(registry.commands) == 1


@pytest.mark.parametrize(
    "relative",
    [
        "source-marker",
        "requirements/node-runtime.lock",
        "requirements/node-tools.lock",
        "scripts/node_wheelhouse.py",
        "scripts/release_image.py",
        "deploy/image/Dockerfile.node-dependencies",
    ],
)
@pytest.mark.parametrize("warm", [False, True])
def test_source_mutation_is_rejected_and_never_cached(node_build, relative, warm):
    root, registry, build = node_build
    if warm:
        build()
    registry.commands.clear()
    path = root / relative

    def mutate(_command):
        path.write_bytes(path.read_bytes() + b"\n# mutated\n")

    if warm:
        registry.on_inspect = mutate
    else:
        registry.on_download = mutate
    with pytest.raises((ValueError, release_image.ReleaseImageError), match="changed"):
        build()
    assert not any(
        command[:3] == ["docker", "buildx", "build"] for command in registry.commands
    ), "changed source was built instead of rejected"
    if not warm:
        assert not NodeImageCache().directory.exists(), (
            "invalid build published a cache receipt"
        )


def test_post_publish_source_mutation_does_not_cache_a_build(node_build):
    root, registry, build = node_build

    def mutate(command):
        if command[-1] in registry.images:
            (root / "source-marker").write_text("after-push", encoding="utf-8")

    registry.on_inspect = mutate
    with pytest.raises(release_image.ReleaseImageError, match="changed"):
        build()
    assert not NodeImageCache().directory.exists(), (
        "changed build published a cache receipt"
    )


def test_unsigned_dist_inventory_is_not_an_early_reuse_source(node_build):
    root, registry, build = node_build
    dist = root / "dist"
    dist.mkdir()
    (dist / "release-runtime-image.json").write_text(json.dumps(IMAGE))
    (dist / "inventory.json").write_text(json.dumps({"files": {}}))
    build()
    assert "download" in registry.commands[0]


def test_forged_receipt_stops_before_registry_or_download(node_build):
    _root, registry, build = node_build
    build()
    receipt = next(NodeImageCache().directory.glob("*.json"))
    value = json.loads(receipt.read_bytes())
    value["receipt"]["image"]["image_inputs"]["wheelhouse_sha256"] = "0" * 64
    receipt.write_text(json.dumps(value), encoding="utf-8")
    registry.commands.clear()
    with pytest.raises(ValueError, match="authentication"):
        build()
    assert not registry.commands, "forged receipt reached registry or build"


@pytest.mark.parametrize("damage", ["inputs", "mutable", "inventory"])
def test_authenticated_but_inconsistent_receipts_are_not_trusted(node_build, damage):
    _root, registry, build = node_build
    build()
    cache = NodeImageCache()
    identity = next(cache.directory.glob("*.json")).stem
    image = cache.load(identity)
    assert image is not None
    if damage == "inputs":
        image["image_inputs"]["platform"] = "linux/arm64"
    elif damage == "mutable":
        image["digest"] = "latest"
    else:
        image["image_inputs"]["wheelhouse_sha256"] = "invalid"
    cache.store(identity, image)
    registry.commands.clear()
    with pytest.raises(release_image.ReleaseImageError, match="identity"):
        build()
    assert not registry.commands, "inconsistent receipt reached registry or build"


@pytest.mark.parametrize("options", [{"push": False}, {"reuse_registry_image": False}])
def test_local_or_forced_builds_do_not_take_the_early_reuse_path(node_build, options):
    _root, registry, build = node_build
    build()
    registry.commands.clear()
    descriptor = build(**options)
    assert sum("download" in command for command in registry.commands) == 2
    assert any(
        command[:3] == ["docker", "buildx", "build"] for command in registry.commands
    ), "forced or local build incorrectly used the receipt shortcut"
    assert descriptor["registry_reused"] is False


def test_warm_cache_does_not_bypass_supported_build_host_check(node_build, monkeypatch):
    _root, registry, build = node_build
    build()
    registry.commands.clear()
    with monkeypatch.context() as scoped:
        scoped.setattr(node_wheelhouse.sys, "version_info", (3, 11, 0))
        with pytest.raises(ValueError, match="Python 3.12"):
            build()
    assert not registry.commands, "unsupported build host reached registry or build"


def test_wheelhouse_detects_a_previous_lock_changing_during_the_second_download(
    node_build, tmp_path
):
    root, registry, _build = node_build

    def mutate(command):
        if command[-1].endswith("node-tools.lock"):
            (root / "requirements/node-runtime.lock").write_text("changed")

    registry.on_download = mutate
    with pytest.raises(ValueError, match="lock changed"):
        build_wheelhouse(root, tmp_path / "wheelhouse", runner=registry.run)


def test_downloaded_bytes_are_rechecked_against_locks_before_image_publication(
    node_build,
):
    _root, registry, build = node_build

    def corrupt(command):
        if command[-1].endswith("node-tools.lock"):
            destination = Path(command[command.index("--dest") + 1])
            filename, _payload = registry.wheels["node-runtime.lock"]
            (destination / filename).write_bytes(b"unlisted wheel bytes")

    registry.on_download = corrupt
    with pytest.raises(ValueError, match="hash-locked requirements"):
        build()
    assert all("download" in command for command in registry.commands), (
        "corrupt wheel bytes reached image publication"
    )
    assert not NodeImageCache().directory.exists(), (
        "corrupt wheel bytes produced a cache receipt"
    )


@pytest.mark.parametrize(
    "damage",
    ["hash", "package", "version", "platform", "missing", "extra", "duplicate"],
)
def test_rewriting_inventory_and_authentication_cannot_authorize_unlocked_wheels(
    node_build, damage
):
    _root, registry, build = node_build
    build()
    cache = NodeImageCache()
    identity = next(cache.directory.glob("*.json")).stem
    image = cache.load(identity)
    assert image is not None
    inventory = image["inventory"]
    files = inventory["files"]
    filename, _payload = registry.wheels["node-runtime.lock"]
    entry = files[filename]
    if damage == "hash":
        entry["sha256"] = hashlib.sha256(b"unlisted wheel bytes").hexdigest()
    elif damage == "package":
        files["unlisted_package-1.0-py3-none-any.whl"] = files.pop(filename)
    elif damage == "version":
        files[filename.replace("-1.0-", "-2.0-")] = files.pop(filename)
    elif damage == "platform":
        files[filename.replace("py3-none-any", "cp312-cp312-win_amd64")] = files.pop(
            filename
        )
    elif damage == "missing":
        del files[filename]
    elif damage == "extra":
        files["unlisted_package-1.0-py3-none-any.whl"] = dict(entry)
    else:
        files[filename.replace("-py3-", "-py312-")] = dict(entry)
    image["image_inputs"]["wheelhouse_sha256"] = hashlib.sha256(
        (json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    cache.store(identity, image)
    registry.commands.clear()
    with pytest.raises(ValueError, match="locked requirement"):
        build()
    assert not registry.commands, "unlocked wheel inventory reached registry or build"


def test_matching_lock_inventory_must_also_match_the_oci_bound_inventory_digest(
    node_build,
):
    _root, registry, build = node_build
    build()
    cache = NodeImageCache()
    identity = next(cache.directory.glob("*.json")).stem
    image = cache.load(identity)
    assert image is not None
    image["image_inputs"]["wheelhouse_sha256"] = "0" * 64
    cache.store(identity, image)
    registry.commands.clear()
    with pytest.raises(release_image.ReleaseImageError, match="inventory digest"):
        build()
    assert not registry.commands, "forged receipt key authorized an image lookup"


@pytest.fixture
def locked_inventory(node_build):
    root, _registry, build = node_build
    build()
    cache = NodeImageCache()
    identity = next(cache.directory.glob("*.json")).stem
    image = cache.load(identity)
    assert image is not None
    return root, image["inventory"]


def replace_inventory_lock(root, inventory, name, text):
    path = root / "requirements" / name
    path.write_text(text, encoding="utf-8")
    digest = sha256(path)
    inventory["locks"][name] = digest
    inventory["files"][name] = {"sha256": digest, "size": path.stat().st_size}


@pytest.mark.parametrize(
    "requirement",
    [
        "node-runtime-fixture>=1.0 --hash=sha256:{sha}",
        "node-runtime-fixture==1.* --hash=sha256:{sha}",
        "node-runtime-fixture[extra]==1.0 --hash=sha256:{sha}",
        "node-runtime-fixture @ https://invalid.example/a.whl --hash=sha256:{sha}",
        "node-runtime-fixture==1.0",
        "node-runtime-fixture==1.0 --hash=sha512:{sha}",
        "node-runtime-fixture==1.0 --hash=sha256:{sha} --trusted-host=invalid.example",
        "-r https://invalid.example/requirements.txt",
        "node-runtime-fixture==$" + "{{VERSION}} --hash=sha256:{sha}",
        "node-runtime-fixture==1.0 --hash=sha256:{sha}\n"
        "node-runtime-fixture==1.0 --hash=sha256:{sha}",
    ],
)
def test_unknown_or_ambiguous_lock_syntax_cannot_bless_an_inventory(
    locked_inventory, requirement
):
    root, inventory = locked_inventory
    digest = inventory["files"]["node_runtime_fixture-1.0-py3-none-any.whl"]["sha256"]
    replace_inventory_lock(
        root, inventory, "node-runtime.lock", requirement.format(sha=digest) + "\n"
    )
    with pytest.raises(ValueError):
        validate_wheelhouse_inventory(root, inventory)


@pytest.mark.parametrize("conflict", ["version", "hash"])
def test_requirements_shared_by_locks_must_agree(locked_inventory, conflict):
    root, inventory = locked_inventory
    version = "2.0" if conflict == "version" else "1.0"
    digest = "0" * 64
    replace_inventory_lock(
        root,
        inventory,
        "node-tools.lock",
        f"node-runtime-fixture=={version} --hash=sha256:{digest}\n",
    )
    with pytest.raises(ValueError, match="conflicting"):
        validate_wheelhouse_inventory(root, inventory)


def test_lock_markers_and_multiline_sha256_options_use_the_requirement_parser(
    locked_inventory,
):
    root, inventory = locked_inventory
    path = root / "requirements/node-runtime.lock"
    requirement = path.read_text(encoding="utf-8")
    marked = requirement.replace(" --hash", ' ; python_version >= "3.12" \\\n --hash')
    replace_inventory_lock(
        root,
        inventory,
        "node-runtime.lock",
        marked
        + 'windows-only==1.0 ; sys_platform == "win32" --hash=sha256:'
        + "0" * 64,
    )
    assert len(validate_wheelhouse_inventory(root, inventory)) == 64
