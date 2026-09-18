from __future__ import annotations

import copy
import csv
import hashlib
import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from gpu_fault_release.regional_release_config import (
    ReleaseError,
    parse_delivery_identity,
)
from scripts import release_image, release_image_set
from scripts.release_identity import (
    ReleaseIdentityError,
    bind_runtime_image,
    build_release_identity,
)
from scripts.verify_release_images import LEGACY_COMPONENT_DOCKERFILE_SHA256
from tests.test_release_identity import identity_root
from tests.test_release_image import fake_component_builder, inspected_image

ROOT = Path(__file__).resolve().parents[1]


def write_node_dependency_locks(root: Path) -> dict[str, tuple[str, bytes]]:
    wheels = {}
    for lock_name in ("node-runtime.lock", "node-tools.lock"):
        project = Path(lock_name).stem + "-fixture"
        payload = project.encode()
        wheels[lock_name] = (
            project.replace("-", "_") + "-1.0-py3-none-any.whl",
            payload,
        )
        (root / "requirements" / lock_name).write_text(
            f"{project}==1.0 --hash=sha256:{hashlib.sha256(payload).hexdigest()}\n",
            encoding="utf-8",
        )
    return wheels


@pytest.fixture
def image_set(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / "checkout"
    root.mkdir()
    identity_root(root)
    for relative in (
        "deploy/image/Dockerfile.component",
        "deploy/image/Dockerfile.node-dependencies",
        "scripts/node_wheelhouse.py",
        "scripts/release_image.py",
        "scripts/release_image_set.py",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    shutil.copyfile(
        root / "config/images.json", root / "config/release-images.lock.json"
    )
    node_wheels = write_node_dependency_locks(root)
    contexts = {}
    images = {}
    lock = threading.Lock()
    monkeypatch.setattr(release_image, "build_component", fake_component_builder)

    def run(command, **kwargs):
        if "download" in command:
            destination = Path(command[command.index("--dest") + 1])
            filename, payload = node_wheels[Path(command[-1]).name]
            (destination / filename).write_bytes(payload)
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:3] == ["docker", "buildx", "imagetools"]:
            with lock:
                result = images.get(command[-1])
            return subprocess.CompletedProcess(
                command,
                0 if result else 1,
                result or "",
                "" if result else "manifest unknown",
            )
        context = Path(command[-1])
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
        with lock:
            contexts[tag] = {
                "wheels": sorted(path.name for path in (context / "wheels").glob("*")),
                "node_files": sorted(
                    path.name for path in (context / "wheelhouse").glob("*")
                ),
                "cache_from": [
                    value
                    for index, value in enumerate(command)
                    if index and command[index - 1] == "--cache-from"
                ],
                "cache_to": [
                    value
                    for index, value in enumerate(command)
                    if index and command[index - 1] == "--cache-to"
                ],
            }
            images[tag] = inspected_image(labels, digest=digest)
        Path(command[command.index("--metadata-file") + 1]).write_text(
            json.dumps({"containerimage.digest": digest})
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    descriptor = release_image_set.build_image_set(
        root,
        repository="registry.example/runtime",
        push=True,
        cache_from=("type=registry,ref=registry.example/cache:buildcache-linux-amd64",),
        cache_to=(
            "type=registry,ref=registry.example/cache:buildcache-linux-amd64,mode=max,"
            "image-manifest=true,oci-mediatypes=true",
        ),
        runner=run,
    )
    return descriptor, contexts, root


def test_each_component_build_contains_only_its_own_wheel(image_set):
    descriptor, contexts, _root = image_set
    images = descriptor["images"]
    assert descriptor["schema_version"] == 3
    assert len({item["reference"] for item in images.values()}) == 3
    for component, prefix in (
        ("control_plane", "gpu_fault_control_plane"),
        ("executor", "gpu_fault_cluster_executor"),
    ):
        files = contexts[images[component]["tag"]]["wheels"]
        assert len(files) == 1 and files[0].startswith(prefix + "-")
        assert set(images[component]["components"]) == {component}
    node = images["node_dependencies"]
    assert node["components"] == {}
    assert "inventory.json" in contexts[node["tag"]]["node_files"]
    assert "node-tools.lock" in contexts[node["tag"]]["node_files"]


def test_each_buildkit_command_receives_only_its_component_cache_export(image_set):
    descriptor, contexts, _root = image_set
    for component, image in descriptor["images"].items():
        context = contexts[image["tag"]]
        scoped_ref = (
            "registry.example/cache:buildcache-linux-amd64-"
            + component.replace("_", "-")
        )
        assert [parsed_cache(value) for value in context["cache_to"]] == [
            {
                "type": "registry",
                "ref": scoped_ref,
                "mode": "max",
                "image-manifest": "true",
                "oci-mediatypes": "true",
            }
        ]
        assert [parsed_cache(value) for value in context["cache_from"]] == [
            {"type": "registry", "ref": scoped_ref},
            {
                "type": "registry",
                "ref": "registry.example/cache:buildcache-linux-amd64",
            },
        ]
        assert image["build_cache"] == {"from_configured": True, "to_configured": True}
        assert scoped_ref not in json.dumps(descriptor), (
            "mutable layer cache identity leaked into the release descriptor"
        )


def test_split_image_binding_and_release_loading_agree(image_set):
    descriptor, _, root = image_set
    bound = bind_runtime_image(root, build_release_identity(root), descriptor)
    manifest = {"schema_version": 4, "deployable": True, "delivery": bound}
    components = {
        **descriptor["components"],
        "node_bundle": {"template_sha256": bound["node_template_inputs"]["sha256"]},
    }
    _, _, _, images, _ = parse_delivery_identity(manifest, components)
    assert images["executor"] != images["runtime"]
    assert (
        images["node_dependencies"]
        == descriptor["images"]["node_dependencies"]["reference"]
    )


@pytest.mark.parametrize(
    "mutation", ["missing", "wheel", "inventory", "source", "mutable", "layout"]
)
def test_split_descriptor_identity_tampering_fails_closed(image_set, mutation):
    root = image_set[2]
    descriptor = copy.deepcopy(image_set[0])
    if mutation == "missing":
        del descriptor["images"]["executor"]
    elif mutation == "wheel":
        descriptor["components"]["executor"]["wheel_sha256"] = "0" * 64
    elif mutation == "inventory":
        descriptor["images"]["node_dependencies"]["wheelhouse_sha256"] = "0" * 64
    elif mutation == "source":
        descriptor["source_identity_sha256"] = "0" * 64
    elif mutation == "mutable":
        descriptor["images"]["executor"]["reference"] = (
            "registry.example/runtime:latest"
        )
    else:
        descriptor["images"]["executor"]["image_inputs"]["build_args"]["COMPONENT"] = (
            "control-plane"
        )
    with pytest.raises(ReleaseIdentityError):
        bind_runtime_image(root, build_release_identity(root), descriptor)


def test_new_candidate_cannot_claim_the_legacy_environment_contract(image_set):
    original, _contexts, root = image_set
    descriptor = copy.deepcopy(original)
    image = descriptor["images"]["control_plane"]
    image["dockerfile_sha256"] = LEGACY_COMPONENT_DOCKERFILE_SHA256
    image["image_inputs"]["dockerfile_sha256"] = LEGACY_COMPONENT_DOCKERFILE_SHA256
    image["image_input_sha256"] = release_image.canonical_sha256(image["image_inputs"])
    with pytest.raises(ReleaseIdentityError, match="dockerfile_sha256"):
        bind_runtime_image(root, build_release_identity(root), descriptor)


def test_v4_cannot_fall_back_to_a_shared_image(image_set):
    root = image_set[2]
    bound = bind_runtime_image(root, build_release_identity(root), image_set[0])
    bound["images"].pop("executor")
    from scripts.release_identity import canonical_sha256

    bound.pop("sha256")
    bound["sha256"] = canonical_sha256(bound)
    with pytest.raises(ReleaseError):
        parse_delivery_identity(
            {"schema_version": 4, "deployable": True, "delivery": bound},
            image_set[0]["components"],
        )


@pytest.fixture
def cache_builds(monkeypatch, tmp_path):
    calls = {}
    identity = {"sha256": "a" * 64}
    monkeypatch.setattr(
        release_image_set, "build_release_identity", lambda _root: identity
    )

    def runtime(_root, *, component, **kwargs):
        calls[component] = kwargs
        return {
            "source_identity_sha256": identity["sha256"],
            "components": {component: {}},
        }

    def node(root, **kwargs):
        return runtime(root, component="node_dependencies", **kwargs)

    monkeypatch.setattr(release_image_set, "build_runtime_image", runtime)
    monkeypatch.setattr(release_image_set, "build_node_dependency_image", node)

    def build(**kwargs):
        return release_image_set.build_image_set(
            tmp_path,
            repository="registry.example/runtime",
            push=kwargs.pop("push", True),
            **kwargs,
        )

    return build, calls, identity


def parsed_cache(value):
    return dict(field.split("=", 1) for field in next(csv.reader([value], strict=True)))


@pytest.mark.parametrize("push", [False, True])
@pytest.mark.parametrize(
    ("reference", "repository", "tag"),
    [
        (
            "registry.example/cache:buildcache-linux-amd64",
            "registry.example/cache",
            "buildcache-linux-amd64",
        ),
        (
            "registry.example:5000/team/cache:base",
            "registry.example:5000/team/cache",
            "base",
        ),
        ("[2001:db8::1]:5000/cache:base", "[2001:db8::1]:5000/cache", "base"),
        ("registry.example/cache", "registry.example/cache", "latest"),
        ("team/cache:base", "team/cache", "base"),
    ],
)
def test_component_registry_cache_exports_are_stable_and_never_write_the_shared_tag(
    cache_builds, push, reference, repository, tag
):
    build, calls, identity = cache_builds
    imported = f"type=registry,ref={reference}"
    exported = (
        f"type=registry,ref={reference},mode=max,image-manifest=true,"
        "oci-mediatypes=true,compression=zstd,compression-level=3"
    )
    options = {"push": push, "cache_from": (imported,), "cache_to": (exported,)}
    build(**options)
    first = copy.deepcopy(calls)
    identity["sha256"] = "b" * 64
    build(**options)
    assert calls == first, "source changes must not churn component cache targets"
    assert set(calls) == {"control_plane", "executor", "node_dependencies"}
    targets = set()
    for component, arguments in calls.items():
        expected = f"{repository}:{tag}-{component.replace('_', '-')}"
        exports = [parsed_cache(value) for value in arguments["cache_to"]]
        assert exports == [{**parsed_cache(exported), "ref": expected}]
        imports = [parsed_cache(value) for value in arguments["cache_from"]]
        assert imports == [
            {"type": "registry", "ref": expected},
            {"type": "registry", "ref": reference},
        ]
        assert imported in arguments["cache_from"]
        assert expected != reference
        targets.add(expected)
    assert len(targets) == 3


def test_local_cache_csv_preserves_paths_and_uses_separate_component_indexes(
    cache_builds, tmp_path
):
    build, calls, _identity = cache_builds
    source = tmp_path / 'old cache, version="one"'
    destination = Path('new cache, version="two"')
    imported = f'type=local,"src={tmp_path}/old cache, version=""one"""'
    exported = 'type=local,"dest=new cache, version=""two""",mode=max,compression=zstd'
    build(cache_from=(imported,), cache_to=(exported,))
    for component, arguments in calls.items():
        suffix = component.replace("_", "-")
        exports = [parsed_cache(value) for value in arguments["cache_to"]]
        assert exports == [
            {
                "type": "local",
                "dest": str(destination / suffix),
                "mode": "max",
                "compression": "zstd",
            }
        ]
        imports = [parsed_cache(value) for value in arguments["cache_from"]]
        assert imports == [
            {"type": "local", "src": str(source / suffix)},
            {"type": "local", "src": str(source)},
        ]
    assert not (tmp_path / destination).exists(), (
        "cache planning must not create or overwrite local indexes"
    )


def test_pinned_or_backend_specific_cache_imports_are_preserved_without_guessing(
    cache_builds,
):
    build, calls, _identity = cache_builds
    digest = "sha256:" + "a" * 64
    imports = (
        f"type=registry,ref=registry.example/cache@{digest}",
        f"type=registry,ref=registry.example/cache:base@{digest}",
        f"type=local,src=cache,digest={digest}",
        "type=local,src=cache,tag=previous",
        "type=gha,scope=previous",
        "type=s3,region=us-east-1,bucket=example-cache,name=previous",
    )
    build(cache_from=imports)
    for arguments in calls.values():
        assert arguments["cache_from"] == imports
        assert arguments["cache_to"] == ()


def test_shorthand_registry_imports_keep_legacy_reuse_and_add_component_reads(
    cache_builds,
):
    build, calls, _identity = cache_builds
    imported = "registry.example:5000/cache:base,registry.example/other:base"
    build(cache_from=(imported, imported))
    for component, arguments in calls.items():
        suffix = component.replace("_", "-")
        assert arguments["cache_from"] == (
            f"type=registry,ref=registry.example:5000/cache:base-{suffix}",
            f"type=registry,ref=registry.example/other:base-{suffix}",
            imported,
        )


@pytest.mark.parametrize(
    "exported",
    [
        "type=gha,scope=shared",
        "type=inline",
        "type=s3,bucket=cache",
        "registry.example/cache:base",
        "ref=registry.example/cache:base",
        "type=registry",
        "type=local,src=cache",
        "type=registry,ref=registry.example/cache:base,scope=shared",
        "type=registry,ref=registry.example/cache:base,dest=cache",
        "type=local,dest=cache,ref=other",
        "type=local,dest=cache,tag=shared",
        "type=registry,type=local,ref=registry.example/cache:base",
        "type=registry,ref=registry.example/cache:base,REF=registry.example/other:base",
        "type=local,dest=cache,dest=other",
        "type=local,dest=",
        "type=local,dest=cache,,mode=max",
        'type=local,dest="cache"',
        'type=local,"dest=cache',
        "type=local,dest=cache\nother",
        "type=local,dest=cache\x00other",
        "type=registry,ref=registry.example/cache@sha256:" + "a" * 64,
        "type=registry,ref=registry.example/cache:",
        "type=registry,ref=registry.example/cache:base:other",
        "type=registry,ref=registry.example/cache:" + "a" * 128,
        "type=registry,ref=https://registry.example/cache:base",
        "type=registry,ref=registry.example/cache:base?token=synthetic-cache-canary",
    ],
)
def test_unsupported_or_ambiguous_cache_exports_stop_before_any_build(
    cache_builds, exported
):
    build, calls, _identity = cache_builds
    with pytest.raises(release_image.ReleaseImageError, match="cache") as caught:
        build(cache_to=("type=registry,ref=registry.example/cache:valid", exported))
    assert not calls, "invalid cache exports started component work"
    assert "synthetic-cache-canary" not in str(caught.value)


@pytest.mark.parametrize(
    "exports",
    [
        (
            "type=registry,ref=registry.example/cache",
            "type=registry,ref=registry.example/cache:latest,mode=max",
        ),
        (
            "type=registry,ref=cache:base",
            "type=registry,ref=index.docker.io/library/cache:base,mode=max",
        ),
        ("type=local,dest=cache", "type=local,dest=./cache,mode=max"),
    ],
)
def test_duplicate_cache_destinations_are_rejected_before_builds(cache_builds, exports):
    build, calls, _identity = cache_builds
    with pytest.raises(release_image.ReleaseImageError, match="duplicate"):
        build(cache_to=exports)
    assert not calls, "duplicate exports reached parallel builders"


@pytest.mark.parametrize(
    "exports",
    [
        (
            "type=registry,ref=registry.example/cache:base",
            "type=registry,ref=registry.example/cache:base-executor",
        ),
        (
            "type=registry,ref=cache:base",
            "type=registry,ref=docker.io/library/cache:base-node-dependencies",
        ),
        ("type=local,dest=cache", "type=local,dest=cache/node-dependencies"),
    ],
)
def test_component_exports_cannot_overwrite_another_configured_shared_cache(
    cache_builds, exports
):
    build, calls, _identity = cache_builds
    with pytest.raises(release_image.ReleaseImageError, match="overlaps a shared"):
        build(cache_to=exports)
    assert not calls, "overlapping cache exports started parallel work"


def test_local_component_cache_symlink_cannot_overwrite_the_legacy_index(
    cache_builds, tmp_path
):
    build, calls, _identity = cache_builds
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "control-plane").symlink_to(cache, target_is_directory=True)
    with pytest.raises(release_image.ReleaseImageError, match="symlink"):
        build(cache_to=(f"type=local,dest={cache}",))
    assert not calls, "a component cache symlink reached the builders"


def test_no_cache_configuration_remains_an_uncached_build(cache_builds):
    build, calls, _identity = cache_builds
    build()
    for arguments in calls.values():
        assert arguments["cache_from"] == ()
        assert arguments["cache_to"] == ()


def test_multiple_cache_exporters_keep_their_own_options_and_component_targets(
    cache_builds,
):
    build, calls, _identity = cache_builds
    build(
        cache_to=(
            "type=registry,ref=registry.example/cache:base,mode=max,"
            "image-manifest=true,oci-mediatypes=true",
            "type=local,dest=cache,mode=min,compression=zstd,force-compression=true",
        )
    )
    for component, arguments in calls.items():
        suffix = component.replace("_", "-")
        assert [parsed_cache(value) for value in arguments["cache_to"]] == [
            {
                "type": "registry",
                "ref": f"registry.example/cache:base-{suffix}",
                "mode": "max",
                "image-manifest": "true",
                "oci-mediatypes": "true",
            },
            {
                "type": "local",
                "dest": f"cache/{suffix}",
                "mode": "min",
                "compression": "zstd",
                "force-compression": "true",
            },
        ]


def test_long_import_tag_is_preserved_when_component_scoping_is_not_possible(
    cache_builds,
):
    build, calls, _identity = cache_builds
    imported = "type=registry,ref=registry.example/cache:" + "a" * 128
    build(cache_from=(imported,))
    for arguments in calls.values():
        assert arguments["cache_from"] == (imported,)
