from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

import pytest

from scripts import component_wheels
from scripts.e2e.regional.notify008_bundle import source_bundle
from scripts.e2e.regional.probes import notify008_identity


@pytest.fixture(name="cpu_wheel", scope="module")
def cpu_wheel_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("notify008-actual-component")
    output = root / "wheels"
    output.mkdir()
    return component_wheels.build_component(
        python=sys.executable,
        name="control_plane",
        build_root=root / "build",
        output=output,
    )


def unpack_component(cpu_wheel, destination):
    wheel, expected_digest, modules = cpu_wheel
    destination.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(destination)
    return expected_digest, modules


def identity_process(package):
    environment = {**os.environ, "PYTHONPATH": str(package)}
    return subprocess.run(
        [sys.executable, str(Path(notify008_identity.__file__).resolve())],
        cwd=package,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@contextmanager
def component_identity(package):
    spec = importlib.util.spec_from_file_location(
        "notify008_component_package", package / "gpu_fault/__init__.py"
    )
    assert spec is not None and spec.loader is not None, (
        "the real component must be importable"
    )
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    info = metadata.Distribution.at(
        next(package.glob("gpu_fault_control_plane-*.dist-info"))
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(notify008_identity, "gpu_fault", runtime)
        patch.setattr(notify008_identity.metadata, "distribution", lambda name: info)
        yield notify008_identity


def payload_process(package, payload_root):
    sources = source_bundle()
    folder = payload_root / "scripts/e2e/regional/probes"
    folder.mkdir(parents=True)
    for name, text in sources.items():
        (folder / name).write_text(text, encoding="utf-8")
    modules = [
        f"scripts.e2e.regional.probes.{Path(name).stem}" for name in sorted(sources)
    ]
    script = """
import importlib, importlib.abc, json, sys
allowed = set(json.loads(sys.argv[1]))
class ClosureGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("scripts.") and fullname not in allowed and fullname not in {
            "scripts.e2e", "scripts.e2e.regional", "scripts.e2e.regional.probes"
        }:
            raise ImportError("unshipped script dependency")
        if fullname.startswith("gpu_fault_release") or fullname == "gpu_fault.admin.cli":
            raise ImportError("forbidden deploy-host dependency")
        return None
sys.meta_path.insert(0, ClosureGuard())
for name in sorted(allowed):
    importlib.import_module(name)
print(json.dumps(sorted(allowed)))
"""
    import json

    return subprocess.run(
        [sys.executable, "-c", script, json.dumps(modules)],
        cwd=payload_root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(package), str(payload_root))),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    ), modules
