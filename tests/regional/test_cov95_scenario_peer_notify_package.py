from __future__ import annotations

import base64
import csv
import hashlib
import shutil
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import gpu_fault
from scripts.component_wheels import (
    MODULES,
    SOURCE_ROOT,
    component_data_files,
    component_definition,
    component_modules,
    component_source_digest,
)
from scripts.e2e.regional.probes import notify008_identity
from tests.regional._cov95_scenario_peer_safety import (
    scenario_peer_transport_guard_fixture as scenario_peer_transport_guard_fixture,
)


def test_real_control_plane_component_closure_is_not_rejected_as_deploy_host(
    monkeypatch, tmp_path
):
    component = component_definition("control_plane")
    selected = component_modules("control_plane")
    staged = tmp_path / "site-packages"
    inputs = [MODULES[name] for name in sorted(selected)]
    inputs.extend(component_data_files("control_plane"))
    copied = []
    for source in inputs:
        destination = staged / source.relative_to(SOURCE_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    package = staged / "gpu_fault"
    metadata = staged / f"gpu_fault_control_plane-{gpu_fault.__version__}.dist-info"
    metadata.mkdir()
    metadata_file = metadata / "METADATA"
    metadata_file.write_text(
        f"Metadata-Version: 2.1\nName: {component.distribution}\n"
        f"Version: {gpu_fault.__version__}\n"
    )
    with (metadata / "RECORD").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows(
            (
                path.relative_to(staged).as_posix(),
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
                .decode()
                .rstrip("="),
                path.stat().st_size,
            )
            for path in [*copied, metadata_file]
        )
        writer.writerow(((metadata / "RECORD").relative_to(staged).as_posix(), "", ""))
    monkeypatch.syspath_prepend(str(staged))
    monkeypatch.setattr(
        notify008_identity,
        "gpu_fault",
        SimpleNamespace(
            __file__=str(package / "__init__.py"),
            module_digest=partial(gpu_fault.module_digest, package_dir=package),
        ),
    )

    assert "gpu_fault.admin" in selected and (package / "admin").is_dir(), (
        "the staged package did not include the real control-plane admin dependency"
    )
    expected_digest = component_source_digest("control_plane")
    assert gpu_fault.module_digest(package_dir=package) == expected_digest, (
        "the temporary component does not match the actual generated module/data closure"
    )
    observed = notify008_identity.runtime_identity()

    assert observed == {
        "distribution": component.distribution,
        "version": gpu_fault.__version__,
        "module_digest": expected_digest,
    }, (
        "a valid staged control-plane distribution did not produce its own runtime identity"
    )
    assert not (Path(staged) / "gpu_fault_release").exists(), (
        "the test accidentally staged a deploy-host release-engine package"
    )
