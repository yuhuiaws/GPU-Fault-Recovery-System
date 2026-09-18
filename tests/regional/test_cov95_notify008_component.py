from __future__ import annotations

import base64
import csv
import hashlib
import json
import runpy
import sys

import pytest

from tests.regional._cov95_notify008_component import (
    component_identity,
    identity_process,
    payload_process,
    unpack_component,
)
from tests.regional._cov95_notify008_component import (
    cpu_wheel_fixture as cpu_wheel_fixture,
)


def test_identity_accepts_the_actual_control_plane_wheel_and_its_shared_admin_closure(
    cpu_wheel, tmp_path
):
    package = tmp_path / "installed"
    expected_digest, modules = unpack_component(cpu_wheel, package)
    assert {name for name in modules if name.startswith("gpu_fault.admin")} == {
        "gpu_fault.admin",
        "gpu_fault.admin.atomic_json",
        "gpu_fault.admin.capacity_evidence",
        "gpu_fault.admin.config",
        "gpu_fault.admin.config_parser",
        "gpu_fault.admin.operation_lock",
        "gpu_fault.admin.python_environment",
        "gpu_fault.admin.site",
    }, (
        "the guard must use the actual reviewed component closure, not an empty admin package"
    )

    completed = identity_process(package)

    assert completed.returncode == 0, (
        f"the real control-plane component wheel was refused: {completed.stdout} {completed.stderr}"
    )
    report = json.loads(completed.stdout)
    assert report["distribution"] == "gpu-fault-control-plane", (
        "identity must come from the installed component metadata"
    )
    assert report["module_digest"] == expected_digest, (
        "the running package must match the real component builder's digest"
    )
    with component_identity(package) as probe:
        assert probe.runtime_identity() == report, (
            "the measured guard and standalone probe must agree on actual wheel identity"
        )


@pytest.mark.parametrize(
    "drift",
    [
        "admin-cli",
        "admin-mutation",
        "release-package",
        "changed-shared-module",
        "changed-python-environment",
    ],
)
def test_identity_refuses_forbidden_or_modified_surfaces_in_a_real_component_wheel(
    cpu_wheel, tmp_path, drift
):
    package = tmp_path / "installed"
    unpack_component(cpu_wheel, package)
    if drift == "admin-cli":
        (package / "gpu_fault/admin/cli.py").write_text(
            "raise RuntimeError('not shipped')\n"
        )
    elif drift == "admin-mutation":
        path = package / "gpu_fault/admin/cluster_removal.py"
        content = b"UNREVIEWED_CHANGE = True\n"
        path.write_bytes(content)
        recorded_hash = (
            base64.urlsafe_b64encode(hashlib.sha256(content).digest())
            .decode()
            .rstrip("=")
        )
        # A matching inventory entry cannot authorize deploy-host mutation code.
        with next(package.glob("*.dist-info/RECORD")).open("a", newline="") as stream:
            csv.writer(stream).writerow(
                (
                    path.relative_to(package).as_posix(),
                    "sha256=" + recorded_hash,
                    len(content),
                )
            )
    elif drift == "release-package":
        (package / "gpu_fault_release").mkdir()
    else:
        name = (
            "python_environment.py"
            if drift == "changed-python-environment"
            else "atomic_json.py"
        )
        path = package / "gpu_fault/admin" / name
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\nUNREVIEWED_CHANGE = True\n")

    completed = identity_process(package)

    assert completed.returncode == 1, (
        "unreviewed deploy-host or changed code must be refused"
    )
    assert json.loads(completed.stdout) == {"error": "RuntimeError"}, (
        "the refusal must not print package contents or credential-like data"
    )
    expected_error = (
        "differ from wheel metadata"
        if drift.startswith("changed-")
        else "forbidden admin or release surfaces"
    )
    with (
        component_identity(package) as probe,
        pytest.raises(RuntimeError, match=expected_error),
    ):
        probe.runtime_identity()


def test_all_shipped_probe_modules_import_against_only_the_actual_cpu_component(
    cpu_wheel, tmp_path
):
    package = tmp_path / "installed"
    unpack_component(cpu_wheel, package)

    completed, modules = payload_process(package, tmp_path / "payload")

    assert completed.returncode == 0, (
        f"the isolated CPU payload reached an unshipped dependency: {completed.stderr}"
    )
    assert json.loads(completed.stdout) == modules, (
        "all portable probes must load without host runner/shared guards or admin CLIs"
    )


@pytest.mark.parametrize(
    "drift",
    [
        "inventory",
        "malformed-inventory",
        "file",
        "hash",
        "hash-mode",
        "outside-root",
        "import-root",
    ],
)
def test_installed_component_requires_complete_record_and_import_origin(
    cpu_wheel, tmp_path, monkeypatch, drift
):
    package = tmp_path / "installed"
    unpack_component(cpu_wheel, package)
    record = next(package.glob("*.dist-info/RECORD"))
    if drift == "inventory":
        record.unlink()
    elif drift == "malformed-inventory":
        record.write_text("gpu_fault/__init__.py\n")
    elif drift == "file":
        (package / "gpu_fault/admin/atomic_json.py").unlink()
    elif drift in {"hash", "hash-mode", "outside-root"}:
        with record.open(newline="") as stream:
            rows = list(csv.reader(stream))
        row = next(item for item in rows if item[0] == "gpu_fault/admin/atomic_json.py")
        if drift == "hash":
            row[1] = ""
        elif drift == "hash-mode":
            row[1] = "sha512=" + row[1].split("=", 1)[1]
        else:
            row[0] = "gpu_fault/../../outside.py"
        with record.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
    with component_identity(package) as probe:
        if drift == "import-root":
            monkeypatch.setattr(
                probe.gpu_fault, "__file__", str(tmp_path / "shadow/__init__.py")
            )
        with pytest.raises(RuntimeError, match="inventory|identity|distribution"):
            probe.runtime_identity()


def test_identity_main_and_script_mode_use_real_metadata_and_sanitize_refusal(
    cpu_wheel, tmp_path, monkeypatch, capsys
):
    package = tmp_path / "installed"
    expected_digest, _ = unpack_component(cpu_wheel, package)
    with component_identity(package) as probe:
        assert probe.main() == 0, (
            "the identity command must accept actual wheel metadata"
        )
        assert (
            json.loads(capsys.readouterr().out)["module_digest"] == expected_digest
        ), "command output must carry the component builder's content identity"
        monkeypatch.setitem(sys.modules, "gpu_fault", probe.gpu_fault)
        with pytest.raises(SystemExit) as raised:
            runpy.run_path(probe.__file__, run_name="__main__")
        assert raised.value.code == 0, (
            "standalone identity entry must use the same guard"
        )
        assert (
            json.loads(capsys.readouterr().out)["module_digest"] == expected_digest
        ), "standalone output must agree with the callable identity entry"
        (package / "gpu_fault/admin/cli.py").write_text("UNAPPROVED = True\n")
        assert probe.main() == 1, (
            "a forbidden CLI surface must prevent identity attestation"
        )
        assert json.loads(capsys.readouterr().out) == {"error": "RuntimeError"}, (
            "identity refusal must expose only its error class"
        )
