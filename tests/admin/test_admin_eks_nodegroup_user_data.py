"""The managed user data part of ``prepare-eks-nodegroup``: merge, scripts, analysis.

The pre-existing launch template states, one test each: (a) no user data,
(b) only the EKS NodeConfig part, (c) an unmarked hand-written shell part
(kept and analysed by default, replaced only when adopted), (d) the managed
part already present -- identical (no-op) or different (only it replaced).
"""

from __future__ import annotations

import base64
import configparser
import email
import gzip
import subprocess
from pathlib import Path

import pytest

from gpu_fault.admin import eks_nodegroup_user_data as user_data
from gpu_fault.admin.bootstrap_common import BootstrapError
from tests.admin._eks_nodegroup_world import (
    NODECONFIG_PART,
    SITE_SCRIPT_PART,
    site_user_data,
)

EMBEDDED = user_data.managed_script()
PIN = user_data.dcgm_pin("4.7.0-2")
HOST = user_data.managed_script(dcgm=PIN)
#: The hand-written part of the live gpu-eks-cluster template, as reported.
HAND_WRITTEN = (
    "MIME-Version: 1.0\n"
    'Content-Type: multipart/mixed; boundary="==GPUFAULT=="\n'
    "\n"
    "--==GPUFAULT==\n"
    'Content-Type: text/x-shellscript; charset="us-ascii"\n'
    'Content-Disposition: attachment; filename="gpu-fault-node-prerequisites.sh"\n'
    "\n"
    "#!/bin/bash\n"
    "# gpu-fault plain-EKS GPU node prerequisites (persisted in the launch template)\n"
    "set -uo pipefail\n"
    "for attempt in 1 2 3 4 5; do\n"
    "  if dnf install -y -q python3.12 strace datacenter-gpu-manager-4-core "
    "datacenter-gpu-manager-4-cuda13; then\n"
    '    echo "installed (attempt ${attempt})"; break\n'
    "  fi\n"
    "  sleep 30\n"
    "done\n"
    "rpm -q python3.12 strace || true\n"
    "\n"
    "--==GPUFAULT==--\n"
)


def _parts(text: str) -> list[tuple[str, str | None, str]]:
    message = email.message_from_string(text)
    return [
        (part.get_content_type(), part.get_filename(), str(part.get_payload()))
        for part in message.walk()
        if not part.is_multipart()
    ]


def test_no_user_data_creates_a_document_with_only_the_managed_part() -> None:
    merge = user_data.merge_managed_part(None, EMBEDDED)

    assert merge.action == "create"
    parts = _parts(merge.text)
    assert [(kind, name) for kind, name, _body in parts] == [
        ("text/x-shellscript", user_data.MANAGED_PART_FILENAME)
    ], "EKS merges its own NodeConfig part; the verb writes only the shell part"
    assert parts[0][2].rstrip("\n") == EMBEDDED.rstrip("\n")
    again = user_data.merge_managed_part(merge.text, EMBEDDED)
    assert (again.action, again.text) == ("keep", merge.text), "a rerun is a no-op"


def test_nodeconfig_only_user_data_gets_the_part_appended_and_keeps_nodeconfig() -> (
    None
):
    current = (
        "MIME-Version: 1.0\n"
        'Content-Type: multipart/mixed; boundary="//"\n'
        "\n"
        "--//\n" + NODECONFIG_PART + "--//--\n"
    )

    merge = user_data.merge_managed_part(current, EMBEDDED)

    assert merge.action == "add"
    assert merge.text.startswith(current.removesuffix("--//--\n")), (
        "the NodeConfig part and everything before it stay byte for byte"
    )
    assert [kind for kind, _name, _body in _parts(merge.text)] == [
        "application/node.eks.aws",
        "text/x-shellscript",
    ]
    assert merge.text.endswith("command -v strace\n--//--\n"), (
        "the managed part is the last part"
    )
    assert [item.content_type for item in merge.parts] == ["application/node.eks.aws"]


def test_site_parts_are_kept_byte_for_byte_beside_the_managed_part() -> None:
    current = site_user_data()

    merge = user_data.merge_managed_part(current, EMBEDDED)

    assert merge.action == "add"
    assert NODECONFIG_PART in merge.text and SITE_SCRIPT_PART in merge.text, (
        "neither the NodeConfig nor the site's own script part is rewritten"
    )
    assert merge.text.index(SITE_SCRIPT_PART) < merge.text.index(
        user_data.MANAGED_PART_MARKER
    ), "the managed part is appended after the existing parts"
    assert [item.managed for item in merge.parts] == [False, False]


def test_crlf_documents_keep_their_line_endings() -> None:
    current = site_user_data().replace("\n", "\r\n")

    merge = user_data.merge_managed_part(current, EMBEDDED)

    assert merge.text.startswith(current.removesuffix("--==SITE==--\r\n")), (
        "the existing CRLF parts are kept byte for byte"
    )
    assert "\n" not in merge.text.replace("\r\n", ""), "no bare LF is introduced"
    again = user_data.merge_managed_part(merge.text, EMBEDDED)
    assert again.action == "keep"


def test_an_identical_managed_part_is_a_no_op() -> None:
    current = user_data.merge_managed_part(site_user_data(), EMBEDDED).text

    merge = user_data.merge_managed_part(current, EMBEDDED)

    assert (merge.action, merge.changed, merge.text) == ("keep", False, current)
    assert merge.managed_script == EMBEDDED


def test_a_different_managed_part_is_replaced_and_nothing_else_moves() -> None:
    current = user_data.merge_managed_part(site_user_data(), EMBEDDED).text

    merge = user_data.merge_managed_part(current, HOST)

    assert merge.action == "replace"
    head = current[
        : current.index('Content-Disposition: attachment; filename="gpu-fault-managed')
    ]
    assert merge.text.startswith(head), "the parts before the managed one are untouched"
    assert merge.text.endswith("dcgmi discovery -l >/dev/null\n--==SITE==--\n"), (
        "the replacement sits where the old managed part was"
    )
    assert merge.managed_script == EMBEDDED, "the replaced script is reported"
    assert user_data.merge_managed_part(merge.text, HOST).action == "keep"


def test_a_hand_written_part_is_kept_and_its_package_lines_are_reported() -> None:
    merge = user_data.merge_managed_part(HAND_WRITTEN, EMBEDDED)

    assert merge.action == "add", "the hand-written part is never dropped by default"
    assert HAND_WRITTEN.removesuffix("--==GPUFAULT==--\n") in merge.text
    (part,) = merge.parts
    described = part.describe()
    assert described["install_lines"] == [
        {
            "line": 5,
            "text": "if dnf install -y -q python3.12 strace "
            "datacenter-gpu-manager-4-core datacenter-gpu-manager-4-cuda13; then",
            "packages": [
                "python3.12",
                "strace",
                "datacenter-gpu-manager-4-core",
                "datacenter-gpu-manager-4-cuda13",
            ],
        }
    ], "the rpm -q line installs nothing and is not listed"
    embedded = user_data.part_findings(
        part, mode="embedded", pin=None, driver_majors=["13"]
    )
    assert embedded["overlaps"] == [
        "part 1 line 5: python3.12 (the managed part installs it too)",
        "part 1 line 5: strace (the managed part installs it too)",
    ], "the duplicated installs are named, not silent"
    assert embedded["contradictions"] == []
    assert embedded["other_packages"] == [
        "part 1 line 5: datacenter-gpu-manager-4-core",
        "part 1 line 5: datacenter-gpu-manager-4-cuda13",
    ]


def test_a_hand_written_part_contradicting_the_pin_or_the_driver_is_flagged() -> None:
    (part,) = user_data.merge_managed_part(HAND_WRITTEN, HOST).parts

    host = user_data.part_findings(part, mode="host", pin=PIN, driver_majors=["12"])

    assert (
        "part 1 line 5: datacenter-gpu-manager-4-core is unpinned (the newest "
        "build), the pin is 4.7.0-2"
    ) in host["contradictions"]
    assert (
        "part 1 line 5: datacenter-gpu-manager-4-cuda13 is the CUDA 13 build; the "
        "nodes' driver reports CUDA 12"
    ) in host["contradictions"]
    pinned = HAND_WRITTEN.replace(
        "datacenter-gpu-manager-4-core ", "datacenter-gpu-manager-4-core-4.7.0-1 "
    )
    (other,) = user_data.merge_managed_part(pinned, HOST).parts
    assert (
        "part 1 line 5: datacenter-gpu-manager-4-core-4.7.0-1 installs 4.7.0-1, the "
        "pin is 4.7.0-2"
    ) in user_data.part_findings(other, mode="host", pin=PIN, driver_majors=["13"])[
        "contradictions"
    ]
    version_only = user_data.dcgm_pin("4.7.0")
    assert not [
        line
        for line in user_data.part_findings(
            other, mode="host", pin=version_only, driver_majors=["13"]
        )["contradictions"]
        if "installs 4.7.0-1" in line
    ], "a version pin accepts every build of that version"


def test_a_hand_written_host_engine_line_contradicts_embedded_mode() -> None:
    script = HAND_WRITTEN.replace(
        "rpm -q python3.12 strace || true\n",
        "sudo systemctl enable --now nvidia-dcgm\n",
    )
    (part,) = user_data.merge_managed_part(script, EMBEDDED).parts

    embedded = user_data.part_findings(
        part, mode="embedded", pin=None, driver_majors=[]
    )
    host = user_data.part_findings(part, mode="host", pin=PIN, driver_majors=[])

    assert embedded["contradictions"] == [
        "part 1 line 10: sudo systemctl enable --now nvidia-dcgm starts a host DCGM "
        "engine, which conflicts with the exporter's embedded engine"
    ]
    assert any("enables it too" in line for line in host["overlaps"]), (
        "host mode shares the engine start with the hand-written part"
    )


def test_adopting_the_hand_written_part_replaces_it_and_names_what_it_drops() -> None:
    merge = user_data.merge_managed_part(
        HAND_WRITTEN, EMBEDDED, adopt=["gpu-fault-node-prerequisites.sh"]
    )

    assert (merge.action, merge.adopted) == ("adopt", (1,))
    assert [name for _kind, name, _body in _parts(merge.text)] == [
        user_data.MANAGED_PART_FILENAME
    ], "the managed part takes the adopted part's place"
    (part,) = merge.parts
    assert user_data.adoption_drops(part, mode="embedded") == [
        "datacenter-gpu-manager-4-core",
        "datacenter-gpu-manager-4-cuda13",
    ]
    assert user_data.adoption_drops(part, mode="host") == [], (
        "host mode installs DCGM itself: nothing is dropped"
    )
    by_index = user_data.merge_managed_part(HAND_WRITTEN, EMBEDDED, adopt=["1"])
    assert by_index.text == merge.text
    again = user_data.merge_managed_part(
        merge.text, EMBEDDED, adopt=["gpu-fault-node-prerequisites.sh"]
    )
    assert again.action == "keep" and again.notes, "a rerun after adoption is a no-op"


@pytest.mark.parametrize(
    ("adopt", "message"),
    [
        (["9"], "names no part"),
        (["missing.sh"], "names no part"),
        (["1"], "only a hand-written shell part"),
    ],
)
def test_adoption_must_name_a_hand_written_shell_part(
    adopt: list[str], message: str
) -> None:
    current = site_user_data()

    with pytest.raises(user_data.UserDataError, match=message):
        user_data.merge_managed_part(current, EMBEDDED, adopt=adopt)


@pytest.mark.parametrize(
    ("current", "message"),
    [
        ("#!/bin/bash\ndnf install -y strace\n", "not a MIME multipart"),
        (
            'MIME-Version: 1.0\nContent-Type: multipart/mixed; boundary="B"\n\n'
            "--B\nContent-Type: text/x-shellscript\n\necho\n",
            "no closing boundary",
        ),
        (
            'Content-Type: multipart/mixed; boundary="B"\n\n--B\n'
            "Content-Type: text/x-shellscript\nContent-Disposition: attachment; "
            f'filename="{user_data.MANAGED_PART_FILENAME}"\n\necho\n--B--\n',
            "uses the managed part's file name",
        ),
    ],
)
def test_user_data_the_merge_cannot_take_apart_is_refused(
    current: str, message: str
) -> None:
    with pytest.raises(user_data.UserDataError, match=message):
        user_data.merge_managed_part(current, EMBEDDED)


def test_two_managed_parts_are_refused() -> None:
    once = user_data.merge_managed_part(site_user_data(), EMBEDDED).text
    managed = once[
        once.index(
            '--==SITE==\nContent-Type: text/x-shellscript; charset="us-ascii"\nContent-Disposition: attachment; filename="gpu-fault-managed'
        ) :
    ]
    twice = once.removesuffix("--==SITE==--\n") + managed.replace(
        user_data.MANAGED_PART_FILENAME, "copy.sh"
    )

    with pytest.raises(user_data.UserDataError, match="2 parts marked"):
        user_data.merge_managed_part(twice, EMBEDDED)


def test_oversized_compressed_or_undecodable_user_data_is_refused() -> None:
    huge = site_user_data().replace(
        "maxPods: 110\n", "maxPods: 110\n" + "# padding\n" * 2000
    )
    with pytest.raises(user_data.UserDataError, match="at most 16384"):
        user_data.merge_managed_part(huge, EMBEDDED)
    compressed = base64.b64encode(gzip.compress(b"#!/bin/bash\n")).decode()
    with pytest.raises(user_data.UserDataError, match="gzip"):
        user_data.decode_user_data(compressed)
    with pytest.raises(user_data.UserDataError, match="not base64"):
        user_data.decode_user_data("%%%")
    assert user_data.decode_user_data(None) is None
    assert user_data.decode_user_data(user_data.encode_user_data("x")) == "x"


def test_host_mode_script_pins_both_packages_in_the_driver_cuda_build() -> None:
    script = user_data.managed_script(dcgm=user_data.dcgm_pin("4.7.0-1"))

    assert "dcgm_version='4.7.0-1'" in script
    assert (
        'gpu_fault_install "datacenter-gpu-manager-4-core-${dcgm_version}" '
        '"datacenter-gpu-manager-4-cuda${cuda_major}-${dcgm_version}"'
    ) in script
    assert 'smi_report="$(nvidia-smi -q)"' in script, (
        "the CUDA major comes from the node's own driver at boot"
    )
    assert "systemctl enable --now nvidia-dcgm" in script
    assert "dcgmi discovery -l >/dev/null" in script
    assert user_data.pinned_dcgm_version(script) == "4.7.0-1"
    assert user_data.pinned_dcgm_version(EMBEDDED) is None
    assert "nvidia-dcgm" not in EMBEDDED, "embedded mode installs no DCGM"
    assert EMBEDDED.startswith("#!/bin/bash\n# " + user_data.MANAGED_PART_MARKER), (
        "the marker is the first comment line"
    )


def test_host_mode_writes_the_restart_dropin_before_it_enables_nvidia_dcgm(
    tmp_path: Path,
) -> None:
    """The package unit (Restart=on-abort, 5 starts in 10 s) leaves a non-zero
    exit or a crash loop down for good; the drop-in restarts both."""

    lines = HOST.splitlines()
    write = next(
        index for index, line in enumerate(lines) if line.startswith("printf '%s\\n'")
    )
    reload = lines.index("systemctl daemon-reload")
    enable = lines.index("systemctl enable --now nvidia-dcgm")
    install = next(
        index
        for index, line in enumerate(lines)
        if line.startswith('gpu_fault_install "datacenter-gpu-manager-4-core')
    )
    assert install < write < reload < enable, (
        "installed, then the drop-in, then systemd reloaded, then enabled and started"
    )
    assert lines[reload - 1] == f"  > {user_data.DCGM_RESTART_DROPIN}"
    # Run the rendered lines with the unit directory moved under tmp_path.
    block = "\n".join(lines[write - 1 : reload]).replace(
        "/etc/systemd/system", str(tmp_path)
    )
    subprocess.run(["bash", "-euo", "pipefail", "-c", block], check=True, timeout=30)
    written = tmp_path / "nvidia-dcgm.service.d" / "90-gpu-fault-restart.conf"
    assert written.read_text(encoding="utf-8") == user_data.DCGM_RESTART_DROPIN_TEXT
    parsed = configparser.ConfigParser()
    parsed.optionxform = str  # type: ignore[assignment,method-assign]
    parsed.read_string(user_data.DCGM_RESTART_DROPIN_TEXT)
    assert {section: dict(parsed[section]) for section in parsed.sections()} == {
        "Unit": {"StartLimitIntervalSec": "300", "StartLimitBurst": "10"},
        "Service": {"Restart": "on-failure", "RestartSec": "5s"},
    }, "on-failure: a clean systemctl stop (the gpu-fault quiesce) is not restarted"
    assert user_data.DCGM_RESTART_DROPIN not in EMBEDDED, (
        "embedded mode writes no drop-in: its managed part is unchanged"
    )


def test_the_plan_view_of_the_dropin_names_its_settings_and_reason() -> None:
    view = user_data.restart_dropin()

    assert view["path"] == (
        "/etc/systemd/system/nvidia-dcgm.service.d/90-gpu-fault-restart.conf"
    )
    assert view["content"] == user_data.DCGM_RESTART_DROPIN_TEXT
    assert view["settings"] == (
        "StartLimitIntervalSec=300 StartLimitBurst=10 Restart=on-failure RestartSec=5s"
    )
    assert "Restart=on-abort" in view["reason"] and "quiesce" in view["reason"]


def test_a_host_part_written_before_the_dropin_is_replaced_once() -> None:
    older = _without_dropin(HOST)
    current = user_data.merge_managed_part(site_user_data(), older).text

    merge = user_data.merge_managed_part(current, HOST)

    assert merge.action == "replace", "the managed part changed: it is replaced"
    assert merge.managed_script == older
    assert user_data.DCGM_RESTART_DROPIN in merge.text
    assert NODECONFIG_PART in merge.text and SITE_SCRIPT_PART in merge.text
    assert user_data.merge_managed_part(merge.text, HOST).action == "keep", (
        "a rerun over the replaced part changes nothing"
    )
    assert user_data.pinned_dcgm_version(merge.managed_script) == "4.7.0-2", (
        "the older part's pin is still read, so a rerun keeps it"
    )


def _without_dropin(script: str) -> str:
    """A host-mode managed part as it was written before the restart drop-in."""

    head, rest = script.split("# nvidia-dcgm restarts after a failure", 1)
    return head + rest[rest.index("systemctl enable --now nvidia-dcgm") :]


@pytest.mark.parametrize(
    ("value", "exact", "accepted", "refused"),
    [
        ("4.7.0-2", True, ["4.7.0-2"], ["4.7.0-1", "4.7.1-2", None]),
        ("4.7.0", False, ["4.7.0-1", "4.7.0-2"], ["4.7.1-1", "4.70-1", None]),
    ],
)
def test_a_pin_is_an_exact_build_or_any_build_of_one_version(
    value: str, exact: bool, accepted: list[str], refused: list[str | None]
) -> None:
    pin = user_data.dcgm_pin(value)

    assert pin.exact is exact
    assert all(pin.accepts(item) for item in accepted), accepted
    assert not any(pin.accepts(item) for item in refused), refused
    assert value in pin.describe()


@pytest.mark.parametrize("value", ["latest", "4.7.0-2;reboot", "4.7.0-1.el9", "", "4"])
def test_a_pin_that_is_not_a_plain_version_is_refused(value: str) -> None:
    with pytest.raises(BootstrapError, match="--dcgm-version"):
        user_data.dcgm_pin(value)


@pytest.mark.parametrize(
    ("spec", "kind"),
    [
        ("python3.12", ("python3.12", None, None)),
        ("python3.12-3.12.14", ("python3.12", None, "3.12.14")),
        ("python3.12-devel", ("other", None, None)),
        ("strace", ("strace", None, None)),
        ("datacenter-gpu-manager-4-core", ("dcgm-core", None, None)),
        ("datacenter-gpu-manager-4-core-4.7.0-1", ("dcgm-core", None, "4.7.0-1")),
        ("datacenter-gpu-manager-4-cuda13", ("dcgm-cuda", "13", None)),
        ("datacenter-gpu-manager-4-cuda12-4.7.0-2", ("dcgm-cuda", "12", "4.7.0-2")),
        ("datacenter-gpu-manager-4-devel", ("other", None, None)),
    ],
)
def test_package_specs_are_classified(
    spec: str, kind: tuple[str, str | None, str | None]
) -> None:
    assert user_data.package_kind(spec) == kind
