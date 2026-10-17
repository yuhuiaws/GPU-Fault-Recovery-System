"""The gpu-fault part of a plain EKS GPU node group's launch template user data.

The ``AL2023_x86_64_NVIDIA`` AMI ships the NVIDIA driver but neither
``/usr/bin/python3.12`` nor ``strace``, which the node runtime needs; they are
installed at first boot by a ``text/x-shellscript`` part of the managed node
group's launch template user data (administrator operations §5.0.1). AL2023
user data is a MIME multipart document: EKS merges its own ``NodeConfig`` part
into a managed node group's copy of the template, and a site may keep its own
scripts and ``NodeConfig`` parts beside ours.

``gpu-fault-admin prepare-eks-nodegroup`` owns exactly one part: the shell part
carrying :data:`MANAGED_PART_MARKER` on a comment line. Merging keeps every
other part byte for byte -- the document is split on its boundary lines rather
than re-serialized by the ``email`` package, which would re-fold headers and
rewrite line endings -- and replaces or appends only the managed part, so a
second run over the result changes nothing. A hand-written shell part is never
dropped or rewritten on its own: the plan lists its package lines with what
they share with or contradict in the managed part (:func:`part_findings`),
and only an explicit ``--adopt-user-data-part`` replaces it, naming what the
replacement drops (:func:`adoption_drops`).

The optional DCGM host engine mode (:data:`DCGM_HOSTENGINE_LABEL` =
:data:`DCGM_HOSTENGINE_HOST` on the node group) adds the DCGM 4 packages to the
same part, pinned (:class:`DcgmPin`) because unpinned installs drifted between
nodes, in the CUDA build matching the driver's ``nvidia-smi`` CUDA major (read
on the node at boot), and enables ``nvidia-dcgm``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from email.parser import HeaderParser
from typing import Any

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault_release.regional_release_rendering import DCGM_HOSTENGINE_ANNOTATION

__all__ = [
    "DCGM_HOSTENGINE_EMBEDDED",
    "DCGM_HOSTENGINE_HOST",
    "DCGM_HOSTENGINE_LABEL",
    "DCGM_PACKAGE_VERSION",
    "MANAGED_PART_FILENAME",
    "MANAGED_PART_MARKER",
    "DcgmPin",
    "PartReport",
    "UserDataError",
    "UserDataMerge",
    "adoption_drops",
    "dcgm_pin",
    "decode_user_data",
    "encode_user_data",
    "existing_managed_script",
    "managed_script",
    "merge_managed_part",
    "package_kind",
    "part_findings",
    "pinned_dcgm_version",
]

#: The comment line that identifies the one part this verb owns.
MANAGED_PART_MARKER = "gpu-fault.io/user-data-part: node-prerequisites"
#: cloud-init stores a script part under its file name: a second part with the
#: same name would overwrite ours, so a foreign part using it is refused.
MANAGED_PART_FILENAME = "gpu-fault-managed-node-prerequisites.sh"
MANAGED_PART_CONTENT_TYPE = 'text/x-shellscript; charset="us-ascii"'
#: The boundary of a document the verb creates from no user data at all.
DEFAULT_BOUNDARY = "==GPU-FAULT-USER-DATA=="
#: EC2 limits user data to 16 KiB before base64.
USER_DATA_LIMIT_BYTES = 16 * 1024
NODE_PACKAGES = ("python3.12", "strace")

#: The node group label declaring the DCGM host engine mode, and its value:
#: the one definition every reader of the declaration points at. The key is
#: the one the exporter DaemonSet already carries as its variant annotation.
DCGM_HOSTENGINE_LABEL = DCGM_HOSTENGINE_ANNOTATION
DCGM_HOSTENGINE_HOST = "host"
#: The mode of a node group without the label: the exporter's own engine.
DCGM_HOSTENGINE_EMBEDDED = "embedded"
#: The DCGM 4 build host mode installs unless ``--dcgm-version`` names another.
DCGM_PACKAGE_VERSION = "4.7.0-2"
DCGM_CORE_PACKAGE = "datacenter-gpu-manager-4-core"
DCGM_CUDA_PACKAGE_PREFIX = "datacenter-gpu-manager-4-cuda"
DCGM_SERVICE = "nvidia-dcgm"
_DCGM_PIN = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}(?:-[0-9]+)?$")
_PINNED_LINE = re.compile(r"^dcgm_version='(?P<version>[^']*)'$", re.MULTILINE)
_CORE_SPEC = re.compile(r"^datacenter-gpu-manager-4-core(?:-(?P<version>[0-9]\S*))?$")
_CUDA_SPEC = re.compile(
    r"^datacenter-gpu-manager-4-cuda(?P<major>[0-9]+)(?:-(?P<version>[0-9]\S*))?$"
)
_GZIP_MAGIC = b"\x1f\x8b"
_INSTALLERS = frozenset({"dnf", "yum", "microdnf", "dnf5"})
_OPERATORS = frozenset({";", "&&", "||", "|", "&", "(", ")", ";;"})
_PREFIXES = frozenset(
    {"if", "then", "else", "elif", "do", "while", "until", "!", "time", "sudo", "exec"}
)


class UserDataError(BootstrapError):
    """The launch template's user data cannot be merged safely by the verb."""


@dataclass(frozen=True)
class DcgmPin:
    """The DCGM 4 packages host mode installs: one build, or one version.

    ``4.7.0-2`` (version-release) installs and accepts exactly that build;
    ``4.7.0`` (version) lets dnf install the newest build of 4.7.0 at boot and
    accepts any installed build of it (``4.7.0-1`` and ``4.7.0-2`` alike), so
    nodes created at different times may differ in build -- the verification
    reports that drift.
    """

    value: str

    @property
    def exact(self) -> bool:
        return "-" in self.value

    def accepts(self, installed: str | None) -> bool:
        if not installed:
            return False
        if self.exact:
            return installed == self.value
        return installed.split("-", 1)[0] == self.value

    def describe(self) -> str:
        if self.exact:
            return f"exactly build {self.value} of both DCGM packages"
        return (
            f"version {self.value}, any build: dnf installs the newest {self.value} "
            "build available when the node boots"
        )


def dcgm_pin(value: str) -> DcgmPin:
    """``value`` as a pin when it is a plain ``version[-release]`` (it lands in
    a script)."""

    if not _DCGM_PIN.fullmatch(value or ""):
        raise BootstrapError(
            "--dcgm-version must be a DCGM 4 version (4.7.0: any build of it) or "
            f"version-release (4.7.0-2: that build); got {value!r}"
        )
    return DcgmPin(value)


def _cuda_major_lines() -> list[str]:
    # ``nvidia-smi -q`` is read whole before parsing: an early-exiting reader
    # (``grep -m1``, ``head``) would SIGPIPE nvidia-smi and fail ``pipefail``.
    return [
        'smi_report="$(nvidia-smi -q)"',
        'cuda_major="$(printf \'%s\\n\' "${smi_report}" | '
        "sed -n 's/^ *CUDA Version *: *\\([0-9][0-9]*\\)\\..*$/\\1/p')\"",
        "cuda_major=\"${cuda_major%%$'\\n'*}\"",
        'if [ -z "${cuda_major}" ]; then',
        '  echo "gpu-fault: nvidia-smi reports no CUDA version" >&2',
        "  exit 1",
        "fi",
    ]


def managed_script(*, dcgm: DcgmPin | None = None) -> str:
    """The shell script of the managed part; DCGM host lines when pinned."""

    lines = [
        "#!/bin/bash",
        f"# {MANAGED_PART_MARKER} v1",
        "# Written by gpu-fault-admin prepare-eks-nodegroup and replaced as a whole",
        "# when its content changes: keep site additions in a part of their own.",
        "# The gpu-fault node runtime needs /usr/bin/python3.12 (venv) and strace;",
        "# the AL2023_x86_64_NVIDIA AMI ships neither.",
        "set -euo pipefail",
        "gpu_fault_install() {",
        "  local attempt",
        "  for attempt in 1 2 3 4 5; do",
        '    if dnf install -y "$@"; then',
        "      return 0",
        "    fi",
        '    echo "gpu-fault: dnf install $* failed (attempt ${attempt} of 5)" >&2',
        "    sleep 30",
        "  done",
        "  return 1",
        "}",
        "gpu_fault_install " + " ".join(NODE_PACKAGES),
        "/usr/bin/python3.12 -c 'import venv'",
        "command -v strace",
    ]
    if dcgm is not None:
        version = dcgm_pin(dcgm.value).value
        lines += [
            f"# DCGM host engine ({DCGM_HOSTENGINE_LABEL}={DCGM_HOSTENGINE_HOST}):",
            "# DCGM 4 pinned, in the CUDA build matching the driver's CUDA major",
            '# (nvidia-smi "CUDA Version").',
            f"dcgm_version='{version}'",
            *_cuda_major_lines(),
            f'gpu_fault_install "{DCGM_CORE_PACKAGE}-${{dcgm_version}}" '
            f'"{DCGM_CUDA_PACKAGE_PREFIX}${{cuda_major}}-${{dcgm_version}}"',
            f"systemctl enable --now {DCGM_SERVICE}",
            "dcgmi discovery -l >/dev/null",
        ]
    return "\n".join(lines) + "\n"


def pinned_dcgm_version(script: str | None) -> str | None:
    """The DCGM pin a managed script carries, ``None`` without one."""

    match = _PINNED_LINE.search(script or "")
    return match.group("version") if match else None


# --- what a part installs -----------------------------------------------------


def package_kind(spec: str) -> tuple[str, str | None, str | None]:
    """``(kind, CUDA major, version)`` of one package spec of an install line.

    Kinds: ``python3.12``, ``strace``, ``dcgm-core``, ``dcgm-cuda`` or
    ``other``; the version is the spec's ``version[-release]`` suffix, ``None``
    for an unpinned spec.
    """

    for name in NODE_PACKAGES:
        if spec == name or (
            spec.startswith(name + "-") and spec[len(name) + 1 :][:1].isdigit()
        ):
            return name, None, spec[len(name) + 1 :] or None
    core = _CORE_SPEC.fullmatch(spec)
    if core:
        return "dcgm-core", None, core.group("version")
    cuda = _CUDA_SPEC.fullmatch(spec)
    if cuda:
        return "dcgm-cuda", cuda.group("major"), cuda.group("version")
    return "other", None, None


@dataclass(frozen=True)
class ScriptLine:
    number: int
    text: str
    packages: tuple[str, ...] = ()


def _words(line: str) -> list[str]:
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        return list(lexer)
    except ValueError:  # an unbalanced quote: plain words are good enough here
        return line.split("#", 1)[0].split()


def _command(words: list[str]) -> list[str]:
    """``words`` without the shell keywords and wrappers in front of the command
    (``if``, ``then``, ``!``, ``sudo``, ``timeout 600``, ``VAR=value``)."""

    while words:
        head = words[0]
        if head in _PREFIXES or ("=" in head and not head.startswith("-")):
            words = words[1:]
        elif head == "timeout":
            words = words[2:]
        else:
            break
    return words


def _commands(words: Sequence[str]) -> list[list[str]]:
    commands: list[list[str]] = [[]]
    for word in words:
        if word in _OPERATORS:
            commands.append([])
        else:
            commands[-1].append(word)
    return [command for item in commands if (command := _command(item))]


def _scan(body: str) -> tuple[tuple[ScriptLine, ...], tuple[ScriptLine, ...]]:
    """The package install lines and the host engine start lines of a script."""

    installs: list[ScriptLine] = []
    engines: list[ScriptLine] = []
    for number, line in enumerate(body.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        for command in _commands(_words(line)):
            head = command[0].rsplit("/", 1)[-1]
            if head in _INSTALLERS and "install" in command:
                rest = command[command.index("install") + 1 :]
                packages = tuple(word for word in rest if not word.startswith("-"))
                if packages:
                    installs.append(ScriptLine(number, line.strip(), packages))
            elif head == "nv-hostengine" or (
                head == "systemctl"
                and {"enable", "start", "restart"} & set(command)
                and any(word.split(".", 1)[0] == DCGM_SERVICE for word in command)
            ):
                engines.append(ScriptLine(number, line.strip()))
    return tuple(installs), tuple(engines)


@dataclass(frozen=True)
class PartReport:
    """One part of the current user data, with what its script installs."""

    index: int
    content_type: str
    filename: str
    managed: bool
    sha256: str
    install_lines: tuple[ScriptLine, ...] = ()
    engine_lines: tuple[ScriptLine, ...] = ()

    @property
    def shell(self) -> bool:
        return self.content_type == "text/x-shellscript"

    def describe(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "index": self.index,
            "content_type": self.content_type,
            "filename": self.filename or None,
            "managed": self.managed,
            "sha256": self.sha256,
        }
        if self.shell and not self.managed:
            report["install_lines"] = [
                {
                    "line": item.number,
                    "text": item.text,
                    "packages": list(item.packages),
                }
                for item in self.install_lines
            ]
            report["host_engine_lines"] = [
                {"line": item.number, "text": item.text} for item in self.engine_lines
            ]
        return report


def part_findings(
    part: PartReport,
    *,
    mode: str,
    pin: DcgmPin | None,
    driver_majors: Sequence[str],
) -> dict[str, list[str]]:
    """What a hand-written part shares with, or contradicts in, the managed part.

    ``driver_majors`` are the CUDA majors the current nodes' drivers report
    (empty when they could not be read): a DCGM CUDA build of another major
    contradicts the driver whatever the mode.
    """

    overlaps: list[str] = []
    contradictions: list[str] = []
    others: list[str] = []
    for line in part.install_lines:
        where = f"part {part.index} line {line.number}"
        for spec in line.packages:
            kind, major, version = package_kind(spec)
            if kind in NODE_PACKAGES:
                overlaps.append(f"{where}: {spec} (the managed part installs it too)")
                continue
            if kind == "other":
                others.append(f"{where}: {spec}")
                continue
            if mode == DCGM_HOSTENGINE_HOST and pin is not None:
                overlaps.append(
                    f"{where}: {spec} (the managed part installs DCGM {pin.value})"
                )
                if version is None:
                    contradictions.append(
                        f"{where}: {spec} is unpinned (the newest build), the pin is "
                        f"{pin.value}"
                    )
                elif not pin.accepts(version):
                    contradictions.append(
                        f"{where}: {spec} installs {version}, the pin is {pin.value}"
                    )
            else:
                others.append(f"{where}: {spec}")
            if kind == "dcgm-cuda" and driver_majors and major not in driver_majors:
                contradictions.append(
                    f"{where}: {spec} is the CUDA {major} build; the nodes' driver "
                    f"reports CUDA {', '.join(sorted(set(driver_majors)))}"
                )
    for line in part.engine_lines:
        where = f"part {part.index} line {line.number}"
        if mode == DCGM_HOSTENGINE_HOST:
            overlaps.append(f"{where}: {line.text} (the managed part enables it too)")
        else:
            contradictions.append(
                f"{where}: {line.text} starts a host DCGM engine, which conflicts "
                "with the exporter's embedded engine"
            )
    return {
        "overlaps": overlaps,
        "contradictions": contradictions,
        "other_packages": others,
    }


def adoption_drops(part: PartReport, *, mode: str) -> list[str]:
    """The package specs an adopted part installs that the managed part does not."""

    kept = set(NODE_PACKAGES)
    if mode == DCGM_HOSTENGINE_HOST:
        kept |= {"dcgm-core", "dcgm-cuda"}
    return [
        spec
        for line in part.install_lines
        for spec in line.packages
        if package_kind(spec)[0] not in kept
    ]


# --- the document ---------------------------------------------------------------


@dataclass(frozen=True)
class _Part:
    """One part, kept as the exact text between its delimiter lines."""

    delimiter: str
    raw: str
    content_type: str
    filename: str
    managed: bool


@dataclass(frozen=True)
class _Document:
    head: str
    parts: tuple[_Part, ...]
    close: str
    tail: str
    boundary: str
    newline: str

    def render(self, parts: Sequence[_Part]) -> str:
        return (
            self.head
            + "".join(item.delimiter + item.raw for item in parts)
            + self.close
            + self.tail
        )


def _header_block(text: str) -> tuple[str, str]:
    """``(headers, rest)`` split at the first empty line."""

    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip("\r\n"):
            return "".join(lines[:index]), "".join(lines[index:])
    return "".join(lines), ""


def _body(raw: str) -> str:
    _headers, rest = _header_block(raw)
    return rest.split("\n", 1)[1] if "\n" in rest else ""


def _part(delimiter: str, raw: str) -> _Part:
    headers, body = _header_block(raw)
    message = HeaderParser().parsestr(headers)
    content_type = message.get_content_type()
    managed = content_type == "text/x-shellscript" and any(
        line.strip().startswith(f"# {MANAGED_PART_MARKER}")
        for line in body.splitlines()
    )
    return _Part(
        delimiter=delimiter,
        raw=raw,
        content_type=content_type,
        filename=str(message.get_filename() or ""),
        managed=managed,
    )


def _report(index: int, item: _Part) -> PartReport:
    installs, engines = (
        _scan(_body(item.raw))
        if item.content_type == "text/x-shellscript" and not item.managed
        else ((), ())
    )
    return PartReport(
        index=index,
        content_type=item.content_type,
        filename=item.filename,
        managed=item.managed,
        sha256=hashlib.sha256(item.raw.encode("utf-8")).hexdigest(),
        install_lines=installs,
        engine_lines=engines,
    )


def _parse(text: str) -> _Document:
    headers, _rest = _header_block(text)
    message = HeaderParser().parsestr(headers)
    boundary = message.get_boundary()
    if not headers.strip() or message.get_content_maintype() != "multipart":
        raise UserDataError(
            "the launch template's user data is not a MIME multipart document "
            "(a managed node group on an EKS AMI takes multipart user data only)"
        )
    if not boundary:
        raise UserDataError("the user data's multipart Content-Type has no boundary")
    newline = "\r\n" if "\r\n" in text else "\n"
    delimiter, close = f"--{boundary}", f"--{boundary}--"
    head: list[str] = []
    parts: list[tuple[str, list[str]]] = []
    closing = ""
    tail: list[str] = []
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n").rstrip(" \t")
        if closing:
            tail.append(line)
        elif bare == close:
            closing = line
        elif bare == delimiter:
            parts.append((line, []))
        elif parts:
            parts[-1][1].append(line)
        else:
            head.append(line)
    if not closing:
        raise UserDataError(
            f"the user data has no closing boundary line {close}; merge it by hand"
        )
    return _Document(
        head="".join(head),
        parts=tuple(_part(line, "".join(body)) for line, body in parts),
        close=closing,
        tail="".join(tail),
        boundary=boundary,
        newline=newline,
    )


def _managed_raw(script: str, newline: str) -> str:
    headers = [
        f"Content-Type: {MANAGED_PART_CONTENT_TYPE}",
        f'Content-Disposition: attachment; filename="{MANAGED_PART_FILENAME}"',
        "",
    ]
    return newline.join(headers) + newline + script.replace("\n", newline)


@dataclass(frozen=True)
class UserDataMerge:
    """The merged document and what the merge did to the current one."""

    #: ``keep`` (already current), ``replace`` (an outdated managed part),
    #: ``adopt`` (the managed part takes an adopted hand-written part's place),
    #: ``add`` (appended beside the existing parts), ``create`` (no user data).
    action: str
    text: str
    parts: tuple[PartReport, ...]
    #: The current managed part's script, ``None`` when there is none.
    managed_script: str | None
    #: Indices of the hand-written parts the managed part replaces.
    adopted: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.action != "keep"

    def summary(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "parts": [item.describe() for item in self.parts],
            "adopted_parts": list(self.adopted),
            "notes": list(self.notes),
            "merged_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "merged_bytes": len(self.text.encode("utf-8")),
        }


def _limited(merge: UserDataMerge) -> UserDataMerge:
    size = len(merge.text.encode("utf-8"))
    if size > USER_DATA_LIMIT_BYTES:
        raise UserDataError(
            f"the merged user data is {size} bytes; EC2 accepts at most "
            f"{USER_DATA_LIMIT_BYTES} before base64"
        )
    return merge


def existing_managed_script(current: str | None) -> str | None:
    """The managed part's script in ``current`` user data, ``None`` without one."""

    if not current:
        return None
    managed = [item for item in _parse(current).parts if item.managed]
    if len(managed) != 1:
        return None
    return _body(managed[0].raw).replace("\r\n", "\n")


def _adopted(
    document: _Document, references: Sequence[str]
) -> tuple[list[int], list[str]]:
    """The 0-based indices ``--adopt-user-data-part`` names, and notes.

    A reference is a 1-based part index or a part's file name and must name a
    hand-written shell part. One that names nothing, or the managed part,
    once the managed part is present, is a no-op (adopted by an earlier run).
    """

    chosen: list[int] = []
    notes: list[str] = []
    has_managed = any(item.managed for item in document.parts)
    for reference in references:
        matches = [
            index
            for index, item in enumerate(document.parts)
            if (reference.isdigit() and int(reference) == index + 1)
            or (not reference.isdigit() and item.filename == reference)
        ]
        if not matches or document.parts[matches[0]].managed:
            if has_managed:
                notes.append(
                    f"--adopt-user-data-part {reference}: nothing to adopt (the "
                    "managed part is in place)"
                )
                continue
            raise UserDataError(
                f"--adopt-user-data-part {reference} names no part of the user data"
            )
        for index in matches:
            if document.parts[index].content_type != "text/x-shellscript":
                raise UserDataError(
                    f"--adopt-user-data-part {reference} names a "
                    f"{document.parts[index].content_type} part; only a hand-written "
                    "shell part can be adopted"
                )
            if index not in chosen:
                chosen.append(index)
    return sorted(chosen), notes


def merge_managed_part(
    current: str | None, script: str, *, adopt: Sequence[str] = ()
) -> UserDataMerge:
    """Put ``script`` into ``current`` as the one managed part.

    Every other part is kept byte for byte, hand-written shell parts included,
    unless ``adopt`` names them: then the managed part takes the first adopted
    part's place and the others go. A document with two managed parts, or a
    foreign part using the managed part's file name, is refused: which one is
    "ours" would be a guess.
    """

    if not current:
        if adopt:
            raise UserDataError("--adopt-user-data-part: the user data is empty")
        newline = "\n"
        text = newline.join(
            (
                "MIME-Version: 1.0",
                f'Content-Type: multipart/mixed; boundary="{DEFAULT_BOUNDARY}"',
                "",
                f"--{DEFAULT_BOUNDARY}",
                _managed_raw(script, newline) + f"--{DEFAULT_BOUNDARY}--",
                "",
            )
        )
        return _limited(UserDataMerge("create", text, (), None))
    document = _parse(current)
    reports = tuple(
        _report(index, item) for index, item in enumerate(document.parts, 1)
    )
    managed = [index for index, item in enumerate(document.parts) if item.managed]
    if len(managed) > 1:
        raise UserDataError(
            f"the user data carries {len(managed)} parts marked "
            f"{MANAGED_PART_MARKER}; keep one and rerun"
        )
    if any(
        not item.managed and item.filename == MANAGED_PART_FILENAME
        for item in document.parts
    ):
        raise UserDataError(
            f"an unmanaged user data part uses the managed part's file name "
            f"{MANAGED_PART_FILENAME} (cloud-init would overwrite one with the "
            "other); rename it and rerun"
        )
    adopted, notes = _adopted(document, adopt)
    desired = _managed_raw(script, document.newline)
    existing = (
        _body(document.parts[managed[0]].raw).replace("\r\n", "\n") if managed else None
    )
    if managed and not adopted and document.parts[managed[0]].raw == desired:
        return _limited(
            UserDataMerge("keep", current, reports, existing, notes=tuple(notes))
        )
    slot = managed[0] if managed else (adopted[0] if adopted else None)
    parts: list[_Part] = []
    for index, item in enumerate(document.parts):
        if index == slot:
            parts.append(
                _Part(
                    item.delimiter,
                    desired,
                    "text/x-shellscript",
                    MANAGED_PART_FILENAME,
                    True,
                )
            )
        elif index not in adopted:
            parts.append(item)
    if slot is None:
        delimiter = f"--{document.boundary}{document.newline}"
        last = document.parts[-1].raw if document.parts else document.head
        if last and not last.endswith(("\n", "\r")):
            # The closing delimiter must start a line of its own.
            delimiter = document.newline + delimiter
        parts.append(
            _Part(delimiter, desired, "text/x-shellscript", MANAGED_PART_FILENAME, True)
        )
    action = "adopt" if adopted else ("replace" if managed else "add")
    return _limited(
        UserDataMerge(
            action,
            document.render(parts),
            reports,
            existing,
            adopted=tuple(index + 1 for index in adopted),
            notes=tuple(notes),
        )
    )


def decode_user_data(value: object) -> str | None:
    """The text of a launch template version's base64 ``UserData``."""

    if value in (None, ""):
        return None
    try:
        raw = base64.b64decode(str(value), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UserDataError("the launch template's UserData is not base64") from exc
    if raw.startswith(_GZIP_MAGIC):
        raise UserDataError(
            "the launch template's user data is gzip-compressed; merge it by hand"
        )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UserDataError("the launch template's user data is not UTF-8") from exc


def encode_user_data(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")
