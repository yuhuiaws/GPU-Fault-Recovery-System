"""Bounded deployment diagnostics that retain causes without echoing credentials."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from gpu_fault.admin.process_supervisor import write_diagnostic

_PRIVATE_KEY = re.compile(
    r"-----BEGIN [^-]*PRIVATE KEY-----.*?(?:-----END [^-]*PRIVATE KEY-----|$)",
    re.S,
)
_URL_CREDENTIALS = re.compile(
    r"(?<![A-Za-z0-9+.-])([A-Za-z][A-Za-z0-9+.-]*://)[^\s/@]+:[^\s/@]+@"
)
# Retry only at field boundaries, not at every character of a long non-key token.
_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?<![a-z0-9_-])
    (?P<key>["']?(?:[a-z0-9_-]*(?:password|token|secret|authorization|private_key|
    access_key|credential|signature)[a-z0-9_-]*)["']?\s*[:=]\s*)
    (?:"[^"]*"|'[^']*'|[^\s,;&}]+)
    """
)
_BEARER = re.compile(r"(?i)\b(?:Bearer|Basic)\s+[A-Za-z0-9+/=._~-]+")
_ACCESS_ID = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SENSITIVE = re.compile(
    r"(?i)password|token|secret|authorization|private.?key|access.?key|credential"
)
_ERROR_CODES = frozenset(
    {
        "AccessDenied",
        "AccessDeniedException",
        "Forbidden",
        "Unauthorized",
        "InvalidClientTokenId",
        "ExpiredToken",
        "ExpiredTokenException",
        "ResourceNotFoundException",
        "ResourceNotFound",
        "NotFound",
        "NoSuchEntity",
        "NoSuchEntityException",
        "TooManyRequestsException",
        "Throttling",
        "ThrottlingException",
        "InternalError",
        "InternalFailure",
        "ServiceUnavailable",
        "InvalidParameter",
        "ValidationError",
    }
)
_TRANSPORT_CODES = {
    "Unable to connect": r"(?i)unable to connect|connection refused|could not connect",
    "RequestTimeout": r"(?i)i/o timeout|request timed out|context deadline exceeded",
    "TLSVerificationFailed": r"(?i)certificate verify failed|x509:|tls handshake",
}


def _redact_value(value: object) -> object:
    if isinstance(value, dict):
        secret = bool(_SENSITIVE.search(str(value.get("name", ""))))
        return {
            str(key): (
                "<redacted>"
                if _SENSITIVE.search(str(key))
                or str(key) in {"data", "stringData", "binaryData"}
                or (secret and key == "value")
                else _redact_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return redact_text(value) if isinstance(value, str) else value


def redact_text(text: str) -> str:
    text = _PRIVATE_KEY.sub("<redacted private key>", text)
    text = _URL_CREDENTIALS.sub(r"\1<redacted>@", text)
    text = _BEARER.sub("<redacted authorization>", text)
    text = _ASSIGNMENT.sub(lambda match: match.group("key") + '"<redacted>"', text)
    return _ACCESS_ID.sub("<redacted access key>", _ANSI.sub("", text))


def diagnostic_text(
    text: str | None, *, sensitive: bool = False, limit: int = 4096
) -> str:
    if sensitive:
        codes = sorted(
            code
            for code in _ERROR_CODES
            if re.search(rf"\b{re.escape(code)}\b", text or "")
        )
        codes.extend(
            name
            for name, pattern in _TRANSPORT_CODES.items()
            if re.search(pattern, text or "")
        )
        return (
            ", ".join(codes) + ": <sensitive output redacted>"
            if codes
            else "<sensitive output redacted>"
            if text
            else ""
        )
    raw = (text or "").strip()
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        safe = redact_text(raw)
    else:
        safe = json.dumps(_redact_value(value), ensure_ascii=True, sort_keys=True)
    safe = "".join(
        character for character in safe if character in "\n\t" or ord(character) >= 32
    )
    return safe if len(safe) <= limit else "[earlier output omitted]\n" + safe[-limit:]


def diagnostic_command(arguments: Sequence[str], *, sensitive: bool = False) -> str:
    if not arguments:
        return "<no command>"
    if sensitive:
        return arguments[0]
    result: list[str] = []
    redact_next = False
    for argument in arguments:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
        else:
            result.append(diagnostic_text(argument))
            redact_next = (
                argument.startswith("-")
                and bool(_SENSITIVE.search(argument))
                and "=" not in argument
            )
    return " ".join(result)


_PROGRESS = re.compile(
    r"(?:"
    r"release-(?:begin|end|phase) [0-9TZ:-]+ "
    r"[a-zA-Z0-9_=/.-]+(?: (?:mode|dry_run|exit_code|total|elapsed|lifecycle|clusters)="
    r"[a-zA-Z0-9_./-]+)*"
    r"|bootstrap task=[a-zA-Z0-9_:./-]+ (?:start|cache-hit|"
    r"failure_type=[a-zA-Z_][a-zA-Z0-9_]*|"
    r"[a-z+]+ status=[a-z]+ duration=[0-9.]+s runner_commands=[0-9]+)"
    r"|deployment-wait command=[a-z0-9_.-]+ state=(?:running|cleanup) "
    r"elapsed=[0-9]+s remaining=[0-9]+s"
    r")"
)


class DriverDiagnostics:
    """Stream only structured progress; redact other output as a complete document.

    Redacting arbitrary stderr line by line loses JSON and multiline credential
    context. Non-progress diagnostics retain the existing whole-document redactor;
    oversized output falls back to safe error categories, not an untrusted tail.
    """

    maximum = 1024 * 1024

    def __init__(self) -> None:
        self._pending = ""
        self._parts: list[str] = []
        self._size = 0
        self._overflow = False
        self._discard_line = False

    def feed(self, text: str) -> None:
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            if not self._discard_line:
                self._line(line)
            self._discard_line = False
        if len(self._pending) > self.maximum:
            self._overflow = True
            self._discard_line = True
            self._pending = ""

    def _line(self, line: str) -> None:
        if _PROGRESS.fullmatch(line):
            if write_diagnostic(diagnostic_text(line) + "\n"):
                return
        if self._size + len(line) + 1 > self.maximum:
            self._overflow = True
            return
        self._parts.append(line + "\n")
        self._size += len(line) + 1

    def finish(self) -> None:
        if self._pending and not self._discard_line:
            self._line(self._pending)
        text = diagnostic_text("".join(self._parts), sensitive=self._overflow)
        if text:
            write_diagnostic(text + "\n", final=True)
        if self._overflow:
            write_diagnostic(
                "deployment diagnostics exceeded their safe size limit\n", final=True
            )
        self._pending = ""
        self._parts.clear()
