"""The S3 destinations a ``site.yaml`` declares and their grammar.

Two destinations share the ``s3://bucket[/prefix]`` grammar: the
control-record archive of ``spec.retention`` (written by the control-worker)
and the diagnostic evidence destination ``spec.diagnosticS3Uri`` (written by
every GPU node's Agent). The latter is the node-side declaration of a site:
the installer writes it on the node as ``GPU_FAULT_DIAGNOSTIC_S3_URI`` from
``--diagnostic-s3-uri``, the release engine hands it to the installer Job
renderer, and ``agent_config_payload`` folds it into the Agent config digest
the deploy pins -- so the digest must be computed with the same value
(``node_agent_environment``).
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from gpu_fault.admin.site_fields import SiteConfigError
from gpu_fault.admin.site_fields import optional_text as _optional_text

S3_URI_PATTERN = re.compile(
    r"^s3://(?P<bucket>[a-z0-9][a-z0-9.-]{1,61}[a-z0-9])(?P<prefix>/.*)?$"
)
#: The node Agent's diagnostic evidence destination (``node_agent/config.py``).
DIAGNOSTIC_S3_URI_ENV = "GPU_FAULT_DIAGNOSTIC_S3_URI"


def archive_s3_prefix_arn(uri: str) -> str:
    """The object ARN pattern one ``s3://bucket/prefix`` archive URI covers."""

    match = S3_URI_PATTERN.fullmatch(uri.strip())
    if match is None:
        raise SiteConfigError("archiveS3Uri must be an s3://bucket[/prefix] URI")
    prefix = (match.group("prefix") or "").strip("/")
    return f"arn:aws:s3:::{match.group('bucket')}/{prefix + '/' if prefix else ''}*"


def parse_diagnostic_s3_uri(value: object, path: str) -> str | None:
    """``spec.diagnosticS3Uri``: an ``s3://bucket[/prefix]`` URI or None.

    Where every GPU node's Agent uploads its diagnostic archives (bundles,
    DCGM quick-diagnostic output) so that a notification's evidence reference
    is an ``s3://`` object an administrator can open, instead of a ``file://``
    path readable only on the faulting node (live 2026-10-25). The bucket is
    required, the prefix optional; a trailing slash is dropped because the
    Agent joins ``<prefix>/<node>/<archive>`` itself. Absent means what every
    site ran with before: archives stay on the node.
    """

    text = _optional_text(value, path)
    if text is None:
        return None
    if S3_URI_PATTERN.fullmatch(text) is None:
        raise SiteConfigError(f"{path} must be an s3://bucket[/prefix] URI")
    return text.rstrip("/")


def site_diagnostic_s3_uri(site: Mapping[str, Any] | None) -> str | None:
    """``spec.diagnosticS3Uri`` of a raw site document, validated; None when
    the document declares none (or is no document yet)."""

    spec = site.get("spec") if isinstance(site, Mapping) else None
    if not isinstance(spec, Mapping):
        return None
    return parse_diagnostic_s3_uri(spec.get("diagnosticS3Uri"), "spec.diagnosticS3Uri")


def node_agent_environment(diagnostic_s3_uri: str | None) -> dict[str, str]:
    """The node-side variables a site declaration adds to every GPU node's
    Agent: what the Agent config digest is computed with and what the node
    installer template carries. Empty when the site declares nothing."""

    if not diagnostic_s3_uri:
        return {}
    return {DIAGNOSTIC_S3_URI_ENV: diagnostic_s3_uri}
