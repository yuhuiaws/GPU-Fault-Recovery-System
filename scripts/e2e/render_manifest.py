from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import yaml


WHEEL_CONFIGMAP_PLACEHOLDER = "REPLACE_WITH_WHEEL_CONFIGMAP"
RUNTIME_PROFILE_PLACEHOLDER = "REPLACE_WITH_RUNTIME_PROFILE_VERSION"
DNS_SUBDOMAIN = re.compile(r"^[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?$")
PROFILE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def render_manifest(
    path: Path,
    wheel_configmap: str | None = None,
    runtime_profile: str | None = None,
) -> str:
    if wheel_configmap is not None and (
        len(wheel_configmap) > 253 or DNS_SUBDOMAIN.fullmatch(wheel_configmap) is None
    ):
        raise ValueError("wheel ConfigMap must be a Kubernetes DNS subdomain")
    if (
        runtime_profile is not None
        and PROFILE_VERSION.fullmatch(runtime_profile) is None
    ):
        raise ValueError("Runtime Profile version has an invalid format")
    text = path.read_text(encoding="utf-8")
    list(yaml.safe_load_all(text))
    placeholders = {
        WHEEL_CONFIGMAP_PLACEHOLDER: wheel_configmap,
        RUNTIME_PROFILE_PLACEHOLDER: runtime_profile,
    }
    present = [placeholder for placeholder in placeholders if placeholder in text]
    if not present:
        raise ValueError(f"{path} contains no supported E2E placeholders")
    missing = [
        placeholder for placeholder in present if placeholders[placeholder] is None
    ]
    if missing:
        raise ValueError("missing values for placeholders: " + ", ".join(missing))
    rendered = text
    for placeholder in present:
        rendered = rendered.replace(placeholder, str(placeholders[placeholder]))
    list(yaml.safe_load_all(rendered))
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a test manifest with approved runtime inputs"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--wheel-configmap")
    parser.add_argument("--runtime-profile")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    rendered = render_manifest(
        args.manifest,
        args.wheel_configmap,
        args.runtime_profile,
    )
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
