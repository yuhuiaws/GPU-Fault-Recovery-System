from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import yaml


WHEEL_CONFIGMAP_PLACEHOLDER = "REPLACE_WITH_WHEEL_CONFIGMAP"
DNS_SUBDOMAIN = re.compile(r"^[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?$")


def render_manifest(path: Path, wheel_configmap: str) -> str:
    if len(wheel_configmap) > 253 or DNS_SUBDOMAIN.fullmatch(wheel_configmap) is None:
        raise ValueError("wheel ConfigMap must be a Kubernetes DNS subdomain")
    text = path.read_text(encoding="utf-8")
    list(yaml.safe_load_all(text))
    count = text.count(WHEEL_CONFIGMAP_PLACEHOLDER)
    if count < 1:
        raise ValueError(f"{path} does not contain {WHEEL_CONFIGMAP_PLACEHOLDER}")
    rendered = text.replace(
        WHEEL_CONFIGMAP_PLACEHOLDER,
        wheel_configmap,
    )
    list(yaml.safe_load_all(rendered))
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a test manifest with the selected wheel ConfigMap"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--wheel-configmap", required=True)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    rendered = render_manifest(
        args.manifest,
        args.wheel_configmap,
    )
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
