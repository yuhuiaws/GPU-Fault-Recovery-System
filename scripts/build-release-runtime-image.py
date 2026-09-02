from __future__ import annotations

import argparse
import json
from pathlib import Path

from release_image import build_runtime_image


ROOT = Path(__file__).resolve().parents[1]


def build_arguments(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, argument = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError(
                "runtime image build args must be unique NAME=VALUE entries"
            )
        result[name] = argument
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--build-arg", action="append", default=[])
    parser.add_argument("--cache-from", action="append", default=[])
    parser.add_argument("--cache-to", action="append", default=[])
    parser.add_argument("--component-artifacts", type=Path)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist/release-runtime-image.json",
    )
    arguments = parser.parse_args()
    descriptor = build_runtime_image(
        ROOT,
        repository=arguments.repository,
        platform=arguments.platform,
        push=arguments.push,
        build_args=build_arguments(arguments.build_arg),
        cache_from=tuple(arguments.cache_from),
        cache_to=tuple(arguments.cache_to),
        component_artifacts=(
            arguments.component_artifacts.resolve()
            if arguments.component_artifacts is not None
            else None
        ),
        reuse_registry_image=not arguments.force_rebuild,
    )
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
