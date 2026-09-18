"""Keep checked CLI and shell-helper Python in the same installed environment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def python_environment(
    environment: Mapping[str, str], *, executable: str | os.PathLike[str]
) -> dict[str, str]:
    # Resolving a venv's interpreter symlink would select the system environment.
    binary_directory = str(Path(executable).absolute().parent)
    inherited = environment.get("PATH", os.defpath).split(os.pathsep)
    result = {
        **environment,
        "PATH": os.pathsep.join(
            [
                binary_directory,
                *(part for part in inherited if part != binary_directory),
            ]
        ),
    }
    result.pop("PYTHONHOME", None)
    return result
