from __future__ import annotations

import site
import sys
from pathlib import Path


def dependency_site_directories() -> tuple[str, ...]:
    """Include active shared dependency layers without replaying their .pth files."""
    directories: list[str] = []
    for entry in [*site.getsitepackages(), *sys.path]:
        path = Path(entry)
        if (
            not path.is_absolute()
            or path.name not in {"site-packages", "dist-packages"}
            or not path.is_dir()
        ):
            continue
        directory = str(path.resolve())
        if directory not in directories:
            directories.append(directory)
    return tuple(directories)
