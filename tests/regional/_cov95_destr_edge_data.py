from __future__ import annotations

from copy import deepcopy
from typing import Any


def change(document: Any, path: tuple[str | int, ...], value: Any) -> None:
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = deepcopy(value)
