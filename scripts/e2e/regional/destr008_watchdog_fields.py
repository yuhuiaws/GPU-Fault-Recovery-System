"""Pure field readers shared by the DESTR-008 watchdog resource contracts.

Split out of ``destr008_watchdog_resources.py`` (architecture file-length
ratchet, 2026-09-20); nothing here touches a cluster.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeGuard

from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RegionalFixtureError("CPU watchdog resource object is malformed")
    return value


def _items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise RegionalFixtureError("CPU watchdog resource list is malformed")
    return value


def _text(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value) and value.strip() == value


def _same(actual: Any, expected: Any) -> bool:
    # JSON equality must not treat False as 0 or True as a replica/exit count.
    try:
        return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True, allow_nan=False
        )
    except (ValueError, TypeError):
        return False


def _defaults(value: dict[str, Any], defaults: dict[str, Any]) -> None:
    for key, default in defaults.items():
        if key in value and _same(value[key], default):
            del value[key]


def _image(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and bool(
        re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9./:_-]*@sha256:[0-9a-f]{64}", value)
    )


def _image_id(value: Any, image: str) -> bool:
    digest_value = image.split("@", 1)[1]
    return isinstance(value, str) and value in {
        image,
        digest_value,
        "docker-pullable://" + image,
        "containerd://" + digest_value,
        "cri-o://" + digest_value,
    }


def _status_image(value: Any, image: str) -> bool:
    # kubelet reports the planned reference or, for a digest-pinned pull with no
    # local tag (containerd 2.x), the bare image config id; ``imageID`` proves it.
    return isinstance(value, str) and (
        value == image or bool(re.fullmatch(r"sha256:[0-9a-f]{64}", value))
    )
