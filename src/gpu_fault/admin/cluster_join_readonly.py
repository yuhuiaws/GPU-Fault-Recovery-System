from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TypeVar, cast

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.site import RenderedSite

NETWORK_BASELINE_CACHE_MAX_AGE_SECONDS = 300
T = TypeVar("T")


def parallel_verify_and_discover(
    verify: Callable[[], None],
    discover: Callable[[], T],
) -> T:
    with ThreadPoolExecutor(max_workers=2) as executor:
        verify_future = executor.submit(verify)
        discovery_future = executor.submit(discover)
        verify_future.result()
        return discovery_future.result()


def load_network_baseline_cache(
    path: Path,
    site: RenderedSite,
) -> list[dict[str, object]] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        observed = float(value["observed_at_epoch"])
        networks = value["networks"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        value.get("schema_version") != 1
        or value.get("site_sha256") != site.source_sha256
        or time.time() - observed > NETWORK_BASELINE_CACHE_MAX_AGE_SECONDS
        or not isinstance(networks, list)
        or any(not isinstance(item, dict) for item in networks)
    ):
        return None
    return [cast(dict[str, object], dict(item)) for item in networks]


def write_network_baseline_cache(
    path: Path,
    site: RenderedSite,
    networks: list[dict[str, object]],
) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "site_sha256": site.source_sha256,
            "observed_at_epoch": time.time(),
            "networks": networks,
        },
    )


def cached_network_baseline(
    path: Path,
    site: RenderedSite,
    discover: Callable[[], list[dict[str, object]]],
) -> list[dict[str, object]]:
    cached = load_network_baseline_cache(path, site)
    if cached is not None:
        return cached
    networks = discover()
    write_network_baseline_cache(path, site, networks)
    return networks
