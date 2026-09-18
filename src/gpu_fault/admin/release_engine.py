"""Construct the existing regional engine without importing a lifecycle command."""

from __future__ import annotations

from typing import Any

from gpu_fault.admin.site import RenderedSite, materialized_release_config
from gpu_fault_release.regional_release_config import ReleaseConfig
from gpu_fault_release.rollout import RegionalRelease, Runner


def build_release(site: RenderedSite) -> Any:
    with materialized_release_config(site) as config_path:
        config = ReleaseConfig.load(config_path)
    return RegionalRelease(config, Runner())
