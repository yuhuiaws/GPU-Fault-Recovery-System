"""One file naming the live site a regional acceptance case runs against.

The committed runners must not contain site topology -- account IDs, kubeconfig
paths, EKS contexts, node names -- and `test_collector_promoted_scripts_contain_
no_site_specific_topology` enforces that. The values still have to come from
somewhere, and for a while they came from a hand-written shell launcher in
`/tmp`: not replayable, lost on reboot, and edited in place whenever a window
moved. A case's evidence could then say what happened without saying which node
it happened to.

A site profile is that launcher's contents as a file the operator keeps outside
the repository (the private state directory is the natural home). Precedence is
the same one the runners already use for their environment fallbacks: an
explicit command-line flag wins, then the profile, then the ambient
environment's own defaults. `applied_site_profile` reports the path and digest
so `build_plan` can record which profile a run used, and `authorize_execution`
can refuse an execute whose profile differs from the one the plan was built
from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

SITE_PROFILE_ENV = "GPU_FAULT_ACCEPTANCE_SITE_PROFILE"
SITE_PROFILE_SECTIONS = ("arguments", "environment")


class SiteProfileError(RuntimeError):
    """The site profile cannot be trusted to name a target."""


def site_profile_path(
    argv: list[str],
    environment: MutableMapping[str, str] | None = None,
) -> Path | None:
    """Resolve the profile a run should use, or None when it uses none."""

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--site-profile", default="")
    known, _ = pre.parse_known_args(list(argv))
    values = os.environ if environment is None else environment
    raw = (known.site_profile or values.get(SITE_PROFILE_ENV, "")).strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def load_site_profile(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SiteProfileError(f"site profile does not exist: {path}")
    mode = path.stat().st_mode
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        # This file decides which cluster and which node a destructive case
        # touches. A writable-by-others profile is a way to redirect a real
        # reboot at a node nobody approved.
        raise SiteProfileError(
            f"site profile is group- or world-writable: {path}; it names the "
            "cluster and node destructive cases act on, so it must be 0600/0640"
        )
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        document = yaml.safe_load(text)
    else:
        document = json.loads(text)
    if not isinstance(document, dict):
        raise SiteProfileError(f"site profile must be a mapping: {path}")
    unknown = sorted(set(document) - set(SITE_PROFILE_SECTIONS))
    if unknown:
        raise SiteProfileError(
            "site profile has unknown sections: "
            + ", ".join(unknown)
            + "; expected "
            + ", ".join(SITE_PROFILE_SECTIONS)
        )
    for section in SITE_PROFILE_SECTIONS:
        if not isinstance(document.get(section, {}), dict):
            raise SiteProfileError(f"site profile {section} must be a mapping")
    return document


def profile_argv(
    profile: dict[str, Any],
    argv: list[str],
    accepted: set[str] | None = None,
) -> list[str]:
    """Expand profile arguments the caller did not already pass on the line.

    A flag typed on the command line suppresses the profile's entry for it
    entirely, including for ``append`` flags such as ``--node``: an explicit
    ``--node`` replaces the profile's node list rather than adding to it,
    because a case that requires exactly two distinct nodes must not silently
    receive three.

    ``accepted`` restricts the expansion to flags a particular runner actually
    defines. One profile describes the site, not one case, and the cases do not
    agree on flag names -- the collector runners take ``--node`` while
    ``DESTR-003`` takes ``--fault-node`` and ``--spare-node``. Without the
    filter, sharing a profile makes every runner that lacks one of its keys exit
    2 on an unrecognised argument.
    """

    supplied = {item.split("=", 1)[0] for item in argv if item.startswith("--")}
    extra: list[str] = []
    for dest, value in sorted(profile.get("arguments", {}).items()):
        flag = "--" + str(dest).replace("_", "-")
        if flag in supplied:
            continue
        if accepted is not None and flag not in accepted:
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            extra.extend([flag, str(item)])
    return extra


def parser_flags(parser: argparse.ArgumentParser) -> set[str]:
    # argparse exposes no public accessor for the option strings it knows, and
    # the alternative -- parse_known_args and discard the rest -- would swallow
    # a typo in the profile instead of reporting it.
    return {option for action in parser._actions for option in action.option_strings}


def bind_site_profile(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Make ``parser`` fill its unsupplied flags from the installed profile.

    Bound at parser-construction time but evaluated at ``parse_args`` time, so a
    runner that adds its own flags after this call still has them considered.
    """

    raw = os.environ.get(SITE_PROFILE_ENV, "").strip()
    if not raw:
        return parser
    profile = load_site_profile(Path(raw))
    original = parser.parse_args

    def parse_args(  # type: ignore[override]
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        values = list(sys.argv[1:] if args is None else args)
        extra = profile_argv(profile, values, accepted=parser_flags(parser))
        return original([*extra, *values], namespace)

    parser.parse_args = parse_args  # type: ignore[method-assign]
    return parser


def apply_site_environment(
    profile: dict[str, Any],
    environment: MutableMapping[str, str] | None = None,
) -> list[str]:
    """Fill the profile's environment keys that are not already set."""

    values = os.environ if environment is None else environment
    applied: list[str] = []
    for key, value in sorted(profile.get("environment", {}).items()):
        if values.get(key, "").strip():
            continue
        values[key] = str(value)
        applied.append(key)
    return applied


def applied_site_profile(
    environment: MutableMapping[str, str] | None = None,
) -> dict[str, str] | None:
    """The profile this process resolved, as a plan-recordable identity."""

    values = os.environ if environment is None else environment
    raw = values.get(SITE_PROFILE_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def install_site_profile(argv: list[str] | None = None) -> dict[str, str] | None:
    """Publish the site profile to ``os.environ``, in place.

    Called as the first statement of a runner's ``main`` so it lands before the
    parser exists: several flags read their default from the environment at
    construction time -- ``--maintenance-window-end`` most importantly -- and a
    profile that only arrived after parsing could not supply them.

    The profile's *arguments* are applied later, by `bind_site_profile`, which
    `add_live_arguments` calls on every live runner's parser. They cannot be
    folded into ``sys.argv`` here: at this point nothing knows which flags this
    runner defines, and a profile naming ``node`` would make every case that
    calls it ``--fault-node`` exit 2.
    """

    values = list(sys.argv[1:] if argv is None else argv)
    path = site_profile_path(values)
    if path is None:
        return None
    profile = load_site_profile(path)
    apply_site_environment(profile)
    os.environ[SITE_PROFILE_ENV] = str(path)
    return applied_site_profile()
