"""Which ``gpu-fault-admin`` may act on a managed state directory.

The first ``deploy`` installs the site's own deploy-host CLI under
``<state-dir>/deployer-venv`` and binds it there with
``gpu-fault-managed-state-dir.json``. Two rules follow, both checked before a
command opens its log:

* a bound CLI acts only on its own state directory: ``--state-dir`` must name
  it, and every managed-site command needs one;
* an unbound CLI -- the developer checkout's ``.venv``, which runs whatever the
  working tree holds -- does not mutate a site that already has a bound CLI.
  ``deploy`` is exempt (it prepares the release and re-execs into the bound CLI
  itself) and so are the read-only verbs. Live 2026-09-15: a join-cluster run
  from the checkout verified its candidate against the checkout's own
  ``dist/`` and rolled a healthy data plane back; nothing had said "wrong CLI".
  The refusal names the CLI to run; ``GPU_FAULT_ADMIN_ALLOW_UNBOUND=1`` runs the
  checkout's code against the site on purpose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Collection
from pathlib import Path
from typing import cast

from gpu_fault.admin.site import SiteConfigError

DEPLOY_HOST_STATE_BINDING = "gpu-fault-managed-state-dir.json"
ALLOW_UNBOUND_ENV = "GPU_FAULT_ADMIN_ALLOW_UNBOUND"
UNBOUND_ALLOWED_COMMANDS = frozenset({"deploy"})
_TRUE = frozenset({"1", "true", "yes", "on"})


def bound_state_dir(prefix: Path | None = None) -> Path | None:
    """The state directory this interpreter's venv is bound to, or None."""

    binding = (prefix or Path(sys.prefix)).resolve() / DEPLOY_HOST_STATE_BINDING
    if not binding.is_file():
        return None
    try:
        value = json.loads(binding.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SiteConfigError("deploy-host state-dir binding is invalid") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise SiteConfigError("deploy-host state-dir binding schema is invalid")
    state_dir = value.get("state_dir")
    if not isinstance(state_dir, str) or not state_dir.strip():
        raise SiteConfigError("deploy-host state-dir binding has no state directory")
    return Path(state_dir).expanduser().resolve()


def site_bound_admin(state_dir: Path) -> Path | None:
    """The CLI ``deploy`` bound to ``state_dir``, or None before the first deploy."""

    venv = state_dir.expanduser().resolve() / "deployer-venv"
    if not (venv / DEPLOY_HOST_STATE_BINDING).is_file():
        return None
    return venv / "bin" / "gpu-fault-admin"


def unbound_allowed(environ: os._Environ[str] | dict[str, str] | None = None) -> bool:
    value = (os.environ if environ is None else environ).get(ALLOW_UNBOUND_ENV, "")
    return value.strip().lower() in _TRUE


def enforce_deploy_host_state_dir(
    arguments: argparse.Namespace,
    *,
    readonly_commands: Collection[str] = (),
) -> None:
    bound = bound_state_dir()
    provided = cast(Path | None, getattr(arguments, "state_dir", None))
    command = str(getattr(arguments, "command", "") or "")
    if bound is not None:
        if provided is not None:
            if provided.expanduser().resolve() == bound:
                return
            raise SiteConfigError(
                f"installed deploy-host is bound to --state-dir {bound}; "
                f"refusing {provided.expanduser().resolve()}"
            )
        explicit = cast(Path | None, getattr(arguments, "file", None))
        if (
            explicit is not None
            and explicit.expanduser().resolve() == bound / "site.yaml"
        ):
            return
        raise SiteConfigError(
            f"installed deploy-host is bound to --state-dir {bound}; "
            f"{command} requires that managed state"
        )
    if (
        provided is None
        or command in UNBOUND_ALLOWED_COMMANDS
        or command in readonly_commands
    ):
        return
    admin = site_bound_admin(provided)
    if admin is None or unbound_allowed():
        return
    raise SiteConfigError(
        f"{provided.expanduser().resolve()} has its own deploy-host CLI; run "
        f"{admin} {command} ... instead of this checkout's gpu-fault-admin, or set "
        f"{ALLOW_UNBOUND_ENV}=1 to run this checkout's code against the site on purpose"
    )
