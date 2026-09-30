"""The live-site tokens a promoted e2e script must never embed.

The drivers under ``scripts/e2e/regional`` were written against one site and
then parameterised through the site profile; these checks prove the site's
values did not creep back in. The account check matches any AWS account
number rather than the live one, so this module -- and the tests that use
it -- carry no site identity themselves.
"""

from __future__ import annotations

import re

STATE_DIRECTORY = "/secure/gpu-fault-bootstrap"
NODE_NAME_PREFIX = "gpu-fault-gpu-1-"
# Twelve digits standing alone (not inside a hex digest or a longer number);
# the all-zero placeholder is the one account number an example may spell out.
AWS_ACCOUNT_NUMBER = re.compile(r"(?<![0-9A-Za-z])(?!0{12})[0-9]{12}(?![0-9A-Za-z])")


def site_topology_leaks(source: str) -> list[str]:
    """Every live-site token ``source`` embeds, empty when it is clean."""

    leaks = [token for token in (STATE_DIRECTORY, NODE_NAME_PREFIX) if token in source]
    leaks.extend(f"AWS account {match}" for match in AWS_ACCOUNT_NUMBER.findall(source))
    return leaks
