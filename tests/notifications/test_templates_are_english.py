"""Every notification the product mails is English.

The operator-facing e-mail templates used to be Chinese; the solution is meant
to be deployable by any team, so the whole notification path -- shared
templates, per-notification subjects and guidance strings, the delivery
context header and the drill banner -- must render without CJK text. The
template *version tags* deliberately keep their historical ``zh`` lineage
because several deduplication keys embed them (see ``notifications/common.py``).
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.notifications import common
from gpu_fault.notifications.common import AdvisoryNotification
from gpu_fault.notifications.delivery_context import CONTEXT_HEADER
from gpu_fault.store.shared import notification_helpers

CJK = re.compile(r"[一-鿿]")
# Every module on the notification path: their imported string constants (subjects,
# guidance, banners, templates) are what the mails are rendered from.
NOTIFICATION_MODULES = sorted(
    [
        *(
            f"gpu_fault.notifications.{path.stem}"
            for path in (Path(common.__file__).parent.glob("*.py"))
            if path.stem != "__init__"
        ),
        "gpu_fault.notification_service",
        "gpu_fault.notification_preview",
        "gpu_fault.store.shared.notification_helpers",
    ]
)
TEMPLATES = {
    name: value
    for name, value in vars(common).items()
    if name.endswith("_EMAIL_TEMPLATE") and isinstance(value, str)
}


def test_every_shared_template_is_english_and_names_its_version() -> None:
    assert len(TEMPLATES) >= 18, sorted(TEMPLATES)
    for name, template in TEMPLATES.items():
        assert not CJK.search(template), f"{name} still carries CJK text"
        assert "{template_version}" in template, f"{name} lost its version footer"


@pytest.mark.parametrize("module_name", NOTIFICATION_MODULES)
def test_notification_module_strings_carry_no_cjk_text(module_name: str) -> None:
    """The rendered mails are built from these modules' string constants
    (subjects, guidance, banners, templates); none may carry CJK text."""

    module = importlib.import_module(module_name)
    offenders = [
        f"{module_name}.{name}: {value[:60]!r}"
        for name, value in vars(module).items()
        if isinstance(value, str) and not name.startswith("__") and CJK.search(value)
    ]
    assert offenders == [], "\n".join(offenders)


def test_delivery_context_and_drill_banner_are_english() -> None:
    assert CONTEXT_HEADER == "Notification context"
    notification = AdvisoryNotification(
        deduplication_key="dedup-drill",
        cluster_name="cluster-a",
        incident_id="incident-drill",
        subject="GPU fault",
        body_text="body",
        support_case_draft="",
    )
    labelled = notification_helpers.with_incident_drill_label(
        notification, SimpleNamespace(drill_id="drill-1")
    )
    assert labelled.subject == "[DRILL:drill-1] GPU fault"
    assert labelled.body_text.startswith(
        "[DRILL NOTIFICATION - NOT A REAL FAULT]\nDrill ID: drill-1\n\n"
    ), labelled.body_text
    assert not CJK.search(labelled.subject + labelled.body_text), labelled


def test_template_version_tags_are_unchanged_dedup_key_material() -> None:
    """A retag would change the deduplication keys that embed the tag and
    re-notify every open incident once; wording changes never bump them."""

    versions = {
        name: value
        for name, value in vars(common).items()
        if name.endswith("_TEMPLATE_VERSION") and isinstance(value, str)
    }
    assert len(versions) >= 18, sorted(versions)
    assert all("-zh-v" in value for value in versions.values()), versions
