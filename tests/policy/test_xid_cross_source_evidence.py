"""The NVIDIA evidence behind treating DCGM field 230 as a second report, pinned.

Batch 237 joins a field-230 change to the kernel line of the same XID because
NVIDIA documents the field as the value of an XID, not as an event of its own:
DCGM Field Identifiers (DCGM documentation "Version: 4.6 (Latest)", re-fetched
for the batch) list ``DCGM_FI_DEV_XID_ERROR 230`` as "XID errors." "The value
is the specific XID error", and NVIDIA's dcgm-exporter 4.8.4 default counters
call ``DCGM_FI_DEV_XID_ERRORS`` the "Value of the last XID error encountered."
The quotes live in the deciding module, in the decisions' reasons, in both
design manuals and in the validation limitations; a quote that changes here
must change on NVIDIA's page first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fault.policy import xid_cross_source

ROOT = Path(__file__).resolve().parents[2]
DESIGNS = ("docs/详细设计.md", "docs/en/detailed-design.md")
FAULT_TABLES = (
    "docs/故障类别与处置动作总表.md",
    "docs/en/fault-categories-and-actions.md",
)


def text_of(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_the_module_carries_the_exact_nvidia_page_version_and_quotes():
    assert xid_cross_source.DCGM_XID_FIELD_REFERENCE == (
        "https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/"
        "dcgm-api-field-ids.html#c.DCGM_FI_DEV_XID_ERROR"
    )
    assert xid_cross_source.DCGM_XID_FIELD_DOCS_VERSION == "4.6"
    assert xid_cross_source.DCGM_XID_FIELD_QUOTE == (
        "XID errors.",
        "The value is the specific XID error",
    )
    assert xid_cross_source.DCGM_EXPORTER_XID_REFERENCE == (
        "https://github.com/NVIDIA/dcgm-exporter/blob/4.8.4/etc/default-counters.csv"
    )
    assert xid_cross_source.DCGM_EXPORTER_XID_QUOTE == (
        "Value of the last XID error encountered."
    )
    assert "The value is the specific XID error" in xid_cross_source.FIELD_230_EVIDENCE
    assert "DCGM docs 4.6" in xid_cross_source.FIELD_230_EVIDENCE
    doc = xid_cross_source.__doc__ or ""
    assert "kernel twin" in doc and "Version: 4.6 (Latest)" in doc
    for quote in (*xid_cross_source.DCGM_XID_FIELD_QUOTE, "Value of the last XID"):
        assert quote in doc.replace("\n", " "), quote


@pytest.mark.parametrize("document", [*DESIGNS, "docs/validation-limitations.md"])
def test_the_manuals_cite_the_field_230_semantics(document: str):
    text = text_of(document)

    assert xid_cross_source.DCGM_XID_FIELD_REFERENCE in text, document
    assert "Version: 4.6 (Latest)" in text, document
    for quote in xid_cross_source.DCGM_XID_FIELD_QUOTE:
        assert quote in text, (document, quote)
    assert xid_cross_source.DCGM_EXPORTER_XID_QUOTE in text, document


@pytest.mark.parametrize("document", [*DESIGNS, *FAULT_TABLES])
def test_the_manuals_name_the_deciding_module(document: str):
    assert "xid_cross_source" in text_of(document), document
    assert "nvlink5_gauge" not in text_of(document), (
        "batch 235's module is generalised into xid_cross_source"
    )


def test_the_design_manuals_cite_the_exporter_counters_at_the_pinned_version():
    for document in DESIGNS:
        assert xid_cross_source.DCGM_EXPORTER_XID_REFERENCE in text_of(document), (
            document
        )
