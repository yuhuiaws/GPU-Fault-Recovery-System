from __future__ import annotations

from pathlib import Path

from scripts.render_xid_catalog_summary import load_catalog, render_summary

ROOT = Path(__file__).parents[2]
CATALOG = ROOT / "src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml"


def test_html_xid_summary_is_derived_from_the_pinned_catalog() -> None:
    document = load_catalog(CATALOG)
    summary = render_summary(document)
    spec = document["spec"]

    assert summary.startswith("# 附录：固定版本 NVIDIA XID Catalog 摘要\n")
    assert "{#nvidia-xid-catalog-summary}" not in summary
    assert document["metadata"]["name"] in summary
    assert document["metadata"]["sourceSha256"] in summary
    assert document["metadata"]["generatedSha256"] in summary
    assert f"| Catalog XID 规则 | {len(spec['catalogRules'])} |" in summary
    assert f"| NVLink5 解码规则 | {len(spec['nvlink5']['decodeRules'])} |" in summary
    assert f"| Resolution Bucket | {len(spec['resolutionBuckets'])} |" in summary

    rules = {item["xid"]: item for item in spec["catalogRules"]}
    for xid in (45, 48, 94, 95, 144, 150, 154, 159, 164, 165):
        assert f"| {xid} |" in summary
        action = rules[xid]["immediateAction"]
        if action:
            assert action in summary


def test_no_handwritten_runtime_incompatible_xid_example_remains() -> None:
    assert not (ROOT / "config/xid-policy.example.yaml").exists()

    config_readme = (ROOT / "config/README.md").read_text(encoding="utf-8")
    html_filter = (ROOT / "html-build.lua").read_text(encoding="utf-8")
    html_builder = (ROOT / "build-html.sh").read_text(encoding="utf-8")

    assert "xid-policy.example.yaml" not in config_readme
    assert "xid-policy.example.yaml" not in html_filter
    assert "render_xid_catalog_summary.py" in html_builder

    summary = render_summary(load_catalog(CATALOG))
    for unsupported_field in (
        "unknownAction",
        "decisionOrder",
        "specialRules",
        "recoveryGates",
    ):
        assert unsupported_field not in summary
