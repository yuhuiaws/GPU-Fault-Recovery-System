from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.e2e.regional import run_boot_acceptance as boot


class BootEntry:
    def __init__(self, root, monkeypatch, *, case="013", execute=True):
        self.root = root
        self.state = root / "state"
        self.state.mkdir()
        self.site = self.state / "site.yaml"
        self.site.write_text("unit site", encoding="utf-8")
        self.identity = {"release_id": "release-unit", "cluster_id": "isolated"}
        self.prerequisite = {"valid": True, "verdict": "PASS"}
        self.calls = []
        self.fixture = SimpleNamespace(
            regional=SimpleNamespace(evidence_identity=self.evidence_identity)
        )
        argv = [
            "--case",
            f"GF-REGIONAL-BOOT-{case}",
            "--run-dir",
            str(root / "run"),
            "--site",
            str(self.site),
            "--production-site",
            str(root / "production-site"),
            "--bootstrap-state-dir",
            str(self.state),
            "--cpu-cluster-arn",
            "arn:aws:eks:us-east-1:000000000000:cluster/cpu-unit",
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:000000000000:cluster/gpu-unit",
            "--admin-email",
            "unit@example.invalid",
            "--retain-bootstrap-site",
        ]
        if execute:
            argv.extend(["--execute", "--confirm", f"BOOT{case}_EXECUTE"])
        self.arguments = boot.parser().parse_args(argv)
        self.path = (
            self.arguments.run_dir
            / "cases"
            / self.arguments.case
            / f"{self.arguments.case}.json"
        )
        monkeypatch.setattr(
            boot, "parser", lambda: SimpleNamespace(parse_args=lambda: self.arguments)
        )
        monkeypatch.setattr(boot, "install_site_profile", self.install_site)
        monkeypatch.setattr(boot, "SiteFixture", self.make_fixture)
        monkeypatch.setattr(
            boot, "predecessor_path", lambda *args: ("unit-predecessor", root / "prior")
        )
        monkeypatch.setattr(boot, "predecessor_evidence", self.predecessor)
        monkeypatch.setattr(boot, "authorize_execution", self.authorize)

    def install_site(self):
        self.calls.append(("site-profile",))

    def evidence_identity(self):
        self.calls.append(("identity",))
        return dict(self.identity)

    def make_fixture(self, site: Path, cluster_id=""):
        self.calls.append(("fixture", site, cluster_id))
        return self.fixture

    def predecessor(self, path, case_id, **identity):
        self.calls.append(("predecessor", path, case_id, identity))
        return dict(self.prerequisite)

    def authorize(self, arguments, **kwargs):
        self.calls.append(("authorize", arguments, kwargs))
