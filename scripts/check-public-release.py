"""Reject live-environment identities from the public repository tree."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pathspec


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".lua",
    ".md",
    ".py",
    ".sh",
    ".sql",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SCANNER_SOURCES = {
    "scripts/check-public-release.py",
    "tests/test_public_release.py",
    "tests/test_documentation_contracts.py",
    "tests/test_fault_scenario_catalog.py",
    "tests/test_script_assets.py",
}
PUBLIC_AWS_ACCOUNTS = {
    "000000000000",
    "111122223333",
    "123456789012",
    # AWS-owned public ECR registry used by the ADOT image.
    "602401143452",
}
SYNTHETIC_INSTANCE_IDS = {
    "i-0000000000000000",
    "i-00000000000000001",
    "i-00000000000000002",
}
PLACEHOLDER_RESOURCE_IDS = {
    "sg-0123456789abcdef0",
}

ARN_ACCOUNT = re.compile(
    r"arn:(?:aws|aws-cn|aws-us-gov):[^:\s]+:[^:\s]*:"
    r"(?P<account>\d{12}):"
)
ECR_ACCOUNT = re.compile(r"\b(?P<account>\d{12})\.dkr\.ecr\.")
AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
INSTANCE_ID = re.compile(r"\bi-[0-9a-f]{16,17}\b", re.IGNORECASE)
RESOURCE_ID = re.compile(
    r"\b(?:vpc|subnet|sg|eni|vol|snap|ami|igw|nat|rtb|vpce|fs)"
    r"-[0-9a-f]{8,17}\b",
    re.IGNORECASE,
)
UUID = re.compile(
    r"(?<![0-9a-f])"
    r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
    r"(?![0-9a-f])",
    re.IGNORECASE,
)
BARE_NODE_ID = re.compile(
    r"(?<![0-9a-f-])[0-9a-f]{17}(?![0-9a-f])",
    re.IGNORECASE,
)
CONCRETE_POD = re.compile(r"\bgpu-fault-[a-z0-9-]+-[0-9a-f]{8,10}-[a-z0-9]{5}\b")
PERSONAL_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@(?:amazon\.com|amazon\.aws)\b",
    re.IGNORECASE,
)
SITE_PRIVATE_ADDRESS = re.compile(r"\b10\.(?:91|92)(?:\.\d{1,3}){2}\b")
LOCAL_REPOSITORY = re.compile(r"/(?:home|Users)/[^/\s]+/GPU_failure_handling")
SITE_MARKERS = (
    re.compile(r"\b(?:hp|eks)-cluster-hypd-[A-Za-z0-9-]+\b", re.IGNORECASE),
    re.compile(r"\bcontrol-plane-GPU-fault-solution\b", re.IGNORECASE),
    re.compile(r"\bhypd-\d+\b", re.IGNORECASE),
    re.compile(r"\bspot-p5en-usw2az3\b", re.IGNORECASE),
    re.compile(r"\baccelerated-liangtest-\d+\b", re.IGNORECASE),
    re.compile(r"\bliang(?:aws|200)\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    label: str
    value: str


def public_text_files(root: Path) -> list[Path]:
    ignore_path = root / ".gitignore"
    ignore = pathspec.GitIgnoreSpec.from_lines(
        ignore_path.read_text(encoding="utf-8").splitlines()
        if ignore_path.is_file()
        else ()
    )
    result = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith(".git/") or ignore.match_file(relative):
            continue
        if relative in SCANNER_SOURCES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data:
            continue
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        result.append(path)
    return sorted(result)


def _append_matches(
    violations: list[Violation],
    relative: str,
    line_number: int,
    line: str,
    label: str,
    pattern: re.Pattern[str],
) -> None:
    violations.extend(
        Violation(relative, line_number, label, match.group())
        for match in pattern.finditer(line)
    )


def scan(root: Path) -> list[Violation]:
    violations: list[Violation] = []
    for path in public_text_files(root):
        relative = path.relative_to(root).as_posix()
        markdown = path.suffix.lower() == ".md"
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            for match in ARN_ACCOUNT.finditer(line):
                account = match.group("account")
                if account not in PUBLIC_AWS_ACCOUNTS:
                    violations.append(
                        Violation(
                            relative,
                            line_number,
                            "customer AWS account",
                            account,
                        )
                    )
            for match in ECR_ACCOUNT.finditer(line):
                account = match.group("account")
                if account not in PUBLIC_AWS_ACCOUNTS:
                    violations.append(
                        Violation(
                            relative,
                            line_number,
                            "customer ECR account",
                            account,
                        )
                    )
            for match in INSTANCE_ID.finditer(line):
                value = match.group().lower()
                if value not in SYNTHETIC_INSTANCE_IDS:
                    violations.append(
                        Violation(
                            relative,
                            line_number,
                            "concrete EC2 instance ID",
                            value,
                        )
                    )
            for match in RESOURCE_ID.finditer(line):
                value = match.group().lower()
                if value not in PLACEHOLDER_RESOURCE_IDS:
                    violations.append(
                        Violation(
                            relative,
                            line_number,
                            "concrete AWS resource ID",
                            value,
                        )
                    )
            _append_matches(
                violations,
                relative,
                line_number,
                line,
                "AWS access key",
                AWS_ACCESS_KEY,
            )
            _append_matches(
                violations,
                relative,
                line_number,
                line,
                "private key",
                PRIVATE_KEY,
            )
            _append_matches(
                violations,
                relative,
                line_number,
                line,
                "personal email",
                PERSONAL_EMAIL,
            )
            _append_matches(
                violations,
                relative,
                line_number,
                line,
                "site private address",
                SITE_PRIVATE_ADDRESS,
            )
            _append_matches(
                violations,
                relative,
                line_number,
                line,
                "local repository path",
                LOCAL_REPOSITORY,
            )
            for pattern in SITE_MARKERS:
                _append_matches(
                    violations,
                    relative,
                    line_number,
                    line,
                    "site-specific resource name",
                    pattern,
                )
            if markdown:
                _append_matches(
                    violations,
                    relative,
                    line_number,
                    line,
                    "raw UUID in public documentation",
                    UUID,
                )
                _append_matches(
                    violations,
                    relative,
                    line_number,
                    line,
                    "raw node identity in public documentation",
                    BARE_NODE_ID,
                )
                _append_matches(
                    violations,
                    relative,
                    line_number,
                    line,
                    "concrete Kubernetes Pod in public documentation",
                    CONCRETE_POD,
                )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    violations = scan(root)
    if violations:
        print("public release safety check failed:", file=sys.stderr)
        for item in violations:
            print(
                f"- {item.path}:{item.line}: {item.label}: {item.value}",
                file=sys.stderr,
            )
        return 1
    print(
        "public release safety check passed: "
        f"scanned {len(public_text_files(root))} text file(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
