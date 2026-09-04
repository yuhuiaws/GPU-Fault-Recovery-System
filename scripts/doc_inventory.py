"""Which Markdown files the documentation gates are allowed to see.

``check-doc-references.py`` and ``check-doc-anchors.py`` must scan exactly the same
surface: the public Markdown of this repository, derived from ``.gitignore`` rather
than from a hand-written list, because a hand-written list drifts the moment a
document moves and a stale exemption silently switches a hard gate off. The
derivation is subtle enough that a second copy would be a second thing to keep
right, so it lives here and both gates import it.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GITIGNORE = ROOT / ".gitignore"
# 外链里的路径不是本仓引用。上游仓库正好也有 ``src/...`` 布局：
# CLOUD_PROVIDER_SOURCE_REVIEW.md 的 cluster-health-scanner 链接里有
# ``src/gpu_healthcheck/gpu_healthcheck.py``，HYPERPOD_SOURCE_REVIEW.md 的
# aws-do-hyperpod 链接里有 ``src/manifests/health-monitoring-agent.yaml``，
# 两者在本仓都不存在也不该存在。同理 ``policy.py`` 这类名字出现在上游
# URL 里也不是本仓的遗留命名。
URL = re.compile(r"<?https?://[^\s)>\]]+")


def private_markdown_rules() -> tuple[frozenset[str], tuple[str, ...]]:
    """Markdown that ``.gitignore`` keeps out of the public repository.

    豁免面必须等于「不进公开仓库的那批文件」，所以直接从 ``.gitignore``
    派生，而不是手抄一份名单。手抄的那份已经在漂移：``IMPLEMENTATION.md``
    等 6 个名字被移进 ``internal-docs/`` 之后，根目录的豁免条目就成了死条目
    ——哪天有人在根目录新建同名公开文档，它会被静默豁免，而这条豁免正是
    ``LEGACY_MONOLITH`` 这类硬错误的开关。

    只解析带 ``/`` 前缀的锚定行：以 ``/`` 结尾的按目录前缀匹配，以 ``.md``
    结尾的按路径精确匹配。其余条目（``__pycache__/``、``artifacts/*`` 等）
    和 Markdown 扫描面无关。
    """
    files: set[str] = set()
    directories: list[str] = []
    for raw in GITIGNORE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("/"):
            continue
        entry = line[1:]
        if entry.endswith("/"):
            directories.append(entry)
        elif entry.endswith(".md"):
            files.add(entry)
    return frozenset(files), tuple(sorted(directories))


PRIVATE_FILES, PRIVATE_DIRECTORIES = private_markdown_rules()


def is_private(relative: str) -> bool:
    return relative in PRIVATE_FILES or any(
        relative.startswith(prefix) for prefix in PRIVATE_DIRECTORIES
    )


def documents() -> list[Path]:
    """Every reviewed Markdown file, not just the ones under docs/.

    The blocklists in the gates that call this also apply to root-level public
    Markdown such as ``README.md``; a ``docs/``-only scan would miss those
    references.
    """
    return [
        path
        for path in sorted(ROOT.glob("*.md")) + sorted(DOCS.rglob("*.md"))
        if not is_private(path.relative_to(ROOT).as_posix())
    ]


def mask_urls(line: str) -> str:
    """Blank out URL spans, keeping offsets so reported columns stay usable."""
    return URL.sub(lambda match: " " * len(match.group(0)), line)
