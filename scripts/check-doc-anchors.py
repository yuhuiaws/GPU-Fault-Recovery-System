"""Every ``#fragment`` link in the public Markdown has to land on a real heading.

The case index alone carries 152 cross-file anchors into
``docs/区域模式端到端验收测试用例.md``, and the deployment manual is navigated almost
entirely by anchor: 「按 [REG-8 区域部署验收清单](部署和运维手册.md#reg-8-区域部署验收清单)
执行」 is how an operator is told where to go mid-procedure. A broken anchor does not
render as an error -- GitHub silently leaves the reader at the top of the file -- so the
failure mode is an operator who reads the wrong section of a destructive runbook. Renaming
a heading is a one-character edit; nothing else in the repository notices.

``check-doc-references.py`` validates references into code (``path.py::symbol``). This
gate validates references into documentation: the link target exists, and the fragment
resolves either to a heading slug or to an explicit ``<a id="...">`` anchor.

Slugs follow GitHub's algorithm (lowercase, drop punctuation, spaces to hyphens,
``-1``/``-2`` for repeats), which keeps CJK intact, so ``## 8.1 GPU 数量变化审批`` is
reachable as ``#81-gpu-数量变化审批`` exactly as the rendered site serves it.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.doc_inventory import ROOT, documents  # noqa: E402

HEADING = re.compile(r"^ {0,3}#{1,6}(?P<text>\s.*|)$")
FENCE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})")
# ``<a id="x">`` / ``<a name="x">``: the deployment manual pins its chapter anchors
# explicitly so that renumbering a section does not silently break every inbound link.
EXPLICIT_ANCHOR = re.compile(r"<a\s+(?:id|name)\s*=\s*[\"'](?P<anchor>[^\"']+)[\"']")
INLINE_CODE = re.compile(r"`+[^`]*`+")
LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?(?P<target>[^)\s>]*)>?(?:\s+\"[^\"]*\")?\s*\)")
# ``http:``, ``mailto:``, ``//host`` -- anything whose fragment lives on another server.
EXTERNAL = re.compile(r"\A(?:[A-Za-z][A-Za-z0-9+.-]*:|//)")
HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
MARKDOWN_LINK_TEXT = re.compile(r"!?\[(?P<text>[^\]]*)\]\((?:[^)]*)\)")
MARKDOWN_REFERENCE_TEXT = re.compile(r"!?\[(?P<text>[^\]]*)\]\[[^\]]*\]")
# ``**``/``*``/``~~`` are markup; a lone ``_`` is not, because identifiers such as
# ``GPU_FAULT_REGION`` and ``__init__`` appear in headings and their underscores survive
# into the slug.
EMPHASIS = re.compile(r"\*\*|\*|~~")
# GitHub keeps letters, marks, digits, ``_`` and ``-``; everything else is dropped, which
# is why 「（唯一生产形态）」 contributes no characters at all to the slug.
SLUG_DROP = re.compile(r"[^\w\- ]", re.UNICODE)


def mask_inline_code(line: str) -> str:
    """Blank code spans, keeping offsets so link brackets still pair up.

    A code span can contain something that reads like a link (the manual quotes
    ``[text](url)`` when it explains the link convention itself), and a fragment inside
    backticks is documentation about anchors rather than an anchor.
    """
    return INLINE_CODE.sub(lambda match: " " * len(match.group(0)), line)


def heading_slug(text: str) -> str:
    """The fragment GitHub serves for a heading, from the heading's rendered text."""
    plain = HTML_TAG.sub("", text)
    plain = MARKDOWN_LINK_TEXT.sub(lambda match: match.group("text"), plain)
    plain = MARKDOWN_REFERENCE_TEXT.sub(lambda match: match.group("text"), plain)
    plain = plain.replace("`", "")
    plain = EMPHASIS.sub("", plain)
    return SLUG_DROP.sub("", plain.strip().lower()).replace(" ", "-")


def prose_lines(path: Path) -> Iterator[tuple[int, str]]:
    """The lines of ``path`` that Markdown renders as prose, skipping fenced blocks.

    These manuals are mostly fenced command blocks, and those blocks are full of
    ``# 说明`` comment lines and of quoted Markdown. A ``#`` comment is not a heading and
    a bracketed example is not a link, so counting either one is how a documentation gate
    starts inventing anchors that the rendered page does not have -- and blocking an
    author who quotes ``[见](#锚点)`` while explaining the convention.
    """
    fence: str | None = None
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        opening = FENCE.match(raw)
        if opening is not None:
            marker = opening.group("fence")
            if fence is None:
                fence = marker[0] * 3
            elif marker.startswith(fence):
                fence = None
            continue
        if fence is None:
            yield line_number, raw


def anchors(path: Path) -> set[str]:
    """Every fragment ``path`` answers to: heading slugs plus explicit HTML anchors."""
    found: set[str] = set()
    seen: dict[str, int] = {}
    for _, raw in prose_lines(path):
        found.update(EXPLICIT_ANCHOR.findall(raw))
        heading = HEADING.match(raw)
        if heading is None:
            continue
        slug = heading_slug(heading.group("text"))
        if not slug:
            continue
        # GitHub disambiguates repeats by appending ``-1``, ``-2``, ... in document order.
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        found.add(slug if count == 0 else f"{slug}-{count}")
    return found


@dataclass(frozen=True)
class Violation:
    document: Path
    line: int
    link: str
    reason: str


@dataclass(frozen=True)
class Link:
    line: int
    target: str
    path: str
    fragment: str


def local_links(document: Path) -> list[Link]:
    """Every link in ``document`` that this repository is responsible for resolving.

    External URLs are somebody else's uptime, so they are skipped; what is left is a
    path inside the repository, a fragment inside a document, or both.
    """
    found: list[Link] = []
    for line_number, raw_line in prose_lines(document):
        for match in LINK.finditer(mask_inline_code(raw_line)):
            target = match.group("target")
            if not target or EXTERNAL.match(target):
                continue
            path_text, _, fragment = target.partition("#")
            found.append(Link(line_number, target, path_text, fragment))
    return found


def check_anchors() -> list[Violation]:
    violations: list[Violation] = []
    known: dict[Path, set[str]] = {}

    for document in documents():
        for link in local_links(document):
            resolved = (
                document.parent / link.path if link.path else document
            ).resolve()
            if not resolved.is_file():
                violations.append(
                    Violation(
                        document,
                        link.line,
                        link.target,
                        "link target does not exist",
                    )
                )
                continue
            if not link.fragment or resolved.suffix != ".md":
                continue
            available = known.setdefault(resolved, anchors(resolved))
            if link.fragment not in available:
                violations.append(
                    Violation(
                        document,
                        link.line,
                        link.target,
                        "no heading or anchor answers this fragment",
                    )
                )
    return violations


def main() -> None:
    violations = check_anchors()
    if violations:
        for item in violations:
            try:
                document = item.document.relative_to(ROOT).as_posix()
            except ValueError:  # pragma: no cover - documents() is repository-relative
                document = str(item.document)
            print(f"{document}:{item.line}: {item.link}: {item.reason}")
        raise SystemExit(1)
    print("Documentation links and anchors are valid.")


if __name__ == "__main__":
    main()
