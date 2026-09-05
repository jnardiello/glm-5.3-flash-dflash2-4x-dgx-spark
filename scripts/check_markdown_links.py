#!/usr/bin/env python3
"""Check local links and heading anchors in public Markdown files."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".claude",
    "bench-results",
    "experiments",
    "reports",
    "todo",
}
SKIP_PREFIXES = {
    Path("scripts/node/moe-tune"),
    Path("scripts/node/nccl-bench"),
}
LINK_RE = re.compile(r"!?\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def is_public(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    return not any(rel == prefix or prefix in rel.parents for prefix in SKIP_PREFIXES)


def markdown_files() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*.md") if is_public(path))


def github_slug(text: str) -> str:
    text = HTML_TAG_RE.sub("", text)
    text = re.sub(r"[\[\]`*_~]", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return re.sub(r"\s", "-", text)


def anchors(path: Path) -> set[str]:
    found: set[str] = set()
    counts: dict[str, int] = {}
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING_RE.match(line)
        if not match:
            continue
        base = github_slug(match.group(1))
        count = counts.get(base, 0)
        counts[base] = count + 1
        found.add(base if count == 0 else f"{base}-{count}")
    return found


def check_file(source: Path) -> list[str]:
    errors: list[str] = []
    in_fence = False
    for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in LINK_RE.finditer(line):
            raw = match.group(1).strip("<>")
            parsed = urlsplit(raw)
            if parsed.scheme or raw.startswith("//"):
                continue
            relative = unquote(parsed.path)
            fragment = unquote(parsed.fragment)
            if not relative:
                target = source
            elif relative.startswith("/"):
                target = ROOT / relative.lstrip("/")
            else:
                target = source.parent / relative
            target = target.resolve()
            try:
                target.relative_to(ROOT)
            except ValueError:
                errors.append(f"{source.relative_to(ROOT)}:{lineno}: link escapes repository: {raw}")
                continue
            if not target.exists():
                errors.append(f"{source.relative_to(ROOT)}:{lineno}: missing target: {raw}")
                continue
            if fragment and target.is_file() and target.suffix.lower() == ".md":
                if fragment not in anchors(target):
                    errors.append(
                        f"{source.relative_to(ROOT)}:{lineno}: missing anchor #{fragment} in "
                        f"{target.relative_to(ROOT)}"
                    )
    return errors


def main() -> int:
    errors = [error for path in markdown_files() for error in check_file(path)]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"markdown-links: PASS ({len(markdown_files())} public files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
