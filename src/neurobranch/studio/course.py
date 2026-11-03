"""Read the existing Master Review course; do not duplicate it in the UI."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from neurobranch.studio.paths import StudioPaths


@dataclass(frozen=True)
class Chapter:
    number: int
    title: str
    body: str


@lru_cache(maxsize=1)
def load_chapters(path: Path | None = None) -> tuple[Chapter, ...]:
    source = path or StudioPaths().course
    text = source.read_text(encoding="utf-8")
    chunks = re.split(r"(?m)^## Chapter ", text)
    chapters: list[Chapter] = []
    for chunk in chunks[1:]:
        header, _, body = chunk.partition("\n")
        match = re.match(r"(\d+)\s*[—-]\s*(.+)", header.strip())
        if not match:
            continue
        chapters.append(
            Chapter(
                number=int(match.group(1)),
                title=match.group(2).strip(),
                body=body.strip(),
            )
        )
    if len(chapters) != 16:
        raise RuntimeError(f"Expected 16 course chapters, found {len(chapters)}")
    return tuple(chapters)


def course_path() -> Path:
    return StudioPaths().course
