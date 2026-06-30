from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SOURCE_SUFFIXES = {".md", ".markdown", ".txt"}


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    source_path: str
    text: str
    metadata: dict[str, str | int]


def iter_source_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in SOURCE_SUFFIXES)


def extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    cleaned = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not cleaned:
        return []

    paragraphs = [paragraph.strip() for paragraph in cleaned.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_paragraph(paragraph, max_chars, overlap))
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current.strip())
            prefix = _tail(current, overlap)
            current = f"{prefix}\n\n{paragraph}".strip() if prefix else paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def _split_long_paragraph(paragraph: str, max_chars: int, overlap: int) -> list[str]:
    words = paragraph.split()
    chunks: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join([*current, word])
        if len(candidate) <= max_chars:
            current.append(word)
            continue
        if current:
            chunk = " ".join(current)
            chunks.append(chunk)
            current = _tail(chunk, overlap).split()
        current.append(word)

    if current:
        chunks.append(" ".join(current))
    return chunks


def _tail(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text if max_chars > 0 else ""
    tail = text[-max_chars:].lstrip()
    first_space = tail.find(" ")
    return tail[first_space + 1 :] if first_space > 0 else tail


def load_documents(root: Path, chunk_size: int = 900, chunk_overlap: int = 120) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for source in iter_source_files(root):
        text = source.read_text(encoding="utf-8")
        title = extract_title(text, source.stem.replace("_", " ").title())
        for index, chunk_text in enumerate(split_text(text, chunk_size, chunk_overlap)):
            digest = hashlib.sha1(f"{source}:{index}:{chunk_text}".encode("utf-8")).hexdigest()[:12]
            chunks.append(
                DocumentChunk(
                    id=f"chunk_{digest}",
                    source_path=str(source),
                    text=chunk_text,
                    metadata={
                        "title": title,
                        "path": str(source),
                        "chunk_index": index,
                    },
                )
            )
    return chunks
