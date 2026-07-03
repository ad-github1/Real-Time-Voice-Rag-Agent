from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .documents import DocumentChunk, iter_source_files, load_documents


INDEX_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "index_manifest.json"
CHUNKS_FILENAME = "chunks.jsonl"


@dataclass(frozen=True)
class IndexBuildResult:
    index_dir: Path
    manifest_path: Path
    chunks_path: Path
    source_count: int
    chunk_count: int
    duplicate_chunks_skipped: int
    created_at_ms: int


def build_persistent_index(
    *,
    data_dir: Path,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> IndexBuildResult:
    index_dir.mkdir(parents=True, exist_ok=True)

    raw_chunks = load_documents(data_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks, duplicate_chunks_skipped = _dedupe_chunks(raw_chunks)

    created_at_ms = int(time.time() * 1000)
    source_fingerprints = collect_source_fingerprints(data_dir)

    chunks_path = index_dir / CHUNKS_FILENAME
    manifest_path = index_dir / MANIFEST_FILENAME

    _write_chunks(chunks_path, chunks)

    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "created_at_ms": created_at_ms,
        "data_dir": str(data_dir.resolve()),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "source_count": len(source_fingerprints),
        "chunk_count": len(chunks),
        "duplicate_chunks_skipped": duplicate_chunks_skipped,
        "source_fingerprints": source_fingerprints,
        "chunks_file": CHUNKS_FILENAME,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return IndexBuildResult(
        index_dir=index_dir,
        manifest_path=manifest_path,
        chunks_path=chunks_path,
        source_count=len(source_fingerprints),
        chunk_count=len(chunks),
        duplicate_chunks_skipped=duplicate_chunks_skipped,
        created_at_ms=created_at_ms,
    )


def load_index_if_valid(
    *,
    data_dir: Path,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk] | None:
    manifest = read_manifest(index_dir)

    if not manifest:
        return None

    if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
        return None

    if manifest.get("chunk_size") != chunk_size:
        return None

    if manifest.get("chunk_overlap") != chunk_overlap:
        return None

    saved_fingerprints = manifest.get("source_fingerprints")
    current_fingerprints = collect_source_fingerprints(data_dir)

    if saved_fingerprints != current_fingerprints:
        return None

    chunks_path = index_dir / str(manifest.get("chunks_file", CHUNKS_FILENAME))

    if not chunks_path.exists():
        return None

    return _read_chunks(chunks_path)


def read_manifest(index_dir: Path) -> dict[str, Any] | None:
    manifest_path = index_dir / MANIFEST_FILENAME

    if not manifest_path.exists():
        return None

    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def index_stats(index_dir: Path, data_dir: Path) -> dict[str, Any]:
    manifest = read_manifest(index_dir)
    current_sources = collect_source_fingerprints(data_dir)

    if not manifest:
        return {
            "exists": False,
            "index_dir": str(index_dir),
            "manifest_path": str(index_dir / MANIFEST_FILENAME),
            "chunks_path": str(index_dir / CHUNKS_FILENAME),
            "current_source_count": len(current_sources),
        }

    saved_sources = manifest.get("source_fingerprints", [])

    return {
        "exists": True,
        "index_dir": str(index_dir),
        "manifest_path": str(index_dir / MANIFEST_FILENAME),
        "chunks_path": str(index_dir / str(manifest.get("chunks_file", CHUNKS_FILENAME))),
        "schema_version": manifest.get("schema_version"),
        "created_at_ms": manifest.get("created_at_ms"),
        "data_dir": manifest.get("data_dir"),
        "chunk_size": manifest.get("chunk_size"),
        "chunk_overlap": manifest.get("chunk_overlap"),
        "source_count": manifest.get("source_count", 0),
        "current_source_count": len(current_sources),
        "chunk_count": manifest.get("chunk_count", 0),
        "duplicate_chunks_skipped": manifest.get("duplicate_chunks_skipped", 0),
        "is_valid_for_current_sources": saved_sources == current_sources,
    }


def clean_index(index_dir: Path) -> bool:
    if not index_dir.exists():
        return False

    shutil.rmtree(index_dir)
    return True


def collect_source_fingerprints(data_dir: Path) -> list[dict[str, Any]]:
    if not data_dir.exists():
        return []

    fingerprints: list[dict[str, Any]] = []

    for source in iter_source_files(data_dir):
        fingerprints.append(_fingerprint_file(source, data_dir))

    urls_file = data_dir / "urls.txt"

    if urls_file.exists():
        fingerprints.append(
            {
                "kind": "urls",
                "path": "urls.txt",
                "urls": _read_urls(urls_file),
            }
        )

    return sorted(fingerprints, key=lambda item: (str(item.get("kind")), str(item.get("path"))))


def format_index_stats(stats: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("Persistent Index Stats")
    lines.append("======================")
    lines.append(f"Index dir: {stats['index_dir']}")
    lines.append(f"Manifest: {stats['manifest_path']}")
    lines.append(f"Chunks: {stats['chunks_path']}")

    if not stats.get("exists"):
        lines.append("")
        lines.append("Index does not exist yet.")
        lines.append("Run: PYTHONPATH=src python3 -m voice_rag_agent.cli index build")
        lines.append(f"Current source files: {stats.get('current_source_count', 0)}")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"Schema version: {stats.get('schema_version')}")
    lines.append(f"Chunk size: {stats.get('chunk_size')}")
    lines.append(f"Chunk overlap: {stats.get('chunk_overlap')}")
    lines.append(f"Indexed source files: {stats.get('source_count')}")
    lines.append(f"Current source files: {stats.get('current_source_count')}")
    lines.append(f"Chunks: {stats.get('chunk_count')}")
    lines.append(f"Duplicate chunks skipped: {stats.get('duplicate_chunks_skipped')}")
    lines.append(f"Valid for current data: {stats.get('is_valid_for_current_sources')}")

    return "\n".join(lines)


def _fingerprint_file(source: Path, data_dir: Path) -> dict[str, Any]:
    resolved_source = source.resolve()
    resolved_data_dir = data_dir.resolve()

    try:
        relative_path = resolved_source.relative_to(resolved_data_dir).as_posix()
    except ValueError:
        relative_path = str(resolved_source)

    stat = resolved_source.stat()

    return {
        "kind": "file",
        "path": relative_path,
        "suffix": source.suffix.lower(),
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(resolved_source),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _read_urls(urls_file: Path) -> list[str]:
    return [
        line.strip()
        for line in urls_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _dedupe_chunks(chunks: list[DocumentChunk]) -> tuple[list[DocumentChunk], int]:
    seen_text_hashes: set[str] = set()
    unique_chunks: list[DocumentChunk] = []
    skipped = 0

    for chunk in chunks:
        text_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()

        if text_hash in seen_text_hashes:
            skipped += 1
            continue

        seen_text_hashes.add(text_hash)
        metadata = dict(chunk.metadata)
        metadata["text_sha256"] = text_hash

        unique_chunks.append(
            DocumentChunk(
                id=chunk.id,
                source_path=chunk.source_path,
                text=chunk.text,
                metadata=metadata,
            )
        )

    return unique_chunks, skipped


def _write_chunks(path: Path, chunks: list[DocumentChunk]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = {
                "id": chunk.id,
                "source_path": chunk.source_path,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")


def _read_chunks(path: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue

        record = json.loads(line)

        chunks.append(
            DocumentChunk(
                id=str(record["id"]),
                source_path=str(record["source_path"]),
                text=str(record["text"]),
                metadata=dict(record.get("metadata") or {}),
            )
        )

    return chunks
