from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MarkdownMetadata:
    """Metadata parsed from the top of a Vinpearl markdown product file."""

    title: str
    url: str
    original_price: str
    current_price: str


@dataclass(frozen=True)
class MarkdownSection:
    """A second-level markdown section and its text content."""

    title: str
    content: str


RagChunk = dict[str, Any]


def parse_markdown_metadata(markdown: str) -> MarkdownMetadata:
    """
    Parse title, URL, original price, and current price from markdown text.

    Expected file shape:
        # Product title
        - URL: ...
        - Gia goc: ...
        - Gia hien tai: ...
    """
    title_match = re.search(r"^#\s+(.+?)\s*$", markdown, flags=re.MULTILINE)
    url_match = re.search(r"^-\s*URL:\s*(.*?)\s*$", markdown, flags=re.MULTILINE)
    original_price_match = re.search(r"^-\s*Giá gốc:\s*(.*?)\s*$", markdown, flags=re.MULTILINE)
    current_price_match = re.search(r"^-\s*Giá hiện tại:\s*(.*?)\s*$", markdown, flags=re.MULTILINE)

    return MarkdownMetadata(
        title=title_match.group(1).strip() if title_match else "",
        url=url_match.group(1).strip() if url_match else "",
        original_price=original_price_match.group(1).strip() if original_price_match else "",
        current_price=current_price_match.group(1).strip() if current_price_match else "",
    )


def split_markdown_sections(markdown: str) -> list[MarkdownSection]:
    """
    Split markdown content into sections using second-level headings (`##`).

    Content before the first `##` is treated as file-level metadata/intro and is
    not returned as a section.
    """
    section_pattern = re.compile(r"^##\s+(.+?)\s*$", flags=re.MULTILINE)
    matches = list(section_pattern.finditer(markdown))
    sections: list[MarkdownSection] = []

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        content = markdown[start:end].strip()
        if content:
            sections.append(MarkdownSection(title=match.group(1).strip(), content=content))

    return sections


def sliding_window_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into character chunks with a sliding window and overlap.

    Args:
        text: Section text to chunk.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared between consecutive chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    cleaned_text = normalize_whitespace(text)
    if not cleaned_text:
        return []
    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(cleaned_text), step):
        chunk = cleaned_text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(cleaned_text):
            break
    return chunks


def normalize_whitespace(text: str) -> str:
    """Normalize repeated whitespace while preserving readable markdown text."""
    lines = [line.strip() for line in text.splitlines()]
    compact_text = "\n".join(line for line in lines if line)
    return re.sub(r"\n{3,}", "\n\n", compact_text).strip()


def build_rag_chunks(
    markdown: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[RagChunk]:
    """
    Convert markdown text into RAG chunks with metadata.

    Each output item has the shape:
        {"text": "...", "metadata": {...}}
    """
    metadata = parse_markdown_metadata(markdown)
    sections = split_markdown_sections(markdown)
    rag_chunks: list[RagChunk] = []
    chunk_index = 0

    for section in sections:
        for chunk_text in sliding_window_chunk(section.content, chunk_size=chunk_size, overlap=overlap):
            rag_chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "title": metadata.title,
                        "url": metadata.url,
                        "original_price": metadata.original_price,
                        "current_price": metadata.current_price,
                        "section": section.title,
                        "chunk_index": chunk_index,
                    },
                }
            )
            chunk_index += 1

    return rag_chunks


def build_rag_chunks_from_file(
    file_path: str | Path,
    chunk_size: int = 500,
    overlap: int = 50,
    encoding: str = "utf-8",
) -> list[RagChunk]:
    """Read a markdown file and convert it into RAG-ready chunks."""
    path = Path(file_path)
    markdown = path.read_text(encoding=encoding)
    return build_rag_chunks(markdown, chunk_size=chunk_size, overlap=overlap)


def build_rag_chunks_from_directory(
    directory_path: str | Path,
    chunk_size: int = 500,
    overlap: int = 50,
    pattern: str = "*.md",
    encoding: str = "utf-8",
) -> list[RagChunk]:
    """Convert all matching markdown files in a directory into RAG chunks."""
    directory = Path(directory_path)
    chunks: list[RagChunk] = []
    for file_path in sorted(directory.glob(pattern)):
        if file_path.is_file():
            chunks.extend(
                build_rag_chunks_from_file(
                    file_path,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    encoding=encoding,
                )
            )
    return chunks
