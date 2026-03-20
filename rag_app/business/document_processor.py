"""
Document processing layer.
Responsible for:
  - Extracting raw text from PDF bytes
  - Splitting text into overlapping chunks
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import pdfplumber

from config.settings import ChunkingConfig


@dataclass
class TextChunk:
    """A single chunk of text with its origin metadata."""
    text: str
    source: str          # Original filename
    chunk_index: int
    page_hint: str = ""  # e.g. "pages 3-4" (best effort)


class DocumentProcessor:
    """Extracts and chunks text from one or more PDF files."""

    def __init__(self, config: ChunkingConfig):
        self._max_words = config.max_words
        self._overlap_words = config.overlap_words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_pdf(self, pdf_bytes: bytes, filename: str) -> List[TextChunk]:
        """Extract text from *pdf_bytes* and return a list of chunks."""
        pages_text = self._extract_pages(pdf_bytes)
        full_text = "\n\n".join(pages_text)
        raw_chunks = self._chunk_text(full_text)
        return [
            TextChunk(text=chunk, source=filename, chunk_index=i)
            for i, chunk in enumerate(raw_chunks)
            if chunk.strip()
        ]

    def process_multiple(
        self, files: List[tuple[str, bytes]]
    ) -> List[TextChunk]:
        """Process multiple (filename, bytes) pairs."""
        all_chunks: List[TextChunk] = []
        for filename, pdf_bytes in files:
            all_chunks.extend(self.process_pdf(pdf_bytes, filename))
        return all_chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_pages(self, pdf_bytes: bytes) -> List[str]:
        """Return a list of page texts using pdfplumber."""
        import io
        pages: List[str] = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)
        return pages

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split *text* on Markdown-style headers or blank lines,
        then merge / split to respect *max_words* with *overlap_words*.
        """
        # Try to split on headers first; fall back to paragraph breaks
        sections = re.split(r"\n(?=#{1,3} )", text)
        if len(sections) <= 1:
            # No headers found – split on double newlines
            sections = re.split(r"\n{2,}", text)

        chunks: List[str] = []
        current_words: List[str] = []

        for section in sections:
            section_words = section.split()
            if not section_words:
                continue

            if len(current_words) + len(section_words) <= self._max_words:
                current_words.extend(section_words)
            else:
                if current_words:
                    chunks.append(" ".join(current_words))
                overlap = (
                    current_words[-self._overlap_words :]
                    if self._overlap_words > 0
                    else []
                )
                current_words = overlap + section_words

                # Section itself is too large – force-split it
                while len(current_words) > self._max_words:
                    chunks.append(" ".join(current_words[: self._max_words]))
                    current_words = current_words[
                        self._max_words - self._overlap_words :
                    ]

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks
