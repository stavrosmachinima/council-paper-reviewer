from __future__ import annotations

import csv
import glob
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    import PyPDF2
except Exception:  # pragma: no cover - optional during bootstrap
    PyPDF2 = None

from .models import BundleChunk, BundleDocument, ExternalEvidenceSpec, ReviewManifest


EXCLUDED_BASENAMES = {
    "dev.db",
    "elsarticle.cls",
    "elsarticle-num.bst",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
}
EXCLUDED_PARTS = {".git", "private", "node_modules", "logs", "__pycache__", "build", "dist"}
EXCLUDED_SUFFIXES = {".db", ".pyc", ".wav", ".mp3", ".ogg", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp"}
TEXT_LIKE_SUFFIXES = {
    ".pdf",
    ".csv",
    ".tex",
    ".bib",
    ".py",
    ".md",
    ".txt",
    ".html",
    ".js",
    ".css",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".yml",
    ".yaml",
    ".mako",
    ".lock",
}
TEXT_LIKE_BASENAMES = {
    "Dockerfile",
    "LICENSE",
    "README",
    "README.md",
    "requirements.txt",
    "package.json",
    "package-lock.json",
    "pyproject.toml",
    "docker-compose.yml",
}


def extract_pdf_text(pdf_path: Path) -> str:
    if PyPDF2 is None:
        return f"[PDF content unavailable in bootstrap environment: {pdf_path.name}]"

    text_parts: List[str] = []
    try:
        with pdf_path.open("rb") as handle:
            reader = PyPDF2.PdfReader(handle)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
    except Exception:
        return f"[PDF text extraction failed for {pdf_path.name}; file was still indexed as evidence.]"
    return "\n\n".join(text_parts)


def read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path)
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        # Insert paragraph breaks every 50 rows so the chunker can split large CSVs
        lines = []
        for i, row in enumerate(rows):
            lines.append(", ".join(cell.strip() for cell in row))
            if i > 0 and i % 50 == 0:
                lines.append("")  # empty line = paragraph break for chunker
        return "\n".join(lines)
    return path.read_text(encoding="utf-8", errors="ignore")


def file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".tex":
        return "latex"
    if suffix == ".bib":
        return "bibliography"
    if suffix == ".csv":
        return "dataset"
    if suffix == ".py":
        return "code"
    if suffix in {".md", ".txt"}:
        return "notes"
    return suffix.lstrip(".") or "file"


def _is_indexable_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name in TEXT_LIKE_BASENAMES:
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return path.suffix.lower() in TEXT_LIKE_SUFFIXES


def _is_excluded_path(path: Path, *, relative_to: Path | None = None, exclude_globs: Sequence[str] | None = None) -> bool:
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return True
    if path.name in EXCLUDED_BASENAMES:
        return True
    if path.name.startswith(".env"):
        return True
    if exclude_globs and relative_to is not None:
        relative = path.resolve().relative_to(relative_to.resolve())
        relative_posix = relative.as_posix()
        for pattern in exclude_globs:
            if relative.match(pattern) or Path(relative_posix).match(pattern):
                return True
    return False


def _iter_external_files(spec: ExternalEvidenceSpec) -> List[Path]:
    root = spec.root.resolve()
    discovered: List[Path] = []
    seen: set[Path] = set()

    include_paths = list(spec.include_paths or [])
    if include_paths:
        for include_path in include_paths:
            candidate = (root / include_path).resolve()
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            if _is_excluded_path(resolved, relative_to=root, exclude_globs=spec.exclude_globs):
                continue
            if not _is_indexable_file(resolved):
                continue
            seen.add(resolved)
            discovered.append(resolved)
        return discovered

    for pattern in spec.include_globs or ["**/*"]:
        for candidate in sorted(root.glob(pattern)):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            if _is_excluded_path(resolved, relative_to=root, exclude_globs=spec.exclude_globs):
                continue
            if not _is_indexable_file(resolved):
                continue
            seen.add(resolved)
            discovered.append(resolved)
    return discovered


def source_group(path: Path, manifest: ReviewManifest) -> str:
    resolved = path.resolve()
    if manifest.manuscript_tex and resolved == manifest.manuscript_tex.resolve():
        return "manuscript"
    if manifest.manuscript_pdf and resolved == manifest.manuscript_pdf.resolve():
        return "manuscript"
    if manifest.bibliography and resolved == manifest.bibliography.resolve():
        return "manuscript"
    if manifest.journal_guide_pdf and resolved == manifest.journal_guide_pdf.resolve():
        return "journal_guide"
    if manifest.target_repo:
        repo = manifest.target_repo.resolve()
        try:
            relative = resolved.relative_to(repo)
        except ValueError:
            relative = None
        if relative:
            first = relative.parts[0] if relative.parts else ""
            if first == "papers":
                return "reference_pdf"
            if first == "data":
                return "dataset"
            if first == "scripts":
                return "code"
            if resolved.name in {"README.md", "CLAUDE.md"}:
                return "repo_note"
    for spec in manifest.external_evidence:
        root = spec.root.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return spec.source_group
    if manifest.legacy_manuscript_src and resolved == manifest.legacy_manuscript_src.resolve():
        return "manuscript"
    return "evidence"


def discover_paths(manifest: ReviewManifest) -> List[Path]:
    paths: List[Path] = []
    seen: set[Path] = set()

    def add(path: Path | None) -> None:
        if not path:
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        if _is_excluded_path(resolved):
            return
        seen.add(resolved)
        paths.append(resolved)

    add(manifest.manuscript_tex)
    add(manifest.manuscript_pdf)
    add(manifest.bibliography)
    add(manifest.journal_guide_pdf)

    if manifest.target_repo and manifest.reference_pdfs_glob:
        for candidate in sorted(glob.glob(str((manifest.target_repo / manifest.reference_pdfs_glob).resolve()))):
            add(Path(candidate))

        for relative in ("README.md", "CLAUDE.md"):
            add(manifest.target_repo / relative)

        data_dir = manifest.target_repo / "data"
        if data_dir.exists():
            for candidate in sorted(data_dir.glob("*.csv")):
                add(candidate)

        scripts_dir = manifest.target_repo / "scripts"
        if scripts_dir.exists():
            for candidate in sorted(scripts_dir.glob("*.py")):
                add(candidate)

    if manifest.mode == "legacy_pdf":
        add(manifest.legacy_manuscript_src)

    for spec in manifest.external_evidence:
        if not spec.root.exists():
            continue
        for candidate in _iter_external_files(spec):
            add(candidate)

    return paths


def extract_tex_sections(text: str) -> List[Dict[str, str]]:
    sections: List[Dict[str, str]] = []
    for match in re.finditer(r"\\(section|subsection|subsubsection)\{([^}]+)\}", text):
        sections.append({"level": match.group(1), "title": match.group(2)})
    return sections


def build_documents(manifest: ReviewManifest) -> List[BundleDocument]:
    documents: List[BundleDocument] = []
    for index, path in enumerate(discover_paths(manifest), start=1):
        try:
            text = read_text_file(path)
        except Exception as exc:
            text = f"[Failed to read {path.name}: {exc}]"

        metadata: Dict[str, Any] = {
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "source_group": source_group(path, manifest),
        }
        if path.suffix.lower() == ".tex":
            metadata["sections"] = extract_tex_sections(text)
        documents.append(
            BundleDocument(
                doc_id=f"doc_{index:02d}",
                kind=file_kind(path),
                path=str(path),
                title=path.name,
                text=text,
                metadata=metadata,
            )
        )
    return documents


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    if paragraphs:
        return paragraphs
    return [line.strip() for line in text.splitlines() if line.strip()]


def _find_current_section(text: str, sections: list) -> str | None:
    """Find the last section heading that appears before or within this text."""
    if not sections:
        return None
    best = None
    for section in sections:
        title = section.get("title", "")
        if title and title.lower() in text.lower():
            best = title
    return best


def chunk_document(document: BundleDocument, chunk_size: int = 2000, overlap: int = 250) -> List[BundleChunk]:
    paragraphs = _split_paragraphs(document.text)
    chunks: List[BundleChunk] = []
    buffer = ""
    ordinal = 1
    sections = document.metadata.get("sections", [])
    current_heading = sections[0].get("title") if sections else None

    for paragraph in paragraphs:
        # Track which section we're in based on paragraph content
        heading_match = _find_current_section(paragraph, sections)
        if heading_match:
            current_heading = heading_match

        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        if buffer:
            chunks.append(
                BundleChunk(
                    chunk_id=f"{document.doc_id}_chunk_{ordinal:03d}",
                    doc_id=document.doc_id,
                    source_path=document.path,
                    title=document.title,
                    text=buffer,
                    ordinal=ordinal,
                    heading=current_heading,
                    metadata={"kind": document.kind, "source_group": document.metadata.get("source_group", "evidence")},
                )
            )
            ordinal += 1
            # Preserve overlap WITHOUT stripping — .strip() was destroying
            # the overlap content when it ended with whitespace
            overlap_text = buffer[-overlap:] if len(buffer) > overlap else buffer
            buffer = f"{overlap_text}\n\n{paragraph}".strip() if overlap_text.strip() else paragraph
            continue
        buffer = paragraph

    if buffer:
        chunks.append(
            BundleChunk(
                chunk_id=f"{document.doc_id}_chunk_{ordinal:03d}",
                doc_id=document.doc_id,
                source_path=document.path,
                title=document.title,
                text=buffer,
                ordinal=ordinal,
                heading=current_heading,
                metadata={"kind": document.kind, "source_group": document.metadata.get("source_group", "evidence")},
            )
        )
    return chunks


def attach_embeddings(chunks: Sequence[BundleChunk], embeddings: Sequence[Sequence[float]]) -> None:
    for chunk, vector in zip(chunks, embeddings):
        chunk.embedding = list(vector)


_STOP_WORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "been",
    "have", "has", "had", "not", "but", "which", "their", "than", "its", "also", "can",
    "will", "each", "these", "those", "such", "into", "over", "more", "only", "most",
    "all", "any", "our", "use", "used", "using", "based", "both", "between", "about",
    "does", "did", "may", "should", "would", "could", "when", "where", "how", "what",
})


def keyword_score(query: str, text: str) -> float:
    query_terms = [
        term for term in re.findall(r"[a-z0-9]+", query.lower())
        if len(term) > 2 and term not in _STOP_WORDS
    ]
    if not query_terms:
        return 0.0
    text_lower = text.lower()
    text_term_set = set(re.findall(r"[a-z0-9]+", text_lower))
    score = 0.0
    for term in query_terms:
        if term in text_term_set:
            # Count occurrences but cap per-term contribution to reduce frequency bias
            count = min(text_lower.count(term), 5)
            score += count
    return score / max(1, len(query_terms))


def cosine_similarity(left: Sequence[float] | None, right: Sequence[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def retrieve_chunks(
    chunks: Sequence[BundleChunk],
    query: str,
    *,
    limit: int = 6,
    query_embedding: Sequence[float] | None = None,
    kinds: Sequence[str] | None = None,
    source_groups: Sequence[str] | None = None,
) -> List[BundleChunk]:
    filtered = [
        chunk
        for chunk in chunks
        if (not kinds or chunk.metadata.get("kind") in kinds)
        and (not source_groups or chunk.metadata.get("source_group") in source_groups)
    ]
    scored = []
    for chunk in filtered:
        score = keyword_score(query, chunk.text)
        if query_embedding and chunk.embedding:
            score += cosine_similarity(query_embedding, chunk.embedding) * 5
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for score, chunk in scored[:limit] if score > 0]


def build_bundle(manifest: ReviewManifest) -> Dict[str, Any]:
    documents = build_documents(manifest)
    chunks: List[BundleChunk] = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return {
        "manifest": manifest.to_dict(),
        "documents": documents,
        "chunks": chunks,
        "stats": {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "total_characters": sum(len(document.text) for document in documents),
        },
    }


def serializable_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "manifest": bundle["manifest"],
        "documents": [document.to_manifest() for document in bundle["documents"]],
        "chunks": [chunk.to_dict() for chunk in bundle["chunks"]],
        "stats": bundle["stats"],
    }


def top_level_section_map(bundle: Dict[str, Any]) -> List[Dict[str, str]]:
    for document in bundle["documents"]:
        if document.kind == "latex":
            return document.metadata.get("sections", [])
    return []


def bundle_summary(bundle: Dict[str, Any]) -> Dict[str, Any]:
    documents = bundle["documents"]
    return {
        "documents": [doc.to_manifest() for doc in documents],
        "section_map": top_level_section_map(bundle),
        "document_titles": [doc.title for doc in documents],
        "stats": bundle["stats"],
    }
