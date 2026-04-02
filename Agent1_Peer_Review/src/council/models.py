from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ..core.config import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REPORT_FORMATS,
    DEFAULT_TARGET_JOURNAL,
    DEFAULT_WEB_RESEARCH_MODE,
)


def normalize_focus(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",")]
        return [part for part in parts if part]
    return [str(value).strip()]


@dataclass
class DomainCalibration:
    """Per-run domain-specific calibration loaded from the manifest."""

    domain_label: str = ""
    context_king_calibration: str = ""
    web_research_search_focus: str = ""
    logic_judge_calibration: List[str] = field(default_factory=list)
    technical_auditor_calibration: List[str] = field(default_factory=list)
    style_targets: List[str] = field(default_factory=list)
    retrieval_queries: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewManifest:
    manifest_path: Path
    target_repo: Path | None = None
    manuscript_tex: Path | None = None
    manuscript_pdf: Path | None = None
    bibliography: Path | None = None
    reference_pdfs_glob: str | None = None
    journal_guide_pdf: Path | None = None
    target_journal: str = DEFAULT_TARGET_JOURNAL
    review_focus: List[str] = field(default_factory=list)
    provider_overrides: Dict[str, Any] = field(default_factory=dict)
    strict_red_team: bool = True
    report_formats: List[str] = field(default_factory=lambda: list(DEFAULT_REPORT_FORMATS))
    output_root: str = DEFAULT_OUTPUT_ROOT
    web_research_mode: str = DEFAULT_WEB_RESEARCH_MODE
    architect_target_files: List[Path] = field(default_factory=list)
    external_evidence: List["ExternalEvidenceSpec"] = field(default_factory=list)
    domain_calibration: DomainCalibration = field(default_factory=DomainCalibration)
    legacy_manuscript_src: Path | None = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return "repo" if self.target_repo else "legacy_pdf"

    @property
    def focus_summary(self) -> str:
        return ", ".join(self.review_focus) if self.review_focus else "general manuscript quality"

    @property
    def architect_enabled(self) -> bool:
        return bool(self.architect_target_files)

    @property
    def web_research_enabled(self) -> bool:
        return self.web_research_mode.lower() not in {"off", "disabled", "local_only", "none"}

    def resolved_output_root(self) -> Path:
        base = self.manifest_path.parent
        return (base / self.output_root).resolve()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["manifest_path"] = str(self.manifest_path)
        for key in ("target_repo", "manuscript_tex", "manuscript_pdf", "bibliography", "journal_guide_pdf", "legacy_manuscript_src"):
            if data.get(key):
                data[key] = str(data[key])
        data["architect_target_files"] = [str(path) for path in self.architect_target_files]
        data["external_evidence"] = [item.to_dict() for item in self.external_evidence]
        data["domain_calibration"] = self.domain_calibration.to_dict()
        return data


@dataclass
class ExternalEvidenceSpec:
    name: str
    root: Path
    include_paths: List[str] = field(default_factory=list)
    include_globs: List[str] = field(default_factory=lambda: ["**/*"])
    exclude_globs: List[str] = field(default_factory=list)
    source_group: str = "external_evidence"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root),
            "include_paths": list(self.include_paths),
            "include_globs": list(self.include_globs),
            "exclude_globs": list(self.exclude_globs),
            "source_group": self.source_group,
        }


@dataclass
class BundleDocument:
    doc_id: str
    kind: str
    path: str
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "kind": self.kind,
            "path": self.path,
            "title": self.title,
            "character_count": len(self.text),
            "metadata": self.metadata,
        }


@dataclass
class BundleChunk:
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    text: str
    ordinal: int
    heading: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] | None = None

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.text) // 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "title": self.title,
            "ordinal": self.ordinal,
            "heading": self.heading,
            "token_estimate": self.token_estimate,
            "metadata": self.metadata,
        }


@dataclass
class ProviderRequest:
    provider: str
    model: str
    messages: List[Dict[str, str]]
    response_format: str = "text"
    temperature: float = 0.2
    top_p: float | None = None
    max_tokens: int = 4000
    thinking_level: str | None = None
    cached_content: str | None = None
    extra_body: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    provider: str
    model: str
    content: str
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderModelInfo:
    name: str
    supported_methods: List[str] = field(default_factory=list)


@dataclass
class ProviderStatus:
    provider: str
    configured: bool
    reachable: bool | None
    available: bool
    selected_model: str
    visible_models: List[str] = field(default_factory=list)
    selected_model_capabilities: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
