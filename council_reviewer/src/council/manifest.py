from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..core.config import DEFAULT_WEB_RESEARCH_MODE, normalize_provider_overrides
from .models import DomainCalibration, ExternalEvidenceSpec, ReviewManifest, normalize_focus


DEFAULT_REPO_LAYOUT = {
    "manuscript_tex": "paper/main.tex",
    "manuscript_pdf": "paper/main.pdf",
    "bibliography": "paper/references.bib",
    "journal_guide_pdf": "guide.pdf",
    "reference_pdfs_glob": "papers/*.pdf",
    "architect_target_files": ["paper/main.tex"],
}

DEFAULT_EXTERNAL_EVIDENCE: List[Dict[str, Any]] = []


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_path_list(base_dir: Path, values: Iterable[str] | None) -> List[Path]:
    resolved: List[Path] = []
    for value in values or []:
        path = _resolve_path(base_dir, str(value))
        if path:
            resolved.append(path)
    return resolved


def _resolve_external_evidence(base_dir: Path, items: Iterable[Dict[str, Any]] | None) -> List[ExternalEvidenceSpec]:
    resolved_specs: List[ExternalEvidenceSpec] = []
    for index, item in enumerate(items or [], start=1):
        root = _resolve_path(base_dir, str(item.get("root") or ""))
        if root is None:
            continue
        resolved_specs.append(
            ExternalEvidenceSpec(
                name=str(item.get("name") or f"external_{index}"),
                root=root,
                include_paths=[str(path) for path in item.get("include_paths") or []],
                include_globs=[str(glob) for glob in item.get("include_globs") or ["**/*"]],
                exclude_globs=[str(glob) for glob in item.get("exclude_globs") or []],
                source_group=str(item.get("source_group") or "external_evidence"),
            )
        )
    return resolved_specs


def load_manifest(manifest_path: str | Path) -> ReviewManifest:
    manifest_file = Path(manifest_path).resolve()
    with manifest_file.open("r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = json.load(handle)

    manifest_dir = manifest_file.parent
    target_repo = _resolve_path(manifest_dir, payload.get("target_repo"))

    if target_repo:
        manuscript_tex = _resolve_path(target_repo, payload.get("manuscript_tex") or DEFAULT_REPO_LAYOUT["manuscript_tex"])
        manuscript_pdf = _resolve_path(target_repo, payload.get("manuscript_pdf") or DEFAULT_REPO_LAYOUT["manuscript_pdf"])
        bibliography = _resolve_path(target_repo, payload.get("bibliography") or DEFAULT_REPO_LAYOUT["bibliography"])
        journal_guide_pdf = _resolve_path(target_repo, payload.get("journal_guide_pdf") or DEFAULT_REPO_LAYOUT["journal_guide_pdf"])
        reference_pdfs_glob = payload.get("reference_pdfs_glob") or DEFAULT_REPO_LAYOUT["reference_pdfs_glob"]
        architect_targets = _resolve_path_list(
            target_repo,
            payload.get("architect_target_files") or DEFAULT_REPO_LAYOUT["architect_target_files"],
        )
        external_evidence = _resolve_external_evidence(
            manifest_dir,
            payload.get("external_evidence", DEFAULT_EXTERNAL_EVIDENCE),
        )
    else:
        legacy_src = payload.get("manuscript_src")
        manuscript_tex = _resolve_path(manifest_dir, payload.get("manuscript_tex"))
        manuscript_pdf = _resolve_path(manifest_dir, payload.get("manuscript_pdf") or legacy_src)
        bibliography = _resolve_path(manifest_dir, payload.get("bibliography"))
        journal_guide_pdf = _resolve_path(manifest_dir, payload.get("journal_guide_pdf"))
        reference_pdfs_glob = payload.get("reference_pdfs_glob")
        architect_targets = _resolve_path_list(manifest_dir, payload.get("architect_target_files") or [])
        external_evidence = _resolve_external_evidence(manifest_dir, payload.get("external_evidence"))

    review_focus = normalize_focus(payload.get("review_focus") or payload.get("reviewFocus"))
    if not review_focus and payload.get("reviewFocus"):
        review_focus = normalize_focus(payload.get("reviewFocus"))

    target_journal = payload.get("target_journal") or payload.get("publicationOutlets") or "(not specified)"

    raw_cal = payload.get("domain_calibration") or {}
    domain_calibration = DomainCalibration(
        domain_label=str(raw_cal.get("domain_label", "")),
        context_king_calibration=str(raw_cal.get("context_king_calibration", "")),
        web_research_search_focus=str(raw_cal.get("web_research_search_focus", "")),
        logic_judge_calibration=list(raw_cal.get("logic_judge_calibration", [])),
        technical_auditor_calibration=list(raw_cal.get("technical_auditor_calibration", [])),
        style_targets=list(raw_cal.get("style_targets", [])),
        retrieval_queries=dict(raw_cal.get("retrieval_queries", {})),
    )

    return ReviewManifest(
        manifest_path=manifest_file,
        target_repo=target_repo,
        manuscript_tex=manuscript_tex,
        manuscript_pdf=manuscript_pdf,
        bibliography=bibliography,
        reference_pdfs_glob=reference_pdfs_glob,
        journal_guide_pdf=journal_guide_pdf,
        target_journal=target_journal,
        review_focus=review_focus,
        provider_overrides=normalize_provider_overrides(payload.get("provider_overrides", {})),
        strict_red_team=bool(payload.get("strict_red_team", True)),
        report_formats=[fmt.lower() for fmt in payload.get("report_formats", ["json", "md"])],
        output_root=payload.get("output_root", "results"),
        web_research_mode=str(payload.get("web_research_mode", DEFAULT_WEB_RESEARCH_MODE)).strip().lower(),
        architect_target_files=architect_targets,
        external_evidence=external_evidence,
        domain_calibration=domain_calibration,
        legacy_manuscript_src=_resolve_path(manifest_dir, payload.get("manuscript_src")),
        raw=payload,
    )
