from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_markdown(
    manifest: Dict[str, Any],
    context_king: Dict[str, Any],
    logic_judge: Dict[str, Any],
    technical_auditor: Dict[str, Any],
    style_scribe: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append(f"# Adversarial Audit Report: {context_king.get('manuscript_title', 'Untitled Manuscript')}")
    lines.append("")
    lines.append(f"- Target journal: {manifest.get('target_journal', 'Unknown')}")
    lines.append(f"- Review focus: {', '.join(manifest.get('review_focus', [])) or 'general manuscript quality'}")
    lines.append(f"- Web research mode: {manifest.get('web_research_mode', 'hybrid')}")
    lines.append("")

    cache = context_king.get("cache", {})
    lines.append("## Context King Premium Path")
    lines.append("")
    lines.append(f"- Local cache mode: {cache.get('mode', 'unknown')}")
    lines.append(f"- Local cache model: {cache.get('selected_model', 'unknown')}")
    lines.append(f"- Local cache TTL: {cache.get('ttl', 'unknown')}")
    lines.append(f"- Premium path used successfully: {context_king.get('premium_path_succeeded', False)}")
    lines.append(f"- Cached bundle token estimate: {cache.get('bundle_token_estimate', 'unknown')}")
    if cache.get("warnings"):
        for warning in cache.get("warnings", []):
            lines.append(f"- Warning: {warning}")
    lines.append("")

    lines.append("## Cached Local Coherence Check")
    lines.append("")
    lines.append(context_king.get("overview", "No overview generated."))
    lines.append("")

    lines.append("## Abstract vs Conclusion Alignment")
    lines.append("")
    for item in context_king.get("abstract_conclusion_alignment", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Citation Support Risks")
    lines.append("")
    for item in context_king.get("citation_support_risks", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Venue Checklist")
    lines.append("")
    for item in context_king.get("venue_checklist", []):
        lines.append(f"- [{item.get('status', 'unknown')}] {item.get('item', 'Unspecified')}: {item.get('notes', '')}")
    lines.append("")

    lines.append("## Cached Local Major Rejection Risks")
    lines.append("")
    if context_king.get("major_rejection_summary"):
        lines.append(context_king.get("major_rejection_summary", ""))
    else:
        lines.append("No major rejection summary generated.")
    lines.append("")
    for item in context_king.get("risk_overview", []):
        lines.append(f"- {item}")
    for item in context_king.get("fatal_consistency_gaps", []):
        lines.append(f"- {item}")
    for blocker in context_king.get("likely_reviewer_rejections", []):
        lines.append(f"- **{blocker.get('severity', 'high').upper()}** {blocker.get('title', 'Issue')}: {blocker.get('details', '')}")
    lines.append("")

    lines.append("## Web-Grounded Journal Fit")
    lines.append("")
    web_research = context_king.get("web_research", {})
    if web_research.get("grounded_summary"):
        lines.append(web_research.get("grounded_summary", ""))
    elif web_research.get("warnings"):
        for warning in web_research.get("warnings", []):
            lines.append(f"- {warning}")
    else:
        lines.append("No web-grounded findings generated.")
    lines.append("")

    lines.append("## Journal Scope Risks")
    lines.append("")
    for item in context_king.get("journal_scope_risks", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Recent Related-Work Gaps")
    lines.append("")
    for item in context_king.get("recent_related_work_gaps", []):
        paper = item.get("paper", "Unspecified paper")
        gap = item.get("gap", "")
        why = item.get("why_it_matters", "")
        url = item.get("url", "")
        details = " ".join(part for part in (gap, why) if part).strip()
        lines.append(f"- **{paper}**: {details} {url}".rstrip())
    lines.append("")

    lines.append("## Logic Judge Priority Blockers")
    lines.append("")
    lines.append(f"- Review mode: {logic_judge.get('review_mode', 'unknown')}")
    lines.append(f"- Provider/model: {logic_judge.get('provider', 'unknown')} / {logic_judge.get('model', 'unknown')}")
    if logic_judge.get("corpus_token_estimate") is not None:
        lines.append(f"- Whole-paper token estimate: {logic_judge.get('corpus_token_estimate')}")
    for blocker in logic_judge.get("priority_blockers", []):
        lines.append(f"- **{blocker.get('severity', 'high').upper()}** {blocker.get('title', 'Issue')}: {blocker.get('details', '')}")
    lines.append("")

    lines.append("## Logic Judge Findings")
    lines.append("")
    for item in logic_judge.get("logical_chain_audit", []):
        lines.append(f"- {item}")
    for item in logic_judge.get("statistical_rigor_audit", []):
        lines.append(f"- {item}")
    for item in logic_judge.get("counter_arguments", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Technical Auditor Findings")
    lines.append("")
    lines.append(f"- Review mode: {technical_auditor.get('review_mode', 'unknown')}")
    lines.append(f"- Provider/model: {technical_auditor.get('provider', 'unknown')} / {technical_auditor.get('model', 'unknown')}")
    if technical_auditor.get("corpus_token_estimate") is not None:
        lines.append(f"- Whole-paper token estimate: {technical_auditor.get('corpus_token_estimate')}")
    for root in technical_auditor.get("external_evidence_roots", []):
        lines.append(f"- External MOS evidence: {root}")
    for item in technical_auditor.get("math_to_prose_sync", []):
        lines.append(f"- {item}")
    for item in technical_auditor.get("methodology_reproducibility", []):
        lines.append(f"- {item}")
    for item in technical_auditor.get("data_leakage_risks", []):
        lines.append(f"- {item}")
    for item in technical_auditor.get("code_result_mismatches", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Style Scribe Suggestions")
    lines.append("")
    for suggestion in style_scribe.get("rewrite_suggestions", []):
        location = suggestion.get("location", "Unspecified location")
        recommendation = suggestion.get("suggestion", "")
        lines.append(f"- **{location}**: {recommendation}")
    for item in style_scribe.get("latex_cleanup", []):
        lines.append(f"- {item}")
    for item in style_scribe.get("narrative_flow", []):
        lines.append(f"- {item}")
    for item in style_scribe.get("ai_residue_flags", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Web Citations")
    lines.append("")
    for citation in web_research.get("web_citations", []):
        title = citation.get("title", "Source")
        url = citation.get("url", "")
        reason = citation.get("reason", "")
        lines.append(f"- **{title}**: {reason} {url}".rstrip())
    lines.append("")

    lines.append("## Architect Targets")
    lines.append("")
    for target in manifest.get("architect_target_files", []):
        lines.append(f"- `{target}`")
    lines.append("")

    lines.append("## Evidence Paths")
    lines.append("")
    evidence_paths = sorted(
        set(context_king.get("evidence_paths", []))
        | set(logic_judge.get("evidence_paths", []))
        | set(technical_auditor.get("evidence_paths", []))
        | set(style_scribe.get("evidence_paths", []))
    )
    for evidence in evidence_paths:
        lines.append(f"- `{evidence}`")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_architect_prompt(
    manifest: Dict[str, Any],
    critique_report_path: Path,
    context_king: Dict[str, Any],
    logic_judge: Dict[str, Any],
    technical_auditor: Dict[str, Any],
    style_scribe: Dict[str, Any],
) -> str:
    targets = manifest.get("architect_target_files", [])
    target_list = "\n".join(f"- {target}" for target in targets) if targets else "- No targets configured."
    return f"""Read `{critique_report_path}`.

You are the Architect for a scientific manuscript revision pass. You already have feedback from:

- Context King
- Logic Judge
- Technical Auditor
- Style Scribe

Prioritization rules:

1. Resolve Logic Judge blockers first.
2. Then resolve Technical Auditor reproducibility, math, code/result, and leakage issues.
3. Apply Style Scribe suggestions only when they do not weaken technical precision.
4. Keep the paper appropriate for {manifest.get('target_journal', 'the target journal')}.

Allowed edit targets:

{target_list}

Do not edit Python scripts or other repo files in this Architect pass.
Do not invent new experiments or claims that are not supported by the critique report.
After editing, summarize the concrete manuscript changes you made and list any findings you intentionally left unresolved.
"""


def save_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
