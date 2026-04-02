from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

from ..core.config import DEFAULT_TARGET_JOURNAL, gemini_context_cache_ttl, ollama_rewrite_model
from .bundle import build_bundle, retrieve_chunks
from .manifest import load_manifest
from .models import BundleChunk, ProviderRequest, ReviewManifest
from .providers import ProviderError, ProviderRouter, build_default_router, model_matches, parse_json_response
from .reporting import render_architect_prompt, render_markdown, save_json, save_markdown


CONTEXT_KING_SAFE_TOKEN_LIMIT = 750000
NVIDIA_HOSTED_CONTEXT_LIMIT = 262144
NEMOTRON_REQUEST_SAFETY_TOKENS = 2048
NEMOTRON_PROMPT_OVERHEAD_TOKENS = 512
NEMOTRON_MIN_OUTPUT_TOKENS = 2000
LOGIC_JUDGE_MAX_TOKENS = 4000
TECHNICAL_AUDITOR_MAX_TOKENS = 6000
CONTEXT_KING_TEMPERATURE = 0.1
NEMOTRON_TEMPERATURE = 0.2
NEMOTRON_TOP_P = 0.9
STYLE_SCRIBE_EVIDENCE_CHAR_LIMIT = 5000
STYLE_SCRIBE_RETRY_EVIDENCE_CHAR_LIMIT = 2500
STYLE_SCRIBE_MAX_CHUNKS = 4
STYLE_SCRIBE_CONTEXT_OVERVIEW_CHAR_LIMIT = 700
STYLE_SCRIBE_PRIORITY_REWRITES_LIMIT = 2
STYLE_SCRIBE_PRIMARY_MAX_TOKENS = 1200
STYLE_SCRIBE_RETRY_MAX_TOKENS = 600
STYLE_SCRIBE_RETRY_NUM_CTX = 4096
GEMINI_WEB_LOCAL_EVIDENCE_CHAR_LIMIT = 4500


def _json_prompt(name: str, schema: Dict[str, Any], instructions: str, evidence: str) -> str:
    return f"""{instructions}

Output ONLY a single valid JSON object. No markdown fences, no preamble, no trailing text.
Every field in the schema below MUST be present with the exact field names shown.

Required schema for `{name}`:
{json.dumps(schema, indent=2)}

Evidence bundle:
{evidence}
"""


def _format_chunks(chunks: Sequence[BundleChunk], max_chars: int = 14000) -> str:
    parts: List[str] = []
    total = 0
    for chunk in chunks:
        block = (
            f"[{chunk.chunk_id}] {chunk.title}\n"
            f"Path: {chunk.source_path}\n"
            f"{chunk.text.strip()}\n"
        )
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n---\n".join(parts) if parts else "No evidence retrieved."


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _truncate_text(text: str, limit: int) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _context_king_document_sort_key(document: Any) -> tuple[int, str]:
    ordering = {
        "manuscript": 0,
        "journal_guide": 1,
        "reference_pdf": 2,
        "dataset": 3,
        "code": 4,
        "mos_app": 5,
        "repo_note": 6,
        "evidence": 7,
    }
    source_group = document.metadata.get("source_group", "evidence")
    return ordering.get(source_group, 99), document.path


_GENERIC_RETRIEVAL_QUERIES: Dict[str, List[str]] = {
    "logic_judge": [
        "statistical tests significance effect size confidence interval",
        "results claims causal inference overclaiming counter argument",
        "methodology design threats to validity limitations",
    ],
    "technical_auditor": [
        "reproducibility methodology evaluation protocol configuration",
        "data preprocessing pipeline hyperparameters training setup",
        "code scripts results verification experimental design",
    ],
    "style_scribe": [
        "academic tone prose cleanup hedging language",
        "citation formatting bibliography latex",
        "narrative flow figures table placement",
    ],
}


def _default_queries(manifest: ReviewManifest) -> Dict[str, List[str]]:
    focus = manifest.focus_summary
    cal_queries = manifest.domain_calibration.retrieval_queries
    return {
        "context_king_local": [
            "Cross-check abstract versus conclusion and introduction promises against results.",
            "Citation audit for unsupported claims, contradictory citations, and missing evidence.",
            f"Global consistency, contribution framing, and venue checklist for {manifest.target_journal}.",
            f"Related work positioning for {focus}.",
            "Threats to validity limitations methodological caveats acknowledged by the authors.",
        ],
        "context_king_web": [
            f"{manifest.target_journal} aims and scope rejection risks",
            f"{manifest.target_journal} guide for authors scope highlights abstract keywords",
            f"2025 2026 state of the art missing related work for {focus}",
        ],
        "logic_judge": cal_queries.get("logic_judge") or _GENERIC_RETRIEVAL_QUERIES["logic_judge"],
        "technical_auditor": cal_queries.get("technical_auditor") or _GENERIC_RETRIEVAL_QUERIES["technical_auditor"],
        "style_scribe": cal_queries.get("style_scribe") or _GENERIC_RETRIEVAL_QUERIES["style_scribe"],
    }


class CouncilRunner:
    def __init__(self, router: ProviderRouter | None = None):
        self.router = router or build_default_router()
        self._active_run_dir: Path | None = None
        self._debug_counter = 0

    def load_manifest(self, manifest_path: str | Path) -> ReviewManifest:
        return load_manifest(manifest_path)

    def _run_dir(self, manifest: ReviewManifest) -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return manifest.resolved_output_root() / stamp

    # ------------------------------------------------------------------
    # Checkpoint / resume helpers
    # ------------------------------------------------------------------

    _AGENT_VALIDATORS = {
        "context_king": lambda d: (
            isinstance(d.get("overview"), str) and len(d["overview"]) > 0
            and isinstance(d.get("major_rejection_summary"), str) and len(d["major_rejection_summary"]) > 0
        ),
        "logic_judge": lambda d: (
            d.get("review_mode") == "whole_paper"
            and isinstance(d.get("section_passes"), list) and len(d["section_passes"]) > 0
        ),
        "technical_auditor": lambda d: (
            d.get("review_mode") == "whole_paper"
            and isinstance(d.get("section_passes"), list) and len(d["section_passes"]) > 0
        ),
        "style_scribe": lambda d: "enabled" in d,
    }

    def _load_cached_agent(self, run_dir: Path, agent_name: str) -> Dict[str, Any] | None:
        """Load a previously saved agent result if it exists and passes validation."""
        path = run_dir / f"{agent_name}.json"
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cached %s at %s is corrupt, will re-run: %s", agent_name, path, exc)
            return None

        validator = self._AGENT_VALIDATORS.get(agent_name)
        if validator and not validator(data):
            logger.warning("Cached %s at %s failed completeness check, will re-run", agent_name, path)
            return None

        logger.info("Resuming: using cached %s from %s", agent_name, path)
        return data

    def _doctor_status_dict(self, status) -> Dict[str, Any]:
        data = status.to_dict()
        visible_models = data.pop("visible_models", [])
        data["visible_model_count"] = len(visible_models)
        data["selected_model_visible"] = any(model_matches(data["selected_model"], item) for item in visible_models) if visible_models else None
        return data

    def _select_corpus_documents(
        self,
        bundle: Dict[str, Any],
        *,
        source_groups: Sequence[str] | None = None,
    ) -> List[Any]:
        documents = sorted(bundle["documents"], key=_context_king_document_sort_key)
        if source_groups:
            allowed = set(source_groups)
            documents = [document for document in documents if document.metadata.get("source_group") in allowed]
        has_manuscript_latex = any(
            document.metadata.get("source_group") == "manuscript" and document.kind == "latex"
            for document in documents
        )
        if has_manuscript_latex:
            documents = [
                document
                for document in documents
                if not (document.metadata.get("source_group") == "manuscript" and document.kind == "pdf")
            ]
        return documents

    def _build_document_corpus(self, documents: Sequence[Any], *, include_text: bool = False) -> Dict[str, Any]:
        blocks = []
        for ordinal, document in enumerate(documents, start=1):
            source_group = document.metadata.get("source_group", "evidence")
            block = (
                f"## Document {ordinal}\n"
                f"Path: {document.path}\n"
                f"Title: {document.title}\n"
                f"Kind: {document.kind}\n"
                f"Source Group: {source_group}\n\n"
                f"{document.text.strip()}\n"
            )
            blocks.append(block)
        corpus = "\n\n".join(blocks).strip()
        data = {
            "text": corpus,
            "sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
            "document_count": len(documents),
            "document_paths": [document.path for document in documents],
            "character_count": len(corpus),
            "token_estimate": _estimate_tokens(corpus),
        }
        if not include_text:
            data.pop("text", None)
        return data

    def _build_filtered_corpus(
        self,
        bundle: Dict[str, Any],
        *,
        include_text: bool = False,
        source_groups: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        documents = self._select_corpus_documents(bundle, source_groups=source_groups)
        return self._build_document_corpus(documents, include_text=include_text)

    def _build_context_king_corpus(self, bundle: Dict[str, Any], *, include_text: bool = False) -> Dict[str, Any]:
        return self._build_filtered_corpus(bundle, include_text=include_text)

    def _estimate_rendered_request_tokens(self, provider: str, prompt: str) -> int:
        system_prompt = self._json_system_prompt(provider)
        return _estimate_tokens(system_prompt) + _estimate_tokens(prompt) + NEMOTRON_PROMPT_OVERHEAD_TOKENS

    def _nvidia_request_budget(self, prompt: str, *, max_output_cap: int) -> Dict[str, int | bool]:
        prompt_token_estimate = self._estimate_rendered_request_tokens("nvidia", prompt)
        available_output_tokens = NVIDIA_HOSTED_CONTEXT_LIMIT - prompt_token_estimate - NEMOTRON_REQUEST_SAFETY_TOKENS
        requested_max_tokens = min(max_output_cap, max(0, available_output_tokens))
        return {
            "prompt_token_estimate": prompt_token_estimate,
            "available_output_tokens": available_output_tokens,
            "requested_max_tokens": requested_max_tokens,
            "reserved_safety_tokens": NEMOTRON_REQUEST_SAFETY_TOKENS,
            "minimum_required_output_tokens": NEMOTRON_MIN_OUTPUT_TOKENS,
            "max_output_cap": max_output_cap,
            "within_safe_limit": requested_max_tokens >= NEMOTRON_MIN_OUTPUT_TOKENS,
        }

    def _require_nvidia_request_budget(
        self,
        *,
        role: str,
        schema_name: str,
        prompt: str,
        corpus: Dict[str, Any],
        max_output_cap: int,
    ) -> Dict[str, int | bool]:
        budget = self._nvidia_request_budget(prompt, max_output_cap=max_output_cap)
        if budget["within_safe_limit"]:
            return budget

        artifact_path = self._write_debug_artifact(
            f"{role}_budget_error",
            {
                "error": (
                    f"{role} request would exceed the NVIDIA hosted context window after reserving "
                    f"{NEMOTRON_REQUEST_SAFETY_TOKENS} safety tokens."
                ),
                "role": role,
                "schema_name": schema_name,
                "prompt_token_estimate": budget["prompt_token_estimate"],
                "available_output_tokens": budget["available_output_tokens"],
                "requested_output_cap": max_output_cap,
                "minimum_required_output_tokens": NEMOTRON_MIN_OUTPUT_TOKENS,
                "document_count": corpus["document_count"],
                "document_paths": corpus["document_paths"],
            },
        )
        raise ProviderError(
            f"{role} request cannot fit within the NVIDIA hosted context budget. Inspect {artifact_path}."
        )

    def _embed_bundle(self, bundle: Dict[str, Any], manifest: ReviewManifest) -> Dict[str, Any]:
        embedded = 0
        failed_chunks: List[Dict[str, Any]] = []
        for chunk in bundle["chunks"]:
            try:
                vector = self.router.embed_texts([chunk.text], overrides=manifest.provider_overrides)[0]
            except Exception as first_exc:
                retry_text = chunk.text[: min(4000, max(512, len(chunk.text) // 2))]
                try:
                    vector = self.router.embed_texts([retry_text], overrides=manifest.provider_overrides)[0]
                except Exception as second_exc:
                    failed_chunks.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "source_path": chunk.source_path,
                            "first_error": str(first_exc),
                            "retry_error": str(second_exc),
                        }
                    )
                    logger.warning(
                        "Skipping embedding for %s (%s) after retry: %s",
                        chunk.chunk_id,
                        chunk.source_path,
                        second_exc,
                    )
                    continue
            chunk.embedding = list(vector)
            embedded += 1

        if embedded == 0:
            mode = "lexical_fallback"
        elif failed_chunks:
            mode = "ollama_partial"
        else:
            mode = "ollama"

        return {
            "mode": mode,
            "embedded_chunk_count": embedded,
            "failed_chunk_count": len(failed_chunks),
            "failed_chunks": failed_chunks,
        }

    def _debug_dir(self) -> Path | None:
        if self._active_run_dir is None:
            return None
        debug_dir = self._active_run_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir

    def _write_debug_artifact(self, label: str, payload: Dict[str, Any]) -> Path | None:
        debug_dir = self._debug_dir()
        if debug_dir is None:
            return None
        self._debug_counter += 1
        path = debug_dir / f"{self._debug_counter:02d}_{label}.json"
        save_json(path, payload)
        return path

    def doctor(self, manifest: ReviewManifest) -> Dict[str, Any]:
        path_checks = {
            "manuscript_tex": bool(manifest.manuscript_tex and manifest.manuscript_tex.exists()),
            "manuscript_pdf": bool(manifest.manuscript_pdf and manifest.manuscript_pdf.exists()),
            "bibliography": bool(not manifest.bibliography or manifest.bibliography.exists()),
            "journal_guide_pdf": bool(not manifest.journal_guide_pdf or manifest.journal_guide_pdf.exists()),
        }
        missing_paths = [name for name, ok in path_checks.items() if not ok and name in {"manuscript_pdf", "manuscript_tex"}]
        external_evidence_checks = [
            {
                "name": spec.name,
                "root": str(spec.root),
                "exists": spec.root.exists(),
                "source_group": spec.source_group,
            }
            for spec in manifest.external_evidence
        ]
        missing_external_evidence = [item["name"] for item in external_evidence_checks if not item["exists"]]

        provider_statuses = {}
        for role in ("context_king", "web_research", "logic_judge", "technical_auditor", "style_scribe", "embedding"):
            status = self.router.status_for_role(role, manifest.provider_overrides)
            provider_statuses[role] = self._doctor_status_dict(status)

        context_king_corpus = {
            "available": False,
            "document_count": 0,
            "character_count": 0,
            "token_estimate": 0,
            "safe_token_limit": CONTEXT_KING_SAFE_TOKEN_LIMIT,
            "within_safe_limit": None,
        }
        logic_judge_corpus = {
            "available": False,
            "document_count": 0,
            "character_count": 0,
            "token_estimate": 0,
            "safe_token_limit": NVIDIA_HOSTED_CONTEXT_LIMIT,
            "within_safe_limit": None,
            "prompt_token_estimate": 0,
            "projected_available_output_tokens": 0,
            "requested_output_cap": LOGIC_JUDGE_MAX_TOKENS,
            "minimum_required_output_tokens": NEMOTRON_MIN_OUTPUT_TOKENS,
            "external_evidence_roots": [],
        }
        technical_auditor_corpus = {
            "available": False,
            "document_count": 0,
            "character_count": 0,
            "token_estimate": 0,
            "safe_token_limit": NVIDIA_HOSTED_CONTEXT_LIMIT,
            "within_safe_limit": None,
            "prompt_token_estimate": 0,
            "projected_available_output_tokens": 0,
            "requested_output_cap": TECHNICAL_AUDITOR_MAX_TOKENS,
            "minimum_required_output_tokens": NEMOTRON_MIN_OUTPUT_TOKENS,
            "external_evidence_roots": [item["root"] for item in external_evidence_checks],
        }
        if not missing_paths:
            bundle = build_bundle(manifest)
            corpus_info = self._build_context_king_corpus(bundle)
            context_king_corpus = {
                "available": True,
                **corpus_info,
                "safe_token_limit": CONTEXT_KING_SAFE_TOKEN_LIMIT,
                "within_safe_limit": corpus_info["token_estimate"] <= CONTEXT_KING_SAFE_TOKEN_LIMIT,
            }
            logic_info = self._build_filtered_corpus(
                bundle,
                include_text=True,
                source_groups=("manuscript", "journal_guide", "reference_pdf", "repo_note"),
            )
            logic_budget = self._nvidia_request_budget(
                self._logic_judge_prompt(manifest, {"overview": "", "major_rejection_summary": ""}, logic_info),
                max_output_cap=LOGIC_JUDGE_MAX_TOKENS,
            )
            logic_judge_corpus = {
                "available": True,
                **{key: value for key, value in logic_info.items() if key != "text"},
                "safe_token_limit": NVIDIA_HOSTED_CONTEXT_LIMIT,
                "within_safe_limit": logic_budget["within_safe_limit"],
                "prompt_token_estimate": logic_budget["prompt_token_estimate"],
                "projected_available_output_tokens": logic_budget["available_output_tokens"],
                "requested_output_cap": LOGIC_JUDGE_MAX_TOKENS,
                "minimum_required_output_tokens": NEMOTRON_MIN_OUTPUT_TOKENS,
                "external_evidence_roots": [],
            }
            ext_source_groups = tuple(item["source_group"] for item in external_evidence_checks)
            technical_info = self._build_filtered_corpus(
                bundle,
                include_text=True,
                source_groups=("manuscript", "dataset", "code", "mos_app") + ext_source_groups,
            )
            technical_roots = [item["root"] for item in external_evidence_checks]
            technical_budget = self._nvidia_request_budget(
                self._technical_auditor_prompt(
                    manifest,
                    {"overview": "", "major_rejection_summary": ""},
                    technical_info,
                    technical_roots,
                ),
                max_output_cap=TECHNICAL_AUDITOR_MAX_TOKENS,
            )
            technical_auditor_corpus = {
                "available": True,
                **{key: value for key, value in technical_info.items() if key != "text"},
                "safe_token_limit": NVIDIA_HOSTED_CONTEXT_LIMIT,
                "within_safe_limit": technical_budget["within_safe_limit"],
                "prompt_token_estimate": technical_budget["prompt_token_estimate"],
                "projected_available_output_tokens": technical_budget["available_output_tokens"],
                "requested_output_cap": TECHNICAL_AUDITOR_MAX_TOKENS,
                "minimum_required_output_tokens": NEMOTRON_MIN_OUTPUT_TOKENS,
                "external_evidence_roots": technical_roots,
            }

        premium_status = self.router.context_king_premium_status(manifest.provider_overrides)
        context_status = provider_statuses["context_king"]
        context_status["cache_mode"] = "explicit_gemini_sdk_cache"
        context_status["required_model"] = "gemini-3.1-pro-preview"
        context_status["sdk_available"] = premium_status["sdk_available"]
        context_status["supports_generate_content"] = premium_status["supports_generate_content"]
        context_status["supports_explicit_caching"] = premium_status["supports_explicit_caching"]
        context_status["bundle_token_estimate"] = context_king_corpus["token_estimate"]
        context_status["safe_token_limit"] = CONTEXT_KING_SAFE_TOKEN_LIMIT

        premium_missing_actions: List[str] = []
        premium_ready = context_status["available"]
        if not model_matches("gemini-3.1-pro-preview", context_status["selected_model"]):
            premium_ready = False
            premium_missing_actions.append(
                "Set GEMINI_CONTEXT_MODEL=gemini-3.1-pro-preview for the paid Context King path."
            )
        if not premium_status["sdk_available"]:
            premium_ready = False
            premium_missing_actions.append(
                "Install `google-genai` in the runtime used to execute this repo; explicit Gemini caching is SDK-only."
            )
        if not premium_status["supports_generate_content"]:
            premium_ready = False
            premium_missing_actions.append(
                f"Selected Context King model `{context_status['selected_model']}` does not advertise generateContent support."
            )
        if not premium_status["supports_explicit_caching"]:
            premium_ready = False
            premium_missing_actions.append(
                f"Selected Context King model `{context_status['selected_model']}` does not advertise explicit caching support."
            )
        if context_king_corpus["available"] and not context_king_corpus["within_safe_limit"]:
            premium_ready = False
            premium_missing_actions.append(
                "The normalized Context King corpus exceeds the safe token ceiling; reduce bundle scope before running the premium cached pass."
            )
        context_status["premium_path_ready"] = premium_ready
        context_status["available"] = context_status["available"] and premium_ready
        for action in premium_missing_actions:
            if action not in context_status["missing_actions"]:
                context_status["missing_actions"].append(action)

        web_status = provider_statuses["web_research"]
        web_status["recommended_model"] = "gemini-2.5-flash"
        if not model_matches("gemini-2.5-flash", web_status["selected_model"]):
            web_warning = (
                f"GEMINI_WEB_MODEL is currently `{web_status['selected_model']}`. "
                "The recommended grounded web model is `gemini-2.5-flash` for lower cost."
            )
            if web_warning not in web_status["warnings"]:
                web_status["warnings"].append(web_warning)

        required_roles = ("context_king", "logic_judge", "technical_auditor")
        ready = (
            not missing_paths
            and not missing_external_evidence
            and all(provider_statuses[role]["available"] for role in required_roles)
            and (not logic_judge_corpus["available"] or bool(logic_judge_corpus["within_safe_limit"]))
            and (not technical_auditor_corpus["available"] or bool(technical_auditor_corpus["within_safe_limit"]))
        )

        missing_actions: List[str] = []
        for role, status in provider_statuses.items():
            if role in required_roles or role == "web_research":
                missing_actions.extend(status["missing_actions"])

        if missing_paths:
            missing_actions.extend(f"Missing required file for `{path_name}` in manifest." for path_name in missing_paths)
        if missing_external_evidence:
            missing_actions.extend(
                f"External evidence root for `{name}` is missing; update the manifest or mount that repo before running."
                for name in missing_external_evidence
            )
        if logic_judge_corpus["available"] and not logic_judge_corpus["within_safe_limit"]:
            missing_actions.append(
                "The Logic Judge whole-paper request would leave fewer than 2000 output tokens on Nemotron; reduce logic evidence scope or prompt size before running."
            )
        if technical_auditor_corpus["available"] and not technical_auditor_corpus["within_safe_limit"]:
            missing_actions.append(
                "The Technical Auditor whole-paper request would leave fewer than 2000 output tokens on Nemotron; reduce technical evidence scope or prompt size before running."
            )

        if manifest.web_research_enabled and not provider_statuses["web_research"]["available"]:
            missing_actions.append("Gemini web grounding is unavailable; the run will fall back to a local-only critique.")

        return {
            "target_journal": manifest.target_journal or DEFAULT_TARGET_JOURNAL,
            "mode": manifest.mode,
            "ready": ready,
            "path_checks": path_checks,
            "external_evidence_checks": external_evidence_checks,
            "context_king_corpus": context_king_corpus,
            "logic_judge_corpus": logic_judge_corpus,
            "technical_auditor_corpus": technical_auditor_corpus,
            "provider_statuses": provider_statuses,
            "missing_actions": sorted(set(missing_actions)),
            "notes": [
                "Local embeddings fall back to lexical retrieval if Ollama embeddings are unavailable.",
                "Style Scribe and the Gemini web pass are optional; the core run still proceeds without them.",
                "Claude Architect is a manual handoff artifact and is not part of the Python runtime.",
                "The local Context King path is premium and cache-first; it hard-fails if the paid Gemini SDK+caching path is unavailable.",
                "Nemotron is the default provider for whole-paper Logic Judge and Technical Auditor passes.",
            ],
        }

    def _query_embedding(self, query: str, manifest: ReviewManifest) -> List[float] | None:
        try:
            vectors = self.router.embed_texts([query], overrides=manifest.provider_overrides)
            return vectors[0] if vectors else None
        except Exception as exc:
            logger.debug("Query embedding failed (falling back to lexical): %s", exc)
            return None

    def _retrieve(
        self,
        bundle: Dict[str, Any],
        manifest: ReviewManifest,
        queries: Sequence[str],
        *,
        limit: int = 8,
        kinds: Sequence[str] | None = None,
        source_groups: Sequence[str] | None = None,
    ) -> List[BundleChunk]:
        selected: Dict[str, BundleChunk] = {}
        for query in queries:
            embedding = self._query_embedding(query, manifest)
            for chunk in retrieve_chunks(
                bundle["chunks"],
                query,
                limit=limit,
                query_embedding=embedding,
                kinds=kinds,
                source_groups=source_groups,
            ):
                selected[chunk.chunk_id] = chunk
        return list(selected.values())

    def _call_role(
        self,
        role: str,
        manifest: ReviewManifest,
        schema_name: str,
        schema: Dict[str, Any],
        prompt: str,
        *,
        temperature: float = 0.2,
        top_p: float | None = None,
        max_tokens: int = 4000,
        thinking_level: str | None = None,
        cached_content: str | None = None,
        extra_body: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return self._call_role_with_metadata(
            role,
            manifest,
            prompt,
            metadata={},
            schema_name=schema_name,
            schema=schema,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            thinking_level=thinking_level,
            cached_content=cached_content,
            extra_body=extra_body,
        )

    def _call_role_with_override(
        self,
        role: str,
        manifest: ReviewManifest,
        schema_name: str,
        schema: Dict[str, Any],
        prompt: str,
        *,
        override_value: str | None = None,
        temperature: float = 0.2,
        top_p: float | None = None,
        max_tokens: int = 4000,
        thinking_level: str | None = None,
        cached_content: str | None = None,
        extra_body: Dict[str, Any] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        overrides = dict(manifest.provider_overrides)
        if override_value:
            overrides[role] = override_value
        temp_manifest = replace(manifest, provider_overrides=overrides)
        return self._call_role_with_metadata(
            role,
            temp_manifest,
            prompt,
            metadata=metadata or {},
            schema_name=schema_name,
            schema=schema,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            thinking_level=thinking_level,
            cached_content=cached_content,
            extra_body=extra_body,
        )

    def _json_system_prompt(self, provider: str) -> str:
        prompt = (
            "You are part of an adversarial scientific manuscript audit council. "
            "Be precise, evidence-based, and concise. Return only the final JSON object that matches the requested schema. "
            "Do not include markdown fences, explanations, prose preambles, or trailing commentary. "
            "Every finding MUST cite specific evidence (section numbers, table references, page locations, or quoted text). "
            "Do not generate vague or generic critique — if you cannot point to concrete evidence, do not include the finding."
        )
        if provider == "nvidia":
            prompt += (
                " Keep reasoning internal. Do not emit <think> tags, chain-of-thought, or hidden reasoning text. "
                "Do not simulate tool calls or function wrappers. Return the schema object directly, not `tool_calls`, "
                "`function`, or `arguments` wrappers."
            )
        return prompt

    def _request_debug_metadata(self, request: ProviderRequest, schema_name: str) -> Dict[str, Any]:
        metadata = dict(request.metadata)
        pass_name = metadata.get("section_pass") or metadata.get("context_king_pass") or schema_name
        return {
            "role": metadata.get("role"),
            "provider": request.provider,
            "model": request.model,
            "schema_name": schema_name,
            "pass_name": pass_name,
            "response_format": request.response_format,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "thinking_level": request.thinking_level,
            "cached_content": request.cached_content,
            "extra_body": request.extra_body,
            "metadata": metadata,
        }

    def _repair_prompt(self, schema_name: str, schema: Dict[str, Any], raw_output: str) -> str:
        bounded_output = raw_output.strip()
        if len(bounded_output) > 16000:
            bounded_output = bounded_output[:16000] + "\n\n[truncated]"
        return (
            f"The previous response for `{schema_name}` was not valid JSON.\n\n"
            "Reformat the content below into one valid JSON object only.\n"
            "Do not add explanation, markdown, code fences, or <think> tags.\n"
            "Preserve only the final answer content needed to satisfy the schema.\n\n"
            f"Required schema:\n{json.dumps(schema, indent=2)}\n\n"
            f"Malformed response:\n{bounded_output}\n"
        )

    def _parse_role_json_response(
        self,
        *,
        resolved,
        request: ProviderRequest,
        response,
        schema_name: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            parsed = parse_json_response(response.content)
            parsed["_provider"] = response.provider
            parsed["_model"] = response.model
            return parsed
        except ProviderError as exc:
            raw_artifact = self._write_debug_artifact(
                f"{request.metadata.get('role', 'role')}_raw_response",
                {
                    "error": str(exc),
                    "request": self._request_debug_metadata(request, schema_name),
                    "raw_output": response.content,
                },
            )

        repair_request = ProviderRequest(
            provider=resolved.provider,
            model=response.model or request.model,
            messages=[
                {"role": "system", "content": self._json_system_prompt(resolved.provider)},
                {"role": "user", "content": self._repair_prompt(schema_name, schema, response.content)},
            ],
            response_format="json",
            temperature=0.0,
            max_tokens=request.max_tokens,
            metadata={**request.metadata, "repair_attempt": 1, "schema_name": schema_name},
        )

        try:
            repair_response = self.router.call(repair_request)
        except Exception as repair_exc:
            artifact_path = raw_artifact or self._debug_dir()
            raise ProviderError(
                f"Provider response was not valid JSON and the repair attempt failed. Inspect {artifact_path}."
            ) from repair_exc

        try:
            parsed = parse_json_response(repair_response.content)
        except ProviderError as repair_exc:
            repair_artifact = self._write_debug_artifact(
                f"{request.metadata.get('role', 'role')}_repair_response",
                {
                    "error": str(repair_exc),
                    "request": self._request_debug_metadata(repair_request, schema_name),
                    "raw_output": repair_response.content,
                },
            )
            artifact_path = repair_artifact or raw_artifact or self._debug_dir()
            raise ProviderError(
                f"Provider response did not contain a JSON object after one repair attempt. Inspect {artifact_path}."
            ) from repair_exc

        parsed["_provider"] = repair_response.provider
        parsed["_model"] = repair_response.model
        return parsed

    def _call_role_with_metadata(
        self,
        role: str,
        manifest: ReviewManifest,
        prompt: str,
        metadata: Dict[str, Any],
        schema_name: str,
        schema: Dict[str, Any],
        *,
        temperature: float = 0.2,
        top_p: float | None = None,
        max_tokens: int = 4000,
        thinking_level: str | None = None,
        cached_content: str | None = None,
        extra_body: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        resolved = self.router.resolve_role(role, manifest.provider_overrides)
        request = ProviderRequest(
            provider=resolved.provider,
            model=resolved.model,
            messages=[
                {"role": "system", "content": self._json_system_prompt(resolved.provider)},
                {"role": "user", "content": prompt},
            ],
            response_format="json",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            thinking_level=thinking_level,
            cached_content=cached_content,
            extra_body=dict(extra_body or {}),
            metadata={"role": role, "schema_name": schema_name, **metadata},
        )
        try:
            response = self.router.call(request)
        except Exception as exc:
            error_detail: Dict[str, Any] = {
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "request": self._request_debug_metadata(request, schema_name),
            }
            http_response = getattr(exc, "response", None)
            if http_response is not None:
                try:
                    error_detail["response_body"] = http_response.text
                    error_detail["status_code"] = http_response.status_code
                except Exception:
                    pass
            artifact_path = self._write_debug_artifact(
                f"{role}_request_error",
                error_detail,
            )
            if artifact_path is not None:
                raise ProviderError(
                    f"{role} request failed before a response was parsed. Inspect {artifact_path}."
                ) from exc
            raise
        return self._parse_role_json_response(
            resolved=resolved,
            request=request,
            response=response,
            schema_name=schema_name,
            schema=schema,
        )

    def _context_king_system_instruction(self, manifest: ReviewManifest) -> str:
        base = (
            f"You are the Context King for an adversarial manuscript audit targeting {manifest.target_journal}. "
            "Read the full cached local evidence bundle as one paper package. Be precise, clinical, and evidence-based."
        )
        cal = manifest.domain_calibration.context_king_calibration
        if cal:
            base += "\n\n" + cal
        return base

    def _build_context_king(self, manifest: ReviewManifest, bundle: Dict[str, Any], doctor: Dict[str, Any]) -> Dict[str, Any]:
        corpus = self._build_context_king_corpus(bundle, include_text=True)
        cache_ttl = gemini_context_cache_ttl()
        cache_metadata = {
            "enabled": False,
            "mode": "explicit_gemini_sdk_cache",
            "selected_model": doctor["provider_statuses"]["context_king"]["selected_model"],
            "ttl": cache_ttl,
            "bundle_token_estimate": corpus["token_estimate"],
            "bundle_character_count": corpus["character_count"],
            "document_count": corpus["document_count"],
            "corpus_sha256": corpus["sha256"],
            "cache_name": None,
            "create_succeeded": False,
            "delete_attempted": False,
            "delete_succeeded": False,
            "warnings": [],
        }

        prompt_context = json.dumps(
            {
                "target_journal": manifest.target_journal,
                "review_focus": manifest.review_focus,
                "cache_mode": cache_metadata["mode"],
                "bundle_token_estimate": corpus["token_estimate"],
                "document_count": corpus["document_count"],
                "document_paths": corpus["document_paths"],
            },
            indent=2,
        )
        coherence_schema = {
            "manuscript_title": "",
            "overview": "",
            "abstract_conclusion_alignment": [""],
            "citation_support_risks": [""],
            "venue_checklist": [{"item": "", "status": "pass", "notes": ""}],
            "section_map": [{"title": "", "notes": ""}],
            "related_work_digest": [""],
            "evidence_pack": [{"path": "", "reason": ""}],
        }
        rejection_schema = {
            "major_rejection_summary": "",
            "risk_overview": [""],
            "fatal_consistency_gaps": [""],
            "likely_reviewer_rejections": [{"severity": "high", "title": "", "details": ""}],
            "evidence_pack": [{"path": "", "reason": ""}],
        }
        coherence_prompt = _json_prompt(
            "context_king_coherence_check",
            coherence_schema,
            f"""Act as the Context King for the coherence check pass targeting {manifest.target_journal}.
Audit title-to-conclusion coherence, abstract promises versus actual results, citation support, venue checklist, and section map.
Pay close attention to whether research questions stated in the introduction are fully answered in the discussion.
For the venue checklist, verify scope fit against {manifest.target_journal}'s focus on computational methods for speech and language.
Use only the cached local evidence bundle for this pass.""",
            prompt_context,
        )
        rejection_prompt = _json_prompt(
            "context_king_major_rejection_risks",
            rejection_schema,
            f"""Act as the Context King for the major rejection risk pass targeting {manifest.target_journal}.
Use deep reasoning to identify the strongest blockers, fatal consistency gaps, and likely desk-reject or reviewer-reject reasons.

CRITICAL: Before listing any risk, read the paper's Threats to Validity and Limitations sections.
- If the paper ALREADY acknowledges a limitation, classify it as 'author-acknowledged' in your details and assess
  whether the acknowledgment is sufficient, rather than treating it as a novel flaw.
- Reserve 'high' severity for issues the paper does NOT address or where the authors' framing is misleading.
- Distinguish between design choices (e.g., dataset-model pairing, single-speaker corpus) and genuine flaws
  (e.g., missing baselines, incorrect statistical tests, unsupported causal claims).
Use only the cached local evidence bundle for this pass.""",
            prompt_context,
        )

        cache_handle = None
        try:
            cache_handle = self.router.create_context_cache(
                overrides=manifest.provider_overrides,
                system_instruction=self._context_king_system_instruction(manifest),
                corpus=corpus["text"],
                ttl=cache_ttl,
                display_name=f"context-king-{corpus['sha256'][:12]}",
            )
            cache_metadata["enabled"] = True
            cache_metadata["create_succeeded"] = True
            cache_metadata["cache_name"] = cache_handle.get("name")
            cache_metadata["selected_model"] = cache_handle.get("model", cache_metadata["selected_model"])

            coherence_pass = self._call_role_with_metadata(
                "context_king",
                manifest,
                coherence_prompt,
                metadata={"premium_context_king": True, "context_king_pass": "coherence_check"},
                schema_name="context_king_coherence_check",
                schema=coherence_schema,
                temperature=CONTEXT_KING_TEMPERATURE,
                thinking_level="medium",
                cached_content=cache_handle["name"],
            )
            coherence_pass["pass_name"] = "coherence_check"
            coherence_pass["cache_mode"] = cache_metadata["mode"]
            coherence_pass["thinking_level"] = "medium"
            coherence_pass["temperature"] = CONTEXT_KING_TEMPERATURE

            rejection_pass = self._call_role_with_metadata(
                "context_king",
                manifest,
                rejection_prompt,
                metadata={"premium_context_king": True, "context_king_pass": "major_rejection_risks"},
                schema_name="context_king_major_rejection_risks",
                schema=rejection_schema,
                temperature=CONTEXT_KING_TEMPERATURE,
                thinking_level="high",
                cached_content=cache_handle["name"],
            )
            rejection_pass["pass_name"] = "major_rejection_risks"
            rejection_pass["cache_mode"] = cache_metadata["mode"]
            rejection_pass["thinking_level"] = "high"
            rejection_pass["temperature"] = CONTEXT_KING_TEMPERATURE
        except Exception as exc:
            raise ProviderError(
                "Context King cache creation or premium cached execution failed. "
                "Check paid Gemini access, explicit caching support, and the google-genai SDK. "
                f"Original error: {exc}"
            ) from exc
        finally:
            if cache_handle and cache_handle.get("name"):
                cache_metadata["delete_attempted"] = True
                try:
                    self.router.delete_context_cache(cache_handle["name"], overrides=manifest.provider_overrides)
                    cache_metadata["delete_succeeded"] = True
                except Exception as exc:
                    cache_metadata["warnings"].append(f"Gemini cache cleanup failed: {exc}")

        web_research = {
            "enabled": False,
            "grounded_summary": "",
            "journal_scope_risks": [],
            "recent_related_work_gaps": [],
            "web_citations": [],
            "warnings": [],
        }
        if manifest.web_research_enabled and doctor["provider_statuses"]["web_research"]["available"]:
            web_schema = {
                "grounded_summary": "",
                "journal_scope_risks": [""],
                "recent_related_work_gaps": [{"paper": "", "gap": "", "why_it_matters": "", "url": ""}],
                "web_citations": [{"title": "", "url": "", "reason": ""}],
            }
            web_search_focus = manifest.domain_calibration.web_research_search_focus
            if not web_search_focus:
                web_search_focus = (
                    f"Look for recent related work, methodological advances, and any "
                    f"{manifest.target_journal} editorial guidance relevant to the submission topic. "
                    f"Flag only gaps where a missing reference would weaken the paper's positioning."
                )
            else:
                web_search_focus = web_search_focus.replace("{target_journal}", manifest.target_journal)
            web_instructions = f"""Act as the web-grounded Context King for a submission targeting {manifest.target_journal}.
Use Google Search grounding to assess journal fit and identify recent 2025-2026 related-work gaps.
Do not repeat local-only critique. Cite the grounded sources you relied on in the output JSON.

SEARCH FOCUS: {web_search_focus}"""
            web_context_chunks = self._retrieve(
                bundle,
                manifest,
                _default_queries(manifest)["context_king_web"],
                limit=6,
                source_groups=("manuscript", "reference_pdf"),
            )
            web_evidence = json.dumps(
                {
                    "target_journal": manifest.target_journal,
                    "review_focus": manifest.review_focus,
                    "local_overview": coherence_pass.get("overview", ""),
                    "local_related_work_digest": coherence_pass.get("related_work_digest", []),
                    "major_rejection_summary": rejection_pass.get("major_rejection_summary", ""),
                },
                indent=2,
            ) + "\n\nRelevant local excerpts:\n" + _format_chunks(
                web_context_chunks,
                max_chars=GEMINI_WEB_LOCAL_EVIDENCE_CHAR_LIMIT,
            )
            try:
                grounded = self._call_role_with_metadata(
                    "web_research",
                    manifest,
                    _json_prompt("web_research", web_schema, web_instructions, web_evidence),
                    metadata={"use_google_search": True},
                    schema_name="web_research",
                    schema=web_schema,
                )
                web_research = {
                    **grounded,
                    "enabled": True,
                    "local_evidence_paths": sorted({chunk.source_path for chunk in web_context_chunks}),
                    "warnings": [],
                }
            except Exception as exc:
                logger.warning("Web research failed at runtime: %s", exc)
                web_research = {
                    **web_research,
                    "warnings": [f"Gemini web grounding failed at runtime: {exc}"],
                }
        elif manifest.web_research_enabled:
            logger.warning("Web research unavailable during doctor checks")
            web_research["warnings"].append("Gemini web grounding was unavailable during doctor checks; continuing with local-only critique.")

        evidence_pack = []
        seen_evidence = set()
        for item in coherence_pass.get("evidence_pack", []) + rejection_pass.get("evidence_pack", []):
            marker = (item.get("path", ""), item.get("reason", ""))
            if marker not in seen_evidence:
                evidence_pack.append(item)
                seen_evidence.add(marker)

        return {
            "manuscript_title": coherence_pass.get("manuscript_title", ""),
            "overview": coherence_pass.get("overview", ""),
            "abstract_conclusion_alignment": coherence_pass.get("abstract_conclusion_alignment", []),
            "citation_support_risks": coherence_pass.get("citation_support_risks", []),
            "venue_checklist": coherence_pass.get("venue_checklist", []),
            "section_map": coherence_pass.get("section_map", []),
            "related_work_digest": coherence_pass.get("related_work_digest", []),
            "major_rejection_summary": rejection_pass.get("major_rejection_summary", ""),
            "risk_overview": rejection_pass.get("risk_overview", []),
            "fatal_consistency_gaps": rejection_pass.get("fatal_consistency_gaps", []),
            "likely_reviewer_rejections": rejection_pass.get("likely_reviewer_rejections", []),
            "evidence_pack": evidence_pack,
            "evidence_paths": corpus["document_paths"],
            "local_passes": {
                "coherence_check": coherence_pass,
                "major_rejection_risks": rejection_pass,
            },
            "cache": cache_metadata,
            "premium_path_succeeded": cache_metadata["create_succeeded"],
            "web_research": web_research,
            "journal_scope_risks": web_research.get("journal_scope_risks", []),
            "recent_related_work_gaps": web_research.get("recent_related_work_gaps", []),
        }

    def _logic_judge_prompt(self, manifest: ReviewManifest, context_king: Dict[str, Any], corpus: Dict[str, Any]) -> str:
        evidence = json.dumps(
            {
                "review_mode": "whole_paper",
                "target_journal": manifest.target_journal,
                "context_overview": context_king.get("overview", ""),
                "major_rejection_summary": context_king.get("major_rejection_summary", ""),
                "bundle_token_estimate": corpus["token_estimate"],
                "document_paths": corpus["document_paths"],
            },
            indent=2,
        ) + "\n\nWhole-paper evidence bundle:\n" + corpus["text"]
        schema = {
            "section_summary": "",
            "logical_chain_audit": [""],
            "statistical_rigor_audit": [""],
            "counter_arguments": [""],
            "consistency_flags": [""],
            "priority_blockers": [{"severity": "high", "title": "", "details": ""}],
            "priority_rewrites": [{"location": "", "issue": "", "instruction": ""}],
        }
        instructions = f"""Act as the Logic Judge for a whole-paper adversarial manuscript audit targeting {manifest.target_journal}.
Read the full manuscript-side evidence bundle in one pass.

Focus on:
1. Logical chain: do the conclusions genuinely follow from the reported evidence?
2. Statistical rigor: are significance, effect sizes, and comparisons interpreted correctly?
3. Counter-arguments: what would a hostile reviewer say to weaken the paper's main claims?
4. Consistency: do the prose claims, tables, and reported numbers agree?

Calibration:
- If the paper already acknowledges a limitation in Threats to Validity or Limitations, label it as author-acknowledged."""
        domain_cal = manifest.domain_calibration.logic_judge_calibration
        if domain_cal:
            instructions += "\n" + "\n".join(f"- {line}" for line in domain_cal)
        return _json_prompt("logic_judge", schema, instructions, evidence)

    def _technical_auditor_prompt(self, manifest: ReviewManifest, context_king: Dict[str, Any], corpus: Dict[str, Any], external_roots: Sequence[str]) -> str:
        evidence = json.dumps(
            {
                "review_mode": "whole_paper",
                "target_journal": manifest.target_journal,
                "context_overview": context_king.get("overview", ""),
                "major_rejection_summary": context_king.get("major_rejection_summary", ""),
                "bundle_token_estimate": corpus["token_estimate"],
                "document_paths": corpus["document_paths"],
                "external_evidence_roots": list(external_roots),
            },
            indent=2,
        ) + "\n\nWhole-paper technical evidence bundle:\n" + corpus["text"]
        schema = {
            "section_summary": "",
            "math_to_prose_sync": [""],
            "methodology_reproducibility": [""],
            "data_leakage_risks": [""],
            "code_result_mismatches": [""],
            "priority_fixes": [{"location": "", "issue": "", "instruction": ""}],
        }
        instructions = f"""Act as the Technical Auditor for a whole-paper adversarial manuscript audit targeting {manifest.target_journal}.
Read the full technical bundle in one pass.

Focus on:
1. Math-to-prose sync across equations, variable names, manuscript claims, and reported results.
2. Reproducibility: hyperparameters, preprocessing, evaluation protocol, and hidden defaults.
3. Data leakage and contamination across train, validation, and test/evaluation paths.
4. Code-result mismatches between manuscript statements, CSVs, scripts, and MOS app behavior.

Focus on concrete inconsistencies that could break replication or invalidate reported results."""
        domain_cal = manifest.domain_calibration.technical_auditor_calibration
        if domain_cal:
            instructions += "\n\nDomain-specific calibration:\n" + "\n".join(f"- {line}" for line in domain_cal)
        return _json_prompt("technical_auditor", schema, instructions, evidence)

    def _build_logic_judge_whole_paper(self, manifest: ReviewManifest, bundle: Dict[str, Any], context_king: Dict[str, Any]) -> Dict[str, Any]:
        corpus = self._build_filtered_corpus(
            bundle,
            include_text=True,
            source_groups=("manuscript", "journal_guide", "reference_pdf", "repo_note"),
        )
        schema = {
            "section_summary": "",
            "logical_chain_audit": [""],
            "statistical_rigor_audit": [""],
            "counter_arguments": [""],
            "consistency_flags": [""],
            "priority_blockers": [{"severity": "high", "title": "", "details": ""}],
            "priority_rewrites": [{"location": "", "issue": "", "instruction": ""}],
        }
        prompt = self._logic_judge_prompt(manifest, context_king, corpus)
        budget = self._require_nvidia_request_budget(
            role="logic_judge",
            schema_name="logic_judge",
            prompt=prompt,
            corpus=corpus,
            max_output_cap=LOGIC_JUDGE_MAX_TOKENS,
        )
        findings = self._call_role_with_metadata(
            "logic_judge",
            manifest,
            prompt,
            metadata={
                "section_pass": "whole_paper",
                "review_mode": "whole_paper",
                "prompt_token_estimate": budget["prompt_token_estimate"],
                "requested_max_tokens": budget["requested_max_tokens"],
                "reserved_safety_tokens": budget["reserved_safety_tokens"],
                "document_count": corpus["document_count"],
            },
            schema_name="logic_judge",
            schema=schema,
            temperature=NEMOTRON_TEMPERATURE,
            top_p=NEMOTRON_TOP_P,
            max_tokens=int(budget["requested_max_tokens"]),
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}},
        )
        findings["pass_name"] = "whole_paper"
        findings["focus_label"] = "Whole-paper logic audit"
        findings["review_mode"] = "whole_paper"
        findings["retrieved_chunk_ids"] = []
        findings["retrieved_chunk_count"] = 0
        findings["prompt_evidence_characters"] = corpus["character_count"]
        findings["prompt_evidence_limit"] = NVIDIA_HOSTED_CONTEXT_LIMIT * 4
        findings["corpus_token_estimate"] = corpus["token_estimate"]
        findings["prompt_token_estimate"] = budget["prompt_token_estimate"]
        findings["requested_max_tokens"] = budget["requested_max_tokens"]
        findings["reserved_safety_tokens"] = budget["reserved_safety_tokens"]
        findings["evidence_paths"] = list(corpus["document_paths"])
        aggregated = {
            "logical_chain_audit": findings.get("logical_chain_audit", []),
            "statistical_rigor_audit": findings.get("statistical_rigor_audit", []),
            "counter_arguments": findings.get("counter_arguments", []),
            "consistency_flags": findings.get("consistency_flags", []),
            "priority_blockers": findings.get("priority_blockers", []),
            "priority_rewrites": findings.get("priority_rewrites", []),
        }
        return {
            "review_mode": "whole_paper",
            "section_passes": [findings],
            "aggregated_findings": aggregated,
            "logical_chain_audit": aggregated["logical_chain_audit"],
            "statistical_rigor_audit": aggregated["statistical_rigor_audit"],
            "counter_arguments": aggregated["counter_arguments"],
            "consistency_flags": aggregated["consistency_flags"],
            "priority_blockers": aggregated["priority_blockers"],
            "priority_rewrites": aggregated["priority_rewrites"],
            "evidence_paths": list(corpus["document_paths"]),
            "provider": findings.get("_provider", ""),
            "model": findings.get("_model", ""),
            "corpus_token_estimate": corpus["token_estimate"],
            "prompt_token_estimate": budget["prompt_token_estimate"],
            "requested_max_tokens": budget["requested_max_tokens"],
            "reserved_safety_tokens": budget["reserved_safety_tokens"],
            "external_evidence_roots": [],
        }

    def _build_logic_judge(self, manifest: ReviewManifest, bundle: Dict[str, Any], context_king: Dict[str, Any]) -> Dict[str, Any]:
        return self._build_logic_judge_whole_paper(manifest, bundle, context_king)

    def _build_technical_auditor_whole_paper(self, manifest: ReviewManifest, bundle: Dict[str, Any], context_king: Dict[str, Any]) -> Dict[str, Any]:
        external_source_groups = tuple(spec.source_group for spec in manifest.external_evidence)
        corpus = self._build_filtered_corpus(
            bundle,
            include_text=True,
            source_groups=("manuscript", "dataset", "code", "mos_app") + external_source_groups,
        )
        schema = {
            "section_summary": "",
            "math_to_prose_sync": [""],
            "methodology_reproducibility": [""],
            "data_leakage_risks": [""],
            "code_result_mismatches": [""],
            "priority_fixes": [{"location": "", "issue": "", "instruction": ""}],
        }
        external_roots = [str(spec.root) for spec in manifest.external_evidence]
        prompt = self._technical_auditor_prompt(manifest, context_king, corpus, external_roots)
        budget = self._require_nvidia_request_budget(
            role="technical_auditor",
            schema_name="technical_auditor",
            prompt=prompt,
            corpus=corpus,
            max_output_cap=TECHNICAL_AUDITOR_MAX_TOKENS,
        )
        findings = self._call_role_with_metadata(
            "technical_auditor",
            manifest,
            prompt,
            metadata={
                "section_pass": "whole_paper",
                "review_mode": "whole_paper",
                "prompt_token_estimate": budget["prompt_token_estimate"],
                "requested_max_tokens": budget["requested_max_tokens"],
                "reserved_safety_tokens": budget["reserved_safety_tokens"],
                "document_count": corpus["document_count"],
            },
            schema_name="technical_auditor",
            schema=schema,
            temperature=NEMOTRON_TEMPERATURE,
            top_p=NEMOTRON_TOP_P,
            max_tokens=int(budget["requested_max_tokens"]),
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}},
        )
        findings["pass_name"] = "whole_paper"
        findings["focus_label"] = "Whole-paper technical audit"
        findings["review_mode"] = "whole_paper"
        findings["retrieved_chunk_ids"] = []
        findings["retrieved_chunk_count"] = 0
        findings["prompt_evidence_characters"] = corpus["character_count"]
        findings["prompt_evidence_limit"] = NVIDIA_HOSTED_CONTEXT_LIMIT * 4
        findings["corpus_token_estimate"] = corpus["token_estimate"]
        findings["prompt_token_estimate"] = budget["prompt_token_estimate"]
        findings["requested_max_tokens"] = budget["requested_max_tokens"]
        findings["reserved_safety_tokens"] = budget["reserved_safety_tokens"]
        findings["evidence_paths"] = list(corpus["document_paths"])
        aggregated = {
            "math_to_prose_sync": findings.get("math_to_prose_sync", []),
            "methodology_reproducibility": findings.get("methodology_reproducibility", []),
            "data_leakage_risks": findings.get("data_leakage_risks", []),
            "code_result_mismatches": findings.get("code_result_mismatches", []),
            "priority_fixes": findings.get("priority_fixes", []),
        }
        return {
            "review_mode": "whole_paper",
            "section_passes": [findings],
            "aggregated_findings": aggregated,
            "math_to_prose_sync": aggregated["math_to_prose_sync"],
            "methodology_reproducibility": aggregated["methodology_reproducibility"],
            "data_leakage_risks": aggregated["data_leakage_risks"],
            "code_result_mismatches": aggregated["code_result_mismatches"],
            "priority_fixes": aggregated["priority_fixes"],
            "evidence_paths": list(corpus["document_paths"]),
            "provider": findings.get("_provider", ""),
            "model": findings.get("_model", ""),
            "corpus_token_estimate": corpus["token_estimate"],
            "prompt_token_estimate": budget["prompt_token_estimate"],
            "requested_max_tokens": budget["requested_max_tokens"],
            "reserved_safety_tokens": budget["reserved_safety_tokens"],
            "external_evidence_roots": external_roots,
        }

    def _build_technical_auditor(self, manifest: ReviewManifest, bundle: Dict[str, Any], context_king: Dict[str, Any]) -> Dict[str, Any]:
        return self._build_technical_auditor_whole_paper(manifest, bundle, context_king)

    def _disabled_style_scribe(self, warning: str, evidence_paths: Sequence[str] | None = None, attempts: Sequence[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        return {
            "enabled": False,
            "warnings": [warning],
            "rewrite_suggestions": [],
            "latex_cleanup": [],
            "narrative_flow": [],
            "ai_residue_flags": [],
            "evidence_paths": sorted(set(evidence_paths or [])),
            "attempts": list(attempts or []),
        }

    def _build_style_scribe_prompt(
        self,
        manifest: ReviewManifest,
        context_king: Dict[str, Any],
        logic_judge: Dict[str, Any],
        evidence_chunks: Sequence[BundleChunk],
        *,
        evidence_char_limit: int,
    ) -> str:
        schema = {
            "rewrite_suggestions": [{"location": "", "suggestion": "", "rationale": ""}],
            "latex_cleanup": [""],
            "narrative_flow": [""],
            "ai_residue_flags": [""],
        }
        instructions = f"""Act as the Style Scribe for an adversarial manuscript audit targeting {manifest.target_journal}.
Focus on prose polish only: academic tone, AI-residue removal, LaTeX/BibTeX sanity, and narrative flow.
Do not make final scientific judgments and do not override Logic Judge blockers.

STYLE TARGETS:"""
        style_targets = manifest.domain_calibration.style_targets
        if style_targets:
            for target in style_targets:
                instructions += "\n- " + target.replace("{target_journal}", manifest.target_journal)
        else:
            instructions += (
                f"\n- Flag hedging language that weakens otherwise well-supported claims ('seems to suggest', 'might indicate')."
                f"\n- Flag AI-residue phrases: 'delve into', 'it is worth noting that', 'in the realm of', 'leverage',"
                f" 'holistic', 'paradigm shift', 'the landscape of', 'a testament to'."
                f"\n- {manifest.target_journal} expects precise technical writing; prefer short, concrete rewrite advice."
            )
        evidence = json.dumps(
            {
                "context_overview": _truncate_text(
                    context_king.get("overview", ""),
                    STYLE_SCRIBE_CONTEXT_OVERVIEW_CHAR_LIMIT,
                ),
                "logic_priority_rewrites": logic_judge.get("priority_rewrites", [])[:STYLE_SCRIBE_PRIORITY_REWRITES_LIMIT],
            },
            indent=2,
        ) + "\n\nRelevant excerpts:\n" + _format_chunks(evidence_chunks, max_chars=evidence_char_limit)
        return _json_prompt("style_scribe", schema, instructions, evidence)

    def _build_style_scribe(self, manifest: ReviewManifest, bundle: Dict[str, Any], context_king: Dict[str, Any], logic_judge: Dict[str, Any], doctor: Dict[str, Any]) -> Dict[str, Any]:
        if not doctor["provider_statuses"]["style_scribe"]["available"]:
            return self._disabled_style_scribe("Style Scribe was unavailable during this run; skipping local prose cleanup.")

        evidence_chunks = self._retrieve(
            bundle,
            manifest,
            _default_queries(manifest)["style_scribe"],
            limit=4,
            source_groups=("manuscript",),
        )[:STYLE_SCRIBE_MAX_CHUNKS]
        evidence_paths = sorted({chunk.source_path for chunk in evidence_chunks})
        schema = {
            "rewrite_suggestions": [{"location": "", "suggestion": "", "rationale": ""}],
            "latex_cleanup": [""],
            "narrative_flow": [""],
            "ai_residue_flags": [""],
        }
        primary_model = self.router.resolve_role("style_scribe", manifest.provider_overrides).model
        configured_rewrite_model = ollama_rewrite_model()
        fallback_override = None if model_matches(primary_model, configured_rewrite_model) else f"ollama:{configured_rewrite_model}"
        attempts = [
            {
                "label": "primary",
                "override_value": None,
                "max_tokens": STYLE_SCRIBE_PRIMARY_MAX_TOKENS,
                "evidence_char_limit": STYLE_SCRIBE_EVIDENCE_CHAR_LIMIT,
                "num_ctx": None,
            },
            {
                "label": "same_model_reduced_budget",
                "override_value": None,
                "max_tokens": STYLE_SCRIBE_RETRY_MAX_TOKENS,
                "evidence_char_limit": STYLE_SCRIBE_RETRY_EVIDENCE_CHAR_LIMIT,
                "num_ctx": STYLE_SCRIBE_RETRY_NUM_CTX,
            },
        ]
        if fallback_override is not None:
            attempts.append(
                {
                    "label": "fallback_model_reduced_budget",
                    "override_value": fallback_override,
                    "max_tokens": STYLE_SCRIBE_RETRY_MAX_TOKENS,
                    "evidence_char_limit": STYLE_SCRIBE_RETRY_EVIDENCE_CHAR_LIMIT,
                    "num_ctx": STYLE_SCRIBE_RETRY_NUM_CTX,
                }
            )

        warnings: List[str] = []
        attempt_summaries: List[Dict[str, Any]] = []
        for attempt in attempts:
            prompt = self._build_style_scribe_prompt(
                manifest,
                context_king,
                logic_judge,
                evidence_chunks,
                evidence_char_limit=attempt["evidence_char_limit"],
            )
            try:
                style_scribe = self._call_role_with_override(
                    "style_scribe",
                    manifest,
                    "style_scribe",
                    schema,
                    prompt,
                    override_value=attempt["override_value"],
                    max_tokens=attempt["max_tokens"],
                    extra_body={"options": {"num_ctx": attempt["num_ctx"]}} if attempt["num_ctx"] else None,
                    metadata={
                        "style_attempt": attempt["label"],
                        "style_retry": attempt["label"] != "primary",
                        "style_prompt_tokens": _estimate_tokens(prompt),
                    },
                )
                style_scribe["enabled"] = True
                style_scribe["warnings"] = warnings
                style_scribe["evidence_paths"] = evidence_paths
                style_scribe["attempts"] = attempt_summaries + [
                    {
                        "label": attempt["label"],
                        "model": style_scribe.get("_model", primary_model),
                        "max_tokens": attempt["max_tokens"],
                        "num_ctx": attempt["num_ctx"],
                        "status": "succeeded",
                    }
                ]
                return style_scribe
            except Exception as exc:
                model_name = primary_model if attempt["override_value"] is None else attempt["override_value"].split(":", 1)[1]
                warnings.append(f"Style Scribe attempt `{attempt['label']}` failed on `{model_name}`: {exc}")
                attempt_summaries.append(
                    {
                        "label": attempt["label"],
                        "model": model_name,
                        "max_tokens": attempt["max_tokens"],
                        "num_ctx": attempt["num_ctx"],
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                logger.warning("Style Scribe attempt `%s` failed: %s", attempt["label"], exc)

        result = self._disabled_style_scribe(
            "Style Scribe failed at runtime; continuing without local prose cleanup.",
            evidence_paths=evidence_paths,
            attempts=attempt_summaries + [{"label": "summary", "status": "disabled"}],
        )
        result["warnings"] = warnings or ["Style Scribe failed at runtime; continuing without local prose cleanup."]
        return result

    def run(
        self,
        manifest_path: str | Path,
        *,
        doctor_only: bool = False,
        resume_dir: Path | None = None,
    ) -> Dict[str, Any]:
        manifest = self.load_manifest(manifest_path)
        if resume_dir is not None:
            run_dir = Path(resume_dir)
            if not run_dir.exists():
                raise ProviderError(f"Resume directory does not exist: {run_dir}")
            logger.info("Resuming from %s", run_dir)
        else:
            run_dir = self._run_dir(manifest)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._active_run_dir = run_dir
        self._debug_counter = 0

        try:
            # Doctor always re-runs (cheap, validates current environment)
            doctor = self.doctor(manifest)
            save_json(run_dir / "doctor.json", doctor)
            if doctor_only:
                return {"run_dir": str(run_dir), "doctor": doctor}
            if not doctor["ready"]:
                raise ProviderError(
                    f"Council doctor check failed. See {run_dir / 'doctor.json'} for exact missing actions."
                )

            # Bundle always re-builds (cheap, manuscript may have changed)
            bundle = build_bundle(manifest)
            try:
                embedding_status = self._embed_bundle(bundle, manifest)
                bundle["stats"]["embedding_mode"] = embedding_status["mode"]
                bundle["stats"]["embedded_chunk_count"] = embedding_status["embedded_chunk_count"]
                bundle["stats"]["embedding_failed_chunk_count"] = embedding_status["failed_chunk_count"]
                if embedding_status["failed_chunks"]:
                    bundle["stats"]["embedding_failed_chunks"] = embedding_status["failed_chunks"]
                logger.info(
                    "Embedded %d/%d chunks via Ollama (%s)",
                    embedding_status["embedded_chunk_count"],
                    len(bundle["chunks"]),
                    embedding_status["mode"],
                )
            except Exception as exc:
                bundle["stats"]["embedding_mode"] = "lexical_fallback"
                bundle["stats"]["embedding_fallback_reason"] = str(exc)
                logger.warning("Embedding unavailable, falling back to lexical retrieval: %s", exc)

            # --- Checkpoint/resume pipeline ---
            # Once any agent re-runs, all downstream agents must also re-run
            # because their input (context_king output, etc.) may have changed.
            invalidated = resume_dir is None  # fresh run = no caching

            context_king = None if invalidated else self._load_cached_agent(run_dir, "context_king")
            if context_king is None:
                invalidated = True
                context_king = self._build_context_king(manifest, bundle, doctor)
                save_json(run_dir / "context_king.json", context_king)

            logic_judge = None if invalidated else self._load_cached_agent(run_dir, "logic_judge")
            if logic_judge is None:
                invalidated = True
                logic_judge = self._build_logic_judge(manifest, bundle, context_king)
                save_json(run_dir / "logic_judge.json", logic_judge)

            technical_auditor = None if invalidated else self._load_cached_agent(run_dir, "technical_auditor")
            if technical_auditor is None:
                invalidated = True
                technical_auditor = self._build_technical_auditor(manifest, bundle, context_king)
                save_json(run_dir / "technical_auditor.json", technical_auditor)

            style_scribe = None if invalidated else self._load_cached_agent(run_dir, "style_scribe")
            if style_scribe is None:
                style_scribe = self._build_style_scribe(manifest, bundle, context_king, logic_judge, doctor)
                save_json(run_dir / "style_scribe.json", style_scribe)

            critique_report = render_markdown(
                manifest.to_dict(),
                context_king,
                logic_judge,
                technical_auditor,
                style_scribe,
            )
            critique_report_path = run_dir / "CRITIQUE_REPORT.md"
            save_markdown(critique_report_path, critique_report)

            architect_prompt = render_architect_prompt(
                manifest.to_dict(),
                critique_report_path,
                context_king,
                logic_judge,
                technical_auditor,
                style_scribe,
            )
            architect_prompt_path = run_dir / "ARCHITECT_PROMPT.md"
            save_markdown(architect_prompt_path, architect_prompt)

            return {
                "run_dir": str(run_dir),
                "doctor": doctor,
                "bundle_stats": bundle["stats"],
                "context_king": context_king,
                "logic_judge": logic_judge,
                "technical_auditor": technical_auditor,
                "style_scribe": style_scribe,
                "critique_report_path": str(critique_report_path),
                "architect_prompt_path": str(architect_prompt_path),
            }
        finally:
            self._active_run_dir = None
