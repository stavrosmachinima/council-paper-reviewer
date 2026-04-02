# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Council Paper Reviewer — a multi-agent adversarial peer review system for academic manuscripts. Orchestrates specialized LLM agents across three providers (Google Gemini, NVIDIA Nemotron, local Ollama) to audit papers before human submission.

Domain calibration is configurable via the `domain_calibration` field in `manuscript.json`. A TTS-specific example is provided in `examples/manuscript_tts.example.json`.

## Commands

```bash
# Install dependencies
pip install -r Agent1_Peer_Review/requirements.txt

# Run pre-flight diagnostics (no LLM calls)
cd Agent1_Peer_Review && python3 run_local_aipeer_review.py --doctor-only

# Run full review pipeline
cd Agent1_Peer_Review && python3 run_local_aipeer_review.py

# Run with verbose logging (debug output to stderr)
cd Agent1_Peer_Review && python3 run_local_aipeer_review.py -v

# Resume a failed run (skips completed agents, saves API costs)
cd Agent1_Peer_Review && python3 run_local_aipeer_review.py --resume results/2026-03-28_17-10-51/

# Run with custom manifest
cd Agent1_Peer_Review && python3 run_local_aipeer_review.py --manifest /path/to/manuscript.json

# Run tests
cd Agent1_Peer_Review && pytest

# Run a single test file
cd Agent1_Peer_Review && pytest tests/test_providers.py

# Run a single test
cd Agent1_Peer_Review && pytest tests/test_providers.py::test_function_name -v
```

## Architecture

### Council-Based Multi-Provider Design

The system uses a **council pattern** — each reviewer role is bound to a specific LLM provider via `ROLE_TO_PROVIDER` in `src/core/config.py`. The pipeline is **fail-fast by design**: if any core agent fails, the entire run fails. This ensures the council produces a complete, coherent audit or nothing.

| Role | Provider | Purpose |
|------|----------|---------|
| `context_king` | Gemini | Deep manuscript reasoning with explicit SDK caching |
| `web_research` | Gemini | Web-grounded journal fit and gap analysis |
| `logic_judge` | NVIDIA Nemotron | Hostile logic/statistics audit (whole-paper) |
| `technical_auditor` | NVIDIA Nemotron | Code/math/reproducibility checks |
| `style_scribe` | Ollama (local) | Prose polish, LaTeX cleanup, AI-residue detection |
| `embedding` | Ollama (local) | Semantic chunk embeddings for retrieval |

Legacy role aliases exist: `librarian` → `context_king`, `red_team` → `logic_judge`, `intern` → `style_scribe`.

### Pipeline Flow

1. **Manifest** (`manuscript.json`) defines the review scope — target repo, documents, focus areas, external evidence, domain calibration
2. **Doctor** validates paths, provider health, token budgets before any LLM calls
3. **Bundle** discovers files, extracts text (PDF/LaTeX/CSV/code), chunks with sliding window (2000 chars, 250 overlap), optionally embeds via Ollama
4. **Council passes** run sequentially: Context King (2 cached passes) → Logic Judge → Technical Auditor → Style Scribe
5. **Reporting** generates `CRITIQUE_REPORT.md`, `ARCHITECT_PROMPT.md`, and per-role JSON artifacts in `results/<timestamp>/`

### Key Modules

- **`src/council/runner.py`** — `CouncilRunner` orchestrates the full pipeline. Each `_build_*()` method constructs role-specific prompts, calls the provider, and parses structured JSON responses. Domain calibration is loaded from `manifest.domain_calibration`.
- **`src/council/providers.py`** — `BaseProvider` abstraction with `GeminiProvider`, `NvidiaHostedProvider`, `OllamaProvider`. `ProviderRouter` resolves roles to providers/models. Includes retry logic, rate-limit detection, and `parse_json_response()` for cleaning LLM output.
- **`src/council/bundle.py`** — File discovery, PDF extraction, chunking with section-aware heading propagation, and hybrid retrieval (keyword with stop-word filtering + cosine similarity on embeddings).
- **`src/council/models.py`** — Dataclasses: `ReviewManifest`, `DomainCalibration`, `BundleDocument`, `BundleChunk`, `ProviderRequest`/`ProviderResponse`.
- **`src/council/manifest.py`** — Manifest loading and path resolution. Parses `domain_calibration` from JSON into a `DomainCalibration` instance.
- **`src/council/reporting.py`** — Markdown and JSON report generation, architect handoff prompt.
- **`src/core/config.py`** — Env loading (`.env`/`.env.local`), provider defaults, token limits, role mapping.

### Domain Calibration

Agent prompts can be tuned per-run via the `domain_calibration` block in `manuscript.json`. This reduces false positives for domain-specific review patterns. All calibration fields are optional — omitting them falls back to generic academic review prompts.

Key calibration fields:
- **`context_king_calibration`**: Domain-specific rules appended to the Context King system instruction.
- **`web_research_search_focus`**: Search focus paragraph for the web-grounded pass. Use `{target_journal}` as a placeholder.
- **`logic_judge_calibration`**: Extra bullet-point rules for the Logic Judge.
- **`technical_auditor_calibration`**: Extra bullet-point rules for the Technical Auditor.
- **`style_targets`**: STYLE TARGETS list for the Style Scribe. Use `{target_journal}` as a placeholder.
- **`retrieval_queries`**: Per-role retrieval query lists (`logic_judge`, `technical_auditor`, `style_scribe`).

See `examples/tts_calibration.json` for a complete reference and `examples/manuscript_tts.example.json` for a full manifest using it.

### Non-Obvious Patterns

- **Gemini explicit caching**: Context King creates a paid SDK cache (`Client.caches.create()`) for the full corpus, reuses it across two passes (coherence + rejection risks), then deletes it. This is the most expensive operation.
- **JSON repair loop**: If a provider returns malformed JSON, `runner.py` re-prompts with a repair request and saves debug artifacts to `run_dir/debug/`.
- **Token budget guards**: Context King has a 750K safe limit; Nemotron roles have 700K. The doctor phase validates corpus size against these before calling.
- **Provider-specific system instructions**: NVIDIA gets extra constraints to suppress `<think>` tags and tool-call wrappers. Gemini gets caching instructions.
- **External evidence**: The manifest can include external repos as read-only evidence for the technical auditor via the `external_evidence` field.
- **Graceful degradation**: Web research and style scribe are optional — failures don't block the core review. Embedding fallback to lexical retrieval is logged with reason.
- **Keyword scoring**: Uses stop-word filtering and per-term frequency capping to prevent common words from dominating retrieval results.

## Environment

Copy `.env.example` to `.env` and fill in API keys. Required: `GEMINI_API_KEY`, `NVIDIA_API_KEY`. Ollama must be running locally on port 11434 with the configured models pulled.
