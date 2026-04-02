# council_reviewer

`council_reviewer` is the 2026 adversarial audit runtime. It reviews a local manuscript repository, builds a retrieval-backed evidence bundle, routes work across multiple providers, and produces a pre-Architect critique package.

## Supported Workflow

Run the CLI from this directory:

```bash
python3 run_local_aipeer_review.py --doctor-only
python3 run_local_aipeer_review.py
```

Copy `manuscript.example.json` to `manuscript.json`, fill in your paths, and run.

## Runtime Roles

- `context_king`
  Paid Gemini 3.1 Pro Preview path with explicit caching for global local-bundle reasoning.
- `web_research`
  Gemini 2.5 Flash grounding pass for journal fit and recent related-work gaps.
- `logic_judge`
  NVIDIA Nemotron whole-paper adversarial logic and statistics review.
- `technical_auditor`
  NVIDIA Nemotron whole-paper formulas, methodology/code sync, reproducibility, and leakage checks.
- `style_scribe`
  Ollama local pass for prose, LaTeX, and AI-residue cleanup.
- `architect`
  Manual Claude Code handoff generated as `ARCHITECT_PROMPT.md`.

## Output Artifacts

Each run writes a timestamped directory under `results/`:

- `doctor.json`
- `context_king.json`
- `logic_judge.json`
- `technical_auditor.json`
- `style_scribe.json`
- `CRITIQUE_REPORT.md`
- `ARCHITECT_PROMPT.md`

## Manifest Fields

The primary repo-review manifest fields are:

- `target_repo`
- `manuscript_tex`
- `manuscript_pdf`
- `bibliography`
- `reference_pdfs_glob`
- `journal_guide_pdf`
- `target_journal`
- `review_focus`
- `web_research_mode`
- `architect_target_files`
- `provider_overrides`
- `report_formats`
- `domain_calibration` — optional block for domain-specific prompt tuning (see below)

## Environment

Required remote credentials:

- `GEMINI_API_KEY`
- `NVIDIA_API_KEY`

Recommended model env vars:

- `GEMINI_CONTEXT_MODEL`
- `GEMINI_WEB_MODEL`
- `GEMINI_CONTEXT_CACHE_TTL`
- `NVIDIA_LOGIC_MODEL`
- `NVIDIA_TECHNICAL_MODEL`
- `OLLAMA_STYLE_MODEL`
- `OLLAMA_REWRITE_MODEL`
- `OLLAMA_EMBED_MODEL`

Deprecated aliases are still accepted during migration:

- `GEMINI_MODEL`
- `OLLAMA_CHAT_MODEL`

## Notes

- The core critique is local-bundle grounded.
- The local Context King path is a paid, explicit-cache Gemini flow and hard-fails if SDK caching support is unavailable.
- The web pass is separate and optional; if grounding fails, the run still completes locally.
- The default judge and auditor path is NVIDIA Nemotron whole-paper review, not section batching.
- The Architect step is manual by design for flexibility with different manuscript formats.

## Domain Calibration

The `domain_calibration` manifest field lets you inject domain-specific review rules without modifying Python. All fields are optional — omitting them falls back to generic academic review prompts.

```json
{
  "domain_calibration": {
    "domain_label": "my research area",
    "context_king_calibration": "DOMAIN CALIBRATION — ... rules for your domain ...",
    "web_research_search_focus": "Look for recent ... relevant to {target_journal}.",
    "logic_judge_calibration": ["Domain-specific rule 1.", "Domain-specific rule 2."],
    "technical_auditor_calibration": ["Domain-specific check 1."],
    "style_targets": ["Formatting target 1.", "{target_journal} writing conventions."],
    "retrieval_queries": {
      "logic_judge": ["query 1", "query 2"],
      "technical_auditor": ["query 1"],
      "style_scribe": ["query 1"]
    }
  }
}
```

Use `{target_journal}` as a placeholder in string fields — it is replaced at runtime with the value of `target_journal`.

See `../examples/tts_calibration.json` for a complete reference example.
