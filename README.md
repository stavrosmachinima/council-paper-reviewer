# Council Paper Reviewer

Multi-agent adversarial peer review system for academic manuscripts. Orchestrates five specialized LLM agents across three providers — Google Gemini, NVIDIA Nemotron, and local Ollama — to audit a paper before human submission.

## The Council

| Role | Model | Provider | Purpose |
|------|-------|----------|---------|
| **Context King** | `gemini-3.1-pro-preview` | Google Gemini | Deep manuscript reasoning with explicit SDK caching. Runs two passes: coherence + rejection risk. |
| **Web Research** | `gemini-2.5-flash` | Google Gemini | Web-grounded journal fit and recent related-work gap analysis. |
| **Logic Judge** | `nemotron-3-super-120b-a12b` | NVIDIA Nemotron | Hostile logic and statistics audit across the full paper. |
| **Technical Auditor** | `nemotron-3-super-120b-a12b` | NVIDIA Nemotron | Code, math, and reproducibility checks. |
| **Style Scribe** | `qwen3:8b` | Ollama (local) | Prose polish, LaTeX cleanup, AI-residue detection. |

## Quick Start

```bash
# 1. Install dependencies
pip install -r Agent1_Peer_Review/requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env — fill in GEMINI_API_KEY and NVIDIA_API_KEY

# 3. Configure your manuscript
cp Agent1_Peer_Review/manuscript.example.json Agent1_Peer_Review/manuscript.json
# Edit manuscript.json — set target_repo, target_journal, and paths

# 4. Start Ollama with required models
ollama pull qwen3:8b && ollama pull gemma3:4b && ollama pull nomic-embed-text

# 5. Run pre-flight diagnostics (no LLM calls)
cd Agent1_Peer_Review && python3 run_local_aipeer_review.py --doctor-only

# 6. Run the full review
cd Agent1_Peer_Review && python3 run_local_aipeer_review.py
```

## Domain Calibration

The council prompts can be tuned for your research domain via a `domain_calibration` block in `manuscript.json`. This lets you inject domain-specific review rules — expected effect size ranges, methodology conventions, formatting targets — without touching Python.

A working example for speech synthesis / TTS research is provided in `examples/manuscript_tts.example.json`. The calibration schema is documented in `Agent1_Peer_Review/README.md`.

## Outputs

All artifacts are written to `Agent1_Peer_Review/results/<timestamp>/`:

- `CRITIQUE_REPORT.md` — consolidated human-readable report
- `ARCHITECT_PROMPT.md` — structured handoff prompt for a follow-up revision pass
- `context_king.json`, `logic_judge.json`, `technical_auditor.json`, `style_scribe.json` — per-role JSON artifacts

## Resume a Failed Run

```bash
cd Agent1_Peer_Review && python3 run_local_aipeer_review.py --resume results/2026-03-28_17-10-51/
```

Skips completed agents and reuses cached LLM responses — saves API costs on partial failures.

## Architecture

See `Agent1_Peer_Review/README.md` for a detailed description of the pipeline, manifest fields, provider configuration, and the non-obvious design patterns (Gemini explicit caching, JSON repair loop, token budget guards).

## Requirements

- Python 3.11+
- [Google AI Studio](https://aistudio.google.com/) API key (Gemini)
- [NVIDIA NIM](https://build.nvidia.com/) API key (Nemotron)
- [Ollama](https://ollama.com/) running locally on port 11434

## License

MIT. Forked from [Agentic-Systems-Lab/rigorous](https://github.com/Agentic-Systems-Lab/rigorous).
