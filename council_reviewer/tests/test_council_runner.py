from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests

from run_local_aipeer_review import render_doctor_summary
from src.council.models import ProviderResponse, ProviderStatus
from src.council.providers import ProviderError
from src.council.runner import CouncilRunner


def create_pdf(path: Path, text: str) -> None:
    path.write_bytes(f"%PDF-1.4\n{text}\n%%EOF".encode("utf-8"))


@dataclass
class FakeResolvedRole:
    role: str
    provider: str
    model: str


class FakeRouter:
    def __init__(self):
        self.calls = []
        self.created_caches = []
        self.deleted_caches = []

    def resolve_role(self, role, overrides=None):
        mapping = {
            "context_king": ("gemini", "gemini-3.1-pro-preview"),
            "web_research": ("gemini", "gemini-2.5-flash"),
            "logic_judge": ("nvidia", "nvidia/nemotron-3-super-120b-a12b"),
            "technical_auditor": ("nvidia", "nvidia/nemotron-3-super-120b-a12b"),
            "style_scribe": ("ollama", "qwen3:8b"),
            "embedding": ("ollama", "nomic-embed-text"),
        }
        if role == "style_scribe" and overrides and isinstance(overrides.get("style_scribe"), str):
            override = overrides["style_scribe"]
            if override.startswith("ollama:"):
                return FakeResolvedRole(role=role, provider="ollama", model=override.split(":", 1)[1])
        provider, model = mapping[role]
        return FakeResolvedRole(role=role, provider=provider, model=model)

    def status_for_role(self, role, overrides=None):
        resolved = self.resolve_role(role, overrides)
        capabilities = []
        if role == "context_king":
            capabilities = ["generateContent", "createCachedContent"]
        elif role == "web_research":
            capabilities = ["generateContent"]
        return ProviderStatus(
            provider=resolved.provider,
            configured=True,
            reachable=True,
            available=True,
            selected_model=resolved.model,
            visible_models=[resolved.model],
            selected_model_capabilities=capabilities,
        )

    def context_king_premium_status(self, overrides=None):
        resolved = self.resolve_role("context_king", overrides)
        return {
            "provider": resolved.provider,
            "model": resolved.model,
            "sdk_available": True,
            "supports_generate_content": True,
            "supports_explicit_caching": True,
            "selected_model_capabilities": ["generateContent", "createCachedContent"],
        }

    def create_context_cache(self, *, overrides=None, system_instruction=None, corpus=None, ttl=None, display_name=None):
        handle = {
            "name": f"cache-{len(self.created_caches) + 1}",
            "model": self.resolve_role("context_king", overrides).model,
            "ttl": ttl,
            "display_name": display_name,
            "corpus_length": len(corpus or ""),
        }
        self.created_caches.append(handle)
        return handle

    def delete_context_cache(self, cache_name, *, overrides=None):
        self.deleted_caches.append(cache_name)

    def call(self, request):
        self.calls.append(request)
        role = request.metadata["role"]
        context_pass = request.metadata.get("context_king_pass")
        payloads = {
            "context_king": {
                "manuscript_title": "Fake Title",
                "overview": "A concise overview.",
                "abstract_conclusion_alignment": ["Abstract and conclusion mostly align."],
                "citation_support_risks": ["One claim needs a stronger supporting citation."],
                "venue_checklist": [{"item": "Highlights", "status": "needs_work", "notes": "Tighten wording."}],
                "section_map": [{"title": "Introduction", "notes": "Present."}],
                "related_work_digest": ["Reference coverage is decent."],
                "evidence_pack": [{"path": "/tmp/fake.tex", "reason": "Main manuscript source."}],
            },
            "web_research": {
                "grounded_summary": "The manuscript is broadly in scope but needs stronger positioning against recent work.",
                "journal_scope_risks": ["The contribution framing must emphasize speech-language relevance."],
                "recent_related_work_gaps": [{"paper": "Recent TTS Paper", "gap": "Missing comparison", "why_it_matters": "Reviewers may expect it.", "url": "https://example.com/paper"}],
                "web_citations": [{"title": "Journal scope page", "url": "https://example.com/scope", "reason": "Scope wording"}],
            },
            "logic_judge": {
                "section_summary": "Whole-paper logic review.",
                "logical_chain_audit": ["Logic finding."],
                "statistical_rigor_audit": ["Stats finding."],
                "counter_arguments": ["Counter argument."],
                "consistency_flags": ["Consistency flag."],
                "priority_blockers": [{"severity": "high", "title": "Blocker", "details": "Tighten causal wording."}],
                "priority_rewrites": [{"location": "Abstract", "issue": "Claim is too broad", "instruction": "Narrow the generalization sentence."}],
            },
            "technical_auditor": {
                "section_summary": "Whole-paper technical review.",
                "math_to_prose_sync": ["Math finding."],
                "methodology_reproducibility": ["Repro finding."],
                "data_leakage_risks": ["Leakage finding."],
                "code_result_mismatches": ["Code mismatch."],
                "priority_fixes": [{"location": "Methods", "issue": "Hidden parameter", "instruction": "Document the preprocessing default."}],
            },
            "style_scribe": {
                "rewrite_suggestions": [{"location": "Abstract", "suggestion": "Replace the broad claim with a constrained-compute framing.", "rationale": "Matches the actual study scope."}],
                "latex_cleanup": ["Review long lines in highlights."],
                "narrative_flow": ["Split the discussion into two shorter paragraphs."],
                "ai_residue_flags": ["Replace 'It is important to note' with a direct academic transition."],
            },
        }
        if role == "context_king" and context_pass == "major_rejection_risks":
            payloads["context_king"] = {
                "major_rejection_summary": "The paper overstates the strength of the evidence.",
                "risk_overview": ["Claim framing should be more constrained."],
                "fatal_consistency_gaps": ["The conclusion is slightly stronger than the reported results."],
                "likely_reviewer_rejections": [{"severity": "high", "title": "Overclaim risk", "details": "Tone down the causal language."}],
                "evidence_pack": [{"path": "/tmp/fake.tex", "reason": "Conclusion wording."}],
            }
        return ProviderResponse(provider=request.provider, model=request.model, content=json.dumps(payloads[role]), raw={"role": role})

    def embed_texts(self, texts, overrides=None):
        return [[float(index + 1), float(len(text) % 7 + 1)] for index, text in enumerate(texts)]


class FakeBrokenLogicRouter(FakeRouter):
    def call(self, request):
        self.calls.append(request)
        if request.metadata.get("role") == "logic_judge" and not request.metadata.get("repair_attempt"):
            return ProviderResponse(provider=request.provider, model=request.model, content="<think>hidden</think>\nnot json", raw={})
        if request.metadata.get("role") == "logic_judge" and request.metadata.get("repair_attempt"):
            return ProviderResponse(provider=request.provider, model=request.model, content="still not json", raw={})
        return super().call(request)


class FakeTimeoutLogicRouter(FakeRouter):
    def call(self, request):
        if request.metadata.get("role") == "logic_judge":
            raise TimeoutError("simulated nvidia timeout")
        return super().call(request)


class FakePartialEmbedRouter(FakeRouter):
    def __init__(self):
        super().__init__()
        self.embed_attempts = []

    def embed_texts(self, texts, overrides=None):
        text = texts[0]
        self.embed_attempts.append(len(text))
        if "TRIGGER_EMBED_400" in text and len(text) > 12000:
            response = requests.Response()
            response.status_code = 400
            response._content = b'{"error":"bad request"}'
            raise requests.HTTPError("embedding 400", response=response)
        return super().embed_texts(texts, overrides=overrides)


class FakeBudgetBlockedRouter(FakeRouter):
    def call(self, request):
        if request.metadata.get("role") == "context_king" and request.metadata.get("context_king_pass") == "coherence_check":
            payload = {
                "manuscript_title": "Fake Title",
                "overview": "OVERFLOW " * 120000,
                "abstract_conclusion_alignment": ["Abstract and conclusion mostly align."],
                "citation_support_risks": ["One claim needs a stronger supporting citation."],
                "venue_checklist": [{"item": "Highlights", "status": "needs_work", "notes": "Tighten wording."}],
                "section_map": [{"title": "Introduction", "notes": "Present."}],
                "related_work_digest": ["Reference coverage is decent."],
                "evidence_pack": [{"path": "/tmp/fake.tex", "reason": "Main manuscript source."}],
            }
            return ProviderResponse(provider=request.provider, model=request.model, content=json.dumps(payload), raw={"role": "context_king"})
        if request.metadata.get("role") in {"logic_judge", "technical_auditor"}:
            raise AssertionError("NVIDIA should not be called when the request budget is insufficient")
        return super().call(request)


class FakeStyleRetryRouter(FakeRouter):
    def __init__(self):
        super().__init__()
        self.style_requests = []

    def call(self, request):
        if request.metadata.get("role") == "style_scribe":
            self.calls.append(request)
            self.style_requests.append(request)
            if len(self.style_requests) < 3:
                raise TimeoutError(f"style timeout {len(self.style_requests)}")
            return ProviderResponse(
                provider=request.provider,
                model=request.model,
                content=json.dumps(
                    {
                        "rewrite_suggestions": [{"location": "Abstract", "suggestion": "Tighten claim.", "rationale": "More precise."}],
                        "latex_cleanup": ["Keep caption formatting consistent."],
                        "narrative_flow": ["Split the conclusion paragraph."],
                        "ai_residue_flags": ["Remove generic transition phrasing."],
                    }
                ),
                raw={"role": "style_scribe"},
            )
        return super().call(request)


class FakeStyleAlwaysFailsRouter(FakeRouter):
    def call(self, request):
        if request.metadata.get("role") == "style_scribe":
            self.calls.append(request)
            raise TimeoutError("style timeout")
        return super().call(request)


def write_manifest(path: Path, repo: Path, eval_app: Path | None = None) -> Path:
    payload = {
        "target_repo": str(repo),
        "web_research_mode": "hybrid",
        "external_evidence": [],
    }
    if eval_app is not None:
        payload["external_evidence"] = [
            {
                "name": "eval_app",
                "root": str(eval_app),
                "include_paths": [
                    "README.md",
                    "app/public/views.py",
                    "app/public/models.py",
                    "app/public/forms.py",
                    "app/templates/public/home.html",
                    "assets/js/script.js",
                    "app/data/models.json",
                    "app/data/samples.json",
                ],
                "include_globs": [],
                "exclude_globs": [
                    ".git/**",
                    "node_modules/**",
                    "logs/**",
                    ".env",
                    ".env.*",
                    "data/dev.db",
                    "__pycache__/**",
                    "*.pyc",
                ],
                "source_group": "external_app",
            }
        ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def prepare_repo(tmp_path: Path, *, with_mos_repo: bool = True) -> tuple[Path, Path | None]:
    repo = tmp_path / "paper_repo"
    (repo / "paper").mkdir(parents=True)
    (repo / "papers").mkdir()
    (repo / "data").mkdir()
    (repo / "scripts").mkdir()
    (repo / "paper" / "main.tex").write_text(
        "\\section{Introduction}\nThis is the introduction.\n\n\\section{Results}\nResults are shown here.",
        encoding="utf-8",
    )
    create_pdf(repo / "paper" / "main.pdf", "Main manuscript")
    (repo / "paper" / "references.bib").write_text("@article{ref,title={Ref}}\n", encoding="utf-8")
    (repo / "README.md").write_text("README", encoding="utf-8")
    (repo / "CLAUDE.md").write_text("CLAUDE", encoding="utf-8")
    (repo / "data" / "results_summary.csv").write_text("model,score\nA,3.1\n", encoding="utf-8")
    (repo / "scripts" / "compute_stats.py").write_text("print('stats')\n", encoding="utf-8")
    create_pdf(repo / "papers" / "reference.pdf", "Reference")
    create_pdf(repo / "guide.pdf", "Guide")

    if not with_mos_repo:
        return repo, None

    eval_app = tmp_path / "eval_app"
    (eval_app / "app" / "public").mkdir(parents=True)
    (eval_app / "app" / "templates" / "public").mkdir(parents=True)
    (eval_app / "app" / "data").mkdir(parents=True)
    (eval_app / "assets" / "js").mkdir(parents=True)
    (eval_app / "README.md").write_text("External app", encoding="utf-8")
    (eval_app / "app" / "public" / "views.py").write_text("print('views')\n", encoding="utf-8")
    (eval_app / "app" / "public" / "models.py").write_text("print('models')\n", encoding="utf-8")
    (eval_app / "app" / "public" / "forms.py").write_text("print('forms')\n", encoding="utf-8")
    (eval_app / "app" / "templates" / "public" / "home.html").write_text("<html></html>\n", encoding="utf-8")
    (eval_app / "assets" / "js" / "script.js").write_text("console.log('ui')\n", encoding="utf-8")
    (eval_app / "app" / "data" / "models.json").write_text("{\"models\": []}\n", encoding="utf-8")
    (eval_app / "app" / "data" / "samples.json").write_text("{\"samples\": []}\n", encoding="utf-8")
    return repo, eval_app


def test_council_runner_writes_minimal_artifacts_and_whole_paper_outputs(tmp_path: Path) -> None:
    repo, eval_app = prepare_repo(tmp_path, with_mos_repo=True)
    manifest_path = write_manifest(tmp_path / "manifest.json", repo, eval_app)

    runner = CouncilRunner(router=FakeRouter())
    result = runner.run(manifest_path)

    run_dir = Path(result["run_dir"])
    expected = {
        "doctor.json",
        "context_king.json",
        "logic_judge.json",
        "technical_auditor.json",
        "style_scribe.json",
        "CRITIQUE_REPORT.md",
        "ARCHITECT_PROMPT.md",
    }
    assert {path.name for path in run_dir.iterdir()} == expected

    doctor = json.loads((run_dir / "doctor.json").read_text(encoding="utf-8"))
    assert "text" not in doctor["context_king_corpus"]
    assert set(doctor["provider_statuses"].keys()) == {
        "context_king",
        "web_research",
        "logic_judge",
        "technical_auditor",
        "style_scribe",
        "embedding",
    }
    assert doctor["logic_judge_corpus"]["prompt_token_estimate"] > 0
    assert doctor["technical_auditor_corpus"]["projected_available_output_tokens"] > 0

    logic_judge = json.loads((run_dir / "logic_judge.json").read_text(encoding="utf-8"))
    assert logic_judge["review_mode"] == "whole_paper"
    assert len(logic_judge["section_passes"]) == 1
    assert logic_judge["section_passes"][0]["pass_name"] == "whole_paper"
    assert set(logic_judge["aggregated_findings"].keys()) == {
        "logical_chain_audit",
        "statistical_rigor_audit",
        "counter_arguments",
        "consistency_flags",
        "priority_blockers",
        "priority_rewrites",
    }

    technical_auditor = json.loads((run_dir / "technical_auditor.json").read_text(encoding="utf-8"))
    assert technical_auditor["review_mode"] == "whole_paper"
    assert len(technical_auditor["section_passes"]) == 1
    assert technical_auditor["section_passes"][0]["pass_name"] == "whole_paper"
    assert str(eval_app) in technical_auditor["external_evidence_roots"]


def test_council_runner_writes_debug_artifacts_when_json_repair_fails(tmp_path: Path) -> None:
    repo, _ = prepare_repo(tmp_path, with_mos_repo=False)
    manifest_path = write_manifest(tmp_path / "manifest.json", repo)

    runner = CouncilRunner(router=FakeBrokenLogicRouter())

    with pytest.raises(ProviderError, match="Inspect"):
        runner.run(manifest_path)

    results_root = tmp_path / "results"
    run_dirs = sorted(path for path in results_root.iterdir() if path.is_dir())
    assert run_dirs
    debug_dir = run_dirs[0] / "debug"
    assert debug_dir.exists()
    debug_files = {path.name for path in debug_dir.iterdir()}
    assert any(name.endswith("_raw_response.json") for name in debug_files)
    assert any(name.endswith("_repair_response.json") for name in debug_files)


def test_council_runner_writes_debug_artifact_for_provider_request_failures(tmp_path: Path) -> None:
    repo, _ = prepare_repo(tmp_path, with_mos_repo=False)
    manifest_path = write_manifest(tmp_path / "manifest.json", repo)

    runner = CouncilRunner(router=FakeTimeoutLogicRouter())

    with pytest.raises(ProviderError, match="logic_judge request failed before a response was parsed"):
        runner.run(manifest_path)

    results_root = tmp_path / "results"
    run_dirs = sorted(path for path in results_root.iterdir() if path.is_dir())
    assert run_dirs
    debug_dir = run_dirs[0] / "debug"
    assert debug_dir.exists()
    debug_files = {path.name for path in debug_dir.iterdir()}
    assert any(name.endswith("_request_error.json") for name in debug_files)


def test_embed_failures_only_degrade_problem_chunks(tmp_path: Path) -> None:
    repo, eval_app = prepare_repo(tmp_path, with_mos_repo=True)
    (eval_app / "app" / "public" / "views.py").write_text("TRIGGER_EMBED_400 " * 2000, encoding="utf-8")
    manifest_path = write_manifest(tmp_path / "manifest.json", repo, eval_app)

    router = FakePartialEmbedRouter()
    runner = CouncilRunner(router=router)
    result = runner.run(manifest_path)

    run_dir = Path(result["run_dir"])
    doctor = json.loads((run_dir / "doctor.json").read_text(encoding="utf-8"))
    assert doctor["ready"] is True
    assert 4000 in router.embed_attempts
    style_scribe = json.loads((run_dir / "style_scribe.json").read_text(encoding="utf-8"))
    assert style_scribe["enabled"] is True


def test_budget_overflow_fails_before_nvidia_request(tmp_path: Path) -> None:
    repo, _ = prepare_repo(tmp_path, with_mos_repo=False)
    manifest_path = write_manifest(tmp_path / "manifest.json", repo)

    runner = CouncilRunner(router=FakeBudgetBlockedRouter())

    with pytest.raises(ProviderError, match="cannot fit within the NVIDIA hosted context budget"):
        runner.run(manifest_path)

    results_root = tmp_path / "results"
    run_dirs = sorted(path for path in results_root.iterdir() if path.is_dir())
    assert run_dirs
    debug_dir = run_dirs[0] / "debug"
    assert any(path.name.endswith("_budget_error.json") for path in debug_dir.iterdir())


def test_style_scribe_retries_same_model_then_fallback_model(tmp_path: Path) -> None:
    repo, eval_app = prepare_repo(tmp_path, with_mos_repo=True)
    manifest_path = write_manifest(tmp_path / "manifest.json", repo, eval_app)

    router = FakeStyleRetryRouter()
    runner = CouncilRunner(router=router)
    result = runner.run(manifest_path)

    style_scribe = json.loads((Path(result["run_dir"]) / "style_scribe.json").read_text(encoding="utf-8"))
    assert style_scribe["enabled"] is True
    assert len(router.style_requests) == 3
    assert router.style_requests[0].model == "qwen3:8b"
    assert router.style_requests[0].max_tokens == 1200
    assert router.style_requests[1].model == "qwen3:8b"
    assert router.style_requests[1].max_tokens == 600
    assert router.style_requests[1].extra_body["options"]["num_ctx"] == 4096
    assert router.style_requests[2].model == "gemma3:4b"
    assert router.style_requests[2].max_tokens == 600
    assert router.style_requests[0].messages[1]["content"].count("Path:") <= 4
    assert any("failed" in warning.lower() for warning in style_scribe["warnings"])


def test_style_scribe_all_failures_do_not_abort_run(tmp_path: Path) -> None:
    repo, eval_app = prepare_repo(tmp_path, with_mos_repo=True)
    manifest_path = write_manifest(tmp_path / "manifest.json", repo, eval_app)

    runner = CouncilRunner(router=FakeStyleAlwaysFailsRouter())
    result = runner.run(manifest_path)

    run_dir = Path(result["run_dir"])
    assert (run_dir / "CRITIQUE_REPORT.md").exists()
    style_scribe = json.loads((run_dir / "style_scribe.json").read_text(encoding="utf-8"))
    assert style_scribe["enabled"] is False
    assert style_scribe["warnings"]


def test_render_doctor_summary_is_compact() -> None:
    summary = render_doctor_summary(
        {
            "run_dir": "/tmp/results/2026-03-28_00-00-00",
            "doctor": {
                "ready": False,
                "mode": "repo",
                "target_journal": "Journal of Example Studies",
                "context_king_corpus": {
                    "document_count": 7,
                    "token_estimate": 12345,
                    "within_safe_limit": True,
                },
                "logic_judge_corpus": {"document_count": 3, "token_estimate": 555, "within_safe_limit": True},
                "technical_auditor_corpus": {"document_count": 4, "token_estimate": 777, "within_safe_limit": True},
                "provider_statuses": {
                    "context_king": {"available": False, "selected_model": "gemini-3.1-pro-preview", "premium_path_ready": False},
                    "web_research": {"available": True, "selected_model": "gemini-2.5-flash", "recommended_model": "gemini-2.5-flash"},
                    "logic_judge": {"available": True, "selected_model": "nvidia/nemotron-3-super-120b-a12b"},
                    "technical_auditor": {"available": True, "selected_model": "nvidia/nemotron-3-super-120b-a12b"},
                    "style_scribe": {"available": False, "selected_model": "qwen3:8b"},
                    "embedding": {"available": False, "selected_model": "nomic-embed-text"},
                },
                "missing_actions": ["Install `google-genai` in the runtime used to execute this repo; explicit Gemini caching is SDK-only."],
            },
        }
    )

    assert "Doctor ready: False" in summary
    assert "Provider status:" in summary
    assert "Logic Judge corpus:" in summary
    assert "Technical Auditor corpus:" in summary
    assert '"provider_statuses"' not in summary
