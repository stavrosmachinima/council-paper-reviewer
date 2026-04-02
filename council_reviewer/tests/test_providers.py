from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from src.council.models import ProviderRequest
from src.council.providers import ProviderError, build_default_router, parse_json_response


def test_resolve_role_prefers_nvidia_logic_judge_by_default(monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_LOGIC_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    router = build_default_router()

    resolved = router.resolve_role("logic_judge")

    assert resolved.provider == "nvidia"
    assert resolved.model == "nvidia/nemotron-3-super-120b-a12b"


def test_legacy_deepseek_and_glm_overrides_fail_fast() -> None:
    router = build_default_router()

    with pytest.raises(ProviderError, match="Legacy DeepSeek/GLM overrides are no longer supported"):
        router.resolve_model_for_value("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

    with pytest.raises(ProviderError, match="Legacy DeepSeek/GLM overrides are no longer supported"):
        router.resolve_role("logic_judge", {"logic_judge": {"provider": "featherless", "model": "zai-org/GLM-4.7"}})


def test_nvidia_health_uses_probe_when_model_listing_unavailable(monkeypatch) -> None:
    router = build_default_router()
    provider = router.provider("nvidia")

    monkeypatch.setenv("NVIDIA_API_KEY", "test")
    monkeypatch.setattr(provider, "list_models", lambda: (_ for _ in ()).throw(requests.ConnectionError("listing blocked")))
    monkeypatch.setattr(provider, "_probe_model", lambda model: None)

    status = provider.health("nvidia/nemotron-3-super-120b-a12b")

    assert status.reachable is True
    assert status.available is True
    assert status.selected_model == "nvidia/nemotron-3-super-120b-a12b"
    assert any("Model listing unavailable" in warning for warning in status.warnings)


def test_nvidia_complete_forwards_top_p_and_extra_body(monkeypatch) -> None:
    router = build_default_router()
    provider = router.provider("nvidia")
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "{\"ok\": true}",
                        }
                    }
                ]
            }

    def fake_post(url, timeout=None, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return FakeResponse()

    monkeypatch.setattr(provider, "_post_with_retries", fake_post)

    response = provider.complete(
        ProviderRequest(
            provider="nvidia",
            model="nvidia/nemotron-3-super-120b-a12b",
            messages=[{"role": "user", "content": "test"}],
            response_format="json",
            temperature=1.0,
            top_p=0.95,
            max_tokens=16000,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
    )

    assert response.content == "{\"ok\": true}"
    assert captured["url"].endswith("/chat/completions")
    assert captured["json"]["top_p"] == 0.95
    assert captured["json"]["chat_template_kwargs"]["enable_thinking"] is True


def test_ollama_complete_forwards_num_predict_top_p_num_ctx_and_timeout(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT", "77")
    router = build_default_router()
    provider = router.provider("ollama")
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "{\"ok\": true}"}}

    def fake_post(url, json=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.council.providers.requests.post", fake_post)

    response = provider.complete(
        ProviderRequest(
            provider="ollama",
            model="qwen3:8b",
            messages=[{"role": "user", "content": "test"}],
            response_format="json",
            temperature=0.2,
            top_p=0.9,
            max_tokens=600,
            extra_body={"options": {"num_ctx": 4096}},
        )
    )

    assert response.content == "{\"ok\": true}"
    assert captured["url"].endswith("/api/chat")
    assert captured["timeout"] == 77
    assert captured["json"]["options"]["num_predict"] == 600
    assert captured["json"]["options"]["top_p"] == 0.9
    assert captured["json"]["options"]["num_ctx"] == 4096


def test_parse_json_response_accepts_wrapped_payloads() -> None:
    parsed = parse_json_response(
        "<think>reasoning omitted</think>\n```json\n{\"tool_calls\":[{\"function\":{\"arguments\":{\"a\":1,\"b\":\"ok\"}}}]}\n```"
    )

    assert parsed == {"a": 1, "b": "ok"}


def test_gemini_list_model_infos_preserves_supported_methods(monkeypatch) -> None:
    router = build_default_router()
    provider = router.provider("gemini")

    def fake_get(url, params=None, timeout=None):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "models": [
                    {
                        "name": "models/gemini-3.1-pro-preview",
                        "supportedGenerationMethods": ["generateContent", "createCachedContent", "countTokens"],
                    }
                ]
            },
        )

    monkeypatch.setattr("src.council.providers.google_genai", None)
    monkeypatch.setattr("src.council.providers.requests.get", fake_get)

    infos = provider.list_model_infos()

    assert len(infos) == 1
    assert infos[0].name == "gemini-3.1-pro-preview"
    assert "createCachedContent" in infos[0].supported_methods


def test_gemini_google_search_request_omits_json_response_mime_type(monkeypatch) -> None:
    router = build_default_router()
    provider = router.provider("gemini")
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "{\"grounded_summary\":\"ok\",\"journal_scope_risks\":[],\"recent_related_work_gaps\":[],\"web_citations\":[]}"
                                }
                            ]
                        }
                    }
                ]
            }

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return FakeResponse()

    monkeypatch.setattr(provider, "_post_with_retries", fake_post)

    response = provider.complete(
        ProviderRequest(
            provider="gemini",
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "test"}],
            response_format="json",
            metadata={"use_google_search": True},
        )
    )

    assert response.content
    assert "responseMimeType" not in captured["json"]["generationConfig"]


def test_gemini_google_search_retries_once_on_timeout(monkeypatch) -> None:
    router = build_default_router()
    provider = router.provider("gemini")
    attempts = {"count": 0}

    class FakeResponse:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "{\"grounded_summary\":\"ok\",\"journal_scope_risks\":[],\"recent_related_work_gaps\":[],\"web_citations\":[]}"
                                }
                            ]
                        }
                    }
                ]
            }

    def fake_post(url, timeout=None, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise requests.ReadTimeout("timed out")
        return FakeResponse()

    monkeypatch.setattr("src.council.providers.requests.post", fake_post)

    response = provider.complete(
        ProviderRequest(
            provider="gemini",
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "test"}],
            response_format="json",
            metadata={"use_google_search": True},
        )
    )

    assert attempts["count"] == 2
    assert "grounded_summary" in response.content
