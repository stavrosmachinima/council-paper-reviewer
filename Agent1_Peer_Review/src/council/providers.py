from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from json import JSONDecodeError, JSONDecoder
from typing import Any, Dict, List, Mapping, Sequence

import requests

from ..core.config import (
    PROVIDER_DEFAULTS,
    REQUIRED_OLLAMA_MODELS,
    ROLE_TO_PROVIDER,
    canonical_role_name,
    gemini_web_model,
    normalize_provider_overrides,
    ollama_request_timeout,
    role_default_model,
)
from .models import ProviderModelInfo, ProviderRequest, ProviderResponse, ProviderStatus

try:
    from google import genai as google_genai
except Exception:  # pragma: no cover - optional dependency
    google_genai = None


REQUEST_TIMEOUT = 30
GEMINI_GROUNDED_TIMEOUT = 120
GEMINI_RATE_LIMIT_RETRIES = 3
GEMINI_NETWORK_RETRIES = 1
NVIDIA_REQUEST_TIMEOUT = int(os.getenv("NVIDIA_REQUEST_TIMEOUT", "900"))
OPENAI_COMPATIBLE_RETRIES = 2
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    pass


def _is_quota_or_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower()


def _is_model_not_found_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 404:
        return True
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 404:
        return True
    text = str(exc)
    return "404" in text or "NOT_FOUND" in text or "not found" in text.lower()


def _retry_delay_seconds(response: requests.Response | None, attempt: int, *, ceiling: float = 12.0) -> float:
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return min(ceiling, 2.0 * (attempt + 1))


def _extract_text_from_content_parts(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if not isinstance(item, Mapping):
                continue
            item_type = str(item.get("type", "")).lower()
            text_value = item.get("text")
            if isinstance(text_value, str) and text_value.strip():
                parts.append(text_value)
                continue
            if item_type == "output_text":
                value = item.get("content") or item.get("text")
                if isinstance(value, str) and value.strip():
                    parts.append(value)
        return "\n".join(part for part in parts if part).strip()
    return ""


def _extract_tool_call_arguments(payload: Any) -> str:
    if isinstance(payload, list):
        for item in payload:
            extracted = _extract_tool_call_arguments(item)
            if extracted:
                return extracted
        return ""
    if not isinstance(payload, Mapping):
        return ""
    function = payload.get("function")
    if isinstance(function, Mapping):
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments.strip():
            return arguments
        if isinstance(arguments, Mapping):
            return json.dumps(arguments)
    arguments = payload.get("arguments")
    if isinstance(arguments, str) and arguments.strip():
        return arguments
    if isinstance(arguments, Mapping):
        return json.dumps(arguments)
    return ""


def _extract_openai_message_content(message: Mapping[str, Any]) -> str:
    content = _extract_text_from_content_parts(message.get("content"))
    if content:
        return content

    tool_call_content = _extract_tool_call_arguments(message.get("tool_calls"))
    if tool_call_content:
        return tool_call_content

    function_call_content = _extract_tool_call_arguments(message.get("function_call"))
    if function_call_content:
        return function_call_content

    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        return refusal
    return ""


def _ollama_variants(name: str) -> set[str]:
    candidate = (name or "").strip()
    if not candidate:
        return set()
    if ":" not in candidate:
        return {candidate, f"{candidate}:latest"}
    base, tag = candidate.split(":", 1)
    if tag == "latest":
        return {candidate, base}
    return {candidate}


def model_matches(selected: str, visible: str) -> bool:
    left = (selected or "").strip()
    right = (visible or "").strip()
    if not left or not right:
        return False
    if left == right:
        return True
    if right in _ollama_variants(left) or left in _ollama_variants(right):
        return True
    return left in right or right in left


@dataclass
class ResolvedRole:
    role: str
    provider: str
    model: str


class BaseProvider:
    def __init__(self, name: str, *, api_key_env: str, base_url: str | None, default_model: str, preferred_models: Sequence[str]):
        self.name = name
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/") if base_url else None
        self.default_model = default_model
        self.preferred_models = tuple(preferred_models)

    @property
    def api_key(self) -> str:
        if not self.api_key_env:
            return ""
        return os.getenv(self.api_key_env, "")

    def has_credentials(self) -> bool:
        return bool(self.api_key or self.name == "ollama")

    def select_model(self, visible_models: Sequence[str] | None = None, override: str | None = None) -> str:
        visible = list(visible_models or [])
        if override:
            if not visible:
                return override
            for item in visible:
                if model_matches(override, item):
                    return item
            return override
        for candidate in self.preferred_models:
            for item in visible:
                if model_matches(candidate, item):
                    return item
        return self.default_model

    def list_models(self) -> List[str]:
        return []

    def list_model_infos(self) -> List[ProviderModelInfo]:
        return [ProviderModelInfo(name=model) for model in self.list_models()]

    def sdk_available(self) -> bool:
        return False

    def health(self, override_model: str | None = None) -> ProviderStatus:
        visible_models: List[str] = []
        model_infos: List[ProviderModelInfo] = []
        warnings: List[str] = []
        missing_actions: List[str] = []

        if not self.has_credentials():
            if self.name == "ollama":
                missing_actions.append(f"Ensure Ollama is reachable at {self.base_url} from the current runtime.")
            else:
                missing_actions.append(f"Set {self.api_key_env} in your environment or .env file.")
            return ProviderStatus(
                provider=self.name,
                configured=False,
                reachable=None,
                available=False,
                selected_model=self.select_model([], override_model),
                visible_models=[],
                missing_actions=missing_actions,
            )

        reachable = True
        try:
            model_infos = self.list_model_infos()
            visible_models = [info.name for info in model_infos]
        except Exception as exc:  # pragma: no cover - network path
            reachable = False
            warnings.append(str(exc))
            if self.name == "ollama":
                missing_actions.append(
                    f"Ensure Ollama is reachable at {self.base_url} from the current runtime."
                )
            else:
                missing_actions.append(
                    f"Ensure provider `{self.name}` is reachable from the current runtime and that the configured credentials are valid."
                )

        selected_model = self.select_model(visible_models, override_model)
        selected_capabilities = next(
            (list(info.supported_methods) for info in model_infos if model_matches(selected_model, info.name)),
            [],
        )
        model_available = not visible_models or any(model_matches(selected_model, item) for item in visible_models)
        if visible_models and not model_available:
            missing_actions.append(
                f"Selected model `{selected_model}` is not visible for provider `{self.name}` in the current runtime."
            )

        return ProviderStatus(
            provider=self.name,
            configured=True,
            reachable=reachable,
            available=reachable and model_available,
            selected_model=selected_model,
            visible_models=visible_models,
            selected_model_capabilities=selected_capabilities,
            warnings=warnings,
            missing_actions=missing_actions,
        )

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        raise NotImplementedError

    def embed_texts(self, texts: Sequence[str], model: str) -> List[List[float]]:
        raise ProviderError(f"{self.name} does not support embeddings in this integration.")

    def create_text_cache(
        self,
        model: str,
        *,
        system_instruction: str,
        corpus: str,
        ttl: str,
        display_name: str,
    ) -> Dict[str, Any]:
        raise ProviderError(f"{self.name} does not support explicit text caching in this integration.")

    def delete_cache(self, cache_name: str) -> None:
        raise ProviderError(f"{self.name} does not support cache deletion in this integration.")


class GeminiProvider(BaseProvider):
    def sdk_available(self) -> bool:
        return google_genai is not None

    def _extract_model_name(self, item: Any) -> str:
        name = item.get("name") if isinstance(item, dict) else getattr(item, "name", "")
        return str(name).split("/")[-1]

    def _extract_supported_methods(self, item: Any) -> List[str]:
        if isinstance(item, dict):
            methods = item.get("supportedGenerationMethods") or item.get("supported_generation_methods") or []
        else:
            methods = getattr(item, "supportedGenerationMethods", None)
            if methods is None:
                methods = getattr(item, "supported_generation_methods", None)
            if methods is None:
                methods = getattr(item, "supported_actions", None)
            methods = methods or []
        return [str(method) for method in methods if str(method)]

    def list_model_infos(self) -> List[ProviderModelInfo]:
        if google_genai:
            client = google_genai.Client(api_key=self.api_key)
            infos = [
                ProviderModelInfo(
                    name=self._extract_model_name(model),
                    supported_methods=self._extract_supported_methods(model),
                )
                for model in client.models.list()
            ]
            return sorted(infos, key=lambda info: info.name)

        response = requests.get(
            f"{self.base_url}/models",
            params={"key": self.api_key},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        infos = [
            ProviderModelInfo(
                name=self._extract_model_name(item),
                supported_methods=self._extract_supported_methods(item),
            )
            for item in payload.get("models", [])
        ]
        return sorted(infos, key=lambda info: info.name)

    def list_models(self) -> List[str]:
        return [info.name for info in self.list_model_infos()]

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        try:
            return self._complete_once(request, model=request.model)
        except Exception as exc:
            if _is_model_not_found_error(exc):
                visible_model = self.select_model(self.list_models(), request.model)
                if visible_model and visible_model != request.model:
                    return self._complete_once(request, model=visible_model)
            if request.metadata.get("premium_context_king"):
                raise ProviderError(
                    f"Premium Context King request failed for `{request.model}`. "
                    f"Check paid-tier access, explicit caching support, and Gemini billing setup. Original error: {exc}"
                ) from exc
            if not _is_quota_or_rate_limit_error(exc):
                raise
            fallback_model = None
            if not request.metadata.get("use_google_search") and request.metadata.get("role") == "context_king":
                candidate = gemini_web_model()
                if candidate and candidate != request.model:
                    fallback_model = candidate
            if fallback_model:
                return self._complete_once(request, model=fallback_model)
            raise

    def _complete_once(self, request: ProviderRequest, *, model: str) -> ProviderResponse:
        if request.metadata.get("use_google_search"):
            return self._complete_with_google_search(request, model=model)

        if google_genai:
            client = google_genai.Client(api_key=self.api_key)
            prompt = "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in request.messages)
            config = {"temperature": request.temperature}
            if request.response_format == "json":
                config["response_mime_type"] = "application/json"
            if request.thinking_level:
                config["thinking_config"] = {"thinking_level": request.thinking_level}
            if request.cached_content:
                config["cached_content"] = request.cached_content
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            content = getattr(response, "text", "") or ""
            return ProviderResponse(provider=self.name, model=model, content=content, raw={"sdk": True})

        if request.cached_content:
            raise ProviderError("Gemini explicit caching requires the google-genai SDK in the current runtime.")

        system_messages = [msg["content"] for msg in request.messages if msg["role"] == "system"]
        user_messages = [msg["content"] for msg in request.messages if msg["role"] != "system"]
        contents = []
        for system in system_messages:
            contents.append({"role": "user", "parts": [{"text": system}]})
        if user_messages:
            contents.append({"role": "user", "parts": [{"text": "\n\n".join(user_messages)}]})
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
            },
        }
        if request.response_format == "json":
            payload["generationConfig"]["responseMimeType"] = "application/json"
        if request.thinking_level:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": request.thinking_level}
        response = self._post_with_retries(
            f"{self.base_url}/models/{model}:generateContent",
            params={"key": self.api_key},
            json=payload,
        )
        raw = response.json()
        content = raw["candidates"][0]["content"]["parts"][0]["text"]
        return ProviderResponse(provider=self.name, model=model, content=content, raw=raw)

    def create_text_cache(
        self,
        model: str,
        *,
        system_instruction: str,
        corpus: str,
        ttl: str,
        display_name: str,
    ) -> Dict[str, Any]:
        if not google_genai:
            raise ProviderError("Gemini explicit caching requires the google-genai SDK in the current runtime.")

        client = google_genai.Client(api_key=self.api_key)
        cache = client.caches.create(
            model=model,
            config={
                "display_name": display_name,
                "system_instruction": system_instruction,
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": corpus}],
                    }
                ],
                "ttl": ttl,
            },
        )
        cache_name = getattr(cache, "name", None)
        if not cache_name and isinstance(cache, dict):
            cache_name = cache.get("name")
        if not cache_name:
            raise ProviderError("Gemini cache creation succeeded but did not return a cache name.")
        return {
            "name": str(cache_name),
            "model": model,
            "display_name": display_name,
            "ttl": ttl,
        }

    def delete_cache(self, cache_name: str) -> None:
        if not google_genai:
            raise ProviderError("Gemini cache deletion requires the google-genai SDK in the current runtime.")
        client = google_genai.Client(api_key=self.api_key)
        try:
            client.caches.delete(name=cache_name)
        except TypeError:
            client.caches.delete(cache_name)

    def _complete_with_google_search(self, request: ProviderRequest, *, model: str) -> ProviderResponse:
        prompt = "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in request.messages)
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}],
                }
            ],
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "temperature": request.temperature,
            },
        }
        response = self._post_with_retries(
            f"{self.base_url}/models/{model}:generateContent",
            headers={
                "x-goog-api-key": self.api_key,
                "content-type": "application/json",
            },
            json=payload,
            timeout_override=GEMINI_GROUNDED_TIMEOUT,
        )
        raw = response.json()
        content = raw["candidates"][0]["content"]["parts"][0]["text"]
        return ProviderResponse(provider=self.name, model=model, content=content, raw=raw)

    def _post_with_retries(self, url: str, timeout_override: int | None = None, **kwargs) -> requests.Response:
        effective_timeout = timeout_override or REQUEST_TIMEOUT
        last_response = None
        last_error: Exception | None = None
        max_attempts = max(GEMINI_RATE_LIMIT_RETRIES, GEMINI_NETWORK_RETRIES)
        for attempt in range(max_attempts + 1):
            try:
                response = requests.post(url, timeout=effective_timeout, **kwargs)
                if response.status_code != 429:
                    response.raise_for_status()
                    return response
                last_response = response
                if attempt >= GEMINI_RATE_LIMIT_RETRIES:
                    break
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after else min(8.0, 1.5 * (2**attempt))
                time.sleep(sleep_seconds)
            except (requests.ReadTimeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt >= GEMINI_NETWORK_RETRIES:
                    raise
                time.sleep(min(8.0, 2.0 * (attempt + 1)))
        if last_response is not None:
            last_response.raise_for_status()
        if last_error is not None:
            raise last_error
        raise ProviderError("Gemini request failed without an HTTP response.")


class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, name: str, *, api_key_env: str, base_url: str | None, default_model: str, preferred_models: Sequence[str], embed_path: str | None = None):
        super().__init__(name, api_key_env=api_key_env, base_url=base_url, default_model=default_model, preferred_models=preferred_models)
        self.embed_path = embed_path or f"{self.base_url}/embeddings"

    def _headers(self) -> Dict[str, str]:
        return {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    def _request_timeout(self) -> int:
        if self.name == "nvidia":
            return NVIDIA_REQUEST_TIMEOUT
        return REQUEST_TIMEOUT

    def _post_with_retries(self, url: str, **kwargs) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(OPENAI_COMPATIBLE_RETRIES + 1):
            try:
                response = requests.post(url, timeout=self._request_timeout(), **kwargs)
                response.raise_for_status()
                return response
            except requests.HTTPError as exc:
                status_code = getattr(exc.response, "status_code", None)
                if status_code not in RETRYABLE_HTTP_STATUS_CODES or attempt >= OPENAI_COMPATIBLE_RETRIES:
                    raise
                last_error = exc
                time.sleep(_retry_delay_seconds(exc.response, attempt, ceiling=20.0))
            except (requests.ReadTimeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt >= OPENAI_COMPATIBLE_RETRIES:
                    raise
                time.sleep(min(20.0, 2.5 * (attempt + 1)))
        if last_error is not None:
            raise last_error
        raise ProviderError(f"{self.name} request failed without an HTTP response.")

    def list_models(self) -> List[str]:
        url = f"{self.base_url}/models"
        response = requests.get(url, headers=self._headers(), timeout=self._request_timeout())
        response.raise_for_status()
        payload = response.json()
        return sorted(model["id"] for model in payload.get("data", []))

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "stream": False,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "response_format": {"type": "json_object"} if request.response_format == "json" else None,
        }
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.extra_body:
            payload.update(request.extra_body)
        response = self._post_with_retries(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
        )
        raw = response.json()
        message = raw["choices"][0]["message"]
        content = _extract_openai_message_content(message)
        return ProviderResponse(provider=self.name, model=request.model, content=content, raw=raw)

    def embed_texts(self, texts: Sequence[str], model: str) -> List[List[float]]:
        response = self._post_with_retries(
            self.embed_path,
            headers=self._headers(),
            json={"model": model, "input": list(texts)},
        )
        raw = response.json()
        return [item["embedding"] for item in raw.get("data", [])]


class NvidiaHostedProvider(OpenAICompatibleProvider):
    def _probe_model(self, model: str) -> None:
        response = self._post_with_retries(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 1,
            },
        )
        response.raise_for_status()

    def health(self, override_model: str | None = None) -> ProviderStatus:
        selected_model = self.select_model([], override_model)
        if not self.has_credentials():
            return ProviderStatus(
                provider=self.name,
                configured=False,
                reachable=None,
                available=False,
                selected_model=selected_model,
                missing_actions=[f"Set {self.api_key_env} in your environment or .env file."],
            )

        try:
            visible_models = self.list_models()
            selected_model = self.select_model(visible_models, override_model)
            available = not visible_models or any(model_matches(selected_model, item) for item in visible_models)
            missing_actions = []
            if visible_models and not available:
                missing_actions.append(
                    f"Selected model `{selected_model}` is not visible for provider `{self.name}` in the current runtime."
                )
            return ProviderStatus(
                provider=self.name,
                configured=True,
                reachable=True,
                available=available,
                selected_model=selected_model,
                visible_models=visible_models,
                warnings=[],
                missing_actions=missing_actions,
            )
        except Exception as listing_exc:
            warnings = [f"Model listing unavailable; using a minimal probe instead. {listing_exc}"]
            try:
                self._probe_model(selected_model)
                return ProviderStatus(
                    provider=self.name,
                    configured=True,
                    reachable=True,
                    available=True,
                    selected_model=selected_model,
                    warnings=warnings,
                )
            except Exception as probe_exc:
                return ProviderStatus(
                    provider=self.name,
                    configured=True,
                    reachable=False,
                    available=False,
                    selected_model=selected_model,
                    warnings=warnings + [str(probe_exc)],
                    missing_actions=[
                        "Ensure the NVIDIA hosted API is reachable from the current runtime and that NVIDIA_API_KEY is valid."
                    ],
                )


class OllamaProvider(BaseProvider):
    def _request_timeout(self) -> int:
        return ollama_request_timeout()

    def list_models(self) -> List[str]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=self._request_timeout())
        response.raise_for_status()
        payload = response.json()
        return sorted(model["name"] for model in payload.get("models", []))

    def health(self, override_model: str | None = None) -> ProviderStatus:
        try:
            visible_models = self.list_models()
            selected_model = self.select_model(visible_models, override_model)
            missing_models = [
                model for model in REQUIRED_OLLAMA_MODELS if not any(model_matches(model, item) for item in visible_models)
            ]
            missing_actions = [f"Run `ollama pull {model}`." for model in missing_models]
            if not any(model_matches(selected_model, item) for item in visible_models):
                missing_actions.append(
                    f"Selected Ollama model `{selected_model}` is not installed. Pull it or point OLLAMA_STYLE_MODEL/override to an installed model."
                )
            return ProviderStatus(
                provider=self.name,
                configured=True,
                reachable=True,
                available=any(model_matches(selected_model, item) for item in visible_models),
                selected_model=selected_model,
                visible_models=visible_models,
                missing_actions=missing_actions,
            )
        except Exception as exc:
            return ProviderStatus(
                provider=self.name,
                configured=True,
                reachable=False,
                available=False,
                selected_model=self.select_model([], override_model),
                warnings=[str(exc)],
                missing_actions=[
                    f"Ensure Ollama is reachable at {self.base_url} from the current runtime. "
                    "If it is already running on the host, run the CLI outside the sandbox or adjust networking."
                ],
            )

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        if request.top_p is not None:
            payload["options"]["top_p"] = request.top_p
        if request.response_format == "json":
            payload["format"] = "json"
        if request.extra_body:
            extra = dict(request.extra_body)
            extra_options = extra.pop("options", None)
            if isinstance(extra_options, Mapping):
                payload["options"].update(extra_options)
            payload.update(extra)

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self._request_timeout(),
        )
        response.raise_for_status()
        raw = response.json()
        content = raw.get("message", {}).get("content", "")
        return ProviderResponse(provider=self.name, model=request.model, content=content, raw=raw)

    def embed_texts(self, texts: Sequence[str], model: str) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            # Truncate oversized inputs — embedding models have limited context
            # (nomic-embed-text: ~8192 tokens ≈ 32K chars)
            truncated = text[:32000] if len(text) > 32000 else text
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": truncated},
                timeout=self._request_timeout(),
            )
            response.raise_for_status()
            raw = response.json()
            embeddings = raw.get("embeddings") or []
            if not embeddings:
                raise ProviderError("Ollama embedding response did not contain embeddings.")
            vectors.append(embeddings[0])
        return vectors


class ProviderRouter:
    def __init__(self, providers: Mapping[str, BaseProvider] | None = None):
        self.providers: Dict[str, BaseProvider] = dict(providers or {})

    def provider(self, name: str) -> BaseProvider:
        try:
            return self.providers[name]
        except KeyError as exc:
            raise ProviderError(f"Unknown provider: {name}") from exc

    def resolve_model_for_value(self, value: str) -> tuple[str, str]:
        if ":" in value:
            prefix, remainder = value.split(":", 1)
            if prefix in self.providers:
                return prefix, remainder
        lowered = value.lower()
        if lowered.startswith("claude"):
            raise ProviderError("Claude models are not part of the Python runtime; use the Architect handoff prompt instead.")
        if lowered.startswith("gemini"):
            return "gemini", value
        if lowered.startswith("nvidia/") or "nemotron" in lowered:
            return "nvidia", value
        if value in self.provider("ollama").preferred_models or ":" in value:
            return "ollama", value
        if (
            lowered.startswith("glm")
            or lowered.startswith("deepseek")
            or "deepseek" in lowered
            or "glm-" in lowered
            or lowered.startswith("zai-org/")
            or lowered.startswith("featherless:")
        ):
            raise ProviderError("Legacy DeepSeek/GLM overrides are no longer supported in this fork. Use NVIDIA Nemotron overrides instead.")
        raise ProviderError(f"Could not infer provider for override value `{value}`.")

    def resolve_role(self, role: str, overrides: Mapping[str, Any] | None = None) -> ResolvedRole:
        canonical = canonical_role_name(role)
        normalized_overrides = normalize_provider_overrides(overrides)
        raw_override = normalized_overrides.get(canonical)
        if isinstance(raw_override, str):
            provider, model = self.resolve_model_for_value(raw_override)
            return ResolvedRole(role=canonical, provider=provider, model=model)
        if isinstance(raw_override, dict):
            provider = raw_override.get("provider") or ROLE_TO_PROVIDER[canonical]
            model = raw_override.get("model") or role_default_model(canonical)
            if provider == "featherless":
                raise ProviderError("Legacy DeepSeek/GLM overrides are no longer supported in this fork. Use NVIDIA Nemotron overrides instead.")
            if provider not in self.providers:
                raise ProviderError(f"Unsupported provider override `{provider}` for role `{canonical}`.")
            return ResolvedRole(role=canonical, provider=provider, model=model)
        provider = ROLE_TO_PROVIDER[canonical]
        model = role_default_model(canonical)
        return ResolvedRole(role=canonical, provider=provider, model=model)

    def status_for_role(self, role: str, overrides: Mapping[str, Any] | None = None) -> ProviderStatus:
        canonical = canonical_role_name(role)
        resolved = self.resolve_role(canonical, overrides)
        return self.provider(resolved.provider).health(resolved.model)

    def context_king_premium_status(self, overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        status = self.status_for_role("context_king", overrides)
        resolved = self.resolve_role("context_king", overrides)
        provider = self.provider(resolved.provider)
        capabilities = list(status.selected_model_capabilities)
        return {
            "provider": resolved.provider,
            "model": status.selected_model,
            "sdk_available": provider.sdk_available(),
            "supports_generate_content": "generateContent" in capabilities,
            "supports_explicit_caching": "createCachedContent" in capabilities,
            "selected_model_capabilities": capabilities,
        }

    def call(self, request: ProviderRequest) -> ProviderResponse:
        return self.provider(request.provider).complete(request)

    def embed_texts(self, texts: Sequence[str], *, overrides: Mapping[str, Any] | None = None) -> List[List[float]]:
        resolved = self.resolve_role("embedding", overrides)
        return self.provider(resolved.provider).embed_texts(texts, resolved.model)

    def create_context_cache(
        self,
        *,
        overrides: Mapping[str, Any] | None = None,
        system_instruction: str,
        corpus: str,
        ttl: str,
        display_name: str,
    ) -> Dict[str, Any]:
        resolved = self.resolve_role("context_king", overrides)
        return self.provider(resolved.provider).create_text_cache(
            resolved.model,
            system_instruction=system_instruction,
            corpus=corpus,
            ttl=ttl,
            display_name=display_name,
        )

    def delete_context_cache(self, cache_name: str, *, overrides: Mapping[str, Any] | None = None) -> None:
        resolved = self.resolve_role("context_king", overrides)
        self.provider(resolved.provider).delete_cache(cache_name)


def build_default_router() -> ProviderRouter:
    providers: Dict[str, BaseProvider] = {
        "gemini": GeminiProvider(**PROVIDER_DEFAULTS["gemini"].__dict__),
        "nvidia": NvidiaHostedProvider(**PROVIDER_DEFAULTS["nvidia"].__dict__),
        "ollama": OllamaProvider(**PROVIDER_DEFAULTS["ollama"].__dict__),
    }
    return ProviderRouter(providers)


def parse_json_response(content: str) -> Dict[str, Any]:
    stripped = (content or "").strip()
    if not stripped:
        raise ProviderError("Provider response was empty and did not contain a JSON object.")

    decoder = JSONDecoder()
    candidates = []
    cleaned_variants = [
        stripped,
        re.sub(r"<think\b[^>]*>.*?</think>", "", stripped, flags=re.IGNORECASE | re.DOTALL),
        re.sub(r"```(?:json)?\s*|\s*```", "", stripped, flags=re.IGNORECASE),
    ]
    seen_variants = set()

    for variant in cleaned_variants:
        variant = re.sub(r"<think\b[^>]*>.*?</think>", "", variant, flags=re.IGNORECASE | re.DOTALL)
        variant = re.sub(r"</?(?:tool_call|analysis)\b[^>]*>", "", variant, flags=re.IGNORECASE)
        variant = variant.strip()
        if not variant or variant in seen_variants:
            continue
        seen_variants.add(variant)
        try:
            parsed = json.loads(variant)
            normalized = _unwrap_json_payload(parsed)
            if normalized is not None:
                return normalized
        except JSONDecodeError:
            pass

        for index, char in enumerate(variant):
            if char not in "[{":
                continue
            try:
                parsed, end = decoder.raw_decode(variant[index:])
            except JSONDecodeError:
                continue
            normalized = _unwrap_json_payload(parsed)
            if normalized is not None:
                candidates.append((end, normalized))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    raise ProviderError("Provider response did not contain a JSON object.")


def _unwrap_json_payload(payload: Any) -> Dict[str, Any] | None:
    if isinstance(payload, dict):
        if "tool_calls" in payload:
            tool_call_payload = _unwrap_json_payload(payload.get("tool_calls"))
            if tool_call_payload is not None:
                return tool_call_payload
        if "function" in payload and isinstance(payload["function"], Mapping):
            function_payload = _unwrap_json_payload(payload["function"])
            if function_payload is not None:
                return function_payload
        if "arguments" in payload and set(payload.keys()).issubset({"arguments", "name", "type", "id", "function"}):
            arguments = payload.get("arguments")
            if isinstance(arguments, Mapping):
                return dict(arguments)
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except JSONDecodeError:
                    return None
                return _unwrap_json_payload(parsed)
        return payload
    if isinstance(payload, list):
        if len(payload) == 1:
            return _unwrap_json_payload(payload[0])
        return None
    return None
