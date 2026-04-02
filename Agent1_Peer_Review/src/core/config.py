from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional during bootstrap
    def load_dotenv(*args, **kwargs):
        return False


ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR.parent / ".env"
LOCAL_ENV_PATH = ROOT_DIR / ".env"

DEFAULT_REPORT_FORMATS = ["json", "md"]
DEFAULT_TARGET_JOURNAL = "(not specified)"
DEFAULT_OUTPUT_ROOT = "results"
DEFAULT_WEB_RESEARCH_MODE = "hybrid"
DEFAULT_GEMINI_CONTEXT_CACHE_TTL = "600s"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

ROLE_ALIASES = {
    "librarian": "context_king",
    "red_team": "logic_judge",
    "intern": "style_scribe",
}

DEFAULT_NVIDIA_LOGIC_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_NVIDIA_TECHNICAL_MODEL = DEFAULT_NVIDIA_LOGIC_MODEL


def _strip_env_value(raw: str) -> str:
    value = raw.strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        value = value[1:-1]
    return value


def load_env_file_fallback(path: str | Path, *, override: bool = False) -> bool:
    env_path = Path(path)
    if not env_path.exists():
        return False

    changed = False
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = _strip_env_value(raw_value)
        if override or key not in os.environ:
            os.environ[key] = value
            changed = True
    return changed


def load_env_defaults() -> None:
    load_dotenv(ENV_PATH)
    load_dotenv(LOCAL_ENV_PATH)
    load_env_file_fallback(ENV_PATH)
    load_env_file_fallback(LOCAL_ENV_PATH)


load_env_defaults()


def env_or_default(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def canonical_role_name(role: str) -> str:
    return ROLE_ALIASES.get(role, role)


def normalize_provider_overrides(overrides: Mapping[str, Any] | None) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    raw = dict(overrides or {})
    for key, value in raw.items():
        canonical = canonical_role_name(key)
        if canonical in normalized and canonical == key:
            normalized[canonical] = value
            continue
        if canonical not in normalized or canonical == key:
            normalized[canonical] = value
    return normalized


def gemini_context_model() -> str:
    return env_or_default("GEMINI_CONTEXT_MODEL", env_or_default("GEMINI_MODEL", "gemini-3.1-pro-preview"))


def gemini_web_model() -> str:
    return env_or_default("GEMINI_WEB_MODEL", "gemini-2.5-flash")


def gemini_context_cache_ttl() -> str:
    return env_or_default("GEMINI_CONTEXT_CACHE_TTL", DEFAULT_GEMINI_CONTEXT_CACHE_TTL)


def nvidia_logic_model() -> str:
    return env_or_default("NVIDIA_LOGIC_MODEL", DEFAULT_NVIDIA_LOGIC_MODEL)


def nvidia_technical_model() -> str:
    return env_or_default("NVIDIA_TECHNICAL_MODEL", DEFAULT_NVIDIA_TECHNICAL_MODEL)


def ollama_style_model() -> str:
    return env_or_default("OLLAMA_STYLE_MODEL", env_or_default("OLLAMA_CHAT_MODEL", "qwen3:8b"))


def ollama_rewrite_model() -> str:
    return env_or_default("OLLAMA_REWRITE_MODEL", "gemma3:4b")


def ollama_embed_model() -> str:
    return env_or_default("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def ollama_request_timeout() -> int:
    raw = env_or_default("OLLAMA_REQUEST_TIMEOUT", "120")
    try:
        return max(1, int(raw))
    except ValueError:
        return 120


DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", gemini_context_model())


@dataclass(frozen=True)
class ProviderDefaults:
    name: str
    api_key_env: str
    base_url: str | None
    default_model: str
    preferred_models: tuple[str, ...]


PROVIDER_DEFAULTS: Dict[str, ProviderDefaults] = {
    "gemini": ProviderDefaults(
        name="gemini",
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model=gemini_context_model(),
        preferred_models=(
            gemini_context_model(),
            "gemini-3.1-pro",
            "gemini-3.1-pro-preview",
            gemini_web_model(),
            "gemini-2.5-pro",
        ),
    ),
    "nvidia": ProviderDefaults(
        name="nvidia",
        api_key_env="NVIDIA_API_KEY",
        base_url=os.getenv("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL),
        default_model=nvidia_logic_model(),
        preferred_models=(
            nvidia_logic_model(),
            nvidia_technical_model(),
            DEFAULT_NVIDIA_LOGIC_MODEL,
        ),
    ),
    "ollama": ProviderDefaults(
        name="ollama",
        api_key_env="",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        default_model=ollama_style_model(),
        preferred_models=(
            ollama_style_model(),
            ollama_rewrite_model(),
            "phi4-mini",
            ollama_embed_model(),
        ),
    ),
}


ROLE_TO_PROVIDER = {
    "context_king": "gemini",
    "web_research": "gemini",
    "logic_judge": "nvidia",
    "technical_auditor": "nvidia",
    "style_scribe": "ollama",
    "embedding": "ollama",
}

ROLE_TO_MODEL_ENV = {
    "context_king": "GEMINI_CONTEXT_MODEL",
    "web_research": "GEMINI_WEB_MODEL",
    "logic_judge": "NVIDIA_LOGIC_MODEL",
    "technical_auditor": "NVIDIA_TECHNICAL_MODEL",
    "style_scribe": "OLLAMA_STYLE_MODEL",
    "embedding": "OLLAMA_EMBED_MODEL",
}

REQUIRED_OLLAMA_MODELS = (
    ollama_style_model(),
    ollama_rewrite_model(),
    "phi4-mini",
    ollama_embed_model(),
)


def role_default_model(role: str) -> str:
    canonical = canonical_role_name(role)
    if canonical == "context_king":
        return gemini_context_model()
    if canonical == "web_research":
        return gemini_web_model()
    if canonical == "logic_judge":
        return nvidia_logic_model()
    if canonical == "technical_auditor":
        return nvidia_technical_model()
    if canonical == "style_scribe":
        return ollama_style_model()
    if canonical == "embedding":
        return ollama_embed_model()
    env_name = ROLE_TO_MODEL_ENV.get(canonical)
    if env_name and os.getenv(env_name):
        return os.getenv(env_name, "")
    provider_name = ROLE_TO_PROVIDER.get(canonical, "")
    provider = PROVIDER_DEFAULTS.get(provider_name)
    if provider:
        return provider.default_model
    return DEFAULT_MODEL


def iterable_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def required_remote_providers() -> Iterable[ProviderDefaults]:
    for provider_name in ("gemini", "nvidia"):
        yield PROVIDER_DEFAULTS[provider_name]
