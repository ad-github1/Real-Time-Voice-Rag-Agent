from __future__ import annotations

from voice_rag_agent.config import Settings

from .base import Reasoner, TemplateReasoner
from .ollama import OllamaReasoner
from .openai_responses import OpenAIResponsesReasoner


def build_reasoner(settings: Settings) -> Reasoner:
    selected = settings.reasoner.strip().lower()
    if selected in {"", "template", "offline", "fake"}:
        return TemplateReasoner()
    if selected in {"ollama", "gemma"}:
        return OllamaReasoner(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
    if selected in {"openai", "responses"}:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when VOICE_RAG_REASONER=openai.")
        if not settings.openai_model:
            raise RuntimeError("OPENAI_MODEL is required when VOICE_RAG_REASONER=openai.")
        return OpenAIResponsesReasoner(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
    raise ValueError(f"Unknown reasoner {settings.reasoner!r}.")
