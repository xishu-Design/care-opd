from __future__ import annotations

from app.llm.schemas import LLMRequest, LLMResult, ProviderHealth


class ProviderUnavailable(RuntimeError):
    pass


class ProviderGenerationError(RuntimeError):
    pass


class BaseLLMProvider:
    name = "base"

    def health(self) -> ProviderHealth:
        return ProviderHealth(name=self.name, configured=False, reachable=False, status="misconfigured", last_error="health_not_implemented")

    def generate(self, request: LLMRequest) -> LLMResult:
        raise NotImplementedError
