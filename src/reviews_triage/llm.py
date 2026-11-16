"""Cliente do LLM: uma camada fina sobre o SDK da Anthropic.

Faz três coisas que o resto do código não deveria repetir: lê o modelo do ambiente,
trata recusa/truncamento como erros explícitos e devolve a telemetria (tokens e
latência) de cada chamada.
"""

import os
import time
from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from reviews_triage.schemas import LLMCall

T = TypeVar("T", bound=BaseModel)

# Teto de saída, não meta: só se paga o que é gerado. O raciocínio do modelo também
# conta contra este limite, então é generoso para não truncar a resposta.
MAX_TOKENS = 4096


class LLMRefusal(Exception):
    """O modelo recusou a requisição (stop_reason == "refusal")."""

    def __init__(self, step: str, call: LLMCall, category: str | None = None):
        super().__init__(f"modelo recusou o passo '{step}' (categoria: {category})")
        self.step, self.call, self.category = step, call, category


class LLMTruncated(Exception):
    """A resposta foi cortada por max_tokens: o resultado não é confiável."""


class LLM(Protocol):
    """O que o agente precisa de um LLM. Os testes e a Fase 5 trocam a implementação."""

    def parse(self, step: str, system: str, user: str, schema: type[T]) -> tuple[T, LLMCall]: ...

    def text(self, step: str, system: str, user: str) -> tuple[str, LLMCall]: ...


class AnthropicLLM:
    def __init__(
        self,
        model: str | None = None,
        effort: str | None = None,
        client: anthropic.Anthropic | None = None,
    ):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "")
        if not self.model:
            raise RuntimeError(
                "ANTHROPIC_MODEL não definido. Preencha no .env (ex.: claude-sonnet-5-5)."
            )
        # Sem temperature/top_p: os modelos atuais rejeitam valores não padrão (erro 400).
        # `effort` é o controle de custo/profundidade, e "low" é aceito por todos eles.
        self.effort = effort or os.environ.get("ANTHROPIC_EFFORT") or "low"
        self.client = client or anthropic.Anthropic()

    def parse(self, step: str, system: str, user: str, schema: type[T]) -> tuple[T, LLMCall]:
        started = time.perf_counter()
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": self.effort},
        )
        call = self._check(step, response, time.perf_counter() - started)
        if response.parsed_output is None:
            raise LLMTruncated(f"passo '{step}': resposta sem saída estruturada")
        return response.parsed_output, call

    def text(self, step: str, system: str, user: str) -> tuple[str, LLMCall]:
        started = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
        )
        call = self._check(step, response, time.perf_counter() - started)
        return "".join(b.text for b in response.content if b.type == "text").strip(), call

    def _check(self, step: str, response, latency_s: float) -> LLMCall:
        usage = response.usage
        call = LLMCall(
            step=step,
            model=self.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            latency_s=latency_s,
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(step, call, getattr(details, "category", None))
        if response.stop_reason == "max_tokens":
            raise LLMTruncated(f"passo '{step}': resposta cortada por max_tokens")
        return call
