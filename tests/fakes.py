"""Dublês de teste: nenhum acessa rede, chave de API ou modelo de embeddings."""

import hashlib
import math
import re

from reviews_triage.llm import LLMRefusal
from reviews_triage.schemas import Classification, LLMCall, ReviewVerdict

APPROVED = ReviewVerdict(aprovada=True, problemas=[])
REJECTED = ReviewVerdict(aprovada=False, problemas=["Promete cupom que a política não prevê."])


def make_call(step: str) -> LLMCall:
    return LLMCall(step=step, model="fake", input_tokens=100, output_tokens=20, latency_s=0.01)


class FakeLLM:
    """Responde de acordo com um roteiro e guarda tudo o que recebeu em `calls`."""

    def __init__(
        self,
        classification: Classification | None = None,
        drafts: list[str] | None = None,
        verdicts: list[ReviewVerdict] | None = None,
        refuse_step: str | None = None,
    ):
        self.classification = classification or Classification(
            categoria="atraso_entrega", sentimento="negativo", urgencia="media"
        )
        self.drafts = drafts or ["Sentimos muito pelo atraso. Vamos analisar o seu caso."]
        self.verdicts = verdicts or [APPROVED]
        self.refuse_step = refuse_step
        self.calls: list[tuple[str, str, str]] = []  # (step, system, user)
        self._n = {"draft": 0, "review": 0}

    def _next(self, step: str, items: list):
        item = items[min(self._n[step], len(items) - 1)]  # o último se repete
        self._n[step] += 1
        return item

    def _record(self, step: str, system: str, user: str) -> LLMCall:
        self.calls.append((step, system, user))
        call = make_call(step)
        if step == self.refuse_step:
            raise LLMRefusal(step, call, "general_harms")
        return call

    def parse(self, step, system, user, schema):
        call = self._record(step, system, user)
        if step == "classify":
            return self.classification, call
        return self._next("review", self.verdicts), call

    def text(self, step, system, user):
        call = self._record(step, system, user)
        return self._next("draft", self.drafts), call

    def steps(self) -> list[str]:
        return [step for step, _, _ in self.calls]


def hash_embed(texts: list[str], dim: int = 256) -> list[list[float]]:
    """Embedding de sacola de palavras por hash. Textos com palavras em comum ficam
    próximos, o que basta para testar a fiação do índice (não a qualidade do modelo)."""
    vectors = []
    for text in texts:
        vector = [0.0] * dim
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        vectors.append([x / norm for x in vector])
    return vectors
