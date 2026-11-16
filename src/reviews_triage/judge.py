"""LLM-as-judge: nota de 1 a 5 para aderência à política, tom e utilidade.

A configuração do juiz é CONGELADA (DECISIONS.md, D40 e D41): modelo, effort e o hash da
rubrica (prompts/judge.md). Mudar qualquer um deles invalida a comparação com rodadas
anteriores, então o `Judge` se recusa a rodar se a rubrica não bater com o hash registrado.
"""

from itertools import groupby
from typing import Literal

from pydantic import BaseModel

from reviews_triage.agent import escape_text
from reviews_triage.llm import LLM
from reviews_triage.policy import PolicyChunk
from reviews_triage.prompts import load_prompt, prompt_version
from reviews_triage.schemas import LLMCall

# Fixo para todas as rodadas e diferente de todos os modelos comparados (Opus, Sonnet, Haiku).
JUDGE_MODEL = "claude-fable-5-2"
JUDGE_EFFORT = "medium"
# Hash de prompts/judge.md no momento do congelamento, antes de qualquer rodada real.
JUDGE_PROMPT_VERSION = "edf6a162"

Score = Literal[1, 2, 3, 4, 5]
CRITERIA = ("aderencia", "tom", "utilidade")


class JudgeScores(BaseModel):
    # A justificativa vem antes das notas: o modelo escreve o porquê e só então decide.
    justificativa: str
    aderencia: Score
    tom: Score
    utilidade: Score


def full_policy_block(chunks: list[PolicyChunk]) -> str:
    """A política inteira, agrupada por documento. O juiz vê tudo, não só o que a busca
    devolveu: se a busca errou, a resposta pode seguir os trechos e ainda assim estar errada."""
    docs = []
    for (source, title), group in groupby(chunks, key=lambda c: (c.source, c.title)):
        sections = "\n".join(f"## {c.section}\n{c.text}" for c in group)
        docs.append(f'<documento arquivo="{source}" titulo="{title}">\n{sections}\n</documento>')
    return "<politica>\n" + "\n".join(docs) + "\n</politica>"


class Judge:
    def __init__(self, llm: LLM, chunks: list[PolicyChunk]):
        self.prompt = load_prompt("judge")
        self.version = prompt_version(self.prompt)
        if self.version != JUDGE_PROMPT_VERSION:
            raise RuntimeError(
                f"A rubrica do juiz mudou (hash {self.version}, congelado {JUDGE_PROMPT_VERSION}). "
                "Isso invalida a comparação com rodadas anteriores; ver DECISIONS.md, D41."
            )
        self.llm = llm
        self.policy = full_policy_block(chunks)

    def score(
        self, review_text: str, review_score: int | None, response: str
    ) -> tuple[JudgeScores, LLMCall]:
        user = "\n".join(
            [
                self.policy,
                f"<nota>{review_score if review_score is not None else 'não informada'}</nota>",
                f"<avaliacao>\n{escape_text(review_text)}\n</avaliacao>",
                f"<resposta>\n{escape_text(response)}\n</resposta>",
            ]
        )
        return self.llm.parse("judge", self.prompt, user, JudgeScores)
