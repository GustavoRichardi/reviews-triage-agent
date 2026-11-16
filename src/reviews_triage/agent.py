"""O agente de triagem: um grafo LangGraph com classify → retrieve_policy → draft → review.

    START → classify → retrieve_policy → draft → review ─┬─ aprovada ─────────────┐
                                           ▲             │                        ▼
                                           └─ reprovada ─┘ (até MAX_REDRAFTS)   finalize → END
                                                           └─ esgotou tentativas ─┘

Uso (precisa de ANTHROPIC_API_KEY e ANTHROPIC_MODEL no .env):
    make triage TEXT="Pedido atrasou 10 dias" SCORE=1
"""

import argparse
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from reviews_triage.labels import Category
from reviews_triage.llm import LLM, AnthropicLLM, LLMRefusal
from reviews_triage.policy import PolicyChunk, PolicyIndex, build_index
from reviews_triage.prompts import PROMPT_NAMES, load_prompt, prompt_version
from reviews_triage.schemas import Classification, LLMCall, ReviewVerdict, TriageResult

TOP_K = 3
# Quantas vezes a revisão pode devolver o rascunho. Total de rascunhos = 1 + MAX_REDRAFTS:
# se a revisão reprovar duas vezes, o caso vai para um humano.
MAX_REDRAFTS = 1

# Dica em linguagem natural acrescentada à consulta de busca (ver DECISIONS.md, D31).
CATEGORY_QUERY_HINT: dict[Category, str] = {
    "atraso_entrega": "atraso na entrega do pedido",
    "produto_errado_ou_faltando": "produto errado ou item faltando no pedido",
    "defeito_ou_qualidade": "produto com defeito ou de baixa qualidade",
    "atendimento_vendedor": "problema no atendimento da loja",
    "elogio": "elogio ao produto ou à loja",
    "outro": "comentário ou sugestão",
}


class TriageState(TypedDict, total=False):
    review_text: str
    review_score: int | None
    classification: Classification
    policy_chunks: list[PolicyChunk]
    draft: str
    draft_attempts: int
    verdict: ReviewVerdict
    escalate_to_human: bool
    escalation_reasons: list[str]
    # `operator.add`: cada nó devolve só as suas chamadas e o LangGraph concatena.
    llm_calls: Annotated[list[LLMCall], operator.add]


# --- Montagem das mensagens ------------------------------------------------------------
# O texto da avaliação vem de terceiros. Ele vai sempre entre tags e com `<` e `>`
# escapados, para que o cliente não consiga "fechar" a tag e escrever fora dela. Isso
# reduz o risco de injeção de prompt; a barreira de verdade é o passo `review`.


def escape_text(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _review_block(state: TriageState) -> str:
    score = state.get("review_score")
    return (
        f"<nota>{score if score is not None else 'não informada'}</nota>\n"
        f"<avaliacao>\n{escape_text(state['review_text'])}\n</avaliacao>"
    )


def _policy_block(chunks: list[PolicyChunk]) -> str:
    items = "\n".join(f'<trecho fonte="{c.ref}">\n{c.text}\n</trecho>' for c in chunks)
    return f"<politica>\n{items}\n</politica>"


def build_query(category: Category, review_text: str) -> str:
    return f"{CATEGORY_QUERY_HINT[category]}. {review_text}"


def escalation_reasons(
    classification: Classification | None,
    verdict: ReviewVerdict | None,
    attempts: int,
    refusal_step: str | None = None,
) -> list[str]:
    """Regras de escalonamento. Código, não LLM: a decisão de chamar um humano precisa
    ser previsível e testável."""
    reasons = []
    if classification is not None and classification.urgencia == "alta":
        reasons.append("urgência alta")
    if refusal_step is not None:
        reasons.append(f"o modelo recusou o passo '{refusal_step}'")
    elif verdict is not None and not verdict.aprovada:
        reasons.append(f"a revisão reprovou a resposta após {attempts} tentativa(s)")
    return reasons


class TriageAgent:
    def __init__(self, llm: LLM, index: PolicyIndex):
        self.llm = llm
        self.index = index
        self.prompts = {name: load_prompt(name) for name in PROMPT_NAMES}
        self.prompt_versions = {name: prompt_version(text) for name, text in self.prompts.items()}
        self.graph = self._build_graph()

    # --- Nós -------------------------------------------------------------------------

    def classify(self, state: TriageState) -> dict:
        result, call = self.llm.parse(
            "classify", self.prompts["classify"], _review_block(state), Classification
        )
        return {"classification": result, "llm_calls": [call]}

    def retrieve_policy(self, state: TriageState) -> dict:
        query = build_query(state["classification"].categoria, state["review_text"])
        return {"policy_chunks": self.index.search(query, k=TOP_K)}

    def draft(self, state: TriageState) -> dict:
        c = state["classification"]
        parts = [
            f"<categoria>{c.categoria}</categoria>",
            f"<sentimento>{c.sentimento}</sentimento>",
            f"<urgencia>{c.urgencia}</urgencia>",
            _review_block(state),
            _policy_block(state["policy_chunks"]),
        ]
        verdict = state.get("verdict")
        if verdict is not None and not verdict.aprovada:  # redação após reprovação
            problems = "\n".join(f"- {p}" for p in verdict.problemas)
            parts += [
                f"<tentativa_anterior>\n{state['draft']}\n</tentativa_anterior>",
                f"<problemas_apontados>\n{problems}\n</problemas_apontados>",
            ]
        text, call = self.llm.text("draft", self.prompts["draft"], "\n".join(parts))
        return {
            "draft": text,
            "draft_attempts": state.get("draft_attempts", 0) + 1,
            "llm_calls": [call],
        }

    def review(self, state: TriageState) -> dict:
        user = "\n".join(
            [
                _review_block(state),
                _policy_block(state["policy_chunks"]),
                f"<resposta>\n{escape_text(state['draft'])}\n</resposta>",
            ]
        )
        verdict, call = self.llm.parse("review", self.prompts["review"], user, ReviewVerdict)
        return {"verdict": verdict, "llm_calls": [call]}

    def finalize(self, state: TriageState) -> dict:
        reasons = escalation_reasons(
            state["classification"], state["verdict"], state["draft_attempts"]
        )
        return {"escalate_to_human": bool(reasons), "escalation_reasons": reasons}

    # --- Grafo -----------------------------------------------------------------------

    @staticmethod
    def route_after_review(state: TriageState) -> str:
        if state["verdict"].aprovada or state["draft_attempts"] > MAX_REDRAFTS:
            return "finalize"
        return "draft"

    def _build_graph(self):
        graph = StateGraph(TriageState)
        graph.add_node("classify", self.classify)
        graph.add_node("retrieve_policy", self.retrieve_policy)
        graph.add_node("draft", self.draft)
        graph.add_node("review", self.review)
        graph.add_node("finalize", self.finalize)

        graph.add_edge(START, "classify")
        graph.add_edge("classify", "retrieve_policy")
        graph.add_edge("retrieve_policy", "draft")
        graph.add_edge("draft", "review")
        graph.add_conditional_edges(
            "review", self.route_after_review, {"draft": "draft", "finalize": "finalize"}
        )
        graph.add_edge("finalize", END)
        return graph.compile()

    # --- Execução --------------------------------------------------------------------

    def run(self, review_text: str, review_score: int | None = None) -> TriageResult:
        # `stream_mode="values"` devolve o estado completo a cada passo. Guardamos o último:
        # se o modelo recusar no meio do caminho, ainda temos o que foi feito até ali.
        state: TriageState = {"review_text": review_text, "review_score": review_score}
        refusal: LLMRefusal | None = None
        try:
            for snapshot in self.graph.stream(state, stream_mode="values"):
                state = snapshot
        except LLMRefusal as exc:
            refusal = exc

        classification = state.get("classification")
        calls = list(state.get("llm_calls", []))
        if refusal is not None:
            calls.append(refusal.call)
            reasons = escalation_reasons(
                classification,
                state.get("verdict"),
                state.get("draft_attempts", 0),
                refusal_step=refusal.step,
            )
            escalate = True
        else:
            reasons = state["escalation_reasons"]
            escalate = state["escalate_to_human"]

        return TriageResult(
            categoria=classification.categoria if classification else None,
            sentimento=classification.sentimento if classification else None,
            urgencia=classification.urgencia if classification else None,
            resposta_sugerida=state.get("draft", ""),
            fontes=[c.ref for c in state.get("policy_chunks", [])],
            escalate_to_human=escalate,
            motivos_escalacao=reasons,
            tentativas_redacao=state.get("draft_attempts", 0),
            llm_calls=calls,
        )


def build_agent() -> TriageAgent:
    """Agente real: modelo e chave vêm do ambiente; embeddings rodam localmente."""
    return TriageAgent(AnthropicLLM(), build_index())


def main() -> None:
    parser = argparse.ArgumentParser(description="Triagem de uma avaliação.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--score", type=int, choices=range(1, 6))
    args = parser.parse_args()

    print(build_agent().run(args.text, args.score).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
