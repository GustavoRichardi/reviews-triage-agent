"""Contratos de dados do agente (saídas estruturadas do LLM e resultado final)."""

from pydantic import BaseModel, Field

from reviews_triage.labels import Category, Sentiment, Urgency


class Classification(BaseModel):
    """Saída do passo `classify`. O schema JSON enviado ao modelo sai desta classe."""

    categoria: Category
    sentimento: Sentiment
    urgencia: Urgency


class ReviewVerdict(BaseModel):
    """Saída do passo `review`: a resposta pode seguir para o atendente?"""

    aprovada: bool
    problemas: list[str]


class LLMCall(BaseModel, frozen=True):
    """Telemetria de uma chamada ao LLM; a Fase 5 usa para calcular custo e latência."""

    step: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_s: float


class TriageResult(BaseModel):
    """Resultado final de uma triagem. `categoria` e `urgencia` são None só se o
    modelo recusou o passo de classificação."""

    categoria: Category | None
    sentimento: Sentiment | None
    urgencia: Urgency | None
    resposta_sugerida: str
    fontes: list[str]
    escalate_to_human: bool
    motivos_escalacao: list[str] = Field(default_factory=list)
    tentativas_redacao: int
    llm_calls: list[LLMCall]
