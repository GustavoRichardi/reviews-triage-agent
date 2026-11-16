"""Rótulos válidos. Fonte única para o gold set, o baseline e o agente.

Os tipos `Literal` são usados pelo Pydantic (o schema enviado ao LLM sai deles) e as
tuplas são derivadas deles, então não há como as duas listas divergirem.
"""

from typing import Literal, get_args

Category = Literal[
    "atraso_entrega",
    "produto_errado_ou_faltando",
    "defeito_ou_qualidade",
    "atendimento_vendedor",
    "elogio",
    "outro",
]
Sentiment = Literal["negativo", "neutro", "positivo"]
Urgency = Literal["baixa", "media", "alta"]

CATEGORIES: tuple[str, ...] = get_args(Category)
SENTIMENTS: tuple[str, ...] = get_args(Sentiment)
URGENCIES: tuple[str, ...] = get_args(Urgency)
