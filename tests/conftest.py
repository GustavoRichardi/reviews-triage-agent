import random

import pandas as pd
import pytest

# Frases-modelo por categoria. Não imitam a distribuição real: servem só para
# testar que o pipeline roda e aprende um sinal óbvio quando ele existe.
TEMPLATES = {
    "atraso_entrega": [
        "Ainda não recebi meu pedido",
        "Chegou muito atrasado",
        "Prazo de entrega estourou",
    ],
    "produto_errado_ou_faltando": [
        "Veio o produto errado",
        "Faltou um item no pacote",
        "Cor diferente da que pedi",
    ],
    "defeito_ou_qualidade": ["Chegou quebrado", "Produto com defeito", "Qualidade péssima, rasgou"],
    "atendimento_vendedor": [
        "Vendedor não responde",
        "Cancelaram sem avisar",
        "Loja não atende o telefone",
    ],
    "elogio": ["Adorei, recomendo", "Excelente produto, parabéns", "Ótima loja, chegou rápido"],
    "outro": ["Ok", "Poderiam ter mais opções", "Sem comentários"],
}
FILLERS = ["", " mesmo", " infelizmente", " de novo", " hoje", " pela segunda vez"]


def make_synthetic_gold(per_class: int = 25, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for category, phrases in TEMPLATES.items():
        for i in range(per_class):
            rows.append(
                {
                    "review_id": f"{category}_{i}",
                    "review_score": 5 if category == "elogio" else rng.randint(1, 4),
                    "text": rng.choice(phrases) + rng.choice(FILLERS),
                    "categoria": category,
                    "urgencia": rng.choice(["baixa", "media", "alta"]),
                    "observacao": "",
                }
            )
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


@pytest.fixture
def synthetic_gold() -> pd.DataFrame:
    return make_synthetic_gold()
