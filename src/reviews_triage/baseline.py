"""Baseline clássico: TF-IDF + Regressão Logística para a categoria.

Uso:
    python -m reviews_triage.baseline     # (ou `make baseline`)

Avalia com validação cruzada no gold set e grava reports/baseline.md.
"""

import argparse
import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from reviews_triage.gold import GOLD_PATH, load_gold
from reviews_triage.labels import CATEGORIES

REPORT_PATH = Path("reports/baseline.md")
SEED = 42
N_SPLITS = 5
N_REPEATS = 5
SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro"}


def build_features(df: pd.DataFrame) -> pd.Series:
    """Texto + nota como um token extra ("nota_1" ... "nota_5").

    O agente também recebe a nota, então o baseline precisa dela para a
    comparação ser justa. Um token no texto é o jeito mais simples de dar
    essa informação ao TF-IDF sem montar um ColumnTransformer.
    """
    return "nota_" + df["review_score"].astype(str) + " " + df["text"]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",  # "não" e "nao" viram o mesmo token
                    ngram_range=(1, 2),  # bigramas pegam "nao chegou", "veio errado"
                    sublinear_tf=True,  # repetir uma palavra 5x não vale 5x mais
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",  # compensa categorias raras
                    random_state=SEED,
                ),
            ),
        ]
    )


def choose_n_splits(y: pd.Series, wanted: int = N_SPLITS) -> int:
    """Estratificar exige pelo menos um exemplo de cada classe por fold."""
    smallest = int(y.value_counts().min())
    if smallest < 2:
        raise ValueError("Toda categoria precisa de pelo menos 2 exemplos para validação cruzada.")
    return min(wanted, smallest)


def evaluate(df: pd.DataFrame, n_repeats: int = N_REPEATS) -> dict:
    """Roda a validação cruzada e devolve tudo o que o relatório precisa.

    - Métricas (média ± desvio): validação cruzada repetida, porque com ~150
      exemplos uma única divisão em folds é muito sensível à sorte.
    - Matriz de confusão e relatório por classe: predições out-of-fold de uma
      rodada de 5 folds (cada exemplo previsto por um modelo que não o viu).
    """
    x, y = build_features(df), df["categoria"]
    n_splits = choose_n_splits(y)
    repeated_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)

    scores = {}
    models = {
        "Classe majoritária (referência)": DummyClassifier(strategy="most_frequent"),
        "TF-IDF + Regressão Logística": build_pipeline(),
    }
    for name, model in models.items():
        result = cross_validate(model, x, y, cv=repeated_cv, scoring=SCORING)
        scores[name] = {m: result[f"test_{m}"] for m in SCORING}

    predictions = cross_val_predict(
        build_pipeline(), x, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    )
    labels = [c for c in CATEGORIES if c in set(y)]

    return {
        "scores": scores,
        "predictions": pd.Series(predictions, index=df.index, name="pred_baseline"),
        "labels": labels,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "top_features": top_features(build_pipeline().fit(x, y)),
    }


def top_features(pipeline: Pipeline, k: int = 8) -> dict[str, list[str]]:
    """Termos com maior peso positivo por categoria (modelo treinado no gold set inteiro).

    Serve para checar se o modelo aprendeu algo com sentido ("atraso", "nao recebi")
    ou se está se apoiando em ruído.
    """
    vocab = pipeline.named_steps["tfidf"].get_feature_names_out()
    clf = pipeline.named_steps["clf"]
    coefs = clf.coef_
    if coefs.shape[0] == 1:  # caso binário: sklearn guarda um vetor só
        coefs = np.vstack([-coefs[0], coefs[0]])
    return {
        label: [vocab[i] for i in np.argsort(row)[::-1][:k]]
        for label, row in zip(clf.classes_, coefs, strict=True)
    }


def file_fingerprint(path: Path) -> str:
    """Hash curto do gold set: identifica com que versão dos rótulos o número foi gerado."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def _mean_std(values: np.ndarray) -> str:
    return f"{values.mean():.3f} ± {values.std():.3f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_report(df: pd.DataFrame, result: dict, gold_path: Path) -> str:
    y, pred, labels = df["categoria"], result["predictions"], result["labels"]

    metrics_rows = [
        [name, _mean_std(s["accuracy"]), _mean_std(s["f1_macro"])]
        for name, s in result["scores"].items()
    ]

    per_class = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    class_rows = [
        [
            label,
            f"{per_class[label]['precision']:.2f}",
            f"{per_class[label]['recall']:.2f}",
            f"{per_class[label]['f1-score']:.2f}",
            int(per_class[label]["support"]),
        ]
        for label in labels
    ]

    matrix = confusion_matrix(y, pred, labels=labels)
    matrix_rows = [[label, *row] for label, row in zip(labels, matrix.tolist(), strict=True)]

    feature_rows = [[label, ", ".join(result["top_features"][label])] for label in labels]

    distribution = y.value_counts().reindex(labels)
    dist_rows = [[label, n, f"{n / len(y):.0%}"] for label, n in distribution.items()]

    cv_desc = (
        f"{result['n_splits']} folds estratificados × {result['n_repeats']} repetições "
        f"({result['n_splits'] * result['n_repeats']} avaliações por modelo)"
    )

    return f"""# Baseline: TF-IDF + Regressão Logística

- **Data:** {date.today().isoformat()}
- **Gold set:** `{gold_path}` (sha256 `{file_fingerprint(gold_path)}`), {len(df)} avaliações
- **Validação:** {cv_desc}
- **Entrada:** texto da avaliação + nota como token (`nota_1` … `nota_5`)
- **Modelo:** TF-IDF (unigramas + bigramas, sem acento, `sublinear_tf`) +
  Regressão Logística (`class_weight="balanced"`)
- **scikit-learn:** {sklearn.__version__}

## Métricas (média ± desvio entre folds)

{_table(["Modelo", "Accuracy", "F1 macro"], metrics_rows)}

**F1 macro** é a métrica principal: cada categoria pesa igual, então acertar só a classe mais
comum não basta. A linha "classe majoritária" mostra o piso: um modelo que sempre chuta a
categoria mais frequente.

## Distribuição das categorias no gold set

{_table(["Categoria", "n", "%"], dist_rows)}

## Por categoria (predições out-of-fold, 1 rodada)

{_table(["Categoria", "Precisão", "Recall", "F1", "Suporte"], class_rows)}

## Matriz de confusão

Linhas = rótulo verdadeiro, colunas = predição.

{_table(["verdadeiro ↓ / predito →", *labels], matrix_rows)}

## Termos mais associados a cada categoria

Modelo treinado no gold set inteiro (apenas para interpretação; as métricas acima são
out-of-fold).

{_table(["Categoria", "Termos"], feature_rows)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    df = load_gold(args.gold)
    result = evaluate(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(df, result, args.gold), encoding="utf-8")

    for name, s in result["scores"].items():
        print(f"{name:<34} F1 macro {_mean_std(s['f1_macro'])}")
    print(f"Relatório em {args.output}")


if __name__ == "__main__":
    main()
