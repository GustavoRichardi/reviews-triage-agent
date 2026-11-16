"""Carga, limpeza e amostragem das avaliações da Olist.

Uso:
    python -m reviews_triage.data            # (ou `make data`)

Lê data/raw/olist_order_reviews_dataset.csv e grava em data/processed/:
- reviews_clean.csv: todas as avaliações com comentário, limpas e sem duplicatas;
- reviews_sample.csv: amostra aleatória reprodutível para desenvolvimento.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/olist_order_reviews_dataset.csv")
PROCESSED_DIR = Path("data/processed")

# Colunas que o resto do projeto usa. `text` é o que os modelos leem.
OUTPUT_COLUMNS = ["review_id", "order_id", "review_score", "review_creation_date", "text"]

_WHITESPACE = re.compile(r"\s+")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Lê o CSV bruto como veio do Kaggle (texto como string, datas parseadas)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Baixe o dataset da Olist no Kaggle (ver README)."
        )
    return pd.read_csv(
        path,
        dtype={"review_comment_title": "string", "review_comment_message": "string"},
        parse_dates=["review_creation_date", "review_answer_timestamp"],
    )


def normalize_text(value: object) -> str:
    """Troca quebras de linha e espaços repetidos por um espaço só.

    Limpeza mínima de propósito: não tira acento, não muda caixa, não remove
    pontuação. Cada modelo faz seu próprio pré-processamento (o TF-IDF
    normaliza do jeito dele; o LLM se beneficia do texto original).
    """
    if value is None or pd.isna(value):
        return ""
    return _WHITESPACE.sub(" ", str(value)).strip()


def build_text(title: str, message: str) -> str:
    """Junta título e mensagem num texto só. Título vazio é ignorado."""
    if not title:
        return message
    separator = " " if title[-1] in ".!?" else ". "
    return f"{title}{separator}{message}"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém só avaliações com comentário, normaliza o texto e remove duplicatas."""
    out = df.copy()
    title = out["review_comment_title"].map(normalize_text)
    message = out["review_comment_message"].map(normalize_text)

    # Requisito do projeto: só avaliações com comentário preenchido.
    # Título sem mensagem não conta (títulos costumam ser genéricos: "Ótimo", "Recomendo").
    has_message = message != ""
    out = out[has_message].copy()
    pairs = zip(title[has_message], message[has_message], strict=True)
    out["text"] = [build_text(t, m) for t, m in pairs]

    # O mesmo review_id aparece em mais de um pedido no dataset original.
    # Mantemos uma linha por avaliação para não contar o mesmo texto duas vezes.
    out = out.drop_duplicates(subset="review_id", keep="first")

    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Amostra aleatória simples e reprodutível (a mesma seed gera a mesma amostra)."""
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = load_raw(args.input)
    cleaned = clean(raw)
    sampled = sample(cleaned, args.sample_size, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output_dir / "reviews_clean.csv", index=False)
    sampled.to_csv(args.output_dir / "reviews_sample.csv", index=False)

    print(f"Linhas brutas:            {len(raw):>7}")
    print(f"Com comentário (limpas):  {len(cleaned):>7} ({len(cleaned) / len(raw):.1%})")
    print(f"Amostra (seed={args.seed}):       {len(sampled):>7}")
    print(f"Arquivos em {args.output_dir}/")


if __name__ == "__main__":
    main()
