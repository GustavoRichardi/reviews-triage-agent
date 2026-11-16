"""Gold set: amostra para rotulagem manual e validação dos rótulos.

Fluxo:
    make gold        -> gera data/gold/to_label.csv (150 avaliações, 30 por nota)
    (você rotula e salva como data/gold/gold.csv)
    make gold-check  -> valida data/gold/gold.csv e mostra o progresso

O gerador nunca escreve em gold.csv: o arquivo rotulado à mão não pode ser
sobrescrito por acidente.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from reviews_triage.data import RAW_PATH, clean, load_raw
from reviews_triage.labels import CATEGORIES, URGENCIES

GOLD_DIR = Path("data/gold")
TO_LABEL_PATH = GOLD_DIR / "to_label.csv"
GOLD_PATH = GOLD_DIR / "gold.csv"

LABEL_COLUMNS = ["categoria", "urgencia", "observacao"]


def build_gold_sample(df: pd.DataFrame, per_score: int = 30, seed: int = 42) -> pd.DataFrame:
    """Amostra estratificada por nota, sem textos repetidos, em ordem embaralhada.

    - Estratificar garante exemplos de reclamação (notas 1-2), que numa amostra
      aleatória seriam minoria frente às notas 5.
    - Textos repetidos (ignorando caixa) são removidos antes de sortear: rotular
      "Muito bom" duas vezes não ensina nada.
    - A ordem final é embaralhada para quem rotula não ver blocos de mesma nota,
      o que induziria a rotular no "piloto automático".
    """
    unique = df.assign(_key=df["text"].str.lower()).drop_duplicates("_key").drop(columns="_key")

    counts = unique["review_score"].value_counts()
    short = counts[counts < per_score]
    if len(short) or counts.size == 0:
        raise ValueError(f"Textos únicos insuficientes por nota (preciso de {per_score}): {short}")

    sampled = unique.groupby("review_score").sample(n=per_score, random_state=seed)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    out = sampled[["review_id", "review_score", "text"]].copy()
    for column in LABEL_COLUMNS:
        out[column] = ""
    return out


def read_labels(path: Path) -> pd.DataFrame:
    """Lê o CSV rotulado tolerando o que planilhas costumam fazer ao salvar
    (BOM do Excel, separador `;`)."""
    return pd.read_csv(
        path, sep=None, engine="python", encoding="utf-8-sig", dtype=str, keep_default_na=False
    )


def validate_gold(df: pd.DataFrame) -> list[str]:
    """Retorna a lista de problemas encontrados (vazia = gold set pronto)."""
    errors = []
    missing = {"review_id", "text", "categoria", "urgencia"} - set(df.columns)
    if missing:
        return [f"colunas ausentes: {sorted(missing)}"]

    if df["review_id"].duplicated().any():
        errors.append("review_id repetido")

    for column, allowed in (("categoria", CATEGORIES), ("urgencia", URGENCIES)):
        values = df[column].str.strip()
        for line, value in values.items():
            # +2: linha 1 é o cabeçalho e o índice começa em 0.
            if value == "":
                errors.append(f"linha {line + 2}: {column} vazia")
            elif value not in allowed:
                errors.append(f"linha {line + 2}: {column}={value!r} não está em {allowed}")
    return errors


def load_gold(path: Path = GOLD_PATH) -> pd.DataFrame:
    """Carrega o gold set rotulado e falha se houver qualquer problema."""
    df = read_labels(path)
    errors = validate_gold(df)
    if errors:
        raise ValueError(f"{path} inválido:\n" + "\n".join(errors))
    df["categoria"] = df["categoria"].str.strip()
    df["urgencia"] = df["urgencia"].str.strip()
    df["review_score"] = df["review_score"].astype(int)
    return df


def _sample_command(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.force:
        sys.exit(f"{args.output} já existe. Use --force para sobrescrever.")
    gold = build_gold_sample(clean(load_raw(args.input)), args.per_score, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: o Excel só reconhece acentos em CSV UTF-8 se houver BOM.
    gold.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"{len(gold)} avaliações em {args.output}")
    print(gold["review_score"].value_counts().sort_index().to_string())


def _check_command(args: argparse.Namespace) -> None:
    if not args.path.exists():
        sys.exit(f"{args.path} não existe. Rotule {TO_LABEL_PATH} e salve como {args.path}.")
    df = read_labels(args.path)
    errors = validate_gold(df)
    if "categoria" in df.columns:
        done = (df["categoria"].str.strip() != "").sum()
        print(f"Rotuladas: {done}/{len(df)}")
    if errors:
        print(f"{len(errors)} problema(s):")
        print("\n".join(f"  - {e}" for e in errors[:30]))
        sys.exit(1)
    print("Gold set válido.\n")
    print(pd.crosstab(df["categoria"], df["urgencia"], margins=True, margins_name="total"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold set: amostragem e validação.")
    sub = parser.add_subparsers(dest="command", required=True)

    sample_cmd = sub.add_parser("sample", help="gera o CSV para rotular")
    sample_cmd.add_argument("--input", type=Path, default=RAW_PATH)
    sample_cmd.add_argument("--output", type=Path, default=TO_LABEL_PATH)
    sample_cmd.add_argument("--per-score", type=int, default=30)
    sample_cmd.add_argument("--seed", type=int, default=42)
    sample_cmd.add_argument("--force", action="store_true")
    sample_cmd.set_defaults(func=_sample_command)

    check_cmd = sub.add_parser("check", help="valida o CSV rotulado")
    check_cmd.add_argument("--path", type=Path, default=GOLD_PATH)
    check_cmd.set_defaults(func=_check_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
