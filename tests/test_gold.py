from pathlib import Path

import pandas as pd
import pytest

from reviews_triage.gold import build_gold_sample, load_gold, read_labels, validate_gold


def make_reviews(per_score: int, duplicates: int = 0) -> pd.DataFrame:
    rows = [
        {"review_id": f"r{score}_{i}", "review_score": score, "text": f"Texto {score}-{i}"}
        for score in range(1, 6)
        for i in range(per_score)
    ]
    # Mesmo texto com caixa diferente: deve contar como repetido.
    rows += [
        {"review_id": f"dup{i}", "review_score": 1, "text": "TEXTO 1-0"} for i in range(duplicates)
    ]
    return pd.DataFrame(rows)


def test_gold_sample_is_stratified_unique_and_unlabeled():
    gold = build_gold_sample(make_reviews(per_score=40, duplicates=5), per_score=30)

    assert len(gold) == 150
    assert gold["review_score"].value_counts().to_dict() == {s: 30 for s in range(1, 6)}
    assert not gold["text"].str.lower().duplicated().any()
    assert (gold[["categoria", "urgencia", "observacao"]] == "").all().all()


def test_gold_sample_is_shuffled_and_reproducible():
    reviews = make_reviews(per_score=40)
    a = build_gold_sample(reviews, per_score=30, seed=1)

    assert a.equals(build_gold_sample(reviews, per_score=30, seed=1))
    # Sem embaralhar, as 30 primeiras linhas seriam todas da nota 1.
    assert a["review_score"].head(30).nunique() > 1


def test_gold_sample_fails_when_a_score_has_too_few_unique_texts():
    # 25 textos únicos na nota 1 (+ duplicatas que não contam) < 30.
    with pytest.raises(ValueError, match="insuficientes"):
        build_gold_sample(make_reviews(per_score=25, duplicates=10), per_score=30)


def labeled(categoria: list[str], urgencia: list[str]) -> pd.DataFrame:
    n = len(categoria)
    return pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(n)],
            "review_score": ["1"] * n,
            "text": ["x"] * n,
            "categoria": categoria,
            "urgencia": urgencia,
        }
    )


def test_validate_gold_accepts_valid_labels():
    assert validate_gold(labeled(["elogio", "atraso_entrega"], ["baixa", "alta"])) == []


def test_validate_gold_reports_empty_and_unknown_labels_with_line_numbers():
    errors = validate_gold(labeled(["elogio", "", "atraso"], ["baixa", "media", "urgente"]))

    assert "linha 3: categoria vazia" in errors
    assert any(e.startswith("linha 4: categoria='atraso'") for e in errors)
    assert any(e.startswith("linha 4: urgencia='urgente'") for e in errors)


def test_read_labels_handles_spreadsheet_exports(tmp_path: Path):
    # Excel pt-BR salva com BOM e ponto e vírgula.
    path = tmp_path / "gold.csv"
    content = (
        "review_id;review_score;text;categoria;urgencia\n"
        "r1;1;Não chegou, cadê?;atraso_entrega;alta\n"
    )
    path.write_bytes(content.encode("utf-8-sig"))

    gold = load_gold(path)

    assert gold.loc[0, "text"] == "Não chegou, cadê?"
    assert gold.loc[0, "categoria"] == "atraso_entrega"
    assert gold.loc[0, "review_score"] == 1
    assert list(read_labels(path).columns)[0] == "review_id"
