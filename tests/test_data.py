from pathlib import Path

import pandas as pd
import pytest

from reviews_triage.data import OUTPUT_COLUMNS, build_text, clean, load_raw, normalize_text, sample

# Mini-CSV com o mesmo esquema do arquivo do Kaggle e os casos chatos que aparecem nele:
# quebra de linha \r\n dentro do comentário, comentário vazio, só espaços,
# título sem mensagem e review_id duplicado.
RAW_CSV = (
    "review_id,order_id,review_score,review_comment_title,review_comment_message,"
    "review_creation_date,review_answer_timestamp\n"
    'r1,o1,5,,"Chegou antes do prazo,\r\nadorei!",2018-01-18 00:00:00,2018-01-18 21:46:59\n'
    "r2,o2,1,Produto errado,Veio outra cor,2018-03-10 00:00:00,2018-03-11 03:05:13\n"
    "r3,o3,4,,,2018-02-17 00:00:00,2018-02-18 14:36:24\n"
    'r4,o4,3,,"   ",2017-04-21 00:00:00,2017-04-21 22:02:06\n'
    "r5,o5,5,Recomendo,,2018-03-01 00:00:00,2018-03-02 10:26:53\n"
    "r2,o6,1,Produto errado,Veio outra cor,2018-03-10 00:00:00,2018-03-11 03:05:13\n"
)


@pytest.fixture
def raw_path(tmp_path: Path) -> Path:
    path = tmp_path / "reviews.csv"
    path.write_bytes(RAW_CSV.encode("utf-8"))
    return path


def test_load_raw_missing_file_has_helpful_message(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_raw(tmp_path / "nao_existe.csv")


def test_normalize_text_collapses_whitespace_and_keeps_accents():
    assert normalize_text("  Não\r\nchegou   ainda ") == "Não chegou ainda"
    assert normalize_text(pd.NA) == ""
    assert normalize_text(None) == ""


def test_build_text_joins_title_and_message():
    assert build_text("", "Veio quebrado") == "Veio quebrado"
    assert build_text("Péssimo", "Veio quebrado") == "Péssimo. Veio quebrado"
    assert build_text("Péssimo!", "Veio quebrado") == "Péssimo! Veio quebrado"


def test_clean_keeps_only_reviews_with_message(raw_path: Path):
    cleaned = clean(load_raw(raw_path))

    assert list(cleaned.columns) == OUTPUT_COLUMNS
    # r3 (vazio), r4 (só espaços) e r5 (só título) saem; r2 duplicado vira uma linha.
    assert cleaned["review_id"].tolist() == ["r1", "r2"]
    assert cleaned["text"].tolist() == [
        "Chegou antes do prazo, adorei!",
        "Produto errado. Veio outra cor",
    ]


def test_sample_is_reproducible_and_capped(raw_path: Path):
    cleaned = clean(load_raw(raw_path))

    assert sample(cleaned, n=1, seed=7).equals(sample(cleaned, n=1, seed=7))
    assert len(sample(cleaned, n=100)) == len(cleaned)
