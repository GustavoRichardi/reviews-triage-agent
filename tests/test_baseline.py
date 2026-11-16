from pathlib import Path

import pandas as pd
import pytest

from reviews_triage.baseline import (
    build_features,
    choose_n_splits,
    evaluate,
    main,
    render_report,
)
from reviews_triage.labels import CATEGORIES


def test_features_include_score_token():
    df = pd.DataFrame({"review_score": [1], "text": ["Não chegou"]})
    assert build_features(df).iloc[0] == "nota_1 Não chegou"


def test_n_splits_is_capped_by_smallest_class():
    assert choose_n_splits(pd.Series(["a"] * 10 + ["b"] * 3)) == 3
    assert choose_n_splits(pd.Series(["a"] * 10 + ["b"] * 10)) == 5
    with pytest.raises(ValueError):
        choose_n_splits(pd.Series(["a"] * 10 + ["b"]))


def test_baseline_learns_obvious_signal_and_beats_majority(synthetic_gold):
    result = evaluate(synthetic_gold, n_repeats=2)
    scores = result["scores"]
    f1_model = scores["TF-IDF + Regressão Logística"]["f1_macro"].mean()
    f1_dummy = scores["Classe majoritária (referência)"]["f1_macro"].mean()

    assert f1_model > 0.8
    assert f1_model > f1_dummy
    # Out-of-fold: uma predição por exemplo, sempre uma categoria válida.
    assert len(result["predictions"]) == len(synthetic_gold)
    assert set(result["predictions"]) <= set(CATEGORIES)
    # Os termos mais fortes de "atraso" vêm das frases de atraso, não de outras categorias.
    atraso_vocab = {"recebi", "atrasado", "entrega", "prazo", "estourou", "chegou", "muito"}
    top_atraso = " ".join(result["top_features"]["atraso_entrega"]).split()
    assert set(top_atraso[:3]) <= atraso_vocab


def test_report_has_all_sections(synthetic_gold, tmp_path: Path):
    gold_path = tmp_path / "gold.csv"
    synthetic_gold.to_csv(gold_path, index=False)

    report = render_report(synthetic_gold, evaluate(synthetic_gold, n_repeats=1), gold_path)

    for section in ["## Métricas", "## Por categoria", "## Matriz de confusão", "sha256"]:
        assert section in report


def test_cli_writes_report(synthetic_gold, tmp_path: Path, monkeypatch):
    gold_path, report_path = tmp_path / "gold.csv", tmp_path / "out" / "baseline.md"
    synthetic_gold.to_csv(gold_path, index=False)
    monkeypatch.setattr(
        "sys.argv", ["baseline", "--gold", str(gold_path), "--output", str(report_path)]
    )

    main()

    assert report_path.read_text(encoding="utf-8").startswith("# Baseline")
