import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import make_synthetic_gold
from fakes import FakeLLM, hash_embed

from reviews_triage.agent import TriageAgent
from reviews_triage.evaluation import (
    EXPECTED_SOURCES,
    bootstrap_f1,
    call_cost,
    estimate_cost,
    execute_run,
    load_run,
    render_comparison,
    retrieval_hit_rates,
    run_eval,
)
from reviews_triage.gold import load_gold
from reviews_triage.judge import JUDGE_MODEL, Judge
from reviews_triage.policy import build_index, load_chunks
from reviews_triage.schemas import Classification, LLMCall

INDEX = build_index(Path("docs/politicas"), embed=hash_embed)


@pytest.fixture
def gold(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    path = tmp_path / "gold.csv"
    make_synthetic_gold(per_class=10).to_csv(path, index=False)
    return load_gold(path), path


def agent_and_judge(model="claude-sonnet-5-5", **llm_kwargs):
    agent = TriageAgent(FakeLLM(model=model, **llm_kwargs), INDEX)
    judge = Judge(FakeLLM(model=JUDGE_MODEL), load_chunks())
    return agent, judge


def test_expected_sources_cover_every_category_with_existing_files():
    files = {p.name for p in Path("docs/politicas").glob("*.md")}
    assert set(EXPECTED_SOURCES) == set(make_synthetic_gold()["categoria"])
    assert all(sources <= files for sources in EXPECTED_SOURCES.values())


def test_call_cost_uses_the_price_table_and_rejects_unknown_models():
    call = LLMCall(
        step="x",
        model="claude-sonnet-5-5",
        input_tokens=1_000_000,
        output_tokens=100_000,
        latency_s=0,
    )
    assert call_cost(call) == pytest.approx(2.00 + 1.00)
    with pytest.raises(KeyError):
        call_cost(call.model_copy(update={"model": "modelo-sem-preco"}))
    agent_cost, judge_cost = estimate_cost("claude-haiku-5-5", 150)
    assert 0 < agent_cost < judge_cost  # Haiku é bem mais barato que o juiz


def test_bootstrap_ci_contains_the_point_estimate_and_is_paired():
    y = np.array(["a", "b"] * 50)
    same = bootstrap_f1(y, y, y, ["a", "b"], n=200)
    assert same["a"] == [1.0, 1.0] and same["diff"] == [0.0, 0.0]  # pareado: diferença exata

    worse = y.copy()
    worse[:20] = "b"
    ci = bootstrap_f1(y, y, worse, ["a", "b"], n=200)
    assert ci["diff"][0] > 0  # A melhor que B em toda reamostragem


def test_run_eval_isolates_errors_and_keeps_gold_order(gold):
    df, _ = gold
    agent, judge = agent_and_judge()
    original = agent.run

    def flaky(text, score):
        if text == df["text"].iloc[3]:
            raise ConnectionError("API fora do ar")
        return original(text, score)

    agent.run = flaky
    records = run_eval(df, agent, judge, workers=4)

    assert [r.review_id for r in records] == df["review_id"].tolist()
    assert sum(r.error is not None for r in records) >= 1
    ok = [r for r in records if r.error is None]
    assert all(r.judge is not None and r.latency_s is not None for r in ok)


def test_execute_run_writes_raw_results_metrics_and_report(gold, tmp_path: Path):
    df, path = gold
    always_atraso = Classification(
        categoria="atraso_entrega", sentimento="negativo", urgencia="alta"
    )
    agent, judge = agent_and_judge(classification=always_atraso)

    report = execute_run(
        df,
        df,
        agent,
        judge,
        "claude-sonnet-5-5",
        "low",
        path,
        reports_dir=tmp_path / "reports",
        workers=2,
    )

    run_dir = next((tmp_path / "reports" / "runs").iterdir())
    meta, summary = load_run(run_dir)
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(df) and json.loads(lines[0])["review_id"] == df["review_id"].iloc[0]
    assert meta.judge_prompt_version == "edf6a162" and meta.judge_model == JUDGE_MODEL
    assert set(meta.prompt_versions) == {"classify", "draft", "review"}

    # Sempre "atraso" em 6 categorias balanceadas: accuracy de 1/6, urgência alta em tudo.
    assert summary["categoria"]["agent_accuracy"] == pytest.approx(1 / 6)
    assert summary["revisao"]["escalada"] == 1.0
    assert summary["juiz"]["aderencia"]["mean"] == 5
    assert summary["custo"]["agente_por_1000"] > 0
    assert set(summary["busca_sem_llm"]) == {"só o texto", "dica da categoria (gold) + texto"}

    text = report.read_text(encoding="utf-8")
    for section in [
        "## Resumo",
        "## Busca de política",
        "## Juiz",
        "## Custo e tokens",
        "edf6a162",
    ]:
        assert section in text


def test_judge_can_never_evaluate_its_own_model(gold, tmp_path: Path):
    df, path = gold
    agent, judge = agent_and_judge(model=JUDGE_MODEL)
    with pytest.raises(ValueError, match="próprio modelo"):
        execute_run(df, df, agent, judge, JUDGE_MODEL, "low", path, reports_dir=tmp_path)


def test_comparison_flags_runs_that_are_not_comparable(gold, tmp_path: Path):
    df, path = gold
    for model in ("claude-sonnet-5-5", "claude-haiku-5-5"):
        agent, judge = agent_and_judge(model=model)
        execute_run(df, df, agent, judge, model, "low", path, reports_dir=tmp_path, workers=2)
    runs = [load_run(p) for p in sorted((tmp_path / "runs").iterdir())]

    assert "✅" in render_comparison(runs)
    changed = runs[1][0].model_copy(update={"judge_prompt_version": "outra"})
    text = render_comparison([runs[0], (changed, runs[1][1])])
    assert "NÃO comparáveis" in text and "rubrica do juiz" in text


def test_retrieval_hit_rates_are_proportions(gold):
    df, _ = gold
    rates = retrieval_hit_rates(df, INDEX)
    for values in rates.values():
        assert 0 <= values["hit@1"] <= values["hit@3"] <= 1


def test_cli_without_yes_only_estimates_and_spends_nothing(gold, monkeypatch, capsys):
    from reviews_triage import evaluation

    _, path = gold
    monkeypatch.setattr(evaluation, "AnthropicLLM", lambda **kw: pytest.fail("não deveria chamar"))
    monkeypatch.setattr(
        "sys.argv",
        ["eval", "run", "--model", "claude-haiku-5-5", "--gold", str(path), "--limit", "5"],
    )

    evaluation.main()

    out = capsys.readouterr().out
    assert "5 avaliações" in out and "US$" in out and "Nada foi executado" in out


def test_cli_explains_missing_gold_set(monkeypatch, tmp_path):
    from reviews_triage import evaluation

    monkeypatch.setattr("sys.argv", ["eval", "retrieval", "--gold", str(tmp_path / "gold.csv")])
    with pytest.raises(SystemExit, match="gold-check"):
        evaluation.main()


# --- Amostra do juiz -------------------------------------------------------------------


def test_judge_sample_is_stratified_exact_and_deterministic():
    from reviews_triage.evaluation import judge_sample_ids

    gold = make_synthetic_gold(per_class=25)  # 150, 25 por categoria
    ids = judge_sample_ids(gold, 60)

    assert len(ids) == 60
    counts = gold[gold["review_id"].isin(ids)]["categoria"].value_counts()
    assert set(counts) == {10}  # 60/150 de 25 = 10 por categoria
    assert judge_sample_ids(gold, 60) == ids  # mesma amostra em toda rodada
    assert judge_sample_ids(gold, None) is None and judge_sample_ids(gold, 500) is None


def test_judge_sample_allocation_sums_exactly_with_unbalanced_categories():
    from reviews_triage.evaluation import judge_sample_ids

    gold = make_synthetic_gold(per_class=25)
    unbalanced = gold[~((gold["categoria"] == "outro") & (gold.index % 3 == 0))]
    for size in (7, 31, 60):
        assert len(judge_sample_ids(unbalanced, size)) == size


def test_execute_run_judges_only_the_sample_and_records_it(gold, tmp_path: Path):
    df, path = gold  # 60 avaliações
    agent, judge = agent_and_judge()

    execute_run(
        df,
        df,
        agent,
        judge,
        "claude-sonnet-5-5",
        "low",
        path,
        reports_dir=tmp_path,
        workers=2,
        judge_sample=12,
    )

    meta, summary = load_run(next((tmp_path / "runs").iterdir()))
    assert summary["juiz"]["n"] == 12 and summary["n"] == 60
    assert meta.judge_sample_size == 12 and meta.judge_sample_sha != "todas"
    assert summary["juiz"]["aderencia"]["ci95"] is not None
    assert (
        estimate_cost("claude-sonnet-5-5", 150, 60)[1] < estimate_cost("claude-sonnet-5-5", 150)[1]
    )


def test_comparison_flags_different_judge_samples(gold, tmp_path: Path):
    df, path = gold
    for sample in (12, 24):
        agent, judge = agent_and_judge()
        execute_run(
            df,
            df,
            agent,
            judge,
            "claude-sonnet-5-5",
            "low",
            path,
            reports_dir=tmp_path / str(sample),
            workers=2,
            judge_sample=sample,
        )
    runs = [load_run(next((tmp_path / str(s) / "runs").iterdir())) for s in (12, 24)]
    assert "amostra do juiz" in render_comparison(runs)


# --- Piloto ----------------------------------------------------------------------------


def clean_reviews(per_score: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"review_id": f"c{s}_{i}", "review_score": s, "text": f"Avaliação {s} número {i}"}
            for s in range(1, 6)
            for i in range(per_score)
        ]
    )


def test_pilot_never_uses_reviews_from_the_gold_set(tmp_path: Path):
    from reviews_triage.evaluation import pilot_sample
    from reviews_triage.gold import build_gold_sample

    reviews = clean_reviews()
    future_gold = set(build_gold_sample(reviews)["review_id"])  # o que `make gold` vai sortear
    already_labeled = tmp_path / "to_label.csv"
    pool = reviews[~reviews["review_id"].isin(future_gold)]
    pool.head(3).assign(categoria="", urgencia="").to_csv(already_labeled, index=False)

    sample = pilot_sample(reviews, 10, [already_labeled, tmp_path / "gold.csv"])

    assert len(sample) == 10
    assert sample["review_score"].value_counts().to_dict() == {s: 2 for s in range(1, 6)}
    assert not set(sample["review_id"]) & future_gold
    assert not set(sample["review_id"]) & set(pool.head(3)["review_id"])
    with pytest.raises(ValueError, match="múltiplo de 5"):
        pilot_sample(reviews, 7, [])


def test_pilot_report_measures_cost_and_hides_judge_scores(tmp_path: Path):
    from reviews_triage.evaluation import execute_pilot, find_runs
    from reviews_triage.judge_validation import export_for_scoring

    sample = clean_reviews(per_score=2)
    agent, judge = agent_and_judge()

    report, run_dir = execute_pilot(
        sample,
        agent,
        judge,
        "claude-sonnet-5-5",
        "low",
        reports_dir=tmp_path,
        validation_dir=tmp_path / "val",
    )
    scoring = export_for_scoring(run_dir, out_dir=tmp_path / "val")

    text = report.read_text(encoding="utf-8")
    assert "## Custo medido" in text and "Projeção das rodadas completas" in text
    assert all(m in text for m in ("claude-opus-5-5", "claude-haiku-5-5", "Três rodadas"))
    assert "Segue a política." not in text  # justificativa do juiz (FakeLLM) não vaza
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert "juiz" not in summary and summary["custo"]["juiz_tokens"]["chamadas"] == 10
    meta, _ = load_run(run_dir)
    assert meta.kind == "pilot" and meta.run_id.endswith("_piloto")
    assert find_runs(kind="gold", runs_dir=tmp_path / "runs") == []  # compare ignora pilotos
    assert str(scoring) in text or scoring.name in text
