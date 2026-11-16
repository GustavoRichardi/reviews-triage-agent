from pathlib import Path

import pandas as pd
import pytest
from fakes import FakeLLM, hash_embed

from reviews_triage.agent import TriageAgent
from reviews_triage.evaluation import EvalRecord, execute_pilot, load_records
from reviews_triage.judge import JUDGE_MODEL, Judge, JudgeScores
from reviews_triage.judge_validation import (
    agreement,
    export_for_scoring,
    load_human_scores,
    render_agreement_report,
)
from reviews_triage.policy import build_index, load_chunks
from reviews_triage.schemas import TriageResult
from reviews_triage.stats import wilson_interval

INDEX = build_index(Path("docs/politicas"), embed=hash_embed)
JUDGE = JudgeScores(justificativa="Cita o prazo certo.", aderencia=4, tom=5, utilidade=3)


@pytest.fixture
def pilot_run(tmp_path: Path) -> Path:
    sample = pd.DataFrame(
        [
            {"review_id": f"p{s}_{i}", "review_score": s, "text": f"Texto {s}-{i}"}
            for s in range(1, 6)
            for i in range(2)
        ]
    )
    agent = TriageAgent(FakeLLM(model="claude-sonnet-5-5"), INDEX)
    judge = Judge(FakeLLM(model=JUDGE_MODEL, judge_scores=JUDGE), load_chunks())
    _, run_dir = execute_pilot(
        sample,
        agent,
        judge,
        "claude-sonnet-5-5",
        "low",
        reports_dir=tmp_path,
        validation_dir=tmp_path / "val",
    )
    return run_dir


def human_file(path: Path, scores: dict[str, int], records) -> pd.DataFrame:
    df = pd.DataFrame([{"review_id": r.review_id, **scores} for r in records])
    df.to_csv(path, index=False)
    return load_human_scores(path)


def test_export_is_blind_shuffled_and_never_overwrites(pilot_run: Path, tmp_path: Path):
    path = export_for_scoring(pilot_run, out_dir=tmp_path / "val")
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    raw = path.read_text(encoding="utf-8-sig")

    assert len(df) == 10 and df["item"].tolist() == list(range(1, 11))
    assert (df[["aderencia", "tom", "utilidade"]] == "").all().all()  # notas em branco
    assert "Cita o prazo certo." not in raw  # nada do juiz no arquivo
    assert df["review_id"].tolist() != [r.review_id for r in load_records(pilot_run)]
    with pytest.raises(FileExistsError):
        export_for_scoring(pilot_run, out_dir=tmp_path / "val")


def test_human_scores_are_validated_with_line_numbers(tmp_path: Path):
    path = tmp_path / "human.csv"
    path.write_bytes("review_id;aderencia;tom;utilidade\nr1;5;4;3\nr2;6;;3,5\n".encode("utf-8-sig"))
    with pytest.raises(ValueError) as exc:
        load_human_scores(path)
    message = str(exc.value)
    assert "linha 3: aderencia='6'" in message and "linha 3: tom=''" in message


def judged_records(scores: list[int]) -> list[EvalRecord]:
    """Registros com notas do juiz variadas (o FakeLLM dá sempre a mesma nota)."""
    result = TriageResult(
        categoria="elogio",
        sentimento="positivo",
        urgencia="baixa",
        resposta_sugerida="Obrigado!",
        fontes=[],
        escalate_to_human=False,
        tentativas_redacao=1,
        llm_calls=[],
    )
    return [
        EvalRecord(
            review_id=f"r{i}",
            review_score=5,
            text="x",
            result=result,
            judge=JudgeScores(justificativa="j", aderencia=s, tom=s, utilidade=s),
        )
        for i, s in enumerate(scores)
    ]


def human_scores(tmp_path: Path, scores: list[int]) -> pd.DataFrame:
    path = tmp_path / "human.csv"
    pd.DataFrame(
        [
            {"review_id": f"r{i}", "aderencia": s, "tom": s, "utilidade": s}
            for i, s in enumerate(scores)
        ]
    ).to_csv(path, index=False)
    return load_human_scores(path)


def test_perfect_agreement_on_varied_scores_passes(tmp_path: Path):
    scores = [5, 5, 4, 4, 3, 2, 1, 5, 4, 2]
    result = agreement(judged_records(scores), human_scores(tmp_path, scores))

    for m in result["criterios"].values():
        assert m["exata"] == 1.0 and m["kappa_quadratico"] == pytest.approx(1.0)
        assert m["veredito"] == "aprovado"
    assert result["discordancias"] == []


def test_lazy_judge_that_always_says_5_is_caught_by_kappa(tmp_path: Path):
    human = [5, 5, 5, 5, 5, 5, 5, 5, 4, 1]  # quase tudo bom, uma resposta péssima
    result = agreement(judged_records([5] * 10), human_scores(tmp_path, human))

    m = result["criterios"]["aderencia"]
    assert m["ate_1_ponto"] == 0.9 and abs(m["vies_juiz_menos_humano"]) <= 0.5  # passaria
    assert m["kappa_quadratico"] == pytest.approx(0.0)  # mas não distingue nada
    assert m["veredito"] == "reprovado"
    assert len(result["discordancias"]) == 1  # a resposta péssima que o juiz deixou passar


def test_no_variation_at_all_is_inconclusive_not_approved(tmp_path: Path):
    result = agreement(judged_records([5] * 10), human_scores(tmp_path, [5] * 10))
    assert {m["veredito"] for m in result["criterios"].values()} == {"inconclusivo"}
    assert "inconclusivo" in render_agreement_report("x", result)


def test_lenient_judge_fails_on_bias_and_disagreements_are_listed(pilot_run: Path, tmp_path: Path):
    records = load_records(pilot_run)
    # Você dá 2 em aderência onde o juiz deu 4: viés +2, nenhuma nota a até 1 ponto.
    human = human_file(tmp_path / "h.csv", {"aderencia": 2, "tom": 4, "utilidade": 3}, records)

    result = agreement(records, human)

    assert result["criterios"]["aderencia"]["vies_juiz_menos_humano"] == pytest.approx(2.0)
    assert result["criterios"]["aderencia"]["veredito"] == "reprovado"
    # Tom: juiz 5 × você 4 em tudo. "Até 1 ponto" dá 100%, mas o juiz é sempre mais generoso:
    # o critério de viés existe exatamente para reprovar esse caso.
    tom = result["criterios"]["tom"]
    assert tom["ate_1_ponto"] == 1.0 and tom["vies_juiz_menos_humano"] == 1.0
    assert tom["veredito"] == "reprovado"
    # Utilidade: juiz 3 × você 3 em tudo. Concorda, mas sem variação não prova nada.
    assert result["criterios"]["utilidade"]["veredito"] == "inconclusivo"
    assert len(result["discordancias"]) == 10
    report = render_agreement_report("piloto-teste", result)
    assert "❌ reprovado" in report and "indicativo" in report  # n = 10 < 30


def test_agreement_rejects_reviews_the_judge_never_saw(pilot_run: Path, tmp_path: Path):
    records = load_records(pilot_run)
    path = tmp_path / "h.csv"
    pd.DataFrame([{"review_id": "nao-existe", "aderencia": 3, "tom": 3, "utilidade": 3}]).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="sem nota do juiz"):
        agreement(records, load_human_scores(path))


def test_wilson_interval_is_honest_about_small_samples():
    low, high = wilson_interval(9, 10)
    assert low < 0.6 < 0.9 < high <= 1.0  # 9 de 10 ainda é compatível com ~60%
    narrow = wilson_interval(90, 100)
    assert narrow[1] - narrow[0] < high - low
    assert wilson_interval(0, 0) is None
