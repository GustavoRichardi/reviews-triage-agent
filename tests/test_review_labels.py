from pathlib import Path

import pandas as pd
import pytest

from reviews_triage.review_labels import (
    blind_ids,
    build_rows,
    labeling_stats,
    parse_answer,
    render_labeling_report,
    review_loop,
    save_rows,
)

TO_LABEL = pd.DataFrame(
    {
        "review_id": [f"r{i}" for i in range(5)],
        "review_score": ["1", "2", "3", "4", "5"],
        "text": ["Não chegou", "Veio errado", "Quebrou", "Loja não responde", "Amei"],
    }
)
PRELABELS = pd.DataFrame(
    {
        "review_id": [f"r{i}" for i in range(5)],
        "categoria_llm": [
            "atraso_entrega",
            "produto_errado_ou_faltando",
            "defeito_ou_qualidade",
            "atraso_entrega",
            "elogio",
        ],
        "urgencia_llm": ["media", "media", "media", "alta", "baixa"],
        "justificativa_llm": ["j"] * 5,
    }
)


def run_session(answers: list[str], blind=frozenset(), progress=None):
    rows = build_rows(TO_LABEL, PRELABELS, progress, set(blind))
    saved, output = [], []
    feed = iter(answers)
    review_loop(
        rows,
        lambda r: saved.append([dict(x) for x in r]),
        ask=lambda _: next(feed),
        say=output.append,
    )
    return rows, saved, "\n".join(output)


def test_parse_answer_applies_category_number_urgency_letter_and_note():
    assert parse_answer("", "elogio", "baixa") == ("elogio", "baixa", "")
    assert parse_answer("3a -- atrasou e quebrou", "elogio", "baixa") == (
        "defeito_ou_qualidade",
        "alta",
        "atrasou e quebrou",
    )
    assert parse_answer("m", "elogio", "baixa")[1] == "media"
    with pytest.raises(ValueError, match="desconhecida"):
        parse_answer("x", "elogio", "baixa")
    with pytest.raises(ValueError, match="informe"):
        parse_answer("3", None, None)  # modo cego exige categoria e urgência


def test_enter_accepts_and_typing_corrects_saving_after_each_answer():
    rows, saved, _ = run_session(["", "", "", "4m", ""])

    assert [r["categoria"] for r in rows] == [
        "atraso_entrega",
        "produto_errado_ou_faltando",
        "defeito_ou_qualidade",
        "atendimento_vendedor",
        "elogio",
    ]
    assert rows[3]["urgencia"] == "media"  # corrigiu "alta" → "media"
    assert len(saved) == 5  # salva a cada resposta


def test_blind_mode_hides_the_suggestion_until_you_answer():
    rows, _, output = run_session(["", "", "", "1a", "t", ""], blind={"r3"})

    item4 = output.split("[4/5]")[1].split("[5/5]")[0]
    before_answer, after_answer = item4.split("✓")
    assert "Sugestão:" not in before_answer and "j" not in before_answer.split('"')[-1]
    assert "igual à sugestão" in after_answer  # só aparece depois que você respondeu
    assert rows[3]["categoria_cega"] == "atraso_entrega" and rows[3]["urgencia_cega"] == "alta"
    assert rows[3]["modo"] == "cego"
    # Rotulou igual à sugestão (atraso/alta), então nem pergunta: "t" vira a resposta da linha 5.
    assert rows[4]["categoria"] == "elogio"


def test_blind_disagreement_lets_you_keep_yours_or_take_the_suggestion():
    rows, _, _ = run_session(["", "", "", "4a", "t", ""], blind={"r3"})
    assert rows[3]["categoria_cega"] == "atendimento_vendedor"
    assert rows[3]["categoria"] == "atraso_entrega"  # trocou pela sugestão depois de ver

    rows, _, _ = run_session(["", "", "", "4a", "", ""], blind={"r3"})
    assert rows[3]["categoria"] == "atendimento_vendedor"  # manteve o seu


def test_back_quit_and_resume_without_losing_decisions(tmp_path: Path):
    rows, _, _ = run_session(["", "6b", "v", "", "q"])  # erra a 2ª, volta, aceita, sai
    assert rows[1]["categoria"] == "produto_errado_ou_faltando"
    assert rows[2]["categoria"] == ""

    path = tmp_path / "gold.csv"
    save_rows(rows, path)
    progress = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    resumed, _, output = run_session(["", "", ""], progress=progress)
    assert "[3/5]" in output and "[1/5]" not in output  # retoma na primeira pendente
    assert all(r["categoria"] for r in resumed)


def test_invalid_input_is_explained_and_asked_again():
    rows, _, output = run_session(["9", "", "", "", "", ""])
    assert "desconhecida" in output and rows[0]["categoria"] == "atraso_entrega"


def test_blind_sample_is_stratified_by_score():
    to_label = pd.DataFrame(
        {
            "review_id": [f"r{s}_{i}" for s in range(1, 6) for i in range(30)],
            "review_score": [str(s) for s in range(1, 6) for _ in range(30)],
            "text": "x",
        }
    )
    ids = blind_ids(to_label, 30)
    assert len(ids) == 30
    assert (
        to_label[to_label["review_id"].isin(ids)]["review_score"].value_counts().tolist() == [6] * 5
    )
    assert blind_ids(to_label, 30) == ids and blind_ids(to_label, 0) == set()


def test_labeling_stats_measure_corrections_and_blind_agreement():
    # Revisão: corrigiu 1 categoria de 3. Cego: 2 itens, concordou em 1.
    gold = pd.DataFrame(
        {
            "categoria": ["elogio", "outro", "atraso_entrega", "elogio", "defeito_ou_qualidade"],
            "urgencia": ["baixa", "baixa", "media", "baixa", "media"],
            "modo": ["revisao", "revisao", "revisao", "cego", "cego"],
            "categoria_llm": ["elogio", "elogio", "atraso_entrega", "elogio", "atraso_entrega"],
            "urgencia_llm": ["baixa", "baixa", "alta", "baixa", "media"],
            "categoria_cega": ["", "", "", "elogio", "defeito_ou_qualidade"],
            "urgencia_cega": ["", "", "", "baixa", "media"],
        }
    )
    stats = labeling_stats(gold)

    assert stats["revisao"]["categoria_corrigida"]["pct"] == pytest.approx(1 / 3)
    assert stats["revisao"]["urgencia_corrigida"]["pct"] == pytest.approx(1 / 3)
    assert stats["cego"]["categoria_concorda"]["pct"] == 0.5
    assert stats["correcoes_mais_comuns"] == {
        "elogio → outro": 1,
        "atraso_entrega → defeito_ou_qualidade": 1,
    }
    report = render_labeling_report(stats, {"model": "claude-fable-5-2", "prompt_version": "abc"})
    assert "claude-fable-5-2" in report and "33%" in report and "ancorando" in report
