"""Revisão rápida dos rótulos do gold set, uma avaliação por vez, no terminal.

    make gold-review   # revisar (retoma de onde parou)
    make gold-report   # % de sugestões corrigidas, concordância às cegas, publica no README

Dois modos, misturados em ordem aleatória:
  - revisão: mostra a sugestão do LLM; Enter aceita, ou você digita a correção.
  - cego (amostra fixa de 30, 6 por nota): você rotula SEM ver a sugestão; ela aparece depois
    e você decide. É o que mede a concordância real LLM × humano e a ancoragem (ver D56).

Correção: número da categoria (1-6) e/ou letra da urgência (b, m, a). Ex.: "3", "a", "3a".
Observação opcional depois de "--": "3a -- atrasou e quebrou".
Comandos: v volta uma avaliação · ? ajuda · q sai (o progresso é salvo a cada resposta).
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from reviews_triage.gold import GOLD_PATH, TO_LABEL_PATH, read_labels
from reviews_triage.labels import CATEGORIES, URGENCIES
from reviews_triage.prelabel import PRELABELS_META_PATH, PRELABELS_PATH
from reviews_triage.stats import kappa, wilson_interval
from reviews_triage.terminal import Back, Quit, ask_command, write_csv_atomic

BLIND_SIZE = 30
KEEP_OR_SWAP = "  Enter mantém o seu · t troca pela sugestão · ou corrija > "
BLIND_SEED = 11
REPORT_PATH = Path("reports/gold_labeling.md")
STATS_PATH = Path("reports/gold_labeling.json")

CATEGORY_KEYS = {str(i): c for i, c in enumerate(CATEGORIES, start=1)}
URGENCY_KEYS = {"b": "baixa", "m": "media", "a": "alta"}
GOLD_COLUMNS = [
    "review_id",
    "review_score",
    "text",
    "categoria",
    "urgencia",
    "observacao",
    "modo",
    "categoria_llm",
    "urgencia_llm",
    "justificativa_llm",
    "categoria_cega",
    "urgencia_cega",
]
LEGEND = "  ".join(f"{k} {v}" for k, v in CATEGORY_KEYS.items()) + "  |  b baixa  m media  a alta"
HELP = f"""
Categorias e urgências:
  {LEGEND}
Enter  aceita a sugestão (modo revisão) ou mantém o seu rótulo (depois de ver a sugestão)
3a     categoria 3 + urgência alta; "3" muda só a categoria, "a" só a urgência
t      troca pelo rótulo sugerido (só depois de ver a sugestão no modo cego)
-- ... observação livre, ex.: "3a -- atrasou e quebrou"
v  volta uma avaliação   ?  esta ajuda   q  sai (o progresso fica salvo)
"""


def parse_answer(text: str, categoria: str | None, urgencia: str | None):
    """Aplica a correção digitada a um rótulo. Devolve (categoria, urgencia, observacao)."""
    keys, _, note = text.partition("--")
    for ch in keys.strip().lower().replace(" ", ""):
        if ch in CATEGORY_KEYS:
            categoria = CATEGORY_KEYS[ch]
        elif ch in URGENCY_KEYS:
            urgencia = URGENCY_KEYS[ch]
        else:
            raise ValueError(f"tecla desconhecida: {ch!r} (digite ? para ajuda)")
    if categoria is None or urgencia is None:
        raise ValueError("informe categoria e urgência, ex.: 3a")
    return categoria, urgencia, note.strip()


def blind_ids(to_label: pd.DataFrame, size: int = BLIND_SIZE, seed: int = BLIND_SEED) -> set[str]:
    """Amostra cega fixa, estratificada por nota (size/5 de cada)."""
    if size <= 0:
        return set()
    per_score = max(1, size // 5)
    sample = to_label.groupby("review_score").sample(n=per_score, random_state=seed)
    return set(sample["review_id"])


def build_rows(
    to_label: pd.DataFrame,
    prelabels: pd.DataFrame | None,
    progress: pd.DataFrame | None,
    blind: set[str],
) -> list[dict]:
    """Uma linha por avaliação, na ordem do to_label.csv, já com o progresso salvo."""
    suggestions = {} if prelabels is None else prelabels.set_index("review_id").to_dict("index")
    saved = {} if progress is None else progress.set_index("review_id").to_dict("index")
    rows = []
    for r in to_label.itertuples():
        s = suggestions.get(r.review_id, {})
        has_suggestion = bool(s.get("categoria_llm"))
        row = {c: "" for c in GOLD_COLUMNS}
        row.update(
            review_id=r.review_id,
            review_score=str(r.review_score),
            text=r.text,
            categoria_llm=s.get("categoria_llm", ""),
            urgencia_llm=s.get("urgencia_llm", ""),
            justificativa_llm=s.get("justificativa_llm", ""),
            modo=("cego" if r.review_id in blind else "revisao")
            if has_suggestion
            else "sem_sugestao",
        )
        previous = saved.get(r.review_id)
        if previous and previous.get("categoria"):
            row.update({c: previous.get(c, "") for c in GOLD_COLUMNS if c in previous})
        rows.append(row)
    return rows


def save_rows(rows: list[dict], path: Path = GOLD_PATH) -> None:
    write_csv_atomic(rows, GOLD_COLUMNS, path)


def _label(categoria: str, urgencia: str) -> str:
    number = CATEGORIES.index(categoria) + 1
    return f"{number} {categoria} · urgência {urgencia}"


def review_one(row: dict, position: str, ask, say) -> None:
    """Preenche categoria/urgencia/observacao (e os campos cegos, no modo cego)."""
    mode = {"revisao": "revisão", "cego": "cego", "sem_sugestao": "sem sugestão"}[row["modo"]]
    say(f"\n[{position}]  nota {row['review_score']}  (modo {mode})")
    say(f'"{row["text"]}"')
    suggestion = (row["categoria_llm"], row["urgencia_llm"])

    if row["modo"] == "revisao":
        say(f"Sugestão: {_label(*suggestion)}")
        say(f"Motivo:   {row['justificativa_llm']}")
        start, prompt = suggestion, "Enter aceita · correção (ex.: 3a) · ? ajuda > "
    else:
        start, prompt = (None, None), "Seu rótulo (ex.: 3a) · ? ajuda > "

    while True:
        answer = ask_command(ask, prompt)
        if answer.strip() == "?":
            say(HELP)
            continue
        try:
            categoria, urgencia, note = parse_answer(answer, *start)
            break
        except ValueError as exc:
            say(f"  {exc}")

    if row["modo"] == "cego":
        row["categoria_cega"], row["urgencia_cega"] = categoria, urgencia
        if (categoria, urgencia) == suggestion:
            say(f"  ✓ igual à sugestão do LLM ({_label(*suggestion)})")
        else:
            say(f"  O LLM sugeriu: {_label(*suggestion)}\n  Motivo: {row['justificativa_llm']}")
            while True:
                answer = ask_command(ask, KEEP_OR_SWAP)
                if answer.strip().lower() == "t":
                    categoria, urgencia = suggestion
                    break
                try:
                    categoria, urgencia, extra = parse_answer(answer, categoria, urgencia)
                    note = extra or note
                    break
                except ValueError as exc:
                    say(f"  {exc}")

    row.update(categoria=categoria, urgencia=urgencia, observacao=note)


def review_loop(rows: list[dict], save: Callable[[list[dict]], None], ask=input, say=print) -> int:
    """Percorre as avaliações pendentes. Devolve quantas ficaram rotuladas."""
    say(f"Teclas: {LEGEND}   (? ajuda, v volta, q sai)")
    i = next((k for k, r in enumerate(rows) if not r["categoria"]), len(rows))
    while i < len(rows):
        if rows[i]["categoria"]:  # já rotulada (ao retomar)
            i += 1
            continue
        try:
            review_one(rows[i], f"{i + 1}/{len(rows)}", ask, say)
        except Back:
            i = max(0, i - 1)
            for c in ("categoria", "urgencia", "observacao", "categoria_cega", "urgencia_cega"):
                rows[i][c] = ""
            save(rows)
            continue
        except Quit:
            break
        save(rows)
        i += 1
    done = sum(bool(r["categoria"]) for r in rows)
    say(f"\n{done}/{len(rows)} rotuladas. Progresso salvo.")
    return done


# --- Relatório da rotulagem ------------------------------------------------------------


def _share(mask: pd.Series) -> dict:
    n, k = len(mask), int(mask.sum())
    return {"n": n, "pct": k / n if n else None, "ic95": wilson_interval(k, n)}


def labeling_stats(gold: pd.DataFrame) -> dict:
    done = gold[gold["categoria"] != ""]
    review = done[(done["modo"] == "revisao") & (done["categoria_llm"] != "")]
    blind = done[(done["modo"] == "cego") & (done["categoria_llm"] != "")]
    order = {u: i for i, u in enumerate(URGENCIES)}

    corrected = review["categoria"] != review["categoria_llm"]
    corrections = Counter(
        f"{a} → {b}"
        for a, b in zip(done["categoria_llm"], done["categoria"], strict=True)
        if a and a != b
    )
    return {
        "n": len(gold),
        "rotuladas": len(done),
        "revisao": {
            "categoria_corrigida": _share(corrected),
            "urgencia_corrigida": _share(review["urgencia"] != review["urgencia_llm"]),
            "alguma_correcao": _share(corrected | (review["urgencia"] != review["urgencia_llm"])),
        },
        "cego": {
            "categoria_concorda": _share(blind["categoria_cega"] == blind["categoria_llm"]),
            "categoria_kappa": kappa(
                blind["categoria_cega"], blind["categoria_llm"], labels=list(CATEGORIES)
            )
            if len(blind)
            else None,
            "urgencia_concorda": _share(blind["urgencia_cega"] == blind["urgencia_llm"]),
            "urgencia_kappa_quadratico": kappa(
                blind["urgencia_cega"].map(order),
                blind["urgencia_llm"].map(order),
                labels=[0, 1, 2],
                weights="quadratic",
            )
            if len(blind)
            else None,
            "trocou_depois_de_ver": int(
                (
                    (blind["categoria"] != blind["categoria_cega"])
                    | (blind["urgencia"] != blind["urgencia_cega"])
                ).sum()
            ),
        },
        "sem_sugestao": int((done["modo"] == "sem_sugestao").sum()),
        "correcoes_mais_comuns": dict(corrections.most_common(8)),
    }


def _pct(share: dict) -> str:
    if share["pct"] is None:
        return "—"
    ci = share["ic95"]
    return f"{share['pct']:.0%} [{ci[0]:.0%}, {ci[1]:.0%}] (n = {share['n']})"


def render_labeling_report(stats: dict, meta: dict) -> str:
    rev, blind = stats["revisao"], stats["cego"]
    k_cat = "—" if blind["categoria_kappa"] is None else f"{blind['categoria_kappa']:.2f}"
    k_urg = (
        "—"
        if blind["urgencia_kappa_quadratico"] is None
        else (f"{blind['urgencia_kappa_quadratico']:.2f}")
    )
    corrections = (
        "\n".join(f"| {pair} | {n} |" for pair, n in stats["correcoes_mais_comuns"].items())
        or "| — | 0 |"
    )
    gap = ""
    if rev["categoria_corrigida"]["pct"] is not None and blind["categoria_concorda"]["pct"]:
        disagree = 1 - blind["categoria_concorda"]["pct"]
        gap = (
            f"\nÀs cegas você discordou do LLM em **{disagree:.0%}** das categorias; vendo a "
            f"sugestão, corrigiu **{rev['categoria_corrigida']['pct']:.0%}**. Se a segunda for "
            "bem menor que a primeira, as sugestões estão ancorando a revisão.\n"
        )
    return f"""# Rotulagem do gold set

Rótulos **pré-gerados por LLM e revisados por humano**: `{meta.get("model", "?")}`
(effort `{meta.get("effort", "?")}`), prompt `prompts/prelabel.md` + critérios do
`LABELING_GUIDE.md`, versão `{meta.get("prompt_version", "?")}`; custo
US$ {meta.get("cost_usd", 0):.2f}. {stats["rotuladas"]}/{stats["n"]} avaliações rotuladas.

## Modo revisão (você via a sugestão)

| | % (IC 95%) |
|---|---|
| Categoria corrigida | {_pct(rev["categoria_corrigida"])} |
| Urgência corrigida | {_pct(rev["urgencia_corrigida"])} |
| Alguma correção | {_pct(rev["alguma_correcao"])} |

## Modo cego (você rotulou sem ver a sugestão)

| | Concordância LLM × você (IC 95%) | Kappa |
|---|---|---|
| Categoria | {_pct(blind["categoria_concorda"])} | {k_cat} |
| Urgência | {_pct(blind["urgencia_concorda"])} | {k_urg} (quadrático) |

Depois de ver a sugestão, você trocou o próprio rótulo em {blind["trocou_depois_de_ver"]} casos.
{gap}
## Correções mais comuns (LLM → você)

| Sugestão → rótulo final | n |
|---|---|
{corrections}
"""


def render_readme_block(stats: dict, meta: dict) -> str:
    rev, blind = stats["revisao"], stats["cego"]
    k_cat = "—" if blind["categoria_kappa"] is None else f"{blind['categoria_kappa']:.2f}"
    return (
        f"Rótulos pré-gerados por `{meta.get('model', '?')}` (prompt `prelabel.md`, versão "
        f"`{meta.get('prompt_version', '?')}`) e revisados por humano "
        f"({stats['rotuladas']}/{stats['n']}).\n\n"
        "| Medida | Valor |\n|---|---|\n"
        "| Sugestões de categoria corrigidas (modo revisão) | "
        f"{_pct(rev['categoria_corrigida'])} |\n"
        f"| Sugestões de urgência corrigidas (modo revisão) | {_pct(rev['urgencia_corrigida'])} |\n"
        f"| Concordância às cegas LLM × humano, categoria | {_pct(blind['categoria_concorda'])}, "
        f"kappa {k_cat} |\n"
        f"| Concordância às cegas LLM × humano, urgência | {_pct(blind['urgencia_concorda'])} |\n\n"
        "Detalhes em [`reports/gold_labeling.md`](reports/gold_labeling.md)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Revisão dos rótulos do gold set.")
    sub = parser.add_subparsers(dest="command", required=True)
    review = sub.add_parser("review")
    review.add_argument("--blind", type=int, default=BLIND_SIZE, help="tamanho da amostra cega")
    sub.add_parser("report")
    args = parser.parse_args()

    if args.command == "review":
        if not TO_LABEL_PATH.exists():
            sys.exit(f"{TO_LABEL_PATH} não existe. Rode `make data` e `make gold` primeiro.")
        if not PRELABELS_PATH.exists():
            sys.exit(f"{PRELABELS_PATH} não existe. Rode `make prelabel YES=1` primeiro.")
        to_label = read_labels(TO_LABEL_PATH)
        progress = read_labels(GOLD_PATH) if GOLD_PATH.exists() else None
        rows = build_rows(
            to_label, read_labels(PRELABELS_PATH), progress, blind_ids(to_label, args.blind)
        )
        review_loop(rows, save_rows)
        print("Para validar: make gold-check  ·  para o relatório: make gold-report")
        return

    from reviews_triage.readme import update_readme  # só o relatório precisa

    gold = read_labels(GOLD_PATH)
    meta = json.loads(PRELABELS_META_PATH.read_text(encoding="utf-8"))
    stats = labeling_stats(gold)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_labeling_report(stats, meta), encoding="utf-8")
    STATS_PATH.write_text(
        json.dumps({"stats": stats, "meta": meta}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    update_readme({"rotulagem": render_readme_block(stats, meta)})
    print(f"Relatório em {REPORT_PATH}; README atualizado.")


if __name__ == "__main__":
    main()
