"""Validação humana do juiz: você dá notas às mesmas respostas, às cegas, e medimos a concordância.

Fluxo:
    make pilot MODEL=... YES=1  -> gera data/judge_validation/<rodada>/to_score.csv
    (você dá notas seguindo prompts/judge.md e salva como human.csv na mesma pasta)
    make judge-agreement        -> reports/judge_agreement_<rodada>.md

O arquivo para dar notas não traz nada do juiz (nem notas, nem justificativa): quem vê a
nota do juiz antes tende a concordar com ela, e a concordância medida sairia inflada.
"""

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from reviews_triage.evaluation import (
    REPORTS_DIR,
    SEED,
    VALIDATION_DIR,
    EvalRecord,
    find_runs,
    load_records,
    load_run,
)
from reviews_triage.gold import read_labels
from reviews_triage.judge import CRITERIA

# Critério de aprovação PRÉ-REGISTRADO (DECISIONS.md, D52): definido antes de existir
# qualquer nota humana, pelo mesmo motivo que a rubrica foi congelada antes de rodar.
MIN_WITHIN_ONE = 0.80  # pelo menos 80% das notas do juiz a no máximo 1 ponto da humana
MAX_ABS_BIAS = 0.5  # e o juiz não pode ser, em média, mais generoso/severo que 0,5 ponto
MIN_KAPPA = 0.40  # e precisa distinguir respostas boas de ruins além do acaso (ver abaixo)
SMALL_N = 30  # abaixo disso o resultado é indicativo: o intervalo de confiança é largo

# Por que o kappa: um juiz "preguiçoso" que dá 5 para tudo passa em "até 1 ponto" e em
# "viés" quando a maioria das respostas é boa, que é o caso típico. O kappa dele é ~0.

VERDICT_LABEL = {
    "aprovado": "✅ aprovado",
    "reprovado": "❌ reprovado",
    "inconclusivo": "⚪ inconclusivo",
}


def verdict(within_one: float, bias: float, kappa: float | None) -> str:
    if within_one < MIN_WITHIN_ONE or abs(bias) > MAX_ABS_BIAS:
        return "reprovado"
    if kappa is None:  # as notas não variaram: não dá para saber se o juiz discrimina
        return "inconclusivo"
    return "aprovado" if kappa >= MIN_KAPPA else "reprovado"


SCORES = ("1", "2", "3", "4", "5")


def export_for_scoring(run_dir: Path, out_dir: Path = VALIDATION_DIR, seed: int = SEED) -> Path:
    """Gera o CSV para notas humanas com as respostas julgadas da rodada, sem nada do juiz."""
    records = [
        r
        for r in load_records(run_dir)
        if r.judge is not None and r.result is not None and r.result.resposta_sugerida
    ]
    if not records:
        raise ValueError(f"Nenhuma resposta julgada em {run_dir}.")
    target = out_dir / run_dir.name / "to_score.csv"
    if target.exists():
        raise FileExistsError(f"{target} já existe; não vou sobrescrever.")

    rows = [
        {
            "review_id": r.review_id,
            "nota_cliente": r.review_score,
            "avaliacao": r.text,
            "resposta": r.result.resposta_sugerida,
            **{criterion: "" for criterion in CRITERIA},
            "observacao": "",
        }
        for r in records
    ]
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "item", range(1, len(df) + 1))
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, encoding="utf-8-sig")  # BOM: acentos certos no Excel
    return target


def load_human_scores(path: Path) -> pd.DataFrame:
    """Lê as notas humanas e falha apontando a linha de cada valor inválido."""
    df = read_labels(path)
    missing = {"review_id", *CRITERIA} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: colunas ausentes {sorted(missing)}")
    errors = []
    for criterion in CRITERIA:
        for line, value in df[criterion].str.strip().items():
            if value not in SCORES:
                errors.append(f"linha {line + 2}: {criterion}={value!r} (use um inteiro de 1 a 5)")
    if errors:
        raise ValueError(f"{path} inválido:\n" + "\n".join(errors))
    for criterion in CRITERIA:
        df[criterion] = df[criterion].str.strip().astype(int)
    return df


def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    """IC de 95% para uma proporção; funciona bem mesmo com n pequeno ou p perto de 0/1."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _weighted_kappa(human: np.ndarray, judge: np.ndarray) -> float | None:
    """Kappa de Cohen com peso quadrático: desconta o acaso e pune mais as discordâncias
    grandes. Indefinido (None) quando as notas quase não variam."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa = cohen_kappa_score(human, judge, labels=[1, 2, 3, 4, 5], weights="quadratic")
    return None if np.isnan(kappa) else float(kappa)


def agreement(records: list[EvalRecord], human: pd.DataFrame) -> dict:
    by_id = {r.review_id: r for r in records if r.judge is not None}
    unknown = set(human["review_id"]) - set(by_id)
    if unknown:
        raise ValueError(f"review_id sem nota do juiz nesta rodada: {sorted(unknown)[:5]}")

    out: dict = {"n": len(human), "criterios": {}, "discordancias": []}
    for criterion in CRITERIA:
        h = human[criterion].to_numpy()
        j = np.array([getattr(by_id[rid].judge, criterion) for rid in human["review_id"]])
        diff = j - h
        n_exact, n_within = int((diff == 0).sum()), int((np.abs(diff) <= 1).sum())
        bias = float(j.mean() - h.mean())
        kappa = _weighted_kappa(h, j)
        out["criterios"][criterion] = {
            "exata": n_exact / len(h),
            "exata_ic95": wilson_interval(n_exact, len(h)),
            "ate_1_ponto": n_within / len(h),
            "ate_1_ponto_ic95": wilson_interval(n_within, len(h)),
            "erro_medio_absoluto": float(np.abs(diff).mean()),
            "vies_juiz_menos_humano": bias,
            "kappa_quadratico": kappa,
            "media_humano": float(h.mean()),
            "media_juiz": float(j.mean()),
            "veredito": verdict(n_within / len(h), bias, kappa),
        }

    for row in human.itertuples():
        judge = by_id[row.review_id].judge
        gaps = {c: getattr(judge, c) - getattr(row, c) for c in CRITERIA}
        if any(abs(g) >= 2 for g in gaps.values()):
            out["discordancias"].append(
                {
                    "review_id": row.review_id,
                    "humano": {c: int(getattr(row, c)) for c in CRITERIA},
                    "juiz": {c: getattr(judge, c) for c in CRITERIA},
                    "resposta": by_id[row.review_id].result.resposta_sugerida,
                    "justificativa_juiz": judge.justificativa,
                }
            )
    return out


def render_agreement_report(run_id: str, result: dict) -> str:
    def pct(value: float, ci: list[float] | None) -> str:
        return f"{value:.0%}" + (f" [{ci[0]:.0%}, {ci[1]:.0%}]" if ci else "")

    rows = []
    for criterion, m in result["criterios"].items():
        kappa = "—" if m["kappa_quadratico"] is None else f"{m['kappa_quadratico']:.2f}"
        rows.append(
            [
                criterion,
                pct(m["ate_1_ponto"], m["ate_1_ponto_ic95"]),
                pct(m["exata"], m["exata_ic95"]),
                f"{m['erro_medio_absoluto']:.2f}",
                f"{m['vies_juiz_menos_humano']:+.2f}",
                kappa,
                f"{m['media_humano']:.2f} / {m['media_juiz']:.2f}",
                VERDICT_LABEL[m["veredito"]],
            ]
        )
    header = [
        "Critério",
        "Até 1 ponto (IC 95%)",
        "Exata (IC 95%)",
        "Erro médio",
        "Viés (juiz − você)",
        "Kappa quadrático",
        "Média você / juiz",
        "Veredito",
    ]
    table = "\n".join(
        ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    )

    disagreements = (
        "\n".join(
            f"- `{d['review_id'][:8]}`: você {d['humano']}, juiz {d['juiz']}.\n"
            f'  Resposta: "{d["resposta"]}"\n  Juiz: "{d["justificativa_juiz"]}"'
            for d in result["discordancias"]
        )
        or "Nenhuma discordância de 2 pontos ou mais."
    )

    small = (
        f"\n> ⚠️ Com n = {result['n']} (< {SMALL_N}) o resultado é **indicativo**: veja como os "
        "intervalos de confiança são largos.\n"
        if result["n"] < SMALL_N
        else ""
    )
    verdicts = {m["veredito"] for m in result["criterios"].values()}
    if verdicts == {"aprovado"}:
        conclusion = "O juiz passou no critério pré-registrado nos três critérios."
    elif "reprovado" in verdicts:
        conclusion = (
            "O juiz **não** passou no critério pré-registrado em pelo menos um critério: as notas "
            "dele nesse critério não devem ser usadas como métrica sem ajuste."
        )
    else:
        conclusion = (
            "Nenhum critério reprovado, mas pelo menos um ficou **inconclusivo**: as respostas não "
            "variaram o bastante para testar se o juiz distingue boas de ruins. Inclua mais "
            "respostas (ou respostas piores) na validação."
        )
    return f"""# Concordância juiz × humano: `{run_id}`

{result["n"]} respostas avaliadas às cegas por você e pelo juiz, com a mesma rubrica.

**Critério pré-registrado** (definido antes das notas): por critério, pelo menos
{MIN_WITHIN_ONE:.0%} das notas do juiz a no máximo 1 ponto da sua, viés médio de no máximo
±{MAX_ABS_BIAS} ponto **e** kappa quadrático de pelo menos {MIN_KAPPA:.2f} (o juiz distingue
respostas boas de ruins além do que o acaso explicaria).
{small}
{table}

{conclusion}

Viés positivo significa que o juiz é mais generoso que você. O kappa desconta a concordância
que aconteceria por acaso; ele fica indefinido (—) quando as notas quase não variam.

## Discordâncias de 2 pontos ou mais

{disagreements}
"""


def _run_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    pilots = find_runs(kind="pilot")
    if not pilots:
        sys.exit("Nenhum piloto encontrado; rode `make pilot` primeiro ou informe RUN=...")
    return pilots[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validação humana do juiz.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "agreement"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--run", help="pasta da rodada em reports/runs (padrão: último piloto)")
    args = parser.parse_args()
    run_dir = _run_dir(args.run)

    if args.command == "export":
        print(f"Arquivo para suas notas: {export_for_scoring(run_dir)}")
        return

    human_path = VALIDATION_DIR / run_dir.name / "human.csv"
    if not human_path.exists():
        sys.exit(f"{human_path} não existe. Dê as notas em to_score.csv e salve como human.csv.")
    result = agreement(load_records(run_dir), load_human_scores(human_path))
    meta, _ = load_run(run_dir)
    out = REPORTS_DIR / f"judge_agreement_{meta.run_id}.md"
    out.write_text(render_agreement_report(meta.run_id, result), encoding="utf-8")
    for criterion, m in result["criterios"].items():
        kappa = m["kappa_quadratico"]
        print(
            f"{criterion:<10} até 1 ponto {m['ate_1_ponto']:.0%}  "
            f"viés {m['vies_juiz_menos_humano']:+.2f}  "
            f"kappa {'—' if kappa is None else f'{kappa:.2f}'}  {m['veredito'].upper()}"
        )
    print(f"Relatório em {out}")


if __name__ == "__main__":
    main()
