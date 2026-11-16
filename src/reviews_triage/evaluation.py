"""Avaliação do agente no gold set (Fase 5).

Uso:
    make eval-retrieval                          # só a busca, sem LLM (custo zero)
    make eval MODEL=claude-sonnet-5-5 LIMIT=10   # mostra a estimativa de custo e PARA
    make eval MODEL=claude-sonnet-5-5 LIMIT=10 YES=1   # piloto: mede o custo real
    make eval MODEL=claude-sonnet-5-5 YES=1      # rodada completa
    make eval-compare                            # compara as rodadas salvas (sem LLM)

Cada rodada grava em reports/runs/<run_id>/:
    meta.json      configuração (modelo, juiz, hashes dos prompts, gold set, commit)
    results.jsonl  resultado bruto por avaliação: é a fonte da verdade
    summary.json   métricas calculadas a partir do results.jsonl
e o relatório legível em reports/eval_<run_id>.md.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix, f1_score

from reviews_triage import baseline
from reviews_triage.agent import MAX_REDRAFTS, TOP_K, TriageAgent, build_query
from reviews_triage.gold import GOLD_PATH, load_gold
from reviews_triage.judge import JUDGE_EFFORT, JUDGE_MODEL, Judge, JudgeScores
from reviews_triage.labels import CATEGORIES
from reviews_triage.llm import AnthropicLLM, LLMRefusal
from reviews_triage.policy import (
    DEFAULT_EMBEDDING_MODEL,
    PolicyIndex,
    build_index,
    load_chunks,
)
from reviews_triage.schemas import LLMCall, TriageResult

REPORTS_DIR = Path("reports")
RUNS_DIR = REPORTS_DIR / "runs"
N_BOOTSTRAP = 1000
SEED = 42
NO_ANSWER = "sem_resposta"  # predição quando o agente falhou ou o modelo recusou

# Preço em US$ por milhão de tokens (entrada, saída), tabela pública da Anthropic.
# Modelo sem preço aqui gera erro, para nunca reportar custo zero por engano.
PRICES_AS_OF = "2026-11"
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),  # vale para prompts de até 100 mil tokens
    "claude-fable-5-2": (8.00, 40.00),
}

# Estimativa grosseira de tokens por avaliação, só para o aviso antes de gastar.
# O número real sai de uma rodada piloto (LIMIT=10).
EST_AGENT_TOKENS = (5_000, 1_000)
EST_JUDGE_TOKENS = (3_500, 800)

# Documento de política que deveria aparecer na busca, por categoria do gold set.
EXPECTED_SOURCES: dict[str, set[str]] = {
    "atraso_entrega": {"atraso_entrega.md"},
    "produto_errado_ou_faltando": {"produto_errado_ou_faltando.md"},
    "defeito_ou_qualidade": {"defeito_ou_qualidade.md"},
    "atendimento_vendedor": {"atendimento_vendedor.md"},
    "elogio": {"elogio_e_outros.md"},
    "outro": {"elogio_e_outros.md"},
}


class RunMeta(BaseModel):
    run_id: str
    created_at: str
    model: str
    effort: str
    judge_model: str
    judge_effort: str
    judge_prompt_version: str
    prompt_versions: dict[str, str]
    gold_path: str
    gold_sha: str
    n_reviews: int
    git_commit: str
    embedding_model: str
    top_k: int
    max_redrafts: int


class EvalRecord(BaseModel):
    review_id: str
    review_score: int
    text: str
    gold_categoria: str
    gold_urgencia: str
    result: TriageResult | None = None
    latency_s: float | None = None
    error: str | None = None
    judge: JudgeScores | None = None
    judge_call: LLMCall | None = None
    judge_error: str | None = None


# --- Custo -----------------------------------------------------------------------------


def call_cost(call: LLMCall) -> float:
    if call.model not in PRICES_PER_MTOK:
        raise KeyError(f"Sem preço para {call.model!r}; atualize PRICES_PER_MTOK.")
    price_in, price_out = PRICES_PER_MTOK[call.model]
    return (call.input_tokens * price_in + call.output_tokens * price_out) / 1_000_000


def estimate_cost(model: str, n_reviews: int) -> tuple[float, float]:
    """(agente, juiz) em US$, pela estimativa grosseira de tokens."""

    def cost(m: str, tokens: tuple[int, int]) -> float:
        return call_cost(
            LLMCall(
                step="est", model=m, input_tokens=tokens[0], output_tokens=tokens[1], latency_s=0
            )
        )

    return n_reviews * cost(model, EST_AGENT_TOKENS), n_reviews * cost(
        JUDGE_MODEL, EST_JUDGE_TOKENS
    )


# --- Execução --------------------------------------------------------------------------


def evaluate_one(row, agent: TriageAgent, judge: Judge) -> EvalRecord:
    record = EvalRecord(
        review_id=row.review_id,
        review_score=int(row.review_score),
        text=row.text,
        gold_categoria=row.categoria,
        gold_urgencia=row.urgencia,
    )
    started = time.perf_counter()
    try:
        record.result = agent.run(row.text, int(row.review_score))
    except Exception as exc:  # erro de API depois dos retries do SDK: registra e segue
        record.error = f"{type(exc).__name__}: {exc}"
        return record
    record.latency_s = time.perf_counter() - started

    if record.result.resposta_sugerida:
        try:
            record.judge, record.judge_call = judge.score(
                row.text, int(row.review_score), record.result.resposta_sugerida
            )
        except LLMRefusal as exc:
            record.judge_call, record.judge_error = exc.call, str(exc)
        except Exception as exc:
            record.judge_error = f"{type(exc).__name__}: {exc}"
    return record


def run_eval(
    gold: pd.DataFrame,
    agent: TriageAgent,
    judge: Judge,
    workers: int = 4,
    on_record: Callable[[EvalRecord], None] | None = None,
) -> list[EvalRecord]:
    """Roda agente + juiz em cada avaliação, em paralelo (as chamadas são I/O)."""
    order = {review_id: i for i, review_id in enumerate(gold["review_id"])}
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate_one, row, agent, judge) for row in gold.itertuples()]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            if on_record:
                on_record(record)
    return sorted(records, key=lambda r: order[r.review_id])


# --- Métricas --------------------------------------------------------------------------


def _f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def bootstrap_f1(
    y_true: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    labels: list[str],
    n: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, list[float]]:
    """IC de 95% para o F1 macro de A, de B e da diferença A − B.

    Pareado: cada reamostragem usa as mesmas avaliações para os dois modelos, então a
    diferença não mistura a variação entre modelos com a variação entre amostras.
    """
    rng = np.random.default_rng(seed)
    a, b = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        a.append(_f1(y_true[idx], y_a[idx], labels))
        b.append(_f1(y_true[idx], y_b[idx], labels))
    a, b = np.array(a), np.array(b)

    def ci(values: np.ndarray) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    return {"a": ci(a), "b": ci(b), "diff": ci(a - b)}


def _approved(reasons: str) -> bool:
    """A revisão aprovou se não houve reprovação nem recusa do modelo."""
    return "reprovou" not in reasons and "recusou" not in reasons


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def retrieval_hit_rates(gold: pd.DataFrame, index: PolicyIndex, k: int = TOP_K) -> dict:
    """Hit rate da busca sem chamar LLM, com e sem a dica de categoria (hipótese D31).

    A variante com dica usa o rótulo do gold set, ou seja, é o teto: o que a busca
    alcançaria com classificação perfeita.
    """
    variants = {
        "só o texto": lambda r: r.text,
        "dica da categoria (gold) + texto": lambda r: build_query(r.categoria, r.text),
    }
    out = {}
    for name, make_query in variants.items():
        hit1 = hitk = 0
        for row in gold.itertuples():
            sources = [c.source for c in index.search(make_query(row), k)]
            expected = EXPECTED_SOURCES[row.categoria]
            hit1 += sources[0] in expected
            hitk += any(s in expected for s in sources)
        out[name] = {"hit@1": hit1 / len(gold), f"hit@{k}": hitk / len(gold)}
    return out


def compute_metrics(records: list[EvalRecord], baseline_pred: pd.Series, baseline_cv: dict) -> dict:
    """Todas as métricas da rodada, em tipos simples (vira summary.json)."""
    ids = [r.review_id for r in records]
    y_true = np.array([r.gold_categoria for r in records])
    y_agent = np.array(
        [r.result.categoria if r.result and r.result.categoria else NO_ANSWER for r in records]
    )
    y_base = baseline_pred.loc[ids].to_numpy()
    labels = [c for c in CATEGORIES if c in set(y_true)]
    boot = bootstrap_f1(y_true, y_agent, y_base, labels)

    per_class = {}
    agent_f1s = f1_score(y_true, y_agent, labels=labels, average=None, zero_division=0)
    base_f1s = f1_score(y_true, y_base, labels=labels, average=None, zero_division=0)
    for label, fa, fb in zip(labels, agent_f1s, base_f1s, strict=True):
        per_class[label] = {
            "agent_f1": float(fa),
            "baseline_f1": float(fb),
            "support": int((y_true == label).sum()),
        }

    # Urgência: acerto geral e, principalmente, quanto do "alta" real o agente pega.
    u_true = np.array([r.gold_urgencia for r in records])
    u_pred = np.array(
        [r.result.urgencia if r.result and r.result.urgencia else NO_ANSWER for r in records]
    )
    true_high, pred_high = u_true == "alta", u_pred == "alta"

    done = [r for r in records if r.result is not None]
    reasons = [" | ".join(r.result.motivos_escalacao) for r in done]

    hits = [
        any(f.split("#")[0] in EXPECTED_SOURCES[r.gold_categoria] for f in r.result.fontes)
        for r in done
        if r.result.fontes
    ]

    judged = [r.judge for r in records if r.judge is not None]
    judge = {}
    for criterion in ("aderencia", "tom", "utilidade"):
        scores = [getattr(j, criterion) for j in judged]
        judge[criterion] = {
            "mean": float(np.mean(scores)) if scores else None,
            "dist": {str(s): scores.count(s) for s in range(1, 6)},
        }
    judge["aderencia_baixa_pct"] = _rate(sum(j.aderencia <= 2 for j in judged), len(judged))

    agent_costs = [sum(call_cost(c) for c in r.result.llm_calls) for r in done]
    judge_cost = sum(call_cost(r.judge_call) for r in records if r.judge_call)
    latencies = [r.latency_s for r in done if r.latency_s is not None]
    by_step: dict[str, list[LLMCall]] = {}
    for r in done:
        for c in r.result.llm_calls:
            by_step.setdefault(c.step, []).append(c)

    return {
        "n": len(records),
        "n_errors": sum(r.error is not None for r in records),
        "n_judge_errors": sum(r.judge_error is not None for r in records),
        "categoria": {
            "agent_f1_macro": _f1(y_true, y_agent, labels),
            "agent_f1_ci95": boot["a"],
            "agent_accuracy": float((y_true == y_agent).mean()),
            "baseline_f1_macro_oof": _f1(y_true, y_base, labels),
            "baseline_f1_ci95": boot["b"],
            "baseline_f1_cv_mean": baseline_cv["mean"],
            "baseline_f1_cv_std": baseline_cv["std"],
            "diff_f1_ci95": boot["diff"],
            "labels": labels,
            "per_class": per_class,
            "confusion": confusion_matrix(y_true, y_agent, labels=[*labels, NO_ANSWER]).tolist(),
        },
        "urgencia": {
            "accuracy": float((u_true == u_pred).mean()),
            "alta_recall": _rate(int((true_high & pred_high).sum()), int(true_high.sum())),
            "alta_precision": _rate(int((true_high & pred_high).sum()), int(pred_high.sum())),
            "n_alta_gold": int(true_high.sum()),
        },
        "busca": {f"hit@{TOP_K}_agente": _rate(sum(hits), len(hits))},
        "revisao": {
            "aprovada_1a": _rate(
                sum(
                    r.result.tentativas_redacao == 1 and _approved(m)
                    for r, m in zip(done, reasons, strict=True)
                ),
                len(done),
            ),
            "aprovada_2a": _rate(
                sum(
                    r.result.tentativas_redacao == 2 and _approved(m)
                    for r, m in zip(done, reasons, strict=True)
                ),
                len(done),
            ),
            "reprovada": _rate(sum("reprovou" in m for m in reasons), len(done)),
            "escalada": _rate(sum(r.result.escalate_to_human for r in done), len(done)),
            "escalada_urgencia": _rate(sum("urgência alta" in m for m in reasons), len(done)),
            "recusas": sum("recusou" in m for m in reasons),
        },
        "juiz": judge,
        "custo": {
            "agente_por_avaliacao": float(np.mean(agent_costs)) if agent_costs else None,
            "agente_por_1000": float(np.mean(agent_costs)) * 1000 if agent_costs else None,
            "juiz_total": judge_cost,
            "agente_total": float(sum(agent_costs)),
            "cache_tokens": sum(
                c.cache_read_tokens + c.cache_write_tokens for cs in by_step.values() for c in cs
            ),
            "tokens_por_passo": {
                step: {
                    "chamadas_por_avaliacao": len(calls) / len(done),
                    "entrada_media": float(np.mean([c.input_tokens for c in calls])),
                    "saida_media": float(np.mean([c.output_tokens for c in calls])),
                }
                for step, calls in by_step.items()
            },
        },
        "latencia_s": {
            "p50": float(np.percentile(latencies, 50)) if latencies else None,
            "p95": float(np.percentile(latencies, 95)) if latencies else None,
            "media": float(np.mean(latencies)) if latencies else None,
        },
    }


# --- Relatórios ------------------------------------------------------------------------


def _fmt(value, pct: bool = False, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.0%}" if pct else f"{value:.{digits}f}"


def _ci(interval: list[float]) -> str:
    return f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_run_report(meta: RunMeta, metrics: dict, records: list[EvalRecord]) -> str:
    cat, urg, rev, jz, cost, lat = (
        metrics[k] for k in ("categoria", "urgencia", "revisao", "juiz", "custo", "latencia_s")
    )
    labels = cat["labels"]

    summary = _table(
        ["Métrica", "Valor"],
        [
            [
                "F1 macro do agente (IC 95%)",
                f"**{_fmt(cat['agent_f1_macro'])}** {_ci(cat['agent_f1_ci95'])}",
            ],
            [
                "F1 macro do baseline, out-of-fold (IC 95%)",
                f"{_fmt(cat['baseline_f1_macro_oof'])} {_ci(cat['baseline_f1_ci95'])}",
            ],
            ["Diferença agente − baseline (IC 95%, pareado)", _ci(cat["diff_f1_ci95"])],
            ["Accuracy do agente (categoria)", _fmt(cat["agent_accuracy"], pct=True)],
            ["Urgência: accuracy", _fmt(urg["accuracy"], pct=True)],
            [
                f"Urgência alta: recall ({urg['n_alta_gold']} casos no gold)",
                _fmt(urg["alta_recall"], pct=True),
            ],
            [
                f"Busca: hit@{meta.top_k} (documento esperado entre as fontes)",
                _fmt(metrics["busca"][f"hit@{meta.top_k}_agente"], pct=True),
            ],
            ["Juiz: aderência à política (1–5)", _fmt(jz["aderencia"]["mean"], digits=2)],
            ["Juiz: tom (1–5)", _fmt(jz["tom"]["mean"], digits=2)],
            ["Juiz: utilidade (1–5)", _fmt(jz["utilidade"]["mean"], digits=2)],
            ["Respostas com aderência ≤ 2", _fmt(jz["aderencia_baixa_pct"], pct=True)],
            ["Escaladas para humano", _fmt(rev["escalada"], pct=True)],
            ["Custo do agente por avaliação (US$)", _fmt(cost["agente_por_avaliacao"], digits=5)],
            [
                "**Custo do agente por 1.000 avaliações (US$)**",
                f"**{_fmt(cost['agente_por_1000'], digits=2)}**",
            ],
            [
                "Latência por avaliação p50 / p95 (s)",
                f"{_fmt(lat['p50'], digits=1)} / {_fmt(lat['p95'], digits=1)}",
            ],
        ],
    )

    per_class = _table(
        ["Categoria", "F1 agente", "F1 baseline", "Suporte"],
        [
            [c, _fmt(v["agent_f1"], digits=2), _fmt(v["baseline_f1"], digits=2), v["support"]]
            for c, v in cat["per_class"].items()
        ],
    )
    confusion = _table(
        ["verdadeiro ↓ / predito →", *labels, NO_ANSWER],
        [[label, *row] for label, row in zip(labels, cat["confusion"], strict=False)],
    )
    retrieval = _table(
        ["Consulta", "hit@1", f"hit@{meta.top_k}"],
        [
            [name, _fmt(v["hit@1"], pct=True), _fmt(v[f"hit@{meta.top_k}"], pct=True)]
            for name, v in metrics.get("busca_sem_llm", {}).items()
        ],
    )
    judge_dist = _table(
        ["Critério", "1", "2", "3", "4", "5", "Média"],
        [
            [c, *jz[c]["dist"].values(), _fmt(jz[c]["mean"], digits=2)]
            for c in ("aderencia", "tom", "utilidade")
        ],
    )
    review = _table(
        ["Desfecho", "%"],
        [
            ["Aprovada no 1º rascunho", _fmt(rev["aprovada_1a"], pct=True)],
            ["Aprovada no 2º rascunho", _fmt(rev["aprovada_2a"], pct=True)],
            ["Reprovada duas vezes (escala)", _fmt(rev["reprovada"], pct=True)],
            ["Escalada por urgência alta", _fmt(rev["escalada_urgencia"], pct=True)],
            ["Recusas do modelo (n)", rev["recusas"]],
        ],
    )
    steps = _table(
        ["Passo", "Chamadas por avaliação", "Tokens de entrada (média)", "Tokens de saída (média)"],
        [
            [
                s,
                _fmt(v["chamadas_por_avaliacao"], digits=2),
                f"{v['entrada_media']:.0f}",
                f"{v['saida_media']:.0f}",
            ]
            for s, v in cost["tokens_por_passo"].items()
        ],
    )

    worst = sorted(
        (r for r in records if r.judge), key=lambda r: (r.judge.aderencia, r.judge.utilidade)
    )
    worst_rows = [
        [
            r.review_id[:8],
            r.gold_categoria,
            r.judge.aderencia,
            r.judge.utilidade,
            r.result.resposta_sugerida.replace("|", "/"),
            r.judge.justificativa.replace("|", "/"),
        ]
        for r in worst[:5]
    ]
    worst_table = _table(
        [
            "review_id",
            "categoria (gold)",
            "aderência",
            "utilidade",
            "resposta",
            "justificativa do juiz",
        ],
        worst_rows,
    )

    errors = f"Erros do agente: {metrics['n_errors']}. Erros do juiz: {metrics['n_judge_errors']}."
    cache_note = ""
    if cost["cache_tokens"]:
        cache_note = (
            f"\n> ⚠️ Houve {cost['cache_tokens']} tokens de cache; o custo acima não os precifica.\n"
        )
    agent_desc = (
        f"`{meta.model}`, effort `{meta.effort}`, top-{meta.top_k}, "
        f"até {1 + meta.max_redrafts} rascunhos"
    )
    judge_desc = (
        f"`{meta.judge_model}`, effort `{meta.judge_effort}`, rubrica `{meta.judge_prompt_version}`"
    )
    hit_agent = _fmt(metrics["busca"][f"hit@{meta.top_k}_agente"], pct=True)

    return f"""# Avaliação: {meta.model}

- **Rodada:** `{meta.run_id}` ({meta.created_at}, commit `{meta.git_commit}`)
- **Gold set:** `{meta.gold_path}` (sha256 `{meta.gold_sha}`), {meta.n_reviews} avaliações
- **Agente:** {agent_desc}
- **Prompts:** {", ".join(f"`{k}` {v}" for k, v in meta.prompt_versions.items())}
- **Juiz:** {judge_desc}
- **Embeddings:** `{meta.embedding_model}`
- **Preços:** tabela da Anthropic de {PRICES_AS_OF}
- {errors}

## Resumo

{summary}

O IC de 95% vem de {N_BOOTSTRAP} reamostragens bootstrap das mesmas avaliações. Se o
intervalo da diferença contém o zero, não dá para afirmar que um modelo é melhor que o outro
com este gold set. O baseline aqui é avaliado out-of-fold (cada avaliação prevista por um
modelo treinado sem ela); na validação cruzada repetida ele fez
{_fmt(cat["baseline_f1_cv_mean"])} ± {_fmt(cat["baseline_f1_cv_std"])}.

## Categoria por classe

{per_class}

### Matriz de confusão do agente

{confusion}

## Busca de política

O agente acertou o documento esperado em {hit_agent} dos casos (consulta = dica da categoria
**prevista** + texto). Sem LLM, comparando as consultas:

{retrieval}

"Esperado" é definido por categoria (`EXPECTED_SOURCES`), não por avaliação: é uma aproximação.

## Juiz (LLM-as-judge)

{judge_dist}

### As 5 respostas com pior aderência

{worst_table}

## Revisão e escalonamento

{review}

## Custo e tokens

{steps}

Custo total do agente nesta rodada: US$ {_fmt(cost["agente_total"], digits=4)}. Custo do juiz
(só para avaliar, não existe em produção): US$ {_fmt(cost["juiz_total"], digits=4)}.
{cache_note}"""


def render_comparison(runs: list[tuple[RunMeta, dict]]) -> str:
    rows = []
    for meta, m in runs:
        cat, jz = m["categoria"], m["juiz"]
        rows.append(
            [
                f"`{meta.model}`",
                f"{_fmt(cat['agent_f1_macro'])} {_ci(cat['agent_f1_ci95'])}",
                _fmt(m["urgencia"]["alta_recall"], pct=True),
                _fmt(m["busca"][f"hit@{meta.top_k}_agente"], pct=True),
                _fmt(jz["aderencia"]["mean"], digits=2),
                _fmt(jz["tom"]["mean"], digits=2),
                _fmt(jz["utilidade"]["mean"], digits=2),
                _fmt(m["revisao"]["escalada"], pct=True),
                _fmt(m["custo"]["agente_por_1000"], digits=2),
                _fmt(m["latencia_s"]["p50"], digits=1)
                + " / "
                + _fmt(m["latencia_s"]["p95"], digits=1),
                f"`{meta.run_id}`",
            ]
        )
    first = runs[0][1]["categoria"]
    rows.append(
        [
            "baseline (TF-IDF + LR)",
            f"{_fmt(first['baseline_f1_macro_oof'])} {_ci(first['baseline_f1_ci95'])}",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "≈ 0",
            "—",
            "—",
        ]
    )
    table = _table(
        [
            "Modelo",
            "F1 macro (IC 95%)",
            "Recall urgência alta",
            "hit@3",
            "Aderência",
            "Tom",
            "Utilidade",
            "Escaladas",
            "US$ / 1.000",
            "Latência p50 / p95 (s)",
            "Rodada",
        ],
        rows,
    )

    # Rodadas só são comparáveis com o mesmo gold set, o mesmo juiz e os mesmos prompts.
    keys = {
        "gold set": lambda m: m.gold_sha,
        "modelo do juiz": lambda m: (m.judge_model, m.judge_effort),
        "rubrica do juiz": lambda m: m.judge_prompt_version,
        "prompts do agente": lambda m: tuple(sorted(m.prompt_versions.items())),
        "n de avaliações": lambda m: m.n_reviews,
    }
    problems = [name for name, key in keys.items() if len({key(meta) for meta, _ in runs}) > 1]
    check = (
        "✅ Todas as rodadas usam o mesmo gold set, juiz, rubrica e prompts."
        if not problems
        else "⚠️ **Rodadas NÃO comparáveis**, diferem em: " + ", ".join(problems) + "."
    )
    return f"""# Comparação entre modelos

{check}

{table}

Juiz: `{runs[0][0].judge_model}` (rubrica `{runs[0][0].judge_prompt_version}`), igual em todas
as rodadas e diferente de todos os modelos avaliados. Preços de {PRICES_AS_OF}.
"""


# --- Orquestração ----------------------------------------------------------------------


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{sha}+alterações" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "desconhecido"


def execute_run(
    gold: pd.DataFrame,
    full_gold: pd.DataFrame,
    agent: TriageAgent,
    judge: Judge,
    model: str,
    effort: str,
    gold_path: Path,
    reports_dir: Path = REPORTS_DIR,
    workers: int = 4,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Path:
    """Roda, salva os resultados brutos, calcula as métricas e escreve o relatório."""
    if model == JUDGE_MODEL:
        raise ValueError(f"O juiz ({JUDGE_MODEL}) não pode avaliar o próprio modelo.")

    now = datetime.now()
    meta = RunMeta(
        run_id=f"{now:%Y-%m-%d_%H%M%S}_{model}",
        created_at=now.isoformat(timespec="seconds"),
        model=model,
        effort=effort,
        judge_model=JUDGE_MODEL,
        judge_effort=JUDGE_EFFORT,
        judge_prompt_version=judge.version,
        prompt_versions=agent.prompt_versions,
        gold_path=str(gold_path),
        gold_sha=file_sha(gold_path),
        n_reviews=len(gold),
        git_commit=git_commit(),
        embedding_model=embedding_model,
        top_k=TOP_K,
        max_redrafts=MAX_REDRAFTS,
    )
    run_dir = reports_dir / "runs" / meta.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")

    # Grava cada resultado assim que chega: se a rodada cair no meio, o que já foi
    # pago não se perde.
    results_path = run_dir / "results.jsonl"
    done = 0
    with results_path.open("w", encoding="utf-8") as fh:

        def save(record: EvalRecord) -> None:
            nonlocal done
            fh.write(record.model_dump_json() + "\n")
            fh.flush()
            done += 1
            if done % 10 == 0 or done == len(gold):
                print(f"  {done}/{len(gold)}", file=sys.stderr)

        records = run_eval(gold, agent, judge, workers=workers, on_record=save)
    results_path.write_text(  # reescreve na ordem do gold set
        "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8"
    )

    base = baseline.evaluate(full_gold)
    base_pred = pd.Series(base["predictions"].to_numpy(), index=full_gold["review_id"])
    f1s = base["scores"]["TF-IDF + Regressão Logística"]["f1_macro"]
    metrics = compute_metrics(
        records, base_pred, {"mean": float(f1s.mean()), "std": float(f1s.std())}
    )
    metrics["busca_sem_llm"] = retrieval_hit_rates(full_gold, agent.index)
    (run_dir / "summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = reports_dir / f"eval_{meta.run_id}.md"
    report.write_text(render_run_report(meta, metrics, records), encoding="utf-8")
    return report


def load_run(run_dir: Path) -> tuple[RunMeta, dict]:
    meta = RunMeta.model_validate_json((run_dir / "meta.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return meta, summary


# --- CLI -------------------------------------------------------------------------------


def _require_gold(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"{path} não existe. Rotule o gold set e rode `make gold-check` (ver Fase 2).")
    return load_gold(path)


def _cmd_run(args: argparse.Namespace) -> None:
    model = args.model or os.environ.get("ANTHROPIC_MODEL", "")
    if not model:
        sys.exit("Informe o modelo: make eval MODEL=claude-sonnet-5-5")
    if model == JUDGE_MODEL:
        sys.exit(f"O juiz ({JUDGE_MODEL}) não pode avaliar o próprio modelo.")
    full_gold = _require_gold(args.gold)
    gold = full_gold.head(args.limit) if args.limit else full_gold

    agent_cost, judge_cost = estimate_cost(model, len(gold))
    print(
        f"{len(gold)} avaliações com {model} (juiz: {JUDGE_MODEL}).\n"
        f"Estimativa GROSSEIRA: agente ~US$ {agent_cost:.2f} + juiz ~US$ {judge_cost:.2f} "
        f"= ~US$ {agent_cost + judge_cost:.2f}. Uma rodada piloto (LIMIT=10) mede o valor real."
    )
    if not args.yes:
        print("Nada foi executado. Para rodar, repita com YES=1 (ou --yes).")
        return

    effort = os.environ.get("ANTHROPIC_EFFORT") or "low"
    index = build_index()
    agent = TriageAgent(AnthropicLLM(model=model, effort=effort), index)
    judge = Judge(AnthropicLLM(model=JUDGE_MODEL, effort=JUDGE_EFFORT), load_chunks())
    report = execute_run(
        gold,
        full_gold,
        agent,
        judge,
        model,
        effort,
        args.gold,
        workers=args.workers,
        embedding_model=os.environ.get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
    )
    print(f"Relatório em {report}")


def _cmd_compare(args: argparse.Namespace) -> None:
    run_dirs = [Path(p) for p in args.runs] or sorted(
        p for p in RUNS_DIR.glob("*") if (p / "summary.json").exists()
    )
    if not run_dirs:
        sys.exit(f"Nenhuma rodada encontrada em {RUNS_DIR}.")
    out = REPORTS_DIR / f"compare_{datetime.now():%Y-%m-%d_%H%M%S}.md"
    out.write_text(render_comparison([load_run(p) for p in run_dirs]), encoding="utf-8")
    print(f"Comparação em {out}")


def _cmd_retrieval(args: argparse.Namespace) -> None:
    rates = retrieval_hit_rates(_require_gold(args.gold), build_index())
    for name, values in rates.items():
        print(f"{name:<36} " + "  ".join(f"{k} {v:.0%}" for k, v in values.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação do agente no gold set.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="roda agente + juiz no gold set")
    run.add_argument("--model")
    run.add_argument("--gold", type=Path, default=GOLD_PATH)
    run.add_argument("--limit", type=int)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--yes", action="store_true", help="confirma o gasto com a API")
    run.set_defaults(func=_cmd_run)

    compare = sub.add_parser("compare", help="tabela comparando rodadas salvas")
    compare.add_argument("runs", nargs="*")
    compare.set_defaults(func=_cmd_compare)

    retrieval = sub.add_parser("retrieval", help="hit rate da busca, sem LLM")
    retrieval.add_argument("--gold", type=Path, default=GOLD_PATH)
    retrieval.set_defaults(func=_cmd_retrieval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
