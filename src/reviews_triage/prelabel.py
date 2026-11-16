"""Pré-rotulagem do gold set com um LLM. Uma pessoa revisa cada linha depois (`make gold-review`).

Uso:
    make prelabel         # mostra a estimativa de custo e PARA
    make prelabel YES=1   # gera data/gold/prelabels.csv e prelabels_meta.json

O modelo é diferente de todos os modelos avaliados como agente (Opus, Sonnet, Haiku 5.5) e do
baseline. O prompt embute o LABELING_GUIDE.md: o guia que você segue é o mesmo que o modelo
segue, e a versão (hash) muda se o prompt OU o guia mudarem.
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from reviews_triage.agent import escape_text
from reviews_triage.gold import GOLD_DIR, TO_LABEL_PATH, read_labels
from reviews_triage.labels import Category, Urgency
from reviews_triage.llm import LLM, AnthropicLLM, LLMRefusal
from reviews_triage.prompts import PROMPTS_DIR, load_prompt, prompt_version
from reviews_triage.schemas import LLMCall

PRELABEL_MODEL = "claude-fable-5-2"
PRELABEL_EFFORT = "medium"
GUIDE_PATH = GOLD_DIR / "LABELING_GUIDE.md"
PRELABELS_PATH = GOLD_DIR / "prelabels.csv"
PRELABELS_META_PATH = GOLD_DIR / "prelabels_meta.json"
PRICE_PER_MTOK = (8.00, 40.00)  # claude-fable-5-2 (mesma tabela de evaluation.py)
EST_TOKENS = (1_500, 400)  # por avaliação: prompt com o guia + raciocínio curto


class PreLabel(BaseModel):
    # A justificativa vem antes do rótulo: o modelo aponta o trecho e só então decide.
    justificativa: str
    categoria: Category
    urgencia: Urgency


def guide_excerpt(path: Path = GUIDE_PATH) -> str:
    """As seções de critério do guia (categoria, desempate, urgência), sem o fluxo de
    trabalho, que só interessa a quem rotula numa planilha."""
    text = path.read_text(encoding="utf-8")
    return text[text.index("## Categoria") : text.index("## Consistência")].strip()


def build_prompt(prompts_dir: Path = PROMPTS_DIR, guide_path: Path = GUIDE_PATH) -> str:
    return load_prompt("prelabel", prompts_dir).replace("{guia}", guide_excerpt(guide_path))


def estimate_cost(n: int) -> float:
    return n * (EST_TOKENS[0] * PRICE_PER_MTOK[0] + EST_TOKENS[1] * PRICE_PER_MTOK[1]) / 1e6


def call_cost(call: LLMCall) -> float:
    return (call.input_tokens * PRICE_PER_MTOK[0] + call.output_tokens * PRICE_PER_MTOK[1]) / 1e6


def prelabel_one(llm: LLM, system: str, row) -> tuple[dict, LLMCall | None]:
    user = f"<nota>{row.review_score}</nota>\n<avaliacao>\n{escape_text(row.text)}\n</avaliacao>"
    out = {
        "review_id": row.review_id,
        "categoria_llm": "",
        "urgencia_llm": "",
        "justificativa_llm": "",
        "erro_llm": "",
    }
    try:
        label, call = llm.parse("prelabel", system, user, PreLabel)
    except LLMRefusal as exc:
        out["erro_llm"] = str(exc)
        return out, exc.call
    except Exception as exc:  # erro de API depois dos retries: sem sugestão, você rotula às cegas
        out["erro_llm"] = f"{type(exc).__name__}: {exc}"
        return out, None
    out.update(
        categoria_llm=label.categoria,
        urgencia_llm=label.urgencia,
        justificativa_llm=label.justificativa,
    )
    return out, call


def run_prelabel(
    to_label: pd.DataFrame, llm: LLM, system: str, workers: int = 4
) -> tuple[pd.DataFrame, list[LLMCall]]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda row: prelabel_one(llm, system, row), to_label.itertuples()))
    labels = pd.DataFrame([r for r, _ in results])  # pool.map preserva a ordem de entrada
    return labels, [c for _, c in results if c is not None]


def write_prelabels(
    labels: pd.DataFrame,
    calls: list[LLMCall],
    system: str,
    to_label_path: Path,
    out_path: Path = PRELABELS_PATH,
    meta_path: Path = PRELABELS_META_PATH,
) -> dict:
    labels.to_csv(out_path, index=False, encoding="utf-8-sig")
    meta = {
        "model": PRELABEL_MODEL,
        "effort": PRELABEL_EFFORT,
        "prompt_version": prompt_version(system),  # prelabel.md + trecho do guia
        "to_label_sha": hashlib.sha256(to_label_path.read_bytes()).hexdigest()[:8],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n": len(labels),
        "n_errors": int((labels["erro_llm"] != "").sum()),
        "cost_usd": round(sum(call_cost(c) for c in calls), 4),
        "input_tokens": sum(c.input_tokens for c in calls),
        "output_tokens": sum(c.output_tokens for c in calls),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Pré-rotula o gold set com um LLM.")
    parser.add_argument("--to-label", type=Path, default=TO_LABEL_PATH)
    parser.add_argument("--yes", action="store_true", help="confirma o gasto com a API")
    parser.add_argument("--force", action="store_true", help="refaz e sobrescreve as sugestões")
    args = parser.parse_args()

    if not args.to_label.exists():
        sys.exit(f"{args.to_label} não existe. Rode `make data` e `make gold` primeiro.")
    if PRELABELS_PATH.exists() and not args.force:
        sys.exit(f"{PRELABELS_PATH} já existe (já foi pago). Use --force para refazer.")
    to_label = read_labels(args.to_label)
    system = build_prompt()
    print(
        f"{len(to_label)} avaliações com {PRELABEL_MODEL} (effort {PRELABEL_EFFORT}), "
        f"prompt {prompt_version(system)}.\n"
        f"Estimativa GROSSEIRA: ~US$ {estimate_cost(len(to_label)):.2f}."
    )
    if not args.yes:
        print("Nada foi executado. Para rodar, repita com YES=1 (ou --yes).")
        return

    llm = AnthropicLLM(model=PRELABEL_MODEL, effort=PRELABEL_EFFORT)
    labels, calls = run_prelabel(to_label, llm, system)
    meta = write_prelabels(labels, calls, system, args.to_label)
    print(
        f"Sugestões em {PRELABELS_PATH}: {meta['n'] - meta['n_errors']} ok, "
        f"{meta['n_errors']} sem sugestão. Custo medido: US$ {meta['cost_usd']:.2f}."
    )


if __name__ == "__main__":
    main()
