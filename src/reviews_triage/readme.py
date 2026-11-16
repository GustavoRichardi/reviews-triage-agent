"""Publica resultados no README a partir dos relatórios salvos (sem copiar número à mão).

O README tem blocos marcados assim:
    <!-- auto:nome -->
    ...conteúdo gerado...
    <!-- /auto:nome -->
`make readme` reescreve todos a partir de reports/; `make gold-report` e `make judge-agreement`
atualizam o seu bloco ao terminar.
"""

import json
import re
from pathlib import Path

README_PATH = Path("README.md")
REPORTS_DIR = Path("reports")


def update_readme(blocks: dict[str, str], path: Path = README_PATH) -> None:
    text = path.read_text(encoding="utf-8")
    for name, content in blocks.items():
        pattern = re.compile(
            rf"(<!-- auto:{re.escape(name)} -->\n).*?(<!-- /auto:{re.escape(name)} -->)", re.S
        )
        if not pattern.search(text):
            raise ValueError(f"README sem o bloco <!-- auto:{name} -->")
        text = pattern.sub(lambda m, c=content: f"{m.group(1)}{c.strip()}\n{m.group(2)}", text)
    path.write_text(text, encoding="utf-8")


def render_judge_block(result: dict, run_id: str) -> str:
    """Kappa obtido por critério, além do veredito (pedido explícito: não só aprovado/reprovado)."""
    rows = []
    for criterion, m in result["criterios"].items():
        kappa = "—" if m["kappa_quadratico"] is None else f"{m['kappa_quadratico']:.2f}"
        rows.append(
            f"| {criterion} | **{kappa}** | {m['ate_1_ponto']:.0%} | "
            f"{m['vies_juiz_menos_humano']:+.2f} | {m['veredito']} |"
        )
    return (
        f"{result['n']} respostas do piloto `{run_id}` avaliadas às cegas por uma pessoa e pelo "
        "juiz, com a mesma rubrica. Critério pré-registrado: até 1 ponto ≥ 80%, |viés| ≤ 0,5 e "
        "kappa quadrático ≥ 0,40.\n\n"
        "| Critério | Kappa quadrático | Até 1 ponto | Viés (juiz − humano) | Veredito |\n"
        "|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
        f"Detalhes em [`reports/judge_agreement_{run_id}.json`]"
        f"(reports/judge_agreement_{run_id}.json)."
    )


def main() -> None:
    from reviews_triage.review_labels import STATS_PATH, render_readme_block

    blocks = {}
    if STATS_PATH.exists():
        saved = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        blocks["rotulagem"] = render_readme_block(saved["stats"], saved["meta"])
    agreements = sorted(REPORTS_DIR.glob("judge_agreement_*.json"))
    if agreements:
        saved = json.loads(agreements[-1].read_text(encoding="utf-8"))
        blocks["validacao-juiz"] = render_judge_block(saved["result"], saved["run_id"])
    update_readme(blocks)
    print(f"README atualizado: {', '.join(blocks) or 'nada a publicar ainda'}.")


if __name__ == "__main__":
    main()
