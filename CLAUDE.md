# reviews-triage-agent: instruções para o Claude Code

Projeto de portfólio do usuário para vagas de estágio em AI Engineer. Ele precisa entender e
defender cada decisão numa entrevista: explique o porquê enquanto trabalha. O histórico de
decisões está em `DECISIONS.md`; leia antes de mudar qualquer coisa já decidida.

## Regras de trabalho (do usuário)

- Código e nomes em inglês; README, comentários e `DECISIONS.md` em português.
- Toda decisão nova vira uma entrada curta no `DECISIONS.md` (o quê, por quê, alternativa descartada).
- Nada de dependência fora da stack sem perguntar: Python 3.11+, uv, pandas, scikit-learn,
  LangGraph, SDK da Anthropic, sentence-transformers, Chroma, FastAPI, pytest, ruff, Docker,
  GitHub Actions; Langfuse opcional (só com chaves no `.env`); `ipykernel` e `matplotlib` só no grupo dev.
- Nenhuma chave no código: tudo via `.env`, que o Makefile carrega (use os alvos do `make`).
- Simples antes de sofisticado; nada de abstração que o projeto não precise.
- Decisão de custo, modelo ou escopo: pergunte antes.
- No fim de cada fase: `make check` passando, mostrar como testar e commit com mensagem clara.

## Chamadas pagas

- `make prelabel`, `make pilot` e `make eval` primeiro **sem** `YES=1`: isso só imprime a
  estimativa. Mostre a estimativa ao usuário e **espere o ok explícito** antes de repetir com `YES=1`.
- Rodadas completas (`make eval ... YES=1`) só depois de o usuário ver o custo **medido** no piloto.
- Não use `--force` no `prelabel` sem pedido: refaz e paga de novo.

## Invariantes de método (não quebre)

- **Juiz congelado** (D40, D41, D53): `claude-fable-5-2`, effort `medium`, rubrica
  `prompts/judge.md` com hash `c4a1e9c6`. Não edite a rubrica nem as constantes de `judge.py`. Se
  `test_judge_configuration_is_frozen` falhar, avise o usuário; não "conserte" o teste.
- **Critério de aprovação do juiz pré-registrado** (D52): não mude `MIN_WITHIN_ONE`, `MAX_ABS_BIAS`
  nem `MIN_KAPPA` em `judge_validation.py`.
- **Validação cega do juiz:** até o usuário terminar as 30 notas (`make judge-score`), não leia,
  imprima nem resuma o `results.jsonl` do piloto (`reports/runs/*_piloto/`) nem nada que contenha
  notas ou justificativas do juiz. O `reports/pilot_*.md` pode ser lido, porque não tem as notas.
- **Amostra cega dos rótulos:** não mostre ao usuário o conteúdo de `data/gold/prelabels.csv`;
  30 avaliações são rotuladas sem ver a sugestão.
- **Rótulos e notas humanas são do usuário:** nunca rotule, nunca dê notas e nunca escreva
  `data/gold/gold.csv` ou `data/judge_validation/*/human.csv`. Só as ferramentas interativas
  (`make gold-review`, `make judge-score`) escrevem esses arquivos. Peça ao usuário para rodá-las num
  terminal próprio, porque elas esperam o teclado dele.
- O piloto roda fora do gold set. `make eval` só com o `gold.csv` completo e validado (`make gold-check`).
- Os números do README vêm dos relatórios (`make readme`): não edite à mão os blocos `<!-- auto:... -->`.

## Roteiro (onde paramos: Fase 5, código pronto, nada rodado com dados reais)

1. `make install` e `cp .env.example .env` (o usuário preenche `ANTHROPIC_API_KEY`). A primeira
   busca baixa o modelo de embeddings (~470 MB).
2. Dataset da Olist em `data/raw/olist_order_reviews_dataset.csv` (o usuário baixa do Kaggle).
   `make data` e `make gold` (grátis). Commit do `to_label.csv`.
3. `make prelabel`: mostrar a estimativa (~US$ 4), esperar o ok, depois `make prelabel YES=1`.
   Commit de `prelabels.csv` e `prelabels_meta.json` (sem abrir o conteúdo para o usuário).
4. Usuário, no terminal dele: `make gold-review`, `make gold-check`, `make gold-report`. Commit.
5. `make pilot MODEL=claude-sonnet-5-5`: estimativa (~US$ 2,40), ok, `YES=1`. Trazer do
   `reports/pilot_*.md` o custo medido contra o estimado e a projeção das rodadas completas.
6. Usuário: `make judge-score`, depois `make judge-agreement` (publica o kappa no README). Commit.
7. Com a aprovação do usuário sobre o custo medido: `make eval MODEL=<modelo> JUDGE_SAMPLE=60`
   para `claude-opus-5-5`, `claude-sonnet-5-5` e `claude-haiku-5-5` (estimativa e ok antes de
   cada uma; `JUDGE_SAMPLE=60` se o usuário confirmar após o piloto). Depois `make eval-compare`,
   `make readme` e commit. Escolher o `ANTHROPIC_MODEL` padrão a partir da comparação (D25).
8. Próximas fases: 6 (API FastAPI `POST /triage`, Docker, docker-compose, GitHub Actions com ruff +
   pytest sem chave, Langfuse opcional) e 7 (README final com diagrama, resultados, custo por 1.000
   avaliações, limitações, como rodar em 3 comandos e GIF da demo).
