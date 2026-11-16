# Decisões de projeto

Registro curto de cada decisão: **o quê**, **por quê** e **alternativa descartada**.
Mais recentes no fim.

## Fase 0 — Estrutura

### D01. uv como gerenciador de ambiente e dependências
- **O quê:** `uv` com `pyproject.toml` + `uv.lock` versionado.
- **Por quê:** um único binário resolve versão do Python, venv e lockfile; é muito rápido
  (importa na CI e no build do Docker) e o lock garante que todos instalam exatamente as
  mesmas versões.
- **Descartado:** `pip` + `requirements.txt` (sem lock real de dependências transitivas) e
  Poetry (mais lento, e o uv já cobre o mesmo papel).

### D02. Layout `src/` com pacote `reviews_triage`
- **O quê:** código em `src/reviews_triage/`, instalado no venv pelo `uv sync`.
- **Por quê:** os testes importam o pacote *instalado*, não arquivos soltos da raiz — evita
  o clássico "funciona na minha máquina porque o diretório atual está no `sys.path`".
- **Descartado:** layout plano (pacote na raiz), mais simples mas mascara erros de empacotamento.

### D03. Dependências entram fase a fase
- **O quê:** a Fase 0 só tem `ruff` e `pytest` (grupo `dev`). pandas, scikit-learn,
  LangGraph etc. entram quando forem usados.
- **Por quê:** cada dependência no `pyproject.toml` tem um motivo rastreável no histórico do
  git; e não baixamos o torch (via sentence-transformers, ~centenas de MB) antes de precisar.
- **Descartado:** declarar a stack inteira já no início.

### D04. Python 3.12 no ambiente local, mínimo 3.11
- **O quê:** `.python-version` = 3.12; `requires-python >= 3.11`.
- **Por quê:** 3.12 é estável e tem wheels prontas para toda a stack (torch, chroma); o
  mínimo 3.11 respeita o requisito do projeto. O ruff checa sintaxe compatível com 3.11.
- **Descartado:** 3.13, porque algumas libs de ML ainda chegam atrasadas em versões novas.

### D05. `.env` carregado pelo Makefile, sem python-dotenv
- **O quê:** o Makefile faz `-include .env` + `export`; o Docker Compose usará `env_file`.
- **Por quê:** evita uma dependência extra só para ler variáveis; o código lê `os.environ`
  e não sabe de onde os valores vieram (12-factor). O `-` faz o `make` funcionar mesmo sem
  `.env` (ex.: na CI, onde os testes não usam chaves).
- **Descartado:** `python-dotenv`/`pydantic-settings` (fora da stack combinada).
- **Atenção:** rodar `uv run python ...` direto *não* carrega o `.env`; use os alvos do `make`.

### D06. ruff para lint e formatação
- **O quê:** `ruff check` (regras E, F, I, UP, B, SIM) + `ruff format`, linha de 100.
- **Por quê:** uma ferramenta substitui flake8 + isort + black, roda em milissegundos.
  As regras escolhidas pegam bugs reais (B), imports bagunçados (I) e sintaxe antiga (UP)
  sem virar ruído.
- **Descartado:** `select = ["ALL"]` — gera muitos falsos positivos para um projeto pequeno.

## Fase 1 — Dados

### D07. Limpeza mínima e compartilhada
- **O quê:** `clean()` só junta quebras de linha/espaços repetidos e tira espaços das pontas.
  Não tira acento, não muda caixa, não remove pontuação nem stopwords.
- **Por quê:** cada modelo tem seu próprio pré-processamento ideal. O TF-IDF pode normalizar
  caixa/acentos dentro do vetorizador; o LLM e o modelo de embeddings funcionam melhor com o
  texto original ("NÃO chegou!!" carrega urgência). Limpar demais aqui jogaria sinal fora para
  todos.
- **Descartado:** um pipeline de NLP "clássico" (lowercase, sem acento, stemming) aplicado a
  todos os dados.

### D08. Título + mensagem num único campo `text`
- **O quê:** `text = "<título>. <mensagem>"`; o filtro "tem comentário" olha só a mensagem.
- **Por quê:** títulos como "Produto errado" carregam a categoria de forma direta; descartá-los
  seria perder sinal. Mas título sozinho ("Recomendo") é genérico demais para triar, então não
  conta como comentário preenchido.
- **Descartado:** usar só a mensagem; ou manter título e mensagem separados (complica todos os
  consumidores para um ganho que não precisamos).

### D09. Uma linha por `review_id`
- **O quê:** `drop_duplicates(subset="review_id")`.
- **Por quê:** no CSV original a mesma avaliação aparece ligada a mais de um pedido. Mantê-las
  duplicaria textos e, pior, poderia colocar a mesma avaliação em treino e teste na validação
  cruzada (vazamento de dados).
- **Descartado:** deduplicar por texto idêntico: "Muito bom" escrito por clientes diferentes é
  um dado real e faz parte da distribuição.

### D10. Saída em CSV, amostra aleatória simples com seed fixa
- **O quê:** `data/processed/*.csv`; amostra de 1.000 com `random_state=42`.
- **Por quê:** CSV é legível em qualquer lugar e o pandas não precisa de dependência extra
  (Parquet exigiria `pyarrow`). O volume (~dezenas de milhares de linhas) não justifica um
  formato colunar. A seed fixa torna a amostra reprodutível.
- **Descartado:** Parquet; amostra estratificada aqui (fica para o gold set, onde importa).

### D11. Notebook só explora; a lógica fica no pacote
- **O quê:** o notebook importa `reviews_triage.data.clean` em vez de reimplementar a limpeza.
  `ipykernel` e `matplotlib` entram só no grupo `dev`.
- **Por quê:** o que exploramos é exatamente o que os modelos vão ver, e a lógica fica testada
  pelo pytest. Notebook com lógica própria tende a divergir do código de produção.
- **Descartado:** fazer limpeza e análise juntas no notebook.
