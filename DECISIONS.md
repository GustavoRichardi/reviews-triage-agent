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
