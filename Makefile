# Carrega o .env (se existir) e exporta as variáveis para os comandos abaixo.
# Assim não precisamos de python-dotenv como dependência.
-include .env
export

.DEFAULT_GOAL := help
.PHONY: help install lint format test check data gold gold-check baseline policy-search triage eval eval-retrieval eval-compare

help: ## Lista os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Cria o .venv e instala dependências (incluindo dev)
	uv sync

lint: ## Checa estilo e erros comuns (não altera arquivos)
	uv run ruff check .
	uv run ruff format --check .

format: ## Corrige o que for automático e formata o código
	uv run ruff check --fix .
	uv run ruff format .

test: ## Roda a suíte de testes
	uv run pytest

check: lint test ## O mesmo que a CI vai rodar

data: ## Limpa as avaliações brutas e gera data/processed/ (requer data/raw/*.csv)
	uv run python -m reviews_triage.data

gold: ## Gera data/gold/to_label.csv (150 avaliações, 30 por nota) para rotular
	uv run python -m reviews_triage.gold sample

gold-check: ## Valida os rótulos em data/gold/gold.csv e mostra o progresso
	uv run python -m reviews_triage.gold check

baseline: ## Avalia TF-IDF + Regressão Logística no gold set e gera reports/baseline.md
	uv run python -m reviews_triage.baseline

policy-search: ## Mostra os trechos de política que a busca devolve (ex.: make policy-search Q="pedido atrasou")
	uv run python -m reviews_triage.policy "$(Q)"

triage: ## Triagem de uma avaliação com o agente real (ex.: make triage TEXT="Atrasou 10 dias" SCORE=1)
	uv run python -m reviews_triage.agent --text "$(TEXT)" $(if $(SCORE),--score $(SCORE),)

eval-retrieval: ## Hit rate da busca no gold set, sem LLM (custo zero)
	uv run python -m reviews_triage.evaluation retrieval

eval: ## Avalia o agente no gold set (MODEL=..., LIMIT=n opcional; só gasta com YES=1)
	uv run python -m reviews_triage.evaluation run $(if $(MODEL),--model $(MODEL),) $(if $(LIMIT),--limit $(LIMIT),) $(if $(YES),--yes,)

eval-compare: ## Tabela comparando as rodadas salvas em reports/runs (sem LLM)
	uv run python -m reviews_triage.evaluation compare $(RUNS)
