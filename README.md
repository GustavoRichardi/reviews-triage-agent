# reviews-triage-agent

Agente que lê avaliações de clientes de e-commerce em português, classifica o problema,
busca a política de atendimento certa e sugere uma resposta — com avaliação automática de
qualidade e custo.

> 🚧 Em construção. Este README é um esqueleto e será completado ao longo das fases.

## Problema de negócio

Plataformas de delivery e e-commerce recebem milhares de avaliações por dia. Triar à mão é
lento e inconsistente: a mesma reclamação recebe respostas diferentes dependendo de quem
atende, e casos urgentes se perdem no volume. O agente classifica, prioriza e rascunha a
resposta para um atendente revisar — o humano continua no controle da decisão final.

## Arquitetura

_(Fase 4 — diagrama Mermaid do grafo classify → retrieve_policy → draft → review.)_

## Dados

Usamos o [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(Kaggle). Os dados brutos **não** são versionados.

1. Baixe o dataset no Kaggle (é preciso estar logado).
2. Copie `olist_order_reviews_dataset.csv` para `data/raw/`.

## Como rodar

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e `make`.

```bash
cp .env.example .env   # preencha as chaves quando chegar na Fase 4
make install
make check             # lint + testes
```

`make help` lista todos os comandos.

## Resultados

_(Fase 5 — baseline vs. agente, custo por 1.000 avaliações.)_

## Limitações e próximos passos

_(Fase 7.)_

## Decisões

As decisões de projeto (o quê, por quê e a alternativa descartada) estão em
[DECISIONS.md](DECISIONS.md).
