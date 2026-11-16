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
3. Rode `make data`. Isso gera em `data/processed/`:
   - `reviews_clean.csv`: só avaliações com comentário, texto normalizado (título + mensagem),
     uma linha por `review_id`;
   - `reviews_sample.csv`: amostra aleatória de 1.000 avaliações (seed 42) para desenvolvimento.

### Gold set (rotulado à mão)

150 avaliações (30 por nota, sem textos repetidos) rotuladas manualmente com categoria e
urgência, usadas para medir o baseline e o agente. `make gold` gera a planilha a rotular,
`make gold-check` valida o resultado. Critérios em
[`data/gold/LABELING_GUIDE.md`](data/gold/LABELING_GUIDE.md).

O gold set contém trechos do dataset da Olist, distribuído sob licença
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); o subconjunto rotulado
segue a mesma licença.

A exploração inicial (distribuição de notas, tamanho dos textos) está em
[`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb).

## Como rodar

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e `make`.

```bash
cp .env.example .env   # preencha as chaves quando chegar na Fase 4
make install
make check             # lint + testes
```

`make help` lista todos os comandos.

## Resultados

**Baseline clássico** (`make baseline`): TF-IDF + Regressão Logística, avaliado com validação
cruzada repetida no gold set. Relatório completo em [`reports/baseline.md`](reports/baseline.md).

_(Fase 5 — tabela baseline vs. agente, custo por 1.000 avaliações.)_

## Limitações e próximos passos

- **O gold set não tem a distribuição da produção.** Ele é estratificado por nota (30 de cada),
  enquanto na base real as notas altas são maioria. Isso dá exemplos suficientes de cada tipo
  de problema, mas as métricas medidas nele (sobretudo accuracy e o custo médio) não são
  exatamente as que se veria em produção. O F1 macro é menos afetado, porque pesa cada
  categoria igualmente.
- **Um único anotador.** Todos os rótulos foram dados por uma pessoa, sem medida de concordância
  entre anotadores; categorias ambíguas carregam um viés de interpretação.

_(Demais itens na Fase 7.)_

## Decisões

As decisões de projeto (o quê, por quê e a alternativa descartada) estão em
[DECISIONS.md](DECISIONS.md).
