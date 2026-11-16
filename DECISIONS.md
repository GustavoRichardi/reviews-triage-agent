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

## Fase 2 — Gold set

### D12. 150 avaliações, estratificadas por nota (30 cada), sem textos repetidos
- **O quê:** `groupby("review_score").sample(30)` sobre textos únicos (ignorando caixa).
- **Por quê:** numa amostra aleatória as notas 5 dominariam e categorias de problema teriam
  poucos exemplos. Com 80 avaliações seriam ~13 por categoria, o que deixa a validação cruzada
  da Fase 3 com variância alta demais; 150 dá ~25 por categoria.
- **Custo da escolha:** o gold set não tem a distribuição da produção (registrado nas
  limitações do README). Métricas por classe e F1 macro continuam comparáveis; accuracy e
  custo médio, não exatamente.
- **Descartado:** amostra aleatória simples (poucos problemas) e estratificar pela categoria
  (impossível: ela é justamente o que ainda não sabemos).

### D13. Rotular categoria e urgência
- **O quê:** colunas `categoria`, `urgencia` e `observacao` (livre).
- **Por quê:** a urgência decide o que vai para um humano (`escalate_to_human`), é a decisão de
  maior impacto no negócio. Sem rótulo, não daria para medir se o agente acerta a prioridade.
- **Descartado:** rotular só categoria.

### D14. Arquivo gerado (`to_label.csv`) separado do arquivo rotulado (`gold.csv`)
- **O quê:** o script só escreve `to_label.csv` (e recusa sobrescrever sem `--force`); o
  humano salva o resultado como `gold.csv`, que nenhum script escreve.
- **Por quê:** horas de rotulagem manual não podem ser apagadas por um `make gold` rodado por
  engano. Artefato gerado por máquina e artefato humano têm ciclos de vida diferentes.
- **Descartado:** rotular direto em `to_label.csv`.

### D15. Guia de rotulagem com regras de desempate
- **O quê:** `data/gold/LABELING_GUIDE.md` com definição, exemplos e regras para casos com mais
  de um problema.
- **Por quê:** sem regra escrita, "atrasou e veio quebrado" recebe rótulos diferentes em dias
  diferentes. O guia também vira a especificação do prompt de classificação na Fase 4, assim
  humano e LLM seguem o mesmo critério.
- **Descartado:** rotular "por intuição".

### D16. Ordem embaralhada; a nota fica visível para quem rotula
- **O quê:** as 150 linhas saem embaralhadas; `review_score` continua no arquivo.
- **Por quê:** blocos de mesma nota induzem a rotular no piloto automático. A nota fica
  visível porque faz parte da avaliação e o agente também a receberá em produção.
- **Descartado:** esconder a nota (o humano teria menos contexto do que o próprio agente).

### D17. Valores dos rótulos e colunas de rótulo em português
- **O quê:** `categoria`/`urgencia` e valores como `atraso_entrega` ficam em português; o código
  (funções, variáveis) segue em inglês.
- **Por quê:** são dados de domínio e coincidem com o contrato da API (`categoria`, `urgencia`)
  definido no escopo. Traduzir criaria um mapeamento a mais sem ganho.
- **Descartado:** traduzir rótulos para inglês.

### D18. Leitura tolerante a planilhas
- **O quê:** o gerador grava UTF-8 com BOM; o leitor detecta `,` ou `;` e aceita BOM.
- **Por quê:** o Excel em pt-BR só mostra acentos com BOM e salva CSV com `;`. Melhor o código
  aceitar isso do que pedir para quem rotula configurar exportação.
- **Descartado:** exigir um formato exato e falhar.

## Fase 3 — Baseline clássico

### D19. TF-IDF de palavras (uni + bigramas) + Regressão Logística
- **O quê:** `TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), sublinear_tf=True)` +
  `LogisticRegression(class_weight="balanced")`, hiperparâmetros fixos (sem busca).
- **Por quê:** é o baseline clássico de classificação de texto: rápido, barato, interpretável
  (dá para listar os termos mais fortes de cada categoria) e difícil de bater com pouco dado.
  Bigramas capturam negação e expressões ("nao chegou", "veio errado"); `strip_accents` junta
  "não"/"nao", comum em avaliações digitadas no celular; `class_weight` compensa categorias raras.
- **Descartado:** n-gramas de caracteres (mais robustos a erro de digitação, mas pouco
  interpretáveis); busca de hiperparâmetros (com 150 exemplos, escolher hiperparâmetros na
  mesma validação cruzada que reporta o resultado inflaria a métrica; validação aninhada seria
  complexidade sem ganho real aqui).

### D20. A nota entra como token (`nota_1` … `nota_5`)
- **O quê:** a entrada do modelo é `"nota_<n> <texto>"`.
- **Por quê:** o agente vai receber a nota, então o baseline também precisa dela para a
  comparação ser justa. Um token é a forma mais simples de dar essa informação ao TF-IDF.
- **Descartado:** `ColumnTransformer` com one-hot da nota (mesmo efeito, mais código).

### D21. Validação cruzada repetida (5 folds × 5 repetições) + referência de classe majoritária
- **O quê:** média ± desvio de accuracy e F1 macro sobre 25 divisões; um `DummyClassifier`
  (sempre a classe mais comum) avaliado da mesma forma.
- **Por quê:** com 150 exemplos, o resultado de uma única divisão em folds depende da sorte; o
  desvio padrão mostra o quanto o número é confiável. O dummy é o piso: se o modelo não o
  supera com folga, não aprendeu nada útil.
- **Descartado:** um único split treino/teste (estimativa ruidosa e desperdiça dados).

### D22. F1 macro como métrica principal
- **O quê:** F1 macro em destaque; accuracy aparece como apoio.
- **Por quê:** o F1 macro dá peso igual a cada categoria. Com classes desbalanceadas, um modelo
  que só acerta `elogio` pode ter accuracy razoável e ser inútil para triagem.
- **Descartado:** accuracy e F1 ponderado (ambos dominados pelas classes grandes).

### D23. Matriz de confusão a partir de predições out-of-fold
- **O quê:** `cross_val_predict` com 5 folds: cada avaliação é prevista por um modelo que não a
  viu no treino. A mesma função vai fornecer as predições do baseline na Fase 5.
- **Por quê:** é a forma honesta de ter uma predição por exemplo do gold set sem um conjunto de
  teste separado (que, com 150 exemplos, ficaria pequeno demais).
- **Descartado:** treinar no gold set inteiro e prever nele mesmo (mede memorização).

### D24. Relatório em Markdown com hash do gold set
- **O quê:** `reports/baseline.md` (sobrescrito a cada execução; o histórico fica no git), com
  data, versão do scikit-learn e o sha256 do `gold.csv`.
- **Por quê:** Markdown renderiza no GitHub sem dependência extra. O hash diz exatamente com
  qual versão dos rótulos o número foi gerado. Se um rótulo for corrigido, o número muda e o
  hash também.
- **Descartado:** imagem PNG da matriz de confusão (exigiria matplotlib em produção).
