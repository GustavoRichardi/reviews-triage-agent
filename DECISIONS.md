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

## Fase 4 — Agente

### D25. Modelo sem padrão no código; Sonnet 5.5 durante o desenvolvimento
- **O quê:** `ANTHROPIC_MODEL` é obrigatório (erro claro se faltar). Na Fase 4 usamos
  `claude-sonnet-5-5`; o padrão do projeto sai da comparação Opus × Sonnet × Haiku na Fase 5.
- **Por quê:** escolher modelo é decisão de custo/qualidade e deve vir de medição, não de
  palpite. Um padrão escondido no código também contaminaria a comparação.
- **Descartado:** fixar um modelo agora e trocar depois.

### D26. Sem `temperature` e sem `thinking`; custo controlado por `effort`
- **O quê:** a requisição só leva `output_config={"effort": "low"}` (configurável por
  `ANTHROPIC_EFFORT`).
- **Por quê:** os modelos atuais rejeitam `temperature`/`top_p` com valor não padrão (erro 400), e
  não existe um jeito de desligar o raciocínio que funcione nos três modelos da comparação
  (Sonnet e Opus 5.5 recusam `thinking: disabled`). `effort: low` é aceito por todos e é o
  controle certo para uma tarefa simples. Consequência: as respostas não são determinísticas;
  a Fase 5 deve repetir medições sensíveis e reportar a variação.
- **Descartado:** `temperature=0` (400 nesses modelos) e `thinking: disabled` (incompatível
  entre modelos).

### D27. Saída estruturada com `messages.parse` + Pydantic
- **O quê:** `classify` e `review` devolvem modelos Pydantic via `output_format`; o schema JSON
  enviado ao modelo é gerado dos próprios tipos (`Literal` em `labels.py`).
- **Por quê:** o schema restringe a geração (não só pede "responda em JSON"), então a categoria
  nunca sai fora da lista, e o SDK já devolve o objeto validado. A lista de rótulos fica em
  um único lugar.
- **Descartado:** pedir JSON no prompt e fazer `json.loads` (frágil); *tool use* forçado
  (`tool_choice` forçado é rejeitado nesses modelos).

### D28. Recusa do modelo é desfecho de negócio, não exceção solta
- **O quê:** `stop_reason == "refusal"` vira `LLMRefusal`; `run()` captura e devolve o resultado
  parcial com `escalate_to_human=True` e o motivo. Resposta truncada por `max_tokens` é erro.
- **Por quê:** avaliações são texto de usuário e podem acionar os classificadores de segurança.
  Um humano deve ver esses casos, e o custo da chamada recusada continua contado.
- **Fallback desativado (ver D38).**

### D29. `LLM` como `Protocol` de dois métodos
- **O quê:** o agente depende de `parse()` e `text()`; `AnthropicLLM` é a implementação real.
- **Por quê:** é a única abstração do projeto, e tem dois usos concretos: testes com LLM
  simulado (exigência da CI) e a Fase 5 medir custo/latência pela mesma interface.
- **Descartado:** `unittest.mock` nos testes (acopla ao formato interno do SDK); camada
  genérica multi-provedor (YAGNI).

### D30. Escalonamento por regra em código, num nó `finalize` sem LLM
- **O quê:** `escalate_to_human` é verdadeiro se a urgência é alta, ou se a revisão não aprovou
  depois das tentativas. Um nó próprio aplica a regra.
- **Por quê:** "quando chamar um humano" precisa ser previsível e testável, não depender de um
  LLM. Isso acrescenta um nó ao desenho de quatro passos, mas mantém o grafo como a única
  fonte do comportamento (API e avaliação só chamam `run()`).
- **Limite de tentativas:** `MAX_REDRAFTS = 1`, ou seja, **2 rascunhos no total**. Se a revisão
  reprovar os dois, o caso escala para um humano (ver D37).

### D31. Busca de política: embeddings locais, Chroma em memória, consulta com dica de categoria
- **O quê:** `paraphrase-multilingual-MiniLM-L12-v2` (local, multilíngue) gera os vetores; o Chroma
  guarda 20 seções e é reconstruído ao iniciar; a consulta é `"<dica da categoria>. <texto>"`;
  top-3.
- **Por quê:** o corpus tem poucas dezenas de seções, então reconstruir o índice leva menos
  tempo do que o modelo leva para carregar, e some o problema de índice desatualizado. A
  dica da categoria usa o resultado do `classify` para orientar a busca.
- **Hipótese ainda não medida:** que a dica melhora o acerto da busca. A Fase 5 mede o hit rate
  com e sem ela, sem chamar LLM (usando o rótulo do gold set).
- **Descartado:** Chroma persistente (invalidação de cache sem ganho); filtro rígido por
  categoria (um erro de classificação tiraria a política certa da busca).

### D32. Seções curtas (≤ 80 palavras), conferidas por teste
- **O quê:** a política é dividida por seção `##`, e um teste falha se alguma passar de 80 palavras.
- **Por quê:** o modelo de embeddings lê no máximo 128 tokens e descarta o resto em silêncio. Seção
  longa demais perde o final e a busca piora sem aviso.
- **Descartado:** *chunking* por janela deslizante (corta regras no meio); um modelo de
  embeddings maior (mais pesado para ganho incerto neste corpus).

### D33. Texto da avaliação tratado como dado não confiável
- **O quê:** a avaliação vai entre tags com `<` e `>` escapados; todos os prompts mandam não seguir
  instruções vindas do texto; o `review` desconfia tanto da avaliação quanto do rascunho.
- **Por quê:** quem escreve a avaliação controla parte do prompt (injeção de prompt). Escapar
  impede "fechar" a tag; o que protege de verdade é o `review` conferir o rascunho contra a
  política, mais o atendente humano no fim.
- **Limite:** não é à prova de falhas; a Fase 5 deveria incluir alguns casos adversariais no gold set.

### D34. Prompts em `prompts/*.md`, versão = hash do conteúdo
- **O quê:** um arquivo por prompt; `prompt_version()` devolve o SHA-256 curto do texto.
- **Por quê:** mudou o texto, mudou a versão, sem depender de lembrar de "subir o número". O
  relatório da Fase 5 registra o hash e o git guarda o histórico e o porquê.
- **Atenção:** `classify.md` espelha o `LABELING_GUIDE.md` (mesmas definições e regras de
  desempate). Se o guia mudar, o prompt deve mudar junto; um teste só garante que todos os
  rótulos aparecem no prompt.
- **Descartado:** número de versão manual; prompts dentro do código Python.

### D35. `pydantic` declarado como dependência direta
- **O quê:** `pydantic` entra no `pyproject.toml`, embora o `anthropic` e o `langgraph` já o
  tragam.
- **Por quê:** o projeto o importa diretamente; depender de uma dependência transitiva quebra
  quando ela deixa de ser instalada. Não instala nenhum pacote novo.

### D36. Cache de prompt e streaming não usados
- **O quê:** nem `cache_control` nem streaming.
- **Por quê:** os prompts de sistema são curtos (abaixo do tamanho mínimo cacheável) e as saídas
  são pequenas. O `LLMCall` já registra tokens de cache lidos/escritos, então a Fase 5 mostra se
  vale a pena (por exemplo, se o prompt crescer).

### D37. No máximo 2 rascunhos; duas reprovações escalam para humano
- **O quê:** `MAX_REDRAFTS = 1`. O `review` pode devolver o rascunho uma vez; se reprovar a
  segunda versão, `escalate_to_human=True` e o último rascunho vai junto, marcado para revisão.
- **Por quê:** se o modelo errou duas vezes no mesmo caso, uma terceira tentativa tende a errar
  igual (o caso costuma ser ambíguo ou a política não o cobre). Um humano resolve melhor, e
  cada tentativa a mais custa duas chamadas (draft + review) e aumenta a latência.
- **Descartado:** 3 rascunhos (custo e latência a mais para um ganho provavelmente pequeno; a
  Fase 5 mostra quantos casos são aprovados só na segunda tentativa).

### D38. Sem fallback automático de modelo quando há recusa
- **O quê:** não usamos o `fallbacks` do lado do servidor (que reenvia a requisição recusada para
  outro modelo). Recusa vira `escalate_to_human` (D28).
- **Por quê:** (1) a Fase 5 compara Opus × Sonnet × Haiku; com fallback, parte das respostas
  de um modelo seria gerada por outro e a comparação mediria uma mistura; (2) custo e latência
  por avaliação ficariam dependentes de um segundo modelo com outro preço; (3) uma avaliação
  que aciona recusa é, ela mesma, um caso que merece olhar humano.
- **Quando rever:** depois que a Fase 5 fixar o modelo padrão, se a taxa de recusa medida for
  relevante a ponto de sobrecarregar a fila humana.
- **Descartado:** fallback ativado por padrão (recomendação geral do SDK, que não se encaixa
  num projeto cujo resultado principal é comparar modelos).

### D39. Atraso acima de 7 dias úteis: o cliente escolhe uma compensação
- **O quê:** acima de 7 dias úteis o cliente escolhe entre manter o pedido com cupom de 15% ou
  cancelar com reembolso total; o reembolso do frete (a partir de 3 dias úteis) vale nos dois
  casos. O texto da política diz explicitamente que as opções não se somam, e um teste confere.
- **Por quê:** decisão de negócio do projeto. Escrever "não se somam" de forma literal importa
  para o agente: uma frase ambígua na política vira promessa errada na resposta.
- **Descartado:** compensações cumulativas (versão do primeiro rascunho da política).
