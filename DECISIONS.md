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

## Fase 5 — Avaliação

### D40. Juiz: `claude-fable-5-2`, fixo em código
- **O quê:** `JUDGE_MODEL = "claude-fable-5-2"`, `JUDGE_EFFORT = "medium"`, como constantes em
  `judge.py`, não variáveis de ambiente. A avaliação se recusa a rodar se o modelo avaliado
  for o próprio juiz.
- **Por quê:** o juiz precisa (1) ser o mesmo nas três rodadas, senão a nota mede o juiz e não o
  agente, e (2) nunca avaliar o próprio modelo, porque LLMs tendem a favorecer textos que
  eles mesmos gerariam (*self-preference bias*). Como Opus, Sonnet e Haiku 5.5 estão todos na
  comparação, nenhum deles serve: qualquer um julgaria a si mesmo em uma rodada. O Fable 5.2
  não é candidato e é mais capaz que os três, o que importa para detectar promessas sutis
  fora da política. `medium` dá raciocínio suficiente para conferir a resposta contra a
  política inteira sem pagar pelo `high`.
- **Custo:** é o modelo mais caro (US$ 8 / 40 por milhão de tokens). Estimativa: ~US$ 0,06
  por avaliação julgada, ~US$ 9 por rodada de 150, ~US$ 27 nas três. É custo de medição, não
  de produção.
- **Risco:** o Fable 5.2 exige retenção de dados de 30 dias; organizações com *zero data
  retention* recebem erro 400.
- **Descartado:** um dos três candidatos (julgaria a si mesmo); juiz variando por rodada (para
  cada candidato, um juiz diferente); `claude-opus-4-8` (mais barato, mas da mesma família do
  Opus 5.5, então o viés de autopreferência pode aparecer em parte).

### D41. Rubrica do juiz congelada antes de qualquer rodada real
- **O quê:** `prompts/judge.md` foi escrito e congelado com hash **`edf6a162`** antes de
  qualquer execução com dados reais. **Substituído pela v2 (`c4a1e9c6`), também antes de
  qualquer rodada real: ver D53.** O hash está em `JUDGE_PROMPT_VERSION`; o `Judge` se recusa
  a rodar se o arquivo não bater, e um teste falha se a rubrica ou a configuração do juiz mudar.
- **Por quê:** rubrica ajustada depois de ver os resultados vira métrica escolhida para parecer
  boa. Congelar antes torna a nota um compromisso prévio.
- **Se precisar mudar:** nova entrada aqui com o motivo, constante atualizada e **todas** as
  rodadas refeitas; a comparação marca como "não comparáveis" rodadas com rubricas diferentes.
- **Descartado:** só versionar pelo git (não impede rodar com uma rubrica editada sem perceber).

### D42. Rubrica com âncoras por nível, critérios independentes, justificativa antes da nota
- **O quê:** cada critério (aderência, tom, utilidade) descreve o que vale 1, 2, 3, 4 e 5. O
  juiz escreve a justificativa citando os trechos e só depois dá as notas, usando saída
  estruturada (`Literal[1..5]`).
- **Por quê:** uma escala sem âncoras deixa cada nota ao gosto do momento e as médias
  derivam. Justificar antes de dar a nota obriga o juiz a olhar a evidência, e a
  justificativa serve para auditar os piores casos no relatório.
- **Descartado:** nota única de 1 a 10 (não diz o que errou); comparação par a par entre
  modelos (custo quadrático e não dá nota absoluta).

### D43. O juiz vê a política inteira, não só os trechos recuperados
- **O quê:** o juiz recebe as 20 seções da política, a avaliação, a nota e a resposta. Não
  recebe a categoria prevista.
- **Por quê:** se a busca trouxe a política errada, a resposta pode seguir os trechos e mesmo
  assim estar errada para o cliente. Julgar contra a política inteira mede o resultado que o
  cliente veria. Sem a categoria prevista, o juiz não é ancorado pelo próprio erro do agente.
- **Descartado:** julgar só contra os trechos recuperados (esconderia erros de busca).

### D44. Resultados brutos são a fonte da verdade; relatórios são derivados
- **O quê:** cada rodada grava `meta.json` (modelo, effort, juiz, hashes de todos os prompts,
  sha do gold set, commit do git), `results.jsonl` (uma linha por avaliação, gravada assim que
  sai) e `summary.json`; o `.md` é gerado a partir deles. `make eval-compare` não chama LLM.
- **Por quê:** a parte cara (chamadas de API) roda uma vez e fica guardada; relatórios e
  comparações podem ser refeitos de graça. A gravação incremental não perde o que já foi pago
  se a rodada cair no meio. O commit e os hashes dizem exatamente o que gerou cada número.
- **Descartado:** só o relatório final em Markdown (não dá para recalcular nem auditar).

### D45. Intervalo de confiança por bootstrap pareado
- **O quê:** 1.000 reamostragens das mesmas avaliações para o F1 macro do agente, do baseline e
  da diferença entre eles.
- **Por quê:** com 150 exemplos, diferenças de alguns pontos de F1 podem ser ruído. Pareado (a
  mesma reamostragem para os dois modelos) isola a diferença entre modelos da variação entre
  amostras. Se o intervalo da diferença contém zero, o relatório não declara vencedor.
- **Descartado:** reportar só o número pontual; teste de McNemar (só olha a accuracy, não o F1 macro).

### D46. Hit rate da busca por categoria, com ablação sem LLM
- **O quê:** cada categoria do gold set tem um documento esperado (`EXPECTED_SOURCES`); hit@3 =
  esse documento aparece entre as 3 fontes. Além do hit@3 do agente (com a categoria prevista),
  `make eval-retrieval` compara a consulta só com o texto e com a dica da categoria do gold set.
- **Por quê:** testa a hipótese da D31 sem gastar com API. A variante com o rótulo do gold set
  é o teto (classificação perfeita); a diferença para o hit@3 do agente mostra quanto o erro
  de classificação prejudica a busca.
- **Limite:** relevância por categoria é uma aproximação; uma avaliação de atraso pode ser bem
  servida pela política de reembolso, e isso conta como erro.

### D47. Custo calculado a partir de tokens medidos e de uma tabela de preços datada
- **O quê:** custo = tokens reais de cada chamada × preço do modelo (`PRICES_PER_MTOK`, preços
  de 2026-11). O custo do agente (o que existiria em produção) é reportado separado do custo do
  juiz (só existe na avaliação). Modelo sem preço na tabela gera erro.
- **Por quê:** custo por 1.000 avaliações é o número que um gestor usa para decidir; misturar o
  custo do juiz inflaria o custo de produção. Erro para modelo sem preço evita reportar zero
  por engano.
- **Limite:** tokens de cache não são precificados (não usamos cache); se aparecerem, o
  relatório avisa.

### D48. Gasto só com confirmação explícita; piloto antes da rodada completa
- **O quê:** `make eval` sem `YES=1` só mostra a estimativa e para. `LIMIT=10` roda um piloto.
  Erros de API numa avaliação ficam registrados e não derrubam a rodada; 4 avaliações em
  paralelo.
- **Por quê:** a rodada completa custa dólares de verdade; um comando digitado errado não deve
  gastar. O piloto mede o custo real (a estimativa de tokens é grosseira) antes do gasto maior.
  O paralelismo reduz uma rodada de ~40 min para ~10 min, porque o tempo é quase todo espera
  de rede.
- **Atenção:** a latência é medida por avaliação, com 4 em paralelo; sob limite de taxa da
  API ela pode sair um pouco maior do que com uma avaliação por vez.

### D49. Piloto fora do gold set, estratificado por nota
- **O quê:** `make pilot` roda agente e juiz em N avaliações (padrão 10, 2 por nota) que **não**
  estão no gold set: exclui o que `make gold` vai sortear (o sorteio é determinístico), os
  arquivos `to_label.csv`/`gold.csv` existentes e textos idênticos. Substitui o "piloto" com
  `make eval LIMIT=10` da D48.
- **Por quê:** (1) o gold set ainda não foi rotulado; quem lê as respostas do agente para
  essas avaliações (e a validação do juiz exige ler) passa a rotulá-las influenciado pelo
  agente, o que infla o F1 dele; (2) o piloto não precisa de rótulos, então não espera a
  rotulagem; (3) estratificar por nota como o gold set deixa o custo medido representativo,
  porque reclamações geram mais reescritas e respostas mais longas que elogios.
- **Saída:** custo medido contra o estimado, tokens por passo, latência e projeção das três
  rodadas com o juiz em 150 e em 60. A projeção para Opus e Haiku usa os tokens medidos com
  Sonnet (aproximação: modelos geram quantidades diferentes de tokens de saída).
- **Descartado:** piloto nas 10 primeiras do gold set (contamina a rotulagem).

### D50. Amostra do juiz configurável e igual para todos os modelos
- **O quê:** `JUDGE_SAMPLE=n` faz o juiz avaliar só n avaliações; F1, busca, custo e latência
  continuam nas 150. A amostra é estratificada por categoria (cotas proporcionais), com seed fixa,
  e calculada sobre o gold set inteiro, então é o mesmo conjunto para Opus, Sonnet e Haiku e
  para qualquer `LIMIT`. O hash do conjunto fica no `meta.json`, e a comparação marca como
  não comparáveis rodadas com amostras diferentes. As médias do juiz ganham IC 95% (bootstrap).
- **Por quê:** com o mesmo conjunto, a diferença de nota entre modelos vem dos modelos e não de
  avaliações diferentes. Estratificar por categoria garante que nenhum tipo de caso suma da
  amostra. O IC mostra o custo estatístico de julgar 60 em vez de 150.
- **Padrão:** julga todas, até você decidir o tamanho depois do piloto (a decisão provável é 60).
- **Descartado:** amostra aleatória por rodada (cada modelo seria julgado em casos diferentes).

### D51. Validação humana do juiz às cegas
- **O quê:** o piloto gera `data/judge_validation/<rodada>/to_score.csv` com avaliação, nota do
  cliente e resposta, em ordem embaralhada, **sem** as notas e a justificativa do juiz. O
  relatório do piloto também não mostra as notas do juiz. Você salva o resultado como
  `human.csv` (o gerador nunca sobrescreve), e `make judge-agreement` mede, por critério:
  concordância exata e a até 1 ponto (com IC de Wilson), erro médio, viés (juiz − humano) e
  kappa de Cohen com peso quadrático. Lista as discordâncias de 2 pontos ou mais, com a
  justificativa do juiz.
- **Por quê:** quem vê a nota do juiz antes tende a concordar com ela, e a concordância sai
  inflada. O IC de Wilson deixa visível o tamanho da incerteza com poucas respostas
  (9 de 10 é compatível com ~60% de concordância real).
- **Limite:** com 10 respostas o resultado é indicativo, e o relatório avisa. Um piloto de
  10 tende a ter poucas respostas ruins, então mal testa se o juiz pega violações.

### D52. Critério de aprovação do juiz pré-registrado (confirmado)
- **O quê:** por critério (aderência, tom, utilidade), o juiz é **aprovado** se: (1) pelo menos 80%
  das notas dele ficam a no máximo 1 ponto da humana; (2) o viés médio fica entre −0,5 e +0,5;
  (3) o kappa quadrático é pelo menos 0,40. É **inconclusivo** quando as notas não variam (kappa
  indefinido) e **reprovado** nos outros casos. As constantes ficam em `judge_validation.py`,
  definidas antes de existir qualquer nota humana. **Confirmado** antes do piloto e antes de
  qualquer nota humana. O README publica o kappa obtido por critério, não só o veredito.
- **Por quê:** é a mesma lógica do congelamento da rubrica: critério escolhido depois de ver a
  concordância vira critério feito para passar. O kappa entrou depois que um teste com um juiz
  falso, que dá sempre a mesma nota, **passou** nos critérios (1) e (2): quando a maioria das
  respostas é boa, um juiz "preguiçoso" que dá 5 para tudo concorda "a até 1 ponto" quase
  sempre e mesmo assim não detecta nada. O kappa dele é ~0.
- **Descartado:** só concordância exata (rígida demais para uma escala de 5 pontos); só
  correlação (não detecta um juiz sistematicamente mais generoso).

### D53. Rubrica do juiz v2, trocada ANTES de qualquer rodada real
- **O quê:** a rubrica passou de `edf6a162` (v1) para **`c4a1e9c6`** (v2). Na `aderencia`, o
  nível 3 deixou de incluir "genérica demais para conferir", o exemplo do nível 4 passou de
  omissão ("omite que o reembolso é no mesmo meio de pagamento") para imprecisão ("diz 'no seu
  cartão' quando a política diz 'no mesmo meio de pagamento'"), e entrou a frase "Omitir não é
  violar...; a falta de informação concreta é medida em utilidade".
- **Quando:** antes de qualquer execução real: nenhum piloto, nenhuma rodada, nenhuma nota do
  juiz existia. A mudança veio de uma revisão da própria rubrica, não de resultado.
- **Por quê:** na v1, uma resposta genérica perdia pontos em aderência **e** em utilidade, ou
  seja, a mesma falha contava duas vezes, contrariando a instrução de avaliar os critérios de
  forma independente. Também punia duas vezes a estratégia segura do agente (encaminhar sem
  prometer nada). Na v2, aderência mede só o que está **dito errado**, que é o risco que o
  passo `review` e o negócio querem controlar; a falta de informação fica em utilidade.
- **Daqui em diante:** vale a regra da D41. Qualquer outra mudança na rubrica exige nova
  entrada aqui e refazer todas as rodadas.

### D54. Piloto com 30 avaliações
- **O quê:** `make pilot` passa a usar N=30 por padrão (6 por nota).
- **Por quê:** com 10 respostas, a validação do juiz é quase inútil: 9 de 10 de concordância
  é compatível com ~60% real (IC de Wilson), e é provável que não haja respostas ruins
  suficientes para testar se o juiz as detecta. 30 estreita o intervalo pelo custo estimado de
  ~US$ 2,40.

### D55. Gold set pré-rotulado por `claude-fable-5-2`, revisado por humano
- **O quê:** `make prelabel` pede ao `claude-fable-5-2` (effort `medium`) categoria, urgência e
  uma justificativa curta para cada uma das 150 avaliações. O prompt é `prompts/prelabel.md` com
  as seções de critério do `LABELING_GUIDE.md` embutidas na hora de rodar; versão =
  hash do prompt resultante (**`101fec44`** hoje), que muda se o prompt **ou** o guia mudarem.
  As sugestões ficam em `data/gold/prelabels.csv` com custo e versão em `prelabels_meta.json`.
  A pessoa revisa cada linha (`make gold-review`); só o rótulo revisado vale.
- **Por quê:** reduz bastante o trabalho manual. Rotular do zero exige decidir; revisar
  exige só conferir. O guia continua sendo a única especificação: humano e LLM seguem o mesmo
  texto.
- **Por que este modelo:** não é nenhum dos modelos avaliados como agente (Opus, Sonnet, Haiku
  5.5) nem o baseline. É o mesmo modelo do juiz, mas para outra tarefa (rótulo de categoria, não
  nota de resposta).
- **Riscos assumidos:** ancoragem e circularidade (ver D56 e o README). Em particular, o Fable
  e os agentes são todos Claude: o simples fato de ser "outro modelo" não elimina erros
  correlacionados.
- **Descartado:** rotulagem 100% manual (mais trabalho); pré-rotular com um dos modelos
  avaliados (o agente seria medido contra os próprios rótulos).

### D56. Amostra cega de 30 e diagnóstico de circularidade
- **O quê:** 30 avaliações (6 por nota, seed fixa) são revisadas **às cegas**: a pessoa rotula sem
  ver a sugestão, a sugestão aparece depois e ela decide se mantém ou troca. Ficam guardados os
  dois rótulos (`categoria_cega` e o final). O relatório de avaliação mostra (1) com que
  frequência o agente concorda com o pré-rótulo, (2) nos casos em que o humano corrigiu o LLM,
  com que frequência o agente repetiu o mesmo erro, e (3) o F1 do agente no subconjunto cego.
- **Por quê:** sem isso, o "% de sugestões corrigidas" é só um limite inferior da discordância
  real, porque a sugestão ancora a revisão. A amostra cega dá a concordância LLM × humano sem
  ancoragem, e a diferença entre as duas mede a ancoragem. O erro compartilhado mede quanto da
  revisão humana estava protegendo a avaliação contra erros correlacionados.
- **Custo:** ~30 decisões feitas do zero, em vez de 30 confirmações (~10 minutos a mais).
  `--blind 0` desliga, mas aí o relatório não consegue separar acerto de ancoragem.
- **Descartado:** só o modo revisão (sem como medir a ancoragem).

### D57. Ferramenta de revisão no terminal
- **O quê:** `make gold-review` mostra uma avaliação por vez. Enter aceita a sugestão; "3",
  "a" ou "3a" corrigem a categoria e/ou a urgência; "-- texto" adiciona observação; `v`
  volta; `q` sai. Salva a cada resposta (troca atômica do arquivo) e retoma de onde parou. A
  ferramenta escreve `gold.csv`, mas só com decisões digitadas pela pessoa. A D14 continua
  valendo: nenhum script gera rótulo sozinho nem apaga uma decisão já tomada (só o `v` desfaz).
- **Por quê:** abrir uma planilha de 150 linhas com 12 colunas é lento e convida a erro de
  digitação. Uma tecla por decisão e a validação na hora (só aceita rótulos válidos) eliminam os
  erros que o `gold-check` pegaria depois.
- **Descartado:** interface web (dependência e servidor para uma tarefa de 30 minutos).

### D58. Resultados publicados no README a partir dos relatórios
- **O quê:** o README tem blocos `<!-- auto:... -->` que `make gold-report`, `make
  judge-agreement` e `make readme` reescrevem a partir dos JSONs em `reports/`. O bloco do juiz
  publica o **kappa obtido** por critério, além do veredito.
- **Por quê:** número copiado à mão envelhece e erra. Gerado a partir do relatório salvo, cada
  número do README tem origem rastreável. Um veredito sem o kappa esconderia o quanto o juiz
  passou ou falhou.
