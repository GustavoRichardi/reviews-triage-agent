# Guia de rotulagem do gold set

> **Rascunho para revisão.** Ajuste as definições *antes* de começar a rotular. Mudar uma regra
> no meio do caminho obriga a revisar o que já foi rotulado.

O gold set é a "verdade" contra a qual o baseline e o agente são medidos. Um rótulo inconsistente
aqui vira erro de medida lá na frente, então o objetivo deste guia é que **a mesma avaliação
receba sempre o mesmo rótulo**, não importa quando você a leia.

## Fluxo

1. `make gold` gera `data/gold/to_label.csv` (150 avaliações, 30 por nota, ordem embaralhada).
2. Abra numa planilha (Google Sheets ou LibreOffice: importar como CSV, UTF-8, separador vírgula).
3. Preencha `categoria` e `urgencia` em todas as linhas; `observacao` é livre e opcional.
4. Salve como **`data/gold/gold.csv`** (CSV). Pode ser vírgula ou ponto e vírgula.
5. `make gold-check` valida os valores e mostra o progresso. Rode quantas vezes quiser.

Use o texto **e** a nota: a nota faz parte da avaliação e também chega ao agente em produção.

## Categoria (uma só por avaliação)

Copie o valor exatamente como está na primeira coluna.

| valor | quando usar | exemplos |
|---|---|---|
| `atraso_entrega` | Pedido não chegou, chegou depois do prazo ou o cliente ainda aguarda. | "Até agora não recebi"; "Chegou 10 dias depois do previsto" |
| `produto_errado_ou_faltando` | Chegou item diferente do comprado (modelo, cor, tamanho), quantidade menor ou faltou parte do pedido. | "Comprei 2, veio 1"; "Pedi azul e veio preto" |
| `defeito_ou_qualidade` | Produto chegou quebrado, com defeito, avariado, parou de funcionar, ou qualidade bem abaixo do anunciado. | "Veio com a tela trincada"; "Material muito fino, não parece a foto" |
| `atendimento_vendedor` | A queixa principal é a interação com a loja: não responde, cancelou sem avisar, problema com nota fiscal, cobrança ou dificuldade de troca/devolução. | "Mandei 3 e-mails e ninguém respondeu"; "Cancelaram meu pedido sem explicação" |
| `elogio` | Avaliação positiva **sem** problema que peça ação. | "Chegou antes do prazo, recomendo" |
| `outro` | Neutro, sugestão, texto sem sentido, ou problema que não se encaixa acima. | "Ok"; "Poderiam ter mais cores" |

### Regras de desempate

1. **Escolha o problema que o cliente precisa que seja resolvido**, não o primeiro mencionado.
   "Atrasou e veio quebrado" → `defeito_ou_qualidade` (o atraso já passou; o defeito ainda exige troca).
2. **`atendimento_vendedor` só quando o atendimento é a queixa em si.** "Não recebi e o vendedor
   não responde" → `atraso_entrega` (o problema de raiz é a entrega; a falta de resposta pesa na urgência).
3. **Elogio com ressalva vira a ressalva** se ela for uma reclamação concreta.
   "Produto ótimo, mas demorou muito" → `atraso_entrega`. Se for só um detalhe sem queixa
   ("ótimo, só a caixa veio amassada mas tudo certo") → `elogio`.
4. **Na dúvida real entre duas categorias**, escolha uma e escreva a outra em `observacao`.
   Essas linhas viram material para a seção de limitações.

## Urgência

Pergunta-guia: *quanto custa para o negócio se essa avaliação esperar alguns dias na fila?*

| valor | quando usar | exemplos |
|---|---|---|
| `alta` | Risco de escalar ou dano relevante: cliente sem produto **e** sem resposta; menciona Procon, Reclame Aqui, processo ou chargeback; cobrança indevida; risco à saúde/segurança. | "Paguei, não recebi e ninguém me responde, vou ao Procon" |
| `media` | Problema real que exige uma ação (troca, reenvio, reembolso), sem sinal de escalada. | "Veio a cor errada, quero trocar" |
| `baixa` | Nada a resolver: elogio, neutro, ou reclamação de algo que já se resolveu. | "Demorou, mas chegou certinho" |

`alta` dispara `escalate_to_human` no agente, então deve ser **rara**. Se você perceber que está
marcando muitas, revise o critério.

## Consistência

- Rotule em uma ou duas sessões, não em dez: o critério deriva com o tempo.
- Ao terminar, releia 15 avaliações aleatórias sem olhar o rótulo que deu e veja se concorda
  consigo mesmo. Se discordar em várias, ajuste o guia e revise.
