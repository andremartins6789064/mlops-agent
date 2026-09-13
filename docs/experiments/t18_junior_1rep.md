# T-18 — Execução reduzida da matriz

Data: 2026-09-13
Notebook: `notebooks/junior_regression.ipynb`
Repetições: `1`
Modelos: `gemma4:e2b`, `nemotron-3-nano:4b`, `smollm2:1.7b`

## Comando executado

```bash
uv run python scripts/run_experiment_matrix.py \
  --notebooks notebooks/junior_regression.ipynb \
  --models gemma4:e2b nemotron-3-nano:4b smollm2:1.7b \
  --repetitions 1 \
  --output-csv output/experiments/t18_junior_1rep.csv \
  --output-root output/experiments/t18_junior_1rep \
  --timeout 120 \
  --max-run-seconds 900
```

## Resultado

As três combinações foram registradas no CSV, mas falharam durante a conversão
com `LLM request failed`. Nenhuma combinação produziu métricas de equivalência,
cobertura, lint, tipos ou mutação.

| Modelo | Duração (s) | Resultado |
|---|---:|---|
| `gemma4:e2b` | 317,610 | falha de requisição LLM |
| `nemotron-3-nano:4b` | 385,447 | falha de requisição LLM |
| `smollm2:1.7b` | 222,723 | falha de requisição LLM |

O Ollama respondeu a uma requisição simples e os três modelos estavam
instalados. A causa operacional mais provável é o timeout fixo de 60 segundos
do cliente OpenAI-compatible por requisição. O erro é sanitizado em
`BaseOpenAICompatibleClient`, portanto a exceção original não aparece no CSV.

Na execução do `smollm2:1.7b`, o pipeline chegou a gerar os quatro módulos e
registrou proveniência no `execution.log`; porém falhou posteriormente durante
o fluxo de revisão por nova requisição à LLM. O estágio `inference` caiu para
template por código Python inválido.

## Validação do projeto

```text
101 passed, 2 skipped
coverage: 88.10%
```

Esta é uma execução parcial e negativa da T-18. Ela não permite concluir sobre
equivalência funcional, efeito do modelo ou efeito da qualidade do notebook.
Também não substitui a execução planejada com o notebook sênior.

O timeout foi posteriormente tornado configurável; a reexecução está registrada
abaixo.

## Reexecução com timeout configurável

O timeout foi tornado configurável em `ConversionRequest` e no CLI do harness,
com padrão de 300 segundos. A matriz foi repetida com:

```bash
uv run python scripts/run_experiment_matrix.py \
  --notebooks notebooks/junior_regression.ipynb \
  --models gemma4:e2b nemotron-3-nano:4b smollm2:1.7b \
  --repetitions 1 \
  --llm-timeout 300 \
  --output-csv output/experiments/t18_junior_1rep_timeout300.csv \
  --output-root output/experiments/t18_junior_1rep_timeout300 \
  --timeout 120 \
  --max-run-seconds 900
```

| Modelo | Duração (s) | Resultado |
|---|---:|---|
| `gemma4:e2b` | 753,182 | conversão concluída; equivalência não executada |
| `nemotron-3-nano:4b` | 900,000 | timeout global da combinação |
| `smollm2:1.7b` | 350,867 | conversão concluída com erro de sintaxe |

O `gemma4:e2b` completou as dez fases, teve duas iterações de revisão, zero
erros de lint, 16 erros de tipo e score de mutação `1.0000`. A equivalência
ficou como `nao_executavel` porque o harness foi executado sem
`--pipeline-command`.

O `smollm2:1.7b` teve origem LLM nos estágios de feature engineering, training
e inference, mas fallback para template em evaluation. O erro final foi
`unterminated string literal`.

O timeout configurável resolveu o primeiro bloqueio de 60 segundos, mas a
execução ainda não é suficiente para concluir a T-18: falta configurar o
comando de execução do pipeline gerado e corrigir ou reportar os artefatos
semanticamente inválidos.
