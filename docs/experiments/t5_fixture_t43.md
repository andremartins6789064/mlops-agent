# T-5 — Reexecução pós T-43 (Groq + Gemini gratuito)

Data: 2026-09-16
Notebook: `tests/fixtures/simple_regression.ipynb`
Reviewer: ligado (`enable_review=true`, padrão `review_max_llm_calls=1`)
T-43 nesta branch: `93112b6`

Cliente de produção, sem patch. Sem plano pago (D8).

## Groq `openai/gpt-oss-20b`

Duração: 108,9 s
Artefato: `output/experiments/t5_fixture_groq20b_t43/`

| Campo | Valor |
|---|---|
| 4 estágios `llm` | Sim, `contract_ok` |
| Suíte | coleta falhou: `from __future__` fora do topo em FE e inference |
| `review_incomplete` | **true** |
| `review_error` | `Reviewer call budget exceeded` |

A T-43 **não se aplica** aqui: há erro de sintaxe nomeando estágios, não só
cobertura. Uma chamada não cobre dois estágios. Comportamento esperado.

## Gemini `gemini-3.6-flash`

Smoke `OK`. Conversão parou no Architecture Agent com **429** (cota
gratuita 20 req/dia). Sem artefato.

## Gemini `gemini-3.1-flash-lite`

Duração: 78,3 s
Artefato: `output/experiments/t5_fixture_gemini31flashlite_t43/`

| Campo | Valor |
|---|---|
| 4 estágios `llm` | Sim, `contract_ok` |
| Pytest | **9 passed**, cobertura **45%** (`main.py` 0/39) |
| Lint/tipos | ruff exit 1; mypy 3 erros (`inference.py`, `main.py`) |
| `review_incomplete` | **true** |
| `review_error` | `Reviewer call budget exceeded` |

Este é o caso “suíte verde, cobertura baixa”. A T-43 evitaria LLM se o
único problema fosse o `main.py`. Não foi: lint/mypy **nomeiam** mais de
um arquivo de estágio, então 1 chamada esgota o orçamento. Correto para
a T-27/T-43; a T-5 ainda não fecha.

## Conclusão

**T-5 continua aberta.** A T-43 cobre o falso incompleto por tabela de
cobertura. Com testes passando, lint/tipo em vários estágios ainda
dispara `Reviewer call budget exceeded`.

Nenhum código do sistema foi alterado nesta execução.
