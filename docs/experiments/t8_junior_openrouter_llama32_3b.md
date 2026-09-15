# T-8 — Reexecução no notebook júnior (OpenRouter / Llama 3.2 3B Instruct)

Data: 2026-09-15
Notebook: `notebooks/junior_regression.ipynb` (somente o júnior)
Provedor: OpenRouter
Modelo: `meta-llama/llama-3.2-3b-instruct`
Reviewer: ligado
Duração aproximada: 55 s
Artefato local (não versionado): `output/experiments/t8_junior_openrouter_llama32_3b/`

Mesmo patch local das outras rodadas OpenRouter: `max_tokens=8192` e
`reasoning.effort=none`. Smoke: `OK`.

Há variante `meta-llama/llama-3.2-3b-instruct:free`; esta execução usou o ID
sem `:free`.

**Tentativa `:free` (2026-09-15, em seguida).** Smoke com
`meta-llama/llama-3.2-3b-instruct:free` devolveu **404**:

```text
This model is unavailable for free. The paid version is available now -
use this slug instead: meta-llama/llama-3.2-3b-instruct
```

A conversão gratuita **não rodou**. O ID pago já foi medido nesta mesma data.

## Comando (equivalente, com o patch)

```text
notebook_path = notebooks/junior_regression.ipynb
output_dir    = output/experiments/t8_junior_openrouter_llama32_3b
use_llm       = True
llm_provider  = openrouter
llm_model     = meta-llama/llama-3.2-3b-instruct
enable_review = True
```

## Itens de verificação

| Item | Resultado |
|---|---|
| Proveniência | FE **template** (`invalid syntax` na linha 1). training / inference / evaluation: `llm`, `fenced`, `contract_ok` |
| ZIP `src/main.py` | presente |
| FE no ZIP | placeholder (`load_data()` devolve `([], [])`) |
| `training.py` | OLS misturado com `load_data` no módulo errado; espera pares `(x, y)` |
| Instalação isolada | **falha** com `random`; declara também `numpy` e `pandas` |
| Suíte do ZIP (sem `random`) | **3 failed, 4 passed**, cobertura **35%**. Os 4 passed são testes de template do FE |
| Entrypoint | **quebra:** `ZeroDivisionError` — FE vazio chega em `train_model` |
| Equivalência | `nao_executavel` |
| Reviewer | cobertura **0%**; reescreveu FE com string JSON inválida (`unterminated string literal`) |
| T-5 | `review_incomplete=true` |

Mesma classe da rodada Groq no júnior: modelo pequeno gera FE inválido, cai no template, o pipeline não executa. Os outros três estágios saíram por LLM, mas não conversam entre si nem com o contrato do `main.py`.

Nenhum código do sistema foi alterado nesta execução.
