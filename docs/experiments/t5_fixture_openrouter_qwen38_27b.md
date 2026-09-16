# T-5 — Tentativa OpenRouter (Qwen 3.8 27B)

Data: 2026-09-16
Notebook: `tests/fixtures/simple_regression.ipynb`
Provedor: OpenRouter
Modelo: `qwen/qwen3.8-27b`
Reviewer: ligado (`enable_review=true`, `review_max_llm_calls=4`,
`review_max_seconds=300`)
Duração da conversão: 452,7 s
Artefato local (não versionado):
`output/experiments/t5_fixture_openrouter_qwen38_27b/`

## Cliente

O caminho vanilla (`convert_notebook` sem `max_tokens`) devolve **402**
nesta conta — o OpenRouter assume um teto enorme. Smoke e conversão usaram
o mesmo patch local da rodada júnior (`t8_junior_openrouter_qwen38_27b.md`):

- `max_tokens=8192`
- `reasoning.effort=none`

Sem isso a conversão não parte. O patch não entrou no produto.

## Resultado da T-5

| Campo | Valor |
|---|---|
| Modelo, durações, 4 estágios no log | **Sim.** 4/4 `origin=llm`, parse `fenced` |
| `review_enabled` | true |
| `review_incomplete` | **true** |
| `review_error` | `evaluation: Reviewer time budget exceeded (300.0s)` |
| `review_iterations` | 1 |
| `test_coverage` | 0.0 |
| `LinearRegression().fit` | **Não** nesta geração |

`feature_engineering.split_data` ausente (`contract_ok=false`). O Reviewer
rodou ~6,7 min depois da geração e estourou o teto de 300 s no estágio
`evaluation` — chegou perto de processar os quatro, mas não concluiu.

O `src/training.py` da árvore de revisão virou um dump JSON de achados de
lint (mesmo tipo de Reviewer destrutivo da rodada júnior OpenRouter).

## Conclusão

**T-5 continua aberta.** OpenRouter + Qwen 27B gera os quatro estágios e
preenche o log, mas o Reviewer não termina no orçamento de tempo.

Uma segunda tentativa com `review_max_seconds=600` **não concluiu**.
Os quatro estágios de código saíram; na geração dos testes o OpenRouter
devolveu **402**: a conta só cobria ~4789 tokens e o runner pedia 8192.
Saldo esgotado pela primeira rodada (~7,5 min) mais esta (parou aos ~2 min).

**T-5 continua aberta.** Sem crédito novo no OpenRouter, não dá para
repetir o Qwen nesta conta.

**Decisão (2026-09-16, D8).** Não contratar API paga para fechar a T-5.
O free-tier explica 429 (Gemini 20 req/dia), 413 (Groq TPM 8k) e 402
(OpenRouter). Não explica o `review_incomplete` com suíte verde no Gemini
(16 passed). Isso é o Reviewer: cobertura &lt; 80% (`main.py`) dispara o
loop, o fallback marca os quatro estágios, a T-27 limita a 1 chamada.
Correção: **T-43**, depois reexecução gratuita (Groq ou Gemini).

Nenhum código do sistema foi alterado (só o runner em `/tmp`).
