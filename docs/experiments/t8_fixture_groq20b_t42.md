# T-8 — Reexecução da fixture pós T-42 (Groq)

Data: 2026-09-16
Notebook: `tests/fixtures/simple_regression.ipynb`
Provedor: Groq
Modelo: `openai/gpt-oss-20b`
Reviewer: ligado (`enable_review=true`, padrão `review_max_llm_calls=1`)
Duração da conversão: 80,2 s
Artefato local (não versionado): `output/experiments/t8_fixture_groq20b_t42/`
T-42 nesta branch: `2873062`

Compara com `docs/experiments/t8_fixture_gemini36flash.md` (item 3 verde;
item 4 era 51% ≠ 54% porque o ZIP não levava o FE corrigido).

## Gemini nesta sessão

Smoke de `gemini-3.6-flash` respondeu `OK`. A conversão completa parou no
Code Generator com **429** de cota gratuita diária (20 req/dia no modelo).
Não foi reexecutada. Groq é o mesmo modelo das reexecuções da fixture.

## Comando executado

Chamada direta a `convert_notebook` (o harness da matriz recusa fixtures):

```text
notebook_path = tests/fixtures/simple_regression.ipynb
output_dir    = output/experiments/t8_fixture_groq20b_t42
use_llm       = True
llm_provider  = groq
llm_model     = openai/gpt-oss-20b
enable_review = True
```

## Itens de verificação

| Item | Resultado |
|---|---|
| 1. `LinearRegression().fit` | **Sim.** `model = LinearRegression()` + `model.fit(...)` via LLM, `contract_ok` |
| 2. LLM ativa | **4/4 `llm`**, parse `fenced`, modelo `openai/gpt-oss-20b` |
| 3. Suíte ZIP em `/tmp` | **Não verde.** Coleta: `NameError: Any` em `tests/test_inference.py`. Deps ok (`joblib`, `numpy`, `pandas`, `scikit-learn`; sem stdlib nem submódulo) |
| 4. Cobertura Reviewer vs ZIP | **Mesma árvore.** sha256 idêntico nos 5 `src/*.py` e nos 4 testes. `validate_output` reportou 0.0% nas duas (coleta interrompida). Ignorando o teste que não coleta: **45% = 45%** (2 failed, 11 passed) |
| 5. Fallback malformado | teste unitário permanece |
| T-5 | **Não.** `review_incomplete=true`, `review_error="Reviewer call budget exceeded"`, 1 iteração |

### Item 3 — ZIP em `/tmp/mlops-t8-fixture-t42`

Instalação isolada ok. Layout `src/` + `tests/` com `sys.path` para `src/`.
`src/main.py` importa `Sequence` (T-40).

```text
ERROR tests/test_inference.py - NameError: name 'Any' is not defined
Interrupted: 1 error during collection
```

Não é empacotamento. O teste LLM usa `Any` sem importar. Item 3 da T-8
continua o resultado Gemini (`16 passed`).

Entrypoint isolado:

```text
final_mse=0.000000000000
exit=0
```

### Item 4 — T-42 confirmada

| Árvore | `src/*.py` vs ZIP | TOTAL (oficial) | TOTAL (sem `test_inference`) |
|---|---|---:|---:|
| Reviewer (`output/.../src/`) | idêntico | 0% (coleta abortou) | 45% |
| ZIP (`/tmp/.../src/`) | idêntico | 0% (coleta abortou) | 45% |

Tabela diagnóstica (mesma nos dois lados):

```text
src/evaluation.py            15      0   100%
src/feature_engineering.py   16      0   100%
src/inference.py             14     14     0%
src/main.py                  39     39     0%
src/training.py              12      0   100%
TOTAL                        96     53    45%
```

O desvio 51% ≠ 54% do Gemini era FE pré-correção no Reviewer e FE
pós-correção no ZIP. Aqui o exportador recebeu os módulos validados:
não há segunda árvore.

## Reviewer

| Campo | Valor |
|---|---|
| `review_enabled` | true |
| `review_incomplete` | true |
| `has_errors` | true |
| `test_coverage` | 0.0 |
| `review_iterations` | 1 |
| `review_error` | Reviewer call budget exceeded |

A mensagem **não** é mais `"Reviewer did not process every affected stage"`.
Com 1 chamada (T-27) e erros em mais de um estágio, o orçamento estoura e
essa causa é a que fica no log — critério da T-42.

`execution.log`: modelo, durações e proveniência `llm` dos quatro estágios;
`review_enabled=true`; Reviewer 1 iteração, incompleto.

## Tentativa extra (`review_max_llm_calls=4`)

Mesmo notebook e modelo, `output/experiments/t8_fixture_groq20b_t42_review4/`.
O Reviewer reescreveu `feature_engineering.py` com `from __future__` fora do
topo (SyntaxError) e falhou em `evaluation` com **413** (25955 tokens vs
TPM 8000 do `gpt-oss-20b`). `review_incomplete=true` com o motivo do 413.
Não fecha a T-5; reforça a T-33 se o `20b` entrar no Reviewer com payload
grande.

## Conclusão

- **T-42:** ZIP = módulos pós-Reviewer. Item 4 deixa de ser dessincronia de
  árvore.
- **T-8 item 3:** não verde nesta rodada Groq (teste LLM). Permanece o Gemini.
- **T-5:** aberta. Com o padrão de 1 chamada o fluxo não conclui quando há
  erros em vários estágios.

Nenhum código do sistema foi alterado nesta execução.
