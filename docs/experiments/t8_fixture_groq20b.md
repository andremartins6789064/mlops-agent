# T-8 — Reexecução da fixture (Groq, pós T-35 / T-38)

Data: 2026-09-16
Notebook: `tests/fixtures/simple_regression.ipynb`
Provedor: Groq
Modelo: `openai/gpt-oss-20b`
Reviewer: ligado (`enable_review=true`)
Duração da conversão: 114,8 s
Artefato local (não versionado): `output/experiments/t8_fixture_groq20b/`

Esta é a reexecução da T-8 **na fixture**, depois da T-35 (`src/main.py` na
árvore de revisão) e da T-38 (stdlib fora do artefato). Não usa o notebook
júnior.

## Comando executado

Chamada direta a `convert_notebook` (o harness da matriz recusa fixtures):

```text
notebook_path = tests/fixtures/simple_regression.ipynb
output_dir    = output/experiments/t8_fixture_groq20b
use_llm       = True
llm_provider  = groq
llm_model     = openai/gpt-oss-20b
enable_review = True
```

## Itens de verificação

| Item | Resultado |
|---|---|
| 1. `training.py` com `LinearRegression().fit(...)` | **Sim.** `origin=llm`, `contract_ok`. `model = LinearRegression()` seguido de `model.fit(train_features, train_labels)`. |
| 2. Conversão com LLM ativa | **Sim.** Quatro estágios `llm`, parse `fenced`, modelo `openai/gpt-oss-20b`. |
| 3. Suíte do ZIP em `/tmp`, isolada | **Não verde.** Instalação ok; 1 failed, 12 passed. Ver abaixo. |
| 4. Cobertura Reviewer vs ZIP | **Iguais: 48%.** T-35 confirmada na fixture: `src/main.py` nas duas árvores (39 statements, 0 cobertos). |
| 5. Resposta malformada → fallback | Confirmado pelo teste unitário `test_code_generator_records_template_provenance_for_malformed_response` (passed, `--no-cov`). |
| T-5. Reviewer concluído | **Não.** `review_incomplete=true`, `review_error="Reviewer did not process every affected stage"`, 1 iteração. |

### Item 3 — ZIP em `/tmp/mlops-t8-fixture`

Layout: `src/main.py` presente; os quatro `test_*.py` inserem `src/` no `sys.path`.

Instalação isolada **passou** (T-38): `requirements.txt` / `pyproject.toml` declaram
`joblib`, `numpy`, `pandas`, `scikit-learn`. Sem `random`. `uv run --no-project
--with-requirements requirements.txt` instalou 17 pacotes.

```text
1 failed, 12 passed
cobertura: 48%  (src/main.py 0/39; evaluation 100%; inference 100%;
                 training 71%; feature_engineering 70%)
```

A falha é `tests/test_training.py::test_training_train_and_save_model`, o
**template** do Test Generator (`train_model([1], [1])`). O `training.py` gerado
é um `LinearRegression` real; o sklearn rejeita X 1-D:

```text
ValueError: Expected 2D array, got 1D array instead: array=[1].
```

Não é assert frágil da LLM nesta execução. O teste de training caiu no
placeholder da T-4, que não é compatível com um estimador sklearn.

O entrypoint do ZIP executa:

```text
final_mse=0.010420222653
exit=0
```

(A fixture original não imprime `final_mse`; equivalência da Etapa 9.0 não se
aplica a este insumo.)

### Item 4 — mesmas árvores

| Árvore | `src/main.py` | TOTAL | Cover |
|---|---|---:|---:|
| Reviewer (`output/.../src/`) | sim (39 stmts) | 97 | 48% |
| ZIP (`/tmp/mlops-t8-fixture/src/`) | sim (39 stmts) | 97 | 48% |

Mesmos arquivos, mesmos missing lines.

## Reviewer

| Campo | Valor |
|---|---|
| `review_enabled` | true |
| `review_incomplete` | true |
| `lint_errors` | 0 |
| `type_errors` | 17 |
| `test_coverage` | 48.0 |
| `review_iterations` | 1 |

## Proveniência (`execution.log`)

| Estágio | origin | duração | parse |
|---|---|---:|---|
| feature_engineering | llm | 2,24 s | fenced |
| training | llm | 1,09 s | fenced |
| inference | llm | 2,16 s | fenced |
| evaluation | llm | 22,32 s | fenced |

`contract_ok=true` nos quatro. Modelo registrado: `openai/gpt-oss-20b`.

## Falhas remanescentes (não silenciadas)

1. **Template de teste de training vs sklearn (T-39).** `train_model([1], [1])`
   quebra `LinearRegression().fit`. Impede a suíte verde do item 3 quando o
   código gerado é o esperado pela Etapa 8.5.
2. **Reviewer incompleto.** T-5 permanece aberta.

T-38 confirmada empiricamente nesta execução: o ZIP isolado instalou sem
módulo da stdlib. Nenhum código do sistema foi alterado nesta execução.
