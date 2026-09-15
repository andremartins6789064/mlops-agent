# T-8 — Reexecução no notebook júnior (Groq)

Data: 2026-09-15
Notebook: `notebooks/junior_regression.ipynb` (somente o júnior; o sênior não rodou)
Provedor: Groq
Modelo: `openai/gpt-oss-20b`
Reviewer: ligado (`enable_review=true`)
Duração aproximada da conversão: 151 s
Artefato local (não versionado): `output/experiments/t8_junior_groq20b/`

Esta execução **não substitui** a T-8 da fixture `tests/fixtures/simple_regression.ipynb`.
O júnior é stdlib puro e não usa `sklearn.LinearRegression`.

## Comando executado

Chamada direta a `convert_notebook` (o harness da matriz recusa fixtures; o júnior
é insumo válido, mas a T-8 histórica não passa pela matriz):

```text
notebook_path = notebooks/junior_regression.ipynb
output_dir    = output/experiments/t8_junior_groq20b
use_llm       = True
llm_provider  = groq
llm_model     = openai/gpt-oss-20b
enable_review = True
```

## Itens de verificação

| Item | Resultado |
|---|---|
| 1. Lógica de treino migrada | Parcial. `training.py` veio da LLM (`origin=llm`, `contract_ok`) com OLS manual e dataclass `LinearRegressionModel`. Não há `sklearn.LinearRegression().fit(...)` — o notebook júnior não usa sklearn. |
| 2. Proveniência dos 4 estágios | `training`/`inference`/`evaluation`: `llm`. `feature_engineering`: **template** (Python inválido: *unterminated triple-quoted string literal*, linha 97). Modelo e durações no `execution.log`. |
| 3. Suíte do ZIP em `/tmp`, isolada | **Não verde.** Ver abaixo. |
| 4. Cobertura Reviewer vs ZIP | **Iguais: 72%.** T-35 confirmada neste artefato: `src/main.py` existe nas duas árvores (39 statements, 0 cobertos pelos testes). Estágios 100%. |
| 5. Resposta malformada → fallback | Confirmado pelo teste unitário `test_code_generator_records_template_provenance_for_malformed_response` (passed). |
| T-5. Reviewer concluído | **Não.** `review_incomplete=true`, `review_error="Reviewer did not process every affected stage"`, 1 iteração. |

### Item 3 — ZIP em `/tmp/mlops-t8-junior`

Layout: `src/main.py` presente; os quatro `test_*.py` inserem `src/` no `sys.path`.

Instalação isolada **falhou** com o `requirements.txt` exportado:

```text
random was not found in the package registry
```

O notebook importa `random` (stdlib). Esse nome foi parar em
`[project].dependencies` e em `requirements.txt`. Não existe pacote `random` no
PyPI. A árvore do Reviewer **não** tinha essa linha: `validate_output` reescreve
os metadados a partir dos `.py` gerados, onde a stdlib já é filtrada.

Diagnóstico (removendo só a linha `random` do ZIP descompactado):

```text
1 failed, 20 passed
cobertura: 72%  (src/main.py 0/39; estágios 98/98)
```

A falha restante é assert frágil da LLM em
`test_evaluation.py::TestEvaluateModel::test_mse_positive_with_misalignments`:
o teste espera MSE `0.333…` e o código correto devolve `0.166…`
(`0.25 + 0.25 + 0) / 3`).

`pyproject.toml` do ZIP ainda declara `joblib`, `numpy` e `pandas`, que o
notebook júnior não usa. `inference.py` gerado importa `joblib`.

### Entrypoint e equivalência

```text
status           : nao_executavel
original_metric  : 0.011202345146
generated_metric : null
error            : Pipeline exited with status 1.
```

Causa: `feature_engineering` caiu no template (`load_data()` devolve `([], [])`);
`train_model` quebra com `ZeroDivisionError: division by zero`.

## Reviewer

| Campo | Valor |
|---|---|
| `review_enabled` | true |
| `review_incomplete` | true |
| `lint_errors` | 0 |
| `type_errors` | 8 |
| `test_coverage` | 72.0 |
| `review_iterations` | 1 |

## Falhas remanescentes (não silenciadas)

1. **Stdlib no artefato.** `random` em `requirements.txt` / `pyproject.toml` impede
   a suíte isolada do ZIP. Candidata a tarefa nova (filtro de stdlib na exportação,
   reusando o filtro que `extract_imported_libraries` já aplica ao código gerado).
2. **Fallback em `feature_engineering`.** String tripla não encerrada; o pipeline
   não executa. Sem isso a equivalência continua `nao_executavel`.
3. **Assert frágil da LLM.** Mesma classe da reexecução T-8 de 2026-09-14.
4. **Reviewer incompleto.** T-5 permanece aberta.
5. **Dependências inventadas.** `joblib`/`numpy`/`pandas` no artefato de um
   notebook só com `random`.

Nenhum código do sistema foi alterado nesta execução.
