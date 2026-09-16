# T-8 — Reexecução da fixture pós T-39 (Groq)

Data: 2026-09-16
Notebook: `tests/fixtures/simple_regression.ipynb`
Provedor: Groq
Modelo: `openai/gpt-oss-20b`
Reviewer: ligado (`enable_review=true`)
Duração da conversão: 93,2 s
Artefato local (não versionado): `output/experiments/t8_fixture_groq20b_t39/`
Commit da T-39 nesta branch: `41ab708`

Compara com `docs/experiments/t8_fixture_groq20b.md` (pré T-39).

## Comando executado

```text
notebook_path = tests/fixtures/simple_regression.ipynb
output_dir    = output/experiments/t8_fixture_groq20b_t39
use_llm       = True
llm_provider  = groq
llm_model     = openai/gpt-oss-20b
enable_review = True
```

## A T-39 resolveu o que se propôs?

**Sim.** O `test_training.py` do ZIP é o template novo (`features = [[1.0], [2.0]]`)
e **passou** contra `LinearRegression().fit`. A falha `Expected 2D array, got 1D`
da execução anterior **não voltou**.

A suíte isolada **ainda não ficou verde**, por outro motivo: asserts frágeis da
LLM em `test_feature_engineering.py` (tamanho do split num dataset de 3 linhas).

## Itens de verificação

| Item | Pré T-39 | Pós T-39 |
|---|---|---|
| 1. `LinearRegression().fit` | sim | **Sim.** `model = LinearRegression()` + `model.fit(X, y)` |
| 2. 4 estágios `llm` | 4/4 | **4/4**, parse `fenced`, `contract_ok` |
| 3. Suíte ZIP isolada | 1 failed, 12 passed (template 1-D) | **3 failed, 17 passed.** `test_training` passou. Falhas: split LLM |
| 4. Cobertura Reviewer vs ZIP | 48% = 48% | **55% = 55%** (`main.py` 0/39 nas duas) |
| 5. Fallback malformado | passed | não reexecutado; teste unitário permanece |
| T-5 Reviewer concluído | `review_incomplete` | **Não.** mesma mensagem, 1 iteração |
| T-38 instalação isolada | ok | **ok** (`joblib`, `numpy`, `pandas`, `scikit-learn`; sem `random`) |

### Item 3 — ZIP em `/tmp/mlops-t8-fixture-t39`

`test_training.py` usa o template T-39 e passa.

As 3 falhas são o teste LLM
`test_split_data_splits_properly_and_is_deterministic[0.2|0.33|0.5]`:
espera `int(n * test_size)` linhas no teste. O notebook tem 3 linhas
(`x=[1,2,3]`). O `train_test_split` do sklearn arredonda para cima
(`0.2` → 1 linha, não 0). É a mesma classe de assert frágil já vista em
2026-09-14, não regressão da T-39.

Estágios (exceto `main.py`) estão em 100% de cobertura.

### Entrypoint

```text
RuntimeError: split_data must return four values
exit=1
```

`split_data` devolve o retorno de `sklearn.model_selection.train_test_split`,
que é uma **lista** de 4 arrays. O `main.py` exige `isinstance(split, tuple)`.
Não faz parte do item 3 da T-8; fica registrado para não silenciar.

## Reviewer

| Campo | Valor |
|---|---|
| `review_enabled` | true |
| `review_incomplete` | true |
| `lint_errors` | 0 |
| `type_errors` | 21 |
| `test_coverage` | 55.0 |
| `review_iterations` | 1 |

## Conclusão

- **T-39:** fechou o defeito estrutural do template vs sklearn.
- **T-8 item 3:** ainda aberto (asserts da LLM, não o template).
- **T-5:** ainda aberta (`review_incomplete=true`).

Nenhum código do sistema foi alterado nesta execução.
