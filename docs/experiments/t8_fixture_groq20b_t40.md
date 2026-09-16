# T-8 — Reexecução da fixture pós T-40 (Groq)

Data: 2026-09-16
Notebook: `tests/fixtures/simple_regression.ipynb`
Provedor: Groq
Modelo: `openai/gpt-oss-20b`
Reviewer: ligado (`enable_review=true`)
Duração da conversão: 105,6 s
Artefato local (não versionado): `output/experiments/t8_fixture_groq20b_t40/`
T-40 nesta branch: `6d2efa0` (junto com o registro T-39)

Compara com `docs/experiments/t8_fixture_groq20b_t39.md`.

## Comando executado

```text
notebook_path = tests/fixtures/simple_regression.ipynb
output_dir    = output/experiments/t8_fixture_groq20b_t40
use_llm       = True
llm_provider  = groq
llm_model     = openai/gpt-oss-20b
enable_review = True
```

## A T-40 resolveu o que se propôs?

**O código entrou no artefato.** O `src/main.py` do ZIP importa `Sequence` e não
usa mais `isinstance(split, tuple)`.

**Esta execução não exercitou o caso sklearn.** `feature_engineering` caiu em
**template** (string tripla não encerrada na LLM). O `split_data` placeholder
devolve **tuple** de listas vazias, não o `list` do `train_test_split`. O
entrypoint passou da checagem das quatro partes e falhou depois em
`train_model` (`Expected 2D array ... array=[]`).

A confirmação empírica do caso list (artefato T-39, só trocando o `main.py`)
permanece a da T-40: `final_mse=0.000000000000`, exit 0.

## Itens de verificação

| Item | Resultado |
|---|---|
| 1. `LinearRegression().fit` | **Sim** em `training.py` (`origin=llm`). FE é template. |
| 2. LLM ativa | 3/4 `llm`; FE **template** |
| 3. Suíte ZIP isolada | **Não instalou.** `sklearn.linear_model` em `requirements.txt` não existe no PyPI. |
| 4. Cobertura Reviewer vs ZIP | Reviewer **62%**. ZIP não mediu até remover a linha inválida; depois **62% = 62%** (`main.py` 0/39). |
| 5. Fallback malformado | não reexecutado; teste unitário permanece |
| T-5 | `review_incomplete=true`, 1 iteração |

### Item 3 — instalação

```text
sklearn.linear_model was not found in the package registry
```

`scikit-learn` também está declarado (mapeamento T-6/T-31). O nome pontilhado
veio da lista `libraries` da análise LLM. O Analyzer heurístico já reduz a
`sklearn`; o `gpt-oss-20b` é que emitiu o submódulo. **Não vira tarefa de
código** — esperar modelo mais capaz.

Diagnóstico (removendo só `sklearn.linear_model`):

```text
4 failed, 18 passed
cobertura: 62%
```

Falhas: asserts LLM em evaluation / inference / training (coeficientes). Não é
o template 1-D da T-39.

### Entrypoint (após remover a dep inválida)

```text
ValueError: Expected 2D array, got 1D array instead: array=[].
exit=1
```

Causa: template de FE (`load_data()` → `[], []`), não a checagem tuple/list.

## Reviewer

| Campo | Valor |
|---|---|
| `review_enabled` | true |
| `review_incomplete` | true |
| `lint_errors` | 0 |
| `type_errors` | 21 |
| `test_coverage` | 62.0 |
| `review_iterations` | 1 |

## Conclusão

- **T-40:** presente no ZIP; caso `train_test_split`→list não apareceu nesta
  rodada (FE template).
- **T-8 item 3:** ainda aberto; instalação falhou por saída inválida da LLM
  (`sklearn.linear_model`) + asserts frágeis. Sem tarefa nova de empacotamento.
- **T-5:** aberta.

Nenhum código do sistema foi alterado nesta execução.
