# T-8 — Reexecução da fixture (Gemini 3.6 Flash, pós T-40)

Data: 2026-09-16
Notebook: `tests/fixtures/simple_regression.ipynb`
Provedor: Gemini (endpoint OpenAI-compatível)
Modelo: `gemini-3.6-flash`
Reviewer: ligado (`enable_review=true`)
Duração da conversão: 154,6 s
Artefato local (não versionado): `output/experiments/t8_fixture_gemini36flash/`
T-40 nesta branch: `6d2efa0`

Compara com `docs/experiments/t8_fixture_groq20b_t40.md` (`gpt-oss-20b`).

## Escolha do modelo

Smoke na mesma chave (sem imprimir credencial):

| Modelo | Smoke curto | Conversão completa |
|---|---|---|
| `gemini-2.5-flash` | 404 (conta nova) | não tentada |
| `gemini-3.1-flash-lite` | OK, devolve código | não usada nesta rodada |
| `gemini-3.5-flash` | timeout | não tentada |
| `gemini-3.8-flash` | OK, conteúdo truncado | não tentada |
| `gemini-3.6-flash` | OK (às vezes 503) | **concluiu** (~155 s) |

A API ainda pede `gemini-3.6-flash` no lugar de `gemini-2.5-flash`. 503 de
demanda voltou num smoke intermediário; a conversão desta rodada passou.

## Comando executado

Chamada direta a `convert_notebook` (o harness da matriz recusa fixtures):

```text
notebook_path = tests/fixtures/simple_regression.ipynb
output_dir    = output/experiments/t8_fixture_gemini36flash
use_llm       = True
llm_provider  = gemini
llm_model     = gemini-3.6-flash
enable_review = True
```

## Itens de verificação

| Item | Groq `gpt-oss-20b` (pós T-40) | Gemini `gemini-3.6-flash` |
|---|---|---|
| 1. `LinearRegression().fit` | Sim (training LLM; FE template) | **Sim.** `LinearRegression()` + `model.fit(...)` |
| 2. LLM ativa | 3/4 `llm`; FE template | **4/4 `llm`**, parse `fenced`. FE `contract_ok=false` (assinatura) |
| 3. Suíte ZIP isolada | não instalou (`sklearn.linear_model`) | **16 passed.** `joblib`, `numpy`, `pandas`, `scikit-learn` |
| 4. Cobertura Reviewer vs ZIP | 62% = 62% (após tirar dep inválida) | **51% ≠ 54%** |
| 5. Fallback malformado | teste unitário permanece | não reexecutado; teste unitário permanece |
| T-5 | `review_incomplete=true` | **Não.** 1 iteração, mesma mensagem |
| Entrypoint | falhou no fit (FE template vazio) | **`final_mse=0.000000000000`, exit 0** |

### Item 3 — ZIP em `/tmp/mlops-t8-fixture-gemini36`

Instalação isolada ok. Sem nomes pontilhados. Sem stdlib.

```text
16 passed
cobertura: 54%  (src/main.py 0/39; estágios 45/45)
```

O `gpt-oss-20b` listava `sklearn.linear_model` no `requirements.txt`. O
`gemini-3.6-flash` não. Confirma a leitura de que isso era qualidade da LLM,
não empacotamento.

### Item 4 — por que 51% ≠ 54%

A árvore que o Reviewer mediu ainda tem o `feature_engineering` original da
LLM (`load_data` devolve `DataFrame`; `prepare_features` com 1 argumento).
Nessa árvore: **6 failed, 10 passed**, cobertura **51%**.

O Reviewer reescreveu o FE para o contrato (`load_data` → `(X, y)`;
`clean_data`/`prepare_features` com features e labels). O ZIP exporta essa
versão: **16 passed**, cobertura **54%**.

É a mesma classe de dessincronia Reviewer vs artefato já vista no júnior
Gemini (`t8_junior_gemini31flashlite.md`). O item 3 mede o ZIP; o item 4
compara dois snapshots diferentes.

### Entrypoint

```text
score=1.0
final_mse=0.000000000000
exit=0
```

`src/main.py` importa `Sequence` (T-40). Com o FE reescrito, `load_data`
devolve duas partes e `split_data` devolve quatro (tuple neste draw).

## Reviewer

| Campo | Valor |
|---|---|
| `review_enabled` | true |
| `review_incomplete` | true |
| `has_errors` | true |
| `test_coverage` | 51.0 |
| `review_iterations` | 1 |
| `review_error` | Reviewer did not process every affected stage |

`lint_errors` / `type_errors` não foram impressos nesta rodada (o script de
coleta quebrou num atributo inexistente depois da conversão).

## Conclusão

- **Item 3 da T-8:** verde nesta rodada Gemini. A suíte isolada passa.
- **Item 4:** ainda aberto (51% ≠ 54%) porque o Reviewer mede o FE pré-correção
  e o ZIP leva o FE pós-correção.
- **T-5:** aberta.
- Nomes pontilhados no artefato **não** reapareceram com este modelo.

Nenhum código do sistema foi alterado nesta execução.
