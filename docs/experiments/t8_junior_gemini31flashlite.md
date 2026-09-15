# T-8 — Reexecução no notebook júnior (Gemini)

Data: 2026-09-15
Notebook: `notebooks/junior_regression.ipynb` (somente o júnior)
Provedor: Gemini (endpoint OpenAI-compatível)
Reviewer: ligado
Artefato local (não versionado): `output/experiments/t8_junior_gemini31flashlite/`

## Escolha do modelo

O README e o `PLAN.md` citam `gemini-2.5-flash`. A lista da API ainda devolve
esse ID, mas o chat completion responde **404 para contas novas**:

```text
This model models/gemini-2.5-flash is no longer available to new users.
Please update your code to use models/gemini-3.6-flash
```

Smoke tests na mesma chave (sem imprimir credencial):

| Modelo | Smoke curto | Conversão completa |
|---|---|---|
| `gemini-2.5-flash` | 404 (conta nova) | não tentada |
| `gemini-3.6-flash` | OK | 503 *high demand* (análise ou code gen) |
| `gemini-3.8-flash` | OK, depois 503 | não chegou a converter |
| `gemini-3.5-flash` | OK, depois 503 | não chegou a converter |
| `gemini-3.7-flash` | timeout | — |
| `gemini-3.1-flash-lite` | OK | **concluiu** (~6 min nesta rodada) |

O retry do cliente só cobre HTTP 429, não 503. Modelo registrado nesta evidência:
**`gemini-3.1-flash-lite`**. Alias `gemini-flash-latest` existe, mas não é
reprodutível (T-25).

## Comando

```text
notebook_path = notebooks/junior_regression.ipynb
output_dir    = output/experiments/t8_junior_gemini31flashlite
use_llm       = True
llm_provider  = gemini
llm_model     = gemini-3.1-flash-lite
enable_review = True
```

## Itens de verificação

| Item | Resultado |
|---|---|
| Proveniência | Os quatro estágios `origin=llm`, `contract_ok=true`, parse `fenced`. Zero fallback. |
| Treino | OLS manual em `train_model`; sem sklearn. Coerente com o notebook júnior. |
| ZIP `src/main.py` | presente |
| Testes do ZIP | os quatro `test_*.py` inserem `src/` no `sys.path` |
| Instalação isolada do ZIP | **falha** com `random` no `requirements.txt` (stdlib; mesmo defeito da rodada Groq) |
| Suíte do ZIP (sem a linha `random`) | **22 passed**, cobertura **60%** (`main.py` 0/39; estágios 100%) |
| Entrypoint | `final_mse` impresso |
| Equivalência | **`equivalente`** (`original=0.011202345146`, `generated≈0.0096`, tolerância 0.05) |
| Cobertura Reviewer vs ZIP | **não iguais:** Reviewer 0% / ZIP 60% |
| T-5 | Reviewer 1 iteração, `review_incomplete=true` |

### Reviewer vs ZIP

O ZIP exporta os módulos **em memória** (pré-correção). O Reviewer reescreveu
`src/feature_engineering.py` na árvore de revisão com um script que chama
`ruff`/`pytest` via `subprocess` — sem `load_data`. A suíte do Reviewer quebra
na coleta (`ImportError: cannot import name 'load_data'`), cobertura 0%.

O ZIP continua com o FE gerado (dados sintéticos + `split_data` no índice 21).
É a mesma classe de problema da T-29/T-35: o artefato medido na revisão não é
o artefato entregue.

`load_data` do ZIP usa `random.uniform` **sem** `random.seed(2026)`. Por isso
duas execuções do `main.py` não batem o MSE bit a bit; a equivalência ainda
passa na tolerância de 0.05.

## Falhas remanescentes

1. `gemini-2.5-flash` documentado está morto para contas novas; atualizar o ID
   no README/`PLAN.md` (candidato: `gemini-3.6-flash` quando a cota responder,
   ou registrar `gemini-3.1-flash-lite` como o que de fato rodou hoje).
2. HTTP 503 não entra no retry (só 429).
3. Stdlib `random` no `requirements.txt` do ZIP.
4. Reviewer incompleto e destrutivo nesta execução; T-5 aberta.
5. Equivalência não é bit-a-bit: o pipeline gerado não replica o seed.

Nenhum código do sistema foi alterado nesta execução.
