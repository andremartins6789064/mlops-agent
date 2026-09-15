# T-8 — Reexecução no notebook júnior (OpenRouter / Laguna S 2.1)

Data: 2026-09-15
Notebook: `notebooks/junior_regression.ipynb` (somente o júnior)
Provedor: OpenRouter
Modelo: `poolside/laguna-s-2.1`
Reviewer: ligado
Duração aproximada: 135 s
Artefato local (não versionado): `output/experiments/t8_junior_openrouter_laguna_s21/`

Mesmo patch local da rodada Qwen (não commitado): `max_tokens=8192` e
`reasoning.effort=none`. Sem isso o OpenRouter recusa com 402 nesta cota.

Há variante gratuita `poolside/laguna-s-2.1:free`; esta execução usou o ID pago.

## Comando (equivalente, com o patch)

```text
notebook_path = notebooks/junior_regression.ipynb
output_dir    = output/experiments/t8_junior_openrouter_laguna_s21
use_llm       = True
llm_provider  = openrouter
llm_model     = poolside/laguna-s-2.1
enable_review = True
```

## Itens de verificação

| Item | Resultado |
|---|---|
| Smoke | `OK.` |
| Proveniência | 4/4 `origin=llm`, `fenced`, `contract_ok=true` |
| ZIP `src/main.py` | presente |
| Instalação isolada | **falha** com `random` no `requirements.txt`; também declara `joblib` e `numpy` |
| Suíte do ZIP (sem `random`) | **5 failed, 36 passed**, cobertura **65%** (`main.py` 0/39) |
| Entrypoint | **quebra:** `TypeError: unsupported operand type(s) for +: 'int' and 'list'` em `train_model` — `prepare_features` devolve matrizes (`[[x]]`) e o treino espera `list[float]` |
| Equivalência | `nao_executavel` (mesmo TypeError) |
| Cobertura Reviewer vs ZIP | **0% vs 65%** — Reviewer substituiu `feature_engineering.py` por código de teste de `inference` (`joblib`/`pytest`); o teste gerado importa `clean_data` e a coleta quebra |
| T-5 | `review_incomplete=true`, 18 erros de tipo, 1 iteração |

Falhas da suíte do ZIP (asserts frágeis / contrato frouxo): MSE esperado 0.4375 vs 0.375; `load_data` sem seed (duas chamadas não batem); teste de `prepare_features` espera quadrado e o código só divide por 3; mensagem de erro de `load_model` diferente da esperada.

Nenhum código do sistema foi alterado nesta execução.
