# T-8 — Reexecução no notebook júnior (OpenRouter / Ministral 3B)

Data: 2026-09-15
Notebook: `notebooks/junior_regression.ipynb` (somente o júnior)
Provedor: OpenRouter
Modelo: `mistralai/ministral-3b-2512`
Reviewer: ligado
Duração aproximada: 84 s
Artefato local (não versionado): `output/experiments/t8_junior_openrouter_ministral3b/`

Mesmo patch local: `max_tokens=8192` e `reasoning.effort=none`. Smoke: `OK`.

## Comando (equivalente, com o patch)

```text
notebook_path = notebooks/junior_regression.ipynb
output_dir    = output/experiments/t8_junior_openrouter_ministral3b
use_llm       = True
llm_provider  = openrouter
llm_model     = mistralai/ministral-3b-2512
enable_review = True
```

## Itens de verificação

| Item | Resultado |
|---|---|
| Proveniência | **4/4 `llm`**, `fenced`, `contract_ok` |
| FE no ZIP | gera `raw_x`/`raw_y` do notebook, mas `load_data` devolve pares `(x, y)` em vez de `(features, labels)` |
| Treino | OLS em `dict` com `coefficient`/`intercept` |
| Inferência | espera `dict`, mas usa `test_features.shape` (NumPy) numa lista |
| Avaliação | anota `np.ndarray` **sem** `import numpy` |
| Instalação isolada | **falha:** `random` no `requirements.txt` |
| Suíte do ZIP (sem `random`) | **não coleta:** `NameError: name 'np' is not defined` em `evaluation.py` |
| Entrypoint | mesmo `NameError` no import |
| Equivalência | `nao_executavel` |
| Reviewer | cobertura 0%, `review_incomplete=true` |

Melhor que o Llama 3.2 3B (que nem gerou o FE). Pior que o Qwen 27B (suíte 54 passed): o Ministral gera os quatro estágios, mas o artefato não importa.

Nenhum código do sistema foi alterado nesta execução.
