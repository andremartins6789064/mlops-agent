# T-8 — Reexecução no notebook júnior (OpenRouter / Qwen)

Data: 2026-09-15
Notebook: `notebooks/junior_regression.ipynb` (somente o júnior)
Provedor: OpenRouter (`https://openrouter.ai/api/v1`)
Modelo: `qwen/qwen3.8-27b`
Reviewer: ligado
Duração aproximada da conversão bem-sucedida: 201 s
Artefato local (não versionado): `output/experiments/t8_junior_openrouter_qwen38_27b/`

## Bloqueios operacionais (cliente do produto)

O slug `qwen/qwen3.8-27b` existe e a chave `OPENROUTER_API_KEY` está configurada.
A primeira chamada com o cliente de produção **falhou com HTTP 402**: o
OpenRouter assume `max_tokens` enorme (32k–131k) quando o campo não é enviado,
e o saldo da conta só cobria ~12k–20k tokens.

Com `max_tokens=4096` o smoke passou (`OK`), mas a análise do notebook
devolveu conteúdo vazio: o modelo gastou o orçamento em *reasoning* (Qwen 3.8
pensa por padrão). A documentação do OpenRouter descreve exatamente esse caso
(`finish_reason=length`, `content` vazio).

A conversão abaixo **não** é o caminho vanilla. Rodou com um patch local,
não commitado, no cliente:

- `max_tokens=8192`
- `reasoning.effort=none` / `enabled=false`

Sem isso, `convert_notebook` não completa nesta conta.

## Comando (equivalente, com o patch)

```text
notebook_path = notebooks/junior_regression.ipynb
output_dir    = output/experiments/t8_junior_openrouter_qwen38_27b
use_llm       = True
llm_provider  = openrouter
llm_model     = qwen/qwen3.8-27b
enable_review = True
```

## Itens de verificação

| Item | Resultado |
|---|---|
| Proveniência | 4/4 `origin=llm`, `parse_method=fenced`, `contract_ok=true` |
| Treino | dataclass `LinearRegressionModel` + OLS; `train_model` existe no ZIP |
| ZIP `src/main.py` | presente |
| Testes do ZIP | bootstrap `sys.path` para `src/` |
| Instalação isolada | **falha** com `random` no `requirements.txt` (stdlib) |
| Suíte do ZIP (sem `random`, com `numpy`) | **54 passed**, cobertura **68%** (`main.py` 0/39; estágios 100%) |
| Entrypoint | **quebra:** `TypeError: 'LinearRegressionModel' object is not subscriptable` — `train_model` devolve dataclass, `predict` trata o modelo como `dict` |
| Equivalência | `nao_executavel` (mesmo TypeError) |
| Cobertura Reviewer vs ZIP | **62% vs 68%** — Reviewer reescreveu `feature_engineering.py` com um dump JSON de lint (` ```json `), inválido como Python |
| T-5 | `review_incomplete=true`, 1 iteração, `type_errors=1` |

## Falhas remanescentes

1. O cliente LLM não envia `max_tokens`; no OpenRouter isso vira 402 nesta cota.
2. Modelos com reasoning (Qwen 3.8) esvaziam `content` se o teto for baixo;
   precisa desligar thinking ou orçar tokens de raciocínio.
3. Stdlib `random` no artefato (mesmo defeito Groq/Gemini).
4. Contrato de forma (`train_model` / `predict` existem) não impede
   incompatibilidade de tipo do objeto modelo.
5. Reviewer destrutivo: JSON no lugar de `feature_engineering.py`.
6. T-5 aberta.

Nenhum código do sistema foi alterado nesta execução (só o runner temporário
em `/tmp`).
