# Fase 6 — Comparação júnior vs sênior e números finais

Data: 2026-09-17
Protocolo: `docs/experiments/protocol_matriz_custo_zero_2026-09-16.md`
CSVs: `docs/experiments/matriz_custo_zero_junior.csv`,
`docs/experiments/matriz_custo_zero_senior.csv`
Commit das métricas do software: `6e91cf2`
Reviewer: desligado (`enable_review=False`)
Repetições: 1

Estudo de viabilidade sob custo zero. Equivalência de `final_mse` (tolerância
0,05) **não foi aplicada**: nenhum pipeline gerado imprimiu a métrica.

## O que se manteve

- Gemini `gemini-3.8-flash`: HTTP 503 *high demand* nos dois notebooks. Sem
  artefato. Sem troca de modelo.
- Granite: inviável nesta GTX 1650 4 GiB. Júnior `Broken pipe`; sênior nem
  executado.
- Groq `openai/gpt-oss-20b`: único ID que **concluiu** conversão nos dois
  notebooks. Contrato `nao_conforme` nos dois (`prepare_features` com 1
  argumento em vez de 2). Suíte inválida nos dois (dependência alucinada).
  Equivalência `nao_executavel`. Reviewer desativada.
- `final_mse` original dos notebooks: `0.011202345146` nos dois.
- Qualidade do `mlops-agent` continua fora desta tabela.

## O que mudou

- **Qwen local.** Júnior concluiu em 161,8 s (4 estágios `llm`, testes
  gerados). Sênior gerou os 4 módulos em ~90 s e travou na geração de
  testes: `timeout after 900s`, máquina presa por horas. O notebook mais
  estruturado **piorou** a viabilidade local.
- **Groq — preservação da lógica.** No júnior, `load_data` e o OLS
  acompanharam o notebook (seed 2026, 30 pontos, `2.75x+1.25`, split 21,
  stdlib). No sênior, o mesmo modelo leu CSV inexistente com pandas/sklearn
  e o estágio `training` caiu para **template** (`invalid syntax`). O
  notebook sênior não ajudou a preservar a lógica; puxou um pipeline
  genérico.
- **Fallback.** Júnior: 0 estágios template nas gerações que fecharam.
  Sênior: 1 fallback (`training` Groq).
- **Dependência alucinada.** Júnior Groq: `joblib`. Sênior Groq: `sklearn`.

## Tabela única (notebook × modelo × repetição)

| Notebook | Modelo | Provedor | Rep. | Duração (s) | Primário (4 módulos + testes) | Estágios LLM | Fallback | Contrato | Equivalência | Classe da falha |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| júnior | `qwen2.5-coder:3b` | ollama | 1 | 161,8 | sim | 4 | 0 | `nao_conforme` | `nao_executavel` | código gerado |
| júnior | `granite-code:3b` | ollama | 1 | 164,0 | não | 0 | 0 | — | — | recurso local |
| júnior | `gemini-3.8-flash` | gemini | 1 | 14,3 | não | 0 | 0 | — | — | provedor gratuito |
| júnior | `openai/gpt-oss-20b` | groq | 1 | 109,1 | sim | 4 | 0 | `nao_conforme` | `nao_executavel` | código gerado |
| sênior | `qwen2.5-coder:3b` | ollama | 1 | 900 | não (timeout; módulos parciais, sem testes) | 4* | 0 | — | — | recurso local |
| sênior | `granite-code:3b` | ollama | 1 | — | não (não executado) | 0 | 0 | — | — | recurso local |
| sênior | `gemini-3.8-flash` | gemini | 1 | 24,3 | não | 0 | 0 | — | — | provedor gratuito |
| sênior | `openai/gpt-oss-20b` | groq | 1 | 131,0 | sim | 3 | 1 (`training`) | `nao_conforme` | `nao_executavel` | código gerado |

\* Qwen sênior chegou a emitir 4 estágios `llm` no log; o harness não
aceitou a combinação (timeout, sem testes/ZIP).

## Totais

| Indicador | Valor |
| --- | ---: |
| Combinações planejadas | 8 |
| Linhas no CSV | 8 |
| Conversões com critério primário (4 módulos + testes) | 3 |
| Estágios `llm` nessas 3 conversões | 11 |
| Estágios em fallback (template) | 1 |
| Pipelines executáveis / `final_mse` gerado | 0 |
| Equivalências dentro da tolerância 0,05 | 0 |

### Distribuição das falhas

| Classe | n | Combinações |
| --- | ---: | --- |
| Código gerado (contrato, suíte, dependência) | 3 | júnior Qwen; júnior Groq; sênior Groq |
| Recurso local (VRAM / hang / timeout) | 3 | júnior Granite; sênior Qwen; sênior Granite |
| Provedor gratuito (503) | 2 | júnior Gemini; sênior Gemini |

OpenRouter (`llama-3.2-3b-instruct:free`, HTTP 404) saiu **antes** da matriz,
no protocolo. Não entra nestes 8.

## Exemplos representativos

### Modularização que preserva o notebook (júnior × Groq)

```python
raw_x = [index / 10.0 for index in range(30)]
raw_y = [
    2.75 * value + 1.25 + random.uniform(-0.15, 0.15)
    for value in raw_x
]
```

OLS fechado em `train_model`, seed 2026, split 21. Melhor evidência de que
a geração sob custo zero **pode** modularizar a lógica do júnior.

### Limite no sênior × Groq (mesmo modelo)

`load_data` ignora a geração in-process do notebook e lê CSV + sklearn:

```python
from sklearn.model_selection import train_test_split
csv_path = Path(path) if path is not None else Path(DATA_PATH)
df = pd.read_csv(csv_path)
```

`training` foi para template após sintaxe inválida:

```python
def train_model(features: Any, labels: Any) -> Any:
    """Train model placeholder using provided data."""
    model = {"features": features, "labels": labels}
    return model
```

### Limite de preservação (júnior × Qwen)

Trocou seed, tamanho e equação (`SEED=42`, `N=100`, `y = 2x + ruído`) e
empacotou a avaliação numa classe, quebrando o contrato.

## Empacotamento do artefato exportado

Nas três conversões que fecharam, o ZIP / árvore exportada contém
`README.md`, `pyproject.toml`, `requirements.txt` e `src/main.py`. Exemplo
júnior Groq: dependências `joblib`, `numpy`. Exemplo sênior Groq:
`joblib`, `numpy`, `pandas`, `scikit-learn` (e a linha inválida
`collections.abc` no `requirements.txt`).

O exportador **não** cria repositório Git, `.gitignore` nem workflow de
CI/CD. Versionamento e integração contínua ficam fora do artefato. Isso é
limitação do protótipo, não falha de um modelo.

## Qualidade do `mlops-agent` (tabela separada)

Procedimento: `uv run pytest` e `uv run pre-commit run --all-files` em
2026-09-16, commit `6e91cf2`. Relatórios brutos não arquivados.

| Item | Valor |
| --- | --- |
| Testes | 165 aprovados, 2 pulados, 0 falhas |
| Cobertura | 88,50% (limite 80%) |
| Pre-commit | passou (ruff, mypy, yaml, eof, whitespace, large files) |
| Python | 3.11.15 |

Não comparar estes números com cobertura/lint dos artefatos gerados.

## Alcance da conclusão (lembrete da Fase 0)

O estudo descreve viabilidade de **gerar** código modular a custo zero, com
os IDs que responderam. Não afirma equivalência funcional, redução de
esforço nem prontidão para produção. APIs pagas e mais de uma repetição
ficam como trabalho futuro, sob o mesmo protocolo.
