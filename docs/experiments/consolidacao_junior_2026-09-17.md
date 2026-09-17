# Consolidação — notebook júnior (Fase 4)

Data: 2026-09-17
Notebook: `notebooks/junior_regression.ipynb`
`final_mse` original: `0.011202345146`
Reviewer: desligado
CSV: `docs/experiments/matriz_custo_zero_junior.csv`
Runs: `docs/experiments/runs_matriz_custo_zero_junior/`

Esta consolidação é a porta para a rodada sênior. Não mistura métricas do
`mlops-agent` (pytest/cobertura do repositório) com a qualidade dos artefatos.

## Tabela por modelo

| Modelo | Duração (s) | Primário | Origem dos 4 estágios | Fallback | Contrato | Equivalência | Classe da falha | Detalhe |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-coder:3b` | 161,8 | gerou | 4× `llm` | 0 | `nao_conforme` | `nao_executavel` | código gerado | `evaluate_model` ausente no módulo `evaluation`; suíte NameError; cobertura 33% |
| `granite-code:3b` | 164,0 | não gerou | — | — | — | — | recurso local | `Broken pipe`; só `execution_started`; travou a GTX 1650 |
| `gemini-3.8-flash` | 14,3 | não gerou | — | — | — | — | provedor gratuito | HTTP 503 *high demand* |
| `openai/gpt-oss-20b` | 109,1 | gerou | 4× `llm` | 0 | `nao_conforme` | `nao_executavel` | código gerado | `prepare_features` com 1 argumento; `import joblib` na suíte |

Nenhum pipeline júnior imprimiu `final_mse`. A tolerância 0,05 **não foi aplicada**.
Zero estágios em fallback para template nas duas gerações que concluíram.

Contagem: 4 execuções; 2 com módulos gerados por LLM (8 estágios `llm` no total);
2 sem artefato; 0 pipelines executáveis.

## O que o notebook pedia

O júnior gera 30 pontos com `random.seed(2026)`, `y = 2.75x + 1.25 + ruído`,
split em 21/9, escala `x / 3.0`, OLS em escopo global e imprime
`final_mse=0.011202345146`. Sem numpy, sklearn ou `joblib`.

## Exemplos de código

### Modularização com preservação da lógica (Groq)

`load_data` e o OLS de `train_model` acompanham o notebook: seed 2026, 30
pontos, coeficiente 2.75, intercepto 1.25, split 21, escala `/ 3.0`.

```python
# Groq — feature_engineering.load_data (trecho)
random.seed(SEED)  # SEED = 2026
raw_x = [index / 10.0 for index in range(30)]
raw_y = [
    2.75 * value + 1.25 + random.uniform(-0.15, 0.15)
    for value in raw_x
]
```

```python
# Groq — training.train_model (trecho)
coefficient = numerator / denominator
intercept = mean_y - coefficient * mean_x
return LinearRegressionModel(coefficient=coefficient, intercept=intercept, mse=mse)
```

Isso mostra que, sob custo zero, um modelo gratuito **pode** modularizar e
preservar a lógica. Não mostra pipeline executável: o contrato quebra em
`prepare_features(features)` (o contrato pede `(features, labels)`), e
`inference.py` importa `joblib` / `model.predict` no estilo sklearn, que o
notebook não usa.

### Limite de preservação (Qwen)

O Qwen também gerou quatro arquivos, mas trocou o problema: numpy, `SEED=42`,
`N=100`, `y = 2x + ruído`. A avaliação foi para uma classe `Evaluation` em vez
da função livre `evaluate_model(test_labels, predictions)` — daí
`contract_ok=false`. Os testes gerados chamam nomes que não existem (`np`,
`load_data` em `test_training` sem import).

```python
# Qwen — feature_engineering.load_data (trecho)
SEED = 42
np.random.seed(SEED)
X = 3 * np.random.rand(N)
y = 2 * X + np.random.normal(size=N, scale=1.0)
```

```python
# Qwen — evaluation.py (trecho)
class Evaluation:
    @staticmethod
    def evaluate_model(test_labels, test_predictions) -> float:
        mse = np.mean((test_labels - test_predictions) ** 2)
        return mse
```

### Granite e Gemini

Sem código gerado. Granite: recurso local (VRAM / `Broken pipe`). Gemini:
indisponibilidade temporária da faixa gratuita (503). Não são defeitos do
artefato.

## Limitações observadas (júnior)

1. **Hardware local.** `gemma4:e2b` nem entrou. `granite-code:3b` entrou e
   travou na conversão. Só `qwen2.5-coder:3b` coube na GTX 1650 4 GiB para
   uma conversão completa.
2. **Faixa gratuita.** Gemini 503 mantido no CSV; OpenRouter 404 já tinha
   saído no protocolo. Sem troca para slug pago.
3. **Gerar ≠ executar.** Qwen e Groq cumpriram o critério primário (4 módulos
   + testes, proveniência `llm`) e falharam nos descritivos (contrato,
   pipeline, suíte).
4. **Contrato frágil.** Qwen omite a função livre de avaliação. Groq muda a
   aridade de `prepare_features`. Sem Reviewer, isso não é corrigido.
5. **Dependências alucinadas.** Groq puxa `joblib`; Qwen puxa `numpy` e dados
   sintéticos diferentes. O notebook é só biblioteca padrão.
6. **Uma repetição.** Não há estimativa de estabilidade das respostas.

## Liberação da rodada sênior

A Fase 4 está concluída. A rodada sênior pode começar, com as mesmas regras:

- os 4 IDs do protocolo;
- sem Reviewer;
- um modelo por vez;
- **não** repetir `granite-code:3b` nesta máquina;
- Gemini 503, se repetir, permanece como resultado, sem troca de modelo;
- Ollama com `LLM_BASE_URL=http://localhost:11434/v1`.
