# Protocolo experimental — matriz de custo zero

Data do protocolo: 2026-09-16
Commit Git: `6e91cf2` (`6e91cf25c63fa91d93a47e6ee11fd13c8d26a52d`)
Python do projeto (`uv`): 3.11.15
Sistema operacional: Linux 6.17.0-20-generic x86_64
Hardware local: NVIDIA GeForce GTX 1650, 4 GiB VRAM

Este arquivo trava o método da matriz principal do TCC. Copiar as decisões
de escopo da Fase 0. Não alterar IDs, timeouts ou critérios no meio da
execução. Se um provedor falhar, registrar a falha e seguir.

## Escopo (Fase 0)

1. **Custo zero.** Só modelos locais (Ollama) ou APIs com faixa gratuita. Sem
   créditos, planos pagos ou troca silenciosa para API paga.
2. **Falhas de provedor são resultado.** Indisponibilidade, limite de
   requisições, `429`, `503` e timeout entram no CSV e na análise.
3. **APIs privadas pagas.** Apenas trabalho futuro, sob o mesmo protocolo.
   Não são requisito para validar o TCC.
4. **Alcance da conclusão.** O estudo conclui sobre a geração de código nas
   condições de custo zero. Não afirma confiabilidade universal, redução
   garantida de esforço humano nem prontidão para produção.
5. **Reviewer desativado.** Matriz com `enable_review=False`. Sem chamadas
   extras de correção por LLM.
6. **Reviewer como limitação.** Aumenta duração, pode exceder o orçamento de
   chamadas e é instável com Ollama. Não é objeto desta matriz.
7. **Dois objetos distintos.** Qualidade do `mlops-agent` (pytest, cobertura,
   pre-commit) ≠ qualidade dos artefatos gerados. Tabelas separadas.

## Casos de entrada

Notebooks formais da matriz (não usar fixtures de `tests/fixtures/`):

| Notebook | Caminho | `final_mse` original (2026-09-16, sem LLM) |
| --- | --- | ---: |
| Júnior | `notebooks/junior_regression.ipynb` | 0.011202345146 |
| Sênior | `notebooks/senior_regression.ipynb` | 0.011202345146 |

Os dois notebooks geram o mesmo problema de regressão determinístico
(`seed=2026`). O júnior mantém o pipeline em escopo global; o sênior usa
funções explícitas. A métrica foi obtida com `read_notebook_metric`, kernel
`python3` do `.venv`, timeout 120 s, antes de qualquer chamada de LLM da
matriz.

Ordem de execução: **júnior completo, consolidar, só então sênior**.

## Modelos no dia da execução

IDs confirmados em 2026-09-16. Não assumir que continuarão disponíveis.

### Entram na matriz (4)

| Provedor | Spec do harness | Smoke |
| --- | --- | --- |
| Ollama | `ollama=qwen2.5-coder:3b` | HTTP 200, ~6,2 s; GPU descarregada depois |
| Ollama | `ollama=granite-code:3b` | HTTP 200, ~9,8 s; GPU descarregada depois |
| Gemini | `gemini=gemini-3.8-flash` | HTTP 200 na faixa gratuita |
| Groq | `groq=openai/gpt-oss-20b` | HTTP 200 na faixa gratuita |

Smoke curto do Gemini e do Groq (`max_tokens` baixo) devolveu corpo vazio com
`finish_reason=length` (modelo com raciocínio). Isso não é 404, 429 nem 503;
os IDs permanecem na matriz. O harness não limita `max_tokens`.

### Excluídos (não substituídos)

| ID | Motivo | Ação |
| --- | --- | --- |
| `gemma4:e2b` (Ollama) | Trava a máquina. Modelo ~7,2 GB; VRAM 4 GiB. | Fora da matriz. Sem substituto local ou pago. |
| `meta-llama/llama-3.2-3b-instruct:free` (OpenRouter) | HTTP 404: indisponível no gratuito. O provedor sugeriu o slug pago `meta-llama/llama-3.2-3b-instruct`. | Fora da matriz. Slug pago **não usado**. |

As duas exclusões entram na seção de limitações do TCC.

## Parâmetros da matriz

| Parâmetro | Valor |
| --- | --- |
| Repetições | 1 por combinação notebook × modelo |
| Reviewer | desligado (`enable_review=False`) |
| Métrica primária | `final_mse` |
| Tolerância de equivalência | 0,05 (absoluta; `rel_tol=0`) |
| Timeout do notebook / pipeline gerado | 120 s (`--timeout`) |
| Timeout por requisição LLM | 300 s (`--llm-timeout`) |
| Retries LLM | 3, backoff 5 s (somente HTTP 429) |
| Teto de parede por combinação | 900 s (`--max-run-seconds`) |
| Comando do pipeline gerado | `{python} src/main.py` no diretório exportado |
| Mutação | ligada no harness; resultado descritivo |

O teto de 900 s vem da T-18: o padrão 180 s corta modelos locais no meio da
conversão. HTTP 429 é retentado; `503` e timeout **não** justificam troca de
modelo.

## GPU local

A GTX 1650 não comporta dois modelos Ollama ao mesmo tempo. Procedimento:

1. Rodar **um** modelo local por vez.
2. Depois de cada combinação local: `ollama stop <modelo>` e conferir
   `nvidia-smi` (~65 MiB ociosos nesta máquina).
3. Só então carregar o próximo modelo local.
4. Gemini e Groq não usam a GPU local; podem rodar com Ollama descarregado.

Não executar `gemma4:e2b`.

## Resultado primário

Uma combinação **gera** se o artefato exportado contém:

- os quatro módulos `feature_engineering`, `training`, `inference`,
  `evaluation`;
- os arquivos de teste do pipeline;
- proveniência de cada estágio registrada como `llm` ou `template`.

A geração não é invalidada por falha de lint, tipos, testes, cobertura,
mutação, contrato ou equivalência.

## Resultados descritivos (não critérios de aprovação)

Registrar no CSV, sem usar como porta de aprovação:

- status do contrato entre estágios;
- execução do pipeline gerado (`python src/main.py`);
- equivalência de `final_mse` (tolerância 0,05) quando o pipeline rodar;
- lint, tipos, testes, cobertura e mutação do artefato.

Erros nessas métricas são diagnóstico do modelo. Relatórios brutos do
artefato não precisam ser arquivados; o CSV e o `execution.log` bastam.

## Registro obrigatório de falha

Manter a linha no CSV quando a combinação for interrompida por:

- limite da faixa gratuita (`429` ou equivalente);
- indisponibilidade (`404`, `503`);
- timeout de LLM ou teto de parede;
- falha sanitizada do cliente.

Não repetir a combinação com modelo pago. Não apagar a linha.

## Comando de referência (júnior, um modelo)

O harness só liga o Reviewer com `--enable-review`. Nesta matriz a flag
**não** é passada.

```bash
uv run python scripts/run_experiment_matrix.py \
  --notebooks notebooks/junior_regression.ipynb \
  --models ollama=qwen2.5-coder:3b \
  --repetitions 1 \
  --llm-timeout 300 \
  --llm-retries 3 \
  --llm-retry-backoff 5 \
  --timeout 120 \
  --max-run-seconds 900 \
  --output-csv docs/experiments/matriz_custo_zero_junior.csv \
  --output-root docs/experiments/runs_matriz_custo_zero_junior
```

Depois de cada modelo Ollama: `ollama stop <modelo>`.

Modelos da rodada júnior, nesta ordem: `ollama=qwen2.5-coder:3b`,
`ollama=granite-code:3b`, `gemini=gemini-3.8-flash`,
`groq=openai/gpt-oss-20b`.

O `.env` local aponta `LLM_BASE_URL` para Gemini. Toda combinação Ollama
precisa exportar no comando:

```bash
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_API_KEY="ollama"
```

Sem isso o cliente chama Gemini e falha com `400 Please pass a valid API key`.
Isso não é resultado do modelo local; é configuração. O rerun correto usa o
override acima.

O CSV júnior deve ser consolidado antes de qualquer execução do notebook
sênior.

## Qualidade do software (fora desta tabela)

Métricas do `mlops-agent` em 2026-09-16, commit `6e91cf2`: 165 testes
aprovados, 2 pulados, cobertura 88,50%, pre-commit passou. Subseção própria
da monografia. Não misturar com a tabela da matriz.
