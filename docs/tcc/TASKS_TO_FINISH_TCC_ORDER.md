# Ordem de execução — tarefas antes da escrita do TCC

Fonte: `TASKS_TO_FINISH_TCC.md`.

A ordem segue as dependências do documento original: o escopo trava as regras, o protocolo trava o método, o júnior alimenta o sênior, e a consolidação só fecha depois da matriz.

**Resumo da sequência:** escopo → qualidade do repo → protocolo + MSE original → júnior → consolidar júnior → sênior → tabelas/exemplos → referências → escrita.

O ponto que mais atrasa o TCC se invertido é o da Fase 4: consolidar o júnior **antes** de tocar no sênior. Também não misturar as métricas do `mlops-agent` (Fase 1) com a tabela da matriz (Fase 6).

---

## Fase 0 — Fixar o escopo (antes de qualquer execução)

Esses itens não geram evidência; eles impedem que o experimento mude de regra no meio do caminho. Fazer primeiro, de uma vez, e copiar as decisões para o protocolo da Fase 2.

- [x] Registrar: só modelos locais ou APIs com faixa gratuita; sem créditos pagos.
- [x] Definir `429`, `503`, timeout e indisponibilidade como **resultado experimental**, não como motivo para trocar de API.
- [x] Registrar: comparação com APIs pagas fica só como trabalho futuro.
- [x] Limitar a conclusão à geração de código em custo zero (sem afirmar confiabilidade universal, redução de esforço ou prontidão para produção).
- [x] Fixar a matriz principal com Reviewer **desativado** (`enable_review=False`).
- [x] Tratar o Reviewer como limitação conhecida do protótipo (duração, orçamento de chamadas, instabilidade no Ollama).
- [x] Separar formalmente: qualidade do `mlops-agent` ≠ qualidade dos artefatos gerados.

### Resultado da Fase 0 — escopo registrado (2026-09-16)

Decisões fixadas. Copiar integralmente para o protocolo da Fase 2.

1. **Custo zero.** Os experimentos usam somente modelos locais (Ollama) ou APIs com faixa gratuita. Não há contratação de créditos, planos pagos nem troca silenciosa para API paga.
2. **Falhas de provedor são resultado.** Indisponibilidade, limite de requisições, `429`, `503` e timeout entram no CSV e na análise. Não justificam substituir o modelo nem repetir a execução com outro provedor pago.
3. **APIs privadas pagas.** Podem aparecer só como sugestão de trabalho futuro, sob o mesmo protocolo. Não são requisito para validar o TCC.
4. **Alcance da conclusão.** O estudo conclui sobre a geração de código pelo protótipo nas condições de custo zero. Não afirma confiabilidade universal, redução garantida de esforço humano nem prontidão para produção.
5. **Reviewer desativado na matriz.** A matriz principal corre com `enable_review=False`. A avaliação mede conversão e artefato exportado, sem chamadas extras de correção por LLM.
6. **Reviewer como limitação.** O Reviewer aumenta a duração, pode exceder o orçamento de chamadas e é especialmente instável com modelos locais do Ollama. Fica registrado como limitação conhecida do protótipo, não como objeto da matriz.
7. **Dois objetos distintos.** As métricas de qualidade do `mlops-agent` (testes, cobertura, lint/tipos do repositório) são controle de desenvolvimento. A qualidade dos artefatos gerados é o objeto da avaliação experimental. Não se comparam entre si nem compartilham a mesma tabela.

## Fase 1 — Qualidade do software (independente da matriz)

Pode rodar em paralelo com a Fase 2. Não depende de LLM nem de notebook.

- [x] Executar `uv run pytest` e `uv run pre-commit run --all-files`.
- [x] Anotar só o resumo: testes aprovados, cobertura, checagens estáticas. Não arquivar relatórios brutos.
- [x] Reservar esses números para uma **subseção própria** da monografia, fora da tabela da matriz.

### Resultado da Fase 1 — qualidade do `mlops-agent` (2026-09-16)

Procedimento: comandos do `README.md`. Relatórios brutos não foram arquivados.

| Item | Valor |
| --- | --- |
| Data | 2026-09-16 |
| Commit | `6e91cf2` |
| Python do projeto (`uv`) | 3.11.15 |
| Sistema | Linux 6.17.0-20-generic x86_64 |
| Comando de testes | `uv run pytest` |
| Testes | 165 aprovados, 2 pulados, 0 falhas |
| Cobertura | 88,50% (`--cov-fail-under=80`) |
| Comando estático | `uv run pre-commit run --all-files` |
| Hooks | todos passaram (`ruff check`, `ruff format`, `mypy`, `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files`) |

Uso na monografia: subseção própria sobre a qualidade do software desenvolvido. Não inserir estes números na tabela da matriz de geração nem compará-los aos artefatos gerados.

## Fase 2 — Protocolo e linha de base (antes da matriz)

Sem isso, a matriz não é reproduzível.

- [x] Criar o protocolo em `mlops-agent/docs/experiments/` com: data, commit Git, Python, SO, modelos, provedores, timeouts, **1 repetição**, tolerância de equivalência do `final_mse`. Incluir as decisões da Fase 0.
- [x] Confirmar os modelos locais que cabem no hardware: `qwen2.5-coder:3b`, `granite-code:3b`.
- [x] Excluir `gemma4:e2b` da matriz: trava a máquina e não há recurso suficiente para rodar.
- [x] Fazer smoke test **sem custo** das três APIs: Gemini (`gemini-3.8-flash`), Groq (`openai/gpt-oss-20b`), OpenRouter (`meta-llama/llama-3.2-3b-instruct:free`).
- [x] Se algum ID gratuito falhar: registrar erro e indisponibilidade; **não** substituir por modelo pago.
- [x] Registrar no protocolo os IDs realmente disponíveis **no dia da execução**.
- [x] Formalizar os casos: `junior_regression.ipynb` e `senior_regression.ipynb`.
- [x] Rodar cada notebook **isolado**, sem LLM, e registrar o `final_mse` original.
- [x] Definir resultado primário: quatro módulos + testes gerados, com proveniência `llm` ou `template`.
- [x] Definir como descritivos (não critérios de aprovação): contrato, execução do pipeline, equivalência de métrica, lint, tipos, testes, cobertura e mutação.

### Resultado parcial da Fase 2 — modelos locais (2026-09-16)

Hardware: NVIDIA GeForce GTX 1650, **4 GiB VRAM**. GPU ociosa em ~65 MiB usados / 3651 MiB livres. Smoke local um modelo por vez, com `keep_alive=0` e `ollama stop` depois de cada um.

| Modelo | Papel na matriz | Smoke | Duração | GPU depois |
| --- | --- | --- | ---: | --- |
| `qwen2.5-coder:3b` | entra | HTTP 200, respondeu | 6,2 s | 65 MiB |
| `granite-code:3b` | entra | HTTP 200, respondeu | 9,8 s | 65 MiB |
| `gemma4:e2b` | **excluído** | não executado | — | — |

**Exclusão de `gemma4:e2b`.** O modelo está instalado (~7,2 GB em disco), mas não entra na matriz: em tentativas anteriores o smoke trava a máquina, e 4 GiB de VRAM não bastam para carregá-lo. Não foi substituído por outro modelo local nem por API paga. A limitação vai para o protocolo e para a seção de limitações do TCC.

Matriz vigente após os locais: **5 candidatos** (2 Ollama + 3 APIs). Smoke das APIs abaixo.

### Resultado da Fase 2 — smoke das APIs (2026-09-16)

Uma chamada por provedor, sem custo extra de plano. Chaves não foram impressas.

| Provedor | Modelo | HTTP | Papel na matriz | Detalhe |
| --- | --- | ---: | --- | --- |
| Gemini | `gemini-3.8-flash` | 200 | entra | ID aceito na faixa gratuita (~2,1 s). Completions curtas (`max_tokens=8` e `32`) voltaram vazias com `finish_reason=length` (padrão de modelo com raciocínio). Não é 404/429/503. |
| Groq | `openai/gpt-oss-20b` | 200 | entra | ID aceito na faixa gratuita (~0,9 s). Mesmo padrão: corpo vazio com `finish_reason=length` no smoke curto. |
| OpenRouter | `meta-llama/llama-3.2-3b-instruct:free` | 404 | **excluído** | Mensagem: modelo indisponível no plano gratuito. O provedor sugeriu o slug pago `meta-llama/llama-3.2-3b-instruct`. **Não usado.** |

**Exclusão do OpenRouter gratuito.** Registrado o HTTP 404. Não houve troca para a variante paga. A limitação vai para o protocolo e para a seção de limitações do TCC.

Matriz vigente: **4 modelos** = `qwen2.5-coder:3b`, `granite-code:3b`, Gemini `gemini-3.8-flash`, Groq `openai/gpt-oss-20b`.

### Resultado da Fase 2 — protocolo e linha de base (2026-09-16)

Arquivo: `mlops-agent/docs/experiments/protocol_matriz_custo_zero_2026-09-16.md`.

| Item | Valor |
| --- | --- |
| Commit | `6e91cf2` |
| Python | 3.11.15 |
| SO | Linux 6.17.0-20-generic x86_64 |
| Notebooks | `junior_regression.ipynb`, `senior_regression.ipynb` |
| `final_mse` original (ambos) | 0.011202345146 |
| Repetições | 1 |
| Reviewer | `enable_review=False` |
| Tolerância `final_mse` | 0,05 absoluta |
| Timeout LLM / pipeline / parede | 300 s / 120 s / 900 s |
| Resultado primário | 4 módulos + testes, proveniência `llm` ou `template` |
| Descritivos | contrato, execução, equivalência, lint, tipos, testes, cobertura, mutação |

Fase 2 concluída. Próximo: Fase 3, notebook júnior, um modelo por vez, começando por `ollama=qwen2.5-coder:3b`.

## Fase 3 — Matriz júnior (4 modelos, sem Reviewer)

Uma repetição por combinação. Não avançar para o sênior antes de consolidar. `gemma4:e2b` e o Llama gratuito do OpenRouter não entram.

- [x] Rodar o notebook júnior com os 4 modelos, `enable_review=False`.
- [x] Salvar CSV + `execution.log` em `docs/experiments/`.
- [x] Para cada execução: duração, origem de cada estágio, status do contrato, motivo de falha.
- [x] Se o pipeline rodar sozinho: comparar `final_mse` com a tolerância do protocolo.
- [x] Se houver métricas de qualidade do artefato: anotar só os valores finais (diagnóstico do modelo; erro nelas não invalida a geração).
- [x] Se houver limite gratuito / timeout / indisponibilidade: registrar causa sanitizada no CSV e **manter** a linha.

### Resultado parcial da Fase 3 — júnior `qwen2.5-coder:3b` (2026-09-16)

CSV: `docs/experiments/matriz_custo_zero_junior_qwen.csv`.
Run: `docs/experiments/runs_matriz_custo_zero_junior/junior_regression-ollama-qwen2_5_coder_3b-run1/`.

Primeira tentativa bateu no Gemini (`400 Please pass a valid API key`) porque o `.env` define `LLM_BASE_URL` da API. Descartada. Rerun com `LLM_BASE_URL=http://localhost:11434/v1` e `LLM_API_KEY=ollama`. GPU descarregada depois (`ollama stop`).

| Campo | Valor |
| --- | --- |
| Duração | 161,8 s |
| Resultado primário | gerou: 4 módulos + testes; todos `origin=llm`; 0 fallback |
| Reviewer | desativada |
| Contrato | `nao_conforme` (`evaluation.evaluate_model` ausente) |
| Equivalência | `nao_executavel` (pipeline exit 1); original `0.011202345146` |
| Mutação | `suite_invalida` |
| Diagnóstico (não invalida) | suíte do artefato falhou (NameError nos testes; cobertura 33%) |

Seguir: `granite-code:3b`, mesmo override de URL do Ollama.

### Resultado parcial da Fase 3 — júnior `granite-code:3b` (2026-09-16)

**Não deu certo.** A conversão não concluiu e a máquina travou na prática.

CSV: `docs/experiments/matriz_custo_zero_junior_granite.csv` (linha mantida).
Log: só `execution_started` em `.../junior_regression-ollama-granite_code_3b-run1/execution.log`. Sem módulos gerados.

| Campo | Valor |
| --- | --- |
| Duração no CSV | 164,0 s |
| Erro | `[Errno 32] Broken pipe` |
| Resultado primário | não gerou (parou na primeira chamada LLM) |
| GPU agora | livre (~65 MiB); `ollama ps` vazio |

O smoke curto do Granite tinha passado (~10 s). A conversão completa do notebook júnior não. Não repetir nesta máquina: mesmo padrão de travamento do `gemma4:e2b`. A linha fica no CSV como falha de recurso local, não como troca de modelo.

### Resultado da Fase 3 — júnior `gemini-3.8-flash` (2026-09-17)

CSV: `docs/experiments/matriz_custo_zero_junior_gemini.csv` (linha mantida). Sem artefato.

| Campo | Valor |
| --- | --- |
| Duração | 14,3 s |
| HTTP | **503** *high demand* / `UNAVAILABLE` |
| Resultado primário | não gerou (falhou na primeira chamada LLM) |
| Substituição | nenhuma (não trocar por outro ID nem API paga) |

### Resultado da Fase 3 — júnior `openai/gpt-oss-20b` (Groq, 2026-09-17)

CSV: `docs/experiments/matriz_custo_zero_junior_groq.csv`.
Run: `docs/experiments/runs_matriz_custo_zero_junior/junior_regression-groq-openai_gpt_oss_20b-run1/`.

| Campo | Valor |
| --- | --- |
| Duração | 109,1 s |
| Resultado primário | gerou: 4 módulos + testes; todos `origin=llm`; 0 fallback |
| Reviewer | desativada |
| Contrato | `nao_conforme` (`prepare_features` com 1 argumento; esperado 2) |
| Equivalência | `nao_executavel`; original `0.011202345146` |
| Mutação | `suite_invalida` |
| Diagnóstico (não invalida) | suíte não coleta: `ModuleNotFoundError: joblib` em `inference.py` |

### Resultado da Fase 3 — CSV único do júnior

Arquivo consolidado: `docs/experiments/matriz_custo_zero_junior.csv` (4 linhas). Nenhum pipeline júnior executou até imprimir `final_mse`; equivalência não se aplica. Logs em `docs/experiments/runs_matriz_custo_zero_junior/`.

| Modelo | Duração (s) | Primário (4 módulos + testes) | Origem | Contrato | Equivalência | Motivo de falha |
| --- | ---: | --- | --- | --- | --- | --- |
| `qwen2.5-coder:3b` | 161,8 | sim | 4× `llm` | `nao_conforme` | `nao_executavel` | suíte do artefato (NameError; cov. 33%) |
| `granite-code:3b` | 164,0 | não | — | — | — | `Broken pipe` (travou a máquina) |
| `gemini-3.8-flash` | 14,3 | não | — | — | — | HTTP 503 high demand |
| `openai/gpt-oss-20b` | 109,1 | sim | 4× `llm` | `nao_conforme` | `nao_executavel` | `joblib` ausente na suíte |

Separação: Qwen e Groq falharam no **código gerado** (contrato/testes). Granite falhou por **recurso local**. Gemini falhou por **provedor gratuito**.

Fase 3 do júnior encerrada. Próximo: Fase 4 (consolidar júnior) antes de qualquer sênior.

## Fase 4 — Consolidar o júnior (porta obrigatória)

- [x] Tabela por modelo, exemplos de código e limitações observadas.
- [x] Só então liberar a rodada sênior.

Arquivo: `mlops-agent/docs/experiments/consolidacao_junior_2026-09-17.md`.

Resumo: 4 execuções; 2 geraram (Qwen, Groq; 8 estágios `llm`; 0 fallback); 0 pipelines executáveis. Groq preservou seed/OLS do notebook; Qwen trocou o dataset. Granite = recurso local; Gemini = 503. Rodada sênior liberada, **sem** repetir Granite nesta GPU.

## Fase 5 — Matriz sênior (mesmas regras)

- [x] Rodar o notebook sênior com os 4 modelos, sem Reviewer.
- [x] Repetir o mesmo registro da Fase 3 (CSV, logs, duração, proveniência, contrato, falhas, MSE quando aplicável).

### Resultado parcial da Fase 5 — sênior `qwen2.5-coder:3b` (2026-09-17)

**Limitação de recurso.** A conversão não fechou nesta GTX 1650.

CSV: `docs/experiments/matriz_custo_zero_senior_qwen.csv` → `timeout after 900s`.
Log: 4 estágios gerados em ~90 s (`feature_engineering`/`training`/`evaluation` `contract_ok`; `inference.predict` com 3 argumentos). Sem testes, sem ZIP. Depois o processo ficou preso até o teto de parede; a sessão durou horas na prática.

| Campo | Valor |
| --- | --- |
| Resultado primário | incompleto (módulos parciais; testes não gerados) |
| Harness | timeout 900 s |
| GPU agora | livre (~65 MiB) |

No júnior o mesmo Qwen tinha concluído (~162 s). No sênior o contexto maior parece estourar a VRAM/tempo. **Não repetir Qwen local no sênior.**

### Resultado da Fase 5 — sênior `granite-code:3b`

Não executado. Mesma limitação de GPU do júnior (`Broken pipe`). Linha no CSV: `not executed: local GPU hang on junior`.

### Resultado da Fase 5 — sênior `gemini-3.8-flash` (2026-09-17)

HTTP **503** *high demand*, 24,3 s. Sem artefato. Sem troca de modelo. Igual ao júnior.

### Resultado da Fase 5 — sênior `openai/gpt-oss-20b` (Groq, 2026-09-17)

CSV: `docs/experiments/matriz_custo_zero_senior_groq.csv`. Conversão concluiu em 131,0 s.

| Campo | Valor |
| --- | --- |
| Resultado primário | gerou 4 módulos + testes |
| Origem | FE/inference/evaluation `llm`; **training `template`** (1 fallback) |
| Contrato | `nao_conforme` (`prepare_features` com 1 argumento) |
| Equivalência | `nao_executavel`; original `0.011202345146` |
| Diagnóstico | suíte não coleta: `ModuleNotFoundError: sklearn` |

### Resultado da Fase 5 — CSV único do sênior

Arquivo: `docs/experiments/matriz_custo_zero_senior.csv` (4 linhas).

| Modelo | Primário | Motivo |
| --- | --- | --- |
| `qwen2.5-coder:3b` | incompleto | timeout 900 s / recurso local |
| `granite-code:3b` | não executado | GPU (mesmo hang do júnior) |
| `gemini-3.8-flash` | não gerou | HTTP 503 |
| `openai/gpt-oss-20b` | gerou (1 fallback) | código gerado: sklearn / contrato |

Fase 5 encerrada. Próximo: Fase 6 (comparar sênior vs júnior e fechar números).

## Fase 6 — Comparar e fechar números

- [x] Comparar sênior vs júnior: o que se manteve e o que mudou.
- [x] Tabela única: uma linha por combinação notebook × modelo × repetição (aqui, repetição = 1).
- [x] Calcular: nº de execuções, módulos gerados por LLM, estágios em fallback, pipelines executáveis, distribuição de falhas.
- [x] Separar falhas de **código gerado** das falhas de **provedor/cota gratuita**.
- [x] Tabela separada com as métricas da Fase 1 (`pytest` / `pre-commit`).
- [x] Escolher exemplos representativos (modularização, preservação da lógica, limites).
- [x] Registrar: o export gera empacotamento reprodutível (`README`, `pyproject.toml`, dependências), mas **não** Git nem CI/CD automático.

Arquivo: `mlops-agent/docs/experiments/consolidacao_fase6_2026-09-17.md`.

Números: 8 combinações; 3 conversões com critério primário; 11 estágios `llm` nessas três; 1 fallback; 0 pipelines executáveis. Falhas: 3 código gerado, 3 recurso local, 2 provedor 503. Groq foi o único a fechar os dois notebooks; no sênior preservou menos a lógica do que no júnior.

## Fase 7 — Preparar o texto (último passo antes de escrever)

- [ ] Atualizar referências, remover notas provisórias e inserir citações na introdução.
- [ ] Conferir o critério de partida: todos os itens concluídos, **exceto** execução impedida por limite gratuito — nesse caso a falha já está no CSV, no protocolo e vai para a seção de limitações.

---

## Critério para iniciar a escrita

Iniciar a redação quando todos os itens acima estiverem concluídos, exceto quando uma limitação de provedor gratuito impedir uma execução. Nesse caso, a limitação deve constar no CSV, no protocolo e na seção de limitações do TCC.

O texto deverá apresentar o projeto como um estudo experimental aplicado de viabilidade sob restrição de custo zero. APIs privadas pagas podem ser mencionadas somente como uma avaliação futura sob o mesmo protocolo experimental.

Mais de uma repetição por combinação deve ser registrada como melhoria futura, para ampliar a análise de estabilidade e variabilidade das respostas dos modelos.
