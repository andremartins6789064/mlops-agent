# Tarefas para iniciar a escrita do TCC

## Escopo fixado

- [ ] Registrar que os experimentos usarão somente modelos locais ou APIs com faixa gratuita, sem contratação de créditos ou planos pagos.
- [ ] Definir que indisponibilidade, limite de requisições, `429`, `503` e timeout são resultados experimentais a registrar, e não motivos para trocar para uma API paga.
- [ ] Registrar que a comparação com APIs privadas pagas será apenas sugestão de trabalho futuro; não é requisito para validar este TCC.
- [ ] Limitar a conclusão do estudo à geração de código pelo protótipo nas condições de custo zero. Não afirmar confiabilidade universal, redução garantida de esforço humano ou prontidão para produção.
- [ ] Executar a matriz principal com o Reviewer desativado (`enable_review=False`). A avaliação deve medir a conversão e o artefato exportado, sem acrescentar chamadas de correção por LLM.
- [ ] Tratar o Reviewer como limitação conhecida do protótipo: ele aumenta a duração, pode exceder o orçamento de chamadas e é especialmente instável com modelos locais do Ollama.
- [ ] Não comparar a qualidade interna do `mlops-agent` com a qualidade dos artefatos gerados. O primeiro é controle de desenvolvimento; o segundo é o objeto da avaliação experimental.

## Métricas de qualidade do mlops-agent

- [ ] Executar os comandos de qualidade já documentados no `mlops-agent/README.md`: `uv run pytest` e `uv run pre-commit run --all-files`.
- [ ] Transcrever na monografia somente o resumo final: testes aprovados, cobertura e resultado das verificações estáticas.
- [ ] Apresentar essas métricas em uma subseção própria sobre a qualidade do software desenvolvido, sem inseri-las na tabela da matriz de geração e sem compará-las aos artefatos gerados.
- [ ] Não salvar relatórios brutos dessas execuções: os comandos do README são o procedimento de reprodução das métricas.

## Preparação da evidência experimental

- [x] Criar um arquivo de protocolo em `mlops-agent/docs/experiments/` com data, commit Git, versão do Python, sistema operacional, modelos, provedores, limites de timeout, número de repetições e tolerância de equivalência.
- [x] Registrar os modelos locais e gratuitos realmente disponíveis no dia da execução, sem assumir que um ID de modelo ou provedor continuará disponível.
- [x] Usar os modelos locais que cabem no hardware: `qwen2.5-coder:3b` e `granite-code:3b`. `gemma4:e2b` foi **excluído da matriz** em 2026-09-16: o smoke test trava a máquina e não há VRAM suficiente (NVIDIA GeForce GTX 1650, 4 GiB; o modelo ocupa ~7,2 GB em disco). Não foi substituído por outro modelo.
- [x] Usar as APIs gratuitas após smoke sem custo: Gemini `gemini-3.8-flash` e Groq `openai/gpt-oss-20b` responderam HTTP 200. OpenRouter `meta-llama/llama-3.2-3b-instruct:free` retornou **404** (indisponível no gratuito). O provedor sugeriu o slug pago `meta-llama/llama-3.2-3b-instruct`; **não foi usado**. Sem substituição.
- [x] Se um modelo de API deixar de estar disponível na conta gratuita, registrar a indisponibilidade e o código de erro; não trocar silenciosamente por um modelo pago.
- [x] Selecionar formalmente os notebooks `junior_regression.ipynb` e `senior_regression.ipynb` como os casos de entrada do experimento.
- [x] Executar cada notebook selecionado isoladamente e registrar o valor original de `final_mse` antes de chamar qualquer LLM.
- [x] Executar somente uma repetição de cada combinação notebook e modelo. Não repetir execuções nesta versão do estudo.
- [x] Definir como resultado primário da conversão: os quatro módulos de pipeline e os arquivos de teste foram gerados, com a proveniência de cada estágio registrada como `llm` ou `template`.
- [x] Definir como resultados descritivos, e não critérios de aprovação: contrato entre estágios, execução do pipeline, equivalência de métrica, lint, tipos, testes, cobertura e mutação.

## Execução da matriz

- [x] Executar primeiro uma rodada completa do notebook júnior com os quatro modelos selecionados (2 locais + 2 APIs), sem Reviewer.
- [x] Salvar o CSV da matriz e os `execution.log` correspondentes em `mlops-agent/docs/experiments/` ou em um caminho versionado equivalente.
- [x] Para cada execução, registrar duração, origem de cada estágio (`llm` ou `template`), status do contrato e motivo de falha, quando houver.
- [x] Quando o pipeline gerado executar sem intervenção manual, comparar o `final_mse` com o notebook usando a tolerância declarada no protocolo.
- [x] Quando as métricas de qualidade do artefato estiverem disponíveis, anotar somente os valores finais relevantes para a monografia como diagnóstico do modelo; erros nessas métricas não invalidam a geração e os relatórios brutos não precisam ser arquivados.
- [x] Para cada execução interrompida por limite gratuito, indisponibilidade ou timeout, registrar a causa sanitizada no CSV e mantê-la na análise; não substituir a execução por uma API paga.
- [x] Consolidar os resultados do notebook júnior antes de executar outro notebook: tabela por modelo, exemplos de código e limitações observadas.
- [x] Só depois da consolidação do notebook júnior, executar a rodada do notebook sênior com os quatro modelos selecionados, sem Reviewer.
- [x] Comparar o resultado do notebook sênior com o resultado consolidado do notebook júnior, destacando o que se manteve e o que mudou.

## Consolidação para a redação

- [x] Produzir uma tabela única com uma linha por combinação notebook, modelo e repetição.
- [x] Calcular e registrar: número de execuções, número de módulos gerados por LLM, número de estágios em fallback, número de pipelines executáveis quando aplicável e distribuição dos motivos de falha.
- [x] Separar nos resultados os comportamentos do código gerado das limitações do provedor ou do modelo gratuito.
- [x] Produzir uma tabela separada com as métricas de qualidade do `mlops-agent` obtidas pelos comandos do README.
- [x] Selecionar exemplos representativos de código gerado para discutir a modularização, a preservação da lógica do notebook e os limites observados.
- [ ] Atualizar a lista de referências, remover notas provisórias e inserir citações no texto da introdução.
- [x] Registrar que o artefato exportado gera empacotamento reprodutível (`README`, `pyproject.toml` e dependências), mas não implementa versionamento Git ou CI/CD automático.

## Critério para iniciar a escrita

Iniciar a redação quando todos os itens acima estiverem concluídos, exceto quando uma limitação de provedor gratuito impedir uma execução. Nesse caso, a limitação deve constar no CSV, no protocolo e na seção de limitações do TCC.

O texto deverá apresentar o projeto como um estudo experimental aplicado de viabilidade sob restrição de custo zero. APIs privadas pagas podem ser mencionadas somente como uma avaliação futura sob o mesmo protocolo experimental.

Mais de uma repetição por combinação deve ser registrada como melhoria futura, para ampliar a análise de estabilidade e variabilidade das respostas dos modelos.
