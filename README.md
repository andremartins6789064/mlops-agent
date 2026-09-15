# mlops-agent

Sistema multi-agente para converter notebooks Jupyter (`.ipynb`) em projetos de Machine Learning mais próximos de produção: código modular, testes automatizados, revisão de qualidade e exportação em `.zip`.

## O que este projeto resolve

Em muitos times, a fase de experimentação acontece em notebooks e a etapa de produção exige refatoração manual intensa. O `mlops-agent` automatiza essa ponte:

- analisa células do notebook e classifica por estágio de pipeline;
- propõe arquitetura de módulos;
- gera código Python por estágio (`feature_engineering`, `training`, `inference`, `evaluation`);
- gera testes para os módulos;
- valida qualidade (lint, tipos e cobertura);
- exporta um pacote final pronto para download.


## Arquitetura (resumo)

Fluxo principal:

1. `NotebookParser` faz parse do `.ipynb`;
2. `NotebookAnalyzerAgent` classifica células e dependências;
3. `ArchitectureAgent` gera plano de módulos;
4. `CodeGeneratorAgent` gera código por estágio;
5. `PipelineTestGeneratorAgent` gera testes;
6. `ReviewerAgent` valida e tenta corrigir até 2 iterações;
7. `ZipExporter` empacota o projeto final.

Camadas:

- `domain`: entidades, value objects e interfaces;
- `application`: casos de uso e orquestração de serviços;
- `agents`: lógica dos agentes;
- `infrastructure`: parser, clientes LLM, prompts e exportadores;
- `ui`: aplicação Streamlit.

## Pré-requisitos

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- Ollama (opcional, para execução local de LLM)

## Instalação

```bash
uv sync
uv run pre-commit install
```

## Configuração de ambiente

Crie `.env` a partir de `.env.example`:

```bash
cp .env.example .env
```

Configuração local padrão com Ollama:

```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=smollm2:1.7b
```

Exemplo usando OpenAI:

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=<sua_chave>
LLM_MODEL=gpt-4o-mini
```

## Executar a interface web

```bash
uv run streamlit run src/ui/app.py
```

## Docker

Imagem da UI para uso interno (sem Ollama e sem chave gravada). URL, API key e modelo continuam na página Configuração. O Reviewer baixa ruff/mypy/pytest com `uv` na hora, então o container precisa de rede de saída.

```bash
docker build -t mlops-agent .
docker run --rm -p 8501:8501 mlops-agent
```

A interface fica em `http://localhost:8501`. Na nuvem, troque o default `http://localhost:11434/v1` por um provedor OpenAI-compatível (Groq, Gemini, OpenRouter, etc.).

Se o plugin Compose estiver instalado (`docker compose version`):

```bash
docker compose up --build
```

Fluxo na UI:

1. `Configuração`: selecionar provedor/modelo e enviar notebook;
2. `Análise`: revisar análise e plano arquitetural;
3. páginas por estágio: revisar código/testes e enviar feedback;
4. `Download`: verificar métricas e baixar o `.zip`.

## Ollama (teste rápido)

```bash
ollama serve
ollama pull smollm2:1.7b
curl http://localhost:11434/v1/models
```

> Dica: para modelos locais pequenos, prompts em inglês costumam melhorar consistência de geração.

## Matriz com provedores

O harness aceita o provedor junto ao modelo, sem usar `:` ou `/` como
separadores:

```bash
uv run python scripts/run_experiment_matrix.py \
  --notebooks notebooks/junior_regression.ipynb notebooks/senior_regression.ipynb \
  --models ollama=gemma4:e2b groq=openai/gpt-oss-20b gemini=gemini-2.5-flash \
  --repetitions 1 \
  --llm-timeout 300 \
  --llm-retries 3 \
  --llm-retry-backoff 5
```

Omitir `--notebooks` usa os dois notebooks da Etapa 9.0. O harness recusa
arquivos sob `tests/fixtures/` e notebooks que não imprimem `final_mse`,
antes de qualquer chamada de LLM. A fixture `simple_regression.ipynb` continua
válida para testes do parser, PASSO 0 e T-8; não é insumo da matriz.

Groq usa `GROQ_API_KEY`, OpenRouter usa `OPENROUTER_API_KEY` e Gemini usa
`GEMINI_API_KEY`. A chave nunca é gravada no CSV, nos logs ou no nome da
pasta de execução. Erros persistentes, como limites `429`, são registrados
com o motivo sanitizado.

## Testes e qualidade

Executar testes:

```bash
uv run pytest
```

Executar gate de qualidade:

```bash
uv run pre-commit run --all-files
```

## Estrutura de saída gerada

O projeto exportado segue o template de `configs/target_structure.yaml`, com:

- `src/feature_engineering.py`
- `src/training.py`
- `src/inference.py`
- `src/evaluation.py`
- `tests/test_*.py`
- `requirements.txt`, `pyproject.toml`, `README.md`

## Contribuição

Contribuições são bem-vindas. Para manter consistência arquitetural:

- implemente sempre respeitando contratos de `src/domain/interfaces.py`;
- evite acoplamento indevido entre camadas;
- mantenha tipagem, docstrings e cobertura de testes;
- rode `pre-commit` antes de abrir PR.
