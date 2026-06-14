# mlops-agent

AI agent for converting Jupyter Notebooks into production-ready ML pipelines.

## Environment setup

```bash
uv sync
uv run pre-commit install
```

## LLM configuration

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Default local configuration (Ollama):

```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=smollm2:1.7b
```

To use OpenAI API:

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=<your_api_key>
LLM_MODEL=gpt-4o-mini
```

## Ollama quick test

```bash
ollama serve
ollama pull smollm2:1.7b
curl http://localhost:11434/v1/models
```

## Prompt language recommendation

For small local models (such as `smollm2:1.7b`), prefer English prompts for better consistency in code generation and instruction following.

## Tests

```bash
uv run pytest tests/unit/
uv run pytest tests/integration/
```
