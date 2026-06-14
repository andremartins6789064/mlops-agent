# benchmark_models.py
# Execução: python benchmark_models.py

import ast
import json
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, cast

# ── Configuração ──────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
N_RUNS = 3  # número de execuções por teste por modelo

MODELS = [
    # (nome_ollama, tem_think_mode)
    ("qwen3.5:0.8b", True),
    ("qwen3:1.7b", True),
    ("qwen3.5:2b", True),
    ("nemotron-3-nano:4b", True),
    ("phi3:mini", False),
    ("deepseek-r1:1.5b", True),
    ("deepseek-coder:1.3b", False),
    ("llama3.2:3b", False),
    ("gemma4:e2b", False),
    ("smollm2:1.7b", False),
    ("codegemma:2b", False),
    ("starcoder2:3b", False),
    ("yi-coder:1.5b", False),
    ("granite-code:3b", False),
    ("stable-code:3b", False),
    ("qwen2.5-coder:3b", False),
]

# ── Prompts ───────────────────────────────────────────────────────────────────

TESTS = {
    "instruction_following": {
        "system": "You are a helpful assistant. Follow instructions precisely.",
        "user": "Reply with exactly one word: OK",
    },
    "json_output": {
        "system": (
            "You are a helpful assistant. Reply ONLY with pure JSON, "
            "no markdown, no explanations."
        ),
        "user": (
            "Return a JSON object with exactly these keys and values: "
            '"status": "ok", "task": "json", "success": true'
        ),
    },
    "code_generation": {
        "system": (
            "You are a Python expert. Reply ONLY with pure Python code, "
            "no markdown, no explanations."
        ),
        "user": (
            "Write a function called clean_data that receives a pandas DataFrame, "
            "removes rows with null values and duplicates, and returns "
            "the cleaned DataFrame. "
            "Use type hints and a docstring."
        ),
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def ollama_chat(model: str, system: str, user: str, think: bool) -> dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "think": think,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return cast(dict[str, Any], json.loads(resp.read().decode()))


def tokens_per_second(raw: dict[str, Any]) -> float:
    ec = raw.get("eval_count", 0)
    ed = raw.get("eval_duration", 1)
    return round(ec / (ed / 1_000_000_000), 1) if ec else 0.0


def has_markdown(text: str) -> bool:
    return "```" in text


def check_instruction(content: str) -> bool:
    return content.strip().lower().rstrip(".").rstrip("!") == "ok"


def check_json(content: str) -> dict[str, bool]:
    text = re.sub(r"```(?:json)?|```", "", content).strip()
    try:
        parsed = json.loads(text)
        valid = True
        fields = all(k in parsed for k in ("status", "task", "success"))
    except json.JSONDecodeError:
        valid = False
        fields = False
    return {
        "valid": valid,
        "no_markdown": not has_markdown(content),
        "fields": fields,
    }


def check_code(content: str) -> dict[str, bool]:
    md = has_markdown(content)
    text = re.sub(r"```(?:python)?|```", "", content).strip()
    try:
        ast.parse(text)
        valid_syntax = True
    except SyntaxError:
        valid_syntax = False
    return {
        "valid_syntax": valid_syntax,
        "no_markdown": not md,
        "type_hints": bool(re.search(r"->\s*\w", text) and re.search(r":\s*\w", text)),
        "docstring": '"""' in text or "'''" in text,
        "correct_logic": "dropna" in text and "drop_duplicates" in text,
    }


# ── Dataclass de resultado ────────────────────────────────────────────────────


@dataclass
class ModelResult:
    model: str

    # taxas de sucesso (0.0 a 1.0) — média de N_RUNS
    instruction_rate: float = 0.0

    json_valid_rate: float = 0.0
    json_nomd_rate: float = 0.0
    json_fields_rate: float = 0.0

    code_syntax_rate: float = 0.0
    code_nomd_rate: float = 0.0
    code_hints_rate: float = 0.0
    code_doc_rate: float = 0.0
    code_logic_rate: float = 0.0

    avg_toks_per_sec: float = 0.0

    # score final: média ponderada das taxas (máx 10.0)
    score: float = 0.0

    error: str = ""


# ── Runner ────────────────────────────────────────────────────────────────────


def run_model(model: str, think: bool) -> ModelResult:
    result = ModelResult(model=model)

    instr_hits: list[int] = []
    json_valid: list[int] = []
    json_nomd: list[int] = []
    json_fields: list[int] = []
    code_syntax: list[int] = []
    code_nomd: list[int] = []
    code_hints: list[int] = []
    code_doc: list[int] = []
    code_logic: list[int] = []
    tps_list: list[float] = []

    for run in range(N_RUNS):
        # --- Teste 1: instruction following ---
        try:
            raw = ollama_chat(model, **TESTS["instruction_following"], think=think)
            content = raw["message"]["content"]
            instr_hits.append(int(check_instruction(content)))
            tps_list.append(tokens_per_second(raw))
        except Exception as e:
            result.error = f"run {run + 1} instruction: {e}"
            return result

        # --- Teste 2: JSON ---
        try:
            raw = ollama_chat(model, **TESTS["json_output"], think=think)
            content = raw["message"]["content"]
            cj = check_json(content)
            json_valid.append(int(cj["valid"]))
            json_nomd.append(int(cj["no_markdown"]))
            json_fields.append(int(cj["fields"]))
            tps_list.append(tokens_per_second(raw))
        except Exception as e:
            result.error = f"run {run + 1} json: {e}"
            return result

        # --- Teste 3: código ---
        try:
            raw = ollama_chat(model, **TESTS["code_generation"], think=think)
            content = raw["message"]["content"]
            cc = check_code(content)
            code_syntax.append(int(cc["valid_syntax"]))
            code_nomd.append(int(cc["no_markdown"]))
            code_hints.append(int(cc["type_hints"]))
            code_doc.append(int(cc["docstring"]))
            code_logic.append(int(cc["correct_logic"]))
            tps_list.append(tokens_per_second(raw))
        except Exception as e:
            result.error = f"run {run + 1} code: {e}"
            return result

        print(".", end="", flush=True)  # progresso visual por run

    def avg(lst: list[int]) -> float:
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    result.instruction_rate = avg(instr_hits)
    result.json_valid_rate = avg(json_valid)
    result.json_nomd_rate = avg(json_nomd)
    result.json_fields_rate = avg(json_fields)
    result.code_syntax_rate = avg(code_syntax)
    result.code_nomd_rate = avg(code_nomd)
    result.code_hints_rate = avg(code_hints)
    result.code_doc_rate = avg(code_doc)
    result.code_logic_rate = avg(code_logic)
    result.avg_toks_per_sec = round(sum(tps_list) / len(tps_list), 1)

    # score = soma das taxas (máx 10.0 com 10 critérios, cada um vale 0.0–1.0)
    result.score = round(
        sum(
            [
                result.instruction_rate,
                result.json_valid_rate,
                result.json_nomd_rate,
                result.json_fields_rate,
                result.code_syntax_rate,
                result.code_nomd_rate,
                result.code_hints_rate,
                result.code_doc_rate,
                result.code_logic_rate,
            ]
        ),
        2,
    )

    return result


# ── Display ───────────────────────────────────────────────────────────────────


def pct(v: float) -> str:
    """Converte taxa 0.0–1.0 em ícone + % legível."""
    p = int(v * 100)
    if p == 100:
        return " ✅"
    if p == 0:
        return " ❌"
    return f"{p:3d}%"


def print_results(results: list[ModelResult]) -> None:
    width = 112
    print("\n" + "═" * width)
    title = (
        "  BENCHMARK — mlops-agent  |  "
        f"N={N_RUNS} runs/teste  |  English prompts  |  think=false"
    )
    print(title)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * width)
    print(
        f"  {'Model':<22} {'tok/s':>6}  "
        f"{'Instr':>5}  "
        f"{'J-Val':>5} {'J-MD':>5} {'J-Fld':>5}  "
        f"{'C-Syn':>5} {'C-MD':>5} {'C-Hnt':>5} {'C-Doc':>5} {'C-Lgc':>5}  "
        f"{'Score':>6}"
    )
    print("─" * width)

    for r in sorted(results, key=lambda x: (x.score, x.avg_toks_per_sec), reverse=True):
        if r.error:
            print(f"  {r.model:<22}  ERROR: {r.error}")
            continue
        print(
            f"  {r.model:<22} {r.avg_toks_per_sec:>5.1f}  "
            f"{pct(r.instruction_rate):>5}  "
            f"{pct(r.json_valid_rate):>5} {pct(r.json_nomd_rate):>5} "
            f"{pct(r.json_fields_rate):>5}  "
            f"{pct(r.code_syntax_rate):>5} {pct(r.code_nomd_rate):>5} "
            f"{pct(r.code_hints_rate):>5} "
            f"{pct(r.code_doc_rate):>5} {pct(r.code_logic_rate):>5}  "
            f"  {r.score:>4.1f}/9"
        )

    print("─" * width)
    best = next(
        (
            r
            for r in sorted(
                results, key=lambda x: (x.score, x.avg_toks_per_sec), reverse=True
            )
            if not r.error
        ),
        None,
    )
    if best:
        best_summary = (
            f"\n  🏆 Melhor modelo: {best.model}  "
            f"(Score {best.score}/9 | {best.avg_toks_per_sec} tok/s)"
        )
        print(best_summary)
    print("═" * width + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    results: list[ModelResult] = []
    total = len(MODELS)

    print(f"\n🚀 Benchmark iniciado — {total} modelos × {N_RUNS} runs × 3 testes")
    print(f"   Estimativa: ~{total * N_RUNS * 3} chamadas ao Ollama\n")

    for i, (model, think) in enumerate(MODELS, 1):
        print(f"  [{i:>2}/{total}] {model:<22} ", end="", flush=True)
        start = time.perf_counter()
        result = run_model(model, think)
        elapsed = round(time.perf_counter() - start, 1)

        if result.error:
            print(f" ERRO: {result.error}")
        else:
            print(f" {elapsed}s — Score {result.score:.1f}/9")

        results.append(result)

    print_results(results)

    output = {
        "timestamp": datetime.now().isoformat(),
        "n_runs": N_RUNS,
        "models_tested": len(MODELS),
        "results": [asdict(r) for r in results],
    }
    filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  📄 Resultados salvos em: {filename}\n")


if __name__ == "__main__":
    main()
