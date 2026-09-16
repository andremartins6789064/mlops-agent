"""Single source of truth for generated-pipeline signatures and metric name."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

PRIMARY_METRIC_NAME = "final_mse"


@dataclass(frozen=True, slots=True)
class FunctionContract:
    """Public function the generated entrypoint is allowed to call."""

    module: str
    name: str
    required_args: tuple[str, ...]
    returns: str
    optional: bool = False


@dataclass(frozen=True, slots=True)
class ContractIssue:
    """One mismatch between generated source and the agreed contract."""

    module: str
    function: str
    message: str


REQUIRED_FUNCTIONS: tuple[FunctionContract, ...] = (
    FunctionContract(
        module="feature_engineering",
        name="load_data",
        required_args=(),
        returns="tuple[features, labels]",
    ),
    FunctionContract(
        module="feature_engineering",
        name="split_data",
        required_args=("features", "labels"),
        returns="sequence[train_features, test_features, train_labels, test_labels]",
    ),
    FunctionContract(
        module="training",
        name="train_model",
        required_args=("train_features", "train_labels"),
        returns="trained model",
    ),
    FunctionContract(
        module="inference",
        name="predict",
        required_args=("model", "test_features"),
        returns="predictions",
    ),
    FunctionContract(
        module="evaluation",
        name="evaluate_model",
        required_args=("test_labels", "predictions"),
        returns=f"Mapping with numeric {PRIMARY_METRIC_NAME}",
    ),
)

OPTIONAL_FUNCTIONS: tuple[FunctionContract, ...] = (
    FunctionContract(
        module="feature_engineering",
        name="clean_data",
        required_args=("features", "labels"),
        returns="tuple[features, labels]",
        optional=True,
    ),
    FunctionContract(
        module="feature_engineering",
        name="prepare_features",
        required_args=("features", "labels"),
        returns="features or tuple[features, labels]",
        optional=True,
    ),
)


def prompt_signatures(*, module_name: str | None = None) -> str:
    """Render the contract as prompt instructions for one or all stages."""
    specs = [
        spec
        for spec in (*REQUIRED_FUNCTIONS, *OPTIONAL_FUNCTIONS)
        if module_name is None or spec.module == module_name
    ]
    lines = [
        "Required call signatures (form only, not the algorithm). "
        "Do not rename functions or change arity. Optional functions may be omitted.",
        f"Notebooks may generate data in-process with no input file. "
        f"`load_data` must not require a path; use `path: str | None = None` "
        f"if a path is useful. Evaluation must return a mapping that includes "
        f"numeric `{PRIMARY_METRIC_NAME}`.",
    ]
    for spec in specs:
        args = ", ".join(spec.required_args)
        suffix = " (optional)" if spec.optional else ""
        lines.append(f"- {spec.module}.{spec.name}({args}) -> {spec.returns}{suffix}")
    return "\n".join(lines)


def entrypoint_source() -> str:
    """Return `src/main.py` derived from the contract, without call heuristics."""
    return _ENTRYPOINT_TEMPLATE.replace("__METRIC_NAME__", PRIMARY_METRIC_NAME)


_ENTRYPOINT_TEMPLATE = '''"""Executable entrypoint for the generated ML pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import evaluation
import feature_engineering
import inference
import training

METRIC_NAME = "__METRIC_NAME__"


def run_pipeline() -> float:
    """Run all generated stages and return the primary metric."""
    loaded = feature_engineering.load_data()
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError("load_data must return (features, labels)")
    features, labels = loaded

    clean_data = getattr(feature_engineering, "clean_data", None)
    if callable(clean_data):
        cleaned = clean_data(features, labels)
        if isinstance(cleaned, tuple) and len(cleaned) == 2:
            features, labels = cleaned

    prepare_features = getattr(feature_engineering, "prepare_features", None)
    if callable(prepare_features):
        prepared = prepare_features(features, labels)
        if isinstance(prepared, tuple) and len(prepared) == 2:
            features, labels = prepared
        elif prepared is not None:
            features = prepared

    split = feature_engineering.split_data(features, labels)
    if (
        isinstance(split, (str, bytes))
        or not isinstance(split, Sequence)
        or len(split) != 4
    ):
        raise RuntimeError("split_data must return four values")
    train_features, test_features, train_labels, test_labels = split

    model = training.train_model(train_features, train_labels)
    predictions = inference.predict(model, test_features)
    metrics = evaluation.evaluate_model(test_labels, predictions)
    if not isinstance(metrics, Mapping):
        raise RuntimeError("evaluate_model must return a metrics mapping")

    value = metrics.get(METRIC_NAME, metrics.get("mse"))
    if value is None:
        raise RuntimeError(f"evaluation metrics must include {METRIC_NAME} or mse")
    return float(value)


if __name__ == "__main__":
    print(f"{METRIC_NAME}={run_pipeline():.12f}")
'''


def check_pipeline_contract(modules: Mapping[str, str]) -> list[ContractIssue]:
    """Return contract mismatches. Never raises for non-conforming source."""
    issues: list[ContractIssue] = []
    for spec in REQUIRED_FUNCTIONS:
        source = modules.get(spec.module)
        if source is None:
            issues.append(
                ContractIssue(
                    module=spec.module,
                    function=spec.name,
                    message="module missing",
                )
            )
            continue
        issues.extend(_check_function(spec, source))
    for spec in OPTIONAL_FUNCTIONS:
        source = modules.get(spec.module)
        if source is None or not _function_exists(source, spec.name):
            continue
        issues.extend(_check_function(spec, source))
    return issues


def format_contract_issues(issues: Sequence[ContractIssue]) -> str:
    """Serialize issues for CSV and logs."""
    return "; ".join(
        f"{issue.module}.{issue.function}: {issue.message}" for issue in issues
    )


def _function_exists(source: str, name: str) -> bool:
    tree = _parse_tree(source)
    if tree is None:
        return False
    return any(
        isinstance(node, ast.FunctionDef) and node.name == name for node in tree.body
    )


def _check_function(spec: FunctionContract, source: str) -> list[ContractIssue]:
    tree = _parse_tree(source)
    if tree is None:
        return [
            ContractIssue(
                module=spec.module,
                function=spec.name,
                message="invalid Python",
            )
        ]
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == spec.name
        ),
        None,
    )
    if function is None:
        return [
            ContractIssue(
                module=spec.module,
                function=spec.name,
                message="missing function",
            )
        ]
    required = _required_positional_count(function)
    expected = len(spec.required_args)
    if required != expected:
        return [
            ContractIssue(
                module=spec.module,
                function=spec.name,
                message=(
                    f"expected {expected} required argument(s) "
                    f"{spec.required_args!r}, found {required}"
                ),
            )
        ]
    return []


def _parse_tree(source: str) -> ast.Module | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    return tree if isinstance(tree, ast.Module) else None


def _required_positional_count(node: ast.FunctionDef) -> int:
    return len(node.args.args) - len(node.args.defaults)
