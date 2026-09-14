"""Utilities for validating and inspecting generated Python source."""

from __future__ import annotations

import ast
import sys

_STDLIB_MODULES = set(sys.stdlib_module_names) | {"__future__"}

GENERATED_TEST_SRC_BOOTSTRAP = (
    "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))"
)


def python_syntax_error(source: str) -> str | None:
    """Return a syntax error message, or ``None`` when source is valid."""
    try:
        ast.parse(source)
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno is not None else "unknown line"
        return f"{exc.msg} ({location})"
    return None


def extract_imported_libraries(source: str) -> set[str]:
    """Return top-level library names imported by valid Python source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    libraries: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            libraries.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            libraries.add(node.module.split(".")[0])
    return libraries - _STDLIB_MODULES


def generated_test_matches_module(
    test_source: str, module_source: str, stage_name: str
) -> bool:
    """Check that generated tests import and call the supplied module safely."""
    try:
        test_tree = ast.parse(test_source)
        module_tree = ast.parse(module_source)
    except SyntaxError:
        return False

    if not generated_test_bootstraps_src(test_source):
        return False

    module_libraries = extract_imported_libraries(module_source)
    allowed_imports = _STDLIB_MODULES | module_libraries | {"pytest", stage_name}
    module_names = {
        node.name
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    function_signatures = {
        node.name: node
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for node in ast.walk(test_tree):
        if isinstance(node, ast.Import):
            has_unknown_import = any(
                alias.name.split(".")[0] not in allowed_imports for alias in node.names
            )
            if has_unknown_import:
                return False
        elif isinstance(node, ast.ImportFrom) and node.module:
            root_name = node.module.split(".")[0]
            if root_name not in allowed_imports:
                return False
            if root_name == stage_name and any(
                alias.name not in module_names for alias in node.names
            ):
                return False
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Attribute):
                continue
            if not isinstance(node.func.value, ast.Name):
                continue
            if node.func.value.id != "stage_module":
                continue
            function = function_signatures.get(node.func.attr)
            if function is None:
                return False
            positional = [
                argument for argument in function.args.args if argument.arg != "self"
            ]
            defaults = len(function.args.defaults)
            minimum = len(positional) - defaults
            maximum = None if function.args.vararg is not None else len(positional)
            argument_count = len(node.args) + len(node.keywords)
            if argument_count < minimum or (
                maximum is not None and argument_count > maximum
            ):
                return False
    return True


def generated_test_bootstraps_src(source: str) -> bool:
    """Return whether tests insert the generated ``src/`` directory on ``sys.path``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    imports_sys = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == "sys" for alias in node.names
        ):
            imports_sys = True
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module == "sys"
            and any(alias.name == "path" for alias in node.names)
        ):
            imports_sys = True
    if not imports_sys:
        return False
    return any(
        _is_src_path_insert(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )


def _is_src_path_insert(node: ast.Call) -> bool:
    """Return True for ``sys.path.insert(0, <path built from __file__ / src>)``."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "insert":
        return False
    if not isinstance(func.value, ast.Attribute) or func.value.attr != "path":
        return False
    if not isinstance(func.value.value, ast.Name) or func.value.value.id != "sys":
        return False
    if len(node.args) < 2:
        return False
    index = node.args[0]
    if not isinstance(index, ast.Constant) or index.value != 0:
        return False
    names: set[str] = set()
    constants: set[object] = set()
    for child in ast.walk(node.args[1]):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Constant):
            constants.add(child.value)
    return "__file__" in names and "src" in constants
