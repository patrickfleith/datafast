"""Pin the declared dependency surface against what the package imports.

`pyproject.toml` once required six packages the code never imported (instructor,
google-generativeai, anthropic, openai, gradio, botocore) while pyarrow and
huggingface_hub were imported without being declared anywhere. These tests fail on
either kind of drift.
"""

import ast
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "datafast"
PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# Imports resolved inside a function body, so the feature stays usable without them.
LAZY_IMPORTS = {"pyarrow": "parquet", "datasets": "hub", "huggingface_hub": "hub"}


def _imports(eager: bool) -> set[str]:
    """Top-level module names imported by the package, by import site.

    `eager` selects module-scope imports (needed to `import datafast` at all);
    otherwise the function-scope ones that back an optional feature.
    """
    found = set()

    def visit(node: ast.AST, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            nested = in_function or isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef)
            )
            if isinstance(child, ast.Import):
                names = [alias.name for alias in child.names]
            elif isinstance(child, ast.ImportFrom):
                # level > 0 is a relative import, always first-party.
                names = [child.module] if child.module and not child.level else []
            else:
                names = []
            if nested is not eager:
                found.update(name.split(".")[0] for name in names)
            visit(child, nested)

    for path in PACKAGE_DIR.rglob("*.py"):
        visit(ast.parse(path.read_text()), in_function=False)

    return {
        name
        for name in found
        if name not in sys.stdlib_module_names and name != "datafast"
    }


def _normalize(name: str) -> str:
    """PEP 503 name normalization: `huggingface_hub` and `huggingface-hub` are one."""
    return name.lower().replace("_", "-")


def _distributions(modules: set[str]) -> set[str]:
    """Map importable module names to the distributions that provide them.

    An uninstalled optional package is absent from the mapping, so fall back to the
    module name — these tests must pass without the extras installed.
    """
    mapping = packages_distributions()
    return {_normalize(dist) for name in modules for dist in mapping.get(name, [name])}


def _requirement_name(spec: str) -> str:
    """`pyarrow>=15.0` -> `pyarrow`, `datafast[parquet,hub]` -> `datafast`."""
    for separator in (">", "<", "=", "[", ";", " "):
        spec = spec.split(separator)[0]
    return spec


def _declared() -> tuple[set[str], dict[str, set[str]]]:
    config = tomllib.loads(PYPROJECT.read_text())["project"]
    runtime = {_normalize(_requirement_name(spec)) for spec in config["dependencies"]}
    extras = {
        name: {_normalize(_requirement_name(spec)) for spec in specs}
        for name, specs in config["optional-dependencies"].items()
    }
    return runtime, extras


def test_runtime_dependencies_match_module_scope_imports():
    """Every runtime dependency is imported eagerly, and vice versa."""
    runtime, _ = _declared()
    assert _distributions(_imports(eager=True)) == runtime


def test_lazily_imported_packages_are_declared_in_an_extra():
    """A lazy import still has to be installable by name."""
    _, extras = _declared()
    for module, extra in LAZY_IMPORTS.items():
        assert _distributions({module}) <= extras[extra], module


def test_lazy_imports_are_not_runtime_dependencies():
    """Optional features must not be forced into the base install."""
    runtime, _ = _declared()
    assert _distributions(set(LAZY_IMPORTS)) & runtime == set()


def test_recorded_lazy_imports_are_the_only_ones():
    """A new function-scope third-party import needs an extra and a line above."""
    assert _imports(eager=False) - _imports(eager=True) == set(LAZY_IMPORTS)
