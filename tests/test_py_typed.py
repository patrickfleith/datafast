"""`py.typed` is the PEP 561 marker. Without it in the wheel a type checker ignores
every annotation datafast ships, so these tests guard the marker and the annotations
it makes visible."""

import inspect
import tomllib
from pathlib import Path
from typing import get_type_hints

import pytest

import datafast

ROOT = Path(__file__).parent.parent


def test_the_marker_file_exists_inside_the_package():
    assert (ROOT / "datafast" / "py.typed").exists()


def test_the_marker_is_declared_as_package_data():
    """setuptools drops any non-.py file that is not declared here."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["tool"]["setuptools"]["package-data"]["datafast"] == ["py.typed"]


def _public_classes() -> list[type]:
    classes = [
        getattr(datafast, name)
        for name in datafast.__all__
        if inspect.isclass(getattr(datafast, name))
    ]
    assert classes, "nothing to check — __all__ exports no classes"
    return classes


@pytest.mark.parametrize("cls", _public_classes(), ids=lambda c: c.__name__)
def test_every_exported_class_has_resolvable_annotations(cls):
    """`values: list[dict[str, any]]` resolved to the builtin and said nothing.

    Shipping py.typed makes an annotation like that a user-visible error, so every
    exported class must resolve against the real typing objects.
    """
    hints = get_type_hints(cls)
    assert not any(hint is any for hint in hints.values()), (
        f"{cls.__name__} annotates a field with the builtin `any`, not typing.Any"
    )


@pytest.mark.parametrize("cls", _public_classes(), ids=lambda c: c.__name__)
def test_every_constructor_forward_reference_names_something_real(cls):
    """Runner annotates `pipeline: "Pipeline"` under TYPE_CHECKING, because Pipeline
    imports Runner back. A type checker resolves that; `get_type_hints` cannot. So the
    check is not that it resolves, but that the name it defers to actually exists."""
    try:
        get_type_hints(cls.__init__)
    except NameError as e:
        assert e.name in datafast.__all__, (
            f"{cls.__name__}.__init__ refers to `{e.name}`, which is not a datafast export"
        )
