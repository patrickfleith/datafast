"""Keep the README and documentation home page on the approved introduction."""

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
TAGLINE = "Generate high-quality and diverse synthetic data with LLMs."
INTRO = """
Datafast is a Python library for generating synthetic datasets with LLMs.

You assemble building blocks for data operations into a custom pipeline that generates
a synthetic dataset matching your project's needs. It works with open-weight and
proprietary models.

Start with seeds or the dimensions your dataset must cover. Connect LLMs and data
processing steps for filtering, transformations, and annotations, then run the pipeline.
Save the dataset locally or push it to the Hugging Face Hub.
"""


@pytest.mark.parametrize(
    "path, start, end, tagline",
    [
        ("README.md", "</div>", "## Install", f"<h3>{TAGLINE}</h3>"),
        ("docs/index.md", TAGLINE, "### Intended Use Cases", TAGLINE),
    ],
)
def test_homepage_intro_stays_as_approved(path, start, end, tagline):
    page = (ROOT / path).read_text(encoding="utf-8")
    assert tagline in page, f"{path} changed the approved tagline"
    _, separator, rest = page.partition(start)
    assert separator, f"{path} is missing {start!r}"
    intro, separator, _ = rest.partition(end)
    assert separator, f"{path} is missing {end!r}"
    # Keep paragraph breaks, but allow editors to rewrap lines.
    paragraphs = [" ".join(p.split()) for p in intro.strip().split("\n\n")]
    expected = [" ".join(p.split()) for p in INTRO.strip().split("\n\n")]
    assert paragraphs == expected, f"{path} changed the approved introduction"
