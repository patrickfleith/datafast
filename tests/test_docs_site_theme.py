"""The documentation site's colour scheme, pinned against the theme that renders it.

Dark mode is three lines of `mkdocs.yml` and nothing else, which is exactly why it is
easy to lose: the nav restructure will rewrite this file, and a dropped `palette` block
fails nothing and builds clean. These tests fail instead.

Every value the config names is checked against the stylesheet Zensical actually ships,
because an unsupported value is not an error — it is silently ignored and the theme
falls back to its default.
"""

import re
from pathlib import Path

import pytest

# Both live in the `docs` extra, which a contributor running `pip install -e ".[dev]"`
# will not have. Nothing here is checkable without them.
zensical = pytest.importorskip("zensical", reason="the docs extra is not installed")
yaml = pytest.importorskip("yaml", reason="the docs extra is not installed")

ROOT = Path(__file__).parent.parent
MKDOCS = ROOT / "mkdocs.yml"


def _config() -> dict:
    """`mkdocs.yml` uses `!!python/name:` tags nowhere, but does use unknown tags in
    plugin blocks, so load it with a permissive loader."""

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor("tag:yaml.org,2002:python/name:", lambda *_: None)
    Loader.add_multi_constructor("!", lambda *_: None)
    return yaml.load(MKDOCS.read_text(), Loader=Loader)


def _palette() -> list[dict]:
    palette = _config()["theme"]["palette"]
    return [palette] if isinstance(palette, dict) else palette


def _theme_dir() -> Path:
    return Path(zensical.__file__).parent / "templates"


def _palette_stylesheet() -> str:
    """The stylesheet the built pages link, not the other one Zensical ships.

    Zensical carries a `classic` and a `modern` build of the Material stylesheets, and
    they do not support the same values. Reading the wrong one would make every check
    below agree with a file the site never loads.
    """
    index = ROOT / "site" / "index.html"
    flavour = "modern"
    if index.exists():
        match = re.search(r"stylesheets/(\w+)/palette", index.read_text())
        if match:
            flavour = match.group(1)
    return next(_theme_dir().glob(f"assets/stylesheets/{flavour}/palette*.css")).read_text()


# --- the palette is a toggle, not a fixed scheme -----------------------------------


def test_the_palette_offers_both_a_light_and_a_dark_scheme():
    schemes = [entry.get("scheme") for entry in _palette()]
    assert "default" in schemes, "no light scheme"
    assert "slate" in schemes, "no dark scheme — dark mode has been dropped"


def test_there_is_exactly_one_entry_per_scheme():
    """Two entries with the same scheme give a toggle that appears to do nothing."""
    schemes = [entry.get("scheme") for entry in _palette()]
    assert len(schemes) == len(set(schemes)), schemes


@pytest.mark.parametrize("entry", _palette(), ids=lambda e: e.get("scheme", "?"))
def test_every_entry_carries_a_toggle(entry):
    """Without `toggle`, the entry is selectable by the OS but not by the reader."""
    toggle = entry.get("toggle")
    assert toggle, f"{entry.get('scheme')} has no toggle"
    assert toggle.get("icon"), "a toggle with no icon renders no button"
    assert toggle.get("name"), "the name is the button's accessible label"


@pytest.mark.parametrize("entry", _palette(), ids=lambda e: e.get("scheme", "?"))
def test_every_entry_names_the_os_preference_it_matches(entry):
    """`media` is what makes the first visit follow the reader's system setting."""
    media = entry.get("media", "")
    assert "prefers-color-scheme" in media, f"{entry.get('scheme')} has no media query"


def test_the_light_and_dark_entries_match_opposite_os_preferences():
    by_scheme = {entry["scheme"]: entry["media"] for entry in _palette()}
    assert "light" in by_scheme["default"]
    assert "dark" in by_scheme["slate"]


def test_the_two_toggles_point_at_each_other():
    """Each button offers the scheme the reader is not currently in."""
    by_scheme = {entry["scheme"]: entry["toggle"]["name"].lower() for entry in _palette()}
    assert "dark" in by_scheme["default"], by_scheme["default"]
    assert "light" in by_scheme["slate"], by_scheme["slate"]


def test_the_two_toggles_use_different_icons():
    icons = [entry["toggle"]["icon"] for entry in _palette()]
    assert len(set(icons)) == len(icons), icons


# --- every value the config names is one the theme understands ----------------------


@pytest.mark.parametrize("entry", _palette(), ids=lambda e: e.get("scheme", "?"))
def test_every_toggle_icon_exists_in_the_shipped_theme(entry):
    """A missing icon is a build error, so this is the check that keeps the build green."""
    icon = entry["toggle"]["icon"]
    assert (_theme_dir() / ".icons" / f"{icon}.svg").exists(), f"no icon {icon}"


@pytest.mark.parametrize("entry", _palette(), ids=lambda e: e.get("scheme", "?"))
def test_every_scheme_is_defined_by_the_stylesheet_the_site_loads(entry):
    scheme = entry["scheme"]
    css = _palette_stylesheet()
    if scheme == "default":
        pytest.skip("`default` is the stylesheet's baseline, not a selector")
    assert f"[data-md-color-scheme={scheme}]" in css, f"{scheme} is not a real scheme"


@pytest.mark.parametrize("entry", _palette(), ids=lambda e: e.get("scheme", "?"))
def test_every_accent_is_defined_by_the_stylesheet_the_site_loads(entry):
    accent = entry.get("accent")
    if accent is None:
        pytest.skip("no accent set")
    css = _palette_stylesheet()
    assert f"[data-md-color-accent={accent}]{{--md-accent-fg-color" in css, accent


def test_the_dark_scheme_redefines_the_code_highlighting_colours():
    """Otherwise every code block on the site keeps its light-mode syntax colours."""
    css = _palette_stylesheet()
    block = re.search(r"\[data-md-color-scheme=slate\]\{(.*?)\}", css, re.S)
    assert block, "no slate block in the stylesheet"
    assert block.group(1).count("--md-code-hl-") >= 10


def test_the_dark_scheme_actually_inverts_the_page():
    """A `slate` that did not darken the background would be a toggle to nowhere."""
    css = _palette_stylesheet()
    block = re.search(r"\[data-md-color-scheme=slate\]\{(.*?)\}", css, re.S).group(1)
    # hsla(hue, saturation%, lightness%, alpha) — the hue is itself a `var(...)`, so
    # match on the two percentages and take the second: that is the lightness.
    lightness = r"[^;]*?\d+%,\s*(\d+)%"
    background = re.search(r"--md-default-bg-color:" + lightness, block)
    foreground = re.search(r"--md-default-fg-color:" + lightness, block)
    assert background and foreground, block[:200]
    assert int(background.group(1)) < 20, "the slate background is not dark"
    assert int(foreground.group(1)) > 80, "the slate text is not light"


def test_links_stay_readable_on_the_dark_background():
    """`--md-typeset-a-color` follows the primary colour, which is nearly black here.

    Material special-cases the dark scheme for exactly this combination. If that rule
    ever disappears, links become black text on a black page.
    """
    css = _palette_stylesheet()
    primary = _palette()[0].get("primary")
    rule = re.search(
        r"\[data-md-color-scheme=slate\]\[data-md-color-primary=" + re.escape(primary)
        + r"\][^{]*\{([^}]*)\}",
        css,
    )
    assert rule, f"no slate link-colour rule for primary={primary}"
    assert "--md-typeset-a-color" in rule.group(1)


def test_the_configured_primary_is_ignored_by_this_stylesheet():
    """A finding, not a requirement — see CONCERNS.md.

    `primary: black` predates dark mode and is not one of the colours Zensical's
    `modern` stylesheet defines, so the header renders in the theme's default indigo in
    both schemes. Pick a supported colour (`grey`, `blue-grey`) or drop the line — and
    when you do, delete this test rather than editing it.
    """
    css = _palette_stylesheet()
    primary = _palette()[0].get("primary")
    assert primary == "black"
    assert f"[data-md-color-primary={primary}]{{--md-primary-fg-color" not in css


def test_both_schemes_use_the_same_primary_and_accent():
    """Only the background and text should change between the two."""
    entries = _palette()
    assert len({entry.get("primary") for entry in entries}) == 1
    assert len({entry.get("accent") for entry in entries}) == 1
