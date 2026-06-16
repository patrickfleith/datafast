import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests marked live or integration",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return

    skip_live = pytest.mark.skip(reason="requires --run-live")
    for item in items:
        if "live" in item.keywords or "integration" in item.keywords:
            item.add_marker(skip_live)
