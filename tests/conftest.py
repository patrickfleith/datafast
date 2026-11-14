import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests marked live",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return

    skip_live = pytest.mark.skip(reason="requires --run-live")
    for item in items:
        # The marker, not `item.keywords`: keywords also hold parametrize ids,
        # so a mocked test with an id of "live" would be skipped too.
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)
