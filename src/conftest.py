"""
pytest configuration
Adds a `--runslow` option to pytest to run slow tests.
Adds a @pytest.mark.slow decorator to tests. These will only run
in pytest if the `--runslow` argument is used.
"""

import pytest


def pytest_addoption(parser):
    """ Add options to pytest """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    """ Configures pytest """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """ Hook which pytest calls after collection of all test items is completed. """
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
