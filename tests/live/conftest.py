"""Shared setup for live provider tests.

Every module under `tests/live/` hits a real provider endpoint, so each one
carries `pytestmark = pytest.mark.live` and the root conftest skips it unless
`--run-live` is passed. Tests also skip themselves when the provider's API key
is missing, so an incomplete `.env` never fails the suite.
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session")
def require_api_key():
    def _require(env_key_name: str) -> None:
        if not os.getenv(env_key_name):
            pytest.skip(f"{env_key_name} is not set")

    return _require
