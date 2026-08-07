"""Shared setup for live provider tests.

Every module under `tests/live/` hits a real provider endpoint, so each one
carries `pytestmark = pytest.mark.live` and the root conftest skips it unless
`--run-live` is passed. Tests also skip themselves when the provider's API key
is missing, so an incomplete `.env` never fails the suite.
"""

import base64
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

ASSETS = Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def require_api_key():
    def _require(env_key_name: str) -> None:
        if not os.getenv(env_key_name):
            pytest.skip(f"{env_key_name} is not set")

    return _require


def _data_uri(name: str, media_type: str) -> str:
    data = base64.standard_b64encode((ASSETS / name).read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{data}"


@pytest.fixture(scope="session")
def image_asset():
    """A solid red 64x64 PNG, as a base64 data URI. The expected answer is "red"."""
    return _data_uri("square.png", "image/png")


@pytest.fixture(scope="session")
def document_asset():
    """A one-page PDF, as a base64 data URI. Its only text is the passphrase
    "ZEBRAFISH-42"."""
    return _data_uri("passphrase.pdf", "application/pdf")
