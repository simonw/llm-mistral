import pytest


@pytest.fixture(autouse=True)
def mock_httpx2(httpx2_mock):
    """Reject unmocked SDK requests in every test using the native plugin."""
