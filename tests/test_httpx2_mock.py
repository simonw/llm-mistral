import socket

import httpx2
import pytest
from pytest_httpx2 import IteratorStream


@pytest.mark.asyncio
@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.httpx2_mock(assert_all_requests_were_expected=False)
async def test_unmocked_httpx2_requests_cannot_use_network(monkeypatch, async_):
    def unexpected_network(*args, **kwargs):
        pytest.fail("An unmocked request attempted network access")

    monkeypatch.setattr(socket.socket, "connect", unexpected_network)
    monkeypatch.setattr(socket, "getaddrinfo", unexpected_network)
    with pytest.raises(httpx2.TimeoutException, match="No response can be found"):
        if async_:
            async with httpx2.AsyncClient() as client:
                await client.get("https://unmocked.invalid/")
        else:
            with httpx2.Client() as client:
                client.get("https://unmocked.invalid/")


@pytest.mark.asyncio
@pytest.mark.parametrize("async_", [False, True])
async def test_httpx2_mock_preserves_stream_chunks(httpx2_mock, async_):
    httpx2_mock.add_response(
        url="https://mocked.invalid/",
        status_code=201,
        headers={"x-test": "stream"},
        stream=IteratorStream([b"first", b"second"]),
    )
    if async_:
        async with httpx2.AsyncClient() as client:
            async with client.stream(
                "POST", "https://mocked.invalid/", content=b"body"
            ) as response:
                chunks = [chunk async for chunk in response.aiter_raw()]
    else:
        with httpx2.Client() as client:
            with client.stream(
                "POST", "https://mocked.invalid/", content=b"body"
            ) as response:
                chunks = list(response.iter_raw())
    assert chunks == [b"first", b"second"]
    assert response.status_code == 201
    assert response.headers["x-test"] == "stream"
    assert httpx2_mock.get_request().content == b"body"
